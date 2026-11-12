"""Exact independent point-head replay and geometry-preserving rendered gray controls."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
from pycocotools import mask as codec

PROTOCOL="761d5b3c008df8e04a9259fe76cd760f1f2b4e6f79398b32808bc2cece2421a9"
PAIR="6a2d1d14c8359f5aed18c54879bed5bf1e10bde575e59e47d3ce4583507f4a11"
PAIR_MATH="8fe7a2c146c08c89f377b1f20b321276a60356de4bdbb27be2a00544dfc77962"
TRAIN="3111c34afd791cb15987ed94c5e898de4c3ff2cad61677b616420651335d812c"
OLD="53ba7adae5d3697cd77e1b147a331889c78ee29d17dea4c51101f9fe421b5aff"
OLD_PRELABEL="7c763fd75a8870a84426b2e8a30f8b1e4be6eefedd68dd2e2c88b8c617992519"
OLD_WAIT="dc254ce8ff521a4508d0e68aff8e82371f7047c41121a249dc21c8e3b45bc07a"
SALT="DFINE-FIT109-gray-joint-v1|"
MODES=("raw","gray_zero","gray_shuffled","full")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(path,digest):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact independent replay source")
    return raw


def load(path,digest,name):
    obj=ModuleType(name)
    obj.__file__=str(path)
    exec(compile(pin(path,digest),str(path),"exec"),obj.__dict__)
    return obj


def shuffle_gray(features, group, rank):
    copied=features.copy()
    for sign in (0,1):
        for depth in (1,2):
            positions=np.flatnonzero(features[:,sign,2]==depth)
            seed=int.from_bytes(sha256(f"{SALT}{group}|{rank}|{sign}|{depth}".encode()).digest()[:16],"little")
            permutation=np.random.Generator(np.random.PCG64(seed)).permutation(positions)
            copied[positions,sign,9:14]=features[permutation,sign,9:14]
    return copied


def ablation_training(path):
    raw=pin(path,TRAIN).decode()
    substitutions=(
        ('HEADS=("independent","shuffled","paired")','HEADS=("full","gray_zero","gray_shuffled")'),
        ('    own_t=torch.from_numpy(standard)\n',''),
        ('        partner=np.concatenate([s["data"]["contexts"][arm] for s in train_slots])\n'
         '        partner_t=torch.from_numpy(np.asarray((partner-center)/scale,dtype=np.float32))',
         '        arm_inputs=standard\n'
         '        if arm=="gray_zero":\n'
         '            arm_inputs=standard.copy()\n'
         '            arm_inputs[:,:,9:14]=0\n'
         '        elif arm=="gray_shuffled":\n'
         '            shuffled=np.concatenate([s["data"]["gray_shuffled"] for s in train_slots])\n'
         '            arm_inputs=np.asarray((shuffled-center)/scale,dtype=np.float32)\n'
         '        own_t=torch.from_numpy(arm_inputs)\n'
         '        partner_t=own_t'),
    )
    for before,after in substitutions:
        need(raw.count(before)==1,"Exact training ablation seam")
        raw=raw.replace(before,after,1)
    obj=ModuleType("_gray_control_equal_trainer")
    obj.__file__=str(path)
    exec(compile(raw,str(path),"exec"),obj.__dict__)
    return obj


def point_prepare(planned):
    counters=Counter()
    for block in planned:
        for rank,data in block["data"].items():
            own=data["features"]
            shuffled=shuffle_gray(own,block["group"],rank)
            need(np.array_equal(own[:,:,:9],shuffled[:,:,:9]),"Gray-only permutation")
            data["gray_shuffled"]=shuffled
            counters.update(candidates=1,pairs=len(own),changed_endpoints=int(np.any(shuffled[:,:,9:14]!=own[:,:,9:14],axis=2).sum()))
    return dict(counters)


def train_and_choose(slots,math,learner,old):
    chosen={m:[[] for _ in slots] for m in MODES}
    audited=[]
    for fold in range(5):
        models,center,scale,audit=learner.train(slots,fold)
        ref=old["training"][fold]["heads"]["independent"]
        check=audit["heads"]["full"]
        need(check==ref,"Exact independent-head original training readout")
        for i,s in enumerate(slots):
            if s["fold"]!=fold or s["data"] is None:
                continue
            data=s["data"]
            own=data["features"]
            zero=own.copy()
            zero[:,:,9:14]=center[9:14]
            values=dict(full=own,gray_zero=zero,gray_shuffled=data["gray_shuffled"])
            for arm,model in models.items():
                f=values[arm]
                gains=learner.gains(model,f,f,center,scale)
                chosen[arm][i]=math.choose_pairs(gains,min(8,s["area"]//10,len(gains)))
        audited.append(dict(fold=fold,**audit))
        print(json.dumps(dict(event="DFINE_FIT109_GRAY_ABLATION_FOLD_COMPLETE",fold=fold,steps=200),
                         sort_keys=True),flush=True)
    return chosen,audited


def materialize(slots,chosen,base,official,gt):
    metrics,budgets={},{}
    for arm in MODES:
        scope=Counter()
        rows=[]
        for s,ids in zip(slots,chosen[arm],strict=True):
            row=s["row"]
            if not ids:
                rows.append(row)
                scope["no_op"]+=1
                continue
            coords=s["data"]["coordinates"][ids]
            rle=base.apply(row["segmentation"],(coords[:,0].tolist(),coords[:,1].tolist()))
            need(int(codec.area(rle))==s["area"],"Exact balanced no-score-change mask")
            rows.append(dict(row,segmentation=rle))
            scope.update(edited=1,swaps=len(ids))
            scope.update({"edited_"+s["role"]:1})
        need(len(rows)==10900 and scope["no_op"]+scope["edited"]==10900,"Exact original candidate scope")
        metrics[arm]=official.official(gt,rows)
        budgets[arm]={k:scope[k] for k in ("edited","swaps","edited_valid","edited_unmatched","edited_neutral","no_op")}
    return metrics,budgets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","pair","pair-math","training","old","base","capture","accepted","prelabel","wait",
                "fit-gt","code","shapely-site","official","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    a=parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Exact CPU-only scope")
    pin(a.protocol,PROTOCOL)
    pair=load(a.pair,PAIR,"_closed_pair")
    base=load(a.base,pair.BASE,"_closed_base")
    math=load(a.pair_math,PAIR_MATH,"_closed_math")
    learner=ablation_training(a.training)
    official=load(a.official,pair.OFFICIAL,"_closed_official")
    old=json.loads(pin(a.old,OLD))
    need(old["advance"] is False and old["FIT"] and not old["DEV"] and not old["TEST"],"Preserve stopped pair mechanism")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _,blocks=base.read_core(a)
    a.prelabel_sha=OLD_PRELABEL
    a.wait_sha=OLD_WAIT
    original,planned=pair.sealed_points(a,base,math,blocks)
    prep=point_prepare(planned)
    need(prep["pairs"]==original["counts"]["pairs"] and prep["candidates"]==original["counts"]["supported_candidates"],
         "Identical closed label-free points")
    a.output.mkdir()
    pre=base.seal(a.output/"PRELABEL_SAFE.json",base.encoded(dict(status="DFINE_FIT109_GRAY_ABLATION_PRELABEL",
                protocol_sha256=PROTOCOL,old_actual_sha256=OLD,old_prelabel_sha256=OLD_PRELABEL,FIT=True,
                DEV=False,TEST=False,GT_opened=False,AP=False,counts=prep)))
    gt,anns=pair.read_labels(a,base,blocks)
    slots,scope=pair.fixed_slots(a,base,blocks,planned,gt,anns)
    chosen,training=train_and_choose(slots,math,learner,old)
    coco,budgets=materialize(slots,chosen,base,official,gt)
    pair.MODES=MODES
    pair.GEOMETRY=(*MODES,"oracle")
    fixed=pair.fixed_geometry(slots,chosen,math)
    fixed["full_minus"]=fixed.pop("paired_minus")
    del fixed["full_minus"]["full"]
    need(coco["full"]==old["cohorts"]["independent"] and budgets["full"]==old["budgets"]["independent"] and
         fixed["metrics"]["full"]==old["fixed_geometry"]["metrics"]["independent"] and
         fixed["metrics"]["oracle"]==old["fixed_geometry"]["metrics"]["oracle"] and
         coco["raw"]==old["cohorts"]["raw"],"Exact closed independent full replay and oracle")
    gates=dict(AP75_retained=coco["full"]["AP75"]>=coco["raw"]["AP75"]-1e-10,
               strict90_retained=fixed["metrics"]["full"]["strict_90"]>=fixed["metrics"]["raw"]["strict_90"])
    for arm in ("gray_zero","gray_shuffled"):
        d=fixed["full_minus"][arm]
        gates["AP_vs_"+arm]=coco["full"]["AP"]-coco[arm]["AP"]>=.30
        gates["geometry_vs_"+arm]=d["delta_points"]>=.20 and d["interval95_points"][0]>0.
    result=dict(status="DFINE_FIT109_GRAY_ABLATION_ACTUAL",FIT=True,DEV=False,TEST=False,GT_opened=True,AP=True,
                detector_forwards=0,detector_optimizer_updates=0,point_optimizer_updates=3000,full_exact_replay=True,
                protocol_sha256=PROTOCOL,prior_actual_sha256=OLD,prelabel_sha256=pre,match_scope=scope,
                counts=prep,training=training,cohorts=coco,budgets=budgets,fixed_geometry=fixed,
                prior_partner_advance=False,gates=gates,advance=all(gates.values()))
    digest=base.seal(a.output/"RESULT_SAFE.json",base.encoded(result))
    print(json.dumps(dict(status=result["status"],result_sha256=digest,advance=result["advance"],gates=gates),
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()