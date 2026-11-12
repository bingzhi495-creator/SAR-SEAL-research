"""Free or fixed endpoint ranking on the exact same BCE point predictions."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np

PROTOCOL="4a55c85ae9e0710846a4a6c8a1ed69acc9b49d8280aefcf3a0219e01b1abd55b"
COMMON="f26da339c97bb4e22044d85e1cc4abb78e117a6bc3fde69500c08e2fca155ae5"
DIRECT="bd8c8067eeea9e7efdc9c8f6f442cf1ab3e239daccbda3cb7c860ea70c368586"
MODES=("raw","fixed","free_matched_count","free_natural")


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def pin(path,digest):
    data=path.read_bytes()
    need(sha256(data).hexdigest()==digest,"Exact immutable source or aggregate")
    return data


def load(path,digest,name):
    out=ModuleType(name)
    out.__file__=str(path)
    exec(compile(pin(path,digest),str(path),"exec"),out.__dict__)
    return out


def adapt_function(path,digest,module,begin,end,replacements):
    raw=pin(path,digest).decode()
    start=raw.index("def "+begin+"(")
    finish=len(raw) if end is None else raw.index("\n\ndef "+end+"(",start)
    source=raw[start:finish]
    for before,after in replacements:
        need(source.count(before)==1,"Exact original function seam")
        source=source.replace(before,after,1)
    exec(compile(source,str(path),"exec"),module.__dict__)


def ranked_endpoints(probabilities):
    p=np.asarray(probabilities)
    need(p.ndim==2 and p.shape[1]==2 and np.isfinite(p).all(),"Exact finite endpoint probabilities")
    ordinals=np.arange(len(p))
    return p,np.lexsort((ordinals,-p[:,0])),np.lexsort((ordinals,p[:,1]))


def choose_free(probabilities,cap,fixed_count=None):
    p,add,drop=ranked_endpoints(probabilities)
    need(0<=cap<=len(p),"Original bounded action cap")
    marginal=p[add[:cap],0]-p[drop[:cap],1]
    k=int((marginal>0).sum()) if fixed_count is None else fixed_count
    need(0<=k<=cap,"Exact admissible free count")
    return list(zip(add[:k].tolist(),drop[:k].tolist(),strict=True))


def endpoint_deltas(values,actions):
    a=np.asarray(actions,dtype=np.intp)
    return values[a[:,0],0]-values[a[:,1],1]


def adapted_readouts(a,common,gray,pair):
    material=(('coords=s["data"]["coordinates"][ids]',
               'indices=np.asarray(ids,dtype=np.intp)\n            points=s["data"]["coordinates"]\n'
               '            coords=np.column_stack((points[indices[:,0],0],points[indices[:,1],1]))'),)
    adapt_function(a.gray,common.GRAY,gray,"materialize","main",material)
    adapt_function(a.pair,gray.PAIR,pair,"fixed_geometry","score",
                   (('+[oracle]','+[[(i,i) for i in oracle]]'),
                    ('(q[action,0]-q[action,1])','endpoint_deltas(q,action)'),
                    ('if m=="paired":','if m==MODES[3]:'),
                    ('paired_minus=effects','free_natural_minus=effects')))
    adapt_function(a.common,COMMON,common,"threshold","summarize",
                   (('(q[ids,0]-q[ids,1])','endpoint_deltas(q,ids)'),))
    common.MODES=gray.MODES=pair.MODES=MODES
    pair.GEOMETRY=(*MODES,"oracle")
    common.endpoint_deltas=pair.endpoint_deltas=endpoint_deltas


def fit(slots,learner,math,prior):
    actions={m:[[] for _ in slots] for m in MODES}
    audits=[]
    usage=Counter()
    learner.HEADS=("full",)
    for fold in range(5):
        models,center,scale,audit=learner.train(slots,fold)
        need(dict(fold=fold,**audit)==prior["training"][fold] | {"heads":{"full":prior["training"][fold]["heads"]["full"]}},
             "Exact prior fixed BCE training")
        for i,s in enumerate(slots):
            if s["fold"]!=fold or s["data"] is None:
                continue
            f=s["data"]["features"]
            p=learner.probabilities(models["full"],f,f,center,scale)
            cap=min(8,s["area"]//10,len(p))
            fixed=[(j,j) for j in math.choose_pairs(p[:,0]-p[:,1],cap)]
            natural=choose_free(p,cap)
            same=choose_free(p,cap,len(fixed))
            actions["fixed"][i]=fixed
            actions["free_matched_count"][i]=same
            actions["free_natural"][i]=natural
            if same:
                need(endpoint_deltas(p.astype(np.float64),same).sum()>=
                     endpoint_deltas(p.astype(np.float64),fixed).sum()-1e-12,"Matched free sum utility optimality")
                n=int((endpoint_deltas(p,same)<=0).sum())
                usage.update(nonpositive_marginal_matched_count_swaps=n,
                             matched_count_candidates_with_nonpositive_marginal=int(n>0),
                             matched_count_candidates_different_endpoint_set=int(
                                 any({v[j] for v in same}!={v[j] for v in fixed} for j in (0,1))))
            usage.update(natural_count_above_fixed=int(len(natural)>len(fixed)),
                         natural_count_below_fixed=int(len(natural)<len(fixed)),
                         natural_count_same_as_fixed=int(len(natural)==len(fixed)))
        audits.append(dict(fold=fold,**audit))
        print(json.dumps(dict(event="DFINE_FIT109_FREE_ENDPOINT_FOLD_COMPLETE",fold=fold,steps=200),
                         sort_keys=True),flush=True)
    return actions,audits,dict(usage)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","common","direct-result","gray","previous","pair","pair-math","training","base","capture",
                "accepted","prelabel","wait","fit-gt","code","shapely-site","official","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    a=parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Exact CPU-only scope")
    pin(a.protocol,PROTOCOL)
    common=load(a.common,COMMON,"_free_count_common")
    old_direct=json.loads(pin(a.direct_result,DIRECT))
    prior=json.loads(pin(a.previous,common.PREVIOUS))
    need(old_direct["exact_natural_bce_replay"] and not old_direct["advance"] and prior["full_exact_replay"] and
         not prior["advance"] and not prior["DEV"] and not prior["TEST"],"Preserve prior scope and stop decisions")
    gray=common.load(a.gray,common.GRAY,"_free_gray")
    pair=common.load_pair(a.pair,gray.PAIR)
    base=common.load(a.base,pair.BASE,"_free_fit109")
    math=common.load(a.pair_math,gray.PAIR_MATH,"_free_points")
    learner=gray.ablation_training(a.training)
    official=common.load(a.official,pair.OFFICIAL,"_free_stock_coco")
    adapt_function(a.training,gray.TRAIN,learner,"gains",None,
                   (('def gains(','def probabilities('),('return points[:,0]-points[:,1]','return points')))
    adapted_readouts(a,common,gray,pair)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _,blocks=base.read_core(a)
    a.prelabel_sha=gray.OLD_PRELABEL
    a.wait_sha=gray.OLD_WAIT
    old,planned=pair.sealed_points(a,base,math,blocks)
    a.output.mkdir()
    pre=base.seal(a.output/"PRELABEL_SAFE.json",base.encoded(dict(status="DFINE_FIT109_FREE_ENDPOINT_PRELABEL",
                  FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,protocol_sha256=PROTOCOL,
                  prior_direct_sha256=DIRECT,prior_gray_sha256=common.PREVIOUS,
                  old_prelabel_sha256=gray.OLD_PRELABEL,counts=old["counts"])))
    gt,anns=pair.read_labels(a,base,blocks)
    slots,scope=pair.fixed_slots(a,base,blocks,planned,gt,anns)
    actions,audits,usage=fit(slots,learner,math,prior)
    coco,budgets=gray.materialize(slots,actions,base,official,gt)
    geometry=pair.fixed_geometry(slots,actions,math)
    transitions,rows=common.threshold(slots,actions)
    common.validate_transitions(transitions,geometry)
    need(coco["fixed"]==prior["cohorts"]["full"] and coco["raw"]==prior["cohorts"]["raw"] and
         budgets["fixed"]==prior["budgets"]["full"]==budgets["free_matched_count"] and
         all(geometry["metrics"][key]==prior["fixed_geometry"]["metrics"][prev]
             for key,prev in (("fixed","full"),("raw","raw"),("oracle","oracle"))) and
         transitions["fixed"]==old_direct["natural"]["strict90_transitions"]["bce"] and scope==prior["match_scope"],
         "Exact prior fixed BCE replay, same-count and original identities")
    witness=base.seal(a.output/"STRICT_TRANSITIONS_PRIVATE.jsonl",b"".join(base.encoded(r) for r in rows))
    natural=geometry["free_natural_minus"]
    pair.MODES=("raw","fixed","free_natural","free_matched_count")
    pair.GEOMETRY=(*pair.MODES,"oracle")
    matched=pair.fixed_geometry(slots,actions,math)["free_natural_minus"]
    # Compare the matched-count arm to the same witnesses using the same bootstrap stream.
    matched.pop("free_natural",None)
    gates=dict(AP_vs_raw=coco["free_natural"]["AP"]-coco["raw"]["AP"]>=.50,
               AP_vs_fixed=coco["free_natural"]["AP"]-coco["fixed"]["AP"]>=.30,
               AP75_retained=coco["free_natural"]["AP75"]>=coco["raw"]["AP75"]-1e-10,
               zero_original_strict90_loss=transitions["free_natural"]["lost"]==0)
    for arm in ("raw","fixed"):
        d=natural[arm]
        gates["geometry_vs_"+arm]=d["delta_points"]>=.20 and d["interval95_points"][0]>0
    result=dict(status="DFINE_FIT109_FREE_ENDPOINT_ACTUAL",FIT=True,DEV=False,TEST=False,GT_opened=True,AP=True,
                detector_forwards=0,detector_optimizer_updates=0,point_optimizer_updates=1000,
                protocol_sha256=PROTOCOL,prelabel_sha256=pre,prior_direct_sha256=DIRECT,
                prior_gray_sha256=common.PREVIOUS,transition_witness_sha256=witness,exact_fixed_bce_replay=True,
                cohorts=coco,budgets=budgets,geometry=geometry,matched_count_minus=matched,usage=usage,
                strict90_transitions=transitions,training=audits,match_scope=scope,gates=gates,advance=all(gates.values()))
    digest=base.seal(a.output/"RESULT_SAFE.json",base.encoded(result))
    print(json.dumps(dict(status=result["status"],result_sha256=digest,gates=gates,advance=result["advance"]),
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
