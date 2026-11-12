"""Freeze and independently replay one all-FIT D-FINE point head."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np

PROTOCOL="3fed15ee224f639add8fdf3db24af18cad0dcbbd5e07c884122bf2693450095d"
COMMON="f26da339c97bb4e22044d85e1cc4abb78e117a6bc3fde69500c08e2fca155ae5"
FREE_SOURCE="d24b7050be95786fffdee386e1e8a6f87ddcb8df8b69c2171eb752e6b6a35fef"
FREE_RESULT="66867e13d7697ac18aff843433d5f2d171773e002a2cf8a1d9bf63da4ccddd7f"
SHAPES={"center":(14,),"scale":(14,),"state.network.0.weight":(24,28),"state.network.0.bias":(24,),
        "state.network.2.weight":(24,24),"state.network.2.bias":(24,),
        "state.network.4.weight":(1,24),"state.network.4.bias":(1,)}


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def pin(path,digest):
    data=path.read_bytes()
    need(sha256(data).hexdigest()==digest,"Exact immutable input")
    return data


def load(path,digest,name):
    out=ModuleType(name)
    out.__file__=str(path)
    exec(compile(pin(path,digest),str(path),"exec"),out.__dict__)
    return out


def float_array(a):
    out=np.ascontiguousarray(a,dtype="<f4")
    need(np.isfinite(out).all(),"Finite float32 scalers and model tensors")
    return out


def array_hash(a):
    return sha256(float_array(a).tobytes()).hexdigest()


def package_bytes(values):
    file=BytesIO()
    np.savez(file,**{k:float_array(v) for k,v in sorted(values.items())})
    return file.getvalue()


def read_archive(raw):
    with np.load(BytesIO(raw),allow_pickle=False) as file:
        need(len(file.files)==len(set(file.files)) and all(file[k].dtype==np.dtype("<f4") for k in file.files),
             "Exact duplicate-free pickleless float32 arrays")
        return {k:float_array(file[k]) for k in file.files}


def validate_arrays(values):
    need({k:v.shape for k,v in values.items()}==SHAPES and np.all(values["scale"]>0) and
         sum(v.size for k,v in values.items() if k.startswith("state."))==1321,
         "Exact unchanged 1321-parameter point head and positive feature scalers")


def restore_head(learner,raw):
    import torch
    values=read_archive(raw)
    validate_arrays(values)
    head=learner.PointHead(len(values["center"]))
    head.load_state_dict({k.removeprefix("state."):torch.from_numpy(v) for k,v in values.items()
                          if k.startswith("state.")},strict=True)
    return head.eval(),values["center"],values["scale"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","common","free","free-result","gray","pair","pair-math","training","base","capture",
                "accepted","prelabel","wait","fit-gt","code","shapely-site","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--verify-root",type=Path)
    parser.add_argument("--verify-result-sha")
    a=parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and
         (a.verify_root is None)==(a.verify_result_sha is None),"Exact CPU-only refit or complete verification receipt")
    pin(a.protocol,PROTOCOL)
    common=load(a.common,COMMON,"_refit_common")
    free=load(a.free,FREE_SOURCE,"_refit_free_witness")
    previous=json.loads(pin(a.free_result,FREE_RESULT))
    need(previous["advance"] and previous["exact_fixed_bce_replay"] and previous["FIT"] and not previous["DEV"] and
         not previous["TEST"],"Exact successful FIT-only endpoint gate")
    gray=common.load(a.gray,common.GRAY,"_refit_gray")
    pair=common.load_pair(a.pair,gray.PAIR)
    base=common.load(a.base,pair.BASE,"_refit_fit109")
    math=common.load(a.pair_math,gray.PAIR_MATH,"_refit_points")
    learner=gray.ablation_training(a.training)
    # Keep the source moment and sampler unchanged, while admitting all original folds.
    train_source=pin(a.training,gray.TRAIN).decode()
    source=train_source[train_source.index("def train("):train_source.index("\n\ndef gains(")]
    for before,after in (
            ('s["fold"]!=fold and s["truth"]','s["truth"]'),
            ('        partner=np.concatenate([s["data"]["contexts"][arm] for s in train_slots])\n'
             '        partner_t=torch.from_numpy(np.asarray((partner-center)/scale,dtype=np.float32))','        partner_t=own_t')):
        need(source.count(before)==1,"Exact all-FIT independent BCE seam")
        source=source.replace(before,after,1)
    exec(compile(source,str(a.training),"exec"),learner.__dict__)
    learner.HEADS=("full",)
    free.adapt_function(a.training,gray.TRAIN,learner,"gains",None,
                        (('def gains(','def probabilities('),('return points[:,0]-points[:,1]','return points')))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _,blocks=base.read_core(a)
    a.prelabel_sha=gray.OLD_PRELABEL
    a.wait_sha=gray.OLD_WAIT
    old,planned=pair.sealed_points(a,base,math,blocks)
    a.output.mkdir()
    pre=base.seal(a.output/"PRELABEL_SAFE.json",base.encoded(dict(status="DFINE_FIT109_BCE_FINAL_REFIT_PRELABEL",
                  FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,protocol_sha256=PROTOCOL,
                  endpoint_result_sha256=FREE_RESULT,old_prelabel_sha256=gray.OLD_PRELABEL,counts=old["counts"])))
    gt,anns=pair.read_labels(a,base,blocks)
    slots,scope=pair.fixed_slots(a,base,blocks,planned,gt,anns)
    models,center,scale,audit=learner.train(slots,0)
    head=models["full"]
    counts=audit["heads"]["full"]
    need((audit["groups"],audit["unique_train_pairs"],counts["steps"],counts["parameters"],
          counts["observed_pair_examples"])==(93,13076,200,1321,12800) and
         counts["initial_tensor_sha256"]==previous["training"][0]["heads"]["full"]["initial_tensor_sha256"] and
         scope==previous["match_scope"],"Exact all-FIT fit support, fold0 initialization and budget")
    arrays=dict(center=center,scale=scale)
    arrays.update({"state."+k:v.detach().numpy() for k,v in head.state_dict().items()})
    validate_arrays(arrays)
    hashes={k:array_hash(v) for k,v in arrays.items()}
    raw=package_bytes(arrays)
    frozen,rc,rs=restore_head(learner,raw)
    witness=next(s["data"]["features"] for s in slots if s["truth"] is not None)
    need(np.array_equal(learner.probabilities(head,witness,witness,center,scale),
                        learner.probabilities(frozen,witness,witness,rc,rs)),"Frozen readback inference parity")
    common_fields=dict(FIT=True,DEV=False,TEST=False,GT_opened=True,AP=False,protocol_sha256=PROTOCOL,
                       prelabel_sha256=pre,endpoint_result_sha256=FREE_RESULT,detector_forwards=0,
                       detector_optimizer_updates=0,point_optimizer_updates=200,audit=audit,array_sha256=hashes,
                       point_supervision_scope=scope,scaler_dimensions=14)
    if a.verify_root is None:
        weights=base.seal(a.output/"POINT_HEAD_PRIVATE.npz",raw)
        result=dict(status="DFINE_FIT109_BCE_FINAL_REFIT_ACTUAL",weight_bytes=len(raw),weight_sha256=weights,
                    **common_fields)
    else:
        prior=json.loads(pin(a.verify_root/"RESULT_SAFE.json",a.verify_result_sha))
        prior_raw=pin(a.verify_root/"POINT_HEAD_PRIVATE.npz",prior["weight_sha256"])
        prior_arrays=read_archive(prior_raw)
        validate_arrays(prior_arrays)
        need(prior["status"]=="DFINE_FIT109_BCE_FINAL_REFIT_ACTUAL" and
             all(prior[k]==v for k,v in common_fields.items()) and len(prior_raw)==prior["weight_bytes"] and
             all(np.array_equal(prior_arrays[k],v) for k,v in arrays.items()),"Independent replay tensor and scaler equality")
        result=dict(status="DFINE_FIT109_BCE_FINAL_REFIT_INDEPENDENT_ACCEPTED",
                    result_sha256=a.verify_result_sha,weight_sha256=prior["weight_sha256"],
                    weight_bytes=len(prior_raw),exact_tensor_replay=True,**common_fields)
    digest=base.seal(a.output/"RESULT_SAFE.json",base.encoded(result))
    print(json.dumps(dict(status=result["status"],safe_sha256=digest,weight_sha256=result["weight_sha256"]),
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
