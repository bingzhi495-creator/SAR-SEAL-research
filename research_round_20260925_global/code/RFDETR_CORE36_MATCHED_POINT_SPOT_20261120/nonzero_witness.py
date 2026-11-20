"""Artificial endpoint predictions check nonzero writes on one fixed real FIT candidate."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType

SOURCES={"run_spot.py":"75ff72a3177e99f06d212a878c8d8921de4db6da8f4249722fb762718083d0ee",
         "fit_step.py":"94e60f5b5557787687d1f228c33c249a1d7889172a225373b97bd789ec8b72b1",
         "feature_adapter.py":"747c4c4b43561f8d073e01854adf241f1cb66c4efa1d29f304ae8837ecd67b73"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("capture","manifest","acceptance","fit-images","code","posterior","output"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parent
    sys.path.insert(0,str(root))
    for name,digest in SOURCES.items():
        if sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("Exact closed source "+name)
    import run_spot as run
    import fit_step as fit
    import feature_adapter as feat
    import numpy as np
    import torch
    from pycocotools import mask as codec
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run.need(not torch.cuda.is_initialized(),"CPU-only nonzero write witness")
    posterior=fit.load(a.posterior,run.ADAPTER_SHA,"_same_locked_probability")
    selected=run.read_cohort(a,feat.select_images)
    data,counts,seal=run.prepare(a,selected,fit,feat,posterior,np,torch,codec)
    reference=None
    for item,candidates,records in data:
        by_rank={row["rank"]:row for row in candidates}
        for record in records:
            candidate=by_rank[record["rank"]]
            cap=min(8,candidate["area"]//10,len(record["coordinates"]))
            if len(record["coordinates"])>=2 and cap>=2:
                reference=(item,candidate,record,cap)
                break
        if reference is not None:
            break
    run.need(reference is not None,"First fixed supported candidate with at least two true pairs")
    item,candidate,record,cap=reference
    coords=record["coordinates"]
    n=len(coords)
    small=np.tile(np.asarray([.1,.9],dtype=np.float32),(n,1))
    small[:2]=[.9,.1]
    large=np.tile(np.asarray([.9,.4],dtype=np.float32),(n,1))
    native=record["features"][:,:,0]
    stock=codec.decode(candidate["rle"]).astype(bool)
    outputs=feat.readouts_sparse(np,stock,coords,small,large,native,cap)
    run.need(outputs["SF"]["edits"]==outputs["SU=F"]["edits"]==outputs["SU"]["edits"]==4 and
             all(outputs[k]["edits"]==4 for k in ("LF=S","LU=S","LW=S","NP=S")),
             "Artificial-q exact positive S action and common four-flip budgets")
    values={}
    for arm,change in outputs.items():
        mask=feat.apply_sparse(np,stock,change)
        actual_xor=int(np.count_nonzero(mask!=stock))
        actual_delta=int(mask.sum())-candidate["area"]
        rle=codec.encode(np.asfortranarray(mask,dtype=np.uint8))
        rle["counts"]=rle["counts"].decode("ascii")
        run.need(actual_xor==change["edits"] and actual_delta==change["area_delta"] and
                 int(codec.area(rle))==candidate["area"]+actual_delta and
                 (arm=="LW=S" or actual_delta==0) and
                 (arm!="N" or rle==candidate["rle"]),"Exact actual write XOR, area, RLE and raw no-op")
        values[arm]={key:change[key] for key in
                     ("adds","drops","edits","area_delta","nonpositive_actions","nonpositive_flips")}
        values[arm].update(actual_xor=actual_xor,actual_area_delta=actual_delta,rle_area=int(codec.area(rle)))
    a.output.mkdir()
    safe=dict(status="RFDETR_CORE36_REAL_POINTS_ARTIFICIAL_PREDICTION_WRITE_WITNESS",FIT=True,DEV=False,TEST=False,
              GT_opened=False,AP=False,effect_evidence=False,point_optimizer_updates=0,model_forward_calls=0,
              first_deterministic_candidate=True,selected_images=4,feature_seal_sha256=seal,
              supported_pairs=n,cap=cap,original_area=candidate["area"],
              artificial_S=dict(default_add=.1,default_drop=.9,first_two_add=.9,first_two_drop=.1),
              artificial_L=dict(all_add=.9,all_drop=.4),native_posterior="exact original RF endpoint posterior",
              verified_policies=values)
    h=run.write(a.output/"RESULT_SAFE.json",safe)
    private=dict(image_id=item["image_id"],rank=candidate["rank"],query_id=candidate["query_id"])
    run.write(a.output/"CANDIDATE_ID_PRIVATE.json",private)
    print(json.dumps(dict(result_sha256=h,**safe)),flush=True)


if __name__=="__main__":
    main()
