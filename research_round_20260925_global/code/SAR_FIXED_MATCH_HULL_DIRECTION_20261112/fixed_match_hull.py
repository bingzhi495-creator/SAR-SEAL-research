"""Fixed baseline COCO pairs and symmetric contour IoU loss on two datasets."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

P404 = "a27aabc220feea27ce3a059cd8b7f7c9fc20c257a03490326ffd33d54cdf18a6"
P406 = "f1666555db09b5e2af935c82664f830de3874cab25fe91aa506c5c33707e7356"
P402 = "68c71e1c22cc6c3c82260b626a3738ba80067948d7a81874540c6ad1d7ea6511"
P374 = "cba94ecd0df862b6a303117a56fa889bc76c0a8c017af50fff2cdb9859f94139"
ARMS = ("native","center")
CELLS = ("base","along_plus","along_minus","across_plus","across_minus")

def module(path,digest,name):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=digest:
        raise ValueError("Expected closed exact analysis source "+path.name)
    obj=ModuleType(name)
    obj.__file__=str(path)
    sys.modules[name]=obj
    exec(compile(raw,str(path),"exec"),obj.__dict__)
    return obj

def score_pairs(gt,streams,polys,reference,template,geometry,adapter,engine,run_lengths,need,np):
    from pycocotools import mask
    ann_images={a["id"]:a["image_id"] for a in gt["annotations"]}
    image_ids=[im["id"] for im in gt["images"]]
    assignments={}
    for arm,rows in streams.items():
        prior=reference[arm]
        result,eval_images,_=template.evaluate_cell(gt,rows,image_ids,adapter)
        need(all(abs(result[k]-prior[k])<=1e-12 for k in ("AP","AP50","AP75","AP90","APs")),
             "Original fixed raster metrics differ before matching")
        assignments[arm]=geometry.match_official(eval_images,rows,ann_images)
    common=sorted(set(assignments["native"])&set(assignments["center"])&set(polys["base"]))
    need(len(common)>0,"Fixed original common match population required")
    values={}
    count_by_size=dict(small=0,medium=0,large=0)
    anns={a["id"]:a for a in gt["annotations"]}
    for aid in common:
        area=anns[aid]["area"]
        count_by_size["small" if area<1024 else "medium" if area<9216 else "large"]+=1
    for arm in ARMS:
        ious=[]
        for aid in common:
            rle=assignments[arm][aid]
            size=rle["size"]
            counts=rle["counts"]
            rectangles=engine.boxes_from_counts(run_lengths(counts.decode("ascii") if isinstance(counts,bytes) else counts),*size)
            area=float(mask.area(rle))
            need(area>0,"A matched detection must contain foreground")
            row=[]
            for cell in CELLS:
                shape,_=polys[cell][aid]
                intersection=engine.overlap_area(rectangles,shape)
                need(0<=intersection<=min(shape.area,area)+1e-7,"Exact matched polygon overlap outside geometry")
                row.append(intersection/(shape.area+area-intersection))
            ious.append(row)
        cells=np.asarray(ious,dtype=float)
        along=100*(cells[:,0]-(cells[:,1]+cells[:,2])/2)
        across=100*(cells[:,0]-(cells[:,3]+cells[:,4])/2)
        values[arm]=dict(baseline=100*cells[:,0],along=along,across=across,extra_across=across-along)
    keys=("baseline","along","across","extra_across")
    summaries={arm:{key:float(values[arm][key].mean()) for key in keys} for arm in ARMS}
    interaction={key:float(np.mean(values["center"][key]-values["native"][key])) for key in keys}
    wins={arm:int(np.count_nonzero(values[arm]["extra_across"]>0)) for arm in ARMS}
    wins["center_more_direction_sensitive"]=int(np.count_nonzero(values["center"]["extra_across"]>values["native"]["extra_across"]))
    return dict(original_matches={arm:len(rows) for arm,rows in assignments.items()},
                common_fixed_matches=len(common),common_by_size=count_by_size,arm_means_points=summaries,
                center_minus_native_points=interaction,counts=wins)

def main():
    parser=ArgumentParser(description=__doc__)
    for k in ("code-root","shapely-site","pair","native-arm-receipt","center-arm-receipt","accepted-result","ssdd-reference",
              "ssdd-integer-gt","ssdd-float-gt","seed5201-native","seed5201-center","seed5202-native","seed5202-center","output"):
        parser.add_argument("--"+k,type=Path,required=True)
    for k in ("pair-sha256","training-protocol","accepted-result-sha256","shift-source-sha256"):
        parser.add_argument("--"+k,required=True)
    a=parser.parse_args()
    started=perf_counter()
    sys.path.append(str(a.shapely_site))
    import numpy as np
    import shapely
    r=a.code_root
    engine=module(r/"HRSID_DEV705_CONTINUOUS_AP_20261111/full_continuous_ap.py",P374,"_fm_engine")
    need=engine.require
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and not a.output.exists() and a.output.is_absolute()
         and shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2"
         and engine.version("numpy")=="1.26.4" and engine.version("pycocotools")=="2.0.11",
         "Fresh isolated previously qualified CPU numerical runtime")
    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("Only closed local DEV")
    sys.addaudithook(offline)
    ss=module(r/"SSDD_DEV187_HULL_AXIS_SHIFT_20261112/directional_probe.py",P404,"_fm_ssdd")
    hr=module(r/"HRSID_DEV705_HULL_AXIS_SHIFT_20261112/hrsid_directional_probe.py",P406,"_fm_hrsid")
    support=module(r/"SSDD_DEV187_CONTINUOUS_TWO_ENCODING_20261112/continuous_ssdd.py",P402,"_fm_support")
    shift=engine.load_source(r/"HRSID_COMMON_DEV705_FIXED_SHIFT_AP_20261111/fixed_shift_ap.py",a.shift_source_sha256,"_fm_shift")
    p311=shift.load_p311(r)
    need((a.pair_sha256,a.training_protocol,a.accepted_result_sha256)==
         (p311.PAIR_SHA,p311.TRAINING_PROTOCOL,p311.PRIMARY_SHA),"Exact closed HRSID pair identity")
    closed=p311.load_closed(r)
    driver=closed.load_gate(r).load_driver()
    accepted=driver.read_json(a.accepted_result,a.accepted_result_sha256)
    need(accepted["status"]==p311.PRIMARY_STATUS and accepted["official_TEST_opened"] is False
         and accepted["training_seed"]==5202,"Accepted HRSID endpoint")
    gt,native,center,provenance=closed.verified_inputs(r,a.pair,a.pair_sha256,
        native_arm_receipt=a.native_arm_receipt,center_arm_receipt=a.center_arm_receipt,training_protocol=a.training_protocol)
    need(len(gt["images"])==705 and len(gt["annotations"])==1782 and
         provenance["training_pair_receipt_sha256"]==accepted["independent_training_pair_sha256"],"Exact HRSID scope")
    wrapper=engine.load_source(r/"SSDD_DEV187_POST_PRIMARY_RANK_20261110/ssdd_fixed_match_geometry.py",
        "5aebe92a3b355874e8a20f24ef19b98ace8c20e3537610ec92a9c0e026b260e3","_fm_adapter")
    _,_,template,_,adapter,geometry=wrapper.load_helpers(r,r/"HRSID_MASK_PHASE_REPEAT_DEV705_20261109/dev705_paired_mask_geometry.py")
    vec=engine.load_source(r/"HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110/vector_label_geometry.py",
        "4877c48d38a03e58e97163381ba544df3c90a78dde85c9141085fa0b92da3b09","_fm_vec")
    valid,neutral,_=engine.polygon_support(gt,vec)
    cells,pop=hr.population(gt,valid,neutral,ss,np,need)
    need(len(cells["base"])==1555,"Exact published HRSID directional population")
    results={}
    prior={arm:accepted["scores"][arm]["primary"] for arm in ARMS}
    results["HRSID_5202"]=score_pairs(gt,dict(native=native,center=center),cells,prior,template,geometry,adapter,engine,
                                      shift.runs,need,np)
    print(json.dumps(dict(event="FIXED_MATCH_DIRECTION_DONE",dataset="HRSID_5202",
        stats=results["HRSID_5202"]["center_minus_native_points"])),flush=True)
    old=json.loads(support.pinned(a.ssdd_integer_gt,support.INTEGER_SHA))
    floating=json.loads(support.pinned(a.ssdd_float_gt,support.FLOAT_SHA))
    ref=json.loads(support.pinned(a.ssdd_reference,support.REFERENCE_SHA))
    oi,no,_=engine.polygon_support(old,vec)
    fi,nf,_=engine.polygon_support(floating,vec)
    need(not no and not nf and len(oi)==len(fi)==380 and old["images"]==floating["images"],"Exact SSDD scope")
    cells,neutral,_=ss.cell_population(old,oi,fi,need)
    need(len(cells["base"])==338,"Exact published SSDD directional population")
    for seed in support.SEEDS:
        row=ref["seeds"][seed]
        streams={arm:json.loads(support.pinned(getattr(a,"seed"+seed+"_"+arm),row["prediction_sha256"][arm])) for arm in ARMS}
        key="SSDD_"+seed
        results[key]=score_pairs(old,streams,cells,row["integer"],template,geometry,adapter,engine,shift.runs,need,np)
        print(json.dumps(dict(event="FIXED_MATCH_DIRECTION_DONE",dataset=key,stats=results[key]["center_minus_native_points"])),
              flush=True)
    payload=dict(status="SAR_FIXED_MATCH_SYMMETRIC_HULL_IOU_ACTUAL",results=results,
                 fixed_original_match_threshold=.5,aspect_min=2,shift_px=.5,dev=True,test=False,fit=False,
                 model_forward_calls=0,optimizer_updates=0,seconds=perf_counter()-started)
    raw=(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as out:
        out.write(raw)
    print(json.dumps(dict(status=payload["status"],sha256=sha256(raw).hexdigest())),flush=True)

if __name__=="__main__":
    main()