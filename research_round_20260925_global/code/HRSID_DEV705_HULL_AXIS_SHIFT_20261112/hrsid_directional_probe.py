"""Replay equal along-hull and across-hull translations on the closed HRSID pair."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

P374_SHA = "cba94ecd0df862b6a303117a56fa889bc76c0a8c017af50fff2cdb9859f94139"
P404_SHA = "a27aabc220feea27ce3a059cd8b7f7c9fc20c257a03490326ffd33d54cdf18a6"
CELLS=("base","along_plus","along_minus","across_plus","across_minus")

def source(path,digest,name):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=digest:
        raise ValueError("Expected byte-pinned closed analysis source "+path.name)
    obj=ModuleType(name)
    obj.__file__=str(path)
    sys.modules[name]=obj
    exec(compile(raw,str(path),"exec"),obj.__dict__)
    return obj

def population(gt,valid,neutral,tools,np,need):
    import shapely
    from shapely.affinity import translate
    image={item["id"]:item for item in gt["images"]}
    cells={k:{} for k in CELLS}
    excluded=dict(invalid_source=len(neutral),low_aspect=0,translated_out_of_frame=0)
    self_iou={k:[] for k in CELLS[1:]}
    areas=dict(small=0,medium=0,large=0)
    guide=np.array([.5,.5])
    ratios=[]
    for ann in gt["annotations"]:
        aid=ann["id"]
        if aid in neutral:
            continue
        geom,rings=valid[aid]
        coords=np.asarray(geom.minimum_rotated_rectangle.exterior.coords)
        edges=coords[1:]-coords[:-1]
        lengths=np.linalg.norm(edges,axis=1)
        need(len(lengths)==4 and lengths.min()>0,"Expected nondegenerate source geometry")
        ratio=lengths.max()/lengths.min()
        if ratio<2:
            excluded["low_aspect"]+=1
            neutral.add(aid)
            continue
        tangent=edges[lengths.argmax()]/lengths.max()
        normal=np.array([-tangent[1],tangent[0]])
        tangent,normal=tools.orient(tangent,guide),tools.orient(normal,guide)
        need(abs(tangent @ normal)<1e-12,"Expected orthogonal directions")
        shifts=dict(along_plus=.5*tangent,along_minus=-.5*tangent,
                    across_plus=.5*normal,across_minus=-.5*normal)
        shifts_geo={k:translate(geom,xoff=x,yoff=y) for k,(x,y) in shifts.items()}
        im=image[ann["image_id"]]
        canvas=shapely.box(0,0,im["width"],im["height"])
        if any(moved.difference(canvas).area>1e-9 for moved in shifts_geo.values()):
            excluded["translated_out_of_frame"]+=1
            neutral.add(aid)
            continue
        ratios.append(float(ratio))
        cells["base"][aid]=(geom,rings)
        for key,delta in shifts.items():
            moved=shifts_geo[key]
            shift=np.asarray([moved.centroid.x-geom.centroid.x,moved.centroid.y-geom.centroid.y])
            need(np.allclose(shift,delta,rtol=0,atol=1e-9)
                 and abs(geom.area-moved.area)<=1e-8*max(1,geom.area),"Equal displacement and shape preservation required")
            cells[key][aid]=(moved,[np.asarray(ring)+delta for ring in rings])
            cross=geom.intersection(moved).area
            self_iou[key].append(cross/(geom.area+moved.area-cross))
        area=ann["area"]
        areas["small" if area<1024 else "medium" if area<9216 else "large"]+=1
    need(all(set(d)==set(cells["base"]) for d in cells.values())
         and len(cells["base"])+len(neutral)==len(gt["annotations"]),"Same cohort for all direction cells")
    return cells,dict(excluded=excluded,median_aspect=float(np.median(ratios)),supported_by_size=areas,
                      mean_self_iou_loss_points={k:float(100*(1-np.mean(v))) for k,v in self_iou.items()})

def main():
    parser=ArgumentParser(description=__doc__)
    for name in ("code-root","pair","native-arm-receipt","center-arm-receipt","accepted-result","shapely-site","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("pair-sha256","training-protocol","accepted-result-sha256","shift-source-sha256"):
        parser.add_argument("--"+name,required=True)
    a=parser.parse_args()
    started=perf_counter()
    if not a.output.is_absolute() or a.output.exists() or os.environ.get("CUDA_VISIBLE_DEVICES")!="-1":
        parser.error("Fresh output and CPU-only execution required")
    sys.path.append(str(a.shapely_site))
    import numpy as np
    import shapely
    root=a.code_root
    engine=source(root/"HRSID_DEV705_CONTINUOUS_AP_20261111/full_continuous_ap.py",P374_SHA,"_h705_axis_engine")
    tools=source(root/"SSDD_DEV187_HULL_AXIS_SHIFT_20261112/directional_probe.py",P404_SHA,"_h705_axis_tools")
    need=engine.require
    need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2"
         and engine.version("numpy")=="1.26.4" and engine.version("pycocotools")=="2.0.11","Exact numeric environment")
    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("Only closed local DEV inputs")
    sys.addaudithook(offline)
    shift=engine.load_source(root/"HRSID_COMMON_DEV705_FIXED_SHIFT_AP_20261111/fixed_shift_ap.py",
                             a.shift_source_sha256,"_h705_axis_closed")
    p311=shift.load_p311(root)
    need((a.pair_sha256,a.training_protocol,a.accepted_result_sha256)==
         (p311.PAIR_SHA,p311.TRAINING_PROTOCOL,p311.PRIMARY_SHA),"Exact closed paired evaluation required")
    closed=p311.load_closed(root)
    driver=closed.load_gate(root).load_driver()
    accepted=driver.read_json(a.accepted_result,a.accepted_result_sha256)
    need(accepted["status"]==p311.PRIMARY_STATUS and accepted["training_seed"]==5202
         and accepted["official_TEST_opened"] is False,"Exact prior endpoint accepted")
    gt,native,center,provenance=closed.verified_inputs(root,a.pair,a.pair_sha256,
        native_arm_receipt=a.native_arm_receipt,center_arm_receipt=a.center_arm_receipt,
        training_protocol=a.training_protocol)
    need(len(gt["images"])==705 and len(gt["annotations"])==1782
         and provenance["training_pair_receipt_sha256"]==accepted["independent_training_pair_sha256"]
         and provenance["original_dev_gt_sha256"]==accepted["original_dev_gt_sha256"],"Exact original HRSID DEV geometry")
    adapter=closed.load_diagnostics(root)["dev705_paired_mask_failure_scope"].load_official_adapter()
    vp=root/"HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110"
    vector=engine.load_source(vp/"vector_label_geometry.py",
        "4877c48d38a03e58e97163381ba544df3c90a78dde85c9141085fa0b92da3b09","_h705_axis_vector")
    foot=engine.load_source(vp/"vector_population_math.py",
        "0821d2cc3b6ca793d33fdfe35b634f3678aad072fc55b86352fa2501e969fc7e","_h705_axis_foot").run_footprint_metrics
    streams=dict(native=native,center=center)
    ranks={}
    for arm,rows in streams.items():
        original,ranks[arm],_=engine.score(gt,rows,adapter)
        old=accepted["scores"][arm]["primary"]
        need(np.allclose(original["all12"],old["all12"],rtol=0,atol=1e-12)
             and abs(original["AP90"]-old["AP90"])<=1e-12,"Exact old raster scores before direction analysis")
    valid,neutral,_=engine.polygon_support(gt,vector)
    cells,pop=population(gt,valid,neutral,tools,np,need)
    metrics,audits={},{}
    for cell,polys in cells.items():
        metrics[cell],audits[cell]={},{}
        for arm,rows in streams.items():
            scores,layout,audit=engine.score(gt,rows,adapter,neutral,polys,shift.runs,foot)
            need(layout==ranks[arm] and audit["dense_geometry_spots"]==16 and audit["pruned_dense_spots"]==8,
                 "Same ranking and independent exact overlap checks required")
            metrics[cell][arm],audits[cell][arm]=scores,audit
        print(json.dumps(dict(event="HRSID_HULL_AXIS_CELL_COMPLETE",cell=cell)),flush=True)
    deltas={cell:{metric:100*(scores["center"][metric]-scores["native"][metric])
                 for metric in engine.METRICS} for cell,scores in metrics.items()}
    result=dict(status="HRSID_DEV705_EQUAL_HULL_DIRECTION_SHIFT_ACTUAL",images=705,gt=1782,
        common=1782-len(neutral),guide_xy=[.5,.5],shift_px=.5,aspect_min=2,population=pop,
        seed=5202,metrics=metrics,delta_points=deltas,response_points=tools.summaries(deltas,engine.METRICS),
        geometry_audits=audits,raster_reproduced=True,fixed_candidates=True,dev=True,test=False,fit=False,
        optimizer_updates=0,model_forward_calls=0,elapsed_seconds=perf_counter()-started)
    raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(status=result["status"],sha256=sha256(raw).hexdigest(),response=result["response_points"]["AP"])),flush=True)

if __name__=="__main__":
    main()