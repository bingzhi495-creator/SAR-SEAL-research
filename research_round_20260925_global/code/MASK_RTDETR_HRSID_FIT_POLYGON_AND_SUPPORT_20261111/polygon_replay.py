"""Replay the frozen Mask RT-DETR views with continuous polygon and lost-support readouts."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from types import SimpleNamespace


SOURCE_PINS = {
    "replay.py":"58ffa34c705e9fdb295ae25ef596ecaabebf495c6695697ad6346d30839cc834",
    "scope.py":"9ce6427b5123849a055874165af57fe913f48520f0e7d06e5c25c3a9ddd933c7",
    "footprint.py":"060a9101ca428ed48d74648bd04b049a969b028f71bbaa0b3db6e56fe78967ff",
    "paddle_geometry.py":"9508c2b1e852b0d0fb35c81e58399e162ae9d5fdda04b9c2f0d37b9cc4ab5240",
    "vector_label_geometry.py":"4877c48d38a03e58e97163381ba544df3c90a78dde85c9141085fa0b92da3b09",
    "vector_population_math.py":"0821d2cc3b6ca793d33fdfE35B634F3678AAD072FC55B86352FA2501E969FC7E".lower(),
}
ORIGINAL_SHA = "da102a21ac7a81316061e2812c97db5761a8cad1b4af21558e7481141c7bfb88"
PLAN_SHA = "3d78297dd49a99be208a91f4bf45d2d56a5d0f1ae55b80328dbb4a42de6102ed"


def require(condition,message):
    if not condition:
        raise ValueError(message)


def run(args):
    for root,names in ((args.paddle_code,tuple(SOURCE_PINS)[:4]),(args.vector_code,tuple(SOURCE_PINS)[4:])):
        for name in names:
            raw=(root/name).read_bytes()
            require(sha256(raw).hexdigest()==SOURCE_PINS[name],"Exact original code required "+name)
    sys.path[:0] = [str(Path(__file__).resolve().parent),str(args.paddle_code),str(args.vector_code)]
    import numpy as np
    import shapely
    import replay
    from polygon_support import continuous_polygon, polygon_cell_summary, polygon_scores, support_cell
    from scope import BATCH_SIZE, canonical
    from stock_trace import StockTrace
    from vector_label_geometry import UnsupportedPolygon

    require(version("shapely")=="2.1.2" and shapely.geos_version_string=="3.13.1","Exact GEOS runtime")
    old_raw=args.original.read_bytes()
    require(sha256(old_raw).hexdigest()==ORIGINAL_SHA,"Exact closed raster reference result")
    old=json.loads(old_raw)
    plan_raw=args.plan.read_bytes()
    require(sha256(plan_raw).hexdigest()==PLAN_SHA,"Exact original source, grouping and seeds")
    plan=json.loads(plan_raw)
    group_order=[item["group"] for item in plan["sources"]]
    trace=StockTrace(args.upstream)
    saved={}
    polygon_records=defaultdict(list)
    census=Counter()
    lost=[]
    lost_source_groups=set()
    lost_instance_keys=set()
    original_transform,original_rows=replay.transform_batch,replay.view_rows

    def witnessed_batch(stock,samples,*,seed):
        if not all("gt_read_order" in row for row in samples):
            return original_transform(stock,samples,seed=seed)
        views,steps=trace.run(samples,seed=seed)
        for view,observed,source in zip(views,steps,samples,strict=True):
            saved[id(view)]=(observed,source["gt_poly"])
        return views

    def polygon_rows(tagged, stock, source_rows, group, counters, records):
        original_rows(tagged,stock,source_rows,group,counters,records)
        steps,polygons=saved.pop(id(tagged))
        census["witnessed_traces"]+=1
        native,center=tagged.output["gt_segm"],tagged.center_masks
        pre=tagged.before_resize["gt_segm"]
        order=tagged.before_resize["gt_read_order"].reshape(-1)
        for row_idx,ref,a,b in zip(order,pre,native,center,strict=True):
            if not ref.any():
                continue
            in_h,in_w=ref.shape
            out_h,out_w=a.shape
            if int(np.count_nonzero(ref))*out_h*out_w>=1024*in_h*in_w:
                continue
            census["old_small_rows"]+=1
            ref_poly=None
            try:
                if not isinstance(polygons[int(row_idx)],list):
                    census["nonpolygon"]+=1
                else:
                    poly,pre_hw,out_hw=continuous_polygon(polygons[int(row_idx)],tagged.source_hw,steps)
                    require(pre_hw==ref.shape and out_hw==a.shape,"Raster and vector shapes must match")
                    if poly.area>0:
                        ref_poly=poly
                    else:
                        census["no_continuous_visible_area"]+=1
            except UnsupportedPolygon:
                census["invalid_polygon"]+=1
            if ref_poly is not None:
                data=polygon_scores(ref_poly,ref.shape,{"native":a,"center":b})
                polygon_records[group].append(data)
                census["positive_polygon"]+=1
            if not b.any() and a.any():
                rescued,raster=support_cell(ref,a.shape)
                raster_scores=replay.exact_footprint_scores(ref,(a,rescued))
                result=dict(raster=raster, native_pixel_count=int(np.count_nonzero(a)),
                            rescue_pixel_in_stock_native=bool((rescued & a).any()),
                            raster_native_iou=raster_scores["native"]["iou"],
                            raster_empty_center_iou=0.,raster_one_cell_iou=raster_scores["center"]["iou"],
                            cropped=tagged.cropped, image_nearest=tagged.image_interpolation==0)
                if ref_poly is not None:
                    ps=polygon_scores(ref_poly,ref.shape,{"native":a,"empty_center":b,"one_cell":rescued})
                    result["polygon"]=polygon_cell_summary(ref_poly,a.shape)
                    result["polygon_iou"]={k:v["iou"] for k,v in ps.items()}
                else:
                    result["polygon"]=None
                lost.append(result)
                lost_source_groups.add(group)
                lost_instance_keys.add((group,int(row_idx)))
            elif not a.any() or not b.any():
                raise ValueError("Unexpected empty-output direction beyond fixed prior report")

    replay.transform_batch=witnessed_batch
    replay.view_rows=polygon_rows
    try:
        replay.run(SimpleNamespace(bundle=args.bundle,plan=args.plan,upstream=args.upstream,
                    protocol_commit=old["protocol_commit"],plan_sha256=PLAN_SHA,output=args.replayed))
    finally:
        replay.transform_batch=original_transform
        replay.view_rows=original_rows
    repeated=json.loads(args.replayed.read_bytes())
    require({k:v for k,v in old.items() if k!="elapsed_seconds"}==
            {k:v for k,v in repeated.items() if k!="elapsed_seconds"}, "Exact prior result except wall time")
    require(not saved and census["witnessed_traces"]==96 and census["old_small_rows"]==581 and len(lost)==3,
            "Exact original view, raster population and all three center-empty cases")
    summary=replay.group_summary(polygon_records,lambda _:True)
    blocks=[]
    for i in range(0,len(group_order),BATCH_SIZE):
        omitted=set(group_order[i:i+BATCH_SIZE])
        stat=replay.group_summary({k:v for k,v in polygon_records.items() if k not in omitted},lambda _:True)
        blocks.append(dict(block=i//BATCH_SIZE+1, valid_polygon_groups=stat["groups"],
                           polygon_iou_gain=stat["iou"]["favorable_delta"],
                           vertical_abs_improvement_px=stat["y_absolute"]["favorable_delta"]))
    corroborates=bool(summary["iou"]["favorable_delta"]>0 and summary["y_absolute"]["favorable_delta"]>0
                     and all(row["polygon_iou_gain"]>0 and row["vertical_abs_improvement_px"]>0 for row in blocks))
    result=dict(status="MASK_RTDETR_HRSID_FIT_POLYGON_AND_EMPTY_SUPPORT_COMPLETE_V1",
                original_raster_sha256=ORIGINAL_SHA, plan_sha256=PLAN_SHA,
                followup_protocol_commit=args.protocol_commit, code_source_sha256=SOURCE_PINS,
                shapely_version=shapely.__version__,geos_version=shapely.geos_version_string,
                original_raster_replay_exact_except_elapsed=True, counts=dict(census),
                polygon_group_stats=summary, leave_one_fixed_batch_block=blocks,
                direction_corroborates_raster=corroborates, known_center_empty_cases=lost,
                center_empty_source_groups=len(lost_source_groups),
                center_empty_distinct_source_instances=len(lost_instance_keys),
                center_empty_batch_blocks=sum(bool(lost_source_groups.intersection(group_order[i:i+BATCH_SIZE]))
                                              for i in range(0,len(group_order),BATCH_SIZE)),
                official_DEV_opened=False,official_TEST_opened=False,model_executed=False,
                optimizer_updates=0,AP_evaluated=False)
    raw=canonical(result)
    with args.output.open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(event=result["status"],sha256=sha256(raw).hexdigest(),corroborates=corroborates),sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("paddle-code","vector-code","upstream","bundle","plan","original","replayed","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--protocol-commit",required=True)
    run(p.parse_args())

