"""Run the prelocked SSDD FIT geometry comparison through stock Paddle operators."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from ssdd_catalog import BATCH_SIZE, REPEATS, SOURCES, canonical, catalog, choose, make_plan, read_member, require, verify_archive

PLAN_SHA="10f671be4aa4ee146d1a7f1a40dfe52a41aa68c8db04587e2f8b5d70e8d0ecb8"
DEPENDENCIES={
    "paddle": {
        "scope.py":"9ce6427b5123849a055874165af57fe913f48520f0e7d06e5c25c3a9ddd933c7",
        "replay.py":"58ffa34c705e9fdb295ae25ef596ecaabebf495c6695697ad6346d30839cc834",
        "footprint.py":"060a9101ca428ed48d74648bd04b049a969b028f71bbaa0b3db6e56fe78967ff",
        "paddle_geometry.py":"9508c2b1e852b0d0fb35c81e58399e162ae9d5fdda04b9c2f0d37b9cc4ab5240",
    },
    "vector": {
        "vector_label_geometry.py":"4877c48d38a03e58e97163381ba544df3c90a78dde85c9141085fa0b92da3b09",
        "vector_population_math.py":"0821d2cc3b6ca793d33fdfe35b634f3678aad072fc55b86352fa2501e969fc7e",
    },
    "polygon": {"stock_trace.py":"7e0894078d5a6a1faa5857d7b7b6355ea108820b531fb60890b3b7b3cf8f66e8","polygon_support.py":"78a893ba62607bf77b7112b277d6e61a9d7c0decace3a2cb64949a799d1ab882"},
}


def deny_network(event,args):
    if event.startswith("socket."):
        raise RuntimeError("Offline SSDD FIT geometry required")


def run(args):
    start=perf_counter()
    dependencies={key:getattr(args,key+"_code") for key in DEPENDENCIES}
    for key,pins in DEPENDENCIES.items():
        for name,expected in pins.items():
            require(sha256((dependencies[key]/name).read_bytes()).hexdigest()==expected,
                    "Expected exact shared geometry source "+name)
    sys.path[:0]=[str(HERE),*(str(path) for path in dependencies.values())]
    import cv2
    import numpy as np
    import shapely
    from footprint import exact_footprint_scores
    from paddle_geometry import SOURCE_SHA256, UPSTREAM_COMMIT, load_stock, transform_batch
    from polygon_support import continuous_polygon, polygon_scores
    from replay import close_enough_dict, group_summary, stock_parser
    from stock_trace import StockTrace
    from vector_label_geometry import UnsupportedPolygon

    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and os.environ.get("OMP_NUM_THREADS")=="1",
            "Expected CPU-only single-thread geometry replay")
    runtime={package:version(package) for package in ("numpy","opencv-python","pycocotools","pillow","pyyaml","shapely")}
    require(runtime==dict(numpy="1.26.4",**{"opencv-python":"4.5.5.64"},pycocotools="2.0.8",pillow="12.3.0",
                          pyyaml="6.0.3",shapely="2.1.2") and cv2.__version__=="4.5.5"
            and shapely.geos_version_string=="3.13.1","Expected locked operator and GEOS versions")
    cv2.setNumThreads(1)
    sys.addaudithook(deny_network)
    verify_archive(args.bundle)
    plan_raw=args.plan.read_bytes()
    require(sha256(plan_raw).hexdigest()==PLAN_SHA,"Expected exact fixed SSDD cohort and schedule")
    plan=json.loads(plan_raw)
    stock=load_stock(args.upstream)
    trace=StockTrace(args.upstream)
    dropped=Counter()
    with ZipFile(args.bundle) as bundle:
        rows,annotations=catalog(bundle)
        require(canonical(make_plan(rows))==plan_raw,"Prospective group roster mismatch")
        selected=choose(rows)
        samples=[]
        for row in selected:
            pin=row["original_image_ref"]
            jpeg=read_member(bundle,row["archive_image_entry"],pin["sha256"],pin["bytes"])
            require(jpeg.startswith(b"\xff\xd8"),"Expected exact original JPEG")
            sample,excluded=stock_parser(jpeg,annotations[row["image_id"]])
            dropped.update(excluded)
            sample["h"],sample["w"]=row["original_shape"]
            require(len(sample["gt_bbox"])>0,"Stock default parser would exclude a zero-box record")
            samples.append(sample)

    census=Counter()
    raster_groups,polygon_groups=defaultdict(list),defaultdict(list)
    for ordinal,seed in enumerate(plan["seeds"]):
        start_index=(ordinal%(SOURCES//BATCH_SIZE))*BATCH_SIZE
        source=samples[start_index:start_index+BATCH_SIZE]
        with_ids=[dict(row,gt_read_order=np.arange(len(row["gt_bbox"]),dtype=np.int32)[:,None]) for row in source]
        native=transform_batch(stock,source,seed=seed)
        witnessed,steps=trace.run(with_ids,seed=seed)
        for j,(unobserved,observed,transforms) in enumerate(zip(native,witnessed,steps,strict=True)):
            entry=selected[start_index+j]
            group=entry["original_image_ref"]["sha256"]
            require(unobserved.source_hw==observed.source_hw==tuple(entry["original_shape"]),
                    "Exact original rectangular dimensions required")
            for flag in ("expanded","cropped","flipped","image_interpolation"):
                require(getattr(unobserved,flag)==getattr(observed,flag),"Trace changed the stock choice "+flag)
            close_enough_dict(unobserved.before_resize,observed.before_resize,{"gt_read_order"})
            close_enough_dict(unobserved.output,observed.output,{"gt_read_order"})
            require(np.array_equal(unobserved.center_masks,observed.center_masks),"Trace changed centered masks")
            original=observed.before_resize["gt_segm"]
            a=observed.output["gt_segm"]
            b=observed.center_masks
            source_ids=observed.before_resize["gt_read_order"].reshape(-1)
            require(np.array_equal(source_ids,observed.output["gt_read_order"].reshape(-1))
                    and len(set(source_ids))==len(source_ids)==len(a)==len(b)==len(original)
                    and all(0<=i<len(source[j]["gt_poly"]) for i in source_ids),"Exact retained rows")
            in_h,in_w=transforms[-1][1]
            out_h,out_w=transforms[-1][2]
            if (in_h,in_w)==(out_h,out_w):
                direction="identity"
                require(np.array_equal(a,b) and np.array_equal(a,original),"Identity resize witness")
            elif in_h<=out_h and in_w<=out_w:
                direction="up"
            elif in_h>=out_h and in_w>=out_w:
                direction="down"
            else:
                direction="mixed"
            nearest=observed.image_interpolation==cv2.INTER_NEAREST
            census["views"]+=1
            census["cropped_views"]+=observed.cropped
            census["expanded_views"]+=observed.expanded
            census["flipped_views"]+=observed.flipped
            census["nearest_image_views"]+=nearest
            census[direction+"_views"]+=1
            census["rows_before_integer_ops"]+=len(source[j]["gt_bbox"])
            census["rows_after_integer_ops"]+=len(source_ids)
            for ref,left,right,idx in zip(original,a,b,source_ids,strict=True):
                require(np.array_equal(left,cv2.resize(ref,(out_w,out_h),interpolation=cv2.INTER_NEAREST)),
                        "Independent OpenCV witness did not match stock mask")
                census["native_masks_witnessed"]+=1
                census["masks_differ"]+=not np.array_equal(left,right)
                count=int(np.count_nonzero(ref))
                if count==0:
                    require(not left.any() and not right.any(),"Empty support generated foreground")
                    census["empty_before_resize"]+=1
                    continue
                if count*out_h*out_w>=1024*in_h*in_w:
                    census["large_reference_rows"]+=1
                    continue
                tags=dict(cropped=observed.cropped,nearest_image=nearest,direction=direction)
                census["small_positive_raster_rows"]+=1
                raster=exact_footprint_scores(ref,(left,right))
                raster_groups[group].append(dict(raster,**tags))
                try:
                    poly,pre_hw,out_hw=continuous_polygon(source[j]["gt_poly"][int(idx)],observed.source_hw,transforms)
                except UnsupportedPolygon:
                    census["invalid_vector_rows"]+=1
                    continue
                require(pre_hw==ref.shape and out_hw==left.shape,"Exact raster and vector coordinate conventions")
                if poly.area<=0:
                    census["no_continuous_visible_area"]+=1
                    continue
                census["small_positive_vector_rows"]+=1
                vector=polygon_scores(poly,pre_hw,{"native":left,"center":right})
                polygon_groups[group].append(dict(vector,**tags))
        print(json.dumps(dict(event="SSDD_MASK_RTDETR_FIT_BATCH_DONE",batches=ordinal+1,views=census["views"])),flush=True)
    require(census["views"]==SOURCES*REPEATS and census["rows_after_integer_ops"]==census["native_masks_witnessed"],
            "Expected all frozen FIT views and retained rows")
    subsets={"all":lambda row:True,"cropped":lambda row:row["cropped"],"uncropped":lambda row:not row["cropped"],
             "nearest_image":lambda row:row["nearest_image"],"other_image":lambda row:not row["nearest_image"],
             **{kind:(lambda row,kind=kind:row["direction"]==kind) for kind in ("up","down","mixed","identity")}}
    raster_result={name:group_summary(raster_groups,predicate) for name,predicate in subsets.items()}
    polygon_result={name:group_summary(polygon_groups,predicate) for name,predicate in subsets.items()}
    blocks=[]
    group_order=[item["content_sha256"] for item in plan["sources"]]
    for offset in range(0,len(group_order),BATCH_SIZE):
        excluded=set(group_order[offset:offset+BATCH_SIZE])
        row={"block":offset//BATCH_SIZE+1}
        for kind,groups in (("raster",raster_groups),("polygon",polygon_groups)):
            kept={name:values for name,values in groups.items() if name not in excluded}
            stat=group_summary(kept,lambda row:True)
            row[kind]=dict(groups=stat["groups"],iou_gain=stat["iou"]["favorable_delta"] if stat["iou"] else None,
                           y_abs_gain=stat["y_absolute"]["favorable_delta"] if stat["y_absolute"] else None)
        blocks.append(row)
    core=raster_result["all"]
    y=core["y_absolute"]
    passed=bool(y and y["groups"]>=16 and y["favorable_delta"]>=.1 and y["interval95"][0]>0
                and core["iou"]["favorable_delta"]>=-.002 and core["retain"]["favorable_delta"]>=-.01)
    result=dict(status="PSEG_SSDD_MASK_RTDETR_FIT_STOCK_PAIRED_GEOMETRY_COMPLETE_V1",
                protocol_commit=args.protocol_commit,plan_sha256=PLAN_SHA,upstream_commit=UPSTREAM_COMMIT,
                upstream_sources=SOURCE_SHA256,runtime=runtime,geos_version=shapely.geos_version_string,
                fit_archive_sha256=plan["fit_archive_sha256"],fit_index_sha256=plan["fit_index_sha256"],
                source_groups=SOURCES,group_kind="distinct original JPEG byte groups",views_per_source=REPEATS,
                source_census=dict(census),parser_exclusions=dict(dropped),raster_reference=raster_result,
                continuous_integer_polygon_reference=polygon_result,leave_one_fixed_batch_block=blocks,
                raster_geometry_gate_passed=passed,elapsed_seconds=perf_counter()-start,
                author_geometry_executed=True,full_paddle_loader_executed=False,
                model_executed=False,optimizer_updates=0,AP_evaluated=False,
                official_DEV_opened=False,official_TEST_opened=False)
    raw=canonical(result)
    with args.output.open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(event=result["status"],sha256=sha256(raw).hexdigest(),passed=passed),sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("paddle-code","vector-code","polygon-code","upstream","bundle","plan","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--protocol-commit",required=True)
    run(p.parse_args())

