"""Full fixed FIT741 SRNet ground-truth crop/paste geometry, without model inference."""
import argparse
from collections import defaultdict, deque
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

P338_SHA="2809a86664716dc8193184534eccb986ee8763e7fe4166f39dea0fe324969cad"
P341_SHA="adb7b489050c8b7f4c206ead9e007b77d6d010dfb3499b41636ea6ceb7e7a4a1"
P341_REPORT_SHA="5f877128215af689eaf61c74704c24db95a6c311e36060719c98e2c5c3c832cc"
ARMS=("interior_full","center_full","interior_contracted","center_contracted")
COMPARISONS=(("center_full","interior_full"),("interior_contracted","interior_full"),
             ("center_full","interior_contracted"),("center_full","center_contracted"),
             ("center_contracted","interior_contracted"),("interior_full","center_contracted"))
STATUS="P342_SRNET_FIT741_CROP_PASTE_FACTORIAL_ORACLE_COMPLETE_V1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(rows):
    """Per-target means preserve undefined-IoU rows and paired denominators."""
    result=dict(targets=len(rows),empty_truth=sum(row["truth_area"]==0 for row in rows),
                arms={},paired_differences={})
    for arm in ARMS:
        metrics=[row["metrics"][arm] for row in rows]
        valid=[item for item in metrics if item["iou"] is not None]
        result["arms"][arm]=dict(
            valid_iou=len(valid),undefined_iou=len(metrics)-len(valid),
            empty_prediction=sum(item["intersection"]+item["fp"]==0 for item in metrics),
            mean_iou=sum(item["iou"] for item in valid)/len(valid) if valid else None,
            at_75=sum(item["strict_075"] for item in metrics),
            at_90=sum(item["strict_090"] for item in metrics),
            intersection_pixels=sum(item["intersection"] for item in metrics),
            false_positive_pixels=sum(item["fp"] for item in metrics),
            false_negative_pixels=sum(item["fn"] for item in metrics))
    for first,second in COMPARISONS:
        pairs=[(row["metrics"][first],row["metrics"][second]) for row in rows]
        differences=[a["iou"]-b["iou"] for a,b in pairs if a["iou"] is not None and b["iou"] is not None]
        result["paired_differences"][first+"_minus_"+second]=dict(
            valid_pairs=len(differences),undefined_pairs=len(pairs)-len(differences),
            mean=sum(differences)/len(differences) if differences else None,
            wins=sum(x>0 for x in differences),ties=sum(x==0 for x in differences),
            losses=sum(x<0 for x in differences))
    return result


def group_names(area, box, boundary, outside_roi, empty_truth):
    size="small" if area<1024 else "mid" if area<9216 else "large"
    aspect="thin" if max(box[2:4])>=3*min(box[2:4]) else "other_aspect"
    return ("all","size_"+size,aspect,"size_"+size+"_"+aspect,
            "mask_touches_canvas" if boundary else "mask_inside_canvas",
            "roi_extends_off_canvas" if outside_roi else "roi_inside_canvas",
            "empty_truth" if empty_truth else "nonempty_truth")


def prior_replay(rows, prior, maximum_error):
    observed=summarize(rows)
    arms={name:{key:observed["arms"][name][key] for key in ("mean_iou","at_75","at_90")} for name in ARMS}
    comparisons={name:{key:observed["paired_differences"][name][key] for key in ("mean","wins","ties","losses")}
                 for name in prior["paired_differences"]}
    require(arms==prior["arms"] and comparisons==prior["paired_differences"]
            and maximum_error==prior["max_abs_coordinate_error"],
            "Prior 20-target four-cell results must reproduce exactly in frozen image and instance order")
    return True


def load_pinned(path, digest, name):
    require(path.is_file() and 0<path.stat().st_size<=1024**2,"Bounded ordinary source required")
    raw=path.read_bytes()
    require(sha256(raw).hexdigest()==digest,"Exact ordinary source digest required")
    module=ModuleType(name)
    module.__file__=str(path)
    exec(compile(raw,str(path),"exec"),module.__dict__)
    return module


def run(a):
    started=perf_counter()
    p338=load_pinned(a.p338_source,P338_SHA,"_p342_p338")
    p341=load_pinned(a.p341_source,P341_SHA,"_p342_p341")
    prior=json.loads(p341.exact(a.p341_report,P341_REPORT_SHA))
    require(prior["status"]==p341.STATUS and prior["images"]==8 and prior["targets"]==20
            and prior["grid_size"]==28 and prior["prior_stock_paste_full_replay_exact"] is True,
            "Exact fixed-cohort parent factorial required")
    p335,base,admission=p338.verified_dataset(a)
    import numpy as np
    import torch
    from mmrotate.core import obb2poly
    from mmrotate.models.dense_heads import SRNet_utils as stock
    from pycocotools import mask as codec

    require(torch.get_num_threads()==1 and not torch.cuda.is_available() and not torch.cuda.is_initialized(),
            "Pinned CPU runtime and single intra-op thread required")
    require(Path(stock.__file__).resolve()==a.source/"mmrotate/models/dense_heads/SRNet_utils.py",
            "Stock SRNet geometry source required")
    raw=json.loads(p335.exact(a.fit/"_annotations.coco.json",p335.FIT_INDEX_SHA,2*1024**2))
    source_rows=defaultdict(lambda:defaultdict(deque))
    images={image["id"]:image for image in raw["images"]}
    for annotation in raw["annotations"]:
        source_rows[annotation["image_id"]][p335.key(annotation["segmentation"])].append(annotation)
    require(len(images)==len(base.data_infos)==741 and len(raw["annotations"])==1661,
            "Fixed full FIT population required")
    records=[]
    by_image=defaultdict(list)
    groups=defaultdict(list)
    max_coordinate_error=0.
    max_canvas_side=0
    max_author_coordinate_error=0.
    all_boxes=[]
    with torch.no_grad():
        for index,image in enumerate(base.data_infos):
            image_id=image["id"]
            original=images[image_id]
            height,width=original["height"],original["width"]
            max_canvas_side=max(max_canvas_side,height,width)
            parsed=base.get_ann_info(index)
            require(len(parsed["bboxes"])==len(parsed["masks"]),"Exact stock parsed instance order required")
            for box,polygon in zip(parsed["bboxes"],parsed["masks"]):
                candidates=source_rows[image_id][p335.key(polygon)]
                require(bool(candidates) and all(row["area"]==candidates[0]["area"] for row in candidates),
                        "Original polygon identity and area must be unambiguous")
                annotation=candidates.popleft()
                truth=p335.bitmap(polygon,height,width,codec)
                truth_area=int(np.count_nonzero(truth))
                boundary=bool(truth[0].any() or truth[-1].any() or truth[:,0].any() or truth[:,-1].any())
                corners=obb2poly(torch.as_tensor(box[None],dtype=torch.float32),"le90")[0].reshape(4,2).numpy()
                outside_roi=bool((corners[:,0]<0).any() or (corners[:,0]>width).any()
                                 or (corners[:,1]<0).any() or (corners[:,1]>height).any())
                metrics=p341.four_paths(box,truth,stock,torch,p338.oracle_metric)
                row=dict(metrics={name:metrics[name] for name in ARMS},truth_area=truth_area,
                         coordinate_error=metrics["matched_center_coordinate_abs_error"])
                records.append(row)
                by_image[image_id].append(row)
                all_boxes.append(box)
                max_coordinate_error=max(max_coordinate_error,row["coordinate_error"])
                for name in group_names(annotation["area"],box,boundary,outside_roi,truth_area==0):
                    groups[name].append(row)
            if (index+1)%100==0:
                print(json.dumps(dict(status="P342_FIT_ORACLE_PROGRESS",images=index+1,targets=len(records),
                                      elapsed_seconds=perf_counter()-started)),flush=True)
        require(len(records)==admission["author_parsed_targets"]==1661
                and not any(rows for source in source_rows.values() for rows in source.values()),
                "Every original FIT annotation must be consumed exactly once")
        _,analytic_errors=p338.stock_geometry(np.asarray(all_boxes,dtype=np.float32),28,stock,torch)
        max_author_coordinate_error=max(analytic_errors.values())
    fixed=p335.choose(raw["images"])
    prior_rows=[row for image in fixed for row in by_image[image["id"]]]
    require(len(prior_rows)==20,"Exact original eight-image cohort required")
    regression=prior_replay(prior_rows,prior,max(row["coordinate_error"] for row in prior_rows))
    require(not torch.cuda.is_initialized(),"CPU scope must remain closed to CUDA")
    summaries={name:summarize(rows) for name,rows in groups.items()}
    for name in ("size_small","size_mid","size_large","thin","other_aspect","mask_touches_canvas",
                 "mask_inside_canvas","roi_extends_off_canvas","roi_inside_canvas","empty_truth","nonempty_truth"):
        summaries.setdefault(name,summarize([]))
    total=summaries["all"]
    source_count=sum(summaries["size_"+name]["targets"] for name in ("small","mid","large"))
    require(source_count==total["targets"]==1661
            and summaries["thin"]["targets"]+summaries["other_aspect"]["targets"]==1661,
            "Declared full denominator must reconcile")
    return dict(status=STATUS,fit=True,dev=False,test=False,ap=False,model_forward_calls=0,optimizer_updates=0,
                source_commit=p335.AUTHOR_COMMIT,author_pins=p338.AUTHOR_PINS,
                p338_source_sha256=P338_SHA,p341_source_sha256=P341_SHA,p341_result_sha256=P341_REPORT_SHA,
                p335_source_sha256=p338.P335_SOURCE_SHA,p335_result_sha256=p338.P335_RESULT_SHA,
                p335_original_terminal_sha256=p338.P335_TERMINAL_SHA,fit_archive_sha256=p335.FIT_SHA,
                fit_index_sha256=p335.FIT_INDEX_SHA,package_versions=admission["package_versions"],
                original_images=741,original_annotations=1661,images=len(base.data_infos),targets=len(records),
                grid_size=28,matched_inverse_scale=28/29,max_abs_coordinate_error=max_coordinate_error,
                independent_analytic_max_abs_coordinate_error=max_author_coordinate_error,
                prior_fixed_8_images_20_targets_regression_exact=regression,
                prior_cohort=summarize(prior_rows),groups=summaries,
                max_square_canvas_side=max_canvas_side,intraop_threads=1,
                definitions=dict(
                    sizes="original COCO area <1024; [1024,9216); >=9216",
                    thin="original author le90 max(width,height)/min(width,height)>=3",
                    mask_boundary="at least one foreground pixel on first/last source image row or column",
                    roi_boundary="at least one original le90 vertex outside continuous [0,width]x[0,height]",
                    mean_iou="equal target weight among defined union>0; undefined and source/predicted empties retained",
                    thresholds="IoU>=.75 / >=.90 by exact integer cross-product",
                    comparisons="same original target and same bitmap truth in all four crop/paste cells",
                    oracle="GT bitmap binary node samples at >=.5; stock bilinear paste then >.5; GT original boxes",
                    inference="no statistical confidence or learned AP inferred from target-level descriptive means"),
                elapsed_seconds=perf_counter()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=("p338-source","p341-source","p341-report","p335-source","p335-result","p335-terminal","source","fit","archive")
    for name in (*inputs,"output"):
        parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args()
    require(all(path.is_absolute() and path.resolve(strict=True)==path
                for path in (getattr(a,name.replace("-","_")) for name in inputs))
            and a.output.is_absolute() and a.output.parent.resolve(strict=True)==a.output.parent
            and not a.output.exists() and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
            and sys.flags.isolated and sys.dont_write_bytecode,
            "Canonical exact inputs, fresh output and isolated CPU runtime required")

    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("Offline full-FIT geometry")

    sys.addaudithook(offline)
    data=(json.dumps(run(a),sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps(dict(status=STATUS,result_sha256=sha256(data).hexdigest())),flush=True)


if __name__=="__main__":
    main()
