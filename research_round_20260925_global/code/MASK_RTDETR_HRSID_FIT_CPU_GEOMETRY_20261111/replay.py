"""Run the prospective FIT-only PaddleDetection stock-operator comparison."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scope import (BATCH_SIZE, BOOTSTRAP_SEED, DRAWS, LIMIT_IOU, LIMIT_RETENTION, LIMIT_Y_PX,
                   MIN_GROUPS, REPEATS, SOURCES, canonical, choose, make_plan, read_catalog,
                   read_pinned, zip_digest)
from footprint import exact_footprint_scores

import cv2
import numpy as np
from PIL import Image
from paddle_geometry import SOURCE_SHA256, UPSTREAM_COMMIT, load_stock, transform_batch

LOADER_SHA = "9508c2b1e852b0d0fb35c81e58399e162ae9d5fdda04b9c2f0d37b9cc4ab5240"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stock_parser(encoded, annotations):
    boxes, polygons, crowds = [], [], []
    excluded = Counter()
    for row in annotations:
        if row.get("ignore", False) or row.get("iscrowd", 0):
            excluded["ignored_or_crowd"] += 1
            continue
        if "bbox" not in row or not any(row["bbox"]):
            excluded["absent_box"] += 1
            continue
        x, y, w, h = row["bbox"]
        if not (row["area"] > 0 and w > 1e-5 and h > 1e-5):
            excluded["invalid_box"] += 1
            continue
        require(row["category_id"] == 1 and all(math.isfinite(v) for v in (x, y, w, h)),
                "Expected one finite ship-category source row")
        polygon = row["segmentation"]
        require(isinstance(polygon, (list, dict)) and bool(polygon), "Expected released nonempty segmentation")
        boxes.append([round(a, 3) for a in (x, y, x+w, y+h)])
        polygons.append(polygon)
        crowds.append(0)
    n = len(boxes)
    return dict(image=encoded, h=800, w=800, gt_bbox=np.asarray(boxes,dtype=np.float32).reshape(n,4),
                gt_class=np.zeros((n,1),dtype=np.int32), is_crowd=np.asarray(crowds,dtype=np.int32).reshape(n,1),
                gt_poly=polygons), excluded


def close_enough_dict(a, b, omit):
    require(a.keys() - omit == b.keys() - omit, "Witness changed stock field names")
    for k in a.keys() - omit:
        if isinstance(a[k], np.ndarray):
            require(a[k].dtype == b[k].dtype and np.array_equal(a[k],b[k]), "Witness changed stock field " + k)
        else:
            require(a[k] == b[k], "Witness changed stock field " + k)


def view_rows(tagged, stock, source_rows, group, census, records):
    for key in ("source_hw", "expanded", "cropped", "flipped", "image_interpolation"):
        require(getattr(tagged,key) == getattr(stock,key), "Witness changed draw " + key)
    close_enough_dict(tagged.before_resize,stock.before_resize,{"gt_read_order"})
    close_enough_dict(tagged.output,stock.output,{"gt_read_order"})
    require(np.array_equal(tagged.center_masks,stock.center_masks), "Witness changed centered masks")
    require(tagged.source_hw == (800,800), "Expected verified native raster size")
    a, b = tagged.output["gt_segm"], tagged.center_masks
    pre = tagged.before_resize["gt_segm"]
    order = tagged.before_resize["gt_read_order"].reshape(-1)
    output_order = tagged.output["gt_read_order"].reshape(-1)
    require(len(order) == len(set(order)) and all(0 <= i < source_rows for i in order)
            and np.array_equal(order,output_order) and len(order) == len(pre) == len(a) == len(b)
            and len(a) == len(tagged.output["gt_bbox"]) == len(tagged.output["gt_class"]),
            "Common retained box, class, and mask row identities required")
    census["views"] += 1
    census["cropped_views"] += tagged.cropped
    census["expanded_views"] += tagged.expanded
    census["flipped_views"] += tagged.flipped
    census["nearest_image_views"] += tagged.image_interpolation == cv2.INTER_NEAREST
    census["rows_before_augment"] += source_rows
    census["rows_after_integer_ops"] += len(pre)
    if not len(pre):
        census["zero_rows_views"] += 1
        return
    out_hw = tuple(a.shape[-2:])
    in_hw = tuple(pre.shape[-2:])
    resize = "down" if out_hw[0] <= in_hw[0] and out_hw[1] <= in_hw[1] else "up" if out_hw[0] >= in_hw[0] and out_hw[1] >= in_hw[1] else "mixed"
    if out_hw == in_hw:
        resize = "identity"
        require(np.array_equal(a,b) and np.array_equal(a,pre), "Identity resize changes mask")
    for ref, left, right in zip(pre,a,b,strict=True):
        legacy = cv2.resize(ref, (out_hw[1],out_hw[0]), interpolation=cv2.INTER_NEAREST)
        require(np.array_equal(left,legacy), "Independent OpenCV witness disagrees with stock mask")
        census["native_masks_witnessed"] += 1
        census["masks_differ"] += not np.array_equal(left,right)
        ref_count = int(np.count_nonzero(ref))
        if not ref_count:
            require(not left.any() and not right.any(), "Empty support cannot generate foreground")
            census["empty_before_resize"] += 1
            continue
        if ref_count*out_hw[0]*out_hw[1] >= 1024*in_hw[0]*in_hw[1]:
            census["large_reference_rows"] += 1
            continue
        scores = exact_footprint_scores(ref,(left,right))
        records[group].append(dict(scores, crop=tagged.cropped,
                                   nearest_image=tagged.image_interpolation==cv2.INTER_NEAREST, resize=resize))


def group_summary(grouped, keep):
    per_group = []
    support = Counter()
    for rows in grouped.values():
        rows = [row for row in rows if keep(row)]
        if not rows:
            continue
        pairs = [(row["native"],row["center"]) for row in rows]
        survived = [tuple(arm["signed_y_px"] is not None for arm in pair) for pair in pairs]
        labels = {(True, True):"both", (True, False):"native_only",
                  (False, True):"center_only", (False, False):"neither"}
        support.update(labels[item] for item in survived)
        common = [pair for pair, valid in zip(pairs,survived,strict=True) if all(valid)]
        data = dict(
            iou_native=float(np.mean([a["iou"] for a,_ in pairs])),
            iou_center=float(np.mean([b["iou"] for _,b in pairs])),
            retain_native=float(np.mean([a for a,_ in survived])),
            retain_center=float(np.mean([b for _,b in survived])),
        )
        if common:
            for axis in "yx":
                key = f"signed_{axis}_px"
                for index, name in enumerate(("native","center")):
                    values = np.asarray([p[index][key] for p in common])
                    data[f"{axis}_signed_{name}"] = float(values.mean())
                    data[f"{axis}_absolute_{name}"] = float(np.abs(values).mean())
        per_group.append(data)
    output = dict(groups=len(per_group), support=dict(support))
    for name in ("iou","retain","y_signed","y_absolute","x_signed","x_absolute"):
        pairs = [(row[name+"_native"], row[name+"_center"]) for row in per_group if name+"_native" in row]
        if not pairs:
            output[name] = None
            continue
        array = np.asarray(pairs)
        delta = array[:,1]-array[:,0]
        if name in ("y_absolute","x_absolute"):
            delta *= -1.
        rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        idx = rng.integers(0,len(pairs),size=(DRAWS,len(pairs)))
        q = np.quantile(delta[idx].mean(1),(.025,.975),method="linear")
        output[name] = dict(groups=len(pairs), native=float(array[:,0].mean()), center=float(array[:,1].mean()),
                            favorable_delta=float(delta.mean()), interval95=[float(v) for v in q],
                            positive_groups=int((delta>0).sum()))
    return output


def deny_network(event, args):
    if event.startswith("socket."):
        raise RuntimeError("Offline FIT geometry required")


def run(args):
    start = perf_counter()
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and os.environ.get("OMP_NUM_THREADS")=="1",
            "CPU-only one-thread replay required")
    require((cv2.__version__, version("opencv-python"), version("pycocotools"), np.__version__)
            == ("4.5.5","4.5.5.64","2.0.8","1.26.4"), "Exact predeclared numeric runtime required")
    cv2.setNumThreads(1)
    sys.addaudithook(deny_network)
    zip_digest(args.bundle)
    plan_raw = args.plan.read_bytes()
    require(len(plan_raw) <= 10000 and sha256(plan_raw).hexdigest() == args.plan_sha256, "Exact prospective cohort lock")
    import paddle_geometry
    require(sha256(Path(paddle_geometry.__file__).read_bytes()).hexdigest() == LOADER_SHA,
            "Exact stock geometry loader source")
    operators = load_stock(args.upstream)
    excluded = Counter()
    records = defaultdict(list)
    census = Counter()
    with ZipFile(args.bundle) as bundle:
        rows,coco = read_catalog(bundle)
        require(plan_raw==canonical(make_plan(rows)), "Exact locked FIT scope and schedule")
        plan=json.loads(plan_raw)
        selected=choose(rows)
        anns=defaultdict(list)
        for annotation in coco["annotations"]:
            anns[annotation["image_id"]].append(annotation)
        samples=[]
        for row in selected:
            raw=read_pinned(bundle,"images/"+row["native_basename"],row["encoded_sha256"],4*2**20)
            with Image.open(BytesIO(raw)) as image:
                rgb=np.asarray(image.convert("RGB"))
                require(image.format=="PNG" and rgb.shape==(800,800,3) and len(raw)==row["encoded_bytes"]
                        and sha256(rgb.tobytes()).hexdigest()==row["decoded_rgb_sha256"],
                        "Exact sealed input image")
            sample, drops=stock_parser(raw,anns[row["image_id"]])
            excluded.update(drops)
            samples.append(sample)
    for cursor, seed in enumerate(plan["seeds"]):
        i=(cursor%(SOURCES//BATCH_SIZE))*BATCH_SIZE
        batch=samples[i:i+BATCH_SIZE]
        with_witness=[dict(sample,gt_read_order=np.arange(len(sample["gt_bbox"]),dtype=np.int32)[:,None])
                      for sample in batch]
        base=transform_batch(operators,batch,seed=seed)
        tagged=transform_batch(operators,with_witness,seed=seed)
        for j,(actual,control) in enumerate(zip(tagged,base,strict=True)):
            view_rows(actual,control,len(batch[j]["gt_bbox"]),selected[i+j]["group"],census,records)
        print(json.dumps(dict(event="PADDLE_FIT_OPERATOR_BATCH_DONE",batches=cursor+1,views=census["views"])),flush=True)
    require(census["views"]==SOURCES*REPEATS,"Exact prospective views")
    limits = {
        "all":lambda r:True, "cropped":lambda r:r["crop"],"uncropped":lambda r:not r["crop"],
        "nearest_image":lambda r:r["nearest_image"], "other_image":lambda r:not r["nearest_image"],
        "down":lambda r:r["resize"]=="down", "up":lambda r:r["resize"]=="up",
        "mixed":lambda r:r["resize"]=="mixed", "identity":lambda r:r["resize"]=="identity",
    }
    stats={key:group_summary(records,predicate) for key,predicate in limits.items()}
    primary=stats["all"]
    y=primary["y_absolute"]
    passed=bool(y is not None and y["groups"]>=MIN_GROUPS and y["favorable_delta"]>=LIMIT_Y_PX
                and y["interval95"][0]>0 and primary["iou"]["favorable_delta"]>=LIMIT_IOU
                and primary["retain"]["favorable_delta"]>=LIMIT_RETENTION)
    report=dict(status="MASK_RTDETR_HRSID_FIT_STOCK_CPU_GEOMETRY_COMPLETE_V1", upstream_commit=UPSTREAM_COMMIT,
                source_sha256=SOURCE_SHA256, loader_sha256=LOADER_SHA,
                runtime={key:version(key) for key in ("numpy","opencv-python","pycocotools","pillow","pyyaml")},
                elapsed_seconds=perf_counter()-start, protocol_commit=args.protocol_commit,
                plan_sha256=args.plan_sha256, sources=SOURCES, views_per_source=REPEATS, batches=len(plan["seeds"]),
                image_box_row_and_rng_witness=True, sampler_reference="pre_resize_raster_pixel_cell_footprint",
                report_group_kind="HRSID filename prefix proxies", counts=dict(census), source_parser_exclusions=dict(excluded),
                small_source_group_equal_stats=stats, gate_passed=passed,
                paddle_training_loader_executed=False, model_executed=False, optimizer_updates=0, AP_evaluated=False,
                official_DEV_opened=False, official_TEST_opened=False)
    payload=canonical(report)
    with args.output.open("xb") as f:
        f.write(payload)
    print(json.dumps(dict(event=report["status"],sha256=sha256(payload).hexdigest(),passed=passed),sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("bundle","plan","upstream","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--protocol-commit",required=True)
    p.add_argument("--plan-sha256",required=True)
    run(p.parse_args())

