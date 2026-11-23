"""COCO rematching on original raster or continuous author-polygon overlaps."""

from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


METRICS = ("AP","AP50","AP75","APs","APm","APl","AR1","AR10","AR100","ARs","ARm","ARl")
PINS = {
    "geometry": ("HRSID_DEV705_CONTINUOUS_AP_20261111/full_continuous_ap.py",
                 "cba94ecd0df862b6a303117a56fa889bc76c0a8c017af50fff2cdb9859f94139"),
    "vector": ("HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110/vector_label_geometry.py",
               "4877c48d38a03e58e97163381ba544df3c90a78dde85c9141085fa0b92da3b09"),
    "shift": ("HRSID_COMMON_DEV705_FIXED_SHIFT_AP_20261111/fixed_shift_ap.py",
              "68a7ec9e0811cc5b18d2b97a09c13b2471f3908a36b39c3dedd4a50b098f97f2"),
    "footprint": ("HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110/vector_population_math.py",
                  "0821d2cc3b6ca793d33fdfe35b634f3678aad072fc55b86352fa2501e969fc7e"),
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def prepare(code, view_gt):
    """Prepare eligible author shapes and caches shared across all arms of this one view."""
    helpers = {}
    for name,(rel,expected) in PINS.items():
        path = code / rel
        data = path.read_bytes()
        require(sha256(data).hexdigest() == expected, "Historical evaluator source differs: "+path.name)
        helper = ModuleType(name)
        helper.__file__ = str(path)
        exec(compile(data,str(path),"exec"),helper.__dict__)
        helpers[name] = helper
    gt = deepcopy(view_gt)
    require(all(type(a["id"]) is int and a["id"] > 0 and a["category_id"] == 1 for a in gt["annotations"])
            and len({a["id"] for a in gt["annotations"]}) == len(gt["annotations"]), "Expected unique positive ship IDs")
    shapes, neutral, census = helpers["geometry"].polygon_support(gt, helpers["vector"])
    support = dict(annotations=len(gt["annotations"]),supported=len(shapes),neutral=len(neutral),**census)
    return SimpleNamespace(gt=gt,shapes=shapes,neutral=neutral,support=support,rectangles={},overlaps={},
                           dense_checks=0,pruned_checks=0,max_error=0.,**helpers)


def key_for(rle):
    counts = rle["counts"]
    text = counts.decode("ascii") if isinstance(counts,bytes) else counts
    return (*rle["size"],text)


def exact_iou(ctx, ann_id, rle, dt_area, audit):
    key = key_for(rle)
    pair = (ann_id,key)
    if pair in ctx.overlaps:
        audit["cached_intersections"] += 1
        return ctx.overlaps[pair]
    shape, rings = ctx.shapes[ann_id]
    if key not in ctx.rectangles:
        h,w,counts = key
        boxes = ctx.geometry.boxes_from_counts(ctx.shift.runs(counts),h,w)
        area = np.sum((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]),dtype=np.float64)
        require(area == dt_area, "Native run rectangle area differs from COCO mask area")
        ctx.rectangles[key] = boxes
        audit["new_run_areas_verified"] += 1
    intersection = ctx.geometry.overlap_area(ctx.rectangles[key],shape)
    require(0 <= intersection <= min(dt_area,shape.area)+1.e-7, "Continuous area outside mask or polygon")
    iou = float(intersection / (dt_area + shape.area - intersection))
    if ctx.dense_checks < 16:
        dense = mask.decode(rle).astype(bool)
        check = ctx.footprint.run_footprint_metrics(rings,dense.shape,{"dt":dense})["by_arm"]["dt"]["iou"]
        error = abs(check-iou)
        require(error < 1.e-10, "Independent horizontal runs and column-major native runs differ")
        ctx.max_error = max(ctx.max_error,error)
        ctx.dense_checks += 1
        audit["dense_independent_checks"] += 1
    ctx.overlaps[pair] = iou
    audit["new_exact_intersections"] += 1
    return iou


def replace_iou(ctx, ev, raster, image_id, category_id, audit):
    if not len(raster):
        return raster
    gts, dts = ev._gts[image_id,category_id], ev._dts[image_id,category_id]
    order = np.argsort([-r["score"] for r in dts],kind="mergesort")[:ev.params.maxDets[-1]]
    require(raster.shape == (len(order),len(gts)), "Native IoU matrix order differs")
    rles = [dts[i]["segmentation"] for i in order]
    areas = np.asarray(mask.area(rles),dtype=np.float64)
    boxes = np.asarray(mask.toBbox(rles),dtype=np.float64)
    boxes[:,2:] += boxes[:,:2]
    for j,gt in enumerate(gts):
        ann_id = gt["id"]
        if ann_id in ctx.neutral:
            audit["neutral_raster_pairs"] += len(rles)
            continue
        shape,rings = ctx.shapes[ann_id]
        bounds = np.asarray(shape.bounds)
        possible = ctx.geometry.plausible(boxes,areas,bounds,shape.area)
        audit["supported_pairs"] += len(rles)
        audit["provably_below_half_pairs"] += int(np.count_nonzero(~possible))
        raster[:,j] = 0.
        for i in np.flatnonzero(possible):
            raster[i,j] = exact_iou(ctx,ann_id,rles[i],areas[i],audit)
        if ctx.pruned_checks < 8:
            upper = ctx.geometry.bound_area(boxes,areas,bounds,shape.area)
            for i in np.flatnonzero((~possible)&(upper>0))[:8-ctx.pruned_checks]:
                dense = mask.decode(rles[i]).astype(bool)
                ref = ctx.footprint.run_footprint_metrics(rings,dense.shape,{"dt":dense})["by_arm"]["dt"]["iou"]
                require(ref < .5, "Conservative upper bound would remove a qualifying detection")
                ctx.pruned_checks += 1
                audit["pruned_independent_checks"] += 1
    return raster


def events90(ev, lookup, i90):
    """Return comparable native identities from the official all-area single-category match records."""
    ids = ev.params.imgIds
    events = {int(i):dict(positive_gt=0,detections=[]) for i in ids}
    seen = set()
    for e in ev.evalImgs:
        if e is None or e["maxDet"] != 100 or e["aRng"] != ev.params.areaRng[0]:
            continue
        image_id = int(e["image_id"])
        require(image_id not in seen, "Duplicate all-area image record")
        seen.add(image_id)
        g = events[image_id]
        positive = {a for a,ig in zip(e["gtIds"],e["gtIgnore"],strict=True) if not ig}
        g["positive_gt"] = len(positive)
        for row_id,score,ignored,matched in zip(e["dtIds"],e["dtScores"],e["dtIgnore"][i90],
                                                 e["dtMatches"][i90],strict=True):
            i,pos,stock_score = lookup[int(row_id)-1]
            require(i == image_id and score == stock_score and
                    (ignored or not matched or int(matched) in positive), "COCO event identity differs")
            g["detections"].append((pos,float(score),bool(ignored),int(matched) if matched and not ignored else 0))
    return events


def event_ap90(events,recall_thresholds):
    n = sum(row["positive_gt"] for row in events.values())
    if not n:
        return None
    records = [r for row in events.values() for r in row["detections"]]
    records.sort(key=lambda r:-r[1])  # Python's sort preserves official image/ordinal ties.
    tp = np.cumsum([not r[2] and bool(r[3]) for r in records],dtype=np.float64)
    fp = np.cumsum([not r[2] and not r[3] for r in records],dtype=np.float64)
    precision = tp / (tp+fp+np.spacing(1))
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(tp/n,recall_thresholds,side="left")
    return float(np.r_[envelope,0.][idx].mean())


def evaluate(ctx,rows,lookup,mode):
    audit = Counter()

    class Evaluator(COCOeval):
        def _prepare(self):
            super()._prepare()
            if mode != "original_raster" and ctx.neutral:
                for entries in self._gts.values():
                    for ann in entries:
                        if ann["id"] in ctx.neutral:
                            ann["ignore"] = 1

        def computeIoU(self,image_id,category_id):
            raster = super().computeIoU(image_id,category_id)
            return replace_iou(ctx,self,raster,image_id,category_id,audit) if mode == "continuous" else raster

    with redirect_stdout(StringIO()):
        ref = COCO()
        ref.dataset = deepcopy(ctx.gt)
        ref.createIndex()
        if rows:
            dt = ref.loadRes(deepcopy(rows))
        else:
            dt = COCO()
            dt.dataset = dict(deepcopy(ctx.gt),annotations=[])
            dt.createIndex()
        ev = Evaluator(ref,dt,"segm")
        ev.params.catIds = [1]
        ev.params.imgIds = [r["id"] for r in ctx.gt["images"]]
        ev.params.maxDets = [1,10,100]
        require(np.isclose(ev.params.iouThrs[0],.50) and len(ev.params.iouThrs)==10,
                "Pruning bound requires native .50:.95 thresholds")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    i90, = np.flatnonzero(np.isclose(ev.params.iouThrs,.9))
    p = ev.eval["precision"][i90,:,0,0,-1]
    scores = dict(zip(METRICS,(float(v) if np.isfinite(v) and v>=0 else None for v in ev.stats),strict=True))
    scores["AP90"] = float(p[p>=0].mean()) if (p>=0).any() else None
    events = events90(ev,lookup,i90)
    recovered = event_ap90(events,ev.params.recThrs)
    require((recovered is None and scores["AP90"] is None) or
            (recovered is not None and scores["AP90"] is not None and abs(recovered-scores["AP90"]) <= 1.e-12),
            "Matched .90 event reconstruction differs from the official precision tensor")
    return scores,events,dict(audit)


def score_arm(ctx,scoped_blocks,arm,expected_raster=None):
    """Score one arm. Events are private, indexed by mode then numerical image ID.

    Each all-area/maxDet=100 event tuple is (compact_position, score, ignored,
    matched_positive_GT_id_or_zero). Original source segmentation and score order are retained.
    """
    require(set(scoped_blocks) == {im["id"] for im in ctx.gt["images"]},"Scoped COCO images differ")
    rows,lookup = [],[]
    for image_id,block in scoped_blocks.items():
        for pos,(rank,source) in enumerate(zip(block["ranks"],block["stock"],strict=True),1):
            require(source["image_id"] == image_id and source["category_id"] == 1,
                    "Candidate does not belong to its original image/category")
            update = block["changes"].get(rank,{}).get(arm)
            rows.append(source if update is None else dict(source,segmentation=update["rle"]))
            lookup.append((image_id,pos,source["score"]))
    original,old_events,_ = evaluate(ctx,rows,lookup,"original_raster")
    if expected_raster is not None:
        require(set(original) == set(expected_raster), "Exactly thirteen original COCO metrics required")
        for k,v in expected_raster.items():
            expected = v if v is not None and np.isfinite(v) and v >= 0 else None
            got = original[k]
            require((got is None and expected is None) or
                    (got is not None and expected is not None and abs(got-expected) <= 1.e-12),
                    "Original archived raster score differs: " + arm + "/" + k)
    if ctx.neutral:
        supported,subset_events,_ = evaluate(ctx,rows,lookup,"supported_raster")
    else:
        supported,subset_events = original,old_events
    continuous,new_events,audit = evaluate(ctx,rows,lookup,"continuous")
    for i in old_events:
        ids = [row[:2] for row in old_events[i]["detections"]]
        require(ids == [row[:2] for row in subset_events[i]["detections"]]
                == [row[:2] for row in new_events[i]["detections"]] and
                subset_events[i]["positive_gt"] == new_events[i]["positive_gt"],
                "Native score order, stable ties or paired support differ")
    audit.update(shared_cache_entries=len(ctx.overlaps),shared_mask_geometries=len(ctx.rectangles),
                 dense_checks_so_far=ctx.dense_checks,pruned_checks_so_far=ctx.pruned_checks,
                 max_independent_iou_error=ctx.max_error)
    return dict(original_raster=original,supported_raster=supported,continuous=continuous,
                audit=audit,events=dict(supported_raster=subset_events,continuous=new_events))
