"""Replay a nonnegative signed area cap on completed SegRefiner-LR masks."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
from scipy.ndimage import distance_transform_edt
import shapely

SEEDS = (5217, 5231, 5279)
UNIFORMS = tuple("cap_uniform_" + str(n) for n in SEEDS)
CAP = "cap_shallow"
RAW, STOCK, ADD, MEAN = "raw", "stock", "add_only", "cap_uniform_mean"
ARMS = (RAW, STOCK, ADD, *UNIFORMS, CAP)
PAIRS = ((MEAN, STOCK), (CAP, MEAN), (CAP, STOCK), (CAP, RAW), (CAP, ADD), (ADD, RAW))
STATUS = "SEGREFINER_DELETION_CAP_SCORE_COMPLETE"
RSDD_SIGN = "cdd7b3f7e1bbbeb528fb5100da76daee54b2a4f61f88e2381ef2b709e48a080e"
RSDD_RESULT = "02f2a09fd50d4cc3112ba43cc9b3ed76badff6720f386fb45143665d5dfb7fc5"
PSEG_SIGN = "a658416573b591d657e49e7deb5db1aaeb754aeb471d567c4077fcad2188d16c"
PSEG_RESULT = "a2570a140ace0104a81c91c148a1ce6f2f88d912a62b5906cbacbf162de6689f"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path, digest):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, "Unexpected signed scorer")
    module = ModuleType("accepted_signed_scorer")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def encode(mask):
    obj = codec.encode(np.asfortranarray(mask, dtype=np.uint8))
    return dict(size=obj["size"], counts=obj["counts"].decode("ascii"))


def cap_masks(raw, stock, image, slot):
    added = stock & ~raw
    drops = raw & ~stock
    indices = np.flatnonzero(drops)
    base = raw | added
    b = min(len(indices), int(added.sum()))
    if not b or b == len(indices):
        shared = stock if b else base
        masks = {arm: shared for arm in (*UNIFORMS, CAP)}
    else:
        depth = distance_transform_edt(np.pad(raw, 1))[1:-1, 1:-1].ravel()
        selections = {CAP: indices[np.argsort(depth[indices], kind="stable")[:b]]}
        for arm, seed in zip(UNIFORMS, SEEDS, strict=True):
            key = sha256(f"{seed}:{image}:{slot}".encode()).digest()
            generator = np.random.default_rng(int.from_bytes(key[:16], "little"))
            selections[arm] = generator.choice(indices, size=b, replace=False)
        masks = {}
        for arm, selection in selections.items():
            mask = base.copy()
            mask.flat[selection] = False
            masks[arm] = mask
    for mask in masks.values():
        kept_drops = raw & ~mask
        require(np.array_equal(mask & ~raw, added) and int(kept_drops.sum()) == b and not (kept_drops & ~drops).any(),
                "Cap masks must preserve stock additions, original delete support and the exact cap")
    masks[ADD] = base
    return {arm: {"rle": encode(mask)} for arm, mask in masks.items()}, len(indices), b, int(added.sum())


def geometry_mask(rle, polygon):
    mask = codec.decode(rle)
    y, x = np.nonzero(mask)
    intersection = shapely.area(shapely.intersection(shapely.box(x, y, x+1, y+1), polygon)).sum()
    iou = intersection / (len(y) + polygon.area - intersection)
    fp = (len(y) - intersection) / polygon.area
    fn = (polygon.area - intersection) / polygon.area
    return 100 * np.array([iou, fp, fn])


def view_geometry(rows, metrics, date, bounds, prior):
    groups = defaultdict(list)
    changed = {pair: Counter() for pair in PAIRS}
    for group, polygon, masks in rows:
        v = {arm: geometry_mask(mask, polygon) for arm, mask in masks.items()}
        v[MEAN] = np.mean([v[a] for a in UNIFORMS], axis=0)
        groups[group].append([v[a] for a in (*ARMS, MEAN)])
        for left, right in PAIRS:
            ls = UNIFORMS if left == MEAN else (left,)
            rs = UNIFORMS if right == MEAN else (right,)
            if any(masks[a] != masks[b] for a in ls for b in rs):
                changed[(left, right)][group] += 1
    parents = sorted(groups)
    means = np.stack([np.mean(groups[k], axis=0) for k in parents])
    arm_idx = {arm: idx for idx, arm in enumerate((*ARMS, MEAN))}
    avg = means.mean(axis=0)
    require(abs(float(avg[arm_idx[STOCK], 0] - avg[arm_idx[ADD], 0]) - prior) < 1e-8, "Closed signed geometry differs")
    measures = {arm: dict(zip(("continuous_IoU_points", "false_positive_truth_area_points",
                              "false_negative_truth_area_points"), avg[i].tolist(), strict=True))
                for arm, i in arm_idx.items()}
    ap = {arm: 100*metrics[arm]["AP"] for arm in ARMS}
    ap[MEAN] = np.mean([ap[arm] for arm in UNIFORMS])
    contrast = {}
    for left, right in PAIRS:
        diff = means[:, arm_idx[left], 0] - means[:, arm_idx[right], 0]
        interval = bounds(diff[:, None], parents)[0]
        edits = changed[(left, right)]
        exposure = sum(edits.values())
        groups_changed = len({date(k) for k in edits})
        delta_ap = float(ap[left] - ap[right])
        contrast[left+"_minus_"+right] = dict(AP_points=delta_ap, continuous_IoU_points=float(diff.mean()),
            interval_95=list(interval), differing_hulls=exposure, differing_resampling_groups=groups_changed,
            advances=bool(delta_ap >= .30 and interval[0] > 0 and exposure >= 20 and groups_changed >= 8))
    return dict(hulls=len(rows), original_groups=len(groups), resampling_groups=len({date(k) for k in groups}),
                pair_geometry_points=measures, contrasts=contrast)


def evaluate_rsdd(a):
    a.cohort, a.support_root = 74, None
    signed = load(a.code / "SEGREFINER_SIGN_20261123/score_rsdd.py", RSDD_SIGN)
    stock = signed.pinned_module(a.code / "SEGREFINER_STOCK_20261123/score_rsdd.py", signed.STOCK_SOURCE)
    exposure = stock.exposure_source()
    _, deps = exposure.sources(a)
    _, api, cfa, history, reserve, old, evaluator, adapter, core = deps
    reference = json.loads(api.pin(a.code.parent/"results/RSDD74_SEGREFINER_SIGN_SCORE_SAFE_20261123.json", RSDD_RESULT))
    report, score_id, score_terminal = api.observed(a.run/"score", a.run/"observed_score/TERMINAL.json",
        "score", stock.STATUS, signed.STOCK_SCORE, worker=signed.STOCK_SOURCE)
    infer, infer_id, infer_terminal = api.observed(a.run/"inference", a.run/"observed_inference/TERMINAL.json",
        "inference", stock.INFER_STATUS, report["inference_result_sha256"], report["inference_terminal_sha256"],
        report["inference_worker_sha256"])
    require((score_id, score_terminal, infer_id, infer_terminal) ==
            tuple(reference[k] for k in ("stock_score_sha256","stock_score_terminal_sha256","inference_result_sha256",
                                         "inference_terminal_sha256")), "Closed lineage differs")
    loader = cfa.source(a.code/"POINTREND_SIGNED_QUOTA_SWAP_20261123/inputs.py", cfa.INPUTS_SHA)
    inputs, lineage = loader.load_population(a)
    require(lineage == report["input_lineage"] == infer["input_lineage"], "Frozen native population differs")
    blocks, areas, positions = exposure.native_blocks(inputs, adapter, core)
    stream = api.pin(a.run/"inference"/stock.LEDGER, report["refined_masks_sha256"])
    counts = stock.attach_verified_masks(inputs, blocks, areas, positions, stream, infer["counts"])
    require(counts == report["counts"], "Frozen native stock census differs")
    writes = Counter(stock_gate_candidates=0, stock_deletions=0, cap_deletions=0, reduced_candidates=0, zero_addition_candidates=0)
    for image, block in blocks.items():
        for slot, det in zip(block["ranks"], block["stock"], strict=True):
            opts = block["changes"].get(slot, {})
            if "stock_lr" not in opts:
                continue
            raw = codec.decode(det["segmentation"]).astype(bool)
            final = codec.decode(opts["stock_lr"]["rle"]).astype(bool)
            caps, amount, cap, added = cap_masks(raw, final, image, slot)
            opts.update(caps)
            writes.update(stock_gate_candidates=1, stock_deletions=amount, cap_deletions=cap, reduced_candidates=int(cap < amount),
                          zero_addition_candidates=int(added == 0))
    require(writes["stock_gate_candidates"] == counts["eligible_candidates"] and
            writes["stock_deletions"] == reference["counts"]["deleted_pixels"], "Cap census differs")
    print(json.dumps(dict(phase="cap_masks_constructed", **writes)), flush=True)
    gt, _, polygons, sets, dates, gt_sha = history.views_and_labels(a, old, reserve, blocks)
    require(gt_sha == reference["author_labels_sha256"], "Original author polygons required")
    views = {}
    for view, count in (("full", 39), ("date_separated", 26)):
        ids = set(sets[view])
        part = dict(gt, images=[im for im in gt["images"] if im["id"] in ids],
                    annotations=[r for r in gt["annotations"] if r["image_id"] in ids])
        subset = {image: blocks[image] for image in ids}
        native, anchors = evaluator.score_one(part, subset, RAW, anchors=True)
        metrics = {RAW: {k: history.available(v) for k,v in native.items()}}
        for arm in ARMS[1:]:
            code = "stock_lr" if arm == STOCK else arm
            vals, _ = evaluator.score_one(part, subset, code, anchors=True)
            metrics[arm] = {k: history.available(v) for k,v in vals.items()}
        orig = reference["views"][view]
        require(all(metrics[k] == orig["metrics_fraction"][v] for k,v in ((RAW,"raw"),(STOCK,"stock_lr"),(ADD,"add_only"))),
                "Original native, stock and addition-only COCO results must be exact")
        rows = []
        for aid,(image,slot,det) in anchors.items():
            poly = polygons[aid]
            if not stock.GATE <= areas[image][slot] < 1024 or not poly.is_valid or poly.area <= 0:
                continue
            opts = blocks[image]["changes"][slot]
            masks = {arm: opts["stock_lr" if arm == STOCK else arm]["rle"] for arm in ARMS[1:]}
            masks[RAW] = det["segmentation"]
            rows.append((blocks[image]["group"], poly, masks))
        require(len(rows) == count, "Original exposed geometry denominator differs")
        value = orig["gate_exposed_geometry"]["paired"]["stock_lr_minus_add_only"]["continuous_iou_points"]["delta"]
        g = view_geometry(rows, metrics, dates.__getitem__, lambda q,p: reserve.bootstrap(q,[dates[k] for k in p]), value)
        views[view] = dict(metrics_fraction=metrics, geometry=g,
                           historical_local_AP=report["views"][view]["metrics_fraction"]["learned_common"]["AP"])
    return views, RSDD_RESULT, dict(writes)


def evaluate_pseg(a):
    signed = load(a.code / "SEGREFINER_SIGN_20261123/score_pseg.py", PSEG_SIGN)
    stock = signed.load_stock(a.stock_source)
    exposure = stock.exposure_module()
    source = exposure.load_source(a.factorial_source)
    ref, ref_hash, _ = source.observed(a.signed_score,a.signed_score_terminal,"score",signed.STATUS,PSEG_SIGN)
    require(ref_hash == PSEG_RESULT, "Expected signed score digest required")
    prior, score_id, score_terminal = source.observed(a.stock_score,a.stock_score_terminal,"score",stock.STATUS,signed.STOCK_WORKER)
    _, scorer, geom, _, capture, roster, audits, original, blocks = source.historical(a)
    records_by_name = exposure.native_records(source,blocks,roster,audits,original)
    infer, result, terminal = stock.verify_receipt(source,a,capture,exposure.PROTOCOL_SHA256)
    require((score_id,score_terminal,result,terminal) ==
            tuple(ref[k] for k in ("stock_score_sha256","stock_score_terminal_sha256","inference_result_sha256",
                                   "inference_terminal_sha256")), "Closed source lineage differs")
    data = source.pin(a.inference/"EDITS_PRIVATE.jsonl",infer["edits_sha256"])
    counts,_ = stock.qualify_edits(source,records_by_name,roster,blocks,map(json.loads,data.splitlines()))
    require(counts == infer["counts"] == ref["counts"], "Closed stock census differs")
    writes = Counter(stock_gate_candidates=0, stock_deletions=0, cap_deletions=0, reduced_candidates=0, zero_addition_candidates=0)
    for image, record in records_by_name.items():
        for pos, opts in record["edits"].items():
            if stock.STOCK not in opts:
                continue
            raw = source.decoded(stock.mask_at(record,int(pos),stock.RAW), record["size"])
            final = source.decoded(opts[stock.STOCK]["rle"], record["size"])
            caps, amount, cap, added = cap_masks(raw,final,image,pos)
            opts.update(caps)
            writes.update(stock_gate_candidates=1,stock_deletions=amount,cap_deletions=cap,reduced_candidates=int(cap < amount),
                          zero_addition_candidates=int(added == 0))
    require(writes["stock_gate_candidates"] == counts["area_admitted"] and
            writes["stock_deletions"] == ref["views"]["official"]["counts"]["removed_pixels"], "Cap census differs")
    print(json.dumps(dict(phase="cap_masks_constructed", **writes)), flush=True)
    gt = json.loads(source.pin(a.author_labels,scorer.GT_SHA))
    require(ref["author_labels_sha256"] == scorer.GT_SHA, "Original author polygons required")
    records = {im["id"]: records_by_name[im["file_name"]] for im in gt["images"]}
    cont = geom.load(a.code/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",geom.GEOM)
    mods = cont.geometry_modules(a.code)
    truth,neutral,_ = mods.geometry.polygon_support(gt,mods.vector)
    require(len(truth) == 546 and not neutral, "Original polygons required")

    def bounds(means, parents):
        rng = np.random.Generator(np.random.PCG64(5217))
        sample = rng.integers(len(means), size=(4096,len(means)))
        return np.percentile(means[sample].mean(axis=1),[2.5,97.5],axis=0).T.tolist()

    keys = {arm: stock.STOCK if arm == STOCK else arm for arm in ARMS}
    views = {}
    for view,count in (("official",67),("exclude_exact_fit_overlap",63)):
        ids = {i for i,r in records.items() if view == "official" or r["name"] not in scorer.EXCLUDED}
        part = dict(gt, images=[im for im in gt["images"] if im["id"] in ids],
                    annotations=[r for r in gt["annotations"] if r["image_id"] in ids])
        order = [im["id"] for im in part["images"]]
        metrics = {}
        for arm,name in keys.items():
            candidates = [dict(image_id=i,category_id=0,score=det["score"],segmentation=stock.mask_at(records[i],rank,name))
                          for i in order for rank,det in enumerate(records[i]["native"],1)]
            metrics[arm] = scorer.evaluate(part,candidates)
        orig = ref["views"][view]
        require(all(metrics[k] == orig["metrics_fraction"][v] for k,v in ((RAW,stock.RAW),(STOCK,stock.STOCK),(ADD,signed.ADD))),
                "Original native, stock and addition-only COCO scores must be exact")
        anchors,_ = geom.evaluate_raw(gt,records,sorted(ids),metrics[RAW])
        rows = []
        for aid,(i,rank) in anchors.items():
            rec = records[i]
            if not stock.GATE <= rec["areas"][rank-1] < 1024:
                continue
            masks = {arm: stock.mask_at(rec,rank,name) for arm,name in keys.items()}
            rows.append((rec["rgb"], truth[aid][0], masks))
        require(len(rows) == count, "Original exposed geometry denominator differs")
        value = orig["primary_gate_exposed_deletion"]["contexts"]["segrefiner_stock_minus_segrefiner_add_only"]["continuous_iou_points"]["delta"]
        g = view_geometry(rows,metrics,lambda k:k,bounds,value)
        views[view] = dict(metrics_fraction=metrics, geometry=g,
                           historical_local_AP=prior["views"][view]["metrics_fraction"]["free_natural"]["AP"],
                           historical_static_AP=prior["views"][view]["metrics_fraction"]["attraction_static_at_dynamic_count"]["AP"])
    return views,PSEG_RESULT,dict(writes)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    sub=parser.add_subparsers(dest="dataset",required=True)
    for name,options in (("rsdd",("run","stock-root","point-root")),
        ("pseg",("signed-score","signed-score-terminal","stock-source","stock-score","stock-score-terminal",
                 "factorial-source","owner","capture","classical-root","inference","inference-terminal"))):
        child=sub.add_parser(name,allow_abbrev=False)
        for key in ("code","author-labels","protocol","output")+options:
            child.add_argument("--"+key,type=Path,required=True)
        child.add_argument("--protocol-sha256",required=True)
        if name=="pseg":
            child.add_argument("--original-readback",type=Path)
            for key in ("checkpoint-sha256","runtime-sha256","inference-worker-sha256"):
                child.add_argument("--"+key,required=True)
    a=parser.parse_args()
    require(sha256(a.protocol.read_bytes()).hexdigest()==a.protocol_sha256 and not a.output.exists(),
            "Frozen diagnostic protocol and fresh output required")
    require((shapely.__version__,shapely.geos_version_string)==("2.0.2","3.11.2"),"Use the closed geometry runtime")
    views, parent, writes=(evaluate_rsdd if a.dataset=="rsdd" else evaluate_pseg)(a)
    out=dict(status=STATUS,phase="score",dataset=a.dataset,views=views,writes=writes,uniform_seeds=list(SEEDS),
             detector_forwards=0,optimizer_updates=0,stock_inference_forwards=0,
             worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=a.protocol_sha256,
             signed_parent_sha256=parent,historical_retrospective=True)
    data=(json.dumps(out,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    a.output.mkdir()
    (a.output/"RESULT_SAFE.json").write_bytes(data)
    print(json.dumps(dict(status=STATUS,result_sha256=sha256(data).hexdigest())),flush=True)


if __name__=="__main__":
    main()
