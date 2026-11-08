"""Fit equal-capacity mean and rank boundary quality heads on the frozen SSDD FIT cache."""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


D = Path(__file__).resolve().parent.parent
MATERIAL = D / "RFDETR_E24_FIT_SELECTOR_MATERIALIZED_ACTUAL_PRIVATE_20261108_01"
MATERIAL_SHA = "742053fc509b9168b02727a8ed58120a471569b597916da764ea8a06e0fee9c4"
SPEC_SHA = "a8f1ebefc3e67ed2a857e7ae88beb1031aa55c37428de4e8588eaa93d387c617"
MATERIAL_API = D / "github_release_20261011/research_round_20260925_global/code/RFDETR_E24_FIT_EQUAL_HEAD_SCOUT_ACTUAL_20261108/fit_selector_materialize.py"
API_SHA = "c3f19a7dab49729c37737bf12f56683ba9ff0bef1432607fdb42c4691234530e"
IOU_THRESHOLDS = np.linspace(.5, .95, 10)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, digest):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Exact closed input " + path.name)
    return raw


def quality_targets(best_iou):
    return (np.asarray(best_iou)[:, None] >= IOU_THRESHOLDS).mean(axis=1)


def pair_concordance(positive, negative):
    d = positive[:, None] - negative
    return float((np.count_nonzero(d > 0) + .5 * np.count_nonzero(d == 0)) / d.size)


def candidate_features(score, area, arrays, start, stop):
    f = arrays["features"][start:stop]
    valid = arrays["valid"][start:stop]
    safe_score = np.clip(score, 1e-6, 1 - 1e-6)
    common = (float(np.log(safe_score / (1 - safe_score))), float(np.log1p(area)),
              float(np.log1p(len(f) / max(1, area))),
              float(np.exp(-f[:, 0]).mean()) if len(f) else 0.0,
              float(valid.mean()) if len(f) else 0.0)
    cues = f[valid, 1:].mean(axis=0) if valid.any() else (0.0, 0.0)
    return common, float(cues[0]), float(cues[1])


def fit_quality(x, y, test):
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-8)

    def design(values):
        return np.column_stack((np.ones(len(values)), np.clip((values - center) / scale, -8, 8)))

    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -40, 40)))

    design_fit = design(x)
    w = np.zeros(design_fit.shape[1])
    for _ in range(128):
        regular = .02 * w
        regular[0] = 0.
        w -= .25 * ((design_fit.T @ (sigmoid(design_fit @ w) - y)) / len(y) + regular)
    return sigmoid(design(test) @ w)


def image_rows(item, material, capture, annotations, api, codec, capture_root):
    document = api.doc(material["match_ref"])
    with np.load(BytesIO(api.checked(material["arrays_ref"], 16 * 1024**2)), allow_pickle=False) as data:
        a = {k: data[k] for k in ("offsets", "features", "valid")}
    source, _ = api.captured_item(capture_root, item, capture)
    records = source["records"]
    need(item["image_id"] == material["image_id"] and len(records) == len(document["rows"])
         and len(a["offsets"]) == len(records) + 1, "Same candidate population and native sparse spans")
    h, w = item["original_hw"]
    regular = [x for x in annotations if not x.get("iscrowd", 0)]
    masks = [dict(size=x["segmentation"]["size"], counts=x["segmentation"]["counts"].encode("ascii"))
             for x in records]
    gt = [codec.merge(codec.frPyObjects(x["segmentation"], h, w)) for x in regular]
    iou = codec.iou(masks, gt, [0] * len(gt)) if gt and masks else np.zeros((len(masks), len(gt)))
    need(iou.shape == (len(masks), len(gt)) and np.isfinite(iou).all()
         and (iou >= 0).all() and (iou <= 1).all(), "Finite exact mask IoU")
    x_mean, x_rank, rows = [], [], []
    for j, (r, m, area) in enumerate(zip(records, document["rows"], codec.area(masks), strict=True)):
        lo, hi = map(int, a["offsets"][j:j+2])
        need(all(r[k] == m[k] for k in ("native_rank", "query_id", "score"))
             and (lo, hi) == (m["start"], m["stop"]), "Same frozen score, rank, query and feature support")
        common, mean, rank = candidate_features(r["score"], int(area), a, lo, hi)
        x_mean.append((*common, mean))
        x_rank.append((*common, rank))
        rows.append(dict(image_id=item["image_id"], category_id=1, score=r["score"], segmentation=r["segmentation"]))
    best = iou.max(axis=1) if gt else np.zeros(len(rows))
    return dict(mean=x_mean, rank=x_rank, y=quality_targets(best), rows=rows, iou=iou,
                gt_areas=[r["area"] for r in regular], image_id=item["image_id"])


def all_metrics(coco, evaluator, gt, rows):
    with redirect_stdout(sys.stderr):
        c = coco.COCO()
        c.dataset = gt
        c.createIndex()
        e = evaluator.COCOeval(c, c.loadRes(rows), "segm")
        e.params.imgIds = [r["id"] for r in gt["images"]]
        e.params.catIds = [1]
        e.params.maxDets = [1, 10, 100]
        e.evaluate()
        e.accumulate()
        e.summarize()
    p = e.eval["precision"][8, :, :, 0, -1]
    return dict(AP=float(e.stats[0]), AP75=float(e.stats[2]), AP90=float(p[p >= 0].mean()),
                APs=float(e.stats[3]), AR10=float(e.stats[7]), AR100=float(e.stats[8]),
                all12=[float(x) for x in e.stats])


def pair_result(images, score):
    groups = {"small": [], "other": []}
    group_images = {"small": set(), "other": set()}
    base = 0
    for data in images:
        iou = data["iou"]
        own = score[base:base+len(iou)]
        base += len(iou)
        unique = (iou >= .5).sum(axis=1) == 1
        for j, area in enumerate(data["gt_areas"]):
            pos = unique & (iou[:, j] >= .9)
            neg = unique & (iou[:, j] >= .5) & (iou[:, j] < .9)
            if pos.any() and neg.any():
                name = "small" if area <= 1024 else "other"
                groups[name].append(pair_concordance(own[pos], own[neg]))
                group_images[name].add(data["image_id"])
    return {k: dict(GT=len(v), images=len(group_images[k]), concordance=float(np.mean(v)) if v else None)
            for k, v in groups.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--protocol", required=True)
    args = ap.parse_args()
    need(len(args.protocol) == 40 and all(c in "0123456789abcdef" for c in args.protocol), "Exact protocol commit")
    start = time.monotonic()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and "torch" not in sys.modules,
         "CPU-only frozen FIT diagnostic")
    args.output_dir.mkdir()
    spec = json.loads(pinned(MATERIAL / "SPEC.json", SPEC_SHA))
    accepted = json.loads(pinned(MATERIAL / "ROOT_INDEPENDENT_MATERIALIZATION_ACCEPTED_PRIVATE.json", MATERIAL_SHA))
    need(spec["DEV"] is spec["TEST"] is accepted["DEV"] is accepted["TEST"] is False
         and accepted["status"] == "ACTUAL_FULL_FIT741_FIXED_MATCH_FEATURES_ACCEPTED",
         "Only original independently accepted FIT materialization")
    loader = importlib.util.spec_from_file_location("_closed_fit_material", MATERIAL_API)
    module = importlib.util.module_from_spec(loader)
    exec(compile(pinned(MATERIAL_API, API_SHA), str(MATERIAL_API), "exec"), module.__dict__)
    _, coco, evaluator, codec = module.runtime(spec)
    items = module.doc(spec["image_manifest"])["items"]
    index = module.doc(accepted["material_index_ref"])
    captured = module.doc(spec["capture_index"])
    need(len(items) == len(index) == len(captured) == 741
         and spec["capture_acceptance"] == accepted["capture_acceptance_ref"], "Exact original full FIT references")
    gt = module.original_gt(spec, items)
    truth = defaultdict(list)
    for r in gt["annotations"]:
        truth[r["image_id"]].append(r)
    split = {"head_fit": [], "head_holdout": []}
    for item, material, capture in zip(items, index, captured, strict=True):
        need(all(item[k] == material[k] == capture[k] for k in ("partition", "image_id", "dataset_index")),
             "Original immutable group roles and ordering")
        if item["partition"] in split:
            split[item["partition"]].append(image_rows(item, material, capture, truth[item["image_id"]],
                module, codec, Path(spec["capture_root"])))
    train, hold = split["head_fit"], split["head_holdout"]
    need(len(train) == 516 and len(hold) == 111, "Original image partitions")
    target = np.concatenate([x["y"] for x in train])
    scores = {"native": np.array([r["score"] for x in hold for r in x["rows"]])}
    need(len(target) == 42481 and len(scores["native"]) == 9083, "Exact frozen candidate population")
    for arm in ("mean", "rank"):
        x = np.asarray([r for d in train for r in d[arm]])
        test = np.asarray([r for d in hold for r in d[arm]])
        scores[arm] = scores["native"] * fit_quality(x, target, test)
    selected = {x["image_id"] for x in hold}
    scoped_gt = {**gt, "images": [r for r in gt["images"] if r["id"] in selected],
                 "annotations": [r for r in gt["annotations"] if r["image_id"] in selected]}
    rows = [r for d in hold for r in d["rows"]]
    metrics = {k: all_metrics(coco, evaluator, scoped_gt,
               [{**r, "score": float(s)} for r, s in zip(rows, v, strict=True)]) for k, v in scores.items()}
    need(abs(metrics["native"]["AP"] - .7247336651207286) < 1e-14
         and abs(metrics["native"]["AP75"] - .9016320389088902) < 1e-14,
         "Original FIT holdout no-edit COCO anchor")
    pairs = {k: pair_result(hold, v) for k, v in scores.items()}
    m, r, n = metrics["mean"], metrics["rank"], metrics["native"]
    p = pairs["rank"]["small"]
    enough = p["GT"] >= 20 and p["images"] >= 15
    pair_delta = p["concordance"] - max(pairs[k]["small"]["concordance"] for k in ("mean", "native")) if enough else None
    passed = (enough and pair_delta >= .02 and r["AP"] >= max(m["AP"], n["AP"]) + .001
              and r["AP75"] >= n["AP75"] and r["AP90"] >= max(m["AP90"], n["AP90"]))
    report = dict(status="ACTUAL_FIT_BILATERAL_QUALITY_EQUAL_HEAD_SCOUT_COMPLETE",
        decision="PROVISIONAL_FIT_UTILITY_PASS" if passed else ("STOP_FIXED_RECIPE" if enough else "STOP_INSUFFICIENT_STRICT_PAIR_SUPPORT"),
        protocol=args.protocol, train_images=516, holdout_images=111, unused_calibration_images=114,
        train_candidates=len(target), holdout_candidates=len(rows), holdout_GT=len(scoped_gt["annotations"]),
        fit_updates_per_head=128, total_seconds=time.monotonic() - start, small_pair_concordance_delta=pair_delta,
        pair_metrics=pairs, metrics=metrics, masks_and_candidate_membership_frozen=True,
        original_DEV=False, TEST=False, segmentation_model_updates=0)
    module.put(args.output_dir / "RESULT_SAFE.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

