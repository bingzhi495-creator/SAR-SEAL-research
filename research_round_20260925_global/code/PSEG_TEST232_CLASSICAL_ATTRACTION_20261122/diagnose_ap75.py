"""Read back the same-count PSeg comparison and explain AP75 at fixed scores."""
import argparse
from copy import deepcopy
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from types import ModuleType

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SCORE_SHA = "3efc20ab1218d2f36b21e5967a9f178488af32d0a3e06c78e64a1ca8062b7365"
CAPTURE_SHA = "e99d0f2bbd9e1839e260a88075f6af187e05ffc2b86abaf6cc33b788372b7b71"
SCORER_SHA = "dd8e435976d2331146c8f83d70f368ee4e0954c05520c54fa9a3b1205838d40f"
ARMS = ("attraction_dynamic", "learned_at_attraction_count")


def guard(ok, label):
    if not ok:
        raise ValueError(label)


def pinned(path, expected):
    raw = path.read_bytes()
    guard(sha256(raw).hexdigest() == expected, path.name)
    return raw


def load(path, expected):
    obj = ModuleType(path.stem)
    obj.__file__ = str(path)
    exec(compile(pinned(path, expected), str(path), "exec"), obj.__dict__)
    return obj


def one_score(gt, blocks, arm, expected):
    dtrows = []
    for im in gt["images"]:
        stock, edits, _ = blocks[im["file_name"]]
        for rank, item in enumerate(stock, 1):
            chosen = edits.get(str(rank), {}).get(arm)
            dtrows.append({"image_id": im["id"], "category_id": 0, "score": item["score"],
                           "segmentation": chosen["rle"] if chosen else item["segmentation"]})
    with redirect_stdout(StringIO()):
        ref = COCO()
        ref.dataset = deepcopy(gt)
        ref.createIndex()
        det = ref.loadRes(deepcopy(dtrows))
        ev = COCOeval(ref, det, "segm")
        ev.params.catIds = [0]
        ev.params.imgIds = [i["id"] for i in gt["images"]]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    ix75, = np.flatnonzero(np.isclose(ev.params.iouThrs, .75))
    prec = ev.eval["precision"][ix75, :, 0, 0, -1]
    guard(float(prec.mean()) == expected["AP75"], "Exact accepted COCO AP75")
    relevant = [row for row in ev.evalImgs if row and row["maxDet"] == 100 and
                list(row["aRng"]) == list(ev.params.areaRng[0])]
    gt_hits = set()
    score_hits = []
    for row in relevant:
        hits = (row["dtMatches"][ix75] > 0) & ~row["dtIgnore"][ix75]
        gt_hits.update(int(v) for v in row["dtMatches"][ix75, hits])
        score_hits.extend(float(row["dtScores"][int(i)]) for i in np.flatnonzero(hits))
    guard(len(gt_hits) == len(score_hits), "Original nonignored once-only TP ids")
    return {"true_positives_at_75": len(gt_hits), "max_recall": len(gt_hits) / len(gt["annotations"]),
            "mean_tp_score": float(np.mean(score_hits))}, prec, gt_hits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("code", "capture", "readback", "owner", "gt", "actual", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    scorer = load(args.code / "PSEG_TEST232_DFINE_FROZEN_SCORE_20261121/score_test232.py", SCORER_SHA)
    actual = json.loads(pinned(args.actual / "score/RESULT_SAFE.json", SCORE_SHA))
    terminal = json.loads((args.actual / "observed_score/TERMINAL.json").read_bytes())
    guard(terminal["actual_exit"] == 0 and terminal["result_sha256"] == SCORE_SHA and
          terminal["output_footer_verified"] is True, "Observed original scoring")
    old = json.loads(pinned(args.readback, CAPTURE_SHA))
    guard(old["status"] == scorer.ACCEPTED, "Accepted original native stream")
    roster = json.loads(pinned(args.owner / "TEST232_IMAGE_ONLY_MANIFEST_PRIVATE.json", old["manifest_sha256"]))["items"]
    blocks_data = [(args.capture / "DEV_MASK_PREDICTIONS_PRIVATE.jsonl", old["input_predictions_sha256"]),
                   (args.capture / "DEV_CANDIDATE_AUDIT_PRIVATE.jsonl", old["candidate_audit_sha256"]),
                   (args.actual / "prepare-edits/EDITS_PRIVATE.jsonl", actual["edits_sha256"])]
    streams = [[json.loads(line) for line in pinned(path, expected).splitlines()] for path, expected in blocks_data]
    scorer.ARMS += ARMS + ("attraction_static_at_dynamic_count",)
    blocks = scorer.join_blocks(roster, *streams)
    gt = json.loads(pinned(args.gt, scorer.GT_SHA))
    result = {}
    for view, exclude in (("official", ()), ("exclude_exact_fit_overlap", scorer.EXCLUDED)):
        ims = [im for im in gt["images"] if im["file_name"] not in exclude]
        ids = {im["id"] for im in ims}
        subset = dict(gt, images=ims, annotations=[ann for ann in gt["annotations"] if ann["image_id"] in ids])
        accepted = actual["views"][view]["metrics_fraction"]
        data = [one_score(subset, blocks, arm, accepted[arm]) for arm in ARMS]
        baseline, changed = data
        prec_delta = changed[1] - baseline[1]
        result[view] = {
            "images": len(ims), "annotations": len(subset["annotations"]),
            "arms": dict(zip(ARMS, (item[0] for item in data))),
            "same_label_retained": len(baseline[2] & changed[2]),
            "same_label_gained": len(changed[2] - baseline[2]),
            "same_label_lost": len(baseline[2] - changed[2]),
            "learned_minus_dynamic_ap75_points": float(100 * prec_delta.mean()),
            "recall_grid_improved": int((prec_delta > 0).sum()),
            "recall_grid_degraded": int((prec_delta < 0).sum()),
            "recall_grid_equal": int((prec_delta == 0).sum()),
        }
    payload = (json.dumps(dict(status="COMPLETED_POSTHOC_AP75_DIAGNOSTIC", posthoc=True, views=result,
                               parent_score_sha256=SCORE_SHA, detector_forwards=0, optimizer_updates=0),
                          sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(payload)
    print(sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()

