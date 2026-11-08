"""Classify original COCO AP90 false positives by same-image mask overlap."""
import numpy as np


def classify_prefix(matches, max_iou, required_tp):
    positions = np.flatnonzero(matches > 0)
    if len(positions) < required_tp:
        raise ValueError("The specified original recall cutoff was not attained")
    end = int(positions[required_tp - 1] + 1)
    fp_ious = max_iou[:end][matches[:end] == 0]
    return dict(
        required_TP=required_tp, rank_of_required_TP=end, FP_before_required_TP=len(fp_ious),
        FP_max_IoU_ge_090=int(np.count_nonzero(fp_ious >= 0.90)),
        FP_max_IoU_ge_050_lt_090=int(np.count_nonzero((fp_ious >= 0.50) & (fp_ious < 0.90))),
        FP_max_IoU_lt_050=int(np.count_nonzero(fp_ious < 0.50)),
    )


def main():
    from contextlib import redirect_stdout
    from copy import deepcopy
    from hashlib import sha256
    from importlib.metadata import version
    import json
    import os
    from pathlib import Path
    import sys

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    directory = Path(__file__).resolve().parent
    helper = directory.parent / "RFDETR_E24_STRICT_OPTIMIZER_EVAL_CANDIDATE_20261109_01"
    pinned = {
        directory / "OFFICIAL_AP90_RANK_SAFE.json": "6bd8ece649a38dcec30aeb39411c02da6a63f3677ed2a17a2a470eecb8118db8",
        directory / "RANK_WITNESS_SAFE.json": "8c2ce61cbc927d205d1d86f8acacffb319a800767993a603269f11bef3b12692",
        directory / "audit_official_ap90_ranking.py": "56f78b82ac424d8128e089b008f0e0b010842023b864159d4c94146a11e25f1e",
        helper / "verify_optimizer_eval_cpu.py": "bfb6f0148e106a15c8d503796b594c51489b8f75140f641b2f8bbcf8a367f135",
    }
    for path, digest in pinned.items():
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Original closed DEV evidence or helper changed")
    sys.path[:0] = [str(directory), str(helper)]
    from audit_official_ap90_ranking import global_ranking
    from verify_optimizer_eval_cpu import CONTROL, OUT, need, raw_ref, read, read_worker_result

    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and version("pycocotools") == "2.0.11",
         "Pinned CPU-only DEV mask runtime required")
    prior = read(directory / "OFFICIAL_AP90_RANK_SAFE.json")
    witness = read(directory / "RANK_WITNESS_SAFE.json")
    wait = read(CONTROL / "TERMINAL_PRIVATE.json")
    need(wait["exit"] == 0 and wait["original_OS_wait"] is True and wait["guard"] is None,
         "Exact closed DEV scoring")
    observed = read_worker_result(raw_ref(wait["stdout_ref"], 1024**2), OUT / "RESULT.json")
    need(observed["protocol_commit"] == prior["protocol_commit"] == witness["protocol_commit"]
         and observed["TEST"] is prior["TEST"] is witness["TEST"] is False
         and observed["integer_GT_ref"]["sha256"] == witness["integer_GT_sha256"],
         "Exact closed P101 DEV sources")
    dataset = json.loads(raw_ref(observed["integer_GT_ref"], 1024**2))
    image_ids = [row["id"] for row in dataset["images"]]
    need(len(image_ids) == len(set(image_ids)) == 187 and len(dataset["annotations"]) == 380
         and all(a["category_id"] == 1 and a["iscrowd"] == 0 for a in dataset["annotations"]),
         "Exact noncrowd DEV187 population")
    arms = {}
    for name in ("E24", "F", "S"):
        ref = observed["arms"][name]["predictions_ref"]
        need(ref["sha256"] == witness["predictions_sha256"][name], "Exact retained predictions")
        predictions = json.loads(raw_ref(ref, 256*1024**2))
        with redirect_stdout(sys.stderr):
            gt = COCO()
            gt.dataset = deepcopy(dataset)
            gt.createIndex()
            dt = gt.loadRes(predictions)
            ev = COCOeval(gt, dt, "segm")
            ev.params.imgIds = image_ids
            ev.params.catIds = [1]
            ev.params.maxDets = [1, 10, 100]
            ev.evaluate()
            ev.accumulate()
        iou_index, = np.flatnonzero(np.isclose(ev.params.iouThrs, 0.90))
        precision = ev.eval["precision"][iou_index, :, 0, 0, ev.params.maxDets.index(100)]
        need(abs(float(precision[precision >= 0].mean()) - prior["arms"][name]["AP90"]) < 1e-14,
             "Reproduce original AP90")
        rows = [row for row in ev.evalImgs if row is not None and row["category_id"] == 1
                and row["aRng"] == ev.params.areaRng[0] and row["maxDet"] == 100]
        need(len(rows) == 187, "Complete original all-area per-image records")
        per_image_ious = []
        for row in rows:
            key = (row["image_id"], row["category_id"])
            dt_rows = ev._dts[key]
            order = np.argsort([-d["score"] for d in dt_rows], kind="mergesort")[:100]
            need([dt_rows[i]["id"] for i in order] == row["dtIds"],
                 "Exact COCO IoU matrix rows by original detection order")
            overlaps = np.asarray(ev.ious[key])
            if len(row["dtIds"]) and len(ev._gts[key]):
                need(overlaps.shape == (len(row["dtIds"]), len(ev._gts[key]))
                     and np.isfinite(overlaps).all(), "Complete same-image ordinary mask IoUs")
                per_image_ious.append(overlaps.max(axis=1))
            else:
                per_image_ious.append(np.zeros(len(row["dtIds"])))
        scores = np.concatenate([row["dtScores"] for row in rows])
        ignored = np.concatenate([row["dtIgnore"][iou_index] for row in rows])
        order = np.argsort(-scores, kind="mergesort")
        max_iou = np.concatenate(per_image_ious)[order][~ignored[order]]
        matches, ignored_count = global_ranking(rows, iou_index)
        need(len(max_iou) == len(matches) and ignored_count == prior["arms"][name]["COCO_ignored_predictions"],
             "Exact score/IoU stream alignment")
        grid = {}
        for cutoff in ("5", "10", "20"):
            expected = prior["arms"][name]["official_recall_grid"][cutoff]
            row = classify_prefix(matches, max_iou, expected["required_TP"])
            need(all(row[key] == expected[key] for key in
                     ("required_TP", "rank_of_required_TP", "FP_before_required_TP")),
                 "Exact original fixed recall cutoff and FP count")
            need(sum(count for key, count in row.items() if key.startswith("FP_max_"))
                 == row["FP_before_required_TP"], "Exhaustive disjoint FP overlap groups")
            grid[cutoff] = row
        arms[name] = grid
    report = dict(
        status="EXPLORATORY_P101_CLOSED_DEV187_AP90_FP_OVERLAP",
        source_commit=prior["source_commit"], protocol_commit=prior["protocol_commit"],
        original_ap90_readout_sha256=pinned[directory / "OFFICIAL_AP90_RANK_SAFE.json"],
        images=187, GT_instances=380, DEV=True, TEST=False, IoU=0.90, area_range="all", maxDets=100,
        official_metrics_unchanged=True, fixed_recall_percent=[5, 10, 20],
        max_iou="Maximum ordinary segmentation IoU over original same-image GT, or zero for no GT.",
        scope="FP status from original COCO greedy matching. Bins are geometric overlap categories and do not certify object absence or identify a score cause.",
        arms=arms,
    )
    with (directory / "AP90_FP_OVERLAP_SAFE.json").open("x", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
