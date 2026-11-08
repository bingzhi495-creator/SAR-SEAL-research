"""Read match totals and TP depth from the unchanged closed DEV COCO evaluation."""
import numpy as np


def global_ranking(rows, iou_index):
    scores = np.concatenate([row["dtScores"] for row in rows])
    matches = np.concatenate([row["dtMatches"][iou_index] for row in rows])
    ignored = np.concatenate([row["dtIgnore"][iou_index] for row in rows])
    order = np.argsort(-scores, kind="mergesort")
    return matches[order][~ignored[order]], int(ignored.sum())


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
    witness_path = directory / "RANK_WITNESS_SAFE.json"
    witness_sha = "8c2ce61cbc927d205d1d86f8acacffb319a800767993a603269f11bef3b12692"
    witness_raw = witness_path.read_bytes()
    if sha256(witness_raw).hexdigest() != witness_sha:
        raise ValueError("Exact prior closed-input witness required")
    rank_witness = json.loads(witness_raw)
    helper = directory.parent / "RFDETR_E24_STRICT_OPTIMIZER_EVAL_CANDIDATE_20261109_01"
    helper_sha = "bfb6f0148e106a15c8d503796b594c51489b8f75140f641b2f8bbcf8a367f135"
    if sha256((helper / "verify_optimizer_eval_cpu.py").read_bytes()).hexdigest() != helper_sha:
        raise ValueError("Original fixed DEV verification helper changed")
    sys.path.insert(0, str(helper))
    from verify_optimizer_eval_cpu import CONTROL, OUT, need, raw_ref, read, read_worker_result

    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and version("pycocotools") == "2.0.11",
         "Pinned CPU-only DEV mask runtime required")
    wait = read(CONTROL / "TERMINAL_PRIVATE.json")
    need(wait["exit"] == 0 and wait["original_OS_wait"] is True and wait["guard"] is None,
         "Exact closed DEV scoring")
    observed = read_worker_result(raw_ref(wait["stdout_ref"], 1024**2), OUT / "RESULT.json")
    need(observed["protocol_commit"] == rank_witness["protocol_commit"] and observed["TEST"] is False
         and observed["integer_GT_ref"]["sha256"] == rank_witness["integer_GT_sha256"],
         "Exact closed P101 DEV sources")
    dataset = json.loads(raw_ref(observed["integer_GT_ref"], 1024**2))
    image_ids = [row["id"] for row in dataset["images"]]
    area = {row["id"]: row["area"] for row in dataset["annotations"]}
    need(len(image_ids) == 187 and len(area) == 380, "Exact DEV187 population")
    recall_grid = (5, 10, 15, 20, 25)
    arms = {}
    for name in ("E24", "F", "S"):
        ref = observed["arms"][name]["predictions_ref"]
        need(ref["sha256"] == rank_witness["predictions_sha256"][name], "Exact retained predictions")
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
        max_index = ev.params.maxDets.index(100)
        precision = ev.eval["precision"][iou_index, :, 0, 0, max_index]
        ap90 = float(precision[precision >= 0].mean())
        original = rank_witness["arms"][name]
        need(abs(ap90 - original["AP90"]) < 1e-14, "Reproduce original AP90")
        rows = [row for row in ev.evalImgs if row is not None
                and row["category_id"] == 1 and row["aRng"] == ev.params.areaRng[0] and row["maxDet"] == 100]
        need(len(rows) == 187, "Complete original per-image all-area COCO records")
        matched, ignored = global_ranking(rows, iou_index)
        tp_positions = np.flatnonzero(matched > 0)
        ids = [int(i) for i in matched[tp_positions]]
        need(len(set(ids)) == len(ids) and all(i in area for i in ids)
             and abs(len(ids) / len(area) - ev.eval["recall"][iou_index, 0, 0, max_index]) < 1e-14,
             "COCO cumulative recall agrees with stable global match order")
        stats = {}
        for percent in recall_grid:
            n = (percent * len(area) + 99) // 100
            rank = int(tp_positions[n - 1] + 1) if n <= len(tp_positions) else None
            grid_index, = np.flatnonzero(np.isclose(ev.params.recThrs, percent / 100))
            stats[str(percent)] = dict(required_TP=n, rank_of_required_TP=rank,
                                       FP_before_required_TP=(rank - n if rank is not None else None),
                                       COCO_interpolated_precision=float(precision[grid_index]))
        maximum = original["maximum_injective_all"]["100"]
        arms[name] = dict(
            AP90=ap90, retained_predictions=len(predictions), COCO_ignored_predictions=ignored,
            COCO_greedy_TP=len(ids), COCO_greedy_TP_small=sum(area[i] < 1024 for i in ids),
            max_injective_TP=maximum, greedy_shortfall=maximum - len(ids), official_recall_grid=stats,
        )
    report = dict(
        status="EXPLORATORY_P101_UNCHANGED_COCO_AP90_MATCH_AND_RANK_READOUT",
        source_commit=rank_witness["source_commit"], protocol_commit=rank_witness["protocol_commit"],
        rank_witness_sha256=witness_sha, images=len(image_ids), GT_instances=len(area), DEV=True, TEST=False,
        official_metrics_unchanged=True, IoU=0.90, area_range="all", maxDets=100, arms=arms,
        limit="Compared original global score sequences and greedy match outcomes across different retained masks, no score-only attribution.",
    )
    with (directory / "OFFICIAL_AP90_RANK_SAFE.json").open("x", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
