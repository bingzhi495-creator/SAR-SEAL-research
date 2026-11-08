"""Compute exact oracle flip counts for frozen coarse P101 false positives."""


def minimum_correct_flips(intersection, union, fn, fp):
    """Find the minimum number of known-correct flips, or None if infeasible."""
    need = max(9 * union - 10 * intersection, 0)
    additions = min(fn, (need + 9) // 10)
    removals = (max(need - 10 * additions, 0) + 8) // 9
    return additions + removals if removals <= fp else None


BAND_SHA = "957a9e6845588c475da9bdce38ba6a1fc70f6c51ddbc65579ed4c2c376a8bdb9"


def load_original_band():
    from hashlib import sha256
    from pathlib import Path
    from types import ModuleType

    root = Path(__file__).resolve().parents[2]
    source = (root / "github_release_20261011/research_round_20260925_global/code"
              / "RFDETR_E24_FIT_NATIVE_CAPTURE_CANDIDATE_20261108/capture_post.py")
    content = source.read_bytes()
    if sha256(content).hexdigest() != BAND_SHA:
        raise ValueError("Exact prior sparse-capture band verifier required")
    module = ModuleType("_p101_original_capture_post")
    exec(compile(content, str(source), "exec"), module.__dict__)
    return module._band_positions


def measure_pair(pred, truth, band_positions):
    import numpy as np

    if pred.ndim != 2 or pred.shape != truth.shape or not (np.any(pred) and np.any(truth)):
        raise ValueError("Expected two nonempty masks on the same original image grid")
    intersection = int(np.count_nonzero(pred & truth))
    union = int(np.count_nonzero(pred | truth))
    false_negative = (~pred & truth).ravel()
    false_positive = (pred & ~truth).ravel()
    fn, fp = int(false_negative.sum()), int(false_positive.sum())
    band = band_positions(np, pred)
    band_fn = int(false_negative[band].sum())
    band_fp = int(false_positive[band].sum())
    return dict(
        intersection=intersection, union=union,
        predicted_pixels=intersection + fp, GT_mask_pixels=intersection + fn,
        full_FN=fn, full_FP=fp, band_pixels=len(band), band_FN=band_fn, band_FP=band_fp,
        full_min_edits=minimum_correct_flips(intersection, union, fn, fp),
        band_min_edits=minimum_correct_flips(intersection, union, band_fn, band_fp),
    )


def summarize(rows):
    from statistics import median

    sizes = ("recorded_GT_area", "GT_mask_pixels", "predicted_pixels", "intersection", "union",
             "full_FN", "full_FP", "band_pixels", "band_FN", "band_FP")
    summary = dict(candidates=len(rows), medians={key: median([row[key] for row in rows]) if rows else None
                                                for key in sizes})
    for domain in ("full", "band"):
        minima = [row[f"{domain}_min_edits"] for row in rows if row[f"{domain}_min_edits"] is not None]
        summary[domain] = dict(feasible=len(minima), infeasible=len(rows) - len(minima),
                               at_most_16=sum(n <= 16 for n in minima),
                               median_min_edits_if_feasible=median(minima) if minima else None)
    return summary


def main():
    import argparse
    from contextlib import redirect_stdout
    from copy import deepcopy
    from hashlib import sha256
    from importlib.metadata import version
    import json
    import os
    from pathlib import Path
    import re
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prelock-commit", required=True, help="Root-verified protocol publication commit")
    options = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", options.prelock_commit):
        parser.error("--prelock-commit must be a complete lower-case commit identifier")

    import numpy as np
    from pycocotools import mask
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    candidate = Path(__file__).resolve().parent
    previous = candidate.parent
    helper = previous.parent / "RFDETR_E24_STRICT_OPTIMIZER_EVAL_CANDIDATE_20261109_01"
    pinned = {
        previous / "AP90_FP_OVERLAP_SAFE.json": "591a020736b2895442ec2b34d589ceb89b8402bd8957af51b139e05427f1707e",
        previous / "RANK_WITNESS_SAFE.json": "8c2ce61cbc927d205d1d86f8acacffb319a800767993a603269f11bef3b12692",
        previous / "audit_ap90_fp_overlap.py": "f82f2e97beddf300cc9eb0ab0fcda0bebd271c947efc21d332b1ffa64efefd2d",
        previous / "audit_official_ap90_ranking.py": "56f78b82ac424d8128e089b008f0e0b010842023b864159d4c94146a11e25f1e",
        helper / "verify_optimizer_eval_cpu.py": "bfb6f0148e106a15c8d503796b594c51489b8f75140f641b2f8bbcf8a367f135",
    }
    for path, digest in pinned.items():
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Original closed DEV evidence or helper changed")
    sys.path[:0] = [str(previous), str(helper)]
    from audit_ap90_fp_overlap import classify_prefix
    from audit_official_ap90_ranking import global_ranking
    from verify_optimizer_eval_cpu import CONTROL, OUT, need, raw_ref, read, read_worker_result

    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and version("pycocotools") == "2.0.11",
         "Pinned CPU-only DEV mask runtime required")
    prior = read(previous / "AP90_FP_OVERLAP_SAFE.json")
    witness = read(previous / "RANK_WITNESS_SAFE.json")
    wait = read(CONTROL / "TERMINAL_PRIVATE.json")
    need(wait["exit"] == 0 and wait["original_OS_wait"] is True and wait["guard"] is None,
         "Exact closed DEV scoring")
    observed = read_worker_result(raw_ref(wait["stdout_ref"], 1024**2), OUT / "RESULT.json")
    need(observed["protocol_commit"] == prior["protocol_commit"] == witness["protocol_commit"]
         and observed["TEST"] is prior["TEST"] is witness["TEST"] is False
         and observed["integer_GT_ref"]["sha256"] == witness["integer_GT_sha256"],
         "Exact closed P101 DEV sources")
    dataset = json.loads(raw_ref(observed["integer_GT_ref"], 1024**2))
    dimensions = {row["id"]: (row["height"], row["width"]) for row in dataset["images"]}
    need(len(dimensions) == len(dataset["images"]) == 187 and len(dataset["annotations"]) == 380
         and all(a["category_id"] == 1 and a["iscrowd"] == 0 and a["area"] > 0 for a in dataset["annotations"]),
         "Exact original noncrowd DEV187 population")
    band = load_original_band()
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
            ev.params.imgIds = list(dimensions)
            ev.params.catIds = [1]
            ev.params.maxDets = [1, 10, 100]
            ev.evaluate()
            ev.accumulate()
        iou_index, = np.flatnonzero(np.isclose(ev.params.iouThrs, 0.90))
        precision = ev.eval["precision"][iou_index, :, 0, 0, ev.params.maxDets.index(100)]
        need(abs(float(precision[precision >= 0].mean()) - witness["arms"][name]["AP90"]) < 1e-14,
             "Reproduce original AP90 on unchanged masks and scores")
        rows = [row for row in ev.evalImgs if row is not None and row["category_id"] == 1
                and row["aRng"] == ev.params.areaRng[0] and row["maxDet"] == 100]
        need(len(rows) == len({row["image_id"] for row in rows}) == 187,
             "Complete original all-area per-image records")
        stream = []
        for row in rows:
            key = (row["image_id"], row["category_id"])
            dt_rows = ev._dts[key]
            gt_rows = ev._gts[key]
            order = np.argsort([-d["score"] for d in dt_rows], kind="mergesort")[:100]
            need([dt_rows[i]["id"] for i in order] == row["dtIds"],
                 "Exact original detection ordering for each COCO IoU row")
            overlap = np.asarray(ev.ious[key])
            if not len(order) or not gt_rows:
                overlap = np.zeros((len(order), len(gt_rows)))
            need(overlap.shape == (len(order), len(gt_rows)) and np.isfinite(overlap).all(),
                 "Complete same-image original mask IoU matrix")
            stream.extend((row["image_id"], dt_rows[d], gt_rows, overlap[i])
                          for i, d in enumerate(order))
        scores = np.concatenate([row["dtScores"] for row in rows])
        ignored = np.concatenate([row["dtIgnore"][iou_index] for row in rows])
        order = np.argsort(-scores, kind="mergesort")
        ranked = [stream[index] for index in order if not ignored[index]]
        matches, _ = global_ranking(rows, iou_index)
        maxima = np.asarray([iou.max() if len(iou) else 0.0 for _, _, _, iou in ranked])
        prefix = classify_prefix(matches, maxima, 38)
        need(prefix == prior["arms"][name]["10"], "Reproduce exact original 10% recall FP prefix")
        accepted, ambiguous, unavailable = [], 0, 0
        for match, maximum, record in zip(matches[:prefix["rank_of_required_TP"]],
                                          maxima[:prefix["rank_of_required_TP"]],
                                          ranked[:prefix["rank_of_required_TP"]], strict=True):
            if match != 0 or not 0.50 <= maximum < 0.90:
                continue
            image_id, prediction, truth_rows, overlap = record
            eligible = np.flatnonzero(overlap >= 0.50)
            if len(eligible) > 1:
                ambiguous += 1
                continue
            if not len(eligible):
                unavailable += 1
                continue
            target = truth_rows[eligible[0]]
            pred = mask.decode(prediction["segmentation"])
            truth = mask.decode(target["segmentation"])
            need(pred.shape == truth.shape == dimensions[image_id] and (pred <= 1).all()
                 and (truth <= 1).all(), "Exact original-grid binary masks required")
            pair = measure_pair(pred.astype(bool), truth.astype(bool), band)
            i, u = pair["intersection"], pair["union"]
            need(2 * i >= u and 10 * i < 9 * u and abs(i / u - maximum) < 1e-14
                 and pair["full_min_edits"] is not None,
                 "Exact integer coarse IoU and feasible full-image oracle")
            if pair["band_min_edits"] is not None:
                need(pair["band_min_edits"] >= pair["full_min_edits"], "Band restriction cannot lower minimum")
            pair["recorded_GT_area"] = target["area"]
            accepted.append(pair)
        need(len(accepted) + ambiguous + unavailable == prefix["FP_max_IoU_ge_050_lt_090"],
             "Exhaustive coarse FP counts")
        arms[name] = dict(
            prefix=prefix, exactly_one_GT_ge_050=len(accepted), ambiguous_multiple_GT_ge_050=ambiguous,
            unavailable_no_GT_ge_050=unavailable,
            strata=dict(all=summarize(accepted),
                        small=summarize([row for row in accepted if row["recorded_GT_area"] < 1024]),
                        other=summarize([row for row in accepted if row["recorded_GT_area"] >= 1024])),
        )
    result = dict(
        status="EXPLORATORY_P101_CLOSED_DEV187_COARSE_FP_PIXEL_ORACLE",
        source_commit=prior["source_commit"], original_evaluation_protocol=prior["protocol_commit"],
        prelock_commit=options.prelock_commit, original_band_source_sha256=BAND_SHA,
        original_fp_prefix_sha256=pinned[previous / "AP90_FP_OVERLAP_SAFE.json"],
        images=187, GT_instances=380, DEV=True, TEST=False, fixed_recall_percent=10,
        target_IoU=0.90, budget_max_edits=16, new_training=False, generated_predictions=False,
        scope="Minimum number of known-correct flips on each unique coarse FP/GT pair independently. "
              "Original masks, scores, and AP remain fixed. No joint one-to-one success or realizability claim.",
        band="Original predicted-mask radius-two clipped Chebyshev window containing both binary signs.",
        small_definition="recorded integer annotation area < 1024",
        arms=arms,
    )
    with (candidate / "COARSE_FP_PIXEL_ORACLE_SAFE.json").open("x", encoding="utf-8", newline="\n") as file:
        json.dump(result, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
