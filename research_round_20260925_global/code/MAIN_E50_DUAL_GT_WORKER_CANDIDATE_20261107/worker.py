"""Unbound four-case E50 DEV annotation sensitivity worker.

The CPU owner must admit and verify original source, shard, GT, and summary
bytes and the external E50 acceptance before calling this function. This
module neither opens those inputs nor establishes resource or result authority.
"""

from copy import deepcopy
import math


ARMS = ("PLAIN", "FULL")
INTEGER = "INTEGER_PRIMARY"
FLOAT = "LOCAL_CANDIDATE_FLOAT"
MODULES = {"population_core", "population_contract", "eval_body", "official",
           "cocoeval", "coco_loader", "gt_sensitivity", "population_metrics"}
COCO_KEYS = {"stats", "metrics", "iouType", "maxDets", "original_GT_preserved",
             "score_filter", "official_COCO_crowd_ignore_semantics"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def check_metrics(metrics):
    need(type(metrics) is dict and set(metrics) == COCO_KEYS
         and metrics["iouType"] == "segm" and metrics["maxDets"] == [1, 10, 100]
         and metrics["original_GT_preserved"] is True
         and metrics["official_COCO_crowd_ignore_semantics"] is True
         and type(metrics["stats"]) is list and len(metrics["stats"]) == 12
         and all(math.isfinite(float(value)) for value in metrics["stats"]),
         "Original official COCO12 segmentation metrics required")


def evaluate_dual_gt(np, control, *, shards, native, integer_gt, floating_gt,
                     accepted_integer_summary, modules, report, save):
    """Evaluate exactly PLAIN and FULL under each of the two admitted GTs.

    `save(report)` is the caller's durable progress hook. The on-disk U67
    summary retains its pending schema even after external Root acceptance.
    Native/R comparison rows and the same-GT raster audit are linked by Root.
    """
    need(type(modules) is dict and set(modules) == MODULES, "Exact admitted helper roles")
    need(type(report) is dict and "dual_gt" not in report and callable(save), "Fresh progress report")
    cases = {variant: {arm: dict(math_started=False, math_completed=False, metrics=None)
                       for arm in ARMS} for variant in (INTEGER, FLOAT)}
    state = dict(schema="Root-E50-DEV187-dual-GT-arms-worker-v1",
                 status="VALIDATING_ADMITTED_DOCUMENTS", results=cases,
                 images=187, annotations=380, candidates_per_arm=18700,
                 fixed_epoch=50, updates_per_arm=37050,
                 integer_COCO12_reproduced=None, qualified=False, TEST=False,
                 new_training=False, formal_training_budget_credit=0,
                 native_R_and_raster_audit_bound=False)
    report["dual_gt"] = state
    save(report)
    control.check()
    core = modules["population_core"]
    core._shards(shards)
    same = lambda left, right: core.signature(left) == core.signature(right)
    original = accepted_integer_summary
    need(original["schema"] == "U67-DGX-fixed50-population-metrics-pending-v1"
         and original["status"] == "METRICS_COMPLETE_PENDING_ROOT_ACCEPTANCE"
         and same(original["endpoint"], shards[0]["endpoint"])
         and all(type(original[key]) is int and original[key] == expected for key, expected in
                 (("images", 187), ("GT", 380), ("queries", 18700),
                  ("fixed_epoch", 50), ("updates_per_arm", 37050)))
         and original["AP"] is original["DEV"] is True
         and original["TEST"] is original["new_training"] is original["qualified"] is False,
         "Exact original complete E50 primary metric summary")
    expected = {arm: original["results"][arm]["COCO"] for arm in ARMS}
    for value in expected.values():
        check_metrics(value)

    modules["gt_sensitivity"].validate_GT_pair(integer_gt, floating_gt)
    ids = [row["id"] for row in integer_gt["images"]]
    need(len(ids) == len(set(ids)) == 187 and all(type(image) is int for image in ids)
         and type(native) is list and len(native) == 18700
         and [native[i*100]["image_id"] for i in range(187)] == ids,
         "Exact ordered DEV187 native query population")
    merge = modules["population_contract"].merge_predictions
    ranges = core.RANGES
    def joined(rows):
        pieces = [dict(dataset_range=span, predictions=records) for span, records in zip(ranges, rows)]
        return merge(pieces, native, ranges)

    need(same(joined([shard["identity_predictions"] for shard in shards]), native),
         "Exact original full-query identity population")
    predictions = {arm: joined([shard["predictions"][arm] for shard in shards]) for arm in ARMS}
    variants = {INTEGER: integer_gt, FLOAT: floating_gt}
    for variant, ground_truth in variants.items():
        state["status"] = "EVALUATING_" + variant
        for arm in ARMS:
            control.check()
            coco = modules["coco_loader"].COCO()
            coco.dataset = deepcopy(ground_truth)
            coco.createIndex()
            need(list(coco.imgs) == ids and len(coco.anns) == 380 and set(coco.cats) == {1},
                 "Both GTs retain all original image IDs, instances and category")
            row = cases[variant][arm]
            row["math_started"] = True
            save(report)
            metrics, per_iou, evaluation = modules["eval_body"].official_metrics(
                np, control, modules["official"], modules["cocoeval"].COCOeval, coco,
                deepcopy(predictions[arm]), ids)
            del evaluation
            row["math_completed"] = True
            save(report)
            check_metrics(metrics)
            need(same(metrics["score_filter"], expected[arm]["score_filter"]),
                 "Original fixed score filtering protocol")
            row["metrics"] = dict(COCO=metrics, per_IoU=per_iou)
            if variant == INTEGER:
                row["accepted_integer_COCO12_exact"] = same(metrics, expected[arm])
            save(report)
        if variant == INTEGER:
            state["integer_COCO12_reproduced"] = all(cases[INTEGER][arm]["accepted_integer_COCO12_exact"] for arm in ARMS)
            save(report)
            need(state["integer_COCO12_reproduced"], "Both saved integer COCO12 dictionaries must reproduce before float")

    delta = modules["population_metrics"]._delta_stats
    stats = {variant: {arm: cases[variant][arm]["metrics"]["COCO"]["stats"] for arm in ARMS}
             for variant in variants}
    state["FULL_minus_PLAIN_COCO12"] = {variant: delta(values["FULL"], values["PLAIN"])
                                      for variant, values in stats.items()}
    state["float_minus_integer_COCO12"] = {arm: delta(stats[FLOAT][arm], stats[INTEGER][arm]) for arm in ARMS}
    control.check()
    state["status"] = "FOUR_CASES_COMPLETE_PENDING_ROOT_ACCEPTANCE"
    save(report)
    return state
