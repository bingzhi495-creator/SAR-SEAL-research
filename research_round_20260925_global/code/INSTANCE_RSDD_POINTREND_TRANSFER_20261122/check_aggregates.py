"""Verify two closed observer receipts and independently reconcile the published score tables."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path


OLD = "59936bb4ceb70db1af4801f8d31d1ba84f223adb688759c795d8a71e58f7da4b"
LOCK = "b4bfdeab8d78c0b7b6c68e3615c3f2c217bb568237749bf43508ea4bf74fa4ad"
FIT = "7401de08367ef31b909ea9c92daa27be02d4efb04460acbf96eab33648fdd7cc"
REPLAY_SOURCE = "9368956a7d47e502c005a6adc1ab39edaca18cd5b3780002073ba672732a2b2f"
SCORE_SOURCE = "45971f735e9e186ce20a38afb319a9a3b7dec815f3778ebe59099e0b5517ad88"
ALIASES = dict(N="raw", C="coarse_paste", P="point_paste", B="point_equal", L="learned_common")
PAIRS = ("LB", "PC", "PN", "LN", "LP", "BN")
KEYS = set("AP AP50 AP75 AP90 APs APm APl AR1 AR10 AR100 ARs ARm ARl".split())
GEOM = ("continuous_iou_points", "fp_per_truth_area_points", "fn_per_truth_area_points")
VIEWS = {"gf3", "full", "terrasar_x", "gf3_deduplicated", "full_deduplicated"}


def check(condition, where):
    if not condition:
        raise ValueError(where)


def close(left, right, where):
    check(math.isclose(left, right, rel_tol=0, abs_tol=2e-10), where)


def terminal(path, phase, source):
    record = json.loads(path.read_bytes())
    check(record["actual_exit"] == 0 and record["output_footer_verified"] is True
          and record["phase"] == phase and record["worker_sha256"] == source, str(path))
    return record["result_sha256"]


def load(path, expected):
    raw = path.read_bytes()
    check(sha256(raw).hexdigest() == expected, str(path))
    return json.loads(raw)


def check_edits(replay, score):
    counts, edits = replay["counts"], score["realized_edits"]
    check(counts["images"] == 740 and counts["mapped_candidates"] == 73226
          and counts["area_eligible"] == edits["area_eligible"] == 43252
          and counts["rendered"] == edits["valid"] == 43252 - counts.get("degenerate", 0)
          and counts["point_evaluations"] == 3136 * counts["rendered"], "Full execution census")
    sensors = replay["counts_by_sensor"]
    check({k: v["images"] for k, v in sensors.items()} == {"GF-3": 583, "TerraSAR-X": 157}, "Sensors")
    for key, total in counts.items():
        check(sum(group.get(key, 0) for group in sensors.values()) == total, "Sensor sum " + key)
    for arm in ("learned_common", "point_equal"):
        check(counts[arm+"_swaps"] == edits[arm+"_adds"] == edits[arm+"_drops"] == 329222
              and counts[arm+"_candidates"] == edits[arm+"_edited_candidates"] == 42829
              and edits[arm+"_area_changed_candidates"] == edits[arm+"_emptied_candidates"] == 0,
              "Full seed-5217 balanced budget " + arm)
    for arm in ("point_paste", "coarse_paste"):
        for suffix, verified in (("adds", "adds"), ("drops", "drops"), ("empty", "emptied_candidates"),
                                 ("candidates", "edited_candidates")):
            check(counts.get(arm+"_"+suffix, 0) == edits.get(arm+"_"+verified, 0), arm+" "+suffix)


def check_view(name, view, old):
    metrics, diffs, geom = view["metrics_fraction"], view["paired_contrasts_points"], view["geometry"]
    original, means = old["geometry"], geom["group_equal_arms"]
    check((view["images"], view["annotations"], view["groups"]) ==
          (old["images"], old["annotations"], old["sensor_groups"]), name+" population")
    check((geom["original_anchors"], geom["eligible_anchors"], geom["groups"]) ==
          (original["original_anchors"], original["counts"]["eligible_anchors"], original["stats"]["groups"]),
          name+" fixed identities")
    check(set(metrics) == set(means) == set(ALIASES.values()), name+" policy arms")
    for arm, row in metrics.items():
        check(set(row) == KEYS and all(v == -1 or 0 <= v <= 1 for v in row.values()), name+" COCO "+arm)
    for arm, prev in (("raw", "raw"), ("learned_common", "free_natural")):
        check(metrics[arm] == old["metrics_fraction"][prev], name+" exact 13 historical metrics "+arm)
        for field in GEOM:
            close(means[arm][field], original["stats"]["group_equal_arms"][prev][field], name+" "+arm+" "+field)
        for t in (".5", ".75", ".9"):
            cut = "0"+t
            for move in ("gained", "lost"):
                check(geom["transitions"][cut][arm][move] == original["transitions"][cut][prev][move],
                      name+" "+arm+" "+cut+" "+move)
    names = {a+b: ALIASES[a]+" minus "+ALIASES[b] for a, b in PAIRS}
    check(set(diffs) == set(geom["paired_contrasts_points"]) == set(names.values()), name+" contrast roster")
    for (a, b), key in names.items():
        left, right = ALIASES[a], ALIASES[b]
        check(set(diffs[key]) == KEYS, name+" metrics "+key)
        for field in KEYS:
            x, y = metrics[left][field], metrics[right][field]
            target = None if min(x, y) < 0 else (x-y)*100
            check(diffs[key][field] == target, name+" "+key+" "+field)
        for field in GEOM:
            interval = geom["paired_contrasts_points"][key][field]
            close(interval["delta"], means[left][field] - means[right][field], name+" "+key+" "+field)
            lo, hi = interval["interval_95"]
            check(math.isfinite(lo) and math.isfinite(hi) and lo <= hi, name+" geometry interval "+key)
            if a+b == "LN":
                previous = original["stats"]["paired_contrasts_points"]["free_natural minus raw"][field]
                for new, prior in zip((lo, hi, interval["delta"]), (*previous["interval_95"], previous["delta"])):
                    close(new, prior, name+" historical geometry interval "+field)
    return {abbr: {**{k: round(100 * metrics[arm][k], 3) for k in ("AP", "AP75", "AP90", "AR100")},
                   "fixed_IoU": round(means[arm][GEOM[0]], 3)} for abbr, arm in ALIASES.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("replay-terminal", "replay", "score-terminal", "score", "prior"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    replay_sha = terminal(a.replay_terminal, "verify-edits", REPLAY_SOURCE)
    score_sha = terminal(a.score_terminal, "score", SCORE_SOURCE)
    replay, score, prior = load(a.replay, replay_sha), load(a.score, score_sha), load(a.prior, OLD)
    check(replay["status"] == "INSTANCE_RSDD_POINTREND_REPLAY_ACCEPTED_V1"
          and score["status"] == "INSTANCE_RSDD_POINTREND_TRANSFER_SCORE_COMPLETE_V1"
          and replay["worker_sha256"] == REPLAY_SOURCE and score["worker_sha256"] == SCORE_SOURCE
          and score["independent_readback_sha256"] == replay_sha, "Scoring parentage")
    check(score["protocol_sha256"] == replay["protocol_sha256"] == LOCK
          and score["fit_weight_sha256"] == replay["fit_weight_sha256"] == FIT
          and score["prelock_commit"] == replay["prelock_commit"] == "11da49604bcc8a7bb871329821754297c159905b"
          and replay["GT_opened"] is False and replay["exact_support_xor_area"] is True
          and replay["exact_replayed_policy_choices"] is True and replay["images"] == 740
          and score["detector_forwards"] == score["optimizer_updates"] == 0, "Fixed protocol")
    check(score["author_test_sha256"] == prior["gt_sha256"] and score["non_simple_ap_retained"] == 1
          and score["evaluator_annotation_ids_rekeyed"] is True, "Shared target and COCO mapping")
    check_edits(replay, score)
    views = score["views"]
    check(set(views) == VIEWS and all(views[v] == views[v+"_deduplicated"] for v in ("gf3", "full")),
          "Five views and exact RGB deduplication equivalence")
    tables = {n: check_view(n, v, prior["views"][n]) for n, v in views.items()}
    main_view = views["gf3"]
    deltas = main_view["paired_contrasts_points"]
    geometry = main_view["geometry"]["paired_contrasts_points"]
    lb = "learned_common minus point_equal"
    gate = deltas[lb]["AP"] > 0 and geometry[lb][GEOM[0]]["interval_95"][0] > 0
    check(gate is score["within_recipe_ranking_criterion"], "Preregistered GF-3 comparison")
    print(json.dumps(dict(status="VERIFIED_CLOSED_AGGREGATES", score_sha256=score_sha,
                          historical_COCO_values_verified=130, COCO_differences_verified=390,
                          balanced_candidates=42829, swaps_per_balanced_policy=329222,
                          gf3_ranking_gate=gate, table_points=tables,
                          gf3_contrasts={k: dict(AP=deltas[k]["AP"], fixed_IoU=g[GEOM[0]]) for k, g in geometry.items()},
                          scope="One HRSID-fitted seed, shared frozen checkpoint, L-imposed edit counts, "
                                "previously evaluated target. Bootstrap intervals concern fixed-match geometry."),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
