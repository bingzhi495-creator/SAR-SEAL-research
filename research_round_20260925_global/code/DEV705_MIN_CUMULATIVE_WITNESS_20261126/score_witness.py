"""Score the prelocked DEV705 witness after an observed no-label coverage pass."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coverage as witness

COVERAGE_CODE_SHA = "2309163796955814f5c2a3921358ae15c741fbbde25062698291e2c036a5e2f2"
OLD_SCORE_SHA = "964260bd342ae132d14a95a6352e5cf9b904c3f120e288f264f2d2235cb1827f"
OLD_SCORE_TERMINAL_SHA = "963bc91c04704263e0c9a95f17d81e5a721e56adae67b1c4fb709b1d6567f920"
METRICS = ("AP", "AP50", "AP75", "AP90", "APs", "AR100")
CTX = None


def operators(a):
    deps = a.code / "LEARNED_BOUNDED_PAIRING_20261126" / "hrsid"
    common_src = witness.pin(deps / "bound_common.py", witness.COMMON_SHA)
    spec = spec_from_file_location("bound_common", str(deps / "bound_common.py"))
    common = module_from_spec(spec)
    sys.modules[spec.name] = common
    exec(compile(common_src, str(deps / "bound_common.py"), "exec"), vars(common))
    archived = common.load(deps / "score_dev.py", witness.READ_ARCHIVE_SHA, "_witness_archived_score")
    return common, archived


def census_and_select(prepared, seeds):
    """Reuse the original I slot in the inherited scorer, preserving F and L."""
    union = set()
    union_groups = set()
    counts = {}
    for seed in seeds:
        count = 0
        changed_groups = set()
        for native, _, _, group, all_edits in prepared:
            for rank, (f, i, l) in all_edits[seed].items():
                can_cap, cf, cl, ci = witness.evaluate(f, i, l)
                choose_independent = not can_cap and ci <= cf and ci <= cl
                if choose_independent:
                    if set(i) == set(l):
                        raise ValueError("Witness does not differ from L")
                    union.add((native["image_id"], rank))
                    union_groups.add(group)
                    changed_groups.add(group)
                    count += 1
                else:
                    all_edits[seed][rank] = (f, l, l)
        counts[str(seed)] = (count, len(changed_groups))
    return counts, len(union), len(union_groups)


def init_score(a, prepared, gt, empties):
    global CTX
    _, archived = operators(a)
    archived.init_coco(a, prepared, gt, empties)
    CTX = archived


def score_w(seed):
    _, _, metrics = CTX.score_coco((seed, "SU=F"))
    return seed, metrics


def deltas(coco, old, seeds, new="SU=F"):
    return {view: {str(seed): {
        metric: 100 * (coco[seed][new][view][metric] - old[str(seed)][view][metric])
        for metric in METRICS if metric in coco[seed][new][view]}
        for seed in seeds} for view in ("primary", "sensitivity")}


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for p in ("code", "source", "bounded", "coverage", "coverage-terminal", "protocol", "output"):
        parser.add_argument("--" + p, required=True, type=Path)
    for p in ("coverage-sha256", "coverage-terminal-sha256", "protocol-sha256"):
        parser.add_argument("--" + p, required=True)
    a = parser.parse_args()
    start = perf_counter()
    if not (sys.flags.isolated and sys.dont_write_bytecode
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and not a.output.exists() and sys.version_info[:3] == (3, 10, 13)
            and all(version(dep) == value for dep, value in witness.EXPECTED_RUNTIME.items())):
        raise ValueError("Requires isolated frozen CPU runtime and fresh output directory")

    common, archived = operators(a)
    require = common.require
    witness.pin(a.protocol, a.protocol_sha256)
    witness.pin(Path(witness.__file__), COVERAGE_CODE_SHA)
    cover = json.loads(witness.pin(a.coverage, a.coverage_sha256))
    terminal = json.loads(witness.pin(a.coverage_terminal, a.coverage_terminal_sha256))
    require(terminal["phase"] == "coverage" and terminal["actual_exit"] == 0
            and terminal["output_footer_verified"] is True
            and terminal["result_sha256"] == a.coverage_sha256
            and terminal["worker_sha256"] == cover["worker_sha256"] == COVERAGE_CODE_SHA
            and cover["status"] == "DEV705_CUMULATIVE_WITNESS_COVERAGE_COMPLETE"
            and cover["protocol_sha256"] == a.protocol_sha256
            and cover["labels_opened"] is cover["mask_predictions_modified"] is False
            and cover["detector_forwards"] == cover["optimizer_updates"] == 0
            and cover["bounded_render_sha256"] == witness.BOUNDED_RENDER_SHA
            and cover["bounded_render_terminal_sha256"] == witness.BOUNDED_TERMINAL_SHA
            and cover["necessary_coverage_gate_passed"] is True
            and cover["scoring_status"] == "ELIGIBLE_NOT_EXECUTED"
            and cover["changed_candidates_union_across_heads"] >= 20
            and cover["changed_prefix_groups_union_across_heads"] >= 8,
            "No natural label-free coverage closure or necessary gate")
    bound_score = common.closed(a.bounded, "score", OLD_SCORE_SHA, OLD_SCORE_TERMINAL_SHA, witness.READ_ARCHIVE_SHA)
    require(bound_score["phase"] == "score" and bound_score["status"] == common.SCORE_STATUS
            and bound_score["render_result_sha256"] == witness.BOUNDED_RENDER_SHA
            and bound_score["render_terminal_sha256"] == witness.BOUNDED_TERMINAL_SHA
            and bound_score["original_score_sha256"] == common.SCORE
            and bound_score["exact_old_coco_all_metrics_and_geometry"] is True
            and bound_score["fixed_matching_counts"] == archived.REFERENCE_COUNTS
            and bound_score["fixed_matching_groups"] == 27
            and bound_score["bootstrap_seed"] == 5217 and bound_score["bootstrap_draws"] == 4096,
            "Archived original comparators or fixed geometry differ")
    common.closed(a.source, "capture", common.CAPTURE, common.CAP_TERMINAL, common.CAPTURE_CODE)
    common.closed(a.source, "replay", common.REPLAY, common.REP_TERMINAL, common.REPLAY_CODE)
    a.intake, a.capture, a.replay = [a.source / stem for stem in ("intake", "capture", "replay")]
    a.render = a.bounded / "render"
    render = json.loads(witness.pin(a.render / "RESULT_SAFE.json", witness.BOUNDED_RENDER_SHA))
    old, ops, sparse = archived.score_operators(a)
    old.admission(a, ops)
    prepared, totals = archived.read_archive(a, old, ops, sparse, render)
    counts, changed, groups = census_and_select(prepared, common.SEEDS)
    require(dict(totals) == cover["population"] == bound_score["totals"]
            and changed == cover["changed_candidates_union_across_heads"]
            and groups == cover["changed_prefix_groups_union_across_heads"]
            and all((cover["head_census"][s]["changed"], cover["head_census"][s]["changed_groups"]) == v
                    for s, v in counts.items()), "Realized no-label W coverage disagrees with observed run")

    a.output.mkdir()
    common.write_result(a.output / "PRELABEL_SAFE.json", {
        "status": "DEV705_CUMULATIVE_WITNESS_PRESCORE_QUALIFIED",
        "coverage_sha256": a.coverage_sha256, "labels_opened": False,
        "changed_candidates_union_across_heads": changed,
        "changed_prefix_groups_union_across_heads": groups,
    })
    gt = ops.native.dev_file("DEV_ORIGINAL_GT_PRIVATE.json")
    empty_rows = ops.native.dev_file("DEV_EMPTY_RASTERS_PRIVATE.json")
    empties = {r["annotation_id"] for r in empty_rows if r["kind"] == "EMPTY_COCO_RASTER"}
    require(len(empties) == len(empty_rows) == 4
            and len(gt["annotations"]) == len({v["id"] for v in gt["annotations"]}) == 1782
            and [c["id"] for c in gt["categories"]] == [1]
            and [r["id"] for r in gt["images"]] == [row[0]["image_id"] for row in prepared],
            "Exact historical DEV705 author reference differs")

    original = bound_score["coco"]
    coco = {s: {arm: original[str(s)][arm] for arm in ("N", "SF", "L")} for s in common.SEEDS}
    with ProcessPoolExecutor(max_workers=3, initializer=init_score, initargs=(a, prepared, gt, empties)) as pool:
        futures = [pool.submit(score_w, s) for s in common.SEEDS]
        values, exposed, fixed_groups = archived.fixed_geometry(ops, prepared, gt)
        for future in futures:
            seed, metrics = future.result()
            coco[seed]["SU=F"] = metrics
            print(json.dumps(dict(event="W_COCO_COMPLETE", head_seed=seed)), flush=True)
    individual, means, contrasts, _ = archived.summarize(coco, values, exposed, fixed_groups)
    for seed, entry in individual.items():
        for arm in ("N", "SF", "L"):
            for metric, value in entry["arms"][arm].items():
                reference = bound_score["per_seed_fixed_geometry"][str(seed)]["arms"][arm][metric]
                require(abs(value - reference) <= 1e-12, "Frozen anchor fixed-match geometry changed")
        entry["arms"]["W"] = entry["arms"].pop("SU=F")
        entry["contrasts"]["W_minus_L"] = entry["contrasts"].pop("I_minus_L")
        entry["contrasts"]["W_minus_F"] = entry["contrasts"].pop("I_minus_F")
    means["W"] = means.pop("SU=F")
    primary = contrasts.pop("I_minus_L")
    contrasts["W_minus_L"] = primary
    contrasts["W_minus_F"] = contrasts.pop("I_minus_F")
    common.write_result(a.output / "RESULT_SAFE.json", {
        "status": "DEV705_CUMULATIVE_WITNESS_SCORE_COMPLETE",
        "worker_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": a.protocol_sha256, "coverage_sha256": a.coverage_sha256,
        "coverage_terminal_sha256": a.coverage_terminal_sha256,
        "original_bounded_score_sha256": OLD_SCORE_SHA,
        "population": dict(totals), "seeds": list(common.SEEDS),
        "primary_annotations": 1782, "empty_raster_sensitivity_annotations": 1778,
        "fixed_matching_counts": archived.REFERENCE_COUNTS, "fixed_matching_groups": len(fixed_groups),
        "bootstrap_draws": 4096, "bootstrap_seed": 5217,
        "bootstrap_unit": "source prefix; common heads averaged within each fixed hull",
        "coverage_gate_passed": True,
        "matched_coverage_gate_passed": primary["changed_hulls"] >= 20 and primary["changed_groups"] >= 8,
        "original_anchor_fixed_geometry_verified": True,
        "contrasts": contrasts, "mean_primary_coco_points": means, "per_head_fixed_geometry": individual,
        "W_coco_by_head": {s: coco[s]["SU=F"] for s in common.SEEDS},
        "W_minus_L_coco_points": deltas(coco, {str(s): original[str(s)]["L"] for s in common.SEEDS}, common.SEEDS),
        "W_minus_F_coco_points": deltas(coco, {str(s): original[str(s)]["SF"] for s in common.SEEDS}, common.SEEDS),
        "detector_forwards": 0, "optimizer_updates": 0,
        "elapsed_seconds": perf_counter() - start,
    })


if __name__ == "__main__":
    main()
