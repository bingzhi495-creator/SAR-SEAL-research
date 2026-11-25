"""Score a naturally closed spatial-pair replay on the bound SSDD development cohort."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType


LOCK = "925c4d5287af89ddac0b6343d53a0edfc32517cf8a1edef451696deb44293d9c"
LEGACY_WORKER = "1d9dd5049ee896e09026f0e128b24e623857d1a6fd0416fbd6c2db8701ac6f64"
LEGACY_RENDER = "08349b799baf91d1982259f50724ad29c6ea8c617ddf429e964dec0aafe18a53"
LEGACY_RENDER_TERMINAL = "cef5097ecb3caef82657649de3f18559acbcb63ec0297c2614522f9dff1b56d5"
LEGACY_SCORE = "5d94e77e9d1d1565648d24a4b44970bfe541894f9530bfe640d1f9a5818f6863"
LEGACY_SCORE_TERMINAL = "6bee2e2806496758169a86fbc5a32701c09697f6b430fcd301b8e612dfce205c"
FIT_WORKER = "280cb7547a3d85d02806ab62dd05519c6dd81626d6903e7fb0a2a9862dd47e47"
RENDER_STATUS = "SPSD_PSEG_SPATIAL_PAIR_RENDER_COMPLETE"
FIT_STATUS = "SPSD_PSEG_SPATIAL_PAIR_FIT_COMPLETE"
PAIR_SOURCES = {
    "joint_selector_sha256": ("select_pairs.py", "c874cfd8608691203db5aab80a16ff7d15b7ce5d4a51c7a9a3d0c6f22daefb0c"),
    "pair_graphs_sha256": ("pair_iou.py", "a904c262f0ce554fb6fc6b6e92864f29566394fdd9f21981bd9a3858cd0791f2"),
    "joint_pair_iou_sha256": ("paired_expectation.py", "1c1c772afd476dcf227c176a2f46706ae83a801b504bab7c3dce0f1952b686c7"),
}
MODELS = ("expectation", "local", "nonspatial")
PAIR_ARMS = tuple(f"{model}_{split}" for model in MODELS for split in ("exact", "fixed"))
COMPARISONS = (
    ("local_exact", "expectation_exact"),
    ("local_exact", "nonspatial_exact"),
    ("nonspatial_exact", "expectation_exact"),
    ("local_fixed", "expectation_fixed"),
    ("local_fixed", "nonspatial_fixed"),
    ("nonspatial_fixed", "expectation_fixed"),
    *((f"{model}_exact", f"{model}_fixed") for model in MODELS),
)
PRIMARY = COMPARISONS[:2]


def load_legacy(path):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != LEGACY_WORKER:
        raise ValueError("Exact released independent scorer required")
    module = ModuleType("independent_expectation")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), vars(module))
    return module


def contrast_key(treatment, reference):
    return f"{treatment}__vs__{reference}"


def promotion(refs, exposure):
    outcomes = {}
    for treatment, reference in PRIMARY:
        key = contrast_key(treatment, reference)
        outcomes[key] = {
            name: value["iou_points"] >= .15 and value["percentile95_points"][0] > 0
            and value["ap_points"] >= -.10
            for name, ref in refs.items() for value in (ref["contrasts"][key],)
        }
    changed = exposure[contrast_key(*PRIMARY[0])]
    enough_changed = changed["hulls"] >= 20 and changed["groups"] >= 8
    return {
        "threshold_passes": outcomes,
        "coverage_pass": enough_changed,
        "advance_source_diagnostic": enough_changed and set(refs) == {"pseg", "spsd"}
        and all(all(results.values()) for results in outcomes.values()),
    }


def verify_render(args, c, legacy):
    root = args.root
    render, render_sha = c.closed(root / "render/RESULT_SAFE.json", root / "observed_render/TERMINAL.json",
                                  root / "run_pairs.py")
    terminal = json.loads((root / "observed_render/TERMINAL.json").read_bytes())
    c.need(terminal["phase"] == "prepare-edits" and render["status"] == RENDER_STATUS
           and render["protocol_sha256"] == LOCK and render["exact_expected_iou_sha256"] == legacy.CORE
           and render["fit_labels_opened"] is render["source_gate_labels_opened"] is render["historical_test_opened"] is False
           and render["no_new_occupancy_updates"] is True
           and render["optimizer_updates"] == render["detector_forwards"] == 0,
           "Originally observed label-free spatial-pair writer required")
    for field, (name, expected) in PAIR_SOURCES.items():
        c.need(render[field] == expected, "Original pair graph and exact selector lineage")
        c.pin(root / name, expected)

    fit_root = root / "fit"
    fit, fit_sha = c.closed(
        fit_root / "actual/RESULT_SAFE.json", fit_root / "observed_fit/TERMINAL.json", fit_root / "fit_pairs.py",
        render["fit_result_sha256"], render["fit_terminal_sha256"], FIT_WORKER,
    )
    fit_terminal = json.loads(c.pin(fit_root / "observed_fit/TERMINAL.json", render["fit_terminal_sha256"]))
    c.need(fit_terminal["phase"] == "prepare-edits" and fit["status"] == FIT_STATUS
           and fit["protocol_sha256"] == LOCK and fit["pair_core_sha256"] == render["pair_graphs_sha256"]
           and fit["fit_only"] is fit["qualified_for_fixed_rerender"] is True
           and fit["source_gate_labels_opened"] is fit["historical_test_opened"] is fit["transfer_labels_opened"] is False
           and fit["full_fit_kappa_state_sha256"] == render["fit_kappa_sha256"] and fit_sha == render["fit_result_sha256"],
           "Previously admitted independently closed FIT gate")
    c.pin(fit_root / "actual/KAPPA_PRIVATE.json", render["fit_kappa_sha256"])
    return render, render_sha, c.digest((root / "observed_render/TERMINAL.json").read_bytes())


def verify_independent(args, c, legacy):
    root = args.independent_run
    worker = root / "run_expected.py"
    render, render_sha = c.closed(
        root / "render/RESULT_SAFE.json", root / "observed_render/TERMINAL.json", worker,
        LEGACY_RENDER, LEGACY_RENDER_TERMINAL, LEGACY_WORKER,
    )
    score, _ = c.closed(
        root / "score/RESULT_SAFE.json", root / "observed_score/TERMINAL.json", worker,
        LEGACY_SCORE, LEGACY_SCORE_TERMINAL, LEGACY_WORKER,
    )
    c.need(render["status"] == legacy.RENDER_STATUS and score["status"] == "SPSD_PSEG_EXPECTED_IOU_SIGNED_SCORE_COMPLETE"
           and render["protocol_sha256"] == score["protocol_sha256"] == legacy.LOCK
           and render["exact_expected_iou_sha256"] == score["exact_expected_iou_sha256"] == legacy.CORE
           and score["render_result_sha256"] == render_sha and score["historical_slot_mask_metric_parity"] is True
           and score["all_actual_signed_rates_verified"] is True and score["fit_or_historical_test_opened"] is False,
           "Exact archived independent expectation mask and score closures")
    return render, score


def patched_score_support(source):
    # Reuse the older common-match routine for each treatment in the new comparison matrix.
    anchor = 'new=masks["variable_iou"]'
    if source.count(anchor) != 1:
        raise ValueError("Bound common-hull exposure source changed")
    return source.replace(anchor, "new=masks[EXPOSURE_TREATMENT]")


def verify_pair_nesting(c, images):
    changes = {contrast_key(*pair): 0 for pair in COMPARISONS}
    for image in images.values():
        for slot in image["candidates"]:
            k = slot["quota"]
            if not k:
                continue
            scores = slot["variable_predicted_iou_gain"]
            counts = slot["variable_signed_counts"]
            masks = slot["changes"]
            for model in MODELS:
                full, fixed = f"{model}_exact", f"{model}_fixed"
                c.need(counts[fixed] == [k, k] and min(scores[full]) + 5e-12 >= min(scores[fixed]),
                       "Same actual signed fixed quota and nested exact objective")
                if counts[full] == [k, k]:
                    c.need(masks[full] == masks[fixed], "Equal allocation retains the exact fixed decision")
            for pair in COMPARISONS:
                changes[contrast_key(*pair)] += int(masks[pair[0]] != masks[pair[1]])
    return changes


def common_exposures(orig, truths, matches, roster, groups, c, expected):
    options = {}
    for treatment, reference in COMPARISONS:
        options.setdefault(treatment, []).append(reference)
    by_pair = {}
    for treatment, references in options.items():
        orig.EXPOSURE_TREATMENT = treatment
        orig.CONTROLS = tuple(references)
        _, details = orig.matched_pairs(truths, matches, roster, groups)
        c.need(details["counts"] == expected, "Exact inherited common native hull identities")
        by_pair.update({contrast_key(treatment, ref): value for ref, value in details["exposure"].items()})
    return by_pair


def score(args, c, cfg, legacy):
    import numpy as np

    start = perf_counter()
    render, render_sha, render_terminal_sha = verify_render(args, c, legacy)
    prior_render, prior_score = verify_independent(args, c, legacy)
    base, base_sha = c.closed(cfg.base_render / "render/RESULT_SAFE.json", cfg.base_render / "RENDER_TERMINAL.json",
                              cfg.base_render / "render_gate.py", legacy.BASE_RENDER_RESULT, legacy.BASE_RENDER_EXIT,
                              legacy.BASE_RENDERER)
    legacy_score, legacy_score_sha = c.closed(
        cfg.base_score / "score/RESULT_SAFE.json", cfg.base_score / "SCORE_TERMINAL.json",
        cfg.base_score / "score_gate.py", legacy.BASE_SCORE_RESULT, legacy.BASE_SCORE_EXIT, legacy.BASE_SCORE,
    )
    c.need(legacy_score["render_result_sha256"] == base_sha and legacy_score["exact_all_archive_score_parity"] is True,
           "Exact already accepted original score and render")
    orig = c.load(cfg.base_score / "score_gate.py", legacy.BASE_SCORE, patched_score_support)
    auth = c.load(cfg.authority, orig.ARCH_AUTHORITY)
    old, _, scorer, reader, anchors, archived, groups, _, _ = auth.archive(cfg)
    c.pin(cfg.group_protocol, orig.GROUP)
    point = json.loads(c.pin(cfg.points / "actual_retry/RESULT_SAFE.json", old.POINTS))
    c.need(render["point_terminal_sha256"] == prior_render["point_terminal_sha256"]
           == base["point_terminal_sha256"] == anchors["point_terminal_sha256"]
           and render["archive_bce_tensor_sha256"] == orig.ARCH_BCE
           and render["archive_render_sha256"] == orig.ARCH_RENDER and render["archive_score_sha256"] == orig.ARCH_SCORE
           and render["decoded_groups"] == 180,
           "Same exact labels-excluded carrier, quotas and originally fitted occupancy")

    new = legacy.records(c, args.root / "render/GATE_RENDER_PRIVATE.jsonl", render["render_stream_sha256"])
    old_rows = legacy.records(c, args.independent_run / "render/GATE_RENDER_PRIVATE.jsonl",
                              prior_render["render_stream_sha256"])
    legacy.require_old_parity(c, new, old_rows)
    del old_rows
    orig.CUES = (*orig.CUES, *PAIR_ARMS)
    mech = c.load(cfg.mechanism, orig.MECHANISM)
    totals, areas = orig.signed_support(cfg, point, new, archived, groups, render, auth.REFERENCE_ARMS, mech)
    changes = verify_pair_nesting(c, new)
    order = sorted(set(groups.values()))
    c.need(len(order) == 180, "Original decoded image-group count")
    print("Independent prelabel replay verified the masks, inherited arms, signed budgets and candidate order.", flush=True)

    truths = reader.labels(cfg, new)
    c.need(set(truths) == {"pseg", "spsd"}, "Exact two released source-development references")
    roster = {iid: row["candidates"] for iid, row in new.items()}
    arms = (*auth.REFERENCE_ARMS, *orig.CUES)
    draws = np.random.Generator(np.random.PCG64(5297)).integers(180, size=(4096, 180))
    refs, matches = {}, {}
    for reference, truth in truths.items():
        inherited = prior_score["references"][reference]
        metrics = {}
        for arm in arms:
            metrics[arm], native = scorer.coco_score(truth, roster, arm)
            if arm == "native":
                matches[reference] = native
        c.need(len(matches[reference]) == inherited["original_matches"]
               and len(truth["annotations"]) == inherited["author_hulls"],
               "Exact original author hulls, including the inherited native match count")
        values = scorer.geometry(arms, truth, roster, matches[reference], groups, order)
        points = dict(zip(arms, values.mean(axis=0).tolist(), strict=True))
        c.need(all(abs(metrics[arm][key] - value) <= 1e-12
                   for arm, result in inherited["metrics"].items() for key, value in result.items())
               and all(abs(points[arm] - value) <= 1e-12 for arm, value in inherited["fixed_iou_points"].items()),
               "Exact metric readback of every inherited arm, including both old expectation arms")
        contrasts = {contrast_key(*pair): orig.contrast(arms, values, metrics, draws, *pair) for pair in COMPARISONS}
        refs[reference] = {
            "metrics": metrics, "fixed_iou_points": points, "contrasts": contrasts,
            "original_matches": inherited["original_matches"], "author_hulls": inherited["author_hulls"],
        }
        print(f"Completed source-development score under {reference}.", flush=True)

    common = prior_score["common_native_hulls"]["counts"]
    exposure = common_exposures(orig, truths, matches, roster, groups, c, common)
    args.output.mkdir()
    c.closeout(
        args.output / "RESULT_SAFE.json", Path(__file__), "SPSD_PSEG_SPATIAL_PAIR_SCORE_COMPLETE",
        render_result_sha256=render_sha, render_terminal_sha256=render_terminal_sha,
        render_stream_sha256=render["render_stream_sha256"],
        fit_result_sha256=render["fit_result_sha256"], fit_terminal_sha256=render["fit_terminal_sha256"],
        independent_result_sha256=LEGACY_SCORE, independent_render_result_sha256=LEGACY_RENDER,
        original_variable_score_sha256=legacy_score_sha,
        independent_exact_mask_parity=True, independent_exact_metric_parity=True, exact_signed_budget_parity=True,
        target_scope="reused source development", fit_or_historical_test_opened=False,
        external_acquisition_evaluation=False, detector_forwards=0, optimizer_updates=0,
        counts=render["counts"], decoded_groups=len(order), bootstrap_seed=5297, bootstrap_draws=len(draws),
        references=refs, common_native_hull_counts=common, common_mask_change_exposure=exposure,
        positive_budget_changed_slots=changes, arm_totals=totals, net_area_change_histogram=areas,
        primary=promotion(refs, exposure), render_reported_seconds=render.get("adapter_seconds"),
        elapsed_seconds=perf_counter() - start,
    )


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("score",))
    for name in ("inputs", "helpers", "root", "independent-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    legacy = load_legacy(args.independent_run / "run_expected.py")
    c = legacy.helper(args.helpers)
    c.LOCK = LOCK
    c.pin(args.root / "PRELOCK.md", LOCK)
    c.need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
           and args.output.resolve() == (args.root / args.phase).resolve() and not args.output.exists(),
           "Fresh isolated CPU output in the prescribed score phase")
    score(args, c, c.settings(args.inputs, legacy.SCORE_KEYS), legacy)


if __name__ == "__main__":
    main()
