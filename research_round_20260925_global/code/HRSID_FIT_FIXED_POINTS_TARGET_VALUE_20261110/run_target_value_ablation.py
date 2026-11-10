"""Compare hard and continuous supervision at identical historical random FIT points."""

from argparse import ArgumentParser
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
from statistics import fmean
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


P253_DIR = "HRSID_FIT_NATIVE_POINT_UPDATE_20261110"
P253_SHA = "7ea206960651b5b2c5874a9a0ddf7601c3b6655314750409c0750afb0fde131b"
REFERENCE_SHA = "70b493edcd9ec247c90995843c07f433329f3ab8c69be6b66d97894b2664eaed"
SUPPORT = {
    "cached_final_query": "bc6bf64a7ff01a716fcd4fe97da9fa292a892bbc5699ff57b46c615581ce4adc",
    "point_update_loss": "99819e406f3a05ae8ad487b54a095fb73058b377bee01b241087443407b99beb",
}
ARMS = ("stepwise_hard", "composed_hard", "raster_occupancy")
ALL = ("anchor", *ARMS)
METRICS = ("polygon_iou", "raster_iou", "center_iou", "ring_bce")
STATUS = "P262_HRSID_FIXED_POINTS_TARGET_VALUE_COMPLETE_V1"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def freeze_target_banks(original, np, hard_c, hard_b, raster, reference_logits, strata, seed):
    """Take the old random bank once and change only its frozen target values."""
    random = original(hard_c, hard_b, raster, reference_logits, strata, seed)["random"]
    indices = random.indices
    require(indices.dtype == np.int64 and not indices.flags.writeable and not random.targets.flags.writeable,
            "Exact immutable historical random point bank required")

    def hard_targets(values):
        selected = np.asarray(values, dtype=np.float64)[indices]
        return SimpleNamespace(indices=indices, targets=np.frombuffer(selected.tobytes(), dtype=np.float64))

    return dict(stepwise_hard=hard_targets(hard_b), composed_hard=hard_targets(hard_c), raster_occupancy=random)


def selected_training(views, np):
    """Reconcile exact indices, target bytes and signed B→C label changes."""
    indices = sha256()
    targets = {arm: sha256() for arm in ARMS}
    count = added = removed = fractional = 0
    for view in views:
        for position, row in enumerate(view["rows"], 1):
            b, c, r = (row["banks"][arm] for arm in ARMS)
            require(all(bank.indices is b.indices and not bank.targets.flags.writeable for bank in (b, c, r)),
                    "Every arm must use identical ordered frozen point indices")
            record = dict(cursor=view["cursor"], row=position, indices=b.indices.tolist())
            indices.update((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
            for arm, bank in zip(ARMS, (b, c, r), strict=True):
                require(bank.targets.shape == b.indices.shape and bank.targets.dtype == np.float64,
                        "Aligned immutable selected target vector required")
                targets[arm].update(bank.targets.astype("<f8", copy=False).tobytes())
            count += len(b.indices)
            added += int(np.count_nonzero((b.targets == 0) & (c.targets == 1)))
            removed += int(np.count_nonzero((b.targets == 1) & (c.targets == 0)))
            fractional += int(np.count_nonzero((r.targets > 0) & (r.targets < 1)))
    require(count == 487 and added + removed > 0 and fractional > 0,
            "Exact 487 historical points and actual hard/continuous target differences required")
    return dict(points=count, stepwise0_composed1=added, stepwise1_composed0=removed,
                hard_target_changed=added + removed, fractional_raster_targets=fractional,
                selected_index_stream_sha256=indices.hexdigest(),
                selected_target_stream_sha256s={arm: value.hexdigest() for arm, value in targets.items()},
                selector="P253_stratified_random_on_composed_C_boundary_strata")


def held_comparison(scores, *, primary):
    require(set(scores) == set(ALL) and all(len(rows) == 8 for rows in scores.values()),
            "Exact eight paired held source groups required")
    means = {arm: {key: fmean(row[key] for row in scores[arm]) for key in METRICS} for arm in ALL}

    def pair(left, right):
        values = [a["polygon_iou"] - b["polygon_iou"]
                  for a, b in zip(scores[left], scores[right], strict=True)]
        return dict(per_source_polygon_iou=values, mean=fmean(values), positive=sum(value > 0 for value in values))

    contrasts = {
        "composed_minus_stepwise": pair("composed_hard", "stepwise_hard"),
        "composed_minus_anchor": pair("composed_hard", "anchor"),
        "composed_minus_raster_occupancy": pair("composed_hard", "raster_occupancy"),
        "raster_occupancy_minus_stepwise": pair("raster_occupancy", "stepwise_hard"),
        "raster_occupancy_minus_anchor": pair("raster_occupancy", "anchor"),
    }
    guard = means["composed_hard"]["center_iou"] - max(means[arm]["center_iou"]
                                                      for arm in ("stepwise_hard", "anchor", "raster_occupancy"))
    report = dict(means=means, contrasts=contrasts, composed_center_iou_guard=guard)
    if primary:
        result = contrasts["composed_minus_stepwise"]
        checks = dict(composed_minus_stepwise_at_least_005=result["mean"] >= .005,
                      positive_held_sources_at_least_5=result["positive"] >= 5,
                      composed_vs_anchor_nonnegative=contrasts["composed_minus_anchor"]["mean"] >= 0,
                      center_iou_guard_at_least_minus_002=guard >= -.002)
        report.update(descriptive_gate_checks=checks, descriptive_gate=all(checks.values()))
    return report


def pinned_bytes(path, expected, limit=64 * 1024):
    require(path.is_absolute() and path.resolve(strict=True) == path, "Exact canonical closed source/reference required")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and sha256(raw).hexdigest() == expected, "Pinned bytes differ: " + path.name)
    return raw


def run(args):
    start = perf_counter()
    require(sys.version_info[:3] == (3, 10, 13) and sys.flags.isolated == sys.flags.utf8_mode == 1
            and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and re.fullmatch(r"[a-f0-9]{40}", args.protocol), "Pinned isolated Windows CPU runtime and protocol required")
    for path in (args.code, args.bundle, args.checkpoint, args.receipts, args.shapely_site):
        require(path.is_absolute() and path.resolve(strict=True) == path, "Canonical sealed input required")
    require(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink(), "Fresh private output required")
    reference = json.loads(pinned_bytes(args.p253_reference, REFERENCE_SHA))
    code = args.code
    old_dir = code / P253_DIR
    source = old_dir / "run_fit_point_updates.py"
    prior = ModuleType("_p262_pinned_original_point_updates")
    prior.__file__ = str(source)
    exec(compile(pinned_bytes(source, P253_SHA), str(source), "exec"), prior.__dict__)

    probe_dir = code / "HRSID_FIT_FROZEN_POINT_PROBE_20261110"
    vector_dir = code / "HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110"
    probe = prior.module(probe_dir / "run_fit_point_probe.py", prior.PROBE_SHA, "_p262_pinned_point_probe")
    pilot = prior.module(vector_dir / "run_fit114_vector_trial.py", probe.PILOT_SHA, "_p262_pinned_geometry_pilot")
    native, np, torch, _ = pilot.install_support(code, args.shapely_site)
    for name, expected in SUPPORT.items():
        pinned_bytes(old_dir / (name + ".py"), expected)
        require(name not in sys.modules, "Fresh exact P253 helper namespace required")
    sys.path[:0] = [str(old_dir), str(probe_dir)]
    loss = import_module("point_update_loss")
    from cached_final_query import projection_parameters
    from rfdetr.models.lwdetr import build_criterion_from_config

    vector_math = prior.module(vector_dir / "vector_population_math.py", prior.MATH_SHA, "_p262_held_vector_math")
    base = probe.accepted_phase_evaluator(code, args.receipts)
    base.RECIPE = code.parent / "configs/RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml"
    base.FIT_AUDIT = code / "HRSID_OFFICIAL_PNG_FIT_AUDIT_20261109/hrsid_png_fit_audit.py"
    for name, expected in dict(base.MODEL_SOURCES, **probe.EXTRA_RUNTIME).items():
        require(sha256(Path(import_module(name).__file__).read_bytes()).hexdigest() == expected,
                "Pinned native runtime: " + name)
    require(version("scipy") == "1.15.3", "Exact frozen CPU assignment backend required")
    model, config, train, _ = base.load_model(args.checkpoint, probe.CHECKPOINT_SHA, torch, args.output.parent,
                                            probe.TRAIN_PROTOCOL)
    model.eval().requires_grad_(False)
    criterion, _ = build_criterion_from_config(config, train)
    original = model.segmentation_head.query_features_proj
    projection_parameters(original)
    before = prior.param_digest(original)
    require(reference["source_checkpoint_sha256"] == probe.CHECKPOINT_SHA
            and reference["fit_bundle_sha256"] == pilot.BUNDLE_SHA and reference["initial_projection_sha256"] == before
            and reference["train_roster"] == list(map(list, prior.TRAIN))
            and reference["held_roster"] == list(map(list, prior.HELD)), "Exact original anchor, FIT data and rosters required")
    args.output.mkdir()

    original_freeze = loss.freeze_row_banks
    with patch.object(loss, "freeze_row_banks", lambda *values: freeze_target_banks(original_freeze, np, *values)):
        views = prior.prepare_views(args, probe, pilot, native, np, torch, model, config, criterion.matcher)
    train_views = [view for view in views if view["train"]]
    selected = selected_training(train_views, np)
    projections = {"anchor": original}
    reports = {}
    for arm in ARMS:
        projections[arm], reports[arm] = prior.train_one(arm, original, train_views, torch, args.output)
        print(json.dumps(dict(event="P262_FIXED_POINT_REAL_PROJECTION_UPDATES", arm=arm, updates=8)), flush=True)
    ledgers = [[{k: event[k] for k in ("cursor", "rows", "points")} for event in reports[arm]["events"]] for arm in ARMS]
    require(ledgers[0] == ledgers[1] == ledgers[2] and sum(row["points"] for row in ledgers[0]) == selected["points"]
            and prior.param_digest(original) == before and all(not p.requires_grad and p.grad is None
                                                               for p in model.parameters())
            and not torch.cuda.is_initialized(), "Equal realized update budget and unchanged frozen carrier required")
    historical_random = reference["arms"]["random"]
    replay = reports["raster_occupancy"]
    require(all(replay[key] == historical_random[key] for key in
                ("optimizer_updates", "events", "initial_sha256", "parameter_sha256")),
            "Continuous occupancy must reproduce original P253 random updates and parameters exactly")
    prior.ALL = ALL
    measured = prior.held_readback(projections, [view for view in views if not view["train"]], np, torch, vector_math)
    require(all(measured[path]["raster_occupancy"] == reference["held_metrics"][path]["random"]
                and measured[path]["anchor"] == reference["held_metrics"][path]["anchor"] for path in ("crop", "direct")),
            "Continuous occupancy and anchor must exactly reproduce all original P253 held metrics")
    result = dict(status=STATUS, protocol=args.protocol, FIT=True, DEV=False, TEST=False, AP=False,
                  optimizer_held_out_seen_by_frozen_source=True, model_forward_calls=len(views), frozen_match_calls=len(views),
                  updated_parameters_per_arm=65792, initial_projection_sha256=before, source_checkpoint_sha256=probe.CHECKPOINT_SHA,
                  fit_bundle_sha256=pilot.BUNDLE_SHA, train_roster=prior.TRAIN, held_roster=prior.HELD, arms=reports,
                  selected_training=selected, held_metrics=measured,
                  held_comparison={path: held_comparison(scores, primary=path == "crop") for path, scores in measured.items()},
                  p253_random_reference_sha256=REFERENCE_SHA, p253_random_numerical_readback_exact=True,
                  p253_random_archive_sha256_match=replay["private_archive_sha256"] == historical_random["private_archive_sha256"],
                  wall_seconds=perf_counter() - start)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with (args.output / "RESULT_SAFE.json").open("xb") as stream:
        require(stream.write(raw) == len(raw), "Complete controlled target-value result required")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest(), forward_calls=len(views),
                          updates_per_arm=8, wall_seconds=result["wall_seconds"])), flush=True)


def main(argv=None):
    cli = ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "bundle", "checkpoint", "receipts", "shapely-site", "output", "p253-reference"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--protocol", required=True)
    run(cli.parse_args(argv))


if __name__ == "__main__":
    main()
