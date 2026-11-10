"""Audit stock stepwise and actual one-pass FIT masks with no model execution."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace


HERE = Path(__file__).resolve().parent
REPLAY = HERE.parent / "HRSID_FIT_RESIZE_PHASE_REPLAY_20261109"
DIRECTION_WORKER = HERE / "ssdd_fit_direction_dgx.py"
DIRECTION_WORKER_SHA = "514f79871f27178aaa6151a81f959e8ace2f0f2c0cb205804d4c5206926226d1"
ONE_PASS_SHA = "05cad566f9b5cc761e91b5283585a5392a7450c42c3133349e6dc21e8bf7496d"
PINS = {
    "fit_query_export": (HERE.parent / "HRSID_FIT_NATIVE_QUERY_RETENTION_20261109/fit_query_export.py",
                         "6a2d7e38d69b82ef3df790a5c359c408b1e74778c39672b47a62c4dc86cabbf5"),
    "reference_resize": (REPLAY / "reference_resize.py", "3ca1d4a6400324c13c777cb4d5f7627713bc625580195d8e05363950872cc8df"),
    "trace_resize": (REPLAY / "trace_resize.py", "902a0e3640a319ba729172847f49ce582f3c0e3954593862ed94971727dc309f"),
    "replay_fit": (REPLAY / "replay_fit.py", "4f95f1b5b84fcf89b005f999cfb8c5fd647a7b262aedd2622860cf8785a14157"),
    "common_mask_supervision": (REPLAY / "common_mask_supervision.py",
                                "59ea82e24353eeb4a14115058d5115333a4e7ea517a0abef581bee137a9a3a57"),
    "paired_fit_transform": (REPLAY / "paired_fit_transform.py",
                             "64f3f63c6ac89b66578f0ada42fa3eb8b4db3b2f1ae2ea9113debe6ab9b270e0"),
    "one_pass_fit_transform": (REPLAY / "one_pass_fit_transform.py", ONE_PASS_SHA),
}
STATUS = "SSDD_FIT741_ONE_PASS_STOCK_AUDIT_COMPLETE"
PROGRESS = "SSDD_FIT_ONE_PASS_AUDIT_PROGRESS"
HISTORICAL_B = {
    "direct": dict(raw_rows=801, b_present=801, b_nonempty=801),
    "crop": dict(raw_rows=860, b_present=644, b_nonempty=644),
}
COUNTS = (
    "raw_rows", "b_present", "c_present", "b_nonempty", "c_nonempty", "b_empty", "c_empty",
    "c_only_present", "b_only_nonempty", "c_only_nonempty", "common_nonempty", "common_changed_masks",
    "common_changed_pixels", "common_small_rows", "common_small_changed_masks",
    "b_conditioned_nonempty", "b_conditioned_changed_masks",
)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def load_stage_support():
    require(DIRECTION_WORKER.resolve(strict=True) == DIRECTION_WORKER, "Canonical frozen SSDD CPU support source")
    with DIRECTION_WORKER.open("rb") as stream:
        raw = stream.read(64 * 1024 + 1)
    require(0 < len(raw) <= 64 * 1024 and sha256(raw).hexdigest() == DIRECTION_WORKER_SHA,
            "Exact frozen SSDD direction worker support")
    support = ModuleType("_ssdd_one_pass_stage_support")
    support.__file__ = str(DIRECTION_WORKER)
    exec(compile(raw, str(DIRECTION_WORKER), "exec"), support.__dict__)
    return support


def load_audit_helpers():
    """Load the ordinary candidate and its exact transitive FIT dependencies."""
    return _load_audit_helpers(load_stage_support())


def _load_audit_helpers(support):
    namespace = {}
    for name, (path, expected) in PINS.items():
        namespace[name] = support.pinned_module(path, expected, namespace)
    return SimpleNamespace(paired=namespace["paired_fit_transform"], one=namespace["one_pass_fit_transform"],
                           common=namespace["common_mask_supervision"], trace=namespace["trace_resize"],
                           replay=namespace["replay_fit"])


def inverse_axis(trace, axis, source_size):
    """Track output centers backwards through each operation in integer rationals."""
    size = trace[-1]["out_hw"][axis]
    numerators = [2 * index + 1 for index in range(size)]
    denominator = 2
    for step in reversed(trace):
        before, after = step["in_hw"][axis], step["out_hw"][axis]
        require(after == size and type(before) is type(after) is int and min(before, after) > 0,
                "Consistent original axis sizes required for independent replay")
        kind = step["op"]
        if kind == "resize":
            numerators = [value * before for value in numerators]
            denominator *= after
        elif kind == "crop":
            offset = step["top" if axis == 0 else "left"]
            require(type(offset) is int and 0 <= offset <= before - after, "Original crop inside its input canvas")
            numerators = [value + offset * denominator for value in numerators]
        elif kind == "hflip":
            require(before == after and type(step["applied"]) is bool, "Exact original horizontal flip trace")
            if axis == 1 and step["applied"]:
                numerators = [before * denominator - value for value in numerators]
        else:
            raise ValueError("Unsupported original geometry operation: " + kind)
        size = before
    indices = [value // denominator for value in numerators]
    require(size == source_size and all(0 <= value < size for value in indices), "Independent source sampling bounds")
    return indices


def audit_view(native_module, stock_transform, image, raw_target, seed, *, set_seed, helpers):
    """Audit one prepared stock view without changing the caller's random stream."""
    import numpy as np
    import torch

    paired, common = helpers.paired, helpers.common
    row_id = common.ROW_ID
    original = paired._rng()
    try:
        raw = paired._prepare(image, raw_target)
        set_seed(seed)
        entry = paired._rng()
        b_input = {key: value.clone() for key, value in raw.items()}
        if not paired._ACTIVE.acquire(blocking=False):
            raise RuntimeError("Concurrent FIT transform audit is unsupported")
        try:
            with helpers.trace.capture_transform(native_module, mode="center") as trace:
                b_image, b = stock_transform(image.copy(), b_input)
        finally:
            paired._ACTIVE.release()
        b_rng = paired._rng_bytes(paired._rng())
        paired._check_output(b_image, b, raw)
        steps = helpers.replay.normalized_trace(trace)

        carrier_ids, observations = [], []

        def observe_carrier(image_input, target_input):
            result_image, carrier = stock_transform(image_input, target_input)
            carrier_ids.append(carrier[row_id].clone())
            return result_image, carrier

        paired._restore(entry)
        c_transform = helpers.one.OnePassFitTransform(native_module, observe_carrier, audit=observations.append)
        c_image, c = c_transform(image, raw)
        require(len(observations) == len(carrier_ids) == 1, "Exactly one natural C transform invocation required")
        witness = observations[0]
        require(b_image.shape == c_image.shape and torch.equal(paired._tensor_bytes(b_image), paired._tensor_bytes(c_image))
                and paired._rng_bytes(paired._rng()) == b_rng
                and witness["rng_before_sha256"] == sha256(paired._rng_bytes(entry)).hexdigest()
                and witness["rng_after_sha256"] == sha256(b_rng).hexdigest()
                and witness["trace_sha256"] == sha256(paired._encoded(trace)).hexdigest(),
                "Stock B and actual natural C image, geometry or random stream differs")
        c[row_id] = carrier_ids[0]
        b_ids, c_ids = b[row_id], c[row_id]
        require(witness["retained_row_ids_sha256"] == paired._tensor_digest(((row_id, c_ids),))
                and witness["masks_sha256"] == paired._training_mask_digest(c["masks"]),
                "Actual natural C masks and original row IDs must match its witness")
        positions = torch.searchsorted(c_ids, b_ids)
        require(not (positions >= len(c_ids)).any() and torch.equal(c_ids[positions], b_ids)
                and all(torch.equal(b[key], c[key][positions]) for key in ("boxes", "labels", "area", "iscrowd"))
                and all(torch.equal(b[key], c[key]) for key in common.IMAGE_FIELDS),
                "Every stock B row must match the positive-box C carrier exactly")
        c_b = helpers.one.substitute_one_pass_masks(raw["masks"], b, trace)
        rows = inverse_axis(trace, 0, raw["masks"].shape[-2])
        cols = inverse_axis(trace, 1, raw["masks"].shape[-1])
        raw_np = raw["masks"].numpy()
        oracle = raw_np[c_ids.numpy()[:, None, None], np.asarray(rows)[None, :, None], np.asarray(cols)[None, None, :]]
        require(np.array_equal(c["masks"].numpy(), oracle)
                and np.array_equal(c_b["masks"].numpy(), oracle[positions.numpy()]),
                "Actual natural C and B-conditioned C must equal the independent inverse-trace oracle")

        shared_b, shared_c = common.common_targets(b, c)
        changed = shared_b["masks"] != shared_c["masks"]
        changed_rows = changed.flatten(1).any(dim=1)
        b_live = b_ids[b["masks"].flatten(1).any(dim=1)]
        c_live = c_ids[c["masks"].flatten(1).any(dim=1)]
        small = raw["area"][shared_b[row_id]] < 1024
        path = "crop" if any(step["kind"] == "crop" for step in steps) else "direct"
        shared_count = len(shared_b[row_id])
        return dict(
            image_id=int(raw["image_id"].item()), seed=seed, path=path, flip=steps[-1]["kind"] == "hflip", steps=steps,
            raw_rows=len(raw[row_id]), b_present=len(b_ids), c_present=len(c_ids),
            b_nonempty=len(b_live), c_nonempty=len(c_live), b_empty=len(b_ids) - len(b_live), c_empty=len(c_ids) - len(c_live),
            c_only_present=len(c_ids) - len(b_ids), b_only_nonempty=len(b_live) - shared_count,
            c_only_nonempty=len(c_live) - shared_count, common_nonempty=shared_count,
            common_changed_masks=int(changed_rows.sum()), common_changed_pixels=int(changed.sum()),
            common_small_rows=int(small.sum()), common_small_changed_masks=int((changed_rows & small).sum()),
            b_conditioned_nonempty=int(c_b["masks"].flatten(1).any(dim=1).sum()),
            b_conditioned_changed_masks=int((b["masks"] != c_b["masks"]).flatten(1).any(dim=1).sum()),
            c_natural_oracle_exact=True, c_b_conditioned_oracle_exact=True,
            direct_fixed_exact=not bool(changed_rows.any()) if path == "direct" else None,
            image_sha256=witness["image_sha256"], trace_sha256=witness["trace_sha256"],
            rng_before_sha256=witness["rng_before_sha256"], rng_after_sha256=witness["rng_after_sha256"],
            b_masks_sha256=paired._training_mask_digest(b["masks"]), c_masks_sha256=witness["masks_sha256"],
            b_row_ids_sha256=paired._tensor_digest(((row_id, b_ids),)), c_row_ids_sha256=witness["retained_row_ids_sha256"],
            c_b_conditioned_masks_sha256=paired._training_mask_digest(c_b["masks"]),
        )
    finally:
        paired._restore(original)


def summarize_audits(rows):
    """Summarize all rows, preserving branch-specific failures and membership."""
    rows = list(rows)
    require(rows and all(row["path"] in ("direct", "crop") for row in rows), "Complete stock FIT audit rows required")

    def totals(selected):
        return dict(views=len(selected), **{key: sum(row[key] for row in selected) for key in COUNTS})

    by_path = {path: totals([row for row in rows if row["path"] == path]) for path in ("direct", "crop")}
    direct_changed_views = sum(row["direct_fixed_exact"] is False for row in rows)
    direct, crop = by_path["direct"], by_path["crop"]
    gate = (direct["common_nonempty"] > 0 and direct_changed_views == 0 and crop["common_changed_masks"] > 0)
    return dict(total=totals(rows), by_path=by_path, direct_fixed_changed_views=direct_changed_views,
                c_natural_oracle_exact_views=sum(row["c_natural_oracle_exact"] is True for row in rows),
                c_b_conditioned_oracle_exact_views=sum(row["c_b_conditioned_oracle_exact"] is True for row in rows),
                decision="SUPPORTS_FIXED_ROW_ONE_PASS_CROP_CHANGE" if gate else "ONE_PASS_FIT_GATE_NOT_MET")


def run(archive, layout, output):
    support = load_stage_support()
    legacy = support.pinned_module(support.HISTORICAL_WORKER, support.HISTORICAL_WORKER_SHA)
    legacy.runtime()
    require(output.is_absolute() and output.parent.resolve(strict=True) == output.parent
            and not output.exists() and not output.is_symlink() and not output.is_relative_to(layout)
            and not layout.is_relative_to(output), "Fresh canonical SSDD output outside the FIT layout")
    sys.path.insert(0, str(HERE))
    from ssdd_phase_init import CONFIG, CONFIG_SHA, E1, E1_SHA, RECIPE, RECIPE_SHA, load_api, offline, pinned_bytes
    sys.addaudithook(offline)
    from rfdetr import config
    from ssdd_fit_only import FitOnlyData
    import ssdd_fit_stage_readback as readback
    import ssdd_phase_fit_replay as historical
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), "SSDD stock FIT audit must leave CUDA unused")
    before = readback.readback_fit_layout(archive, layout)
    groups = support.verified_groups(archive, before, readback)
    ids = tuple(sorted(groups))
    pinned_bytes(RECIPE, RECIPE_SHA)
    model_config, train = load_api(CONFIG, CONFIG_SHA)._load_configs(RECIPE, layout)
    load_api(E1, E1_SHA).native_recipe(model_config, train, config)
    data = FitOnlyData(model_config, train, layout)
    data.setup("fit")
    dataset = data._dataset_train
    require(type(dataset) is historical.CocoDetection and tuple(dataset.ids) == ids and callable(dataset._transforms),
            "Exact original SSDD FIT741 loader and sorted ID roster")
    transform = dataset._transforms
    helpers = _load_audit_helpers(support)
    output.mkdir()
    ledger = sha256()
    rows = []
    start = perf_counter()
    with (output / "VIEWS_PRIVATE.jsonl").open("xb") as stream:
        try:
            for ordinal, image_id in enumerate(ids):
                seed = support.SEED_BASE + ordinal
                view = None

                def observe(image, raw):
                    nonlocal view
                    require(view is None and int(raw["image_id"].item()) == image_id, "Exact stock SSDD input identity")
                    source = dataset.coco.imgs[image_id]
                    source_hw = source["height"], source["width"]
                    historical._target(raw, image_id, source_hw, source_hw, with_rows=False)
                    view = audit_view(historical._torchvision, transform, image, raw, seed,
                                      set_seed=historical._seed_cpu, helpers=helpers)
                    return image, raw

                dataset._transforms = observe
                dataset[ordinal]
                require(view is not None and view["steps"][-1]["after"] == list(historical.CANVAS)
                        and max(view["b_present"], view["c_present"]) <= historical.QUERY_CAPACITY,
                        "Exact stock SSDD final geometry and Q100 target capacity")
                view.update(ordinal=ordinal, group=groups[image_id])
                raw = support.compact(view) + b"\n"
                require(stream.write(raw) == len(raw), "Incomplete private one-pass ledger")
                ledger.update(raw)
                rows.append(view)
                if (ordinal + 1) % 50 == 0 or ordinal + 1 == len(ids):
                    print(support.compact(dict(event=PROGRESS, images=ordinal + 1)).decode(), flush=True)
        except Exception as error:
            legacy.write_json(output / "FIRST_REPLAY_FAILURE_PRIVATE.json",
                              dict(ordinal=ordinal, image_id=image_id, seed=seed, cause_type=type(error).__name__))
            raise
        finally:
            dataset._transforms = transform
        stream.flush()
        os.fsync(stream.fileno())
    result = summarize_audits(rows)
    paths = {path: values["views"] for path, values in result["by_path"].items()}
    require(len(rows) == support.FIT_IMAGES and paths == support.HISTORICAL_PATHS
            and all(result["by_path"][path][key] == value for path, counts in HISTORICAL_B.items()
                    for key, value in counts.items())
            and not torch.cuda.is_initialized() and torch.get_num_threads() == torch.get_num_interop_threads() == 1
            and before == readback.readback_fit_layout(archive, layout), "Historical view, CPU runtime or FIT input differs")
    legacy.write_json(output / "RESULT_SAFE.json", dict(
        status=STATUS, images=len(ids), views=len(rows), groups=len(set(groups.values())), seed_base=support.SEED_BASE,
        paths=paths, audit=result, fit_archive_sha256=before["fit_archive_sha256"], fit_index_sha256=before["fit_index_sha256"],
        content_roster_sha256=before["content_roster_sha256"], ordered_fit_ids_sha256=support.FIT_ORDER_SHA,
        candidate_sha256=ONE_PASS_SHA, view_ledger_sha256=ledger.hexdigest(), elapsed_seconds=perf_counter() - start,
        platform="Linux_aarch64_DGX_CPU", packages=legacy.PACKAGES, cpu_affinity=[2, 3], torch_threads=1,
        exact_shared_images_geometry_rng=True, exact_b_nonmask_fields=True,
        cuda_initialized=False, FIT=True, DEV=False, TEST=False, AP=False, forward_calls=0, optimizer_updates=0,
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("archive", "layout", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.archive, args.layout, args.output)


if __name__ == "__main__":
    main()
