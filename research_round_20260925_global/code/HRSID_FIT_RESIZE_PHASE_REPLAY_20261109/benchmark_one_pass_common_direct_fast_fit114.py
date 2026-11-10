"""Compare original and direct-fast matched one-pass loader cost on FIT114."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median
import sys
from types import ModuleType
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
BENCHMARK_SHA = "e75dc88bda98ac0c18b876dbe498f1ef70e4d05f41eae2bf663ccbd1e43393d7"
AUDIT_MODES = ("audit_off", "audit_on")


def summarize_costs(pairs, common_rows, warm_hits, timed_hits=0):
    deltas = [fast - original for original, fast in pairs]
    return dict(
        paired_rounds=len(pairs),
        matched_rows=common_rows,
        warm_fast_hits=warm_hits,
        timed_fast_hits=timed_hits,
        original_paired_median_ms=median(original for original, _ in pairs),
        direct_fast_median_ms=median(fast for _, fast in pairs),
        paired_delta_median_ms=median(deltas),
        paired_delta_min_ms=min(deltas),
        paired_delta_max_ms=max(deltas),
        paired_ratio_median=median(fast / original for original, fast in pairs),
    )


def run(args):
    support = args.support_code
    sys.path[:0] = [str(HERE), str(support / "HRSID_FIT_RESIZE_PHASE_REPLAY_20261109"),
                   str(support / "SSDD_MASK_PHASE_TRANSFER_20261110"),
                   str(support / "HRSID_FIT_NATIVE_QUERY_RETENTION_20261109")]
    original_benchmark = HERE / "benchmark_one_pass_common_fit114.py"
    if sha256(original_benchmark.read_bytes()).hexdigest() != BENCHMARK_SHA:
        raise ValueError("Exact original timing protocol required")
    from benchmark_one_pass_common_fit114 import B, C, ROUNDS, SMOKE_SHA, measure_view, select_views
    from run_continuous_fit_dgx import (
        AUDIT, AUDIT_SHA, BASE_SEED, COHORT_SHA, HISTORY_SHA, REPEATS, digest, file_digest, pinned_read, require,
    )
    require(digest((HERE / "smoke_one_pass_common_fit114.py").read_bytes()) == SMOKE_SHA,
            "Exact published matched smoke pins required")
    from smoke_one_pass_common_fit114 import AUDITED_SHA, BUNDLE_SHA, MATCHED_SHA, PATHS, SCOPES, SUBSET_SHA
    from ssdd_fit_replay_dgx import runtime
    runtime()
    fast_path = HERE / "one_pass_common_direct_fast_transform.py"
    require(digest(fast_path.read_bytes()) == args.fast_sha256, "Exact prospectively bound direct-fast source")
    require(digest((HERE / "one_pass_common_pair_transform.py").read_bytes()) == MATCHED_SHA, "Exact matched transform")
    from one_pass_common_direct_fast_transform import OnePassCommonDirectFastTransform
    from one_pass_common_pair_transform import OnePassCommonPairTransform
    from paired_fit_transform import _rng, _rng_bytes, _tensor_digest, _training_mask_digest
    import numpy as np
    import torch

    source = AUDIT.read_bytes()
    require(digest(source) == AUDIT_SHA, "Exact original augmentation audit")
    native = ModuleType("_hrsid_fit114_direct_fast_benchmark")
    native.__file__ = str(AUDIT)
    exec(compile(source, str(AUDIT), "exec"), native.__dict__)
    sys.addaudithook(native.offline)
    require(all(native.version(k) == v for k, v in native.VERSIONS.items())
            and all(digest(Path(m.__file__).read_bytes()) == v for m, v in native.SOURCE_SHA.items()),
            "Exact original HRSID runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized()
            and all(os.environ.get(name) == "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "CPU-only single-thread benchmark")

    for path, limit in ((args.bundle, 192 * 1024**2), (args.closed_ledger, 4 * 1024**2)):
        require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
                and 0 < path.stat().st_size <= limit, "Canonical bounded closed FIT input")
    with args.bundle.open("rb") as stream:
        require(file_digest(stream) == BUNDLE_SHA, "Exact sealed original FIT114 archive")
    raw_ledger = args.closed_ledger.read_bytes()
    require(digest(raw_ledger) == AUDITED_SHA, "Exact independently verified closed ledger")
    audited = [json.loads(line) for line in raw_ledger.splitlines()]

    audit = native.NativeInputAudit()
    strata = {}
    with ZipFile(args.bundle) as bundle:
        cohort = json.loads(pinned_read(bundle, "COHORT_PRIVATE.json", COHORT_SHA, 1024**2))
        subset = json.loads(pinned_read(bundle, "FIT114_COCO_PRIVATE.json", SUBSET_SHA, 4 * 1024**2))
        history = [json.loads(line) for line in pinned_read(
            bundle, "HISTORICAL_VIEWS_PRIVATE.jsonl", HISTORY_SHA, 1024**2,
        ).splitlines()]
        images = {row["id"]: row["file_name"] for row in subset["images"]}
        annotations = defaultdict(list)
        for row in subset["annotations"]:
            require(row["image_id"] in images, "Only pinned FIT images")
            annotations[row["image_id"]].append(row)
        require(cohort["fit_only"] is True and len(cohort["images"]) == len(images) == 114
                and len(history) == len(audited) == 456 and len(subset["annotations"]) == 1109,
                "Exact historical cohort")
        for (layer, path), indices in select_views(history, SCOPES, PATHS).items():
            pairs = {mode: [] for mode in AUDIT_MODES}
            warm_hits = dict.fromkeys(AUDIT_MODES, 0)
            timed_hits = dict.fromkeys(AUDIT_MODES, 0)
            common_rows, changed_masks = 0, 0
            for index, cursor in enumerate(indices):
                old, ref = history[cursor], audited[cursor]
                ordinal, seed = cursor // REPEATS, BASE_SEED + cursor
                row = cohort["images"][ordinal]
                iid, name = row["image_id"], row["native_basename"]
                require(images[iid] == name and old["ordinal"] == ref["ordinal"] == ordinal
                        and old["seed"] == ref["seed"] == seed and ref["image_id"] == iid
                        and old["path"] == ref["path"] == path and old["layer"] == ref["layer"] == layer
                        and old["steps"] == ref["steps"] and old["flip"] == ref["flip"],
                        "Exact historical stratum and view")
                data = pinned_read(bundle, "images/" + name, row["encoded_sha256"], 4 * 1024**2)
                image, scale = native.io_utils.decode_pil_image_bytes(data, draft_size=None)
                require(len(data) == row["encoded_bytes"] and image.mode == "RGB" and image.size == (800, 800)
                        and scale == (1., 1.) and digest(np.asarray(image).tobytes(order="C")) == row["decoded_rgb_sha256"],
                        "Exact prepared RGB source")
                image, targets = audit.convert(image, dict(image_id=iid, annotations=annotations[iid]))
                native.target_summary(targets, 800, iid, normalized=False)
                require(not bool(native.mask_empty(targets).any()), "Exact historical prepared nonempty masks")
                common_rows += ref["common_nonempty"]
                changed_masks += ref["common_changed_masks"]

                reverse = bool(index % 2)
                modes = AUDIT_MODES[::-1] if reverse else AUDIT_MODES
                for mode in modes:
                    witnesses = {B: [], C: []}
                    audit_on = mode == "audit_on"
                    stock = audit.transform
                    original = OnePassCommonPairTransform(
                        native._torchvision, stock, "one_pass", audit=witnesses[B].append if audit_on else None,
                    )
                    fast = OnePassCommonDirectFastTransform(
                        native._torchvision, stock, "one_pass", audit=witnesses[C].append if audit_on else None,
                    )
                    arms = {B: original, C: fast}
                    outputs = {}
                    for arm in (C, B) if reverse else (B, C):
                        native.set_seed(seed)
                        require(digest(_rng_bytes(_rng())) == ref["rng_before_sha256"], "Exact warm initial RNG")
                        outputs[arm] = arms[arm](image, targets)
                        require(digest(_rng_bytes(_rng())) == ref["rng_after_sha256"], "Exact warm terminal RNG")
                    b_image, b_target = outputs[B]
                    c_image, c_target = outputs[C]
                    require(
                        b_target.keys() == c_target.keys() and torch.equal(b_image, c_image)
                        and all(b_target[k].dtype == c_target[k].dtype and torch.equal(b_target[k], c_target[k])
                                for k in b_target)
                        and len(b_target["labels"]) == ref["b_present"] == ref["b_nonempty"]
                        == ref["b_conditioned_nonempty"] == ref["common_nonempty"]
                        and _tensor_digest((("image", b_image),)) == ref["image_sha256"]
                        and _training_mask_digest(b_target["masks"]) == ref["c_b_conditioned_masks_sha256"],
                        "Warm original and fast actual outputs must equal closed evidence",
                    )
                    del outputs, b_image, c_image, b_target, c_target
                    if audit_on:
                        require(len(witnesses[B]) == len(witnesses[C]) == 1 and witnesses[B] == witnesses[C],
                                "Both warm audit witnesses must match exactly")
                        witness = witnesses[B][0]
                        for field in ("image_sha256", "trace_sha256", "rng_before_sha256", "rng_after_sha256",
                                      "common_nonempty", "common_changed_masks"):
                            require(witness[field] == ref[field], "Exact archived warm witness")
                        require(witness["common_row_ids_sha256"] == ref["b_row_ids_sha256"]
                                and witness["native_common_masks_sha256"] == ref["b_masks_sha256"]
                                and witness["center_common_masks_sha256"] == ref["c_b_conditioned_masks_sha256"],
                                "Exact archived fixed-B rows and masks")

                    hit = fast.fast_direct_visits
                    require(hit in (0, 1) and (path == "direct" or hit == 0), "Warm actual direct-only fast hit")
                    pairs[mode].extend(measure_view(
                        lambda arm: arms[arm](image, targets), lambda: native.set_seed(seed), reverse=reverse,
                    ))
                    timed = fast.fast_direct_visits - hit
                    require(timed == 2 * ROUNDS * hit
                            and all(len(rows) == (1 + 2 * ROUNDS if audit_on else 0) for rows in witnesses.values()),
                            "Exact repeated branch and audit visit accounting")
                    warm_hits[mode] += hit
                    timed_hits[mode] += timed
            require(warm_hits["audit_off"] == warm_hits["audit_on"], "Audit must not change fast admission")
            require(path != "direct" or warm_hits["audit_off"] > 0, "Observe an actual fast hit in both direct strata")
            strata[f"{layer}_{path}"] = dict(
                views=len(indices), changed_masks=changed_masks,
                **{mode: summarize_costs(pairs[mode], common_rows, warm_hits[mode], timed_hits[mode])
                   for mode in AUDIT_MODES},
            )
    require(not torch.cuda.is_initialized() and torch.get_num_threads() == torch.get_num_interop_threads() == 1,
            "Exact single-thread CPU-only benchmark")
    return dict(
        status="HRSID_FIT114_DIRECT_FAST_PAIRED_CPU_OVERHEAD_COMPLETE",
        scope="prepared_input_transform_only_no_decode_no_forward",
        baseline="original_one_pass_common_pair_transform_mode_one_pass",
        treatment="direct_fast_one_pass_common_transform_mode_one_pass",
        selection="first_four_historical_B_nonempty_views_per_layer_and_path",
        schedule="for_each_audit_mode_one_untimed_warm_call_each_then_five_paired_ABBA_or_BAAB_rounds_per_view",
        clock="perf_counter_ns",
        views=sum(item["views"] for item in strata.values()),
        torch_threads=1, cpu_affinity=sorted(os.sched_getaffinity(0)),
        strata=strata, original_timing_source_sha256=BENCHMARK_SHA, matched_source_sha256=MATCHED_SHA,
        fast_source_sha256=args.fast_sha256, closed_ledger_sha256=AUDITED_SHA,
        FIT=True, DEV=False, TEST=False, AP=False, forward_calls=0, optimizer_updates=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("support-code", "bundle", "closed-ledger"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fast-sha256", required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()