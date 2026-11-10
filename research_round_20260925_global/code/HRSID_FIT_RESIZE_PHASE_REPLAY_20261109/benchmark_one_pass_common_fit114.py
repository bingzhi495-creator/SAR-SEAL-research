"""Time matched one-pass loader overhead on a fixed subset of historical FIT114 views."""

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import islice
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns
from types import ModuleType
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
SMOKE_SHA = "dacb89854b17985a9c068ff657b25ad1eabf542c814021276716ef59ed52f6c6"
VIEWS_PER_STRATUM = 4
ROUNDS = 5
B = "stock_center"
C = "paired_one_pass"


def select_views(history, layers, paths):
    selected = {}
    for layer in layers:
        for path in paths:
            eligible = (i for i, row in enumerate(history)
                        if row["layer"] == layer and row["path"] == path
                        and row["rows"]["center_nonempty"] > 0)
            indices = list(islice(eligible, VIEWS_PER_STRATUM))
            if len(indices) != VIEWS_PER_STRATUM:
                raise ValueError("Too few historical B-nonempty views in a fixed stratum")
            selected[layer, path] = indices
    return selected


def measure_view(apply, reset, clock=perf_counter_ns, *, reverse=False):
    pairs = []
    for index in range(ROUNDS):
        order = (C, B, B, C) if (index + reverse) % 2 else (B, C, C, B)
        times = {B: [], C: []}
        for arm in order:
            reset()
            start = clock()
            output = apply(arm)
            elapsed_ms = (clock() - start) / 1_000_000
            del output
            if elapsed_ms <= 0:
                raise ValueError("Nonpositive CPU timing")
            times[arm].append(elapsed_ms)
        pairs.append((median(times[B]), median(times[C])))
    return pairs


def summarize(pairs, common_rows, changed_masks):
    deltas = [c - b for b, c in pairs]
    return dict(
        views=len(pairs) // ROUNDS,
        paired_rounds=len(pairs),
        matched_rows=common_rows,
        changed_masks=changed_masks,
        stock_center_median_ms=median(b for b, _ in pairs),
        paired_one_pass_median_ms=median(c for _, c in pairs),
        paired_delta_median_ms=median(deltas),
        paired_delta_min_ms=min(deltas),
        paired_delta_max_ms=max(deltas),
        paired_ratio_median=median(c / b for b, c in pairs),
    )


def run(args):
    support = args.support_code
    sys.path[:0] = [str(HERE), str(support / "HRSID_FIT_RESIZE_PHASE_REPLAY_20261109"),
                   str(support / "SSDD_MASK_PHASE_TRANSFER_20261110"),
                   str(support / "HRSID_FIT_NATIVE_QUERY_RETENTION_20261109")]
    smoke = HERE / "smoke_one_pass_common_fit114.py"
    if sha256(smoke.read_bytes()).hexdigest() != SMOKE_SHA:
        raise ValueError("Exact published matched smoke pins required")
    from smoke_one_pass_common_fit114 import AUDITED_SHA, BUNDLE_SHA, MATCHED_SHA, PATHS, SCOPES, SUBSET_SHA
    from run_continuous_fit_dgx import (
        AUDIT, AUDIT_SHA, BASE_SEED, COHORT_SHA, HISTORY_SHA, REPEATS, digest, file_digest, pinned_read, require,
    )
    from ssdd_fit_replay_dgx import runtime
    runtime()
    from common_mask_supervision import ROW_ID
    from one_pass_common_pair_transform import OnePassCommonPairTransform
    from paired_fit_transform import (
        _check_output, _encoded, _prepare, _rng, _rng_bytes, _tensor_digest, _training_mask_digest,
    )
    from trace_resize import capture_transform
    import numpy as np
    import torch

    require(digest((HERE / "one_pass_common_pair_transform.py").read_bytes()) == MATCHED_SHA, "Exact matched transform")
    source = AUDIT.read_bytes()
    require(digest(source) == AUDIT_SHA, "Exact original augmentation audit")
    native = ModuleType("_hrsid_fit114_matched_benchmark")
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
    paired = OnePassCommonPairTransform(native._torchvision, audit.transform, "one_pass")

    def stock(image, targets):
        raw = _prepare(image, targets)
        independent = {key: value.clone() for key, value in raw.items()}
        with capture_transform(native._torchvision, mode="center") as trace:
            result_image, result = audit.transform(image.copy(), independent)
        _check_output(result_image, result, raw)
        return result_image, result, trace

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
            pairs, common_rows, changed_masks = [], 0, 0
            for index, cursor in enumerate(indices):
                old, ref = history[cursor], audited[cursor]
                ordinal, seed = cursor // REPEATS, BASE_SEED + cursor
                row = cohort["images"][ordinal]
                iid, name = row["image_id"], row["native_basename"]
                require(images[iid] == name and old["ordinal"] == ref["ordinal"] == ordinal
                        and old["seed"] == ref["seed"] == seed and ref["image_id"] == iid
                        and old["path"] == ref["path"] == path and old["layer"] == ref["layer"] == layer
                        and old["steps"] == ref["steps"] and old["flip"] == ref["flip"],
                        "Exact first historical B-nonempty views in each stratum")
                data = pinned_read(bundle, "images/" + name, row["encoded_sha256"], 4 * 1024**2)
                image, scale = native.io_utils.decode_pil_image_bytes(data, draft_size=None)
                require(len(data) == row["encoded_bytes"] and image.mode == "RGB" and image.size == (800, 800)
                        and scale == (1., 1.) and digest(np.asarray(image).tobytes(order="C")) == row["decoded_rgb_sha256"],
                        "Exact prepared RGB source")
                image, targets = audit.convert(image, dict(image_id=iid, annotations=annotations[iid]))
                native.target_summary(targets, 800, iid, normalized=False)
                require(not bool(native.mask_empty(targets).any()), "Exact historical prepared nonempty masks")

                # Untimed warm pass doubles as independent fixed-support and RNG verification.
                native.set_seed(seed)
                before = digest(_rng_bytes(_rng()))
                b_image, b_target, b_trace = stock(image, targets)
                after = digest(_rng_bytes(_rng()))
                native.set_seed(seed)
                c_image, c_target = paired(image, targets)
                count = len(b_target["labels"])
                changed = int((b_target["masks"] != c_target["masks"]).flatten(1).any(dim=1).sum())
                require(
                    b_target.keys() - {ROW_ID} == c_target.keys()
                    and count == ref["b_present"] == ref["b_nonempty"] == ref["b_conditioned_nonempty"] == ref["common_nonempty"]
                    and _tensor_digest(((ROW_ID, b_target[ROW_ID]),)) == ref["b_row_ids_sha256"]
                    and _training_mask_digest(b_target["masks"]) == ref["b_masks_sha256"]
                    and _training_mask_digest(c_target["masks"]) == ref["c_b_conditioned_masks_sha256"]
                    and changed == ref["common_changed_masks"] and (path != "direct" or changed == 0)
                    and torch.equal(b_image, c_image)
                    and all(torch.equal(b_target[k], c_target[k]) for k in c_target if k != "masks")
                    and _tensor_digest((("image", b_image),)) == ref["image_sha256"]
                    and digest(_encoded(b_trace)) == ref["trace_sha256"]
                    and before == ref["rng_before_sha256"] and after == ref["rng_after_sha256"]
                    and digest(_rng_bytes(_rng())) == after,
                    "Fixed B support and one-pass output must match the independent closed ledger",
                )
                del b_image, c_image, b_target, c_target, b_trace
                common_rows += count
                changed_masks += changed

                def apply(arm):
                    return (stock if arm == B else paired)(image, targets)

                pairs.extend(measure_view(apply, lambda: native.set_seed(seed), reverse=bool(index % 2)))
            strata[f"{layer}_{path}"] = summarize(pairs, common_rows, changed_masks)
    require(not torch.cuda.is_initialized() and torch.get_num_threads() == torch.get_num_interop_threads() == 1,
            "Exact single-thread CPU-only benchmark")
    require(any(strata[f"{layer}_crop"]["changed_masks"] for layer in SCOPES), "Real crop intervention")
    return dict(
        status="HRSID_FIT114_MATCHED_ONE_PASS_CPU_OVERHEAD_COMPLETE",
        scope="prepared_input_transform_only_audit_off_no_decode_no_forward",
        baseline="stock_stepwise_center_same_returned_rows",
        treatment="one_pass_common_pair_transform_mode_one_pass_audit_none",
        selection="first_four_historical_B_nonempty_views_per_layer_and_path",
        clock="perf_counter_ns",
        schedule="one_untimed_warm_call_each_then_five_paired_ABBA_or_BAAB_rounds_per_view",
        views=sum(item["views"] for item in strata.values()),
        torch_threads=1, cpu_affinity=sorted(os.sched_getaffinity(0)),
        strata=strata, matched_source_sha256=MATCHED_SHA, closed_ledger_sha256=AUDITED_SHA,
        FIT=True, DEV=False, TEST=False, AP=False, forward_calls=0, optimizer_updates=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("support-code", "bundle", "closed-ledger"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()