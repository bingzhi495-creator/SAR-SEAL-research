"""Check the matched direct optimization on all 456 sealed HRSID FIT114 views."""

import argparse
import builtins
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import ModuleType
from zipfile import ZIP_STORED, ZipFile


REPLAY = "HRSID_FIT_RESIZE_PHASE_REPLAY_20261109"
SSDD = "SSDD_MASK_PHASE_TRANSFER_20261110"
OPTIMIZED = "one_pass_common_direct_fast_transform.py"
MODES = ("stepwise_center", "one_pass")
ROUTES = ("fast_direct", "direct_fallback", "crop_fallback")
BUNDLE_SHA = "af12a4e86ea36e9d7fe6ac47b17c684a4812baba6aa9be019f889f833e0f577c"
SUBSET_SHA = "6f25f80ed0e0b9c3f0c57ab6cc3b25e947a102353757a33bca4d3823df3d547d"
AUDITED_SHA = "2e4603a8bf45f6ffeb541ecd3b92fc99b5ccba0aff8982217903124703d1babe"
MATCHED_SHA = "5db1d9024a7ffa03608488b07669e5692d378f166f8773b1a5475089519d08b7"
STRATA = {"core_direct": (235, 1203, 0), "core_crop": (201, 638, 635),
          "stress_direct": (13, 1567, 0), "stress_crop": (7, 737, 734)}
PINS = {
    "common_mask_supervision": (REPLAY + "/common_mask_supervision.py", "59ea82e24353eeb4a14115058d5115333a4e7ea517a0abef581bee137a9a3a57"),
    "trace_resize": (REPLAY + "/trace_resize.py", "902a0e3640a319ba729172847f49ce582f3c0e3954593862ed94971727dc309f"),
    "fit_query_export": ("HRSID_FIT_NATIVE_QUERY_RETENTION_20261109/fit_query_export.py", "6a2d7e38d69b82ef3df790a5c359c408b1e74778c39672b47a62c4dc86cabbf5"),
    "reference_resize": (REPLAY + "/reference_resize.py", "3ca1d4a6400324c13c777cb4d5f7627713bc625580195d8e05363950872cc8df"),
    "replay_fit": (REPLAY + "/replay_fit.py", "4f95f1b5b84fcf89b005f999cfb8c5fd647a7b262aedd2622860cf8785a14157"),
    "paired_fit_transform": (REPLAY + "/paired_fit_transform.py", "64f3f63c6ac89b66578f0ada42fa3eb8b4db3b2f1ae2ea9113debe6ab9b270e0"),
    "one_pass_fit_transform": (REPLAY + "/one_pass_fit_transform.py", "05cad566f9b5cc761e91b5283585a5392a7450c42c3133349e6dc21e8bf7496d"),
    "one_pass_common_pair_transform": (REPLAY + "/one_pass_common_pair_transform.py", MATCHED_SHA),
}
HISTORICAL = (REPLAY + "/run_continuous_fit_dgx.py", "1c967ad78557a0530fc086e834f6e43ce8ad51775f00f01fd793ac2d65cc4c3f")
RUNTIME = (SSDD + "/ssdd_fit_replay_dgx.py", "892d7e66114c4c78f0f55ffc2e7b666bdae363ec727780e97a7465b70bed4dee")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(raw):
    return sha256(raw).hexdigest()


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def canonical_file(path, limit):
    require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
            "Canonical bounded sealed input required")
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= limit, "Bounded regular sealed input required")
    return identity(info)


def pinned_bytes(path, expected, limit):
    original = canonical_file(path, limit)
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
        require(len(raw) == original[2] and digest(raw) == expected
                and original == identity(os.fstat(stream.fileno())) == identity(path.stat()),
                "Exact sealed bytes required: " + path.name)
    return raw


def load(path, expected, namespace=None):
    raw = pinned_bytes(path, expected, 64 * 1024)
    module = ModuleType("_fit114_fast_audit_" + path.stem)
    module.__file__ = str(path)
    if namespace is not None:
        def use_pinned(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name in namespace:
                return namespace[name]
            return builtins.__import__(name, globals, locals, fromlist, level)
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=use_pinned)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def load_adapters(support, optimized_sha):
    pinned = {}
    for name, (relative, expected) in PINS.items():
        pinned[name] = load(support / relative, expected, pinned)
    optimized = load(support / REPLAY / OPTIMIZED, optimized_sha, pinned)
    original = pinned["one_pass_common_pair_transform"]
    require(original.MODES == MODES, "Exact original fixed-support mode order required")
    return original.OnePassCommonPairTransform, optimized.OnePassCommonDirectFastTransform, pinned["paired_fit_transform"]


def same_tensor(a, b, paired, torch):
    return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.device.type == b.device.type == "cpu"
            and a.dtype == b.dtype and a.shape == b.shape and torch.equal(paired._tensor_bytes(a), paired._tensor_bytes(b)))


def actual_call(adapter, records, image, target, native, paired, seed):
    records.clear()
    native.set_seed(seed)
    before = digest(paired._rng_bytes(paired._rng()))
    image, selected = adapter(image, target)
    after = digest(paired._rng_bytes(paired._rng()))
    require(len(records) == 1 and records[0]["rng_before_sha256"] == before
            and records[0]["rng_after_sha256"] == after, "Exactly one audit and the original post-transform RNG required")
    return image, selected, records[0].copy(), after


def verify_output(mode, old, fast, reference, paired, torch):
    a_image, a_target, a_audit, a_rng = old
    b_image, b_target, b_audit, b_rng = fast
    require(same_tensor(a_image, b_image, paired, torch) and a_target.keys() == b_target.keys()
            and all(same_tensor(a_target[key], b_target[key], paired, torch) for key in a_target)
            and compact(a_audit) == compact(b_audit) and a_rng == b_rng,
            "Original and optimized selected image, labels, audit or RNG differ")
    require(all(a_audit[key] == reference[key] for key in (
                "image_sha256", "trace_sha256", "rng_before_sha256", "rng_after_sha256", "raw_rows",
                "common_nonempty", "common_changed_masks"))
            and a_audit["crop"] is (reference["path"] == "crop") and a_audit["flipped"] is reference["flip"]
            and a_audit["native_present"] == a_audit["center_present"] == reference["b_present"]
            and a_audit["native_nonempty"] == reference["b_nonempty"]
            and a_audit["center_nonempty"] == reference["b_conditioned_nonempty"],
            "Actual paired witness differs from the closed HRSID view")
    require(reference["b_present"] == reference["b_nonempty"] == reference["b_conditioned_nonempty"]
            == reference["common_nonempty"] == len(a_target["labels"])
            and "_p198_row_id" not in a_target and a_audit["common_row_ids_sha256"] == reference["b_row_ids_sha256"],
            "Matched original rows differ from the independently closed fixed-support view")
    selected = "native_common_masks_sha256" if mode == "stepwise_center" else "center_common_masks_sha256"
    require(a_audit["native_common_masks_sha256"] == reference["b_masks_sha256"]
            and a_audit["center_common_masks_sha256"] == reference["c_b_conditioned_masks_sha256"]
            and paired._training_mask_digest(a_target["masks"]) == a_audit[selected]
            and paired._tensor_digest((("image", a_image),)) == reference["image_sha256"],
            "Actual selected outputs must match the independently closed image and masks")
    return a_audit


def checked_route(before, after, path):
    require(type(before) is type(after) is int and 0 <= before <= after <= before + 1,
            "Exact optimization counter transition required per invocation")
    require(after == before or path == "direct", "A crop view must use the original fallback")
    return "fast_direct" if after > before else path + "_fallback"


def run(args):
    support = args.support_code
    require(support.is_absolute() and support.resolve(strict=True) == support and support.is_dir()
            and re.fullmatch(r"[0-9a-f]{64}", args.optimized_source_sha256) is not None,
            "Canonical pinned ordinary source tree and approved candidate digest required")
    load(support / RUNTIME[0], RUNTIME[1]).runtime()
    historic = load(support / HISTORICAL[0], HISTORICAL[1])
    native = load(historic.AUDIT, historic.AUDIT_SHA)
    sys.addaudithook(native.offline)
    require(all(native.version(key) == expected for key, expected in native.VERSIONS.items())
            and all(digest(Path(module.__file__).read_bytes()) == expected
                    for module, expected in native.SOURCE_SHA.items()), "Exact original HRSID augmentation runtime")
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), "GPU stays hidden")
    original_cls, optimized_cls, paired = load_adapters(support, args.optimized_source_sha256)
    audit = native.NativeInputAudit()
    records = {mode: ([], []) for mode in MODES}
    adapters = {
        mode: (original_cls(native._torchvision, audit.transform, mode, audit=records[mode][0].append),
               optimized_cls(native._torchvision, audit.transform, mode, audit=records[mode][1].append))
        for mode in MODES
    }
    ledger_state = canonical_file(args.closed_ledger, 4 * 1024**2)
    audited = [json.loads(line) for line in pinned_bytes(args.closed_ledger, AUDITED_SHA, 4 * 1024**2).splitlines()]
    strata = {name: dict(views=0, common_rows=0, changed_masks=0,
                        routes={mode: Counter(dict.fromkeys(ROUTES, 0)) for mode in MODES}) for name in STRATA}
    evidence = sha256()
    bundle_state = canonical_file(args.bundle, 192 * 1024**2)
    with args.bundle.open("rb") as source:
        require(identity(os.fstat(source.fileno())) == bundle_state and historic.file_digest(source) == BUNDLE_SHA,
                "Exact sealed original HRSID FIT114 archive")
        with ZipFile(source) as bundle:
            compare_bundle(bundle, historic, native, audit, paired, adapters, records, audited, strata, evidence, np, torch)
        require(bundle_state == identity(os.fstat(source.fileno())) == identity(args.bundle.stat())
                and ledger_state == identity(args.closed_ledger.stat()), "Sealed FIT and closed reference unchanged")
    require(not torch.cuda.is_initialized() and torch.get_num_threads() == torch.get_num_interop_threads() == 1,
            "Exact single-thread CPU-only audit")
    for name, expected in STRATA.items():
        current = strata[name]
        require(tuple(current[key] for key in ("views", "common_rows", "changed_masks")) == expected
                and all(sum(route.values()) == current["views"] for route in current["routes"].values()),
                "Exact closed stratum counts and actual route census required")
    totals = {mode: {route: sum(item["routes"][mode][route] for item in strata.values()) for route in ROUTES}
              for mode in MODES}
    require(all(totals[mode]["fast_direct"] == adapters[mode][1].fast_direct_visits for mode in MODES),
            "Measured optimization path counts must agree with actual candidate counters")
    exercised = all(totals[mode]["fast_direct"] > 0 for mode in MODES)
    return dict(status="HRSID_FIT114_MATCHED_DIRECT_FAST_EQUIVALENCE_COMPLETE",
                decision="BITWISE_EQUIVALENT_FAST_DIRECT_EXERCISED" if exercised else "FAST_DIRECT_PATH_NOT_EXERCISED",
                images=114, views=456, original_adapter_calls=912, optimized_adapter_calls=912, strata=strata,
                routes=totals, exact_selected_targets_images_audits_and_rng=True, exact_closed_reference_views=456,
                optimized_source_sha256=args.optimized_source_sha256, matched_source_sha256=MATCHED_SHA,
                closed_audit_ledger_sha256=AUDITED_SHA, historical_view_ledger_sha256=historic.HISTORY_SHA,
                fit_bundle_sha256=BUNDLE_SHA, fit_subset_sha256=SUBSET_SHA, verification_sha256=evidence.hexdigest(),
                cpu_affinity=[2, 3], torch_threads=1, cuda_initialized=False,
                FIT=True, DEV=False, TEST=False, AP=False, forward_calls=0, optimizer_updates=0)


def compare_bundle(bundle, historic, native, audit, paired, adapters, records, audited, strata, evidence, np, torch):
    cohort = json.loads(historic.pinned_read(bundle, "COHORT_PRIVATE.json", historic.COHORT_SHA, 1024**2))
    subset = json.loads(historic.pinned_read(bundle, "FIT114_COCO_PRIVATE.json", SUBSET_SHA, 4 * 1024**2))
    history = [json.loads(line) for line in historic.pinned_read(
        bundle, "HISTORICAL_VIEWS_PRIVATE.jsonl", historic.HISTORY_SHA, 1024**2,
    ).splitlines()]
    images = {row["id"]: row["file_name"] for row in subset["images"]}
    annotations = defaultdict(list)
    for row in subset["annotations"]:
        require(row["image_id"] in images, "Only exact selected FIT114 annotations permitted")
        annotations[row["image_id"]].append(row)
    roster = {"COHORT_PRIVATE.json", "FIT114_COCO_PRIVATE.json", "HISTORICAL_VIEWS_PRIVATE.jsonl"}
    roster.update("images/" + row["native_basename"] for row in cohort["images"])
    require(len(roster) == len(bundle.infolist()) == 117 and {item.filename for item in bundle.infolist()} == roster
            and all(item.compress_type == ZIP_STORED for item in bundle.infolist())
            and len(cohort["images"]) == len(subset["images"]) == len(images) == 114 and cohort["fit_only"] is True
            and len(history) == len(audited) == 456 and len(subset["annotations"]) == 1109,
            "Exact isolated and complete original HRSID FIT114 cohort")
    for ordinal, row in enumerate(cohort["images"]):
        image_id, name = row["image_id"], row["native_basename"]
        require(images[image_id] == name and row["core"] is not row["stress"], "Exact historic FIT114 image identity")
        layer = "core" if row["core"] else "stress"
        raw = historic.pinned_read(bundle, "images/" + name, row["encoded_sha256"], 4 * 1024**2)
        image, scale = native.io_utils.decode_pil_image_bytes(raw, draft_size=None)
        require(len(raw) == row["encoded_bytes"] and image.mode == "RGB" and image.size == (800, 800) and scale == (1., 1.)
                and digest(np.asarray(image).tobytes(order="C")) == row["decoded_rgb_sha256"], "Exact encoded and decoded FIT image")
        image, target = audit.convert(image, dict(image_id=image_id, annotations=annotations[image_id]))
        native.target_summary(target, 800, image_id, normalized=False)
        require(not bool(native.mask_empty(target).any()), "Exact original prepared nonempty FIT rows")
        for repeat in range(historic.REPEATS):
            cursor = ordinal * historic.REPEATS + repeat
            seed = historic.BASE_SEED + cursor
            old, ref = history[cursor], audited[cursor]
            path = ref["path"]
            require(old["ordinal"] == ref["ordinal"] == ordinal and old["seed"] == ref["seed"] == seed
                    and old["layer"] == ref["layer"] == layer and old["group"] == row["group"] and ref["image_id"] == image_id
                    and old["path"] == path and old["flip"] is ref["flip"] and old["steps"] == ref["steps"]
                    and all(old["rows"][legacy] == ref[current] for legacy, current in (
                        ("raw_rows", "raw_rows"), ("center_present", "b_present"), ("center_nonempty", "b_nonempty"))),
                    "Exact historic view, geometry and stepwise support required")
            summary = strata[f"{layer}_{path}"]
            for mode in MODES:
                base, candidate = adapters[mode]
                previous = candidate.fast_direct_visits
                original = actual_call(base, records[mode][0], image, target, native, paired, seed)
                optimized = actual_call(candidate, records[mode][1], image, target, native, paired, seed)
                witness = verify_output(mode, original, optimized, ref, paired, torch)
                route = checked_route(previous, candidate.fast_direct_visits, path)
                summary["routes"][mode][route] += 1
                evidence.update(compact(dict(cursor=cursor, mode=mode, route=route, witness=witness)) + b"\n")
            summary["views"] += 1
            summary["common_rows"] += ref["common_nonempty"]
            summary["changed_masks"] += ref["common_changed_masks"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("support-code", "bundle", "closed-ledger"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--optimized-source-sha256", required=True)
    print(compact(run(parser.parse_args(argv))).decode("utf-8"), flush=True)


if __name__ == "__main__":
    main()
