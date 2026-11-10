"""Check the matched one-pass training adapter on four preselected FIT114 views."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType
from zipfile import ZipFile

BUNDLE_SHA = "af12a4e86ea36e9d7fe6ac47b17c684a4812baba6aa9be019f889f833e0f577c"
SUBSET_SHA = "6f25f80ed0e0b9c3f0c57ab6cc3b25e947a102353757a33bca4d3823df3d547d"
AUDITED_SHA = "2e4603a8bf45f6ffeb541ecd3b92fc99b5ccba0aff8982217903124703d1babe"
MATCHED_SHA = "5db1d9024a7ffa03608488b07669e5692d378f166f8773b1a5475089519d08b7"
SCOPES = ("core", "stress")
PATHS = ("direct", "crop")


def run(args):
    here = Path(__file__).resolve().parent
    support = args.support_code
    sys.path[:0] = [str(here), str(support / "HRSID_FIT_RESIZE_PHASE_REPLAY_20261109"),
                   str(support / "SSDD_MASK_PHASE_TRANSFER_20261110"),
                   str(support / "HRSID_FIT_NATIVE_QUERY_RETENTION_20261109")]
    from run_continuous_fit_dgx import (
        AUDIT, AUDIT_SHA, BASE_SEED, COHORT_SHA, HISTORY_SHA, REPEATS, digest, file_digest, pinned_read, require,
    )
    from ssdd_fit_replay_dgx import runtime
    runtime()
    from one_pass_common_pair_transform import MODES, OnePassCommonPairTransform
    from paired_fit_transform import _training_mask_digest
    import numpy as np
    import torch

    require(digest((here / "one_pass_common_pair_transform.py").read_bytes()) == MATCHED_SHA, "Exact matched transform")
    source = AUDIT.read_bytes()
    require(digest(source) == AUDIT_SHA, "Exact original augmentation audit")
    native = ModuleType("_hrsid_fit114_matched_smoke")
    native.__file__ = str(AUDIT)
    exec(compile(source, str(AUDIT), "exec"), native.__dict__)
    sys.addaudithook(native.offline)
    require(all(native.version(k) == v for k, v in native.VERSIONS.items())
            and all(digest(Path(m.__file__).read_bytes()) == v for m, v in native.SOURCE_SHA.items()),
            "Exact original HRSID runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    require(not torch.cuda.is_initialized() and args.bundle.is_file()
            and 0 < args.bundle.stat().st_size <= 192 * 1024**2
            and args.closed_ledger.is_file() and 0 < args.closed_ledger.stat().st_size <= 4 * 1024**2,
            "Bounded sealed FIT inputs and CPU runtime")
    with args.bundle.open("rb") as stream:
        require(file_digest(stream) == BUNDLE_SHA, "Exact sealed original FIT114 archive")
    raw_audited = args.closed_ledger.read_bytes()
    require(digest(raw_audited) == AUDITED_SHA, "Exact closed and independently verified one-pass ledger")
    audited = [json.loads(line) for line in raw_audited.splitlines()]
    result = {}
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
        audit = native.NativeInputAudit()
        for layer in SCOPES:
            for path in PATHS:
                cursor = next(i for i, old in enumerate(history)
                              if old["layer"] == layer and old["path"] == path
                              and old["rows"]["center_nonempty"] > 0)
                old, reference = history[cursor], audited[cursor]
                ordinal, seed = cursor // REPEATS, BASE_SEED + cursor
                row = cohort["images"][ordinal]
                iid, name = row["image_id"], row["native_basename"]
                require(images[iid] == name and old["ordinal"] == reference["ordinal"] == ordinal
                        and old["seed"] == reference["seed"] == seed and reference["image_id"] == iid
                        and old["path"] == reference["path"] == path and old["layer"] == reference["layer"] == layer
                        and old["steps"] == reference["steps"] and old["flip"] == reference["flip"],
                        "Exact first historical B-nonempty view in each stratum")
                data = pinned_read(bundle, "images/" + name, row["encoded_sha256"], 4 * 1024**2)
                image, scale = native.io_utils.decode_pil_image_bytes(data, draft_size=None)
                require(len(data) == row["encoded_bytes"] and image.mode == "RGB" and image.size == (800, 800)
                        and scale == (1., 1.) and digest(np.asarray(image).tobytes(order="C")) == row["decoded_rgb_sha256"],
                        "Exact prepared RGB source")
                prepared, targets = audit.convert(image, dict(image_id=iid, annotations=annotations[iid]))
                native.target_summary(targets, 800, iid, normalized=False)
                require(not bool(native.mask_empty(targets).any()), "Exact historical prepared nonempty masks")
                outputs, witnesses = [], []
                for mode in MODES:
                    visits = []
                    def stock(rgb, target):
                        visits.append(True)
                        return audit.transform(rgb, target)
                    native.set_seed(seed)
                    outputs.append(OnePassCommonPairTransform(
                        native._torchvision, stock, mode, audit=witnesses.append,
                    )(prepared, targets))
                    require(len(visits) == 1, "Exactly one stock augmentation per arm")
                require(len(witnesses) == 2 and witnesses[0] == witnesses[1], "Exact paired visits")
                witness = witnesses[0]
                for key in ("image_sha256", "trace_sha256", "rng_before_sha256", "rng_after_sha256",
                            "common_nonempty", "common_changed_masks"):
                    require(witness[key] == reference[key], "Matched view differs from independently read back stock evidence")
                b, c = outputs[0][1], outputs[1][1]
                require(torch.equal(outputs[0][0], outputs[1][0]) and b.keys() == c.keys()
                        and all(torch.equal(b[k], c[k]) for k in b if k != "masks")
                        and reference["b_nonempty"] == reference["b_conditioned_nonempty"] == witness["common_nonempty"]
                        and witness["common_row_ids_sha256"] == reference["b_row_ids_sha256"]
                        and _training_mask_digest(b["masks"]) == witness["native_common_masks_sha256"] == reference["b_masks_sha256"]
                        and _training_mask_digest(c["masks"]) == witness["center_common_masks_sha256"] == reference["c_b_conditioned_masks_sha256"],
                        "Actual returned labels must equal independently witnessed fixed-B targets")
                require(path != "direct" or torch.equal(b["masks"], c["masks"]), "Direct mask control")
                result[f"{layer}_{path}"] = dict(common_rows=witness["common_nonempty"],
                                                 changed_masks=witness["common_changed_masks"])
    require(any(result[f"{layer}_crop"]["changed_masks"] > 0 for layer in SCOPES)
            and not torch.cuda.is_initialized(), "Real crop intervention without GPU or training")
    return dict(status="HRSID_MATCHED_ONE_PASS_REAL_FIT_SMOKE_PASSED", views=4, stock_calls=8, strata=result,
                matched_source_sha256=MATCHED_SHA, closed_ledger_sha256=AUDITED_SHA,
                FIT=True, DEV=False, TEST=False, AP=False, forward_calls=0, optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("support-code", "bundle", "closed-ledger"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
