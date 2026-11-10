"""Read-only CPU diagnosis of the independently accepted SSDD-retrained P295 pair."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from types import ModuleType


ACCEPTED_SHA256 = "97c53f1df007ddfc2cc0c200041a5b6246a40b4130c50e90151483e113d70a62"
GEOMETRY_SHA256 = "5aebe92a3b355874e8a20f24ef19b98ace8c20e3537610ec92a9c0e026b260e3"
GEOMETRY_WRAPPER = "SSDD_DEV187_POST_PRIMARY_RANK_20261110/ssdd_fixed_match_geometry.py"
GEOMETRY_CORE = "HRSID_MASK_PHASE_REPEAT_DEV705_20261109/dev705_paired_mask_geometry.py"
STATUS = "P299_SSDD_RETRAINED_DEV187_CPU_DIAGNOSTIC_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE"
ARMS = ("native", "center")
MIB = 1024**2


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def pinned_bytes(path, digest, maximum, label, exact_size=None):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            f"Canonical closed {label} file required")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum and (exact_size is None or len(raw) == exact_size),
            f"Closed {label} size differs")
    require(sha256(raw).hexdigest() == digest, f"Accepted {label} digest differs")
    return raw


def unique_pairs(pairs):
    value = dict(pairs)
    require(len(pairs) == len(value), "Duplicate closed JSON keys are forbidden")
    return value


def reject_nonfinite(_):
    raise ValueError("Nonfinite closed JSON values are forbidden")


def decode(raw, expected):
    value = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject_nonfinite)
    require(type(value) is expected, "Unexpected closed JSON shape")
    return value


def diagnose(code, accepted_path, integer_gt, native, center):
    """Bind original closed bytes, reproduce the primary scores, and diagnose the pair."""
    receipt = decode(pinned_bytes(accepted_path, ACCEPTED_SHA256, MIB, "P295 receipt"), dict)
    require(receipt.get("status") == "P286_SSDD_DEV187_INDEPENDENTLY_ACCEPTED"
            and receipt.get("coordinator_closed") is True and receipt.get("coordinator_original_os_wait") is True
            and type(receipt.get("coordinator_original_os_exit")) is int and receipt["coordinator_original_os_exit"] == 0
            and receipt.get("primary_GT") == "INTEGER_PRIMARY" and receipt.get("DEV") is True
            and receipt.get("TEST") is False and receipt.get("AP") is True and receipt.get("FIT") is False
            and receipt.get("training") is False and receipt.get("floating_gt_sha256") is None,
            "Original closed SSDD P295 integer DEV acceptance required")
    require(native != center and all(path.name == "PREDICTIONS.json" for path in (native, center)),
            "Two distinct original SSDD PREDICTIONS.json artifacts required")

    packages = {name: version(name) for name in ("numpy", "pycocotools", "scipy")}
    require(packages["numpy"] == "1.26.4" and packages["pycocotools"] == "2.0.11",
            "Exact original NumPy and COCO packages required")
    geometry_path = code / GEOMETRY_WRAPPER
    geometry_source = pinned_bytes(geometry_path, GEOMETRY_SHA256, MIB, "ordinary geometry source")
    geometry = ModuleType("_p299_closed_ssdd_geometry")
    geometry.__file__ = str(geometry_path)
    exec(compile(geometry_source, str(geometry_path), "exec"), geometry.__dict__)
    helpers = geometry.load_helpers(code, code / GEOMETRY_CORE)
    rank, p250, template, scoring, official, _ = helpers

    gt = decode(pinned_bytes(integer_gt, receipt["integer_gt_sha256"], MIB, "integer GT",
                             exact_size=p250.INTEGER_GT_BYTES), dict)
    images = gt["images"]
    ids = [image["id"] for image in images]
    require(len(ids) == len(set(ids)) == receipt["images"] == scoring.DEV_IMAGES
            and len(gt["annotations"]) == receipt["annotations"] == scoring.DEV_ANNOTATIONS,
            "Exact original SSDD DEV187 integer population required")
    predictions = {}
    for arm, path in zip(ARMS, (native, center), strict=True):
        expected = receipt["arms"][arm]
        predictions[arm] = decode(pinned_bytes(path, expected["prediction_sha256"], 64 * MIB,
                                               f"{arm} predictions"), list)
        groups = template.nonempty_by_image(predictions[arm], images)
        count = len(predictions[arm])
        nonempty = sum(map(len, groups))
        require(count == expected["prediction_count"] and nonempty == expected["nonempty_masks"]
                and count - nonempty == expected["all_zero_masks"], f"{arm} accepted candidate census differs")

    primary = scoring.paired_scores(official, gt, predictions, ids)
    require(primary == receipt["primary"], "Original primary scores differ from accepted P295 scores")
    pair = (gt, predictions["native"], predictions["center"])
    candidate = rank.analyze_pair(*pair, rank.load_support(code))
    fixed_match = geometry.evaluate_pair(*pair, helpers)
    require(fixed_match["prefix_rows_per_arm"] == candidate["prefix_counts"]["retained_per_arm"]
            and fixed_match["prefix_images"] == candidate["prefix_counts"]["images_with_prefix"],
            "Rank and fixed-match diagnostics use different candidate prefixes")
    contrasts = candidate["contrasts_points"]
    first = contrasts["score_on_native_geometry"]
    second = contrasts["score_on_center_geometry"]
    interaction = {key: None if first[key] is None or second[key] is None else second[key] - first[key]
                   for key in first}
    return dict(
        status=STATUS, DEV=True, TEST=False, training=False, model_forward_calls=0, device="CPU",
        posthoc_development_diagnostic=True, training_seed=5201, primary_GT="INTEGER_PRIMARY",
        images=receipt["images"], annotations=receipt["annotations"], distinct_jpeg=receipt["distinct_jpeg"],
        accepted_p295_sha256=ACCEPTED_SHA256, integer_gt_sha256=receipt["integer_gt_sha256"],
        prediction_sha256={mode: receipt["arms"][mode]["prediction_sha256"] for mode in ARMS},
        training_protocol=receipt["training_protocol"], evaluation_protocol=receipt["evaluation_protocol"],
        package_versions=packages, full_primary_scores_exact=True, primary=primary,
        equal_nonempty_candidate_analysis=candidate, score_donor_by_candidate_interaction_points=interaction,
        fixed_official_iou50_common_gt_geometry=fixed_match,
        candidate_effect_scope="different selected mask contents and native within-image order at equal counts",
        donor_effect_scope="cross-image ranking at fixed selected masks, within-image order and COCO matches",
        fixed_geometry_scope="paired mask measurement conditional on both arms matching the same GT at IoU 0.50",
    )


def deny_network(event, _):
    if event.startswith("socket."):
        raise PermissionError("P299 is a local CPU diagnostic")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("code", "accepted", "integer-gt", "native", "center", "output"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    args = parser.parse_args()
    require(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists(), "New absolute output in an existing canonical directory required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.addaudithook(deny_network)
    result = diagnose(args.code, args.accepted, args.integer_gt, args.native, args.center)
    payload = json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    print(STATUS)


if __name__ == "__main__":
    main()
