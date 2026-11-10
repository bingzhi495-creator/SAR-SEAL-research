"""Analyze the closed common-support HRSID DEV705 pair without rerunning inference."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
GROUP_SOURCE = "HRSID_MASK_PHASE_REPEAT_DEV705_20261109"
GROUP_ADMISSION_SHA = "079a3a4d1d00ed8719a15f376c2aabb34056c4da8d1b1db1d85c621b23c25033"
MANIFEST_SHA = "39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f"
PRIMARY_STATUS = "HRSID_MASK_COMMON_FIT_E24_DEV705_INDEPENDENTLY_ACCEPTED_V1"


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code-root", "pair", "native-arm-receipt", "center-arm-receipt",
                 "image-manifest", "accepted-result", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("pair-sha256", "training-protocol", "accepted-result-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if (not args.output.is_absolute() or args.output.parent.resolve(strict=True) != args.output.parent
            or args.output.exists() or args.output.is_symlink()):
        parser.error("A fresh canonical aggregate output file is required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        parser.error("Run this closed-prediction diagnostic on the CPU")
    if args.code_root.resolve(strict=True) != args.code_root:
        parser.error("Canonical code root required")

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(args.code_root / GROUP_SOURCE))
    from closed_common_diagnostics import analyze_diagnostics, load_source, verified_inputs
    from area_group_sensitivity import analyze_group_composition

    admission = load_source(args.code_root / GROUP_SOURCE / "dev705_group_admission.py",
                            GROUP_ADMISSION_SHA, "_common_large_group_admission")
    driver = admission.load_driver()
    primary = driver.read_json(args.accepted_result, args.accepted_result_sha256)
    driver.require(primary["status"] == PRIMARY_STATUS and primary["images"] == 705
                   and primary["primary_annotations"] == 1782 and primary["official_TEST_opened"] is False
                   and primary["training_protocol"] == args.training_protocol and primary["training_seed"] == 5202,
                   "Exact closed common-support DEV705 primary result required")
    gt, native, center, provenance = verified_inputs(
        args.code_root, args.pair, args.pair_sha256, native_arm_receipt=args.native_arm_receipt,
        center_arm_receipt=args.center_arm_receipt, training_protocol=args.training_protocol,
    )
    driver.require(provenance["training_pair_receipt_sha256"] == primary["independent_training_pair_sha256"]
                   and provenance["evaluation_protocol"] == primary["evaluation_protocol"]
                   and provenance["original_dev_gt_sha256"] == primary["original_dev_gt_sha256"],
                   "The supplied score result must bind the same evaluated training pair and GT")
    pair = driver.read_json(args.pair, args.pair_sha256)
    manifest_ref = dict(path=args.image_manifest.as_posix(), sha256=MANIFEST_SHA)
    group_gt, _, groups = admission.load_population(driver, manifest_ref, pair["gt"])
    driver.require(group_gt == gt, "Group membership and geometry must use the same admitted ground truth")

    expected = {
        mode: dict(AP=primary["scores"][mode]["primary"]["AP"],
                   AP75=primary["scores"][mode]["primary"]["AP75"],
                   APl=primary["scores"][mode]["primary"]["all12"][5])
        for mode in ("native", "center")
    }
    composition = analyze_group_composition(gt, native, center, groups, expected)
    print("CLOSED_COMMON_DEV705_GROUP_COMPOSITION_COMPLETE", flush=True)
    diagnostics = analyze_diagnostics(args.code_root, gt, native, center)
    result = dict(status="HRSID_COMMON_E24_DEV705_LARGE_FAILURE_DIAGNOSTIC_COMPLETE_V1",
                  provenance=provenance, accepted_result_sha256=args.accepted_result_sha256,
                  image_manifest_sha256=MANIFEST_SHA, group_map_sha256=admission.GROUP_MAP_SHA256,
                  source_sha256={
                      name: sha256((HERE / name).read_bytes()).hexdigest() for name in (
                          "run_common_large_followup.py", "closed_common_diagnostics.py", "area_group_sensitivity.py")},
                  closed_predictions_only=True, official_TEST_opened=False,
                  primary_original_scores=expected, group_composition=composition, diagnostics=diagnostics)
    encoded = (json.dumps(result, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(encoded)
    print(json.dumps(dict(status=result["status"], bytes=len(encoded), sha256=sha256(encoded).hexdigest())),
          flush=True)


if __name__ == "__main__":
    main()
