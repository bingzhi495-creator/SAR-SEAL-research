"""One offline intersection of two closed HRSID DEV705 large-mask associations."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
import sys
from types import ModuleType, SimpleNamespace


PROTOCOL = "39f88c2a6a09c013eb5ebcaabe5909feeb9c145fe828dac4019a576c08ebaaa4"
RF_RESULT = "415977eae1f821f615762994b1b7336c6a59f5fac07c27f7108444734a34db82"
MASK_RESULT = "7fc39fe3a49b92b0739a5261a0ec5ec1601f4dc6f81e49eb74daf82989046a75"
MASK_PRIMARY = "a6f71845ab7e0488e9ef7899005f1baf78ea6f2360f1fcc15a0e61f98d184e58"
GT = "393b3da233991dc5fafc98b4ddeaa341b042d9571434ef8ce53c3f2fdada5628"
MANIFEST = "39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f"
ARMS = ("native", "center")
FIELDS = ("continuous_iou_points", "fp_per_gt_area_points", "fn_per_gt_area_points")
AREA = "predicted_area_change_per_gt_area_points"
RF_FIELDS = dict(zip(FIELDS, ("polygon_iou", "polygon_fp_rate", "polygon_fn_rate")))


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path, digest):
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
         and 0 < path.stat().st_size <= 2**20, "Canonical bounded receipt or source required")
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Receipt or source digest mismatch: " + path.name)
    return raw


def load(path, digest, name):
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(pin(path, digest), str(path), "exec"), module.__dict__)
    return module


def original_args(receipt, digest):
    saved = json.loads(pin(receipt, digest))
    items = saved["args"]
    need(len(items) % 2 == 0 and all(key.startswith("--") for key in items[::2]),
         "Original CLI pair structure required")
    mapping = {key[2:].replace("-", "_"): Path(value) if not key.endswith("sha256")
               and key != "--training-protocol" else value for key, value in zip(items[::2], items[1::2])}
    need(len(mapping) == len(items) // 2, "Unique original CLI options required")
    return saved, SimpleNamespace(**mapping)


def tool_closed(terminal, result_sha):
    wait = terminal["exact_tool_wait"]
    message = json.loads(wait["output"].splitlines()[-1])
    need(wait["exit_code"] == 0 and message["result_sha256"] == result_sha,
         "Successful exact original tool wait required")
    return message


def load_rf(args, result, p360):
    accepted = json.loads(pin(args.accepted_result, p360.PRIMARY_SHA))
    prior = json.loads(pin(args.prior_result, p360.BASELINE_SHA))
    need(args.pair_sha256 == p360.PAIR_SHA and args.training_protocol == p360.TRAIN_PROTOCOL
         and args.accepted_result_sha256 == p360.PRIMARY_SHA and accepted["official_TEST_opened"] is False
         and accepted["training_seed"] == 5202 and result["image_manifest_sha256"] == MANIFEST,
         "Exact closed RF seed and original score lineage required")
    closed = p360.load(args.code_root / p360.P282_PATH, p360.P282_SHA, "_rf_closed_pair")
    gt, native, center, provenance = closed.verified_inputs(
        args.code_root, args.pair, p360.PAIR_SHA, native_arm_receipt=args.native_arm_receipt,
        center_arm_receipt=args.center_arm_receipt, training_protocol=p360.TRAIN_PROTOCOL)
    need(provenance == result["provenance"] == prior["provenance"]
         and provenance["original_dev_gt_sha256"] == accepted["original_dev_gt_sha256"] == GT
         and provenance["training_pair_receipt_sha256"] == accepted["independent_training_pair_sha256"],
         "Exact original RF prediction and vector GT provenance required")
    diagnostics = closed.load_diagnostics(args.code_root)
    failure = diagnostics["dev705_paired_mask_failure_scope"]
    score = diagnostics["dev705_score_template"]
    ordered, counts = score.prefix_rows(score.nonempty_by_image(native, gt["images"]),
                                       score.nonempty_by_image(center, gt["images"]))
    need(sum(counts) == result["equal_nonempty_prefix_rows_per_arm"] == 41482,
         "Exact RF equal-nonempty prefix required")
    adapter = failure.load_official_adapter()
    matches = {arm: p360.fixed_matches(gt, ordered[arm], adapter, failure.match_official) for arm in ARMS}
    need(len(matches["native"].keys() & matches["center"].keys()) == result["paired_gt"] == 1620,
         "Exact original RF paired associations required")
    return gt, matches


def measure_pairs(large, polygons, matches, geom, modules):
    measured = {}
    for aid in sorted(large & matches["native"].keys() & matches["center"].keys()):
        reference, _ = polygons[aid]
        axes = geom.hull(reference)
        values = {arm: geom.measure(reference, matches[arm][aid], axes, modules)[0] for arm in ARMS}
        measured[aid] = {arm: {key: float(values[arm][key]) for key in FIELDS} for arm in ARMS}
    return measured


def parity(arm_values, published):
    need(abs(mean(arm_values) - published) <= 1e-8, "Original published arm geometry mean differs")


def deltas(arms):
    result = {key: arms["center"][key] - arms["native"][key] for key in FIELDS}
    result[AREA] = result[FIELDS[1]] - result[FIELDS[2]]
    return result


def summary(rows, chips):
    chip_count = len(set(chips.values()))
    need(set(rows) == set(chips) and chip_count > 1, "Paired target and chip supports must reconcile")
    output = {}
    for field in (*FIELDS, AREA):
        values = [row[field] for row in rows.values()]
        dropped = [mean(row[field] for aid, row in rows.items() if chips[aid] != chip)
                   for chip in sorted(set(chips.values()))]
        positive = sum(value > 1e-12 for value in values)
        negative = sum(value < -1e-12 for value in values)
        output[field] = dict(mean=mean(values), positive=positive, negative=negative,
                             ties=len(values)-positive-negative, leave_one_chip_range=[min(dropped), max(dropped)])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("protocol", "rf-command", "rf-wait", "mask-command", "mask-wait", "mask-primary-wait", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    pin(args.protocol, PROTOCOL)
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
         and not args.output.exists() and not args.output.is_symlink(), "Fresh aggregate-only destination required")

    def offline(event, _):
        if event.startswith("socket."):
            raise PermissionError("Offline CPU diagnostic required")

    sys.addaudithook(offline)
    _, rf = original_args(args.rf_command, "55784a22c6a60ecc26f611195e3eb4948d3b4c7b4a7820f345be907c0fa8c4d0")
    mask_command, mask = original_args(args.mask_command,
        "e5927a6ff808060fe553e9e9593abf54e975479f85e9e4872eb6ecfdf76d1470")
    rf_wait = json.loads(pin(args.rf_wait, "3d9253592d24f764ef9d354fc4338d644882e06388655a1426a8c261236ab876"))
    need(rf_wait["original_os_exit"] == 0 and rf_wait["process_group_closed"] is True
         and rf_wait["result"]["sha256"] == RF_RESULT, "Original closed RF fixed-match result required")
    mask_wait = json.loads(pin(args.mask_wait, "1c4ad8b16bc10a3f33ba5986703ea15d097dfe6fde132bb81fad24990a852410"))
    marker = tool_closed(mask_wait, MASK_RESULT)
    need(marker["job_active_after_wait"] == marker["original_os_exit"] == 0,
         "Original Mask fixed-match job must be drained")
    primary_wait = json.loads(pin(args.mask_primary_wait,
        "008403e77abb68015498f74d904046cac14d3a5dea6d8437cbf22ac50d05a1a2"))
    tool_closed(primary_wait, MASK_PRIMARY)
    need(primary_wait["original_score_result_sha256"] == mask.score_sha256 == MASK_PRIMARY
         and primary_wait["original_inner_score_terminal_sha256"] == mask.score_terminal_sha256
         and primary_wait["original_cuda_export_owner_wait_sha256"] == mask.export_wait_sha256
         and primary_wait["closure_sha256s"] == dict(native=mask.native_closure_sha256, center=mask.center_closure_sha256),
         "Independent original Mask primary score exact binding required")
    code = mask.code_root
    need(rf.code_root == code and rf.shapely_site == mask.shapely_site
         and args.output not in (rf.output, mask.output), "One exact geometry runtime and independent output required")
    rf_result, mask_result = json.loads(pin(rf.output, RF_RESULT)), json.loads(pin(mask.output, MASK_RESULT))
    need(rf_result["DEV"] is mask_result["DEV"] is True
         and rf_result["official_TEST_opened"] is mask_result["official_TEST_opened"] is False,
         "Closed DEV-only results required")
    guard = load(code / "MASK_RTDETR_DEV705_EQUAL_PREFIX_20261112/equal_prefix.py",
                 "503ef0b1c7e808ba1e5d9b6e0ac39ae9ef5b967493e6965fb1bbe1f2b0767699", "_cpu_guard")
    guard.admit_worker()
    sys.path.append(str(mask.shapely_site))
    import shapely
    need(Path(shapely.__file__).resolve().is_relative_to(mask.shapely_site)
         and (shapely.__version__, shapely.geos_version_string) == ("2.0.2", "3.11.2"),
         "Original continuous geometry runtime required")
    geom = load(code / "MASK_RTDETR_DEV705_FIXED_MATCH_GEOMETRY_20261112/fixed_match_geometry.py",
                mask_command["worker_sha256"], "_mask_fixed_geometry")
    admission = geom.load_module(mask.admission_source, geom.ADMISSION_SHA, "_mask_closed_inputs")
    p360 = load(code / "HRSID_COMMON_DEV705_LARGE_FAILURE_20261111/large_core_fn.py",
                "224dd916d4d5672d8aaa9707f8b9233926d4590043ab754ff151b25ce5546acc", "_rf_fixed_matches")
    need(mask_result["original_gt_sha256"] == admission.GT_SHA == GT
         and mask_result["image_manifest_sha256"] == admission.MANIFEST_SHA == MANIFEST,
         "Identical byte-bound original reference geometry and image population required")

    rf_gt, rf_matches = load_rf(rf, rf_result, p360)
    mask_gt, streams, priors, provenance = admission.load_inputs(mask)
    need(mask_gt == rf_gt and len(mask_gt["images"]) == 705
         and all(len(streams[arm]) == 70500 for arm in ARMS)
         and provenance == mask_result["provenance_by_arm"],
         "Equal original GT and exact full native Mask streams required")
    mask_matches = geom.original_matches(mask_gt, streams, priors, geom.load_match_helpers(admission, code), admission)
    modules = admission.geometry_modules(code)
    polygons, _, _ = modules.geometry.polygon_support(mask_gt, modules.vector)
    ann = {row["id"]: row for row in mask_gt["annotations"]}
    large = {aid for aid, row in ann.items() if row["area"] >= 9216}
    need(len(large) == 16 and all(aid in polygons and not ann[aid].get("ignore", 0)
                                 and not ann[aid].get("iscrowd", 0) for aid in large),
         "Exact valid original large annotation support required")
    rf_values = measure_pairs(large, polygons, rf_matches, geom, modules)
    mask_values = measure_pairs(large, polygons, mask_matches, geom, modules)
    need(len(rf_values) == rf_result["strata"]["large"]["pairs"] == 15 and len(mask_values) == 14,
         "Original own-cohort support must reproduce")
    for arm in ARMS:
        for field in FIELDS:
            parity([row[arm][field] for row in rf_values.values()],
                   100 * rf_result["strata"]["large"]["arms"][arm][RF_FIELDS[field]])
            target = mask_result["paired_geometry"]["large"][field]
            need(target["eligible_pairs"] == len(mask_values), "Exact original Mask support required")
            parity([row[arm][field] for row in mask_values.values()], target[arm + "_mean"])

    common = sorted(rf_values.keys() & mask_values.keys())
    need(13 <= len(common) <= 14, "Mathematical intersection bound of original eligible pairs")
    chips = {aid: ann[aid]["image_id"] for aid in common}
    rf_changes = {aid: deltas(rf_values[aid]) for aid in common}
    mask_changes = {aid: deltas(mask_values[aid]) for aid in common}
    diff = {aid: {key: mask_changes[aid][key] - rf_changes[aid][key] for key in rf_changes[aid]} for aid in common}
    payload = dict(
        status="HRSID_DEV705_LARGE_FIXED_MATCH_INTERSECTION_ACTUAL", protocol_sha256=PROTOCOL,
        RF_original_fixed_geometry_sha256=RF_RESULT, Mask_original_fixed_geometry_sha256=MASK_RESULT,
        original_GT_sha256=GT, image_manifest_sha256=MANIFEST, original_own_cohort_arm_means_reproduced=True,
        DEV=True, official_TEST_opened=False, image_payload_opened=False, model_forward_calls=0, optimizer_updates=0,
        original_large_annotations=len(large), RF_both=len(rf_values), Mask_both=len(mask_values),
        both_architectures=len(common), contributing_DEV_chips=len(set(chips.values())),
        RF_candidate_scope="original equal-nonempty prefix, 41482 rows per arm",
        Mask_candidate_scope="original full native top-100, 70500 rows per arm",
        RF_center_minus_native=summary(rf_changes, chips), Mask_center_minus_native=summary(mask_changes, chips),
        Mask_delta_minus_RF_delta=summary(diff, chips))
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as out:
        need(out.write(raw) == len(raw), "Complete original result required")
    print(json.dumps(dict(status=payload["status"], result_sha256=sha256(raw).hexdigest()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()