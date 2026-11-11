"""Float-primary paired SSDD DEV187 endpoint adapter for independently accepted float E24 arms."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import math
from pathlib import Path
import os
import sys
from types import ModuleType, SimpleNamespace


ARMS = ("native", "center")
STAGE_PATH = "SSDD_DEV187_STAGE_CUDA_INIT_20261111/ssdd_dev187_stage_worker.py"
STAGE_SHA = "a491f847fab5411240967172792c73cc47574120be07367a3cf931b998aa0be0"
PROTOCOL_SHA = "4db9c71edda60dfa89bf2e9065ae5fcecdf23dbb4199e0aceb1a3556855d7ffc"
FLOAT_GT_SHA = "8d98e6f49200d5390c17bee66697de1f582133d08063259a561ed5f710be7b6c"
INTEGER_GT_SHA = "5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55"
REFERENCE_SHA = "9f273cab09b086d7f303cc98a2915349cf69cbfa5f36b03394e481b62f27354a"
REFERENCE_STATUS = "P370_SSDD_TWO_SEED_FIXED_DEV187_FLOAT_SENSITIVITY_COMPLETE"
HISTORICAL_AP = 0.7047324561474607
HISTORICAL_ACCEPTANCE_SHA = "97c53f1df007ddfc2cc0c200041a5b6246a40b4130c50e90151483e113d70a62"
TRAINING_PROTOCOL = "194e16f289a644430015f310dd63a3c1c1618cfd"
FIT_SOURCE_MANIFEST_SHA = "56fdd4be8d19b4e2e42ff736e89775f32b1e6ce2a9c91d2a90afad509640bc14"
FLOAT_SHORT_SHA = "78d2c6d1b99a1b05755324e061c1ad1bd4c5a460578e06e73720fa8c748ea8e0"
PAIR_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_E24_PAIR_INDEPENDENTLY_ACCEPTED_V1"
ARM_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_E24_ARM_INDEPENDENTLY_ACCEPTED_V1"
REQUEST_STATUS = "SSDD_FLOAT_PRIMARY_DEV187_STAGE_REQUEST_V1"
EXPORT_STATUS = "SSDD_FLOAT_COMMON_DEV187_IMAGE_ONLY_EXPORT_PENDING_INDEPENDENT_ACCEPTANCE_V1"
SCORE_STATUS = "SSDD_FLOAT_COMMON_DEV187_FLOAT_PRIMARY_SCORE_PENDING_INDEPENDENT_ACCEPTANCE_V1"
REPORT_EVENT = "SSDD_FLOAT_COMMON_DEV187_STAGE_REPORT_WRITTEN_V1"
EXPORT_ACCEPTED_STATUS = "SSDD_FLOAT_COMMON_DEV187_EXPORT_INDEPENDENTLY_ACCEPTED_V1"
FIT_ONLY = dict(FIT=True, DEV=False, TEST=False, AP=False)
METRICS = ("AP", "AP50", "AP75", "AP90", "APs", "APm", "APl", "AR100")
MIB = 1024**2


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def file_ref(path, expected):
    return dict(path=path.as_posix(), bytes=path.stat().st_size, sha256=expected)


def load_stock(code_root):
    path = code_root / STAGE_PATH
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
         "Exact canonical published stage source required")
    raw = path.read_bytes()
    need(len(raw) <= 128 * 1024 and sha256(raw).hexdigest() == STAGE_SHA, "Stock CUDA stage source changed")
    stage = ModuleType("_original_ssdd_dev_stage")
    stage.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), stage.__dict__)
    return stage, stage.load_published(code_root, scoring=True)


def score_apis(scorer):
    return SimpleNamespace(**{role: scorer.load_api(scorer.CODE / path, digest)
                              for role, (path, digest) in scorer.SOURCES.items()})


def score_record(value):
    return dict(value, APm=value["all12"][4], APl=value["all12"][5])


def delta(a, b):
    return {key: 100 * (b[key] - a[key]) if min(a[key], b[key]) >= 0 else None for key in METRICS}


def score_variants(scorer, apis, integer_gt, float_gt, predictions, ids):
    apis.sensitivity.validate_GT_pair(integer_gt, float_gt)
    out = {}
    for key, gt in (("primary", float_gt), ("sensitivity", integer_gt)):
        cell = {mode: score_record(scorer.score_one(apis.official, gt, predictions[mode], ids)) for mode in ARMS}
        out[key] = dict(cell, center_minus_native_points=delta(cell["native"], cell["center"]))
    return out


def historical_native(stage, reference):
    need(reference["sha256"] == REFERENCE_SHA, "Exact independently closed two-seed raster reference required")
    raw = stage.read_artifact(reference, MIB)
    value = json.loads(raw, object_pairs_hook=stage.unique, parse_constant=stage.reject_constant)
    need(value["status"] == REFERENCE_STATUS and value["DEV"] is True and value["TEST"] is False
         and value["regenerated_float_equal"] is True and value["images"] == 187 and value["annotations"] == 380
         and value["candidate_float_gt_sha256"] == FLOAT_GT_SHA and value["integer_gt_sha256"] == INTEGER_GT_SHA
         and value["max_dets"] == [1, 10, 100], "Exact closed historical raster population required")
    prior = value["seeds"]["5201"]
    metrics = prior["candidate_float"]["native"]
    need(prior["training_seed"] == 5201 and prior["primary_exact"] is True
         and prior["acceptance_sha256"] == HISTORICAL_ACCEPTANCE_SHA
         and metrics["AP"] == HISTORICAL_AP and len(metrics["all12"]) == 12,
         "Exact seed-5201 integer-trained native, same float-score reference required")
    return score_record(metrics)


def gate(native, center, reference):
    threshold = 1.0
    paired = 100 * (center["AP"] - native["AP"])
    against_old = 100 * (center["AP"] - reference["AP"])
    controlling = min(paired, against_old)
    if controlling >= threshold - 1e-12:
        status = "QUALIFIES_NEXT_FIXED_SEED"
    elif controlling > 1e-12:
        status = "AT_MOST_ONE_PRELOCKED_REPLICATION"
    else:
        status = "NO_PROMOTION_NONPOSITIVE"
    return dict(status=status, threshold_points=threshold, paired_center_minus_native_AP_points=paired,
                center_minus_integer_trained_native_AP_points=against_old,
                native_minus_integer_trained_native_AP_points=100 * (native["AP"] - reference["AP"]),
                controlling_margin_AP_points=controlling)


ARM_DIGESTS = ("launch_sha256", "checkpoint_sha256", "result_sha256", "epoch_audit_sha256",
               "visit_ledger_sha256", "stdout_sha256", "terminal_sha256", "original_outer_wait_sha256",
               "stock_audit_sha256", "selected_mask_stream_sha256")
PAIR_SHARED_DIGESTS = ("common_ledger_sha256", "identical_fields_sha256")
PAIR_SHARED_COUNTS = ("common_targets_total", "common_empty_visits", "changed_mask_visits")


def read_training_pair(api, pair_ref, *, native_arm_ref, center_arm_ref, training_protocol, evaluation_protocol):
    need(training_protocol == TRAINING_PROTOCOL and api.is_hex(evaluation_protocol, 40),
         "Exact accepted SSDD float training and fixed evaluation protocols required")
    pair = api.read_receipt(pair_ref)
    api.exact(pair, dict(status=PAIR_STATUS, training_protocol=TRAINING_PROTOCOL,
                         source_manifest_sha256=FIT_SOURCE_MANIFEST_SHA, qualification_pair_sha256=FLOAT_SHORT_SHA,
                         seed=5201, fit_images=741, completed_epochs_per_arm=24, visits_per_arm=17856,
                         optimizer_updates_per_arm=4464, ema_updates_per_arm=4464,
                         identical_nonmask_inputs_and_rng=True, identical_common_support=True, **FIT_ONLY),
              "Exact independently accepted floating/common E24 pair required",
              extra=("arms", "selected_mask_stream_sha256", *PAIR_SHARED_DIGESTS, *PAIR_SHARED_COUNTS))
    need(all(api.is_hex(pair[key]) for key in PAIR_SHARED_DIGESTS)
         and type(pair["common_targets_total"]) is int and pair["common_targets_total"] > 0
         and type(pair["common_empty_visits"]) is int and 0 <= pair["common_empty_visits"] < 17856
         and type(pair["changed_mask_visits"]) is int and 0 < pair["changed_mask_visits"] <= 17856,
         "Original common FIT accounting required")
    need(type(pair["arms"]) is type(pair["selected_mask_stream_sha256"]) is dict
         and set(pair["arms"]) == set(pair["selected_mask_stream_sha256"]) == set(ARMS),
         "Original distinct accepted floating native and center roles required")
    arms = {}
    for mode, ref in zip(ARMS, (native_arm_ref, center_arm_ref), strict=True):
        arm = api.read_receipt(ref)
        api.exact(pair["arms"][mode], dict(sha256=ref["sha256"], bytes=ref["bytes"]),
                  "Accepted float pair must bind exact arm receipt")
        api.exact(arm, dict(status=ARM_STATUS, qualification_pair_sha256=FLOAT_SHORT_SHA,
                            original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True,
                            completed_epochs=24, visits=17856, optimizer_updates=4464, ema_updates=4464,
                            filtered_empty_rows_total=0, **FIT_ONLY,
                            **{key: pair[key] for key in (*PAIR_SHARED_DIGESTS, *PAIR_SHARED_COUNTS)}),
                  "Exact independently closed float E24 original arm required",
                  extra=("phase", "operation_commit", "checkpoint_bytes", *ARM_DIGESTS))
        need(api.is_hex(arm["operation_commit"], 40) and all(api.is_hex(arm[key]) for key in ARM_DIGESTS)
             and type(arm["checkpoint_bytes"]) is int and 0 < arm["checkpoint_bytes"] <= 1024 * MIB
             and pair["selected_mask_stream_sha256"][mode] == arm["selected_mask_stream_sha256"],
             "Exact bounded original checkpoint and actual closed ledger identity required")
        fit_ids = arm["phase"]["fit_ids_sha256"]
        need(api.is_hex(fit_ids), "Original SSDD FIT IDs digest required")
        api.exact(arm["phase"], api.phase_marker(mode, TRAINING_PROTOCOL, fit_ids, FIT_SOURCE_MANIFEST_SHA),
                  "Exact saved float E24 phase marker required")
        arms[mode] = arm
    need(native_arm_ref["sha256"] != center_arm_ref["sha256"]
         and arms["native"]["phase"]["fit_ids_sha256"] == arms["center"]["phase"]["fit_ids_sha256"]
         and arms["native"]["visit_ledger_sha256"] == arms["center"]["visit_ledger_sha256"]
         and all(arms["native"][key] != arms["center"][key] for key in
                 ("checkpoint_sha256", "terminal_sha256", "original_outer_wait_sha256", "result_sha256",
                  "selected_mask_stream_sha256")),
         "Independent arm origins, shared FIT IDs and different selected mask streams required")
    return pair, arms


def export_float_arm(api, pair_ref, *, native_arm_ref, center_arm_ref, training_protocol, evaluation_protocol,
                     mode, archive_ref, dev_manifest_ref, dev_root, model_config, train_config,
                     output_directory, event_sink):
    need(mode in ARMS, "Exact original float-trained SSDD endpoint role required")
    _, arms = read_training_pair(api, pair_ref, native_arm_ref=native_arm_ref, center_arm_ref=center_arm_ref,
                                 training_protocol=training_protocol, evaluation_protocol=evaluation_protocol)
    arm = arms[mode]
    archive = api.reference(archive_ref, max_bytes=1024 * MIB, suffix=".ckpt")
    need(archive.name == api.ARCHIVE_NAME and archive_ref["sha256"] == arm["checkpoint_sha256"]
         and archive_ref["bytes"] == arm["checkpoint_bytes"], "Exact original independently accepted float E24 archive")
    items, ids = api.load_image_manifest(dev_manifest_ref, dev_root)
    original = api.load_sources()
    result = original["worker"].run_e24_dev_export(
        archive_ref, items, ids, model_config, train_config, recipe_sha256=api.RECIPE_SHA,
        validate_archive_progress=api.bind_archive_validator(arm, long_worker=original["long"]),
        boundary_api=original["boundary"], pilot_api=original["pilot"],
        export_e1_image=original["raw_export"].export_e1_image,
        canonical_export=original["canonical"].export_coco_mask_predictions,
        output_directory=output_directory, event_sink=event_sink,
    )
    return dict(status=EXPORT_STATUS, training_protocol=training_protocol, evaluation_protocol=evaluation_protocol,
                mode=mode, training_seed=5201, training_pair_sha256=pair_ref["sha256"],
                training_arm_sha256=(native_arm_ref if mode == "native" else center_arm_ref)["sha256"],
                image_manifest_sha256=api.IMAGE_MANIFEST_SHA, result=result)


def float_api(original):
    return SimpleNamespace(
        read_pair=lambda pair_ref, **kwargs: read_training_pair(original, pair_ref, **kwargs),
        export_arm=lambda pair_ref, **kwargs: export_float_arm(original, pair_ref, **kwargs),
        reference=original.reference, load_image_manifest=original.load_image_manifest, EXPORT_STATUS=EXPORT_STATUS)


def read_json(stage, ref, maximum=MIB):
    return json.loads(stage.read_artifact(ref, maximum), object_pairs_hook=stage.unique,
                      parse_constant=stage.reject_constant)


def read_request(stage_api, path, expected_sha, stage_name):
    need(stage_name in stage_api.STAGES and stage_api.hex_digest(expected_sha), "Exact float stage request required")
    value = read_json(stage_api, file_ref(path, expected_sha))
    export = stage_name in stage_api.EXPORTS
    extras = {"archive_ref", "dev_root"} if export else {"float_gt_ref", "integer_gt_ref", "historical_ref", *ARMS}
    stage_api.keys(value, stage_api.COMMON | extras, "Exact new float stage request roles required")
    need(value["status"] == REQUEST_STATUS and value["stage"] == stage_name
         and value["training_protocol"] == TRAINING_PROTOCOL
         and stage_api.hex_digest(value["evaluation_protocol"], 40), "Exact new SSDD float protocol binding required")
    refs = [stage_api.ref_path(value[key], MIB) for key in
            ("training_pair_ref", "native_arm_ref", "center_arm_ref", "image_manifest_ref")]
    need(value["image_manifest_ref"]["sha256"] == stage_api.IMAGE_MANIFEST_SHA, "Fixed DEV187 image manifest required")
    out = stage_api.path_value(value["output_directory"])
    need(out.resolve() == out and out.parent.is_dir() and not out.exists() and not out.is_symlink(),
         "Fresh separate float-stage output required")
    if export:
        archive = stage_api.ref_path(value["archive_ref"], 1024 * MIB, name="checkpoint_23.ckpt")
        dev_root = stage_api.path_value(value["dev_root"])
        need(not out.is_relative_to(dev_root) and not dev_root.is_relative_to(out) and not archive.is_relative_to(out),
             "Separate original archive, DEV images and export destination required")
        refs.append(archive)
    else:
        for key, digest in (("float_gt_ref", FLOAT_GT_SHA), ("integer_gt_ref", INTEGER_GT_SHA),
                            ("historical_ref", REFERENCE_SHA)):
            ref = value[key]
            refs.append(stage_api.ref_path(ref, MIB))
            need(ref["sha256"] == digest, "Exact prelocked score reference required: " + key)
        need(value["integer_gt_ref"]["bytes"] == stage_api.INTEGER_GT_BYTES, "Original integer GT length differs")
        for mode in ARMS:
            arm = value[mode]
            stage_api.keys(arm, {"closure", "report_ref", "predictions_ref", "export_manifest_ref",
                                 "checkpoint_ref", "acceptance_ref"}, "Exact separate accepted float export roles required")
            refs.extend(stage_api.validate_arm_request({key: item for key, item in arm.items() if key != "acceptance_ref"}))
            refs.append(stage_api.ref_path(arm["acceptance_ref"], MIB))
        need(value["native"]["closure"]["terminal_sha256"] != value["center"]["closure"]["terminal_sha256"],
             "Distinct original export child terminals required")
    need(len(set(refs)) == len(refs) and all(not ref.is_relative_to(out) for ref in refs),
         "All float-stage input roles and output must stay separate")
    return value


EXPORT_WAIT_STATUS = "SSDD_FLOAT_COMMON_DEV187_EXPORT_ORIGINAL_TOOL_WAIT_V1"
EXPORT_OWNER_CLOSED = "SSDD_FLOAT_COMMON_DEV187_EXPORT_ORIGINAL_OWNER_CLOSED_V1"


def verify_export_acceptance(stage, original, record, mode, *, pair_sha, evaluation_protocol):
    """Check the independent acceptance and the original process proofs it references."""
    accepted = read_json(stage, record["acceptance_ref"])
    closure = record["closure"]
    report_sha = closure["report_sha256"]
    original.exact(accepted, dict(status=EXPORT_ACCEPTED_STATUS, mode=mode, evaluation_protocol=evaluation_protocol,
                                  training_pair_sha256=pair_sha, export_report_sha256=report_sha),
                   "Exact independent float export acceptance required",
                   extra=("request_sha256", "original_child_terminal_ref", "original_outer_wait_ref"))
    req_sha = accepted["request_sha256"]
    terminal_ref = accepted["original_child_terminal_ref"]
    need(original.is_hex(req_sha) and terminal_ref["sha256"] == closure["terminal_sha256"],
         "Original request and independent float child terminal identities required")
    terminal_path = stage.ref_path(terminal_ref, MIB, name="OWNER_TERMINAL_PRIVATE.json")
    terminal = read_json(stage, terminal_ref)
    need(original.matches(terminal, dict(original_os_wait=True, original_os_exit=0, job_assigned=True,
                                         job_active_before_drain=0, job_active_after_wait=0))
         and terminal["stop_reason"] is None and type(terminal["owner"]) is dict,
         "Exact natural float export JobObject terminal required")
    stdout_ref = terminal["stdout_ref"]
    need(stage.ref_path(stdout_ref, 8 * MIB, name="OWNER.stdout").parent == terminal_path.parent,
         "Original stdout must be tied to the contained float export terminal")
    marker = json.loads(stage.read_artifact(stdout_ref, 8 * MIB).splitlines()[-1])
    stage_name = "EXPORT_" + mode.upper()
    need(marker == dict(status=REPORT_EVENT, stage=stage_name, request_sha256=req_sha, report_sha256=report_sha),
         "Exact float image-only worker footer must bind its original export")
    outer = read_json(stage, accepted["original_outer_wait_ref"], 8 * MIB)
    join = dict(stage=stage_name, request_sha256=req_sha, export_report_sha256=report_sha,
                original_child_terminal_sha256=terminal_ref["sha256"])
    original.exact(outer, dict(status=EXPORT_WAIT_STATUS, **join),
                   "Exact retained original external float export tool wait required",
                   extra=("original_session_id", "exact_tool_wait"))
    wait = outer["exact_tool_wait"]
    need(type(outer["original_session_id"]) is int and outer["original_session_id"] > 0
         and type(wait["exit_code"]) is int and wait["exit_code"] == 0 and wait.get("session_id") is None
         and json.loads(wait["output"].splitlines()[-1]) == dict(status=EXPORT_OWNER_CLOSED, **join),
         "Exact original outer float export wait and footer required")
    return outer["original_session_id"]


def load_export(stage, original, scorer, record, mode, trained, *, pair_sha, arm_sha, evaluation_protocol):
    """Validate the new float report and reuse the original candidate format."""
    archive = record["checkpoint_ref"]
    need(archive["sha256"] == trained["checkpoint_sha256"] and archive["bytes"] == trained["checkpoint_bytes"],
         "Float scored arm must retain its own independently accepted E24 archive")
    report = read_json(stage, record["report_ref"])
    original.exact(report, dict(status=EXPORT_STATUS, training_protocol=TRAINING_PROTOCOL,
                                evaluation_protocol=evaluation_protocol, mode=mode, training_seed=5201,
                                training_pair_sha256=pair_sha, training_arm_sha256=arm_sha,
                                image_manifest_sha256=stage.IMAGE_MANIFEST_SHA),
                   "Exact float report and original arm/protocol/image identities required", extra=("result",))
    result = report["result"]
    need(original.matches(result, dict(status="NATIVE_E24_EMA_DEV187_EXPORT_COMPLETE_PENDING_ROOT", fixed_epoch=24,
                                       optimizer_updates=4464, checkpoint_sha256=archive["sha256"],
                                       recipe_sha256=original.RECIPE_SHA, image_count=187, native_count_total=18700,
                                       full_DEV=True, qualified=False, terminal_self_certified=False, GT_read=False,
                                       TEST=False, AP=False, training=False, tensor_precision="float32",
                                       autocast=False, checkpoint_weights_only_cpu=True,
                                       float32_matmul_precision="highest", cudnn_tf32=False)),
         "Exact original image-only native CUDA FP32 fixed endpoint required")
    prediction_ref, index_ref = record["predictions_ref"], record["export_manifest_ref"]
    need(result["predictions_ref"] == prediction_ref and result["image_manifest_ref"] == index_ref,
         "Exact unchanged output prediction and native index identities required")
    rows = read_json(stage, prediction_ref, 64 * MIB)
    index = read_json(stage, index_ref, 4 * MIB)
    totals = {name: result[name] for name in scorer.EXPORT_TOTALS}
    need(all(type(value) is int and 0 <= value <= 18700 for value in totals.values())
         and type(rows) is list and totals["prediction_count"] == len(rows)
         and totals["prediction_count"] + totals["unmapped_slot1_total"] == 18700
         and totals["nonempty_masks"] + totals["all_zero_masks"] == len(rows)
         and totals["codec_roundtrip_masks"] == len(rows), "Exact original SSDD mapped and native candidate accounting")
    original.exact(index, dict(schema="RFDETR-E24-canonical-DEV-image-index-v1",
                               checkpoint_sha256=archive["sha256"], recipe_sha256=original.RECIPE_SHA,
                               prediction_offset_base=0, native_rank_base=1, predictions_ref=prediction_ref, **totals),
                   "Original SSDD native index and candidate accounting required", extra=("images",))
    timing = result["timing_seconds"]
    need(type(timing) is dict and all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                                     for value in timing.values())
         and {"synchronized_cuda_forward", "worker_elapsed"} <= timing.keys()
         and all(type(result[name]) is int and result[name] >= 0 for name in
                 ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")),
         "Original SSDD runtime aggregates required")
    safe = dict(checkpoint_sha256=archive["sha256"], training_arm_sha256=arm_sha,
                export_report_sha256=record["report_ref"]["sha256"], prediction_sha256=prediction_ref["sha256"],
                export_manifest_sha256=index_ref["sha256"],
                independent_export_acceptance_sha256=record["acceptance_ref"]["sha256"],
                original_export_terminal_sha256=record["closure"]["terminal_sha256"], timing_seconds=timing,
                cuda_peak_allocated_bytes=result["cuda_peak_allocated_bytes"],
                cuda_peak_reserved_bytes=result["cuda_peak_reserved_bytes"], **totals)
    return index, rows, safe


def score_closed_float_pair(stage, original, scorer, request):
    pair_ref = request["training_pair_ref"]
    train = stage.train_args(request)
    _, trained = read_training_pair(original, pair_ref, **train)
    need(request["integer_gt_ref"]["sha256"] == INTEGER_GT_SHA
         and request["integer_gt_ref"]["bytes"] == stage.INTEGER_GT_BYTES
         and request["float_gt_ref"]["sha256"] == FLOAT_GT_SHA,
         "Exact historical integer and authenticated source-float score references required")
    evaluation = request["evaluation_protocol"]
    pair_sha = pair_ref["sha256"]
    sessions = {mode: verify_export_acceptance(stage, original, request[mode], mode, pair_sha=pair_sha,
                                                evaluation_protocol=evaluation) for mode in ARMS}
    need(len(set(sessions.values())) == len(ARMS), "Distinct natural float export owner waits required")
    sys.addaudithook(stage.scope_guard(allowed=())[0])
    manifest = request["image_manifest_ref"]
    image_artifact = scorer.Artifact(manifest, stage.read_artifact(manifest, MIB))
    _, ids, items = scorer.fixed_images(image_artifact)
    inputs, arms = {}, {}
    for mode in ARMS:
        index, rows, arms[mode] = load_export(stage, original, scorer, request[mode], mode, trained[mode],
                                              pair_sha=pair_sha, arm_sha=train[mode + "_arm_ref"]["sha256"],
                                              evaluation_protocol=evaluation)
        inputs[mode] = (index, rows)
    integer_gt = read_json(stage, request["integer_gt_ref"])
    float_gt = read_json(stage, request["float_gt_ref"])
    for gt in (integer_gt, float_gt):
        scorer.validate_gt(gt, ids, items)
    apis = score_apis(scorer)
    predictions = {}
    for mode, (index, rows) in inputs.items():
        document = apis.bridge.group_canonical_rows(index, rows, items)
        flat, _ = apis.adapter.normalize_export(document, float_gt["images"])
        need(flat == rows, "Retain exact mapped candidate streams, mask rows, rank and confidence")
        predictions[mode] = flat
    output = score_variants(scorer, apis, integer_gt, float_gt, predictions, ids)
    reference = historical_native(stage, request["historical_ref"])
    a, b = (output["primary"][mode] for mode in ARMS)
    return dict(status=SCORE_STATUS, training_protocol=TRAINING_PROTOCOL, evaluation_protocol=evaluation,
                protocol_sha256=PROTOCOL_SHA, training_pair_sha256=pair_sha, image_manifest_sha256=manifest["sha256"],
                integer_gt_sha256=INTEGER_GT_SHA, floating_gt_sha256=FLOAT_GT_SHA,
                images=187, annotations=380, max_dets=[1,10,100], seed=5201,
                primary_GT="OFFICIAL_SOURCE_FLOAT_PRIMARY", sensitivity_GT="LEGACY_INTEGER_SENSITIVITY",
                historical_integer_trained_native_float_score=reference, historical_reference_sha256=REFERENCE_SHA,
                center_minus_historical_native_float_points=delta(reference, b),
                native_minus_historical_native_float_points=delta(reference, a),
                first_seed_gate=gate(a, b, reference), arms=arms, FIT=False, DEV=True, TEST=False,
                AP=True, training=False, model_forward_calls=0, **output)


def main(argv=None):
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--stage", required=True, choices=("EXPORT_NATIVE", "EXPORT_CENTER", "SCORE"))
    args = parser.parse_args(argv)
    stage, modules = load_stock(args.code_root)
    stage.admit_runtime(args.stage)
    need(args.stage != "SCORE" or all(os.environ.get(key) == "1" for key in
                                      ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
         "Independent SSDD CPU score must run on one thread")
    stage.read_artifact(file_ref(args.protocol, PROTOCOL_SHA), MIB)
    request = read_request(stage, args.request, args.request_sha256, args.stage)
    sys.addaudithook(stage.reject_network)
    original = modules["ssdd_phase_dev_admission"]
    output = stage.path_value(request["output_directory"])
    if args.stage == "SCORE":
        result = score_closed_float_pair(stage, original, modules["ssdd_phase_dev_score"], request)
        output.mkdir()
        report = output / stage.SCORE_REPORT
    else:
        result = stage.run_export(request, float_api(original), modules["ssdd_phase_init"], event_sink=stage.print_event)
        report = output / stage.EXPORT_REPORT
    raw = stage.json_bytes(result)
    with report.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    stage.print_event(dict(status=REPORT_EVENT, stage=args.stage, request_sha256=args.request_sha256,
                           report_sha256=sha256(raw).hexdigest()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
