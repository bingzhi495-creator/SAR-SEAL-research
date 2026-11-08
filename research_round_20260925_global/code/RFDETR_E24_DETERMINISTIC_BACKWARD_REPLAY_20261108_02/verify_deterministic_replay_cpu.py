"""Verify the closed P94 numerical replay without opening any active science inputs."""
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
D = HERE.parent
PAIR = D / "RFDETR_E24_DETERMINISTIC_BACKWARD_REPLAY_ACTUAL_PRIVATE_20261108_02"
NATIVE = D / "RFDETR_E24_FIRST_GRAD_REPLAY_ACTUAL_PRIVATE_20261108_01"
COMPARATOR = D / "RFDETR_E24_FIRST_GRAD_REPLAY_CANDIDATE_20261108_01/compare_first_gradient_replay.py"
WORKER = D / "RFDETR_E24_DETERMINISTIC_BACKWARD_REPLAY_CANDIDATE_20261108_02/deterministic_sampling_replay.py"
WORKER_SHA = "7939d724a8b7b5ff66c9ab7fa7022efe272382f7db31b427875cc02d81c7a4fa"
PROTOCOL = "dca8e55719146b64b234059772e83b254871ebe0"
NATIVE_PROTOCOL = "21e20c368d130cd635cf0459a46e6b53327571a8"
STRICT_STATUS = "STRICT_NATIVE_BACKWARD_AND_CPU_SAMPLING_REPLAY_COMPLETE_NO_UPDATE"
COMBINED_STATUS = "STOCK_CUDA_FORWARD_CPU_GRID_AND_RESIZE_BACKWARD_PROBE_COMPLETE_NO_UPDATE"
NATIVE_STATUS = "RFDETR_FIRST_GRADIENT_REPLAY_COMPLETE_NO_UPDATE"
NAMES = ("F0", "F1")
POLICIES = ("original_training_mode", "float32_matmul_precision", "cudnn_allow_tf32", "deterministic_algorithms")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def gate(forward_exact, relative, cosine):
    if not forward_exact:
        return "FORWARD_IDENTITY_GATE_FAILED"
    if relative is None or cosine is None or not math.isfinite(relative) or not math.isfinite(cosine):
        return "FIRST_GRADIENT_GATE_FAILED"
    return "FIRST_GRADIENT_GATE_PASS" if relative <= 1e-5 and cosine >= .999999 else "FIRST_GRADIENT_GATE_FAILED"


def get_comparator():
    pins = {
        COMPARATOR: "cc3268707e29cb3668f0107515c7a0c56ad030741db795be562b47e89ab6a1f3",
        COMPARATOR.with_name("first_gradient_replay.py"): "b35340218f6b557a915417a352ce6ca5270a368375a94e2f6b0b3178d2545314",
    }
    for path, digest in pins.items():
        need(path.resolve(strict=True) == path and sha256(path.read_bytes()).hexdigest() == digest,
             "Exact closed comparator support")
    spec = importlib.util.spec_from_file_location("_strict_replay_closed_comparator", COMPARATOR)
    cmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp)
    return cmp


def no_updates(record):
    return (record["optimizer_updates"] == record["ema_updates"] == 0
            and all(record[key] is False for key in ("AP", "DEV", "TEST")))


def check_native(result, log, attempt, protocol):
    need(result["status"] == log["status"] == NATIVE_STATUS
         and result["attempt"] == log["attempt"] == attempt
         and result["arm"] == "F" and result["seed"] == 5201 and result["protocol_commit"] == protocol
         and len(result["microbatches"]) == log["microbatches"] == 4
         and result["clip"]["preclip_global_norm"] == log["preclip_global_norm"] > 0
         and math.isfinite(log["preclip_global_norm"])
         and result["original_training_mode"] is True and result["deterministic_algorithms"] is False
         and result["init"]["initial_adam_step"] == result["init"]["initial_state_slots"] == 0
         and log["optimizer_updates"] == log["ema_updates"] == 0 and no_updates(result),
         "Exact closed zero-update first-window result")


def check_terminal(cmp, root, protocol, exits):
    terminal = cmp.read_json(root / "ORIGINAL_TERMINAL_PRIVATE.json")
    need(terminal["protocol"] == protocol and terminal["exits"] == exits
         and terminal["original_OS_waits"] is True and all(
             terminal[k] is False for k in ("automatic_retry", "training", "DEV", "TEST")),
         "Exact genuine zero-exit closed original worker pair")


def strict_attempt(cmp, name):
    base = PAIR / name
    control = PAIR / "CONTROL" / name
    receipt = cmp.read_json(control / "TERMINAL_PRIVATE.json")
    log = cmp.worker_stdout(PAIR, name)
    lines = cmp.read_ref(receipt["stdout_ref"], control / "WORKER.stdout", cmp.MIB, read=True).splitlines()
    need(len(lines) == 3, "Exact three-stage worker output")
    native_log, combined_log, strict_log = map(json.loads, lines)
    native_path = base / "RESULT.json"
    combined_path = base / "COMBINED_SAMPLING_BACKWARD_PROBE_PRIVATE.json"
    native = cmp.read_json(native_path)
    combined = cmp.read_json(combined_path)
    strict = cmp.read_json(base / "DETERMINISTIC_BACKWARD_PRIVATE.json")
    check_native(native, native_log, name, PROTOCOL)
    native_sha = sha256(native_path.read_bytes()).hexdigest()
    need(log == strict_log == strict and strict["status"] == STRICT_STATUS and no_updates(strict)
         and strict["strict_backward_calls"] == 4
         and math.isfinite(strict["strict_backward_seconds"]) and strict["strict_backward_seconds"] > 0
         and strict["strict_warn_only"] is False and strict["cuda_forward_deterministic_flag"] is False
         and strict["cublas_workspace"] is None
         and strict["native_forward_result_sha256"] == native_sha
         and strict["combined_result_sha256"] == sha256(combined_path.read_bytes()).hexdigest(),
         "Strict worker log binds both completed trace files")
    need(combined["status"] == combined_log["status"] == COMBINED_STATUS
         and combined["attempt"] == combined_log["attempt"] == name
         and combined["protocol_commit"] == PROTOCOL
         and combined["native_replay_result_sha256"] == native_sha
         and combined_log["optimizer_updates"] == combined_log["ema_updates"] == 0
         and no_updates(combined), "Combined route log binds the native replay")
    routes = {}
    for key, count, logged, unhandled in (
        ("grid_sample", 36, "grid_cpu_backward_calls", "unchanged_differentiable_cuda_bilinear_calls"),
        ("bilinear_interpolate", 8, "resize_cpu_backward_calls", "unchanged_cuda_bilinear_antialias_calls"),
    ):
        route = combined[key]
        shapes = route["observed_shapes"]
        need(route[unhandled] == 0 and route["routed_forward_calls"] == route["cpu_backward_calls"]
             == combined_log[logged] == count
             and sum(row["cpu_backward_calls"] for row in shapes) == count
             and all(row["cpu_backward_calls"] == row["routed_forward_calls"] > 0 for row in shapes),
             "Exact completed CPU backward calls by operation and observed shape")
        routes[key] = count
    return native, strict["strict_backward_seconds"], routes, receipt["wall_seconds"]


def main():
    import torch

    need(not torch.cuda.is_available(), "CPU-only verifier with CUDA hidden")
    torch.set_num_threads(2)
    cmp = get_comparator()
    check_terminal(cmp, PAIR, PROTOCOL, dict.fromkeys(NAMES, 0))
    request = cmp.read_json(PAIR / "ORIGINAL_REQUEST_PRIVATE.json")
    need(request["protocol"] == PROTOCOL and request["attempts"] == list(NAMES)
         and request["planned_per_attempt_FIT_microbatches"] == 4 and request["new_optimizer_updates"] == 0
         and request["cublas_workspace_override"] is None
         and all(request[key] is False for key in ("AP", "DEV", "TEST", "automatic_retry"))
         and request["worker"]["sha256"] == WORKER_SHA, "Exact prospective worker request")
    cmp.read_ref(request["worker"], WORKER, cmp.MIB)
    check_terminal(cmp, NATIVE, NATIVE_PROTOCOL, dict.fromkeys(cmp.ATTEMPTS, 0))
    source = cmp.load_pair()
    sys.addaudithook(source.offline)
    results, gradients, checks, routing, wall, strict_seconds, drift = {}, {}, {}, {}, {}, {}, {}
    for name in NAMES:
        result, strict_seconds[name], routing[name], wall[name] = strict_attempt(cmp, name)
        baseline = cmp.read_json(NATIVE / name / "RESULT.json")
        check_native(baseline, cmp.worker_stdout(NATIVE, name), name, NATIVE_PROTOCOL)
        need(all(result[key] == baseline[key] for key in (*POLICIES, "init", "tape")),
             "Identical archived start, FIT tape and native forward policy")
        stage_rows = cmp.compare_stages(baseline, result)
        checks[name] = {key: sum(key not in row["differing_stages"] for row in stage_rows) for key in cmp.STAGES[:11]}
        checks[name]["rng_after_backward_and_clip"] = int(
            result["clip"]["rng_after_backward_and_clip"] == baseline["clip"]["rng_after_backward_and_clip"])
        gradients[name] = cmp.gradient(source, PAIR, name, result["gradient_ref"],
                                       result["clipped_gradient_sha256"], expected_tensors=543)
        native_gradient = cmp.gradient(source, NATIVE, name, baseline["gradient_ref"],
                                       baseline["clipped_gradient_sha256"], expected_tensors=543)
        drift[name] = cmp.gradient_comparison(native_gradient, gradients[name])
        del native_gradient
        results[name] = result
    stages = cmp.compare_stages(results["F0"], results["F1"])
    exact = all(check[key] == 4 for check in checks.values() for key in cmp.STAGES[:11])
    exact = exact and all(check["rng_after_backward_and_clip"] == 1 for check in checks.values())
    diff = cmp.gradient_comparison(gradients["F0"], gradients["F1"])
    outcome = gate(exact, diff["all"]["rel_l2_to_left"], diff["all"]["cosine"])
    summary = dict(status="ACTUAL_STRICT_BACKWARD_FIRST_GRADIENT_CPU_INDEPENDENTLY_VERIFIED",
                   decision=outcome, protocol_commit=PROTOCOL, original_OS_waits=True, attempts=list(NAMES),
                   native_forward_and_rng_exact=exact, native_comparison_checks=checks,
                   strict_replay_pair_stage_differences=stages, actual_CPU_backward_routes=routing,
                   actual_strict_backward_calls=dict.fromkeys(NAMES, 4),
                   strict_backward_seconds=strict_seconds, worker_wall_seconds=wall,
                   preclip_global_norm={name: result["clip"]["preclip_global_norm"] for name, result in results.items()},
                   strict_replay_pair_gradient=diff, native_vs_strict_gradient_drift=drift,
                   optimizer_updates=0, ema_updates=0, AP=False, DEV=False, TEST=False)
    with (HERE / "STRICT_REPLAY_CPU_VERIFIED_SAFE.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(decision=outcome, exact=exact, gradient=diff["all"],
                          first_differences=[row["first_difference"] for row in stages]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
