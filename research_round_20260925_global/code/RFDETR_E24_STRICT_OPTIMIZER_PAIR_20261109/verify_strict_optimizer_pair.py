"""Read back a closed P96 training pair and independently qualify the fixed DEV handoff."""
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
D = HERE.parent
BASE = D / "RFDETR_E24_STRICT_OPTIMIZER_ACTUAL_PRIVATE_20261109_01"
OWNER = D / "RFDETR_E24_STRICT_OPTIMIZER_OWNER_PRIVATE_20261109_01"
STATE = D / "authoritative_operations_20261021_01/ACTIVE_RESEARCH_STATE.json"
PROFILE = "RFDETR_E24_STRICT_OPTIMIZER_PAIR_GPU_20261109_01"
CANDIDATE = D / "RFDETR_E24_OPTIMIZER_PRESERVATION_TRAIN_CANDIDATE_20261108_01"
WORKER = D / "RFDETR_E24_STRICT_OPTIMIZER_TRAIN_CANDIDATE_20261109_01/run_strict_optimizer_arm.py"
P95 = D / "P95_RFDETR_E24_STRICT_BACKWARD_REPLAY_ACTUAL_REMOTE_READBACK_20261109.json"
P96 = D / "P96_RFDETR_E24_STRICT_OPTIMIZER_PAIR_PROTOCOL_REMOTE_READBACK_20261109.json"
REPLAY = D / "RFDETR_E24_DETERMINISTIC_BACKWARD_REPLAY_VERIFIER_PRIVATE_20261109_01/STRICT_REPLAY_CPU_VERIFIED_SAFE.json"
REPLAY_SHA = "cf88ff4580df2abf2c013b3fca90e673987911c950311bc7c95177f78c21d6b5"
WORKER_SHA = "09532725bad85f9cd1e5f61cd652f5f2d5203dae406f777702c8328b4817f43a"
PROTOCOL = "5c0d26c34483a002d0534241d665c7cbb8e5aa0d"
FIRST_GRADIENT_SHA256 = "ddad103f09d69255ee35f5e7cc755d850be88b49844e2550a4f1fa0c15914f21"
HELPERS = {
    "compare_state_pair.py": "8eedc1e10435c12ed913bddba13a1e235ff1259a8901ce9d4860716dcde058e0",
    "state_preservation.py": "8dda41131a2930aacf00c51dbe8cfa4817c0a3cd4b02c6bb606948db77341678",
}
SAFE = BASE / "ROOT_SAFE_TRAINING_RESULT.json"
CLOSED = BASE / "ROOT_CLOSED_PAIR_ACCEPTANCE_PRIVATE.json"
RUNNING = "ACTUAL_E24_STRICT_OPTIMIZER_STATE_PAIR_RUNNING"
ACCEPTED = "ACTUAL_E24_OPTIMIZER_STATE_PAIR_ACCEPTED"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def verify_backward_control(row):
    need(type(row["strict_calls"]) is int and row["strict_calls"] == 1488
         and row["first_counts"] == [4, 36, 8] and row["first_gradient_sha256"] == FIRST_GRADIENT_SHA256
         and row["cublas_workspace_override"] is None and math.isfinite(row["strict_seconds"])
         and row["strict_seconds"] > 0, "Exact strict-backward policy and first window")
    counts = dict(strict_calls=row["strict_calls"], strict_seconds=row["strict_seconds"],
                  first_counts=row["first_counts"])
    for name, unchanged, minimum in (
        ("grid_sample", "unchanged_differentiable_cuda_bilinear_calls", 36),
        ("bilinear_interpolate", "unchanged_cuda_bilinear_antialias_calls", 8),
    ):
        route = row[name]
        calls = route["cpu_backward_calls"]
        shapes = route["observed_shapes"]
        need(type(calls) is int and calls >= minimum and route["routed_forward_calls"] == calls
             and route[unchanged] == 0 and shapes
             and all(type(x["cpu_backward_calls"]) is int
                     and x["routed_forward_calls"] == x["cpu_backward_calls"] > 0 for x in shapes)
             and sum(x["cpu_backward_calls"] for x in shapes) == calls, "Complete routed CPU backward for " + name)
        counts[name + "_cpu_backward_calls"] = calls
    return counts


def bounded_bytes(path, limit=1024**2):
    need(path.resolve(strict=True) == path and path.is_file(), "Exact closed file " + path.name)
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    need(0 < len(data) <= limit, "Bounded closed file " + path.name)
    return data


def read(path, limit=1024**2):
    return json.loads(bounded_bytes(path, limit))


def last_line(path):
    return json.loads(bounded_bytes(path, 4*1024**2).splitlines()[-1])


def main():
    sys.path.insert(0, str(D / "RFDETR_FIT_TEACHER_BOUNDARY_CANDIDATE_20261108_01"))
    from accept_optimizer_pair import live
    from root_fit_teacher_boundary import RUNTIME, RSS_CAP, ref, windows_runtime, write

    profile = read(STATE, 16*1024**2)[PROFILE]
    birth = profile["active_attempt"]
    need(profile["status"] == RUNNING and birth["protocol_commit"] == PROTOCOL
         and not live(birth), "Original pair owner must be closed before artifact readback")
    external = read(OWNER / "OWNER_BIRTH_PRIVATE.json")
    start = datetime.fromisoformat(external["start_utc"][:26] + "+00:00").timestamp()
    need(external["pid"] == birth["pid"] and external["protocol_commit"] == PROTOCOL
         and abs(start - birth["creation_time"]) < .002, "Exact external owner birth")
    terminal = read(BASE / "PAIR_ORIGINAL_TERMINAL_PRIVATE.json")
    need(terminal == dict(exits=dict(F=0, S=0), original_OS_waits=True, automatic_retry=False,
                          TEST=False, DEV=False, protocol=PROTOCOL)
         and last_line(OWNER / "OWNER.stdout") == dict(original_pair_exits=dict(F=0, S=0), protocol=PROTOCOL),
         "Exact closed original pair terminal and owner log")
    request = read(BASE / "PAIR_REQUEST_PRIVATE.json")
    need(request["owner"] == birth and request["protocol"] == PROTOCOL
         and request["seed"] == 5201 and request["arms"] == ["F", "S"]
         and request["planned_microbatches"] == 1488 and request["planned_updates"] == 372
         and request["arm_wall_seconds_cap"] == 1800 and request["raw_and_ema_e24"] is True
         and request["DEV"] is request["TEST"] is request["automatic_retry"] is False,
         "Exact locked pair request")
    p95, p96 = read(P95), read(P96)
    need(p96["commit"] == PROTOCOL and p96["parent"] == p95["commit"] == "55380b6ea1db5f4ef7a35ac356b92792d31e2621"
         and all(row["repository"] == "bingzhi495-creator/SAR-SEAL-research"
                 and row["status"] == "REMOTE_HEAD_PARENT_TREE_AND_ALL_OWN_BLOBS_VERIFIED" for row in (p95, p96))
         and request["publication"] == ref(P96)
         and any(row["path"].endswith("/" + WORKER.name) and row["sha256"] == WORKER_SHA for row in p96["files"])
         and any(row["path"].endswith("/RFDETR_E24_STRICT_BACKWARD_REPLAY_ACTUAL_20261109_SAFE.json")
                 and row["sha256"] == REPLAY_SHA for row in p95["files"]), "Exact verified P96 publication and P95 parent")
    need(request["source"] == ref(WORKER) and request["source"]["sha256"] == WORKER_SHA
         and request["prior_strict_replay_safe"] == ref(REPLAY)
         and request["prior_strict_replay_safe"]["sha256"] == REPLAY_SHA
         and set(request["helpers"]) == set(HELPERS)
         and all(request["helpers"][name] == ref(CANDIDATE / name) and request["helpers"][name]["sha256"] == digest
                 for name, digest in HELPERS.items()), "Exact closed scientific sources and replay proof")
    need(not SAFE.exists() and not CLOSED.exists(), "Fresh strict pair acceptance output")
    reports, training_steps = {}, {}
    for arm, step in (("F", 372), ("S", 4836)):
        control, out = BASE / "CONTROL" / arm, BASE / arm
        need(not live(read(control / "LIVE_PRIVATE.json")), "Original worker must be closed")
        wait = read(control / "TERMINAL_PRIVATE.json")
        need(wait["original_OS_wait"] is True and wait["exit"] == 0 and wait["guard"] is None
             and 0 < wait["peak_worker_RSS_bytes"] <= RSS_CAP and 0 < wait["wall_seconds"]
             and all(wait[k] == ref(control / name) for k, name in
                     (("stdout_ref", "WORKER.stdout"), ("stderr_ref", "WORKER.stderr"))),
             "Same genuine closed original worker " + arm)
        last, report = last_line(control / "WORKER.stdout"), read(out / "RESULT.json")
        meta, result = report["meta"], report["result"]
        need(last["event"] == "E24_OPTIMIZER_STATE_ARM_TERMINAL" and last["arm"] == meta["arm"] == arm
             and last["updates"] == meta["updates"] == meta["ema_updates"] == result["updates"] == result["ema_updates"] == 372
             and last["microbatches"] == meta["microbatches"] == result["microbatches"] == 1488
             and meta["epochs"] == 2 and meta["starting_ema_clock"] == 4464 and meta["ema_absolute_updates"] == 4836
             and [(x["epoch"], x["microbatches"], x["updates"]) for x in result["epoch_stats"]]
                 == [(1, 744, 186), (2, 1488, 372)]
             and meta["adam_step_counts"] == {str(step): 543}
             and meta["protocol_commit"] == PROTOCOL and meta["seed"] == 5201
             and result["visit_sha256"] == meta["visit_sha256"] == meta["planned_visit_sha256"]
             and 0 < result["train_seconds"] == last["train_seconds"] <= wait["wall_seconds"]
             and report["AP"] is report["DEV"] is report["TEST"] is False,
             "Exact actual paired counts and closed worker result " + arm)
        backward = verify_backward_control(result["backward_control"])
        first = report["first_step"]
        need(first["clipped_grad_sha256"] == FIRST_GRADIENT_SHA256
             and first["clipped_grad_tensors"] == 543 and first["first_microbatches"] == 4
             and first["adam_counts_before"] == ({"4464": 543} if arm == "S" else {})
             and first["adam_counts_after"] == ({"4465": 543} if arm == "S" else {"1": 543})
             and backward["strict_seconds"] <= result["train_seconds"], "Exact P94 gradient and first AdamW step " + arm)
        endpoint = out / "ENDPOINT.pt"
        need(endpoint.resolve(strict=True) == endpoint and 0 < endpoint.stat().st_size <= 256*1024**2
             and report["endpoint_ref"] == ref(endpoint), "Exact closed endpoint " + arm)
        reports[arm] = report
        training_steps[arm] = dict(updates=372, microbatches=1488, absolute_ema=4836,
            adam_steps=meta["adam_step_counts"], train_seconds=result["train_seconds"],
            peak_reserved_bytes=result["peak_reserved_bytes"], original_OS_wait=True,
            worker_wall_seconds=wait["wall_seconds"], worker_peak_RSS_bytes=wait["peak_worker_RSS_bytes"],
            backward_control=backward)

    compare = CANDIDATE / "compare_state_pair.py"
    control = BASE / "GRADIENT_READBACK_CONTROL"
    control.mkdir()
    scratch = control / "scratch"
    scratch.mkdir()
    env = windows_runtime(scratch)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    summary_path = control / "COMPARISON_PRIVATE.json"
    with (control / "COMPARATOR.stdout").open("xb") as stdout, (control / "COMPARATOR.stderr").open("xb") as stderr:
        comparison = subprocess.run([str(RUNTIME), "-I", "-B", "-X", "utf8", str(compare), "--pair-dir", str(BASE),
                                     "--output", str(summary_path)], env=env, cwd=control, stdout=stdout, stderr=stderr,
                                    timeout=180, creationflags=subprocess.CREATE_NO_WINDOW)
    write(control / "TERMINAL_PRIVATE.json", dict(exit=comparison.returncode, original_OS_wait=True, CPU=True, TEST=False))
    need(comparison.returncode == 0, "Independent CPU comparator must return an actual zero OS wait")
    summary = read(summary_path)
    need(summary == last_line(control / "COMPARATOR.stdout") and summary["formal_training_accepted"] is False
         and summary["AP"] is summary["DEV"] is summary["TEST"] is False, "Exact closed independent CPU result")
    diff = summary["numerical_first_gradient"]
    agree = summary["same_initial_state_data_lr"] is summary["same_initial_forward_conditions"] is True
    exact = summary["same_clipped_gradient"] is True and diff["changed_elements"] == 0
    within = diff["relative_l2_to_S"] <= 1e-5 and diff["cosine"] >= .999999
    passed = agree and exact and within
    status = ACCEPTED if passed else "PRESERVED_FIRST_GRADIENT_GATE_FAILED"
    safe = dict(status=status, protocol_commit=PROTOCOL, seed=5201, same_initial_input_state_lr=agree,
                first_gradient="EXACT" if exact else "FAIL", gradient_numeric=diff,
                raw_update_l2=summary["raw_update_l2"], mask_head_update_l2=summary["mask_head_update_l2"],
                training_steps=training_steps, original_OS_waits=True, comparator_CPU_OS_wait=True,
                AP=False, DEV=False, TEST=False)
    need(read(STATE, 16*1024**2)[PROFILE] == profile, "Pair authority unchanged during readback")
    write(SAFE, safe)
    if passed:
        write(CLOSED, dict(schema="RFDETR_E24_OPTIMIZER_STATE_PAIR_CLOSED_V1", status=status,
                           protocol_commit=PROTOCOL,
                           arms={arm: dict(closed=True, checkpoint_ref=report["endpoint_ref"]) for arm, report in reports.items()},
                           train_metrics=training_steps))
    print(json.dumps(safe, sort_keys=True, allow_nan=False), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

