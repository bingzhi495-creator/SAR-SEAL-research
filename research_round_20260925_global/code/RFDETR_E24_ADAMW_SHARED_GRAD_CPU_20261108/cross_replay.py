"""Cross two real closed clipped gradients with both initial E24 AdamW states on CPU."""
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import torch


OUT = Path(__file__).resolve().parent
D = OUT.parent
CANDIDATE = D / "RFDETR_E24_OPTIMIZER_PRESERVATION_TRAIN_CANDIDATE_20261108_01"
ORIGINAL = D / "RFDETR_E24_OPTIMIZER_PRESERVATION_ACTUAL_PRIVATE_20261108_01"
PROTOCOL = "fb2baedab2a5b1a9970f968ec0b98f6584744cf1"
SAFE_SHA = "be9067a2b0862a93f658325bcdd5f4d2a71c757221bb2c05ae8a45270b864cec"
SOURCE_SHA = {
    "run_state_arm.py": "f870227ecce5da591c1bdc9b975591fff720b36540e0f01818277105d3c2176c",
    "state_preservation.py": "8dda41131a2930aacf00c51dbe8cfa4817c0a3cd4b02c6bb606948db77341678",
}
ARMS = ("F", "S")
MATCHED_INITIAL_KEYS = (
    "archive_start_sha256", "starting_ema_clock", "initial_online_sha256", "initial_ema_sha256",
    "parameter_group_order_sha256", "applied_group_lrs", "archived_unused_slot",
)


def need(ok, why):
    if not ok:
        raise ValueError(why)


def compare_updates(left, right):
    """Compute norm ratios, distances, and cosine on full and mask-head update vectors."""
    need(list(left) == list(right), "Exact update parameter identities and order")
    moments = {"all": [], "mask_head": []}
    for name, a in left.items():
        b = right[name]
        need(a.shape == b.shape and a.dtype == b.dtype and a.is_floating_point(),
             "Exact finite update vector geometry")
        a, b = a.double(), b.double()
        need(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "Finite updates required")
        row = (float(a.square().sum()), float(b.square().sum()),
               float((a-b).square().sum()), float((a*b).sum()))
        moments["all"].append(row)
        if name.startswith("segmentation_head."):
            moments["mask_head"].append(row)
    result = {}
    for group, rows in moments.items():
        a2, b2, diff2, dot = (math.fsum(row[i] for row in rows) for i in range(4))
        left_l2, right_l2, diff_l2 = (math.sqrt(x) for x in (a2, b2, diff2))
        need(left_l2 > 0 and right_l2 > 0, "Expected nonzero paired updates")
        result[group] = dict(left_l2=left_l2, right_l2=right_l2, difference_l2=diff_l2,
            right_to_left_l2=right_l2/left_l2, relative_difference_to_left=diff_l2/left_l2,
            cosine=max(-1.0, min(1.0, dot/(left_l2*right_l2))), tensors=len(rows))
    return result


def json_file(path):
    need(path.resolve(strict=True) == path and 0 < path.stat().st_size <= 1024**2, "Bounded closed JSON")
    return json.loads(path.read_bytes())


def exact_bytes(path, ref, maximum, keep=False):
    need(path.resolve(strict=True) == path == Path(ref["path"]), "Exact closed input path")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        need(0 <= before.st_size == ref["bytes"] <= maximum, "Exact bounded original size")
        h = sha256()
        blocks = []
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
            if keep:
                blocks.append(block)
        after = os.fstat(stream.fileno())
    need(h.hexdigest() == ref["sha256"] and (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
         "Closed original byte identity")
    return b"".join(blocks)


def read_gradients(source):
    terminal = json_file(ORIGINAL/"PAIR_ORIGINAL_TERMINAL_PRIVATE.json")
    need(terminal["exits"] == dict(F=0, S=0) and terminal["protocol"] == PROTOCOL and
         terminal["original_OS_waits"] is True and all(terminal[k] is False for k in ("DEV", "TEST", "automatic_retry")),
         "Exact closed original pair with both original OS waits")
    safe_path = ORIGINAL/"ROOT_SAFE_TRAINING_RESULT.json"
    need(sha256(safe_path.read_bytes()).hexdigest() == SAFE_SHA, "Exact independently compared original")
    safe = json_file(safe_path)
    need(safe["status"] == "PRESERVED_FIRST_GRADIENT_GATE_FAILED" and
         safe["same_initial_input_state_lr"] is True and
         all(safe[k] is False for k in ("DEV", "TEST", "AP")), "Original gate failure must remain preserved")

    gradients, reports, evidence = {}, {}, {}
    for arm in ARMS:
        control = ORIGINAL/"CONTROL"/arm
        receipt = json_file(control/"TERMINAL_PRIVATE.json")
        need(receipt["exit"] == 0 and receipt["original_OS_wait"] is True and receipt["guard"] is None,
             "Genuine original worker exit zero")
        stdout = exact_bytes(control/"WORKER.stdout", receipt["stdout_ref"], 1024**2, keep=True)
        exact_bytes(control/"WORKER.stderr", receipt["stderr_ref"], 1024**2)
        last = json.loads(stdout.splitlines()[-1])
        report = json_file(ORIGINAL/arm/"RESULT.json")
        first, meta = report["first_step"], report["meta"]
        need(last["event"] == "E24_OPTIMIZER_STATE_ARM_TERMINAL" and last["arm"] == meta["arm"] == first["arm"] == arm
             and last["updates"] == meta["updates"] == 372 and last["microbatches"] == meta["microbatches"] == 1488
             and meta["protocol_commit"] == PROTOCOL and first["first_microbatches"] == 4
             and first["applied_group_lrs"] == meta["applied_group_lrs"]
             and all(report[k] is False for k in ("DEV", "TEST", "AP")), "Original report and last event agree")
        path = ORIGINAL/arm/"FIRST_CLIPPED_GRAD_PRIVATE.pt"
        ref = first["gradient_ref"]
        exact_bytes(path, ref, 256*1024**2)
        with path.open("rb") as stream:
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        grad = payload["tensors"]
        need(set(payload) == {"schema", "tensors"} and payload["schema"] == "RFDETR_FIRST_CLIPPED_GRAD_V1"
             and len(grad) == first["clipped_grad_tensors"] == 543
             and source.tensor_dict_sha256(grad) == first["clipped_grad_sha256"], "Exact closed real clipped tensors")
        norm = math.sqrt(math.fsum(float(g.double().square().sum()) for g in grad.values()))
        need(math.isclose(norm, first["clipped_grad_l2"], rel_tol=1e-12), "Original first clipped gradient norm")
        gradients[arm], reports[arm] = grad, report
        evidence[arm] = dict(archive_sha256=ref["sha256"], tensor_sha256=first["clipped_grad_sha256"],
                             tensor_count=len(grad), clipped_l2=norm, original_OS_exit=0)
    need(list(gradients["F"]) == list(gradients["S"]) and
         reports["F"]["first_step"]["first_input_sha256"] == reports["S"]["first_step"]["first_input_sha256"] and
         all(reports["F"]["meta"][k] == reports["S"]["meta"][k] for k in MATCHED_INITIAL_KEYS),
         "Exact common original input, model state and LR")
    return gradients, reports, evidence


def one_update(source, native_source, state, gradient, expected_initial):
    started = time.monotonic()
    # Match the original setup seed even though both archived model states are loaded exactly.
    torch.random.default_generator.manual_seed(5201)
    _, model, criterion, opt, ema, _, init = source.initialize(state, torch.device("cpu"), native_source)
    del criterion
    need(all(init[k] == expected_initial[k] for k in MATCHED_INITIAL_KEYS)
         and init["initial_state_slots"] == (0 if state == "F" else 543)
         and init["initial_adam_step"] == (0 if state == "F" else 4464)
         and type(opt) is torch.optim.AdamW, "Exact archived start and official optimizer")
    need(source.tensor_dict_sha256(model.state_dict()) == init["initial_online_sha256"],
         "Exact online archive after native initialization")
    ema_hash_before = source.tensor_dict_sha256(ema.module.state_dict())
    need(ema_hash_before == init["initial_ema_sha256"] and int(ema.n_averaged.item()) == 4464,
         "Exact archived EMA held unchanged")
    named = dict(model.named_parameters())
    index = {}
    for live_group, saved_group in zip(opt.param_groups, opt.state_dict()["param_groups"], strict=True):
        for parameter, slot in zip(live_group["params"], saved_group["params"], strict=True):
            index[id(parameter)] = slot
    unused = {index[id(p)] for name, p in named.items() if name not in gradient}
    need(unused == {init["archived_unused_slot"]} and len(index) == 544, "Exact shared unused AdamW slot")
    before = {name: p.detach().clone() for name, p in named.items()}
    for name, values in gradient.items():
        p = named[name]
        need(p.device.type == values.device.type == "cpu" and p.shape == values.shape and p.dtype == values.dtype
             and bool(torch.isfinite(values).all()), "Native parameter and supplied gradient agree")
        p.grad = values.clone()
    step_start = time.monotonic()
    opt.step()
    update_wall = time.monotonic()-step_start
    step = 1 if state == "F" else 4465
    counters = Counter(int(v["step"].item()) for v in opt.state.values())
    need(counters == {step: 543} and int(ema.n_averaged.item()) == 4464
         and source.tensor_dict_sha256(ema.module.state_dict()) == ema_hash_before
         and [g["lr"] for g in opt.param_groups] == init["applied_group_lrs"],
         "Exactly one AdamW step and zero EMA changes")
    del opt, ema
    update = {name: p.detach().double()-before[name].double() for name, p in named.items()}
    need(all(not bool(torch.count_nonzero(update[k])) for k in set(named)-set(gradient)),
         "Unused parameter must remain unchanged")
    norms = compare_updates(update, update)
    return update, dict(update_l2=norms["all"]["left_l2"], mask_head_update_l2=norms["mask_head"]["left_l2"],
        adam_state_step_counts={str(k): v for k, v in counters.items()}, setup_and_step_seconds=time.monotonic()-started,
        adam_step_seconds=update_wall)


def main():
    need(sys.platform == "win32" and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and not torch.cuda.is_available() and not torch.cuda.is_initialized(), "CPU-only exact Windows runtime")
    need(os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1", "Pinned offline route")
    for name, digest in SOURCE_SHA.items():
        p = CANDIDATE/name
        need(p.resolve(strict=True) == p and sha256(p.read_bytes()).hexdigest() == digest, "Exact closed initialization source")
    spec = importlib.util.spec_from_file_location("_closed_native_optim_pair_source", CANDIDATE/"run_state_arm.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    need(Path(sys.executable).resolve(strict=True) == source.RUNTIME.resolve(strict=True), "Pinned interpreter")
    sys.addaudithook(source.offline)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    gradients, reports, evidence = read_gradients(source)
    native = source.p77_source()
    started = time.monotonic()
    updates, cells = {}, {}
    for state, witness in ((a,b) for a in ARMS for b in ARMS):
        cell = state+witness
        update, record = one_update(source, native, state, gradients[witness], reports[state]["meta"])
        updates[cell] = update
        cells[cell] = dict(optimizer_state=state, gradient_witness=witness, **record)
        print(json.dumps(dict(cell=cell, **cells[cell]), sort_keys=True), flush=True)

    fixed = {g: compare_updates(updates["F"+g],updates["S"+g]) for g in ARMS}
    sensitivity = {a: compare_updates(updates[a+"F"],updates[a+"S"]) for a in ARMS}
    result = dict(status="ACTUAL_CPU_ADAMW_SHARED_REAL_GRADIENT_CROSS_REPLAY_COMPLETE",
        protocol=PROTOCOL, backend="NATIVE_TORCH_ADAMW_CPU", cells=cells,
        shared_gradient_state_effect=fixed, within_state_gradient_sensitivity=sensitivity,
        gradients=evidence, original_pair_os_waits_verified=True, original_AP_qualification_passed=False,
        one_cpu_optimizer_update_per_cell=True, optimizer_updates_total=4, ema_updates=0,
        FIT_data_opened=False, model_forward=False, DEV=False, TEST=False, AP=False, GPU=False,
        elapsed_seconds=time.monotonic()-started, initialization_source_sha256=SOURCE_SHA)
    path = OUT/"SAFE_SUMMARY.json"
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(dict(status=result["status"],summary_sha256=sha256(path.read_bytes()).hexdigest(),
        elapsed_seconds=result["elapsed_seconds"]),sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
