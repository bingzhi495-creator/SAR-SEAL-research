"""Independently compare first gradients from four genuinely closed replay subprocesses."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
from itertools import combinations
import json
import math
import os
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
D = HERE.parent
sys.path.insert(0, str(HERE))
from first_gradient_replay import load_pair, need

ATTEMPTS = ("F0", "S0", "F1", "S1")
ACTUAL = D / "RFDETR_E24_FIRST_GRAD_REPLAY_ACTUAL_PRIVATE_20261108_01"
ORIGINAL = D / "RFDETR_E24_OPTIMIZER_PRESERVATION_ACTUAL_PRIVATE_20261108_01"
MIB = 1024**2
STAGES = (
    "rng_before_fetch", "input_sha256", "rng_after_fetch", "patch_projection_output_sha256",
    "backbone_output_sha256", "forward_outputs", "rng_after_forward", "matcher", "point_coords",
    "losses", "rng_after_loss", "backbone_first_feature_grad_sha256",
    "patch_projection_grad_out_sha256", "cumulative_gradient",
)


def gradient_comparison(left, right):
    need(list(left) == list(right), "Exact gradient parameter identities and order")
    groups = {k: [] for k in ("all", "backbone", "non_backbone", "patch_projection_weight")}
    for name, a in left.items():
        b = right[name]
        need(a.shape == b.shape and a.dtype == b.dtype and a.is_floating_point()
             and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
             "Finite matching gradient tensors")
        a = a.double()
        b = b.double()
        delta = a - b
        row = (float(a.square().sum()), float(b.square().sum()), float(delta.square().sum()),
               float((a*b).sum()), float(delta.abs().max()) if delta.numel() else 0.,
               int(torch.count_nonzero(delta)), a.numel())
        groups["all"].append(row)
        groups["backbone" if name.startswith("backbone.") else "non_backbone"].append(row)
        if name.endswith("patch_embeddings.projection.weight"):
            groups["patch_projection_weight"].append(row)
    total_diff = math.fsum(r[2] for r in groups["all"])
    output = {}
    for key, rows in groups.items():
        norm_a = math.sqrt(math.fsum(r[0] for r in rows))
        norm_b = math.sqrt(math.fsum(r[1] for r in rows))
        sq = math.fsum(r[2] for r in rows)
        diff = math.sqrt(sq)
        output[key] = dict(tensors=len(rows), elements=sum(r[6] for r in rows),
            changed_elements=sum(r[5] for r in rows), max_abs=max((r[4] for r in rows), default=0.),
            left_l2=norm_a, right_l2=norm_b, diff_l2=diff, rel_l2_to_left=diff/norm_a if norm_a else None,
            cosine=math.fsum(r[3] for r in rows)/(norm_a*norm_b) if norm_a and norm_b else None,
            squared_diff_share=sq/total_diff if total_diff else None)
    return output


def read_json(path, limit=MIB):
    need(path.resolve(strict=True) == path and path.is_file() and 0 < path.stat().st_size <= limit,
         "Exact bounded closed JSON")
    return json.loads(path.read_bytes())


def read_ref(ref, path, maximum, read=False):
    need(Path(ref["path"]) == path and path.resolve(strict=True) == path and path.is_file(),
         "Exact closed artifact path")
    with path.open("rb") as f:
        a = os.fstat(f.fileno())
        need(0 <= a.st_size == ref["bytes"] <= maximum, "Exact bounded closed artifact size")
        h = sha256()
        data = bytearray() if read else None
        for block in iter(lambda: f.read(MIB), b""):
            h.update(block)
            if data is not None:
                data.extend(block)
        b = os.fstat(f.fileno())
        need((a.st_size, a.st_mtime_ns) == (b.st_size, b.st_mtime_ns) and h.hexdigest() == ref["sha256"],
             "Closed artifact checksum and stable bytes")
    return data


def worker_stdout(base, name):
    control = base / "CONTROL" / name
    receipt = read_json(control / "TERMINAL_PRIVATE.json")
    need(receipt["exit"] == 0 and receipt["original_OS_wait"] is True and receipt["guard"] is None,
         "Genuine closed original worker OS wait")
    raw = read_ref(receipt["stdout_ref"], control / "WORKER.stdout", MIB, read=True)
    read_ref(receipt["stderr_ref"], control / "WORKER.stderr", MIB)
    return json.loads(raw.splitlines()[-1])


def gradient(source, base, name, ref, digest, expected_tensors=None):
    path = base / name / "FIRST_CLIPPED_GRAD_PRIVATE.pt"
    read_ref(ref, path, 256*MIB)
    with path.open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    values = payload["tensors"]
    need(set(payload) == {"schema", "tensors"} and payload["schema"] == "RFDETR_FIRST_CLIPPED_GRAD_V1"
         and values and (expected_tensors is None or len(values) == expected_tensors)
         and source.tensor_dict_sha256(values) == digest,
         "Exact closed first gradient tensor identity")
    return values


def replay_reports(source, base):
    terminal = read_json(base / "ORIGINAL_TERMINAL_PRIVATE.json")
    need(terminal["exits"] == dict.fromkeys(ATTEMPTS, 0) and terminal["original_OS_waits"] is True
         and all(terminal[k] is False for k in ("automatic_retry", "training", "DEV", "TEST")),
         "Exact four genuine replay originals closed")
    logs = {name: worker_stdout(base, name) for name in ATTEMPTS}
    reports = {}
    grads = {}
    for name in ATTEMPTS:
        doc = read_json(base / name / "RESULT.json")
        log = logs[name]
        need(doc["status"] == log["status"] == "RFDETR_FIRST_GRADIENT_REPLAY_COMPLETE_NO_UPDATE"
             and doc["attempt"] == log["attempt"] == name and doc["arm"] == name[0] and doc["seed"] == 5201
             and doc["protocol_commit"] == terminal["protocol"] and len(doc["microbatches"]) == log["microbatches"] == 4
             and doc["optimizer_updates"] == doc["ema_updates"] == log["optimizer_updates"] == log["ema_updates"] == 0
             and doc["clip"]["preclip_global_norm"] == log["preclip_global_norm"]
             and all(doc[k] is False for k in ("AP", "DEV", "TEST")), "Complete matching closed replay outputs")
        reports[name] = doc
        grads[name] = gradient(source, base, name, doc["gradient_ref"], doc["clipped_gradient_sha256"])
    init_keys = ("archive_start_sha256", "starting_ema_clock", "initial_online_sha256",
                 "initial_ema_sha256", "parameter_group_order_sha256", "applied_group_lrs")
    baseline = reports["F0"]
    for name, doc in reports.items():
        need(doc["tape"] == baseline["tape"] and all(doc["init"][k] == baseline["init"][k] for k in init_keys)
             and all(doc[k] == baseline[k] for k in ("original_training_mode", "float32_matmul_precision",
                                                    "cudnn_allow_tf32", "deterministic_algorithms"))
             and doc["init"]["initial_state_slots"] == (543 if name[0] == "S" else 0)
             and doc["init"]["initial_adam_step"] == (4464 if name[0] == "S" else 0),
             "Equivalent fixed replay state and runtime apart from AdamW history")
    return terminal["protocol"], reports, grads


def compare_stages(left, right):
    rows = []
    for index, (a, b) in enumerate(zip(left["microbatches"], right["microbatches"], strict=True)):
        need(a["index"] == b["index"] and a["slot"] == b["slot"], "Same recorded replay microbatch position")
        different = [key for key in STAGES if a.get(key) != b.get(key)]
        rows.append(dict(microbatch=index+1, differing_stages=different, first_difference=different[0] if different else None))
    return rows


def original_gradients(source, base):
    terminal = read_json(base / "PAIR_ORIGINAL_TERMINAL_PRIVATE.json")
    need(terminal["exits"] == {"F": 0, "S": 0} and terminal["original_OS_waits"] is True,
         "Original F/S pair genuinely closed")
    logs = {name: worker_stdout(base, name) for name in ("F", "S")}
    result = {}
    for name in ("F", "S"):
        doc = read_json(base / name / "RESULT.json")
        log = logs[name]
        need(log["event"] == "E24_OPTIMIZER_STATE_ARM_TERMINAL" and log["arm"] == doc["meta"]["arm"] == name
             and log["updates"] == doc["meta"]["updates"] == 372 and log["microbatches"] == doc["meta"]["microbatches"] == 1488,
             "Original first gradients belong to complete endpoints")
        first = doc["first_step"]
        result[name] = gradient(source, base, name, first["gradient_ref"], first["clipped_grad_sha256"], first["clipped_grad_tensors"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--original-dir", type=Path)
    args = parser.parse_args()
    need(args.actual_dir.resolve(strict=True) == args.actual_dir == ACTUAL
         and args.output.resolve() == args.output and D in args.output.parents
         and args.output.parent.is_dir() and not args.output.exists()
         and (args.original_dir is None or args.original_dir.resolve(strict=True) == args.original_dir == ORIGINAL),
         "Exact private closed roots and fresh private result")
    need(not torch.cuda.is_available(), "CPU-only independent comparison")
    torch.set_num_threads(2)
    source = load_pair()
    sys.addaudithook(source.offline)
    protocol, reports, grads = replay_reports(source, args.actual_dir)
    pairs = {}
    for left, right in combinations(ATTEMPTS, 2):
        comparisons = compare_stages(reports[left], reports[right])
        need(all("input_sha256" not in row["differing_stages"] for row in comparisons), "Realized replay inputs differ")
        pairs[left+"_"+right] = dict(kind="within_arm" if left[0] == right[0] else "cross_arm",
            stages=comparisons, gradient=gradient_comparison(grads[left], grads[right]))
    origin = None
    if args.original_dir is not None:
        before = original_gradients(source, args.original_dir)
        origin = {f"original_{name[0]}_vs_{name}": gradient_comparison(before[name[0]], grads[name])
                  for name in ATTEMPTS}
        origin["original_F_vs_S"] = gradient_comparison(before["F"], before["S"])
    output = dict(status="ACTUAL_FOUR_FIRST_GRADIENT_REPLAYS_CPU_COMPARED", protocol_commit=protocol,
        attempts=list(ATTEMPTS), original_OS_waits_verified=True, optimizer_updates=0, ema_updates=0,
        preclip_global_norm={n: r["clip"]["preclip_global_norm"] for n,r in reports.items()},
        clip_scale_from_norm={n: min(1.0, .1/(r["clip"]["preclip_global_norm"]+1e-6)) for n,r in reports.items()},
        numeric_precision=dict(float32_matmul=reports["F0"]["float32_matmul_precision"],
                               cudnn_allow_tf32=reports["F0"]["cudnn_allow_tf32"],
                               deterministic_algorithms=reports["F0"]["deterministic_algorithms"]),
        all_fixed_pairs=pairs, original_comparisons=origin, CPU=True, GPU=False, AP=False, TEST=False, DEV=False)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=output["status"], fixed_pairs=len(pairs), original_comparisons=origin is not None,
        relative={key:r["gradient"]["all"]["rel_l2_to_left"] for key,r in pairs.items()})), flush=True)


if __name__ == "__main__":
    main()
