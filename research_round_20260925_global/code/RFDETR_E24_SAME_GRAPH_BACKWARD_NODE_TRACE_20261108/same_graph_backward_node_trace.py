"""Locate observed local differences over two backward traversals of one retained FIT graph."""
import argparse
from collections import Counter, defaultdict, deque
from contextlib import ExitStack
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from types import ModuleType

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
D = HERE.parent
REPLAY = D / "RFDETR_E24_FIRST_GRAD_REPLAY_CANDIDATE_20261108_01/first_gradient_replay.py"
REPLAY_SHA = "b35340218f6b557a915417a352ce6ca5270a368375a94e2f6b0b3178d2545314"
PROTOCOL = "a0a828a1cc372a4ad9a1885023509b82a0320a27"
MIB = 1024**2
MAX_GRAPH_NODES = 60000
MAX_HOOKS = 128
MAX_TUPLE_BYTES = 128 * MIB
MAX_PASS_BYTES = 3072 * MIB

OPERATOR_FAMILIES = {
    "conv": ("ConvolutionBackward", "CudnnConvolution", "MkldnnConvolution"),
    "attention": ("ScaledDotProduct", "FlashAttention", "EfficientAttention", "SoftmaxBackward"),
    "sampling": ("GridSampler", "UpsampleBilinear", "UpsampleNearest", "InterpolateBackward"),
    "index": ("IndexBackward", "IndexPutBackward", "IndexSelectBackward", "GatherBackward", "Scatter"),
    "projector": ("NativeLayerNormBackward", "NativeGroupNormBackward", "AddmmBackward", "BmmBackward"),
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def selected_nodes(root):
    queue = deque([(root, 0)])
    scheduled = {root}
    available = Counter()
    buckets = defaultdict(list)
    ordinal = 0
    while queue:
        node, depth = queue.popleft()
        operator = node.name()
        for family, names in OPERATOR_FAMILIES.items():
            if any(label in operator for label in names):
                available[family] += 1
                if len(buckets[family]) < MAX_HOOKS:
                    buckets[family].append((ordinal, depth, family, operator, node))
                break
        for parent, _ in node.next_functions:
            if parent is not None and parent not in scheduled:
                need(len(scheduled) < MAX_GRAPH_NODES, "Fixed retained graph node ceiling")
                scheduled.add(parent)
                queue.append((parent, depth + 1))
        ordinal += 1
    chosen = []
    for offset in range(MAX_HOOKS):
        for family in OPERATOR_FAMILIES:
            if offset < len(buckets[family]):
                chosen.append(buckets[family][offset])
                if len(chosen) == MAX_HOOKS:
                    return sorted(chosen), dict(available), ordinal
    return sorted(chosen), dict(available), ordinal


def gradient_digest(values, budget):
    present = [x for x in values if x is not None]
    if any(x.layout != torch.strided for x in present):
        return None
    size = sum(x.numel() * x.element_size() for x in present)
    if size > MAX_TUPLE_BYTES or budget[0] + size > MAX_PASS_BYTES:
        return None
    budget[0] += size
    digest = sha256()
    for value in values:
        if value is None:
            digest.update(b"null\n")
        else:
            digest.update(json.dumps([list(value.shape), str(value.dtype)]).encode() + b"\n")
            cpu = value.detach().cpu().contiguous().reshape(-1)
            digest.update(memoryview(cpu.view(torch.uint8).numpy()))
    return digest.hexdigest()


def leaf_comparison(first, second):
    need(list(first) == list(second) and bool(first), "Same nonempty leaf gradient identities")
    groups = defaultdict(list)
    for name in first:
        a, b = first[name], second[name]
        need(a.shape == b.shape and a.dtype == b.dtype, "Same leaf gradient shape and type")
        need(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), "Finite repeated leaf gradients")
        delta = a.double() - b.double()
        row = (float(delta.square().sum()), float(a.double().square().sum()), int(torch.count_nonzero(delta)))
        groups["all"].append(row)
        if name.startswith("backbone."):
            groups["backbone"].append(row)
    result = {}
    for label, rows in groups.items():
        diff = math.sqrt(math.fsum(r[0] for r in rows))
        norm = math.sqrt(math.fsum(r[1] for r in rows))
        result[label] = dict(tensors=len(rows), relative_l2_to_first=diff/norm if norm else None,
                             difference_l2=diff, changed_elements=sum(r[2] for r in rows))
    return result


def trace_backwards(loss, leaves, device, anchors=None):
    """Run two backward passes with identical RNG and passive node tuple witnesses."""
    need(loss.grad_fn is not None and loss.ndim == 0 and bool(torch.isfinite(loss)), "Finite scalar retained loss")
    selected, available, graph_count = selected_nodes(loss.grad_fn)
    need(bool(selected), "At least one supported graph node")
    calls = {}
    anchor_calls = {}
    budget = [0]
    snapshots = []
    data = []
    timings = []
    rng_before = torch.random.get_rng_state()
    cuda_rng_before = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    rng_after = []

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def capture(ordinal, grad_inputs, grad_outputs):
        # In backward, grad_outputs is received and grad_inputs is emitted.
        need(ordinal not in calls, "Each selected node should execute once per backward")
        calls[ordinal] = (gradient_digest(grad_outputs, budget), gradient_digest(grad_inputs, budget))

    def capture_anchor(name, grad):
        need(name not in anchor_calls, "Each requested boundary tensor should execute once per backward")
        anchor_calls[name] = gradient_digest((grad,), budget)

    with ExitStack() as stack:
        for ordinal, _, _, _, node in selected:
            stack.callback(node.register_hook(
                lambda grad_inputs, grad_outputs, key=ordinal: capture(key, grad_inputs, grad_outputs)
            ).remove)
        for name, tensor in (anchors or {}).items():
            need(tensor.requires_grad, "The observed boundary must propagate a gradient")
            stack.callback(tensor.register_hook(lambda grad, key=name: capture_anchor(key, grad)).remove)
        for _ in range(2):
            calls.clear()
            anchor_calls.clear()
            budget[0] = 0
            for tensor in leaves.values():
                tensor.grad = None
            torch.random.set_rng_state(rng_before)
            if cuda_rng_before is not None:
                torch.cuda.set_rng_state(cuda_rng_before, device)
            sync()
            started = time.perf_counter()
            loss.backward(retain_graph=True)
            sync()
            timings.append(time.perf_counter() - started)
            rng_after.append((
                bytes(torch.random.get_rng_state().numpy()),
                bytes(torch.cuda.get_rng_state(device).cpu().numpy()) if cuda_rng_before is not None else None,
            ))
            data.append((dict(calls), dict(anchor_calls), budget[0]))
            snapshots.append({name: value.grad.detach().to("cpu", copy=True)
                              for name, value in leaves.items() if value.grad is not None})
    by_node = []
    for ordinal, depth, family, operator, _ in selected:
        first = data[0][0].get(ordinal)
        second = data[1][0].get(ordinal)
        covered = first is not None and second is not None and None not in (*first, *second)
        same_in = first[0] == second[0] if covered else None
        same_out = first[1] == second[1] if covered else None
        by_node.append(dict(ordinal=ordinal, graph_depth=depth, family=family, operator=operator,
                            compared=covered, exact_backward_inputs=same_in, exact_backward_outputs=same_out))
    leaf = leaf_comparison(*snapshots)
    boundaries = {name: a.get(name) == b.get(name) if a.get(name) is not None and b.get(name) is not None else None
                  for name in (anchors or {}) for a, b in [(data[0][1], data[1][1])]}
    candidates = [row for row in by_node if row["exact_backward_inputs"] is True
                  and row["exact_backward_outputs"] is False]
    return dict(graph_node_count=graph_count, selected_node_count=len(selected), operator_available=available,
                selected_by_family=dict(Counter(r["family"] for r in by_node)),
                compared_node_count=sum(r["compared"] for r in by_node),
                local_difference_count=len(candidates), candidate_local_differences=candidates,
                node_comparisons=by_node, exact_boundary_gradients=boundaries,
                leaf_gradient_exact=leaf["all"]["changed_elements"] == 0, leaf_gradients=leaf,
                backward_rng_states_equal=rng_after[0] == rng_after[1],
                observer_hashed_bytes=[data[0][2], data[1][2]], backward_wall_seconds_with_hooks=timings,
                selection_order="BFS from loss, ordered next_functions, family-round-robin cap",
                limits=dict(graph_nodes=MAX_GRAPH_NODES, hooks=MAX_HOOKS,
                            tuple_hash_bytes=MAX_TUPLE_BYTES, pass_hash_bytes=MAX_PASS_BYTES))


class TraceComplete(Exception):
    """Stop the reused exact replay entry before it can write a normal window result."""


def load_replay():
    b = REPLAY.read_bytes()
    need(sha256(b).hexdigest() == REPLAY_SHA, "Exact qualified native replay source")
    replay = ModuleType("_pinned_same_graph_replay")
    replay.__file__ = str(REPLAY)
    sys.modules[replay.__name__] = replay
    exec(compile(b, str(REPLAY), "exec"), replay.__dict__)
    return replay


def run(args):
    replay = load_replay()
    result = {}

    def observe(model, criterion, optimizer, native_api, device, original, ema, fetch, tape):
        need(len(tape) == 4 and model.training and not optimizer.state and int(ema.n_averaged.item()) == 0,
             "Exact fixed F replay prefix and clocks")
        before = replay.tree_hash(dict(model.named_parameters()))
        ema_before = replay.tree_hash(ema.average.state_dict())
        slot, micro_seed = tape[0]
        random.seed(micro_seed)
        np.random.seed(micro_seed)
        torch.manual_seed(micro_seed)
        samples, targets, _ = fetch(slot)
        anchors = {}

        def backbone(_module, _args, output):
            anchors["backbone_first_feature"] = output[0][0].tensors

        handle = model.backbone.register_forward_hook(backbone)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(samples, targets)
                losses = criterion(outputs, targets)
                loss = torch.stack([v*criterion.weight_dict[k] for k, v in losses.items()
                                    if k in criterion.weight_dict]).sum()
        finally:
            handle.remove()
        need(set(anchors) == {"backbone_first_feature"}, "Expected fixed backbone boundary")
        trace = trace_backwards(loss/4, dict(model.named_parameters()), device, anchors)
        need(replay.tree_hash(dict(model.named_parameters())) == before and not optimizer.state
             and int(ema.n_averaged.item()) == 0 and replay.tree_hash(ema.average.state_dict()) == ema_before,
             "No optimizer, EMA or parameter mutation")
        result.update(status="ACTUAL_ONE_FIT_FORWARD_TWO_BACKWARDS_NO_UPDATE", protocol_commit=args.protocol_commit,
                      attempt=args.attempt, FIT_microbatches=1, forward_passes=1, backward_passes=2,
                      optimizer_updates=0, ema_updates=0, AP=False, DEV=False, TEST=False,
                      native_forward_sha256=replay.tree_hash(outputs), fixed_input_sha256=replay.tree_hash((samples, targets)),
                      forward_output_sha256_by_name={k: replay.tree_hash(v) for k, v in sorted(outputs.items())},
                      replay_source_sha256=REPLAY_SHA, peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      precision="native BF16 autocast, high float32 matmul, cuDNN TF32", **trace)
        raise TraceComplete

    replay.trace_first_window = observe
    try:
        replay.run(args)
    except TraceComplete:
        pass
    else:
        raise ValueError("Single-graph trace did not stop the native replay entry")
    out = args.output_dir / "RESULT.json"
    with out.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(dict(status=result["status"], optimizer_updates=0, ema_updates=0,
                          selected_nodes=result["selected_node_count"], compared_nodes=result["compared_node_count"],
                          local_difference_count=result["local_difference_count"],
                          leaf_gradient_exact=result["leaf_gradient_exact"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt", choices=("F0",), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--protocol-commit", required=True)
    args = p.parse_args()
    need(args.protocol_commit == PROTOCOL, "Exact verified P90 protocol")
    run(args)
