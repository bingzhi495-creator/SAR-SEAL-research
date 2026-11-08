"""Reuse the one-graph trace with two CPU sampling backward routes and DINO-focused observers."""
import argparse
from collections import Counter, defaultdict, deque
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
D = HERE.parent
NODE = D / "RFDETR_E24_SAME_GRAPH_BACKWARD_NODE_TRACE_CANDIDATE_20261108_01/same_graph_backward_node_trace.py"
NODE_SHA = "a7cdbff31a38da45640eb46392d29eb6d173a9bb763b64a81aca14124acc85e5"
COMBINED = D / "RFDETR_E24_COMBINED_SAMPLING_BACKWARD_PROBE_CANDIDATE_20261108_01/combined_sampling_backward_probe.py"
COMBINED_SHA = "b90dcaf0cc408b078c2fcad2adb2c646f0e2632fe034f6a4cf50792cff8822e2"
PROTOCOL = "1d81e039340b8b3333cc7bd0d18d35faafc0eb3f"
REGIONS = {"DINO_ENCODER": 96, "BACKBONE_NON_DINO": 24, "OUTSIDE_BACKBONE": 8}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load_pinned(path, digest, name):
    data = path.read_bytes()
    need(sha256(data).hexdigest() == digest, "Exact previously qualified source")
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def collect_tensors(value, key):
    """Preserve a deterministic path to every tensor at a native nested boundary."""
    if torch.is_tensor(value):
        return {key: value}
    if value is None:
        return {}
    if hasattr(value, "tensors") and hasattr(value, "mask"):
        return collect_tensors(value.tensors, key + "_tensors") | collect_tensors(value.mask, key + "_mask")
    need(isinstance(value, (list, tuple)), "Unsupported native backbone output at " + key)
    captured = {}
    for i, child in enumerate(value):
        captured.update(collect_tensors(child, f"{key}_{i}"))
    return captured


def graph_nodes(roots, ceiling):
    queue = deque((node, 0) for node in roots)
    seen = set(roots)
    need(len(seen) == len(queue), "Unique graph roots required")
    while queue:
        node, depth = queue.popleft()
        yield node, depth
        for parent, _ in node.next_functions:
            if parent is not None and parent not in seen:
                need(len(seen) < ceiling, "Fixed graph node ceiling")
                seen.add(parent)
                queue.append((parent, depth + 1))


def reachable(tensors, ceiling):
    roots = list(dict.fromkeys(x.grad_fn for x in tensors if x.grad_fn is not None))
    return {node for node, _ in graph_nodes(roots, ceiling)}


def focused_nodes(root, dino, joined, base=None):
    """Select bounded graph observations prioritized by the actual backbone cut."""
    if base is None:
        base = load_pinned(NODE, NODE_SHA, "_combined_probe_cpu_node_helpers")
    ceiling = base.MAX_GRAPH_NODES
    dino_nodes = reachable(dino, ceiling)
    backbone_nodes = reachable(joined, ceiling)
    need(bool(dino_nodes) and dino_nodes <= backbone_nodes, "DINO output ancestry must be inside the joined backbone")
    buckets = defaultdict(lambda: defaultdict(list))
    available = Counter()
    regions = {}
    total = 0
    for ordinal, (node, depth) in enumerate(graph_nodes([root], ceiling)):
        total += 1
        operator = node.name()
        for family, names in base.OPERATOR_FAMILIES.items():
            if not any(label in operator for label in names):
                continue
            region = ("DINO_ENCODER" if node in dino_nodes else
                      "BACKBONE_NON_DINO" if node in backbone_nodes else "OUTSIDE_BACKBONE")
            available[family] += 1
            regions[ordinal] = region
            buckets[region][family].append((ordinal, depth, family, operator, node))
            break
    chosen = []
    unused = []
    for region, limit in REGIONS.items():
        rows = []
        family_rows = buckets[region]
        for offset in range(max(map(len, family_rows.values()), default=0)):
            for family in base.OPERATOR_FAMILIES:
                sequence = family_rows.get(family, ())
                if offset < len(sequence):
                    rows.append(sequence[offset])
        chosen.extend(rows[:limit])
        unused.extend(rows[limit:])
    chosen.extend(unused[:base.MAX_HOOKS-len(chosen)])
    return sorted(chosen), dict(available), total, regions


def run(args):
    nodes = load_pinned(NODE, NODE_SHA, "_pinned_combined_same_graph_nodes")
    combined = load_pinned(COMBINED, COMBINED_SHA, "_pinned_combined_sampling_backward")
    grid = combined.load_grid_probe()
    replay = nodes.load_replay()
    grid_trace = grid.GridBackwardTrace()
    resize_trace = combined.ResizeBackwardTrace()
    captured = {}
    regions = {}
    base_run = replay.run
    base_trace = nodes.trace_backwards

    def observed_replay(replay_args):
        stock_window = replay.trace_first_window

        def observe_window(model, criterion, optimizer, native_api, device, original, ema, fetch, tape):
            def joiner(_module, _args, value):
                need(len(value) == 3, "Expected native features, positions and cross-attention outputs")
                for prefix, sequence in zip(("feature", "position", "cross_attention"), value, strict=True):
                    captured.update(collect_tensors(sequence, prefix))

            def encoder(_module, _args, value):
                captured.update(collect_tensors(value, "encoder"))

            def patch_projection(_module, _args, value):
                need(torch.is_tensor(value), "Expected native patch projection tensor")
                captured["patch_projection"] = value

            projections = [module for name, module in model.named_modules()
                           if name.endswith("patch_embeddings.projection") and isinstance(module, torch.nn.Conv2d)]
            need(len(projections) == 1 and tuple(projections[0].weight.shape) == (384, 3, 12, 12),
                 "Exact native DINO patch projection")
            with ExitStack() as stack:
                stack.callback(model.backbone.register_forward_hook(joiner).remove)
                stack.callback(model.backbone[0].encoder.register_forward_hook(encoder).remove)
                stack.callback(projections[0].register_forward_hook(patch_projection).remove)
                return stock_window(model, criterion, optimizer, native_api, device, original, ema, fetch, tape)

        with patch.object(replay, "trace_first_window", observe_window):
            return base_run(replay_args)

    def select(root):
        dino = [v for k, v in captured.items() if k.startswith("encoder_") and v.requires_grad]
        joined = [v for k, v in captured.items() if k.startswith(("feature_", "position_", "cross_attention_"))
                  and v.requires_grad]
        chosen, available, count, labels = focused_nodes(root, dino, joined, nodes)
        regions.update(labels)
        return chosen, available, count

    def trace(loss, leaves, device, anchors):
        need(captured.get("feature_0_tensors") is anchors.get("backbone_first_feature")
             and "patch_projection" in captured, "Exact observed backbone boundaries")
        trainable = {k: v for k, v in captured.items() if v.requires_grad}
        with patch.object(nodes, "selected_nodes", select):
            observed = base_trace(loss, leaves, device, trainable)
        equality = observed["exact_boundary_gradients"]
        observed["boundaries"] = {k: dict(requires_grad=v.requires_grad, compared=equality.get(k) is not None,
                                           exact_gradient=equality.get(k), dtype=str(v.dtype), shape=list(v.shape))
                                  for k, v in captured.items()}
        equality["backbone_first_feature"] = equality["feature_0_tensors"]
        for row in observed["node_comparisons"]:
            row["region"] = regions[row["ordinal"]]
        observed["selected_by_region"] = dict(Counter(r["region"] for r in observed["node_comparisons"]))
        observed["compared_by_region"] = dict(Counter(r["region"] for r in observed["node_comparisons"] if r["compared"]))
        observed["selection_order"] = "BFS per operator family, 96 DINO + 24 non-DINO backbone + 8 outside; unused slots shared"
        grid_result, resize_result = grid_trace.summary(), resize_trace.summary()
        for route, skipped in ((grid_result, "unchanged_differentiable_cuda_bilinear_calls"),
                               (resize_result, "unchanged_cuda_bilinear_antialias_calls")):
            need(route[skipped] == 0 and route["routed_forward_calls"] > 0
                 and route["cpu_backward_calls"] == 2*route["routed_forward_calls"],
                 "All two-backward sampling route counts must close")
        observed["cpu_sampling_backward_routes"] = dict(grid_sample=grid_result, bilinear_interpolate=resize_result)
        observed["combined_sampling_backward_source_sha256"] = COMBINED_SHA
        observed["same_graph_source_sha256"] = NODE_SHA
        return observed

    need(not torch.are_deterministic_algorithms_enabled(), "Expected unchanged CUDA forward routing")
    with (patch.object(replay, "run", observed_replay), patch.object(nodes, "load_replay", lambda: replay),
          patch.object(nodes, "trace_backwards", trace),
          patch.object(F, "grid_sample", grid.routed_grid_sample(grid_trace)),
          patch.object(F, "interpolate", combined.routed_interpolate(resize_trace))):
        nodes.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", choices=("F0",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-commit", required=True)
    args = parser.parse_args()
    need(args.protocol_commit == PROTOCOL, "Exact verified P92 prospective protocol")
    run(args)
