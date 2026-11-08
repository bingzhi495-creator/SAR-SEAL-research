"""FIT-only dual-phase query geometry and exact scalar-threshold oracle."""
import argparse
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

SIDE = 576
WIDTH = 100
GROUPS = 13
CHUNK = 2048
REPLAY_SHA = "5a5001c18da150436576cc1cdb4e0d581a5e516e7e7e3ab0776e07b308e98c85"


def meets(intersection, union, numerator, denominator):
    return (union > 0) & (denominator * intersection >= numerator * union)


def threshold_counts(scores, truth, threshold):
    foreground = scores > threshold
    return int((foreground & truth).sum()), int((foreground | truth).sum())


def best_threshold(scores, truth):
    """Choose a strict threshold on phase A, grouping equal scores."""
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    split = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1, len(scores)]
    true_positive = np.r_[0, np.cumsum(truth[order], dtype=np.int64)[split - 1]]
    cut = np.r_[sorted_scores[0], sorted_scores[split[:-1]], -np.inf]
    ratios = true_positive / (truth.sum() + np.r_[0, split] - true_positive)
    i, u = threshold_counts(scores, truth, 0)
    if i/u == ratios.max():
        return 0.
    tied = cut[ratios == ratios.max()]
    return float(tied[np.argmin(np.abs(tied))])


def phase_geometry(torch, packet, phase, fixed_queries, cap):
    """Replicate native sparse point logits and accumulate integer overlaps."""
    from rfdetr.models.heads.segmentation import point_sample

    sparse = {key: value.cuda() for key, value in packet["sparse"].items()}
    masks = packet["target_masks"].cuda()
    n = masks.shape[0]
    fixed = torch.from_numpy(fixed_queries).cuda()

    def samples(points):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = point_sample(sparse["spatial_features"], points[None], align_corners=False)
            logits = torch.einsum("bcp,bnc->bnp", features, sparse["query_features"]) + sparse["bias"]
            labels = point_sample(masks.to(logits.dtype).unsqueeze(1),
                                  points[None].repeat(n, 1, 1), align_corners=False, mode="nearest")
        return logits[0], labels[:, 0]

    original = packet["uniform_draw"].cuda()
    z, y = samples(original)
    cap.need(torch.equal(z.float().cpu(), packet["native_pred_at_draw"])
             and torch.equal(y.float().cpu(), packet["native_target_at_draw"]), "Native point anchor changed")
    intersection = torch.zeros((GROUPS*WIDTH, n), dtype=torch.int64, device="cuda")
    predicted = torch.zeros(GROUPS*WIDTH, dtype=torch.int64, device="cuda")
    annotated = torch.zeros(n, dtype=torch.int64, device="cuda")
    chosen = np.empty((n, SIDE*SIDE), np.float32)
    labels = np.empty((n, SIDE*SIDE), np.bool_)
    with torch.no_grad():
        for lo in range(0, SIDE*SIDE, CHUNK):
            hi = min(lo + CHUNK, SIDE*SIDE)
            y, x = np.divmod(np.arange(lo, hi), SIDE)
            xy = np.column_stack(((x + phase[0])/SIDE, (y + phase[1])/SIDE)).astype(np.float32)
            z, mask = samples(torch.from_numpy(xy).cuda())
            pred = (z > 0).float()
            target = mask.float()
            # Binary inputs and each chunk sum <=2048, so integer counts are exactly representable.
            intersection += (pred @ target.T).to(torch.int64)
            predicted += pred.sum(1).to(torch.int64)
            annotated += target.sum(1).to(torch.int64)
            chosen[:, lo:hi] = z[fixed].float().cpu().numpy()
            labels[:, lo:hi] = mask.bool().cpu().numpy()
    union = predicted[:, None] + annotated[None] - intersection
    return intersection.cpu().numpy(), union.cpu().numpy(), chosen, labels


def describe_view(torch, packet, math, cap):
    n = len(packet["is_small"])
    assigned = math.assignment_from_native_pairs(packet["native_src"].numpy(), packet["native_dst"].numpy(),
                                                 targets=n)[0]
    a, b = [phase_geometry(torch, packet, phase, assigned, cap) for phase in math.DENSE_PHASES]
    small = packet["is_small"].numpy()
    scopes = {"all": np.ones(n, dtype=bool), "small": small}
    availability = {}
    for tag, threshold in (("iou75", (3, 4)), ("iou90", (9, 10))):
        good_a, good_b = [meets(*row[:2], *threshold).reshape(GROUPS, WIDTH, n) for row in (a, b)]
        for label, good in (("phase_a", good_a), ("phase_b", good_b), ("both_same_query", good_a & good_b)):
            for scope, select in scopes.items():
                absent = ~good.any(axis=1)[:, select]
                availability[f"{tag}_{label}_{scope}"] = [int(absent[0].sum()), int(absent[1:].sum()),
                                                           int(select.sum()), int((GROUPS-1)*select.sum())]
    a_good = meets(*a[:2], 9, 10)
    b_good = meets(*b[:2], 9, 10)
    any_strict = (a_good[:WIDTH] & b_good[:WIDTH]).any(0)
    localized = np.logical_and.reduce([meets(row[0][assigned, np.arange(n)], row[1][assigned, np.arange(n)], 1, 2)
                                      for row in (a, b)])
    eligible = np.flatnonzero(~any_strict & localized)
    repaired_a = []
    repaired_both = []
    for i in eligible:
        t = best_threshold(a[2][i], a[3][i])
        good_a = meets(*threshold_counts(a[2][i], a[3][i], t), 9, 10)
        good_b = meets(*threshold_counts(b[2][i], b[3][i], t), 9, 10)
        repaired_a.append(bool(good_a))
        repaired_both.append(bool(good_a and good_b))
    return availability, np.array([n, (~any_strict & ~localized).sum(), len(eligible), sum(repaired_a),
                                   sum(repaired_both)], dtype=np.int64)


def run(args):
    start = time.perf_counter()
    path = args.replay_runtime
    raw = path.read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != REPLAY_SHA:
        raise ValueError("Pinned closed replay runtime differs")
    replay = types.ModuleType("_closed_matcher_replay")
    replay.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), replay.__dict__)
    need = replay.need
    need(sys.version_info[:3] == (3, 12, 14) and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
         and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
         "Exact isolated DGX runtime")
    replay.canonical(args.capture_dir, directory=True)
    need(args.output.is_absolute() and replay.canonical(args.output.parent, directory=True) == args.output.parent
         and not args.output.exists() and not args.output.is_symlink()
         and not args.output.is_relative_to(args.capture_dir), "Fresh output outside archived capture")
    cap = replay.module(args.capture_source, replay.CAPTURE_SHA, "_pinned_capture_matcher_cost")
    math = replay.module(path.with_name("matcher_cost_math.py"), replay.MATH_SHA, "_pinned_matcher_math")
    report = json.loads(replay.checked_bytes(args.capture_dir/"CAPTURE_SAFE.json", args.capture_report_sha256))
    receipt = json.loads(replay.checked_bytes(args.capture_closure, args.capture_closure_sha256))
    rows = replay.validate_closed(report, receipt, args.capture_protocol, args.capture_report_sha256, cap)
    items = json.loads(replay.checked_bytes(args.fit_index, replay.FIT_MANIFEST_SHA))["items"]
    plan = cap.view_plan(items, report["tranche"])
    need(report["tranche"] == "cohort32" and len(items) == 741
         and len({item["image_sha256"] for item, _ in plan}) == 32, "Exact unique FIT scope")
    sys.addaudithook(cap.offline)
    import torch
    need(all(version(name) == value for name, value in cap.VERSIONS.items())
         and all(replay.digest(Path(import_module(name).__file__).read_bytes()) == digest
                 for name, digest in cap.NATIVE_SOURCE.items()), "Exact capture numerical runtime and source")
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One DGX CUDA device")
    torch.cuda.set_per_process_memory_fraction(min(.35, cap.CAP/torch.cuda.get_device_properties(0).total_memory))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    totals = {}
    outcomes = []
    for row in rows:
        if row["status"] != "STOCK_TRACE_EXACT":
            continue
        packet = replay.load_packet(torch, args.capture_dir, row, plan, args.capture_protocol, cap)
        result, counts = describe_view(torch, packet, math, cap)
        for k, v in result.items():
            totals[k] = totals.get(k, np.zeros(4, dtype=np.int64)) + v
        outcomes.append(counts)
        print(json.dumps(dict(event="DUAL_PHASE_MASK_GEOMETRY_VIEW_COMPLETE", ordinal=row["ordinal"])), flush=True)
    totals = {k: v.tolist() for k, v in totals.items()}
    for tag, expected in (("iou75_phase_a_all", 25), ("iou75_phase_a_small", 13),
                          ("iou90_phase_a_all", 482), ("iou90_phase_a_small", 260)):
        need(sum(totals[tag][:2]) == expected, "Phase A independently agrees with fixed P126 absence counts")
    counts = np.stack(outcomes)
    n, unlocalized, eligible, rescued_a, rescued_both = counts.sum(axis=0).tolist()
    selected = counts[:, 2] > 0
    repair_rate = replay.image_pair([(float(r[4]/r[2]), 0.) for r in counts[selected]])
    go = (int(selected.sum()) >= 15 and eligible >= 25 and repair_rate["iid"] >= .15
          and repair_rate["percentile95"][0] > 0)
    safe = dict(status="SSDD_FIT_DUAL_PHASE_MASK_GEOMETRY_COMPLETE", protocol=args.protocol,
                capture_protocol=args.capture_protocol, FIT_only=True, DEV=False, TEST=False, AP=False,
                optimizer_updates=0, traced_views=len(outcomes), empty_views=len(rows)-len(outcomes),
                thresholds=[.75, .90], phases=math.DENSE_PHASES,
                availability_order=["first_group_absent", "other_groups_absent", "first_group_targets",
                                    "other_group_targets"], availability=totals,
                fixed_match_threshold=dict(targets=n, not_localized_without_strict=unlocalized,
                                           eligible=eligible, eligible_images=int(selected.sum()),
                                           phase_a_rescued=rescued_a, both_phases_rescued=rescued_both,
                                           image_mean_confirmed=repair_rate, screen_gate_pass=go),
                phase_a_reference_counts_match=True, seconds=time.perf_counter()-start,
                cuda_peak_bytes=torch.cuda.max_memory_allocated())
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(safe, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(safe, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("replay-runtime", "capture-dir", "capture-source", "capture-closure", "fit-index", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("protocol", "capture-protocol", "capture-report-sha256", "capture-closure-sha256"):
        parser.add_argument("--"+name, required=True)
    run(parser.parse_args())
