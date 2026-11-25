"""Retrospective topology diagnosis on the closed Instance-RSDD stock deletions."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
from scipy.ndimage import generate_binary_structure, label
import shapely


SIGN_WORKER = "cdd7b3f7e1bbbeb528fb5100da76daee54b2a4f61f88e2381ef2b709e48a080e"
SIGN_RESULT = "02f2a09fd50d4cc3112ba43cc9b3ed76badff6720f386fb45143665d5dfb7fc5"
STATUS = "RSDD74_SEGREFINER_DELETION_TOPOLOGY_COMPLETE"
CONTEXTS = (("delete_only", "raw"), ("stock_lr", "add_only"))
FG8 = generate_binary_structure(2, 2)
BG4 = generate_binary_structure(2, 1)


def topology(mask):
    _, components = label(mask, structure=FG8)
    _, background_components = label(~np.pad(mask, 1), structure=BG4)
    return int(components), int(background_components - 1)


def verify_topology():
    ring = np.ones((3, 3), dtype=bool)
    ring[1, 1] = False
    diagonal = np.eye(3, dtype=bool)
    split = np.array([[1, 0, 1]], dtype=bool)
    assert topology(ring) == (1, 1)
    assert topology(diagonal) == (1, 0)
    assert topology(split) == (2, 0)
    assert topology(np.zeros_like(ring)) == (0, 0)


def import_signed(path):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != SIGN_WORKER:
        raise ValueError("Use the original accepted signed scorer")
    signed = ModuleType(path.stem)
    signed.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), signed.__dict__)
    return signed


def summarize(rows):
    counts = Counter(parent for parent, *_ in rows)
    observed = Counter()
    loss = loss_at_topology_event = mean_delta = 0.0
    for parent, delta, before, after in rows:
        weight = 1 / (len(counts) * counts[parent])
        more_components = after[0] > before[0]
        more_holes = after[1] > before[1]
        erased = before[0] > 0 and after[0] == 0
        event = more_components or more_holes or erased
        adverse = delta < -1e-8
        observed.update(more_components=int(more_components), more_holes=int(more_holes),
                        erased=int(erased), any_event=int(event), counts_unchanged=int(before == after),
                        adverse=int(adverse), adverse_with_event=int(adverse and event),
                        adverse_with_unchanged_counts=int(adverse and before == after))
        loss += weight * max(0.0, -delta)
        loss_at_topology_event += weight * max(0.0, -delta) * event
        mean_delta += weight * delta
    return dict(hulls=len(rows), parent_groups=len(counts), counts=dict(observed),
                equal_parent_mean_continuous_IoU_delta=mean_delta,
                equal_parent_gross_continuous_IoU_loss=loss,
                gross_loss_fraction_on_event=loss_at_topology_event/loss if loss else None)


def analyze(args):
    verify_topology()
    args.cohort, args.support_root = 74, None
    args.score_terminal = args.run / "observed_score" / "TERMINAL.json"
    args.inference_terminal = args.run / "observed_inference" / "TERMINAL.json"
    signed = import_signed(args.code / "SEGREFINER_SIGN_20261123" / "score_rsdd.py")
    stock = signed.pinned_module(args.code / "SEGREFINER_STOCK_20261123" / "score_rsdd.py", signed.STOCK_SOURCE)
    exposure = stock.exposure_source()
    whole, deps = exposure.sources(args)
    _, api, cfa, history, reserve, old, evaluator, adapter, core = deps
    reference = json.loads(api.pin(args.code.parent / "results/RSDD74_SEGREFINER_SIGN_SCORE_SAFE_20261123.json", SIGN_RESULT))
    report, score_sha, score_tsha = api.observed(args.run / "score", args.score_terminal, "score", stock.STATUS,
                                                signed.STOCK_SCORE, worker=signed.STOCK_SOURCE)
    infer, infer_sha, infer_tsha = api.observed(args.run / "inference", args.inference_terminal, "inference",
                                               stock.INFER_STATUS, report["inference_result_sha256"],
                                               report["inference_terminal_sha256"], report["inference_worker_sha256"])
    signed.require(reference["stock_score_sha256"] == score_sha and reference["inference_result_sha256"] == infer_sha
                   and reference["inference_terminal_sha256"] == infer_tsha and infer["target_labels_opened"] is False,
                   "The original observed inference and score are required")
    loader = cfa.source(args.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123" / "inputs.py", cfa.INPUTS_SHA)
    rows, lineage = loader.load_population(args)
    signed.require(lineage == infer["input_lineage"] == report["input_lineage"] == reference["input_lineage"],
                   "Original population required")
    blocks, areas, positions = exposure.native_blocks(rows, adapter, core)
    stream = api.pin(args.run / "inference" / stock.LEDGER, report["refined_masks_sha256"])
    counts = stock.attach_verified_masks(rows, blocks, areas, positions, stream, infer["counts"])
    signed.require(counts == report["counts"], "Actual original stock counts differ")
    writes = signed.attach_signed_masks(blocks, areas, counts, stock.GATE)
    signed.require(writes == reference["counts"], "Original signed counts required")
    print(json.dumps(dict(phase="qualified_original_writes", gate_candidates=writes["gate_candidates"])), flush=True)

    gt, _, polygons, sets, dates, label_sha = history.views_and_labels(args, old, reserve, blocks)
    signed.require(label_sha == reference["author_labels_sha256"], "Same author polygons required")
    views = {}
    for view, eligible_count in (("full", 39), ("date_separated", 26)):
        ids = set(sets[view])
        subset = {i: blocks[i] for i in ids}
        partial = dict(gt, images=[im for im in gt["images"] if im["id"] in ids],
                       annotations=[a for a in gt["annotations"] if a["image_id"] in ids])
        _, anchors = evaluator.score_one(partial, subset, "raw", anchors=True)
        output = {f"{arm}_minus_{base}": [] for arm, base in CONTEXTS}
        for gid, anchor in anchors.items():
            image, slot, _ = anchor
            poly = polygons[gid]
            if not stock.GATE <= areas[image][slot] < 1024 or not poly.is_valid or poly.area <= 0:
                continue
            masks = dict(zip(signed.ARMS, signed.rles_for(anchor, subset), strict=True))
            features = {}
            for name, mask in masks.items():
                shape = topology(codec.decode(mask).astype(bool))
                iou = 100 * core.geometric_iou(mask, poly, shapely)
                features[name] = (shape, iou)
            parent = subset[image]["group"]
            for arm, base in CONTEXTS:
                pair = (parent, features[arm][1] - features[base][1], features[base][0], features[arm][0])
                output[f"{arm}_minus_{base}"].append(pair)
        views[view] = {}
        for name, data in output.items():
            scored = summarize(data)
            matched = reference["views"][view]["gate_exposed_geometry"]
            expected = matched["group_equal_arms"][name.split("_minus_")[0]]["continuous_iou_points"]
            expected -= matched["group_equal_arms"][name.split("_minus_")[1]]["continuous_iou_points"]
            signed.require(scored["hulls"] == eligible_count and scored["parent_groups"] == matched["represented_groups"]
                           and abs(scored["equal_parent_mean_continuous_IoU_delta"] - expected) < 1e-8,
                           "Original paired geometry and eligibility must reproduce")
            views[view][name] = scored
    result = dict(status=STATUS, phase="score", retrospective=True, views=views,
                  new_model_forwards=0, new_optimizer_updates=0, topology="8-connected foreground, 4-connected padded exterior",
                  event_definition="new component count, new hole count or full erasure", weights="equal original parents",
                  original_deletion_geometry_reproduced=True, signed_parent_sha256=SIGN_RESULT,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.mkdir()
    stamp = api.write(args.output / "RESULT_SAFE.json", api.encode(result))
    print(json.dumps(dict(status=STATUS, result_sha256=stamp)), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("code", "run", "stock-root", "point-root", "author-labels", "output"):
        cli.add_argument("--" + name, required=True, type=Path)
    analyze(cli.parse_args())
