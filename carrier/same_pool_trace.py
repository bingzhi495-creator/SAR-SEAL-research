"""Opt-in pre-merge RLE trace writer for MaskTTAModel.

The writer is called only when ``tta_cfg.trace_dir`` is set. It never changes
the model's returned instances, scores, masks, ordering, or evaluation.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_util


def _rle(mask: np.ndarray) -> dict:
    encoded = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(x) for x in encoded["size"]],
            "counts": encoded["counts"].decode("ascii")}


def write_vote_trace(trace_dir, data_samples, aug_masks, merged_scores,
                     merged_labels, clusters, keep, results, tta_cfg,
                     has_model_mask_probs):
    """Save the exact source candidates and branch decisions of one image.

    ``clusters`` and ``keep`` come from the original GPU merge loop, so CPU
    reconstruction never decides which candidate belonged to a cluster.
    """
    start = time.perf_counter()
    image_id = int(data_samples[0].metainfo["img_id"])
    scores = merged_scores.detach().cpu().tolist()
    labels = merged_labels.detach().cpu().tolist()
    candidates = []
    offset = 0
    for view_id, masks in enumerate(aug_masks):
        batch = masks.detach().to(device="cpu", dtype=torch.uint8).numpy()
        for slot, mask in enumerate(batch):
            candidate_id = offset + slot
            candidates.append({
                "candidate_id": candidate_id,
                "view_id": view_id,
                "slot_in_view": slot,
                "score": float(scores[candidate_id]),
                "label": int(labels[candidate_id]),
                "mask": _rle(mask),
            })
        offset += len(batch)
        del batch
    output_masks = results.masks.detach().to(device="cpu", dtype=torch.uint8).numpy()
    output_scores = results.scores.detach().cpu().tolist()
    output_labels = results.labels.detach().cpu().tolist()
    outputs = [{"output_index": index,
                "score": float(output_scores[index]),
                "label": int(output_labels[index]),
                "mask": _rle(mask)}
               for index, mask in enumerate(output_masks)]
    view_meta = []
    for sample in data_samples:
        meta = sample.metainfo
        view_meta.append({
            "flip": bool(meta.get("flip", False)),
            "flip_direction": meta.get("flip_direction"),
            "scale_factor": [float(x) for x in meta.get("scale_factor", ())],
        })
    record = {
        "schema": "same_pool_premerge_vote_trace_v1",
        "image_id": image_id,
        "n_views": len(aug_masks),
        "view_meta": view_meta,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "clusters": clusters,
        "kept_cluster_indices": [int(x) for x in keep],
        "outputs": outputs,
        "merge_cfg": {
            "merge": tta_cfg.get("merge", "bbox_nms"),
            "vote_iou": float(tta_cfg.get("vote_iou", 0.5)),
            "vote_frac": float(tta_cfg.get("vote_frac", 0.5)),
            "vote_gamma": float(tta_cfg.get("vote_gamma", 0.0)),
            "vote_mask": tta_cfg.get("vote_mask", "majority"),
            "vote_min_support": int(tta_cfg.get("vote_min_support", 2)),
            "vote_max_support": int(tta_cfg.get("vote_max_support", 5)),
            "max_per_img": int(tta_cfg.max_per_img),
        },
        "has_model_mask_probs": bool(has_model_mask_probs),
        "trace_encoding_seconds": time.perf_counter() - start,
    }
    trace_path = Path(trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)
    path = trace_path / f"img_{image_id:08d}.json.gz"
    part = path.with_name(path.name + ".part")
    writing_started = time.perf_counter()
    with gzip.open(part, "wt", encoding="utf-8", compresslevel=4) as stream:
        json.dump(record, stream, ensure_ascii=False, separators=(",", ":"))
    os.replace(part, path)
    write_seconds = time.perf_counter() - writing_started
    cuda_peak_bytes = (int(torch.cuda.max_memory_allocated())
                       if results.masks.is_cuda else None)
    print("SAME_POOL_TRACE image_id=%d candidates=%d clusters=%d kept=%d "
          "encode_s=%.4f write_s=%.4f bytes=%d cuda_peak_bytes=%s path=%s" % (
              image_id, len(candidates), len(clusters), len(outputs),
              record["trace_encoding_seconds"], write_seconds,
              path.stat().st_size, cuda_peak_bytes, path), flush=True)
    return path


def _decode_rle(encoded):
    return mask_util.decode({"size": encoded["size"],
                             "counts": encoded["counts"].encode("ascii")}).astype(bool)


def _encode_bool(mask):
    return _rle(mask.astype(np.uint8))


def _majority(candidates, ids, fraction):
    count = np.zeros_like(_decode_rle(candidates[ids[0]]["mask"]), dtype=np.int16)
    for candidate_id in ids:
        count += _decode_rle(candidates[candidate_id]["mask"])
    return _encode_bool(count >= len(ids) * fraction)


def replay_vote_trace(record, per_view_top1=False):
    """Replay saved GPU clusters on CPU, with optional simple top-1 readout.

    The per-view control keeps cluster identities/counts, labels, seed scores,
    and top-K order. It changes only each cluster's binary mask readout.
    """
    if record.get("schema") != "same_pool_premerge_vote_trace_v1":
        raise ValueError("unsupported trace schema")
    cfg = record["merge_cfg"]
    if cfg["merge"] != "vote" or cfg["vote_gamma"] != 0.0:
        raise ValueError("replay currently supports vote merge with gamma=0 only")
    candidates = record["candidates"]
    by_cluster = []
    for cluster in record["clusters"]:
        seed = int(cluster["seed_id"])
        members = [int(x) for x in cluster["member_ids"]]
        if per_view_top1:
            best_by_view = {}
            for candidate_id in members:
                candidate = candidates[candidate_id]
                view = int(candidate["view_id"])
                previous = best_by_view.get(view)
                if previous is None or candidate["score"] > candidates[previous]["score"]:
                    best_by_view[view] = candidate_id
            selected = sorted(best_by_view.values())
            mask = _majority(candidates, selected, 0.5)
        else:
            mode = cfg["vote_mask"]
            if mode == "seed":
                mask = candidates[seed]["mask"]
            elif mode == "consensus":
                if cfg["vote_min_support"] <= len(members) <= cfg["vote_max_support"]:
                    mask = _majority(candidates, members, cfg["vote_frac"])
                else:
                    mask = candidates[seed]["mask"]
            elif mode == "majority":
                mask = _majority(candidates, members, cfg["vote_frac"])
            else:
                raise ValueError(f"readout {mode!r} is not supported by CPU replay")
        by_cluster.append({"score": float(candidates[seed]["score"]),
                           "label": int(candidates[seed]["label"]),
                           "mask": mask})
    return [by_cluster[int(index)] for index in record["kept_cluster_indices"]]


def validate_vote_trace(record):
    """Return exact source-to-output replay mismatches, without reading GT/AP."""
    problems = []
    candidates = record.get("candidates", [])
    if [c.get("candidate_id") for c in candidates] != list(range(len(candidates))):
        problems.append("candidate IDs are not contiguous pooled order")
    seen = []
    for cluster in record.get("clusters", []):
        members = cluster["member_ids"]
        if cluster["seed_id"] not in members:
            problems.append("seed absent from cluster")
        seen.extend(members)
    if sorted(seen) != list(range(len(candidates))):
        problems.append("cluster membership does not partition candidate pool")
    try:
        replayed = replay_vote_trace(record)
    except (KeyError, IndexError, ValueError) as error:
        return problems + [f"replay failed: {error}"]
    outputs = record.get("outputs", [])
    if len(replayed) != len(outputs):
        problems.append("output count differs")
        return problems
    for index, (actual, expected) in enumerate(zip(replayed, outputs)):
        if actual["score"] != expected["score"]:
            problems.append(f"output {index}: score differs")
        if actual["label"] != expected["label"]:
            problems.append(f"output {index}: label differs")
        if not np.array_equal(_decode_rle(actual["mask"]),
                              _decode_rle(expected["mask"])):
            problems.append(f"output {index}: mask differs")
    return problems
