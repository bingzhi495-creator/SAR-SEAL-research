"""Fit two disposable point heads on four unchanged, independently accepted RF-DETR FIT images."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter

MANIFEST_SHA = "af43530e7f8f118261b5523cdde738cca467ead813d38f1d4c989c8f4102e184"
PREDICTIONS_SHA = "2ed2f67671705a8b5b502a3661190c6c16b354c8a4d940d28a8c67943786f9df"
ACCEPTANCE_SHA = "47bed472492920b3f8950c60ea77fcf3b5904665f8bcf6a374bfc5bc1274289f"
GT_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
PAIR_SHA = "8fe7a2c146c08c89f377b1f20b321276a60356de4bdbb27be2a00544dfc77962"
RANK_SHA = "cb438f8937f8e29dbdcbc441abdf5a788ac9c442fd63f1b3ad504b8ae0fcca0b"
ADAPTER_SHA = "5b9a25a4e9c8c125342dd9e8fd7e0d1aeb3ecb546f5cf95b8512dfe09116b0f2"
ARMS = ("N", "SF", "SU=F", "SU", "LF=S", "LU=S", "LW=S", "NP=S")
MIB = 1024**2


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pinned(path, digest, limit):
    with path.open("rb") as src:
        data = src.read(limit + 1)
    need(0 < len(data) <= limit and sha256(data).hexdigest() == digest, "Pinned bytes: " + path.name)
    return data


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write(path, value):
    raw = encode(value)
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    return sha256(raw).hexdigest()


def native_candidate_rows(block, joint_count=100):
    """Use the frozen joint rank with its unmapped background slot gaps."""
    joint_q = block["joint_query_ids"]
    joint_labels = block["joint_labels"]
    joint_scores = block["joint_scores"]
    need(len(joint_q) == len(joint_labels) == len(joint_scores) == joint_count
         and len(block["ship_scores"]) == len(block["all_rles"]) == 164
         and all(0 <= q < 164 for q in joint_q) and set(joint_labels) <= {0, 1}
         and all(0 <= score <= 1 for score in joint_scores)
         and all(a >= b for a, b in zip(joint_scores, joint_scores[1:])),
         "Original joint candidate identities and scores")
    expected = [(rank, q, score) for rank, (q, label, score) in
                enumerate(zip(joint_q, joint_labels, joint_scores, strict=True), 1) if label == 0]
    need([(rank, q) for rank, q, _ in expected] == list(zip(block["native_ranks"], block["native_ids"], strict=True))
         and len({q for _, q, _ in expected}) == len(expected)
         and all(score == block["ship_scores"][q] for _, q, score in expected),
         "Original class-zero map in joint rank order")
    return [dict(image_id=block["image_id"], category_id=1, rank=rank, query_id=q,
                 score=score, rle=block["all_rles"][q]) for rank, q, score in expected]


def read_cohort(args, select_images):
    accepted = json.loads(pinned(args.acceptance, ACCEPTANCE_SHA, 64 * 1024))
    need(accepted["status"] == "FIT114_CORE36_CENTER5202_Q164_INDEPENDENT_READBACK_V1"
         and accepted["original_prediction_sha256"] == PREDICTIONS_SHA
         and accepted["images"] == 36 and accepted["mapped"] == 3460
         and accepted["raw_logit_rows"] == accepted["raw_logit_raster_RLE_exact"] == 2130
         and accepted["FIT"] and not any(accepted[k] for k in ("GT_opened", "DEV", "TEST")), "Closed CPU carrier")
    manifest = json.loads(pinned(args.manifest, MANIFEST_SHA, 64 * 1024))
    need(manifest["FIT"] and not any(manifest[k] for k in ("DEV", "TEST"))
         and len(manifest["images"]) == 36, "Original FIT image manifest only")
    rows = [json.loads(line) for line in pinned(args.capture / "QUERIES_PRIVATE.jsonl", PREDICTIONS_SHA, 8 * MIB).splitlines()]
    need(len(rows) == 36 and [b["image_id"] for b in rows] == [r["image_id"] for r in manifest["images"]],
         "Identical original image and prediction order")
    selected = select_images(manifest["images"], 4)
    positions = {row["image_id"]: (i, b) for i, (row, b) in enumerate(zip(manifest["images"], rows, strict=True), 1)}
    return [(item, *positions[item["image_id"]]) for item in selected]


def author_boxes(raw, torch):
    boxes = torch.as_tensor(raw, dtype=torch.float32)
    need(boxes.shape == (164, 4) and bool(torch.isfinite(boxes).all()), "Original finite box tensor")
    cx, cy, w, h = boxes.unbind(-1)
    return (torch.stack((cx - .5 * w, cy - .5 * h, cx + .5 * w, cy + .5 * h), dim=-1) * 800).clamp(0, 800).numpy()


def has_author_support(box):
    x1, y1, x2, y2 = box
    need(x2 >= x1 and y2 >= y1, "Non-inverted original clipped author box")
    return x2 > x1 and y2 > y1


def prepare(args, roster, fit_step, feat, posterior, np, torch, codec):
    from PIL import Image

    base = args.code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24"
    pair = fit_step.load(base / "fit109_paired_point/pair_math.py", PAIR_SHA, "_rfd_pair")
    rank = fit_step.load(base / "fit109_probability_rank/rank_math.py", RANK_SHA, "_rfd_rank")
    data, counts, seal = [], Counter(), sha256()
    for item, ordinal, block in roster:
        name = item["native_basename"]
        need(re.fullmatch(r"P\d+_[A-Za-z0-9_]+\.png", name) and name == block["native_basename"]
             and item["group"] == block["source_prefix"]
             and item["decoded_rgb_sha256"] == block["image_content_sha256"]
             and item["encoded_sha256"] == block["encoded_sha256"], "Exact image and candidate provenance")
        rgb = pinned(args.fit_images / name, item["encoded_sha256"], 4 * MIB)
        with Image.open(BytesIO(rgb)) as image:
            need(image.mode == "RGB" and image.size == (800, 800), "Original full FIT PNG shape and mode")
            pixels = np.asarray(image)
        need(len(rgb) == item["encoded_bytes"] and sha256(pixels.tobytes()).hexdigest() == item["decoded_rgb_sha256"],
             "Exact decoded RGB bytes")
        image_features = pair.image_features(pixels)
        candidates = native_candidate_rows(block)
        for cand in candidates:
            cand["area"] = int(codec.area(cand["rle"]))
        eligible = [row for row in candidates if 32 <= row["area"] < 1024]
        cap = block["logit_capture"]
        location = f"RAW_LOGITS/{ordinal:03d}.npy"
        need(cap["relative_path"] == location, "Original logit sidecar")
        payload = pinned(args.capture / location, cap["sha256"], 9 * MIB)
        logits = np.load(BytesIO(payload), allow_pickle=False)
        need(logits.dtype == np.dtype("<f4") and logits.shape == (len(eligible), 144, 144)
             and list(logits.shape) == cap["shape"] and len(payload) == cap["bytes"]
             and bool(np.isfinite(logits).all()), "Exact retained raw logit planes")
        boxes = author_boxes(block["raw_boxes_cxcywh"], torch)
        records = []
        for k, (row, proof, raw) in enumerate(zip(eligible, cap["candidates"], logits, strict=True)):
            qid = row["query_id"]
            need((proof["logit_row"], proof["query_id"], proof["native_rank"], proof["score"], proof["predicted_area"])
                 == (k, qid, row["rank"], row["score"], row["area"]), "Original probability-to-rank mapping")
            stock = codec.decode(row["rle"]).astype(bool)
            prob = feat.render_posterior(torch, np, posterior, raw, stock)
            counts["exact_original_masks"] += 1
            if not has_author_support(boxes[qid]):
                counts["zero_area_author_boxes"] += 1
                continue
            points = feat.extract_candidate(np, pair, rank, stock, boxes[qid], row["score"], prob, image_features)
            if points is None:
                continue
            points.update(image_id=item["image_id"], rank=row["rank"], query_id=qid, group=item["group"])
            records.append(points)
            seal.update(encode(dict(image_id=item["image_id"], rank=row["rank"], query_id=qid, group=item["group"])))
            seal.update(points["coordinates"].tobytes())
            seal.update(points["features"].tobytes())
            counts["pairs"] += len(points["coordinates"])
        counts["mapped"] += len(candidates)
        counts["eligible"] += len(eligible)
        counts["candidate_point_sets"] += len(records)
        counts["original_empty_masks"] += sum(row["area"] == 0 for row in candidates)
        data.append((item, candidates, records))
    return data, dict(counts), seal.hexdigest()


def attach(args, data, fit_step):
    source = json.loads(pinned(args.fit_gt, GT_SHA, 32 * MIB))
    need(len(source["images"]) == 2937 and len(source["annotations"]) == 9260
         and [row["id"] for row in source["categories"]] == [1], "Original native HRSID FIT labels only")
    chosen = {item["image_id"] for item, _, _ in data}
    images = {row["id"]: row for row in source["images"] if row["id"] in chosen}
    annotations = defaultdict(list)
    for ann in source["annotations"]:
        if ann["image_id"] in chosen:
            annotations[ann["image_id"]].append(ann)
    helpers = fit_step.load_geometry(args.code, args.shapely_site)
    records, counts = [], Counter()
    for item, candidates, points in data:
        image = images[item["image_id"]]
        need(image["file_name"] == item["native_basename"] and image["height"] == image["width"] == 800,
             "Original FIT image and annotation identity")
        joined, audit = fit_step.attach_truth(image, annotations[image["id"]], candidates, points, helpers)
        counts.update(audit)
        records.extend(joined)
    counts["selected_annotations"] = sum(map(len, annotations.values()))
    return records, dict(counts)


def verify_edits(data, records, fitted, feat, np, codec):
    output = {arm: Counter(mapped=sum(len(cands) for _, cands, _ in data)) for arm in ARMS}
    candidates = {(row["image_id"], row["rank"]): row for _, rows, _ in data for row in rows}
    for row, small, large in zip(records, fitted["prediction"]["S"], fitted["prediction"]["L"], strict=True):
        candidate = candidates[row["image_id"], row["rank"]]
        need(row["query_id"] == candidate["query_id"], "Same query for original candidate and refined points")
        stock = codec.decode(candidate["rle"]).astype(bool)
        coords = row["coordinates"]
        cap = min(8, candidate["area"] // 10, len(coords))
        edits = feat.readouts_sparse(np, stock, coords, small, large, row["features"][:, :, 0], cap)
        need(set(edits) == set(ARMS), "Complete control matrix")
        need(edits["SF"]["edits"] == edits["SU=F"]["edits"] and
             all(edits[k]["edits"] == edits["SU"]["edits"] for k in ("LF=S", "LU=S", "LW=S", "NP=S")),
             "Actual per-candidate equal editing")
        for arm, edit in edits.items():
            mask = feat.apply_sparse(np, stock, edit)
            rle = codec.encode(np.asfortranarray(mask, dtype=np.uint8))
            rle["counts"] = rle["counts"].decode("ascii")
            need(int(codec.area(rle)) == candidate["area"] + edit["area_delta"] and
                 (arm == "LW=S" or edit["area_delta"] == 0) and
                 (edit["edits"] != 0 or rle == candidate["rle"]), "RLE area and no-op identity")
            counter = output[arm]
            counter["verified_point_candidates"] += 1
            counter["edited"] += bool(edit["edits"])
            for key in ("edits", "adds", "drops", "area_delta", "nonpositive_actions", "nonpositive_flips"):
                counter[key] += edit[key]
    return {arm: dict(value, unchanged=value["mapped"] - value["edited"]) for arm, value in output.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "manifest", "acceptance", "fit-images", "fit-gt", "code", "posterior", "shapely-site", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only source spot")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fit_step
    import feature_adapter as feat
    import numpy as np
    import torch
    from pycocotools import mask as codec

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    need(not torch.cuda.is_initialized(), "No accelerator context")
    start = perf_counter()
    posterior = fit_step.load(args.posterior, ADAPTER_SHA, "_rfd_closed_posterior")
    cohort = read_cohort(args, feat.select_images)
    data, counts, feature_sha = prepare(args, cohort, fit_step, feat, posterior, np, torch, codec)
    need(counts["mapped"] > 0 and counts["exact_original_masks"] == counts["eligible"]
         and counts["candidate_point_sets"] > 0, "Complete original parity before FIT labels")
    args.output.mkdir()
    prelabel = dict(status="RFDETR_CORE36_FOUR_FIT_IMAGES_PRELABEL", selected_images=len(data),
                    independent_acceptance_sha256=ACCEPTANCE_SHA, manifest_sha256=MANIFEST_SHA,
                    selection_sha256=sha256(encode([(item["decoded_rgb_sha256"], item["image_id"]) for item,_,_ in data])).hexdigest(),
                    feature_seal_sha256=feature_sha, **counts, FIT=True, DEV=False, TEST=False, GT_opened=False)
    prelabel_sha = write(args.output / "PRELABEL_SAFE.json", prelabel)
    print(json.dumps(prelabel), flush=True)
    records, labels = attach(args, data, fit_step)
    trainer = fit_step.load_trainer(args.code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_paired_point/train_point.py")
    fitted = fit_step.fit_heads(records, trainer)
    edits = verify_edits(data, records, fitted, feat, np, codec)
    need(not torch.cuda.is_initialized(), "No GPU")
    result = dict(status="RFDETR_CORE36_FOUR_FIT_IMAGES_ACTUAL", images=len(data), preparation=counts,
                  prelabel_sha256=prelabel_sha, supervision=labels, fit=fitted["audit"], policies=edits,
                  model_forward_calls=0, detector_optimizer_updates=0, AP=False, FIT=True, DEV=False,
                  TEST=False, FIT_GT_opened=True, torch_version=torch.__version__, wall_seconds=perf_counter()-start)
    result_sha = write(args.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(result_sha256=result_sha, **result)), flush=True)


if __name__ == "__main__":
    main()
