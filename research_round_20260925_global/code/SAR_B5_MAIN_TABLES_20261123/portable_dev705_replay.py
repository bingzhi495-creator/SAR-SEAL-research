"""Reproduce all eight RF-DETR center5202 point policies on DEV705, without labels."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("_dev_capture_constants", str(HERE / "portable_dev705_capture.py"))
cap = module_from_spec(spec)
spec.loader.exec_module(cap)
require = cap.require
CHECKPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
HEADS = {
    5217: "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1",
    5218: "08b39b9911f77c039b26f8bb9bc34e940f5a350390911586f47290b2c3d88869",
    5219: "93a3be4356816f9d29560d40d9bff8cc5d8730e9e600b29db07959612f504604",
}
STOCK_SHA = "da55f2d3f9bbf08b26d5fae76606b2496a5b482c5fb61f4fab53c23172e5c06e"
EDITS_SHA = "f8c20ce94715e651a1e5cd90e2213a3a2793c2b9454de45de212d07c69335a6b"
EXPECTED = dict(eligible=42325, empty=15114, exact_original_masks=42325, images=705, mapped=67386,
                point_pairs=1298958, supported=40897, unsupported=1428, zero_area_author_boxes=1041)
ARMS = ("N", "SF", "SU=F", "SU", "LF=S", "LU=S", "LW=S", "NP=S")
CTX = None


def budgets():
    return {seed:{arm:Counter() for arm in ARMS} for seed in HEADS}


def initialize(args, repair):
    global CTX
    import numpy as np
    import torch
    from pycocotools import mask as codec

    require(not torch.cuda.is_initialized(), "CPU readout")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    portable = cap.loader(args.code)
    treatment = portable.module(args.code / "HRSID_OFFICIAL_FROZEN_REFINEMENT_CAPTURE_READBACK_20261121/capture.py",
                                "a50da3d740d9620a23c98b978b7cfaa383040f1e7542e2eb5da32ca4b0106da6")
    policy = treatment.load_policy(args.code)
    heads = {s:policy.plan.restore_bundle(args.heads / f"{s}_all_fit.npz", digest, policy.trainer, policy.fit_step)
             for s, digest in HEADS.items()}
    formal = args.code / "RFDETR_MATCHED_POINT_FORMAL_20261120"
    verify = portable.module(formal / "verify_dev_readout.py", "203286c6eea046836faf8ce74c8a815ae004d2c198616020d9c5dc0e5cc9c4d7")
    sparse = portable.module(formal / "score_dev.py", "271554ea01f5e5e24e48ff2c53c3229755615aa514624d7d89e5c9ec559fb79c")
    png = portable.module(args.code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705/dfine_dev705.py",
                          policy.readout.PNG_READER)
    require(tuple(policy.readout.ARMS) == ARMS and tuple(policy.readout.SEEDS) == tuple(HEADS), "Locked point policy roster")
    CTX = SimpleNamespace(np=np, torch=torch, codec=codec, policy=policy, reader=policy.readout, heads=heads, verify=verify,
                          sparse=sparse, png=png, archive=ZipFile(args.archive), root=args.capture, repair=repair)


def verified_masks(c, stock, candidate, coords, chosen, expected):
    # Both verifier routines are separate published implementations. They read the
    # sparse pixel indices, reconstruct the rasters, and check each arm and budget.
    cache = {}
    def get_edit(seed, rank, arm):
        return chosen[str(seed)].get(arm, c.sparse.ZERO[arm])
    for seed in HEADS:
        values = c.verify.verify_masks(c.np, c.codec, stock, candidate["rle"], coords, ARMS, get_edit,
                                      c.policy.features, seed, candidate["rank"], cache=cache)
        for arm, observed in values.items():
            expected[seed][arm].update(verified=1, **observed)
        require(values["SF"]["edits"] == values["SU=F"]["edits"] and all(
                values[name]["edits"] == values["SU"]["edits"] for name in ("LF=S","LU=S","LW=S","NP=S")),
                "Independent per-candidate edit quotas")


def replay_image(task):
    ordinal, item, query_line, stock_line = task
    c = CTX
    p, r = c.policy, c.reader
    block = json.loads(query_line)
    image_id = item["image_id"]
    require(block["image_id"] == image_id and block["image_content_sha256"] == item["decoded_rgb_sha256"] and
            block["encoded_sha256"] == item["encoded_sha256"], "Native query and source image differ")
    pool = p.spot.native_candidate_rows(block)
    require(r.legacy_stock_bytes(image_id, pool) == stock_line, "Native rank, score and RLE parity")
    for cand in pool:
        cand["area"] = int(c.codec.area(cand["rle"]))
    eligible = [cand for cand in pool if 32 <= cand["area"] < 1024]
    ref = block["logit_capture"]
    require(ref["relative_path"] == f"RAW_LOGITS/{ordinal:03d}.npy", "Original plane identity")
    path = c.root / ref["relative_path"]
    raw = r.pin(path, ref["sha256"], 9 * r.MIB)
    buf = BytesIO(raw)
    values = c.np.load(buf, allow_pickle=False)
    require(not buf.read(1) and len(raw) == ref["bytes"] and values.dtype == c.np.dtype("<f4") and
            list(values.shape) == ref["shape"] == [len(eligible),144,144] and c.np.isfinite(values).all(),
            "Exact saved logit file")
    src = dict(relative_path=ref["relative_path"], path=str(path), bytes=len(raw), sha256=ref["sha256"])
    rgb, hw = c.png.read_png(c.archive, item)
    require(hw == (800,800), "Original 800px canvas")
    features = p.pair.image_features(rgb)
    boxes = p.spot.author_boxes(block["raw_boxes_cxcywh"], c.torch)
    counts = Counter(images=1, mapped=len(pool), eligible=len(eligible), empty=sum(cand["area"] == 0 for cand in pool))
    original, independent = budgets(), budgets()
    selected = {str(s):{} for s in HEADS}
    points = []
    used_before = c.repair["_consumed"]
    for i, (cand, proof, plane) in enumerate(zip(eligible, ref["candidates"], values, strict=True)):
        q, rank = cand["query_id"], cand["rank"]
        require(proof == dict(logit_row=i, query_id=q, native_rank=rank, score=cand["score"], predicted_area=cand["area"]),
                "Original retained candidate identity")
        stock = c.codec.decode(cand["rle"]).astype(bool)
        identity = dict(ordinal=ordinal, image_id=image_id, query_id=q, native_rank=rank, logit_row=i)
        prob = r.probability_with_repair(c.np, c.torch, p.features, p.posterior, plane, stock, cand["rle"],
                                         identity, src, c.repair)
        counts["exact_original_masks"] += 1
        record = None
        if p.spot.has_author_support(boxes[q]):
            record = p.features.extract_candidate(c.np, p.pair, p.rank, stock, boxes[q], cand["score"], prob, features)
        coords = c.np.empty((0,2), dtype=c.np.int32) if record is None else record["coordinates"]
        degenerate = c.verify.verify_points(c.np, stock, boxes[q], coords, p.pair, p.rank)
        counts["zero_area_author_boxes"] += int(degenerate)
        points.append(dict(rank=rank, query_id=q, coordinates=coords.tolist()))
        if record is None:
            counts["unsupported"] += 1
            continue
        counts["supported"] += 1
        counts["point_pairs"] += len(coords)
        record["rle"] = cand["rle"]
        chosen = r.emit_candidate(c.np, stock, record, c.heads, p.fit_step, p.features, c.codec, original)
        verified_masks(c, stock, cand, coords, chosen, independent)
        for s, arms in chosen.items():
            if arms:
                selected[s][str(rank)] = arms
    for seed in HEADS:
        for arm in ARMS:
            require(dict(original[seed][arm]) == dict(independent[seed][arm]), "Independent raster budget readback")
    row = dict(ordinal=ordinal, image_id=image_id, group=block["source_prefix"], original_hw=[800,800],
               mapped_identity_sha256=r.mapped_identity_sha256(pool), points=points, seeds=selected)
    return r.encoded(row), counts, original, c.repair["_consumed"] - used_before


def capture_inputs(args):
    path = args.capture / "RESULT_SAFE.json"
    capture_raw = path.read_bytes()
    report = json.loads(capture_raw)
    require(report["status"] == "B5_RFDETR_CENTER5202_DEV705_CAPTURE_COMPLETE" and
            report["checkpoint_sha256"] == CHECKPOINT and report["historical_CUDA_stock_exact"] is True and
            report["model_forwards"] == report["images"] == 705 and report["native_mapped_rows"] == 67386 and
            report["GT_opened"] is report["AP"] is False and report["optimizer_updates"] == 0, "Completed fresh capture")
    pairs = ((args.capture / cap.QUERIES, "queries_sha256", cap.QUERIES_SHA),
             (args.capture / cap.STOCK, "stock_predictions_sha256", STOCK_SHA),
             (args.capture / "IMAGE_MANIFEST_PRIVATE.json", "image_manifest_sha256", cap.OUTPUT_MANIFEST),
             (args.intake / "DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json", "input_manifest_sha256", cap.INPUT_MANIFEST),
             (args.capture / cap.PROBABILITY, "native_probability_file_sha256", cap.REPAIR_FILE_SHA))
    for file, key, expected in pairs:
        require(cap.file_hash(file) == report[key] == expected, "Historical frozen input differs: " + file.name)
    payload = (args.capture / cap.REPAIR).read_bytes()
    require(cap.sha(payload) == report["repair_sha256"], "Regenerated native repair lineage")
    repair = json.loads(payload)
    require(all(repair[k] == v for k, v in cap.REPAIR_EXPECTED.items()) and
            repair["native_probability_ref"]["relative_path"] == cap.PROBABILITY and
            repair["native_probability_ref"]["sha256"] == cap.REPAIR_FILE_SHA and
            repair["source_logit_ref"]["relative_path"] == f"RAW_LOGITS/{repair['ordinal']:03d}.npy",
            "Original exact raw/sign/probability correction")
    for k in ("source_logit_ref", "native_probability_ref"):
        repair[k]["path"] = str(args.capture / repair[k]["relative_path"])
    repair["_consumed"] = 0
    return cap.sha(capture_raw), report["repair_sha256"], repair


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("archive", "intake", "capture", "heads", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--code", type=Path, default=HERE.parent)
    parser.add_argument("--workers", type=int, default=1, help="Independent single-thread CPU image workers")
    args = parser.parse_args()
    for k, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, k, value.resolve(strict=k != "output"))
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    require(sys.version_info[:3] == (3,10,13) and all(version(n) == v for n,v in
            {"torch":"2.10.0+cu130","numpy":"1.26.4","pycocotools":"2.0.11","Pillow":"12.3.0"}.items()),
            "Use the locked CPU point-replay runtime")
    require(args.workers > 0 and not args.output.exists(), "Use a positive worker count and fresh output")
    stamp, patch_sha, repair = capture_inputs(args)
    rows = json.loads((args.intake / "DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json").read_bytes())
    capture_rows = json.loads((args.capture / "IMAGE_MANIFEST_PRIVATE.json").read_bytes())
    require(len(rows) == len(capture_rows) == 705 and all(all(a[k] == b[k] for k in
            ("image_id","encoded_sha256","decoded_rgb_sha256")) for a,b in zip(rows, capture_rows, strict=True)),
            "Ordered source and capture manifest identities")
    args.output.mkdir()
    output = args.output / "SPARSE_EDITS_PRIVATE.jsonl"
    total, sums, repairs = Counter(), budgets(), 0
    digest = sha256()
    with (args.capture / cap.QUERIES).open("rb") as queries, (args.capture / cap.STOCK).open("rb") as native, \
            output.open("xb") as edits, ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                                            initargs=(args, repair)) as pool:
        tasks = ((i,r,q,n) for i,(r,q,n) in enumerate(zip(rows, queries, native, strict=True), 1))
        for line, counts, actual, used in pool.map(replay_image, tasks):
            edits.write(line)
            digest.update(line)
            total.update(counts)
            repairs += used
            for s in HEADS:
                for arm in ARMS:
                    sums[s][arm].update(actual[s][arm])
            if total["images"] == 1 or total["images"] % 25 == 0:
                print(json.dumps(dict(images=total["images"], mapped=total["mapped"])), flush=True)
    require(dict(total) == EXPECTED and repairs == 1, "Original complete native/eligible population and one CUDA correction")
    require(digest.hexdigest() == cap.file_hash(output) == EDITS_SHA, "Fresh canonical sparse edits differ from the historical bytes")
    for arms in sums.values():
        for v in arms.values():
            v["unchanged"] = total["mapped"] - v["edited"]
    result = dict(status="B5_RFDETR_CENTER5202_DEV705_REPLAY_COMPLETE", seeds=list(HEADS), arms=list(ARMS),
                  counts=dict(total), budgets=sums, images=705, native_mapped_rows=67386, edits_sha256=EDITS_SHA,
                  stock_predictions_sha256=STOCK_SHA, queries_sha256=cap.QUERIES_SHA, checkpoint_sha256=CHECKPOINT,
                  point_heads_sha256=HEADS, capture_result_sha256=stamp, probability_repair_sha256=patch_sha,
                  probability_repair_consumed=repairs, exact_candidate_identity=True, exact_native_support=True,
                  exact_per_candidate_budgets=True, actual_XOR_area_RLE_verified=True, exact_historical_sparse_bytes=True,
                  optimizer_updates=0, point_optimizer_updates=0, model_forward_calls=0, AP=False, GT_opened=False,
                  FIT=False, DEV=True, TEST=False, worker_sha256=cap.file_hash(Path(__file__)))
    path = args.output / "RESULT_SAFE.json"
    with path.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps(dict(status=result["status"], result_sha256=cap.file_hash(path))), flush=True)


if __name__ == "__main__":
    main()

