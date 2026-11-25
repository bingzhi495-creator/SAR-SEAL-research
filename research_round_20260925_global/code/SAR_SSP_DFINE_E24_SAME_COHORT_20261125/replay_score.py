"""Independent readback, exact edit replay and final author-raster scoring."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

PRELOCK = "6dfdffcbe0e29ef0c28b82870d84f16852f4b3d23629d8baf67d858077c4ea51"
CAPTURE = "0e15ca971ac2a6f9318b16710977ea5b86961c233a5effeca5649d348bc84155"
CAPTURE_WORKER = "03699208afeea40b4cfe9ec130fc4b5b931144b6d634d0fa4380c5e9ba44c798"
REPLAY = "f1b6e83e98851bf867bd72f3ecb082ca2997632bdd1aedaaf64110cf06b04fd5"
ATTRACTION_ADAPTER = "782d6b12c5d22d933827d05e78feb7bdc9864224bfe55a190fed117f706d94bd"
SCORE = "d4d16cdfab0903480c7e6f3686de973886b726d54f91b7e235686c1502d44735"
ROOT = Path(__file__).parent


def need(ok, message):
    if not ok:
        raise ValueError(message)


def file_hash(path):
    with path.open("rb") as stream:
        return __import__("hashlib").file_digest(stream, "sha256").hexdigest()


def pin(path, digest):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, f"Exact bound file: {path.name}")
    return raw


def load(path, digest, key):
    m = ModuleType(key)
    m.__file__ = str(path)
    exec(compile(pin(path, digest), str(path), "exec"), m.__dict__)
    return m


def terminal_at(phase, result_sha):
    t = json.loads((ROOT / (phase.upper() + "_TERMINAL.json")).read_bytes())
    need(t["phase"] == phase and t["exit_code"] == 0 and t["footer_verified"] is True
         and t["result_sha256"] == result_sha and t["protocol_sha256"] == PRELOCK
         and type(t["original_tool_handle"]) is int and t["original_tool_handle"] > 0, "Observed terminal and footer")
    return t


def configure(args):
    pin(ROOT / "PRELOCK.md", PRELOCK)
    capture = json.loads(pin(ROOT / "capture/RESULT_SAFE.json", CAPTURE))
    terminal_at("capture", CAPTURE)
    args.capture = ROOT / "capture"
    sys.path.insert(0, str(args.capture / "CODEC_PRIVATE"))
    original = load(ROOT / "capture.py", CAPTURE_WORKER, "_intake")
    replay = load(args.code / "PSEG_TEST232_DFINE_FROZEN_REPLAY_20261121/replay.py", REPLAY, "_legacy_transfer")
    m = replay.modules(args.code, args)
    m.author.cpu_runtime()
    roster = original.validate_manifest(args.manifest, original.MANIFEST, m.author)
    need(capture["protocol_sha256"] == PRELOCK
         and capture["export_source_sha256"] == original.CAPTURE_SOURCE
         and capture["manifest_sha256"] == original.MANIFEST
         and capture["ema_sha256"] == replay.EMA
         and capture["status"] == "DFINE_HRSID_E24_SAR_SSP_VAL232_PROBABILITY_EXPORT_COMPLETE_V1"
         and capture["images"] == capture["model_forwards"] == capture["probability_files"] == 232
         and capture["rows_at_100"] == capture["probability_rows"] == capture["exact_top100_binary_replay"] == 23200
         and capture["post_nms_rows"] == capture["exact_author_binary_replay"]
         and capture["original_author_postprocess_parity"] is True
         and capture["GT_opened"] is capture["AP"] is False and capture["optimizer_updates"] == 0,
         "Qualified actual D-FINE capture")
    return m, replay, roster, capture


def replay_all(args, m, legacy, roster, capture):
    import torch
    adapter = load(args.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/ueikap_transfer.py",
                   ATTRACTION_ADAPTER, "_exact_classical")
    adapter.extended_edits(args, m)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    models = m.old.restore(args)
    models[0].raw = m.legacy.pin(args.pair_math, m.native.PAIR)
    counts, support_counts = Counter(), Counter()
    digest = sha256()
    phases = (("fixed", "free_matched_count"), ("free_natural", "native_matched_count"), tuple(adapter.ARM_EXTRA))
    verify = args.phase == "verify"
    prepared = read_phase("prepare", args.prior_sha) if verify else None
    edit_path = ROOT / "prepare/EDITS_PRIVATE.jsonl"
    if not verify:
        edit_path.parent.mkdir()
    with edit_path.open("rb" if verify else "xb") as file:
        for ordinal, (item, coco, audit, planes) in enumerate(
                m.legacy.blocks(args, m.author, roster, capture, True), 1):
            image = m.legacy.rgb(args, item)
            row = legacy.emit(m, args, item, coco, audit, planes, image, models, counts)
            need(row["rgb_sha256"] == item["rgb_sha256"], "Exact closed image and candidate stream join")
            for family in phases:
                adapter.same_count(row["changes"], family)
            line = m.author.encoded(row)
            if verify:
                need(file.readline(4 * 1024**2) == line, "Independent byte parity for every replayed candidate")
                legacy.check_support(models[1], coco, audit, row["changes"], support_counts)
            else:
                need(file.write(line) == len(line), "Complete edit line")
            digest.update(line)
            counts["candidates"] += len(coco)
            if ordinal % 50 == 0 or ordinal == 232:
                print(json.dumps(dict(phase=args.phase, images=ordinal)), flush=True)
        if verify:
            need(not file.read(1) and prepared["counts"] == dict(counts) and
                 prepared["edits_sha256"] == digest.hexdigest()
                 and all(counts[k] == v for k, v in support_counts.items()), "Independent stream and support parity")
        else:
            file.flush()
            os.fsync(file.fileno())
    need(ordinal == 232 and counts["candidates"] == 23200
         and counts["area_eligible"] == counts["zero_support"] + counts["supported"]
         and all(len({counts[arm + "_swaps"] for arm in family}) == 1 for family in phases),
         "Full image and exact matched budgets")
    return dict(counts=dict(counts), edits_sha256=digest.hexdigest(),
                head_acceptance_sha256=m.old.REFIT_ACCEPTED,
                independent_replay=verify, target_labels_opened=False,
                original_author_readback=True, support_verified=verify)


def read_phase(phase, checksum):
    result = json.loads(pin(ROOT / phase / "RESULT_SAFE.json", checksum))
    terminal_at(phase, checksum)
    need(result["phase"] == phase and result["capture_sha256"] == CAPTURE
         and result["worker_sha256"] == file_hash(Path(__file__))
         and result["protocol_sha256"] == PRELOCK and result["target_labels_opened"] is False
         and result["optimizer_updates"] == result["model_forwards"] == 0, "Exact observed phase scope")
    return result


def score(args, m, roster, capture):
    from pycocotools import mask as codec
    scored = load(args.code / "SAR_SSP_AIR_FROZEN_20261125/score_sarssp.py", SCORE, "_shared_sar_score")
    verified = read_phase("verify", args.prior_sha)
    need(verified["independent_replay"] and verified["support_verified"], "Closed verified label-free edits")
    records, boxes, names = {}, {}, {}
    edits = scored.read_stream(ROOT / "prepare/EDITS_PRIVATE.jsonl", verified["edits_sha256"])
    for (item, coco, audit, _), row in zip(m.legacy.blocks(args, m.author, roster, capture, False), edits, strict=True):
        image_id = item["image_id"]
        need(row["image_id"] == image_id and row["rgb_sha256"] == item["rgb_sha256"], "Exact scoring order")
        path = item["image_ref"]
        names[image_id] = (path["path"], path["sha256"])
        ledger = row["changes"]
        records[image_id], boxes[image_id] = [], []
        for position, (prediction, original) in enumerate(zip(coco, audit, strict=True), 1):
            stock = prediction["segmentation"]
            masks, counts = {}, {}
            for label, data in ledger.get(str(position), {}).items():
                arm = scored.ARMS[label]
                mask, k = data["rle"], data["swaps"]
                area = original["mask_area"]
                need(scored.signed_counts(stock, mask, area) == (k, k), "Independent actual signed count")
                masks[arm], counts[arm] = mask, k
            for family in scored.FAMILIES:
                need(len({counts.get(arm, 0) for arm in family}) == 1, "Same candidate scoring count")
            records[image_id].append(dict(native=stock, score=prediction["score"], masks=masks, counts=counts))
            a, b, c, d = original["box_xyxy"]
            boxes[image_id].append([a, b, max(0., c - a), max(0., d - b)])
    need(len(records) == 232 and sum(map(len, records.values())) == 23200, "All captured candidates scored")
    print("Verified D-FINE label-free edit stream before author-label access.", flush=True)
    need(file_hash(args.archive) == scored.ARCHIVE_SHA, "Identical author archive")
    with ZipFile(args.archive) as package:
        truth, groups, edges = scored.exact_ground_truth(package, names)
    evaluate = scored.load_evaluator(args.code / scored.EVALUATOR_PATH)
    result = scored.score(truth, groups, edges, records, evaluate)
    native_ap = evaluate(truth, scored.coco_rows(records, "native"))["AP"]
    native = scored.native_matches(truth, records, native_ap)
    bbox_cover = Counter()
    for row in truth["annotations"]:
        image_id = row["image_id"]
        if row["id"] in native:
            bbox_cover["native_matched"] += 1
        else:
            bbox_cover["native_unmatched"] += 1
        values = codec.iou(boxes[image_id], [row["bbox"]], [0])
        for threshold in (0.25, 0.50, 0.75):
            if (values >= threshold).any():
                bbox_cover[f"all_at_{threshold:.2f}"] += 1
                if row["id"] not in native:
                    bbox_cover[f"mask_unmatched_at_{threshold:.2f}"] += 1
    return dict(result, target_labels_opened=True, native_top100_bbox_coverage=dict(bbox_cover),
                prior_verify_sha256=args.prior_sha, counts=verified["counts"], fixed_head=True,
                same_cohort_second_carrier_sensitivity=True, newly_prospective=False)


def main():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("phase", choices=("prepare", "verify", "score"))
    for key in ("code", "manifest", "images", "refit-root"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--prior-sha")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
         "Fresh isolated CPU-only phase")
    def offline(event, _):
        if event.startswith("socket."):
            raise PermissionError("Offline frozen evaluation")
    sys.addaudithook(offline)
    result_dir = ROOT / args.phase
    need(not result_dir.exists(), "Fresh phase destination")
    for key, suffix in (("refit_result", "ACTUAL/RESULT_SAFE.json"),
                        ("refit_acceptance", "INDEPENDENT/RESULT_SAFE.json"),
                        ("weights", "ACTUAL/POINT_HEAD_PRIVATE.npz")):
        setattr(args, key, args.refit_root / suffix)
    started = perf_counter()
    m, replay, roster, capture = configure(args)
    outcome = (score(args, m, roster, capture) if args.phase == "score"
               else replay_all(args, m, replay, roster, capture))
    result_dir.mkdir(exist_ok=args.phase == "prepare")
    result = dict(outcome, phase=args.phase, elapsed_seconds=perf_counter()-started,
                  model_forwards=0, optimizer_updates=0, protocol_sha256=PRELOCK, capture_sha256=CAPTURE,
                  worker_sha256=file_hash(Path(__file__)))
    digest = m.author.write_new(result_dir / "RESULT_SAFE.json", result)
    print(json.dumps(dict(phase=args.phase, result_sha256=digest), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
