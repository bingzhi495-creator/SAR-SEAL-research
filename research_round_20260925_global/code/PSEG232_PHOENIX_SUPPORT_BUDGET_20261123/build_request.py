"""Prepare the label-free Phoenix transport request for the original PSeg-SSDD TEST232 cohort."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from PIL import Image
from pycocotools import mask as codec


REQUEST = "PSEG232_PHOENIX_SUPPORT_BUDGET_REQUEST"
LOCK = "5307bcbc32f94f11d27092945e1993692b6ef8347e85a2d3231080dae54c81df"
COMPARE = "633da1df76185bb32177422eba0c3b51f2c0b1a01931ce432b993641c10aa6c7"
ATTRACTION = "8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6"
HISTORY = {
    "prepare-edits": ("fae27e53ed3ca8253b8f1768e6d541525b2ea79dcb931d0edfe6c94ab4a837f3",
                      "cc2873bfe3a30bd1de1ad93e6c8d70125b05a998714371ecf65450ae851c69d0"),
    "verify-edits": ("6042fe2ef69c30c9ae4567c9deae7c0982425cc07576f79b7733259b0a1071b7",
                     "39ce2d6b0aefeef9a9447342e69d05a250c2cb61beafaaecb39298ad896b53aa"),
}
EDITS = "49d42e65eb6982aaa4d34a462b519d78dd9a2f884f3bdb95512585ad024ac4e5"
CENSUS = dict(images=232, mapped_candidates=23200, area_eligible=11209,
              positive_budget_candidates=11054, swaps_per_arm=73770)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write(path, data):
    with path.open("xb") as file:
        need(file.write(data) == len(data), "Incomplete output " + path.name)
    return digest(data)


def pack_candidate(position, native_rle, stock, coords, dynamic, static, arms, old, compare):
    k = len(dynamic)
    need(len(static) == k == compare.validate_extra_budget(arms), "Reconstructed dynamic quota differs from archive")
    if not k:
        return None
    need(coords.ndim == 2 and coords.shape[1] == 2 and coords.dtype.kind in "iu" and
         0 < k <= min(8, int(stock.sum()) // 10, len(coords)) and len(coords) <= 32 and
         np.all((0 <= coords) & (coords < stock.size)) and np.unique(coords).size == coords.size and
         not stock.flat[coords[:, 0]].any() and stock.flat[coords[:, 1]].all(), "Invalid native endpoint cells")
    for mode, selection in zip(compare.NEW_ARMS[:2], (dynamic, static), strict=True):
        need(old.edit(stock, coords, selection) == arms[mode]["rle"], "Reconstructed classical mask differs")
    learned = codec.decode(arms[compare.NEW_ARMS[2]]["rle"]).astype(bool)
    adds, drops = np.flatnonzero(learned & ~stock), np.flatnonzero(stock & ~learned)
    need(len(adds) == len(drops) == k and np.isin(adds, coords[:, 0]).all() and np.isin(drops, coords[:, 1]).all(),
         "Archived learned edit falls outside the same sampled signed support")
    return dict(position=position, segmentation=native_rle, k=k, coords=coords.tolist())


def transport_image(ordinal, image_id, rgb, candidates, expected_rgb):
    need(rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8 and
         digest(rgb.tobytes(order="C")) == expected_rgb, "Original decoded EXIF-transposed RGB required")
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    png = buf.getvalue()
    with Image.open(BytesIO(png)) as check:
        need(check.mode == "RGB" and check.format == "PNG" and check.size == rgb.shape[:2][::-1]
             and check.tobytes() == rgb.tobytes(order="C"), "Lossless native RGB round trip required")
    return dict(image_id=image_id, image_file=f"image_{ordinal:03d}.png", hw=list(rgb.shape[:2]),
                encoded_sha256=digest(png), decoded_rgb_sha256=expected_rgb, candidates=candidates), png


def admit(a):
    file = a.code / "PSEG_TEST232_CLASSICAL_ATTRACTION_20261122/compare.py"
    raw = file.read_bytes()
    need(digest(raw) == COMPARE, "Original classical admission source required")
    compare = ModuleType("historical_compare")
    compare.__file__ = str(file)
    exec(compile(raw, str(file), "exec"), compare.__dict__)
    replay = compare.load_module(a.code / "PSEG_TEST232_DFINE_FROZEN_REPLAY_20261121/replay.py", compare.REPLAY_SHA)
    readback = compare.document(a.original_readback, compare.READBACK_SHA)
    original = compare.document(a.original_replay / "RESULT_SAFE.json", readback["replay_sha256"])
    terminal = compare.document(a.owner / "REPLAY_TERMINAL_PRIVATE.json", readback["original_tool_wait_sha256"])
    replay.check_terminal(terminal, readback["replay_sha256"])
    need(readback["status"] == replay.VERIFIED and readback["exact_native_support"] is True and
         original["status"] == replay.PREPARED and original["edits_sha256"] == readback["edits_sha256"] and
         readback["GT_opened"] is original["GT_opened"] is False, "Closed original label-free replay required")
    a.manifest = a.owner / "TEST232_IMAGE_ONLY_MANIFEST_PRIVATE.json"
    a.accepted = a.owner / "CAPTURE_READBACK/INDEPENDENT_SAFE.json"
    a.images = a.owner / "IMAGE_ONLY"
    a.manifest_sha, a.accepted_sha = readback["manifest_sha256"], readback["acceptance_sha256"]
    modules = replay.modules(a.code, a)
    modules.author.cpu_runtime()
    roster, receipt = replay.admit(modules, a)
    need(receipt["probability_rows"] == CENSUS["mapped_candidates"] and
         receipt["predictions_sha256"] == readback["input_predictions_sha256"] and
         all(receipt[key] == readback[key] for key in ("candidate_audit_sha256", "probability_index_sha256")),
         "Original independent readback and capture disagree")
    compare.pinned(a.capture / "RESULT_SAFE.json", receipt["export_result_sha256"])
    compare.pinned(a.owner / "CAPTURE_READBACK/TERMINAL_PRIVATE.json", receipt["export_terminal_sha256"])
    compare.pinned(a.owner / "CAPTURE_READBACK/ORIGINAL_TOOL_WAIT_PRIVATE.json", receipt["original_tool_wait_sha256"])
    a.source_sha, a.phase_root = COMPARE, a.classical_root
    counts = dict(candidates=23200, extra_candidate_edits=11054, extra_swaps_per_arm=73770)
    for phase, status in (("prepare-edits", compare.PREPARED), ("verify-edits", compare.VERIFIED)):
        result_sha, terminal_sha = HISTORY[phase]
        compare.pinned(a.phase_root / ("observed_" + phase) / "TERMINAL.json", terminal_sha)
        compare.pinned(a.phase_root / phase / "RESULT_SAFE.json", result_sha)
        result = compare.closed(a, phase, status)
        need(result["counts"] == counts and result["edits_sha256"] == EDITS and result["GT_opened"] is False
             and result["original_readback_sha256"] == compare.READBACK_SHA, "Closed classical prelabel chain differs")
    lineage = dict(manifest_sha256=a.manifest_sha, acceptance_sha256=a.accepted_sha,
                   ema_sha256=receipt["ema_sha256"], **{key: receipt[key] for key in
                   ("predictions_sha256", "candidate_audit_sha256", "probability_index_sha256", "export_result_sha256",
                    "export_terminal_sha256", "original_tool_wait_sha256")})
    history = dict(classical_worker_sha256=COMPARE, edits_sha256=EDITS, prepare_result_sha256=HISTORY["prepare-edits"][0],
                   prepare_terminal_sha256=HISTORY["prepare-edits"][1], verify_result_sha256=HISTORY["verify-edits"][0],
                   verify_terminal_sha256=HISTORY["verify-edits"][1], original_readback_sha256=compare.READBACK_SHA,
                   original_edits_sha256=readback["edits_sha256"], original_replay_sha256=readback["replay_sha256"],
                   original_replay_terminal_sha256=readback["original_tool_wait_sha256"])
    old = compare.block_stream(a.original_replay / modules.legacy.EDITS, readback["edits_sha256"])
    extended = compare.block_stream(a.classical_root / "prepare-edits" / modules.legacy.EDITS, EDITS)
    need(len(old) == len(extended) == len(roster) == CENSUS["images"], "Complete source ledgers required")
    return compare, modules, roster, receipt, old, extended, lineage, history


def build_image(ordinal, block, earlier, archived, original_pair, rank, attraction, modules, compare, rgb):
    item, native, audit, planes = block
    image_id = item["image_id"]
    need(image_id == earlier["image_id"] == archived["image_id"] and list(rgb.shape[:2]) == item["original_hw"]
         and len(native) == len(audit) == len(planes) == 100, "Invalid image identity or top-100 native order")
    previous = compare.crosscheck(archived, earlier)
    features = original_pair.image_features(rgb)
    pair, probability = modules.native.make_native(original_pair, rank, *item["original_hw"])
    counts = Counter(images=1, mapped_candidates=len(native), area_eligible=0, endpoint_pairs=0)
    candidates = []
    for position, (det, info, plane) in enumerate(zip(native, audit, planes, strict=True), 1):
        arms = archived["changes"].get(str(position), {})
        area, rle = info["mask_area"], det["segmentation"]
        need(info["native_rank"] == position and info["score"] == det["score"] and rle["size"] == item["original_hw"]
             and int(codec.area(rle)) == area, "Native rank, score, size or area differs")
        if not 32 <= area < 1024:
            need(not arms, "Unexpected archived edits on an area-ineligible mask")
            continue
        counts["area_eligible"] += 1
        stock = codec.decode(rle).astype(bool, order="C")
        data = pair.paired_points(stock, info["box_xyxy"], area, det["score"], plane, features, probability)
        if data is None:
            need(not arms, "Nonempty archived decisions with no signed support")
            continue
        coords = data[0]
        dynamic, static = attraction.select(stock, coords, min(8, area // 10, len(coords)))
        entry = pack_candidate(position, rle, stock, coords, dynamic, static, arms, modules.old, compare)
        if entry is not None:
            candidates.append(entry)
            counts["endpoint_pairs"] += len(coords)
    counts["positive_budget_candidates"] = len(candidates)
    counts["swaps_per_arm"] = sum(row["k"] for row in candidates)
    need(len(candidates) == previous["extra_candidate_edits"] and counts["swaps_per_arm"] == previous["extra_swaps_per_arm"],
         "Image-specific positive candidate or swap census differs from independent replay")
    image, png = transport_image(ordinal, image_id, rgb, candidates, archived["rgb_sha256"])
    return image, png, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "owner", "capture", "original-replay", "original-readback", "classical-root", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    a = parser.parse_args()
    for name, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, name, value.resolve(strict=name != "output"))
    need(sys.flags.isolated and sys.dont_write_bytecode and not a.output.exists(), "Isolated runtime and fresh output required")
    need(a.protocol.name == "PSEG232_PHOENIX_SUPPORT_BUDGET_PRELOCK_20261123.md" and
         digest(a.protocol.read_bytes()) == a.protocol_sha256 == LOCK, "Exact PSeg factorial protocol required")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    started = perf_counter()
    compare, m, roster, receipt, old, extended, lineage, history = admit(a)
    pair, rank = m.native.load_old(a.pair_math, a.rank_math)
    attraction = compare.load_module(a.code / "FIT109_CLASSICAL_ATTRACTION_20261121/spatial_attraction.py", ATTRACTION)
    a.output.mkdir()
    images, counts = [], Counter()
    blocks = m.legacy.blocks(a, m.author, roster, receipt, True)
    for j, (block, before, after) in enumerate(zip(blocks, old, extended, strict=True), 1):
        rgb = m.legacy.rgb(a, block[0])
        row, png, image_counts = build_image(j, block, before, after, pair, rank, attraction, m, compare, rgb)
        need(write(a.output / row["image_file"], png) == row["encoded_sha256"], "Transport bytes differ")
        images.append(row)
        counts.update(image_counts)
        if j == 1 or j % 25 == 0 or j == CENSUS["images"]:
            print(json.dumps(dict(images=j, positive_budget_candidates=counts["positive_budget_candidates"])), flush=True)
    need(all(counts[k] == v for k, v in CENSUS.items()) and
         counts["swaps_per_arm"] <= counts["endpoint_pairs"] <= 32 * counts["positive_budget_candidates"],
         "Official PSeg image, support, and realized budget census required")
    payload = dict(status=REQUEST, protocol_sha256=LOCK, input_lineage=lineage, source_replay=history, images=images)
    request_sha = write(a.output / "REQUEST_PRIVATE.json", encode(payload))
    result = dict(status=REQUEST + "_COMPLETE", phase="request", protocol_sha256=LOCK, input_lineage=lineage,
                  source_replay=history, counts=dict(counts), request_sha256=request_sha, target_labels_opened=False,
                  detector_forwards=0, point_head_forwards=0, optimizer_updates=0, worker_sha256=digest(Path(__file__).read_bytes()),
                  seconds=perf_counter() - started)
    result_sha = write(a.output / "RESULT_SAFE.json", encode(result))
    print(json.dumps(dict(status=result["status"], result_sha256=result_sha)), flush=True)


if __name__ == "__main__":
    main()
