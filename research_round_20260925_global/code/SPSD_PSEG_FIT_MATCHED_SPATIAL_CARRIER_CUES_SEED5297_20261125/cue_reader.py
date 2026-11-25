"""Verify the actual closed feature intake and join only original FIT endpoints."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np


POINTS = "87693897f873a2129764781611b0f7c35cadde51c0c4e830d7f8187c3b684049"
STATUS = "SPSD_PSEG_TRAIN915_DFINE_SPATIAL_CUES_COMPLETE"
VERIFY_STATUS = "SPSD_PSEG_TRAIN915_DFINE_SPATIAL_CUES_READBACK_COMPLETE"
NATIVE = "6f58ba8b486950431f4aebebf237e60029b447e5ec21a7f51ad2202d065f904a"
NATIVE_READBACK = "e461828b04f166d98030210fe42df0d590d24e2452ea5da8630fb0fc3219d795"
CAPTURE_PROTOCOL = "7f0fb2c8ee98f0dcb784dc356b5a118fe926ef40dabc28f08c306d36f61da1f8"
CAPTURE_WORKER = "f63cacf94e67bd1c627fe78ee000f05e98a3d7df8cf1c34c88089f63948387e3"
LOCK_STATUS = "FIT_SPATIAL_CUE_ACQUISITION_LOCKED"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    need(isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected), "Well-formed closure digest")
    raw = path.read_bytes()
    need(digest(raw) == expected, "Exact admitted artifact " + path.name)
    return raw


def closed(result, observed, requirement, worker):
    need(observed["actual_exit"] == 0 and observed["output_footer_verified"] is True
         and observed["result_sha256"] == requirement
         and observed["worker_sha256"] == worker and result["worker_sha256"] == worker,
         "Bound natural observer exit and footer")


def admitted_cues(a):
    receipt = json.loads(pin(a.cue_lock, a.cue_lock_sha))
    need(receipt["status"] == LOCK_STATUS and receipt["source_gate_labels_opened"] is False
         and receipt["historical_test_opened"] is False and receipt["point_result_sha256"] == POINTS
         and receipt["cue_capture_worker_sha256"] == receipt["cue_capture_readback_worker_sha256"] == CAPTURE_WORKER,
         "Exact admitted independent acquisition scope")
    capture = a.cue_capture
    results = json.loads(pin(capture/"actual/RESULT_SAFE.json", receipt["cue_capture_result_sha256"]))
    term = json.loads(pin(capture/"observed_capture_active/TERMINAL.json", receipt["cue_capture_terminal_sha256"]))
    back = json.loads(pin(capture/"readback/RESULT_SAFE.json", receipt["cue_capture_readback_sha256"]))
    end = json.loads(pin(capture/"observed_readback/TERMINAL.json", receipt["cue_capture_readback_terminal_sha256"]))
    closed(results, term, receipt["cue_capture_result_sha256"], receipt["cue_capture_worker_sha256"])
    closed(back, end, receipt["cue_capture_readback_sha256"], receipt["cue_capture_readback_worker_sha256"])
    need(results["status"] == STATUS and back["status"] == VERIFY_STATUS
         and term["phase"] == "capture" and end["phase"] == "verify"
         and results["partitions"] == back["partitions"]
         and results["protocol_sha256"] == back["protocol_sha256"] == CAPTURE_PROTOCOL
         and results["partitions"]["fit"]["cue_index_sha256"] == receipt["cue_fit_index_sha256"]
         and results["projection_seed"] == 5297 and results["cue_dimensions"] == 6
         and results["spatial_feature_shape"] == [1,256,160,160] and results["detector_forwards"] == 915
         and back["detector_forwards"] == 0 and back["integrity_images"] == 915
         and back["spatial_capture_result_sha256"] == receipt["cue_capture_result_sha256"]
         and back["spatial_capture_terminal_sha256"] == receipt["cue_capture_terminal_sha256"]
         and all(doc["point_result_sha256"] == POINTS and doc["capture_result_sha256"] == NATIVE
                 and doc["capture_readback_sha256"] == NATIVE_READBACK and doc["optimizer_updates"] == 0
                 and all(doc[k] is False for k in ("AP", "train_labels_opened", "test_labels_opened", "test_images_opened"))
                 for doc in (results, back)),
         "Capture and independent readback exact independently admitted FIT scope")
    original = json.loads(pin(a.points/"actual_retry/RESULT_SAFE.json", POINTS))
    oldp = a.points/"actual_retry/fit"
    points = [json.loads(line) for line in pin(oldp/"POINT_INDEX_PRIVATE.jsonl",
                 original["partitions"]["fit"]["point_index_sha256"]).splitlines()]
    cues = [json.loads(line) for line in pin(capture/"actual/fit/CUE_INDEX_PRIVATE.jsonl",
                 receipt["cue_fit_index_sha256"]).splitlines()]
    fields = ("native_archive_ordinal", "image_id", "source_group", "original_hw", "rgb_sha256",
              "supported_candidates", "paired_sites", "realized_static_exchanges")
    rows = {}
    need(len(points) == len(cues) == 732, "All original FIT image indices")
    for point, cue in zip(points, cues, strict=True):
        ordinal = point["native_archive_ordinal"]
        need(all(point[k] == cue[k] for k in fields)
             and cue["point_sha256"] == point["sha256"] and ordinal not in rows
             and point["filename"] == f"POINTS_PRIVATE/{ordinal:04}.npz"
             and cue["filename"] == f"CUES_PRIVATE/{ordinal:04}.npz",
             "Exact point and cue image identity, order and counts")
        rows[ordinal] = (point, cue)
    return rows, receipt


def image_arrays(old_root, cue_root, point, cue):
    keep = ("native_ranks", "point_offsets", "signed_coordinates")
    raw_point = pin(old_root/point["filename"], point["sha256"])
    raw_cue = pin(cue_root/cue["filename"], cue["sha256"])
    need(len(raw_point) == point["bytes"] and len(raw_cue) == cue["bytes"], "Exact NPZ lengths")
    with np.load(BytesIO(raw_point), allow_pickle=False) as source, np.load(BytesIO(raw_cue), allow_pickle=False) as added:
        need(all(np.array_equal(source[k], added[k]) and source[k].dtype == added[k].dtype for k in keep),
             "No native rank, offset, coordinate or dtype changes")
        ordinal = added["native_archive_ordinal"].item()
        cue_point_sha = added["point_sha256"].item()
        if isinstance(cue_point_sha, bytes):
            cue_point_sha = cue_point_sha.decode("ascii")
        feature, static, projection = source["features"], source["static_offsets"], added["cue"]
        ranks, offsets, coords = (source[k] for k in keep)
        n, p = len(ranks), len(coords)
        need(ordinal == point["native_archive_ordinal"] and cue_point_sha == point["sha256"]
             and len(set(ranks)) == n == point["supported_candidates"] and p == point["paired_sites"]
             and offsets.shape == static.shape == (n+1,) and offsets[0] == 0 and offsets[-1] == p
             and coords.shape == (p,2) and feature.shape == (p,2,14)
             and projection.shape == (p,2,6) and projection.dtype == np.float32 and np.isfinite(projection).all(),
             "Exact signed-cue preparation and original point binding")
    index = {int(rank):i for i, rank in enumerate(ranks)}
    return index, offsets, feature, static, projection


def join_fit(a, admission, bank, incumbent):
    rows, lock = admission
    old_root = a.points/"actual_retry/fit"
    cue_root = a.cue_capture/"actual/fit"
    cache = {}
    used = set()
    arrays = []
    digested = sha256()
    for packet, (key, _) in zip(bank, incumbent, strict=True):
        ordinal, rank = (int(i) for i in key.split("|"))
        need(key not in used, "Unique original FIT instance")
        used.add(key)
        point, cue = rows[ordinal]
        need(point["image_id"] == packet.image_id, "Exact native FIT image identity")
        if ordinal not in cache:
            cache[ordinal] = image_arrays(old_root, cue_root, point, cue)
        index, offsets, feature, static, projection = cache[ordinal]
        j = index[rank]
        start, stop = offsets[j:j+2]
        vector = projection[start:stop].copy()
        need(np.array_equal(feature[start:stop], packet.features) and len(vector) == len(packet.features)
             and int(static[j+1] - static[j]) == packet.k, "Exact original per-candidate raw features and k")
        rowsn = len(vector)
        digested.update(f"{ordinal}|{rank}|{rowsn}\n".encode("ascii"))
        digested.update(vector.astype("<f4",copy=False).tobytes(order="C"))
        arrays.append((vector, ordinal, rank))
    need(len(arrays) == 848 and len(cache) == 447, "Complete original labeled FIT fit roster joined")
    keys = ("cue_capture_result_sha256", "cue_capture_terminal_sha256", "cue_capture_readback_sha256",
            "cue_fit_index_sha256")
    receipt = {k:lock[k] for k in keys}
    receipt.update(cue_acquisition_lock_sha256=a.cue_lock_sha, cue_reader_sha256=a.cue_reader_sha,
                   cue_joined_candidates=len(arrays), cue_joined_images=len(cache),
                   cue_fit_vector_payload_sha256=digested.hexdigest())
    return arrays, receipt
