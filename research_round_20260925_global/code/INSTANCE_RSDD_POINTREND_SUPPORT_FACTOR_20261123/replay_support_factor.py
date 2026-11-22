"""Splice closed PointRend masks on and off the frozen native endpoint sets."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
from struct import pack
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec


PREPARED = "INSTANCE_RSDD_POINTREND_SUPPORT_FACTOR_PREPARED_V1"
ACCEPTED = "INSTANCE_RSDD_POINTREND_SUPPORT_FACTOR_ACCEPTED_V1"
PRELOCK_SHA = "dab26421fd6647d5f329efa2739a185e0373f115d3a2cb387a24ecf5e09ca7f6"
PRELOCK_COMMIT = "b5c45220977e5b5b85e7ac5e107deaae53b020ac"
SOURCE_WORKER = "9368956a7d47e502c005a6adc1ab39edaca18cd5b3780002073ba672732a2b2f"
SOURCE_CAPTURE = "476a8ff4a9c705df3fa6530fca2c4f58d020dc7995b444480476eeac1e98b8a1"
SOURCE_RESULT = "ed6e7d48b5f5ec0ba9a98f37327ed4cd075a95c5ffc3857167a79da53f6a0cbd"
SOURCE_TERMINAL = "2dcbecab56ac7dfab07c91f47302ee4f4c2dd5e70bd99ca3069272123cc5b848"
SOURCE_EDITS = "b32ef296dcaaf6253662048d8957091cc3a1a93d611abca2e3605d24a85ed7da"
ARMS = ("p_on_support", "p_off_support")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def factor(native, balanced, paste, adds, drops):
    """Return P on S/B elsewhere and B on S/P elsewhere."""
    need(native.shape == balanced.shape == paste.shape and native.ndim == 2
         and native.dtype == balanced.dtype == paste.dtype == np.bool_, "One common native boolean raster")
    for side in (adds, drops):
        need(side.ndim == 1 and side.dtype.kind in "iu" and len(side) <= 32
             and np.all((side >= 0) & (side < native.size))
             and len(np.unique(side)) == len(side), "Unique valid endpoints on each native side")
    need(len(adds) == len(drops) and not native.flat[adds].any() and native.flat[drops].all(),
         "Equal disjoint endpoint tables with correct native signs")
    on = balanced.copy(order="C")
    off = paste.copy(order="C")
    support = np.concatenate((adds, drops))
    on.flat[support] = paste.flat[support]
    off.flat[support] = balanced.flat[support]
    outside = np.ones(native.size, dtype=bool)
    outside[support] = False
    need(np.array_equal(balanced.flat[outside], native.flat[outside]),
         "Balanced arm edits only the original native support")
    on_change, off_change = on ^ balanced, off ^ balanced
    need(not (on_change & off_change).any() and np.array_equal(on_change | off_change, paste ^ balanced),
         "Disjoint masked updates exactly recompose the PointRend paste")
    return on, off


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, f"Exact frozen bytes for {path.name}")
    return raw


def module(path, expected):
    source = ModuleType("_support_factor_" + path.stem)
    source.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), source.__dict__)
    return source


def dependencies(code):
    path = code / "INSTANCE_RSDD_POINTREND_TRANSFER_20261122"
    return module(path / "replay_pointrend.py", SOURCE_WORKER), module(path / "capture_basis.py", SOURCE_CAPTURE)


def authority(payload, independent):
    paths = {"code", "protocol", "roster", "capture", "capture_receipt", "capture_terminal", "source_replay"}
    stamps = {"protocol_sha"}
    if independent:
        paths.add("prepare")
        stamps |= {"prepare_terminal_sha", "prepare_result_sha"}
    need(set(payload) == paths | stamps | {"workers", "prelock_commit"}
         and type(payload["workers"]) is int and 1 <= payload["workers"] <= 8
         and all(Path(payload[k]).is_absolute() for k in paths)
         and all(re.fullmatch("[0-9a-f]{64}", payload[k]) for k in stamps)
         and payload["prelock_commit"] == PRELOCK_COMMIT and payload["protocol_sha"] == PRELOCK_SHA,
         "Exact authorized phase inputs")
    result = SimpleNamespace(**{k: Path(v).resolve(strict=True) if k in paths else v for k, v in payload.items()})
    pin(result.protocol, result.protocol_sha)
    return result


def read_population(a, prior, capture):
    _, native, edit = capture.input_modules(a.code)
    a.capture_terminal_sha = capture.CAPTURE_TERMINAL
    a.capture_receipt_sha = capture.CAPTURE_RECEIPT
    rows, orig = native.read_capture(a, edit)
    source, terminal = prior.closed(a.source_replay / "observed_verify-edits/TERMINAL.json", SOURCE_TERMINAL,
                                    a.source_replay / "verify-edits/RESULT_SAFE.json", SOURCE_RESULT)
    need(source["status"] == prior.ACCEPTED and terminal["phase"] == "verify-edits"
         and source["worker_sha256"] == terminal["worker_sha256"] == SOURCE_WORKER
         and source["images"] == 740 and source["GT_opened"] is False
         and source["exact_replayed_policy_choices"] is source["exact_support_xor_area"] is True
         and source["edits_sha256"] == SOURCE_EDITS and source["protocol_sha256"] == capture.LOCK_SHA
         and source["capture_receipt_sha256"] == a.capture_receipt_sha
         and source["predictions_sha256"] == orig["predictions_sha256"]
         and source["candidate_audit_sha256"] == orig["audit_sha256"],
         "Closed independently verified PointRend edits on the exact native roster")
    changes = edit.load_stream(a.source_replay / "prepare-edits/EDITS_PRIVATE.jsonl", SOURCE_EDITS)
    need(len(rows) == len(changes) == 740, "All author test rows must remain in scope")
    return list(zip(rows, changes, strict=True)), orig, source


def initialize(a):
    global worker
    prior, capture = dependencies(a.code)
    prior.install_guard()
    import torch
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_initialized()
         and sys.version_info[:3] == (3, 10, 13) and np.__version__ == "1.26.4"
         and torch.__version__ == "2.10.0+cu130" and version("pycocotools") == "2.0.11",
         "Qualified archived CPU support reconstruction")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    _, _, edit = capture.input_modules(a.code)
    base = edit.load_module(a.code / edit.BASE / "replay.py", edit.REPLAY_SHA)
    worker = SimpleNamespace(capture=a.capture, base=base, rank=base.methods(a.code).rank)


def decode(rle, shape):
    need(set(rle) == {"size", "counts"} and rle["size"] == list(shape) and isinstance(rle["counts"], str),
         "Canonical native RLE")
    return codec.decode(rle).astype(bool)


def encode(bitmap):
    rle = codec.encode(np.asfortranarray(bitmap, dtype=np.uint8))
    return dict(size=rle["size"], counts=rle["counts"].decode("ascii"))


def tally_change(counts, name, native, edited, expected):
    add, drop = int(np.count_nonzero(edited & ~native)), int(np.count_nonzero(native & ~edited))
    need((add, drop) == expected, "Exact signed actual changes relative to native mask")
    counts[name + "_adds"] += add
    counts[name + "_drops"] += drop
    counts[name + "_candidates"] += int(add + drop > 0)
    counts[name + "_empty"] += int(add + drop > 0 and not edited.any())
    return add + drop


def replay_image(work):
    (row, native), past_row = work
    shape = tuple(row["original_hw"])
    image_id = row["image_id"]
    old = past_row["changes"]
    need(shape == (512, 512) and native["image_id"] == past_row["image_id"] == image_id
         and not row["posterior_overrides"]
         and set(old).issubset({str(idx) for idx in range(1, len(row["candidates"])+1)}),
         "Exact author image, native canvas and compact edit indices")
    planes = worker.base.load_array(worker.capture, row["logits_ref"])
    logits = dict(zip(row["logit_ranks"], planes, strict=True))
    need(planes.shape == (len(logits), 144, 144), "Archived eligible query planes")
    counts = Counter(images=1, mapped_candidates=len(row["candidates"]), area_eligible=len(logits))
    changes = {}
    support_seal = sha256(pack("<I", image_id))
    for pos, (audit, pred) in enumerate(zip(row["candidates"], native["coco"], strict=True), 1):
        key = str(pos)
        rank = audit["native_rank"]
        raw_rle = pred["segmentation"]
        area = int(codec.area(raw_rle))
        need(audit["position"] == pos and pred["score"] == audit["score"] and pred["image_id"] == image_id
             and rank == row["native_ranks"][pos-1] and audit["query_id"] == row["query_ids"][pos-1]
             and area == audit["mask_area"] and (rank in logits) == (32 <= area < 1024),
             "Exact original candidate identity, score, size and gate")
        previous = old.get(key, {})
        need(set(previous) <= {"point_paste", "coarse_paste", "point_equal", "learned_common"},
             "Only closed PointRend transfer arms")
        box = np.asarray(audit["box_xyxy"], dtype=np.float32)
        need(box.shape == (4,) and np.isfinite(box).all() and np.all((0 <= box) & (box <= 512))
             and np.all(box[:2] <= box[2:]), "Exact archived clipped native box")
        degenerate = np.any(box[:2] == box[2:])
        if rank not in logits or degenerate:
            need(not previous, "Excluded original-mask fallback")
            counts["degenerate"] += int(rank in logits and degenerate)
            continue
        counts["rendered"] += 1
        stock = decode(raw_rle, shape)
        probability = worker.base.native_probability(logits[rank], stock, None)
        adds, drops = worker.base.expected_support(worker.rank, stock, box, probability)
        support_seal.update(pack("<II", rank, len(adds)))
        for side in (adds, drops):
            support_seal.update(np.asarray(side, dtype="<u4").tobytes())
        counts["endpoints_add"] += len(adds)
        counts["endpoints_drop"] += len(drops)
        counts["editable_supports"] += int(bool(len(adds)))
        point_paste = previous.get("point_paste", {})
        point_equal = previous.get("point_equal", {})
        swaps = point_equal.get("swaps", 0)
        need(type(swaps) is int and 0 <= swaps <= min(8, area//10, len(adds))
             and (swaps > 0) == ("point_equal" in previous), "Exact archived positive balanced budget")
        paste = decode(point_paste["rle"], shape) if point_paste else stock
        balanced = decode(point_equal["rle"], shape) if point_equal else stock
        on, off = factor(stock, balanced, paste, adds, drops)
        paste_adds, paste_drops = (point_paste.get(name, 0) for name in ("adds", "drops"))
        tally_change(counts, "point_equal", stock, balanced, (swaps, swaps))
        tally_change(counts, "point_paste", stock, paste, (paste_adds, paste_drops))
        counts["point_equal_swaps"] += swaps
        on_adds = int(paste.flat[adds].sum())
        on_drops = int((~paste.flat[drops]).sum())
        expected = ((on_adds, on_drops), (swaps+paste_adds-on_adds, swaps+paste_drops-on_drops))
        counts["paste_off_support_adds"] += paste_adds-on_adds
        counts["paste_off_support_drops"] += paste_drops-on_drops
        support = np.concatenate((adds, drops))
        p_side, b_side = paste.flat[support], balanced.flat[support]
        on_vs_b = (int((p_side & ~b_side).sum()), int((~p_side & b_side).sum()))
        tally_change(counts, "p_on_support_vs_b", balanced, on, on_vs_b)
        tally_change(counts, "p_off_support_vs_b", balanced, off, (paste_adds-on_adds, paste_drops-on_drops))
        tally_change(counts, "p_vs_b", balanced, paste,
                     (on_vs_b[0]+paste_adds-on_adds, on_vs_b[1]+paste_drops-on_drops))
        record = {}
        for name, edited, signed in zip(ARMS, (on, off), expected, strict=True):
            if tally_change(counts, name, stock, edited, signed):
                record[name] = {"rle": encode(edited)}
        if record:
            changes[key] = record
    return dict(image_id=image_id, changes=changes), counts, support_seal.hexdigest()


def run(a, output, independent, prior, capture):
    population, orig, source = read_population(a, prior, capture)
    source_sha = digest(Path(__file__).read_bytes())
    if independent:
        ready, terminal = prior.closed(a.prepare.parent / "observed_prepare-edits/TERMINAL.json",
                                       a.prepare_terminal_sha, a.prepare / "RESULT_SAFE.json", a.prepare_result_sha)
        need(ready["status"] == PREPARED and terminal["phase"] == "prepare-edits"
             and ready["worker_sha256"] == terminal["worker_sha256"] == source_sha
             and ready["source_replay_receipt_sha256"] == SOURCE_RESULT
             and ready["GT_opened"] is False and ready["protocol_sha256"] == a.protocol_sha
             and ready["prelock_commit"] == a.prelock_commit, "Observed identical prelock and source on preparation")
        expected = pin(a.prepare / "EDITS_PRIVATE.jsonl", ready["edits_sha256"]).splitlines(keepends=True)
        need(len(expected) == len(population), "Exact whole-population prepared rows")
    counts, by_sensor, changes = Counter(), {"GF-3": Counter(), "TerraSAR-X": Counter()}, []
    support = sha256()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=initialize, initargs=(a,)) as pool:
        for i, (row, number, stamp) in enumerate(pool.map(replay_image, population, chunksize=2)):
            audit = population[i][0][0]
            need(row["image_id"] == audit["image_id"], "Exact author order")
            raw = (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            if independent:
                need(raw == expected[i], "Independent byte-identical splice output by image")
            counts.update(number)
            by_sensor[audit["sensor"]].update(number)
            changes.append(raw)
            support.update(bytes.fromhex(stamp))
            if (i+1) % 50 == 0:
                print(json.dumps(dict(images=i+1, mapped_candidates=counts["mapped_candidates"])), flush=True)
    scope_fields = ("images", "mapped_candidates", "area_eligible", "rendered", "degenerate", "editable_supports",
                    "endpoints_add", "endpoints_drop", "point_equal_swaps", "point_equal_candidates",
                    "point_paste_adds", "point_paste_drops", "point_paste_empty", "point_paste_candidates")
    for actual, expected_counts in [(counts, source["counts"])] + [
            (c, source["counts_by_sensor"][sensor]) for sensor, c in by_sensor.items()]:
        need(all(actual[k] == expected_counts[k] for k in scope_fields),
             "Complete archived source edit and full original support census")
    raw = b"".join(changes)
    edits_sha, support_sha = digest(raw), support.hexdigest()
    if independent:
        need(edits_sha == ready["edits_sha256"] and support_sha == ready["support_sha256"]
             and counts == ready["counts"] and by_sensor == ready["counts_by_sensor"],
             "Independent support seal and aggregation parity")
    output.mkdir()
    if not independent:
        (output / "EDITS_PRIVATE.jsonl").write_bytes(raw)
    return dict(status=ACCEPTED if independent else PREPARED, images=len(population), GT_opened=False,
                author_TEST=True, model_forwards=0, optimizer_updates=0, AP=False,
                counts=dict(counts), counts_by_sensor=by_sensor, support_sha256=support_sha, edits_sha256=edits_sha,
                protocol_sha256=a.protocol_sha, prelock_commit=a.prelock_commit, worker_sha256=source_sha,
                source_replay_receipt_sha256=SOURCE_RESULT, source_replay_terminal_sha256=SOURCE_TERMINAL,
                source_edits_sha256=SOURCE_EDITS, capture_receipt_sha256=a.capture_receipt_sha,
                predictions_sha256=orig["predictions_sha256"], candidate_audit_sha256=orig["audit_sha256"],
                exact_support_xor_area=True, exact_replayed_policy_choices=independent, workers=a.workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("prepare-edits", "verify-edits"))
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    cfg = parser.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and cfg.output.is_absolute() and cfg.output.name == cfg.phase and not cfg.output.exists(),
         "Fresh isolated CPU support-factor replay")
    independent = cfg.phase == "verify-edits"
    a = authority(json.loads(pin(cfg.authority, cfg.authority_sha)), independent)
    prior, capture = dependencies(a.code)
    prior.install_guard()
    start = perf_counter()
    result = run(a, cfg.output, independent, prior, capture)
    result.update(seconds=perf_counter()-start, input_authority_sha256=cfg.authority_sha)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    (cfg.output / "RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=result["status"], result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()
