"""Independently replay frozen edit choices and validate the twelve native-format streams."""

import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid
import zlib

import numpy as np
from pycocotools import mask as codec


SEEDS = ("5217", "5218", "5219")
ARMS = ("SF", "SU=F", "SU", "LW=S")
PHASE = "HRSID_OFFICIAL_TEST_FROZEN_REFINEMENT_HISTORICAL_DIAGNOSTIC_V1"
FIT = "476b0f3fa748d1d3fd2393bdeb1af2d83e1532995d58b12625d57222e3f2638a"
ENDPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
BASE_NATIVE = "9c16ecb5275ae1ba23926a46aa74a5e10c979a3761838d7ff279859986a742b4"
BASE_READBACK = "18ef3b71b4bf9ad3ba4d9d4a80d86c542b1b48a7e373aff14d3a7bb398e89295"
CAPTURE = "a50da3d740d9620a23c98b978b7cfaa383040f1e7542e2eb5da32ca4b0106da6"


def need(test, message):
    if not test:
        raise ValueError(message)


def encode(obj):
    return (json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(data):
    return sha256(data).hexdigest()


def pinned_json(path, expected):
    data = path.read_bytes()
    need(digest(data) == expected, "Pinned file differs: " + path.name)
    return json.loads(data)


def fields(obj, expected):
    # Exact JSON scalar types matter for ranks, zero-valued scores and flags.
    need(all(k in obj and type(obj[k]) is type(v) and obj[k] == v for k, v in expected.items()),
         "Inconsistent metadata for " + ", ".join(expected))


def float_array(text, shape):
    count = math.prod(shape)
    need(isinstance(text, str) and len(text) == count * 8, "Float32 witness size")
    raw = bytes.fromhex(text)
    need(len(raw) == count * 4 and raw.hex() == text, "Canonical little-endian float witness")
    arr = np.frombuffer(raw, dtype="<f4").reshape(shape)
    need(np.isfinite(arr).all(), "Nonfinite witness")
    need(((arr >= 0) & (arr <= 1)).all(), "Posterior outside its domain")
    return arr


def recompute(coords, small, large, cap):
    """Apply the four frozen definitions to archived posteriors using independent stable sorts."""
    k = len(coords)
    fixed = sorted(range(k), key=lambda j: -float(small[j, 0] - small[j, 1]))
    sf = [(j, j) for j in fixed if small[j, 0] > small[j, 1]][:cap]
    adds = sorted(range(k), key=lambda j: -float(small[j, 0]))[:cap]
    drops = sorted(range(k), key=lambda j: float(small[j, 1]))[:cap]
    free = list(zip(adds, drops))
    su = [(i, j) for i, j in free if small[i, 0] > small[j, 1]]
    plans = {}
    half = np.float32(.5)
    for arm, pairs in (("SF", sf), ("SU=F", free[:len(sf)]), ("SU", su)):
        flips = [(int(coords[i, 0]), small[i, 0] - half, True) for i, j in pairs]
        others = [(int(coords[j, 1]), half - small[j, 1], False) for i, j in pairs]
        ordered = [v for pair in zip(flips, others) for v in pair]
        plans[arm] = summarize(ordered, [small[i, 0] - small[j, 1] for i, j in pairs], "swap")
    ranked = [(int(coords[i, side]), large[i, side] - half if side == 0 else half - large[i, side],
               side == 0) for i in range(k) for side in (0, 1)]
    ranked.sort(key=lambda v: (-float(v[1]), v[0]))
    chosen = ranked[:2 * len(su)]
    plans["LW=S"] = summarize(chosen, [v[1] for v in chosen], "flip")
    return plans


def summarize(selected, action_gains, kind):
    gains = [v[1] for v in selected]
    nadd = sum(v[2] for v in selected)
    return dict(flipped=[v[0] for v in selected], edits=len(selected), adds=nadd,
                drops=len(selected) - nadd, area_delta=2 * nadd - len(selected),
                actions=len(action_gains), action_kind=kind,
                nonpositive_actions=sum(float(x) <= 0 for x in action_gains),
                nonpositive_flips=sum(float(x) <= 0 for x in gains),
                predicted_gain_sum=float(np.sum(gains, dtype=np.float64)))


def validate_image(block, index, line, totals):
    need(line.endswith(b"\n"), "Incomplete stock line")
    native = json.loads(line)
    need(encode(native) == line, "Noncanonical stock block")
    image_id, rows, ranks = native["image_id"], native["coco"], native["native_ranks"]
    fields(block, dict(index_zero_based=index, image_id=image_id, stock_block_sha256=digest(line)))
    fields(native, dict(raw_query_count=164, native_count=100, unmapped_slot1_count=100-len(rows)))
    witness = block["witness"]
    need(set(witness) == {"image_id", "mapped", "eligible", "supported", "candidates", "native_line_sha256",
                          "joint_scores_f32le_hex"}, "Native witness schema")
    fields(witness, dict(image_id=image_id, mapped=len(rows), native_line_sha256=digest(line)))
    joint = float_array(witness["joint_scores_f32le_hex"], (328,))
    top = sorted(range(328), key=lambda j: -float(joint[j]))[:100]
    mapped = [(rank, j // 2, float(joint[j])) for rank, j in enumerate(top, 1) if j % 2 == 0]
    need([c[0] for c in mapped] == ranks and len(mapped) == len(rows) == len(witness["candidates"]),
         "Joint top-100 / mapped rank identity")
    arms = block["predictions"]
    need(set(arms) == set(SEEDS) and all(set(a) == set(ARMS) for a in arms.values()), "Incomplete arms")
    for outputs in arms.values():
        for output in outputs.values():
            need(list(output) == list(native) and len(output["coco"]) == len(rows)
                 and all(encode(output[k]) == encode(native[k]) for k in native if k != "coco"),
                 "Candidate roster differs")
    eligible = supported = 0
    for i, ((rank, qid, confidence), row, candidate) in enumerate(zip(mapped, rows, witness["candidates"])):
        fields(row, dict(image_id=image_id, category_id=1, score=confidence))
        rle = row["segmentation"]
        need(rle["size"] == [800, 800], "Native canvas")
        area = int(codec.area(rle))
        fields(candidate, dict(rank=rank, query_id=qid, area=area, score=confidence))
        is_eligible = 32 <= area < 1024
        eligible += int(is_eligible)
        edits = candidate.get("edits", {})
        coords = candidate.get("coordinates", [])
        base_keys = {"rank", "query_id", "area", "score"}
        if is_eligible:
            base_keys.update(("edits", "coordinates"))
        if coords:
            base_keys.update(("feature_sha256", "point_scores_f32le_hex"))
        need(set(candidate) == base_keys and isinstance(coords, list) and isinstance(edits, dict),
             "Candidate witness for this support")
        if coords:
            supported += 1
            need(is_eligible and len(coords) <= 32 and all(isinstance(pair, list) and len(pair) == 2
                 and all(type(j) is int and 0 <= j < 640000 for j in pair) for pair in coords),
                 "Point coordinate domain")
            arr = np.array(coords, dtype=np.int32)
            stock = codec.decode(rle).astype(bool)
            need(len(np.unique(arr)) == arr.size and not stock.flat[arr[:, 0]].any()
                 and stock.flat[arr[:, 1]].all() and int(stock.sum()) == area, "Native signs, support or area")
            need(re.fullmatch("[0-9a-f]{64}", candidate["feature_sha256"]) is not None, "Feature digest")
            need(set(candidate["point_scores_f32le_hex"]) == set(SEEDS), "Head seed witness")
        else:
            need(not edits, "Edits outside supported candidates")
        need(set(edits).issubset(SEEDS), "Unexpected seed")
        cached = {}
        for seed in SEEDS:
            plans = {}
            if coords:
                scores = candidate["point_scores_f32le_hex"][seed]
                need(set(scores) == {"S", "L"}, "Frozen heads")
                plans = recompute(arr, float_array(scores["S"], (len(coords), 2)),
                                  float_array(scores["L"], (len(coords), 2)), min(8, area // 10, len(coords)))
            selected = {a: p for a, p in plans.items() if p["edits"]}
            recorded = edits.get(seed, {})
            need(set(selected) == set(recorded) and (seed not in edits or bool(recorded)), "Selected arm set")
            for arm in ARMS:
                out = arms[seed][arm]["coco"][i]
                need(list(out) == list(row) and all(encode(out[k]) == encode(row[k]) for k in row
                     if k != "segmentation"), "Mapped candidate fields changed")
                if arm not in selected:
                    need(rle == out["segmentation"], "No-op native mask changed")
                    continue
                plan = selected[arm]
                need(set(recorded[arm]) == set(plan), "Sparse witness schema")
                fields(recorded[arm], plan)
                need(all(type(j) is int for j in recorded[arm]["flipped"]), "Integer edit indices")
                key = tuple(sorted(plan["flipped"]))
                if key not in cached:
                    target = stock.copy(order="C")
                    ids = np.array(key)
                    target.flat[ids] = ~stock.flat[ids]
                    actual = codec.encode(np.asfortranarray(target, dtype=np.uint8))
                    cached[key] = (dict(size=actual["size"], counts=actual["counts"].decode("ascii")),
                                   int(codec.area(actual)), int(stock.flat[ids].sum()))
                expected, final_area, drops = cached[key]
                need(out["segmentation"] == expected and drops == plan["drops"] and len(key)-drops == plan["adds"]
                     and final_area-area == plan["area_delta"] and (arm == "LW=S" or final_area == area),
                     "Actual raster XOR, sign, or area")
                totals[f"{seed}/{arm}/edited_candidates"] += 1
                for k in ("edits", "adds", "drops", "area_delta"):
                    totals[f"{seed}/{arm}/{k}"] += plan[k]
    fields(witness, dict(eligible=eligible, supported=supported))
    totals.update(mapped=len(rows), images=1, eligible=eligible, supported=supported)
    return native, arms


def checked_provenance(admission_path, admission_sha, terminal_path, terminal_sha):
    terminal = pinned_json(terminal_path, terminal_sha)
    fields(terminal, dict(status="HRSID_FROZEN_REFINEMENT_OWNER_TERMINAL_OBSERVED",
                          original_process_terminal=True, original_exit=0, admission_sha256=admission_sha,
                          footer_status="HRSID_FROZEN_REFINEMENT_READOUT_COMPLETE"))
    admission = pinned_json(admission_path, admission_sha)
    fields(admission, dict(status="OWNER_ADMITTED_HRSID_FROZEN_REFINEMENT_READOUT_V1", phase=PHASE,
                           policy=list(ARMS), seeds=[int(s) for s in SEEDS], GT_opened=False,
                           historical_test_selection_exposure=True))
    fields(admission["sha256"], dict(fit_acceptance=FIT, capture_source=CAPTURE))
    paths = {k: Path(v) for k, v in admission["paths"].items()}
    owner = paths["output"]
    expected_result = terminal["owner_result_sha256"]
    need(expected_result == terminal["stdout_output_sha256"], "Observed stdout result identity")
    result = pinned_json(owner / "RESULT_SAFE.json", expected_result)
    fields(result, dict(status=terminal["footer_status"], phase=PHASE, images=1962, mapped_rows=187796,
                        admission_sha256=admission_sha, baseline_readback_sha256=BASE_READBACK,
                        baseline_predictions_sha256=BASE_NATIVE, detector_forwards=1962,
                        optimizer_updates=0, GT_opened=False, AP=False))
    opened = json.loads((owner / "PRELABEL_SAFE.json").read_bytes())
    fields(opened, dict(status="HRSID_FROZEN_REFINEMENT_WORKER_OPENED", phase=PHASE,
                        admission_sha256=admission_sha, baseline_readback_sha256=BASE_READBACK,
                        original_endpoint_sha256=ENDPOINT, GT_opened=False, AP=False))
    baseline = pinned_json(paths["native_readback"], BASE_READBACK)
    fields(baseline, dict(status="HRSID_OFFICIAL_TEST_NATIVE_EXPORT_INDEPENDENT_READBACK_ACCEPTED",
                          original_process_terminal=True, original_exit=0, GT_opened=False, AP=False,
                          endpoint_sha256=ENDPOINT, predictions_sha256=BASE_NATIVE, images=1962,
                          mapped_rows=187796))
    return paths, result, baseline, terminal


def write_record(path, record):
    with path.open("xb") as out:
        out.write(encode(record))
        out.flush()
        os.fsync(out.fileno())


def multihash(stream):
    h = sha256()
    for data in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(data)
    return h.hexdigest()


def replay(rows_dir, stock_path, result, baseline, output, *, native_sha):
    """Publish verified streams only when the caller subsequently seals the receipt."""
    count = baseline["images"]
    names = {f"{i:05d}.json.zlib" for i in range(count)}
    need({p.name for p in rows_dir.iterdir()} == names, "Missing or extra captured image block")
    writers, references, path_index = {}, {}, {}
    tally = Counter()
    owner_hash, native_hash, feature_hash = sha256(), sha256(), sha256()
    packed_bytes = native_bytes = 0
    image_ids = set()
    timing = Counter()
    with ExitStack() as stack:
        native = stack.enter_context(stock_path.open("rb"))
        need(os.fstat(native.fileno()).st_size == baseline["prediction_bytes"]
             and multihash(native) == native_sha, "Closed native stream identity")
        native.seek(0)
        for seed in SEEDS:
            writers[seed], references[seed], path_index[seed] = {}, {}, {}
            for arm in ARMS:
                filename = seed + "_" + arm + ".jsonl"
                writers[seed][arm] = (stack.enter_context((output / filename).open("xb")), sha256())
                path_index[seed][arm] = filename
        for i in range(count):
            line = native.readline()
            native_hash.update(line)
            native_bytes += len(line)
            compressed = (rows_dir / f"{i:05d}.json.zlib").read_bytes()
            owner_hash.update(sha256(compressed).digest())
            packed_bytes += len(compressed)
            decoder = zlib.decompressobj()
            raw = decoder.decompress(compressed)
            need(decoder.eof and not decoder.unused_data, "Trailing or incomplete compressed capture")
            block = json.loads(raw)
            need(set(block) == {"index_zero_based", "image_id", "stock_block_sha256", "predictions",
                                "witness", "timings"}, "Image envelope")
            parsed, arms = validate_image(block, i, line, tally)
            need(parsed["image_id"] not in image_ids, "Duplicate native image")
            image_ids.add(parsed["image_id"])
            for c in block["witness"]["candidates"]:
                if "feature_sha256" in c:
                    feature_hash.update(encode([parsed["image_id"], c["rank"], c["query_id"], c["feature_sha256"]]))
            times = block["timings"]
            need(set(times) == {"detector_forward_seconds", "native_postprocess_parity_seconds",
                                "refinement_cpu_seconds"} and all(type(t) in (float, int) and math.isfinite(t)
                                and t >= 0 for t in times.values()), "Phase timings")
            timing.update(times)
            for seed in SEEDS:
                for arm, out in arms[seed].items():
                    obj = encode(out)
                    handle, h = writers[seed][arm]
                    need(handle.write(obj) == len(obj), "Short treatment write")
                    h.update(obj)
        need(not native.read(1) and native_bytes == baseline["prediction_bytes"] and native_hash.hexdigest() == native_sha,
             "Complete native readback")
        need(packed_bytes == result["image_row_bytes"] and owner_hash.hexdigest() == result["image_rows_sha256"],
             "Owner aggregate per-block bytes or hash")
        need(tally["images"] == count and tally["mapped"] == baseline["mapped_rows"], "Complete population")
        need(dict(timing) == result["phase_seconds"], "Owner time totals")
        for seed in SEEDS:
            for arm, (handle, h) in writers[seed].items():
                handle.flush()
                os.fsync(handle.fileno())
                references[seed][arm] = dict(sha256=h.hexdigest(), bytes=handle.tell())
    for seed in SEEDS:
        for arm in ARMS:
            with (output / path_index[seed][arm]).open("rb") as handle:
                ref = references[seed][arm]
                need(os.fstat(handle.fileno()).st_size == ref["bytes"] and multihash(handle) == ref["sha256"],
                     "Closed materialized stream readback")
    return references, path_index, dict(tally), feature_hash.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("admission", "terminal", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("admission-sha256", "terminal-sha256"):
        parser.add_argument("--" + key, required=True)
    a = parser.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode, "Run the reader with -I -B")
    paths, result, baseline, terminal = checked_provenance(a.admission, a.admission_sha256, a.terminal, a.terminal_sha256)
    destination = a.output
    owner = paths["output"].resolve(strict=True)
    need(destination.is_absolute() and destination.parent.resolve(strict=True) == destination.parent
         and destination.parent == owner.parent and not destination.exists(), "Fresh independent sibling output")
    draft = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
    draft.mkdir()
    refs, filenames, totals, feature_digest = replay(owner / "IMAGE_ROWS_PRIVATE", paths["native_predictions"], result,
                                                    baseline, draft, native_sha=BASE_NATIVE)
    path_index = {s: {a: str(destination / name) for a, name in names.items()} for s, names in filenames.items()}
    write_record(draft / "PATHS_PRIVATE.json", path_index)
    receipt = dict(status="HRSID_OFFICIAL_TEST_FROZEN_REFINEMENT_READOUT_INDEPENDENT_ACCEPTED",
                   original_process_terminal=True, original_exit=0, TEST=True, GT_opened=False, AP=False,
                   historical_test_selection_exposure=True, optimizer_updates=0, images=totals["images"],
                   mapped_rows=totals["mapped"], endpoint_sha256=ENDPOINT, fit_accepted_sha256=FIT,
                   native_predictions_sha256=BASE_NATIVE, native_readback_sha256=BASE_READBACK,
                   exact_candidate_identity=True, exact_native_per_image=True, exact_per_candidate_budgets=True,
                   actual_XOR_area_RLE_verified=True, point_score_decisions_replayed=True,
                   descriptor_identity_ledger_sha256=feature_digest, image_features_recomputed=False,
                   owner_result_sha256=terminal["owner_result_sha256"], terminal_sha256=a.terminal_sha256,
                   admission_sha256=a.admission_sha256, reader_sha256=digest(Path(__file__).read_bytes()),
                   owner_blocks_sha256=result["image_rows_sha256"], streams=refs, counts=totals)
    write_record(draft / "READOUT_ACCEPTED.json", receipt)
    need(not destination.exists(), "Fresh independent output")
    draft.rename(destination)
    print(json.dumps(dict(status=receipt["status"], sha256=digest(encode(receipt)))), flush=True)


if __name__ == "__main__":
    main()
