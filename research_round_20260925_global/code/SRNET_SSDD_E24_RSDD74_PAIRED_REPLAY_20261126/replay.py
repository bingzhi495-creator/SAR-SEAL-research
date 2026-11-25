"""Replay the frozen RF HRSID point head on exact native SRNet candidates."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec


BASE = "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/replay.py"
PROTOCOL_SHA = "8cc6abb79faea6af12fa7d22aeec18d43e277323092dd5cdf91aba4c8d744826"
BASE_SHA = "be1516489123153a018ebc721806190d83f05285617e7c108e9b3c1064f66655"
MANIFEST_SHA = "154d0d748b96df0d56ac249adcb8b6ae63275a09dfd7a0d7480839005e044338"
WEIGHTS_SHA = "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1"
SRNET_WEIGHTS_SHA = "8998d0f6c11082250a46af48ec0c5455fb7606de8b6bb6a00ad2956046ea95ea"
PILOT_RESULT_SHA = "d914fcb254a5dcf143c496ceeedfed43080cc3ad28903f2054eae3e4818be112"
CAPTURE = "SRNET_SSDD_E24_RSDD74_NATIVE_CAPTURE_COMPLETE"
PREPARED = "SRNET_SSDD_E24_RSDD74_EDITS_PREPARED"
VERIFIED = "SRNET_SSDD_E24_RSDD74_EDITS_VERIFIED"
F, I = "fixed", "free_matched_count"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    data = path.read_bytes()
    require(digest(data) == expected, "Exact digest: " + path.name)
    return data


def json_pin(path, expected):
    return json.loads(pin(path, expected))


def encoded(row):
    return (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def module(path, expected):
    out = ModuleType(path.stem)
    out.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), out.__dict__)
    return out


def closed(terminal_path, terminal_sha, result_path, result_sha, phase, worker_sha=None):
    event = json_pin(terminal_path, terminal_sha)
    require(event["phase"] == phase and event["actual_exit"] == 0 and event["output_footer_verified"] is True
            and event["result_sha256"] == result_sha
            and (worker_sha is None or event["worker_sha256"] == worker_sha), "Observed closed " + phase)
    result = json_pin(result_path, result_sha)
    require(result.get("worker_sha256", event["worker_sha256"]) == event["worker_sha256"],
            "Observed worker identity: " + phase)
    return result


def edit_modules(code):
    base = module(code / BASE, BASE_SHA)
    return base, base.methods(code)


def render_pair(base, methods, stock, coords, q):
    require(q.shape == coords.shape and q.dtype == np.float32 and np.isfinite(q).all()
            and np.all((q >= 0) & (q <= 1)), "Frozen source FP32 posteriors")
    cap = min(8, int(stock.sum()) // 10, len(coords))
    fixed = methods.pair.choose_pairs(q[:, 0] - q[:, 1], cap)
    free = methods.free.choose_free(q, cap, fixed_count=len(fixed))
    require(len(free) == len(fixed), "Exact signed quota per candidate")
    out = {}
    if fixed:
        out = {F: base.mask_edit(stock, coords, [(j, j) for j in fixed]),
               I: base.mask_edit(stock, coords, free)}
    base.check_edits(stock, coords, out)
    return out


def checked_capture(a):
    report = closed(a.capture / "observed_capture/TERMINAL.json", a.capture_terminal_sha,
                    a.capture / "actual/RESULT_SAFE.json", a.capture_result_sha, "capture", a.capture_worker_sha)
    require(report["status"] == CAPTURE and report["protocol_sha256"] == a.protocol_sha
            and report["weights_sha256"] == SRNET_WEIGHTS_SHA and report["pilot_result_sha256"] == PILOT_RESULT_SHA
            and report["intake_image_manifest_sha256"] == MANIFEST_SHA
            and report["model_forwards"] == 74 and report["reference_files_opened"] == report["optimizer_updates"] == 0
            and report["all_native_rle_and_probability_masks_exact"] is True
            and report["all_original_rank_score_and_rotated_box_same_call_exact"] is True
            and report["all_archive_jpeg_and_rgb_exact"] is True and report["no_image_or_candidate_filtering"] is True
            and report["threshold_operator"] == ">" and report["mask_threshold"] == .50, "Exact E24 capture lineage")
    index = [json.loads(line) for line in pin(a.capture / "actual/INDEX_PRIVATE.jsonl", report["index_sha256"]).splitlines()]
    original = json_pin(a.manifest, MANIFEST_SHA)
    require(len(index) == len(original["items"]) == 74
            and [row["image_id"] for row in index] == original["expected_ids"], "Whole 74 population and order")
    for ordinal, (current, previous) in enumerate(zip(index, original["items"], strict=True), 1):
        require(current["ordinal"] == ordinal and all(current[key] == previous[key] for key in (
            "image_id", "source_image_name", "acquisition_group", "sensor", "original_hw", "rgb_sha256"))
            and current["image_sha256"] == previous["image_ref"]["sha256"]
            and current["original_hw"] == [512, 512], "Canonical reserve image identity")
    return report, index


def native_rows(row):
    candidates = row["candidates"]
    require(len(candidates) <= 2000, "Original SRNet max_per_img")
    output = []
    for rank, cand in enumerate(candidates, 1):
        require(type(cand["rank"]) is int and cand["rank"] == rank and cand["category_id"] == 1
                and np.isfinite(cand["score"]) and .30 <= cand["score"] <= 1
                and len(cand["rbox_xywha"]) == 5 and np.isfinite(cand["rbox_xywha"]).all()
                and cand["rle"]["size"] == [512, 512] and isinstance(cand["rle"]["counts"], str),
                "Original SRNet candidate identity, orientation and mask")
        output.append(dict(image_id=row["image_id"], category_id=1, score=cand["score"], segmentation=cand["rle"]))
    return output


def native_probabilities(a, row):
    root = a.capture / "actual"
    expected_name = f"PROBABILITIES_PRIVATE/{row['ordinal']:03d}.npz"
    ref = row["probability_file"]
    require(ref["path"] == expected_name, "Capture-local ordered native probability path")
    raw = pin(root / expected_name, ref["sha256"])
    require(len(raw) == ref["bytes"], "Exact compressed probability size")
    with np.load(BytesIO(raw), allow_pickle=False) as data:
        require(data.files == ["probability"], "Exact private native probability field")
        probabilities = data["probability"]
    require(probabilities.shape == (len(row["candidates"]), 512, 512)
            and probabilities.dtype == np.float32 and np.isfinite(probabilities).all()
            and np.all((probabilities >= -1e-6) & (probabilities <= 1 + 1e-6)),
            "Unmodified finite native projected SRNet probabilities")
    return probabilities


def render_image(a, row, base, methods, point, sampler, predict, verify):
    original = native_rows(row)
    probabilities = native_probabilities(a, row)
    # RGB is decoded from the canonical reserve bytes, separate from SRNet's inference normalization.
    rgb = base.load_rgb(a.manifest.parent / "images", row)
    image = methods.pair.image_features(rgb)
    changes = {}
    counts = Counter(images=1, native_candidates=len(original), no_candidate_images=int(not original))
    for cand, native, prob in zip(row["candidates"], original, probabilities, strict=True):
        stock = codec.decode(native["segmentation"]).astype(bool)
        require(np.array_equal(prob > .5, stock), "Author strict-threshold full-frame parity")
        area = int(stock.sum())
        counts["zero_native_masks"] += int(area == 0)
        if not 32 <= area < 1024:
            continue
        counts["native_area_eligible_candidates"] += 1
        box = np.asarray(cand["clipped_corner_xyxy"], dtype=np.float32)
        require(box.shape == (4,) and np.isfinite(box).all() and np.all((box >= 0) & (box <= 512))
                and np.all(box[2:] >= box[:2]), "Fixed clipped corner box")
        if np.any(box[2:] == box[:2]):
            counts["degenerate_support_boxes"] += 1
            continue
        data = point.paired_points(stock, box, area, native["score"], prob, image, sampler)
        coords = data[0] if data is not None else np.empty((0, 2), dtype=np.int32)
        if verify:
            base.check_support(methods, stock, box, prob, coords)
        if data is None:
            continue
        features = data[1]
        require(np.isfinite(features).all() and features.dtype == np.float32
                and features.shape == (len(coords), 2, 14), "Exact native per-endpoint features")
        counts["editable_supports"] += 1
        counts["native_half_probability_endpoints"] += int((features[:, :, 0] == .5).sum())
        options = render_pair(base, methods, stock, coords, predict(features))
        if not options:
            continue
        changes[str(cand["rank"])] = options
        counts["positive_matched_budget_candidates"] += 1
        counts["exact_adds_per_policy"] += options[F]["swaps"]
        counts["exact_drops_per_policy"] += options[F]["swaps"]
        counts["different_policy_candidates"] += int(options[F]["rle"] != options[I]["rle"])
    return dict(image_id=row["image_id"], changes=changes), counts


def run(a, output, verify, authority_sha):
    start = perf_counter()
    capture, index = checked_capture(a)
    own = digest(Path(__file__).read_bytes())
    previous, prior = None, None
    if verify:
        prior = closed(a.prepare / "observed_prepare-edits/TERMINAL.json", a.prepare_terminal_sha,
                       a.prepare / "prepare-edits/RESULT_SAFE.json", a.prepare_result_sha, "prepare-edits", own)
        require(prior["status"] == PREPARED and prior["capture_result_sha256"] == a.capture_result_sha
                and prior["protocol_sha256"] == a.protocol_sha
                and prior["label_files_opened"] == 0, "Original label-free edit closure")
        previous = pin(a.prepare / "prepare-edits/EDITS_PRIVATE.jsonl", prior["edits_sha256"]).splitlines(keepends=True)
        require(len(previous) == len(index), "Whole first edit ledger")

    import torch
    require(torch.__version__ == "2.10.0+cu130" and not torch.cuda.is_initialized(), "Qualified CPU point runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base, methods = edit_modules(a.code)
    predict = base.predictor(a.code, methods, a.fit_result, a.fit_accepted, a.weights)
    point, sampler = base.canvas(methods, (512, 512))
    rows, totals = [], Counter()
    for i, row in enumerate(index):
        item, counts = render_image(a, row, base, methods, point, sampler, predict, verify)
        out = encoded(item)
        require(previous is None or out == previous[i], "Closed independent replay byte equality")
        rows.append(out)
        totals.update(counts)
    payload = b"".join(rows)
    checksum = digest(payload)
    require(totals["images"] == 74 and totals["native_candidates"] == capture["counts"]["candidates"]
            and totals["no_candidate_images"] == capture["counts"].get("no_candidates", 0) and (not verify or (
        checksum == prior["edits_sha256"] and dict(totals) == prior["counts"])), "Complete candidate flow")
    output.mkdir()
    if not verify:
        (output / "EDITS_PRIVATE.jsonl").write_bytes(payload)
    result = dict(status=VERIFIED if verify else PREPARED, protocol_sha256=a.protocol_sha,
                  prelock_commit=a.prelock_commit, input_authority_sha256=authority_sha, worker_sha256=own,
                  capture_result_sha256=a.capture_result_sha, capture_terminal_sha256=a.capture_terminal_sha,
                  capture_index_sha256=capture["index_sha256"], manifest_sha256=MANIFEST_SHA, seed=5217,
                  fit_result_sha256=base.FIT_SHA, fit_accepted_sha256=base.FIT_ACCEPTED_SHA, fit_weights_sha256=WEIGHTS_SHA,
                  labels_opened=False, label_files_opened=0, detector_forwards=0, optimizer_updates=0,
                  all_stock_threshold_exact=True,
                  same_per_candidate_signed_budget=True, supports_through_unchanged_frozen_core=True,
                  exact_independent_replay=verify, edits_sha256=checksum, counts=dict(totals),
                  elapsed_seconds=perf_counter()-start)
    if verify:
        result.update(prepare_result_sha256=a.prepare_result_sha, prepare_terminal_sha256=a.prepare_terminal_sha)
    payload = encoded(result)
    (output / "RESULT_SAFE.json").write_bytes(payload)
    print(json.dumps(dict(status=result["status"], result_sha256=digest(payload))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("prepare-edits", "verify-edits"))
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    cmd = parser.parse_args()
    verify = cmd.phase == "verify-edits"
    raw = json_pin(cmd.authority, cmd.authority_sha)
    paths = {"code", "protocol", "capture", "manifest", "fit_result", "fit_accepted", "weights"}
    hashes = {"protocol_sha", "capture_terminal_sha", "capture_result_sha", "capture_worker_sha"}
    if verify:
        paths.add("prepare")
        hashes.update(("prepare_terminal_sha", "prepare_result_sha"))
    require(set(raw) == paths | hashes | {"prelock_commit"} and all(Path(raw[key]).is_absolute() for key in paths)
            and all(re.fullmatch(r"[0-9a-f]{64}", raw[key]) for key in hashes)
            and re.fullmatch(r"[0-9a-f]{40}", raw["prelock_commit"]) and sys.flags.isolated and sys.dont_write_bytecode
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not cmd.output.exists(),
            "Fresh isolated admitted CPU run")
    a = SimpleNamespace(**{key: Path(value).resolve(strict=True) if key in paths else value for key, value in raw.items()})
    require(a.protocol_sha == PROTOCOL_SHA, "Published SRNet paired evaluation prelock")
    pin(a.protocol, a.protocol_sha)
    run(a, cmd.output, verify, cmd.authority_sha)


if __name__ == "__main__":
    main()

