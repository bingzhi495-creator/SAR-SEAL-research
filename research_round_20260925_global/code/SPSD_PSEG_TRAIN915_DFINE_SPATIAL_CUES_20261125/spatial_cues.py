"""Capture and independent readback of frozen D-FINE spatial endpoint cues."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np


BASE: Path
POINTS: Path
AUTH: Path
REFERENCE: Path
MIRROR: Path
ACCEPT: Path
EMA: Path
CODEC: Path
LOCK = "7f0fb2c8ee98f0dcb784dc356b5a118fe926ef40dabc28f08c306d36f61da1f8"
NATIVE = "6f58ba8b486950431f4aebebf237e60029b447e5ec21a7f51ad2202d065f904a"
NATIVE_READBACK = "e461828b04f166d98030210fe42df0d590d24e2452ea5da8630fb0fc3219d795"
POINT_RESULT = "87693897f873a2129764781611b0f7c35cadde51c0c4e830d7f8187c3b684049"
TRAIN_WRAPPER = "0f10812a7d2409f69334670e7dae16a9ef51b4f84a20ee1278a4187d06e4c9a0"
NATIVE_SOURCE = "8a7dd193e1d4e14eae528cba453941af9bc7d646646f34888726ec9287e0ce12"
FIT = dict(fit=(732, 34796, 1041568), source_gate=(183, 8821, 265943))
CAPTURE_STATUS = "SPSD_PSEG_TRAIN915_DFINE_SPATIAL_CUES_COMPLETE"
VERIFY_STATUS = "SPSD_PSEG_TRAIN915_DFINE_SPATIAL_CUES_READBACK_COMPLETE"
HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("ascii")


def pin(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, f"Pinned bytes: {path.name}")
    return raw


def module(path, expected, name):
    out = ModuleType(name)
    out.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), vars(out))
    return out


def closed(terminal, result, expected, phase):
    raw = terminal.read_bytes()
    doc = json.loads(raw)
    require(doc["phase"] == phase and doc["actual_exit"] == 0
            and doc["output_footer_verified"] is True and doc["result_sha256"] == expected,
            f"Observed closed run: {terminal.parent.name}")
    return json.loads(pin(result, expected)), digest(raw), doc["worker_sha256"]


def no_labels(record):
    return not any(record[field] for field in (
        "AP", "train_labels_opened", "test_labels_opened", "test_images_opened", "optimizer_updates"))


def original_inputs():
    native, _, worker = closed(BASE / "observed_capture/TERMINAL.json",
                               BASE / "capture/RESULT_SAFE.json", NATIVE, "capture")
    checked, _, _ = closed(BASE / "observed_readback/TERMINAL.json",
                          BASE / "READBACK_SAFE.json", NATIVE_READBACK, "verify")
    points, _, _ = closed(POINTS / "observed_points_retry/TERMINAL.json",
                         POINTS / "actual_retry/RESULT_SAFE.json", POINT_RESULT, "prepare-edits")
    require(worker == TRAIN_WRAPPER and native["FIT"] is True and native["TEST"] is False
            and native["GT_opened"] is False and native["images"] == 915 and native["rows_at_100"] == 91500
            and points["capture_result_sha256"] == NATIVE and points["capture_readback_sha256"] == NATIVE_READBACK
            and no_labels(points) and not points["detector_forwards"]
            and checked is not None, "Closed TRAIN-only original streams and preparation")
    rows = {}
    for part, (n, _, _) in FIT.items():
        split = points["partitions"][part]
        path = POINTS / "actual_retry" / part / "POINT_INDEX_PRIVATE.jsonl"
        records = [json.loads(raw) for raw in pin(path, split["point_index_sha256"]).splitlines()]
        require(len(records) == n, "All old point rows")
        last = 0
        for row in records:
            ordinal = row["native_archive_ordinal"]
            require(last < ordinal <= 915 and ordinal not in rows
                    and row["filename"] == f"POINTS_PRIVATE/{ordinal:04}.npz",
                    "Exact disjoint original point ordinal roster")
            rows[ordinal] = (part, row)
            last = ordinal
    require(set(rows) == set(range(1, 916)), "Exact original 915 point ordinals")
    return native, rows


def point_fields(part, meta):
    old = pin(POINTS / "actual_retry" / part / meta["filename"], meta["sha256"])
    require(len(old) == meta["bytes"], "Source point file size")
    with np.load(BytesIO(old), allow_pickle=False) as points:
        ranks, offsets, coordinates = (points[key] for key in (
            "native_ranks", "point_offsets", "signed_coordinates"))
    h, w = meta["original_hw"]
    require(ranks.shape == (meta["supported_candidates"],) and ranks.dtype == np.int16
            and bool(((1 <= ranks) & (ranks <= 100)).all()) and bool((np.diff(ranks) > 0).all())
            and offsets.shape == (len(ranks) + 1,) and offsets.dtype == np.int32
            and offsets[0] == 0 and offsets[-1] == meta["paired_sites"] and bool((np.diff(offsets) > 0).all())
            and coordinates.shape == (offsets[-1], 2) and coordinates.dtype == np.int32
            and bool(((coordinates >= 0) & (coordinates < h*w)).all()),
            "Exact original point topology and native geometry")
    return ranks, offsets, coordinates


def projection():
    signs = np.random.Generator(np.random.PCG64(5297)).integers(0, 2, (256, 6), dtype=np.int8)
    return (2 * signs - 1).astype(np.float32) / np.float32(16.0)


def sample(torch, feature, weight, coordinates, hw):
    if not len(coordinates):
        return np.empty((0, 2, 6), dtype=np.float32)
    h, w = hw
    y, x = np.divmod(coordinates, w)
    x = np.asarray(2.0 * (x + .5) / w - 1.0, dtype=np.float32)
    y = np.asarray(2.0 * (y + .5) / h - 1.0, dtype=np.float32)
    grid = torch.as_tensor(np.stack((x, y), axis=-1)[None], device=feature.device)
    projected = torch.einsum("bchw,cd->bdhw", feature, weight)
    sampled = torch.nn.functional.grid_sample(
        projected, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return sampled[0].permute(1, 2, 0).contiguous().cpu().numpy()


def synthetic(torch):
    cpu_weight = torch.as_tensor(projection())
    # Two channels encode a row/column ramp. A rectangular native grid checks
    # coordinate orientation, half-pixel placement, border behavior and side order.
    h, w = 4, 6
    field = torch.zeros((1, 256, 2, 3), dtype=torch.float32)
    field[0, 0] = torch.tensor([[0, 1, 2], [10, 11, 12]])
    field[0, 1] = 4.
    coords = np.array([[0, 7], [22, 23]], dtype=np.int32)
    expected = np.array([[0., 2.75], [11.75, 12.]], dtype=np.float32)
    expected = expected[..., None] * projection()[0] + 4. * projection()[1]
    require(np.allclose(sample(torch, field, cpu_weight, coords, (h, w)), expected,
                        rtol=0, atol=1e-6), "Analytic rectangle and projection parity")
    require(sample(torch, field, cpu_weight, np.empty((0, 2), dtype=np.int32), (h, w)).shape
            == (0, 2, 6), "Exact empty feature dimensions")


def write_new(path, raw):
    with path.open("xb") as output:
        require(output.write(raw) == len(raw), "Full private cue write")
        output.flush()
        os.fsync(output.fileno())


def finish(a, result):
    result.update(worker_sha256=digest(Path(__file__).read_bytes()), protocol_sha256=LOCK,
                  capture_result_sha256=NATIVE, capture_readback_sha256=NATIVE_READBACK,
                  point_result_sha256=POINT_RESULT, optimizer_updates=0, AP=False,
                  train_labels_opened=False, test_labels_opened=False, test_images_opened=False)
    raw = encode(result)
    write_new(a.output / "RESULT_SAFE.json", raw)
    print(json.dumps(dict(status=result["status"], result_sha256=digest(raw))), flush=True)


def actual_capture(a):
    from contextlib import ExitStack

    start = perf_counter()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
            and shutil.disk_usage(a.output.parent).free >= 1024**3, "GPU visibility and disk floor")
    prior, point_rows = original_inputs()
    native = module(REFERENCE / "native_capture_reference.py", NATIVE_SOURCE, "_native_original")
    train = module(BASE / "capture.py", TRAIN_WRAPPER, "_train_original")
    original = train.upstream_capture(MIRROR)
    intake = BASE / "intake_retry"
    train.bind_intake(intake)
    base, torch, wrapper, author = native.initialize(SimpleNamespace(
        base=REFERENCE / "author_export_reference.py", acceptance=ACCEPT,
        author_source=AUTH / "AUTHOR_SOURCE_VERIFIED",
        author_tree=AUTH / "AUTHOR_SOURCE_TREE_PRIVATE.json", ema=EMA))
    rows = train.manifest_rows(intake / "TRAIN_IMAGE_ONLY_MANIFEST_PRIVATE.json", train.MANIFEST, base)
    synthetic(torch)
    weight_np = projection()
    weight = torch.as_tensor(weight_np, device="cuda")
    peak_start, _ = torch.cuda.mem_get_info()
    feature = []

    def capture_feature(_, inputs, output):
        require(len(inputs) == 1 and tuple(output.shape) == (1, 256, 160, 160)
                and output.dtype == torch.float32 and output.device.type == "cuda",
                "Original author MaskDecoder pre-query feature [1,256,160,160]")
        feature.append(output)

    a.output.mkdir()
    for part in FIT:
        (a.output / part / "CUES_PRIVATE").mkdir(parents=True)
    base.prepare_export_codec(CODEC, a.output / "CODEC_PRIVATE")
    counts = {part: Counter() for part in FIT}
    cue_index = {part: sha256() for part in FIT}
    old_sha = {field: sha256() for field in ("predictions_sha256", "candidate_audit_sha256", "probability_index_sha256")}
    seconds = 0.
    n_all = 0
    n_top = 0
    forward, nms, hook, orig_nms = native.bind_observers(wrapper, author, torch)
    feature_hook = wrapper.model.decoder.mask_decoder.register_forward_hook(capture_feature)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        with ExitStack() as stack:
            capture = BASE / "capture"
            pred_stream = stack.enter_context((capture / base.PREDICTIONS).open("rb"))
            audit_stream = stack.enter_context((capture / base.AUDIT).open("rb"))
            prob_stream = stack.enter_context((capture / "PROBABILITIES_PRIVATE.jsonl").open("rb"))
            destination = {part: stack.enter_context((a.output / part / "CUE_INDEX_PRIVATE.jsonl").open("xb"))
                           for part in FIT}
            stack.enter_context(torch.inference_mode())
            for ordinal, row in enumerate(rows, 1):
                part, meta = point_rows[ordinal]
                require(meta["image_id"] == row["image_id"] and meta["original_hw"] == row["original_hw"]
                        and meta["source_group"] == row["source_group"] and meta["rgb_sha256"] == row["rgb_sha256"],
                        "Exact original image/point join")
                rgb = original.read_rgb(intake / "FIT_IMAGES", row, base)
                require(digest(rgb.tobytes(order="C")) == row["rgb_sha256"], "Exact native RGB")
                ranks, offsets, coordinates = point_fields(part, meta)
                feature.clear()
                begin = perf_counter()
                predicted, with_query, planes = native.capture_row(rgb, row, wrapper, author, torch, np,
                                                                    base, forward, nms)
                require(len(feature) == 1, "One original mask spatial activation per admitted image")
                cues = sample(torch, feature[0], weight, coordinates, meta["original_hw"])
                torch.cuda.synchronize()
                seconds += perf_counter() - begin
                require(cues.shape == (*coordinates.shape, 6) and cues.dtype == np.float32
                        and bool(np.isfinite(cues).all()), "Fixed, finite projected cues")
                candidates = with_query["candidates"]
                audit = dict(with_query, candidates=[{k: v for k, v in c.items()
                                                       if k not in ("query_id", "probability_row")} for c in candidates])
                raw_prediction, raw_audit, raw_probability = (f.readline() for f in (pred_stream, audit_stream, prob_stream))
                require(raw_prediction == base.encoded(predicted) and raw_audit == base.encoded(audit),
                        "Exact frozen all-candidate scores, masks and native audit")
                prob = json.loads(raw_probability)
                prob_ref = prob["probability"]
                plane_bytes = BytesIO()
                np.save(plane_bytes, planes, allow_pickle=False)
                data = plane_bytes.getvalue()
                require(prob["image_id"] == row["image_id"] and len(planes) == prob_ref["rows"] == 100
                        and prob_ref["file"] == f"PROBABILITY_PRIVATE/{ordinal:04d}.npy"
                        and prob["query_ids"] == [c["query_id"] for c in candidates]
                        and len(data) == prob_ref["bytes"] and digest(data) == prob_ref["sha256"],
                        "Exact archived top100 raw probability NPY bytes and query order")
                for key, raw in zip(old_sha, (raw_prediction, raw_audit, raw_probability), strict=True):
                    old_sha[key].update(raw)
                out = BytesIO()
                np.savez(out, native_archive_ordinal=np.int32(ordinal), native_ranks=ranks,
                         point_offsets=offsets, signed_coordinates=coordinates, cue=cues,
                         point_sha256=np.asarray(meta["sha256"], dtype="S64"))
                raw = out.getvalue()
                relative = f"CUES_PRIVATE/{ordinal:04d}.npz"
                write_new(a.output / part / relative, raw)
                cue_line = encode(dict(meta, filename=relative, sha256=digest(raw), bytes=len(raw),
                                       point_sha256=meta["sha256"]))
                require(destination[part].write(cue_line) == len(cue_line), "Full cue index row")
                cue_index[part].update(cue_line)
                counts[part].update(images=1, supported_candidates=len(ranks), paired_sites=len(coordinates),
                                    cue_bytes=len(raw), endpoints_by_side=2*len(coordinates),
                                    candidates_without_pairs=100-len(ranks))
                n_all += len(candidates)
                n_top += len(planes)
                if ordinal % 100 == 0 or ordinal == 915:
                    print(json.dumps(dict(progress_images=ordinal, nms_candidates=n_all,
                                          exact_top100_probability_rows=n_top)), flush=True)
            require(not any(stream.read(1) for stream in (pred_stream, audit_stream, prob_stream)),
                    "All three old streams exhausted simultaneously")
            for stream in destination.values():
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        feature_hook.remove()
        hook.remove()
        author.nms = orig_nms
    require(all(value.hexdigest() == prior[key] for key, value in old_sha.items())
            and n_all == prior["post_nms_rows"] and n_top == prior["rows_at_100"],
            "Exact original streams and candidate scope")
    for part, expected in FIT.items():
        require(tuple(counts[part][key] for key in ("images", "supported_candidates", "paired_sites")) == expected,
                "Exact fixed point denominators")
    finish(a, dict(status=CAPTURE_STATUS, partitions={
                part: dict(counts[part], cue_index_sha256=cue_index[part].hexdigest()) for part in FIT},
                detector_forwards=915, exact_post_nms_rows=n_all, exact_probability_rows=n_top,
                exact_candidate_audit_images=915, exact_author_mask_replays=n_all,
                projection_seed=5297, projection_sha256=digest(weight_np.tobytes()),
                cue_dimensions=6, spatial_feature_shape=[1, 256, 160, 160], synthetic_checks=2,
                author_commit=base.AUTHOR_COMMIT, ema_sha256=native.EMA,
                inherited_gpu_free_at_start_bytes=peak_start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                forward_postprocess_sampling_seconds=seconds, elapsed_seconds=perf_counter()-start))


def verify(a):
    start = perf_counter()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "Independent CPU-only cue verification")
    _, points = original_inputs()
    term = HERE / "observed_capture_active/TERMINAL.json"
    closed_doc = json.loads(term.read_bytes())
    receipt, terminal_sha, worker = closed(term, HERE / "actual/RESULT_SAFE.json",
                                           closed_doc["result_sha256"], "capture")
    require(worker == receipt["worker_sha256"] == digest(Path(__file__).read_bytes())
            and receipt["status"] == CAPTURE_STATUS and receipt["protocol_sha256"] == LOCK
            and receipt["point_result_sha256"] == POINT_RESULT and receipt["capture_result_sha256"] == NATIVE
            and no_labels(receipt) and receipt["detector_forwards"] == 915 and receipt["exact_probability_rows"] == 91500
            and receipt["projection_sha256"] == digest(projection().tobytes()), "Closed cue capture bindings")
    checked = 0
    for part, (n, support_n, pair_n) in FIT.items():
        folder = HERE / "actual" / part
        parsed = [json.loads(raw) for raw in pin(folder / "CUE_INDEX_PRIVATE.jsonl",
                  receipt["partitions"][part]["cue_index_sha256"]).splitlines()]
        legacy = [meta for assigned, meta in points.values() if assigned == part]
        require(len(parsed) == len(legacy) == n and len(list((folder / "CUES_PRIVATE").glob("*.npz"))) == n,
                "Complete split files")
        sums = Counter()
        for cue_row, original in zip(parsed, legacy, strict=True):
            ordinal = original["native_archive_ordinal"]
            expect = dict(original, filename=f"CUES_PRIVATE/{ordinal:04d}.npz", bytes=cue_row["bytes"],
                          sha256=cue_row["sha256"], point_sha256=original["sha256"])
            require(cue_row == expect, "Exact partition index row")
            raw = pin(folder / cue_row["filename"], cue_row["sha256"])
            require(len(raw) == cue_row["bytes"], "Exact feature archive size")
            ranks, offsets, coordinates = point_fields(part, original)
            with np.load(BytesIO(raw), allow_pickle=False) as archive:
                cue = archive["cue"]
                require(set(archive.files) == {"native_archive_ordinal", "native_ranks", "point_offsets",
                        "signed_coordinates", "cue", "point_sha256"}
                        and archive["native_archive_ordinal"].item() == ordinal
                        and archive["native_archive_ordinal"].dtype == np.int32
                        and archive["point_sha256"].item() == original["sha256"].encode("ascii")
                        and all(archive[key].dtype == expected.dtype and np.array_equal(archive[key], expected)
                                for key, expected in (("native_ranks", ranks), ("point_offsets", offsets),
                                                      ("signed_coordinates", coordinates)))
                        and cue.shape == (*coordinates.shape, 6) and cue.dtype == np.float32
                        and bool(np.isfinite(cue).all()), "Independent exact cue/coordinate readback")
            sums.update(images=1, supported_candidates=len(ranks), paired_sites=len(coordinates), cue_bytes=len(raw),
                        endpoints_by_side=2*len(coordinates), candidates_without_pairs=100-len(ranks))
            checked += 1
        require(tuple(sums[key] for key in ("images", "supported_candidates", "paired_sites")) == (n, support_n, pair_n)
                and all(receipt["partitions"][part][key] == val for key, val in sums.items()),
                "Expected partition scientific denominators")
    a.output.mkdir()
    finish(a, dict(status=VERIFY_STATUS, spatial_capture_result_sha256=closed_doc["result_sha256"],
                   spatial_capture_terminal_sha256=terminal_sha, partitions=receipt["partitions"],
                   detector_forwards=0, integrity_images=checked, elapsed_seconds=perf_counter()-start))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("capture", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--point-source", type=Path, required=True)
    parser.add_argument("--author-intake", type=Path, required=True)
    parser.add_argument("--reference-capture", type=Path, required=True)
    parser.add_argument("--capture-adapter", type=Path, required=True)
    parser.add_argument("--ema-acceptance", type=Path, required=True)
    parser.add_argument("--ema-checkpoint", type=Path, required=True)
    parser.add_argument("--codec-wheel", type=Path, required=True)
    a = parser.parse_args()
    a.output = a.output.resolve()
    global BASE, POINTS, AUTH, REFERENCE, MIRROR, ACCEPT, EMA, CODEC
    BASE = a.native_source.resolve()
    POINTS = a.point_source.resolve()
    AUTH = a.author_intake.resolve()
    REFERENCE = a.reference_capture.resolve()
    MIRROR = a.capture_adapter.resolve()
    ACCEPT = a.ema_acceptance.resolve()
    EMA = a.ema_checkpoint.resolve()
    CODEC = a.codec_wheel.resolve()
    require(sys.flags.isolated and sys.dont_write_bytecode and a.output.parent == HERE
            and not a.output.exists(), "Isolated private fresh operation")
    pin(HERE / "PRELOCK.md", LOCK)

    def deny_network(event, _args):
        if event.startswith("socket."):
            raise PermissionError("Offline locked spatial cue acquisition")

    sys.addaudithook(deny_network)
    (actual_capture if a.mode == "capture" else verify)(a)


if __name__ == "__main__":
    main()



