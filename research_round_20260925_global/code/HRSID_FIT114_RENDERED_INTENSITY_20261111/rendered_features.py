"""Add prediction-conditioned rendered-intensity features to closed FIT114 points."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from zipfile import ZipFile


STATUS = "P318_HRSID_FIT114_RENDERED_INTENSITY_FEATURES_COMPLETE"
P316_STATUS = "P316_HRSID_FIT114_SPARSE_SIGNED_LOGIT_CAPTURE_COMPLETE"
P316_REPORT_SHA = "091333f7790d5fb5b7b9791b40777b55b480e01dd116dc92e51d27da36128e83"
P316_TERMINAL_SHA = "d4c10b4b5eec73dbb3d8a0927dd527ec9fcdb97d0c18a0a6bdde824d70300e95"
P316_SOURCE_SHA = "c129e18d6ffaab843f2901581425711f6daca12deeaace0ee0f1150b3dcd7ba1"
P268 = "HRSID_FIT114_INPUT_SHIFT_20261111/input_shift.py"
P268_SHA = "bf32a994b00ce9e9e30c976b8aa7e16b0dd2dce3d4e4942c2305b1f7d07f9c71"
SIZE, RADIUS, MIB = 800, 2, 2**20
FEATURE_NAMES = ("foreground_minus_background", "center_minus_foreground", "center_minus_background",
                 "local_luminance_dispersion")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(record):
    return (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def require_unchanged(path, stream, original_stat):
    need(identity(path.stat()) == identity(os.fstat(stream.fileno())) == identity(original_stat),
         "Frozen input changed during feature extraction")


def verify_stream(path, stream, expected_size, expected_sha):
    original_stat = os.fstat(stream.fileno())
    need(original_stat.st_size == expected_size, "Exact frozen input byte count required")
    digest = sha256()
    for chunk in iter(lambda: stream.read(MIB), b""):
        digest.update(chunk)
    need(digest.hexdigest() == expected_sha, "Exact frozen input digest required")
    require_unchanged(path, stream, original_stat)
    stream.seek(0)
    return original_stat


def augment_image(record, rgb, codec):
    """Read only each fixed point's same-side and opposite-side rendered intensity."""
    import numpy as np

    need(rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8, "Original decoded RGB uint8 required")
    need(set(record) == {"image_id", "group", "encoded_sha256", "core", "stress", "candidates"},
         "Exact original candidate image record required")
    height, width = rgb.shape[:2]
    candidates = record["candidates"]
    last_rank, query_ids, points = 0, set(), []
    for candidate in candidates:
        need(set(candidate) == {"native_rank", "query_id", "score", "category_id", "segmentation", "points"},
             "Exact original native candidate fields required")
        rank, query = candidate["native_rank"], candidate["query_id"]
        score, rle = candidate["score"], candidate["segmentation"]
        need(type(rank) is int and last_rank < rank <= 100 and type(query) is int and 0 <= query < 164
             and query not in query_ids and type(score) is float and 0. <= score <= 1.
             and type(candidate["category_id"]) is int and candidate["category_id"] == 1
             and set(rle) == {"size", "counts"} and rle["size"] == [height, width]
             and isinstance(rle["counts"], str) and bool(rle["counts"]) and rle["counts"].isascii(),
             "Exact ordered native candidate identity required")
        last_rank = rank
        query_ids.add(query)
        side_counts = Counter()
        positions = set()
        for p in candidate["points"]:
            need(set(p) == {"y", "x", "inside", "logits"} and type(p["y"]) is type(p["x"]) is int
                 and 0 <= p["y"] < height and 0 <= p["x"] < width and (p["y"], p["x"]) not in positions
                 and type(p["inside"]) is bool and isinstance(p["logits"], list) and len(p["logits"]) == 9
                 and all(type(z) is float and math.isfinite(z) for z in p["logits"])
                 and (p["logits"][0] > 0.) is p["inside"], "Exact fixed finite signed point fields required")
            positions.add((p["y"], p["x"]))
            side_counts[p["inside"]] += 1
            points.append(p)
        need(all(count <= 32 for count in side_counts.values()), "Exact captured point cap per side required")

    if not points:
        return dict(record), Counter(candidate_rows=len(candidates), points=0)
    # This is a fixed luminance of the delivered rendered pixels, scaled to [0,1].
    rgb_int = rgb.astype(np.uint16)
    lum = (77*rgb_int[..., 0] + 150*rgb_int[..., 1] + 29*rgb_int[..., 2]) / (256. * 255.)
    y = np.array([p["y"] for p in points], dtype=np.intp)
    x = np.array([p["x"] for p in points], dtype=np.intp)
    d = np.arange(-RADIUS, RADIUS+1)
    dy, dx = np.repeat(d, len(d)), np.tile(d, len(d))
    rows, cols = y[:, None]+dy, x[:, None]+dx
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows = rows.clip(0, height-1)
    cols = cols.clip(0, width-1)
    values = lum[rows, cols]
    n = valid.sum(axis=1)
    whole_mean = (values*valid).sum(axis=1) / n
    sigma = np.sqrt((((values-whole_mean[:, None])**2)*valid).sum(axis=1) / n)
    scale = np.maximum(sigma, 4./255.)
    center = lum[y, x]

    output, counters = [], Counter(candidate_rows=len(candidates), points=len(points))
    start = 0
    for candidate in candidates:
        count = len(candidate["points"])
        if not count:
            output.append(candidate)
            continue
        idx = slice(start, start+count)
        rle = candidate["segmentation"]
        pred = codec.decode(dict(rle, counts=rle["counts"].encode("ascii"))).astype(bool, copy=False)
        need(pred.shape == (height, width) and
             np.array_equal(pred[y[idx], x[idx]], [p["inside"] for p in candidate["points"]]),
             "Captured prediction-side signs must equal its native COCO mask")
        mask = pred[rows[idx], cols[idx]]
        foreground, background = mask & valid[idx], ~mask & valid[idx]
        nf, nb = foreground.sum(axis=1), background.sum(axis=1)
        missing_fg, missing_bg = nf == 0, nb == 0
        # Use the window mean for a missing side and retain all original points.
        mean_fg = np.where(missing_fg, whole_mean[idx], (values[idx]*foreground).sum(axis=1) / np.maximum(1, nf))
        mean_bg = np.where(missing_bg, whole_mean[idx], (values[idx]*background).sum(axis=1) / np.maximum(1, nb))
        features = np.column_stack((
            np.clip((mean_fg-mean_bg)/scale[idx], -4., 4.),
            np.clip((center[idx]-mean_fg)/scale[idx], -4., 4.),
            np.clip((center[idx]-mean_bg)/scale[idx], -4., 4.),
            np.clip(4.*sigma[idx], 0., 1.),
        ))
        need(bool(np.isfinite(features).all()), "Every fixed feature vector must be finite")
        enriched = [dict(point, radiometric=vector) for point, vector in
                    zip(candidate["points"], features.tolist(), strict=True)]
        output.append(dict(candidate, points=enriched))
        counters["points_missing_foreground"] += int(missing_fg.sum())
        counters["points_missing_background"] += int(missing_bg.sum())
        counters["points_clipped_window"] += int((n[idx] < (2*RADIUS+1)**2).sum())
        counters["points_sigma_floor"] += int((sigma[idx] < 4./255.).sum())
        start += count
    need(start == len(points) and len(output) == len(candidates), "Candidate and point identity must reconcile")
    return dict(record, candidates=output), counters


def load_scope(code_root):
    path = code_root / P268
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == P268_SHA, "Exact released FIT114 input reader source required")
    reader = ModuleType("_p318_exact_p268")
    reader.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), reader.__dict__)
    return reader


def read_closed_capture(args, reader):
    need((args.p316_report_sha256, args.p316_terminal_sha256) == (P316_REPORT_SHA, P316_TERMINAL_SHA),
         "Exact original closed P316 report and terminal identities required")
    report = json.loads(reader.pinned(args.p316_root / "CAPTURE_SAFE.json", P316_REPORT_SHA, 32*1024))
    terminal = json.loads(reader.pinned(args.p316_terminal, P316_TERMINAL_SHA, 32*1024))
    need(terminal["original_os_wait"] is terminal["worker_closed"] is True
         and type(terminal["original_os_exit"]) is int and terminal["original_os_exit"] == 0
         and terminal["resource_stop"] is False and terminal["source_sha256"] == P316_SOURCE_SHA,
         "Exact original successful P316 CPU worker closure required")
    need(report["status"] == P316_STATUS and report["device"] == "cpu" and report["FIT"] is True
         and report["cpu_native_signed_raster_exact"] is True and report["optimizer_updates"] == 0
         and report["images"] == report["forward_calls"] == 114 and report["core_images"] == 109
         and report["stress_images"] == 5 and report["boundary_radius_original_pixels"] == 1
         and report["max_points_per_side"] == 32
         and all(report[name] is False for name in ("GT_opened", "DEV", "TEST", "training", "AP"))
         and report["fit_bundle_sha256"] == reader.BUNDLE_SHA and report["historical_cohort_sha256"] == reader.COHORT_SHA,
         "Original closed FIT114 no-GT P316 candidate scope required")
    return report


def fit_image(archive, row, reader):
    import numpy as np
    from PIL import Image

    name = row["native_basename"]
    need(re.fullmatch(r"P\d+_[A-Za-z0-9_]+\.png", name) and name.split("_", 1)[0] == row["group"],
         "Exact native FIT-only filename and source prefix required")
    data = reader.member(archive, "images/"+name, row["encoded_sha256"], 4*MIB)
    with Image.open(BytesIO(data)) as image:
        need(len(data) == row["encoded_bytes"] and image.size == (SIZE, SIZE) and image.format == "PNG",
             "Exact closed original 800-square FIT PNG required")
        rgb = np.asarray(image.convert("RGB"))
    need(sha256(rgb.tobytes(order="C")).hexdigest() == row["decoded_rgb_sha256"],
         "Exact decoded P316 original RGB bytes required")
    return rgb


def enrich(args, reader, report):
    from pycocotools import mask as codec

    source = args.p316_root/"CANDIDATES_PRIVATE.jsonl"
    out_sha, counts = sha256(), Counter()
    with TemporaryDirectory(prefix=".rendered_features_", dir=args.output.parent) as temp:
        target = Path(temp)/"features"
        target.mkdir()
        feature_file = target/"FEATURES_PRIVATE.jsonl"
        with args.bundle.open("rb") as bundle, source.open("rb") as original, feature_file.open("xb") as output:
            bundle_stat = verify_stream(args.bundle, bundle, 151_320_868, reader.BUNDLE_SHA)
            candidate_stat = verify_stream(source, original, report["candidate_stream_bytes"], report["candidate_stream_sha256"])
            with ZipFile(bundle) as archive:
                manifest = json.loads(reader.member(archive, "COHORT_PRIVATE.json", reader.COHORT_SHA, MIB))
                roster = manifest["images"]
                need(manifest["fit_only"] is True and len(roster) == 114
                     and sum(row["core"] for row in roster) == 109 and sum(row["stress"] for row in roster) == 5,
                     "Exact closed original FIT114-only roster required")
                for raw, row in zip_longest(original, roster):
                    need(raw is not None and row is not None and 0 < len(raw) <= 4*MIB and raw.endswith(b"\n"),
                         "One complete original candidate block per exact FIT114 image required")
                    record = json.loads(raw)
                    keys = ("image_id", "group", "encoded_sha256", "core", "stress")
                    need(tuple(record[key] for key in keys) == tuple(row[key] for key in keys),
                         "Exact original FIT114 source group and row order")
                    rgb = fit_image(archive, row, reader)
                    result, observed = augment_image(record, rgb, codec)
                    line = encoded(result)
                    need(output.write(line) == len(line), "Complete feature stream row required")
                    out_sha.update(line)
                    counts.update(observed)
                    counts["images"] += 1
                    if counts["images"] % 20 == 0:
                        print(json.dumps(dict(event="P318_RENDERED_FEATURE_PROGRESS", images=counts["images"])), flush=True)
            output.flush()
            os.fsync(output.fileno())
            require_unchanged(args.bundle, bundle, bundle_stat)
            require_unchanged(source, original, candidate_stat)
        need(counts["images"] == report["images"] == 114
             and counts["candidate_rows"] == report["historical_gpu_vs_current_cpu"]["cpu_rows"]
             and counts["points"] == report["counters"]["inner_points"]+report["counters"]["outer_points"],
             "Exact original CPU candidate and fixed point denominators required")
        result = dict(status=STATUS, images=114, core_images=109, stress_images=5, GT_opened=False, FIT=True,
                      DEV=False, TEST=False, training=False, model_forward_calls=0, device="cpu",
                      candidate_values_preserved=True, candidate_rows=counts["candidate_rows"], points=counts["points"],
                      feature_stream_sha256=out_sha.hexdigest(), feature_stream_bytes=feature_file.stat().st_size,
                      p316_candidate_stream_sha256=report["candidate_stream_sha256"], p316_report_sha256=P316_REPORT_SHA,
                      p316_original_wait_sha256=P316_TERMINAL_SHA, fit_bundle_sha256=reader.BUNDLE_SHA,
                      cohort_sha256=reader.COHORT_SHA, feature_names=FEATURE_NAMES, support_window_pixels=5,
                      normalized_contrast_clip=4., luminance_sigma_floor=4./255.,
                      intensity="uint8 decoded rendered RGB weighted 77,150,29 then divided by 256*255",
                      dispersion="clip(4 * local population luminance standard deviation,0,1)",
                      window_scope="in-canvas square, original predicted mask foreground and background",
                      missing_side_policy="whole local mean when a prediction side is absent",
                      support_reader_sha256=P268_SHA, counters=dict(counts))
        raw = encoded(result)
        with (target/"FEATURES_SAFE.json").open("xb") as stream:
            need(stream.write(raw) == len(raw), "Complete feature receipt required")
            stream.flush()
            os.fsync(stream.fileno())
        need(not args.output.exists() and not args.output.is_symlink(), "Fresh output directory changed")
        target.rename(args.output)
    return dict(status=STATUS, report_sha256=sha256(raw).hexdigest(), feature_stream_sha256=out_sha.hexdigest())


def deny_network(event, _):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise PermissionError("Offline FIT-only rendered feature extraction required")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code-root", "p316-root", "p316-terminal", "bundle", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("p316-report-sha256", "p316-terminal-sha256"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args(argv)
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.flags.isolated == sys.flags.utf8_mode == 1
         and sys.dont_write_bytecode, "Exact isolated CPU invocation required")
    for path in (args.code_root, args.p316_root, args.p316_terminal, args.bundle):
        need(path.is_absolute() and path.resolve(strict=True) == path, "Exact canonical closed source path required")
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
         and not args.output.exists() and not args.output.is_symlink(), "Fresh canonical output required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    need({name: version(name) for name in ("numpy", "Pillow", "pycocotools")}
         == dict(numpy="1.26.4", Pillow="12.3.0", pycocotools="2.0.11"), "Exact original FIT CPU dependencies required")
    sys.addaudithook(deny_network)
    reader = load_scope(args.code_root)
    report = read_closed_capture(args, reader)
    print(json.dumps(enrich(args, reader, report), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
