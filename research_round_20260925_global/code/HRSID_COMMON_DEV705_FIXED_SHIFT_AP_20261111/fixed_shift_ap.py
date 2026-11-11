"""Re-score fixed vertical translations of closed common-support HRSID masks."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


P311_PATH = "HRSID_COMMON_DEV705_STRICT_RESIDUAL_20261111/strict_residual.py"
P311_SHA = "6fbefd4e62e7715518c1e460db9b0c32f5c407f11adef8de6cd5737b65346b54"
STATUS = "HRSID_COMMON_DEV705_ALL_CANDIDATES_FIXED_VERTICAL_SHIFT_AP_ACTUAL"
METRICS = ("AP", "AP50", "AP75", "AP90", "APs", "APm", "APl", "AR100")
DIRECTIONS = {"up_one": -1, "down_one": 1}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def runs(text):
    """Decode COCO's signed base-32 delta counts without decoding the mask."""
    counts = []
    value = shift = 0
    for byte in text.encode("ascii"):
        code = byte - 48
        need(0 <= code <= 63, "Malformed compressed COCO run")
        value |= (code & 31) << shift
        shift += 5
        if code & 32:
            continue
        if code & 16:
            value -= 1 << shift
        if len(counts) > 2:
            value += counts[-2]
        need(value >= 0, "Negative decoded COCO run")
        counts.append(value)
        value = shift = 0
    need(counts and shift == 0, "Incomplete compressed COCO run")
    return counts


def translate(rle, bbox, dy):
    """Move a non-edge mask in column-major coordinates, preserving its area."""
    need(dy in (-1, 1), "A single fixed vertical pixel displacement is required")
    height, width = rle["size"]
    x, y, w, h = map(int, bbox)
    need(tuple(bbox) == (x, y, w, h), "Integer prediction bounding box required")
    if w == 0 or h == 0:
        return rle, "empty_unchanged"
    need(0 <= x < x+w <= width and 0 <= y < y+h <= height, "Prediction bbox outside its canvas")
    if not (0 <= y + dy and y + h + dy <= height):
        return rle, "edge_unchanged"
    counts = runs(rle["counts"])
    need(sum(counts) == height*width, "RLE runs must account for every image pixel")
    # In Fortran order this is an exact flat offset if no pixel crosses a row boundary.
    if len(counts) % 2 == 0:
        counts.append(0)
    counts[0] += dy
    counts[-1] -= dy
    need(counts[0] >= 0 and counts[-1] >= 0 and sum(counts) == height*width,
         "The no-clipping translation produced an invalid background run")
    if counts[-1] == 0:
        counts.pop()
    encoded = mask.frPyObjects(dict(size=[height, width], counts=counts), height, width)
    return dict(size=encoded["size"], counts=encoded["counts"].decode("ascii")), "shifted"


def dense_spot(original, shifted, dy, status):
    baseline = mask.decode(original)
    expected = baseline
    if status == "shifted":
        expected = np.zeros_like(baseline)
        if dy < 0:
            expected[:-1] = baseline[1:]
        else:
            expected[1:] = baseline[:-1]
    need(np.array_equal(mask.decode(shifted), expected), "Compressed translation failed full decode spot check")


def build_variants(rows):
    """Use predictions alone to construct complete, rank-preserving streams."""
    original = tuple(rows)
    rles = [row["segmentation"] for row in original]
    need(all(rle["size"] == [800, 800] for rle in rles), "Exact original canvas required")
    areas, boxes = mask.area(rles), mask.toBbox(rles)
    variants = {"raw": original}
    census = {"raw": dict(mapped_rows=len(rows), empty_rows=int((areas == 0).sum()))}
    for direction, dy in DIRECTIONS.items():
        output, edited_boxes, work, spots = [], [], Counter(), Counter()
        for row, bbox in zip(original, boxes, strict=True):
            rle = row["segmentation"]
            shifted, status = translate(rle, bbox, dy)
            work[status] += 1
            output.append(row if shifted is rle else dict(row, segmentation=shifted))
            edited_boxes.append((bbox[0], bbox[1] + (dy if status == "shifted" else 0), bbox[2], bbox[3]))
            if spots[status] < 8:
                dense_spot(rle, shifted, dy, status)
                spots[status] += 1
        result_rles = [row["segmentation"] for row in output]
        need(np.array_equal(mask.area(result_rles), areas)
             and np.array_equal(mask.toBbox(result_rles), np.asarray(edited_boxes)),
             "One-pixel shifts must preserve every foreground count and move only the bbox y")
        need(len(output) == len(original)
             and work["empty_unchanged"] == census["raw"]["empty_rows"]
             and sum(work.values()) == len(original), "Complete per-direction candidate census required")
        variants[direction] = tuple(output)
        census[direction] = dict(work, full_decode_spots=dict(spots))
    return variants, census


def digest_rows(rows):
    digest = sha256()
    for row in rows:
        digest.update((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def score_rows(gt, rows, adapter):
    captured = []
    coco = COCO()
    coco.dataset = deepcopy(gt)
    image_ids = [row["id"] for row in gt["images"]]

    def observe(*args):
        evaluation = COCOeval(*args)
        captured.append(evaluation)
        return evaluation

    with redirect_stdout(StringIO()):
        coco.createIndex()
        values = adapter.evaluate_population(dict(COCO=COCO, COCOeval=observe), coco, list(rows), image_ids)
    evaluator, = captured
    params = evaluator.params
    area = list(params.areaRngLbl).index("all")
    max_det = list(params.maxDets).index(100)
    index90, = np.flatnonzero(np.isclose(params.iouThrs, .90))
    precision = evaluator.eval["precision"][index90, :, 0, area, max_det]
    valid = precision[precision > -1]
    need(len(values) == 12 and np.isfinite(values).all() and valid.size,
         "Original stock COCO segmentation metrics required")
    scores = dict(zip(("AP", "AP50", "AP75", "APs", "APm", "APl"), values[:6], strict=True))
    scores.update(AP90=float(valid.mean()), AR100=values[8], all12=list(values))
    start = area * len(image_ids)
    ranks = tuple(None if row is None else (row["image_id"], tuple(row["dtIds"]), tuple(row["dtScores"]))
                  for row in evaluator.evalImgs[start:start + len(image_ids)])
    return scores, ranks


def load_p311(code_root):
    path = code_root / P311_PATH
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == P311_SHA, "Exact closed analysis gate source required")
    module = ModuleType("_p365_p311")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def point_delta(reference, candidate):
    return {name: 100 * (candidate[name] - reference[name]) for name in METRICS}


def run(a):
    started = perf_counter()
    need(version("numpy") == "1.26.4" and version("pycocotools") == "2.0.11",
         "Exact original CPU evaluator runtime required")
    p311 = load_p311(a.code_root)
    need((a.pair_sha256, a.training_protocol, a.accepted_result_sha256)
         == (p311.PAIR_SHA, p311.TRAINING_PROTOCOL, p311.PRIMARY_SHA), "Exact closed common training pair required")
    closed = p311.load_closed(a.code_root)
    driver = closed.load_gate(a.code_root).load_driver()
    accepted = driver.read_json(a.accepted_result, a.accepted_result_sha256)
    need(accepted["status"] == p311.PRIMARY_STATUS and accepted["training_seed"] == 5202
         and accepted["official_TEST_opened"] is False, "Exact independent original DEV acceptance required")
    gt, native, center, provenance = closed.verified_inputs(a.code_root, a.pair, a.pair_sha256,
        native_arm_receipt=a.native_arm_receipt, center_arm_receipt=a.center_arm_receipt,
        training_protocol=a.training_protocol)
    need(len(gt["images"]) == 705 and len(gt["annotations"]) == 1782
         and provenance["training_pair_receipt_sha256"] == accepted["independent_training_pair_sha256"]
         and provenance["evaluation_protocol"] == accepted["evaluation_protocol"]
         and provenance["original_dev_gt_sha256"] == accepted["original_dev_gt_sha256"],
         "Exact accepted DEV scope, predictions and original integer COCO GT required")

    variants, work, pins = {}, {}, {}
    for arm, rows in (("native", native), ("center", center)):
        variants[arm], work[arm] = build_variants(rows)
        pins[arm] = {name: digest_rows(value) for name, value in variants[arm].items()}
    print(json.dumps(dict(event="P365_SIX_FULL_PREDICTION_STREAMS_FROZEN", sha256=pins)), flush=True)
    build_seconds = perf_counter() - started
    adapter = closed.load_diagnostics(a.code_root)["dev705_paired_mask_failure_scope"].load_official_adapter()
    metrics = {}
    for arm, streams in variants.items():
        metrics[arm] = {}
        original_ranks = None
        for name, rows in streams.items():
            result, ranks = score_rows(gt, rows, adapter)
            if name == "raw":
                original_ranks = ranks
                expected = accepted["scores"][arm]["primary"]
                need(np.allclose(result["all12"], expected["all12"], rtol=0, atol=1e-12)
                     and abs(result["AP90"] - expected["AP90"]) <= 1e-12,
                     "Raw all-candidate stock scoring does not reproduce accepted results")
            else:
                need(ranks == original_ranks, "Score, candidate rank or per-image layout changed")
            need(digest_rows(rows) == pins[arm][name], "Prediction stream changed after freezing")
            metrics[arm][name] = result
            print(json.dumps(dict(event="P365_STOCK_COCO_VARIANT_SCORED", arm=arm, variant=name)), flush=True)
    a_raw, b_raw = (metrics[arm]["raw"] for arm in ("native", "center"))
    a_up = metrics["native"]["up_one"]
    full = point_delta(a_raw, b_raw)
    residual = point_delta(a_up, b_raw)
    gain = point_delta(a_raw, a_up)
    return dict(
        status=STATUS, metrics=metrics, variants=work, prediction_sha256=pins,
        delta_vs_raw_points={arm: {name: point_delta(streams["raw"], result) for name, result in streams.items()}
                             for arm, streams in metrics.items()},
        contrast=dict(center_raw_minus_native_raw_points=full, center_raw_minus_native_up_one_points=residual,
                      native_up_one_recovery_fraction={key: gain[key] / full[key] if full[key] > 0 else None for key in METRICS},
                      near_recovery_absolute_AP_residual_limit_points=1.0,
                      near_recovery_AP=abs(residual["AP"]) <= 1.0),
        policy=dict(shifts_y_pixels=DIRECTIONS, all_mapped_rows=True, edge_or_empty_remains_raw=True,
                    foreground_area_preserved=True, score_order_class_and_candidate_metadata_preserved=True,
                    transform_uses_gt=False, all_six_streams_frozen_before_scoring=True),
        scope=dict(images=705, annotations=1782, DEV=True, FIT=False, official_TEST_opened=False,
                   COCO_mask_AP=True, model_forward_calls=0, optimizer_updates=0, transformed_masks_saved=False),
        provenance=provenance, build_including_admission_seconds=build_seconds, total_seconds=perf_counter() - started,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code-root", "pair", "native-arm-receipt", "center-arm-receipt", "accepted-result", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("pair-sha256", "training-protocol", "accepted-result-sha256"):
        parser.add_argument("--" + name, required=True)
    a = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and os.environ.get("OMP_NUM_THREADS") == "1"
         and a.code_root.resolve(strict=True) == a.code_root and a.output.is_absolute()
         and a.output.parent.resolve(strict=True) == a.output.parent
         and not a.output.exists() and not a.output.is_symlink(), "Fresh isolated CPU DEV output required")

    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("Closed offline diagnostic only")

    sys.addaudithook(offline)
    report = run(a)
    data = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with a.output.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(data).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
