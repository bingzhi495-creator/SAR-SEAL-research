"""Integer-threshold score-free coverage over exact closed endpoint predictions."""
import base64
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
import types

MIB = 1024 ** 2
THRESHOLDS = {"0.50": (1, 2), "0.75": (3, 4), "0.90": (9, 10)}
CASES = ("INDEPENDENT_E24", "E24", "E50", "E100")
GT_SHA = "5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55"
HELPERS_SHA = "54cab5b7ae5aa5b0293954a738ad88d9716fd7fa0ab23b87373715417099899d"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def measure(intersection, union, numerator, denominator):
    """Return per-GT coverage and maximum one-to-one cardinality, inclusive at threshold."""
    need(intersection.shape == union.shape and intersection.ndim == 2,
         "Expected aligned intersection and union matrices")
    adjacency = (union > 0) & (denominator * intersection >= numerator * union)
    edges = [row.nonzero()[0].tolist() for row in adjacency]
    matched = {}

    def augment(target, seen):
        for candidate in edges[target]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate not in matched or augment(matched[candidate], seen):
                matched[candidate] = target
                return True
        return False

    for target in range(len(edges)):
        augment(target, set())
    return [bool(row) for row in edges], len(matched)


def intersection_union(truth, predictions, mask, np):
    target_count, candidate_count = len(truth), len(predictions)
    if not target_count or not candidate_count:
        empty = np.empty((target_count, candidate_count), dtype=np.int64)
        return empty, empty.copy()
    predicted_area = np.asarray(mask.area(predictions), dtype=np.int64)
    target_area = np.asarray(mask.area(truth), dtype=np.int64)
    area_sum = target_area[:, None] + predicted_area[None, :]
    need((area_sum < 2 ** 31).all(), "Integer area below exact inversion bound")
    ordinary = np.asarray(mask.iou(predictions, truth, [0] * target_count)).T
    need(ordinary.shape == area_sum.shape and np.isfinite(ordinary).all()
         and ((0 <= ordinary) & (ordinary <= 1)).all(), "Finite ordinary IoU")
    estimate = ordinary * area_sum / (1 + ordinary)
    inter = np.rint(estimate).astype(np.int64)
    union = area_sum - inter
    reconstructed = np.divide(inter, union, out=np.zeros_like(ordinary), where=union > 0)
    need((np.abs(estimate - inter) < 1e-6).all()
         and np.allclose(reconstructed, ordinary, rtol=0, atol=1e-14)
         and ((0 <= inter) & (inter <= np.minimum(target_area[:, None], predicted_area[None, :]))).all(),
         "C IoU implies unique valid integer intersection and roundtrip")
    return inter, union


def as_rle(value, height, width, mask):
    if isinstance(value, list):
        return mask.merge(mask.frPyObjects(value, height, width))
    if isinstance(value["counts"], list):
        return mask.frPyObjects(value, height, width)
    need(value["size"] == [height, width], "Compressed mask at original raster size")
    return dict(size=value["size"], counts=value["counts"].encode("ascii")
                if isinstance(value["counts"], str) else value["counts"])


def aggregate(truth, predictions, threshold, image_order, mask, np):
    flags, injectable, small_injectable, other_injectable = [], 0, 0, 0
    for image, height, width in image_order:
        targets = truth.get(image, ())
        candidates = predictions.get(image, ())
        target_rle = [x[1] for x in targets]
        candidate_rle = [as_rle(x, height, width, mask) for x in candidates]
        inter, union = intersection_union(target_rle, candidate_rle, mask, np)
        covered, matched = measure(inter, union, *threshold)
        small = np.asarray([row[0] for row in targets], dtype=bool)
        small_injectable += measure(inter[small], union[small], *threshold)[1]
        other_injectable += measure(inter[~small], union[~small], *threshold)[1]
        flags.extend(covered)
        injectable += matched
    return flags, injectable, small_injectable, other_injectable


def run(request):
    started = time.monotonic()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and sys.flags.isolated and sys.flags.no_site,
         "CPU-only isolated diagnostic process")
    need(Path("/proc/sys/kernel/random/boot_id").read_text().strip() == request["boot_id"]
         and sys.executable == request["python"], "Exact remote boot and interpreter")
    root = Path(request["output_root"])
    source = base64.b64decode(request["helpers_base64"], validate=True)
    need(sha256(source).hexdigest() == HELPERS_SHA, "Exact previously accepted readback helpers")
    helpers = types.ModuleType("_closed_dgx_geometry_reads")
    exec(compile(source, "score_closed.py", "exec"), helpers.__dict__)
    runtime = types.ModuleType("_closed_dgx_geometry_runtime")
    data = helpers.read(request["runtime"], root / "E24/SOURCE/runtime.py", MIB)
    exec(compile(data, "runtime.py", "exec"), runtime.__dict__)
    need(all(p.resolve(strict=True) == p for p in runtime.SITES), "Pinned venv paths")
    sys.path.extend(map(str, runtime.SITES))
    import numpy as np
    from pycocotools import mask

    need(version("pycocotools") == "2.0.11", "Exact official mask implementation")
    gt_ref = request["gt"]
    need(gt_ref["sha256"] == GT_SHA, "Frozen integer ground-truth bytes")
    gt = json.loads(helpers.read(gt_ref, Path(gt_ref["path"]), MIB))
    images = [(row["id"], row["height"], row["width"]) for row in gt["images"]]
    dimensions = {image: (height, width) for image, height, width in images}
    need(len(images) == len(dimensions) == 187 and len(gt["annotations"]) == 380
         and [c["id"] for c in gt["categories"]] == [1], "Full unchanged DEV187 population")
    truth = defaultdict(list)
    crowd_annotations = 0
    for row in gt["annotations"]:
        image = row["image_id"]
        height, width = dimensions[image]
        need(row["category_id"] == 1 and row["area"] > 0, "Single category nonempty ground truth")
        crowd_annotations += bool(row.get("iscrowd", 0))
        truth[image].append((row["area"] < 1024, as_rle(row["segmentation"], height, width, mask)))
    small_flags = [small for image, _, _ in images for small, _ in truth.get(image, ())]
    denominator = len(small_flags)
    need(denominator == 380, "Full exact per-instance order")
    small_count = sum(small_flags)
    summaries, all_flags, pred_shas, pred_counts = {}, {}, {}, {}
    for name in CASES:
        case = root / name
        owner = json.loads(helpers.read(request["owners"][name], case / "OWNER_RECEIPT.json", MIB))
        worker = request["workers"][name]
        helpers.check_owner(owner, request["boot_id"], request["spec_refs"][name], worker)
        result = json.loads(helpers.read(worker, case / "RESULT.json", MIB))["result"]
        accepted = request["accepted_cases"][name]
        pred_ref = result["predictions_ref"]
        need(pred_ref["sha256"] == accepted["predictions_sha256"]
             and result["prediction_count"] == accepted["mapped_rows"]
             and result["checkpoint_sha256"] == accepted["checkpoint_sha256"],
             "Exact previously accepted endpoint predictions and checkpoint identity")
        rows = json.loads(helpers.read(pred_ref, case / "ARTIFACTS/PREDICTIONS.json", 256 * MIB))
        need(len(rows) == accepted["mapped_rows"], "Full exact prediction rows")
        predictions = defaultdict(list)
        for row in rows:
            need(row["image_id"] in dimensions and row["category_id"] == 1,
                 "Only mapped predictions for the locked DEV images")
            predictions[row["image_id"]].append(row["segmentation"])
        need(all(len(items) <= 100 for items in predictions.values()), "Native candidate top-k bound")
        pred_shas[name], pred_counts[name] = pred_ref["sha256"], len(rows)
        all_flags[name], summaries[name] = {}, {}
        for label, threshold in THRESHOLDS.items():
            flags, unique, unique_small, unique_other = aggregate(truth, predictions, threshold, images, mask, np)
            need(len(flags) == denominator and unique <= sum(flags), "Unique coverage bound")
            covered_small = sum(c and small for c, small in zip(flags, small_flags, strict=True))
            all_flags[name][label] = flags
            summaries[name][label] = dict(
                per_gt_max=sum(flags), max_injective=unique,
                small_per_gt_max=covered_small, other_per_gt_max=sum(flags) - covered_small,
                small_only_max_injective=unique_small, other_only_max_injective=unique_other)
    changes = {}
    for label in THRESHOLDS:
        paired = list(zip(all_flags["E24"][label], all_flags["E100"][label], small_flags, strict=True))
        scopes = {"all": paired, "small": [row for row in paired if row[2]],
                  "other": [row for row in paired if not row[2]]}
        changes[label] = {
            name: dict(gained=sum(not a and b for a, b, _ in group),
                       lost=sum(a and not b for a, b, _ in group),
                       both_covered=sum(a and b for a, b, _ in group),
                       neither_covered=sum(not a and not b for a, b, _ in group))
            for name, group in scopes.items()
        }
        counts = changes[label]["all"]
        need(counts["gained"] - counts["lost"] == summaries["E100"][label]["per_gt_max"]
             - summaries["E24"][label]["per_gt_max"], "Paired transitions close against the marginal totals")
    return dict(status="ACTUAL_DGX_CLOSED_ENDPOINT_SCORE_FREE_GEOMETRY", images=len(images),
                GT_instances=denominator, small_GT=small_count, other_GT=denominator-small_count,
                crowd_annotations=crowd_annotations, input_role="INTERNAL_DEV187_INTEGER_GT",
                geometry="ordinary mask intersection over union, with exact inclusive rational thresholds",
                scope="mapped retained predictions only, ignores scores; no raw-query recovery claim",
                small_definition="recorded annotation area < 1024; subset matching excludes other GT",
                prediction_sha256=pred_shas, prediction_counts=pred_counts, integer_GT_sha256=GT_SHA,
                coverage=summaries, same_run_E100_vs_E24_per_GT=changes,
                seconds=time.monotonic()-started, GPU=False, AP=False, new_forward=False, training=False, TEST=False)


if __name__ == "__main__":
    print(json.dumps(run(json.load(sys.stdin)), sort_keys=True, allow_nan=False), flush=True)
