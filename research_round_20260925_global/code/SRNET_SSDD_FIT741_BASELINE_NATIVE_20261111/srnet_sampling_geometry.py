"""Fit-only audit of SRNet's training and inference mask sampling coordinates."""
import argparse
from collections import defaultdict, deque
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


P335_SOURCE_SHA = "c078c51822ceabf107426d32fc500ccacef19cfe857b4c11cc2a6455be58c077"
P335_RESULT_SHA = "81c6f27647f19cdad1013f36620be457887b49713fe9fec425911df005102f68"
P335_TERMINAL_SHA = "931f56b9017ed7656938ba3f9247cfdbe61e3e1db80dd0d02a0644d772632f3d"
AUTHOR_PINS = {
    "mmrotate/models/dense_heads/SRNet_utils.py": "7d9210f7619fa31ad4bfacfc30e380cb1b4fce5d77e7217de15c4c14b5632c7f",
    "mmrotate/models/dense_heads/SRNet_head.py": "144552cf27b5970c46ff4e3af1b4e0f59b1b273daa0aadf89cdcc513c8e2dba1",
    "mmrotate/core/bbox/transforms.py": "899315d58f3004da58b639df9d369ce01de3e8ca33076100ed99a704ab6308d2",
}
STATUS = "P338_SRNET_FIT741_TRAIN_TEST_GRID_GEOMETRY_COMPLETE_V1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(path, digest, maximum=1024 * 1024):
    require(path.is_file() and 0 < path.stat().st_size <= maximum, "Bounded exact file required")
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, "Pinned source or closure digest differs")
    return raw


def analytic_nodes(boxes, n):
    """Analytic le90 positions and train minus test displacement, in image pixels."""
    boxes = np.asarray(boxes, dtype=np.float64)
    require(boxes.ndim == 2 and boxes.shape[1] == 5 and n >= 2
            and np.isfinite(boxes).all() and (boxes[:, 2:4] > 0).all(),
            "Finite positive le90 boxes and a grid size >= 2 required")
    cyx = boxes[:, None, :2]
    width, height, angle = (boxes[:, k, None] for k in (2, 3, 4))
    cos, sin = np.cos(angle), np.sin(angle)
    row, col = np.indices((n, n), dtype=np.float64).reshape(2, -1)

    def rotate(u, v):
        x, y = u * width, v * height
        return np.stack((x * cos - y * sin, x * sin + y * cos), axis=-1)

    train = cyx + rotate((col + 1) / (n + 1) - .5, (row + 1) / (n + 1) - .5)
    test = cyx + rotate((col + .5) / n - .5, (row + .5) / n - .5)
    displacement = rotate(((n - 1) / 2 - col) / (n * (n + 1)),
                          ((n - 1) / 2 - row) / (n * (n + 1)))
    return train, test, displacement


def oracle_metric(estimate, truth):
    require(estimate.shape == truth.shape and estimate.dtype == truth.dtype == np.bool_,
            "Paired source-size boolean masks required")
    intersection = int(np.count_nonzero(estimate & truth))
    fp = int(np.count_nonzero(estimate & ~truth))
    fn = int(np.count_nonzero(~estimate & truth))
    union = intersection + fp + fn
    return dict(intersection=intersection, union=union, fp=fp, fn=fn,
                iou=intersection / union if union else None,
                strict_075=bool(union and 4 * intersection >= 3 * union),
                strict_090=bool(union and 10 * intersection >= 9 * union))


def stock_geometry(boxes, n, stock, torch):
    """Run the actual FP32 author coordinates, verify independent FP64 algebra."""
    output = []
    errors = np.zeros(3, dtype=np.float64)
    for start in range(0, len(boxes), 32):
        block = boxes[start:start + 32]
        rois = torch.as_tensor(block, dtype=torch.float32)
        actual_train = stock.sample_points_train(rois, "le90", n).numpy().astype(np.float64)
        actual_test = stock.sample_points_test(rois, "le90", n).numpy().astype(np.float64)
        train, test, delta = analytic_nodes(block, n)
        errors = np.maximum(errors, [
            np.abs(actual_train - train).max(),
            np.abs(actual_test - test).max(),
            np.abs(actual_train - actual_test - delta).max(),
        ])
        displacement = np.linalg.norm(actual_train - actual_test, axis=-1)
        output.extend(np.stack((displacement.mean(axis=1),
                               np.sqrt((displacement ** 2).mean(axis=1)),
                               displacement.max(axis=1)), axis=-1))
    require(len(output) == len(boxes) and errors.max() <= .001,
            "FP32 author grid diverges from independent FP64 le90 geometry")
    return np.asarray(output), dict(train=float(errors[0]), test=float(errors[1]),
                                    train_minus_test=float(errors[2]))


def oracle_pair(box, truth, n, stock, torch):
    """Same polygon raster, target threshold, original box, and author paste."""
    import torch.nn.functional as F

    height, width = truth.shape
    side = max(height, width)
    canvas = torch.zeros((1, 1, side, side), dtype=torch.float32)
    canvas[0, 0, :height, :width] = torch.from_numpy(truth)
    rois = torch.as_tensor(box[None], dtype=torch.float32)
    metadata = dict(pad_shape=(side, side, 3), img_shape=(height, width, 3),
                    ori_shape=(height, width, 3))

    def sample(points):
        grid = points / (side / 2) - 1
        sampled = F.grid_sample(canvas, grid[:, None], align_corners=False)
        return (sampled.reshape(1, 1, n, n) >= .5).to(torch.float32)

    train = sample(stock.sample_points_train(rois, "le90", n))
    test = sample(stock.sample_points_test(rois, "le90", n))
    _, actual_target = stock.sample_feature_and_gt_masks(
        rois, canvas[0], canvas[:, 0], metadata, version="le90", output_size=n)
    require(torch.equal(train[0], actual_target), "Oracle train grid must reproduce author binary target")

    def paste(mask):
        return stock.get_mask(rois, [0], mask, version="le90", output_size=n,
                              img_meta=metadata, thr=.5, num_classes=1, rescale=False)[0][0]

    return dict(train=oracle_metric(paste(train), truth), test=oracle_metric(paste(test), truth),
                sampled_binary_nodes_changed=int(torch.count_nonzero(train != test)))


def strata(areas, boxes):
    small = np.asarray(areas) < 1024
    thin = np.max(boxes[:, 2:4], axis=1) >= 3 * np.min(boxes[:, 2:4], axis=1)
    return dict(all=np.ones(len(boxes), dtype=bool), small=small, other=~small,
                thin=thin, other_aspect=~thin, small_thin=small & thin)


def summarize_geometry(values):
    if not len(values):
        return {"targets": 0}
    maximum = values[:, 2]
    return dict(targets=len(values), mean_per_target_node_displacement_px=float(values[:, 0].mean()),
                mean_per_target_rms_displacement_px=float(values[:, 1].mean()),
                median_per_target_max_displacement_px=float(np.median(maximum)),
                p90_per_target_max_displacement_px=float(np.quantile(maximum, .90)),
                max_displacement_px=float(maximum.max()),
                targets_max_displacement_gt_half_px=int(np.count_nonzero(maximum > .5)),
                targets_max_displacement_gt_one_px=int(np.count_nonzero(maximum > 1)))


def summarize_oracle(rows):
    valid = [row for row in rows if row["train"]["iou"] is not None
             and row["test"]["iou"] is not None]
    result = dict(targets=len(rows), valid=len(valid),
                  empty_truth=sum(row["empty_truth"] for row in rows))
    if not valid:
        return result
    train = np.asarray([row["train"]["iou"] for row in valid])
    test = np.asarray([row["test"]["iou"] for row in valid])
    delta = test - train
    result.update(train_node_mean_iou=float(train.mean()), test_node_mean_iou=float(test.mean()),
                  test_minus_train_mean_iou=float(delta.mean()),
                  test_minus_train_median_iou=float(np.median(delta)),
                  test_wins=int(np.count_nonzero(delta > 0)),
                  ties=int(np.count_nonzero(delta == 0)),
                  train_wins=int(np.count_nonzero(delta < 0)),
                  targets_with_changed_sampled_binary_nodes=sum(
                      row["sampled_binary_nodes_changed"] > 0 for row in valid),
                  train_ge_075=sum(row["train"]["strict_075"] for row in valid),
                  test_ge_075=sum(row["test"]["strict_075"] for row in valid),
                  train_ge_090=sum(row["train"]["strict_090"] for row in valid),
                  test_ge_090=sum(row["test"]["strict_090"] for row in valid))
    return result


def verified_dataset(a):
    exact(a.p335_source, P335_SOURCE_SHA)
    previous = json.loads(exact(a.p335_result, P335_RESULT_SHA))
    terminal = json.loads(exact(a.p335_terminal, P335_TERMINAL_SHA))
    require(terminal["status"] == "P335_SRNET_NATIVE_DATA_ORIGINAL_CPU_WAIT_V2"
            and terminal["original_os_wait"] is True and type(terminal["original_os_exit"]) is int
            and terminal["original_os_exit"] == 0 and terminal["worker_closed"] is True
            and terminal["resource_stop"] is False and terminal["source_sha256"] == P335_SOURCE_SHA,
            "Original accepted FIT-only loader wait required")
    for name, digest in AUTHOR_PINS.items():
        exact(a.source / name, digest)
    spec = importlib.util.spec_from_file_location("p335_pinned_fit_loader", a.p335_source)
    p335 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p335)
    sys.path.insert(0, str(a.source))
    import mmrotate.datasets as author_datasets

    original_build = author_datasets.build_dataset
    built = []

    def observe(settings):
        result = original_build(settings)
        built.append(result)
        return result

    with patch.object(author_datasets, "build_dataset", observe):
        report = p335.run(SimpleNamespace(source=a.source, fit=a.fit, archive=a.archive))
    require(len(built) == 1 and {k: v for k, v in report.items() if k != "elapsed_seconds"}
            == {k: v for k, v in previous.items() if k != "elapsed_seconds"},
            "Current original author FIT admission must reproduce the closed P335 safe report")
    return p335, built[0].dataset, previous


def run(a):
    started = perf_counter()
    p335, base, previous = verified_dataset(a)
    import torch
    from mmrotate.models.dense_heads import SRNet_utils as stock
    from pycocotools import mask as codec

    require(Path(stock.__file__).resolve() == a.source / "mmrotate/models/dense_heads/SRNet_utils.py"
            and not torch.cuda.is_available() and not torch.cuda.is_initialized(),
            "Exact author CPU sampler and no CUDA required")
    raw = json.loads(p335.exact(a.fit / "_annotations.coco.json", p335.FIT_INDEX_SHA, 2 * 1024 ** 2))
    original = defaultdict(lambda: defaultdict(deque))
    for ann in raw["annotations"]:
        original[ann["image_id"]][p335.key(ann["segmentation"])].append(ann)
    by_image = defaultdict(list)
    boxes, areas = [], []
    for idx, image in enumerate(base.data_infos):
        parsed = base.get_ann_info(idx)
        image_id = image["id"]
        for box, segmentation in zip(parsed["bboxes"], parsed["masks"]):
            candidates = original[image_id][p335.key(segmentation)]
            require(bool(candidates), "Parsed target cannot consume an absent original polygon")
            if len(candidates) > 1:
                require(all(x["area"] == candidates[0]["area"] for x in candidates),
                        "Ambiguous area strata for duplicate original polygons")
            ann = candidates.popleft()
            boxes.append(box)
            areas.append(ann["area"])
            by_image[image_id].append((box, segmentation, ann["area"]))
    require(len(boxes) == previous["author_parsed_targets"] == 1661
            and not any(rows for source in original.values() for rows in source.values()),
            "Every original FIT polygon and parsed le90 box must reconcile")
    boxes = np.asarray(boxes, dtype=np.float32)
    cohort = p335.choose(raw["images"])
    oracle_inputs = []
    for image in cohort:
        for box, segmentation, area in by_image[image["id"]]:
            truth = p335.bitmap(segmentation, image["height"], image["width"], codec)
            oracle_inputs.append((box, truth, area))
    require(len(cohort) == 8 and len(oracle_inputs) == previous["actual_pipeline_counts"]["native_masks_verified"],
            "Fixed P335 eight-image oracle scope")
    geometry_masks = strata(areas, boxes)
    oracle_masks = strata([row[2] for row in oracle_inputs],
                          np.asarray([row[0] for row in oracle_inputs]))
    results = {}
    with torch.no_grad():
        for n in (28, 56):
            values, errors = stock_geometry(boxes, n, stock, torch)
            oracle = [dict(oracle_pair(box, truth, n, stock, torch), empty_truth=not bool(truth.any()))
                      for box, truth, _ in oracle_inputs]
            results[str(n)] = dict(
                used_in_author_head=n == 28,
                stock_vs_independent_max_abs_coordinate_error_px=errors,
                geometry={key: summarize_geometry(values[selector]) for key, selector in geometry_masks.items()},
                original_polygon_raster_oracle={
                    key: summarize_oracle([row for row, keep in zip(oracle, selector) if keep])
                    for key, selector in oracle_masks.items()},
            )
    require(not torch.cuda.is_initialized(), "No CUDA state permitted")
    return dict(status=STATUS, author_commit=p335.AUTHOR_COMMIT, author_pins=AUTHOR_PINS,
                p335_source_sha256=P335_SOURCE_SHA, p335_result_sha256=P335_RESULT_SHA,
                p335_original_terminal_sha256=P335_TERMINAL_SHA, fit_archive_sha256=p335.FIT_SHA,
                fit_index_sha256=p335.FIT_INDEX_SHA, package_versions=previous["package_versions"],
                FIT=True, DEV=False, TEST=False, AP=False, model_forward_calls=0, optimizer_updates=0,
                geometry_images=741, geometry_targets=len(boxes), oracle_images=len(cohort),
                oracle_targets=len(oracle_inputs), sizes=results,
                strata_definition=dict(small="original COCO annotation area < 1024",
                                       thin="author original le90 max side / min side >= 3"),
                oracle_scope="original COCO polygon bitmap, original parsed GT box, identical stock square paste;"
                             " paired grid-only diagnostic with GT information; no fitted quality inference",
                elapsed_seconds=perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = ("p335-source", "p335-result", "p335-terminal", "source", "fit", "archive")
    for flag in (*inputs, "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    a = parser.parse_args()
    require(all(p.is_absolute() and p.resolve(strict=True) == p for p in
                (getattr(a, name.replace("-", "_")) for name in inputs))
            and a.output.is_absolute() and a.output.parent.resolve(strict=True) == a.output.parent
            and not a.output.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and sys.flags.isolated and sys.dont_write_bytecode,
            "Exact canonical inputs, fresh output and isolated CPU runtime required")

    def offline(event, _):
        if event.startswith("socket."):
            raise PermissionError("Offline FIT-only geometry")

    sys.addaudithook(offline)
    raw = (json.dumps(run(a), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with a.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()

