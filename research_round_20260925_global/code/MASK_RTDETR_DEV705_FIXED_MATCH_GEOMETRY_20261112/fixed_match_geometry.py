"""Measure continuous hull geometry on frozen original Mask RT-DETR COCO matches."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask


ARMS = ("native", "center")
STATUS = "MASK_RTDETR_DEV705_FIXED_MATCH_GEOMETRY_COMPLETE"
ADMISSION_SHA = "7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"
MATCH_WRAPPER = "SSDD_DEV187_POST_PRIMARY_RANK_20261110/ssdd_fixed_match_geometry.py"
MATCH_WRAPPER_SHA = "5aebe92a3b355874e8a20f24ef19b98ace8c20e3537610ec92a9c0e026b260e3"
MATCH_GEOMETRY = "HRSID_MASK_PHASE_REPEAT_DEV705_20261109/dev705_paired_mask_geometry.py"
PROTOCOL_SHA = "ff783c685bf9fcf9ef68ab9894df9dd59348f133f1a8f8144ee2d79825534b3d"
CENSUS_STATES = ("both", "native_only", "center_only", "neither")
COCO_GROUPS = ("all", "small", "medium", "large")
VECTOR_GROUPS = (*COCO_GROUPS, "aspect_ge_2", "thin")
ASPECT_MIN = 2.
THIN_SHORT_SIDE_MAX = 8.
METRICS = ("continuous_iou_points", "fp_per_gt_area_points", "fn_per_gt_area_points",
           "centroid_abs_short_px", "centroid_abs_long_px")
TIE = 1e-10


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load_module(path, expected, name):
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
         and 0 < path.stat().st_size <= 1024**2, "Canonical bounded pinned ordinary source required")
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == expected, "Pinned ordinary source differs: " + path.name)
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def load_match_helpers(admission, code_root):
    wrapper = load_module(code_root / MATCH_WRAPPER, MATCH_WRAPPER_SHA, "_closed_match_loader")
    _, _, template, _, _, geometry = wrapper.load_helpers(code_root, code_root / MATCH_GEOMETRY)
    adapter = admission.load_module(code_root / admission.OFFICIAL_ADAPTER, admission.OFFICIAL_ADAPTER_SHA,
                                    "_original_p392_match_adapter")
    return SimpleNamespace(template=template, adapter=adapter, match=geometry.match_official)


def original_matches(gt, streams, priors, helpers, admission):
    images = [row["id"] for row in gt["images"]]
    annotation_images = {ann["id"]: ann["image_id"] for ann in gt["annotations"]}
    need(len(annotation_images) == len(gt["annotations"]), "Unique original annotation identities required")
    output = {}
    for arm in ARMS:
        all12 = []

        def capture(*args):
            values = helpers.adapter.evaluate_population(*args)
            all12.extend(values)
            return values

        metrics, records, ranks = helpers.template.evaluate_cell(
            gt, streams[arm], images, SimpleNamespace(evaluate_population=capture))
        prior = priors[arm]
        need(len(all12) == len(prior["all12"]) == 12
             and np.allclose(all12, prior["all12"], rtol=0., atol=1e-12)
             and abs(metrics["AP90"] - prior["AP90"]) <= 1e-12,
             "Sealed original all12/AP90 parity required before retaining associations: " + arm)
        need(ranks == admission.native_layout(streams[arm]),
             "Sealed original candidates, ordering, ties and scores must agree: " + arm)
        output[arm] = helpers.match(records, streams[arm], annotation_images)
    return output


def hull(geometry):
    """Return aspect, short-side width, long unit axis and short unit axis."""
    vertices = np.asarray(geometry.minimum_rotated_rectangle.exterior.coords, dtype=float)
    edges = np.diff(vertices, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    need(len(lengths) == 4 and np.isfinite(lengths).all() and lengths.min() > 0.,
         "Positive-area reference must have a finite nondegenerate minimum rectangle")
    width = float(lengths.min())
    index = int(lengths.argmax())
    tangent = edges[index] / lengths[index]
    normal = np.asarray((-tangent[1], tangent[0]))
    return float(lengths[index] / width), width, tangent, normal


def measure(reference, rle, axes, modules):
    compressed = rle["counts"]
    text = compressed.decode("ascii") if isinstance(compressed, bytes) else compressed
    rectangles = modules.geometry.boxes_from_counts(modules.shift.runs(text), *rle["size"])
    weights = np.prod(rectangles[:, 2:] - rectangles[:, :2], axis=1)
    area = float(mask.area(rle))
    ref_area = float(reference.area)
    need(area > 0 and area == weights.sum() and ref_area > 0,
         "Official matched nonempty foreground and disjoint compressed footprint required")
    intersection = modules.geometry.overlap_area(rectangles, reference)
    need(0. <= intersection <= min(area, ref_area) + 1e-7, "Continuous overlap outside corresponding areas")
    intersection = min(intersection, area, ref_area)
    center = np.average((rectangles[:, 2:] + rectangles[:, :2]) / 2, axis=0, weights=weights)
    measures = dict(continuous_iou_points=100 * intersection / (area + ref_area - intersection),
                    fp_per_gt_area_points=100 * (area - intersection) / ref_area,
                    fn_per_gt_area_points=100 * (ref_area - intersection) / ref_area)
    aspect, _, tangent, normal = axes
    if aspect >= ASPECT_MIN:
        offset = center - np.array((reference.centroid.x, reference.centroid.y))
        measures.update(centroid_abs_short_px=abs(float(normal @ offset)),
                        centroid_abs_long_px=abs(float(tangent @ offset)))
    return measures, center


def dense_check(rings, rle, measures, predicted_centroid, modules):
    dense = mask.decode(rle).astype(bool)
    independent = modules.footprint.run_footprint_metrics(rings, dense.shape, {"row": dense})
    stats = independent["by_arm"]["row"]
    reference_area = independent["visible_polygon_area"]
    y, x = np.nonzero(dense)
    expected_centroid = np.asarray((x.mean() + .5, y.mean() + .5))
    errors = dict(
        continuous_iou_points=abs(100 * stats["iou"] - measures["continuous_iou_points"]),
        fp_per_gt_area_points=abs(100 * stats["fp_area"] / reference_area - measures["fp_per_gt_area_points"]),
        fn_per_gt_area_points=abs(100 * stats["fn_area"] / reference_area - measures["fn_per_gt_area_points"]),
        centroid_max_abs_px=float(np.abs(predicted_centroid - expected_centroid).max()))
    need(max(errors.values()) < 1e-8, "Exact compressed footprint differs from independent dense run geometry")
    return errors


def coverage_state(ann_id, matches):
    native, center = (ann_id in matches[arm] for arm in ARMS)
    if native and center:
        return "both"
    if native:
        return "native_only"
    if center:
        return "center_only"
    return "neither"


def summarize(rows, metric, footprint, draws):
    pairs = [(image, a[metric], b[metric]) for image, a, b in rows if metric in a]
    need(all((metric in a) == (metric in b) for _, a, b in rows), "Both arms must share metric support")
    if not pairs:
        return dict(eligible_pairs=0, contributing_DEV_chips=0, native_mean=None, center_mean=None,
                    delta_center_minus_native=None, center_better=0, center_worse=0, ties=0,
                    paired_chip_cluster_percentile95=None)
    native, center = (np.asarray([row[index] for row in pairs]) for index in (1, 2))
    delta = center - native
    favorable = delta if metric == "continuous_iou_points" else -delta
    boot = footprint.paired_cluster_stats([(row[0], difference)
                                          for row, difference in zip(pairs, delta, strict=True)], draws)
    return dict(eligible_pairs=len(pairs), contributing_DEV_chips=boot["source_images"],
                native_mean=float(native.mean()), center_mean=float(center.mean()),
                delta_center_minus_native=boot["row_weighted_mean_delta"],
                center_better=int((favorable > TIE).sum()), center_worse=int((favorable < -TIE).sum()),
                ties=int((np.abs(favorable) <= TIE).sum()),
                paired_chip_cluster_percentile95=boot["row_weighted_image_cluster_percentile95"])


def analyze(gt, matches, modules, *, expected_support, dense_spots):
    polygons, neutral, support = modules.geometry.polygon_support(gt, modules.vector)
    need((len(polygons), len(neutral)) == expected_support, "Exact fixed source polygon support required")
    full = {key: Counter() for key in COCO_GROUPS}
    valid = {key: Counter() for key in VECTOR_GROUPS}
    pairs = {key: [] for key in VECTOR_GROUPS}
    images = {row["id"]: index for index, row in enumerate(gt["images"])}
    annotations = {row["id"]: row for row in gt["annotations"]}
    audited = Counter()
    errors = {arm: {} for arm in ARMS}
    for ann_id, ann in sorted(annotations.items()):
        size = modules.geometry.stratum(ann["area"])
        state = coverage_state(ann_id, matches)
        for key in ("all", size):
            full[key][state] += 1
        if ann_id not in polygons:
            continue
        reference, rings = polygons[ann_id]
        axes = hull(reference)
        aspect, width = axes[:2]
        groups = ["all", size]
        if aspect >= ASPECT_MIN:
            groups.append("aspect_ge_2")
            if width <= THIN_SHORT_SIDE_MAX:
                groups.append("thin")
        for group in groups:
            valid[group][state] += 1
        if state != "both":
            continue
        values = {}
        for arm in ARMS:
            rle = matches[arm][ann_id]
            values[arm], centroid = measure(reference, rle, axes, modules)
            if audited[arm] < dense_spots:
                observed = dense_check(rings, rle, values[arm], centroid, modules)
                for key, error in observed.items():
                    errors[arm][key] = max(error, errors[arm].get(key, 0.))
                audited[arm] += 1
        record = (images[ann["image_id"]], values["native"], values["center"])
        for group in groups:
            pairs[group].append(record)
    need(full["all"].total() == len(annotations) and valid["all"].total() == len(polygons)
         and all(valid[group]["both"] == len(pairs[group]) for group in VECTOR_GROUPS)
         and all(audited[arm] == dense_spots for arm in ARMS),
         "Full and paired match support must reconcile with exact geometry audits")
    draws = modules.footprint.bootstrap_draws(len(images))
    paired = {group: {metric: summarize(rows, metric, modules.footprint, draws) for metric in METRICS}
              for group, rows in pairs.items()}
    return dict(original_full_coverage={group: dict((state, counts[state]) for state in CENSUS_STATES)
                                       for group, counts in full.items()},
                supported_coverage={group: dict((state, counts[state]) for state in CENSUS_STATES)
                                    for group, counts in valid.items()},
                supported_unique_annotations=len(polygons), neutral_unique_annotations=len(neutral),
                support=support, paired_geometry=paired,
                dense_geometry_checks={arm: dict(count=audited[arm], maximum_absolute_error=errors[arm])
                                       for arm in ARMS})


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("admission-source", "protocol", "code-root", "p392-source", "score-dir", "export-root",
                 "shapely-site", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("score-terminal-sha256", "score-sha256", "export-wait-sha256", "paired-receipt-sha256",
                 "native-closure-sha256", "center-closure-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    need(os.name == "nt" and sys.version_info[:3] == (3, 10, 13) and sys.flags.isolated == 1
         and sys.flags.utf8_mode == 1 and sys.dont_write_bytecode
         and all(os.environ.get(key) == value for key, value in
                 (("CUDA_VISIBLE_DEVICES", "-1"), ("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1")))
         and version("numpy") == "1.26.4" and version("pycocotools") == "2.0.11",
         "Exact isolated one-thread CPU scoring runtime required")
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
         and not args.output.exists() and not args.output.is_symlink()
         and not args.output.is_relative_to(args.export_root) and not args.output.is_relative_to(args.score_dir)
         and all(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir()
                 for path in (args.code_root, args.p392_source, args.score_dir, args.export_root)),
         "Canonical closed input paths and fresh outside result required")
    need(all(re.fullmatch("[0-9a-f]{64}", getattr(args, name)) for name in
             ("score_terminal_sha256", "score_sha256", "export_wait_sha256", "paired_receipt_sha256",
              "native_closure_sha256", "center_closure_sha256")), "Exact original receipt digests required")
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    need(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
         "Exact continuous geometry runtime required")

    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("Closed offline CPU diagnostic only")

    sys.addaudithook(offline)
    started = perf_counter()
    admission = load_module(args.admission_source, ADMISSION_SHA, "_closed_p392_admission")
    admission.pinned(args.protocol, PROTOCOL_SHA)
    modules = admission.geometry_modules(args.code_root)
    helpers = load_match_helpers(admission, args.code_root)
    gt, streams, priors, provenance = admission.load_inputs(args)
    matches = original_matches(gt, streams, priors, helpers, admission)
    result = analyze(gt, matches, modules, expected_support=(1761, 21), dense_spots=8)
    payload = dict(status=STATUS, protocol_sha256=PROTOCOL_SHA, training_protocol=admission.PROTOCOL,
                   original_score_sha256=args.score_sha256, original_score_terminal_sha256=args.score_terminal_sha256,
                   original_export_wait_sha256=args.export_wait_sha256, paired_receipt_sha256=args.paired_receipt_sha256,
                   original_gt_sha256=admission.GT_SHA, image_manifest_sha256=admission.MANIFEST_SHA,
                   provenance_by_arm=provenance, original_images=705, original_unique_annotations=1782,
                   raw_prediction_rows={arm: len(streams[arm]) for arm in ARMS}, original_all12_AP90_reproduced=True,
                   fixed_original_nongeometric_reassignment=False, association_iou_threshold=.5,
                   unfiltered_full_top100=True, aspect_min=ASPECT_MIN, thin_short_side_max_px=THIN_SHORT_SIDE_MAX,
                   paired_uncertainty=dict(cluster="DEV chip", independent_source_acquisition_inference=False,
                                           draws=modules.footprint.BOOTSTRAP_DRAWS, seed=modules.footprint.BOOTSTRAP_SEED),
                   DEV=True, FIT=False, official_TEST_opened=False, official_TEST_AP=False,
                   model_forward_calls=0, optimizer_updates=0, elapsed_seconds=perf_counter() - started, **result)
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
