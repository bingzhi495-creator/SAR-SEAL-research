"""Post-hoc 8-connected-component diagnostic on the 34 measured NASTaR score_02 anchors.

Run with the closed score's Python environment and an unused --output directory.
Only official ZIP directory metadata may be requested. All AIS bytes come from
the verified score_02 cache. The output contains cohort-level aggregates only.
"""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy import ndimage


SCORE_WORKER = "02b912430f91a3b6f407b86a1c1601dddeb578034acf576f1601f51c7936138b"
SCORE_TERMINAL = "4223c59905a6de071cdc8550bfd7ba16c73cf28512b483f1f552da4f7b88eb79"
PREPARE_WORKER = "ad53d9a20dc11edfd356b9c90e9633ba0affcbbe026deb0d4e38796cd4087731"
STATUS = "NASTAR_CLOSED_EXTENT_COMPONENT_DIAGNOSTIC_COMPLETE"


def check(test, detail):
    if not test:
        raise ValueError(detail)


def digest(data):
    return sha256(data).hexdigest()


def load_score(code):
    path = code / "score_nastar_ais.py"
    raw = path.read_bytes()
    check(digest(raw) == SCORE_WORKER, "Closed scoring implementation changed")
    mod = ModuleType("closed_nastar_score")
    mod.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), mod.__dict__)
    return mod


def center_steps(geo, score, prepare, intake, row):
    """WGS84 distances from cell (255,255)'s center to +1 column and +1 row centers."""
    raw = score.pinned(intake / row["tiff_path"], row["tiff_sha256"])
    tags = dict(model_transformation=34264, model_pixel_scale=33550, model_tiepoint=33922,
                geo_key_directory=34735, geo_double_params=34736, geo_ascii_params=34737, gdal_nodata=42113)
    with Image.open(io.BytesIO(raw)) as image:
        check((image.format, image.mode, image.size) == ("TIFF", "L", (512, 512)), "Unexpected source raster")
        metadata = {k: prepare.plain_tag(image.tag_v2.get(number)) for k, number in tags.items()}
    check(metadata == row["geometry"], "Original TIFF tags and closed intake geometry differ")
    crs, affine = score.grid_from_tags(metadata)
    origin, matrix = geo._grid(affine)
    x, y = origin[:, None] + matrix @ np.asarray([[255.5, 256.5, 255.5], [255.5, 255.5, 256.5]])
    lon, lat = geo.Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y, errcheck=True)
    _, _, distances = geo._GEOD.inv(np.repeat(lon[0], 2), np.repeat(lat[0], 2), lon[1:], lat[1:])
    check(np.isfinite(distances).all() and np.all(distances > 0), "Invalid adjacent-center distances")
    return distances


def retained_component(mask, near):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    area = np.bincount(labels.ravel(), minlength=count + 1)
    near_count = np.bincount(labels[near], minlength=count + 1)
    # ndimage.label emits components in row-major source discovery order.
    chosen = max(range(1, count + 1), key=lambda n: (near_count[n], area[n], -n))
    check(near_count[chosen] > 0, "Selected component misses the locked disk")
    return labels == chosen, int(count)


def component_extent(score, geo, intake, row, fields, stock, record):
    crs, transform = score.grid_from_tags(row["geometry"])
    lon, lat, heading = fields[:3]
    near = score.within_radius(geo, lon, lat, crs, transform)
    valid = score.validity(intake, row)
    for det in stock:
        raw = score.codec.decode(det["segmentation"]).astype(bool)
        if not raw[near].any():
            continue
        # Verify the chosen native anchor against the result of the unmodified official scorer.
        original = geo.heading_extents_m(raw, lon, lat, heading, crs, transform, valid)
        check(np.array_equal(original, record["arms"]["raw"]), "Anchor replay differs")
        component, count = retained_component(raw, near)
        extent = geo.heading_extents_m(component, lon, lat, heading, crs, transform, valid)
        check(np.all(np.asarray(extent) <= np.asarray(original) + 1e-8), "Subset enlarged the footprint")
        return extent, count
    raise ValueError("Failed to recover the closed native anchor")


def average_passes(group_arrays):
    return np.mean([np.mean(a, axis=0) for a in group_arrays], axis=0)


def aggregate(groups):
    arrays = [np.asarray(items) for items in groups.values()]
    result = {}
    for col, key in enumerate(("native", "retained_component")):
        errors = [a[:, col] - a[:, 2] for a in arrays]
        result[key] = dict(
            mean_extent_m=average_passes([a[:, col] for a in arrays]).tolist(),
            mean_bias_m=average_passes(errors).tolist(),
            mean_absolute_error_m=average_passes([np.abs(e) for e in errors]).tolist(),
        )
    reduction = [np.mean(np.abs(a[:, 0] - a[:, 2]) - np.abs(a[:, 1] - a[:, 2]), axis=0) for a in arrays]
    signs = np.sign(np.where(np.abs(reduction) < 1e-8, 0, reduction))
    result.update(mean_reference_m=average_passes([a[:, 2] for a in arrays]).tolist(),
                  native_minus_component_mae_m=np.mean(reduction, axis=0).tolist(),
                  passes_with_lower_component_mae=(signs > 0).sum(axis=0).tolist(),
                  passes_with_equal_mae=(signs == 0).sum(axis=0).tolist(),
                  passes_with_higher_component_mae=(signs < 0).sum(axis=0).tolist())
    return result


def run(root, geo_site):
    code = root / "released_score_02"
    score = load_score(code)
    terminal = json.loads(score.pinned(root / "observed_score_02/TERMINAL.json", SCORE_TERMINAL))
    check(terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
          and terminal["worker_sha256"] == SCORE_WORKER, "Closed score terminal failed verification")
    closed = json.loads(score.pinned(root / "score_02/RESULT_SAFE.json", terminal["result_sha256"]))
    check(closed["status"] == score.STATUS and closed["worker_sha256"] == SCORE_WORKER, "Closed score mismatch")
    args = SimpleNamespace(intake=root / "intake", replay=root / "replay", capture=root / "capture",
                           protocol=root / "protocol.md", replay_terminal=root / "observed_replay/TERMINAL.json")
    blocks, replay, replay_sha = score.closed_inputs(args)
    check(replay_sha == closed["replay_result_sha256"] and closed["intake_sha256"] == replay["manifest_sha256"]
          and closed["capture_result_sha256"] == replay["capture_result_sha256"], "Closed scientific data lineage changed")
    if geo_site is not None:
        sys.path.append(str(geo_site))
    geo = score.module(code / "ais_geometry.py", score.GEOMETRY)
    import pyproj
    check((pyproj.__version__, pyproj.proj_version_str) == ("3.7.1", "9.5.1"), "Unexpected geodesic version")
    prepare = score.module(code / "prepare_nastar.py", PREPARE_WORKER)
    directory = prepare.pinned_directory()
    cache = root / "score_02/ais_private"
    # Fail on a missing private file. Only the already-verified ZIP index is allowed over the network.
    with patch.object(prepare, "pinned_directory", return_value=directory), \
            patch.object(score.requests, "get", side_effect=RuntimeError("AIS cache miss; label fetch forbidden")):
        labels, ais_digest, packaging = score.selected_labels(blocks, prepare, cache)
    check(ais_digest == closed["selected_product_csv_hash_aggregate"] and packaging == closed["ais_product_packaging"],
          "Private labels differ from closed score")
    primary = closed["primary_distinct_images"]
    check(len(blocks) == closed["roster_selected"] == 142 and closed["exact_decoded_duplicates"] == 0,
          "Unexpected original cohort")
    scales, groups = defaultdict(list), defaultdict(list)
    fragments, failures = defaultdict(list), {arm: Counter() for arm in score.ARMS}
    exclusions = Counter()
    for item, row, stock, changes in blocks:
        group = item["pass_id"]
        scales[group].append(center_steps(geo, score, prepare, args.intake, row))
        check(row["duplicate_of_image_id"] is None, "Unexpected duplicate")
        label, hits = labels[item["image_id"]]
        check(label is not None and hits == 1, "Expected exactly one closed reference")
        fields, validity = score.clean_labels(label)
        if not all(validity.values()):
            exclusions["incomplete_physical_reference"] += 1
            continue
        record = score.physical_measure(geo, args.intake, row, fields, stock, changes)
        if "reason" in record:
            exclusions[record["reason"]] += 1
            continue
        if record["invalid"]:
            exclusions["any_arm_measurement_failure"] += 1
            for arm, reason in record["invalid"].items():
                failures[arm][reason] += 1
            continue
        check(set(record["arms"]) == set(score.ARMS), "Incomplete four-arm anchor")
        component, count = component_extent(score, geo, args.intake, row, fields, stock, record)
        fragments[group].append(count)
        groups[group].append((record["arms"]["raw"], component, record["reference"]))
    counts = np.asarray([n for values in fragments.values() for n in values])
    check(len(counts) == primary["coverage"]["all_arm_measured"] == 34 and len(groups) == 18 and len(scales) == 24,
          "Measured anchor population changed")
    check(sum(exclusions.values()) + len(counts) == len(blocks) and failures == primary["failure_by_arm"]
          and all(n == primary["coverage"][k] for k, n in exclusions.items()), "Anchor exclusions changed")
    summary = aggregate(groups)
    official = primary["arm_measurements"]["raw"]
    for ours, theirs in (("mean_extent_m", "length_beam_mean_m"), ("mean_bias_m", "length_beam_bias_m"),
                         ("mean_absolute_error_m", "length_beam_mae_m")):
        np.testing.assert_allclose(summary["native"][ours], official[theirs], rtol=0, atol=1e-8)
    np.testing.assert_allclose(summary["mean_reference_m"], official["ais_length_beam_mean_m"], rtol=0, atol=1e-8)
    spacings = np.concatenate([np.asarray(v) for v in scales.values()])
    return dict(status=STATUS, analysis="Post-hoc native anchor component extent diagnostic",
                axes_measurement=["along_true_heading_length", "across_true_heading_beam"],
                physical_units="metres", selected_chips=len(blocks), selected_passes=len(scales),
                measured_anchors=len(counts), measured_passes=len(groups), exclusions=dict(exclusions),
                failure_by_arm=failures, fragmented_anchors=int((counts > 1).sum()),
                fragmented_percent=100 * float(np.mean(counts > 1)),
                fragmented_percent_equal_pass=100 * float(average_passes([np.asarray(x) > 1 for x in fragments.values()])),
                passes_with_fragmented_anchors=sum(any(n > 1 for n in v) for v in fragments.values()),
                component_count_median=float(np.median(counts)), component_count_max=int(counts.max()),
                adjacent_center_spacing_m=dict(axes=["plus_column", "plus_row"], min=spacings.min(axis=0).tolist(),
                                               median=np.median(spacings, axis=0).tolist(), max=spacings.max(axis=0).tolist(),
                                               equal_pass_mean=average_passes(scales.values()).tolist()),
                measurement_equal_pass=summary, score_result_sha256=terminal["result_sha256"],
                ais_content_aggregate_sha256=ais_digest, official_directory_sha256=prepare.DIRECTORY_SHA,
                replay_result_sha256=replay_sha, intake_sha256=replay["manifest_sha256"], ais_product_packaging=packaging,
                definition="8-neighbor native components ranked by cells in the closed 18 m reference disk, then area, then source scan order",
                model_forwards=0, optimizer_updates=0, new_label_fetches=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--closed-root", type=Path, required=True)
    parser.add_argument("--geo-site", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.closed_root.resolve(), args.output.resolve()
    check(output != root and root not in output.parents, "Place the fresh diagnostic outside the closed run")
    output.mkdir(parents=True, exist_ok=False)
    result = run(root, args.geo_site)
    result["worker_sha256"] = digest(Path(__file__).read_bytes())
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with (output / "RESULT_SAFE.json").open("xb") as handle:
        handle.write(raw)
    print(json.dumps(dict(status=STATUS, result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()
