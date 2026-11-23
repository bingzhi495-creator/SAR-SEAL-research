"""Diagnose the already evaluated Gulf anchors with VV XML sample geometry."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import re
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np


LOCK = "3e9e2cccd4246ef416bc08cbadc6032d547ca5a57abc1df6eb7d16997d5c509c"
OLD_LOCK = "6001a3d9be9cf5cccb71e4156603d91c2dec1858ba46816a8380f2d315272cbb"
ANCHORS = "af6ce6a393610d93b7c47bf08b5d7c7b0994b86892432cf27c23f78526ded194"
GRID = "e4642f41961dcbbdbdaf6ba12756a1fcaa625ec2985caf4fbc10d048e6c51050"
STATUS = "GULF_AIS_GEOSCALE_DIAGNOSTIC_COMPLETE"
ARMS = ("raw", "fixed", "free_matched_count", "free_natural")
OFF_X = np.array([-.5, -.5, .5, .5], dtype=np.float64)
OFF_Y = np.array([-.5, .5, -.5, .5], dtype=np.float64)
SCENE_TIMES = re.compile(r"_([0-9]{8}T[0-9]{6})_([0-9]{8}T[0-9]{6})_")


def require(ok, text):
    if not ok:
        raise ValueError(text)


def digest(data):
    return sha256(data).hexdigest()


def pinned(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Pinned input bytes changed: " + path.name)
    return raw


def sibling(name, sha):
    path = Path(__file__).with_name(name)
    spec = spec_from_file_location("_gulf_diagnostic_" + path.stem, path)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(pinned(path, sha), str(path), "exec"), module.__dict__)
    return module


def east_north(geod, lon0, lat0, lon, lat):
    lon, lat = np.broadcast_arrays(np.asarray(lon, np.float64), np.asarray(lat, np.float64))
    az, _, dist = geod.inv(np.full_like(lon, lon0), np.full_like(lat, lat0), lon, lat)
    require(np.isfinite(az).all() and np.isfinite(dist).all(), "Finite WGS84 geodesic coordinates")
    angle = np.deg2rad(az)
    return np.stack((np.sin(angle) * dist, np.cos(angle) * dist), axis=-1)


def local_geometry(grid, geod, col, row, image_axis):
    lon0, lat0 = grid.forward(col, row)
    lon, lat = grid.forward(np.array([col + 1, col]), np.array([row, row + 1]))
    step = east_north(geod, lon0, lat0, lon, lat)
    unit = np.asarray(image_axis, dtype=np.float64)
    anchor_axis = unit @ step
    norm = float(np.linalg.norm(anchor_axis))
    lengths = np.linalg.norm(step, axis=1)
    require(np.isclose(np.linalg.norm(unit), 1.) and (lengths > 0).all() and norm > 0,
            "Nondegenerate fixed native PCA axis and local grid")
    cos = float(np.dot(*step) / np.prod(lengths))
    axis_angle = float(np.rad2deg(np.arccos(np.clip(cos, -1., 1.))))
    return (lon0, lat0), anchor_axis / norm, [float(v) for v in lengths], norm, axis_angle


def physical_cells(grid, geod, col, row, center, indices):
    y, x = np.divmod(indices.astype(np.int64), 512)
    cols = col + x[:, None] - 256 + OFF_X
    rows = row + y[:, None] - 256 + OFF_Y
    lon, lat = grid.forward(cols, rows)
    return east_north(geod, *center, lon, lat).reshape(-1, 2)


def fixed_mid_time(scene):
    matched = SCENE_TIMES.search(scene)
    require(matched is not None, "Original source scene clock required")
    start, end = [datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).timestamp()
                  for value in matched.groups()]
    require(end >= start, "Ordered scene clock")
    return (start + end) / 2


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    require(np.isfinite(values).all(), "Finite aggregate diagnostic")
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                minimum=float(values.min()), p95=float(np.percentile(values, 95)), maximum=float(values.max()))


def paired_reduction(old, rows, a, b, arm="raw"):
    groups = old.by_group(rows)
    if not groups:
        return None
    effects = np.array([np.mean([abs(r[a][arm]-r["length"])-abs(r[b][arm]-r["length"]) for r in g])
                        for g in groups], dtype=np.float64)
    draws = np.random.Generator(np.random.PCG64(5217)).integers(len(groups), size=(4096,len(groups)))
    return dict(ships=len(rows), groups=len(groups), mae_reduction_m=float(effects.mean()),
                ci95_m=np.percentile(effects[draws].mean(axis=1), (2.5,97.5)).tolist())


def run(a):
    elapsed = perf_counter()
    pinned(a.diagnostic_lock, LOCK)
    require(a.diagnostic_lock_sha256 == LOCK and not a.output.exists(), "Fixed posthoc plan and fresh output")
    sys.path.insert(0, str(a.geo_site))
    import pyproj
    require(pyproj.__version__ == "3.7.1" and pyproj.proj_version_str == "9.5.1"
            and Path(pyproj.__file__).resolve().is_relative_to(a.geo_site), "Qualified archived WGS84 dependency")
    geod = pyproj.Geod(ellps="WGS84")
    anchors, grid_code = sibling("anchors.py", ANCHORS), sibling("grid.py", GRID)
    root = a.closed_root
    old_args = SimpleNamespace(code=a.code, protocol=a.gulf_protocol, protocol_sha256=OLD_LOCK, include_heading=True)
    for phase in ("intake","capture","replay","score"):
        setattr(old_args, phase, root / phase)
        setattr(old_args, phase + "_terminal", root / ("observed_" + phase) / "TERMINAL.json")
    previous, old_module, manifest, rows = anchors.load_frozen(old_args)
    require(len(rows) == 25 and len(old_module.by_group(rows)) == 19, "Original complete acceptance")
    corrected_count = sum(r["fixed_k"] > 0 for r in rows)
    require(corrected_count == 23, "Original positive fixed edit subset")
    grids, spatial, temporal = {}, [], []
    origin_values = {k: [] for k in ("column_step_m","row_step_m","mapped_axis_m_per_pixel","local_axes_angle_deg",
                                       "center_to_ais_m","inverse_center_col_abs_px","inverse_center_row_abs_px",
                                       "native_axis_to_ais_heading_deg")}
    time_values = {k: [] for k in ("xml_time_minus_mid_s","scene_midpoint_lag_s","low_possible_lag_s",
                                    "high_possible_lag_s","old_distance_proxy_m","low_distance_proxy_m","high_distance_proxy_m")}
    counts = Counter()
    for r in rows:
        scene = r["row"]["scene_id"]
        expected = manifest["scene_geometry"][scene]
        if scene not in grids:
            xml = pinned(old_args.intake / expected["xml_path"], expected["xml_sha256"])
            grid = grid_code.read_grid(xml)
            require((grid.width, grid.height) == (expected["width"], expected["height"]), "Original XML scene dimensions")
            grids[scene] = grid
        grid = grids[scene]
        col, row = r["pixel_col"], r["pixel_row"]
        center, axis, steps, axis_step, angle = local_geometry(grid, geod, col, row, r["local_axis"])
        origin_values["column_step_m"].append(steps[0])
        origin_values["row_step_m"].append(steps[1])
        origin_values["mapped_axis_m_per_pixel"].append(axis_step)
        origin_values["local_axes_angle_deg"].append(angle)
        counts["physical_rows"] += 1

        out = {k: r[k] for k in ("length","fixed_k","pass_id","spans")}
        corrected, heading_spans = {}, {}
        heading = r.get("heading")
        actual_heading = None
        if heading is not None:
            bearing = np.deg2rad(heading)
            actual_heading = np.array([np.sin(bearing), np.cos(bearing)])
            theta = float(np.rad2deg(np.arccos(np.clip(abs(axis @ actual_heading), 0, 1))))
            origin_values["native_axis_to_ais_heading_deg"].append(theta)
        for arm in ARMS:
            en = physical_cells(grid, geod, col, row, center, r["components"][arm])
            corrected[arm] = float(np.ptp(en @ axis))
            if actual_heading is not None:
                heading_spans[arm] = float(np.ptp(en @ actual_heading))
        out.update(geo_spans=corrected)
        if heading_spans:
            out["heading_spans"] = heading_spans
            counts["heading_rows"] += 1
        if r["fixed_k"] == 0:
            require(corrected["raw"] == corrected["fixed"] == corrected["free_matched_count"],
                    "Zero-budget footprint and actual projected length remain identical")
        spatial.append(out)

        if r["lon"] is not None and r["lat"] is not None:
            actual_col, actual_row = grid.inverse_polynomial(r["lon"], r["lat"])
            delta_x, delta_y = abs(actual_col-col), abs(actual_row-row)
            origin_values["inverse_center_col_abs_px"].append(delta_x)
            origin_values["inverse_center_row_abs_px"].append(delta_y)
            _, _, d = geod.inv(*center, r["lon"], r["lat"])
            require(np.isfinite(d), "Finite original AIS-point/crop-center separation")
            origin_values["center_to_ais_m"].append(float(d))
            counts["registration_rows"] += 1
            counts["exact_truncated_index_rows"] += int(int(actual_col) == col and int(actual_row) == row)
            counts["within_one_truncated_pixel_per_axis_rows"] += int(max(abs(int(actual_col)-col),abs(int(actual_row)-row)) <= 1)
        else:
            counts["missing_ais_coordinate_rows"] += 1

        rel = grid.time_seconds(col, row) - fixed_mid_time(scene)
        lag, speed = float(r["dt_sec"]), float(r["sog"]) * 1852 / 3600
        lo, hi = abs(lag-abs(rel)), lag+abs(rel)
        proxy0, proxy_lo, proxy_hi = speed*lag, speed*lo, speed*hi
        require(0 <= lag <= 120 and np.isfinite(rel) and speed >= 0 and proxy0 <= 50, "Original timing admission")
        counts["proxy_upper_over_50"] += int(proxy_hi > 50.)
        counts["proxy_lower_over_50"] += int(proxy_lo > 50.)
        counts["reported_speed_zero"] += int(speed == 0.)
        for key, value in zip(time_values, (rel, lag, lo, hi, proxy0, proxy_lo, proxy_hi), strict=True):
            time_values[key].append(value)
        if len(spatial) % 5 == 0:
            print(json.dumps(dict(phase="score", anchors=len(spatial))), flush=True)
    require(len(spatial) == counts["physical_rows"] == 25, "Complete frozen physical diagnostic")
    heading = [r for r in spatial if "heading_spans" in r]
    expected0 = old_module.arm_summary(spatial, "spans")
    require(expected0 == previous["arm_measurements"], "Exact original baseline means on identical frozen anchors")
    whole = dict(nominal=expected0, geolocated_native_axis=old_module.arm_summary(spatial,"geo_spans"),
                 nominal_pairing=old_module.contrast(spatial), geolocated_pairing=old_module.contrast(spatial,"geo_spans"),
                 paired_mae_reduction_geolocated_vs_nominal={arm:paired_reduction(old_module,spatial,"spans","geo_spans",arm)
                                                           for arm in ARMS})
    if heading:
        subset = dict(nominal=old_module.arm_summary(heading,"spans"), geolocated_native_axis=old_module.arm_summary(heading,"geo_spans"),
                      geolocated_ais_axis=old_module.arm_summary(heading,"heading_spans"),
                      nominal_pairing=old_module.contrast(heading), geo_native_pairing=old_module.contrast(heading,"geo_spans"),
                      ais_axis_pairing=old_module.contrast(heading,"heading_spans"),
                      raw_mae_reduction_ais_axis_vs_geo_native=paired_reduction(old_module,heading,"geo_spans","heading_spans"))
    else:
        subset = None
    result = dict(status=STATUS, phase="score", protocol_sha256=LOCK, source_protocol_sha256=OLD_LOCK,
                  worker_sha256=digest(Path(__file__).read_bytes()), anchors_worker_sha256=ANCHORS, grid_worker_sha256=GRID,
                  source_score_result_sha256=digest((root / "score/RESULT_SAFE.json").read_bytes()),
                  fixed_ship_count=len(spatial), fixed_groups=len(old_module.by_group(spatial)), positive_fixed_count=corrected_count,
                  evaluated_scenes=len(grids), counts=dict(counts), old_arm_metrics_exact=True, original_association_reused=True,
                  new_model_forwards=0, optimizer_updates=0, dependency_versions=dict(pyproj=pyproj.__version__,proj=pyproj.proj_version_str),
                  geometry={k:summarize(v) for k,v in origin_values.items()},
                  time_proxy={k:summarize(v) for k,v in time_values.items()}, whole_cohort=whole, same_positive_heading_subset=subset,
                  retrospective_diagnostics=True, wall_seconds=perf_counter()-elapsed)
    a.output.mkdir()
    raw = (json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False, indent=2)+"\n").encode()
    (a.output / "RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=STATUS,result_sha256=digest(raw))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code","closed-root","gulf-protocol","diagnostic-lock","geo-site","output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--diagnostic-lock-sha256",required=True)
    a = parser.parse_args()
    for name, entry in vars(a).items():
        if isinstance(entry,Path):
            setattr(a,name,entry.resolve(strict=name!="output"))
    run(a)


if __name__ == "__main__":
    main()
