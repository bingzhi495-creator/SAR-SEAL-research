"""Frozen native-component length scoring on the historical SAR Ship Gulf cohort."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import platform
import re
from time import perf_counter
from types import ModuleType
from zipfile import BadZipFile

import numpy as np
from pycocotools import mask as codec
from scipy import ndimage


PROTOCOL = "6001a3d9be9cf5cccb71e4156603d91c2dec1858ba46816a8380f2d315272cbb"
FROZEN = "78207f87a0c0003f3e768d7307df991f2cefdf5a8e4e51cde0c8a085ee9c16d1"
HEAD = "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1"
STATUS = "GULF_AIS_FROZEN_SCORE_COMPLETE"
ARMS = ("raw", "fixed", "free_matched_count", "free_natural")
SCENE = re.compile(r"S1A_IW_GRDH_1SDV_(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{6})_[A-F0-9]{6}_[A-F0-9]{4}")
MEMBER = re.compile(r"(\d{9})_(\d{8}T\d{6})\.npz")
STAMP = "%Y%m%dT%H%M%S"
CONNECT8 = np.ones((3, 3), dtype=bool)
NEAR_ROW, NEAR_COL = np.nonzero(np.sum((np.indices((7, 7)) - 3) ** 2, axis=0) <= 9)
NEAR_ROW, NEAR_COL = NEAR_ROW + 253, NEAR_COL + 253
DIST2 = (NEAR_ROW - 256) ** 2 + (NEAR_COL - 256) ** 2


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pinned(path, expected):
    data = path.read_bytes()
    require(digest(data) == expected, "Input digest mismatch: " + path.name)
    return data


def observed(directory, terminal_path, phase):
    terminal_bytes = terminal_path.read_bytes()
    terminal = json.loads(terminal_bytes)
    require(terminal["phase"] == phase and type(terminal["actual_exit"]) is int
            and terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True,
            "Observed successful " + phase + " required")
    result = json.loads(pinned(directory / "RESULT_SAFE.json", terminal["result_sha256"]))
    require(result["status"] == "GULF_AIS_FROZEN_" + phase.upper() + "_COMPLETE"
            and result["phase"] == phase and result["worker_sha256"] == terminal["worker_sha256"]
            and result["protocol_sha256"] == PROTOCOL and result["target_labels_opened"] is False
            and result["optimizer_updates"] == 0, "Unqualified previous phase: " + phase)
    return result, terminal["result_sha256"], digest(terminal_bytes)


def validate_edits(stock, edits):
    require(all(k.isascii() and k.isdigit() and str(int(k)) == k and 1 <= int(k) <= len(stock)
                for k in edits), "Unknown compact candidate position")
    for key, updates in edits.items():
        base = stock[int(key) - 1]["segmentation"]
        area = int(codec.area(base))
        require(32 <= area < 1024 and set(updates) <= set(ARMS[1:]) and
                updates.get("fixed", {}).get("swaps", 0) == updates.get("free_matched_count", {}).get("swaps", 0),
                "Unpermitted arm, candidate area, or matched budget")
        for change in updates.values():
            after, count = change["rle"], change["swaps"]
            require(set(change) == {"rle", "swaps"} and after["size"] == [512, 512]
                    and type(count) is int and 1 <= count <= min(8, area // 10)
                    and int(codec.area(after)) == area
                    and area - int(codec.area(codec.merge([base, after], intersect=True))) == count,
                    "Signed edit counts differ from archived masks")


def scalar(archive, key):
    require(archive.files.count(key) == 1, "Missing or ambiguous scalar")
    value = archive[key]
    require(value.shape == (), "Expected scalar field")
    return value.item()


def numeric(archive, key):
    value = scalar(archive, key)
    require(type(value) in (int, float), "Expected numerical scalar")
    return value


def integer(archive, key):
    value = numeric(archive, key)
    require(np.isfinite(value) and int(value) == value, "Expected integer scalar")
    return int(value)


def scalar_failures(archive, row, geometry):
    scene = row["scene_id"]
    path = PurePosixPath(row["path"])
    product, member = SCENE.fullmatch(scene), MEMBER.fullmatch(path.name)
    require(product and member and path.parts == ("chips", scene, path.name)
            and row["pass_id"] == product[3] + "-" + product[1][:8], "Invalid admitted source identity")
    begin, end = (datetime.strptime(product[n], STAMP) for n in (1, 2))
    midpoint = begin + (end - begin) / 2
    failures = set()
    try:
        raw_time = scalar(archive, "acq_time")
        actual = datetime.fromisoformat(raw_time)
        if actual.tzinfo is not None:
            actual = actual.astimezone(timezone.utc).replace(tzinfo=None)
        source_scene = scalar(archive, "scene_id")
        identity = integer(archive, "mmsi")
        if not (source_scene in (scene, scene + ".SAFE") and str(identity) == member[1]
                and actual == midpoint and member[2] == midpoint.strftime(STAMP)):
            failures.add("source_identity_or_time")
    except (TypeError, ValueError, OverflowError, KeyError, OSError, BadZipFile, EOFError):
        failures.add("source_identity_or_time")
    try:
        lag, speed = numeric(archive, "dt_sec"), numeric(archive, "sog")
        if not (np.isfinite(lag) and 0 <= lag <= 120):
            failures.add("invalid_or_outside_120s")
        if not (np.isfinite(speed) and 0 <= speed <= 60):
            failures.add("invalid_or_outside_speed")
        if np.isfinite(lag) and np.isfinite(speed) and lag >= 0 and speed >= 0:
            if lag * speed * 1852 / 3600 > 50:
                failures.add("speed_lag_over_50m")
    except (TypeError, ValueError, OverflowError, KeyError, OSError, BadZipFile, EOFError):
        failures.add("invalid_time_or_speed_scalar")
    try:
        if integer(archive, "n_other_ships") != 0:
            failures.add("other_ships_nonzero")
    except (TypeError, ValueError, OverflowError, KeyError, OSError, BadZipFile, EOFError):
        failures.add("invalid_other_ships_scalar")
    try:
        col, line = (integer(archive, key) for key in ("pixel_col", "pixel_row"))
        if not (256 <= col < geometry["width"] - 256 and 256 <= line < geometry["height"] - 256):
            failures.add("source_center_incomplete")
    except (TypeError, ValueError, OverflowError, KeyError, OSError, BadZipFile, EOFError):
        failures.add("invalid_source_center")
    return failures


def axis_for(indices):
    """Exact integer scatter decides the isotropic tie before conversion to floating point."""
    yy, xx = np.divmod(indices.astype(np.int64), 512)
    n = len(xx)
    sx, sy = int(xx.sum()), int(yy.sum())
    xx_scatter = n * int(xx @ xx) - sx * sx
    yy_scatter = n * int(yy @ yy) - sy * sy
    xy_scatter = n * int(xx @ yy) - sx * sy
    if xy_scatter == 0:
        return (1., 0.) if xx_scatter >= yy_scatter else (0., 1.)
    angle = .5 * np.arctan2(2 * xy_scatter, xx_scatter - yy_scatter)
    return float(np.cos(angle)), float(np.sin(angle))


def issue(indices):
    if not len(indices):
        return "missing_positive_overlap"
    yy, xx = np.divmod(indices, 512)
    if (yy == 0).any() or (xx == 0).any() or (yy == 511).any() or (xx == 511).any():
        return "edge_contact"
    if len(indices) < 2:
        return "fewer_than_two_centers"
    return None


def span(indices, axis):
    yy, xx = np.divmod(indices, 512)
    x, y = axis
    projection = x * xx + y * yy
    return float(10 * (np.ptp(projection) + abs(x) + abs(y)))


def native_component(mask):
    if not mask[NEAR_ROW, NEAR_COL].any():
        return None
    labels, _ = ndimage.label(mask, structure=CONNECT8)
    near = labels[NEAR_ROW, NEAR_COL]
    best_key, best = None, None
    for label in np.unique(near[near > 0]):
        indices = np.flatnonzero(labels == label)
        key = (int(DIST2[near == label].min()), -len(indices), int(indices[0]))
        if best_key is None or key < best_key:
            best_key, best = key, indices
    return best


def overlap_component(mask, native):
    labels, _ = ndimage.label(mask, structure=CONNECT8)
    values = labels.ravel()[native]
    ids, intersections = np.unique(values[values != 0], return_counts=True)
    if not len(ids):
        return np.array([], dtype=np.intp)
    best_key, best = None, None
    for label in ids[intersections == intersections.max()]:
        indices = np.flatnonzero(labels == label)
        key = (-len(indices), int(indices[0]))
        if best_key is None or key < best_key:
            best_key, best = key, indices
    return best


def measurements(stock, changes):
    for pos, candidate in enumerate(stock, 1):
        raw = codec.decode(candidate["segmentation"]).astype(bool)
        native = native_component(raw)
        if native is not None:
            break
    else:
        return None
    update = changes.get(str(pos), {})
    axes_exist = len(native) >= 2
    axis = axis_for(native) if axes_exist else None
    failures, spans, whole = {}, {}, {}
    for arm in ARMS:
        if arm in update:
            mask = codec.decode(update[arm]["rle"]).astype(bool)
            selected = overlap_component(mask, native)
        else:
            mask, selected = raw, native
        failure = issue(selected)
        if failure or not axes_exist:
            failures[arm] = failure or "native_axis_undefined"
            continue
        spans[arm] = span(selected, axis)
        all_indices = np.flatnonzero(mask)
        if issue(all_indices) is None:
            whole[arm] = span(all_indices, axis)
    return dict(spans=spans, whole=whole, failures=failures, fixed_k=update.get("fixed", {}).get("swaps", 0),
                natural_k=update.get("free_natural", {}).get("swaps", 0))


def by_group(rows, date=False):
    groups = defaultdict(list)
    for r in rows:
        key = r["pass_id"].split("-")[1] if date else r["pass_id"]
        groups[key].append(r)
    return [groups[k] for k in sorted(groups)]


def contrast(rows, field="spans", date=False):
    groups = by_group(rows, date)
    positive = [sum(r["fixed_k"] > 0 for r in g) for g in groups]
    result = dict(paired_ships=len(rows), paired_groups=len(groups), positive_fixed_ships=sum(positive),
                  positive_fixed_groups=sum(n > 0 for n in positive), zero_fixed_ships=len(rows)-sum(positive),
                  mae_reduction_m=None, ci95_m=None)
    if groups:
        values = np.asarray([np.mean([abs(r[field]["fixed"] - r["length"]) -
                                      abs(r[field]["free_matched_count"] - r["length"]) for r in g]) for g in groups])
        draws = np.random.Generator(np.random.PCG64(5217)).integers(len(values), size=(4096, len(values)))
        result.update(mae_reduction_m=float(np.mean(values)),
                      ci95_m=np.percentile(values[draws].mean(axis=1), [2.5, 97.5], method="linear").tolist())
    return result


def arm_summary(rows, field):
    result = {}
    for arm in ARMS:
        subset = [r for r in rows if arm in r[field]]
        groups = by_group(subset)
        means = []
        for group in groups:
            actual = np.array([r["length"] for r in group])
            proxy = np.array([r[field][arm] for r in group])
            means.append([proxy.mean(), actual.mean(), (proxy-actual).mean(), abs(proxy-actual).mean()])
        names = ("apparent_length_m", "ais_length_m", "signed_bias_m", "mae_m")
        values = np.mean(means, axis=0).tolist() if groups else [None] * len(names)
        result[arm] = dict(ships=len(subset), groups=len(groups), **dict(zip(names, values, strict=True)))
    return result


def closed_sources(a):
    require(a.protocol_sha256 == PROTOCOL, "Wrong prelock")
    pinned(a.protocol, PROTOCOL)
    path = Path(__file__).with_name("frozen.py")
    frozen = ModuleType("_gulf_frozen")
    frozen.__file__ = str(path)
    exec(compile(pinned(path, FROZEN), str(path), "exec"), frozen.__dict__)
    state = frozen.load_items(a)
    cap, cap_sha, population = frozen.verify_capture(a, state)
    rr, replay_sha, replay_term_sha = observed(a.replay, a.replay_terminal, "replay")
    _, cap_term_result, cap_term_sha = observed(a.capture, a.capture_terminal, "capture")
    intervals = []
    for term, expected_sha in ((a.intake_terminal, state.intake_terminal_sha256),
                               (a.capture_terminal, cap_term_sha), (a.replay_terminal, replay_term_sha)):
        metadata = json.loads(pinned(term, expected_sha))
        begin, end = (datetime.fromisoformat(metadata[key]) for key in ("started_utc", "ended_utc"))
        require(begin.tzinfo is not None and end.tzinfo is not None and begin <= end,
                "Invalid observer time interval")
        intervals.append((begin, end))
    require(intervals[0][1] <= intervals[1][0] and intervals[1][1] <= intervals[2][0],
            "Overlapping intake, capture, or replay phases")
    require(cap_term_result == cap_sha and rr["worker_sha256"] == cap["worker_sha256"] == FROZEN
            and rr["capture_result_sha256"] == cap_sha and rr["independent_decisions_equal"] is True
            and rr["ais_fields_opened"] is False and rr["point_head_sha256"] == HEAD
            and rr["detector_forwards"] == rr["model_forward_calls"] == 0
            and rr["counts"]["fixed_swaps"] == rr["counts"]["free_matched_count_swaps"]
            and all(rr[k] == cap[k] for k in ("images", "checkpoint_sha256", "intake_result_sha256",
                    "intake_terminal_sha256", "protocol_sha256", "lock_sha256", "manifest_sha256",
                    "predictions_sha256", "audit_sha256")), "Capture/replay linkage differs")
    edits = [json.loads(line) for line in pinned(a.replay / state.operator.EDITS, rr["edits_sha256"]).splitlines()]
    blocks = {}
    for item, (audit, pred), edit in zip(state.items, population, edits, strict=True):
        idx = item["image_id"]
        require(idx == audit["image_id"] == pred["image_id"] == edit["image_id"]
                and idx not in blocks, "Incomplete or reordered replay")
        validate_edits(pred["coco"], edit["changes"])
        blocks[idx] = pred["coco"], edit["changes"]
    require(len(blocks) == rr["counts"]["images"] == rr["images"], "Closed stream length differs")
    lineage = dict(protocol_sha256=PROTOCOL, intake_result_sha256=state.intake_result_sha256,
                   intake_terminal_sha256=state.intake_terminal_sha256, intake_sha256=state.intake_sha256,
                   census_result_sha256=state.manifest["census_result_sha256"], roster_sha256=state.roster_sha256,
                   capture_result_sha256=cap_sha, capture_terminal_sha256=cap_term_sha,
                   replay_result_sha256=replay_sha, replay_terminal_sha256=replay_term_sha,
                   predictions_sha256=rr["predictions_sha256"], audit_sha256=rr["audit_sha256"],
                   edits_sha256=rr["edits_sha256"], checkpoint_sha256=rr["checkpoint_sha256"],
                   point_head_sha256=HEAD, frozen_worker_sha256=FROZEN)
    return state.manifest, blocks, lineage


def original_npz(intake, row):
    return np.load(BytesIO(pinned(intake / row["archive_path"], row["archive_sha256"])), allow_pickle=False)


def analyze(intake, manifest, blocks):
    rows = manifest["items"]
    coverage = Counter(selected=len(rows), intake_ready=0, ready_first_decoded_identity=0,
                       duplicate_decoded_on_roster=0, admitted_ais_metadata=0, associated_native_mask=0)
    exclusions, scalar_rejections, intake_failures = Counter(), Counter(), Counter()
    failures = {arm: Counter() for arm in ARMS}
    measured = Counter({arm: 0 for arm in ARMS})
    seen, locked = set(), []
    for row in rows:
        identity = row["gray_sha256"]
        duplicate = identity is not None and identity in seen
        if identity is not None:
            seen.add(identity)
            coverage["duplicate_decoded_on_roster"] += int(duplicate)
        if row["status"] != "ready":
            exclusions["intake_failure"] += 1
            intake_failures[row["failure_category"]] += 1
            continue
        coverage["intake_ready"] += 1
        if duplicate:
            exclusions["decoded_duplicate"] += 1
            continue
        coverage["ready_first_decoded_identity"] += 1
        with original_npz(intake, row) as archive:
            reasons = scalar_failures(archive, row, manifest["scene_geometry"][row["scene_id"]])
        if reasons:
            exclusions["association_metadata"] += 1
            scalar_rejections.update(reasons)
            continue
        coverage["admitted_ais_metadata"] += 1
        record = measurements(*blocks[row["image_id"]])
        if record is None:
            exclusions["no_associated_native_mask"] += 1
            continue
        coverage["associated_native_mask"] += 1
        for arm in ARMS:
            if arm in record["failures"]:
                failures[arm][record["failures"][arm]] += 1
            else:
                measured[arm] += 1
        if record["failures"]:
            exclusions["any_arm_geometry_failure"] += 1
            continue
        record["pass_id"] = row["pass_id"]
        if record["fixed_k"] == 0:
            require(record["spans"]["fixed"] == record["spans"]["free_matched_count"] == record["spans"]["raw"],
                    "Zero fixed budget must yield exactly zero paired change")
        locked.append((row, record))
    # No AIS length member is opened until all candidate identities, axes and spans are fixed.
    cohort = []
    for row, record in locked:
        with original_npz(intake, row) as archive:
            try:
                value = numeric(archive, "length_m")
                valid = np.isfinite(value) and value >= 50
            except (TypeError, ValueError, OverflowError, KeyError, OSError, EOFError, BadZipFile):
                valid = False
        if valid:
            record["length"] = value
            cohort.append(record)
        else:
            exclusions["invalid_or_below_50m_length"] += 1
    coverage["all_arm_geometry_fixed"] = len(locked)
    coverage["length_members_requested"] = len(locked)
    coverage["all_arm_measured_with_valid_length"] = len(cohort)
    coverage["common_positive_fixed_budget"] = sum(r["fixed_k"] > 0 for r in cohort)
    coverage["common_zero_fixed_budget"] = len(cohort) - coverage["common_positive_fixed_budget"]
    dates = sorted({row["pass_id"].split("-")[1] for row in rows})
    require(len(rows) == sum(exclusions.values()) + len(cohort), "Unaccounted roster outcomes")
    primary = contrast(cohort)
    gates = dict(exposure=primary["positive_fixed_ships"] >= 20 and primary["positive_fixed_groups"] >= 8,
                 length_mean_at_least_half_m=primary["mae_reduction_m"] is not None and primary["mae_reduction_m"] >= .5,
                 length_lower_positive=primary["ci95_m"] is not None and primary["ci95_m"][0] > 0)
    complete_whole = [r for r in cohort if len(r["whole"]) == len(ARMS)]
    matched_extent = {}
    for arm in ARMS:
        available = by_group([r for r in cohort if arm in r["whole"]])
        differences = [np.mean([r["whole"][arm] - r["spans"][arm] for r in g]) for g in available]
        matched_extent[arm] = float(np.mean(differences)) if available else None
    dates_sensitivity = contrast(cohort, date=True)
    dates_sensitivity["multiple_dates_available"] = dates_sensitivity["paired_groups"] >= 2
    return dict(coverage=dict(coverage), selected_passes_with_chips=len({r["pass_id"] for r in rows}),
                selected_scene_start_date_range_utc=[dates[0], dates[-1]], distinct_selected_dates=len(dates),
                roster_exclusions_exclusive=dict(exclusions), intake_failure_categories=dict(intake_failures),
                metadata_rejections_nonexclusive=dict(scalar_rejections),
                associated_arm_measurement_failures={arm: dict(failures[arm]) for arm in ARMS},
                associated_arm_geometry_measurable=dict(measured), arm_measurements=arm_summary(cohort, "spans"),
                primary=primary, utc_date_sensitivity=dates_sensitivity,
                whole_mask_diagnostic=dict(arm_measurements=arm_summary(cohort, "whole"),
                                          paired_whole_minus_component_extent_m=matched_extent,
                                          all_four_uncensored_ships=len(complete_whole),
                                          paired_all_four=contrast(complete_whole, "whole")),
                gates=gates, passes_all_gates=all(gates.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("code", "intake", "intake-terminal", "capture", "capture-terminal", "replay",
                "replay-terminal", "protocol", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    a = parser.parse_args()
    started = perf_counter()
    require(not a.output.exists(), "Fresh score destination required")
    manifest, blocks, lineage = closed_sources(a)
    validated = perf_counter()
    result = analyze(a.intake, manifest, blocks)
    result.update(status=STATUS, phase="score", **lineage, worker_sha256=digest(Path(__file__).read_bytes()),
                  units="m", endpoint="Full native pixel cells projected on the frozen raw component major axis",
                  paired_contrast="absolute error fixed minus absolute error independent at exact fixed count",
                  bootstrap=dict(draws=4096, seed=5217, generator="PCG64", percentile=[2.5, 97.5]),
                  dependency_versions={k: version(k) for k in ("numpy", "scipy", "pycocotools", "Pillow")},
                  python_version=platform.python_version(), model_forwards=0, detector_forwards=0,
                  optimizer_updates=0, heading_opened=False, length_members_opened_only_after_geometry=True,
                  closure_verification_seconds=validated-started, scoring_seconds=perf_counter()-validated,
                  wall_seconds=perf_counter()-started)
    data = (json.dumps(result, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    a.output.mkdir()
    path = a.output / "RESULT_SAFE.json"
    with path.open("xb") as output:
        require(output.write(data) == len(data), "Incomplete score output")
    pinned(path, digest(data))
    print(json.dumps(dict(status=STATUS, result_sha256=digest(data))), flush=True)


if __name__ == "__main__":
    main()
