"""Replay closed frozen-head occupancy and add equal-magnitude hull-axis targets."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean
from types import ModuleType, SimpleNamespace


PREVIOUS = "HRSID_FIT_FROZEN_VECTOR_OCCUPANCY_20261111/vector_occupancy_ablation.py"
PREVIOUS_SHA = "b7d218ac22c562d7d4e6ece664d5bb9aeb7a23242100a42ea04e968863682e06"
PREVIOUS_RESULT_SHA = "d0c13432257a374c010e6bdecbfdb621a0b61d2756a2440842205c0f934f18e1"
PRELOCK_SHA = "6406ccbb4ad080db0c068cd04c6fb65f7aed8d98d2e0ac6c28f8213a5fdd2070"
ORIENTED = ("along_occupancy", "across_occupancy", "four_direction_occupancy")
CONTROL = "composed_hard"
STATUS = "HRSID_FIT_FIXED_POINTS_ORIENTED_OCCUPANCY_COMPLETE"
PROJECTION_FIELDS = ("optimizer_updates", "events", "initial_sha256", "parameter_sha256", "private_archive_sha256")


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def pinned(path, expected):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "Canonical closed input required")
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Exact pinned input changed: " + path.name)
    return raw


def area_values(shape, footprint, inverse):
    import numpy as np
    import shapely

    values = np.asarray(shapely.area(shapely.intersection(footprint, shape)), dtype=np.float64)
    require(np.isfinite(values).all() and np.all((values >= -1e-10) & (values <= 1 + 1e-10)),
            "Continuous polygon intersection must be contained within its unit cell")
    return np.clip(values, 0, 1)[inverse]


def oriented_targets(shape, hw, cells, unshifted):
    """Measure the same continuous visible shape at equal displacement magnitudes."""
    import numpy as np
    import shapely
    from shapely.affinity import translate

    fallback = dict.fromkeys(ORIENTED, unshifted)
    if shape.is_empty or shape.area == 0:
        return fallback, "empty_visible"
    require(shape.is_valid, "Valid transformed visible polygon union required")
    rectangle = shape.minimum_rotated_rectangle
    require(rectangle.geom_type == "Polygon", "Nondegenerate enclosing rectangle required")
    vertex = np.asarray(rectangle.exterior.coords, dtype=np.float64)
    edges = vertex[1:] - vertex[:-1]
    lengths = np.linalg.norm(edges, axis=1)
    require(len(lengths) == 4 and np.isfinite(lengths).all() and lengths.min() > 0,
            "Finite four-sided enclosing rectangle required")
    if lengths.max() / lengths.min() < 2.:
        return fallback, "aspect_below_2"
    along = edges[lengths.argmax()] / lengths.max()
    across = np.array([-along[1], along[0]])
    shifts = [sign * .5 * axis for axis in (along, across) for sign in (1., -1.)]
    require(all(abs(np.linalg.norm(shift) - .5) < 1e-12 for shift in shifts)
            and abs(along @ across) < 1e-12, "Equal half-pixel orthogonal translations required")
    moved = [translate(shape, xoff=dx, yoff=dy) for dx, dy in shifts]
    height, width = hw
    frame = shapely.box(0, 0, width, height)
    if any(g.difference(frame).area > 1e-10 for g in moved):
        return fallback, "frame_crossing"
    require(all(g.is_valid and abs(g.area - shape.area) <= 1e-9 * max(1., shape.area) for g in moved),
            "Each translation must preserve the original visible continuous area")
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    x, y = unique.T
    footprint = shapely.box(x, y, x + 1, y + 1)
    require(np.allclose(area_values(shape, footprint, inverse), unshifted, atol=1e-10, rtol=0),
            "Exact unit-cell treatment must agree with the verified unshifted occupancy")
    along_plus, along_minus, across_plus, across_minus = [area_values(g, footprint, inverse) for g in moved]
    along_values = .5 * (along_plus + along_minus)
    across_values = .5 * (across_plus + across_minus)
    return dict(zip(ORIENTED, (along_values, across_values, .5*(along_values + across_values)), strict=True)), "eligible"


def add_oriented_banks(views, np, prior, selected, args, previous, old):
    import shapely
    from shapely.geometry import Polygon

    vector_counts = previous.add_vector_banks(views, np, prior, selected, args)
    require(vector_counts == old["vector_targets"], "Original P398 vector target stream must replay exactly")
    heights = prior.SIZE
    height, width = heights
    frame = shapely.box(0, 0, width, height)
    digests = {arm: sha256() for arm in ORIENTED}
    counts, row_reasons, point_reasons = Counter(), Counter(), Counter()
    by_arm = {arm: Counter() for arm in ORIENTED}
    mass_delta = {arm: 0. for arm in ORIENTED}
    for view in views:
        for row in view["rows"]:
            vector = row["banks"][previous.VECTOR]
            base = vector.targets
            rings = row["rings"]
            shape = shapely.union_all([Polygon(ring) for ring in rings]).intersection(frame)
            values, reason = oriented_targets(shape, heights, row["cells"][vector.indices], base)
            row_reasons[reason] += 1
            point_reasons[reason] += len(base)
            base_fp32 = base.astype("<f4")
            for arm, target in values.items():
                frozen = np.frombuffer(target.astype("<f8", copy=False).tobytes(), dtype=np.float64)
                require(frozen.shape == base.shape and not frozen.flags.writeable
                        and np.isfinite(frozen).all() and np.all((frozen >= 0) & (frozen <= 1)),
                        "Aligned immutable target values in unit interval required")
                row["banks"][arm] = SimpleNamespace(indices=vector.indices, targets=frozen)
                digests[arm].update(frozen.astype("<f8", copy=False).tobytes())
                by_arm[arm]["different_from_vector_gt_1e_10"] += int(np.count_nonzero(np.abs(frozen - base) > 1e-10))
                by_arm[arm]["changed_consumed_fp32"] += int(np.count_nonzero(frozen.astype("<f4") != base_fp32))
                by_arm[arm]["fractional_targets"] += int(np.count_nonzero((frozen > 0) & (frozen < 1)))
                mass_delta[arm] += float(np.sum(frozen - base, dtype=np.float64))
            counts["points"] += len(base)
            counts["along_diff_across_fp32"] += int(np.count_nonzero(
                values[ORIENTED[0]].astype("<f4") != values[ORIENTED[1]].astype("<f4")))
    require(sum(row_reasons.values()) == 31 and counts["points"] == sum(point_reasons.values()) == 487,
            "Exact 31-row and 487-point historical scope required")
    require(counts["along_diff_across_fp32"] > 0 and all(row["changed_consumed_fp32"] > 0 for row in by_arm.values()),
            "The equal-displacement orientation trial must realize distinct point targets")
    return dict(unshifted=vector_counts, oriented=dict(
        point_counts=dict(counts), row_reasons=dict(row_reasons), point_reasons=dict(point_reasons),
        by_arm={arm: dict(values, target_stream_sha256=digests[arm].hexdigest(),
                          selected_target_mass_delta=mass_delta[arm]) for arm, values in by_arm.items()}))


def compare_orientation(measured, p262, original, reports, selected, previous, old):
    require(selected == old["selected_training"], "Original selected points and targets must replay exactly")
    old_held = {path: {arm: scores[arm] for arm in previous.ALL} for path, scores in measured.items()}
    require(old_held == old["held_metrics"], "Every old per-source held metric must replay exactly")
    require(all(all(reports[arm][key] == old["arms"][arm][key] for key in PROJECTION_FIELDS)
                for arm in (*previous.BASE_ARMS, previous.VECTOR)),
            "Original projection updates, losses, state and private archive digests must replay exactly")
    require(previous.compare_vector(old_held, p262, original, reports, selected) == old["held_comparison"],
            "Exact prior point-value comparison required")
    for path, scores in measured.items():
        require(set(scores) == set((*previous.ALL, *ORIENTED)) and all(len(rows) == 8 for rows in scores.values()),
                "Eight exact held sources per path and arm required")
    result = {}
    for path, scores in measured.items():
        means = {arm: {key: fmean(row[key] for row in rows) for key in rows[0]} for arm, rows in scores.items()}
        contrasts = {}
        along = ORIENTED[0]
        for arm, rows in scores.items():
            if arm == along:
                continue
            delta = [a["polygon_iou"] - b["polygon_iou"] for a, b in zip(scores[along], rows, strict=True)]
            contrasts[arm] = dict(per_source_polygon_iou=delta, mean=fmean(delta),
                                  positive=sum(x > 0 for x in delta), ties=sum(x == 0 for x in delta))
        item = dict(means=means, along_minus=contrasts)
        if path == "crop":
            primary = contrasts[CONTROL]
            checks = dict(mean_at_least_005=primary["mean"] >= .005, positive_groups_at_least_5=primary["positive"] >= 5)
            item.update(primary_control=CONTROL, gate_checks=checks, feasibility_gate=all(checks.values()))
        result[path] = item
    return result


def adapted(previous, source):
    text = previous.adapted(source)
    replacements = (
        ('(*ARMS, VECTOR)', '(*ARMS, VECTOR, *ORIENTED)', 2),
        ('prior.ALL = (*ALL, VECTOR)', 'prior.ALL = (*ALL, VECTOR, *ORIENTED)', 1),
        ('held_comparison=compare_vector(measured, p262_reference, held_comparison, reports, selected),',
         'held_comparison=compare_orientation(measured, p262_reference, held_comparison, reports, selected),', 1),
        ('vector_targets=vector_targets, vector_protocol_sha256=VECTOR_PRELOCK_SHA,',
         'target_geometry=vector_targets, orientation_protocol_sha256=VECTOR_PRELOCK_SHA,\n'
         '                  p398_reference_sha256=P398_REFERENCE_SHA, p398_all_controls_exact=True,', 1),
    )
    for before, after, count in replacements:
        require(text.count(before) == count, "Exact P398 adaptation point changed")
        text = text.replace(before, after)
    return text


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "bundle", "checkpoint", "receipts", "shapely-site", "output",
                 "p253-reference", "p262-reference", "p398-reference", "prelock"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    pinned(args.prelock, PRELOCK_SHA)
    path = args.code / PREVIOUS
    previous = ModuleType("_p398_closed_vector_extension")
    previous.__file__ = str(path)
    exec(compile(pinned(path, PREVIOUS_SHA), str(path), "exec"), previous.__dict__)
    old = json.loads(pinned(args.p398_reference, PREVIOUS_RESULT_SHA))
    p262 = json.loads(previous.pinned(args.p262_reference, previous.REFERENCE_SHA))
    require(old["status"] == previous.STATUS and p262["status"] == "P262_HRSID_FIXED_POINTS_TARGET_VALUE_COMPLETE_V1",
            "Successful closed P398 and P262 required")
    path = args.code / previous.BASE
    original = ModuleType("_closed_p262_orientation_extension")
    original.__file__ = str(path)
    original.__dict__.update(
        p262_reference=p262, VECTOR=previous.VECTOR, ORIENTED=ORIENTED, VECTOR_STATUS=STATUS,
        VECTOR_PRELOCK_SHA=PRELOCK_SHA, VECTOR_REFERENCE_SHA=previous.REFERENCE_SHA, P398_REFERENCE_SHA=PREVIOUS_RESULT_SHA,
        add_vector_banks=lambda *x: add_oriented_banks(*x, previous, old),
        compare_orientation=lambda *x: compare_orientation(*x, previous, old))
    source = previous.pinned(path, previous.BASE_SHA).decode("utf-8")
    exec(compile(adapted(previous, source), str(path), "exec"), original.__dict__)
    args.protocol = previous.BASE_PROTOCOL
    original.run(args)


if __name__ == "__main__":
    main()
