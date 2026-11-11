"""Add vector-area supervision to a pinned, closed three-arm FIT replay."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean
from types import ModuleType, SimpleNamespace


BASE = "HRSID_FIT_FIXED_POINTS_TARGET_VALUE_20261110/run_target_value_ablation.py"
BASE_SHA = "e51e1d27036dfeac514c90679a00e8d7025c50ca716989f35eec434aa1038c49"
REFERENCE_SHA = "c59e8968fc2149d73760dc401ed4dfc90cd814ae25acf4ac58895e9df8a362d2"
OCCUPANCY_SHA = "1a54d094c39db2620444b580030e210cc1d2d3ed7ed6cec1456fc72134a46f06"
PRELOCK_SHA = "aa13fb2deac73ee873648b016efd0ca80981338fbfda49a13ca98f99c9f546ee"
BASE_PROTOCOL = "81f95dd7612aaf39362cb9d789e54ed3a84f164b"
BASE_ARMS = ("stepwise_hard", "composed_hard", "raster_occupancy")
VECTOR = "vector_occupancy"
ALL = ("anchor", *BASE_ARMS, VECTOR)
STATUS = "HRSID_FIT_FIXED_POINTS_VECTOR_OCCUPANCY_COMPLETE"


def pinned(path, expected):
    if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("Canonical dependency required: " + path.name)
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Pinned dependency changed: " + path.name)
    return raw


def add_vector_banks(views, np, prior, selected, args):
    import point_occupancy
    expected = args.code / "HRSID_FIT_FROZEN_POINT_PROBE_20261110/point_occupancy.py"
    if Path(point_occupancy.__file__).resolve() != expected:
        raise ValueError("Unexpected polygon occupancy module")
    pinned(expected, OCCUPANCY_SHA)
    digest = sha256()
    points = fractional = changed = empty_vectors = 0
    for view in views:
        for row in view["rows"]:
            raster = row["banks"]["raster_occupancy"]
            # The original loader has already applied the sealed geometry to these rings.
            final_segmentation = [ring.reshape(-1).tolist() for ring in row["rings"]]
            if final_segmentation:
                values = point_occupancy.polygon_cell_occupancy(
                    final_segmentation, prior.SIZE, (), row["cells"][raster.indices],
                )
            else:
                values = np.zeros(raster.indices.shape, dtype=np.float64)
                empty_vectors += 1
            target = np.frombuffer(values.astype("<f8", copy=False).tobytes(), dtype=np.float64)
            if target.shape != raster.indices.shape or target.flags.writeable:
                raise ValueError("Vector occupancy must retain the historical immutable point bank")
            row["banks"][VECTOR] = SimpleNamespace(indices=raster.indices, targets=target)
            digest.update(target.astype("<f8", copy=False).tobytes())
            points += target.size
            fractional += int(np.count_nonzero((target > 0) & (target < 1)))
            changed += int(np.count_nonzero(np.abs(target - raster.targets) > 1e-10))
    if points != selected["points"] or points != 487 or min(changed, fractional) <= 0:
        raise ValueError("Exact historical point count and an actual vector treatment are required")
    return dict(points=points, fractional_targets=fractional, different_from_raster=changed,
                empty_visible_vector_rows=empty_vectors,
                selected_target_stream_sha256=digest.hexdigest())


def compare_vector(measured, reference, original, reports, selected):
    for arm in BASE_ARMS:
        if any(reports[arm][key] != reference["arms"][arm][key] for key in (
                "optimizer_updates", "events", "initial_sha256", "parameter_sha256",
                "private_archive_sha256")):
            raise ValueError("A historical control did not reproduce exactly: " + arm)
    if selected != reference["selected_training"]:
        raise ValueError("The original point and target ledgers must match exactly")
    result = {}
    for path, scores in measured.items():
        baseline = {arm: scores[arm] for arm in ("anchor", *BASE_ARMS)}
        if baseline != reference["held_metrics"][path]:
            raise ValueError("The full held per-source control results did not reproduce: " + path)
        if set(scores) != set(ALL) or any(len(rows) != 8 for rows in scores.values()):
            raise ValueError("Every comparison requires the exact eight paired held sources")
        metrics = tuple(scores[VECTOR][0])
        means = {arm: {metric: fmean(row[metric] for row in rows) for metric in metrics}
                 for arm, rows in scores.items()}
        contrasts = {}
        for arm, rows in baseline.items():
            values = [a["polygon_iou"] - b["polygon_iou"] for a, b in zip(scores[VECTOR], rows, strict=True)]
            contrasts[arm] = dict(per_source_polygon_iou=values, mean=fmean(values),
                                  positive=sum(value > 0 for value in values))
        output = dict(means=means, vector_minus=contrasts,
                      original_comparison=original(baseline, primary=path == "crop"))
        if path == "crop":
            guard = means[VECTOR]["center_iou"] - max(means[arm]["center_iou"] for arm in baseline)
            primary = contrasts["raster_occupancy"]
            gates = dict(paired_iou_at_least_005=primary["mean"] >= .005,
                         positive_sources_at_least_5=primary["positive"] >= 5,
                         versus_anchor_nonnegative=contrasts["anchor"]["mean"] >= 0,
                         hard_center_iou_guard_at_least_minus_002=guard >= -.002)
            output.update(gate_checks=gates, feasibility_gate=all(gates.values()), center_iou_guard=guard)
        result[path] = output
    return result


def adapted(source):
    edits = (
        ('STATUS = "P262_HRSID_FIXED_POINTS_TARGET_VALUE_COMPLETE_V1"', 'STATUS = VECTOR_STATUS'),
        ('    selected = selected_training(train_views, np)\n', '    selected = selected_training(train_views, np)\n'
         '    vector_targets = add_vector_banks(train_views, np, prior, selected, args)\n'),
        ('    for arm in ARMS:\n        projections[arm]', '    for arm in (*ARMS, VECTOR):\n        projections[arm]'),
        ('for event in reports[arm]["events"]] for arm in ARMS]', 'for event in reports[arm]["events"]] for arm in (*ARMS, VECTOR)]'),
        ('ledgers[0] == ledgers[1] == ledgers[2] and', 'all(ledger == ledgers[0] for ledger in ledgers) and'),
        ('    prior.ALL = ALL', '    prior.ALL = (*ALL, VECTOR)'),
        ('                  held_comparison={path: held_comparison(scores, primary=path == "crop") for path, scores in measured.items()},',
         '                  held_comparison=compare_vector(measured, p262_reference, held_comparison, reports, selected),\n'
         '                  vector_targets=vector_targets, vector_protocol_sha256=VECTOR_PRELOCK_SHA,\n'
         '                  p262_reference_sha256=VECTOR_REFERENCE_SHA, p262_all_controls_exact=True,'),
    )
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError("Original P262 insertion point differs")
        source = source.replace(old, new, 1)
    return source


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "bundle", "checkpoint", "receipts", "shapely-site", "output",
                 "p253-reference", "p262-reference", "prelock"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    pinned(args.prelock, PRELOCK_SHA)
    reference = json.loads(pinned(args.p262_reference, REFERENCE_SHA))
    if reference["status"] != "P262_HRSID_FIXED_POINTS_TARGET_VALUE_COMPLETE_V1":
        raise ValueError("Successful closed P262 required")
    path = args.code / BASE
    original = ModuleType("_closed_p262_vector_extension")
    original.__file__ = str(path)
    original.__dict__.update(p262_reference=reference, VECTOR=VECTOR, VECTOR_STATUS=STATUS,
                             VECTOR_PRELOCK_SHA=PRELOCK_SHA, VECTOR_REFERENCE_SHA=REFERENCE_SHA,
                             add_vector_banks=add_vector_banks, compare_vector=compare_vector)
    exec(compile(adapted(pinned(path, BASE_SHA).decode()), str(path), "exec"), original.__dict__)
    args.protocol = BASE_PROTOCOL
    original.run(args)


if __name__ == "__main__":
    main()
