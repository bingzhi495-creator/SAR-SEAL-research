"""Exchange reference centroids while preserving each of the two closed polygon shapes."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

CLOSED = "be5ce158dc85426999bbc8b237e5edf50fd766751d02dd5b24bfc287ccf48a52"
SOURCE = "68c71e1c22cc6c3c82260b626a3738ba80067948d7a81874540c6ad1d7ea6511"
CELLS = ("integer_integer", "integer_float", "float_integer", "float_float")

def source(path, expected, name):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Exact closed source required: " + path.name)
    obj = ModuleType(name)
    obj.__file__ = str(path)
    sys.modules[name] = obj
    exec(compile(raw, str(path), "exec"), obj.__dict__)
    return obj

def make_cells(gt, left, right, checker):
    import numpy as np
    from shapely import box
    from shapely.affinity import translate

    images = {row["id"]: row for row in gt["images"]}
    cells = {cell: {} for cell in CELLS}
    neutral = set()
    displacement = []
    for ann in gt["annotations"]:
        aid = ann["id"]
        image = images[ann["image_id"]]
        canvas = box(0, 0, image["width"], image["height"])
        pi, ri = left[aid]
        pf, rf = right[aid]
        dx = pf.centroid.x - pi.centroid.x
        dy = pf.centroid.y - pi.centroid.y
        displacement.append((dx, dy))
        moved_i = translate(pi, xoff=dx, yoff=dy)
        moved_f = translate(pf, xoff=-dx, yoff=-dy)
        if moved_i.difference(canvas).area > 1e-9 or moved_f.difference(canvas).area > 1e-9:
            neutral.add(aid)
            continue
        for original, moved, desired in ((pi, moved_i, pf), (pf, moved_f, pi)):
            checker(abs(original.area - moved.area) <= 1e-8 * max(1, original.area)
                    and moved.centroid.distance(desired.centroid) < 1e-9, "Translation failed shape/centroid invariants")
        cells["integer_integer"][aid] = (pi, ri)
        cells["float_float"][aid] = (pf, rf)
        cells["integer_float"][aid] = (moved_i, [np.asarray(ring) + [dx, dy] for ring in ri])
        cells["float_integer"][aid] = (moved_f, [np.asarray(ring) - [dx, dy] for ring in rf])
    checker(len(neutral) + len(cells["integer_integer"]) == 380
            and all(set(v) == set(cells["integer_integer"]) for v in cells.values()),
            "Exact shared centroid/shape scoring population required")
    return cells, neutral, np.median(np.asarray(displacement), axis=0).tolist()

def contrasts(cells, metrics):
    ii, iff, fi, ff = (cells[key] for key in CELLS)
    return {key: dict(
        center_effect=.5*((iff[key] - ii[key]) + (ff[key] - fi[key])),
        shape_effect=.5*((fi[key] - ii[key]) + (ff[key] - iff[key])),
        interaction=(ff[key] - fi[key]) - (iff[key] - ii[key]))
        for key in metrics}

def run(a):
    started = perf_counter()
    sys.path.append(str(a.shapely_site))
    import shapely
    core = a.code_root
    prior = source(core/"SSDD_DEV187_CONTINUOUS_TWO_ENCODING_20261112/continuous_ssdd.py", SOURCE, "_ssdd_previous")
    need = prior.need
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and shapely.__version__ == "2.0.2"
         and shapely.geos_version_string == "3.11.2" and prior.version("numpy") == "1.26.4"
         and prior.version("pycocotools") == "2.0.11", "Exact original isolated CPU packages")
    def offline(event, _):
        if event.startswith("socket."):
            raise PermissionError("Closed local DEV diagnostics only")
    sys.addaudithook(offline)
    engine = prior.module(core/"HRSID_DEV705_CONTINUOUS_AP_20261111/full_continuous_ap.py",
        "cba94ecd0df862b6a303117a56fa889bc76c0a8c017af50fff2cdb9859f94139", "_ssdd_factor_engine")
    wrapped = prior.module(core/"SSDD_DEV187_POST_PRIMARY_RANK_20261110/ssdd_fixed_match_geometry.py",
        "5aebe92a3b355874e8a20f24ef19b98ace8c20e3537610ec92a9c0e026b260e3", "_ssdd_factor_adapter")
    adapter = wrapped.load_helpers(core, core/"HRSID_MASK_PHASE_REPEAT_DEV705_20261109/dev705_paired_mask_geometry.py")[4]
    runs = prior.module(core/"HRSID_COMMON_DEV705_FIXED_SHIFT_AP_20261111/fixed_shift_ap.py",
        "68a7ec9e0811cc5b18d2b97a09c13b2471f3908a36b39c3dedd4a50b098f97f2", "_ssdd_factor_runs").runs
    vec = core/"HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110"
    vector = prior.module(vec/"vector_label_geometry.py",
        "4877c48d38a03e58e97163381ba544df3c90a78dde85c9141085fa0b92da3b09", "_ssdd_factor_vector")
    footprint = prior.module(vec/"vector_population_math.py",
        "0821d2cc3b6ca793d33fdfe35b634f3678aad072fc55b86352fa2501e969fc7e", "_ssdd_factor_fp").run_footprint_metrics
    integer = json.loads(prior.pinned(a.integer_gt, prior.INTEGER_SHA))
    floating = json.loads(prior.pinned(a.float_gt, prior.FLOAT_SHA))
    need(integer["images"] == floating["images"] and len(integer["images"]) == 187
         and len(integer["annotations"]) == len(floating["annotations"]) == 380, "Exact DEV ground truth")
    reference = json.loads(prior.pinned(a.reference, prior.REFERENCE_SHA))
    prior_cont = json.loads(prior.pinned(a.closed_continuous, CLOSED))
    need(prior_cont["raster_full_exact"] and prior_cont["test"] is False
         and prior_cont["valid_common_polygons"] == 380, "Exact closed original continuous score")
    li, ln, _ = engine.polygon_support(integer, vector)
    ri, rn, _ = engine.polygon_support(floating, vector)
    need(len(li) == len(ri) == 380 and not ln and not rn, "Both accepted GT polygon sets must be valid")
    polygons, neutral, med = make_cells(integer, li, ri, need)
    data = {}
    for seed in prior.SEEDS:
        rows = {arm: json.loads(prior.pinned(getattr(a, "seed"+seed+"_"+arm),
                    reference["seeds"][seed]["prediction_sha256"][arm])) for arm in prior.ARMS}
        results, audits = {}, {}
        ranks = {}
        for cell in CELLS:
            metrics, audit = {}, {}
            for arm in prior.ARMS:
                metrics[arm], order, audit[arm] = engine.score(integer, rows[arm], adapter,
                    neutral, polygons[cell], runs, footprint)
                if arm in ranks:
                    need(ranks[arm] == order, "Original candidate ordering changed")
                ranks[arm] = order
                need(audit[arm]["dense_geometry_spots"] == 16 and audit[arm]["pruned_dense_spots"] == 8,
                     "Independent dense real overlap checks required")
                if not neutral and cell in ("integer_integer", "float_float"):
                    which = "integer" if cell == "integer_integer" else "candidate_float"
                    prior.exact_original(metrics[arm], prior_cont["seeds"][seed]["cells"][which]["continuous_common"][arm])
            results[cell] = metrics
            audits[cell] = audit
        deltas = {key: prior.paired_delta(value) for key,value in results.items()}
        data[seed] = dict(cells=results, deltas_points=deltas, factors_points=contrasts(deltas, prior.METRICS), audits=audits)
        print(json.dumps(dict(event="SSDD_CENTROID_SHAPE_SEED_COMPLETE", seed=seed, AP={k:v["AP"] for k,v in deltas.items()})), flush=True)
    report = dict(status="SSDD_DEV187_REFERENCE_CENTROID_SHAPE_FACTORIAL_ACTUAL", images=187, gt=380,
        neutral_crossed_out_of_frame=len(neutral), valid_common=380-len(neutral),
        median_author_float_centroid_delta_xy=med, seeds=data, source_continuous_sha256=CLOSED,
        fixed_prediction_scores_and_metadata=True, fit=False, dev=True, test=False,
        model_forward_calls=0, optimizer_updates=0, wall_seconds=perf_counter()-started)
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    with a.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(status=report["status"], sha256=sha256(raw).hexdigest())), flush=True)

def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("code-root", "shapely-site", "reference", "closed-continuous", "integer-gt", "float-gt", "output",
                 "seed5201-native", "seed5201-center", "seed5202-native", "seed5202-center"):
        parser.add_argument("--"+name, type=Path, required=True)
    a = parser.parse_args()
    if not a.output.is_absolute() or a.output.exists():
        parser.error("A fresh output path is required")
    run(a)

if __name__ == "__main__":
    main()