"""Measure fixed-prediction sensitivity to equal along-hull and across-hull contour translations."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

CENTROID_SOURCE = "c06e52a577682d0faa0475b18a9765904b902f36d83e981d4d9a89f2e26ac963"
SOURCE = "68c71e1c22cc6c3c82260b626a3738ba80067948d7a81874540c6ad1d7ea6511"
CELLS = ("base", "along_plus", "along_minus", "across_plus", "across_minus")
SHIFT = 0.5
ASPECT = 2.0

def load_module(path, expected, name):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Expected exact analysis dependency " + path.name)
    out = ModuleType(name)
    out.__file__ = str(path)
    sys.modules[name] = out
    exec(compile(raw, str(path), "exec"), out.__dict__)
    return out

def orient(axis, offset):
    direction = axis @ offset
    if direction < 0 or (direction == 0 and (axis[0] < 0 or (axis[0] == 0 and axis[1] < 0))):
        return -axis
    return axis

def cell_population(integer, old, floating, need):
    import numpy as np
    from shapely import box
    from shapely.affinity import translate

    by_id = {row["id"]: row for row in integer["images"]}
    polygons = {name:{} for name in CELLS}
    excluded = {"aspect_below_2":0, "crossed_canvas":0}
    neutral, counts = set(), {"small":0,"medium":0,"large":0}
    natural, self_iou, ratios = [], {key:[] for key in CELLS[1:]}, []
    for ann in integer["annotations"]:
        aid = ann["id"]
        geo, rings = old[aid]
        reference, _ = floating[aid]
        vertex = np.asarray(geo.minimum_rotated_rectangle.exterior.coords, dtype=float)
        edges = vertex[1:] - vertex[:-1]
        lengths = np.linalg.norm(edges, axis=1)
        need(len(lengths) == 4 and lengths.min() > 0, "Expected nondegenerate hull rectangle")
        ratio = lengths.max()/lengths.min()
        if ratio < ASPECT:
            excluded["aspect_below_2"] += 1
            neutral.add(aid)
            continue
        tangent = edges[lengths.argmax()]/lengths.max()
        normal = np.array([-tangent[1], tangent[0]])
        offset = np.asarray([reference.centroid.x-geo.centroid.x, reference.centroid.y-geo.centroid.y])
        tangent, normal = orient(tangent, offset), orient(normal, offset)
        need(abs(tangent @ normal) < 1e-12 and min(tangent @ offset, normal @ offset) >= -1e-12,
             "Axes must be orthogonal and orient consistently toward the observed label shift")
        shifts = dict(along_plus=SHIFT*tangent, along_minus=-SHIFT*tangent,
                      across_plus=SHIFT*normal, across_minus=-SHIFT*normal)
        image = by_id[ann["image_id"]]
        canvas = box(0,0,image["width"],image["height"])
        moved = {key:translate(geo, xoff=dx, yoff=dy) for key,(dx,dy) in shifts.items()}
        if any(shape.difference(canvas).area > 1e-9 for shape in moved.values()):
            excluded["crossed_canvas"] += 1
            neutral.add(aid)
            continue
        polygons["base"][aid] = (geo, rings)
        for key, shift in shifts.items():
            shape = moved[key]
            actual_shift = np.array([shape.centroid.x-geo.centroid.x,shape.centroid.y-geo.centroid.y])
            need(np.allclose(actual_shift,shift,rtol=0,atol=1e-9)
                 and abs(shape.area-geo.area) <= 1e-8*max(1,geo.area),
                 "Every translation must preserve shape area and realize equal displacement")
            polygons[key][aid] = (shape,[np.asarray(ring)+shift for ring in rings])
            cross = geo.intersection(shape).area
            self_iou[key].append(cross/(geo.area+shape.area-cross))
        natural.append([abs(tangent @ offset), abs(normal @ offset)])
        ratios.append(float(ratio))
        area = ann["area"]
        counts["small" if area < 1024 else "medium" if area < 9216 else "large"] += 1
    need(len(neutral)+len(natural)==380 and len(natural)>0
         and all(set(poly)==set(polygons["base"]) for poly in polygons.values()), "Common directional cohort required")
    loss = {key: float(100*(1-np.mean(vals))) for key,vals in self_iou.items()}
    return polygons, neutral, dict(excluded=excluded, supported_by_size=counts,
        median_natural_projected_shift_px=dict(zip(("along","across"),np.median(np.asarray(natural),axis=0).tolist())),
        median_aspect_ratio=float(np.median(ratios)), polygon_self_iou_drop_points=loss)

def summaries(deltas, names):
    out = {}
    for k in names:
        base = deltas["base"][k]
        along_odd = (deltas["along_plus"][k]-deltas["along_minus"][k])/2
        across_odd = (deltas["across_plus"][k]-deltas["across_minus"][k])/2
        out[k] = dict(along_odd=along_odd, across_odd=across_odd, across_minus_along_odd=across_odd-along_odd,
            along_even=(deltas["along_plus"][k]+deltas["along_minus"][k])/2-base,
            across_even=(deltas["across_plus"][k]+deltas["across_minus"][k])/2-base)
    return out

def main():
    parser = ArgumentParser(description=__doc__)
    for key in ("code-root","shapely-site","reference","integer-gt","float-gt","output",
                "seed5201-native","seed5201-center","seed5202-native","seed5202-center"):
        parser.add_argument("--"+key,type=Path,required=True)
    a = parser.parse_args()
    started = perf_counter()
    sys.path.append(str(a.shapely_site))
    import shapely
    core = a.code_root
    old = load_module(core/"SSDD_DEV187_CENTROID_SHAPE_FACTORIAL_20261112/centroid_factor.py",CENTROID_SOURCE,"_ssdd_dir_centroid")
    shared = old.source(core/"SSDD_DEV187_CONTINUOUS_TWO_ENCODING_20261112/continuous_ssdd.py",SOURCE,"_ssdd_dir_shared")
    need = shared.need
    need(not a.output.exists() and a.output.is_absolute() and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
         and shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2"
         and shared.version("numpy")=="1.26.4" and shared.version("pycocotools")=="2.0.11",
         "Exact closed CPU scope and fresh output required")
    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("Offline closed DEV only")
    sys.addaudithook(offline)
    engine = shared.module(core/"HRSID_DEV705_CONTINUOUS_AP_20261111/full_continuous_ap.py",
        "cba94ecd0df862b6a303117a56fa889bc76c0a8c017af50fff2cdb9859f94139","_ssdd_dir_engine")
    adapter = shared.module(core/"SSDD_DEV187_POST_PRIMARY_RANK_20261110/ssdd_fixed_match_geometry.py",
        "5aebe92a3b355874e8a20f24ef19b98ace8c20e3537610ec92a9c0e026b260e3","_ssdd_dir_adapter")
    official = adapter.load_helpers(core,core/"HRSID_MASK_PHASE_REPEAT_DEV705_20261109/dev705_paired_mask_geometry.py")[4]
    runs = shared.module(core/"HRSID_COMMON_DEV705_FIXED_SHIFT_AP_20261111/fixed_shift_ap.py",
        "68a7ec9e0811cc5b18d2b97a09c13b2471f3908a36b39c3dedd4a50b098f97f2","_ssdd_dir_runs").runs
    vector_root = core/"HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110"
    vector = shared.module(vector_root/"vector_label_geometry.py",
        "4877c48d38a03e58e97163381ba544df3c90a78dde85c9141085fa0b92da3b09","_ssdd_dir_vec")
    foot = shared.module(vector_root/"vector_population_math.py",
        "0821d2cc3b6ca793d33fdfe35b634f3678aad072fc55b86352fa2501e969fc7e","_ssdd_dir_fp").run_footprint_metrics
    ref = json.loads(shared.pinned(a.reference,shared.REFERENCE_SHA))
    gt = json.loads(shared.pinned(a.integer_gt,shared.INTEGER_SHA))
    floating = json.loads(shared.pinned(a.float_gt,shared.FLOAT_SHA))
    pin, nn, _ = engine.polygon_support(gt,vector)
    pfloat, nf, _ = engine.polygon_support(floating,vector)
    need(gt["images"]==floating["images"] and not nn and not nf and len(pin)==len(pfloat)==380,
         "Exact 380 paired geometry inputs required")
    cells, neutral, population = cell_population(gt,pin,pfloat,need)
    all_results = {}
    for seed in shared.SEEDS:
        paths = {arm:getattr(a,"seed"+seed+"_"+arm) for arm in shared.ARMS}
        rows = {arm:json.loads(shared.pinned(path,ref["seeds"][seed]["prediction_sha256"][arm]))
                for arm,path in paths.items()}
        cell_metrics, ranks, checks = {}, {}, {}
        for mode, poly in cells.items():
            cell_metrics[mode], checks[mode] = {}, {}
            for arm in shared.ARMS:
                m, r, audit = engine.score(gt,rows[arm],official,neutral,poly,runs,foot)
                need(arm not in ranks or ranks[arm]==r, "Unchanged original candidates and order required")
                ranks[arm] = r
                need(audit["dense_geometry_spots"]==16 and audit["pruned_dense_spots"]==8,
                     "Require real dense checks of continuous overlap")
                cell_metrics[mode][arm], checks[mode][arm] = m,audit
        deltas = {mode:shared.paired_delta(metrics) for mode,metrics in cell_metrics.items()}
        effect = summaries(deltas,shared.METRICS)
        all_results[seed] = dict(cells=cell_metrics,delta_points=deltas,direction_response_points=effect,geometry_audit=checks)
        print(json.dumps(dict(event="SSDD_HULL_SHIFT_SEED_COMPLETE",seed=seed,AP=effect["AP"])),flush=True)
    payload = dict(status="SSDD_DEV187_EQUAL_HULL_DIRECTION_SHIFT_ACTUAL",images=187,gt=380,valid_common=380-len(neutral),
        shift_px=SHIFT,aspect_threshold=ASPECT,population=population,seeds=all_results,fixed_candidates=True,
        fit=False,dev=True,test=False,model_forward_calls=0,optimizer_updates=0,seconds=perf_counter()-started)
    raw=(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(status=payload["status"],result_sha256=sha256(raw).hexdigest())),flush=True)

if __name__=="__main__":
    main()