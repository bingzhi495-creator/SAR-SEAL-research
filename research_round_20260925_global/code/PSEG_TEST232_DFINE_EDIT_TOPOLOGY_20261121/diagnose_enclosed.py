"""Explain whether newly enclosed deletions overlap the original reference polygon."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
from scipy.ndimage import label, generate_binary_structure


GEOMETRY_SHA = "107f31864bbb7e807a4194e552c7a1778d410f0e564f80437c5717fe6b47098d"
ARMS = ("fixed", "free_matched_count", "native_matched_count", "free_natural")
BACKGROUND = generate_binary_structure(2, 1)


def hole_pixels(mask):
    rows, cols = np.nonzero(mask)
    result = np.zeros(mask.shape,dtype=bool)
    if len(rows):
        y, x = slice(rows.min(), rows.max()+1), slice(cols.min(), cols.max()+1)
        padded = np.pad(mask[y,x], 1)
        components, _ = label(~padded, structure=BACKGROUND)
        result[y,x] = (~padded & (components != components[0,0]))[1:-1,1:-1]
    return result


def measure(raw, rows, expected_swaps, shape, shapely, polygon):
    modified = codec.decode(rows).astype(bool)
    if modified.shape != shape:
        raise ValueError("Original native raster")
    deleted = raw & ~modified
    if np.count_nonzero(deleted) != expected_swaps or np.count_nonzero(~raw & modified) != expected_swaps:
        raise ValueError("Area and edit counts from original writeback")
    if not expected_swaps:
        return np.zeros(4,dtype=np.float64)
    outside = hole_pixels(modified)
    y, x = np.nonzero(deleted)
    cell = shapely.box(x,y,x+1,y+1)
    truth_fraction = shapely.area(shapely.intersection(cell,polygon))
    enclosed = outside[y,x]
    return np.array([len(y), enclosed.sum(), truth_fraction.sum(), truth_fraction[enclosed].sum()])


def summarize(sample, shapely, truth):
    group = defaultdict(list)
    totals = {k:np.zeros(4,dtype=np.float64) for k in ARMS}
    affected = Counter()
    for annotation, (record, rank) in sample:
        native = record["native"][rank-1]["segmentation"]
        area = int(codec.area(native))
        if not 32 <= area < 1024:
            continue
        polygon = truth[annotation][0]
        raw = codec.decode(native).astype(bool)
        edit = record["edit"].get(str(rank),{})
        values = []
        for arm in ARMS:
            row = edit.get(arm, {})
            v = measure(raw, row.get("rle",native), row.get("swaps",0), raw.shape, shapely,polygon)
            totals[arm] += v
            affected.update({arm+"_candidates_with_enclosed_deletions":int(v[1]>0)})
            values.append(100*v[3]/polygon.area)
        group[record["rgb"]].append(values)
    average = np.asarray([np.mean(group[k],axis=0) for k in sorted(group)])
    population = sum(map(len,group.values()))
    rng = np.random.Generator(np.random.PCG64(5217))
    iid = rng.integers(len(average), size=(4096,len(average)))
    effects = {}
    for a,b in ((1,0),(3,2)):
        differences = average[:,a] - average[:,b]
        effects[ARMS[a]+" minus "+ARMS[b]] = dict(
            mean_points=float(differences.mean()),
            interval95_points=np.percentile(differences[iid].mean(axis=1),[2.5,97.5]).tolist())
    keys=("all_deleted_cells","enclosed_deleted_cells","true_polygon_area_all_deleted_cells","true_polygon_area_enclosed_deleted_cells")
    return dict(eligible_pairs=population, contributing_decoded_images=len(group),
                pooled_edits={a:dict(zip(keys,values.tolist())) for a,values in totals.items()},
                candidates_affected=dict(affected),
                average_area_of_enclosed_true_deletions_points=dict(zip(ARMS,average.mean(axis=0).tolist())),
                paired_contrasts=effects)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ("diagnostic","score","gt","manifest","native","audit","edits","geometry","code","shapely-site","output"):
        parser.add_argument("--"+n,type=Path,required=True)
    a=parser.parse_args()
    if not sys.flags.isolated or not sys.dont_write_bytecode or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Exact CPU-only diagnostic")
    source=a.diagnostic.read_bytes()
    if sha256(source).hexdigest()!=GEOMETRY_SHA:
        raise ValueError("Exact closed geometry provenance source")
    audit=ModuleType("_prior_pseg_geometry_audit")
    audit.__file__=str(a.diagnostic)
    exec(compile(source,str(a.diagnostic),"exec"),audit.__dict__)
    sys.path.append(str(a.shapely_site.resolve(strict=True)))
    import shapely
    if shapely.__version__!="2.0.2" or shapely.geos_version_string!="3.11.2":
        raise ValueError("Prior exact polygon geometry runtime")
    start=perf_counter()
    score,gt,records=audit.read_data(a)
    geom=audit.load(a.geometry,audit.GEOM)
    engine=geom.geometry_modules(a.code)
    truth,neutral,_=engine.geometry.polygon_support(gt,engine.vector)
    if neutral:
        raise ValueError("All original floating polygons valid")
    views={}
    for scope,omit in (("official",set()),("exclude_exact_fit_overlap",audit.OMIT)):
        ids=sorted(i for i,row in records.items() if int(row["name"].split(".")[0]) not in omit)
        result,_=audit.evaluate_raw(gt,records,ids,score["views"][scope]["metrics_fraction"]["raw"])
        pairs=[(annotation,(records[i],rank)) for annotation,(i,rank) in sorted(result.items())]
        views[scope]=summarize(pairs,shapely,truth)
    if views["official"]["eligible_pairs"]!=293 or views["exclude_exact_fit_overlap"]["eligible_pairs"]!=279:
        raise ValueError("Original paired geometry sample")
    result=dict(status="PSEG_TEST232_ENCLOSED_DELETION_GEOMETRY_COMPLETE",
                comparison="Post hoc historical test localized to original COCO .50 fixed pairs",
                original_score_sha256=audit.SCORE, geometry_audit_source_sha256=GEOMETRY_SHA,
                source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                detector_forwards=0,optimizer_updates=0,
                definition="Deleted stock foreground cells in a 4-connected enclosed background region of the edited raster",
                views=views,seconds=perf_counter()-start)
    raw=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as out:
        out.write(raw)
    print(json.dumps({"status":result["status"],"result_sha256":sha256(raw).hexdigest()}),flush=True)


if __name__=="__main__":
    main()
