"""Observe one frozen carrier cohort under two exact overlap conventions."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np


LOCK = "453cc178e7409f7cd61708b59f90dac94720db981b37c1d1596ce02af69daadd"
INPUTS = "8d5d6b28279bd492de9585f570ca0a8f23919cdfadc0e4214602681ebc94e554"
CONTINUOUS = "2879b8e946e8ceb34736a9cbc4f497b2559d8d555130eac43368681f3cbdc392"
AP90 = "2c9baee1159a25c2e5f8f077c233b4d497b9ffd6c0a45abd49d8a0c66eab3753"
OLD = "7039d7d0336f6a0cc3b64bfbe4b0c446b38b073cc90764f446a4dc6d09f90d85"
RESERVE = "5ff671b2b83bb12e322b0c766897bf3cdacaedd233bf0d6edcdfcccf2c938df4"
REPRS = ("supported_raster", "continuous")
STATUS = "RSDD_STRICT_CONTINUOUS_SCORE_COMPLETE"


def check(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    check(digest(raw) == expected, "Unexpected content fingerprint: " + path.name)
    return raw


def import_pinned(path, expected):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), module.__dict__)
    return module


def author_scope(a, blocks, original, reserve):
    full = set(blocks)
    gf3 = {i for i, b in blocks.items() if b["sensor"] == "GF-3"}
    if a.cohort == 740:
        label_hash = original.AUTHOR_TEST
        gt = {"info": {}, **json.loads(pin(a.author_labels, label_hash))}
        check(len(gf3) == 583 and len(gt["annotations"]) == 872
              and sum(x["image_id"] in gf3 for x in gt["annotations"]) == 631, "Author test scope differs")
        views = {"gf3": gf3, "full": full}
        groups = {i: b["group"] for i, b in blocks.items()}
        check(len(set(groups.values())) == 112 and len({groups[i] for i in gf3}) == 76,
              "Unexpected author acquisition groups")
    else:
        label_hash = reserve.TRAIN
        gt = reserve.select_labels(pin(a.author_labels, label_hash), blocks)
        census = json.loads(pin(a.code.parent / "results/INSTANCE_RSDD_PARENT_RESERVE_CENSUS_SAFE_20261123.json",
                               reserve.CENSUS))
        dates = {p["paper_scene"]: (p["sensor"], p["calendar_date"]) for p in census["per_parent"]}
        selected = set(census["date_separated_sensitivity"]["parent_scene_ids"])
        views = {"full": full, "date_separated": {i for i, b in blocks.items() if b["group"] in selected}}
        check(len(views["date_separated"]) == 42 and len(gt["annotations"]) == 81, "Author reserve differs")
        groups = {i: dates[b["group"]] for i, b in blocks.items()}
    check(len(gt["images"]) == len(full) == a.cohort and {r["id"] for r in gt["images"]} == full
          and all(r["file_name"] == blocks[r["id"]]["file"] and r["width"] == r["height"] == 512 for r in gt["images"])
          and all(r["category_id"] == 1 and len(r["segmentation"]) == 1 for r in gt["annotations"]),
          "Bound image-to-author annotation join failed")
    gt, rekeyed = original.coco_annotation_ids(gt)
    return gt, groups, views, dict(sha256=label_hash, evaluator_ids_rekeyed=rekeyed)


def contrasts(arms, curves, seeds, ap90, reference):
    rows = {}
    for seed in map(str, seeds):
        learned = f"{seed}/learned"
        summaries = {}
        for kind in REPRS:
            x, y = (arms[arm][kind] for arm in (learned, reference))
            summaries[kind] = {m: 100*(x[m]-y[m]) if x[m] is not None and y[m] is not None else None for m in x}
        boot = {k: curves[learned][k]-curves[reference][k] for k in REPRS}
        rows[seed] = (summaries, boot)
    summary = {seed: dict(metrics_delta_points=m, AP90_paired={k: ap90.interval(x) for k, x in b.items()},
                          representation_interaction_AP90=ap90.interval(b["continuous"]-b["supported_raster"]))
               for seed, (m,b) in rows.items()}
    mean_trace = {kind: np.mean([b[kind] for _,b in rows.values()], axis=0) for kind in REPRS}
    mean_deltas = {}
    for k in REPRS:
        m = [metrics[k] for metrics,_ in rows.values()]
        mean_deltas[k] = {j: float(np.mean([row[j] for row in m])) if all(row[j] is not None for row in m) else None
                         for j in m[0]}
    summary["head_mean"] = dict(metrics_delta_points=mean_deltas,
                               AP90_paired={k: ap90.interval(x) for k,x in mean_trace.items()},
                               representation_interaction_AP90=ap90.interval(mean_trace["continuous"]-mean_trace["supported_raster"]))
    return summary


def view(a, name, ids, gt, groups, blocks, old, seeds, continuous, ap90):
    support = dict(gt, images=[x for x in gt["images"] if x["id"] in ids],
                   annotations=[x for x in gt["annotations"] if x["image_id"] in ids])
    scoped = {i: b for i,b in blocks.items() if i in ids}
    group = {i: groups[i] for i in ids}
    keys, weights = ap90.weights_for(group)
    ctx = continuous.prepare(a.code, support)
    arm_results, curves = {}, {}
    for arm in ("raw", "dynamic", "static", *(f"{s}/learned" for s in seeds)):
        result = continuous.score_arm(ctx, scoped, arm, expected_raster=old[name][arm])
        events = result.pop("events")
        curves[arm] = {k: ap90.score_weighted(e, group, keys, weights) for k,e in events.items()}
        for k, curve in curves[arm].items():
            expected = result[k]["AP90"]
            check(expected is not None and abs(curve[0]-expected) <= 1e-12 and np.isfinite(curve).all(),
                  "Exact unweighted AP90 parity or positive bootstrap support failed")
        result["AP90_transitions"] = ap90.transitions(events["supported_raster"], events["continuous"])
        result["AP90_supported_gt"] = sum(x["positive_gt"] for x in events["supported_raster"].values())
        arm_results[arm] = result
        print(json.dumps(dict(phase="score", view=name, arm=arm, images=len(ids))), flush=True)
    computed = {b: contrasts(arm_results, curves, seeds, ap90, b) for b in ("dynamic", "static", "raw")}
    return dict(images=len(ids), annotations=len(support["annotations"]), blocks=len(keys),
                mapped_candidates=sum(len(x["stock"]) for x in scoped.values()), support=ctx.support,
                original_raster_exact=True, arms=arm_results, learned_minus=computed)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--detector-seed", type=int, choices=(5202,5203), required=True)
    p.add_argument("--cohort", type=int, choices=(74,740), required=True)
    p.add_argument("--protocol-sha256", required=True)
    for k in ("code", "run-root", "author-labels", "protocol", "output"):
        p.add_argument("--"+k, required=True, type=Path)
    a = p.parse_args()
    check((a.detector_seed,a.cohort) in {(5202,740),(5203,740),(5203,74)}
          and a.protocol_sha256 == LOCK and not a.output.exists() and sys.flags.isolated
          and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "Fixed scope and CPU worker")
    pin(a.protocol, LOCK)
    start = perf_counter()
    directory = Path(__file__).resolve().parent
    adapter = import_pinned(directory / "inputs.py", INPUTS)
    continuous = import_pinned(directory / "continuous.py", CONTINUOUS)
    ap90 = import_pinned(directory / "ap90.py", AP90)
    original = import_pinned(a.code / "INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py", OLD)
    reserve = import_pinned(a.code / "INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/score_reserve.py", RESERVE)
    blocks, expected, lineage = adapter.load_inputs(a.code, a.run_root, a.detector_seed, a.cohort)
    gt, groups, views, labels = author_scope(a, blocks, original, reserve)
    args = (gt,groups,blocks,expected,lineage["seeds"],continuous,ap90)
    result = {name:view(a,name,ids,*args) for name, ids in views.items()}
    safe = dict(status=STATUS, phase="score", protocol_sha256=LOCK,
                worker_sha256=digest(Path(__file__).read_bytes()), continuous_sha256=CONTINUOUS, bootstrap_sha256=AP90,
                detector_seed=a.detector_seed, images=a.cohort, label_reference=labels, input_lineage=lineage,
                views=result, optimizer_updates=0, detector_forwards=0, bootstrap_draws=4096, bootstrap_seed=5217,
                elapsed_seconds=perf_counter()-start)
    a.output.mkdir()
    raw = (json.dumps(safe, sort_keys=True, allow_nan=False)+"\n").encode()
    with (a.output / "RESULT_SAFE.json").open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(status=STATUS, result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()
