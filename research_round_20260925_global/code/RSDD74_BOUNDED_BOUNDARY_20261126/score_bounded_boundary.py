"""Read only the closed reserve writers and score original native Boundary IoU."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask


STATUS = "RSDD74_BOUNDED_PAIRING_NATIVE_BOUNDARY_COMPLETE"
PROTOCOL_SHA = "c88160e734b8c9aebaa1f5387e78cc6bbd765907827528ba10a5b6f518a8835d"
PRELOCK_COMMIT = "eabb2cfc9b54fb50638d2a5f3ef8acd12118be5c"
BOUNDARY_DIR = "INSTANCE_RSDD_NATIVE_BOUNDARY_IOU_20261126"
METRIC_SHA = "b477c918a1168eb6b734313b18c4df8e338a591305c164a444eb61a560e873c5"
STATS_SHA = "7e0e8e29018f6a9c7e19b03f17976a5de78ef09bfc35edfc88e1f9e914cb141f"
BOUNDED = "LEARNED_BOUNDED_PAIRING_20261126/rsdd"
BOUNDED_PROTOCOL = "f6b235aa2d5abb76e6948f04e6ed78ddc2edcb72fced3d11691a8e1e62e8b1dc"
BOUNDED_SCORE_SHA = "bb751d3d42875b4cc9e980534697430283428b9ecfbb1c3c9bc09d26b63028e5"
BOUNDED_SCORE_RESULT = "648f9509bf889d11da57a5680e14dee95438f0318a32a93eae0085cab6a1c97c"
BOUNDED_SCORE_TERMINAL = "844d94c6532599bd97aa8c513683be11ad40bf2ac576210895fad18e7dd34fef"
BOUNDED_RENDER_SHA = "34d4e588dc757dd60c975ebe9498d479a80265bf568422d57b3fa0fa1f6975f5"
BOUNDED_RENDER_RESULT = "4a015eef90344a4d06bd77fedb5fd5a1ea9d5742811f7de01a96dcb6a24ac994"
BOUNDED_RENDER_TERMINAL = "92fea4ab053d319397b823bafe3cab7991c793c55a17802df0bd6bb64b11bcee"
BOUNDED_STREAM = "e97a774e67ae5d4485947c45e8cc08f8715db0dbeca8c594063cf95a426a2919"
OLD_BOUNDARY_RESULT = "a8cae7278bdd3d5485fc6e539632be2e80e0d27615a3e127803fd34477d14c3a"
OLD_BOUNDARY_TERMINAL = "67ef5f0f99c840c36ce00d378d1dbc776ad83dcc48c12f356edef659665e3499"
OLD_BOUNDARY_WORKER = "d9b5024dcf18299d7d5fab997134964fcae8c1e46193cd061833ce1c3de74c7a"
NATIVE_MATCH_SOURCE = "SRNET_RSDD74_THRESHOLD_ORACLE_20261126/oracle.py"
NATIVE_MATCH_SHA = "2273d873a639ee53487269bdf2e110fd029fb03e1a4f76dc9a6f240ad0e8a4a1"
TOL = 1e-12
ARMS = ("F", "L", "I")
CANDIDATE_ARMS = {"F":"5217/fixed", "L":"5217/bounded", "I":"5217/free_matched_count"}
CONTRASTS = {"I_minus_L":(("I","L"),), "L_minus_F":(("L","F"),), "I_minus_F":(("I","F"),)}
VIEWS = {"full":(74,81,47,13,12,11), "date_separated":(42,47,32,8,8,8)}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Exact lineage: " + path.name)
    return raw


def load(path, expected):
    out = ModuleType(path.stem)
    out.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), out.__dict__)
    return out


def compare(value, reference, name):
    if isinstance(reference, dict):
        require(set(value) == set(reference), "Original statistics fields: " + name)
        for key in reference:
            compare(value[key], reference[key], name + "." + key)
    elif isinstance(reference, list):
        require(len(value) == len(reference), "Original interval shape: " + name)
        for this, old in zip(value, reference, strict=True):
            compare(this, old, name)
    elif isinstance(reference, float):
        require(abs(value - reference) <= TOL, "Original floating statistic: " + name)
    else:
        require(value == reference, "Original denominator/setting: " + name)


def verify_boundary_parity(current, historical):
    mapping = {"native":"native", "F":"SF", "I":"SU_equal"}
    for key in ("anchors", "acquisition_groups", "bootstrap_seed", "bootstrap_draws"):
        compare(current[key], historical[key], key)
    for key, old_name in mapping.items():
        compare(current["empty_masks"][key], historical["empty_masks"][old_name], "empty " + key)
    for width in ("1", "2"):
        view, prior = current["widths"][width], historical["widths"][width]
        for key, old_name in mapping.items():
            compare(view["mean_boundary_iou_points"][key], prior["mean_boundary_iou_points"][old_name],
                    "absolute points " + width + " " + key)
            for count in ("arms_whole_mask", "both_reference_and_arm_whole_mask"):
                compare(view["saturation"][count][key], prior["saturation"][count][old_name],
                        "saturation " + width + " " + key)
        compare(view["saturation"]["reference_whole_mask"], prior["saturation"]["reference_whole_mask"],
                "reference saturation " + width)
        branches = (None, "native_reference_non_saturated", "all_author_zero_delta")
        for subset in branches:
            now, previous = (view, prior) if subset is None else (view[subset], prior[subset])
            if subset:
                for count in ("anchors", "acquisition_groups"):
                    compare(now[count], previous[count], subset + " " + count)
            compare(now["contrasts"].get("I_minus_F", {}),
                    previous["contrasts"].get("SU_equal_minus_SF", {}),
                    ("primary" if subset is None else subset) + " " + width)


def geometry(title, gt, blocks, anchors, dates, audit, shapely, archive, bounded):
    annotations = {x["id"]: x for x in gt["annotations"]}
    author_groups = Counter(blocks[x["image_id"]]["group"] for x in annotations.values())
    groups = set()
    changed = defaultdict(lambda:[0,set()])
    counts = Counter(invalid_polygons=0, area_excluded=0, eligible=0, positive_budget=0)
    for key, (image, rank, native) in anchors.items():
        ann = annotations[key]
        stock = native["segmentation"]
        require(ann["image_id"] == image, "Exact unrematched author image")
        poly = shapely.Polygon(np.asarray(ann["segmentation"][0]).reshape(-1, 2))
        if not poly.is_valid or poly.area <= 0:
            counts["invalid_polygons"] += 1
            continue
        if not 32 <= int(mask.area(stock)) < 1024:
            counts["area_excluded"] += 1
            continue
        source = blocks[image]["changes"].get(rank, {})
        options = {name: source.get(key, {}) for name, key in CANDIDATE_ARMS.items()}
        k = options["F"].get("swaps", 0)
        require(k == options["L"].get("swaps", 0) == options["I"].get("swaps", 0),
                "Identical signed realized count on eligible fixed anchor")
        counts.update(eligible=1, positive_budget=int(k > 0))
        group = blocks[image]["group"]
        groups.add(group)
        truth = mask.merge(mask.frPyObjects(ann["segmentation"], 512, 512))
        rles = {name: opt.get("rle", stock) for name, opt in options.items()}
        audit.add_rles(group, truth, stock, rles)
        for name, ((new, old),) in CONTRASTS.items():
            if k and rles[new] != rles[old]:
                changed[name][0] += 1
                changed[name][1].add(group)
    expected = VIEWS[title]
    observed = (len(blocks),len(anchors),counts["eligible"],len(author_groups),len(groups),len({dates[g] for g in groups}))
    require(observed == expected and len(annotations) == len(anchors)
            and counts["invalid_polygons"] == 0 and counts["area_excluded"] == len(anchors) - counts["eligible"]
            and counts["positive_budget"] == archive["exposure"]["positive_SF"],
            "Original four-arm view roster and eligibility")
    contrasts = {}
    for name in CONTRASTS:
        num, parents = changed[name]
        nblocks = len({dates[g] for g in parents})
        reference = bounded["pooled_contrasts"][name]
        compare((num,len(parents),nblocks),
                (reference["changed_hulls"],reference["changed_parent_groups"],reference["changed_analysis_blocks"]),
                name + " original geometry activation")
        contrasts[name] = dict(anchors=num, parents=len(parents), sensor_date_blocks=nblocks)
    summary = audit.report(author_groups=author_groups, blocks={g:dates[g] for g in author_groups})
    verify_boundary_parity(summary, archive["boundary"])
    return dict(images=len(blocks),author_hulls=len(annotations),eligible_matches=counts["eligible"],
                source_parent_groups=len(author_groups),eligible_parent_groups=len(groups),
                eligible_sensor_date_blocks=observed[-1],exposure=dict(counts),changed_valid_positive=contrasts,
                boundary=summary,original_boundary_i_minus_f_parity=True,
                archived_pooled_ap_context_only={name:bounded["pooled_contrasts"][name]["ap_points"]
                                                 for name in CONTRASTS})


def run(c):
    start = perf_counter()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11"
            and c.prelock_commit == PRELOCK_COMMIT and c.protocol_sha == PROTOCOL_SHA and not c.output.exists(),
            "Fresh qualified native CPU audit")
    pin(c.protocol, c.protocol_sha)
    metric = load(c.code / BOUNDARY_DIR / "boundary_metric.py", METRIC_SHA)
    stats = load(c.code / BOUNDARY_DIR / "boundary_stats.py", STATS_SHA)
    anchor_source = load(c.code / NATIVE_MATCH_SOURCE, NATIVE_MATCH_SHA)
    prior = load(c.code / BOUNDED / "score.py", BOUNDED_SCORE_SHA)
    score = prior.closed(c.bounded / "reserve74_score/RESULT_SAFE.json",
                         c.bounded / "observed_reserve74_score/TERMINAL.json",
                         BOUNDED_SCORE_RESULT, BOUNDED_SCORE_TERMINAL, BOUNDED_SCORE_SHA, "score", prior.STATUS)
    boundary = prior.closed(c.boundary / "actual_retry/RESULT_SAFE.json",
                            c.boundary / "observed_score_retry/TERMINAL.json",
                            OLD_BOUNDARY_RESULT, OLD_BOUNDARY_TERMINAL, OLD_BOUNDARY_WORKER, "score",
                            "INSTANCE_RSDD_RESERVE74_BOUNDARY_IOU_COMPLETE")
    require(score["view"] == "reserve74" and score["protocol_sha256"] == BOUNDED_PROTOCOL
            and score["render_result_sha256"]["5217"] == BOUNDED_RENDER_RESULT
            and score["render_worker_sha256"] == BOUNDED_RENDER_SHA
            and score["detector_forwards"] == score["optimizer_updates"] == 0
            and boundary["metric_sha256"] == METRIC_SHA and boundary["stats_sha256"] == STATS_SHA
            and boundary["frozen_candidate_count"] == 7346, "Exact evaluated older boundary and continuous histories")
    scope = SimpleNamespace(code=c.code, archive=c.archive, author_labels=c.author_train,
                            protocol_sha=BOUNDED_PROTOCOL, render_worker_sha=BOUNDED_RENDER_SHA, view="reserve74")
    archive = c.bounded
    record = dict(result=str(archive / "reserve74_5217_render/RESULT_SAFE.json"),
                  terminal=str(archive / "observed_reserve74_5217_render/TERMINAL.json"),
                  result_sha=BOUNDED_RENDER_RESULT,terminal_sha=BOUNDED_RENDER_TERMINAL)
    stream, render = prior.verify_render(scope, 5217, record)
    original = load(c.code / "INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py", prior.ORIGINAL_SCORE_SHA)
    evaluator = original.module(c.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", original.SCORE_SOURCE)
    adapter = original.module(c.code / "UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", original.GEOMETRY_SOURCE)
    core = adapter.load_geometry(c.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    blocks, historic, reserve, dates, nested = prior.historical_reserve(scope, original, core, adapter)
    require(render["actual_bounded_stream_sha256"] == BOUNDED_STREAM
            and render["prior"]["prior_verified_result_sha256"] == historic["receipt_sha256"]["replay"]
            and render["counts"]["positive_candidates"] == 4092 and render["counts"]["swaps"] == 30841
            and score["author_reference_sha256"] == boundary["author_train_sha256"] == reserve.TRAIN
            and boundary["author_census_sha256"] == reserve.CENSUS,
            "Original accepted signed budgets and reference identities")
    prior.verify_and_extend(blocks, {5217:(stream,render)}, core, scope.view)
    print(json.dumps({"stage":"BOUNDARY_INPUTS_AND_ALLOCATION_VERIFIED_BEFORE_LABELS"}), flush=True)

    sys.path.append(str(c.shapely_site))
    import shapely

    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Exact original polygon backend")
    gt, rekeyed = original.coco_annotation_ids(reserve.select_labels(pin(c.author_train, reserve.TRAIN), blocks))
    require(rekeyed == historic["evaluator_annotation_ids_rekeyed"], "Original canonical author IDs")
    anchors = anchor_source.native_anchors(evaluator, gt, blocks)
    views = {}
    for title, ids in (("full",set(blocks)),("date_separated",nested)):
        selected = {i:b for i,b in blocks.items() if i in ids}
        auth = dict(gt,images=[im for im in gt["images"] if im["id"] in ids],
                    annotations=[ann for ann in gt["annotations"] if ann["image_id"] in ids])
        fixed_matches = {key:value for key,value in anchors.items() if value[0] in ids}
        if title == "date_separated":
            require(fixed_matches == anchor_source.native_anchors(evaluator, auth, selected),
                    "Nested stock matches identical to original matcher with nested images only")
        views[title] = geometry(title, auth, selected, fixed_matches, dates, stats.Audit(metric, ARMS, CONTRASTS),
                                shapely, boundary["views"][title], score["views"][title])
    main_width = views["full"]["boundary"]["widths"]["1"]
    entire = main_width["contrasts"]["I_minus_L"]
    subset = main_width["native_reference_non_saturated"]
    non_sat = subset["contrasts"].get("I_minus_L")
    enough = subset["anchors"] >= 20 and subset["acquisition_groups"] >= 6
    supported = bool(enough and non_sat and all(row["paired_95_interval_points"][0] > 0 for row in (entire, non_sat)))
    adverse = bool(enough and non_sat and all(row["paired_95_interval_points"][1] < 0 for row in (entire, non_sat)))
    robust = bool((supported and all(row["block_95_interval_points"][0] > 0 for row in (entire, non_sat)))
                  or (adverse and all(row["block_95_interval_points"][1] < 0 for row in (entire, non_sat))))
    decision = dict(parent_width_one="positive" if supported else "negative" if adverse else "inconclusive",
                    nonsaturated_coverage_met=enough, matching_block_sign_excludes_zero=robust)
    result = dict(status=STATUS,phase="score",worker_sha256=digest(Path(__file__).read_bytes()),
                  protocol_sha256=c.protocol_sha,prelock_commit=c.prelock_commit,
                  boundary_metric_sha256=METRIC_SHA,boundary_stats_sha256=STATS_SHA,
                  original_i_f_boundary_result_sha256=OLD_BOUNDARY_RESULT,bounded_score_result_sha256=BOUNDED_SCORE_RESULT,
                  bounded_render_result_sha256=BOUNDED_RENDER_RESULT,bounded_stream_sha256=BOUNDED_STREAM,
                  author_train_sha256=reserve.TRAIN,original_candidate_count=7346,original_positive_candidates=4092,
                  exact_swaps_per_side_per_arm=30841,scored_arms=["N","F","L","I"],
                  original_anchor_threshold=.50,original_seed=5217,bootstrap_draws=4096,
                  pure_geometry_only=True,retrospective_author_train=True,detector_forwards=0,optimizer_updates=0,
                  AP_computed=False,author_test_labels_opened=False,primary_decision=decision,views=views,elapsed_seconds=perf_counter()-start)
    data = (json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    c.output.mkdir()
    (c.output/"RESULT_SAFE.json").write_bytes(data)
    print(json.dumps(dict(status=STATUS,result_sha256=digest(data))),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("code","protocol","bounded","boundary","archive","author-train","shapely-site","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--protocol-sha",required=True)
    p.add_argument("--prelock-commit",required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()

