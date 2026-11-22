"""Compare two frozen rankings on capped and complete native boundary layers."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np


ORIGINAL_SCORE = "2b3791a794511bb437b9cd47830fb3b01f19bb30cb68107a224acc33e5a75c31"
TRANSFER_SCORER = "45971f735e9e186ce20a38afb319a9a3b7dec815f3778ebe59099e0b5517ad88"
ACCEPTED = "INSTANCE_RSDD_EXPANDED_RANKING_ACCEPTED_V1"
STATUS = "INSTANCE_RSDD_EXPANDED_RANKING_SCORE_COMPLETE_V1"
N, P, BS, LS = "raw", "point_paste", "point_equal", "learned_common"
LSN, BE, LE = "learned_common_native_tie", "point_expanded", "learned_expanded"
OLD = (N, P, BS, LS)
NEW = (LSN, BE, LE)
ARMS = OLD + NEW
FIELDS = ("continuous_iou_points", "fp_per_truth_area_points", "fn_per_truth_area_points")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, expected):
    payload = path.read_bytes()
    require(sha256(payload).hexdigest() == expected, f"Exact bytes required for {path.name}")
    return payload


def contrasts(v):
    """Vectorized over any dimensions retained in each arm."""
    expanded = v[LE] - v[BE]
    small = v[LSN] - v[BS]
    return {
        "L_E minus B_E": expanded,
        "L_SN minus B_S": small,
        "L_S minus B_S": v[LS] - v[BS],
        "L_SN minus L_S": v[LSN] - v[LS],
        "L_E minus L_SN": v[LE] - v[LSN],
        "B_E minus B_S": v[BE] - v[BS],
        "ranking by support interaction": expanded - small,
        "L_E minus P": v[LE] - v[P],
        "B_E minus P": v[BE] - v[P],
        "L_E minus N": v[LE] - v[N],
        "B_E minus N": v[BE] - v[N],
        "P minus N": v[P] - v[N],
    }


def join_rows(blocks, rows, core):
    require(len(rows) == len(blocks) == 740, "The full ordered author roster is required")
    by_sensor = defaultdict(Counter)
    for (image, frame), row in zip(blocks.items(), rows, strict=True):
        require(set(row) == {"image_id", "changes"} and row["image_id"] == image,
                "Original author image order")
        totals = by_sensor[frame["sensor"]]
        totals["images"] += 1
        totals["mapped_candidates"] += len(frame["stock"])
        expected = {str(i) for i, e in frame["changes"].items() if e.get(LS, {}).get("swaps", 0) > 0}
        require(set(row["changes"]) == expected, "Exactly the archived positive-k candidates")
        for key, arms in row["changes"].items():
            rank = int(key)
            legacy = frame["changes"][rank]
            stock = frame["stock"][rank-1]["segmentation"]
            k = legacy[LS]["swaps"]
            require(set(arms) == set(NEW) and 1 <= k <= 8 and legacy[BS]["swaps"] == k,
                    "Three complete archived-budget comparators")
            area = int(core.codec.area(stock))
            for arm, edit in arms.items():
                require(set(edit) == {"swaps", "rle"} and type(edit["swaps"]) is int and edit["swaps"] == k,
                        "Signed budget must match the archived natural writer on every candidate")
                after = core.checked_rle(edit["rle"], 512, 512)
                overlap = int(core.codec.area(core.codec.merge([stock, after], intersect=True)))
                require(int(core.codec.area(after)) == area and area-overlap == k,
                        "Measured additions and removals must both equal k")
                legacy[arm] = edit
            totals["positive_k_candidates"] += 1
            totals["swaps_per_arm"] += k
            for name, a, b in (("native_tie_changed_candidates", LSN, LS),
                               ("learned_expansion_changed_candidates", LE, LSN),
                               ("point_expansion_changed_candidates", BE, BS)):
                totals[name] += legacy[a]["rle"]["counts"] != legacy[b]["rle"]["counts"]
    totals = sum(by_sensor.values(), Counter())
    require(totals["mapped_candidates"] == 73226 and totals["swaps_per_arm"] == 329222
            and totals["positive_k_candidates"] == 42829 and by_sensor["GF-3"]["swaps_per_arm"] == 268347
            and by_sensor["GF-3"]["positive_k_candidates"] == 34808, "Reproduce complete archived budgets")
    return {"full": dict(totals), "by_sensor": {s:dict(c) for s,c in by_sensor.items()}}


def admit(args, prior, old, adapter, core, previous):
    root = args.expanded_replay
    old.terminal(root / "observed_verify-edits/TERMINAL.json", args.expanded_terminal_sha,
                 args.expanded_receipt_sha, "verify-edits")
    receipt = json.loads(pin(root / "verify-edits/RESULT_SAFE.json", args.expanded_receipt_sha))
    capture = json.loads(pin(args.cap_receipt, args.cap_receipt_sha))
    require(receipt["status"] == ACCEPTED and receipt["images"] == 740 and receipt["GT_opened"] is False
            and receipt["protocol_sha256"] == args.protocol_sha and receipt["prelock_commit"] == args.prelock_commit
            and receipt["exact_replayed_policy_choices"] is True and receipt["exact_support_xor_area"] is True
            and receipt["source_replay_receipt_sha256"] == args.new_receipt_sha == previous["independent_readback_sha256"]
            and receipt["candidate_audit_sha256"] == capture["audit_sha256"]
            and receipt["predictions_sha256"] == capture["predictions_sha256"],
            "Independent expanded-support replay on the original capture and PointRend archive")
    blocks, historic = prior.load_blocks(args, old, adapter, core)
    require(historic == previous["realized_edits"], "Reproduce original masks and complete editing census")
    rows = old.records(root / "prepare-edits/EDITS_PRIVATE.jsonl", receipt["edits_sha256"])
    counts = join_rows(blocks, rows, core)
    return blocks, counts, receipt


def geometry(assignments, polygons, blocks, core, shapely):
    groups = defaultdict(list)
    for gid, (image, rank, original) in assignments.items():
        truth = polygons[gid]
        stock = original["segmentation"]
        if not truth.is_valid or not 32 <= core.codec.area(stock) < 1024:
            continue
        frame = blocks[image]
        cache = {}
        values = []
        for arm in ARMS:
            mask = frame["changes"].get(rank, {}).get(arm, {}).get("rle", stock)
            ident = mask["counts"]
            if ident not in cache:
                area = int(core.codec.area(mask))
                iou = core.geometric_iou(mask, truth, shapely)
                overlap = iou * (area+truth.area) / (1+iou)
                cache[ident] = [100*iou, 100*(area-overlap)/truth.area, 100*(truth.area-overlap)/truth.area]
            values.append(cache[ident])
        groups[frame["group"]].append(values)
    points = np.stack([np.mean(groups[g], axis=0) for g in sorted(groups)])
    by_arm = {arm:points[:, i] for i, arm in enumerate(ARMS)}
    sample = np.random.Generator(np.random.PCG64(5217)).integers(len(groups), size=(4096, len(groups)))
    estimates = {}
    for name, diff in contrasts(by_arm).items():
        ci = np.percentile(diff[sample].mean(axis=1), (2.5, 97.5), axis=0).T
        estimates[name] = {field:dict(delta=float(mu), interval_95=limits.tolist())
                           for field, mu, limits in zip(FIELDS, diff.mean(axis=0), ci, strict=True)}
    means = {arm:dict(zip(FIELDS, val.mean(axis=0).tolist(), strict=True)) for arm,val in by_arm.items()}
    return dict(original_anchors=len(assignments), eligible_anchors=sum(map(len, groups.values())),
                groups=len(groups), group_equal_arms=means, paired_contrasts_points=estimates)


def score_view(gt, ids, blocks, reference, evaluator, core, polygons, shapely):
    chosen = dict(gt, images=[r for r in gt["images"] if r["id"] in ids],
                  annotations=[r for r in gt["annotations"] if r["image_id"] in ids])
    subset = {i:frame for i,frame in blocks.items() if i in ids}
    scores = {}
    for arm in ARMS:
        scores[arm], matched = evaluator.score_one(chosen, subset, arm, anchors=arm == N)
        if arm == N:
            assignments = matched
        if arm in OLD:
            historical = reference["metrics_fraction"][arm]
            require(scores[arm].keys() == historical.keys()
                    and all(abs(float(v) - historical[k]) < 1e-12 for k,v in scores[arm].items()),
                    "Reproduce every historical COCO metric in this view")
    metrics = {}
    for field in scores[N]:
        values = {arm:100*scores[arm][field] for arm in ARMS}
        # COCO uses -1 for an unavailable stratum. The ledger represents this as null.
        metrics[field] = {name:float(v) if min(values.values()) >= 0 else None
                          for name,v in contrasts(values).items()}
    geom = geometry(assignments, polygons, subset, core, shapely)
    previous = reference["geometry"]
    require(all(geom[k] == previous[k] for k in ("original_anchors","eligible_anchors","groups"))
            and all(abs(geom["group_equal_arms"][arm][k]-previous["group_equal_arms"][arm][k]) < 1e-10
                    for arm in OLD for k in FIELDS), "Reproduce historical acquisition-weighted cohort and means")
    return dict(images=len(ids), annotations=len(chosen["annotations"]), metrics_fraction=scores,
                metrics_contrasts_points=metrics, geometry=geom)


def run(args, prior, previous):
    import shapely
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Original geometry runtime")
    old = prior.predecessor(args.code)
    evaluator = old.module(args.code/"PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", old.SCORE_SOURCE)
    adapter = old.module(args.code/"UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", old.GEOMETRY_SOURCE)
    core = adapter.load_geometry(args.code/"UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    blocks, counts, receipt = admit(args, prior, old, adapter, core, previous)
    # All new masks, eligibility, exact signed budgets and original streams are accepted above.
    gt = {"info": {}, **json.loads(pin(args.author_test, old.AUTHOR_TEST))}
    require(len(gt["images"]) == 740 and len(gt["annotations"]) == 872
            and set(blocks) == {r["id"] for r in gt["images"]}
            and all(r["width"] == r["height"] == 512 and r["file_name"] == blocks[r["id"]]["file"]
                    for r in gt["images"])
            and all(r["category_id"] == 1 and len(r["segmentation"]) == 1 for r in gt["annotations"]),
            "Original author image identities and polygon population")
    gt, rekeyed = old.coco_annotation_ids(gt)
    polygons = {r["id"]:shapely.Polygon(np.asarray(r["segmentation"][0]).reshape(-1, 2)) for r in gt["annotations"]}
    require(sum(not poly.is_valid for poly in polygons.values()) == 1, "Archived original polygon treatment")
    full = set(blocks)
    gf3 = {i for i,frame in blocks.items() if frame["sensor"] == "GF-3"}
    require(len(gf3) == 583, "Predeclared primary GF-3 view")
    selections = dict(gf3=gf3, full=full, terrasar_x=full-gf3,
                      gf3_deduplicated=old.representatives(blocks, gf3),
                      full_deduplicated=old.representatives(blocks, full))
    views, evaluated = {}, {}
    for name, ids in selections.items():
        identity = frozenset(ids)
        if identity not in evaluated:
            evaluated[identity] = score_view(gt, ids, blocks, previous["views"][name],
                                             evaluator, core, polygons, shapely)
        views[name] = evaluated[identity]
        historical = previous["views"][name]
        require(len(ids) == historical["images"] and all(
            views[name]["metrics_fraction"][arm] == historical["metrics_fraction"][arm] for arm in OLD),
            "Original deduplication population and reference scores")
    primary = "L_E minus B_E"
    ap = views["gf3"]["metrics_contrasts_points"]["AP"][primary]
    lower = views["gf3"]["geometry"]["paired_contrasts_points"][primary][FIELDS[0]]["interval_95"][0]
    return dict(status=STATUS, protocol_sha256=args.protocol_sha, prelock_commit=args.prelock_commit,
                original_score_sha256=ORIGINAL_SCORE, independent_readback_sha256=args.expanded_receipt_sha,
                expanded_replay_terminal_sha256=args.expanded_terminal_sha, edits_sha256=receipt["edits_sha256"],
                native_prediction_sha256=receipt["predictions_sha256"], author_test_sha256=old.AUTHOR_TEST,
                evaluator_annotation_ids_rekeyed=rekeyed, previously_evaluated=True, detector_forwards=0,
                optimizer_updates=0, label_free_realized_counts=counts, views=views,
                within_recipe_expanded_ranking_criterion=bool(ap > 0 and lower > 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    cmd = parser.parse_args()
    values = json.loads(pin(cmd.authority, cmd.authority_sha))
    paths = {"reference_authority", "original_score", "protocol", "expanded_replay"}
    hashes = {"reference_authority_sha", "protocol_sha", "expanded_terminal_sha", "expanded_receipt_sha"}
    require(set(values) == paths | hashes | {"prelock_commit"}
            and all(Path(values[k]).is_absolute() for k in paths)
            and all(re.fullmatch("[a-f0-9]{64}", values[k]) for k in hashes)
            and re.fullmatch("[a-f0-9]{40}", values["prelock_commit"]) and not cmd.output.exists()
            and sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11", "Original fresh CPU scorer")
    new = {k:Path(v) if k in paths else v for k,v in values.items()}
    previous = json.loads(pin(new["original_score"], ORIGINAL_SCORE))
    require(previous["authority_sha256"] == new["reference_authority_sha"], "Exact archived score inputs")
    reference = json.loads(pin(new["reference_authority"], new["reference_authority_sha"]))
    legacy = {k:v if k.endswith("_sha") else Path(v) for k,v in reference.items()}
    pin(new["protocol"], new["protocol_sha"])
    # Preserve the old authority's protocol path for all historical input checks.
    del new["protocol"]
    args = SimpleNamespace(**legacy, **new)
    source = args.code/"INSTANCE_RSDD_POINTREND_TRANSFER_20261122/score_transfer.py"
    prior = ModuleType("expanded_support_prior_score")
    prior.__file__ = str(source)
    exec(compile(pin(source, TRANSFER_SCORER), str(source), "exec"), prior.__dict__)
    pin(args.protocol, prior.PROTOCOL)
    sys.path.append(str(args.shapely_site))
    started = perf_counter()
    result = run(args, prior, previous)
    result.update(authority_sha256=cmd.authority_sha, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=perf_counter()-started)
    payload = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    cmd.output.mkdir()
    (cmd.output/"RESULT_SAFE.json").write_bytes(payload)
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(payload).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
