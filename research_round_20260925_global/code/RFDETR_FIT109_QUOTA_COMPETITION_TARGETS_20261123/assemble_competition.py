"""Exact unilateral COCO-AP action labels from the closed HRSID FIT109 pool."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


STATUS = "RFDETR_FIT109_QUOTA_COMPETITION_TARGETS_COMPLETE"
PROTOCOL = "ecfc612cd941ef4f276e103cd30bca223d291244ddf7d7e72b2178d82239a7ff"
ACTION_RESULT = "bec04364e1df66fe3073b9cbe6e6c3e563ffdfb9d38ee8933cba53a3c6a13111"
ACTION_TERMINAL = "0097515808d3a73aadc23d3de15dee17001e9f2c13cf6dd50218cf8ed7ffe363"
ACTION_SOURCE = "c3834371916148546c2dd3b842839808214798911bf04d7065cdea0221a8f4ae"
FEATURE_MANIFEST = "3193de2d4cbd6404c20a41e109424558b25a2bd6f70486879af063efc421715a"
GEOMETRIC_MANIFEST = "ed22e33e9e0cadef4cd8c8a526dcc85f78aef2c2dabb5d3004eb60f1308c61ab"
ORIGINAL_GT = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
ORIGINAL_CAPTURE = "7f3f6dd112d287623360c8e3a174bae3a12a039c7f5505e4bf7c5d76b2e5ed6f"
COMPETITION = "8ede9d7c65e82f2b5956d2256ba02f7e7c9ff2fccfcb8e41390d4b117ee83da9"
SPARSE_RLE = "cb9b8e839de85588fdf116ce5554ed6cadbc20bdbe76e38180561528bfd870d5"
SPOT = "75ff72a3177e99f06d212a878c8d8921de4db6da8f4249722fb762718083d0ee"
SEEDS = (5217, 5218, 5219)
MIB = 1024**2
AP_TOL = 1e-12


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, digest, limit=MIB):
    with path.open("rb") as stream:
        raw = stream.read(limit+1)
    need(0 < len(raw) <= limit and sha256(raw).hexdigest() == digest, "Exact bounded input: "+path.name)
    return raw


def document(path, digest, limit=MIB):
    return json.loads(pin(path, digest, limit))


def source(path, digest, name):
    loaded = ModuleType(name)
    loaded.__file__ = str(path)
    sys.modules[name] = loaded
    exec(compile(pin(path, digest), str(path), "exec"), loaded.__dict__)
    return loaded


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8")


def write(path, payload):
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(payload).hexdigest()


def originals(a, spot):
    capture = document(a.capture_acceptance, ORIGINAL_CAPTURE)
    terminal = document(a.source_actions/"observed_prepare/TERMINAL.json", ACTION_TERMINAL)
    phase = a.source_actions/"prepare"
    action = document(phase/"RESULT_SAFE.json", ACTION_RESULT)
    need(terminal["phase"] == "prepare-edits" and terminal["actual_exit"] == 0 and terminal["output_footer_verified"]
         and terminal["result_sha256"] == ACTION_RESULT and terminal["worker_sha256"] == action["worker_sha256"] == ACTION_SOURCE
         and action["status"] == "RFDETR_FIT109_QUOTA_SOURCE_ACTIONS_COMPLETE" and action["FIT"]
         and not any(action[k] for k in ("DEV", "TEST", "AP")), "Observed exact closed source action phase")
    need(action["prelabel_feature_manifest_sha256"] == FEATURE_MANIFEST, "Exact admitted source feature manifest")
    receipt = document(phase/"FEATURE_MANIFEST_PRIVATE.json", FEATURE_MANIFEST)
    need(receipt["GT_opened"] is False and len(receipt["groups"]) == len({f["group"] for f in receipt["groups"]}) == 109
         and action["original_stock_rle_sha256"] == receipt["original_stock_rle_sha256"] == capture["stock_predictions_sha256"]
         and action["original_candidate_roster_sha256"] == receipt["original_candidate_roster_sha256"] == capture["queries_sha256"]
         and receipt["original_image_manifest_sha256"] == capture["image_manifest_sha256"]
         and capture["original_os_exit"] == 0 and capture["mode"] == "fit" and capture["GT_opened"] is False,
         "Independent FIT carrier and pre-label manifest agreement")
    stocks = [json.loads(row) for row in pin(a.capture/"STOCK_NATIVE_PREDICTIONS_PRIVATE.jsonl",
                                            capture["stock_predictions_sha256"],128*MIB).splitlines()]
    queries = [json.loads(row) for row in pin(a.capture/"QUERIES_PRIVATE.jsonl",
                                             capture["queries_sha256"],128*MIB).splitlines()]
    image_index = document(a.capture/"IMAGE_MANIFEST_PRIVATE.json", capture["image_manifest_sha256"])
    need(len(stocks) == len(queries) == len(image_index) == len(receipt["groups"]) == 109,
         "Complete closed standard FIT109 streams")
    originals_by_img, packets = {}, {}
    for row, query, meta, spec in zip(stocks, queries, image_index, receipt["groups"], strict=True):
        image_id = row["image_id"]
        pool = spot.native_candidate_rows(query)
        expected = [dict(image_id=image_id, category_id=1, score=c["score"], segmentation=c["rle"]) for c in pool]
        need(image_id == query["image_id"] == meta["image_id"] == spec["image_id"] and image_id not in originals_by_img
             and expected == row["coco"] and [c["rank"] for c in pool] == row["native_ranks"]
             and all(c["rle"]["size"] == [800,800] for c in pool), "Exact original stock RLE, scores, ranks and query identity")
        need(len(spec["file"]) == 7 and spec["file"][:3].isdigit() and spec["file"].endswith(".npz"), "Private shard filename")
        payload = pin(phase/"FEATURE_SHARDS_PRIVATE"/spec["file"], spec["sha256"], 8*MIB)
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in ("identity", "endpoints", "rank_order", "endpoint_count",
                                             "balanced_count", "actions", "action_offsets")}
        ids, exposures, offsets = arrays["identity"], arrays["actions"], arrays["action_offsets"]
        need(ids.shape == (spec["candidate_rows"], 4) and exposures.shape == (spec["actions"], 2)
             and offsets.shape == (len(ids)+1,) and offsets[0] == 0 and offsets[-1] == len(exposures)
             and (np.diff(offsets) > 0).all() and np.all(ids[:,1] == image_id), "Same full original feature shard actions")
        originals_by_img[image_id] = pool
        packets[image_id] = (spec, arrays)
    need(sum(map(len, originals_by_img.values())) == 10488 and
         sum(len(p[1]["identity"]) for p in packets.values()) == 19263 and
         sum(len(p[1]["actions"]) for p in packets.values()) == 313135,
         "All full-pool balanced candidates and every preexisting feasible action")
    return originals_by_img, packets, capture


def action_prefix(points, order, moves):
    add, drop = moves
    return np.r_[points[order[:add, 0], 0], points[order[:drop, 1], 1]]


def original_geometry_roles(a, packets):
    """Only the previously admitted valid-match flag, no geometry target values."""
    phase = a.source_actions/"prepare"
    receipt = document(phase/"TARGET_MANIFEST_PRIVATE.json",GEOMETRIC_MANIFEST)
    need(receipt["feature_manifest_sha256"] == FEATURE_MANIFEST and len(receipt["groups"]) == 109,
         "Use exact archived fixed-geometry exposure classification")
    masks = {}
    for row in receipt["groups"]:
        feature = row["feature_manifest"]
        image_id = feature["image_id"]
        need(packets[image_id][0] == feature and image_id not in masks, "Origin geometry role joins to the exact feature shard")
        raw = pin(phase/"TARGET_SHARDS_PRIVATE"/feature["file"],row["target_sha256"],8*MIB)
        with np.load(BytesIO(raw),allow_pickle=False) as stored:
            mask = stored["supervised_candidate"]
        need(mask.dtype == np.bool_ and mask.shape == (feature["candidate_rows"],), "Defined valid-match role mask")
        masks[image_id] = mask
    need(sum(np.count_nonzero(v) for v in masks.values()) == 1176, "Exact old valid candidate exposure")
    return masks


def balanced_populations(original, packets, sparse):
    worlds = {s:{} for s in SEEDS}
    identities = {}
    for image_id, pool in original.items():
        spec, q = packets[image_id]
        identities[image_id] = {(c["rank"], c["query_id"]): i for i, c in enumerate(pool)}
        need(len(identities[image_id]) == len(pool), "Unique score-ordered source candidate identity")
        for seed in SEEDS:
            worlds[seed][image_id] = [dict(image_id=image_id, category_id=1, score=c["score"],
                                           segmentation=c["rle"]) for c in pool]
        for i, (seed, iid, rank, query) in enumerate(q["identity"]):
            position = identities[image_id][rank, query]
            n = int(q["endpoint_count"][i])
            k = int(q["balanced_count"][i])
            begin, end = q["action_offsets"][i:i+2]
            moves = q["actions"][begin:end]
            need(np.array_equal(moves[:,0],np.arange(max(0,2*k-n),min(2*k,n)+1))
                 and np.all(moves.sum(axis=1) == 2*k), "All feasible counts retain the archived edit budget")
            points, order = q["endpoints"][i,:n], q["rank_order"][i,:n]
            need(len(np.unique(points)) == 2*n and all(np.array_equal(np.sort(order[:,j]),np.arange(n)) for j in (0,1)),
                 "Disjoint signed endpoints and exact frozen probability rankings")
            stock = pool[position]["rle"]
            flip = action_prefix(points, order, (k,k))
            result = sparse.xor_rle(stock, flip)
            area = int(codec.area(stock))
            need(len(flip) == 2*k and 32 <= area < 1024 and int(codec.area(result)) == area and
                 int(codec.area(codec.merge([stock,result],intersect=True))) == area-k,
                 "Balanced reference exact XOR and native area")
            worlds[int(seed)][image_id][position]["segmentation"] = result
    return worlds, identities


def annotations(a, original):
    gt = document(a.fit_gt, ORIGINAL_GT,32*MIB)
    need(len(gt["images"]) == 2937 and len(gt["annotations"]) == 9260
         and [c["id"] for c in gt["categories"]] == [1], "Exact original native HRSID FIT author annotations")
    imgs = [x for x in gt["images"] if x["id"] in original]
    anns = [x for x in gt["annotations"] if x["image_id"] in original]
    need(len(imgs) == 109 and len(anns) == 496 and all(x["width"] == x["height"] == 800 for x in imgs)
         and len({x["id"] for x in anns}) == len(anns)
         and all(type(x["id"]) is int and x["id"] > 0 and x["category_id"] == 1 and 0 <= x["area"] <= 640000 for x in anns),
         "Original 109-image matching population and author area ambiguity checks")
    return dict(info={}, images=imgs, annotations=anns, categories=gt["categories"])


def direct_coco(gt, outputs):
    ordered = [row for image_id in sorted(outputs) for row in outputs[image_id]]
    with redirect_stdout(StringIO()):
        coco = COCO()
        coco.dataset = deepcopy(gt)
        coco.createIndex()
        detection = coco.loadRes(deepcopy(ordered))
        scorer = COCOeval(coco, detection, "segm")
        scorer.params.catIds = [1]
        scorer.params.imgIds = sorted(outputs)
        scorer.params.maxDets = [1, 10, 100]
        scorer.evaluate()
        scorer.accumulate()
    precision = scorer.eval["precision"][:,:,0,0,-1]
    need(np.all(precision >= 0), "A nonempty pooled truth denominator at every IoU threshold")
    official = {d["image_id"]: d for d in scorer.evalImgs
                if d is not None and d["aRng"] == scorer.params.areaRng[0] and d["maxDet"] == 100}
    need(set(official) == set(outputs), "Complete official all-area evaluation")
    return official, scorer, precision.mean(axis=1)


def direct_status(match):
    return np.where(match["dtIgnore"], -1, np.asarray(match["dtMatches"] != 0, dtype=np.int8))


def reference_world(gt, outputs, comp):
    reports, coco, official_ap = direct_coco(gt, outputs)
    images, counts = {}, {}
    for image_id, match in reports.items():
        details = coco._gts[image_id,1]
        expected = np.array([row["iscrowd"] for row in details],dtype=bool)
        shape = (len(outputs[image_id]),len(details))
        ious = np.asarray(coco.ious[image_id,1],dtype=np.float64).reshape(shape)
        image = comp.FrozenImage(ious,expected)
        official_gt_id = np.asarray([0]+[q["id"] for q in details],dtype=np.int64)
        need(np.array_equal(np.sign(image.baseline), direct_status(match)) and
             np.array_equal(official_gt_id[np.abs(image.baseline)],match["dtMatches"]) and
             np.array_equal(np.asarray(match["dtScores"]),np.asarray([q["score"] for q in outputs[image_id]])),
             "Direct COCO reproduces every image truth ID, TP/FP/ignored status and fixed rank score")
        images[image_id] = image
        counts[image_id] = int(np.count_nonzero(~expected))
    ids = sorted(outputs)
    score = np.concatenate([np.asarray(reports[i]["dtScores"],dtype=np.float64) for i in ids])
    order = np.argsort(-score,kind="stable")
    number = np.concatenate([np.full(len(outputs[i]), i,dtype=np.int64) for i in ids])
    positions = {i: np.flatnonzero(number[order] == i) for i in ids}
    direct_ids = np.concatenate([np.asarray(reports[i]["dtIds"],dtype=np.int64) for i in ids])
    need(np.array_equal(direct_ids,np.arange(1,len(direct_ids)+1)), "Original insertion order resolves equal-score ties")
    global_status = np.concatenate([images[i].baseline for i in ids],axis=1)[:,order]
    reconstructed = comp.ap_curve(global_status,sum(counts.values()))
    need(np.allclose(official_ap,reconstructed,atol=AP_TOL,rtol=0), "Reproduce each global COCO 101-point AP value")
    total_matched = sum(np.count_nonzero(image.baseline > 0) for image in images.values())
    return SimpleNamespace(images=images, positions=positions, global_status=global_status,
                           positives=sum(counts.values()), by_threshold=official_ap, exact_ap=reconstructed,
                           total_matches=total_matched)


def integer_intersection(raw_iou, areas, target_areas, crowd):
    a = np.asarray(areas,dtype=np.int64)[:,None]
    g = np.asarray(target_areas,dtype=np.int64)[None,:]
    x = np.where(crowd,raw_iou*a,raw_iou*(a+g)/(1+raw_iou))
    integer = np.rint(x).astype(np.int64)
    rebuilt = categorical_iou(integer,a,g,crowd)
    need(np.all(np.abs(x-integer) <= 1e-6) and np.array_equal(rebuilt,raw_iou),
         "Round-trip native COCO IoUs via exact integer intersections")
    return integer


def categorical_iou(inter, area, gt, crowd):
    denom = np.where(crowd,area,area+gt-inter)
    need(np.all(inter >= 0) and np.all(inter <= np.minimum(area,gt)), "Physical exact mask intersections")
    return np.divide(inter,denom,out=np.zeros(inter.shape,dtype=np.float64),where=inter>0)


def action_overlaps(moves, points, order, baselines, area, gt_area, crowd, sample_gt):
    add, drop = moves.T
    hits_add = sample_gt[points[order[:,0],0]]
    hits_drop = sample_gt[points[order[:,1],1]]
    zero = np.zeros((1,len(gt_area)),dtype=np.int64)
    inc = np.concatenate((zero,np.cumsum(hits_add,axis=0,dtype=np.int64)),axis=0)
    dec = np.concatenate((zero,np.cumsum(hits_drop,axis=0,dtype=np.int64)),axis=0)
    after = baselines[None,:]+inc[add]-dec[drop]
    return categorical_iou(after,(area+add-drop)[:,None],gt_area[None,:],crowd)


def role(status):
    return "native_tp50" if status>0 else "native_ignored50" if status<0 else "native_unmatched50"


def target_groups(gt, originals, worlds, packets, identities, geometric_roles, sparse, comp, output):
    native = {i:[dict(image_id=i,category_id=1,score=c["score"],segmentation=c["rle"]) for c in p]
              for i,p in originals.items()}
    native_reports, native_coco, native_ap = direct_coco(gt,native)
    references = {seed:reference_world(gt, worlds[seed], comp) for seed in SEEDS}
    need(len({r.positives for r in references.values()}) == 1, "Positive truth denominator is fixed for all seeds")
    private = output/"COMPETITION_SHARDS_PRIVATE"
    private.mkdir()
    by_seed, role_stats = {s:Counter() for s in SEEDS}, {}
    geometry_stats = {k:Counter() for k in ("native_fixed_geometry", "without_native_fixed_geometry")}
    meaningful_images = {s:set() for s in SEEDS}
    meaningful_groups = {s:set() for s in SEEDS}
    geometry_groups = {k:set() for k in geometry_stats}
    probes, manifests = {}, []
    for image_id in sorted(packets):
        spec, data = packets[image_id]
        pool = originals[image_id]
        local = identities[image_id]
        out = np.zeros((len(data["actions"]),10),dtype=np.float64)
        displayed = native_reports[image_id]
        native_roles = direct_status(displayed)[0]
        truths = native_coco._gts[image_id,1]
        crowd = np.array([row["iscrowd"] for row in truths],dtype=bool)
        gt_masks = [row["segmentation"] for row in truths]
        gt_area = np.asarray(codec.area(gt_masks),dtype=np.int64) if gt_masks else np.empty(0,dtype=np.int64)
        gt_samples = codec.decode(gt_masks).reshape(800*800,len(truths)) if gt_masks else np.empty((800*800,0),dtype=np.uint8)
        areas = np.array(codec.area([c["rle"] for c in pool]),dtype=np.int64)
        raw_iou = np.asarray(native_coco.ious[image_id,1],dtype=np.float64).reshape((len(pool),len(truths)))
        native_inter = integer_intersection(raw_iou,areas,gt_area,crowd)
        for i,(seed,iid,rank,query) in enumerate(data["identity"]):
            pos = local[rank,query]
            reference = references[int(seed)]
            image = reference.images[image_id]
            start, end = data["action_offsets"][i:i+2]
            moves = data["actions"][start:end]
            n = int(data["endpoint_count"][i])
            k = int(data["balanced_count"][i])
            coords, order = data["endpoints"][i,:n], data["rank_order"][i,:n]
            alternatives = action_overlaps(moves,coords,order,native_inter[pos],areas[pos],gt_area,crowd,gt_samples)
            original_role = role(native_roles[pos])
            balanced_role = role(image.baseline[0,pos]).replace("native_","balanced_",1)
            outcomes = role_stats.setdefault(original_role,Counter())
            condition = "native_fixed_geometry" if geometric_roles[image_id][i] else "without_native_fixed_geometry"
            cohort = geometry_stats[condition]
            outcomes.update(candidates=1, actions=len(moves))
            cohort.update(candidates=1,actions=len(moves))
            stats = by_seed[seed]
            stats.update(candidates=1, actions=len(moves), **{balanced_role+"_candidates":1})
            b, = np.flatnonzero(moves[:,0] == k)
            need(np.array_equal(alternatives[b],image.iou[pos]), "Exact analytical IoU parity with sparse balanced COCO RLE")
            stats["balanced_overlap_rows_verified"] += 1
            original_status = np.sign(image.baseline)
            memo = {}
            alt_result = None
            for j, hypothesis in enumerate(alternatives):
                if np.array_equal(hypothesis,image.iou[pos]):
                    stats["identical_iou_actions"] += 1
                    continue
                successors = np.sign(image.substitute(pos,hypothesis))
                if np.array_equal(successors,original_status):
                    stats["unchanged_outcome_actions"] += 1
                    continue
                outcomes["changed_status_actions"] += 1
                cohort["changed_status_actions"] += 1
                stats["changed_status_actions"] += 1
                key = successors.tobytes()
                if key not in memo:
                    if alt_result is None:
                        alt_result = reference.global_status.copy()
                    alt_result[:,reference.positions[image_id]] = successors
                    memo[key] = comp.ap_curve(alt_result,reference.positives)-reference.exact_ap
                    stats["global_ap_recomputations"] += 1
                delta = memo[key]
                out[start+j] = delta
                mean = float(delta.mean())
                positive,negative = int(mean > AP_TOL),int(mean < -AP_TOL)
                for ledger in (stats,outcomes,cohort):
                    ledger["positive_mean_ap_actions"] += positive
                    ledger["negative_mean_ap_actions"] += negative
                # Tiny inexact cancellations may occur from reference engine averaging order.
                nonzero = bool(np.any(np.abs(delta) > AP_TOL))
                if nonzero:
                    outcomes["nonzero_ap_actions"] += 1
                    cohort["nonzero_ap_actions"] += 1
                    stats["nonzero_ap_actions"] += 1
                    meaningful_images[seed].add(image_id)
                    meaningful_groups[seed].add(spec["group"])
                    geometry_groups[condition].add(spec["group"])
                if seed not in probes or (nonzero and not probes[seed][-1]):
                    probes[seed] = (image_id,pos,action_prefix(coords,order,moves[j]),delta.copy(),nonzero)
        with (private/spec["file"]).open("xb") as stream:
            np.savez_compressed(stream,identity=data["identity"],actions=data["actions"],
                                action_offsets=data["action_offsets"],delta_ap_by_iou=out,delta_ap=out.mean(axis=1))
        digest = sha256((private/spec["file"]).read_bytes()).hexdigest()
        manifests.append(dict(file=spec["file"], source_feature_sha256=spec["sha256"], sha256=digest,
                              group=spec["group"], fold=spec["fold"], image_id=image_id, actions=len(out)))

    direct_deltas = {}
    for seed,(i,pos,flips,wanted,_) in probes.items():
        copy = {key: list(rows) for key,rows in worlds[seed].items()}
        copy[i][pos] = dict(copy[i][pos],segmentation=sparse.xor_rle(originals[i][pos]["rle"],flips))
        _,_,stats = direct_coco(gt,copy)
        calculated = stats-references[seed].by_threshold
        need(np.allclose(wanted,calculated,atol=AP_TOL,rtol=0), "Direct mutated global COCO AP proves one actual action")
        direct_deltas[int(seed)] = dict(max_abs_difference=float(np.max(np.abs(wanted-calculated))),
                                        nonzero_thresholds=int(np.count_nonzero(np.abs(calculated)>AP_TOL)))
    need(sum(r["actions"] for r in by_seed.values()) == sum(x["actions"] for x in manifests) == 313135
         and sum(r["balanced_overlap_rows_verified"] for r in by_seed.values()) == 19263
         and all(r["identical_iou_actions"]+r["unchanged_outcome_actions"]+r["changed_status_actions"]==r["actions"]
                 for r in by_seed.values()), "Every legal action receives an exact scalar and ten AP components")
    for ledger in (*by_seed.values(), *role_stats.values(), *geometry_stats.values()):
        ledger["zero_mean_ap_actions"] = ledger["actions"]-ledger["positive_mean_ap_actions"]-ledger["negative_mean_ap_actions"]
    return (manifests, dict(native_ap=native_ap.tolist(), per_seed={
        int(seed):dict(stats, balanced_ap=references[seed].by_threshold.tolist(), positives=references[seed].positives,
                       images_with_nonzero_ap=len(meaningful_images[seed]), groups_with_nonzero_ap=len(meaningful_groups[seed]),
                       direct_baseline_image_statuses_verified=10488*10)
        for seed,stats in by_seed.items()}, counts_by_original_role=role_stats,
        counts_by_original_geometric_role={k:dict(v,groups_with_nonzero_ap=len(geometry_groups[k]))
                                            for k,v in geometry_stats.items()},
        direct_action_probes=direct_deltas))


def main():
    ap = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    ap.add_argument("--authority",type=Path,required=True)
    ap.add_argument("--authority-sha",required=True)
    ap.add_argument("--output",type=Path,required=True)
    cfg=ap.parse_args()
    paths={"code","protocol","source_actions","capture","capture_acceptance","fit_gt"}
    authority=document(cfg.authority,cfg.authority_sha)
    need(set(authority)==paths|{"protocol_sha"} and all(Path(authority[k]).is_absolute() for k in paths)
         and cfg.output.is_absolute() and not cfg.output.exists() and cfg.output.parent.is_dir()
         and sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1",
         "Fresh isolated host-only source evaluation")
    a=SimpleNamespace(**{k:Path(v).resolve(strict=True) if k in paths else v for k,v in authority.items()})
    need(a.protocol_sha == PROTOCOL, "The verified source competition protocol is required")
    pin(a.protocol,PROTOCOL)
    start=perf_counter()
    comp=source(a.code/"RFDETR_FIT109_QUOTA_COMPETITION_20261123/competition.py",COMPETITION,"_fit_competition")
    sparse=source(a.code/"RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py",SPARSE_RLE,"_fit_sparse")
    spot=source(a.code/"RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/run_spot.py",SPOT,"_fit_native_spot")
    original,packets,capture=originals(a,spot)
    worlds,identities=balanced_populations(original,packets,sparse)
    cfg.output.mkdir()
    world_digests = {seed:sha256(encode(worlds[seed])).hexdigest() for seed in SEEDS}
    before=write(cfg.output/"PRELABEL_SAFE.json",encode(dict(feature_phase_sha256=ACTION_RESULT,
        original_stock_sha256=capture["stock_predictions_sha256"],fixed_legal_actions=313135,
        full_candidates=10488,groups=109,roles_from_labels=False,GT_opened=False,balanced_pool_sha256=world_digests)))
    middle=perf_counter()
    gt=annotations(a,original)
    geometric_roles=original_geometry_roles(a,packets)
    result,details=target_groups(gt,original,worlds,packets,identities,geometric_roles,sparse,comp,cfg.output)
    manifest=write(cfg.output/"COMPETITION_MANIFEST_PRIVATE.json",encode(dict(groups=result,source_prelabel_sha256=before,
                 original_feature_result_sha256=ACTION_RESULT,feature_manifest_sha256=FEATURE_MANIFEST,
                 score_components=comp.IOU_LEVELS.tolist())))
    safe=dict(status=STATUS,**details,worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
              protocol_sha256=a.protocol_sha,authority_sha256=cfg.authority_sha,images=109,annotations=496,
              candidates=10488,legal_actions=313135,changed_area_is_part_of_exact_iou=True,
              prelabel_sha256=before,competition_manifest_sha256=manifest,feature_manifest_sha256=FEATURE_MANIFEST,
              source_feature_result_sha256=ACTION_RESULT,
              prelabel_seconds=middle-start,compilation_seconds=perf_counter()-middle,
              detector_forwards=0,optimizer_updates=0,FIT=True,DEV=False,TEST=False,FIT_GT_opened=True,
              source_only_AP=True)
    stamp=write(cfg.output/"RESULT_SAFE.json",encode(safe))
    print(json.dumps(dict(status=STATUS,result_sha256=stamp)),flush=True)


if __name__=="__main__":
    main()
