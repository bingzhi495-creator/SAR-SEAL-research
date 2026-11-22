"""Retrospective PointRend support attribution with independently replayed masks."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np


LOCK = "dab26421fd6647d5f329efa2739a185e0373f115d3a2cb387a24ecf5e09ca7f6"
COMMIT = "b5c45220977e5b5b85e7ac5e107deaae53b020ac"
ORIGINAL_SCORE = "2b3791a794511bb437b9cd47830fb3b01f19bb30cb68107a224acc33e5a75c31"
TRANSFER_SCORER = "45971f735e9e186ce20a38afb319a9a3b7dec815f3778ebe59099e0b5517ad88"
READBACK = "INSTANCE_RSDD_POINTREND_SUPPORT_FACTOR_ACCEPTED_V1"
STATUS = "INSTANCE_RSDD_POINTREND_SUPPORT_FACTOR_SCORED_V1"

ARMS = ("raw", "point_equal", "p_on_support", "p_off_support", "point_paste")
HYBRIDS = ARMS[2:4]


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pinned(path, expected):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == expected, f"Pinned input {path.name}")
    return data


def load(path, expected):
    pinned(path, expected)
    spec = spec_from_file_location(path.stem, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def four_way(values):
    b, on, off, p = (values[..., i] for i in range(1, 5))
    on_first, on_second = on-b, p-off
    off_first, off_second = off-b, p-on
    return dict(P_minus_B=p-b, inside_first=on_first, inside_second=on_second,
                outside_first=off_first, outside_second=off_second,
                interaction=p-on-off+b, inside_order_average=(on_first+on_second)/2,
                outside_order_average=(off_first+off_second)/2)


def seen_inputs(a, reference, old):
    old.terminal(a.factor_terminal, a.factor_terminal_sha, a.factor_receipt_sha, "verify-edits")
    new = json.loads(pinned(a.factor_receipt, a.factor_receipt_sha))
    require(new["status"] == READBACK and new["protocol_sha256"] == LOCK and new["prelock_commit"] == COMMIT
            and new["images"] == 740 and new["GT_opened"] is False
            and new["exact_support_xor_area"] is True and new["exact_replayed_policy_choices"] is True,
            "Exact independent support replay required before scoring")
    require(new["source_replay_receipt_sha256"] == a.new_receipt_sha
            and new["source_replay_terminal_sha256"] == a.new_terminal_sha
            and new["source_replay_receipt_sha256"] == reference["independent_readback_sha256"],
            "Same original PointRend replay as accepted in the comparison")
    cap = json.loads(pinned(a.cap_receipt, a.cap_receipt_sha))
    original = json.loads(pinned(a.new_receipt, a.new_receipt_sha))
    require(new["predictions_sha256"] == cap["predictions_sha256"]
            and new["candidate_audit_sha256"] == cap["audit_sha256"]
            and new["source_edits_sha256"] == original["edits_sha256"],
            "Same candidate and PointRend mask streams")
    return new, old.records(a.factor_edits, new["edits_sha256"])


def join_hybrids(blocks, rows, core):
    require(len(rows) == len(blocks) == 740, "Same complete candidate image roster for both masks")
    observed = Counter()
    for (image_id, block), row in zip(blocks.items(), rows, strict=True):
        require(image_id == row["image_id"], "Same author order in support replay")
        for k, updates in row["changes"].items():
            rank = int(k)
            require(str(rank) == k and rank in block["ranks"] and set(updates) <= set(HYBRIDS),
                    "Existing compact candidate identity and known hybrid keys")
            stock = block["stock"][rank-1]["segmentation"]
            for arm, edit in updates.items():
                require(set(edit) == {"rle"} and edit["rle"] != stock, "Sparse hybrid contains changed mask only")
                core.checked_rle(edit["rle"], 512, 512)
                block["changes"].setdefault(rank, {})[arm] = edit
                observed[arm] += 1
    return dict(observed)


def coco_rows(blocks, arm):
    rows, ids = [], []
    for image_id, frame in blocks.items():
        for rank, original in zip(frame["ranks"], frame["stock"], strict=True):
            edit = frame["changes"].get(rank, {}).get(arm)
            rows.append(original if edit is None else dict(original, segmentation=edit["rle"]))
            ids.append((image_id, rank, original))
    return rows, ids


def anchor_rows(ev, ids):
    associations = []
    for row in ev.evalImgs:
        if row is None or row["maxDet"] != 100 or row["aRng"] != ev.params.areaRng[0]:
            continue
        for g, d, ignore in zip(row["gtIds"], row["gtMatches"][0], row["gtIgnore"], strict=True):
            if d and not ignore:
                associations.append((g, ids[int(d)-1]))
    require(len({g for g, _ in associations}) == len(associations), "Native fixed-match IDs are unique")
    return associations


def curves(ev):
    precision = ev.eval["precision"][:, :, 0, 0, -1]
    recall = ev.eval["recall"][:, 0, 0, -1]
    reached = ev.params.recThrs[None, :] <= recall[:, None]
    require(precision.shape == (10, 101) and (precision >= 0).all()
            and np.all(precision[~reached] == 0), "Original COCO interpolation and reach grid")
    reach = reached.mean(axis=1)
    quality = (precision*reached).sum(axis=1) / reached.sum(axis=1)
    ap = precision.mean(axis=1)
    require(np.allclose(quality*reach, ap, rtol=0, atol=1e-12), "Exact precision-reach identity")
    return np.stack((ap, recall, reach, quality), axis=1)


def overlap_geometry(core, shape, associations, polygons, blocks, symmetric):
    by_group = defaultdict(list)
    fields = ("continuous_iou", "fp_over_truth", "fn_over_truth")
    for gid, (image_id, rank, original) in associations:
        truth = polygons[gid]
        stock = original["segmentation"]
        if not truth.is_valid or not 32 <= core.codec.area(stock) < 1024:
            continue
        frame = blocks[image_id]
        changes = frame["changes"].get(rank, {})
        raster = [changes.get(arm, {}).get("rle",stock) for arm in ARMS]
        values, intersections, areas = [], [], []
        for rle in raster:
            area = int(core.codec.area(rle))
            iou = core.geometric_iou(rle, truth, shape)
            inter = iou*(area+truth.area)/(1+iou)
            values.append((100*iou, 100*(area-inter)/truth.area, 100*(truth.area-inter)/truth.area))
            intersections.append(inter)
            areas.append(area)
        gained, size = symmetric(intersections[4], areas[4], intersections[1], areas[1], truth.area)
        require(abs(100*(gained+size)-(values[4][0]-values[1][0])) < 1e-10, "Exact realized decomposition")
        by_group[frame["group"]].append((values, [100*gained, 100*size]))
    groups = [by_group[k] for k in sorted(by_group)]
    point = np.stack([np.mean([v for v, _ in g], axis=0) for g in groups], axis=1)
    parts = np.stack([np.mean([v for _, v in g], axis=0) for g in groups])
    draws = np.random.Generator(np.random.PCG64(5217)).integers(len(groups), size=(4096,len(groups)))
    changes = four_way(point.transpose(1, 2, 0))
    changes.update(overlap_component=parts[:, :1], area_component=parts[:, 1:])
    stats = {}
    for k, effect in changes.items():
        names = fields if effect.shape[1] == len(fields) else (fields[0],)
        limits = np.percentile(effect[draws].mean(axis=1), (2.5, 97.5), axis=0).T
        stats[k] = {field:dict(delta=float(mu), interval_95=ci.tolist())
                    for field, mu, ci in zip(names, effect.mean(axis=0), limits, strict=True)}
    return dict(anchors=sum(map(len, groups)), groups=len(groups), contrasts_points=stats,
                means_points={arm:dict(zip(fields, means.tolist(), strict=True))
                              for arm, means in zip(ARMS, point.mean(axis=1), strict=True)})


def score_view(gt, keep, blocks, reference, geometry, metric_keys, modules):
    diag, gap, core, shapely = modules
    chosen = dict(gt, images=[r for r in gt["images"] if r["id"] in keep],
                  annotations=[r for r in gt["annotations"] if r["image_id"] in keep])
    subset = {k:v for k,v in blocks.items() if k in keep}
    scores, arrays, rank_data = {}, [], {}
    for arm in ARMS:
        rows, ids = coco_rows(subset, arm)
        ev = diag.evaluate(chosen, rows)
        arr = curves(ev)
        arrays.append(arr)
        scores[arm] = dict(zip(metric_keys, map(float,ev.stats), strict=True), AP90=float(arr[8,0]))
        if arm in ("raw","point_equal","point_paste"):
            old = reference["metrics_fraction"][arm]
            require(scores[arm].keys() == old.keys()
                    and all(abs(value-old[m]) < 1e-12 for m,value in scores[arm].items()),
                    "Official score of every archived arm exactly reproduced")
        if arm == "raw":
            association = anchor_rows(ev, ids)
        if arm in ("point_equal","point_paste"):
            rank_data[arm] = diag.ranking(ev)
    db, sb, mb, ib = rank_data["point_equal"]
    dp, sp, mp, ip = rank_data["point_paste"]
    require(np.array_equal(db,dp) and np.array_equal(sb,sp), "Exact frozen ranked COCO candidate sequence")
    b, p = arrays[1], arrays[4]
    reach = 50*(p[:,2]-b[:,2])*(p[:,3]+b[:,3])
    precision = 50*(p[:,3]-b[:,3])*(p[:,2]+b[:,2])
    require(np.allclose(reach+precision,100*(p[:,0]-b[:,0]),rtol=0,atol=1e-10), "AP attribution adds exactly")
    threshold_rows = []
    for j in range(10):
        valid = ~(ib[j]|ip[j])
        threshold_rows.append(dict(threshold=float(ev.params.iouThrs[j]),
          ap_points={arm:float(v[j,0]*100) for arm,v in zip(ARMS,arrays,strict=True)},
          recall_points={arm:float(v[j,1]*100) for arm,v in zip(ARMS,arrays,strict=True)},
          reach_component_points=float(reach[j]), interpolation_component_points=float(precision[j]),
          P_gained=int(np.count_nonzero(valid & (mp[j]>0) & (mb[j]==0))),
          P_lost=int(np.count_nonzero(valid & (mb[j]>0) & (mp[j]==0))),
          gt_identity_changes=int(np.count_nonzero(valid & (mp[j]>0)&(mb[j]>0)&(mp[j]!=mb[j])))))
    effect = {}
    for field in ("AP","AP50","AP75","AP90","AR100"):
        effect[field] = {k:float(v*100) for k,v in four_way(np.array([scores[a][field] for a in ARMS])).items()}
    geo = overlap_geometry(core, shapely, association, geometry, subset, gap.decomposition)
    return dict(images=len(keep), annotations=len(chosen["annotations"]), metrics_fraction=scores,
                metrics_contrasts_points=effect, geometry=geo, threshold_diagnostics=threshold_rows,
                mean_reach_points=float(reach.mean()), mean_interpolation_points=float(precision.mean()))


def run(a, prior):
    import shapely
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Original GEOS stack")
    old = prior.predecessor(a.code)
    accepted_score = json.loads(pinned(a.original_score, ORIGINAL_SCORE))
    new, rows = seen_inputs(a, accepted_score, old)
    adapter = old.module(a.code/"UEIKAP_RFDETR_CARRIER_20261122/score_rf.py",old.GEOMETRY_SOURCE)
    core = adapter.load_geometry(a.code/"UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    blocks, _ = prior.load_blocks(a, old, adapter, core)
    observed = join_hybrids(blocks, rows, core)
    # This is the first access to label geometry in this follow-up process.
    gt = {"info": {}, **json.loads(pinned(a.author_test,old.AUTHOR_TEST))}
    require(len(gt["annotations"]) == 872 and len(gt["images"]) == 740
            and {r["category_id"] for r in gt["annotations"]} == {1}
            and set(blocks) == {r["id"] for r in gt["images"]}
            and all(r["file_name"] == blocks[r["id"]]["file"] and [r["height"],r["width"]] == [512,512]
                    for r in gt["images"]), "Exact admitted author annotations and image mapping")
    gt, rekeyed = old.coco_annotation_ids(gt)
    polys = {r["id"]:shapely.Polygon(np.asarray(r["segmentation"][0]).reshape(-1,2))
             for r in gt["annotations"]}
    require(sum(not p.is_valid for p in polys.values()) == 1,"Original non-simple polygon kept in COCO")
    base = a.code/"FIT109_MATCHED_IOU_WRITER_20261121"
    diag = load(base/"diagnose_bj.py","93f57ab95493352a0566e88626e0e5b1aead4a34fde4c8eb86d5ace7b0699de4")
    gap = load(base/"analyze_pointrend_gap.py","c5abb2ed1fb262bbc4fb64acaed9fabdd6d685570c65dab01ce87c8250019307")
    evaluator = old.module(a.code/"PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py",old.SCORE_SOURCE)
    full, gf3 = set(blocks), {i for i,r in blocks.items() if r["sensor"] == "GF-3"}
    views = {name:score_view(gt, ids, blocks, accepted_score["views"][name], polys, evaluator.METRICS,
                              (diag, gap, core, shapely))
             for name,ids in (("gf3",gf3),("full",full),("terrasar_x",full-gf3))}
    return dict(status=STATUS, protocol_sha256=LOCK, prelock_commit=COMMIT,
                native_prediction_sha256=new["predictions_sha256"], previously_evaluated=True,
                hybrid_candidates_changed=observed, new_detector_forwards=0, optimizer_updates=0,
                author_ids_rekeyed=rekeyed, original_score_sha256=ORIGINAL_SCORE,
                supported_replay_sha256=a.factor_receipt_sha, views=views)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = json.loads(pinned(args.authority,args.authority_sha))
    pathkeys = {"reference_authority","original_score","factor_protocol","factor_terminal","factor_receipt","factor_edits"}
    hashkeys = {"reference_authority_sha","factor_terminal_sha","factor_receipt_sha"}
    require(set(values) == pathkeys|hashkeys and not args.output.exists()
            and sys.flags.isolated and sys.dont_write_bytecode
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and np.__version__ == "1.26.4"
            and version("pycocotools") == "2.0.11", "Fresh original CPU runtime")
    inputs = {k:Path(v) if k in pathkeys else v for k,v in values.items()}
    reference = json.loads(pinned(inputs["reference_authority"],inputs["reference_authority_sha"]))
    original = json.loads(pinned(inputs["original_score"],ORIGINAL_SCORE))
    require(original["authority_sha256"] == inputs["reference_authority_sha"], "Use already-observed prior score inputs")
    reference = {k:Path(v) if not k.endswith("_sha") else v for k,v in reference.items()}
    a = SimpleNamespace(**reference, **inputs)
    pinned(a.factor_protocol,LOCK)
    sys.path.append(str(a.shapely_site))
    started = perf_counter()
    previous = load(a.code/"INSTANCE_RSDD_POINTREND_TRANSFER_20261122/score_transfer.py",TRANSFER_SCORER)
    result = run(a,previous)
    result.update(authority_sha256=args.authority_sha, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  seconds=perf_counter()-started)
    payload = (json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    args.output.mkdir()
    (args.output/"RESULT_SAFE.json").write_bytes(payload)
    print(json.dumps(dict(status=STATUS,result_sha256=sha256(payload).hexdigest())),flush=True)


if __name__ == "__main__":
    main()
