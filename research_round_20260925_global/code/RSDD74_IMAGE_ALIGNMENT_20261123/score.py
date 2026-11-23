"""Score the prelocked horizontal image-only displacement at fixed native masks, ranks and counts."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
import shapely


LOCK = "b5a74bebb9f2ab99c523d57a6074cd978fb23916dda8c9075ea008e51177f2e6"
SIDES = "a4f729f0154a56b17a75af06c3c1e880ca25f0fe029f7b9eccab175fc597e1f3"
SEEDS = (5217, 5218, 5219)
HEADS = ("phoenix", *(f"{s}/learned" for s in SEEDS))
OLD = ("raw", "dynamic", "static", *HEADS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def sources(args):
    path = args.code / "RSDD74_PHOENIX_SIGNED_SIDES_20261123/score.py"
    data = path.read_bytes()
    require(digest(data) == SIDES, "Original COCO, fixed matches and cluster bootstrap required")
    module = ModuleType("alignment_reference")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module, module.sources(args)


def centroid_x(mask, area, core):
    rows, cols = core.codec.decode(mask).nonzero()
    require(len(rows) == area and area > 0, "Mask area differs when measuring centroid")
    # Index or cell-center centroids differ by 0.5 identically in all arms.
    return float(cols.sum(dtype=np.int64)) / area + .5


def changed(pair, area, core):
    a, b = pair
    return a != b and int(core.codec.area(core.codec.merge([a, b], intersect=True))) != area



SHIFTED = {head: {d: (f"phoenix/{d}" if head == "phoenix" else f"{head.split('/')[0]}/{d}")
                  for d in ("minus", "plus")} for head in HEADS}
ARMS = (*OLD, *(name for halves in SHIFTED.values() for name in halves.values()))
OCC_ARMS = {v for h in HEADS[1:] for v in SHIFTED[h].values()}
OCC_WORKER = "1d022b13283ffabc6c8c824a0f862576a45576fe4b058369dec078434e62104a"
INFER_WORKER = "2b9d2e691bd721d5d3b6889698297bdeb519005e3d0c93d20ba0bb3aa81ca72a"
EDIT_STATUS = "RSDD74_IMAGE_ALIGNMENT_EDITS_COMPLETE"
SCORE_STATUS = "RSDD74_IMAGE_ALIGNMENT_SCORE_COMPLETE"
REQ_RESULT = "08ba5d901c7c6d0b17a8b17cd8a3bb5f646139e66e2aae2459554cacb0fabd2a"
REQ_TERM = "f0e8fbaa487e0819c5f6e7ac2ecaadcee86edbf7d871accb27600021089008de"
OLD_RESULT = "fe94442d5e58c5b2568269f9cefdc5dbb86e307da43cde6b5368db09298786ec"
OLD_TERM = "6684f7d858e40fbe78bc631baa153cf5a011461104df4348bdba4b48b8c89770"
CENSUS = dict(mapped_candidates=7346, eligible_candidates=4137, positive_budget_candidates=4107, swaps_per_arm=29900)


def load_inputs(a, phx, api, cfa, sides):
    request, _, _ = api.observed(a.request, a.request_terminal, "request", api.STATUS, REQ_RESULT, REQ_TERM, phx.BUILDER)
    old, _, _ = api.observed(a.phx_edits, a.phx_edits_terminal, "edits", phx.EDIT, OLD_RESULT, OLD_TERM, sides.PHX_WORKER)
    occ, os, ot = api.observed(a.occ, a.occ_terminal, "edits", "RSDD74_IMAGE_ALIGNMENT_OCCUPANCY_COMPLETE", worker=OCC_WORKER)
    inputs = cfa.source(a.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123/inputs.py", cfa.INPUTS_SHA)
    rows, lineage = inputs.load_population(a)
    req_hash = request["request_sha256"]
    require(request["counts"] == api.TOTALS and all(occ["counts"][k] == v for k,v in api.TOTALS.items())
            and request["protocol_sha256"] == old["protocol_sha256"] == api.LOCK and occ["protocol_sha256"] == LOCK
            and request["target_labels_opened"] is old["target_labels_opened"] is occ["target_labels_opened"] is False
            and occ["GT_opened"] is False and occ["original_all_seeds_exact"] is occ["per_candidate_signed_counts_exact"] is True
            and occ["non_image_features_exact"] is occ["static_geometry_exact"] is True
            and occ["detector_forwards"] == occ["optimizer_updates"] == 0
            and all(occ["counts"][f"{s}_original_exact"] == 4107 for s in SEEDS)
            and occ["request_result_sha256"] == REQ_RESULT and occ["request_terminal_sha256"] == REQ_TERM
            and occ["request_sha256"] == old["request_sha256"] == req_hash
            and occ["input_lineage"] == old["input_lineage"] == request["input_lineage"] == lineage
            and occ["source_replay"] == old["source_replay"] == request["source_replay"], "Native/occupancy source linkage differs")
    req = json.loads(api.pin(a.request / "REQUEST_PRIVATE.json", req_hash))
    require(req["input_lineage"] == lineage and req["source_replay"] == old["source_replay"] and req["protocol_sha256"] == api.LOCK,
            "Original endpoint request lineage differs")
    original = api.pin(a.phx_edits / "EDITS_PRIVATE.jsonl", old["edits_sha256"])
    occ_raw = api.pin(a.occ / "EDITS_PRIVATE.jsonl", occ["edits_sha256"])
    chain = dict(request_result_sha256=REQ_RESULT, request_terminal_sha256=REQ_TERM, request_sha256=req_hash,
                 phx_edits_result_sha256=OLD_RESULT, phx_edits_terminal_sha256=OLD_TERM, phx_edits_sha256=old["edits_sha256"],
                 occ_result_sha256=os, occ_terminal_sha256=ot, occ_edits_sha256=occ["edits_sha256"], input_lineage=lineage,
                 occupancy_worker_sha256=OCC_WORKER, inference_worker_sha256=INFER_WORKER, source_replay=old["source_replay"])
    probs = {}
    for key, sign in (("minus",-1),("plus",1)):
        directory, terminal = getattr(a,key), getattr(a,key+"_terminal")
        receipt, rs, ts = api.observed(directory, terminal, "inference", "RSDD74_IMAGE_ALIGNMENT_PHOENIX_COMPLETE",
                                      worker=INFER_WORKER)
        require(receipt["shift_columns"] == sign and receipt["diagnostic_protocol_sha256"] == LOCK
                and receipt["protocol_sha256"] == api.LOCK and receipt["original_worker_sha256"] == phx.INFER_SHA
                and receipt["target_labels_opened"] is False
                and receipt["native_detector_forwards"] == receipt["optimizer_updates"] == 0
                and (receipt["seed"],receipt["batch_size"],receipt["refine_iters"]) == (5202,2,5)
                and receipt["author_commit"] == phx.AUTHOR and receipt["weight_sha256"] == phx.WEIGHT_SHA
                and receipt["input_lineage"] == lineage and receipt["source_replay"] == old["source_replay"]
                and receipt["request_sha256"] == req_hash
                and all(receipt["counts"][k] == old["inference_counts"][k]
                        for k in ("images","image_encodes","decoder_batches","decoded_candidates","native_count_sum",
                                  "endpoint_pairs","default_area_rule_active","candidate_refinement_iterations")),
                "Wrong Phoenix shift, data, source or session settings")
        probs[key] = api.pin(directory / "ENDPOINT_PROBABILITIES_PRIVATE.jsonl", receipt["endpoint_sha256"])
        chain[key+"_result_sha256"], chain[key+"_terminal_sha256"] = rs, ts
        chain[key+"_endpoint_sha256"] = receipt["endpoint_sha256"]
    return rows, req["images"], original, occ_raw, probs, chain


def as_rows(raw):
    return map(json.loads, raw.splitlines())


def assemble(rows, images, old_raw, occ_raw, endpoint_raw, phx, core, cfa, adapter):
    edits, counts = [], Counter()
    locations = {h: Counter() for h in HEADS}
    for row, req, old, occ, m, p in zip(rows, images, as_rows(old_raw), as_rows(occ_raw), as_rows(endpoint_raw["minus"]),
                                         as_rows(endpoint_raw["plus"]), strict=True):
        item, audit, pred = row[:3]
        require(all(x["image_id"] == item["image_id"] for x in (req,old,occ,m,p))
                and all(set(x) == {"image_id","candidates"} for x in (m,p))
                and set(old["changes"]) == set(occ["changes"]), "Unexpected image or changed candidate roster")
        positive = sorted(int(k) for k in old["changes"])
        require([x["position"] for x in req["candidates"]] == positive, "Positive request roster differs")
        changes = {}
        for pos, cand, minus, plus in zip(positive, req["candidates"], m["candidates"], p["candidates"], strict=True):
            k, coords = cand["k"], np.asarray(cand["coords"])
            old_edits, shifted = old["changes"][str(pos)], occ["changes"][str(pos)]
            det, info = pred["coco"][pos-1], audit["candidates"][pos-1]
            require(set(old_edits) == set(OLD)-{"raw"} and set(shifted) == OCC_ARMS
                    and cand["segmentation"] == det["segmentation"] and cand["score"] == det["score"] == info["score"]
                    and cand["native_rank"] == info["native_rank"] and cand["query_id"] == info["query_id"]
                    and all(v["swaps"] == k for v in (*old_edits.values(),*shifted.values())), "Archived identity or quota differs")
            entries = {**old_edits, **shifted}
            for side, prob in (("minus",minus),("plus",plus)):
                entries[SHIFTED["phoenix"][side]] = phx.ranked_mask(cand, prob, Counter())
            # checked_blocks below checks the exact total number of signed changes.
            # Counting k changes on each requested sign now rules out any off-support change.
            for arm in OCC_ARMS:
                mask = core.codec.decode(entries[arm]["rle"]).astype(bool,copy=False).ravel()
                require(mask[coords[:,0]].sum() == k and (~mask[coords[:,1]]).sum() == k, "Occupancy wrote outside original signed endpoints")
            area = int(core.codec.area(det["segmentation"]))
            for head in HEADS:
                baseline = entries[head]["rle"]
                left, right = (entries[SHIFTED[head][d]]["rle"] for d in ("minus","plus"))
                with_any = changed((baseline,left),area,core) or changed((baseline,right),area,core)
                locations[head]["either_shift_vs_unshifted"] += int(with_any)
                locations[head]["plus_vs_minus"] += int(changed((left,right),area,core))
            counts["positive_budget_candidates"] += 1
            counts["swaps_per_arm"] += k
            counts["endpoint_pairs_per_direction"] += len(coords)
            changes[str(pos)] = entries
        edits.append(dict(image_id=item["image_id"],changes=changes))
    cfa.EDITED = tuple(a for a in ARMS if a != "raw")
    blocks, budget, _ = cfa.checked_blocks(rows, edits, adapter, core)
    require(budget == CENSUS and counts["swaps_per_arm"] == 29900 and counts["positive_budget_candidates"] == 4107
            and counts["endpoint_pairs_per_direction"] == 129621, "Complete roster, endpoints and realized budgets required")
    return blocks, edits, {**budget, **counts}, locations

def exposures(counts, dates):
    return {name: dict(matches=sum(by_group.values()), parents=len(by_group),
                       sensor_date_blocks=len({dates[g] for g in by_group})) for name, by_group in counts.items()}


def geometry(anchors, blocks, polygons, dates, core, sides):
    groups = defaultdict(list)
    coverage = {k:Counter() for k in (*HEADS,"any_head","any_shift_vs_unshifted")}
    counts = Counter(native_matches=len(anchors), eligible_matches=0, area_excluded_matches=0, invalid_polygon_matches=0)
    for gid, (image, pos, detection) in anchors.items():
        truth = polygons[gid]
        if not truth.is_valid or truth.area <= 0:
            counts["invalid_polygon_matches"] += 1
            continue
        native = detection["segmentation"]
        area = int(core.codec.area(native))
        if not 32 <= area < 1024:
            counts["area_excluded_matches"] += 1
            continue
        counts["eligible_matches"] += 1
        edits = blocks[image]["changes"].get(pos,{})
        counts["zero_budget_anchors"] += int(not edits)
        masks = {a: edits[a]["rle"] if a in edits else native for a in ARMS}
        group = blocks[image]["group"]
        cache, entries = {}, []
        for mask in masks.values():
            key = mask["counts"]
            if key not in cache:
                cache[key] = (*sides.geometry_values(mask, truth, area, core), centroid_x(mask,area,core))
            entries.append(cache[key])
        within, from_old = [], []
        for h in HEADS:
            lo, hi = (masks[SHIFTED[h][d]] for d in ("minus","plus"))
            flag = changed((lo,hi),area,core)
            within.append(flag)
            from_old.append(changed((lo,masks[h]),area,core) or changed((hi,masks[h]),area,core))
            if flag:
                coverage[h][group] += 1
        if any(within):
            coverage["any_head"][group] += 1
        if any(from_old):
            coverage["any_shift_vs_unshifted"][group] += 1
        groups[group].append(entries)
    parents = sorted(groups)
    require(bool(parents), "No eligible fixed anchor in the original view")
    keys = [dates[g] for g in parents]
    boot = sides.PairedBootstrap(keys)
    parent_boot = sides.PairedBootstrap(parents)
    means = np.stack([np.mean(groups[g],axis=0) for g in parents])
    indices = {a:i for i,a in enumerate(ARMS)}
    columns = {a:means[:,j] for a,j in indices.items()}
    sides_by_head, responses, fidelity = {}, {}, {}
    for h in (*HEADS,"static"):
        pair = SHIFTED[h] if h in SHIFTED else dict(minus="static",plus="static")
        old, left, right = columns[h], columns[pair["minus"]], columns[pair["plus"]]
        response = (right[:,-1]-left[:,-1])/2
        responses[h] = response
        sides_by_head[h] = dict(signed_horizontal_response=boot.effect(response),
                                minus_from_unshifted=boot.effect(left[:,-1]-old[:,-1]),
                                plus_from_unshifted=boot.effect(right[:,-1]-old[:,-1]))
        gap = old[:,:-1]-(right[:,:-1]+left[:,:-1])/2
        fidelity[h] = {k:boot.effect(gap[:,j]) for j,k in enumerate(sides.GEOMETRY)}
    mean_occ = np.mean([responses[h] for h in HEADS[1:]],axis=0)
    net = responses["phoenix"]-mean_occ
    profile = exposures(coverage, dates)
    exposure = profile["any_head"]
    enough = exposure["matches"] >= 20 and exposure["sensor_date_blocks"] >= 8
    primary = boot.effect(net)
    primary.update(exposure=exposure, changed_20_hulls_8_blocks=enough, lower_bound_positive=primary["interval"][0] > 0,
                   directional_gate=bool(enough and primary["interval"][0] > 0),
                   parent_only_sensitivity=parent_boot.effect(net))
    require(np.all(responses["static"] == 0), "Static geometry-only response must vanish")
    aggs = means.mean(axis=0)
    return dict(counts=dict(counts), represented_groups=len(parents), sensor_date_blocks=len(set(keys)), exposure=profile,
                response_by_head=sides_by_head, occupancy_seed_mean=boot.effect(mean_occ), primary=primary,
                per_seed_phoenix_minus_occupancy={h:boot.effect(responses["phoenix"]-responses[h]) for h in HEADS[1:]},
                nonshifted_minus_mean_shifted=fidelity,
                group_equal_arms={a:dict(zip(sides.GEOMETRY,aggs[i,:-1].tolist(),strict=True)) for a,i in indices.items()})


def score_view(ids, gt, all_blocks, polygons, dates, evaluator, h, core, sides, ref):
    blocks = {i:b for i,b in all_blocks.items() if i in ids}
    part = dict(gt, images=[i for i in gt["images"] if i["id"] in ids],
                annotations=[a for a in gt["annotations"] if a["image_id"] in ids])
    metrics = {}
    for arm in ARMS:
        raw, matches = evaluator.score_one(part,blocks,arm,anchors=arm=="raw")
        metrics[arm] = {k:h.available(v) for k,v in raw.items()}
        if arm == "raw":
            anchors = matches
    geo = geometry(anchors,blocks,polygons,dates,core,sides)
    require(all(geo["counts"][k] == v for k,v in ref["geometry"]["counts"].items())
            and len(ids) == ref["images"] and len(part["annotations"]) == ref["annotations"],
            "Historical original anchors or view differ")
    for arm in OLD:
        prior = ref["geometry"]["group_equal_arms"][arm]
        require(metrics[arm] == ref["metrics_fraction"][arm] and
                all(abs(geo["group_equal_arms"][arm][k]-v) <= 1e-9 for k,v in prior.items()),
                "Original COCO or continuous metrics differ: "+arm)
    quality = {}
    for head, pair in SHIFTED.items():
        unshifted, minus, plus = metrics[head], metrics[pair["minus"]], metrics[pair["plus"]]
        quality[head] = {k: None if any(v[k] is None for v in (unshifted, minus, plus)) else
                         100*(unshifted[k]-(minus[k]+plus[k])/2) for k in unshifted}
    return dict(images=len(ids), annotations=len(part["annotations"]), source_groups=len({b["group"] for b in blocks.values()}),
                metrics_fraction=metrics, nonshifted_minus_mean_shifted_coco_points=quality, geometry=geo)


def run(a):
    started = perf_counter()
    require(not a.output.exists() and digest(a.protocol.read_bytes()) == a.protocol_sha256 == LOCK, "Fresh output and exact lock required")
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Frozen polygon runtime required")
    worker = digest(Path(__file__).read_bytes())
    sides, (phx, api, cfa, h, reserve, old, evaluator, adapter, core) = sources(a)
    if a.phase == "score":
        saved, er, et = api.observed(a.edits, a.edits_terminal,"edits",EDIT_STATUS,worker=worker)
        require(saved["protocol_sha256"] == LOCK and saved["target_labels_opened"] is False, "Missing observed label-free assembly")
    rows, images, original, occ_raw, probs, chain = load_inputs(a,phx,api,cfa,sides)
    blocks, edits, counts, locations = assemble(rows,images,original,occ_raw,probs,phx,core,cfa,adapter)
    raw_edits = b"".join(api.encode(r) for r in edits)
    base = dict(phase=a.phase, protocol_sha256=LOCK, worker_sha256=worker, **chain, counts=counts,
                arms=list(ARMS), all_candidate_mask_difference=locations, edits_sha256=digest(raw_edits),
                signed_counts_by_arm={name: dict(edited_candidates=0, additions=0, removals=0) if name=="raw" else
                                      dict(edited_candidates=counts["positive_budget_candidates"],
                                           additions=counts["swaps_per_arm"], removals=counts["swaps_per_arm"])
                                      for name in ARMS},
                original_arms_exact=True, native_signed_counts_exact=True, detector_forwards=0, optimizer_updates=0)
    if a.phase == "edits":
        a.output.mkdir()
        api.write(a.output / "EDITS_PRIVATE.jsonl", raw_edits)
        result = dict(**base,status=EDIT_STATUS,target_labels_opened=False)
    else:
        require(all(saved[k] == v for k,v in base.items() if k != "phase")
                and api.pin(a.edits/"EDITS_PRIVATE.jsonl",base["edits_sha256"]) == raw_edits,
                "Reconstructed edits differ from independently closed assembly")
        ref, ref_hash, ref_term = api.observed(a.phx_score,a.phx_score_terminal,"score",phx.SCORE,
                                              sides.PHX_SCORE,sides.PHX_TERM,sides.PHX_WORKER)
        require(ref["input_lineage"] == chain["input_lineage"] and ref["edits_sha256"] == chain["phx_edits_sha256"],
                "Reference score lineage differs")
        gt,rekeyed,polygons,sets,dates,label_hash = h.views_and_labels(a,old,reserve,blocks)
        views = {name: score_view(sets[name],gt,blocks,polygons,dates,evaluator,h,core,sides,ref["views"][name])
                 for name in ("full","date_separated")}
        full = views["full"]["geometry"]
        require((full["counts"]["eligible_matches"], full["represented_groups"], full["sensor_date_blocks"]) == (47,12,11)
                and views["date_separated"]["geometry"]["counts"]["eligible_matches"] == 32, "Original fixed denominators required")
        result = dict(**base,status=SCORE_STATUS,target_labels_opened=True,historical_control_metrics_exact=True,
                edits_result_sha256=er,edits_terminal_sha256=et,phx_score_result_sha256=ref_hash,phx_score_terminal_sha256=ref_term,
                author_labels_sha256=label_hash,author_ids_rekeyed=rekeyed,bootstrap_draws=4096,bootstrap_seed=5217,
                cluster_unit="sensor/date, equal parent weight",primary_view="full",primary=full["primary"],views=views)
        a.output.mkdir()
    result["seconds"] = perf_counter()-started
    stamp = api.write(a.output/"RESULT_SAFE.json",api.encode(result))
    print(json.dumps(dict(status=result["status"],result_sha256=stamp)),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    commands = p.add_subparsers(dest="phase",required=True)
    for phase in ("edits","score"):
        sub = commands.add_parser(phase,allow_abbrev=False)
        opts = ["code","stock-root","point-root","request","request-terminal","phx-edits","phx-edits-terminal",
                "occ","occ-terminal","minus","minus-terminal","plus","plus-terminal","protocol","output"]
        if phase == "score":
            opts += ["edits","edits-terminal","phx-score","phx-score-terminal","author-labels"]
        for name in opts:
            sub.add_argument("--"+name,required=True,type=Path)
        sub.add_argument("--protocol-sha256",required=True)
    a = p.parse_args()
    for k,v in vars(a).items():
        if isinstance(v,Path):
            setattr(a,k,v.resolve(strict=k!="output"))
    a.cohort,a.support_root = 74,None
    run(a)


if __name__ == "__main__":
    main()
