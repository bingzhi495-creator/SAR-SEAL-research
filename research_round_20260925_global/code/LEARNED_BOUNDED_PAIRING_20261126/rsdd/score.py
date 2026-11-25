"""Score the frozen bounded-pairing diagnostic only after verified prelabel output."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from types import ModuleType, SimpleNamespace
import json
import os
import sys

import numpy as np


INPUTS_SHA = "8d5d6b28279bd492de9585f570ca0a8f23919cdfadc0e4214602681ebc94e554"
ORIGINAL_SCORE_SHA = "7039d7d0336f6a0cc3b64bfbe4b0c446b38b073cc90764f446a4dc6d09f90d85"
RESERVE_SCORE_SHA = "5ff671b2b83bb12e322b0c766897bf3cdacaedd233bf0d6edcdfcccf2c938df4"
RESERVE_AUTHORITY_SHA = "07e76ed55e1f83663e74b0db91c9a636d57c33a9c3706b046190c208385644e7"
RESERVE_RESULT_SHA = "b7f46a2332866f8898ed27cd027b21019c10d9d056f3010b256a67bd8887feda"
RESERVE_TERMINAL_SHA = "f6c19153e4b57449126df492bdbaf8558531df70a368fcfac8168a81e8d7a089"
SOLVER_SHA = "99817f192798f52f03efe5e3ac422a453019bf458d33096a03844cb60360436c"
ARMS = ("fixed", "bounded", "free_matched_count")
FIELDS = ("continuous_iou_points", "fp_per_truth_area_points", "fn_per_truth_area_points")
CONTRASTS = (("I_minus_L", 2, 1), ("L_minus_F", 1, 0), ("I_minus_F", 2, 0))
STATUS = "LEARNED_BOUNDED_PAIRING_RSDD_SCORE_COMPLETE"
RENDER_STATUS = "LEARNED_BOUNDED_PAIRING_RSDD_RENDER_COMPLETE"


def need(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    payload = Path(path).read_bytes()
    need(digest(payload) == expected, f"Unexpected SHA-256 for {Path(path).name}")
    return payload


def module(path, expected):
    imported = ModuleType(path.stem)
    imported.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), imported.__dict__)
    return imported


def closed(result_path, terminal_path, result_sha, terminal_sha, worker_sha, phase, status):
    result = json.loads(pin(result_path, result_sha))
    event = json.loads(pin(terminal_path, terminal_sha))
    need(event["phase"] == phase and event["actual_exit"] == 0 and event["output_footer_verified"] is True
         and event["worker_sha256"] == result["worker_sha256"] == worker_sha
         and event["result_sha256"] == result_sha and result["status"] == status,
         "Expected observed natural closure for the exact worker and result")
    return result


def verify_render(a, seed, seed_record):
    rpath = Path(seed_record["result"])
    result = closed(rpath, seed_record["terminal"], seed_record["result_sha"], seed_record["terminal_sha"],
                    a.render_worker_sha, "prepare-edits", RENDER_STATUS)
    need(result["protocol_sha256"] == a.protocol_sha and result["solver_sha256"] == SOLVER_SHA
         and result["view"] == a.view and result["head_seed"] == seed
         and result["label_files_opened"] == result["detector_forwards"] == result["optimizer_updates"] == 0
         and result["exact_historical_baseline_masks"] is True and result["exact_per_candidate_signed_counts"] is True,
         "Verified label-free same-reference renderer and frozen signed counts")
    raw = pin(rpath.parent / "BOUNDED_PRIVATE.jsonl", result["actual_bounded_stream_sha256"])
    ordered = tuple(json.loads(line) for line in raw.splitlines())
    need(len(ordered) == result["counts"]["images"] and len({r["image_id"] for r in ordered}) == len(ordered),
         "Complete observed render stream")
    return ordered, result


def historical_gf(a, source):
    inputs = module(a.code / "RSDD_STRICT_CONTINUOUS_20261123/inputs.py", INPUTS_SHA)
    blocks, raster, lineage = inputs.load_inputs(a.code, a.archive, 5202, 740)
    gf = {i: block for i, block in blocks.items() if block["sensor"] == "GF-3"}
    need(len(gf) == 583 and len({v["group"] for v in gf.values()}) == 76
         and len({v["rgb"] for v in gf.values()}) == 583, "Same original complete GF3 view")
    reports = {seed:json.loads(pin(a.archive/f"score{seed}/RESULT_SAFE.json",
                                 lineage["phase_result_sha256"][f"score_{seed}"]))["views"]["gf3"]["geometry"]["stats"]["group_equal_arms"]
               for seed in (5217,5218)}
    need(tuple(lineage["seeds"]) == (5217,5218), "Both original source head seeds required")
    return gf, raster["gf3"], reports, lineage


def historical_reserve(a, source, core, adapter):
    reserve = module(a.code / "INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/score_reserve.py", RESERVE_SCORE_SHA)
    path = a.archive / "AUTHORITY_SCORE_PRIVATE.json"
    settings = json.loads(pin(path, RESERVE_AUTHORITY_SHA))
    scope = SimpleNamespace(**{k: Path(v) if isinstance(v, str) and ":/" in v else v for k, v in settings.items()})
    scope.code = a.code
    need(Path(scope.author_train) == a.author_labels, "Original author training file for the selected reserve")
    report = closed(a.archive / "score/RESULT_SAFE.json", a.archive / "observed_score/TERMINAL.json",
                    RESERVE_RESULT_SHA, RESERVE_TERMINAL_SHA, RESERVE_SCORE_SHA, "score", reserve.STATUS)
    census = json.loads(pin(scope.census, reserve.CENSUS))
    blocks, exposure = reserve.admitted(scope, source, core, adapter, census)
    need(len(blocks) == 74 and sum(len(r["stock"]) for r in blocks.values()) == 7346,
         "Exact reserve historical full candidate stream")
    dates = {r["paper_scene"]: (r["sensor"], r["calendar_date"]) for r in census["per_parent"]}
    selected = set(census["date_separated_sensitivity"]["parent_scene_ids"])
    reduced = {i for i, row in blocks.items() if row["group"] in selected}
    need(len(reduced) == 42 and len(selected) == 8, "Exact nested reserve date-separated population")
    return blocks, report, reserve, dates, reduced


def verify_and_extend(blocks, streams, core, view):
    if view == "reserve74":
        for block in blocks.values():
            block["changes"] = {rank:{f"5217/{k}":v for k,v in row.items()}
                                for rank,row in block["changes"].items()}
    for seed, (rows, report) in streams.items():
        need([row["image_id"] for row in rows] == list(blocks), "Published candidate identity and order")
        quota = changed = 0
        for item in rows:
            block = blocks[item["image_id"]]
            active = {rank for rank,row in block["changes"].items() if f"{seed}/fixed" in row}
            need({int(rank) for rank in item["changes"]} == active, "Exact frozen positive candidate roster")
            for rank, bounded in item["changes"].items():
                rank = int(rank)
                old = block["changes"][rank]
                stock = block["stock"][rank-1]["segmentation"]
                rle = core.checked_rle(bounded["rle"], 512, 512)
                area = int(core.codec.area(stock))
                retained = int(core.codec.area(core.codec.merge([stock, rle], intersect=True)))
                count = bounded["swaps"]
                need(type(count) is int and 0 < count <= 8 and int(core.codec.area(rle)) == area
                     and area-retained == count == old[f"{seed}/fixed"]["swaps"] == old[f"{seed}/free_matched_count"]["swaps"],
                     "Same RLE signed budget and candidate area")
                old[f"{seed}/bounded"] = dict(swaps=count, rle=rle)
                quota += count
                changed += 1
        need(changed == report["counts"]["positive_candidates"] and quota == report["counts"]["swaps"]
             and sum(len(v["stock"]) for v in blocks.values()) == report["counts"]["mapped_candidates"],
             "Aggregate stream parity")


def geometry(gt, blocks, anchors, seeds, core, shapely):
    polygons = {r["id"]:shapely.Polygon(np.asarray(r["segmentation"][0]).reshape(-1,2)) for r in gt["annotations"]}
    groups = defaultdict(list)
    exposures = {name:{seed:[set(),set()] for seed in seeds} for name,_,_ in CONTRASTS}
    eligible = 0
    for key,(i,rank,native) in anchors.items():
        orig = native["segmentation"]
        area = int(core.codec.area(orig))
        polygon = polygons[key]
        if not polygon.is_valid or polygon.area <= 0 or not 32 <= area < 1024:
            continue
        block = blocks[i]
        group = block["group"]
        scores, cache = [], {}
        eligible += 1
        variants = block["changes"].get(rank,{})
        for seed in seeds:
            masks = [variants.get(f"{seed}/{arm}",{}).get("rle",orig) for arm in ARMS]
            for name,left,right in CONTRASTS:
                if masks[left]["counts"] != masks[right]["counts"]:
                    pair = exposures[name][seed]
                    pair[0].add(key)
                    pair[1].add(group)
            values = []
            for rle in masks:
                compact = rle["counts"]
                if compact not in cache:
                    iou = core.geometric_iou(rle,polygon,shapely)
                    intersection = iou*(area+polygon.area)/(1+iou)
                    cache[compact] = [100*iou,100*(area-intersection)/polygon.area,
                                      100*(polygon.area-intersection)/polygon.area]
                values.append(cache[compact])
            scores.append(values)
        groups[group].append(scores)
    return groups, exposures, eligible


def assess(ap, ci, changed, independent_groups):
    eligible = changed >= 20 and independent_groups >= 8
    if eligible and ap >= .30 and ci[0] > 0:
        evidence = "positive"
    elif eligible and ap <= -.30 and ci[1] < 0:
        evidence = "negative"
    else:
        evidence = "descriptive"
    return dict(decision=evidence, exposure=eligible,
                practical_within_bands=abs(ap)<=.15 and ci[0]>=-.15 and ci[1]<=.15)


def close_enough(current, archive):
    need(set(current) == set(archive), "Complete original COCO metric list")
    for key,value in current.items():
        saved = archive[key]
        if saved is None:
            need(value is None or value < 0, "Archived undefined metric parity")
        else:
            need(value is not None and abs(value-saved) <= 1e-12, "Archived raster score parity: " + key)


def measure(gt, ids, blocks, seeds, archived_ap, archived_geo, evaluator, core, shapely, dates=None,
            reserve_bootstrap=None):
    chosen = dict(gt,images=[r for r in gt["images"] if r["id"] in ids],
                  annotations=[r for r in gt["annotations"] if r["image_id"] in ids])
    scoped = {i:r for i,r in blocks.items() if i in ids}
    native, anchors = evaluator.score_one(chosen,scoped,"raw",anchors=True)
    close_enough(native,archived_ap["raw"])
    by_head = {}
    for seed in seeds:
        metrics = {arm:evaluator.score_one(chosen,scoped,f"{seed}/{arm}",anchors=False)[0] for arm in ARMS}
        for arm in ("fixed","free_matched_count"):
            close_enough(metrics[arm],archived_ap[f"{seed}/{arm}"])
        by_head[seed] = metrics
    groups, exposure, eligible = geometry(chosen,scoped,anchors,seeds,core,shapely)
    parent_keys = sorted(groups)
    values = np.stack([np.mean(groups[key],axis=0) for key in parent_keys])
    baselines = values.mean(axis=0)
    for j,seed in enumerate(seeds):
        for arm in ("fixed","free_matched_count"):
            want = archived_geo[seed][arm]["continuous_iou_points"]
            need(abs(baselines[j,ARMS.index(arm),0]-want)<1e-10, "Archived continuous geometry parity")
    draws = np.random.Generator(np.random.PCG64(5217)).integers(len(parent_keys),size=(4096,len(parent_keys)))

    def stats(head_ids):
        seed_slots = [seeds.index(x) for x in head_ids]
        average = values[:,seed_slots].mean(axis=1)
        ap = np.array([[by_head[seed][arm]["AP"] for arm in ARMS] for seed in head_ids]).mean(axis=0)*100
        result = {}
        for name,left,right in CONTRASTS:
            difference = average[:,left]-average[:,right]
            original = [exposure[name][seed] for seed in head_ids]
            hulls = set().union(*(row[0] for row in original))
            parents = set().union(*(row[1] for row in original))
            independent_groups = len(parents) if dates is None else len({dates[p] for p in parents})
            parent_interval = np.percentile(difference[draws].mean(axis=1),[2.5,97.5],axis=0).T.tolist()
            intervals = parent_interval if dates is None else reserve_bootstrap(difference,[dates[p] for p in parent_keys])
            effect = float(ap[left]-ap[right])
            result[name] = dict(ap_points=effect, changed_hulls=len(hulls), changed_parent_groups=len(parents),
                                changed_analysis_blocks=independent_groups,
                                geometry={field:dict(delta=float(mu), interval_95=ci, parent_interval_95=pi)
                                          for field,mu,ci,pi in zip(FIELDS,difference.mean(axis=0),intervals,parent_interval,strict=True)},
                                **assess(effect,intervals[0],len(hulls),independent_groups))
        result["practical_recovery"] = result["L_minus_F"]["decision"] == "positive" and result["I_minus_L"]["practical_within_bands"]
        return result

    return dict(images=len(ids), author_hulls=len(chosen["annotations"]), original_native_matches=len(anchors),
                eligible_original_matches=eligible, represented_parent_groups=len(parent_keys),
                represented_analysis_blocks=len(parent_keys) if dates is None else len({dates[p] for p in parent_keys}),
                native_metrics=native, head_metrics=by_head,
                head_contrasts={str(seed):stats((seed,)) for seed in seeds}, pooled_contrasts=stats(seeds),
                exact_archived_raster_and_continuous_parity=True)


def run(a):
    started = perf_counter()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11"
         and not a.output.exists(), "Fresh verified CPU scoring")
    pin(a.protocol,a.protocol_sha)
    pin(Path(__file__).with_name("render.py"),a.render_worker_sha)
    seeds = (5217,5218) if a.view == "gf3" else (5217,)
    settings = json.loads(pin(a.render_manifest,a.render_manifest_sha))
    need(set(settings) == {str(seed) for seed in seeds}, "Prespecified frozen seeds")
    replays = {seed:verify_render(a,seed,settings[str(seed)]) for seed in seeds}
    original = module(a.code/"INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py",ORIGINAL_SCORE_SHA)
    evaluator = original.module(a.code/"PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py",original.SCORE_SOURCE)
    adaptor = original.module(a.code/"UEIKAP_RFDETR_CARRIER_20261122/score_rf.py",original.GEOMETRY_SOURCE)
    core = adaptor.load_geometry(a.code/"UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    if a.shapely_site is not None:
        sys.path.append(str(a.shapely_site))
    import shapely
    need(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
         "Original exact continuous polygon evaluation runtime")
    if a.view == "gf3":
        blocks, raster, geo, lineage = historical_gf(a,original)
        for seed, (_,report) in replays.items():
            want = lineage["canonical_stream_sha256"]
            need(report["prior"]["prior_candidate_sha256"]==want["predictions_sha256"]
                 and report["prior"]["prior_candidate_audit_sha256"]==want["audit_sha256"]
                 and report["prior"]["prior_edit_sha256"]==want[f"edits_{seed}"],"Matched original historical payloads")
    else:
        blocks, prior, reserve, dates, nested = historical_reserve(a,original,core,adaptor)
        need(replays[5217][1]["prior"]["prior_verified_result_sha256"] == prior["receipt_sha256"]["replay"],
             "Original reserve verification and score use the same edit archive")
    verify_and_extend(blocks,replays,core,a.view)
    print("Observed label-free render and exact native candidate and signed-budget parity verified.",flush=True)
    if a.view == "gf3":
        author_sha = original.AUTHOR_TEST
        gt, _ = original.coco_annotation_ids({"info":{}, **json.loads(pin(a.author_labels,author_sha))})
        need(len(gt["images"]) == 740 and len(gt["annotations"]) == 872,"Exact original author TEST release")
        report = measure(gt,set(blocks),blocks,seeds,raster,geo,evaluator,core,shapely)
        need((report["images"],report["author_hulls"],report["original_native_matches"],report["eligible_original_matches"],
              report["represented_parent_groups"]) == (583,631,629,425,68),"Exact GF3 score evaluability")
        views = {"gf3":report}
    else:
        author_sha = reserve.TRAIN
        gt, _ = original.coco_annotation_ids(reserve.select_labels(pin(a.author_labels,author_sha),blocks))
        views = {}
        for title,ids,expected in (("full",set(blocks),(74,81,47)),("date_separated",nested,(42,47,32))):
            archived = prior["views"][title]
            raster = {key if key=="raw" else f"5217/{key}":value for key,value in archived["metrics_fraction"].items()}
            geo = {5217:archived["geometry"]["group_equal_arms"]}
            report = measure(gt,ids,blocks,seeds,raster,geo,evaluator,core,shapely,dates,reserve.bootstrap)
            need((report["images"],report["author_hulls"],report["eligible_original_matches"])==expected,
                 "Exact historical reserve view denominator")
            views[title] = report
    result = dict(status=STATUS, phase="score", view=a.view, worker_sha256=digest(Path(__file__).read_bytes()),
                  protocol_sha256=a.protocol_sha, render_worker_sha256=a.render_worker_sha,
                  render_manifest_sha256=a.render_manifest_sha, author_reference_sha256=author_sha,
                  render_result_sha256={key:values["result_sha"] for key,values in settings.items()},
                  original_historical_data_only=True, detector_forwards=0, optimizer_updates=0,
                  views=views, elapsed_seconds=perf_counter()-started)
    a.output.mkdir()
    data = (json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with (a.output/"RESULT_SAFE.json").open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=STATUS,result_sha256=digest(data))),flush=True)


def main():
    parser = ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("code","archive","protocol","render-manifest","author-labels","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("protocol-sha","render-manifest-sha","render-worker-sha"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--view",choices=("gf3","reserve74"),required=True)
    parser.add_argument("--shapely-site",type=Path)
    run(parser.parse_args())


if __name__=="__main__":
    main()
