"""Audit native-pixel Boundary IoU on closed Instance-RSDD reserve74 pairs."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask
import shapely


SIDE_INPUTS = "ae600c98bcb353a1a2b4dce4a4b5b2aa9edca46b15a7ddb68fe50ec5b549ff0f"
SIDE_SCORE = "548258c6041187918c564c87e45d1744b3763d1895782bf0b26f6834d45ff89c"
SIDE_SCORE_RESULT = "d506ca71df57dba1560305b8d38564d8f53892b71d54276560cdb25f941d6aec"
SIDE_SCORE_TERMINAL = "3cade7664a76d0d408505a038e53ae43ae50557f49f348e8b3d8f3e9895b98aa"
SIDE_REPLAY_WORKER = "aa590649fb2fa0a70d7fd239b26e9f865efac2d8c0dfe86c4525933f6720f214"
SIDE_REPLAY_RESULT = "d46eaf850a59c9e9de14935bd15e2aa9cd6133f1d333e3c6658ce358df6166e1"
SIDE_REPLAY_TERMINAL = "bb407a8ad84eb0137c9c11793b6a78564ab363b4da9f944662a21e47796c876d"
POINT_SCORE = "5838206836ad765ff589b0f2a476f6b7989eb1760aed8b35a45b36d171603ebf"
POINT_SCORE_RESULT = "c54b9a53a5bfd3f6a54b6d31d93a0789eb176b07bd2cb52c58c6f2ceb4e77b27"
POINT_SCORE_TERMINAL = "c427a4d9af1fbd12cd876f33b56b40115f4f94a3559079060bed08d4efcb858d"
POINT_REPLAY_WORKER = "f8a0ab392bba3134a9a100731d0cbfdcefae0c590c5d0aa64536eeea6efb55fe"
POINT_REPLAY_RESULT = "dbc6e2dbce37cdd8fea34ccefbce16af0ed1ce58db2495de682e278921626fc8"
POINT_REPLAY_TERMINAL = "62b9eecb1411169db4e3620822ff957e555a66981b5af8ee570582044ce04131"
ORIGINAL_SCORE = "b7f46a2332866f8898ed27cd027b21019c10d9d056f3010b256a67bd8887feda"
ORIGINAL_TERMINAL = "f6c19153e4b57449126df492bdbaf8558531df70a368fcfac8168a81e8d7a089"
VERIFIED_B5 = "2c344df83223493ec1009a80afa36bee56ac4656ffe3ab31e3864c0cda09888e"
ARMS = ("SF","SU_equal","PointRend","occupancy")
CONTRASTS = {"SU_equal_minus_SF":(("SU_equal","SF"),),
             "occupancy_minus_PointRend":(("occupancy","PointRend"),)}
STATUS = "INSTANCE_RSDD_RESERVE74_BOUNDARY_IOU_COMPLETE"
VIEWS = {"full":(74,81,47,13,12,11), "date_separated":(42,47,32,8,8,8)}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    data=path.read_bytes()
    need(digest(data)==expected,"SHA-256 mismatch: "+path.name)
    return data


def load(path, expected):
    module=ModuleType(path.stem)
    module.__file__=str(path)
    exec(compile(pin(path,expected),str(path),"exec"),vars(module))
    return module


def closed(root, phase, result_sha, terminal_sha, worker_sha, status):
    receipt=json.loads(pin(root/phase/"RESULT_SAFE.json",result_sha))
    terminal=json.loads(pin(root/f"observed_{phase}"/"TERMINAL.json",terminal_sha))
    need(terminal["phase"]==phase and terminal["actual_exit"]==0 and terminal["output_footer_verified"] is True
         and terminal["result_sha256"]==result_sha and terminal["worker_sha256"]==receipt["worker_sha256"]==worker_sha
         and receipt["status"]==status,"Closed historical "+phase+" receipt")
    return receipt


def require_metric_parity(measured, archived):
    need(set(measured)==set(archived),"Archived COCO metric roster")
    for name, value in measured.items():
        expected=archived[name]
        scalar=float(value) if np.isfinite(value) and value>=0 else None
        need((scalar is None and expected is None) or
             (scalar is not None and expected is not None and abs(expected-scalar)<=1e-12),
             "Archived COCO metric identity: "+name)


def load_context(a):
    sides_dir=a.code/"INDEPENDENT_SIDE_FACTORIAL_20261123"
    point_dir=a.code/"INSTANCE_RSDD_RESERVE_POINTREND_20261123"
    sides=load(sides_dir/"score_sides.py",SIDE_SCORE)
    source=load(sides_dir/"side_inputs.py",SIDE_INPUTS)
    point=load(point_dir/"score_reserve_pointrend.py",POINT_SCORE)
    portable=point.load_portable(a.code)
    historical=closed(a.prior_reserve,"score",ORIGINAL_SCORE,ORIGINAL_TERMINAL,portable.SCORE_SOURCE,
                      "INSTANCE_RSDD_RESERVE74_SCORE_COMPLETE_V1")
    side_score=closed(a.side_run,"score",SIDE_SCORE_RESULT,SIDE_SCORE_TERMINAL,SIDE_SCORE,
                      "INDEPENDENT_SIDE_FACTORIAL_SCORE_COMPLETE")
    side_replay=closed(a.side_run,"replay",SIDE_REPLAY_RESULT,SIDE_REPLAY_TERMINAL,SIDE_REPLAY_WORKER,
                      "INDEPENDENT_SIDE_FACTORIAL_REPLAY_COMPLETE")
    point_score=closed(a.point_run,"score",POINT_SCORE_RESULT,POINT_SCORE_TERMINAL,POINT_SCORE,
                       "INSTANCE_RSDD_RESERVE74_POINTREND_SCORE_COMPLETE_V1")
    point_replay=closed(a.point_run,"replay",POINT_REPLAY_RESULT,POINT_REPLAY_TERMINAL,POINT_REPLAY_WORKER,point.ACCEPTED)
    pin(a.stock_root/"B5_VERIFIED_SAFE.json",VERIFIED_B5)
    reserve=portable.module(a.code/portable.RESERVE/"score_reserve.py",portable.SCORE_SOURCE)
    transfer=portable.module(a.code/"INSTANCE_RSDD_POINTREND_TRANSFER_20261122/score_transfer.py",point.TRANSFER)
    prior=transfer.predecessor(a.code)
    evaluator=prior.module(a.code/"PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py",prior.SCORE_SOURCE)
    adapter=prior.module(a.code/portable.UEIKAP/"score_rf.py",prior.GEOMETRY_SOURCE)
    geometry=adapter.load_geometry(a.code/"UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    require_versions=(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2")
    need(require_versions,"Exact prior Shapely and geometry runtime")
    need(historical["author_train_sha256"]==point_score["author_train_sha256"]==reserve.TRAIN and
         historical["census_sha256"]==point_score["census_sha256"]==reserve.CENSUS,
         "Same historical author reference and parent census")
    cohort=SimpleNamespace(code=a.code,cohort=74,stock_root=a.stock_root)
    rows, provenance=source.load_population(cohort)
    paired, numbers=sides.checked_blocks(rows,pin(a.side_run/"replay/EDITS_PRIVATE.jsonl",side_replay["edits_sha256"]),
                                        (5217,),adapter,geometry)
    need(numbers[5217]["positive_budget_candidates"]==4092 and numbers[5217]["swaps_per_arm"]==30841
         and side_replay["edits_sha256"]==side_score["edits_sha256"],"Archived exact fixed and free per-arm signed count")
    point_args=SimpleNamespace(code=a.code,capture=a.stock_root/"capture",replay=a.point_run/"replay")
    items=[item for item,_,_,_ in rows]
    learned, point_bytes, point_result, _ = point.checked_blocks(
        point_args,portable,transfer,adapter,geometry,items,provenance["manifest_sha256"])
    need(point_bytes==point_replay and point_result==POINT_REPLAY_RESULT==point_score["replay_result_sha256"]
         and point_replay["counts"]["point_equal_swaps"]==point_replay["counts"]["learned_common_swaps"]==31391
         and point_replay["counts"]["point_equal_candidates"]==point_replay["counts"]["learned_common_candidates"]==4114,
         "Archived exact PointRend common endpoint signed count")
    need(list(paired)==list(learned) and len(paired)==74 and sum(len(b["stock"]) for b in paired.values())==7346
         and all(all(paired[i][key]==learned[i][key] for key in ("stock","ranks","group","sensor","file","rgb")) for i in paired),
         "Both archived comparisons share every original mask identity, rank and score")
    census=json.loads(pin(a.code.parent/"results/INSTANCE_RSDD_PARENT_RESERVE_CENSUS_SAFE_20261123.json",reserve.CENSUS))
    need(Counter((v["group"],v["sensor"]) for v in paired.values())==
         {(p["paper_scene"],p["sensor"]):p["images"] for p in census["per_parent"]},"Exact archived parent roster")
    labels, rekeyed=prior.coco_annotation_ids(reserve.select_labels(pin(a.author_train,reserve.TRAIN),paired))
    need(len(labels["images"])==74 and len(labels["annotations"])==81 and
         rekeyed==historical["evaluator_annotation_ids_rekeyed"]==point_score["evaluator_annotation_ids_rekeyed"],
         "Original selected author training polygons and canonical annotation identities")
    dates={p["paper_scene"]:(p["sensor"],p["calendar_date"]) for p in census["per_parent"]}
    date_parents=set(census["date_separated_sensitivity"]["parent_scene_ids"])
    return SimpleNamespace(evaluator=evaluator,paired=paired,learned=learned,gt=labels,historical=historical,
                           side_score=side_score,point_score=point_score,dates=dates,date_parents=date_parents,
                           source_sha=provenance["capture_result_sha256"],census_sha=reserve.CENSUS,
                           author_sha=reserve.TRAIN,roster_sha=portable.ROSTER)


def preflight(ctx, name, ids):
    cohort=dict(ctx.gt,images=[a for a in ctx.gt["images"] if a["id"] in ids],
                annotations=[a for a in ctx.gt["annotations"] if a["image_id"] in ids])
    paired={i:ctx.paired[i] for i in ctx.paired if i in ids}
    learned={i:ctx.learned[i] for i in paired}
    checks=((paired,"raw","raw"),(paired,"5217/FF","fixed"),(paired,"5217/II","free_matched_count"),
            (learned,"point_equal",None),(learned,"learned_common",None))
    metric_count=0
    anchors=None
    for blocks, arm, historical_name in checks:
        values, match=ctx.evaluator.score_one(cohort,blocks,arm,anchors=arm=="raw")
        need(len(values)==13,"All original COCO mask summary fields")
        if historical_name is not None:
            if arm=="raw":
                anchors=match
            require_metric_parity(values,ctx.historical["views"][name]["metrics_fraction"][historical_name])
            require_metric_parity(values,ctx.side_score["views"][name]["metrics_fraction"][arm])
            metric_count+=26
        else:
            require_metric_parity(values,ctx.point_score["views"][name]["metrics_fraction"][arm])
            metric_count+=13
    need(anchors is not None and len(anchors)==len(cohort["annotations"]),"Every historical author hull has its original 0.50 native anchor")
    return paired, learned, cohort, anchors, metric_count


def score_view(ctx, name, ids, utility, stats):
    paired, learned, gt, anchors, metric_count=preflight(ctx,name,ids)
    annotations={a["id"]:a for a in gt["annotations"]}
    author=Counter(paired[a["image_id"]]["group"] for a in annotations.values())
    audit=stats.Audit(utility,ARMS,CONTRASTS)
    groups=set()
    actual=Counter()
    for gid,(i,rank,native) in anchors.items():
        ann=annotations[gid]
        raw=native["segmentation"]
        need(ann["image_id"]==i,"Original author and .50 anchor image")
        polygon=shapely.Polygon(np.asarray(ann["segmentation"][0]).reshape(-1,2))
        if not polygon.is_valid or polygon.area<=0:
            actual["invalid_polygons"]+=1
            continue
        if not 32<=int(mask.area(raw))<1024:
            actual["area_excluded"]+=1
            continue
        side=paired[i]["changes"].get(rank,{})
        point=learned[i]["changes"].get(rank,{})
        incoming={arm:change.get(tag,{}) for arm,change,tag in
                  (("SF",side,"5217/FF"),("SU_equal",side,"5217/II"),
                   ("PointRend",point,"point_equal"),("occupancy",point,"learned_common"))}
        sf=incoming["SF"].get("adds",0)
        pr=incoming["PointRend"].get("swaps",0)
        need(sf==incoming["SU_equal"].get("adds",0) and pr==incoming["occupancy"].get("swaps",0),
             "Original positive or zero realized budgets independently match")
        actual.update(eligible=1,positive_SF=int(sf>0),positive_PointRend=int(pr>0))
        group=paired[i]["group"]
        groups.add(group)
        truth=mask.merge(mask.frPyObjects(ann["segmentation"],512,512))
        variant={arm:u.get("rle",raw) for arm,u in incoming.items()}
        audit.add_rles(group,truth,raw,variant)
    expected=VIEWS[name]
    observed=(len(ids),len(anchors),actual["eligible"],len(author),len(groups),len({ctx.dates[p] for p in groups}))
    need(observed==expected and actual["invalid_polygons"]==0 and actual["area_excluded"]==len(anchors)-actual["eligible"]
         and actual["positive_SF"]==actual["positive_PointRend"]==actual["eligible"]
         and ctx.point_score["views"][name]["geometry"]["counts"]["eligible_matches"]==actual["eligible"],
         "Exact historical .50 matches, editable hulls, eligible parents and sensor-date blocks")
    return dict(images=len(ids),author_hulls=len(anchors),author_parents=len(author),eligible_parents=len(groups),
                eligible_sensor_date_blocks=observed[-1], original_coco_values_matched=metric_count,
                exposure=dict(actual), boundary=audit.report(author_groups=author,blocks={p:ctx.dates[p] for p in author}))


def main():
    parser=ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("code","stock-root","prior-reserve","side-run","point-run","author-train",
                 "metric","stats","protocol","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    for name in ("metric-sha256","stats-sha256","protocol-sha256"):
        parser.add_argument("--"+name,required=True)
    a=parser.parse_args()
    start=perf_counter()
    need(sys.flags.isolated and sys.dont_write_bytecode and not a.output.exists()
         and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Fresh isolated CPU-only score")
    pin(a.protocol,a.protocol_sha256)
    utility=load(a.metric,a.metric_sha256)
    stats=load(a.stats,a.stats_sha256)
    ctx=load_context(a)
    ids={"full":set(ctx.paired),"date_separated":{i for i,b in ctx.paired.items() if b["group"] in ctx.date_parents}}
    views={name:score_view(ctx,name,images,utility,stats) for name,images in ids.items()}
    value=dict(status=STATUS,phase="score",worker_sha256=digest(Path(__file__).read_bytes()),
               protocol_sha256=a.protocol_sha256,metric_sha256=a.metric_sha256,stats_sha256=a.stats_sha256,
               study="retrospective previously evaluated frozen reserve",native_anchor_coco_raster_iou=.50,
               parent_weighting=True,source_capture_result_sha256=ctx.source_sha,author_roster_sha256=ctx.roster_sha,
               author_census_sha256=ctx.census_sha,author_train_sha256=ctx.author_sha,
               previous_score_sha256=ORIGINAL_SCORE,side_replay_sha256=SIDE_REPLAY_RESULT,
               side_score_sha256=SIDE_SCORE_RESULT,point_replay_sha256=POINT_REPLAY_RESULT,
               point_score_sha256=POINT_SCORE_RESULT,frozen_candidate_count=7346,
               policy_budget=dict(fixed_free=dict(positive_candidates=4092,swaps_per_arm=30841),
                                  occupancy_pointrend=dict(positive_candidates=4114,swaps_per_arm=31391)),
               views=views,detector_forwards=0,optimizer_updates=0,historical_test_labels_opened=False,
               previously_unevaluated_reference_labels_opened=False,elapsed_seconds=perf_counter()-start)
    a.output.mkdir()
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with (a.output/"RESULT_SAFE.json").open("xb") as result:
        result.write(raw)
    print(json.dumps(dict(status=STATUS,result_sha256=digest(raw))),flush=True)


if __name__=="__main__":
    main()
