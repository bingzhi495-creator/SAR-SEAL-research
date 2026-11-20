"""Fixed external SSDD float-primary official mask and continuous geometry score."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

REPLAY="ee1db45898c42262ba263e15a14ee24c72ef1ee00d5015b5bd243daaeeee01d3"
OLD_SCORE="77159ea493a95973fe422dac1d303eaeba603e6d8af7ae179c72aecc2e4301b0"
FLOAT="8d98e6f49200d5390c17bee66697de1f582133d08063259a561ed5f710be7b6c"
INTEGER="5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55"


def need(ok,why):
    if not ok:
        raise ValueError(why)


def load_worker(path):
    from types import ModuleType

    data=path.read_bytes()
    need(sha256(data).hexdigest()==REPLAY,"Exact locked target replay worker")
    module=ModuleType("_closed_ssdd_replay")
    module.__file__=str(path)
    exec(compile(data,str(path),"exec"),module.__dict__)
    return module


def label_pair(float_gt,int_gt,roster):
    def original(x):
        return {k:v for k,v in x.items() if k not in ("segmentation","area","bbox")}
    need(len(float_gt["images"])==len(int_gt["images"])==187 and
         len(float_gt["annotations"])==len(int_gt["annotations"])==380 and
         float_gt["images"]==int_gt["images"] and float_gt["categories"]==int_gt["categories"] and
         [i["id"] for i in float_gt["images"]]==[r["image_id"] for r in roster] and
         [(i["height"],i["width"]) for i in float_gt["images"]]==[tuple(r["original_hw"]) for r in roster] and
         all(original(a)==original(b) for a,b in zip(float_gt["annotations"],int_gt["annotations"],strict=True)) and
         len({a["id"] for a in float_gt["annotations"]})==380 and
         all(a["image_id"] in {i["image_id"] for i in roster} for a in float_gt["annotations"]),
         "Exact paired 187/380 official float and integer populations")


def geometry_adapter(args,worker,old,old_score,blocks):
    raw=worker.pin(args.old_score,OLD_SCORE).decode()
    start=raw.index("def geometry(")
    stop=raw.index("\n\ndef main(",start)
    raw=raw[start:stop]
    for before,after in (
        ('counts["stock_matches"]==1652 and counts["stock_matches"]==','counts["stock_matches"]=='),
        ('70500-edits[arm]["edited"]','sum(len(b[1]) for b in blocks)-edits[arm]["edited"]')):
        need(raw.count(before)==1,"Exact native geometry cohort seam")
        raw=raw.replace(before,after,1)
    by_id={item["image_id"]:item["rgb_sha256"] for item,_,_,_ in blocks}
    old_score.as_group=lambda item:by_id[item["image_id"]]
    exec(compile(raw,str(args.old_score),"exec"),old_score.__dict__)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","replay-source","old","old-score","native","dev-source","manifest","capture",
                "accepted","replay","replay-accepted","code","score-adapter","shapely-site","float-gt","int-gt","output"):
        p.add_argument("--"+key,type=Path,required=True)
    for key in ("accepted-sha","replay-sha","replay-accepted-sha"):
        p.add_argument("--"+key,required=True)
    a=p.parse_args()
    w=load_worker(a.replay_source)
    w.pin(a.protocol,w.PROTOCOL)
    old=w.load(a.old,w.OLD,"_closed_historical_point_replay")
    dev=w.load(a.dev_source,w.DEV,"_source_native_author_contract")
    dev.cpu_runtime()
    accepted=w.accepted(a,dev)
    roster=w.manifest(a)
    prepared=w.require_prepare(a)
    verified=json.loads(w.pin(a.replay_accepted,a.replay_accepted_sha))
    need(verified["status"]==w.VERIFIED and verified["protocol_sha256"]==w.PROTOCOL and
         verified["replay_sha256"]==a.replay_sha and verified["acceptance_sha256"]==a.accepted_sha and
         verified["edits_sha256"]==prepared["edits_sha256"] and verified["images"]==187 and
         verified["DEV"] is True and all(verified[k] is False for k in ("FIT","TEST","GT_opened","AP")),
         "Exact independent completed label-free mask closure before labels")
    b=list(w.edit_blocks(a,dev,old,roster,accepted,prepared))
    # Give each fixed image the immutable decoded-RGB cluster, without changing any candidates.
    blocks=[(dict(item,rgb_sha256=row["rgb_sha256"]),coco,audit,row["changes"]) for item,coco,audit,row in b]
    need(sum(len(row) for _,row,_,_ in blocks)==prepared["counts"]["candidates"] and len(blocks)==187,
         "Identical native ranked candidate population")
    old_score=w.load(a.old_score,OLD_SCORE,"_original_mask_and_shape_score")
    adapter=w.load(a.score_adapter,dev.SCORE_ADAPTER_SHA,"_official_mask_adapter")
    geom_source=w.load(a.code/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                       old_score.GEOM,"_original_continuous_geometry")
    stock=w.load(a.code/"HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py",old_score.MATCH,"_original_stock_match")
    sys.path.append(str(a.shapely_site.resolve(strict=True)))
    import shapely
    need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Exact continuous native geometry runtime")
    modules=geom_source.geometry_modules(a.code)
    geometry_adapter(a,w,old,old_score,blocks)
    a.output.mkdir()
    pre=dev.write_new(a.output/"PRELABEL_SAFE.json",dict(status="DFINE_SSDD_POINT_SCORE_PRELABEL",
                      protocol_sha256=w.PROTOCOL,acceptance_sha256=a.accepted_sha,replay_sha256=a.replay_sha,
                      replay_accepted_sha256=a.replay_accepted_sha,FIT=False,DEV=True,TEST=False,GT_opened=False,AP=False))
    float_gt=json.loads(w.pin(a.float_gt,FLOAT))
    int_gt=json.loads(w.pin(a.int_gt,INTEGER))
    label_pair(float_gt,int_gt,roster)
    started=perf_counter()
    flat={arm:[r for _,rows,_,changes in blocks for r in old_score.candidate_rows(rows,changes,arm)] for arm in w.MODES}
    scores={}
    for kind,gt in (("float_primary",float_gt),("integer_sensitivity",int_gt)):
        scores[kind]={arm:old_score.official(gt,rows,adapter) for arm,rows in flat.items()}
        for v in scores[kind].values():
            v["AP50"]=v["all12"][1]
    selected=[]
    seen=set()
    for item,_,_,_ in blocks:
        rgb=item["rgb_sha256"]
        if rgb not in seen:
            selected.append(item["image_id"])
            seen.add(rgb)
    need(len(selected)==182,"Prediction-only decoded RGB dedup representatives")
    dedup_gt=deepcopy(float_gt)
    ids=set(selected)
    dedup_gt["images"]=[r for r in dedup_gt["images"] if r["id"] in ids]
    dedup_gt["annotations"]=[r for r in dedup_gt["annotations"] if r["image_id"] in ids]
    dedup={arm:old_score.official(dedup_gt,[r for r in rows if r["image_id"] in ids],adapter)
           for arm,rows in flat.items()}
    for v in dedup.values():
        v["AP50"]=v["all12"][1]
    fixed,witness=old_score.geometry(float_gt,blocks,stock,modules)
    need(all(fixed["budgets"][arm].get("swaps",0)==prepared["counts"].get(arm+"_swaps",0) for arm in w.MODES) and
         all(fixed["budgets"][arm].get("edited",0)==prepared["counts"].get(arm+"_edited",0) for arm in w.MODES),
         "Exact fixed score and prelabel realized budgets")
    raw=dev.encoded
    path=a.output/"STRICT_TRANSITIONS_PRIVATE.jsonl"
    with path.open("xb") as file:
        for row in witness:
            file.write(raw(row))
    main=scores["float_primary"]
    gates=dict(AP_vs_raw=100*(main["free_natural"]["AP"]-main["raw"]["AP"])>=.50,
               AP_vs_fixed=100*(main["free_natural"]["AP"]-main["fixed"]["AP"])>=.30,
               AP75_retained=main["free_natural"]["AP75"]>=main["raw"]["AP75"]-1e-12,
               zero_original_strict90_loss=fixed["strict90_transitions"]["free_natural"]["lost"]==0)
    for arm in ("raw","fixed"):
        c=fixed["comparisons"]["free_natural_minus_"+arm]
        gates["geometry_vs_"+arm]=c["delta_points"]>=.20 and c["interval95_points"][0]>0
    result=dict(status="DFINE_HRSID_TO_SSDD_DEV187_POINT_ACTUAL",protocol_sha256=w.PROTOCOL,
                input_acceptance_sha256=a.accepted_sha,replay_sha256=a.replay_sha,prelabel_score_sha256=pre,
                float_gt_sha256=FLOAT,integer_gt_sha256=INTEGER,images=187,gt_instances=380,
                candidates=prepared["counts"]["candidates"],decoded_RGB_groups=182,
                scores_fraction=scores,dedup_float_fraction=dedup,dedup_gt_instances=len(dedup_gt["annotations"]),
                geometry_points=fixed,transition_witness_sha256=w.file_hash(path),
                prelabel_counts=prepared["counts"],replay_seconds=prepared["seconds"],score_seconds=perf_counter()-started,
                point_optimizer_updates=0,detector_forwards=0,FIT=False,DEV=True,TEST=False,GT_opened=True,AP=True,
                official_TEST_AP=False,source_only_frozen_head=True,gates=gates,advance=all(gates.values()))
    h=dev.write_new(a.output/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=result["status"],result_sha256=h,gates=gates,advance=result["advance"])),flush=True)


if __name__=="__main__":
    main()
