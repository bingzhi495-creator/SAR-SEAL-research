"""Read accepted Instance-RSDD masks, signed edits and prior raster metrics without target labels."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace


NEW_SCORER = "06e5da31b3f480f32600cda3af924fdc5ecd5c14bec2fe31a08892d53927a3c3"
B5 = "1269ecbc18820594bcaa1996d96d78e7a659e5adada3f0310a131751e6bb0ffa"
B5_ACCEPT = "5400911314237ff6fe71dbf50887fbb48366034d369494c70ae7b32c1cfb118a"
NEW_CAPTURE = "56fe4d823171c5e7652637aec28a7aad340046ff2772a7e320f9b9dff9d5d8fc"
NEW_REPLAY = "f6e68014db4321d5c73d2e56bc1a0cf4c0337a3b00bb3f3f080a4868a2f40b5f"
NEW_SCORES = {
    74: ("fe0581a4803a7133b672f2588ac50cc9166e59a676e46758c1973f02f4aa4b1d",
         "3214c179b4fd6b52a199b2b60c2f6e3b3792da193f54f299a2c3d442953e7ba8"),
    740: ("fd0d832de7b518dd96da11aedcca8a62ca99bbb91cf782d815b5369b9c93a039",
          "dc6bca9906e813d0ecda381e2915e01948bf4e61a033945696a3c99949b42c43")}
B5_TERMINALS = {
    "capture":"836826714d978469c225994ff8d77adde73d01cb357df4ce6acb118e2d966576",
    "replay5217":"bded198793fa9b99a6a88bac85de917be4195be33e40bc8e91e48aacd1998375",
    "replay5218":"c3579aba79476781a4ffc7dea08f66ec98ef364f09ae1dba77cf7821c9b5c9ac",
    "score5217":"2b5541f0b7ded48bdaac61247be5311c9693c9cd2bbab2bac92259535e01e3a0",
    "score5218":"e0547509678678ea68884ade2165d98b824628df9dfa0a0b62bc35984a953907"}
CLASSICAL = {"attraction_dynamic":"dynamic", "attraction_static_at_dynamic_count":"static"}
HEAD_ARMS = ("fixed","free_matched_count","free_natural","learned")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, expected):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==expected,"Closed input changed: "+path.name)
    return raw


def load(path, expected):
    m=ModuleType(path.stem)
    m.__file__=str(path)
    exec(compile(pinned(path,expected),str(path),"exec"),m.__dict__)
    return m


def arm_names(seeds):
    return ("raw","dynamic","static",*(f"{seed}/{arm}" for seed in seeds for arm in HEAD_ARMS))


def observed(root,stage,worker,phase,observer_phase,result_sha,term_sha):
    result=json.loads(pinned(root/stage/"RESULT_SAFE.json",result_sha))
    term=json.loads(pinned(root/("observed_"+stage)/"TERMINAL.json",term_sha))
    need(type(term["actual_exit"]) is int and term["actual_exit"]==0 and term["output_footer_verified"] is True
         and term["phase"]==observer_phase and result["phase"]==phase and term["result_sha256"]==result_sha
         and term["worker_sha256"]==result["worker_sha256"]==worker,"Exact successful phase closure required")
    return result


def from_5203(root,cohort,new,adapter,core):
    score_sha,term_sha=NEW_SCORES[cohort]
    prior=observed(root,"score",NEW_SCORER,"score","score",score_sha,term_sha)
    need(prior["status"]=="RSDD_CARRIER5203_SCORE_COMPLETE" and prior["detector_seed"]==5203
         and prior["images"]==cohort and prior["head_seeds"]==list(new.SEEDS)
         and prior["checkpoint_sha256"]==new.CHECKPOINT and prior["protocol_sha256"]==new.LOCK
         and prior["signed_budgets_verified"] is True,"Historical second-carrier score differs")
    args=SimpleNamespace(capture=root/"capture",capture_terminal=root/"observed_capture/TERMINAL.json",
                         replay=root/"replay",replay_terminal=root/"observed_replay/TERMINAL.json",cohort=cohort,
                         capture_worker_sha256=NEW_CAPTURE,replay_worker_sha256=NEW_REPLAY)
    # The previous scorer's inputs() independently rechecks every arm's RLE, signed count,
    # target-free replay lineage and candidate order. Its separate labels() is never called here.
    blocks,mapped,budgets,chain=new.inputs(args,adapter,core)
    need(all(prior[k]==v for k,v in chain.items()) and budgets==prior["budget"]
         and mapped==prior["mapped_candidates"],"Original score and image-only streams differ")
    cap=json.loads(pinned(root/"capture/RESULT_SAFE.json",chain["capture_result_sha256"]))
    replay=json.loads(pinned(root/"replay/RESULT_SAFE.json",chain["replay_result_sha256"]))
    reference={view:entry["metrics_fraction"] for view,entry in prior["views"].items()}
    lineage=dict(detector_seed=5203,cohort=cohort,seeds=list(new.SEEDS),arms=list(new.ARMS),
                 checkpoint_sha256=new.CHECKPOINT,original_protocol_sha256=new.LOCK,
                 manifest_sha256=cap["manifest_sha256"],roster_sha256=cap["roster_sha256"],
                 head_sha256=replay["frozen_head_sha256"],readback_result_sha256=cap["readback_result_sha256"],
                 score_result_sha256=score_sha,score_terminal_sha256=term_sha,mapped_candidates=mapped,**chain)
    return blocks,reference,lineage


def merged_choices(seed_edits):
    """Retain one exact pair of classical masks and give each learned policy its seed."""
    seeds=tuple(seed_edits)
    merged={}
    positions=set().union(*(seed_edits[s] for s in seeds))
    for pos in positions:
        row={}
        for original,canonical in CLASSICAL.items():
            values=[seed_edits[s].get(pos,{}).get(original) for s in seeds]
            need(all(x==values[0] for x in values),"Classical choices differ across seed ledgers")
            if values[0] is not None:
                k,mask=values[0]
                row[canonical]=dict(swaps=k,rle=mask)
        for seed in seeds:
            for arm in HEAD_ARMS:
                source="learned_at_attraction_count" if arm=="learned" else arm
                choice=seed_edits[seed].get(pos,{}).get(source)
                if choice is not None:
                    k,mask=choice
                    row[f"{seed}/{arm}"]=dict(swaps=k,rle=mask)
        if row:
            merged[pos]=row
    return merged


def original_metrics(scores):
    seeds=tuple(scores)
    views=scores[seeds[0]]["views"]
    need(all(set(score["views"])==set(views) for score in scores.values()),"Seed score views differ")
    out={}
    for view in views:
        metrics={s:scores[s]["views"][view]["metrics_fraction"] for s in seeds}
        row={}
        for old,name in {"raw":"raw",**CLASSICAL}.items():
            reference=metrics[seeds[0]][old]
            need(all(v[old]==reference for v in metrics.values()),"Archived classical/native raster scores differ")
            row[name]=reference
        for s in seeds:
            for arm in HEAD_ARMS:
                row[f"{s}/{arm}"]=metrics[s]["learned_at_attraction_count" if arm=="learned" else arm]
        out[view]=row
    return out


def from_5202(code,root,adapter,core):
    old=load(code/"SAR_B5_MAIN_TABLES_20261123/portable_rsdd740.py",B5)
    shared,native=old.native_modules(code)
    verified=json.loads(pinned(root/"VERIFIED_SAFE.json",B5_ACCEPT))
    need(verified["passed"] is True and verified["difference_count"]==0
         and verified["maximum_absolute_difference_points"]==0,"Closed frozen 5202 reproduction differs")
    sha=verified["fresh_summary_sha256"]
    cap=observed(root,"capture",B5,"capture","capture",sha["capture"],B5_TERMINALS["capture"])
    a=SimpleNamespace(intake=root/"intake",capture=root/"capture")
    items=old.load_items(a,shared,native)
    checked,checked_sha=old.capture_info(a,shared,items)
    need(checked==cap and checked_sha==sha["capture"] and cap["optimizer_updates"]==0,"Original native capture differs")
    seeds=tuple(old.HEADS)
    references,replays={},{}
    streams=verified["canonical_streams"]
    hashes={"image_manifest":old.MANIFEST_SHA,"predictions_sha256":cap["predictions_sha256"],"audit_sha256":cap["audit_sha256"]}
    for seed in seeds:
        rp,sc=f"replay{seed}",f"score{seed}"
        rsha,ssha=sha[f"edits_{seed}"],sha[f"score_{seed}"]
        replay=observed(root,rp,B5,"replay","verify-edits",rsha,B5_TERMINALS[rp])
        score=observed(root,sc,B5,"score","score",ssha,B5_TERMINALS[sc])
        need(replay["status"]==old.REPLAYED and replay["target_labels_opened"] is False
             and replay["model_forward_calls"]==replay["optimizer_updates"]==0 and replay["exact_support_xor_area"] is True
             and replay["exact_replayed_policy_choices"] is True and score["status"]=="B5_RSDD740_SCORE_COMPLETE"
             and score["point_head_seed"]==replay["point_head_seed"]==seed and score["point_head_sha256"]
             ==replay["point_head_sha256"]==old.HEADS[seed] and score["capture_sha256"]==replay["capture_sha256"]==checked_sha
             and score["replay_sha256"]==rsha and score["edits_sha256"]==replay["edits_sha256"]
             and all(replay[k]==score[k]==cap[k] for k in ("checkpoint_sha256","manifest_sha256","predictions_sha256",
                                                            "audit_sha256","source_partition","images")),
             "Seed-specific frozen image-only replay or score lineage differs")
        hashes[f"edits_{seed}"]=replay["edits_sha256"]
        replays[seed],references[seed]=replay,score
    for k,h in hashes.items():
        need(streams[k]==dict(equal=True,observed_sha256=h,expected_sha256=h),"Original publication digest differs: "+k)
    rows=old.capture_rows(a,shared,items,cap)
    edits=[old.records(root/f"replay{s}"/shared.EDITS,replays[s]["edits_sha256"]) for s in seeds]
    blocks,counts,swaps={},Counter(),Counter()
    for item,(audit,pred),*curves in zip(items,rows,*edits,strict=True):
        i=item["image_id"]
        need(i not in blocks and all(row["image_id"]==i for row in curves)
             and audit["original_hw"]==[512,512],"Lost a candidate image or seed edit row")
        masks=adapter.native_masks(core,pred,audit,i)
        by_seed={s:core.verify_edits(row["changes"],masks,512,512) for s,row in zip(seeds,curves,strict=True)}
        updates=merged_choices(by_seed)
        for options in updates.values():
            for arm,entry in options.items():
                counts[arm]+=int(entry["swaps"]>0)
                swaps[arm]+=entry["swaps"]
        blocks[i]=dict(stock=pred["coco"],ranks=[cand["position"] for cand in audit["candidates"]],
                       changes=updates,rgb=audit["rgb_sha256"],sensor=audit["sensor"],group=audit["acquisition_group"],
                       file=audit["source_image_name"])
    mapped=sum(len(b["stock"]) for b in blocks.values())
    need(len(blocks)==740 and mapped==cap["counts"]["mapped"],"Incomplete 5202 full native denominator")
    for s,replay in replays.items():
        aliases={**CLASSICAL,**{"learned_at_attraction_count" if arm=="learned" else arm:f"{s}/{arm}" for arm in HEAD_ARMS}}
        for key,arm in aliases.items():
            need((counts[arm],swaps[arm])==(replay["counts"][key+"_candidates"],replay["counts"][key+"_swaps"]),
                 "Restored signed budget differs for "+arm)
    lineage=dict(detector_seed=5202,cohort=740,seeds=list(seeds),arms=list(arm_names(seeds)),mapped_candidates=mapped,
                 checkpoint_sha256=shared.CHECKPOINT,original_protocol_sha256=references[seeds[0]]["protocol_sha256"],
                 manifest_sha256=old.MANIFEST_SHA,head_sha256=old.HEADS,b5_verification_sha256=B5_ACCEPT,
                 phase_result_sha256=sha,phase_terminal_sha256=B5_TERMINALS,canonical_stream_sha256=hashes,
                 original_published_reference_sha256=verified["published_reference_sha256"])
    return blocks,original_metrics(references),lineage


def load_inputs(code,run_root,detector_seed,cohort):
    """Return blocks, exact raster_metrics[view][arm], and aggregate lineage for one accepted run."""
    code,run_root=Path(code).resolve(strict=True),Path(run_root).resolve(strict=True)
    need((detector_seed,cohort) in ((5202,740),(5203,740),(5203,74)),"Unknown accepted detector/cohort")
    new=load(code/"INSTANCE_RSDD_CARRIER5203_20261123/score.py",NEW_SCORER)
    prior=new.load(code/"INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py",new.OLD_SCORE)
    adapter=prior.module(code/"UEIKAP_RFDETR_CARRIER_20261122/score_rf.py",prior.GEOMETRY_SOURCE)
    core=adapter.load_geometry(code/"UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    if detector_seed==5203:
        blocks,metrics,lineage=from_5203(run_root,cohort,new,adapter,core)
    else:
        blocks,metrics,lineage=from_5202(code,run_root,adapter,core)
    need(all(set(row)==set(lineage["arms"]) for row in metrics.values()),"Missing archived arm or raster reference")
    lineage["inputs_worker_sha256"]=sha256(Path(__file__).read_bytes()).hexdigest()
    return blocks,metrics,lineage
