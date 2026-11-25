"""Read and score two matched BCE repeat seeds on the exact original group roster."""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from pycocotools import mask as codec


LOCK = "3b51839c19c0ff03f2371d45be7c9d84e161531f8a70962d1c85a4472c93801a"
LEGACY_SCORE = "b5c61ebbc9d02994b2d9cac210a19e4091693055ec97e257cc8063095c4b86f2"
HELPER = "09c542d2fc700d2105bd5d2c248437547b0590f77ce9271743640f79090eed7d"
ORIGINAL_RENDER = "d12a193bf80e6fb24b3c853c8da7d342f6408aa22e9f3920b873f2d5eb3de1e5"
ORIGINAL_RENDER_TERMINAL = "cb84320a93db76c5d11156c40aca0b2b1bd4d0fa2bd746a7d68c6c86ccaef9d2"
ORIGINAL_SCORE = "f86390623ffd658e91c17e8a7b99e0859bcc3918b9fce0afda2ff332d72d023a"
ORIGINAL_SCORE_TERMINAL = "a3ad0ea5274ecfd7c11c7baf1d4ad2047b1926790871268922c5b90fc97edeb4"
SEEDS = (5298,5299)
REFERENCES = ("pseg","spsd")
SAME_ARMS = ("same_bce_robust","same_bce_mean","same_bce_cell_pessimistic")
NAMES_IN_HISTORICAL = {k:k.removeprefix("same_") for k in SAME_ARMS}
POOLED_ARMS = ("same_bce_robust","fixed_iou","same_bce_mean","static","near_iou",
               "variable_mean_iou","variable_raw_min","same_bce_cell_pessimistic")


def need(condition,message):
    if not condition:
        raise ValueError(message)


def repeat_artifacts(a,seed,path,old,root,base,legacy):
    checksum=getattr(a,f"result{seed}")
    t_sha=getattr(a,f"terminal{seed}")
    observer=json.loads(root.pin(path/"RENDER_TERMINAL.json",t_sha))
    need(observer["actual_exit"]==0 and observer["output_footer_verified"] is True and observer["phase"]=="render"
         and observer["worker_sha256"]==a.renderer_sha and observer["result_sha256"]==checksum
         and observer["seed"]==seed,"Exact observed natural closure for both label-free seed replays")
    entry=json.loads(root.pin(path/"render/RESULT_SAFE.json",checksum))
    need(entry["status"]=="SPSD_PSEG_REPEAT_VARIABLE_SIGNED_BUDGET_GATE_RENDER_COMPLETE"
         and entry["seed"]==seed and entry["worker_sha256"]==a.renderer_sha and entry["protocol_sha256"]==LOCK
         and entry["repeat_fit_result_sha256"]==a.fit_sha and entry["repeat_fit_readback_sha256"]==a.fit_readback_sha
         and entry["repeat_bce_tensor_sha256"]==getattr(a,f"tensor{seed}")
         and entry["method_sha256"]=="fb49070f10d064fbe1bcce89e7baf1fb59d3ae81950863437adb95c37c7c4b81"
         and entry["archive_render_sha256"]==legacy.ARCH_RENDER and entry["archive_score_sha256"]==legacy.ARCH_SCORE
         and entry["archive_bce_tensor_sha256"]==legacy.ARCH_BCE and entry["exact_archive_rle_preserved"] is True
         and entry["point_terminal_sha256"]==base["point_terminal_sha256"]
         and entry["point_result_sha256"]==old.POINTS and entry["decoded_groups"]==180
         and entry["detector_forwards"]==entry["optimizer_updates"]==0
         and entry["source_gate_labels_opened"] is entry["fit_labels_opened"] is entry["historical_test_opened"] is False,
         "Exactly matched seed, source archive, FIT origin and labels absent from render")
    return entry


def indexed_rows(path,doc,loader):
    rows=[json.loads(x) for x in loader.pin(path/"render/GATE_RENDER_PRIVATE.jsonl",doc["render_stream_sha256"]).splitlines()]
    by_id={p["image_id"]:p for p in rows}
    need(len(rows)==len(by_id)==183,"All exact source rows")
    return by_id


def same_bce_support(a, point, repeats, root, mech):
    from io import BytesIO

    point_root=a.points/"actual_retry/source_gate"
    index=[json.loads(line) for line in root.pin(point_root/"POINT_INDEX_PRIVATE.jsonl",
             point["partitions"]["source_gate"]["point_index_sha256"]).splitlines()]
    bases=0
    for row in index:
        iid=row["image_id"]
        packet=root.pin(point_root/row["filename"],row["sha256"])
        with np.load(BytesIO(packet),allow_pickle=False) as data:
            ranks,offset,coords,k=[data[f] for f in ("native_ranks","point_offsets","signed_coordinates","static_offsets")]
        for j,r in enumerate(ranks):
            quota=int(k[j+1]-k[j])
            if not quota:
                continue
            p=repeats[SEEDS[0]][iid]["candidates"][int(r)-1]
            mask=codec.decode(p["native"]).astype(bool,order="C")
            points=coords[offset[j]:offset[j+1]]
            for seed in SEEDS:
                q=repeats[seed][iid]["candidates"][int(r)-1]
                need(p["native"]==q["native"] and p["quota"]==q["quota"]==quota,"Same per-seed baseline geometry")
                for arm in SAME_ARMS:
                    encoded=q["changes"][arm]
                    corrected=codec.decode(encoded).astype(bool,order="C")
                    need(encoded["size"]==q["native"]["size"] and mech.signed_changes(mask,corrected,points,quota)==(quota,quota),
                         "Same-seed BCE comparator independently preserves original add/drop support and equal split")
            bases+=1
    need(bases==8785,"All 8,785 repeated same-seed native positive quotas")


def classification_rows(native,same_seed):
    if not same_seed:
        return native
    rows={}
    for iid,v in native.items():
        rows[iid]=[dict(q,changes=dict(q["changes"],**{old:q["changes"][new] for new,old in NAMES_IN_HISTORICAL.items()}))
                   if q["quota"] else q for q in v]
    return rows


def main():
    parser=ArgumentParser(description=__doc__)
    for key in ("gate","archive","authority","legacy-root","original-render","points","labels","protocol","group-protocol","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    for key in ("renderer-sha","fit-sha","fit-readback-sha","result5298","result5299","terminal5298","terminal5299",
                "tensor5298","tensor5299"):
        parser.add_argument("--"+key,required=True)
    a=parser.parse_args()
    start=perf_counter()
    from hashlib import sha256
    from types import ModuleType

    legacy_path=a.legacy_root/"score_gate.py"
    source=legacy_path.read_bytes()
    need(sha256(source).hexdigest()==LEGACY_SCORE,"Exact trusted initial variable scorer")
    legacy=ModuleType("prior_variable_score")
    legacy.__file__=str(legacy_path)
    exec(compile(source,str(legacy_path),"exec"),legacy.__dict__)
    root=Path(__file__).parent
    legacy.pin(root/"render_gate.py",a.renderer_sha)
    legacy.pin(a.protocol,LOCK)
    legacy.pin(a.group_protocol,legacy.GROUP)
    pool=legacy.load(root/"repeat_summary.py",HELPER)
    mech=legacy.load(a.legacy_root/"mechanism.py",legacy.MECHANISM)
    auth=legacy.load(a.authority,legacy.ARCH_AUTHORITY)
    old,archive_root,scorer,reader,base,archived,groups,anchor,_=auth.archive(a)
    accepted={}
    for seed in SEEDS:
        path=root/f"seed{seed}"
        accepted[seed]=repeat_artifacts(a,seed,path,old,archive_root,base,legacy)
    need(accepted[5298]["repeat_fit_checkpoint_sha256"]==accepted[5299]["repeat_fit_checkpoint_sha256"]
         and accepted[5298]["repeat_fit_terminal_sha256"]==accepted[5299]["repeat_fit_terminal_sha256"]
         and accepted[5298]["repeat_bce_tensor_sha256"]!=accepted[5299]["repeat_bce_tensor_sha256"],
         "Distinct weights from the same predeclared FIT pair")
    point=json.loads(legacy.pin(a.points/"actual_retry/RESULT_SAFE.json",old.POINTS))
    need(point["train_labels_opened"] is point["test_labels_opened"] is point["test_images_opened"] is False,
         "Closed label-free input to original signed supports")
    repeat_rows={seed:indexed_rows(root/f"seed{seed}",record,legacy) for seed,record in accepted.items()}
    geom={}
    for seed in SEEDS:
        geom[seed]=legacy.signed_support(a,point,repeat_rows[seed],archived,groups,accepted[seed],
                                      (*auth.REFERENCE_ARMS,*SAME_ARMS),mech)[0]
    same_bce_support(a,point,repeat_rows,legacy,mech)
    history_doc=json.loads(legacy.pin(a.original_render/"render/RESULT_SAFE.json",ORIGINAL_RENDER))
    history_term=json.loads(legacy.pin(a.original_render/"RENDER_TERMINAL.json",ORIGINAL_RENDER_TERMINAL))
    hist_check=json.loads(legacy.pin(a.legacy_root/"SCORE_TERMINAL.json",ORIGINAL_SCORE_TERMINAL))
    history=json.loads(legacy.pin(a.legacy_root/"score/RESULT_SAFE.json",ORIGINAL_SCORE))
    need(history_term["actual_exit"]==hist_check["actual_exit"]==0 and history_term["output_footer_verified"] is True
         and hist_check["output_footer_verified"] is True and history_term["result_sha256"]==ORIGINAL_RENDER
         and hist_check["result_sha256"]==ORIGINAL_SCORE and history["render_result_sha256"]==ORIGINAL_RENDER
         and history["render_stream_sha256"]==history_doc["render_stream_sha256"]
         and history["exact_original_support_and_actual_total_budget"] is True,
         "Exact already closed historical variable and fixed-score anchor")
    active_rows={5297:indexed_rows(a.original_render,history_doc,legacy),**repeat_rows}
    need(all(set(rows)==set(archived) for rows in active_rows.values()),"Identical historical and repeated image membership")
    print("Both repeat archived slots, original signed endpoint bounds and same-seed controls verified before any reference.",
          flush=True)
    truths=reader.labels(a,archived)
    native={seed:{i:r["candidates"] for i,r in items.items()} for seed,items in active_rows.items()}
    basic_native={i:r["candidates"] for i,r in archived.items()}
    order=sorted(set(groups.values()))
    draws=np.random.Generator(np.random.PCG64(5297)).integers(180,size=(4096,180))
    refs={seed:{} for seed in (5297,*SEEDS)}
    vectors={seed:{} for seed in refs}
    parts={seed:{} for seed in SEEDS}
    matching={}
    for ref,truth in truths.items():
        archived_metrics={}
        for arm in auth.REFERENCE_ARMS:
            metric,assoc=scorer.coco_score(truth,basic_native,arm)
            need(all(abs(metric[k]-v)<=1e-12 for k,v in anchor["references"][ref]["metrics"][arm].items()),
                 "Exact unchanged archived mask AP and recall benchmark")
            if arm=="native":
                matching[ref]=assoc
                need(len(assoc)==anchor["references"][ref]["original_matches"],"Original exact native match count")
            archived_metrics[arm]=metric
        for seed in (5297,*SEEDS):
            historical=seed==5297
            arms=(*auth.REFERENCE_ARMS,*legacy.CUES) if historical else (*auth.REFERENCE_ARMS,*legacy.CUES,*SAME_ARMS)
            metrics=dict(archived_metrics)
            for arm in arms[len(archived_metrics):]:
                metrics[arm]=history["references"][ref]["metrics"][arm] if historical else scorer.coco_score(truth,native[seed],arm)[0]
            table=scorer.geometry(arms,truth,native[seed],matching[ref],groups,order)
            fixed=dict(zip(arms,table.mean(axis=0).tolist(),strict=True))
            known=history["references"][ref] if historical else anchor["references"][ref]
            for arm,expected in known["fixed_iou_points"].items():
                need(abs(fixed[arm]-expected)<=1e-12,"Exact historical and archived group IoU reconstruction")
            contrasts={}
            vectors[seed][ref]={}
            comparison=tuple(arm for arm in arms if arm!="variable_iou")
            old_geometry=table[:,arms.index("variable_iou")]
            for arm in comparison:
                delta=old_geometry-table[:,arms.index(arm)]
                contrasts[arm]=pool.paired_seed_pool((delta,),(metrics["variable_iou"]["AP"]-metrics[arm]["AP"],),draws)
                vectors[seed][ref][arm]=delta
            if not historical:
                parts[seed][ref]={}
                for arm in ("fixed_iou","same_bce_robust"):
                    decomp,details=legacy.decomposition(truth,native[seed],matching[ref],groups,order,arm,draws,mech)
                    need(np.allclose(decomp,vectors[seed][ref][arm],rtol=0,atol=1e-12),
                         "Exact true overlap-area decomposition of each paired contrast")
                    parts[seed][ref][arm]=details
            refs[seed][ref]=dict(metrics=metrics,fixed_iou_points=fixed,variable_iou_minus=contrasts,
                                 original_matches=len(matching[ref]),author_hulls=len(truth["annotations"]))
        print("Scored both seeds plus frozen historical geometry under",ref,flush=True)

    exposure,report,decision={},{},{}
    for seed in (5297,*SEEDS):
        e,m=legacy.matched_pairs(truths,matching,classification_rows(native[seed],seed in SEEDS),groups)
        exposure[seed]={**e,**{new:e[old] for new,old in NAMES_IN_HISTORICAL.items()}}
        alias={v:k for k,v in NAMES_IN_HISTORICAL.items()}
        m["exposure"]={alias.get(k,k):v for k,v in m["exposure"].items()}
        m["paired_reference_signed_facts"]={alias.get(k,k):v for k,v in m["paired_reference_signed_facts"].items()}
        report[seed]=m
        if seed in SEEDS:
            decision[seed]=pool.thresholds(refs[seed],exposure[seed])
        else:
            need(m["counts"]==history["shared_hull_diagnostic"]["counts"],"Exact frozen historical exposure cohort")
    pooled={}
    for seeds in (SEEDS,(5297,*SEEDS)):
        group_results={}
        for ref in REFERENCES:
            group_results[ref]={"variable_iou_minus":{}}
            for arm in POOLED_ARMS:
                keys=[NAMES_IN_HISTORICAL.get(arm,arm) if seed==5297 else arm for seed in seeds]
                v,a_p=[],[]
                for seed,key in zip(seeds,keys,strict=True):
                    v.append(vectors[seed][ref][key])
                    a_p.append(refs[seed][ref]["variable_iou_minus"][key]["ap_points"])
                group_results[ref]["variable_iou_minus"][arm]=pool.paired_seed_pool(v,a_p,draws)
        observed={arm:tuple(min(exposure[seed][arm][j] for seed in seeds) for j in (0,1)) for arm in POOLED_ARMS}
        pooled["repeat_only" if seeds==SEEDS else "historical_and_repeats"]=dict(seeds=seeds,references=group_results,
                    conservative_per_seed_minimum_exposure=observed,advancement=pool.thresholds(group_results,observed))
    out=dict(status="SPSD_PSEG_REPEAT_VARIABLE_SIGNED_BUDGET_GATE_SCORE_COMPLETE",protocol_sha256=LOCK,
             worker_sha256=legacy.digest(Path(__file__).read_bytes()),matched_seed_fits=a.fit_sha,fit_readback_sha256=a.fit_readback_sha,
             renderer_sha256=a.renderer_sha,render_terminals={seed:getattr(a,f"terminal{seed}") for seed in SEEDS},
             render_results={seed:getattr(a,f"result{seed}") for seed in SEEDS},
             historical_score_sha256=ORIGINAL_SCORE,historical_render_sha256=ORIGINAL_RENDER,
             native_mask_candidate_slots=18300,images=183,decoded_groups=180,paired_group_resamples=4096,
             exact_archive_and_actual_signed_budget_parity=True,repeated_source_development=True,
             references_by_seed=refs,geometry_decomposition=parts,release_safe_signed_usage=geom,
             shared_hull_diagnostic=report,per_seed_advancement=decision,pooled=pooled,
             repeats_confirmed=all(x["advances"] for x in decision.values()) and pooled["repeat_only"]["advancement"]["advances"],
             detector_forwards=0,optimizer_updates=0,fit_labels_opened=False,historical_test_opened=False,
             external_acquisition_evaluation=False,elapsed_seconds=perf_counter()-start)
    data=(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    a.output.mkdir()
    with (a.output/"RESULT_SAFE.json").open("xb") as result:
        result.write(data)
    print(json.dumps(dict(status=out["status"],result_sha256=legacy.digest(data))),flush=True)


if __name__=="__main__":
    main()
