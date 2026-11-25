"""Label-free signed RankIoU replay and separate author source-development scoring."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

HERE=Path(__file__).resolve().parent
HELPER="fde6b39f7fb6d5e2c825c91129731ca5355db1db943a187fcbc562b96baf9c26"
LOCK="b310bb8de9d73f4abda9ab62db4995c27c3f010fcc2939f0dbd9f5d8ef8c20a2"
CORE="d303ecfa5d88af0e6cfb102c905fbc71d5cab9b1b5dd3b18fd173600e331efff"
WRITER="fb49070f10d064fbe1bcce89e7baf1fb59d3ae81950863437adb95c37c7c4b81"
BASE_RENDERER="a0c11c5c37c3a8569e1a54047fbc09598604710861bcd19f4096a0e3abe35959"
BASE_RENDER_RESULT="d12a193bf80e6fb24b3c853c8da7d342f6408aa22e9f3920b873f2d5eb3de1e5"
BASE_RENDER_EXIT="cb84320a93db76c5d11156c40aca0b2b1bd4d0fa2bd746a7d68c6c86ccaef9d2"
BASE_SCORE="b5c61ebbc9d02994b2d9cac210a19e4091693055ec97e257cc8063095c4b86f2"
BASE_SCORE_RESULT="f86390623ffd658e91c17e8a7b99e0859bcc3918b9fce0afda2ff332d72d023a"
BASE_SCORE_EXIT="a3ad0ea5274ecfd7c11c7baf1d4ad2047b1926790871268922c5b90fc97edeb4"
RENDER_STATUS="SPSD_PSEG_EXPECTED_IOU_SIGNED_MASKS_COMPLETE"
EXPECTED=("expectation_exact","expectation_fixed")
RENDER_KEYS=("group_protocol","archive_loader","variable_writer","gate","archive","original_fit",
             "points","core","selector","attraction","base_renderer")
SCORE_KEYS=("gate","archive","authority","points","labels","group_protocol","base_score","base_render","mechanism")
PAIRS=(("expectation_exact","variable_iou"),("expectation_exact","expectation_fixed"),
       ("expectation_fixed","fixed_iou"),("expectation_exact","bce_robust"),("expectation_exact","bce_mean"))


def helper(path):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=HELPER:
        raise ValueError("Exact common receipt helpers")
    m=ModuleType("receipt_helpers")
    m.__file__=str(path)
    exec(compile(raw,str(path),"exec"),vars(m))
    m.LOCK=LOCK
    return m


def render(a,c,cfg):
    start=perf_counter()
    moment=c.load(HERE/"expected_iou.py",CORE)
    diag=Counter()
    def amend(source):
        anchor='    raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\\n").encode()'
        c.need(source.count(anchor)==1,"Bound renderer closeout location")
        return source.replace(anchor,"    result.update(extra_facts())\n"+anchor)
    worker=c.load(cfg.base_renderer,BASE_RENDERER,amend)
    original_loader=worker.load

    def add_exact_policy(path,hash_value):
        prior=original_loader(path,hash_value)
        if hash_value!=WRITER:
            return prior

        def select(p,static,k,area):
            raw=prior.select(p,static,k,area)
            exact,counts=moment.select(prior,p,static,k,area)
            c.need(raw["menu_actions"]==counts["menu_actions"],"Identical signed endpoint menu for both objectives")
            diag["maximum_jensen_gap_bound"]=max(diag["maximum_jensen_gap_bound"],counts.pop("max_gap_bound"))
            diag.update(counts)
            for new,old in zip(EXPECTED,("variable_iou","fixed_iou"),strict=True):
                x,y=(exact[new]["action"],raw[old]["action"])
                diag[new+"_changes"]+=int(x!=y)
                diag[new+"_split_changes"]+=int(len(x[0])!=len(y[0]))
            return raw|exact

        return SimpleNamespace(ARMS=(*prior.ARMS,*EXPECTED), select=select,write_mask=prior.write_mask)

    worker.load=add_exact_policy
    worker.LOCK=LOCK
    worker.STATUS=RENDER_STATUS
    worker.__file__=__file__
    worker.extra_facts=lambda:dict(exact_expected_iou_sha256=CORE,model_independent_endpoint_condition=True,
                                   moment_screen=dict(diag),adapter_seconds=perf_counter()-start)
    cfg.protocol=HERE/"PRELOCK.md"
    cfg.output=a.output
    worker.render(cfg)


def records(c,path,digest):
    rows=[json.loads(raw) for raw in c.pin(path,digest).splitlines()]
    result={i["image_id"]:i for i in rows}
    c.need(len(rows)==len(result)==183,"Exact unique source-development population")
    return result


def require_old_parity(c,new,old):
    c.need(list(new)==list(old),"Exact historical image membership and order")
    for iid, prior in old.items():
        current=new[iid]
        c.need(all(current[key]==value for key,value in prior.items() if key!="candidates"),
               "Exact original image metadata")
        for item,anchor in zip(current["candidates"],prior["candidates"],strict=True):
            for key,metadata in anchor.items():
                if key=="changes" or (key.startswith("variable_") and isinstance(metadata,dict)):
                    c.need(all(item[key][key2]==value for key2,value in metadata.items()),
                           "Exact original salience, native and complete historical masks")
                else:
                    c.need(item[key]==metadata,"Exact old quota, score and candidate anchor")
            if item.get("quota") and item["variable_signed_counts"][EXPECTED[0]][0]==item["quota"]:
                c.need(item["changes"][EXPECTED[0]]==item["changes"][EXPECTED[1]],"Fixed-split exact consistency")


def score(a,c,cfg):
    import numpy as np
    start=perf_counter()
    bound,render_sha=c.closed(a.root/"render/RESULT_SAFE.json",a.root/"observed_render/TERMINAL.json",Path(__file__))
    c.need(bound["status"]==RENDER_STATUS and bound["protocol_sha256"]==LOCK and bound["exact_expected_iou_sha256"]==CORE
           and bound["fit_labels_opened"] is bound["source_gate_labels_opened"] is bound["historical_test_opened"] is False
           and bound["optimizer_updates"]==bound["detector_forwards"]==0,
           "Mandatory naturally closed label-free writer")
    old_render,prior_sha=c.closed(cfg.base_render/"render/RESULT_SAFE.json",cfg.base_render/"RENDER_TERMINAL.json",
                  cfg.base_render/"render_gate.py",BASE_RENDER_RESULT,BASE_RENDER_EXIT,BASE_RENDERER)
    previous,previous_sha=c.closed(cfg.base_score/"score/RESULT_SAFE.json",cfg.base_score/"SCORE_TERMINAL.json",
                  cfg.base_score/"score_gate.py",BASE_SCORE_RESULT,BASE_SCORE_EXIT,BASE_SCORE)
    c.need(previous["render_result_sha256"]==prior_sha and previous["exact_all_archive_score_parity"],
           "Previously accepted score shares exact historical replay")

    def add_exposure(source):
        anchor='new=masks["variable_iou"]'
        c.need(source.count(anchor)==1,"Exact matched-hull diagnostic location")
        return source.replace(anchor,'new=masks["expectation_exact"]')

    orig=c.load(cfg.base_score/"score_gate.py",BASE_SCORE,add_exposure)
    auth=c.load(cfg.authority,orig.ARCH_AUTHORITY)
    old,_,scorer,reader,anchors,archived,groups,_,_=auth.archive(cfg)
    c.pin(cfg.group_protocol,orig.GROUP)
    point=json.loads(c.pin(cfg.points/"actual_retry/RESULT_SAFE.json",old.POINTS))
    c.need(bound["point_terminal_sha256"]==old_render["point_terminal_sha256"]==anchors["point_terminal_sha256"]
           and bound["archive_bce_tensor_sha256"]==orig.ARCH_BCE and bound["archive_render_sha256"]==orig.ARCH_RENDER,
           "Same admitted probability head, native carrier, group and support")
    new=records(c,a.root/"render/GATE_RENDER_PRIVATE.jsonl",bound["render_stream_sha256"])
    base_rows=records(c,cfg.base_render/"render/GATE_RENDER_PRIVATE.jsonl",old_render["render_stream_sha256"])
    require_old_parity(c,new,base_rows)
    del base_rows
    orig.CUES=(*orig.CUES,*EXPECTED)
    mech=c.load(cfg.mechanism,orig.MECHANISM)
    totals,area=orig.signed_support(cfg,point,new,archived,groups,bound,auth.REFERENCE_ARMS,mech)
    c.need(totals[EXPECTED[1]]["grew"]==totals[EXPECTED[1]]["shrank"]==0,
           "Independently verified exact fixed-arm equal actual signs")
    print("All source masks, score order and remaining signed edit counts independently verified before labels.",flush=True)
    truths=reader.labels(cfg,new)
    roster={iid:row["candidates"] for iid,row in new.items()}
    order=sorted(set(groups.values()))
    arms=(*auth.REFERENCE_ARMS,*orig.CUES)
    draws=np.random.Generator(np.random.PCG64(5297)).integers(180,size=(4096,180))
    refs,matches={},{}
    for ref,truth in truths.items():
        metrics={}
        for arm in arms:
            metrics[arm],native=scorer.coco_score(truth,roster,arm)
            if arm=="native":
                matches[ref]=native
        known=previous["references"][ref]
        c.need(len(matches[ref])==known["original_matches"] and len(truth["annotations"])==known["author_hulls"],
               "Exact author denominator and fixed native matching")
        values=scorer.geometry(arms,truth,roster,matches[ref],groups,order)
        result=dict(zip(arms,values.mean(axis=0).tolist(),strict=True))
        c.need(all(abs(metrics[arm][k]-val)<=1e-12 for arm,m in known["metrics"].items() for k,val in m.items())
               and all(abs(result[arm]-val)<=1e-12 for arm,val in known["fixed_iou_points"].items()),
               "Exact rerun of every previous AP and fixed IoU")
        paired={x+"__vs__"+y:orig.contrast(arms,values,metrics,draws,x,y) for x,y in PAIRS}
        refs[ref]=dict(metrics=metrics,fixed_iou_points=result,contrasts=paired,author_hulls=known["author_hulls"],
                       original_matches=known["original_matches"])
    orig.CONTROLS=("variable_iou","expectation_fixed","fixed_iou","bce_robust","bce_mean","near_iou")
    expose,shared=orig.matched_pairs(truths,matches,roster,groups)
    adoption=[doc["contrasts"]["expectation_exact__vs__variable_iou"] for doc in refs.values()]
    qualifies=expose["variable_iou"][0]>=20 and expose["variable_iou"][1]>=8 and all(
        b["iou_points"]>=.15 and b["percentile95_points"][0]>0 and b["ap_points"]>=-.10 for b in adoption)
    a.output.mkdir()
    c.closeout(a.output/"RESULT_SAFE.json",Path(__file__),"SPSD_PSEG_EXPECTED_IOU_SIGNED_SCORE_COMPLETE",
          target_scope="reused source development",fit_or_historical_test_opened=False,detector_forwards=0,optimizer_updates=0,
          render_result_sha256=render_sha,prior_score_sha256=previous_sha,exact_expected_iou_sha256=CORE,
          historical_slot_mask_metric_parity=True,all_actual_signed_rates_verified=True,
          references=refs,common_native_hulls=shared,arm_totals=totals,net_area_change_histogram=area,
          exposure_pass=expose["variable_iou"][0]>=20 and expose["variable_iou"][1]>=8,
          evaluate_promotion=qualifies,elapsed_seconds=perf_counter()-start)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("render","score"))
    for n in ("inputs","helpers","root","output"):
        p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args()
    c=helper(a.helpers)
    c.pin(HERE/"PRELOCK.md",LOCK)
    c.need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
           and a.output.resolve()==(a.root/a.phase).resolve() and not a.output.exists(),
           "Fresh isolated CPU output in the prescribed phase")
    (render if a.phase=="render" else score)(a,c,c.settings(a.inputs,RENDER_KEYS if a.phase=="render" else SCORE_KEYS))

if __name__=="__main__":
    main()
