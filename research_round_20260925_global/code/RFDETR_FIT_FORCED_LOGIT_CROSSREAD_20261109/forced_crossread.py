"""Fixed-match attribution of the closed forced local-logit edit output."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np

SOURCE_SHA="4840f5511d4225ef1701e88f0b96da2669149ed529b7c071b4dd908d882b074b"
RESULT_SHA="17c79b4645e70d585d48ee1fab33048fc4f65bac0e4ed48a44fadb0bf0ec36a4"
POLICY_SHA="44c79d7459a59d9dd601d5192c9743ecce815725dd24e1456ba14c70573e0031"
ARMS=("native","forced_all","matched_only","unmatched_only","overlap_unmatched_only","other_unmatched_only")


def fixed_substitutions(native,changed,matched,overlap_unmatched):
    eligible=set(changed)
    m=eligible&matched
    u=eligible-m
    if not overlap_unmatched<=u:
        raise ValueError("Overlap is a subset of original unmatched edited rows")
    sets=dict(native=set(),forced_all=eligible,matched_only=m,unmatched_only=u,
              overlap_unmatched_only=overlap_unmatched,other_unmatched_only=u-overlap_unmatched)
    return {name:[changed[j] if j in positions else r for j,r in enumerate(native)]
            for name,positions in sets.items()}


def run(args,base,teacher,prior,local,policy):
    start=time.perf_counter()
    raw=args.selective_result.read_bytes()
    if sha256(raw).hexdigest()!=RESULT_SHA:
        raise ValueError("Exact closed P154 result bytes")
    reference=json.loads(raw)
    env=teacher.Environment(args)
    point_items=base.parts(env.fit)["point_fit"]
    env.s.need(len(point_items)==290 and len({r["image_sha256"] for r in point_items})==288,
               "Exact point fitting role")
    point_rows=base.point_training(env,teacher,prior,local,point_items)
    point,loss=base.train_model(env.torch,np.concatenate([r["x"] for r in point_rows]),
                                np.concatenate([r["y"] for r in point_rows]),prior.weights(point_rows),True)
    env.s.need(len(point_rows)==reference["point_fit_pairs"]
               and sum(len(r["y"]) for r in point_rows)==reference["point_fit_points"]
               and loss==reference["point_fit_loss"],"Exact point training replay")
    pred={k:[] for k in ARMS}
    geometry=Counter()
    identity=[]
    for i,item in enumerate(env.hold,1):
        c=env.image(item)
        actions=base.proposals(c,item["dataset_index"],point,teacher,local,policy)
        edited={a["position"]:policy.edited_mask(c["native"][a["position"]],a["add"],a["delete"])
                for a in actions}
        native=env.s.coerce_rles(c["native"],env.codec)
        forced={j:env.codec.encode(np.asfortranarray(p.astype(np.uint8))) for j,p in edited.items()}
        # Both unconditioned variants exist before reading the fixed matching.
        _,truth,matches,check=env.matches(item,c)
        env.s.need(all(a==b for a,b in zip(native,check,strict=True)),"Exact original raster encodings")
        matched=dict(matches)
        overlap=env.codec.iou(native,env.s.coerce_rles(truth,env.codec),[0]*len(truth)) if native and len(truth) else None
        overlap_unmatched={j for j in forced if j not in matched and overlap is not None and overlap[j].max()>=.50}
        streams=fixed_substitutions(native,forced,set(matched),overlap_unmatched)
        iid=item["image_id"]
        identity.extend((iid,rank,q,score.hex()) for rank,q,score in zip(c["rank"],c["query"],c["scores"],strict=True))
        geometry.update(images=1,candidates=len(native),edited_rows=len(edited),original_matches=len(matches),
                        original_strict=sum(env.geometry.signed_counts(c["native"][j],truth[k])["strict"] for j,k in matches),
                        gt_overlap_unmatched=len(overlap_unmatched))
        for j,p in edited.items():
            o=base.action_outcome(c["native"][j],p,truth[matched[j]] if j in matched else None,env.geometry)
            geometry.update(matched_edits=o["matched"],unmatched_edits=not o["matched"],useful=o["useful"],
                            lost_strict=o["strict_loss"],new_strict=o.get("strict_gain",False),margin=o["margin"],
                            fp_delta=o.get("fp_delta",0),fn_delta=o.get("fn_delta",0))
        for k in ARMS:
            pred[k].extend(dict(image_id=iid,category_id=1,score=score,segmentation=r)
                           for score,r in zip(c["scores"],streams[k],strict=True))
        if i%25==0 or i==len(env.hold):
            print(json.dumps(dict(event="FORCED_ACTION_CROSSREAD_PROGRESS",images=i)),flush=True)
    env.close()
    digest=sha256(json.dumps(identity,separators=(",",":")).encode()).hexdigest()
    env.s.need(digest==reference["all_candidate_identities_sha256"] and
               all(geometry[k]==reference["diagnostics"]["forced_all"][k] for k in geometry),
               "Exact candidate identity and forced geometry replay")
    ids=[x["image_id"] for x in env.hold]
    s=set(ids)
    gt=dict(info=env.gt["info"],images=[x for x in env.gt["images"] if x["id"] in s],
            annotations=[x for x in env.gt["annotations"] if x["image_id"] in s],categories=env.gt["categories"])
    metrics={k:env.s.fit_stats(gt,v,ids) for k,v in pred.items()}
    env.s.need(all(metrics[k]==reference["metrics"][k] for k in ("native","forced_all")),
               "Exact native and forced full AP endpoints")
    keys=("AP","AP75","AP90","APsmall","AR100")
    delta={k:{m:metrics[k][m]-metrics["native"][m] for m in keys} for k in ARMS if k!="native"}
    interaction={m:delta["forced_all"][m]-delta["matched_only"][m]-delta["unmatched_only"][m] for m in keys}
    matched_gate=(delta["matched_only"]["AP"]>=.005 and delta["matched_only"]["AP75"]>=0
                  and delta["matched_only"]["AP90"]>=.01 and geometry["lost_strict"]<=3)
    result=dict(status="SSDD_FIT_FORCED_LOGIT_CROSSREAD_COMPLETE",protocol=args.protocol,
                reference_protocol=reference["protocol"],metrics=metrics,delta_vs_native=delta,
                full_minus_matched_minus_unmatched=interaction,geometry=geometry,
                matched_gate=bool(matched_gate),exact_endpoints=True,point_fit_loss=loss,
                optimizer_updates=dict(point_replay=96,detector=0),all_candidate_identities_sha256=digest,
                uses_gt_for_attribution=True,AP=True,DEV=False,TEST=False,elapsed_s=time.perf_counter()-start,
                cuda_peak_bytes=env.torch.cuda.max_memory_allocated())
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selective-source",type=Path,required=True)
    p.add_argument("--teacher",type=Path,required=True)
    first,_=p.parse_known_args()
    raw=first.selective_source.read_bytes()
    if sha256(raw).hexdigest()!=SOURCE_SHA:
        raise ValueError("Exact published selective action source")
    base=ModuleType("_forced_action_base")
    base.__file__=str(first.selective_source)
    sys.modules[base.__name__]=base
    exec(compile(raw,str(first.selective_source),"exec"),base.__dict__)
    teacher=base.teacher_module(first.teacher)
    teacher.common_args(p)
    for name in ("local-source","local-math","policy","selective-result"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    prior=teacher.module(args.local_source,base.LOCAL_SHA,"_crossread_local_source")
    local=teacher.module(args.local_math,base.MATH_SHA,"_crossread_local_math")
    policy=teacher.module(args.policy,POLICY_SHA,"_crossread_action_policy")
    run(args,base,teacher,prior,local,policy)

