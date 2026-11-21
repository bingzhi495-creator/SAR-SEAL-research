"""Execute frozen-PointRend matched-IoU writer probe, FIT only."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from pycocotools import mask as codec


PRELOCK = "422d4df009291013d58af0def766154c2637e87f5c9ffd3f4678bbba517ce191"
PEER = "c791b3eb637acf5e30341768a53fa1acb8a3c8db5d6153717474b0b6958fc319"
VERIFY = "daa3f6dc7751fe546700b8297b89c0f7295238b3876631f073c8f863d72d449f"
ARMS = ("N","B","J","T","R")
MIB = 1024**2


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return sha256(raw).hexdigest()


def write(path, value):
    raw = (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return sha(raw)


@torch.inference_mode()
def infer_one_image(head, image_basis, r, mods):
    items = []
    complete = {}
    for row, box, logit, rle, k, saved in r:
        item = dict(row,box_xyxy01=box,raw_logits=logit,rle=rle,k=k)
        identity, coords, k, box, skip = mods.infer._row_inputs(item,(800,800))
        complete[identity] = dict(T=[],R=[])
        if not skip:
            items.append((identity,coords,k,box,logit,rle,saved))
    feature = torch.from_numpy(image_basis).unsqueeze(0)
    for first in range(0,len(items),12):
        entry = items[first:first+12]
        boxes = torch.stack([x[3] for x in entry])
        raw = torch.from_numpy(np.stack([x[4] for x in entry]))[:,None]
        coarse = mods.adapter.coarse_roi_logits(raw, boxes, (800,800))
        def predict(xy, samples):
            return head(mods.adapter.fine_features_at_roi_points(feature,boxes,xy,(800,800)), samples)
        pred = mods.render.subdivide_binary_mask(coarse,predict,initial_size=(7,7),native_size=(224,224),points_per_step=784)
        need(pred.point_evaluations == 3136*len(entry),"Exact frozen subdivision calls")
        for i,(key,c,k,b,_,rle,saved) in enumerate(entry):
            if not k:
                need(not saved["B"] and not saved["J"],"Original zero budget")
                continue
            logits, box = pred.logits[i:i+1], boxes[i:i+1]
            idx = torch.from_numpy(c.reshape(-1))
            xy = torch.stack((idx%800,torch.div(idx,800,rounding_mode="floor")),dim=-1)[None]
            point = mods.adapter.paste_probabilities_at_pixels(logits,box,xy,(800,800)).numpy().reshape(-1,2)
            old, _ = mods.infer._edits(c,point,k,mods.budget)
            need(old["B"].tolist()==saved["B"] and old["J"].tolist()==saved["J"],
                 "Exact original balanced and joint posteriors through their edits")
            crop, pixels = mods.infer._crop_grid(b,(800,800))
            xu,yu=xy[0,:,0].numpy(),xy[0,:,1].numpy()
            need(np.all((yu>=crop[0].start)&(yu<crop[0].stop)&(xu>=crop[1].start)&(xu<crop[1].stop)),
                 "All edited posterior support is in the original crop")
            mask = codec.decode(rle).astype(bool,copy=False)
            q = mods.adapter.paste_probabilities_at_pixels(logits,box,pixels,(800,800)).numpy().reshape(mask[crop].shape)
            need(np.allclose(point.ravel(),q[yu-crop[0].start,xu-crop[1].start],rtol=0,atol=1e-6),
                 "Identical endpoint and dense crop posterior")
            chosen = mods.policy.expected_ratio_writes(c,point,k,int(codec.area(rle)),
                                                       float(q[mask[crop]].sum(dtype=np.float64)),
                                                       float(q.sum(dtype=np.float64)))
            complete[key] = {arm:ids.tolist() for arm,ids in chosen.items()}
    return complete


def score(gt,pools,rows,budgets,old,new,helpers,coco_curve,lock):
    refs = {helpers.geometry.identity(r):r for r in rows}
    keys, streams, groups = [], {a:[] for a in ARMS}, defaultdict(list)
    counts,strict = Counter(),Counter()
    for image in gt["images"]:
        iid = image["id"]
        for c in pools[iid]:
            key = (iid,c["rank"],c["query_id"])
            keys.append(key)
            masks = dict.fromkeys(ARMS,c["rle"])
            if key in refs:
                rec = refs[key]
                options = {arm:old[key][arm] for arm in ("B","J")}
                options.update(new[key])
                cxy = rec["coordinates"]
                vals = {}
                for a in ARMS:
                    indices = options.get(a,[])
                    budget = 2*budgets[key]["k"] if a!="N" else 0
                    need(len(indices)==budget==len(set(indices)),"Realized original matched edits")
                    adds = int(np.isin(cxy[:,0],indices).sum())
                    drops = int(np.isin(cxy[:,1],indices).sum())
                    edit = dict(flipped=indices,adds=adds,drops=drops,edits=budget,area_delta=adds-drops)
                    helpers.geometry.sparse_edit(rec,edit)
                    masks[a] = helpers.sparse.xor_rle(c["rle"],np.asarray(indices,dtype=np.int64))
                    need(int(codec.area(masks[a]))==c["area"]+adds-drops,"Exact exported native area")
                    counts[a+"_adds"] += adds
                    counts[a+"_drops"] += drops
                    if budget:
                        counts[a+"_net_shrink"] += int(drops>adds)
                    if rec["role"]=="valid":
                        g = helpers.geometry.candidate_geometry(rec,edit)
                        vals[a] = g["iou"]
                        strict[a] += int(g["raw_iou"]>.90 and g["iou"]<=.90)
                if vals:
                    groups[rec["group"]].append([vals[a] for a in ARMS])
            for a in ARMS:
                streams[a].append(dict(image_id=iid,category_id=1,score=c["score"],segmentation=masks[a]))
    need(len(keys)==lock["mapped"] and len(groups)==92 and sum(len(v) for v in groups.values())==392,
         "Exact original scoring and fixed geometry populations")
    means=100*np.array([np.mean(groups[g],axis=0) for g in sorted(groups)])
    arms = {}
    for ix,a in enumerate(ARMS):
        curve,_=coco_curve(gt,streams.pop(a),keys)
        arms[a]=dict(AP=float(np.mean(curve)),AP75=curve[5],AP90=curve[8],AP_thresholds=curve,
                     geometry=float(means[:,ix].mean()),strict_losses=strict[a],adds=counts[a+"_adds"],
                     drops=counts[a+"_drops"],net_shrink=counts[a+"_net_shrink"])
    rng = np.random.Generator(np.random.PCG64(5217))
    pick = rng.integers(len(means),size=(4096,len(means)))
    comparisons,checks = {},{}
    for b in ARMS[:-1]:
        delta=means[:,-1]-means[:,ARMS.index(b)]
        ap=arms["R"]["AP"]-arms[b]["AP"]
        center=float(delta.mean())
        ci=np.quantile(delta[pick].mean(axis=1),[.025,.975]).tolist()
        comparisons[b]=dict(AP_delta_points=ap,geometry_delta_points=center,geometry_ci95_points=ci)
        checks[b]=dict(ap=ap>=.30)
        if b in ("B","T"):
            checks[b].update(geometry=center>=.20,ci=ci[0]>0,AP75=arms["R"]["AP75"]>=arms[b]["AP75"],
                             strict=arms["R"]["strict_losses"]<=arms[b]["strict_losses"])
    return dict(arms=arms, R_minus=comparisons, checks=checks,
                advance=all(all(result.values()) for result in checks.values()))


def main():
    p=ArgumentParser(allow_abbrev=False)
    for n in ("work","code","run","owner","accepted","capture","carrier-readback","old-support","old-score","raw-capture",
              "raw-acceptance","source-cohort","fit-gt","shapely-site","protocol","output"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--owner-sha256",required=True)
    p.add_argument("--carrier-readback-sha",required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
         and not torch.cuda.is_initialized() and not a.output.exists(),"Specific isolated CPU run and fresh output")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started=perf_counter()
    folder=Path(__file__).parent
    def script(path,digest):
        raw=path.read_bytes()
        need(sha(raw)==digest,"Verified ordinary source "+path.name)
        env={"__file__":str(path),"__name__":"_ratio_inputs"}
        exec(compile(raw,str(path),"exec"),env)
        return SimpleNamespace(**env)
    inputs=script(a.work/"verify_point_fit109.py",VERIFY)
    load,read=inputs.checked_module,inputs.read_doc
    lock=read(a.protocol,PRELOCK)
    need(lock["FIT"] and not lock["DEV"] and not lock["TEST"],"Only FIT authorized")
    owner=inputs.checked_terminal(a.owner,a.owner_sha256)
    peer=read(a.accepted,PEER)
    result=read(a.run/"RESULT_SAFE.json",owner["result_sha256"])
    need(peer["result_sha256"]==owner["result_sha256"]==lock["source_result_sha256"] and peer["COCO12_bit_exact_all_six_arms"],
         "Original independently accepted reference")
    pin=read(a.work/"POINTREND_FIT109_MODULE_PINS.json",owner["module_manifest_sha256"])
    worker=load(a.work/"run_point_fit109.py",pin["run_point_fit109.py"])
    policy_path=folder/"ratio_writer.py"
    policy_sha=sha(policy_path.read_bytes())
    curve_path=folder/"analyze_pointrend_gap.py"
    curve_sha=sha(curve_path.read_bytes())
    shared=a.code/"RFDETR_POINTREND_SUBDIVISION_20261121"
    mods=SimpleNamespace(
       adapter=load(shared/"roi_point_head.py","672bdd2aed8c0056f29c816f829bae9ceae5f9a685309beebf107aa59e547274"),
       render=load(shared/"pointrend_subdivision.py","6c6be609317dc7960ba901b690e661d92588d88ee1f3210c9156c81a4d3219ad"),
       budget=load(shared/"matched_budget.py","4ca4207435c03cbd45567c4aff470f21b892c2d6889b5fad4b0f2b6e94316b7a"),
       infer=load(a.work/"infer_point_policies.py",pin["infer_point_policies.py"]), policy=load(policy_path,policy_sha))
    curve=load(curve_path,curve_sha).coco_curve
    orig=read(a.work/"POINTREND_FIT109_PRELOCK.json",worker.PROTOCOL_SHA)
    a.capture_readback,a.capture_readback_sha=a.carrier_readback,a.carrier_readback_sha
    _,index=worker.admit_capture(a,orig)
    former=a.code/"RFDETR_QUERY_FEATURE_FIT109_20261121"
    old=load(former/"run_fit109_query.py","97fdc3987462aabbcd7d4fbe672c9788a3b969a221243e13d82473093170a15a")
    spot=old.load(a.code/"RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/run_spot.py")
    formal=a.code/"RFDETR_MATCHED_POINT_FORMAL_20261120"
    prior,plan=(old.load(formal/n) for n in ("run_fit109.py","fit_plan.py"))
    captured=a.capture
    a.capture,a.acceptance,a.acceptance_sha256=a.raw_capture,a.raw_acceptance,old.OLD_RAW_ACCEPTANCE
    roster,_=prior.read_population(a,spot)
    helpers=old.load(former/"fit109_score.py").load_helpers(a.code,a.shapely_site)
    pools,rows=worker.collect_old_support(a,SimpleNamespace(old=old,spot=spot),roster,orig)
    budgets,folds=worker.read_budgets(a.old_score,orig,rows,plan,roster,helpers.geometry)
    saved={tuple(x["key"]):x for x in (json.loads(line) for line in inputs.pinned(a.run/"PREDICTION_WITNESS_PRIVATE.jsonl",
                   result["prediction_witness_sha256"],80*MIB).splitlines())}
    heads={}
    for fold in range(5):
        name=f"HEAD_FOLD_{fold}_PRIVATE.npz"
        raw=inputs.pinned(a.run/name,peer["fold_weight_file_sha256"][fold],2*MIB)
        with np.load(BytesIO(raw),allow_pickle=False) as z:
            head=mods.adapter.BinaryPointHead()
            head.load_state_dict({n:torch.from_numpy(z[n]) for n in z.files},strict=True)
        heads[fold]=head.eval()
    all_rows=defaultdict(list)
    for r in rows:
        all_rows[r["image_id"]].append(r)
    choices={}
    a.output.mkdir()
    for item,ordinal,_ in roster:
        iid=item["image_id"]
        spatial,values=worker.captured_arrays(captured,index[iid])
        by_rank={c["rank"]:c for c in pools[iid]}
        payload=[]
        for row in all_rows[iid]:
            key=helpers.geometry.identity(row)
            rank=row["rank"]
            qry,area,confidence,box,logit=values[rank]
            c=by_rank[rank]
            need(qry==row["query_id"] and area==c["area"] and confidence==c["score"],"Frozen candidate")
            payload.append((row,box,logit,c["rle"],budgets[key]["k"],saved[key]))
        choices.update(infer_one_image(heads[folds[item["group"]]],spatial,payload,mods))
        if ordinal%25==0 or ordinal==len(roster):
            print(json.dumps(dict(event="RATIO_PRELABEL_PREDICTIONS",images=ordinal)),flush=True)
    need(set(choices)==set(budgets)==set(saved) and len(choices)==lock["area_eligible"],"Complete prelabel properties")
    witness_sha=write(a.output/"EDITS_PRIVATE.json",[[list(key),v] for key,v in choices.items()])
    write(a.output/"PRELABEL_SAFE.json",dict(FIT=True,DEV=False,TEST=False,GT_opened=False,
         witness_sha256=witness_sha,protocol_sha256=PRELOCK,policy_sha256=policy_sha))
    gt,attached=old.attach_fit(a,roster,rows,pools,helpers,spot)
    metric=score(gt,pools,attached,budgets,saved,choices,helpers,curve,lock)
    for b in ("N","B","J"):
        need(abs(metric["arms"][b]["AP"]-100*result["metrics"]["arms"][b]["AP"])<1e-10,"Original matched scoring replay")
    sig=write(a.output/"RESULT_SAFE.json",dict(status="FIT109_MATCHED_IOU_WRITER_ACTUAL_COMPLETE",
         FIT=True,DEV=False,TEST=False,point_head_updates=0,model_forward_calls=0,protocol_sha256=PRELOCK,
         policy_sha256=policy_sha,source_sha256=sha(Path(__file__).read_bytes()),witness_sha256=witness_sha,
         original_benchmark_sha256=owner["result_sha256"],exact_old_balanced_and_joint_edits=True,
         candidate_count=len(choices),wall_seconds=perf_counter()-started,metrics=metric))
    print(json.dumps(dict(status="FIT109_MATCHED_IOU_WRITER_ACTUAL_COMPLETE",result_sha256=sig,advance=metric["advance"])),flush=True)


if __name__=="__main__":
    main()
