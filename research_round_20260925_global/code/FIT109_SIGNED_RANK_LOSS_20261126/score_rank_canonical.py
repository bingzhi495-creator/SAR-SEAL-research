"""Exact same-budget held-fold rendering and paired FIT109 scoring."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np


SEEDS = (5217, 5218, 5219)
ARMS = ("native", "legacy_f", "legacy_i", "bce", "bce_rank")
COMPARATORS = ("bce", "legacy_i")
EXPECTED_FIT_GT = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
LEGACY_RESULT = "114ac529e2196fa4f60d8fe56e07381fb6814e0fdb09e7d5615626fd74a85f64"
LEGACY_RECEIPT = "476b0f3fa748d1d3fd2393bdeb1af2d83e1532995d58b12625d57222e3f2638a"
LEGACY_AP = "ffd9c3d31dbbcc933b88454d1e27e8ae9b1f7afa1df115f9ee212ebaa4afdb49"
PAIR_ROOT = "code/DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_paired_point/"
MODULES = {
    "pair": (PAIR_ROOT+"pair_math.py","8fe7a2c146c08c89f377b1f20b321276a60356de4bdbb27be2a00544dfc77962"),
    "trainer": (PAIR_ROOT+"train_point.py","3111c34afd791cb15987ed94c5e898de4c3ff2cad61677b616420651335d812c"),
    "step": ("code/RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/fit_step.py",
             "94e60f5b5557787687d1f228c33c249a1d7889172a225373b97bd789ec8b72b1"),
    "plan": ("code/RFDETR_MATCHED_POINT_FORMAL_20261120/fit_plan.py",
             "1fd0fe8f815b5ad4e01d0d8eeca0aa0e1480433292fc48f94fee17e5837e5c35"),
    "sparse": ("code/RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py",
               "cb9b8e839de85588fdf116ce5554ed6cadbc20bdbe76e38180561528bfd870d5"),
    "sibling": ("code/HRSID_FIT109_SIBLING_ENDPOINT_RANK_20261125/pilot.py",
                "dd35a407d09786af0f6aa3cc61fe86f25edc9a3c3af7a212a9310639720b6960"),
    "annotation": ("code/RFDETR_FIT109_QUOTA_COMPETITION_TARGETS_20261123/assemble_competition.py",
                   "3b0be851e38028248d5b646a108598585b6a261989496bf535d5ea8b54e6dd5b"),
    "oracle": ("code/HRSID_FIT109_NESTED_SUPPORT_ORACLE_20261125/oracle.py",
               "10b1b0fde04d207bcbd220526b627fa7924158675707fdbfb7423e94b53ea8b2"),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Unexpected bound input: " + path.name)
    return raw


def read_json(path, expected):
    return json.loads(pin(path, expected))


def module(round_root, name):
    path, fingerprint = MODULES[name]
    path = round_root / path
    loaded = ModuleType("_signed_rank_" + name)
    loaded.__file__ = str(path)
    sys.modules[loaded.__name__] = loaded
    exec(compile(pin(path, fingerprint), str(path), "exec"), loaded.__dict__)
    return loaded


def fixed_signed_slots(q, k):
    n = len(q)
    need(q.shape == (n,2) and q.dtype == np.float32 and np.isfinite(q).all()
         and np.all((q >= 0) & (q <= 1)) and 0 <= k <= min(n,8), "Exact FP32 signed action")
    order = np.arange(n)
    return np.column_stack((np.lexsort((order,-q[:,0]))[:k],
                            np.lexsort((order,q[:,1]))[:k])).astype(np.int32)


def conditional_geometry(intersection, mask_area, truth_area, truth, edits):
    if len(edits):
        intersection += float(truth[edits[:,0],0].sum() - truth[edits[:,1],1].sum())
    union = mask_area + truth_area - intersection
    need(union > 0 and -1e-7 <= intersection <= min(mask_area,truth_area)+1e-7,
         "Finite continuous physical intersection")
    return intersection / union, (mask_area-intersection)/truth_area, (truth_area-intersection)/truth_area


def conditional_iou(intersection, mask_area, truth_area, truth, edits):
    return conditional_geometry(intersection, mask_area, truth_area, truth, edits)[0]


def bootstrap(delta):
    delta = np.asarray(delta,dtype=np.float64)
    pick = np.random.Generator(np.random.PCG64(5217)).integers(len(delta),size=(4096,len(delta)))
    return np.percentile(delta[pick].mean(axis=1),[2.5,97.5]).tolist()


def closed(directory, terminal, result_sha, terminal_sha, put="actual"):
    result = read_json(directory/put/"RESULT_SAFE.json",result_sha)
    receipt = read_json(directory/terminal/"TERMINAL.json",terminal_sha)
    need(receipt["actual_exit"] == 0 and receipt["output_footer_verified"] is True
         and receipt["result_sha256"] == result_sha
         and receipt["worker_sha256"] == result["worker_sha256"], "Observed natural closure and result footer")
    return result


def safe_ref(root, relative, expected):
    rel = Path(relative)
    need(not rel.is_absolute() and ".." not in rel.parts,"Bound relative asset")
    path = (root/rel).resolve(strict=True)
    need(path.is_relative_to(root.resolve(strict=True)),"Asset remains in frozen phase root")
    return pin(path,expected)


def source_inputs(a):
    source = closed(a.source,"observed_source",a.source_sha,a.source_terminal_sha)
    need(source["status"] == "RFDETR_FIT109_SIGNED_RANK_SOURCE_COMPLETE"
         and source["author_fit_gt_sha256"] == EXPECTED_FIT_GT, "Qualified exact FIT-only source")
    stock = pin(a.source/"actual/STOCK_PRIVATE.jsonl",source["stock_sha256"])
    pages = [json.loads(line) for line in stock.splitlines()]
    candidates = [row for page in pages for row in page["candidates"]]
    infos = {p["image"]["image_id"]:(p["image"]["group"],p["fold"]) for p in pages}
    need(len(pages) == len(infos) == len({r[0] for r in infos.values()}) == 109
         and len(candidates) == 10488 and
         all(row["image_id"] == page["image"]["image_id"] and row["category_id"] == 1
             for page in pages for row in page["candidates"]), "Exact FIT stock exposure")
    data = np.load(BytesIO(pin(a.source/"actual/POINTS_PRIVATE.npz",source["points_sha256"])),allow_pickle=False)
    return source, candidates, infos, data


def members(data, candidates, infos, plan):
    names = ("candidate_index","image_id","group","fold","rank","query_id","offset","coordinates","features",
             "native_area","archived_seeds","archived_f_k")
    m = {name:data[name] for name in names}
    count = len(m["candidate_index"])
    index, offsets, xy, feats = (m[k] for k in ("candidate_index","offset","coordinates","features"))
    need(count == 6421 and index.dtype.kind in "iu" and np.all(np.diff(index)>0)
         and index[0] >= 0 and index[-1] < len(candidates)
         and offsets.shape == (count+1,) and offsets[0] == 0 and np.all(np.diff(offsets)>0)
         and xy.shape == (offsets[-1],2) and xy.dtype.kind in "iu"
         and feats.shape == (len(xy),2,14) and feats.dtype == np.float32 and np.isfinite(feats).all()
         and m["archived_seeds"].tolist() == list(SEEDS) and m["archived_f_k"].shape == (count,3),
         "Exact ordered stock, paired endpoint and legacy quotas")
    assignments = plan.source_folds([group for group,_ in infos.values()])
    for j, i in enumerate(index):
        c = candidates[int(i)]
        group, fold = infos[c["image_id"]]
        need((c["image_id"],c["rank"],c["query_id"],c["area"]) ==
             (m["image_id"][j],m["rank"][j],m["query_id"][j],m["native_area"][j])
             and group == m["group"][j] and fold == m["fold"][j] == assignments[group],
             "Canonical image/group/fold and identity join")
    return m


def restore_small(path_bytes, trainer, step):
    torch = step.torch
    model = trainer.PointHead(14)
    with np.load(BytesIO(path_bytes),allow_pickle=False) as z:
        center,scale = z["center"],z["scale"]
        state = model.state_dict()
        need(set(z.files) == {"center","scale"}|{"S."+k for k in state} and
             all(v.shape == (14,) and v.dtype == np.float32 and np.isfinite(v).all() for v in (center,scale))
             and np.all(scale>0), "Exact new S checkpoint/scaler")
        for k,v in state.items():
            x = z["S."+k]
            need(x.shape == tuple(v.shape) and x.dtype == np.float32 and np.isfinite(x).all(),
                 "Exact S state")
            state[k] = torch.from_numpy(x)
        model.load_state_dict(state,strict=True)
    return model.eval(),center,scale


def render(a):
    pair,trainer,step,plan,sparse = [module(a.public_round,k) for k in ("pair","trainer","step","plan","sparse")]
    torch = step.torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source,candidates,infos,data = source_inputs(a)
    m = members(data,candidates,infos,plan)
    fit = closed(a.fit,"observed_fit",a.fit_sha,a.fit_terminal_sha)
    need(fit["status"] == "FIT109_SIGNED_RANK_LOSS_FIT_COMPLETE" and
         (fit["prelock_commit"],fit["protocol_sha256"],fit["source_result_sha256"],
          fit["source_terminal_sha256"],fit["source_points_sha256"]) ==
         (a.prelock_commit,a.protocol_sha,a.source_sha,a.source_terminal_sha,source["points_sha256"])
         and fit["equal_initialization_and_rows"] and fit["group_fold_train_only_standardization"]
         and fit["optimizer_updates"] == 6000 and fit["n_parameters_per_arm"] == 1321
         and fit["updates_per_arm_fold_seed"] == 200 and not fit["additional_refits"]
         and fit["rank_weight"] == .25 and fit["adamw_learning_rate"] == .002
         and fit["adamw_weight_decay"] == .01,
         "Matched two-arm held tensor closure under the same source and lock")
    fit_root = a.fit/"actual"
    head_rows = {(r["seed"],r["fold"],r["arm"]):r for r in fit["weights"]}
    pred_rows = {r["seed"]:r for r in fit["held_predictions"]}
    need(len(fit["weights"]) == len(head_rows) == 30 and
         set(head_rows) == {(s,f,t) for s in SEEDS for f in range(5) for t in ARMS[3:]}
         and set(pred_rows) == set(SEEDS), "Every seed/fold/arm is independently represented")
    old = read_json(a.legacy/"RESULT_SAFE.json",LEGACY_RESULT)
    old_closed = read_json(a.legacy/"INDEPENDENT_SAFE.json",LEGACY_RECEIPT)
    need(old_closed["fit_result_sha256"] == LEGACY_RESULT and old_closed["run_natural_exit"] == 0
         and old_closed["checkpoint_bundles_verified"] == 18, "Exact old completed head acceptance")
    orig = {(r["seed"],r["fold"]):r for r in old["all_initializations"] if r["fold"] is not None}
    need(len(orig)==15, "Every original legacy held-fold bundle is available")

    output = np.full((3,len(m["candidate_index"]),4,8,2),-1,dtype=np.int32)
    k_all = np.zeros((3,len(m["candidate_index"])),dtype=np.int8)
    for si, seed in enumerate(SEEDS):
        counts = pred_rows[seed]
        with np.load(BytesIO(safe_ref(fit_root,counts["relative_predictions"],counts["sha256"])),
                     allow_pickle=False) as z:
            held = {name:z[name] for name in ARMS[3:]}
        need(all(q.shape == m["features"].shape[:2] and q.dtype==np.float32 and np.isfinite(q).all()
                 for q in held.values()), "Exact held endpoint arrays")
        models = {}
        for f in range(5):
            entry = orig[seed,f]
            heads,center,scale = plan.restore_bundle(a.legacy/"POINT_HEADS_PRIVATE"/entry["relative_weights"],
                                                    entry["weights_sha256"],trainer,step)
            models[f,"legacy"] = (heads["S"],center,scale)
            b,r = (head_rows[seed,f,n] for n in ARMS[3:])
            need(b["initial_sha256"] == r["initial_sha256"] and b["common_batch_sha256"] == r["common_batch_sha256"]
                 and b["scaler_sha256"] == r["scaler_sha256"]
                 and b["steps"] == r["steps"] == 200, "Matched initial state, schedule and optimizer budget")
            for name,entry in zip(ARMS[3:],(b,r),strict=True):
                models[f,name] = restore_small(safe_ref(fit_root,entry["relative_weights"],entry["weights_sha256"]),
                                               trainer,step)
        for j, pos in enumerate(m["candidate_index"]):
            f = int(m["fold"][j])
            sl = slice(m["offset"][j],m["offset"][j+1])
            xy, feats = m["coordinates"][sl],m["features"][sl]
            need(len(xy) <= 32 and np.unique(xy).size == 2*len(xy) and
                 np.all((xy >= 0) & (xy < 800*800)), "Exact bounded unique signed endpoint pool")
            probs=step.predict(models[f,"legacy"][0],feats,*models[f,"legacy"][1:])
            cap=min(8,int(m["native_area"][j])//10,len(xy))
            selected=pair.choose_pairs(probs[:,0]-probs[:,1],cap)
            k = len(selected)
            need(k == int(m["archived_f_k"][j,si]), "Independent exact legacy F quota readback")
            k_all[si,j] = k
            need(all(np.array_equal(step.predict(models[f,t][0],feats,*models[f,t][1:]),held[t][sl])
                     for t in ARMS[3:]), "Independent exact held fold posterior readback")
            if k:
                original = candidates[int(pos)]
                rle = original["rle"]
                need(rle["size"] == [800,800], "Original native RLE")
                runs = np.cumsum(sparse.decode_runs(rle["counts"]))
                linear = (xy%800)*800+xy//800
                signs = np.searchsorted(runs,linear,side="right")%2
                need(np.all(signs[:,0]==0) and np.all(signs[:,1]==1), "Signed support matches actual frozen mask")
                choices = [np.tile(np.asarray(selected)[:,None],(1,2)),fixed_signed_slots(probs,k)]
                choices.extend(fixed_signed_slots(held[t][sl],k) for t in ARMS[3:])
                for ai, choice in enumerate(choices):
                    output[si,j,ai,:k] = xy[choice,np.arange(2)]
        expected_count,expected_k = ((6310,46581),(6325,46888),(6333,46925))[si]
        need(np.count_nonzero(k_all[si])==expected_count and k_all[si].sum()==expected_k,
             "Exact accepted legacy active-pool and per-side edit count")
    result = dict(status="FIT109_SIGNED_RANK_HELD_EDITS_PREPARED" if a.phase=="render" else
                  "FIT109_SIGNED_RANK_HELD_EDITS_VERIFIED",candidate_rows=len(candidates),
                  complete_source_images=len(infos),support_rows=len(m["candidate_index"]),
                  archived_counts_unchanged=True,held_weights_reconstructed_per_candidate=True,
                  optimizer_updates=0,label_geometry_opened=False,
                  per_seed_budget={str(s):dict(positive_candidates=int(np.count_nonzero(k)),
                                               exact_adds=int(k.sum()),exact_drops=int(k.sum()))
                                   for s,k in zip(SEEDS,k_all,strict=True)})
    stream = BytesIO()
    np.savez_compressed(stream,edits=output,k_all=k_all)
    raw = stream.getvalue()
    if a.phase=="verify":
        prepared=closed(a.render_root,"observed_render",a.render_sha,a.render_terminal_sha)
        need(prepared["status"]=="FIT109_SIGNED_RANK_HELD_EDITS_PREPARED" and
             prepared["protocol_sha256"]==a.protocol_sha and
             prepared["source_result_sha256"]==a.source_sha and prepared["fit_result_sha256"]==a.fit_sha and
             pin(a.render_root/"actual/EDITS_PRIVATE.npz",prepared["edits_sha256"])==raw,
             "Independent whole edit ledger byte-for-byte replay")
        result.update(render_result_sha256=a.render_sha,render_terminal_sha256=a.render_terminal_sha)
    a.output.mkdir(exist_ok=False)
    (a.output/"EDITS_PRIVATE.npz").write_bytes(raw)
    result["edits_sha256"]=digest(raw)
    release(a,result)


def to_group_matrix(values):
    groups = sorted(values)
    return groups,np.asarray([np.asarray(values[g],dtype=np.float64).mean(axis=0)*100 for g in groups])


def score(a):
    verified = closed(a.verify_root,"observed_verify",a.verify_sha,a.verify_terminal_sha)
    need(verified["status"]=="FIT109_SIGNED_RANK_HELD_EDITS_VERIFIED" and
         verified["protocol_sha256"]==a.protocol_sha and verified["source_result_sha256"]==a.source_sha and
         verified["fit_result_sha256"]==a.fit_sha, "Exact independent terminal-bound held renders")
    plan,sparse,sibling,oracle = [module(a.public_round,k) for k in ("plan","sparse","sibling","oracle")]
    source,candidates,infos,data=source_inputs(a)
    m=members(data,candidates,infos,plan)
    with np.load(BytesIO(pin(a.verify_root/"actual/EDITS_PRIVATE.npz",verified["edits_sha256"])),
                 allow_pickle=False) as z:
        all_edits,k_all=z["edits"],z["k_all"]
    need(all_edits.shape==(3,6421,4,8,2) and k_all.shape==(3,6421) and
         np.array_equal(k_all,m["archived_f_k"].T),"Exact seal for all policies")
    annotation=module(a.public_round,"annotation")
    need(annotation.ORIGINAL_GT==EXPECTED_FIT_GT,"Canonical historical author FIT annotation hash")
    gt=annotation.annotations(a,infos)
    zero_original_gt_id=max(x["id"] for x in gt["annotations"])
    originals={iid:[] for iid in infos}
    local_to_global={}
    global_to_local=[]
    for ix,c in enumerate(candidates):
        iid=c["image_id"]
        pos=len(originals[iid])
        local_to_global[iid,pos]=ix
        global_to_local.append((iid,pos))
        originals[iid].append(dict(image_id=iid,category_id=1,score=c["score"],segmentation=c["rle"]))
    coco,matches=oracle.fixed_coco(gt,originals)
    need(len(matches)==453,"Exact frozen stock .50 matches across the 496-hull denominator")
    idx={int(c):j for j,c in enumerate(m["candidate_index"])}
    roles=data["role"]
    truth=data["truth"]
    truth_area=data["truth_area"]
    native_intersection=data["native_intersection"]
    matched_gt=data["matched_gt_id"]
    need(roles.shape==(6421,) and truth.dtype==np.float64 and truth.shape==m["coordinates"].shape
         and sum(roles==2)==392, "Exact eligible continuous-label anchor population")
    old=read_json(a.public_round/"results/HRSID_FIT109_CROSS_DETECTOR_ENDPOINT_SAFE_20261125.json",LEGACY_AP)
    per_seed={}
    group_tables={"all_hull_discrete":[],"fixed_match_continuous":[],
                  "fixed_excess_area":[],"fixed_missed_area":[]}
    active_arms=ARMS[:3] if a.phase=="legacy" else ARMS
    for si,seed in enumerate(SEEDS[:1] if a.phase=="legacy" else SEEDS):
        worlds={"native":originals,**{t:deepcopy(originals) for t in active_arms[1:]}}
        signed_indices={}
        changed_candidate={t:0 for t in COMPARATORS}
        for j,i in enumerate(m["candidate_index"]):
            i=int(i)
            c=candidates[i]
            iid,pos=global_to_local[i]
            k=int(k_all[si,j])
            sl=slice(m["offset"][j],m["offset"][j+1])
            xy=m["coordinates"][sl]
            for ai,t in enumerate(active_arms[1:]):
                flips=all_edits[si,j,ai,:k]
                need(np.all(all_edits[si,j,ai,k:]==-1) and len(np.unique(flips))==2*k,
                     "Exact signed ledger and no substitutions")
                slot_indices=np.column_stack([np.array([lookup[int(v)] for v in flips[:,col]])
                                              for col in range(2)
                                              for lookup in ({int(v):ix for ix,v in enumerate(xy[:,col])},)])
                arr=slot_indices.astype(np.int32).reshape(k,2)
                if roles[j]==2:
                    signed_indices[j,t]=arr
                rle=sparse.xor_rle(c["rle"],flips.ravel())
                need(int(sibling.codec.area(rle))==int(m["native_area"][j]),"Every edit preserves actual mask area")
                worlds[t][iid][pos]["segmentation"]=rle
            for t in (() if a.phase=="legacy" else COMPARATORS):
                changed_candidate[t] += worlds["bce_rank"][iid][pos]["segmentation"] != worlds[t][iid][pos]["segmentation"]
        metrics={t:sibling.full_metrics(gt,worlds[t],originals) for t in active_arms[:3]}
        group_raster,group_cont=defaultdict(list),defaultdict(list)
        changes={t:set() for t in COMPARATORS}
        change_groups={t:set() for t in COMPARATORS}
        for ann in gt["annotations"]:
            aid,iid=ann["id"],ann["image_id"]
            group=infos[iid][0]
            if aid not in matches:
                group_raster[group].append([0.] * len(active_arms))
                continue
            im,pos=matches[aid]
            need(im==iid,"Fixed native match remains in the source image")
            target=coco.annToRLE(ann)
            rs=[worlds[t][iid][pos]["segmentation"] for t in active_arms]
            values=sibling.codec.iou(rs,[target],[0]).reshape(-1)
            need(values[0] >= .50-1e-12,"Original fixed raster match stays present")
            group_raster[group].append(values.tolist())
            j=idx.get(local_to_global[iid,pos])
            if j is None or roles[j]!=2:
                continue
            need(int(matched_gt[j])==(0 if aid==zero_original_gt_id else aid), "Fractional geometry anchor uses the original stock identity")
            sl=slice(m["offset"][j],m["offset"][j+1])
            ys=truth[sl]
            need(np.isfinite(ys).all() and np.all((ys>=-1e-10)&(ys<=1+1e-10)),
                 "Original finite fractional-occupancy truth")
            change_cont = [conditional_geometry(native_intersection[j],m["native_area"][j],truth_area[j],
                          ys,np.empty((0,2),np.int32) if t=="native" else signed_indices[j,t]) for t in active_arms]
            group_cont[group].append(change_cont)
            for t in (() if a.phase=="legacy" else COMPARATORS):
                if k_all[si,j] > 0 and rs[-1] != rs[ARMS.index(t)]:
                    changes[t].add(aid)
                    change_groups[t].add(group)
        need(len(group_raster)==109 and sum(len(x) for x in group_raster.values())==496 and
             len(group_cont)==92 and sum(len(x) for x in group_cont.values())==392,"Exact two estimand denominators")
        asm, raster=to_group_matrix(group_raster)
        bsm, cont=to_group_matrix(group_cont)
        if si:
            need(raster_order==asm and continuous_order==bsm,"Same paired prefix axes across seeds")
        raster_order,continuous_order=asm,bsm
        for dest,table in zip(group_tables,(raster,cont[:,:,0],cont[:,:,1],cont[:,:,2]),strict=True):
            group_tables[dest].append(table)
        if seed==5217:
            refs=(("native","native"),("legacy_f","distance"),("legacy_i","independent"))
            ap_parity={t:metrics[t]==old["full_coco_points"][b] for t,b in refs}
            discrete_parity={t:abs(float(raster[:,ARMS.index(t)].mean())-
                         old["fixed_all_hull_geometry"][b]["parent_discrete_iou_points"])<1e-10 for t,b in refs}
            if a.phase=="legacy":
                ap_deltas={t:{m:(metrics[t][m]-old["full_coco_points"][b][m])
                              for m in metrics[t] if metrics[t][m] != old["full_coco_points"][b][m]
                              and metrics[t][m] is not None and old["full_coco_points"][b][m] is not None}
                           for t,b in refs}
                raster_deltas={t:float(raster[:,ARMS.index(t)].mean())-
                         old["fixed_all_hull_geometry"][b]["parent_discrete_iou_points"] for t,b in refs}
                a.output.mkdir(exist_ok=False)
                release(a,dict(status="FIT109_SIGNED_RANK_LEGACY_DIAGNOSTIC_COMPLETE",coco_metric_parity=ap_parity,
                               discrete_parity=discrete_parity,legacy_ap_deltas=ap_deltas,
                               legacy_discrete_iou_deltas=raster_deltas,observed_arm_scope=list(active_arms)))
                return
            need(all(ap_parity.values()) and all(discrete_parity.values()),
                 "Exact seed5217 published legacy N/F/I COCO metrics and fixed-group discrete geometry")
        metrics.update({t:sibling.full_metrics(gt,worlds[t],originals) for t in active_arms[3:]})
        per_seed[str(seed)]=dict(coco_points=metrics,
                  prefix_equal_discrete_points={t:float(raster[:,ix].mean()) for ix,t in enumerate(ARMS)},
                  prefix_equal_continuous_points={t:float(cont[:,ix,0].mean()) for ix,t in enumerate(ARMS)},
                  prefix_equal_excess_per_truth_area_points={t:float(cont[:,ix,1].mean()) for ix,t in enumerate(ARMS)},
                  prefix_equal_missed_per_truth_area_points={t:float(cont[:,ix,2].mean()) for ix,t in enumerate(ARMS)},
                  changed_candidates=changed_candidate,
                  changed_continuous_anchors={t:dict(matches=len(changes[t]),prefixes=len(change_groups[t]))
                                               for t in COMPARATORS})

    comparisons={}
    directional={}
    for t in COMPARATORS:
        ix=ARMS.index(t)
        deltas={str(s):per_seed[str(s)]["coco_points"]["bce_rank"]["AP"]-
                         per_seed[str(s)]["coco_points"][t]["AP"] for s in SEEDS}
        geo={}
        for estimand,table in group_tables.items():
            sums=np.asarray(table)
            d=sums[:,:,-1]-sums[:,:,ix]
            center=d.mean(axis=0)
            geo[estimand]=dict(delta_points=float(center.mean()),prefix_bootstrap_ci95=bootstrap(center),
                                per_seed_delta_points={str(s):float(b.mean()) for s,b in zip(SEEDS,d,strict=True)},
                                per_seed_prefix_ci95={str(s):bootstrap(b) for s,b in zip(SEEDS,d,strict=True)})
        ap_mean=float(np.mean(list(deltas.values())))
        coverage=all(m["changed_continuous_anchors"][t]["matches"] >= 20 and
                     m["changed_continuous_anchors"][t]["prefixes"] >= 8 for m in per_seed.values())
        lower,upper=geo["fixed_match_continuous"]["prefix_bootstrap_ci95"]
        positive=coverage and ap_mean >= .30 and lower > 0 and all(d > 0 for d in deltas.values())
        negative=coverage and ap_mean <= -.30 and upper < 0 and all(d < 0 for d in deltas.values())
        directional[t]=dict(coverage=coverage,positive=positive,negative=negative,
                            decision="positive" if positive else ("negative" if negative else "inconclusive"))
        comparisons[t]=dict(mask_ap_delta_points=ap_mean,mask_ap_deltas_by_seed=deltas,geometry=geo)
    a.output.mkdir(exist_ok=False)
    release(a,dict(status="FIT109_SIGNED_RANK_HELD_SCORE_COMPLETE",images=109,candidates=10488,
                   hulls=496,frozen_native_matches=453,frozen_unmatched_hulls=43,continuous_anchors=392,
                   continuous_prefixes=92,all_hull_prefixes=109,paired_4096_group_bootstrap=True,
                   legacy_seed5217_nfi_ap_and_discrete_geometry_verified=True,all_source_detector_fit_exposure=True,
                   point_head_groups_held_out=True,author_fit_only=True,other_data_opened=False,
                   verification_result_sha256=a.verify_sha,verification_terminal_sha256=a.verify_terminal_sha,
                   per_seed_budget=verified["per_seed_budget"],per_seed=per_seed,
                   contrasts_rank_minus=comparisons,directional_diagnostic=directional,
                   advance=all(x["positive"] for x in directional.values())))


def release(a,result):
    result.update(worker_sha256=digest(Path(__file__).read_bytes()),protocol_sha256=a.protocol_sha,
                  prelock_commit=a.prelock_commit,source_result_sha256=a.source_sha,
                  source_terminal_sha256=a.source_terminal_sha,fit_result_sha256=a.fit_sha,
                  fit_terminal_sha256=a.fit_terminal_sha)
    raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    (a.output/"RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=result["status"],result_sha256=digest(raw))),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--phase",required=True,choices=("render","verify","score","legacy"))
    for n in ("source","fit","legacy","public-round","protocol","output","coco-site"):
        p.add_argument("--"+n,type=Path,required=True)
    for n in ("protocol-sha","prelock-commit","source-sha","source-terminal-sha","fit-sha","fit-terminal-sha"):
        p.add_argument("--"+n,required=True)
    for n in ("render-root","verify-root","fit-gt"):
        p.add_argument("--"+n,type=Path)
    for n in ("render-sha","render-terminal-sha","verify-sha","verify-terminal-sha"):
        p.add_argument("--"+n)
    a=p.parse_args()
    need(len(a.prelock_commit)==40 and a.output.parent.is_dir() and not a.output.exists(),
         "Published prelock witness and fresh isolated output")
    pin(a.protocol,a.protocol_sha)
    sys.path.append(str(a.coco_site))
    (score if a.phase in ("score","legacy") else render)(a)

