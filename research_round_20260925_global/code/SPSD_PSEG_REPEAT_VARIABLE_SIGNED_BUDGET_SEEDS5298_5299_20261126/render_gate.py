"""Actual label-free same-seed fixed and variable split replay for repeats 5298 and 5299."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType

import torch
import numpy as np
from pycocotools import mask as codec


LOCK = "3b51839c19c0ff03f2371d45be7c9d84e161531f8a70962d1c85a4472c93801a"
CONSTRAINT = "fb49070f10d064fbe1bcce89e7baf1fb59d3ae81950863437adb95c37c7c4b81"
ARCHIVE_LOADER = "25a0837e5147e95421483d73b6e3146e28f8f95059c922e21514dc9e36f3a0e3"
GROUP = "4030a3cb5684a6dac0ea1e7765a48d05caec113c5c6e0e6de2c82b991e9d5f15"
STATUS = "SPSD_PSEG_REPEAT_VARIABLE_SIGNED_BUDGET_GATE_RENDER_COMPLETE"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw)==expected,"Exact artifact "+path.name)
    return raw


def load(path, expected):
    m = ModuleType(path.stem)
    m.__file__ = str(path)
    exec(compile(pin(path,expected),str(path),"exec"),m.__dict__)
    return m


def repeat_fitted(a,old,root,roots):
    pin(a.fit/"fit_repeat_bce.py",a.fit_worker_sha)
    pin(a.fit/"PRELOCK.md",a.fit_lock_sha)
    term=root.terminal(a.fit/"observed_fit/TERMINAL.json","prepare-edits",a.fit_sha,a.fit_worker_sha)
    doc=json.loads(pin(a.fit/"actual/RESULT_SAFE.json",a.fit_sha))
    back=json.loads(pin(a.fit/"READBACK_SAFE.json",a.fit_readback_sha))
    need(doc["status"]=="SPSD_PSEG_BCE_REPEAT_SEEDS_FIT_COMPLETE"
         and doc["protocol_sha256"]==a.fit_lock_sha and doc["worker_sha256"]==a.fit_worker_sha
         and doc["source_fit_result_sha256"]==roots.ARCH_FIT and doc["archive_5297_bce_tensor_sha256"]==roots.ARCH_BCE
         and doc["archive_5297_used_as_historical_anchor"] is True and doc["current_5297_parity"] is True
         and doc["seeds"]==[5298,5299] and set(doc["train_audit"])=={"5298","5299"}
         and doc["source_gate_labels_opened"] is doc["source_gate_points_opened"] is doc["historical_test_opened"] is False
         and doc["detector_forwards"]==0 and doc["total_optimizer_updates"]==2000 and doc["verification_optimizer_updates"]==1000,
         "Exact naturally closed matched BCE repeat fit scope")
    need(back["status"]=="SPSD_PSEG_BCE_REPEAT_SEEDS_READBACK_COMPLETE"
         and back["fit_result_sha256"]==a.fit_sha and back["terminal_sha256"]==term
         and back["protocol_sha256"]==a.fit_lock_sha and back["worker_sha256"]==a.fit_worker_sha
         and back["completed"] is True and back["natural_exit"]==0
         and back["input_sha256"]==doc["input_receipt_sha256"]
         and back["checkpoint_sha256"]==doc["output_checkpoint_sha256"],
         "Independent matched repeat checkpoint readback")
    pin(a.fit/"actual/INPUT_SAFE.json",doc["input_receipt_sha256"])
    checkpoint=torch.load(BytesIO(pin(a.fit/"actual/FITTED_WEIGHTS_PRIVATE.pt",doc["output_checkpoint_sha256"])),
                          weights_only=True,map_location="cpu")
    need(checkpoint["seeds"]==[5298,5299] and set(checkpoint["models"])=={"5298","5299"},
         "Exact two unselected predeclared fit seeds")
    core=load(a.core,old.CORE)
    seed=str(a.seed)
    state=core.PointHead()
    state.load_state_dict(checkpoint["models"][seed],strict=True)
    tensors=np.concatenate([p.detach().numpy().ravel() for p in state.parameters()]).astype("<f4",copy=False)
    tensor_sha=digest(tensors.tobytes())
    audit=doc["train_audit"][seed]
    center,scale=(checkpoint[key].numpy() for key in ("center","scale"))
    need(audit["final_tensor_sha256"]==back["final_tensor_sha256"][seed]==tensor_sha and audit["parameters"]==1346 and audit["optimizer_updates"]==1000
         and audit["seed"]==a.seed and np.isfinite(tensors).all() and center.shape==scale.shape==(14,)
         and np.isfinite(center).all() and np.isfinite(scale).all() and np.all(scale>0),
         "Exactly matched per-seed BCE tensors, original features and FIT-only moments")
    return core,state.eval(),center,scale,tensor_sha,term,doc


def render(a):
    start = perf_counter()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1", "CPU-only label-free writer")
    pin(a.protocol,LOCK)
    pin(a.group_protocol,GROUP)
    policy = load(a.variable_writer,CONSTRAINT)
    roots = load(a.archive_loader,ARCHIVE_LOADER)
    old,root,_,_,base,archived,groups,_,arch_term=roots.archive(a)
    torch.set_num_threads(1)
    core,model,center,scale,model_tensor,fit_term,fit=repeat_fitted(a,old,root,roots)
    selector=load(a.selector,old.SELECTOR)
    attraction=load(a.attraction,old.ATTRACTION)
    pt=json.loads(pin(a.points/"actual_retry/RESULT_SAFE.json",old.POINTS))
    point_terminal=root.terminal(a.points/"observed_points_retry/TERMINAL.json",
                                 "prepare-edits",old.POINTS,old.POINT_WORKER)
    need(point_terminal==base["point_terminal_sha256"] and pt["train_labels_opened"] is pt["test_labels_opened"] is False,
         "Original closed label-free point preparation")
    point_root=a.points/"actual_retry/source_gate"
    point_index=pt["partitions"]["source_gate"]["point_index_sha256"]
    index=[json.loads(s) for s in pin(point_root/"POINT_INDEX_PRIVATE.jsonl",point_index).splitlines()]
    need(len(index)==len(archived)==183 and set(archived)=={r["image_id"] for r in index}
         and len(set(groups.values()))==180,"Exact label-free image and group scope")
    totals=Counter()
    arm_stats={arm:Counter() for arm in policy.ARMS}
    areas={arm:Counter() for arm in policy.ARMS}
    row_digest=sha256()
    a.output.mkdir()
    with (a.output/"GATE_RENDER_PRIVATE.jsonl").open("xb") as stream:
        for item in index:
            iid=item["image_id"]
            previous=archived[iid]
            need(item["source_group"]==previous["source_group"]==groups[iid]
                 and item["original_hw"]==previous["original_hw"],"Exact groups and native geometry")
            raw=pin(point_root/item["filename"],item["sha256"])
            need(len(raw)==item["bytes"],"Exact point packet length")
            with np.load(BytesIO(raw),allow_pickle=False) as f:
                ranks,offset,coords,features,quotas,booked=(f[k] for k in ("native_ranks","point_offsets",
                     "signed_coordinates","features","static_offsets","static_side_indices"))
            n=len(ranks)
            need(n==item["supported_candidates"]==len(set(ranks)) and len(coords)==item["paired_sites"]
                 and len(booked)==item["realized_static_exchanges"] and quotas.shape==offset.shape==(n+1,)
                 and np.all((1<=ranks)&(ranks<=100)) and features.shape==(len(coords),2,14)
                 and offset[0]==quotas[0]==0 and offset[-1]==len(coords) and quotas[-1]==len(booked),
                 "Exact original point geometry and range of native ranks")
            all_p=core.predict(model,features,center,scale)
            candidates=[dict(p,changes=dict(p["changes"])) for p in previous["candidates"]]
            processed=set()
            for i,r in enumerate(ranks):
                rank=int(r)
                p=candidates[rank-1]
                k=int(quotas[i+1]-quotas[i])
                need(k==p["quota"],"Exact original quota on every supported candidate")
                if not k:
                    continue
                processed.add(rank)
                span=slice(offset[i],offset[i+1])
                sub,yy=all_p[span],coords[span]
                incumbent_array=booked[quotas[i]:quotas[i+1]]
                incumbent=root.actions(incumbent_array)
                legacy=selector.select(sub[:,0,:],sub[:,1,:],*incumbent,k=k)
                native=codec.decode(p["native"]).astype(bool,order="C")
                area=int(native.sum())
                dynamic,static=attraction.select(native,yy,min(8,area//10,len(yy)))
                need(len(dynamic)==len(static)==k and np.array_equal(static,incumbent_array)
                     and root.edited_mask(p["native"],native,yy,root.actions(dynamic),k)==p["changes"]["dynamic"],
                     "Exact older dynamic and static budget")
                need(root.edited_mask(p["native"],native,yy,legacy["static"],k)==p["changes"]["static"],
                     "Exact frozen static RLE")
                for branch in ("robust","mean","cell_pessimistic"):
                    p["changes"]["same_bce_"+branch]=root.edited_mask(p["native"],native,yy,legacy[branch],k)
                decisions=policy.select(sub,incumbent,k,area)
                p["variable_signed_counts"]={}
                p["variable_predicted_iou_gain"]={}
                p["variable_predicted_intersection_gain"]={}
                p["variable_menu_actions"]=decisions["menu_actions"]
                cache={}
                for arm in policy.ARMS:
                    action=decisions[arm]["action"]
                    if action not in cache:
                        cache[action]=policy.write_mask(native,yy,action,k,codec)
                    encoded,(adds,drops)=cache[action]
                    need(adds+drops==2*k and int(codec.area(encoded))==area+adds-drops,
                         "Exact every writer pixel quota and remaining area")
                    p["changes"][arm]=encoded
                    p["variable_signed_counts"][arm]=[adds,drops]
                    p["variable_predicted_iou_gain"][arm]=decisions[arm]["iou_gain"]
                    p["variable_predicted_intersection_gain"][arm]=decisions[arm]["intersection_gain"]
                    arm_stats[arm].update(adds=adds,drops=drops,total=adds+drops,candidates=1,
                       enlarged=int(adds>drops),shrunk=int(drops>adds),area_unchanged=int(adds==drops),
                       differs_from_static=int(encoded!=p["changes"]["static"]),
                       differs_from_same_seed_bce_robust=int(encoded!=p["changes"]["same_bce_robust"]))
                    areas[arm][str(adds-drops)]+=1
                need(p["variable_signed_counts"]["fixed_iou"]==[k,k]
                     and abs(p["variable_signed_counts"]["near_iou"][0]-k)<=1
                     and all(p["changes"][name]==p["changes"]["fixed_iou"]
                         for name in ("variable_iou","near_iou") if p["variable_signed_counts"][name]==[k,k]),
                     "Exact fixed and near controls and zero net-area tie parity")
                totals.update(positive=1,exchanges=k,enumerated_actions=decisions["menu_actions"])
            need(processed=={j for j,p in enumerate(candidates,1) if p["quota"]},
                 "All original positive quotas retained including no-allocation-ablation edge cases")
            line=(json.dumps(dict(previous,candidates=candidates),sort_keys=True,
                          separators=(",",":"),allow_nan=False)+"\n").encode()
            need(stream.write(line)==len(line),"Complete label-free encoded row")
            row_digest.update(line)
            totals.update(images=1,supported=n,candidates=len(candidates))
        stream.flush()
        os.fsync(stream.fileno())
    need(totals["images"]==183 and totals["candidates"]==18300 and totals["supported"]==8821
         and totals["positive"]==8785 and totals["exchanges"]==57692
         and all(v["total"]==115384 and v["candidates"]==8785 for v in arm_stats.values()),
         "Exactly same realized native write budget across all mask arms")
    result=dict(status=STATUS,protocol_sha256=LOCK,worker_sha256=digest(Path(__file__).read_bytes()),
                method_sha256=CONSTRAINT,archive_loader_sha256=ARCHIVE_LOADER,
                archive_render_sha256=roots.ARCH_RENDER,archive_score_sha256=roots.ARCH_SCORE,
                archive_score_terminal_sha256=arch_term,archive_bce_tensor_sha256=roots.ARCH_BCE,
                archive_fit_result_sha256=roots.ARCH_FIT,point_terminal_sha256=point_terminal,
                point_result_sha256=old.POINTS,point_index_sha256=point_index,render_stream_sha256=row_digest.hexdigest(),
                counts=dict(totals),arm_totals=arm_stats,net_area_change_histograms=areas,decoded_groups=180,
                seed=a.seed, repeat_fit_result_sha256=a.fit_sha, repeat_fit_readback_sha256=a.fit_readback_sha,
                repeat_fit_terminal_sha256=fit_term, repeat_fit_protocol_sha256=a.fit_lock_sha,
                repeat_fit_worker_sha256=a.fit_worker_sha, repeat_fit_checkpoint_sha256=fit["output_checkpoint_sha256"],
                repeat_bce_tensor_sha256=model_tensor, exact_archive_rle_preserved=True,
                exact_bce_archive_rle_parity=True,exact_dynamic_archive_rle_parity=True,
                source_gate_labels_opened=False,fit_labels_opened=False,historical_test_opened=False,
                detector_forwards=0,optimizer_updates=0,elapsed_seconds=perf_counter()-start)
    raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with (a.output/"RESULT_SAFE.json").open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(dict(status=STATUS,result_sha256=digest(raw))),flush=True)


def main():
    p=ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("protocol","group-protocol","archive-loader","variable-writer","gate","archive",
                 "fit","points","core","selector","attraction","output"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("fit-sha","fit-worker-sha","fit-lock-sha","fit-readback-sha"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--seed",type=int,choices=(5298,5299),required=True)
    render(p.parse_args())


if __name__=="__main__":
    main()
