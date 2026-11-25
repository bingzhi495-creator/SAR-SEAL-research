"""Replay within-candidate radiometric features on the frozen development gate without labels."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from io import BytesIO
import gzip
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec
import torch

LOCK = "d25a4ff51bc266bdab598a2fb013ff96a401c8368f2aacbcf68873f32e8cbd63"
MENU_WORKER = "6f7402b8ad6d9ccde478a36df178d84ce376a6c315b7fcfd6382e1d05f2487c9"
GROUP = "4030a3cb5684a6dac0ea1e7765a48d05caec113c5c6e0e6de2c82b991e9d5f15"
PREVIOUS_RENDER = "4753e73b6056f405ef5faac7d0de2c23392bd6003b24339272c8fb745a4422d7"
PREVIOUS_SCORE = "38291650ed596a91c3fd677ecfb3d662b3024b6a693ac203c49d8f4323845c66"
PREVIOUS_RENDER_WORKER = "c5cb6feaa85cd1f3131c10df93f1ea8b47f331bd0a91a07013881b8c86de2d07"
PREVIOUS_SCORE_WORKER = "9689c4044abea34b32f14977b3e5564b1cccb4020e5c26bed51078347b6799ce"
REGRESSION_PARAMETERS = "d41d2538ae03ff76b385f7e25ca5288fea9930920678e50ceb0e1863047ef456"
ACTION_CORE = "5e7deb9fb35a2b22f0d86dc4a4a1daca24a87d5134a7d87dacfb1af9824c5463"
CORE = "0f1537f12ed4890bb60f71957e5d81926aeaab4760c8b01291383e4b5c53ee0d"
SELECTOR = "7c7a5415dad9c70804a0cf4dd07d0be46f8cbc5b3c0d1666e3cd067d25846205"
ATTRACTION = "8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6"
OLD_RENDER = "4681265afbbf7acae7389a937e988e85218f558ad7ff99e6d87fa5d0f69e51b3"
OLD_SCORE = "e8268d0a4fdf3c36b92e8ac2ae604905b51a5337c9f6dc43463a0f42ccad53f9"
RENDER_WORKER = "17fb4db0858785a312d1c761d97a39f8de448e9b19230aeb727a8ff5d286f5a7"
SCORE_WORKER = "64b4df4f9b6af079eb56a5099d1552a05d2c701d020102f3007ad64e2d754a69"
POINTS = "87693897f873a2129764781611b0f7c35cadde51c0c4e830d7f8187c3b684049"
POINT_WORKER = "926f87b220c4ba9f1a7074088a52641cd66c17d4016bac91fba20c84f450f766"
HEADS = {"regression": "regression", "joint": "joint", "per_side": "per_side"}
POLICIES = ("robust", "mean", "cell_pessimistic")
ARMS = ("native", "dynamic", "static",
        "regression_robust", "regression_mean", "regression_cell_pessimistic",
        "joint_robust", "joint_mean", "joint_cell_pessimistic",
        "per_side_robust", "per_side_mean", "per_side_cell_pessimistic",
        "archive_robust", "archive_mean", "archive_cell_pessimistic")
ANCHORS = {"native": "native", "dynamic": "dynamic", "static": "static",
           **{"archive_" + p: "deployment_" + p for p in POLICIES}}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def hash_bytes(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(hash_bytes(raw) == expected, "Exact file digest: " + path.name)
    return raw


def load(path, expected):
    m = ModuleType(path.stem)
    m.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), m.__dict__)
    return m


def prior(gate):
    ro = load(gate/"render_gate.py", RENDER_WORKER)
    reader = load(gate/"score_gate.py", SCORE_WORKER)
    complete = json.loads(pin(gate/"score/RESULT_SAFE.json", OLD_SCORE))
    end = ro.terminal(gate/"SCORE_TERMINAL.json", "score", OLD_SCORE, SCORE_WORKER)
    ready, term, _, native, records, group = reader.exact_readback(SimpleNamespace(render_root=gate))
    need(term["result_sha256"] == OLD_RENDER == complete["render_result_sha256"]
         and ready["render_stream_sha256"] == complete["render_stream_sha256"]
         and complete["historical_test_opened"] is False and ready["source_gate_labels_opened"] is False,
         "Prior full exact frozen data authority")
    return ro, reader, ready, native, records, group, complete, end


def previous_loss_gate(root, authority):
    score_term = authority.terminal(root/"SCORE_TERMINAL.json", "score", PREVIOUS_SCORE, PREVIOUS_SCORE_WORKER)
    render_term = authority.terminal(root/"RENDER_TERMINAL.json", "render", PREVIOUS_RENDER, PREVIOUS_RENDER_WORKER)
    score = json.loads(pin(root/"score/RESULT_SAFE.json", PREVIOUS_SCORE))
    render = json.loads(pin(root/"render/RESULT_SAFE.json", PREVIOUS_RENDER))
    need(score["render_result_sha256"] == PREVIOUS_RENDER and score["render_terminal_sha256"] == render_term
         and score["render_stream_sha256"] == render["render_stream_sha256"]
         and score["fit_result_sha256"] == render["fit_result_sha256"]
         and score["fit_labels_opened"] is score["historical_test_opened"] is False
         and render["source_gate_labels_opened"] is False, "Closed prior-style replay and score authority")
    rows = [json.loads(i) for i in pin(root/"render/GATE_RENDER_PRIVATE.jsonl", render["render_stream_sha256"]).splitlines()]
    bank = {r["image_id"]: r for r in rows}
    need(len(bank) == len(rows) == 183 and all(len(r["candidates"]) == 100 for r in rows),
         "Exact prior matched-action data scope")
    return score, bank, score_term


def fitted(a, ro):
    pin(a.fit/"fit_features.py", a.fit_worker_sha)
    pin(a.fit/"PRELOCK.md", a.fit_lock_sha)
    checked = json.loads(pin(a.fit/"READBACK_SAFE.json", a.fit_readback_sha))
    need(checked["status"] == "SPSD_PSEG_MATCHED_RADIOMETRIC_FEATURE_READBACK_COMPLETE",
         "Independent fit readback completed")
    end = ro.terminal(a.fit/"observed_fit/TERMINAL.json", "prepare-edits", a.fit_sha, a.fit_worker_sha)
    doc = json.loads(pin(a.fit/"actual/RESULT_SAFE.json", a.fit_sha))
    need(doc["status"] == "SPSD_PSEG_MATCHED_RADIOMETRIC_FEATURE_FIT_COMPLETE"
         and doc["protocol_sha256"] == a.fit_lock_sha and doc["worker_sha256"] == a.fit_worker_sha
         and doc["total_optimizer_updates"] == 3000 and doc["out_of_fold_actual_updates"] == 5000
         and doc["feature_worker_sha256"] == a.transform_sha and doc["all_raw_14_scalers_frozen"] is True
         and doc["capacity_aware_replacement_width"] == 5
         and doc["regression_baseline_tensor_sha256"] == REGRESSION_PARAMETERS
         and doc["detector_forwards"] == 0 and doc["source_gate_labels_opened"] is doc["historical_test_opened"] is False
         and doc["train_audit"]["seed"] == 5297,
         "Exact feature-only paired fit and archived regression tensor parity")
    inp = json.loads(pin(a.fit/"actual/INPUT_SAFE.json", doc["input_receipt_sha256"]))
    need(inp["status"] == "FIT_MATCHED_RADIOMETRIC_FEATURE_PACKETS_ADMITTED"
         and inp["fit_only"] is True and inp["source_gate_labels_opened"] is inp["historical_test_opened"] is False,
         "Exact pre-fit source-only admitted packet")
    core = load(a.core, CORE)
    radiometric = load(a.fit/"radiometric.py", a.transform_sha)
    checkpoint = torch.load(BytesIO(pin(a.fit/"actual/FITTED_WEIGHTS_PRIVATE.pt",doc["output_checkpoint_sha256"])),
                            weights_only=True,map_location="cpu")
    need(checkpoint["seed"] == 5297 and set(checkpoint["models"]) == set(HEADS)
         and set(checkpoint["center"]) == set(checkpoint["scale"]) == set(HEADS),
         "Exact final states and FIT-only scaling for all heads")
    audit = doc["train_audit"]["heads"]
    parity = ("initial_tensor_sha256","batch_sha256","observed_candidates","observed_endpoints","observed_actions")
    need(all(audit["regression"][k] == audit[head][k] for k in parity for head in ("joint","per_side"))
         and audit["regression"]["final_tensor_sha256"] == REGRESSION_PARAMETERS,
         "Matched sample, seed and original-feature archived parity")
    models, center, scale = {}, {}, {}
    for name,state in checkpoint["models"].items():
        model = core.PointHead()
        model.load_state_dict(state,strict=True)
        metadata = audit[name]
        sha = hash_bytes(b"".join(p.detach().numpy().astype("<f4",copy=False).tobytes() for p in model.parameters()))
        need(metadata["final_tensor_sha256"] == sha and metadata["optimizer_updates"] == 1000
             and metadata["parameters"] == 1346, "Exact final matched tensor identities and steps")
        models[name] = model.eval()
        center[name], scale[name] = (np.asarray(checkpoint[tag][name], dtype=np.float32) for tag in ("center","scale"))
        shape = (14,) if name == "regression" else (28,)
        need(center[name].shape == scale[name].shape == shape and np.isfinite(center[name]).all()
             and np.isfinite(scale[name]).all() and (scale[name] > 0).all(), "Exact feature-domain standardization shape")
    need(all(np.array_equal(values[arm][:14],values["regression"]) for values in (center,scale)
             for arm in ("joint","per_side")), "Original 14 feature moments retained exactly")
    return core, radiometric, models, center, scale, doc, end


def relative(estimates, picked, incumbent):
    add, drop = (estimates[:,s,:] for s in (0,1))
    per_side = []
    for side, current, old in zip((add,drop),picked,incumbent,strict=True):
        p = side[np.asarray(current,dtype=np.intp)].sum(axis=0,dtype=np.float64)
        q = side[np.asarray(old,dtype=np.intp)].sum(axis=0,dtype=np.float64)
        per_side.append(p-q)
    return (per_side[0]-per_side[1]).tolist()


def render(a):
    start=perf_counter()
    ro,_,old,native,records,groups,_,prior_term = prior(a.gate)
    _, old_bank, old_bank_term = previous_loss_gate(a.previous_gate, ro)
    need(set(old_bank) == set(native), "Exact same previous-style gate roster")
    core,radiometric,models,center,scale,fit,fit_term = fitted(a,ro)
    selector,attraction = load(a.selector,SELECTOR),load(a.attraction,ATTRACTION)
    menu = load(Path(__file__).with_name("common_menu.py"), MENU_WORKER)
    pt = json.loads(pin(a.points/"actual_retry/RESULT_SAFE.json",POINTS))
    pt_term = ro.terminal(a.points/"observed_points_retry/TERMINAL.json","prepare-edits",POINTS,POINT_WORKER)
    need(pt["worker_sha256"] == POINT_WORKER and pt["capture_readback_sha256"] == old["capture_readback_sha256"]
         and pt["train_labels_opened"] is pt["test_labels_opened"] is False, "Exact already verified source points")
    root=a.points/"actual_retry/source_gate"
    index=[json.loads(x) for x in pin(root/"POINT_INDEX_PRIVATE.jsonl",
            pt["partitions"]["source_gate"]["point_index_sha256"]).splitlines()]
    need(len(index)==len(native)==183 and set(native)=={p["image_id"] for p in index}
         and len(set(groups.values()))==180,"Original full gate membership")
    totals, counts, indexed = Counter(), {arm:Counter() for arm in ARMS[1:]}, sha256()
    a.output.mkdir()
    menu_path = a.output/"GATE_MENU_PRIVATE.jsonl.gz"
    with ((a.output/"GATE_RENDER_PRIVATE.jsonl").open("xb") as stream, menu_path.open("xb") as raw_menu,
          gzip.GzipFile(fileobj=raw_menu,filename="",mode="wb",mtime=0) as menu_stream):
        for entry in index:
            iid=entry["image_id"]
            original=records[iid]
            historical = old_bank[iid]
            need(entry["source_group"] == original["source_group"] == historical["source_group"]
                 and entry["original_hw"] == original["original_hw"] == historical["original_hw"]
                 and historical["filename"] == original["filename"], "Original native frame and group")
            for previous, stock in zip(historical["candidates"], native[iid], strict=True):
                need(all(previous[key] == stock[key] for key in ("native", "score", "quota")),
                     "All prior-style candidate identities, native states, scores and budgets")
            rows=[dict(native=x["native"],score=x["score"],quota=x["quota"],changes={}) for x in native[iid]]
            raw=pin(root/entry["filename"],entry["sha256"])
            need(len(raw)==entry["bytes"],"Exact point packet length")
            with np.load(BytesIO(raw),allow_pickle=False) as npz:
                ranks,offset,coords,features,quota,booked=(npz[x] for x in ("native_ranks","point_offsets",
                    "signed_coordinates","features","static_offsets","static_side_indices"))
            n,p,s=len(ranks),len(coords),len(booked)
            need(n==entry["supported_candidates"]==len(set(ranks)) and p==entry["paired_sites"]
                 and s==entry["realized_static_exchanges"] and offset.shape==quota.shape==(n+1,)
                 and offset[0]==quota[0]==0 and offset[-1]==p and quota[-1]==s
                 and np.all((1<=ranks)&(ranks<=100)) and coords.shape==(p,2) and booked.shape==(s,2)
                 and features.shape==(p,2,14),"Exact prepared point scope")
            raw_probs = core.predict(models["regression"], features, center["regression"], scale["regression"])
            menus = {}
            for j,r0 in enumerate(ranks):
                r=int(r0)-1
                pos,prev=rows[r],native[iid][r]
                y=coords[offset[j]:offset[j+1]]
                state=codec.decode(pos["native"]).astype(bool,order="C")
                area=int(codec.area(pos["native"]))
                dyn,sta=attraction.select(state,y,min(8,area//10,len(y)))
                k=int(quota[j+1]-quota[j])
                need(k==len(dyn)==len(sta)==pos["quota"] and np.array_equal(booked[quota[j]:quota[j+1]],
                     np.asarray(sta,dtype=np.int16).reshape((-1,2))),"Exact original static/dynamic paired quota")
                if not k:
                    continue
                inc=ro.actions(sta)
                bank={"static":inc,"dynamic":ro.actions(dyn)}
                f = features[offset[j]:offset[j+1]]
                post={"regression":raw_probs[offset[j]:offset[j+1]]}
                for name in ("joint","per_side"):
                    post[name]=radiometric.predict(models[name], f, center[name], scale[name], name)
                need(all(v.shape == (len(y),2,2) and np.isfinite(v).all() and (0 <= v).all() and (v <= 1).all()
                         for v in post.values()), "Exact native endpoint order and finite candidate features")
                for name,pred in post.items():
                    choice=selector.select(pred[:,0,:],pred[:,1,:],*inc,k=k)
                    need(choice["static"]==inc,"Exact frozen static incumbent")
                    for policy in POLICIES:
                        bank[name+"_"+policy]=choice[policy]
                cache={}
                for arm,action in bank.items():
                    if action not in cache:
                        cache[action]=ro.edited_mask(pos["native"],state,y,action,k)
                    pos["changes"][arm]=cache[action]
                    counts[arm].update(candidates=1,exchanges=k)
                need(pos["changes"]["static"] == prev["changes"]["static"]
                     and pos["changes"]["dynamic"] == prev["changes"]["dynamic"], "Previous native anchor RLE parity")
                archived = historical["candidates"][r]["changes"]
                for current, earlier in ANCHORS.items():
                    if current.startswith("archive_"):
                        pos["changes"][current] = archived[earlier]
                        counts[current].update(candidates=1, exchanges=k)
                need(all(pos["changes"][current] == archived[earlier]
                         for current, earlier in ANCHORS.items() if current != "native"),
                     "Exact historical dynamic/static and all three archived RLEs")
                need(all(pos["changes"]["regression_" + policy] == pos["changes"]["archive_" + policy]
                         for policy in POLICIES),"Exact regression and archive writer parity")
                menus[str(r)] = menu.freeze_menu(y, post, inc, bank, k, selector)
                totals["common_menu_actions"] += menus[str(r)]["actions_count"]
                pos["margin_estimates"]={writer:{est:relative(arr,bank[writer+"_robust"],inc)
                    for est,arr in post.items()} for writer in HEADS.values()}
                totals.update(positive_candidates=1,exchanges=k,
                              changed_joint_vs_regression_robust=int(pos["changes"]["joint_robust"] !=
                                                                    pos["changes"]["regression_robust"]),
                              changed_joint_vs_per_side_robust=int(pos["changes"]["joint_robust"] !=
                                                                  pos["changes"]["per_side_robust"]))
            need(all(bool(row["quota"])==bool(row["changes"]) for row in rows),"Exact supported set and pass-through")
            content=(json.dumps(dict(image_id=iid, source_group=groups[iid],filename=original["filename"],
                              original_hw=entry["original_hw"],candidates=rows),
                              sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
            need(stream.write(content)==len(content),"Complete frozen output row")
            indexed.update(content)
            menu_row = (json.dumps(dict(image_id=iid, menus=menus),sort_keys=True,separators=(",",":"),
                                   allow_nan=False)+"\n").encode()
            need(menu_stream.write(menu_row) == len(menu_row), "Complete common menu row")
            totals.update(images=1,candidates=100,supported=n,common_menu_candidates=len(menus))
    need(totals["common_menu_candidates"] == 8785 and totals["candidates"]==18300 and totals["positive_candidates"]==8785 and totals["exchanges"]==57692
         and totals["supported"]==8821 and all(v==dict(candidates=8785,exchanges=57692) for v in counts.values()),
         "Complete identical per-arm actual signed budgets")
    doc=dict(status="SPSD_MATCHED_RADIOMETRIC_FEATURE_GATE_RENDER_COMPLETE", protocol_sha256=LOCK,
         fit_protocol_sha256=a.fit_lock_sha, fit_worker_sha256=a.fit_worker_sha,
         fit_readback_sha256=a.fit_readback_sha, radiometric_worker_sha256=a.transform_sha,
         worker_sha256=hash_bytes(Path(__file__).read_bytes()), fit_result_sha256=a.fit_sha,
         previous_action_score_sha256=PREVIOUS_SCORE, previous_action_score_terminal_sha256=old_bank_term,
         previous_action_render_sha256=PREVIOUS_RENDER, archive_action_rle_parity=True,
         regression_archive_rle_parity=True, regression_tensor_sha256=REGRESSION_PARAMETERS,
         common_menu_worker_sha256=MENU_WORKER, common_menu_input_sha256=hash_bytes(menu_path.read_bytes()),
         fit_terminal_sha256=fit_term,fit_checkpoint_sha256=fit["output_checkpoint_sha256"],
         prior_score_sha256=OLD_SCORE,prior_render_sha256=OLD_RENDER,prior_score_terminal_sha256=prior_term,
         point_result_sha256=POINTS,point_terminal_sha256=pt_term,render_stream_sha256=indexed.hexdigest(),
         counts=totals,arms=counts,images=183,decoded_groups=180,detector_forwards=0,optimizer_updates=0,
         source_gate_labels_opened=False,fit_labels_opened=False,historical_test_opened=False,
         elapsed_seconds=perf_counter()-start)
    data=(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with (a.output/"RESULT_SAFE.json").open("xb") as out:
        out.write(data)
    print(json.dumps(dict(status=doc["status"],result_sha256=hash_bytes(data))),flush=True)


def main():
    p=ArgumentParser(description=__doc__)
    for x in ("gate","previous-gate","fit","points","core","selector","attraction","protocol","group-protocol","output"):
        p.add_argument("--"+x,type=Path,required=True)
    for name in ("fit-sha","fit-worker-sha","fit-lock-sha","fit-readback-sha","transform-sha"):
        p.add_argument("--"+name,required=True)
    a=p.parse_args()
    pin(a.protocol,LOCK)
    pin(a.group_protocol,GROUP)
    torch.set_num_threads(1)
    render(a)


if __name__=="__main__":
    main()

