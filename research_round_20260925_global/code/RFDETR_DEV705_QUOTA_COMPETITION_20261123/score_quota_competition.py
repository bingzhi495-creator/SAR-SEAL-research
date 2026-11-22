"""Score a sealed one-seed full-population DEV705 quota comparison."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np


LOCK = "ceb5e6d0615cc794f1232adc8dc4853e6839580ee41ad7f95af5fe5273353a29"
COMMIT = "fc9b7ed391e6f19449bb51e1decdee909c5680f1"
PRIOR_QUOTA = "d8a6ed489903bfb504b0202d875ee889546360c757593aa9d6a92637e12683ee"
PRIOR_QUOTA_SOURCE = "cd6d1a732a01f8804d28991e4debf30c88f50057b0b2ad0754b7db44c46691a2"
SPARSE_RLE = "cb9b8e839de85588fdf116ce5554ed6cadbc20bdbe76e38180561528bfd870d5"
SEED = "5217"
REPLAY_ARMS = ("SU", "B", "R", "J", "C", "A", "M", "G")
ARCHIVE_ARMS = ("LU=S", "LW=S")
ARMS = ("N", "SU", "J", "R", "C", "A", "M", "G", *ARCHIVE_ARMS)
CONTRASTS = (("A","SU"), ("A","M"), ("M","G"), ("A","N"), ("A","J"), ("A","R"), ("A","C"))
FIELDS = ("iou", "fp_over_truth", "fn_over_truth")
STATUS = "HRSID_DEV705_FROZEN_QUOTA_COMPETITION_SCORE_COMPLETE"


def need(value, reason):
    if not value:
        raise ValueError(reason)


def pin(path, expected, limit=64*1024**2):
    with path.open("rb") as stream:
        raw = stream.read(limit+1)
    need(0 < len(raw) <= limit and sha256(raw).hexdigest() == expected, "Pinned input " + path.name)
    return raw


def module(path, expected, name):
    item = ModuleType(name)
    item.__file__ = str(path)
    sys.modules[name] = item
    exec(compile(pin(path,expected),str(path),"exec"), item.__dict__)
    return item


def replay_closure(a):
    root = a.new_replay
    phases = {
        "prepare-edits": ("RFDETR_DEV705_QUOTA_COMPETITION_PREPARED", False),
        "verify-edits": ("RFDETR_DEV705_QUOTA_COMPETITION_REPLAY_ACCEPTED", True),
    }
    accepted, receipts = {}, {}
    for phase, (status, independent) in phases.items():
        tpath = root/("observed_"+phase)/"TERMINAL.json"
        expected = getattr(a, phase.replace("-","_") + "_terminal_sha")
        term = json.loads(pin(tpath,expected))
        value = json.loads(pin(root/phase/"RESULT_SAFE.json",term["result_sha256"]))
        need(term["phase"] == phase and term["actual_exit"] == 0 and term["output_footer_verified"] is True
             and value["worker_sha256"] == term["worker_sha256"] and value["status"] == status
             and value["protocol_sha256"] == LOCK and value["seed"] == int(SEED) and value["images"] == 705
             and tuple(value["arms"]) == REPLAY_ARMS and value["policy_model_sha256"] == a.policy_model_sha
             and value["capture_accepted_sha256"] == a.capture_accepted_sha
             and value["readout_accepted_sha256"] == a.readout_accepted_sha
             and value["independent_replayed"] is independent and value["DEV"] is True
             and all(value[k] is True for k in ("exact_candidate_identity","exact_native_support",
                                                "exact_realized_counts","exact_baseline_order"))
             and all(value[k] is False for k in ("FIT","TEST","AP","GT_opened"))
             and value["detector_forwards"] == value["optimizer_updates"] == 0,
             "Independently observed prelabel replay, exact scope and supplied policy")
        accepted[phase] = value
        receipts[phase] = dict(terminal_sha256=expected, result_sha256=term["result_sha256"])
    before, after = accepted.values()
    need(after["prepared_terminal_sha256"] == a.prepare_edits_terminal_sha
         and all(before[k] == after[k] for k in ("edits_sha256","budgets","counts","worker_sha256",
                                                 "policy_source_sha256","stock_predictions_sha256",
                                                 "reference_edits_sha256")),
         "Same model, ancestry, ledger and per-arm write counts in both replays")
    payload = pin(root/"prepare-edits/SPARSE_EDITS_PRIVATE.jsonl",after["edits_sha256"],512*1024**2)
    return after, receipts, [json.loads(line) for line in payload.splitlines()]


def join_frozen(inputs, new_rows, receipt, spot, readout, codec):
    prior, report, images, blocks, old_rows, anchors, witness = inputs
    need(len(new_rows) == len(old_rows) == len(images) == len(blocks) == 705
         and receipt["reference_edits_sha256"] == report["edits_sha256"]
         and receipt["stock_predictions_sha256"] == readout.STOCK,
         "Exactly the accepted input cohort, stock and earlier SU budget")
    joined = {}
    for item, block, old, new in zip(images,blocks,old_rows,new_rows,strict=True):
        image_id = item["image_id"]
        need(image_id == old["image_id"] == new["image_id"] and image_id not in joined
             and {k:v for k,v in old.items() if k != "seeds"} == {k:v for k,v in new.items() if k != "seeds"}
             and set(new["seeds"]) == {SEED}, "Exact projected image, rank and support records")
        rows = spot.native_candidate_rows(block)
        for row in rows:
            row["area"] = int(codec.area(row["rle"]))
        need(readout.mapped_identity_sha256(rows) == new["mapped_identity_sha256"], "Exact native candidate identities")
        native = readout.stock_block(image_id, rows)["coco"]
        coords = {r["rank"]: np.asarray(r["coordinates"],dtype=np.int64).reshape((-1,2)) for r in new["points"]}
        a,b = old["seeds"][SEED], new["seeds"][SEED]
        need(set(b) == {r for r,v in a.items() if "SU" in v}, "Every positive original SU opportunity")
        mapping = {row["rank"]:row for row in rows}
        for key, actions in b.items():
            rank = int(key)
            need(str(rank) == key and rank in coords and 32 <= mapping[rank]["area"] < 1024
                 and set(actions) == set(REPLAY_ARMS) and actions["B"] == actions["SU"] == a[key]["SU"],
                 "Exact archived balanced act on original supported candidate")
            total = actions["SU"]["edits"]
            need(total > 0 and total % 2 == 0 and total <= 16
                 and all(actions[arm]["edits"] == total for arm in REPLAY_ARMS)
                 and all(a[key][arm]["edits"] == total for arm in ARCHIVE_ARMS),
                 "All controls use the same exact original candidate budget")
        joined[image_id] = SimpleNamespace(native=native, pool=rows, coordinates=coords, group=new["group"],
                                            anchors=anchors.get(image_id,{}), old=a, new=b)
    need(sum(len(r.pool) for r in joined.values()) == 67386
         and sum(len(r.coordinates) for r in joined.values()) == 42325
         and sum(len(r.anchors) for r in joined.values()) == 1461, "Original candidates, support and match census")
    return joined, prior, witness[int(SEED)]


def edit_for(frame, rank, arm, score):
    if arm == "N":
        return score.zero_edit("N")
    stream = frame.old if arm in ARCHIVE_ARMS else frame.new
    entry = stream.get(str(rank),{}).get(arm)
    return score.zero_edit("N") if entry is None else entry


def masks_for(arm, joined, score, codec, editor):
    predictions = []
    measured = Counter(edited=0,edits=0,adds=0,drops=0,area_delta=0,quota_differs_from_balance=0,
                       nonpositive_flips=0)
    for frame in joined.values():
        for old, row in zip(frame.native,frame.pool,strict=True):
            rank = row["rank"]
            action = edit_for(frame,rank,arm,score)
            if not action["edits"]:
                need(not action["flipped"], "No-op has no latent sparse action")
                predictions.append(old)
                continue
            coords = frame.coordinates[rank]
            indices = np.asarray(action["flipped"])
            need(indices.ndim == 1 and indices.dtype.kind in "iu" and len(set(indices)) == len(indices)
                 and np.isin(indices,coords.reshape(-1)).all(), "Actions are unique within the original endpoint support")
            rle = editor.xor_rle(old["segmentation"],indices)
            both = int(codec.area(codec.merge([old["segmentation"],rle],intersect=True)))
            added, dropped = int(codec.area(rle))-both, row["area"]-both
            balance = edit_for(frame,rank,"SU",score)["adds"]
            need(added == action["adds"] and dropped == action["drops"]
                 and added+dropped == len(indices) == action["edits"] == 2*balance
                 and added-dropped == action["area_delta"]
                 and type(action["nonpositive_flips"]) is int
                 and 0 <= action["nonpositive_flips"] <= action["edits"],
                 "RLE intersection independently verifies exact XOR and net area")
            measured.update(edited=1,edits=added+dropped,adds=added,drops=dropped,area_delta=added-dropped,
                            quota_differs_from_balance=int(added != balance),
                            nonpositive_flips=action["nonpositive_flips"])
            predictions.append(dict(old,segmentation=rle))
    need(len(predictions) == 67386, "All original native rows retained")
    return predictions, dict(measured,unchanged=67386-measured["edited"])


def fixed_geometry(gt, joined, witnesses, fit_step, helpers, arithmetic, score):
    by_image, by_group = defaultdict(list), defaultdict(list)
    counts = {arm:Counter() for arm in ARMS}
    for ann in gt["annotations"]:
        by_image[ann["image_id"]].append(ann)
    for image in gt["images"]:
        frame = joined[image["id"]]
        if not frame.anchors:
            continue
        records = [dict(image_id=image["id"],rank=rank,query_id=query,group=frame.group,
                        coordinates=frame.coordinates[rank]) for rank,query in frame.anchors.items()]
        attached, _ = fit_step.attach_truth(image,by_image[image["id"]],frame.pool,records,helpers)
        for row in attached:
            key = arithmetic.identity(row)
            previous = witnesses[key]
            need(row["role"] == "valid" and row["matched_gt_id"] == previous["matched_gt_id"]
                 and row["group"] == previous["group"], "Same previous valid native match and source group")
            values = []
            for arm in ARMS:
                action = edit_for(frame,row["rank"],arm,score)
                got = arithmetic.candidate_geometry(row,action)
                if arm in previous["arms"]:
                    initial = previous["arms"][arm]
                    need(all(abs(got[name]-initial[name]) <= 1e-12 for name in ("I","P","T","iou")),
                         "Continuous scores of published controls exactly replayed")
                values.append([got["iou"], (got["P"]-got["I"])/got["T"], (got["T"]-got["I"])/got["T"]])
                counts[arm].update(edited=bool(action["edits"]), net_increased=action["area_delta"]>0,
                                   net_decreased=action["area_delta"]<0,
                                   strict75=got["iou"]>.75, strict90=got["iou"]>.90)
            by_group[frame.group].append(values)
    means = np.array([np.mean(by_group[g],axis=0) for g in sorted(by_group)])
    need(len(means) == 27 and sum(map(len,by_group.values())) == 1461, "Whole original continuous cohort")
    sample = np.random.Generator(np.random.PCG64(5217)).integers(len(means),size=(4096,len(means)))
    effects = {}
    for left,right in CONTRASTS:
        delta = 100*(means[:,ARMS.index(left)]-means[:,ARMS.index(right)])
        interval = np.percentile(delta[sample].mean(axis=1),(2.5,97.5),axis=0)
        effects[left+" minus "+right] = {key:dict(delta=float(delta[:,j].mean()),interval95=interval[:,j].tolist())
                                         for j,key in enumerate(FIELDS)}
    return dict(groups=len(means), anchors=1461, counts=counts, contrasts_points=effects,
                group_equal_points={arm:dict(zip(FIELDS,(100*means[:,i].mean(axis=0)).tolist(),strict=True))
                                    for i,arm in enumerate(ARMS)})


def run(a, output):
    import torch
    from pycocotools import mask as codec

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    q = module(a.code/"RFDETR_DEV705_FROZEN_ORDER_QUOTA_20261123/quota.py",PRIOR_QUOTA_SOURCE,"_quota_history")
    d = q.module(a.code/"RFDETR_OCCUPANCY_BOUND_ADAPTER_20261120/run_diagnostic.py",q.DIAGNOSTIC,"_quota_prior_diag")
    formal = a.code/"RFDETR_MATCHED_POINT_FORMAL_20261120"
    original = d.load_original(formal/"run_score_dev.py")
    fit_step = original.load_fit_step(a.spot_source/"fit_step.py")
    spot = fit_step.load(a.spot_source/"run_spot.py",original.SOURCE_SPOT,"_quota_spot")
    readout = fit_step.load(formal/"run_dev_readout.py",original.SOURCE_READOUT,"_quota_readout")
    score = fit_step.load(formal/"score_dev.py",original.MATHEMATICS,"_quota_native_eval")
    arithmetic = fit_step.load(formal/"evaluate_geometry.py",original.GEOMETRY,"_quota_native_occupancy")
    editor = module(a.code/"RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py",SPARSE_RLE,"_qualified_sparse_rle")
    native = fit_step.load(a.code/"HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py",
                           original.NATIVE_SOURCE,"_quota_native_source")
    native.DEV = a.original_dev_dir
    native.OFFICIAL = a.code/"DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/worker.py"
    replay, receipts, rows = replay_closure(a)
    originals = q.original_inputs(a,d,original,spot,readout)
    joined, prior, witnesses = join_frozen(originals,rows,replay,spot,readout,codec)
    output.mkdir()
    seal = spot.write(output/"PRELABEL_SAFE.json",dict(status="DEV705_QUOTA_COMPETITION_SCORE_PRELABEL",
                      protocol_sha256=LOCK,closed_replay=receipts,images=705,decisions_sha256=replay["edits_sha256"],
                      policy_model_sha256=a.policy_model_sha,historical_geometry_witness_used=True,
                      DEV=True,GT_opened=False,AP=False,FIT=False,TEST=False))
    gt, findings = d.load_dev_geometry_inputs(native)
    need(len(gt["images"]) == 705 and len(gt["annotations"]) == 1782
         and [r["id"] for r in gt["images"]] == list(joined), "Original all-area DEV test geometry identity")
    empty = {r["annotation_id"] for r in findings}
    measured, budgets, times = {}, {}, {}
    for arm in ARMS:
        start = perf_counter()
        masks, budgets[arm] = masks_for(arm,joined,score,codec,editor)
        times[arm] = dict(mask_reconstruction_seconds=perf_counter()-start)
        if arm in replay["budgets"]:
            need(all(k in replay["budgets"][arm] and value == replay["budgets"][arm][k]
                     for k,value in budgets[arm].items()), "Identical realized counts to closed independent replay")
        start = perf_counter()
        measured[arm] = score.score_official(native,gt,masks,empty)
        times[arm]["coco_seconds"] = perf_counter()-start
        if arm in prior["coco"][SEED]:
            need(measured[arm] == prior["coco"][SEED][arm], "Exact historical score for " + arm)
            budget = prior["budgets"][SEED][arm]
            need(all(budgets[arm][k] == budget[k] for k in
                     ("edited","edits","adds","drops","area_delta","nonpositive_flips")),
                 "Exact published edit cost for " + arm)
    helpers = fit_step.load_geometry(a.code,a.shapely_site)
    geo = fixed_geometry(gt,joined,witnesses,fit_step,helpers,arithmetic,score)
    main = {arm:r["primary"]["AP"] for arm,r in measured.items()}
    best = max((arm for arm in ARMS if arm != "A"),key=main.get)
    gain = 100*(main["A"]-main[best])
    need(not torch.cuda.is_initialized(), "No accelerator context or training")
    return dict(status=STATUS,protocol_sha256=LOCK,prelock_commit=COMMIT,phase_receipts=receipts,
                prelabel_sha256=seal,policy_model_sha256=a.policy_model_sha,observed_replay_seconds=replay["wall_seconds"],
                detector_forwards=0,optimizer_updates=0,images=705,mapped=67386,DEV_GT_opened=True,
                DEV=True,AP=True,FIT=False,TEST=False,COCO=measured,geometry=geo,budgets=budgets,arm_seconds=times,
                strongest_reference=best,primary_AP_gain_points=gain,full_repeat_qualifies=gain>=1,
                one_follow_up_qualifies=0<gain<1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--authority",required=True,type=Path)
    parser.add_argument("--authority-sha",required=True)
    parser.add_argument("--output",required=True,type=Path)
    cfg = parser.parse_args()
    arguments = json.loads(pin(cfg.authority,cfg.authority_sha))
    paths = {"protocol","previous_authority","previous_result","new_replay"}
    hashes = {"previous_authority_sha","prepare_edits_terminal_sha","verify_edits_terminal_sha","policy_model_sha"}
    need(set(arguments) == paths|hashes and sys.flags.isolated and sys.dont_write_bytecode
         and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not cfg.output.exists(), "Qualified isolated fresh score")
    args = {k:Path(v) if k in paths else v for k,v in arguments.items()}
    pin(args["protocol"],LOCK)
    old = json.loads(pin(args["previous_result"],PRIOR_QUOTA))
    need(old["authority_sha256"] == args["previous_authority_sha"], "Reuse the fully admitted prior DEV lineage")
    raw = json.loads(pin(args["previous_authority"],args["previous_authority_sha"]))
    inherited = {k:Path(v) if not k.endswith("_sha") else v for k,v in raw.items() if k != "protocol"}
    a = SimpleNamespace(**inherited,**args)
    start = perf_counter()
    result = run(a,cfg.output)
    result.update(worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  wall_seconds=perf_counter()-start,authority_sha256=cfg.authority_sha)
    payload = (json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    (cfg.output/"RESULT_SAFE.json").write_bytes(payload)
    print(json.dumps(dict(status=STATUS,result_sha256=sha256(payload).hexdigest())),flush=True)


if __name__ == "__main__":
    main()
