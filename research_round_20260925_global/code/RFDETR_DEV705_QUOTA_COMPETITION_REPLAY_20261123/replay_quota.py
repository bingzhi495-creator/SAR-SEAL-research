"""Reconstruct six quota controls and the reference on label-free DEV705 masks."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

import numpy as np


SEED = 5217
ARMS = ("SU", "B", "R", "J", "C", "A", "M", "G")
PREPARED = "RFDETR_DEV705_QUOTA_COMPETITION_PREPARED"
ACCEPTED = "RFDETR_DEV705_QUOTA_COMPETITION_REPLAY_ACCEPTED"
MIB = 1024**2
ORIGINAL_LOADER = "3a77e660fd31570cf1f4b44e818b28cc6743d414591216a5f274a11139955b05"
QUOTA = "cd6d1a732a01f8804d28991e4debf30c88f50057b0b2ad0754b7db44c46691a2"
SOURCE_ACTIONS = "c3834371916148546c2dd3b842839808214798911bf04d7065cdea0221a8f4ae"
PROTOCOL = "ceb5e6d0615cc794f1232adc8dc4853e6839580ee41ad7f95af5fe5273353a29"
FITTED = "RFDETR_FIT109_QUOTA_COMPETITION_MODELS_COMPLETE"
INPUT_PATHS = {
    "protocol", "code", "capture", "capture_accepted", "image_manifest", "archive",
    "readout_root", "readout_accepted", "spot_source", "probability_repair", "fit_result",
    "fit_accepted", "head_root", "policy_source", "policy_model", "policy_fit_result", "policy_fit_terminal",
}
INPUT_HASHES = {
    "protocol_sha", "capture_accepted_sha", "readout_accepted_sha", "probability_repair_sha",
    "fit_result_sha", "fit_accepted_sha", "policy_source_sha", "policy_model_sha", "policy_fit_result_sha",
    "policy_fit_terminal_sha",
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, digest, limit=MIB):
    need(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= limit,
         "Bounded exact input " + path.name)
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Input digest " + path.name)
    return raw


def document(path, digest, limit=MIB):
    return json.loads(pin(path,digest,limit))


def module(path, digest, name):
    obj = ModuleType(name)
    obj.__file__ = str(path)
    sys.modules[name] = obj
    exec(compile(pin(path,digest),str(path),"exec"),obj.__dict__)
    return obj


def sparse_rows(path, expected_sha):
    need(path.is_file() and 0 < path.stat().st_size <= 512*MIB, "Bounded frozen sparse edits")
    digest = sha256()
    with path.open("rb") as stream:
        for raw in stream:
            need(raw.endswith(b"\n"), "Complete sparse row")
            digest.update(raw)
            yield raw
    need(digest.hexdigest() == expected_sha, "Exact complete sparse stream")


def admitted_refit(a, original, readout, spot, fit_step):
    report = document(a.readout_accepted,a.readout_accepted_sha)
    source = document(a.readout_root/"RESULT_SAFE.json",report["readout_result_sha256"])
    observed = document(a.readout_root/"ORIGINAL_TOOL_WAIT_PRIVATE.json",report["original_tool_wait_sha256"])
    footer = json.loads(observed["terminal"]["output"].splitlines()[-1])
    original.verify_readout_pair(report,source,readout)
    need(report["status"] == "RFDETR_CENTER5202_DEV705_MATCHED_READOUT_INDEPENDENT_ACCEPTED"
         and source["status"] == readout.STATUS and report["images"] == 705
         and report["original_exit"] == observed["terminal"]["exit_code"] == 0
         and footer["result_sha256"] == report["readout_result_sha256"]
         and footer["status"] == source["status"]
         and all(report[k] for k in ("exact_candidate_identity","exact_native_support",
                                     "exact_per_candidate_budgets","actual_XOR_area_RLE_verified"))
         and report["capture_accepted_sha256"] == source["capture_accepted_sha256"] == a.capture_accepted_sha
         and report["probability_repair_sha256"] == source["probability_repair_sha256"] == a.probability_repair_sha
         and report["probability_repair_consumed"] == source["probability_repair_consumed"] == 1
         and report["edits_sha256"] == source["edits_sha256"]
         and report["counts"] == source["counts"]
         and source["fit_result_sha256"] == a.fit_result_sha and source["fit_accepted_sha256"] == a.fit_accepted_sha
         and all(x["DEV"] is True and not any(x[k] for k in ("FIT","TEST","AP","GT_opened"))
                 for x in (report,source)), "Same observed and independently verified frozen DEV readout")
    images, blocks, capture = readout.original_population(a,spot)
    plan = module(a.code/"RFDETR_MATCHED_POINT_FORMAL_20261120/fit_plan.py",
                  readout.PINS["fit_plan.py"],"_quota_dev_fit_plan")
    fitted = document(a.fit_result,a.fit_result_sha,8*MIB)
    accepted = document(a.fit_accepted,a.fit_accepted_sha)
    refit = readout.validated_refits(fitted,accepted,a.fit_result_sha)[SEED]
    base = a.code/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24"
    trainer = fit_step.load_trainer(base/"fit109_paired_point/train_point.py")
    heads, center, scale = plan.restore_bundle(a.head_root/refit["relative_weights"],refit["weights_sha256"],
                                               trainer,fit_step)
    return dict(report=report, baseline=source, images=images, blocks=blocks, capture=capture,
                head=heads["S"], center=center, scale=scale, base=base)


def edit_for_action(stock, coords, q, order, adds, drops):
    """Write a signed quota using unchanged native coordinates and stable prefixes."""
    take, remove = order[0][:adds], order[1][:drops]
    add_idx, drop_idx = coords[take,0], coords[remove,1]
    n = adds+drops
    need(0 <= adds <= len(coords) and 0 <= drops <= len(coords) and n <= 16
         and np.unique(coords).size == coords.size and np.all((coords >= 0) & (coords < stock.size))
         and not stock.flat[coords[:,0]].any() and stock.flat[coords[:,1]].all(),
         "Unique bounded endpoint signs and feasible signed quota")
    flips = np.r_[add_idx,drop_idx]
    gain = np.r_[q[take,0]-np.float32(.5),np.float32(.5)-q[remove,1]]
    cost = int(np.count_nonzero(gain <= 0))
    need(len(flips) == n, "Exact sparse XOR and area obtained by distinct native positions")
    return dict(flipped=flips.tolist(),edits=n,adds=adds,drops=drops,area_delta=adds-drops,
                actions=n,action_kind="flip",nonpositive_actions=cost,nonpositive_flips=cost,
                predicted_gain_sum=float(np.sum(gain,dtype=np.float64)))


def replay_candidates(a, ordinal, item, block, old, readout, fit_step, spot, codec, torch, stage, h):
    expected = {}
    pool = spot.native_candidate_rows(block)
    for cand in pool:
        cand["area"] = int(codec.area(cand["rle"]))
    eligible = [c for c in pool if 32 <= c["area"] < 1024]
    need(old["ordinal"] == ordinal and old["image_id"] == item["image_id"] == block["image_id"]
         and old["group"] == block["source_prefix"] and old["original_hw"] == [800,800]
         and old["mapped_identity_sha256"] == readout.mapped_identity_sha256(pool),
         "Archived score, rank, query and image identity")
    ref = block["logit_capture"]
    name = ref["relative_path"]
    need(name == f"RAW_LOGITS/{ordinal:03d}.npy", "Original retained raw plane order")
    path = a.capture/name
    data = pin(path,ref["sha256"],9*MIB)
    raw = np.load(BytesIO(data),allow_pickle=False)
    need(len(data) == ref["bytes"] and raw.dtype == np.dtype("<f4") and np.isfinite(raw).all()
         and list(raw.shape) == ref["shape"] == [len(eligible),144,144], "Exact saved original logits")
    source = dict(relative_path=name,path=str(path.resolve()),bytes=ref["bytes"],sha256=ref["sha256"])
    boxes = spot.author_boxes(block["raw_boxes_cxcywh"],torch)
    rgb, hw = h.png.read_png(h.archive,item)
    need(hw == (800,800), "Exact original native canvas")
    features = h.pair.image_features(rgb)
    h.counts.update(images=1,mapped=len(pool),eligible=len(eligible),empty=sum(c["area"] == 0 for c in pool))
    old_arms = old["seeds"][str(SEED)]
    baseline_ranks = set()
    for i,(cand,proof,plane,point) in enumerate(zip(eligible,ref["candidates"],raw,old["points"],strict=True)):
        qid, rank, score, area = (cand[key] for key in ("query_id","rank","score","area"))
        need((proof["logit_row"],proof["query_id"],proof["native_rank"],proof["score"],proof["predicted_area"])
             == (i,qid,rank,score,area) and (point["rank"],point["query_id"]) == (rank,qid),
             "Exact retained candidate and support identity")
        key = str(rank)
        historical = old_arms.get(key,{})
        stock = codec.decode(cand["rle"]).astype(bool)
        identity = dict(ordinal=ordinal,image_id=item["image_id"],query_id=qid,native_rank=rank,logit_row=i)
        posterior = readout.probability_with_repair(np,torch,h.feat,h.posterior,plane,stock,cand["rle"],
                                                   identity,source,h.repair)
        h.counts["exact_original_masks"] += 1
        record = None
        if spot.has_author_support(boxes[qid]):
            record = h.feat.extract_candidate(np,h.pair,h.ranker,stock,boxes[qid],score,posterior,features)
        else:
            h.counts["zero_area_author_boxes"] += 1
        coords = [] if record is None else record["coordinates"].tolist()
        need(coords == point["coordinates"], "Reconstructed supported endpoints exactly match old readout")
        if record is None:
            need(not historical,"Unsupported candidates are original in every arm")
            h.counts["unsupported"] += 1
            continue
        coords = record["coordinates"]
        h.counts.update(supported=1,point_pairs=len(coords),head_calls=1,head_endpoint_values=2*len(coords))
        q = fit_step.predict(stage["head"],record["features"],stage["center"],stage["scale"])
        ranks = h.quota.original_ranks(q)
        actual = historical.get("SU")
        old_flips = () if actual is None else actual["flipped"]
        h.quota.frozen_prefix(coords,*ranks,old_flips)
        h.quota.frozen_prefix(coords,*ranks,historical.get("SU=F",{}).get("flipped",()))
        k,moves,action_features = h.actions.actions_and_features(coords,q,ranks,area,score)
        need(len(old_flips) == 2*k, "Exact archived natural positive same-count decision")
        h.counts["feasible_actions"] += len(moves)
        if not k:
            h.counts["zero_budget"] += 1
            continue
        baseline_ranks.add(key)
        need(actual["edits"] == len(old_flips) and actual["adds"] == actual["drops"] == k
             and actual["area_delta"] == 0, "Archived natural balanced mask budget and signs")
        choices = h.policy.select_actions(h.model,k,moves,action_features)
        need(set(choices) == set(ARMS[1:]) and all(type(i) is int and 0 <= i < len(moves) for i in choices.values())
             and np.array_equal(moves[choices["B"]],(k,k)), "Original policy bundle and balanced reference")
        arms = {"SU":actual,"B":actual}
        for arm in ARMS[2:]:
            adds,drops = (int(v) for v in moves[choices[arm]])
            need(adds+drops == 2*k,"Every active arm inherits the exact same number of native flips")
            arms[arm] = edit_for_action(stock,coords,q,ranks,adds,drops)
        for arm, edit in arms.items():
            h.budgets[arm].update(edited=1,quota_differs_from_balance=int(edit["adds"] != k),
                                  **{key:edit[key] for key in ("edits","adds","drops","area_delta","nonpositive_flips")})
        expected[key] = arms
    need(baseline_ranks == {k for k,v in old_arms.items() if "SU" in v},
         "No omitted archived nonzero natural action")
    return {key:old[key] for key in ("ordinal","image_id","group","original_hw","mapped_identity_sha256","points")} | {
        "seeds":{str(SEED):expected}}


def verify_prepared(a, prepared, terminal, worker_sha):
    raw = (prepared/"RESULT_SAFE.json").read_bytes()
    old = json.loads(raw)
    actual = document(terminal,a.prepared_terminal_sha)
    need(actual["phase"] == "prepare-edits" and actual["actual_exit"] == 0
         and actual["output_footer_verified"] is True and actual["result_sha256"] == sha256(raw).hexdigest()
         and actual["worker_sha256"] == worker_sha == old["worker_sha256"] and old["status"] == PREPARED
         and old["DEV"] is True and not any(old[k] for k in ("FIT","TEST","AP","GT_opened"))
         and old["protocol_sha256"] == a.protocol_sha and old["policy_model_sha256"] == a.policy_model_sha
         and old["policy_source_sha256"] == a.policy_source_sha and old["readout_accepted_sha256"] == a.readout_accepted_sha,
         "Require an actual exited and content-verified label-free preparation for comparison")
    need(old["policy_fit_result_sha256"] == a.policy_fit_result_sha
         and old["policy_fit_terminal_sha256"] == a.policy_fit_terminal_sha, "Same actual source-only fit")
    return old


def admitted_model(a, policy):
    receipt = document(a.policy_fit_result,a.policy_fit_result_sha)
    observer = document(a.policy_fit_terminal,a.policy_fit_terminal_sha)
    need(observer["phase"] == "prepare-edits" and observer["actual_exit"] == 0
         and observer["output_footer_verified"] is True and observer["result_sha256"] == a.policy_fit_result_sha
         and receipt["status"] == FITTED and receipt["model_sha256"] == a.policy_model_sha
         and receipt["FIT"] is True and receipt["DEV"] is receipt["TEST"] is False,
         "Actual closed source-only policy fit bound to the supplied model")
    return policy.load_model(a.policy_model,a.policy_model_sha)


def main():
    ap = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    ap.add_argument("--phase",required=True,choices=("prepare-edits","verify-edits"))
    ap.add_argument("--authority",type=Path,required=True)
    ap.add_argument("--authority-sha",required=True)
    ap.add_argument("--output",type=Path,required=True)
    cfg = ap.parse_args()
    verifying = cfg.phase == "verify-edits"
    extra_paths = {"prepared","prepared_terminal"} if verifying else set()
    extra_hashes = {"prepared_terminal_sha"} if verifying else set()
    path_keys = INPUT_PATHS | extra_paths
    raw = document(cfg.authority,cfg.authority_sha)
    need(set(raw) == path_keys | INPUT_HASHES | extra_hashes
         and all(Path(raw[k]).is_absolute() for k in path_keys)
         and cfg.output.is_absolute() and not cfg.output.exists() and cfg.output.parent.is_dir()
         and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.flags.isolated and sys.dont_write_bytecode,
         "Fresh isolated CPU-only label-free DEV authority")
    a = SimpleNamespace(**{k:Path(v).resolve(strict=True) if k in path_keys else v for k,v in raw.items()})
    need(a.protocol_sha == PROTOCOL,"Exact published quota fit and evaluation protocol")
    pin(a.protocol,PROTOCOL)
    start = perf_counter()
    worker_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    old = verify_prepared(a,a.prepared,a.prepared_terminal,worker_sha) if verifying else None
    formal = a.code/"RFDETR_MATCHED_POINT_FORMAL_20261120"
    original = module(formal/"run_score_dev.py",ORIGINAL_LOADER,"_quota_dev_original_loader")
    fit_step = original.load_fit_step(a.spot_source/"fit_step.py")
    spot = fit_step.load(a.spot_source/"run_spot.py",original.SOURCE_SPOT,"_quota_dev_spot")
    readout = fit_step.load(formal/"run_dev_readout.py",original.SOURCE_READOUT,"_quota_dev_readout")
    quota = module(a.code/"RFDETR_DEV705_FROZEN_ORDER_QUOTA_20261123/quota.py",QUOTA,"_quota_dev_frozen_order")
    actions = module(a.code/"RFDETR_FIT109_FROZEN_QUOTA_ACTIONS_20261123/source_actions.py",SOURCE_ACTIONS,"_quota_dev_actions")
    policy = module(a.policy_source,a.policy_source_sha,"_quota_dev_policy")
    model = admitted_model(a,policy)
    import torch
    from pycocotools import mask as codec
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    need(not torch.cuda.is_initialized(),"No accelerator context")
    stage = admitted_refit(a,original,readout,spot,fit_step)
    base = stage["base"]
    h = SimpleNamespace(
        feat=fit_step.load(a.spot_source/"feature_adapter.py",readout.PINS["feature_adapter.py"],"_quota_dev_features"),
        posterior=fit_step.load(a.code/"RFDETR_MATCHED_POINT_REFINEMENT_20261120/adapter.py",
                                spot.ADAPTER_SHA,"_quota_dev_posterior"),
        png=readout.load(base/"dev705/dfine_dev705.py",readout.PNG_READER,"_quota_dev_png"),
        pair=fit_step.load(base/"fit109_paired_point/pair_math.py",spot.PAIR_SHA,"_quota_dev_pairs"),
        ranker=fit_step.load(base/"fit109_probability_rank/rank_math.py",spot.RANK_SHA,"_quota_dev_ranks"),
        repair=readout.read_probability_repair(a), actions=actions, quota=quota, policy=policy, model=model,
        counts=Counter(), budgets={name:Counter() for name in ARMS})
    original_rows = sparse_rows(a.readout_root/"SPARSE_EDITS_PRIVATE.jsonl",stage["report"]["edits_sha256"])
    cfg.output.mkdir()
    if verifying:
        original_run = sparse_rows(a.prepared/"SPARSE_EDITS_PRIVATE.jsonl",old["edits_sha256"])
    else:
        outgoing = (cfg.output/"SPARSE_EDITS_PRIVATE.jsonl").open("xb")
    digest = sha256()
    try:
        with a.archive.open("rb") as file:
            need(a.archive.stat().st_size == h.png.ARCHIVE_BYTES,"Original complete source PNG archive")
            file.seek(h.png.CENTRAL_OFFSET)
            need(sha256(file.read(h.png.CENTRAL_BYTES)).hexdigest() == h.png.CENTRAL_SHA,
                 "Exact original archive central directory")
            with ZipFile(file) as archive:
                h.archive = archive
                for number,(item,block,previous) in enumerate(zip(stage["images"],stage["blocks"],original_rows,strict=True),1):
                    replayed = replay_candidates(a,number,item,block,json.loads(previous),readout,fit_step,spot,codec,torch,stage,h)
                    line = readout.encoded(replayed)
                    if verifying:
                        need(next(original_run,None) == line,"Independent byte-exact replay by author image position")
                    else:
                        need(outgoing.write(line) == len(line),"Complete sparse edit write")
                    digest.update(line)
                    if number % 25 == 0 or number == 705:
                        print(json.dumps(dict(event="DEV705_FROZEN_QUOTA_PROGRESS",images=number)),flush=True)
        if verifying:
            need(next(original_run,None) is None,"No trailing prepared rows")
        else:
            outgoing.flush()
            os.fsync(outgoing.fileno())
    finally:
        if not verifying:
            outgoing.close()
    counts = dict(h.counts)
    prior = stage["baseline"]
    packed = {arm:dict(value,unchanged=67386-value["edited"]) for arm,value in h.budgets.items()}
    need(all(counts[k] == value for k,value in prior["counts"].items())
         and all(packed[arm]["edits"] == prior["budgets"][str(SEED)]["SU"]["edits"]
                 and packed[arm]["edited"] == prior["budgets"][str(SEED)]["SU"]["edited"] for arm in ARMS)
         and not torch.cuda.is_initialized(),"All original candidates and exact previously realized budgets")
    correction = readout.repair_summary(stage["capture"],h.repair)
    new_sha = digest.hexdigest()
    if verifying:
        need(old["edits_sha256"] == new_sha and old["budgets"] == packed and old["counts"] == counts
             and old["stock_predictions_sha256"] == readout.STOCK,
             "Every independently reproduced edit and count accepted before scoring")
    result = dict(status=ACCEPTED if verifying else PREPARED,protocol_sha256=a.protocol_sha,
                  authority_sha256=cfg.authority_sha,worker_sha256=worker_sha,
                  readout_accepted_sha256=a.readout_accepted_sha,capture_accepted_sha256=a.capture_accepted_sha,
                  base_refit_result_sha256=a.fit_result_sha,base_refit_accepted_sha256=a.fit_accepted_sha,
                  policy_model_sha256=a.policy_model_sha,policy_source_sha256=a.policy_source_sha,
                  policy_fit_result_sha256=a.policy_fit_result_sha,policy_fit_terminal_sha256=a.policy_fit_terminal_sha,
                  prepared_terminal_sha256=a.prepared_terminal_sha if verifying else None,
                  stock_predictions_sha256=readout.STOCK,reference_edits_sha256=stage["report"]["edits_sha256"],
                  edits_sha256=new_sha,images=705,seed=SEED,arms=list(ARMS),counts=counts,budgets=packed,
                  exact_candidate_identity=True,exact_native_support=True,exact_realized_counts=True,
                  exact_baseline_order=True,independent_replayed=verifying,**correction,
                  wall_seconds=perf_counter()-start,detector_forwards=0,optimizer_updates=0,
                  FIT=False,DEV=True,GT_opened=False,TEST=False,AP=False)
    result_sha = readout.write(cfg.output/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=result["status"],result_sha256=result_sha)),flush=True)


if __name__ == "__main__":
    main()
