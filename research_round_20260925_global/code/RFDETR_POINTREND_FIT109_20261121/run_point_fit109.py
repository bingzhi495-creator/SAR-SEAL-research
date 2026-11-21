"""Run the prelocked FIT109 binary PointRend comparison on independently read captures."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from pycocotools import mask as codec


PROTOCOL_SHA = "648db29c9f841f33eb54fdcb6d883c743442940dcd7cea475ad1a91b424c1aa4"
OLD_SUPPORT_RECEIPT = "fc759d9cf1982f3970f3b75627f8eccbca0b474c97ecb141ae487ba7dacc3e94"
OLD_SCORE_RESULT = "7311c19a1fc187665e4f38ed616cf5c007a9818606a1b910520259ec74df270d"
OLD_SCORE_ACCEPTED = "4fbfea6f681342fac77518f47e43245324c1ccd18f48bb8e92fac52f1b6917ed"
MIB = 1024**2
SOURCES = ("fit_point_heads.py", "infer_point_policies.py", "score_point_fit109.py")


def need(ok, why):
    if not ok:
        raise ValueError(why)


def pinned(path, digest, bound):
    with path.open("rb") as fh:
        raw = fh.read(bound+1)
    need(0 < len(raw) <= bound and sha256(raw).hexdigest() == digest, "Pinned input " + path.name)
    return raw


def load(path, digest):
    raw = pinned(path, digest, MIB)
    module = ModuleType("_pointrend_fit109_" + path.stem)
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def write(path, data):
    raw = (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return sha256(raw).hexdigest()


def collect_old_support(args, modules, roster, protocol):
    old, spot = modules.old, modules.spot
    current, _ = old.read_acceptance(SimpleNamespace(new_capture=args.old_support,
                                                      capture_acceptance_sha=OLD_SUPPORT_RECEIPT))
    entries = []
    feature_hash = sha256()
    pools, records = {}, []
    counts = Counter()
    for item, ordinal, block in roster:
        iid = item["image_id"]
        pop = spot.native_candidate_rows(block)
        pools[iid] = pop
        counts["mapped"] += len(pop)
        name = f"NATIVE_ENDPOINT_BASIS_{ordinal:02d}_PRIVATE.npz"
        with (args.old_support/name).open("rb") as f:
            raw = f.read(48*MIB+1)
        need(0 < len(raw) <= 48*MIB, "Original bounded endpoint archive")
        entries.append(dict(ordinal=ordinal, sha256=sha256(raw).hexdigest(), bytes=len(raw)))
        with np.load(BytesIO(raw), allow_pickle=False) as z:
            need(int(z["image_id"]) == iid, "Original FIT image for endpoint archive")
            ranks, qids, off, coords, old14 = (z[key] for key in ("ranks", "query_ids", "offsets", "coords", "old14"))
        need(len(ranks) == len(qids) and off.shape == (len(ranks)+1,) and off[0] == 0
             and off[-1] == len(coords) == len(old14) and np.all(np.diff(ranks)>0), "Ordered endpoint slices")
        lookup = {int(k): index for index,k in enumerate(ranks)}
        seen = set()
        for stock in pop:
            area = stock["area"] = int(codec.area(stock["rle"]))
            if not 32 <= area < 1024:
                continue
            rank, query = stock["rank"], stock["query_id"]
            ident = dict(image_id=iid, rank=rank, query_id=query, group=item["group"])
            if rank in lookup:
                j = lookup[rank]
                loc = slice(off[j], off[j+1])
                x, v = coords[loc], old14[loc]
                need(qids[j] == query and 0 < len(x) <= 32 and x.shape[1:]==(2,)
                     and v.shape==(len(x),2,14) and v.dtype==np.float32, "Original paired support shape")
                for part in (spot.encode(ident), x.tobytes(), v.tobytes()):
                    feature_hash.update(part)
                seen.add(rank)
            else:
                x = np.empty((0,2), dtype=np.int32)
                v = np.empty((0,2,14), dtype=np.float32)
            records.append(dict(ident, coordinates=x, old14=v, area=area, cap=min(8, area//10, len(x))))
        need(seen == set(lookup), "All original nonempty supports accounted for")
    need(counts["mapped"] == protocol["data"]["mapped_candidates"]
         and len(records) == protocol["data"]["area_eligible_candidates"]
         and sha256(spot.encode(entries)).hexdigest() == current["private_output_sha256"]
         and feature_hash.hexdigest() == protocol["prior_oof_support_sha256"],
         "Exact originally accepted point support and full mapped population")
    return pools, records


def read_budgets(path, protocol, records, plan, roster, geom):
    receipt = json.loads(pinned(path/"INDEPENDENT_SCORE_SAFE.json", OLD_SCORE_ACCEPTED, 64*1024))
    need(receipt["result_sha256"] == OLD_SCORE_RESULT and receipt["original_os_exit"] == 0
         and receipt["independent_readback_accepted"] and receipt["job_drained"]
         and receipt["original_old_k_before_GT_replayed"]
         and receipt["native_budget_sha256"] == protocol["prior_oof_budget_sha256"],
         "Closed and prelabel-replayed FIT budgets")
    raw = pinned(path/"BUDGETS_PRIVATE.jsonl", receipt["native_budget_sha256"], 4*MIB)
    entries = [json.loads(line) for line in raw.splitlines()]
    rows = {tuple(r["key"]):r for r in entries}
    folds = plan.source_folds([item["group"] for item,_,_ in roster])
    need(len(rows) == len(entries) == len(records), "Exact prior budget cardinality")
    for row in records:
        k = rows[geom.identity(row)]
        need(type(k["k"]) is int and 0 <= k["k"] <= k["cap"] == row["cap"]
             and k["fold"] == folds[row["group"]], "Frozen per-candidate budget and held group")
    need(sum(r["k"] for r in rows.values()) == protocol["matched_swaps_per_arm"], "Exact original replay counts")
    return rows, folds


def admit_capture(args, protocol):
    accepted = json.loads(pinned(args.capture_readback, args.capture_readback_sha, 64*1024))
    need(accepted["status"] == "RFDETR_POINTREND_FIT109_CARRIER_INDEPENDENT_ACCEPTED"
         and accepted["original_exit"] == 0 and accepted["images"] == protocol["data"]["images"]
         and accepted["eligible_rows"] == protocol["data"]["area_eligible_candidates"]
         and accepted["FIT"] and not accepted["DEV"] and not accepted["TEST"],
         "Original terminal and independent FIT-only capture closure")
    frozen = json.loads(pinned(args.capture/"RESULT_SAFE.json", accepted["result_sha256"], 64*1024))
    need(frozen["status"] == "RFDETR_POINTREND_FIT109_CARRIER_CAPTURE_COMPLETE"
         and frozen["model_forwards"] == protocol["data"]["images"]
         and frozen["index_sha256"] == accepted["index_sha256"], "Bound captured input")
    index = json.loads(pinned(args.capture/"INDEX_PRIVATE.json", accepted["index_sha256"], MIB))
    if isinstance(index, dict):
        index = index["images"]
    by_id = {data["image_id"]:data for data in index}
    need(len(by_id) == len(index) == protocol["data"]["images"], "Full unique captured set")
    return accepted, by_id


def captured_arrays(path, entry):
    # Previously verified original files are given fresh SHA checks before label consumers.
    sd, rd = entry["spatial"], entry["row_archive"]
    spatial_path = path/sd["name"]
    # The full file is 21.24 MB. Open once and keep the observed bytes in this phase.
    spatial = np.load(BytesIO(pinned(spatial_path, sd["sha256"], 22*MIB)), allow_pickle=False)
    need(spatial.shape == (256,144,144) and spatial.dtype == np.float32, "Fixed projected full feature map")
    row_bytes = pinned(path/rd["name"], rd["sha256"], 12*MIB)
    with np.load(BytesIO(row_bytes), allow_pickle=False) as z:
        need(int(z["image_id"]) == entry["image_id"], "Captured native identity")
        rank, query, area, scores, boxes, logits = (
            z[key] for key in ("ranks","query_ids","areas","scores","boxes_xyxy","raw_logits"))
    need(len(rank) == entry["rows"] == len(query) == len(area) == len(scores)
         and rank.shape == query.shape == area.shape == scores.shape
         and boxes.shape == (len(rank),4) and logits.shape == (len(rank),144,144),
         "Fixed candidate carrier archive")
    return spatial, {int(r): (int(q), int(a), float(s), box, logit)
                     for r,q,a,s,box,logit in zip(rank,query,area,scores,boxes,logits,strict=True)}


def witnesses(fh, new):
    for key, row in new.items():
        payload = {"key":list(key), "diagnostics":row["diagnostics"]}
        for arm in "CP":
            rle = row[arm]
            counts = rle["counts"]
            payload[arm] = {"size":list(rle["size"]), "counts":counts.decode("ascii") if isinstance(counts,bytes) else counts}
        for arm in "FBJ":
            payload[arm] = row[arm].tolist()
        fh.write((json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("code","protocol","module-manifest","capture","capture-readback","old-support","old-score",
              "raw-capture","raw-acceptance","source-cohort","fit-gt","shapely-site","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--module-manifest-sha", required=True)
    ap.add_argument("--capture-readback-sha", required=True)
    a = ap.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and sys.version_info[:3] == (3,10,13) and np.__version__ == "1.26.4"
         and torch.__version__ == "2.10.0+cu130" and not torch.cuda.is_initialized(), "Pinned CPU runtime")
    need(a.output.is_absolute() and a.output.parent.is_dir() and not a.output.exists(), "New private work path")
    start = perf_counter()
    protocol = json.loads(pinned(a.protocol, PROTOCOL_SHA, 64*1024))
    need(protocol["phase"] == "FIT_ONLY" and protocol["FIT"] and not protocol["DEV"] and not protocol["TEST"],
         "Explicit source-only scientific scope")
    manifest = json.loads(pinned(a.module_manifest, a.module_manifest_sha, 64*1024))
    need(set(manifest) == {*SOURCES, Path(__file__).name} and
         sha256(Path(__file__).read_bytes()).hexdigest() == manifest[Path(__file__).name],
         "This runner and all new experimental modules pinned")
    local = Path(__file__).parent
    heads, infer, scoring = (load(local/n, manifest[n]) for n in SOURCES)
    previous = "RFDETR_QUERY_FEATURE_FIT109_20261121/"
    old = load(a.code/(previous+"run_fit109_query.py"), "97fdc3987462aabbcd7d4fbe672c9788a3b969a221243e13d82473093170a15a")
    spot = old.load(a.code/"RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/run_spot.py")
    former = a.code/"RFDETR_MATCHED_POINT_FORMAL_20261120"
    prior, plan = (old.load(former/n) for n in ("run_fit109.py","fit_plan.py"))
    score_base = old.load(a.code/(previous+"fit109_score.py"))
    shared = a.code/"RFDETR_POINTREND_SUBDIVISION_20261121"
    adapter = load(shared/"roi_point_head.py","672bdd2aed8c0056f29c816f829bae9ceae5f9a685309beebf107aa59e547274")
    render = load(shared/"pointrend_subdivision.py","6c6be609317dc7960ba901b690e661d92588d88ee1f3210c9156c81a4d3219ad")
    budgeter = load(shared/"matched_budget.py","4ca4207435c03cbd45567c4aff470f21b892c2d6889b5fad4b0f2b6e94316b7a")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    accepted, capture_index = admit_capture(a, protocol)
    a.capture_source = a.capture
    a.capture, a.acceptance, a.acceptance_sha256 = a.raw_capture, a.raw_acceptance, old.OLD_RAW_ACCEPTANCE
    roster, _ = prior.read_population(a, spot)
    a.capture = a.capture_source
    need(set(capture_index) == {item["image_id"] for item,_,_ in roster}, "Same captured source images")
    helpers = score_base.load_helpers(a.code, a.shapely_site)
    pools, rows = collect_old_support(a, SimpleNamespace(old=old,spot=spot), roster, protocol)
    budgets, folds = read_budgets(a.old_score, protocol, rows, plan, roster, helpers.geometry)
    a.output.mkdir()
    pre = write(a.output/"PRELABEL_SAFE.json", dict(status="POINTREND_FIT109_PRELABEL_BUDGET_REVERIFIED",
                capture_readback_sha256=a.capture_readback_sha, protocol_sha256=PROTOCOL_SHA,
                native_budget_sha256=protocol["prior_oof_budget_sha256"], candidates=len(rows),
                swaps=sum(r["k"] for r in budgets.values()), FIT=True, DEV=False, TEST=False,
                optimizer_updates=0, GT_opened=False))
    print(json.dumps(dict(event="FIT109_POINTREND_PRELABEL_CLOSED", sha256=pre)), flush=True)

    gt, attached = old.attach_fit(a,roster,rows,pools,helpers,spot)
    truth = {ann["id"]:ann for ann in gt["annotations"]}
    positive = [dict(r,k=budgets[helpers.geometry.identity(r)]["k"]) for r in attached
                if r["role"] == "valid" and r["truth"] is not None and len(r["old14"]) > 0]
    need(len(positive) == protocol["data"]["supervised_matches"], "Exact original supported training matches")

    @lru_cache(maxsize=12)
    def get_image(iid):
        return captured_arrays(a.capture, capture_index[iid])

    @lru_cache(maxsize=96)
    def get_label(aid):
        ann = truth[aid]
        return codec.decode(codec.merge(codec.frPyObjects(ann["segmentation"],800,800))).astype(bool)

    def load_roi(row):
        spatial, values = get_image(row["image_id"])
        query, _, _, box, logits = values[row["rank"]]
        need(query == row["query_id"], "Fixed training ROI identity")
        return spatial, logits, box, get_label(row["matched_gt_id"])

    job = perf_counter()
    fitted, audit = heads.train_oof_heads(positive, folds, load_roi, adapter=adapter,
                                         seed=protocol["seed"], steps=protocol["optimization"]["steps_per_fold"],
                                         cpu_threads=4)
    fit_seconds = perf_counter()-job
    need(set(fitted) == set(audit) == set(range(5)), "Complete held-out head fits")
    updates = sum(row["updates"] for row in audit.values())
    targets = sum(row["supervised_points"] for row in audit.values())
    need(updates == 5*protocol["optimization"]["steps_per_fold"]
         and targets == updates*protocol["optimization"]["points_per_roi"], "Executed optimizer and label budget")
    train_sha = write(a.output/"HEAD_AUDITS_PRIVATE.json", audit)
    for fold, head in sorted(fitted.items()):
        with (a.output/f"HEAD_FOLD_{fold}_PRIVATE.npz").open("xb") as f:
            np.savez(f, **{k:v.detach().numpy() for k,v in head.state_dict().items()})
    print(json.dumps(dict(event="FIT109_POINTREND_HEADS_FIT", folds=5, seconds=round(fit_seconds,2))), flush=True)
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    all_outputs, counts = {}, Counter()
    job = perf_counter()
    witness_path = a.output/"PREDICTION_WITNESS_PRIVATE.jsonl"
    with witness_path.open("xb") as stream:
        for item, ordinal, _ in roster:
            iid = item["image_id"]
            spatial, values = get_image(iid)
            row_inputs = []
            stock = {c["rank"]:c for c in pools[iid]}
            for row in by_image[iid]:
                rank = row["rank"]
                query, area, score, box, logits = values[rank]
                need(query == row["query_id"] and area == row["area"] and score == stock[rank]["score"],
                     "Exact stock score, area and identity")
                row_inputs.append(dict(row, box_xyxy01=box, raw_logits=logits,
                                       rle=stock[rank]["rle"], k=budgets[helpers.geometry.identity(row)]["k"]))
            result, totals = infer.infer_image(fitted[folds[item["group"]]], spatial, row_inputs,
                                              adapter=adapter, render=render, budget=budgeter, chunk_size=12)
            all_outputs.update(result)
            witnesses(stream, result)
            counts.update(totals)
            if ordinal % 12 == 0 or ordinal == len(roster):
                print(json.dumps(dict(event="FIT109_POINTREND_HELD_IMAGES", images=ordinal)), flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    infer_seconds = perf_counter()-job
    witness_sha = sha256(witness_path.read_bytes()).hexdigest()
    need(counts["rows"] == len(all_outputs) == len(rows)
         and counts["point_evaluations"] == protocol["subdivision"]["point_evaluations_per_valid_roi"]*counts["rendered"],
         "Complete held predictions and exact realizations")
    job = perf_counter()
    metrics = scoring.evaluate(gt,pools,attached,all_outputs,budgets,helpers,score_base.pooled,protocol)
    report = dict(status="RFDETR_POINTREND_FIT109_OOF_EVALUATION_COMPLETE", FIT=True, DEV=False, TEST=False, AP=True,
        model_forward_calls=0, frozen_carrier_image_forwards=109, point_head_updates=updates,
        supervised_points=targets, n_params=sum(p.numel() for p in fitted[0].parameters()),
        held_sources=109, seed=protocol["seed"], source_group_holdout_for_point_head_only=True, gt_training_covered_by_fit=True,
        protocol_sha256=PROTOCOL_SHA, capture_readback_sha256=a.capture_readback_sha, prelabel_sha256=pre,
        source_manifest_sha256=a.module_manifest_sha, native_budget_sha256=protocol["prior_oof_budget_sha256"],
        prediction_witness_sha256=witness_sha, train_audit_sha256=train_sha, inference_counts=dict(counts),
        fit_seconds=fit_seconds, inference_seconds=infer_seconds, score_seconds=perf_counter()-job,
        wall_seconds=perf_counter()-start, metrics=metrics)
    digest=write(a.output/"RESULT_SAFE.json",report)
    print(json.dumps(dict(status=report["status"], result_sha256=digest, advance=metrics["advance"])), flush=True)


if __name__ == "__main__":
    main()
