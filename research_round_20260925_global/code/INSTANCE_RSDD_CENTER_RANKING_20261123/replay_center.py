"""Freeze a shared native pixel-center menu and test PointRend paste padding."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np


PROTOCOL = "0c517bf9994d684fd2701036017e75346f4de05d20b600954b6ae8b0567b89d7"
COMMIT = "0811472445c3dd9a0bffaf7346733305b79ad4c0"
EXP_RELEASE = "INSTANCE_RSDD_EXPANDED_RANKING_20261123/replay_expanded.py"
EXP_SOURCE = "c5d96adec0fdc03aa51b2b098a9928391382d0a87ccf6e4ca2d3a04d57e229f5"
EXP_TERMINAL = "5cfd066a6425421a05751332b6942fbcc66df364333f8b24130de0878febca6b"
EXP_RESULT = "cb071251e94f8a82af74193d713ae534cd4b01126dbb0b381e1708b1de493f62"
ARMS = ("point_center","learned_center","point_center_border")
PREPARED = "INSTANCE_RSDD_CENTER_RANKING_PREPARED_V1"
ACCEPTED = "INSTANCE_RSDD_CENTER_RANKING_ACCEPTED_V1"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, expected):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == expected, "Pinned content: " + path.name)
    return raw


def module(path, digest):
    item = ModuleType(path.stem)
    item.__file__ = str(path)
    exec(compile(pin(path,digest),str(path),"exec"),item.__dict__)
    return item


def previous_inputs(a, old, pr):
    root = a.expanded_replay
    receipt, term = pr.closed(root/"observed_verify-edits/TERMINAL.json",EXP_TERMINAL,
                              root/"verify-edits/RESULT_SAFE.json",EXP_RESULT)
    need(term["phase"] == "verify-edits" and term["worker_sha256"] == EXP_SOURCE
         and receipt["status"] == old.ACCEPTED and receipt["exact_support_xor_area"] is True
         and receipt["exact_replayed_policy_choices"] is True and receipt["GT_opened"] is False
         and receipt["protocol_sha256"] == old.PROTOCOL, "Closed exact full-band replay")
    prior = json.loads(pin(root/"AUTHORITY_verify-edits_PRIVATE.json",receipt["input_authority_sha256"]))
    args = SimpleNamespace(**{k:(Path(v).resolve(strict=True) if k in {"code","images","source_replay","protocol"} else v)
                               for k,v in prior.items() if k not in {"prepare","prepare_terminal_sha","prepare_receipt_sha"}})
    need(args.code == a.code and args.images == a.images and args.support_sha == receipt["support_sha256"],
         "Reuse exact previous input lineage")
    point, occ, weight, point_receipt, population = old.load_inputs(args,pr)
    need(point_receipt["predictions_sha256"] == receipt["predictions_sha256"]
         and point_receipt["candidate_audit_sha256"] == receipt["candidate_audit_sha256"], "Same frozen native cohort")
    expanded = [json.loads(line) for line in pin(root/"prepare-edits/EDITS_PRIVATE.jsonl",receipt["edits_sha256"]).splitlines()]
    need(len(population) == len(expanded) == 740, "Original complete E readout")
    return args, point, occ, weight, receipt, list(zip(population,expanded,strict=True))


def initialize(auth, args, point, occ, weight):
    global old, w, math
    old = module(auth.code/EXP_RELEASE,EXP_SOURCE)
    old.initialize(args,point,occ,weight)
    w = old.worker
    math = module(Path(__file__).with_name("center_math.py"),auth.center_math_sha)


def replay_one(item):
    ((native_pair, basis, occ_edits), point_edits), expanded = item
    row, native = native_pair
    need(row["image_id"] == point_edits["image_id"] == expanded["image_id"]
         and tuple(row["original_hw"]) == (512,512) and not row["posterior_overrides"], "Pinned original image")
    hw = (512,512)
    logits = w.base.load_array(w.a.capture,row["logits_ref"])
    basis_map = w.base.load_array(w.a.basis,basis["spatial"])
    planes = dict(zip(row["logit_ranks"],logits,strict=True))
    pixels = w.m.pair.image_features(w.base.load_rgb(w.images,row))
    torch = w.torch
    occupancy_precision = torch.get_float32_matmul_precision()
    need(occupancy_precision == "high", "Historical occupancy FP32 precision")
    rows, identities = [], []
    counts = Counter(images=1,mapped_candidates=len(native["coco"]))
    for pos,(info,pred) in enumerate(zip(row["candidates"],native["coco"],strict=True),1):
        key = str(pos)
        former = occ_edits["changes"].get(key,{}).get("free_natural")
        k = former["swaps"] if former else 0
        rank,query,area = (info[name] for name in ("native_rank","query_id","mask_area"))
        box = np.asarray(info["box_xyxy"],dtype=np.float32)
        rle = pred["segmentation"]
        need(pos == info["position"] and info["score"] == pred["score"]
             and area == int(w.codec.area(rle)), "Original candidate confidence and RLE")
        if rank not in planes or np.any(box[2:] == box[:2]):
            need(k == 0 and key not in expanded["changes"], "Exact original fallback")
            continue
        stock = w.codec.decode(rle).astype(bool)
        prob = w.base.native_probability(planes[rank],stock,None)
        sides = w.base.expected_support(w.m.rank,stock,box,prob)
        coords = w.m.pair.pair_assignment(*sides,hw[1]) if len(sides[0]) else np.empty((0,2),np.int64)
        rows.append(dict(image_id=row["image_id"],rank=rank,query_id=query,raw_logits=planes[rank],rle=rle,
                         box_xyxy01=box/np.float32(512),coordinates=coords,k=k))
        identities.append((key,info,stock,prob,box,former))
    retained = old.RecordedRender(w.render)
    before = w.pr.RecordedBudget(w.budget)
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.inference_mode():
            rendered, detail = w.infer.infer_image(w.head,basis_map,rows,adapter=w.adapter,render=retained,
                                                    budget=before,image_hw=hw,chunk_size=12)
    finally:
        torch.set_float32_matmul_precision(occupancy_precision)
    need(detail["point_evaluations"] == 3136*len(rows), "Exact original render budget")
    counts["point_evaluations"] = detail["point_evaluations"]
    changes = {}
    for descriptor, fields, render, observed in zip(rows,identities,retained.logits,before.observed,strict=True):
        key, info, stock, prob, box, former = fields
        rle,coords,k = (descriptor[n] for n in ("rle","coordinates","k"))
        identity = tuple(descriptor[n] for n in ("image_id","rank","query_id"))
        original = point_edits["changes"].get(key,{})
        need(rendered[identity]["P"] == original.get("point_paste",{}).get("rle",rle), "Reproduce original full paste")
        if not k:
            need(key not in expanded["changes"], "No extra zero-budget row")
            continue
        history = expanded["changes"][key]
        indices, features = w.helper.expanded_features(w.m.rank,stock,box,info["mask_area"],info["score"],
                                                       prob,pixels,coords)
        q_l = w.helper.predict_sides(w.predict,features)
        box_tensor = torch.as_tensor(box,dtype=torch.float32)[None]
        q_b = old.point_sides(render,box_tensor,indices,observed[:2],len(coords),hw)
        for arm,sides,probabilities in (("point_expanded",indices,q_b),("learned_expanded",indices,q_l),
              ("learned_common_native_tie",tuple(x[:len(coords)] for x in indices),tuple(x[:len(coords)] for x in q_l))):
            edits = w.helper.select(sides,probabilities,k,tie="native")
            need(old.mask_edit(rle,stock,edits,k) == history[arm], "Reproduce archived equal-budget masks")
        mask = tuple(math.eligible(side,hw[1],box) for side in indices)
        center = tuple(side[use] for side,use in zip(indices,mask,strict=True))
        kc = min(k,*(len(side) for side in center))
        scores_l = tuple(q[use] for q,use in zip(q_l,mask,strict=True))
        scores_b = tuple(q[use] for q,use in zip(q_b,mask,strict=True))
        scores_z,scores_r,edge = math.border_and_zero(render,box_tensor,center,hw)
        need(all(np.array_equal(old_q,new_q) for old_q,new_q in zip(scores_b,scores_z,strict=True)),
             "Zero-padding sampler exactly reproduces archived E probabilities at retained centers")
        counts["positive_original_k"] += 1
        counts["positive_center_k"] += bool(kc)
        counts["clamped_candidates"] += (kc < k)
        counts["center_swaps"] += kc
        counts["original_swaps"] += k
        changes[key] = {}
        for sign,full,allowed,zero,border,edge_mask in zip(("add","drop"),indices,center,scores_z,scores_r,edge,strict=True):
            counts[f"candidates_points_{sign}"] += len(full)
            counts[f"center_points_{sign}"] += len(allowed)
            y, x = np.divmod(full,hw[1])
            counts[f"center_at_upper_edge_{sign}"] += int(((x+.5 == box[2]) | (y+.5 == box[3])).sum())
            counts[f"edge_footprint_{sign}"] += int(edge_mask.sum())
            counts[f"padding_score_changed_{sign}"] += int((zero!=border).sum())
        for arm, scores in zip(ARMS,(scores_b,scores_l,scores_r),strict=True):
            chosen = w.helper.select(center,scores,kc,tie="native")
            need(all(math.eligible(side,hw[1],box).all() for side in chosen), "Every realized edit is centered in the RoI")
            changes[key][arm] = old.mask_edit(rle,stock,chosen,kc)
            for side,used,possible,near in zip(("add","drop"),chosen,center,edge,strict=True):
                counts[f"{arm}_edited_edge_{side}"] += int(np.isin(used,possible[near]).sum())
        counts["padding_choice_changed"] += changes[key]["point_center"] != changes[key]["point_center_border"]
    need(set(changes) == set(expanded["changes"]), "Every previous positive-k candidate preserved")
    return dict(image_id=row["image_id"],changes=changes), counts


def run(a, out, phase, auth_sha):
    source = module(a.code/EXP_RELEASE,EXP_SOURCE)
    source.label_guard(a.images)
    pr = source.module(a.code/"INSTANCE_RSDD_POINTREND_TRANSFER_20261122/replay_pointrend.py",source.PRIOR,"_prior_point")
    args,point,occ,weight,expanded,pop = previous_inputs(a,source,pr)
    worker_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    independent = phase == "verify-edits"
    expected = None
    if independent:
        previous, term = pr.closed(a.prepare/"observed_prepare-edits/TERMINAL.json",a.prepare_terminal_sha,
                                    a.prepare/"prepare-edits/RESULT_SAFE.json",a.prepare_receipt_sha)
        need(term["phase"] == "prepare-edits" and term["worker_sha256"] == worker_sha
             and previous["status"] == PREPARED and previous["protocol_sha256"] == PROTOCOL
             and previous["center_math_sha256"] == a.center_math_sha and previous["GT_opened"] is False
             and previous["expanded_replay_receipt_sha256"] == EXP_RESULT, "Observed first replay must match")
        expected = pin(a.prepare/"prepare-edits/EDITS_PRIVATE.jsonl",previous["edits_sha256"]).splitlines(keepends=True)
        need(len(expected) == len(pop), "Complete prior ledger")
    started = perf_counter()
    pieces, totals = [], Counter()
    sensor = {"GF-3":Counter(),"TerraSAR-X":Counter()}
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(a,args,point,occ,weight)) as pool:
        for i,(output, counts) in enumerate(pool.map(replay_one,pop,chunksize=2)):
            audit = pop[i][0][0][0][0]
            need(output["image_id"] == audit["image_id"], "Original authoritative image order")
            content = source.encode(output)
            need(expected is None or expected[i] == content, "Identical independent mask row")
            pieces.append(content)
            totals.update(counts)
            sensor[audit["sensor"]].update(counts)
            if (i+1)%25 == 0:
                print(json.dumps(dict(images=i+1,center_swaps=totals["center_swaps"])),flush=True)
    for tally, historic in [(totals,expanded["counts"]),*((sensor[k],expanded["counts_by_sensor"][k]) for k in sensor)]:
        need(all(tally[k] == historic[k] for k in ("images","mapped_candidates","point_evaluations"))
             and (tally["positive_original_k"],tally["original_swaps"]) == (historic["positive_k"],historic["swaps"])
             and 0 <= tally["center_swaps"] <= tally["original_swaps"], "Full frozen population and inherited budget")
    raw = b"".join(pieces)
    sha = sha256(raw).hexdigest()
    need(not independent or (sha == previous["edits_sha256"] and totals == previous["counts"]
                              and sensor == previous["counts_by_sensor"]), "Exact independent edit ledger and counts")
    out.mkdir()
    if not independent:
        (out/"EDITS_PRIVATE.jsonl").write_bytes(raw)
    result = dict(status=ACCEPTED if independent else PREPARED,images=740,protocol_sha256=PROTOCOL,prelock_commit=COMMIT,
                  expanded_replay_receipt_sha256=EXP_RESULT,predictions_sha256=expanded["predictions_sha256"],
                  candidate_audit_sha256=expanded["candidate_audit_sha256"],center_math_sha256=a.center_math_sha,
                  worker_sha256=worker_sha,input_authority_sha256=auth_sha,edits_sha256=sha,
                  GT_opened=False,author_TEST=True,AP=False,detector_forwards=0,optimizer_updates=0,workers=a.workers,
                  exact_replayed_policy_choices=independent,exact_support_xor_area=True,counts=dict(totals),
                  counts_by_sensor={k:dict(v) for k,v in sensor.items()},seconds=perf_counter()-started)
    payload = source.encode(result)
    (out/"RESULT_SAFE.json").write_bytes(payload)
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(payload).hexdigest())),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("phase",choices=("prepare-edits","verify-edits"))
    p.add_argument("--authority",required=True,type=Path)
    p.add_argument("--authority-sha",required=True)
    p.add_argument("--output",required=True,type=Path)
    cli = p.parse_args()
    values = json.loads(pin(cli.authority,cli.authority_sha))
    paths = {"protocol","code","images","expanded_replay"}
    numbers = {"workers","center_math_sha"}
    if cli.phase == "verify-edits":
        paths.add("prepare")
        numbers |= {"prepare_terminal_sha","prepare_receipt_sha"}
    need(set(values) == paths|numbers and type(values["workers"]) is int and 1 <= values["workers"] <= 8
         and sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and cli.output.name == cli.phase and not cli.output.exists(), "Fresh CPU-only authority")
    a = SimpleNamespace(**{k:Path(v).resolve(strict=True) if k in paths else v for k,v in values.items()})
    pin(a.protocol,PROTOCOL)
    run(a,cli.output,cli.phase,cli.authority_sha)


if __name__ == "__main__":
    main()
