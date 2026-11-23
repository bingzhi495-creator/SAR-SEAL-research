"""Replay the frozen HRSID point heads on the second detector's own Instance-RSDD masks."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType


LOCK = "f7ac19ff63c2c84fb5e2aac772f5efa0bf16a8e8ac519b0a2469d74b6bfd05cf"
CHECKPOINT = "36205b47714057ee24a3916f2bc0d307779ad4aa60e5b39b949bbdf381f31054"
CAPTURE_WORKER = "56fe4d823171c5e7652637aec28a7aad340046ff2772a7e320f9b9dff9d5d8fc"
COMMON = "0ccbbb90640990481f25b4aae5e91c877ef9634f05c826f6638a6b12e2ee4f9a"
STATUS = "RSDD_CARRIER5203_REPLAY_COMPLETE"
SEEDS = (5217,5218,5219)
MODES = ("fixed", "free_matched_count", "free_natural", "learned")
ARMS = ("dynamic", "static") + tuple(f"{seed}/{mode}" for seed in SEEDS for mode in MODES)
RUNTIME = {"torch":"2.10.0+cu130", "numpy":"1.26.4", "Pillow":"12.3.0",
           "pycocotools":"2.0.11", "rfdetr":"1.12.0"}


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def module(path, expected):
    raw = path.read_bytes()
    check(sha256(raw).hexdigest() == expected, "Historical source differs: " + path.name)
    obj = ModuleType(path.stem)
    obj.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), obj.__dict__)
    return obj


def encode(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def initialize(a, items):
    global w, edit
    common = module(a.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123/replay_quota_swap.py", COMMON)
    common.initialize(a, items)
    w = common.w
    edit = module(a.code / "UEIKAP_RFDETR_CARRIER_20261122/edit_rf.py", common.EDIT)
    check(set(w.models) == set(SEEDS), "Frozen source head identities differ")


def policy_orders(stock, coords, features):
    np = w.np
    cap = min(8, int(stock.sum()) // 10, len(coords))
    dynamic, static = w.methods.attraction.select(stock, coords, cap)
    orders = dict(dynamic=dynamic, static=static)
    check(len(dynamic) == len(static) <= cap, "Geometry control quotas differ")
    for seed in SEEDS:
        head, center, scale = w.models[seed]
        q = w.step.predict(head, features, center, scale)
        check(q.shape == coords.shape and q.dtype == np.float32 and np.isfinite(q).all()
              and np.all((q >= 0) & (q <= 1)), "Original source posterior required")
        gain = q[:, 0] - q[:, 1]
        fixed = [(i,i) for i in w.methods.pair.choose_pairs(gain, cap)]
        expected_fixed = [(i,i) for i in sorted(range(len(gain)), key=lambda i: (-float(gain[i]),i))
                          if gain[i] > 0][:cap]
        check(fixed == expected_fixed, "Strict positive paired gain or historical tie order differs")
        orders[f"{seed}/fixed"] = fixed
        budgets = (("free_matched_count",len(fixed)), ("free_natural",None), ("learned",len(dynamic)))
        for name, count in budgets:
            actual = w.methods.free.choose_free(q, cap, count)
            check(actual == edit.ordered_pairs(q, cap, count),
                  "Independent reconstruction of signed policies differs")
            orders[f"{seed}/{name}"] = actual
    return orders


def masks_for(stock, stock_rle, coords, orders, counts):
    np = w.np
    result, known = {}, {}
    for arm, choices in orders.items():
        if not choices:
            continue
        adds = tuple(sorted(int(coords[i,0]) for i,_ in choices))
        drops = tuple(sorted(int(coords[j,1]) for _,j in choices))
        change = known.get((adds,drops))
        if change is None:
            k = len(choices)
            check(k == len(set(adds)) == len(set(drops)) and not stock.flat[list(adds)].any()
                  and stock.flat[list(drops)].all(), "Signed selected locations must be distinct on each side")
            change = w.base.mask_edit(stock,coords,choices)
            after = w.codec.decode(change["rle"]).astype(bool)
            added = np.flatnonzero(after & ~stock)
            removed = np.flatnonzero(stock & ~after)
            check(change["swaps"] == k and np.array_equal(added,adds) and np.array_equal(removed,drops)
                  and int(w.codec.area(change["rle"])) == int(w.codec.area(stock_rle)), "Signed native XOR differs")
            known[(adds,drops)] = change
        result[arm] = change
        counts[arm+"_candidates"] += 1
        counts[arm+"_swaps"] += change["swaps"]
    return result


def process_image(pair):
    audit, pred = pair
    np = w.np
    image_id = audit["image_id"]
    check(pred["image_id"] == image_id and audit["original_hw"] == [512,512], "Original native grid differs")
    rgb = w.base.load_rgb(w.a.image_root, audit)
    image = w.methods.pair.image_features(rgb)
    # As in the original CFA replay, load_rgb first imports the qualified RF-DETR image decoder.
    check(w.torch.get_float32_matmul_precision() == "high", "Original occupancy numerical precision required")
    planes = w.base.load_array(w.a.capture, audit["logits_ref"])
    logits = dict(zip(audit["logit_ranks"], planes, strict=True))
    check(planes.shape == (len(logits),144,144), "Eligible native logit plane count differs")
    overrides = {x["native_rank"]:x for x in audit["posterior_overrides"]}
    check(len(overrides) == len(audit["posterior_overrides"]) and set(overrides) <= set(logits),
          "Invalid native CUDA posterior override")
    counts = Counter(images=1,mapped_candidates=len(pred["coco"]),eligible_planes=len(logits),overrides=len(overrides))
    changes = {}
    for pos, (item,det) in enumerate(zip(audit["candidates"],pred["coco"],strict=True),1):
        rank, area = item["native_rank"], int(w.codec.area(det["segmentation"]))
        check((item["position"],rank,item["query_id"],item["score"]) ==
              (pos,pred["native_ranks"][pos-1],audit["query_ids"][pos-1],det["score"]) and area == item["mask_area"]
              and (rank in logits) == (32 <= area < 1024), "Original rank, query, score or area gate differs")
        if rank not in logits:
            continue
        stock = w.codec.decode(det["segmentation"]).astype(bool)
        override = overrides.get(rank)
        check(override is None or override["query_id"] == item["query_id"], "Native override query differs")
        probability = w.base.native_probability(logits[rank], stock,
                          None if override is None else w.base.load_array(w.a.capture, override))
        box = np.asarray(item["box_xyxy"],dtype=np.float32)
        check(box.shape == (4,) and np.isfinite(box).all() and np.all((0 <= box) & (box <= 512))
              and np.all(box[:2] <= box[2:]), "Invalid native ROI")
        if np.any(box[:2] == box[2:]):
            w.base.check_support(w.methods, stock, box, probability, np.empty((0,2),np.int32))
            counts["degenerate"] += 1
            continue
        data = w.pair.paired_points(stock, box, area, det["score"], probability, image, w.sampler)
        coords = data[0] if data is not None else np.empty((0,2),np.int32)
        w.base.check_support(w.methods, stock, box, probability, coords)
        if not len(coords):
            counts["empty_support"] += 1
            continue
        check(coords.ndim == 2 and coords.shape[1] == 2 and len(np.unique(coords)) == coords.size,
              "Original endpoint identities must all be distinct")
        actions = policy_orders(stock, coords, data[1])
        edits = masks_for(stock, det["segmentation"], coords, actions, counts)
        if edits:
            changes[str(pos)] = edits
        counts["editable_supports"] += 1
        counts["point_head_forwards"] += len(SEEDS)
    return dict(image_id=image_id,changes=changes), counts


def run(a):
    started = perf_counter()
    check(sys.flags.isolated and sys.dont_write_bytecode and sys.version_info[:3] == (3,10,13)
          and all(version(k) == v for k,v in RUNTIME.items()), "Qualified isolated B5 CPU runtime required")
    check(sha256(a.protocol.read_bytes()).hexdigest() == a.protocol_sha256 == LOCK and not a.output.exists(),
          "Exact new protocol and fresh destination required")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    cap_path = Path(__file__).with_name("capture.py")
    if not cap_path.is_file():
        cap_path = a.code / "INSTANCE_RSDD_CARRIER5203_20261123/capture.py"
    cap = module(cap_path,CAPTURE_WORKER)
    _, _, intake, items, lineage = cap.load_population(a.code,a.stock_root,a.cohort)
    source, pairs, hashes = cap.read_capture(a.capture,a.capture_terminal,items,lineage)
    check(source["phase"] == "capture" and source["status"] == "RSDD_CARRIER5203_CAPTURE_COMPLETE"
          and source["protocol_sha256"] == LOCK and source["checkpoint_sha256"] == CHECKPOINT
          and source["worker_sha256"] == CAPTURE_WORKER and source["cohort"] == a.cohort
          and source["images"] == len(items) == a.cohort and source["target_labels_opened"] is False,
          "Closed captured cohort or carrier differs")
    a.image_root = intake / "images"
    common = module(a.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123/replay_quota_swap.py",COMMON)
    a.output.mkdir()
    counts, digest = Counter(), sha256()
    with (a.output / "EDITS_PRIVATE.jsonl").open("xb") as out, ProcessPoolExecutor(
            max_workers=a.workers, initializer=initialize, initargs=(a,items)) as pool:
        for j,(row,tally) in enumerate(pool.map(process_image,pairs,chunksize=2),1):
            check(row["image_id"] == items[j-1]["image_id"], "Replay changed native image order")
            line = encode(row)
            check(out.write(line) == len(line), "Incomplete native ledger")
            digest.update(line)
            counts.update(tally)
            if j == 1 or j % 50 == 0 or j == len(items):
                print(json.dumps(dict(phase="replay",images=j,total_images=len(items))),flush=True)
    check(counts["images"] == a.cohort and counts["mapped_candidates"] == source["counts"]["mapped"]
          and counts["eligible_planes"] == source["counts"]["eligible"] and counts["overrides"] == source["counts"]["overrides"]
          and counts["eligible_planes"] == counts["degenerate"]+counts["empty_support"]+counts["editable_supports"]
          and counts["point_head_forwards"] == len(SEEDS)*counts["editable_supports"], "Incomplete native replay population")
    arm_counts = {}
    for arm in ARMS:
        k = counts.pop(arm+"_swaps",0)
        arm_counts[arm] = dict(changed_candidates=counts.pop(arm+"_candidates",0),swaps=k,adds=k,drops=k)
    for seed in SEEDS:
        check(arm_counts[f"{seed}/fixed"] == arm_counts[f"{seed}/free_matched_count"]
              and arm_counts[f"{seed}/learned"] == arm_counts["dynamic"] == arm_counts["static"],
              "Realized equal signed counts differ")
    for k in ("degenerate","empty_support","editable_supports","point_head_forwards"):
        counts.setdefault(k,0)
    readback = sha256()
    with (a.output / "EDITS_PRIVATE.jsonl").open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            readback.update(chunk)
    check(readback.digest() == digest.digest(), "On-disk native ledger readback differs")
    result = dict(status=STATUS,phase="replay",cohort=a.cohort,source_partition=source["source_partition"],
                  images=a.cohort,seeds=list(SEEDS),arms=list(ARMS),input_lineage=lineage,**hashes,
                  predictions_sha256=source["predictions_sha256"],audit_sha256=source["audit_sha256"],
                  candidate_audit_sha256=source["audit_sha256"],manifest_sha256=source["manifest_sha256"],
                  roster_sha256=source["roster_sha256"],edits_sha256=digest.hexdigest(),worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  protocol_sha256=LOCK,checkpoint_sha256=CHECKPOINT,frozen_head_sha256=common.HEADS,counts=dict(counts),
                  counts_by_arm=arm_counts,exact_support_xor_area=True,independent_signed_orders_exact=True,
                  per_candidate_fixed_matched_budget_equal=True,per_candidate_dynamic_matched_budget_equal=True,
                  target_labels_opened=False,GT_opened=False,detector_forwards=0,optimizer_updates=0,
                  point_head_forwards=counts["point_head_forwards"],workers=a.workers,runtime=RUNTIME,seconds=perf_counter()-started)
    data = encode(result)
    with (a.output / "RESULT_SAFE.json").open("xb") as out:
        check(out.write(data) == len(data), "Incomplete replay receipt")
    print(json.dumps(dict(status=STATUS,result_sha256=sha256(data).hexdigest())),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--cohort",type=int,choices=(74,740),required=True)
    p.add_argument("--workers",type=int,default=4,choices=range(1,9))
    p.add_argument("--protocol-sha256",required=True)
    for key in ("code","stock-root","capture","capture-terminal","weights-dir","protocol","output"):
        p.add_argument("--"+key,type=Path,required=True)
    a = p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):
            setattr(a,key,value.resolve(strict=key!="output"))
    run(a)


if __name__ == "__main__":
    main()
