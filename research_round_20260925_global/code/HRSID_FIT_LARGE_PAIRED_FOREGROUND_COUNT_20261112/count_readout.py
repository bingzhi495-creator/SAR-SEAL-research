"""Compare fixed large-ship query geometry at each arm's original foreground count."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask
import torch

READOUT_SHA = "1ba6728b6cef1eead3e94d79353351b80a0f58875964f46a87e12f87f13c1763"
EXACT_BASELINE_PREFIX = (89.510, 88.185)
METHODS = ("native_native", "center_native", "native_center", "center_center")


def require(condition, why):
    if not condition:
        raise ValueError(why)


def top_area(logits, count):
    flat = logits.reshape(-1)
    require(0 <= count <= len(flat) and np.isfinite(flat).all(), "Finite bounded top count")
    if count == 0:
        return np.zeros_like(logits, dtype=bool)
    threshold = np.partition(flat, len(flat)-count)[len(flat)-count]
    picked = flat > threshold
    remainder = count - np.count_nonzero(picked)
    picked[np.flatnonzero(flat == threshold)[:remainder]] = True
    require(int(picked.sum()) == count, "Exact foreground count")
    return picked.reshape(logits.shape)


def metrics(prediction, truth):
    area = int(truth.sum())
    inter = int((prediction & truth).sum())
    selected = int(prediction.sum())
    return dict(iou=inter/(area+selected-inter), fp_gt=(selected-inter)/area, fn_gt=(area-inter)/area)


def independent_fixture():
    sample = np.asarray([[9, 1, 1], [1, -1, -2]], dtype=np.float32)
    for k in range(sample.size+1):
        ordered = np.argsort(-sample.ravel(), kind="stable")[:k]
        reference = np.zeros(sample.size, dtype=bool)
        reference[ordered] = True
        require(np.array_equal(reference.reshape(sample.shape), top_area(sample, k)), "Independent tie comparison")
    return dict(status="FIXED_COUNT_TIE_FIXTURE_PASS", cases=7)


def emit(path, value):
    encoded = (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)+"\n").encode()
    with path.open("xb") as handle:
        handle.write(encoded)
    return sha256(encoded).hexdigest()


def summary(rows, groups):
    values = ("iou", "fp_gt", "fn_gt")
    average = lambda records, key, field: mean(row["scores"][key][field] for row in records) * 100
    per_group = {group: [row for row in rows if row["group"] == group] for group in groups}
    result = {
        "group_counts": {group: len(group_rows) for group, group_rows in per_group.items()},
        "object_equal": {k: {f: average(rows,k,f) for f in values} for k in METHODS},
        "prefix_equal": {k: {f: mean(average(g,k,f) for g in per_group.values()) for f in values} for k in METHODS},
        "image_equal": {},
        "contrasts": {},
    }
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    result["image_equal"] = {k: {f: mean(average(g,k,f) for g in by_image.values()) for f in values} for k in METHODS}
    for tag, b, a in (("stock", "center_center", "native_native"), ("native_count", "center_native", "native_native"),
                      ("center_count", "center_center", "native_center")):
        deltas = [average(records,b,"iou") - average(records,a,"iou") for records in per_group.values()]
        loo = [mean(deltas[:i] + deltas[i+1:]) for i in range(len(groups))]
        result["contrasts"][tag] = dict(prefix_equal=mean(deltas), prefix_values=deltas, leave_one_prefix=loo,
            object_equal=result["object_equal"][b]["iou"]-result["object_equal"][a]["iou"],
            image_equal=result["image_equal"][b]["iou"]-result["image_equal"][a]["iou"],
            fp_gt_prefix=result["prefix_equal"][b]["fp_gt"]-result["prefix_equal"][a]["fp_gt"],
            fn_gt_prefix=result["prefix_equal"][b]["fn_gt"]-result["prefix_equal"][a]["fn_gt"])
    return result


def run(output, protocol_sha):
    start = perf_counter()
    root = output.parent.parent
    draft = Path(__file__).parent
    protocol = (draft/"PROTOCOL.md").read_bytes()
    require(sha256(protocol).hexdigest() == protocol_sha, "Protocol freeze digest")
    source_path = root/"P381_HRSID_FIT_LARGE_FIXED_QUERY_READOUT_DRAFT_PRIVATE_20261111_01"/"readout.py"
    raw = source_path.read_bytes()
    require(sha256(raw).hexdigest() == READOUT_SHA, "Exact closed P381 readout source")
    readout = ModuleType("_fixed_count_closed_readout")
    readout.__file__ = str(source_path)
    exec(compile(raw,str(source_path),"exec"), readout.__dict__)
    previous = readout.load_module("_fixed_count_closed_p379",
        readout.folder(readout.P379_NAME,"DRAFT")/"score.py",
        "8741bebb448120e37e0c10b58467a83d724f2294d20fddbc93720301285084aa")
    capture = readout.load_module("_fixed_count_closed_p380",
        readout.folder(readout.P380_NAME,"DRAFT")/"capture.py",
        "ddd08360cd51cc39e8ade9ea22a566026e1f74967e48688075a4a52b8d24fd9a")
    stocks, captures, old = readout.read_sources(previous)
    roster = json.loads(readout.pinned(previous.P378_DRAFT/"IMAGE_ONLY_MANIFEST_PRIVATE.json",previous.MANIFEST_SHA,65536))
    require(roster["FIT"] and roster["DEV"] is roster["TEST"] is False, "Original FIT-only roster")
    frozen = json.loads(readout.pinned(previous.INDEX,previous.INDEX_SHA,32*1024**2))
    ids = {r["image_id"] for r in roster["images"]}
    images = {i["id"]:i for i in frozen["images"] if i["id"] in ids}
    anns = defaultdict(list)
    for ann in frozen["annotations"]:
        if ann["image_id"] in ids:
            anns[ann["image_id"]].append(ann)
    rows = []
    reference = {arm:{r["annotation_id"]:r for r in old[arm]} for arm in previous.SIDES}
    with torch.inference_mode():
        for ix,item in enumerate(roster["images"]):
            iid = item["image_id"]
            group = item["native_basename"].split("_",1)[0]
            maps, anchors, all_targets = {}, {}, {}
            large = [r for r in anns[iid] if float(r["area"]) >= 9216]
            for arm in previous.SIDES:
                block = stocks[arm][0][ix]
                directory,entries = captures[arm]
                entry = entries[ix]
                require(iid == entry["image_id"] == block["image_id"], "Bound common image order")
                pool, _, _ = previous.ranked_candidates(block)
                match, rles, _ = previous.freeze_coco(images[iid], anns[iid], pool)
                by_rank = {candidate["rank"]:candidate for candidate in pool}
                packet = readout.pinned(directory/"FEATURES_PRIVATE"/entry["file"], entry["feature_sha256"], readout.PACKET_BYTES)
                features, query, bias = capture.restore_packet(packet,readout.SHAPES,entry["native_tensor_strides"],torch)
                native = (torch.einsum("bchw,bnc->bnhw",features,query)+bias)[0]
                maps[arm], anchors[arm], all_targets[arm] = {}, {}, {}
                for target in large:
                    aid = target["id"]
                    orig = reference[arm][aid]
                    rank = match.get(aid)
                    require(rank is not None and orig["group"] == group and orig["anchor_rank"] == rank, "Fixed P379 matching")
                    candidate = by_rank[rank]
                    image_map = readout.interpolate(native[candidate["query"]],readout.SIZE)
                    anchor = mask.decode(candidate["rle"]).astype(bool)
                    require(np.array_equal(image_map>0.,anchor), "Exact original selected query mask")
                    maps[arm][aid], anchors[arm][aid] = image_map, anchor
                    all_targets[arm][aid] = mask.decode(rles[aid]).astype(bool)
            for target in large:
                aid = target["id"]
                truth = all_targets["native"][aid]
                require(np.array_equal(truth,all_targets["center"][aid]) and truth.any(), "Exact shared nonempty GT")
                scores = {}
                counts = {arm:int(anchors[arm][aid].sum()) for arm in previous.SIDES}
                for arm in previous.SIDES:
                    for donor in previous.SIDES:
                        pred = top_area(maps[arm][aid],counts[donor])
                        if arm == donor:
                            require(np.array_equal(pred,anchors[arm][aid]),"Own count preserves stock anchor")
                        scores[f"{arm}_{donor}"] = metrics(pred,truth)
                rows.append(dict(image_id=iid,annotation_id=aid,group=group,counts=counts,scores=scores))
    groups = sorted({r["group"] for r in rows})
    require(len(ids)==28 and len(rows)==50 and len(groups)==4, "Exact prespecified target count")
    result = summary(rows,groups)
    base = result["prefix_equal"]
    stock_check = (base["native_native"]["iou"],base["center_center"]["iou"])
    require(all(abs(x-y)<.0006 for x,y in zip(stock_check,EXACT_BASELINE_PREFIX)), "Same previous four-prefix stock aggregates")
    require(not torch.cuda.is_initialized(),"CPU only")
    compar = result["contrasts"]
    fixed = [compar["native_count"],compar["center_count"]]
    persistent = all(c["prefix_equal"] <= -.50 and sum(v<0 for v in c["prefix_values"]) >= 3
                    and max(c["leave_one_prefix"]) < 0 for c in fixed)
    compatible = compar["stock"]["prefix_equal"] <= -.50 and all(abs(c["prefix_equal"]) <= .25 for c in fixed)
    gate = "PERSISTENT_RANKING_DEFICIT" if persistent else "COUNT_SENSITIVITY_COMPATIBLE" if compatible else "MIXED"
    audit_sha = emit(output.parent/"ROW_AUDIT_PRIVATE.json",rows)
    result.update(status="HRSID_FIT_LARGE_PAIRED_FOREGROUND_COUNT_ACTUAL", FIT=True, DEV=False, TEST=False, AP=False,
        model_forward_calls=0, optimizer_updates=0, images=28, large_targets=50, gate=gate, protocol_sha256=protocol_sha,
        private_row_audit_sha256=audit_sha, worker_seconds=perf_counter()-start)
    result_sha = emit(output,result)
    print(json.dumps(dict(status=result["status"],result_sha256=result_sha)),flush=True)


def offline(event, _):
    if event.startswith("socket."):
        raise RuntimeError("Frozen FIT-only worker is offline")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--protocol-sha")
    args = parser.parse_args()
    sys.addaudithook(offline)
    torch.set_num_threads(1)
    if args.check:
        print(json.dumps(independent_fixture()))
    else:
        run(args.output,args.protocol_sha)

