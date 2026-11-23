"""Choose Phoenix signed edit counts on the archived endpoints without opening target labels."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from math import fsum
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from pycocotools import mask as codec


ARMS = ("balance", "relaxed", "free")
PROTOCOL = "ae73858a9dec877575a800616b8da17b73717358042b81aa86016678b42db57e"
OLD_PROTOCOL = "0160a2ee707bb95d2c8342cec229cc0c06e0edb4d9860e857701c23cf6b61052"
SOURCE = {
    "request": ("08ba5d901c7c6d0b17a8b17cd8a3bb5f646139e66e2aae2459554cacb0fabd2a",
                "f0e8fbaa487e0819c5f6e7ac2ecaadcee86edbf7d871accb27600021089008de"),
    "inference": ("ec061d588abcef68d3ced78296b3be5ca9b3f9ad93b3d1138ed90d5b6e6e7ddc",
                  "7b643eeb7b73c2c56bff6b22d58a32945e3d392f20ad332296bd7374290196d3"),
    "edits": ("fe94442d5e58c5b2568269f9cefdc5dbb86e307da43cde6b5368db09298786ec",
              "6684f7d858e40fbe78bc631baa153cf5a011461104df4348bdba4b48b8c89770"),
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def quota_choices(probabilities, k):
    q = np.asarray(probabilities, dtype=np.float64)
    require(q.ndim == 2 and q.shape[1] == 2 and np.isfinite(q).all() and np.all((0 <= q) & (q <= 1))
            and type(k) is int and 1 <= k <= len(q), "Expected a positive common budget and signed posterior rows")
    # These are the two monotone utility orders before centering. Exact posterior ties retain the archived ordinal.
    add_order = np.argsort(-q[:,0], kind="stable")
    drop_order = np.argsort(q[:,1], kind="stable")
    add_utility = q[add_order,0] - .5
    drop_utility = .5 - q[drop_order,1]
    feasible = []
    for adds in range(max(0, 2*k-len(q)), min(2*k, len(q))+1):
        drops = 2*k-adds
        utility = fsum([*add_utility[:adds], *drop_utility[:drops]])
        feasible.append((-utility, abs(adds-drops), adds))
    selected = {}
    for arm, allowance in zip(ARMS, (0, 2, 2*k),strict=True):
        neg, _, a = min(row for row in feasible if row[1] <= allowance)
        selected[arm] = add_order[:a], drop_order[:2*k-a], -neg
    return selected


def encoded(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def pinned(path, digest):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == digest, "Historical fingerprint mismatch for " + path.name)
    return data


def inputs(root):
    stages = {}
    for phase, (result_sha, terminal_sha) in SOURCE.items():
        receipt = json.loads(pinned(root/phase/"RESULT_SAFE.json", result_sha))
        end = json.loads(pinned(root/("observed_"+phase)/"TERMINAL.json", terminal_sha))
        require(end["phase"] == receipt["phase"] == phase and type(end["actual_exit"]) is int and end["actual_exit"] == 0
                and end["result_sha256"] == result_sha and end["output_footer_verified"] is True
                and receipt["target_labels_opened"] is False and receipt["protocol_sha256"] == OLD_PROTOCOL
                and receipt["worker_sha256"] == end["worker_sha256"], "Successful original phase required")
        stages[phase] = receipt
    request, actor, archive = (stages[k] for k in ("request","inference","edits"))
    require(request["request_sha256"] == actor["request_sha256"] == archive["request_sha256"]
            and archive["endpoint_sha256"] == actor["endpoint_sha256"] and archive["archived_arms_exact"] is True
            and archive["independent_ranking_exact"] is True and archive["inference_result_sha256"] == SOURCE["inference"][0]
            and request["input_lineage"] == actor["input_lineage"] == archive["input_lineage"],
            "Original endpoint and mask receipts do not agree")
    payload = json.loads(pinned(root/"request/REQUEST_PRIVATE.json", request["request_sha256"]))
    require(payload["input_lineage"] == archive["input_lineage"] and payload["protocol_sha256"] == OLD_PROTOCOL
            and payload["source_replay"] == archive["source_replay"] and len(payload["images"]) == 74,
            "Original source inventory differs")
    probs = pinned(root/"inference/ENDPOINT_PROBABILITIES_PRIVATE.jsonl", actor["endpoint_sha256"])
    edits = pinned(root/"edits/EDITS_PRIVATE.jsonl", archive["edits_sha256"])
    lineage = dict(request_result_sha256=SOURCE["request"][0], request_terminal_sha256=SOURCE["request"][1],
                   inference_result_sha256=SOURCE["inference"][0], inference_terminal_sha256=SOURCE["inference"][1],
                   archived_edits_result_sha256=SOURCE["edits"][0], archived_edits_terminal_sha256=SOURCE["edits"][1],
                   request_sha256=request["request_sha256"], endpoint_sha256=actor["endpoint_sha256"],
                   archived_edits_sha256=archive["edits_sha256"], input_lineage=archive["input_lineage"])
    return zip(payload["images"], map(json.loads, probs.splitlines()), map(json.loads, edits.splitlines()), strict=True), lineage


def render(candidate, endpoints, historical, by_arm):
    k = candidate["k"]
    old = historical["phoenix"]
    require(k == old["swaps"] and endpoints["position"] == candidate["position"], "Historical budget or position differs")
    native = codec.decode(candidate["segmentation"]).astype(bool)
    coords = np.asarray(candidate["coords"])
    q = np.asarray(endpoints["probabilities"], dtype=np.float64)
    require(native.shape == (512,512) and 32 <= native.sum() < 1024 and type(k) is int and
            1 <= k <= min(8, native.sum()//10, len(coords)) and len(coords) <= 32 and coords.dtype.kind in "iu"
            and coords.shape == q.shape == (len(coords),2) and np.all((coords >= 0) & (coords < native.size))
            and np.unique(coords).size == coords.size and not native.flat[coords[:,0]].any()
            and native.flat[coords[:,1]].all(), "Invalid signed native endpoint support")
    options = quota_choices(q,k)
    require(options["balance"][2] <= options["relaxed"][2] <= options["free"][2], "Nested objective violation")
    changes = {}
    for arm,(add_rows,drop_rows,utility) in options.items():
        mask = native.copy(order="C")
        mask.flat[coords[add_rows,0]] = True
        mask.flat[coords[drop_rows,1]] = False
        added, dropped = np.flatnonzero(mask & ~native), np.flatnonzero(native & ~mask)
        require(np.array_equal(added,np.sort(coords[add_rows,0])) and np.array_equal(dropped,np.sort(coords[drop_rows,1]))
                and len(added)+len(dropped) == 2*k, "Writeback did not preserve its signed choices")
        rle = codec.encode(np.asfortranarray(mask,dtype=np.uint8))
        require(np.array_equal(codec.decode(rle),mask), "COCO RLE readback failed")
        result = dict(size=list(mask.shape),counts=rle["counts"].decode("ascii"))
        if arm == "balance":
            require(result == old["rle"], "Balanced arm failed historical Phoenix mask parity")
        changes[arm] = dict(rle=result)
        a,d = len(added),len(dropped)
        delta = a-d
        selected = np.concatenate((q[add_rows,0]-.5,.5-q[drop_rows,1]))
        require(fsum(selected) == utility and int(mask.sum()) == int(native.sum())+delta, "Native area or utility mismatch")
        by_arm[arm].update(changed_candidates=1,adds=a,drops=d,flips=a+d,signed_net_area=delta,
                           changed_quota_vs_balance=int(delta != 0),unchanged_quota=int(delta == 0),
                           more_add_candidates=int(delta > 0),more_drop_candidates=int(delta < 0),
                           all_add_candidates=int(d == 0),all_drop_candidates=int(a == 0),
                           negative_selected=int(np.count_nonzero(selected < 0)),zero_selected=int(np.count_nonzero(selected == 0)),
                           positive_selected=int(np.count_nonzero(selected > 0)),zero_margin_candidates=int(utility == 0),
                           negative_margin_candidates=int(utility < 0),sum_signed_utility=utility,
                           sum_utility_gain_over_balance=utility-options["balance"][2])
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("matched-root","protocol","output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    require(args.protocol_sha256 == PROTOCOL and not args.output.exists(), "Fresh output and current protocol required")
    pinned(args.protocol,PROTOCOL)
    start=perf_counter()
    rows,lineage=inputs(args.matched_root)
    totals,by_arm=Counter(),{arm:Counter() for arm in ARMS}
    args.output.mkdir()
    ledger=args.output/"EDITS_PRIVATE.jsonl"
    stream_hash=sha256()
    image_ids=set()
    with ledger.open("xb") as out:
        for image,prob,history in rows:
            i=image["image_id"]
            require(i == prob["image_id"] == history["image_id"] and i not in image_ids, "Different historical images")
            image_ids.add(i)
            expected=sorted(map(int,history["changes"]))
            require(expected == [c["position"] for c in image["candidates"]], "Historical positive candidate roster differs")
            changes={}
            for pos,cand,endpoint in zip(expected,image["candidates"],prob["candidates"],strict=True):
                key=str(pos)
                changes[key]=render(cand,endpoint,history["changes"][key],by_arm)
                totals.update(positive_budget_candidates=1,endpoint_pairs=len(cand["coords"]),swaps_per_balanced_arm=cand["k"])
            raw=encoded(dict(image_id=i,changes=changes))
            require(out.write(raw) == len(raw), "Incomplete sparse image record")
            stream_hash.update(raw)
            totals["images"]+=1
            if totals["images"]==1 or totals["images"]%10==0 or totals["images"]==74:
                print(json.dumps(dict(phase="edits", images=totals["images"], candidates=totals["positive_budget_candidates"])),flush=True)
        out.flush()
        os.fsync(out.fileno())
    require(totals == dict(images=74,positive_budget_candidates=4107,endpoint_pairs=129621,swaps_per_balanced_arm=29900)
            and all(v["flips"] == 59800 and v["changed_candidates"] == 4107 for v in by_arm.values()),
            "Aggregated positive coverage or total flip count differs")
    totals["mapped_candidates"]=lineage["input_lineage"]["mapped_candidates"]
    totals["untouched_candidates"]=totals["mapped_candidates"]-totals["positive_budget_candidates"]
    totals["total_flips_per_arm"]=2*totals["swaps_per_balanced_arm"]
    require(totals["mapped_candidates"] == 7346 and by_arm["balance"]["adds"] == by_arm["balance"]["drops"] == 29900,
            "Native or historical signed census differs")
    edits_sha=stream_hash.hexdigest()
    pinned(ledger,edits_sha)
    status="RSDD74_PHOENIX_AREA_ABLATION_EDITS_COMPLETE"
    result=dict(status=status,phase="edits",protocol_sha256=PROTOCOL,worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                **lineage,edits_sha256=edits_sha,counts=dict(totals),counts_by_arm=by_arm,arms=list(ARMS),balanced_rles_exact=True,
                target_labels_opened=False,optimizer_updates=0,detector_forwards=0,phoenix_forwards=0,seconds=perf_counter()-start)
    data=encoded(result)
    with (args.output/"RESULT_SAFE.json").open("xb") as f:
        require(f.write(data) == len(data), "Incomplete receipt")
    print(json.dumps(dict(status=status,result_sha256=sha256(data).hexdigest())),flush=True)


if __name__ == "__main__":
    main()
