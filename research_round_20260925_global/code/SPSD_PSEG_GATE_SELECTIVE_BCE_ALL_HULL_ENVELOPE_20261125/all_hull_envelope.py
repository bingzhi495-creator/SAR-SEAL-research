"""Exact fixed-capacity all-hull envelope for archived BCE-robust drafts."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import itertools
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD: Path
LOCK = "c73a0a24eae29834e70cfda1544027be86f77504b89f5fda98f0b1468720ce75"
PLAN = "86baa9a69c543680a38abe51e3cce860fcfea7d37168012a398c8e762f3c4fbd"
PLAN_TERMINAL = "c35e49ed7c7ef42af5be2ed43ab0fff517e54f62ecbded22a8864abdb1c1c1d1"
WORKER = "344fea17198fd8a70128aaabfcc9b13dbe188bda503a761e3222011aaf87fce0"
STATUS = "SPSD_GATE_SELECTIVE_BCE_ALL_HULL_ENVELOPE_COMPLETE"
ALPHAS = (0, .25, .50, .75, 1)
THRESHOLD = .15
ARM = "bce_robust"
REFS = ("pseg", "spsd")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, f"Pinned exact bytes: {path.name}")
    return raw


def load_previous():
    source = OLD/"selective_bound.py"
    prev = ModuleType("_exact_closed_budget")
    prev.__file__ = str(source)
    exec(compile(pin(source, WORKER), str(source), "exec"), vars(prev))
    return prev


def admitted(prev):
    path = OLD/"observed_plan/TERMINAL.json"
    need(prev.terminal(path,"capture",WORKER,PLAN) == PLAN_TERMINAL, "Exact label-free quota terminal")
    plan = json.loads(pin(OLD/"plan/RESULT_SAFE.json",PLAN))
    need(plan["status"] == prev.STATUS_PLAN and plan["gate_labels_opened"] is False
         and plan["worker_sha256"] == WORKER and plan["protocol_sha256"] == prev.LOCK,
         "Exact prior label-free quota protocol")
    reader, records, native, groups, score, _ = prev.inputs()
    quota, counts = prev.quota_plan(native, records)
    need(b"".join(map(prev.enc,quota)) == pin(OLD/"plan/QUOTAS_PRIVATE.jsonl",plan["quota_stream_sha256"])
         and counts == plan["counts"] and counts[ARM]["eligible"] == 8706 and counts[ARM]["fixed_adoptions"] == 2844,
         "Recomputed immutable eligible ranks and signed decisions")
    return reader, records, native, groups, score, quota


def restored_geometry(truth, matches, native, codec):
    # Preserve the original annotation traversal and image-level averaging
    # to reproduce the terminal geometry to its existing double precision.
    image_values = defaultdict(list)
    image_deltas = defaultdict(dict)
    for entry in truth["annotations"]:
        iid = entry["image_id"]
        if entry["id"] not in matches:
            image_values[iid].append(np.zeros(2))
            continue
        known_image, rank = matches[entry["id"]]
        need(known_image == iid and rank not in image_deltas[iid], "One native .50 slot for each admitted hull")
        row = native[iid][rank-1]
        static = row["changes"].get("static",row["native"])
        learned = row["changes"].get(ARM,row["native"])
        iou = codec.iou([static,learned],[entry["segmentation"]],[0])[:,0]
        need(np.isfinite(iou).all() and np.all((iou>=0)&(iou<=1)), "Finite exact raster IoU")
        image_values[iid].append(iou)
        image_deltas[iid][rank] = float(iou[1] - iou[0])

    values, deltas = {}, {}
    for iid in native:
        n = len(image_values[iid])
        values[iid] = np.mean(image_values[iid],axis=0) if n else np.zeros(2)
        deltas[iid] = {rank:value/n for rank,value in image_deltas[iid].items()} if n else {}
    return values,deltas


def group_mean(image_values, groups):
    by_group = defaultdict(list)
    for iid,g in groups.items():
        by_group[g].append(image_values[iid])
    need(len(by_group) == 180, "All exact decoded RGB groups")
    return np.stack([np.asarray(by_group[g]).mean(axis=0) for g in sorted(by_group)]).mean(axis=0)*100


def optimize(strata, delta, alpha):
    weights = np.array([alpha,1-alpha])
    adopted = []
    for stratum in strata:
        options = stratum["eligible_ranks"]
        n = stratum["adopted"]
        # Group/image denominators shared by candidates in an image have
        # already been applied, except common positive factors.
        chosen = sorted(options,key=lambda rank:(-float(delta[rank] @ weights),rank))[:n]
        adopted.extend(chosen)
    return adopted


def synthetic():
    deltas = np.array([[0.,0.],[.16,.02],[.06,.15],[.0,.0],[.02,.26],[-.1,.25]])
    rows = [dict(k=1,eligible_ranks=[1,2,3],adopted=1),dict(k=2,eligible_ranks=[4,5],adopted=1)]
    pairs = [deltas[list(pair)].sum(axis=0) for pair in itertools.product([1,2,3],[4,5])]
    supports, witnessed = [], []
    for a in ALPHAS:
        weights = np.array([a,1-a])
        selected = deltas[optimize(rows,deltas,a)].sum(axis=0)
        supports.append(float(selected @ weights))
        witnessed.append(float(min(selected)))
        need(abs(supports[-1]-max(float(pair@weights) for pair in pairs)) <= 1e-15, "Exact top-k mixed support")
    exact_joint = max(float(min(pair)) for pair in pairs)
    need(max(witnessed) <= exact_joint + 1e-15 <= min(supports) + 1e-15, "Valid simultaneous envelope and witness")
    need(optimize([dict(k=1,eligible_ranks=[3,2,1],adopted=1)],np.zeros((4,2)),.5) == [1],
         "Fixed original native tie rank")


def run(output):
    start = perf_counter()
    prev = load_previous()
    sys.path.insert(0,str(prev.CODEC))
    from pycocotools import mask as codec

    reader, records, native, groups, prior, quota = admitted(prev)
    truths = reader.labels(SimpleNamespace(labels=prev.LABELS),records)
    base, all_delta = {}, {}
    for ref, expected in zip(REFS,((411,311),(410,308)),strict=True):
        truth = truths[ref]
        matches = prev.native_050(truth,native)
        need((len(truth["annotations"]),len(matches)) == expected, "Full original reference population")
        values, all_delta[ref] = restored_geometry(truth,matches,native,codec)
        static,always = group_mean(values,groups)
        required = prior["references"][ref]["fixed_iou_points"]
        need(abs(static-required["static"]) <= 1e-12 and abs(always-required[ARM]) <= 1e-12,
             "Restored static and always BCE-robust group-equal fixed IoU parity")
        base[ref] = dict(static=float(static),always=float(always),hulls=expected[0],native_matched_hulls=expected[1])
    print("Exact PSeg and SPSD static and always BCE full-hull values restored.",flush=True)
    delta = {}
    eligible_counts = np.zeros(2,dtype=int)
    eligible_image_counts = np.zeros(2,dtype=int)
    for image in quota:
        iid = image["image_id"]
        delta[iid] = np.zeros((101,2),dtype=np.float64)
        eligible = {r for item in image["writers"][ARM] for r in item["eligible_ranks"]}
        for j,ref in enumerate(REFS):
            matched = all_delta[ref][iid]
            eligible_counts[j] += len(eligible & matched.keys())
            eligible_image_counts[j] += int(bool(eligible & matched.keys()))
            for rank,value in matched.items():
                delta[iid][rank,j] = value
    baseline_delta = np.array([base[ref]["always"] - base[ref]["static"] for ref in REFS])
    envelope = []
    for alpha in ALPHAS:
        image_gain = {}
        adopted_counts = np.zeros(2,dtype=int)
        count = 0
        for row in quota:
            iid = row["image_id"]
            selected = optimize(row["writers"][ARM],delta[iid],alpha)
            count += len(selected)
            for j,ref in enumerate(REFS):
                adopted_counts[j] += sum(rank in all_delta[ref][iid] for rank in selected)
            image_gain[iid] = delta[iid][selected].sum(axis=0)
        need(count == 2844, "Full fixed quota")
        vs_static = group_mean(image_gain,groups)
        vs_always = vs_static-baseline_delta
        envelope.append(dict(alpha=alpha, vs_static_points=vs_static.tolist(),
                             vs_always_points=vs_always.tolist(), worst_reference_points=float(min(vs_always)),
                             support_points=float(vs_always @ [alpha,1-alpha]), adoptions=count,
                             adopted_original_matched_hulls=adopted_counts.tolist()))
    upper = min(envelope,key=lambda entry:entry["support_points"])
    witness = max(envelope,key=lambda entry:entry["worst_reference_points"])
    need(witness["worst_reference_points"] <= upper["support_points"] + 1e-12, "Valid ordering of bounds")
    output.mkdir()
    result = dict(status=STATUS, protocol_sha256=LOCK, worker_sha256=digest(Path(__file__).read_bytes()),
                  quota_result_sha256=PLAN, quota_terminal_sha256=PLAN_TERMINAL,
                  author_reference_order=list(REFS), baseline_parity=base, support_grid=envelope,
                  individual_upper_points=dict(pseg=envelope[-1]["vs_always_points"][0],
                                               spsd=envelope[0]["vs_always_points"][1]),
                  simultaneous_upper_points=upper["support_points"], upper_grid_alpha=upper["alpha"],
                  best_witness_lower_points=witness["worst_reference_points"], best_witness_alpha=witness["alpha"],
                  threshold_points=THRESHOLD, simultaneous_normalized_upper=upper["support_points"]/THRESHOLD,
                  rules_out_threshold=upper["support_points"] < THRESHOLD,
                  eligible_original_matched_hulls=eligible_counts.tolist(),
                  images_with_eligible_original_matched_hulls=eligible_image_counts.tolist(),
                  images=183, decoded_groups=180, candidate_slots=18300, no_shared_match_required=True,
                  gate_labels_opened=True, AP=False, fit_labels_opened=False, history_test_opened=False,
                  detector_forwards=0, optimizer_updates=0, synthetic_checks=3, elapsed_seconds=perf_counter()-start)
    raw = prev.enc(result)
    prev.write(output/"RESULT_SAFE.json",raw)
    print(json.dumps(dict(status=STATUS,result_sha256=digest(raw))),flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    a = parser.parse_args()
    a.output = a.output.resolve()
    global OLD
    OLD = a.oracle_root.resolve()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and a.output.parent == ROOT and not a.output.exists(), "Fresh isolated CPU scope")
    pin(ROOT/"PRELOCK.md",LOCK)
    synthetic()
    run(a.output)


if __name__ == "__main__":
    main()

