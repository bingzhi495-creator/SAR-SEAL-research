"""Exact historical gate oracle on a sealed selective BCE budget."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD: Path
PRIOR: Path
LABELS: Path
CODEC: Path
LOCK = "9705fc99bac75d384a74a0306cba50734a9546a058e64e01b874c0b6098fe519"
OLD_RENDER = "7aab5753eafe1e18464d0740a6b440fe09997dca2020c64a0d48180c228728fd"
OLD_SCORE = "b93e0f4d355f538ae751b8622bf359541dd5705f25d1dfe6eca836984521b052"
RENDER_WORKER = "c4841925513aeee64078064d483c73557b48c352594dd0d1edbe67a15be4b5b8"
SCORE_WORKER = "f65219d4677f93fa0347989ac3f3b5d7b51c4b2342d6aa6ab645c472d8dc3ad7"
WRITERS = ("bce_robust", "bce_mean")
REFS = ("pseg", "spsd")
OBJECTIVES = ("minimum", "pseg_only", "spsd_only")
STATUS_PLAN = "SPSD_GATE_SELECTIVE_BCE_ORACLE_QUOTAS_COMPLETE"
STATUS_SCORE = "SPSD_GATE_SELECTIVE_BCE_ORACLE_SCORE_COMPLETE"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(data):
    return sha256(data).hexdigest()


def enc(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def pin(path, expected):
    data = path.read_bytes()
    need(sha(data) == expected, f"Original exact file {path.name}")
    return data


def terminal(path, phase, worker, result):
    data = path.read_bytes()
    close = json.loads(data)
    need(close["actual_exit"] == 0 and close["output_footer_verified"] is True
         and close["phase"] == phase and close["worker_sha256"] == worker
         and close["result_sha256"] == result, f"Closed receipt {path.name}")
    return sha(data)


def write(path, data):
    with path.open("xb") as stream:
        need(stream.write(data) == len(data), "Full readback and result write")
        stream.flush()
        os.fsync(stream.fileno())


def finish(folder, result):
    result.update(protocol_sha256=LOCK, worker_sha256=sha(Path(__file__).read_bytes()),
                  prior_render_sha256=OLD_RENDER, prior_score_sha256=OLD_SCORE,
                  candidate_slots=18300, images=183, decoded_groups=180,
                  fit_labels_opened=False, historical_test_opened=False, AP=False,
                  detector_forwards=0, optimizer_updates=0)
    data = enc(result)
    write(folder / "RESULT_SAFE.json", data)
    print(json.dumps(dict(status=result["status"], result_sha256=sha(data))), flush=True)


def inputs():
    terminal(OLD / "RENDER_TERMINAL.json", "render", RENDER_WORKER, OLD_RENDER)
    term = terminal(OLD / "SCORE_TERMINAL.json", "score", SCORE_WORKER, OLD_SCORE)
    prior = json.loads(pin(OLD / "score/RESULT_SAFE.json", OLD_SCORE))
    scorer = ModuleType("_prior_frozen_scorer")
    source = OLD / "score_gate.py"
    scorer.__file__ = str(source)
    exec(compile(pin(source, SCORE_WORKER), str(source), "exec"), vars(scorer))
    _, reader, _, rsha, _, render, native, rows, groups = scorer.accepted(SimpleNamespace(gate=PRIOR), OLD)
    need(rsha == OLD_RENDER == prior["render_result_sha256"]
         and render["render_stream_sha256"] == prior["render_stream_sha256"]
         and prior["exposure"]["shared_native_matches"] == 276 and len(groups) == 183
         and len(set(groups.values())) == 180, "Exact historical cohort configuration")
    return reader, rows, native, groups, prior, term


def allocate(strata):
    base = {k: len(ranks)//3 for k, ranks in strata.items()}
    total = sum(map(len, strata.values()))//3
    remain = total - sum(base.values())
    largest = sorted(strata, key=lambda k: (-(len(strata[k]) % 3), k))
    for k in largest[:remain]:
        base[k] += 1
    need(sum(base.values()) == total and all(0 <= base[k] <= len(strata[k]) for k in strata),
         "Exact within-image largest-remainder adoption")
    return total, base


def quota_plan(native, images):
    lines, results = [], {arm: Counter() for arm in WRITERS}
    for iid, candidates in native.items():
        image = dict(image_id=iid, source_group=images[iid]["source_group"], writers={})
        for arm in WRITERS:
            strata = defaultdict(list)
            for rank, row in enumerate(candidates, 1):
                if row["quota"] > 0 and row["changes"][arm] != row["changes"]["static"]:
                    strata[row["quota"]].append(rank)
            total, capacities = allocate(strata)
            by_k = [dict(k=k, eligible_ranks=strata[k], adopted=capacities[k]) for k in sorted(strata)]
            image["writers"][arm] = by_k
            results[arm].update(eligible=sum(map(len, strata.values())), fixed_adoptions=total,
                                images_with_adoptions=int(total>0), nonempty_strata=len(strata))
        lines.append(image)
    return lines, results


def plan(folder):
    started = perf_counter()
    _, records, native, _, _, prior_terminal = inputs()
    lines, totals = quota_plan(native, records)
    raw = b"".join(map(enc, lines))
    folder.mkdir()
    write(folder / "QUOTAS_PRIVATE.jsonl", raw)
    finish(folder, dict(status=STATUS_PLAN, quota_stream_sha256=sha(raw), counts=totals,
                        prior_score_terminal_sha256=prior_terminal, gate_labels_opened=False,
                        elapsed_seconds=perf_counter()-started))


def native_050(truth, native):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    slots, detections = {}, []
    for iid, records in native.items():
        for rank,row in enumerate(records, 1):
            slots[len(detections)+1] = iid, rank
            detections.append(dict(image_id=iid, category_id=0, score=row["score"], segmentation=row["native"]))
    with redirect_stdout(StringIO()):
        gt = COCO()
        gt.dataset = deepcopy(truth)
        gt.createIndex()
        ev = COCOeval(gt, gt.loadRes(detections), "segm")
        ev.params.catIds, ev.params.imgIds, ev.params.maxDets = [0], sorted(native), [100]
        ev.params.iouThrs = np.array([.5])
        ev.params.areaRng = [ev.params.areaRng[0]]
        ev.params.areaRngLbl = ["all"]
        ev.evaluate()
    associated = {}
    for block in ev.evalImgs:
        if block is None:
            continue
        for aid, did, ignore in zip(block["gtIds"], block["gtMatches"][0], block["gtIgnore"], strict=True):
            if did and not ignore:
                slot = slots[int(did)]
                need(slot[0] == block["image_id"] and aid not in associated, "Exact slot owner")
                associated[int(aid)] = slot
    return associated


def paired_truth(codec, native, groups, truths, matches):
    from scipy.optimize import linear_sum_assignment

    objects = {ref: defaultdict(list) for ref in REFS}
    for ref in REFS:
        for hull in truths[ref]["annotations"]:
            objects[ref][hull["image_id"]].append(hull)
    denominator = {iid: np.array([len(objects[ref][iid]) for ref in REFS], dtype=int) for iid in groups}
    counts, shared = Counter(), {iid: {} for iid in groups}
    for iid in groups:
        pseg, spsd = [objects[ref][iid] for ref in REFS]
        if not pseg or not spsd:
            continue
        p, s = [[hull["segmentation"] for hull in records] for records in (pseg, spsd)]
        iou = codec.iou(p,s,[0]*len(s))
        pi, si = linear_sum_assignment(iou, maximize=True)
        counts["raw_pairs"] += len(pi)
        for x, y in zip(pi, si, strict=True):
            if iou[x,y] < .5:
                continue
            counts["reference_iou_ge_050"] += 1
            r, z = matches["pseg"].get(pseg[x]["id"]), matches["spsd"].get(spsd[y]["id"])
            if r is None or r != z:
                continue
            j, rank = r
            need(iid == j and rank not in shared[iid], "Unique original candidate pair")
            shared[iid][rank] = (p[x], s[y])
            counts["shared_matches"] += 1
            counts["shared_positive_quota"] += int(native[iid][rank-1]["quota"] > 0)
    need(tuple(counts[key] for key in ("raw_pairs","reference_iou_ge_050","shared_matches",
                                      "shared_positive_quota")) == (410,370,276,206), "Exact prior common coverage")
    return shared, denominator, counts


def deltas_for(codec, rows, shared, arm, ranks):
    values = {}
    for rank in ranks:
        if rank not in shared:
            continue
        rles = rows[rank-1]["changes"]
        each = []
        for label in shared[rank]:
            iou = codec.iou([rles[arm],rles["static"]],[label],[0])[:,0]
            inter = [int(codec.area(codec.merge([rles[key], label], intersect=True))) for key in (arm,"static")]
            each.append((float(iou[0]-iou[1]), inter[0]-inter[1]))
        values[rank] = tuple(x[0] for x in each), tuple(x[1] for x in each)
    return values


def choice(eligible, n, values, objective):
    def rank(r):
        delta = values.get(r, ((0.,0.), (0,0)))[0]
        merit = min(delta) if objective == "minimum" else delta[REFS.index(objective.removesuffix("_only"))]
        return -merit, r
    return sorted(eligible, key=rank)[:n]


def select_image(strata, values):
    fixed = {objective: [] for objective in OBJECTIVES}
    eligible = []
    for stratum in strata:
        options, n = stratum["eligible_ranks"], stratum["adopted"]
        eligible.extend(options)
        for mode in OBJECTIVES:
            fixed[mode].extend(choice(options, n, values, mode))
    fixed["always"] = eligible
    fixed["unlimited"] = [r for r in values if min(values[r][0]) > 0]
    return fixed


def summarize_selected(positions, values, n):
    relevant = [values[r] for r in positions if r in values]
    if not relevant:
        return np.zeros(3), Counter(total_adoptions=len(positions))
    delta = np.asarray([x[0] for x in relevant])
    pixels = np.asarray([x[1] for x in relevant])
    minima = delta.min(axis=1)
    sign = pixels.min(axis=1)
    effect = np.r_[minima.sum()/max(n), delta.sum(axis=0)/np.maximum(n,1)]
    return effect, Counter(total_adoptions=len(positions), shared_hulls=len(relevant),
                           positive_min=int((sign>0).sum()), zero_min=int((sign==0).sum()),
                           negative_min=int((sign<0).sum()))


def synthetic():
    n, strata = allocate({1:[1,2],2:[3,4],3:[5,6,7,8,9]})
    need((n,strata) == (3,{1:1,2:1,3:1}), "Largest-remainder lower k tie")
    values = {1: ((.2,.1),(2,1)), 3: ((-.01,.2),(-1,2))}
    need(choice([1,2,3],2,values,"minimum") == [1,2], "Shared negative yields to unshared zero")
    need(choice([1,3],2,values,"minimum") == [1,3] and choice([1,3],1,values,"spsd_only") == [3],
         "Forced exact capacity and reference-specific optimistic ranking")


def score(folder):
    from pycocotools import mask as codec

    started = perf_counter()
    worker = sha(Path(__file__).read_bytes())
    proof = ROOT / "observed_plan/TERMINAL.json"
    receipt_hash = json.loads(proof.read_bytes())["result_sha256"]
    terminal_hash = terminal(proof,"capture",worker,receipt_hash)
    receipt = json.loads(pin(ROOT/"plan/RESULT_SAFE.json",receipt_hash))
    need(receipt["status"] == STATUS_PLAN and receipt["gate_labels_opened"] is False
         and receipt["protocol_sha256"] == LOCK and receipt["worker_sha256"] == worker,
         "Exact locked no-label budget stage")
    raw_quotas = pin(ROOT/"plan/QUOTAS_PRIVATE.jsonl",receipt["quota_stream_sha256"])
    reader, records, native, groups, previous, _ = inputs()
    plan_rows, totals = quota_plan(native, records)
    need(raw_quotas == b"".join(map(enc,plan_rows)) and totals == receipt["counts"],
         "Independent exact frozen quota readback before gate references")
    truths = reader.labels(SimpleNamespace(labels=LABELS), records)
    matches = {ref:native_050(truths[ref],native) for ref in REFS}
    need(tuple(len(matches[r]) for r in REFS) == (311,308) and
         tuple(len(truths[r]["annotations"]) for r in REFS) == (411,410), "Full fixed original reference denominators")
    shared, denominator, exposure = paired_truth(codec,native,groups,truths,matches)
    ordered = sorted(set(groups.values()))
    draws = np.random.Generator(np.random.PCG64(5297)).integers(180,size=(4096,180))
    modes = (*OBJECTIVES,"always","unlimited")
    output = {}
    for arm in WRITERS:
        group_rows = {mode:defaultdict(list) for mode in modes}
        metrics = {mode:Counter() for mode in modes}
        exposure_groups = {mode:set() for mode in modes}
        for image in plan_rows:
            iid = image["image_id"]
            group = groups[iid]
            strata = image["writers"][arm]
            eligible = [r for row in strata for r in row["eligible_ranks"]]
            values = deltas_for(codec,native[iid],shared[iid],arm,eligible)
            decisions = select_image(strata,values)
            for mode, ranks in decisions.items():
                d, counts = summarize_selected(ranks,values,denominator[iid])
                group_rows[mode][group].append(d)
                metrics[mode].update(counts)
                if counts["shared_hulls"]:
                    exposure_groups[mode].add(group)
        need(all(metrics[mode]["total_adoptions"] == totals[arm]["fixed_adoptions"] for mode in OBJECTIVES),
             "All prespecified adoption counts filled")
        arrays = {mode:np.stack([np.asarray(group_rows[mode][g]).mean(axis=0) for g in ordered])*100
                  for mode in modes}
        need(np.all(arrays["unlimited"][:,0] >= arrays["minimum"][:,0]-1e-12)
             and np.all(arrays["pseg_only"][:,1] >= arrays["minimum"][:,1]-1e-12)
             and np.all(arrays["spsd_only"][:,2] >= arrays["minimum"][:,2]-1e-12), "Independent upper-bound inequalities")

        def stat(x):
            return dict(mean_points=x.mean(axis=0).tolist(),
                        percentile95_points=np.percentile(x[draws].mean(axis=1),[2.5,97.5],axis=0).T.tolist())

        output[arm] = dict(eligibility=totals[arm], solution={
            mode:dict(stat(arrays[mode]), counts=dict(metrics[mode]), changed_shared_groups=len(exposure_groups[mode]))
            for mode in modes}, selected_minus_always_on_shared=stat(arrays["minimum"]-arrays["always"]),
            original_full_native_fixed_iou_points={ref:previous["references"][ref]["fixed_iou_points"][arm] for ref in REFS})
    folder.mkdir()
    finish(folder, dict(status=STATUS_SCORE, budget_result_sha256=receipt_hash, budget_terminal_sha256=terminal_hash,
                        eligible_matches=exposure, author_hulls=[411,410], native_matches=[311,308],
                        metrics_order=["per_hull_minimum","pseg","spsd"], writers=output,
                        bootstrap_draws=4096, random_seed=5297, gate_labels_opened=True,
                        elapsed_seconds=perf_counter()-started))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("plan","score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signed-action-gate", type=Path, required=True)
    parser.add_argument("--original-gate", type=Path, required=True)
    parser.add_argument("--label-export", type=Path, required=True)
    parser.add_argument("--codec-directory", type=Path, required=True)
    a = parser.parse_args()
    a.output = a.output.resolve()
    global OLD, PRIOR, LABELS, CODEC
    OLD = a.signed_action_gate.resolve()
    PRIOR = a.original_gate.resolve()
    LABELS = a.label_export.resolve()
    CODEC = a.codec_directory.resolve()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and a.output.parent == ROOT and not a.output.exists(), "Fresh isolated CPU scope")
    pin(ROOT/"PRELOCK.md",LOCK)
    sys.path.insert(0,str(CODEC))
    synthetic()
    (plan if a.phase == "plan" else score)(a.output)


if __name__=="__main__":
    main()

