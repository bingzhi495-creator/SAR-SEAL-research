"""Audit closed predicted maximin search with full-quota upper bounds, without labels."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
import torch


STATUS = "SPSD_DUAL_GATE_PREDICTED_MAXIMIN_BOUND_COMPLETE"
PROTOCOL = "c5906ccc76da02bbfe308b5552ae9061d6c3e2f047f7cce3e4510e3af94b68bf"
RENDER_WORKER = "17fb4db0858785a312d1c761d97a39f8de448e9b19230aeb727a8ff5d286f5a7"
RENDER_RESULT = "4681265afbbf7acae7389a937e988e85218f558ad7ff99e6d87fa5d0f69e51b3"
TOL = 1e-8


def need(test, message):
    if not test:
        raise ValueError(message)


def pin(path, expected):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == expected, "Bound file digest: " + path.name)
    return raw


def gain(q, pair):
    return q[np.asarray(pair[0], dtype=np.intp), 0].sum(axis=0) - q[np.asarray(pair[1], dtype=np.intp), 1].sum(axis=0)


def support_upper(q, k, incumbent):
    """Any convex mixture's optimal linear signed gain is a valid upper bound."""
    base = gain(q, incumbent)

    def support(w):
        mixed = q[:, :, 1] + w * (q[:, :, 0] - q[:, :, 1])
        add = np.argsort(-mixed[:, 0], kind="stable")[:k]
        drop = np.argsort(mixed[:, 1], kind="stable")[:k]
        v = gain(q, (add, drop)) - base
        return w * v[0] + (1 - w) * v[1], v[0] - v[1]

    left, right = 0.0, 1.0
    upper = min(support(left)[0], support(right)[0])
    for _ in range(32):
        mid = (left + right) / 2
        value, slope = support(mid)
        upper = min(value, upper)
        if slope < 0:
            left = mid
        else:
            right = mid
    return float(upper)


def refine(q, k, incumbent, lower, upper):
    """Keep all valid primal/dual information when an exact solve times out."""
    m = len(q)
    base = gain(q, incumbent)
    c = np.zeros(2*m+1)
    c[-1] = -1
    rows = np.zeros((4, 2*m+1))
    rows[0, :m] = rows[1, m:2*m] = 1
    rows[2:, :m] = -q[:, 0, :].T
    rows[2:, m:2*m] = q[:, 1, :].T
    rows[2:, -1] = 1
    constraints = LinearConstraint(rows, [k,k,-np.inf,-np.inf], [k,k,-base[0],-base[1]])
    fit = milp(c, integrality=np.r_[np.ones(2*m), 0],
               bounds=Bounds(np.zeros(2*m+1), np.r_[np.ones(2*m), 2*k]),
               constraints=constraints, options={"time_limit": .50, "mip_rel_gap": 0.0})
    if fit.status in (0, 1):
        if fit.x is not None:
            body = fit.x[:2*m]
            if np.all(np.abs(body - np.rint(body)) <= 1e-6):
                a, d = (np.flatnonzero(x > .5) for x in (body[:m], body[m:]))
                if len(a) == len(d) == k:
                    lower = max(lower, float(np.min(gain(q, (a,d)) - base)))
        bound = getattr(fit, "mip_dual_bound", None)
        if bound is not None and np.isfinite(bound):
            # A dual bound on the minimization of -t bounds the gain from above.
            upper = min(upper, -float(bound) + TOL)
    need(upper >= lower - TOL, "Continuous and integer bounds enclose a feasible witness")
    return lower, max(lower, upper), str(fit.status)


def inputs(args):
    mod = SimpleNamespace(__file__=str(args.render_worker))
    exec(compile(pin(args.render_worker, RENDER_WORKER), str(args.render_worker), "exec"), vars(mod))
    result = json.loads(pin(args.gate/"render/RESULT_SAFE.json", RENDER_RESULT))
    mod.terminal(args.gate/"RENDER_TERMINAL.json", "render", RENDER_RESULT, RENDER_WORKER)
    need(result["source_gate_labels_opened"] is False and result["historical_test_opened"] is False
         and result["fit_result_sha256"] == mod.FIT, "Closed label-free gate render")
    _, _, pts, core, select, _, models, center, scale, _ = mod.load_artifacts(args)
    entries = args.points/"actual_retry/source_gate"
    index = [json.loads(line) for line in mod.pin(entries/"POINT_INDEX_PRIVATE.jsonl",
                         pts["partitions"]["source_gate"]["point_index_sha256"]).splitlines()]
    render = [json.loads(line) for line in pin(args.gate/"render/GATE_RENDER_PRIVATE.jsonl",
                                               result["render_stream_sha256"]).splitlines()]
    candidates, seen, frame = [], Counter(), []

    for item, image in zip(index, render, strict=True):
        need(item["image_id"] == image["image_id"] and item["source_group"] == image["source_group"]
             and item["original_hw"] == image["original_hw"] and len(image["candidates"]) == 100,
             "Exact image, group and native frame")
        payload = mod.pin(entries/item["filename"], item["sha256"])
        need(len(payload) == item["bytes"], "Point packet byte parity")
        with np.load(BytesIO(payload), allow_pickle=False) as packet:
            rank, offsets, coordinates, features, quotas, static = (
                packet[k] for k in ("native_ranks","point_offsets","signed_coordinates","features",
                                    "static_offsets","static_side_indices"))
        predicted = core.predict(models["dual"], features, center, scale)
        need(len(rank) == item["supported_candidates"] and len(set(rank)) == len(rank)
             and len(coordinates) == item["paired_sites"] and len(static) == item["realized_static_exchanges"],
             "Exact original point and static budget population")
        for j, rid in enumerate(rank):
            row = image["candidates"][int(rid)-1]
            lo, hi = offsets[j:j+2]
            qlo, qhi = quotas[j:j+2]
            k = int(qhi-qlo)
            need(k == row["quota"], "Stored original edit budget")
            if k == 0:
                continue
            inc = mod.actions(static[qlo:qhi])
            q = predicted[lo:hi].astype(np.float64)
            choice = select.select(q[:,0], q[:,1], *inc, k=k)
            raw = row["native"]
            stock = mod.codec.decode(raw).astype(bool, order="C")
            for arm, plan in (("static", inc), ("dual_robust", choice["robust"]),
                        ("dual_mean",choice["mean"]),("dual_cell_pessimistic",choice["cell_pessimistic"])):
                mask = mod.edited_mask(raw, stock, coordinates[lo:hi], plan, k)
                need(mask == row["changes"][arm], "Exact old RLE reconstruction: " + arm)
            base = gain(q, inc)
            finite = float(np.min(gain(q,choice["robust"]) - base))
            upper = support_upper(q,k,inc)
            need(finite >= -TOL and upper >= finite - TOL, "Known feasible search and upper bound")
            candidates.append(dict(group=item["source_group"], image=item["image_id"], k=k,
                                   q=q, inc=inc, finite=finite, lower=finite, upper=max(upper,finite)))
            seen.update(positive=1, quota=k, robust_changed=int(choice["robust"]!=inc))
        seen.update(images=1, candidates=100, supported=len(rank))
        frame.append((item["source_group"], item["image_id"]))
    need(seen["images"] == 183 and seen["candidates"] == 18300 and seen["supported"] == 8821
         and seen["positive"] == 8785 and seen["quota"] == 57692
         and seen["robust_changed"] == result["arms"]["dual_robust"]["changed_vs_static"]
         and len({group for group, _ in frame}) == 180, "Full gate parity")
    seen["positive_groups"] = len({c["group"] for c in candidates})
    seen["zero_quota_images"] = len(frame) - len({c["image"] for c in candidates})
    return candidates, dict(seen), frame


def summarize(candidates, frame):
    groups = defaultdict(lambda: defaultdict(list))
    for group, image in frame:
        groups[group][image] = []
    by_k = defaultdict(list)
    for entry in candidates:
        value = [entry["finite"], entry["lower"]-entry["finite"], entry["upper"]-entry["finite"]]
        groups[entry["group"]][entry["image"]].append(value)
        by_k[entry["k"]].append(value)
    group_rows = np.asarray([np.mean([np.mean(x,axis=0) if x else np.zeros(3) for x in groups[k].values()],axis=0) for k in sorted(groups)])
    draw = np.random.Generator(np.random.PCG64(5297)).integers(len(group_rows),size=(4096,len(group_rows)))
    raw = np.asarray([v for rows in by_k.values() for v in rows])
    return dict(
        columns=["finite_margin","lower_bound_gap","upper_bound_gap"],
        candidate_means=raw.mean(axis=0).tolist(),
        group_means=group_rows.mean(axis=0).tolist(),
        paired_group95=np.percentile(group_rows[draw].mean(axis=1),[2.5,97.5],axis=0).T.tolist(),
        upper_gap_percentiles=dict(zip(("median","p90","p95","p99","max"),
                    np.percentile(raw[:,2],[50,90,95,99,100]).tolist(),strict=True)),
        supported_exceed_001_by_witness=int((raw[:,1] > .01).sum()),
        possibly_exceed_001=int((raw[:,2] > .01).sum()),
        upper_within_001=int((raw[:,2] <= .01).sum()),
        numerically_equal_bound=int(((raw[:,2]-raw[:,1]) <= TOL).sum()),
        expand_search_threshold=.05,
        expand_estimated_search_by_threshold=bool(group_rows[:,2].mean()>.05),
        quota_strata={str(k):dict(count=len(v),candidate_means=np.mean(v,axis=0).tolist())
                       for k,v in sorted(by_k.items())}
    )


def main():
    p = ArgumentParser(description=__doc__)
    for name in ("protocol","render-worker","gate","fit","capture","points","core","selector","attraction","out"):
        p.add_argument("--"+name,type=Path,required=True)
    args = p.parse_args()
    start = perf_counter()
    pin(args.protocol, PROTOCOL)
    torch.set_num_threads(1)
    roster, seen, frame = inputs(args)
    initial = summarize(roster, frame)
    order = sorted(range(len(roster)), key=lambda i: (-(roster[i]["upper"]-roster[i]["finite"]),i))
    solicit = [i for i in order if roster[i]["upper"]-roster[i]["finite"] > .01][:128]
    status = Counter()
    for i in solicit:
        row = roster[i]
        row["lower"],row["upper"],s = refine(row["q"],row["k"],row["inc"],row["lower"],row["upper"])
        status[s] += 1
    result = dict(status=STATUS, protocol_sha256=PROTOCOL, prior_render_sha256=RENDER_RESULT,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  matched_complete_prior_mask_parity=True, source_gate_labels_opened=False,
                  historical_test_opened=False, new_detector_forwards=0, optimizer_updates=0,
                  counts=seen, initial=initial, final=summarize(roster, frame),
                  integer_solve_count=len(solicit), integer_solver_status=dict(status), elapsed_seconds=perf_counter()-start)
    raw = (json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with args.out.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"status":STATUS,"result_sha256":sha256(raw).hexdigest()}),flush=True)


if __name__ == "__main__":
    main()
