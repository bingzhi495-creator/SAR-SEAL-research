"""Fit nested mask readouts on a closed SSDD FIT cohort with common score-space loss."""
import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit

PHASE_SHA = "6512bef6c4e4021716d2ba1d5477d0df213ccdcc09ed80162ec87fb48c973c60"
PENALTY = .002
ITERATIONS = 240
ARMS = ("native_bf16", "native_continuous", "full_a_scalar", "offset", "coefficients", "free_nodes")


@dataclass
class Fit:
    delta: np.ndarray
    offset: float
    objective: float
    initial_objective: float
    steps: int
    solver_success: bool
    gradient_inf: float


def sampling_matrix(shape, side, phase):
    """Align-corners-false bilinear border sampling on a regular phase grid."""
    h, w = shape
    y, x = np.divmod(np.arange(side*side), side)
    # Use the same normalized FP32 point coordinates as the native anchor.
    xx = np.clip(((x+phase[0])/side).astype(np.float32).astype(np.float64)*w-.5, 0, w-1)
    yy = np.clip(((y+phase[1])/side).astype(np.float32).astype(np.float64)*h-.5, 0, h-1)
    x0, y0 = xx.astype(np.int64), yy.astype(np.int64)
    dx, dy = xx-x0, yy-y0
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    cols = np.column_stack((y0*w+x0, y0*w+x1, y1*w+x0, y1*w+x1))
    vals = np.column_stack(((1-dy)*(1-dx), (1-dy)*dx, dy*(1-dx), dy*dx))
    return csr_matrix((vals.ravel(), cols.ravel(), np.arange(0, 4*len(x)+1, 4)),
                      shape=(len(x), h*w))


def fit_linear(design, rms, start, native, target, weights):
    """Shared convex weighted logistic loss plus regularization on score changes."""
    scale = np.maximum(rms, .001)
    target = target.astype(np.float64)

    def evaluate(value):
        delta = design @ (value/scale)
        scores = native + delta
        loss = weights @ (np.logaddexp(0., scores) - target*scores + PENALTY/2*delta**2)
        residual = weights * (expit(scores) - target + PENALTY*delta)
        return float(loss), (design.T @ residual)/scale

    x0 = start*scale
    initial = evaluate(x0)[0]
    r = minimize(evaluate, x0, jac=True, method="L-BFGS-B",
                 options=dict(maxiter=ITERATIONS, ftol=1e-11, gtol=1e-7, maxls=40))
    if not np.isfinite(r.fun) or not np.isfinite(r.x).all() or r.fun > initial+1e-10:
        raise ValueError("Finite convex objective must not increase")
    return r.x/scale, float(r.fun), initial, int(r.nit), bool(r.success), float(np.abs(r.jac).max())


def nested_fit(native, target, weights, grid, basis):
    """Fit offset, then a coefficient adjustment, then unrestricted native-grid nodes."""
    bias = np.ones((len(native), 1))
    p, *meta = fit_linear(bias, np.ones(1), np.zeros(1), native, target, weights)
    a = Fit(np.zeros(basis.shape[1]), p[0], *meta)
    design = np.column_stack((grid @ basis, bias))
    rms = np.sqrt(weights @ design**2)
    p, *meta = fit_linear(design, rms, np.r_[a.delta, a.offset], native, target, weights)
    b = Fit(p[:-1], p[-1], *meta)
    start = basis @ b.delta + b.offset
    if not np.allclose(grid @ start, design @ p, atol=1e-9, rtol=1e-9):
        raise ValueError("Free nodes must exactly embed the coefficient arm")
    rms = np.sqrt(grid.power(2).T @ weights)
    p, *meta = fit_linear(grid, rms, start, native, target, weights)
    c = Fit(p, 0., *meta)
    if not (abs(b.initial_objective-a.objective) < 1e-9
            and abs(c.initial_objective-b.objective) < 1e-9):
        raise ValueError("Exact nesting of shared objectives")
    return dict(offset=a, coefficients=b, free_nodes=c)


def support(truth, baseline, side):
    y, x = np.divmod(np.arange(side*side), side)
    outer = truth | (baseline > 0)
    tile = ((x//4 + y//4) % 2) == 0
    roi = ((x >= x[outer].min()-8) & (x <= x[outer].max()+8)
           & (y >= y[outer].min()-8) & (y <= y[outer].max()+8))
    # Sparse whole-frame anchors curb unintended far background changes.
    anchors = ((x % 16 == 2) | (x % 16 == 6)) & (y % 16 == 2)
    if not (tile & truth).any():
        tile = ~tile
    return tile & (roi | anchors), ~tile


def counts(logits, truth, threshold=0.):
    pred = logits > threshold
    i = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    u = i+fp+fn
    return dict(strict=int(u > 0 and 10*i >= 9*u), fp=fp, fn=fn)


def case_metrics(scores_a, scores_b, truth_a, truth_b, held_a, threshold=0.):
    a = counts(scores_a, truth_a, threshold)
    b = counts(scores_b, truth_b, threshold)
    held = counts(scores_a[held_a], truth_a[held_a], threshold)
    return dict(phase_a=a["strict"], phase_b=b["strict"], both=a["strict"]*b["strict"],
                held_a=held["strict"], a_fp=a["fp"], a_fn=a["fn"], b_fp=b["fp"], b_fn=b["fn"])


def phase_cases(torch, packet, phases, phase, math, cap):
    n = len(packet["is_small"])
    assigned = math.assignment_from_native_pairs(packet["native_src"].numpy(), packet["native_dst"].numpy(),
                                                 targets=n)[0]
    a, b = [phase.phase_geometry(torch, packet, p, assigned, cap) for p in phases]
    strictly = [phase.meets(x[0][:100], x[1][:100], 9, 10) for x in (a,b)]
    local = [phase.meets(x[0][assigned, np.arange(n)], x[1][assigned, np.arange(n)], 1, 2) for x in (a,b)]
    missing = ~(strictly[0] & strictly[1]).any(0)
    eligible = np.flatnonzero(missing & local[0] & local[1])
    return assigned, a, b, eligible, int(missing.sum())


def report_pair(image_rows, left, right, bootstrap):
    pairs = [(np.mean([r[left]["both"] for r in rows]), np.mean([r[right]["both"] for r in rows]))
             for rows in image_rows if rows]
    j = bootstrap(pairs)
    return dict(images=j["images"], first=j["iid"], second=j["jitter"],
                first_minus_second=j["iid_minus_jitter"], percentile95=j["percentile95"])


def audit_native_readout(torch, basis, q, bias, maps, phases, shape, side, need):
    from rfdetr.models.heads.segmentation import point_sample

    # Check both border and interior against stock point sampling outside autocast.
    ids = np.array([0, side-1, side, side*side-side, side*side-1, side*37+25, side*201+233])
    y, x = np.divmod(ids, side)
    tensor = torch.from_numpy(basis.T.astype(np.float32).reshape(1,-1,*shape)).cuda()
    coeff = torch.from_numpy(q.astype(np.float32)).cuda()
    for phase, matrix in zip(phases, maps, strict=True):
        xy = np.column_stack(((x+phase[0])/side, (y+phase[1])/side)).astype(np.float32)
        feat = point_sample(tensor, torch.from_numpy(xy).cuda()[None], align_corners=False)
        native = (coeff @ feat[0] + bias).cpu().numpy()
        analytic = matrix[ids] @ (basis @ q) + bias
        need(np.allclose(native, analytic, atol=.001, rtol=.0001), "Analytic border/interior readout matches stock")


def module(path, expected, name):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Exact closed dependency digest")
    result = types.ModuleType(name)
    result.__file__ = str(path)
    sys.modules[name] = result
    exec(compile(raw, str(path), "exec"), result.__dict__)
    return result


def run(args):
    started = time.perf_counter()
    phase = module(args.phase_source, PHASE_SHA, "_pinned_phase_geometry")
    replay = module(args.replay_runtime, phase.REPLAY_SHA, "_closed_matcher_replay")
    need = replay.need
    need(sys.version_info[:3] == (3,12,14) and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
         and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
         "Exact isolated DGX runtime")
    need(version("scipy")=="1.15.3", "Exact SciPy optimizer")
    replay.canonical(args.capture_dir, directory=True)
    need(args.output.is_absolute() and replay.canonical(args.output.parent, directory=True) == args.output.parent
         and not args.output.exists() and not args.output.is_symlink()
         and not args.output.is_relative_to(args.capture_dir), "Fresh output outside closed capture")
    cap = replay.module(args.capture_source, replay.CAPTURE_SHA, "_pinned_capture_matcher_cost")
    math = replay.module(args.replay_runtime.with_name("matcher_cost_math.py"), replay.MATH_SHA, "_pinned_matcher_math")
    report = json.loads(replay.checked_bytes(args.capture_dir/"CAPTURE_SAFE.json", args.capture_report_sha256))
    receipt = json.loads(replay.checked_bytes(args.capture_closure, args.capture_closure_sha256))
    rows = replay.validate_closed(report, receipt, args.capture_protocol, args.capture_report_sha256, cap)
    items = json.loads(replay.checked_bytes(args.fit_index, replay.FIT_MANIFEST_SHA))["items"]
    plan = cap.view_plan(items, report["tranche"])
    need(report["tranche"] == "cohort32" and len(items)==741 and len({v["image_sha256"] for v,_ in plan})==32,
         "Exact original unique FIT scope")
    sys.addaudithook(cap.offline)
    import torch
    need(all(version(k)==v for k,v in cap.VERSIONS.items())
         and all(replay.digest(Path(import_module(k).__file__).read_bytes())==v for k,v in cap.NATIVE_SOURCE.items()),
         "Exact native numeric runtime")
    need(torch.cuda.is_available() and torch.cuda.device_count()==1, "One DGX CUDA device")
    torch.cuda.set_per_process_memory_fraction(min(.35, cap.CAP/torch.cuda.get_device_properties(0).total_memory))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    phases = math.DENSE_PHASES
    maps, lattice = None, None
    views, fit = [], {arm: [] for arm in ARMS[3:]}
    totals = Counter()
    for row in rows:
        if row["status"]!="STOCK_TRACE_EXACT":
            continue
        packet = replay.load_packet(torch, args.capture_dir, row, plan, args.capture_protocol, cap)
        assigned,a,b,eligible,missing = phase_cases(torch,packet,phases,phase,math,cap)
        totals.update(targets=len(assigned), missing=missing, eligible=len(eligible))
        sparse = packet["sparse"]
        _, channels, h, w = sparse["spatial_features"].shape
        need((h,w)==(144,144), "Exact native 144-grid support")
        basis = sparse["spatial_features"][0].float().numpy().reshape(channels,-1).T.astype(np.float64)
        coeff = sparse["query_features"][0].float().numpy().astype(np.float64)
        bias = float(sparse["bias"].float().item())
        if maps is None:
            lattice=(h,w,channels)
            maps=[sampling_matrix((h,w),phase.SIDE,p) for p in phases]
        need(lattice==(h,w,channels) and np.array_equal(a[3],b[3]), "Fixed lattice and identical phase GT labels")
        paired = []
        for i in eligible:
            q = coeff[assigned[i]]
            truth = a[3][i]
            anchor_a, anchor_b = a[2][i],b[2][i]
            g = basis @ q
            za,zb = [p@g+bias for p in maps]
            if i==eligible[0]:
                torch.set_float32_matmul_precision("highest")
                audit_native_readout(torch,basis,q,bias,maps,phases,(h,w),phase.SIDE,need)
                torch.set_float32_matmul_precision("high")
            mask,held = support(truth,anchor_a,phase.SIDE)
            y = truth[mask]
            need(y.any() and not y.all(), "Every locked instance must have both fit labels")
            weights=np.where(y,.5/y.sum(),.5/(~y).sum())
            native_threshold=phase.best_threshold(anchor_a,truth)
            outcomes = dict(native_bf16=case_metrics(anchor_a,anchor_b,truth,truth,held),
                            native_continuous=case_metrics(za,zb,truth,truth,held),
                            full_a_scalar=case_metrics(anchor_a,anchor_b,truth,truth,held,native_threshold))
            fitted = nested_fit(za[mask],y,weights,maps[0][mask],basis)
            for arm,f in fitted.items():
                if arm=="offset":
                    da,db = f.offset,f.offset
                else:
                    nodes = basis@f.delta if arm=="coefficients" else f.delta
                    da,db = [p@nodes+f.offset for p in maps]
                outcomes[arm]=case_metrics(za+da,zb+db,truth,truth,held)
                fit[arm].append(dict(objective=f.objective,initial=f.initial_objective,steps=f.steps,
                                     success=f.solver_success,gradient_inf=f.gradient_inf))
            paired.append(outcomes)
            totals.update(fit_points=int(mask.sum()), fit_positives=int(y.sum()),
                          held_gt=int(truth[held].any()), small=int(packet["is_small"][i]))
        views.append(paired)
        print(json.dumps(dict(event="NESTED_MASK_ORACLE_VIEW_COMPLETE",ordinal=row["ordinal"],
                              eligible=len(eligible))),flush=True)
    cases=[x for view in views for x in view]
    need(len(views)==30 and sum(bool(v) for v in views)==21
         and (totals["targets"],totals["missing"],len(cases))==(55,42,42)
         and sum(x["full_a_scalar"]["phase_a"] for x in cases)==7
         and sum(x["full_a_scalar"]["both"] for x in cases)==3,
         "Reproduce exact P128 eligibility, supports and scalar reference")
    by_arm={arm:{k:sum(x[arm][k] for x in cases) for k in cases[0][arm]} for arm in ARMS}
    pairs={name:report_pair(views,a,b,replay.image_pair)
           for name,a,b in (("coeff_vs_offset","coefficients","offset"),
                            ("coeff_vs_full_a_scalar","coefficients","full_a_scalar"),
                            ("free_vs_coeff","free_nodes","coefficients"))}
    qualified=lambda j: j["first_minus_second"]>=.10 and j["percentile95"][0]>0
    fits={k:dict(solver_success=sum(x["success"] for x in v),
                 reached_iteration_limit=sum(x["steps"]==ITERATIONS for x in v),
                 gradient_inf_max=max(x["gradient_inf"] for x in v),
                 mean_final_loss=float(np.mean([x["objective"] for x in v])),
                 mean_gain=float(np.mean([x["initial"]-x["objective"] for x in v]))) for k,v in fit.items()}
    safe=dict(status="SSDD_FIT_NESTED_MASK_BASIS_ORACLE_COMPLETE", protocol=args.protocol,
              FIT_only=True,DEV=False,TEST=False,AP=False,optimizer_updates=0,
              traced_views=len(views), empty_views=len(rows)-len(views), eligible_images=21,
              capture_protocol=args.capture_protocol, phases=phases, lattice=lattice,
              scope=dict(totals), arms=by_arm, paired=pairs, solver=fits,
              coefficients_gate=qualified(pairs["coeff_vs_offset"]) and qualified(pairs["coeff_vs_full_a_scalar"]),
              free_nodes_gate=qualified(pairs["free_vs_coeff"]),
              shared_score_penalty=PENALTY, max_solver_iterations=ITERATIONS,
              cuda_peak_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-started)
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf-8") as f:
        json.dump(safe,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(safe,sort_keys=True),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("phase-source","replay-runtime","capture-dir","capture-source","capture-closure","fit-index","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("protocol","capture-protocol","capture-report-sha256","capture-closure-sha256"):
        parser.add_argument("--"+name,required=True)
    run(parser.parse_args())
