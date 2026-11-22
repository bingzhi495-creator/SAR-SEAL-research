"""Fit three fixed-capacity quota ridge controls on closed HRSID source shards."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


POLICIES = "9873890710accea913138ea92c2a874d51a51fc2ef5cec0815ca7cf52b103cdf"
PROTOCOL = "ceb5e6d0615cc794f1232adc8dc4853e6839580ee41ad7f95af5fe5273353a29"
SOURCE_RESULT = "bec04364e1df66fe3073b9cbe6e6c3e563ffdfb9d38ee8933cba53a3c6a13111"
SOURCE_TERMINAL = "0097515808d3a73aadc23d3de15dee17001e9f2c13cf6dd50218cf8ed7ffe363"
SOURCE_STATUS = "RFDETR_FIT109_QUOTA_SOURCE_ACTIONS_COMPLETE"
AP_STATUS = "RFDETR_FIT109_QUOTA_COMPETITION_TARGETS_COMPLETE"
STATUS = "RFDETR_FIT109_QUOTA_COMPETITION_MODELS_COMPLETE"
SEEDS = (5217, 5218, 5219)
OFFSETS = np.arange(-8, 9, dtype=np.int16)
LIMIT = 16*1024**2


def check(ok, why):
    if not ok:
        raise ValueError(why)


def pin(path, expected, limit=LIMIT):
    with path.open("rb") as stream:
        data = stream.read(limit+1)
    check(0 < len(data) <= limit and sha256(data).hexdigest() == expected, "Pinned bounded input " + path.name)
    return data


def read(path, expected):
    return json.loads(pin(path, expected))


def load_policies():
    path = Path(__file__).with_name("policies.py")
    pin(path, POLICIES)
    spec = spec_from_file_location("_frozen_quota_policies", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def closed_source(root, result_sha, terminal_sha, status):
    observer = read(root/"observed_prepare/TERMINAL.json", terminal_sha)
    check(observer["actual_exit"] == 0 and observer["output_footer_verified"] is True
          and observer["result_sha256"] == result_sha and observer["phase"] == "prepare-edits",
          "Source worker has a successful observed exit and footer")
    result = read(root/"prepare/RESULT_SAFE.json", result_sha)
    check(result["status"] == status and result["FIT"] is True and result["DEV"] is False and result["TEST"] is False
          and observer["worker_sha256"] == result["worker_sha256"], "Exact closed FIT-only worker role")
    return result


def input_refs(source, source_result, competition, ap_result, policies):
    feature_sha = source_result["prelabel_feature_manifest_sha256"]
    features = read(source/"FEATURE_MANIFEST_PRIVATE.json", feature_sha)
    geometry = read(source/"TARGET_MANIFEST_PRIVATE.json", source_result["target_manifest_sha256"])
    ap_rows = read(competition/"COMPETITION_MANIFEST_PRIVATE.json", ap_result["competition_manifest_sha256"])
    check(ap_result["feature_manifest_sha256"] == ap_rows["feature_manifest_sha256"]
          == geometry["feature_manifest_sha256"] == feature_sha
          and tuple(features["feature_names"]) == policies.FEATURES and features["GT_opened"] is False,
          "Both FIT-only supervision channels refer to the same sealed action features")
    indexed = {row["file"]:row for row in ap_rows["groups"]}
    rows = features["groups"]
    check(len(indexed) == len(ap_rows["groups"]) == len(rows) == len(geometry["groups"]) == 109
          and len({r["group"] for r in rows}) == len({r["image_id"] for r in rows}) == 109
          and [r["file"] for r in rows] == [f"{i:03d}.npz" for i in range(1,110)],
          "One uniquely named feature and target shard per original source group")
    for row, geo in zip(rows, geometry["groups"], strict=True):
        ap = indexed[row["file"]]
        check(geo["feature_manifest"] == row and ap["source_feature_sha256"] == row["sha256"]
              and all(ap[key] == row[key] for key in ("group","fold","image_id","actions")),
              "Original candidate populations and all feasible actions aligned for both channels")
        yield row, geo, ap


def candidate_weights(identity, eligible):
    """Each active seed contributes equally inside a single source group."""
    weight = np.zeros(len(eligible), dtype=np.float64)
    active = [(identity[:,0] == seed) & eligible for seed in SEEDS]
    nonempty = [mask for mask in active if mask.any()]
    for mask in nonempty:
        weight[mask] = 1./(len(nonempty)*int(mask.sum()))
    return weight


def load_shard(source, competition, ref, geo_ref, ap_ref, policies):
    name = ref["file"]
    f_path = source/"FEATURE_SHARDS_PRIVATE"/name
    g_path = source/"TARGET_SHARDS_PRIVATE"/name
    a_path = competition/"COMPETITION_SHARDS_PRIVATE"/name
    with (np.load(BytesIO(pin(f_path,ref["sha256"])),allow_pickle=False) as f,
          np.load(BytesIO(pin(g_path,geo_ref["target_sha256"])),allow_pickle=False) as g,
          np.load(BytesIO(pin(a_path,ap_ref["sha256"])),allow_pickle=False) as a):
        identity, actions, offsets, count = (f[k] for k in ("identity","actions","action_offsets","balanced_count"))
        matrix = f["features"]
        geometry, mask = g["delta_iou"], g["supervised_candidate"]
        ap, levels = a["delta_ap"], a["delta_ap_by_iou"]
        check(all(np.array_equal(f[key],a[key]) for key in ("identity","actions","action_offsets"))
              and tuple(f["feature_names"]) == policies.FEATURES,
              "Byte-pinned AP targets have the original feature action identities and order")
    n, m = ref["candidate_rows"], ref["actions"]
    check(identity.shape == (n,4) and identity.dtype == np.int64 and len({tuple(i) for i in identity}) == n
          and set(identity[:,0]).issubset(SEEDS) and np.all(identity[:,1] == ref["image_id"])
          and count.shape == mask.shape == (n,) and mask.dtype == np.bool_
          and offsets.shape == (n+1,) and offsets.dtype.kind in "iu" and offsets[0] == 0 and offsets[-1] == m
          and np.all(np.diff(offsets) >= 1) and actions.shape == (m,2)
          and matrix.shape == (m,len(policies.FEATURES)) and matrix.dtype == np.float64,
          "Legal source cohort sizes, unique identities, and complete candidate action slices")
    check(geometry.shape == ap.shape == (m,) and geometry.dtype == ap.dtype == np.float64
          and levels.shape == (m,10) and levels.dtype == np.float64 and np.isfinite(levels).all()
          and np.isfinite(ap).all() and np.all(np.abs(ap) <= 1)
          and np.allclose(ap,levels.mean(axis=1),rtol=0,atol=1e-15)
          and int(mask.sum()) == geo_ref["supervised_candidate_seed_rows"],
          "Continuous geometry and ten-threshold unilateral AP targets in unit fractions")
    alternatives = np.diff(offsets) > 1
    all_w = candidate_weights(identity,alternatives)
    match_w = candidate_weights(identity,alternatives & mask)
    masses = np.zeros((m,2),dtype=np.float64)
    constant_worth = np.zeros(len(OFFSETS), dtype=np.float64)
    trainable = np.zeros(m,dtype=bool)
    for i,k in enumerate(count):
        start,stop = int(offsets[i]),int(offsets[i+1])
        span = slice(start,stop)
        moves, sub = actions[span], matrix[span]
        balance = policies.feasible(k,moves,sub)
        check(ap[start+balance] == 0 and np.all(levels[start+balance] == 0)
              and (np.isfinite(geometry[span]).all() and geometry[start+balance] == 0
                   if mask[i] else np.isnan(geometry[span]).all()),
              "Exactly zero target for balanced action and explicit matched-only geometry labels")
        if not alternatives[i]:
            continue
        trainable[span] = True
        trainable[start+balance] = False
        masses[span,0] = all_w[i]/(len(moves)-1)
        masses[span,1] = match_w[i]/(len(moves)-1)
        masses[start+balance] = 0
        picks = np.clip(k+OFFSETS,moves[0,0],moves[-1,0])-moves[0,0]
        constant_worth += all_w[i]*ap[start+picks]
    check(int(np.isfinite(geometry).sum()) == geo_ref["supervised_actions"], "Complete published geometry exposure")
    counts = Counter(candidate_seed_rows=n, legal_actions=m, zero_budget_rows=int((count == 0).sum()),
                     single_action_rows=int((~alternatives).sum()),
                     nonzero_budget_single_action_rows=int(np.count_nonzero(~alternatives & (count > 0))),
                     geometry_single_action_rows=int(np.count_nonzero(~alternatives & mask)),
                     geometry_candidate_seed_rows=int(mask.sum()), geometry_legal_actions=int(np.isfinite(geometry).sum()),
                     ap_nonzero_actions=int(np.count_nonzero(ap)), ap_positive_actions=int(np.count_nonzero(ap > 0)),
                     ap_negative_actions=int(np.count_nonzero(ap < 0)), ap_trainable_actions=int(trainable.sum()),
                     matched_trainable_actions=int(np.count_nonzero(masses[:,1])),
                     unmatched_ap_nonzero_actions=int(np.count_nonzero((ap != 0)&~np.isfinite(geometry))))
    return (policies.basis(matrix[trainable]),masses[trainable],ap[trainable],geometry[trainable]),counts,constant_worth


def fit_ridge(standard, target, weights, lam):
    use = weights > 0
    x, y, w = standard[use],target[use],weights[use]
    check(np.isfinite(y).all() and np.isclose(w.sum(),1.,atol=1e-12,rtol=0)
          and x.shape[0]>0, "Finite supervised rows and normalized group weighting")
    st = x*np.sqrt(w[:,None])
    coef = np.linalg.solve(st.T@st+lam*np.eye(x.shape[1]), x.T@(w*y))
    fit = x@coef-y
    return coef, float(np.sum(w*fit**2))


def fit_all(source, src_result, competition, ap_result, policies):
    blocks, counts = [], Counter()
    offsets = np.zeros(len(OFFSETS),dtype=np.float64)
    for row, geo, ap in input_refs(source, src_result, competition, ap_result, policies):
        block, summary, constant_values = load_shard(source, competition, row, geo, ap, policies)
        counts.update(summary)
        counts["source_groups"] += 1
        counts["active_all_groups"] += int(np.any(block[1][:,0] > 0))
        counts["active_matched_groups"] += int(np.any(block[1][:,1] > 0))
        blocks.append(block)
        offsets += constant_values
    check(all(counts[k] == v for k,v in dict(candidate_seed_rows=19263,legal_actions=313135,
          geometry_candidate_seed_rows=1176,geometry_legal_actions=18308,source_groups=109).items()),
          "Complete previously admitted FIT source denominators")
    design, weights, delta_ap, delta_iou = (np.concatenate(x,axis=0) for x in zip(*blocks,strict=True))
    weights /= weights.sum(axis=0)
    all_mask, matched = weights.T
    squared = np.einsum("n,np,np->p",all_mask,design,design)
    scale = np.sqrt(squared)
    scale[scale == 0] = 1.
    standardized = design/scale
    coef, errors = zip(*(fit_ridge(standardized,y,w,policies.REGULARIZER) for y,w in
                         ((delta_ap,all_mask),(delta_ap,matched),(delta_iou,matched))),strict=True)
    offsets /= counts["active_all_groups"]
    offset = int(OFFSETS[policies.best_index(offsets,OFFSETS,0)])
    return (scale, np.stack(coef), offset),dict(counts),dict(zip(policies.FIT_ARMS,errors,strict=True)),offsets


def write(path, body):
    raw = (json.dumps(body,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.authority,args.authority_sha)
    paths = {"protocol","source_root","ap_root"}
    hashes = {"protocol_sha256","ap_result_sha256","ap_terminal_sha256"}
    check(set(config) == paths|hashes and config["protocol_sha256"] == PROTOCOL
          and all(Path(config[k]).is_absolute() for k in paths)
          and args.output.is_absolute() and not args.output.exists() and args.output.parent.is_dir()
          and sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
          and all(os.environ.get(key) == "1" for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")),
          "Fresh isolated single-thread CPU source fit")
    pin(Path(config["protocol"]),config["protocol_sha256"])
    started = perf_counter()
    source = Path(config["source_root"])
    ap = Path(config["ap_root"])
    source_result = closed_source(source,SOURCE_RESULT,SOURCE_TERMINAL,SOURCE_STATUS)
    ap_result = closed_source(ap,config["ap_result_sha256"],config["ap_terminal_sha256"],AP_STATUS)
    check(ap_result["source_feature_result_sha256"] == SOURCE_RESULT,
          "Observed AP targets derived from the exact independently closed source features")
    policies = load_policies()
    (scale, coef, offset),counts,errors,values = fit_all(source/"prepare",source_result,ap/"prepare",ap_result,policies)
    args.output.mkdir()
    model_path = args.output/"MODEL_PRIVATE.json"
    model_sha = write(model_path,dict(status=policies.MODEL_STATUS, input_feature_names=list(policies.FEATURES),
                      protocol_sha256=config["protocol_sha256"], source_result_sha256=SOURCE_RESULT,
                      ap_result_sha256=config["ap_result_sha256"], fit_arms=list(policies.FIT_ARMS),
                      capacity_per_arm=policies.CAPACITY, regularizer=policies.REGULARIZER,
                      rms=scale.tolist(), coefficients=coef.tolist(), constant_signed_offset=offset))
    verified = policies.load_model(model_path,model_sha)
    check(np.array_equal(verified.scale,scale) and np.array_equal(verified.coefficients,coef)
          and verified.constant_signed_offset==offset,"Private fitted model exact readback")
    result = dict(status=STATUS, model_sha256=model_sha, protocol_sha256=config["protocol_sha256"],
                  authority_sha256=args.authority_sha, source_result_sha256=SOURCE_RESULT,
                  ap_result_sha256=config["ap_result_sha256"], ap_terminal_sha256=config["ap_terminal_sha256"],
                  AP_targets_consumed=True, FIT=True, DEV=False, TEST=False, endpoint_scoring=False,
                  counts=counts, ridge_fits=3, capacity_per_arm=policies.CAPACITY,
                  weighted_source_training_mse=errors, constant_signed_offset=offset,
                  fixed_offset_marginal_utilities={str(i):float(v) for i,v in zip(OFFSETS,values,strict=True)},
                  runtime_seconds=perf_counter()-started,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), policies_sha256=POLICIES)
    digest = write(args.output/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=STATUS,result_sha256=digest)),flush=True)


if __name__ == "__main__":
    main()
