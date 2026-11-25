"""Fit capacity-matched radiometric inputs with one frozen action bank and objective."""
from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
import torch
from torch.nn import functional as F


PROTOCOL = "1ea013fecd33b6626e69faf9764d9358808a519c73ad3a539f4b23d4fd631201"
BANK_ADMISSION_SOURCE = "b29f2a654831e4cd512d88b87e7dfe016dc711f897173f5ce20aa50cc9ea0d80"
FEATURE_SOURCE = "7015378c2b82db61cf223ee78ad7469a5d29dcaf00fc69490d70aaa50beef2f4"
STATUS = "SPSD_PSEG_MATCHED_RADIOMETRIC_FEATURE_FIT_COMPLETE"
INPUT_STATUS = "FIT_MATCHED_RADIOMETRIC_FEATURE_PACKETS_ADMITTED"
SEED = 5297
BETA = .1
AUX_WEIGHT = 1.
MODES = ("joint", "per_side")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Exact source: " + path.name)
    return raw


def previous_admission(path, expected=BANK_ADMISSION_SOURCE):
    module = ModuleType("verified_previous_admission")
    module.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), module.__dict__)
    return module


def radiometric_fits(packets, action, source, matched, feature_core, raw_center, raw_scale):
    own = [action._packet_arrays(row, source["_feature_array"]) for row in packets]
    groups = source["_eligible_rows"]([row.group_id for row in packets], [row.image_id for row in packets], len(packets))
    raw, (targets, weights, quota, valid_points, valid_actions, endpoints, actions) = action._dense(own)
    native = torch.from_numpy(raw[..., 0].copy())
    true_gain = action.signed_action_deltas(targets, weights, quota)
    draws = source["_batches"](groups, SEED)
    batches = torch.from_numpy(draws)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        initial = source["PointHead"]().train()
    shared = dict(parameters=sum(p.numel() for p in initial.parameters()),
                  initial_tensor_sha256=source["_parameter_digest"](initial),
                  batch_sha256=digest(draws.astype("<i8", copy=False).tobytes()),
                  observed_candidates=int(batches.numel()), observed_endpoints=int(2 * endpoints[batches].sum().item()),
                  observed_actions=int(actions[batches].sum().item()))
    need(all(shared[k] == matched[k] for k in shared), "Identical architecture, initialization and action exposures")
    transformed = {mode:[feature_core.transform_candidate(row[0], mode) for row in own] for mode in MODES}
    different = sum(int(not np.array_equal(j,s)) for j,s in zip(transformed["joint"], transformed["per_side"], strict=True))
    need(different > 0, "A realized distribution-reference intervention")
    models, audits, moments, feature_audit = {}, {}, {}, {"different_joint_vs_per_side_candidates":different}
    raw_dup_center = np.concatenate((raw_center, raw_center[:9]))
    raw_dup_scale = np.concatenate((raw_scale, raw_scale[:9]))
    for mode, tables in transformed.items():
        packed = [(x, targets, weights, k) for x, (_, targets, weights, k) in zip(tables, own, strict=True)]
        center, scale = action._fit_moments(packed, groups)
        need(all(np.array_equal(x[..., :14], original[0]) and np.array_equal(x[..., 14:23], original[0][..., :9])
                 for x, original in zip(tables, own, strict=True)), "Exact shared raw feature and duplicate-coordinate parity")
        center[:23] = raw_dup_center
        scale[:23] = raw_dup_scale
        moments[mode] = center, scale
        crafted = np.zeros((*raw.shape[:-1], 28), dtype=np.float32)
        cue_hash = sha256()
        nonzero = 0
        for i, x in enumerate(tables):
            crafted[i, :len(x)] = x
            cues = x[..., 23:]
            cue_hash.update(cues.astype("<f4", copy=False).tobytes(order="C"))
            nonzero += int(np.count_nonzero(cues))
        feature_audit[mode] = dict(fit_cue_tensor_sha256=cue_hash.hexdigest(), nonzero_cue_values=nonzero,
                                   cue_center=center[23:].tolist(), cue_scale=scale[23:].tolist())
        standard = torch.from_numpy(np.ascontiguousarray((crafted - center) / scale, dtype=np.float32))
        started = perf_counter()
        fitted = deepcopy(initial)
        params = list(fitted.parameters())
        opt = torch.optim.AdamW(params, lr=.002, weight_decay=.01)
        for idx in batches:
            opt.zero_grad(set_to_none=True)
            logits = feature_core.forward_cues(fitted, standard[idx], native[idx])
            pixel_bce = F.binary_cross_entropy_with_logits(logits, targets[idx], reduction="none").mean(dim=(-1, -2))
            bce = ((pixel_bce * valid_points[idx]).sum(dim=-1) / endpoints[idx]).mean()
            estimated = action.signed_action_deltas(logits.sigmoid(), weights[idx], quota[idx])
            error = F.smooth_l1_loss(estimated, true_gain[idx], beta=BETA, reduction="none").mean(dim=-1)
            gain = ((error * valid_actions[idx]).sum(dim=-1) / actions[idx]).mean()
            objective = bce + AUX_WEIGHT * gain
            need(bool(torch.isfinite(objective) & torch.isfinite(gain)), "Original finite fixed auxiliary loss")
            objective.backward()
            opt.step()
        observed = {int(v["step"].item()) for v in opt.state.values()}
        need(len(opt.state) == len(params) and observed == {1_000}, "Actual matched optimizer steps")
        models[mode] = fitted.eval()
        audits[mode] = dict(shared, optimizer_updates=1000, final_tensor_sha256=source["_parameter_digest"](fitted),
                             last_total_loss=float(objective.detach()), last_bce_loss=float(bce.detach()),
                             last_signed_action_loss=float(gain.detach()), elapsed_seconds=perf_counter()-started)
    return models, moments, audits, feature_audit


def main():
    p = ArgumentParser(description=__doc__, allow_abbrev=False)
    for field in ("protocol", "admission-worker", "feature-source", "deployment-fit", "capture", "points", "labels", "prior-fit",
                  "previous", "action-core", "source-core", "attraction", "selector", "out"):
        p.add_argument("--" + field, type=Path, required=True)
    a = p.parse_args()
    start = perf_counter()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and torch.get_num_threads() == 1
         and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
         "Exact existing fit CPU runtime")
    pin(a.protocol, PROTOCOL)
    need(not a.out.exists(), "Fresh location required")
    admission = previous_admission(a.admission_worker)
    loader, core, bank, source_result, folds = admission.admitted_bank(a)
    a.out.mkdir()
    input_sha = loader.write(a.out / "INPUT_SAFE.json", dict(
        status=INPUT_STATUS, protocol_sha256=PROTOCOL, fit_only=True, source_fit_result_sha256=admission.BANK_RESULT,
        aligned_action_packet_sha256=source_result["aligned_action_packet_sha256"], counts=source_result["counts"],
        out_of_fold_probabilities_sha256=source_result["out_of_fold_probabilities_sha256"],
        source_gate_labels_opened=False, source_gate_points_opened=False, historical_test_opened=False))
    print(json.dumps(dict(status=INPUT_STATUS, input_receipt_sha256=input_sha)), flush=True)
    core.MODES = ("bce_signed_action",)
    received, center, scale, shared = core.train_matched_action(bank, source_core=a.source_core, seed=SEED)
    reg_audit = shared.pop("heads")["bce_signed_action"]
    previous = source_result["train_audit"]["heads"]["deployment_bank"]
    need(all(reg_audit[k] == value for k, value in previous.items() if k != "elapsed_seconds")
         and reg_audit["final_tensor_sha256"] == admission.REGRESSION,
         "Exact deployment-bank reference-wise regression control")
    feature_core = previous_admission(a.feature_source, FEATURE_SOURCE)
    source = core._source(a.source_core)
    models, moments, audits, feature_audit = radiometric_fits(bank, core, source, reg_audit, feature_core, center, scale)
    need(len({entry["final_tensor_sha256"] for entry in (reg_audit, *audits.values())}) == 3,
         "Realized distinct three-arm endpoints")
    checkpoint = a.out / "FITTED_WEIGHTS_PRIVATE.pt"
    states = dict(regression=received["bce_signed_action"].state_dict())
    states.update({name:model.state_dict() for name, model in models.items()})
    moments["regression"] = center, scale
    with checkpoint.open("xb") as stream:
        torch.save(dict(models=states, center={m:torch.from_numpy(v[0]) for m,v in moments.items()},
                        scale={m:torch.from_numpy(v[1]) for m,v in moments.items()}, seed=SEED), stream)
        stream.flush()
        os.fsync(stream.fileno())
    result = dict(status=STATUS, protocol_sha256=PROTOCOL, worker_sha256=digest(Path(__file__).read_bytes()),
                  action_core_sha256=loader.ACTION_CORE, training_core_sha256=loader.OCCUPANCY_CORE,
                  admission_worker_sha256=BANK_ADMISSION_SOURCE, input_receipt_sha256=input_sha,
                  source_fit_result_sha256=admission.BANK_RESULT, source_fit_readback_sha256=admission.BANK_READBACK,
                  aligned_action_packet_sha256=source_result["aligned_action_packet_sha256"],
                  out_of_fold_probabilities_sha256=source_result["out_of_fold_probabilities_sha256"],
                  head_pair_baseline_tensor_parity=True, regression_baseline_tensor_sha256=admission.REGRESSION,
                  out_of_fold_actual_updates=sum(fold["train_head"]["optimizer_updates"] for fold in folds),
                  total_optimizer_updates=3000, detector_forwards=0, smooth_l1_beta=BETA, auxiliary_weight=AUX_WEIGHT,
                  counts=source_result["counts"], feature_audit=feature_audit, feature_worker_sha256=FEATURE_SOURCE,
                  all_raw_14_scalers_frozen=True, capacity_aware_replacement_width=5,
                  source_gate_labels_opened=False, source_gate_points_opened=False,
                  historical_test_opened=False, output_checkpoint_sha256=digest(checkpoint.read_bytes()),
                  train_audit=dict(shared, heads=dict(regression=reg_audit, **audits)),
                  elapsed_seconds=perf_counter()-start)
    sha = loader.write(a.out / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=STATUS, result_sha256=sha)), flush=True)


if __name__ == "__main__":
    main()
