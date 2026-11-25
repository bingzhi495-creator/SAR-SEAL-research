"""Fit three equal-budget gain supervision variants on one exact FIT action bank."""
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


PROTOCOL = "8c33dd1743ecdcd8e329fb5003a04d2c069466ec7f59fd1751906708e384fc4a"
BANK_ADMISSION_SOURCE = "b29f2a654831e4cd512d88b87e7dfe016dc711f897173f5ce20aa50cc9ea0d80"
STATUS = "SPSD_PSEG_MATCHED_GAIN_CALIBRATION_FIT_COMPLETE"
INPUT_STATUS = "FIT_MATCHED_GAIN_CALIBRATION_PACKETS_ADMITTED"
SEED = 5297
BETA = .1
AUX_WEIGHT = 1.
MODES = ("midpoint", "gap")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Exact source: " + path.name)
    return raw


def previous_admission(path):
    module = ModuleType("verified_previous_admission")
    module.__file__ = str(path)
    exec(compile(pin(path, BANK_ADMISSION_SOURCE), str(path), "exec"), module.__dict__)
    return module


def component(gains, mode):
    if mode == "midpoint":
        return (gains[..., 0] + gains[..., 1]) / 2
    if mode == "gap":
        return (gains[..., 0] - gains[..., 1]).abs() / 2
    raise ValueError("Use midpoint or gap")


def candidate_loss(prediction, target, valid_actions, mode):
    predicted = component(prediction, mode)
    actual = component(target, mode)
    error = F.smooth_l1_loss(predicted, actual, beta=BETA, reduction="none")
    return ((error * valid_actions).sum(dim=-1) / valid_actions.sum(dim=-1)).mean()


def geometry_count(numerators, valid_actions):
    midpoint = component(numerators, "midpoint")
    gap = component(numerators, "gap")
    return dict(real_actions=int(valid_actions.sum().item()),
                exact_zero_true_midpoint=int(((midpoint == 0) & valid_actions).sum().item()),
                exact_zero_true_gap=int(((gap == 0) & valid_actions).sum().item()),
                positive_true_gap=int(((gap > 0) & valid_actions).sum().item()))


def component_fits(packets, action, source, matched):
    own = [action._packet_arrays(row, source["_feature_array"]) for row in packets]
    groups = source["_eligible_rows"]([row.group_id for row in packets], [row.image_id for row in packets], len(packets))
    center, scale = action._fit_moments(own, groups)
    features, (targets, weights, quota, valid_points, valid_actions, points, actions) = action._dense(own)
    standard, native = source["_inputs"](features, center, scale)
    true_gain = action.signed_action_deltas(targets, weights, quota)
    numerators = weights.flatten(-2) @ targets.flatten(-3, -2)
    need(torch.equal(numerators, numerators.round()), "Original binary occupancy yields exact integer gains")
    static_gains = action.signed_action_deltas(targets, torch.zeros_like(weights[:, :1]), quota)
    need(not static_gains.count_nonzero().item()
         and all(not component(static_gains, arm).count_nonzero().item() for arm in MODES),
         "Zero relative static action has zero transformed targets")
    draws = source["_batches"](groups, SEED)
    batches = torch.from_numpy(draws)
    draws_flat = batches.flatten()
    action_geometry = geometry_count(numerators, valid_actions)
    exposure_geometry = geometry_count(numerators[draws_flat], valid_actions[draws_flat])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        initial = source["PointHead"]().train()
    shared = dict(parameters=sum(p.numel() for p in initial.parameters()),
                  initial_tensor_sha256=source["_parameter_digest"](initial),
                  batch_sha256=digest(draws.astype("<i8", copy=False).tobytes()),
                  observed_candidates=int(batches.numel()), observed_endpoints=int(2 * points[batches].sum().item()),
                  observed_actions=int(actions[batches].sum().item()))
    need(all(shared[k] == matched[k] for k in shared), "Exact equal initial tensors, draws, and input exposures")
    models, audits = {}, {}
    for mode in MODES:
        started = perf_counter()
        model = deepcopy(initial)
        parameters = list(model.parameters())
        opt = torch.optim.AdamW(parameters, lr=.002, weight_decay=.01)
        for idx in batches:
            opt.zero_grad(set_to_none=True)
            logits = model(standard[idx], native[idx])
            pixel_bce = F.binary_cross_entropy_with_logits(logits, targets[idx], reduction="none").mean(dim=(-1, -2))
            bce = ((pixel_bce * valid_points[idx]).sum(dim=-1) / points[idx]).mean()
            estimated = action.signed_action_deltas(logits.sigmoid(), weights[idx], quota[idx])
            auxiliary = candidate_loss(estimated, true_gain[idx], valid_actions[idx], mode)
            loss = bce + AUX_WEIGHT * auxiliary
            need(bool(torch.isfinite(loss) & torch.isfinite(auxiliary)), "Finite component training loss")
            loss.backward()
            opt.step()
        counts = {int(v["step"].item()) for v in opt.state.values()}
        need(len(opt.state) == len(parameters) and counts == {1_000}, "Exactly 1,000 optimizer updates")
        models[mode] = model.eval()
        audits[mode] = dict(shared, optimizer_updates=1000, final_tensor_sha256=source["_parameter_digest"](model),
                            last_total_loss=float(loss.detach()), last_bce_loss=float(bce.detach()),
                            last_component_loss=float(auxiliary.detach()), elapsed_seconds=perf_counter()-started)
    return models, center, scale, audits, action_geometry, exposure_geometry


def main():
    p = ArgumentParser(description=__doc__, allow_abbrev=False)
    for field in ("protocol", "admission-worker", "deployment-fit", "capture", "points", "labels", "prior-fit",
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
    source = core._source(a.source_core)
    models, c, s, audits, true_counts, draw_counts = component_fits(bank, core, source, reg_audit)
    need(np.array_equal(c, center) and np.array_equal(s, scale)
         and len({entry["final_tensor_sha256"] for entry in (reg_audit, *audits.values())}) == 3,
         "Exact common scaling and distinct three fitted endpoints")
    checkpoint = a.out / "FITTED_WEIGHTS_PRIVATE.pt"
    states = dict(regression=received["bce_signed_action"].state_dict())
    states.update({name:model.state_dict() for name, model in models.items()})
    with checkpoint.open("xb") as stream:
        torch.save(dict(models=states, center=torch.from_numpy(center), scale=torch.from_numpy(scale), seed=SEED), stream)
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
                  counts=source_result["counts"], target_geometry=true_counts, exposure_geometry=draw_counts,
                  zero_static_components=True, source_gate_labels_opened=False, source_gate_points_opened=False,
                  historical_test_opened=False, output_checkpoint_sha256=digest(checkpoint.read_bytes()),
                  train_audit=dict(shared, heads=dict(regression=reg_audit, **audits)),
                  elapsed_seconds=perf_counter()-start)
    sha = loader.write(a.out / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=STATUS, result_sha256=sha)), flush=True)


if __name__ == "__main__":
    main()
