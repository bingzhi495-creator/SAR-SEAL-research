"""Fit equal-budget robust pair ranking and signed action regression on the same FIT bank."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import NamedTuple

import numpy as np
import torch
from torch.nn import functional as F


PROTOCOL = "fb9db9f23fd81d95ed2537b396ebb610f2b7a422c18931c78b940288643a7b1b"
BANK_WORKER = "5419c9179f1ca82bf21436049450e623725c7e45b1cb2b7b19c2541e57e26fb7"
BANK_RESULT = "4872b63ee24c8021e09612314161c691006607dceb3ea92a2ce27690016836e4"
BANK_TERMINAL = "52e572c8916cafff8297f2308b95aa364fd6de7329e747f1e54275a32664548d"
BANK_READBACK = "8b280c6777d53e7125fb1c1f342d3eb7c33811ee18c700bf0f7be580a06f8af1"
REGRESSION = "d41d2538ae03ff76b385f7e25ca5288fea9930920678e50ceb0e1863047ef456"
STATUS = "SPSD_PSEG_MATCHED_ROBUST_RANKING_FIT_COMPLETE"
INPUT_STATUS = "FIT_MATCHED_ROBUST_RANKING_PACKETS_ADMITTED"
SEED = 5297
TEMPERATURE = .1
AUXILIARY_WEIGHT = 1.


class Pairs(NamedTuple):
    left: torch.Tensor
    right: torch.Tensor
    sign: torch.Tensor
    valid: torch.Tensor
    possible: torch.Tensor

    def subset(self, indices):
        return Pairs(self.left, self.right, self.sign[indices], self.valid[indices], self.possible[indices])


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(blob):
    return sha256(blob).hexdigest()


def pin(path, expected):
    blob = path.read_bytes()
    need(digest(blob) == expected, "Exact required digest: " + path.name)
    return blob


def load(path, expected):
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), mod.__dict__)
    return mod


def rank_targets(numerators, action_mask):
    """Build exact sign comparisons with the zero static anchor at column A."""
    need(numerators.ndim == 3 and numerators.shape[-1] == 2 and action_mask.dtype == torch.bool
         and action_mask.shape == numerators.shape[:2]
         and bool(torch.isfinite(numerators).all() & torch.equal(numerators, numerators.round())),
         "Expected two integer-valued reference numerators per action")
    m = numerators.shape[1]
    left, right = torch.triu_indices(m + 1, m + 1, 1, device=numerators.device)
    robust = F.pad(numerators.amin(dim=-1), (0, 1))
    all_valid = F.pad(action_mask, (0, 1), value=True)
    possible = all_valid[:, left] & all_valid[:, right]
    sign = (robust[:, left] - robust[:, right]).sign()
    return Pairs(left, right, sign, possible & (sign != 0), possible)


def rank_penalty(estimated, pairs):
    """Every sampled candidate contributes its own valid-pair average."""
    robust = F.pad(estimated.amin(dim=-1), (0, 1))
    difference = robust[:, pairs.left] - robust[:, pairs.right]
    costs = TEMPERATURE * F.softplus(-pairs.sign * difference / TEMPERATURE)
    return ((costs * pairs.valid).sum(dim=-1) / pairs.valid.sum(dim=-1).clamp(min=1)).mean()


def pair_counts(pairs, actions):
    anchor = pairs.right == actions
    eligible = pairs.valid.any(dim=-1)
    return dict(possible_comparisons=int(pairs.possible.sum().item()),
                supervised_comparisons=int(pairs.valid.sum().item()),
                ties_ignored=int((pairs.possible & ~pairs.valid).sum().item()),
                static_comparisons=int(pairs.possible[:, anchor].sum().item()),
                static_comparisons_supervised=int(pairs.valid[:, anchor].sum().item()),
                candidates_with_supervised_comparisons=int(eligible.sum().item()),
                candidate_instances=int(eligible.shape[0]))


def fit_rank(packets, action, source, saved):
    own = [action._packet_arrays(row, source["_feature_array"]) for row in packets]
    groups = source["_eligible_rows"]([p.group_id for p in packets], [p.image_id for p in packets], len(packets))
    center, scale = action._fit_moments(own, groups)
    feature, (target, weight, quota, valid_points, valid_actions, points, actions) = action._dense(own)
    standardized, native = source["_inputs"](feature, center, scale)
    numerators = weight.flatten(-2) @ target.flatten(-3, -2)
    pairs = rank_targets(numerators, valid_actions)
    draws = source["_batches"](groups, SEED)
    batches = torch.from_numpy(draws)
    train_pairs = pair_counts(pairs, weight.shape[1])
    exposure = pair_counts(pairs.subset(batches.flatten()), weight.shape[1])
    need(train_pairs["supervised_comparisons"] > 0 and train_pairs["candidate_instances"] == 848,
         "Actual plan has nonempty robust rank supervision")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        model = source["PointHead"]().train()
    parameters = list(model.parameters())
    match = dict(parameters=sum(p.numel() for p in parameters),
                 observed_candidates=int(batches.numel()), observed_endpoints=int(2 * points[batches].sum().item()),
                 observed_actions=int(actions[batches].sum().item()),
                 initial_tensor_sha256=source["_parameter_digest"](model),
                 batch_sha256=digest(draws.astype("<i8", copy=False).tobytes()))
    need(all(v == saved[k] for k, v in match.items()), "Ranking arm initialization, samples and dimensions match")
    start = perf_counter()
    optimizer = torch.optim.AdamW(parameters, lr=.002, weight_decay=.01)
    for idx in batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model(standardized[idx], native[idx])
        pixel_bce = F.binary_cross_entropy_with_logits(logits, target[idx], reduction="none").mean(dim=(-1, -2))
        bce = ((pixel_bce * valid_points[idx]).sum(dim=-1) / points[idx]).mean()
        estimated = action.signed_action_deltas(logits.sigmoid(), weight[idx], quota[idx])
        ranking = rank_penalty(estimated, pairs.subset(idx))
        loss = bce + AUXILIARY_WEIGHT * ranking
        need(bool(torch.isfinite(loss) & torch.isfinite(ranking)), "Finite ranking objective")
        loss.backward()
        optimizer.step()
    optimizer_steps = {int(v["step"].item()) for v in optimizer.state.values()}
    need(len(optimizer.state) == len(parameters) and optimizer_steps == {1_000},
         "Observed ranking optimizer update parity")
    match.update(optimizer_updates=1000, final_tensor_sha256=source["_parameter_digest"](model),
                 last_total_loss=float(loss.detach()), last_bce_loss=float(bce.detach()),
                 last_ranking_loss=float(ranking.detach()), elapsed_seconds=perf_counter()-start)
    return model.eval(), center, scale, match, train_pairs, exposure


def admitted_bank(a):
    bank = load(a.deployment_fit / "fit_banks.py", BANK_WORKER)
    done = json.loads(pin(a.deployment_fit / "observed_fit/TERMINAL.json", BANK_TERMINAL))
    result = json.loads(pin(a.deployment_fit / "actual/RESULT_SAFE.json", BANK_RESULT))
    readback = json.loads(pin(a.deployment_fit / "READBACK_SAFE.json", BANK_READBACK))
    need(done["actual_exit"] == 0 and done["output_footer_verified"] is True
         and done["worker_sha256"] == BANK_WORKER and done["result_sha256"] == BANK_RESULT
         and result["worker_sha256"] == BANK_WORKER and result["protocol_sha256"] == bank.PROTOCOL
         and readback["terminal_sha256"] == BANK_TERMINAL and readback["fit_result_sha256"] == BANK_RESULT
         and readback["final_tensor_sha256"]["deployment_bank"] == REGRESSION
         and result["train_audit"]["heads"]["deployment_bank"]["final_tensor_sha256"] == REGRESSION,
         "Closed, verified and unchanged deployment-bank FIT authority")
    original = load(a.previous / "fit_action.py", bank.ORIGINAL)
    earlier = json.loads(pin(a.previous / "actual/RESULT_SAFE.json", bank.ORIGINAL_RESULT))
    action = load(a.action_core, original.ACTION_CORE)
    occupancy = load(a.source_core, original.OCCUPANCY_CORE)
    attraction = load(a.attraction, original.ATTRACTION)
    selector = load(a.selector, bank.SELECTOR)
    packets, counts, packet_sha, incumbents = bank.original_packets(a, original, action, attraction, earlier)
    probabilities, folds, fold_sha, prob_sha = bank.crossfit_occupancy(packets, action, occupancy, a.source_core)
    aligned, observed_audit, action_sha = bank.aligned_packets(packets, probabilities, incumbents, selector)
    need(counts == result["counts"] and packet_sha == result["fit_packet_sha256"]
         and fold_sha == result["out_of_fold_group_sha256"] and prob_sha == result["out_of_fold_probabilities_sha256"]
         and action_sha == result["aligned_action_packet_sha256"] and observed_audit == result["bank_audit"],
         "Exact FIT tensors, OOF predictions and deployment action geometry")
    return original, action, aligned, result, folds


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for field in ("protocol", "deployment-fit", "capture", "points", "labels", "prior-fit", "previous",
                  "action-core", "source-core", "attraction", "selector", "out"):
        parser.add_argument("--" + field, type=Path, required=True)
    a = parser.parse_args()
    started = perf_counter()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and torch.get_num_threads() == 1
         and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
         "Use the paired control's exact current CPU runtime")
    pin(a.protocol, PROTOCOL)
    need(not a.out.exists(), "Use a fresh fit location")
    loader, core, packets, original, folds = admitted_bank(a)
    a.out.mkdir()
    receipt = loader.write(a.out / "INPUT_SAFE.json", dict(
        status=INPUT_STATUS, fit_only=True, protocol_sha256=PROTOCOL, source_fit_result_sha256=BANK_RESULT,
        counts=original["counts"], aligned_action_packet_sha256=original["aligned_action_packet_sha256"],
        out_of_fold_probabilities_sha256=original["out_of_fold_probabilities_sha256"],
        source_gate_labels_opened=False, source_gate_points_opened=False, historical_test_opened=False))
    print(json.dumps(dict(status=INPUT_STATUS, input_receipt_sha256=receipt)), flush=True)
    core.MODES = ("bce_signed_action",)
    received, center, scale, original_audit = core.train_matched_action(packets, source_core=a.source_core, seed=SEED)
    regression = received["bce_signed_action"]
    audit = original_audit.pop("heads")["bce_signed_action"]
    expected = original["train_audit"]["heads"]["deployment_bank"]
    need(all(audit[k] == value for k, value in expected.items() if k != "elapsed_seconds")
         and audit["final_tensor_sha256"] == REGRESSION, "Exact current fixed-bank regression control reproduced")
    source = core._source(a.source_core)
    rank, rank_center, rank_scale, rank_audit, comparisons, exposures = fit_rank(packets, core, source, audit)
    need(np.array_equal(center, rank_center) and np.array_equal(scale, rank_scale)
         and rank_audit["final_tensor_sha256"] != REGRESSION, "Equal FIT moments and distinct trained objectives")
    checkpoint = a.out / "FITTED_WEIGHTS_PRIVATE.pt"
    with checkpoint.open("xb") as stream:
        torch.save(dict(models=dict(regression=regression.state_dict(), robust_rank=rank.state_dict()),
                        center=torch.from_numpy(center), scale=torch.from_numpy(scale), seed=SEED), stream)
        stream.flush()
        os.fsync(stream.fileno())
    result = dict(status=STATUS, protocol_sha256=PROTOCOL, worker_sha256=digest(Path(__file__).read_bytes()),
                  action_core_sha256=loader.ACTION_CORE, training_core_sha256=loader.OCCUPANCY_CORE,
                  source_fit_result_sha256=BANK_RESULT, source_fit_readback_sha256=BANK_READBACK,
                  counts=original["counts"], input_receipt_sha256=receipt,
                  aligned_action_packet_sha256=original["aligned_action_packet_sha256"],
                  out_of_fold_probabilities_sha256=original["out_of_fold_probabilities_sha256"],
                  head_pair_baseline_tensor_parity=True, regression_baseline_tensor_sha256=REGRESSION,
                  out_of_fold_actual_updates=sum(v["train_head"]["optimizer_updates"] for v in folds),
                  total_optimizer_updates=2000, detector_forwards=0, temperatures=dict(rank=TEMPERATURE, smooth_l1=.1),
                  auxiliary_weight=AUXILIARY_WEIGHT, robust_rank_comparisons=comparisons, robust_rank_exposures=exposures,
                  source_gate_labels_opened=False, source_gate_points_opened=False, historical_test_opened=False,
                  output_checkpoint_sha256=digest(checkpoint.read_bytes()),
                  train_audit=dict(original_audit, heads=dict(regression=audit, robust_rank=rank_audit)),
                  elapsed_seconds=perf_counter()-started)
    checksum = loader.write(a.out / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=STATUS, result_sha256=checksum)), flush=True)


if __name__ == "__main__":
    main()
