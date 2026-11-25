"""Train equal-budget aligned and scrambled frozen spatial cues."""
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


PROTOCOL = "fbc0db22a2784e5da2cedb7260d9a0c182070c33654000e5ae1b4a1f35fa57a6"
BANK_ADMISSION_SOURCE = "b29f2a654831e4cd512d88b87e7dfe016dc711f897173f5ce20aa50cc9ea0d80"
FEATURE_SOURCE = "3f06e771ac41d03073aec68400f481235a17aa9c66a6a0c310c0a58be69b7558"
STATUS = "SPSD_PSEG_MATCHED_SPATIAL_CARRIER_CUE_FIT_COMPLETE"
INPUT_STATUS = "FIT_MATCHED_SPATIAL_CARRIER_CUE_PACKETS_ADMITTED"
SEED = 5297
BETA = .1
AUX_WEIGHT = 1.
MODES = ("spatial", "scrambled")


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


def admitted_bank(a, old, acquisition, closed_cues):
    bank = old.load(a.deployment_fit / "fit_banks.py", old.BANK_WORKER)
    done = json.loads(pin(a.deployment_fit/"observed_fit/TERMINAL.json", old.BANK_TERMINAL))
    result = json.loads(pin(a.deployment_fit/"actual/RESULT_SAFE.json", old.BANK_RESULT))
    back = json.loads(pin(a.deployment_fit/"READBACK_SAFE.json", old.BANK_READBACK))
    need(done["actual_exit"] == 0 and done["output_footer_verified"] is True
         and done["worker_sha256"] == old.BANK_WORKER and done["result_sha256"] == old.BANK_RESULT
         and result["worker_sha256"] == old.BANK_WORKER and result["protocol_sha256"] == bank.PROTOCOL
         and back["terminal_sha256"] == old.BANK_TERMINAL and back["fit_result_sha256"] == old.BANK_RESULT
         and back["final_tensor_sha256"]["deployment_bank"] == old.REGRESSION
         and result["train_audit"]["heads"]["deployment_bank"]["final_tensor_sha256"] == old.REGRESSION,
         "Exact previous natural and independently verified deployment bank")
    original = old.load(a.previous/"fit_action.py", bank.ORIGINAL)
    earlier = json.loads(pin(a.previous/"actual/RESULT_SAFE.json", bank.ORIGINAL_RESULT))
    core = old.load(a.action_core, original.ACTION_CORE)
    occupancy = old.load(a.source_core, original.OCCUPANCY_CORE)
    attraction = old.load(a.attraction, original.ATTRACTION)
    selector = old.load(a.selector, bank.SELECTOR)
    packets, counts, packet_sha, incumbent = bank.original_packets(a, original, core, attraction, earlier)
    joined, cue_receipt = acquisition.join_fit(a, closed_cues, packets, incumbent)
    estimates, folds, fold_sha, prediction_sha = bank.crossfit_occupancy(packets, core, occupancy, a.source_core)
    aligned, feature_audit, cue_packet_sha = bank.aligned_packets(packets, estimates, incumbent, selector)
    need(counts == result["counts"] and packet_sha == result["fit_packet_sha256"]
         and fold_sha == result["out_of_fold_group_sha256"] and prediction_sha == result["out_of_fold_probabilities_sha256"]
         and cue_packet_sha == result["aligned_action_packet_sha256"] and feature_audit == result["bank_audit"],
         "Identical original FIT proposal selection and action tensors")
    return original, core, aligned, result, folds, joined, cue_receipt


def spatial_fits(packets, action, source, matched, feature_core, raw_center, raw_scale, joined):
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

    def filled(mode):
        return [feature_core.transform_candidate(row[0], cue, mode, ordinal, rank)
                for row, (cue, ordinal, rank) in zip(own, joined, strict=True)]

    real = filled("spatial")
    control = filled("scrambled")
    differ = sum(int(not np.array_equal(j[...,22:],s[...,22:])) for j,s in zip(real, control, strict=True))
    need(differ > 0, "A realized correspondence intervention")
    for aligned, scrambled in zip(real, control, strict=True):
        for side in (0,1):
            order = np.lexsort(aligned[:,side,22:].T)
            permuted_order = np.lexsort(scrambled[:,side,22:].T)
            need(np.array_equal(aligned[order,side,22:], scrambled[permuted_order,side,22:]),
                 "Complete per-side six-coordinate vectors preserve their multiset")
    moments_input = [(x, y, w, k) for x, (_, y, w, k) in zip(real, own, strict=True)]
    center, scale = action._fit_moments(moments_input, groups)
    center[:22] = np.concatenate((raw_center, raw_center[:8]))
    scale[:22] = np.concatenate((raw_scale, raw_scale[:8]))
    models, audits = {}, {}
    moments = {mode:(center, scale) for mode in MODES}
    feature_audit = dict(different_spatial_vs_scrambled_candidates=differ, shared_cue_center=center[22:].tolist(),
                         shared_cue_scale=scale[22:].tolist())

    for mode, tables in (("spatial", real), ("scrambled", control)):
        need(all(np.array_equal(x[...,:14], orig[0]) and np.array_equal(x[...,14:22], orig[0][...,:8])
                 for x, orig in zip(tables, own, strict=True)), "Exact raw and retained duplicate feature parity")
        crafted = np.zeros((*raw.shape[:-1], 28), dtype=np.float32)
        cues = sha256()
        for i, x in enumerate(tables):
            crafted[i,:len(x)] = x
            cues.update(x[...,22:].astype("<f4", copy=False).tobytes(order="C"))
        feature_audit[mode] = dict(fit_cue_tensor_sha256=cues.hexdigest())
        standard = torch.from_numpy(np.ascontiguousarray((crafted - center) / scale, dtype=np.float32))
        started = perf_counter()
        model = deepcopy(initial)
        params = list(model.parameters())
        opt = torch.optim.AdamW(params, lr=.002, weight_decay=.01)
        for idx in batches:
            opt.zero_grad(set_to_none=True)
            logits = feature_core.forward_cues(model, standard[idx], native[idx])
            pixel_bce = F.binary_cross_entropy_with_logits(logits, targets[idx], reduction="none").mean(dim=(-1, -2))
            bce = ((pixel_bce * valid_points[idx]).sum(dim=-1) / endpoints[idx]).mean()
            estimated = action.signed_action_deltas(logits.sigmoid(), weights[idx], quota[idx])
            error = F.smooth_l1_loss(estimated, true_gain[idx], beta=BETA, reduction="none").mean(dim=-1)
            gain = ((error * valid_actions[idx]).sum(dim=-1) / actions[idx]).mean()
            objective = bce + AUX_WEIGHT * gain
            need(bool(torch.isfinite(objective) & torch.isfinite(gain)), "Exact original finite total objective")
            objective.backward()
            opt.step()
        observed = {int(v["step"].item()) for v in opt.state.values()}
        need(len(opt.state) == len(params) and observed == {1_000}, "Actual matched optimizer updates")
        models[mode] = model.eval()
        audits[mode] = dict(shared, optimizer_updates=1000, final_tensor_sha256=source["_parameter_digest"](model),
                             last_total_loss=float(objective.detach()), last_bce_loss=float(bce.detach()),
                             last_signed_action_loss=float(gain.detach()), elapsed_seconds=perf_counter()-started)
    return models, moments, audits, feature_audit


def main():
    p = ArgumentParser(description=__doc__, allow_abbrev=False)
    for field in ("protocol", "admission-worker", "feature-source", "cue-reader", "cue-lock", "cue-capture", "deployment-fit", "capture", "points", "labels", "prior-fit",
                  "previous", "action-core", "source-core", "attraction", "selector", "out"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--cue-reader-sha", required=True)
    p.add_argument("--cue-lock-sha", required=True)
    a = p.parse_args()
    start = perf_counter()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and torch.get_num_threads() == 1
         and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
         "Exact existing fit CPU runtime")
    pin(a.protocol, PROTOCOL)
    need(not a.out.exists(), "Fresh location required")
    acquisition = previous_admission(a.cue_reader, a.cue_reader_sha)
    closed_cues = acquisition.admitted_cues(a)
    admission = previous_admission(a.admission_worker)
    loader, core, bank, source_result, folds, joined, cue_receipt = admitted_bank(a, admission, acquisition, closed_cues)
    a.out.mkdir()
    input_sha = loader.write(a.out / "INPUT_SAFE.json", dict(
        status=INPUT_STATUS, protocol_sha256=PROTOCOL, fit_only=True, source_fit_result_sha256=admission.BANK_RESULT,
        aligned_action_packet_sha256=source_result["aligned_action_packet_sha256"], counts=source_result["counts"],
        out_of_fold_probabilities_sha256=source_result["out_of_fold_probabilities_sha256"],
        source_gate_labels_opened=False, source_gate_points_opened=False, historical_test_opened=False,
        **cue_receipt))
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
    models, moments, audits, feature_audit = spatial_fits(bank, core, source, reg_audit, feature_core, center, scale, joined)
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
                  all_raw_14_scalers_frozen=True, capacity_aware_replacement_width=6, cue_shared_scaler=True,
                  source_gate_labels_opened=False, source_gate_points_opened=False,
                  historical_test_opened=False, output_checkpoint_sha256=digest(checkpoint.read_bytes()),
                  train_audit=dict(shared, heads=dict(regression=reg_audit, **audits)),
                  elapsed_seconds=perf_counter()-start, **cue_receipt)
    sha = loader.write(a.out / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=STATUS, result_sha256=sha)), flush=True)


if __name__ == "__main__":
    main()
