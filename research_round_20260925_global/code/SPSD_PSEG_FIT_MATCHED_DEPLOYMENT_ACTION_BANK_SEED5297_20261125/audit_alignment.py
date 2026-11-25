"""Recompute OOF FIT proposal geometry from the exact closed fit receipts."""
from argparse import ArgumentParser
from collections import defaultdict
import json
import os
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np


FIT = "4872b63ee24c8021e09612314161c691006607dceb3ea92a2ce27690016836e4"
READBACK = "8b280c6777d53e7125fb1c1f342d3eb7c33811ee18c700bf0f7be580a06f8af1"
WORKER = "5419c9179f1ca82bf21436049450e623725c7e45b1cb2b7b19c2541e57e26fb7"
TERMINAL = "52e572c8916cafff8297f2308b95aa364fd6de7329e747f1e54275a32664548d"
STATUS = "SPSD_PSEG_FIT_OOF_ACTION_ALIGNMENT_COMPLETE"


def span_line(bank, vector):
    data = bank.reshape((len(bank), -1)).astype(np.float64)
    response = vector.ravel().astype(np.float64)
    _, singular, basis = np.linalg.svd(data, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
    if not np.any(response):
        return rank, None, None, False
    residual = response - basis[:rank].T @ (basis[:rank] @ response)
    energy = float(residual @ residual / (response @ response))
    distance = int(np.square(data - response).sum(axis=1).min())
    if not 0 <= energy <= 1 + 1e-10:
        raise ValueError("Invalid squared residual fraction")
    return rank, energy, distance, distance == 0


def summarize(packets, observations):
    by_group = defaultdict(lambda: defaultdict(list))
    for packet, (rank, residual, distance, exact) in zip(packets, observations, strict=True):
        by_group[packet.group_id][packet.image_id].append(
            (0 if residual is None else residual, 0 if distance is None else distance))
    groups = [np.mean([np.mean(v, axis=0) for v in images.values()], axis=0) for images in by_group.values()]
    group_mean = np.mean(groups, axis=0)
    active = [r for r in observations if r[1] is not None]
    results = {}
    for key, j in (("residual", 1), ("nearest_xor", 2)):
        values = np.asarray([r[j] for r in active])
        results[key] = dict(changed_only_mean=float(values.mean()), points=np.quantile(values, (.1, .5, .9)).tolist(),
                            all_groups_equal_mean=float(group_mean[j-1]))
    results["matched_nonstatic"] = int(sum(row[3] for row in active))
    results["nonstatic"] = len(active)
    results["rank_median"] = float(np.median([row[0] for row in observations]))
    return results


def main():
    p = ArgumentParser(description=__doc__)
    for name in ("fit", "capture", "points", "labels", "prior-fit", "previous", "action-core",
                 "source-core", "attraction", "selector", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    start = perf_counter()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or any(os.environ.get(k) != "1" for k in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise ValueError("Exact original CPU runtime")
    fit_worker = a.fit / "fit_banks.py"
    import hashlib
    if hashlib.sha256(fit_worker.read_bytes()).hexdigest() != WORKER:
        raise ValueError("Exact pinned closed FIT worker")
    m = run_path(str(fit_worker))
    pin, load, need = (m[k] for k in ("pin", "source", "need"))
    fit = json.loads(pin(a.fit / "actual/RESULT_SAFE.json", FIT))
    readback = json.loads(pin(a.fit / "READBACK_SAFE.json", READBACK))
    term = json.loads(pin(a.fit / "observed_fit/TERMINAL.json", TERMINAL))
    need(term["actual_exit"] == 0 and term["output_footer_verified"] is True
         and term["result_sha256"] == readback["fit_result_sha256"] == FIT
         and readback["terminal_sha256"] == TERMINAL and fit["worker_sha256"] == WORKER,
         "Natural closed FIT and independent readback")
    prev = load(a.previous / "fit_action.py", m["ORIGINAL"])
    prior = json.loads(pin(a.previous / "actual/RESULT_SAFE.json", m["ORIGINAL_RESULT"]))
    core = load(a.action_core, prev.ACTION_CORE)
    occupancy = load(a.source_core, prev.OCCUPANCY_CORE)
    attraction = load(a.attraction, prev.ATTRACTION)
    selector = load(a.selector, m["SELECTOR"])
    packets, counts, packet_sha, incumbents = m["original_packets"](a, prev, core, attraction, prior)
    estimates, audits, groups_sha, probabilities_sha = m["crossfit_occupancy"](packets, core, occupancy, a.source_core)
    aligned, bank_audit, bank_sha = m["aligned_packets"](packets, estimates, incumbents, selector)
    need(fit["fit_packet_sha256"] == packet_sha and probabilities_sha == fit["out_of_fold_probabilities_sha256"]
         and groups_sha == fit["out_of_fold_group_sha256"] and bank_sha == fit["aligned_action_packet_sha256"]
         and bank_audit == fit["bank_audit"], "Exact independent deterministic OOF reconstruction")
    observations = defaultdict(list)
    for packet, alternate, posterior, (_, static) in zip(packets, aligned, estimates, incumbents, strict=True):
        plan = selector.select(posterior[:,0,:], posterior[:,1,:], *static, k=packet.k)["robust"]
        response = m["signed_weights"](plan, static, len(posterior))
        observations["previous_bank"].append(span_line(packet.action_weights, response))
        observations["deployment_bank"].append(span_line(alternate.action_weights, response))
    metrics = {arm:summarize(packets, rows) for arm,rows in observations.items()}
    need(metrics["previous_bank"]["nonstatic"] == metrics["deployment_bank"]["nonstatic"] == 847
         and metrics["deployment_bank"]["matched_nonstatic"] == 847,
         "All exact nonstatic deployed menu wins are supported")
    result = dict(status=STATUS, fit_result_sha256=FIT, fit_readback_sha256=READBACK,
                  worker_sha256=m["digest"](Path(__file__).read_bytes()),
                  protocol_sha256=m["digest"]((Path(__file__).parent/"PRELOCK.md").read_bytes()),
                  fit_bank_audit=bank_audit, metrics=metrics,
                  fit_candidates=848, fit_images=447, fit_groups=439, repeated_oof_fit_updates=5000,
                  probe_optimizer_updates=0, source_gate_labels_opened=False, historical_test_opened=False,
                  overlap_or_ap_measured=False, elapsed_seconds=perf_counter()-start)
    checksum = prev.write(a.output, result)
    print(json.dumps(dict(status=STATUS, result_sha256=checksum)),flush=True)


if __name__ == "__main__":
    main()
