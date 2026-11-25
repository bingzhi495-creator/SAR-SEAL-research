"""Actual FIT-only ridge for selective adoption of the frozen BCE robust writer."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
import torch


PROTOCOL = "ba8ebebd5b6f0fa92c13dfee6f2da971296837f81e7f329eca9190e9d17fc389"
FEATURE = "81b59ddaaa902f9ade813cf82dc68f84c34b43b455a5515c847c5db66afd8e1f"
ADMISSION_WORKER = "b29f2a654831e4cd512d88b87e7dfe016dc711f897173f5ce20aa50cc9ea0d80"
FROZEN_BCE = "a2c169f895b1435e689480836ebf4f04fd6e05d84652f51d74e0448ed099faff"
INPUT_STATUS = "FIT_SELECTIVE_WRITER_OOF_PACKETS_ADMITTED"
STATUS = "SPSD_PSEG_SELECTIVE_WRITER_FIT_COMPLETE"


def need(flag, reason):
    if not flag:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Exact source digest " + path.name)
    return raw


def load(path, expected):
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(pin(path,expected),str(path),"exec"),mod.__dict__)
    return mod


def admitted(a):
    prior = load(a.admission_worker,ADMISSION_WORKER)
    bank = load(a.deployment_fit/"fit_banks.py",prior.BANK_WORKER)
    prior_done = json.loads(pin(a.deployment_fit/"observed_fit/TERMINAL.json",prior.BANK_TERMINAL))
    prior_readback = json.loads(pin(a.deployment_fit/"READBACK_SAFE.json",prior.BANK_READBACK))
    prior_stats = json.loads(pin(a.deployment_fit/"actual/RESULT_SAFE.json",prior.BANK_RESULT))
    need(prior_done["actual_exit"] == 0 and prior_done["output_footer_verified"] is True
         and prior_done["result_sha256"] == prior.BANK_RESULT and prior_done["worker_sha256"] == prior.BANK_WORKER
         and prior_readback["terminal_sha256"] == prior.BANK_TERMINAL and prior_readback["fit_result_sha256"] == prior.BANK_RESULT
         and prior_stats["paired_previous_fit_tensor_parity"] is True and prior_stats["source_gate_labels_opened"] is False
         and prior_stats["historical_test_opened"] is False, "Prior independently closed OOF authority")
    source = load(a.frozen_parent/"fit_action.py",bank.ORIGINAL)
    source_done = json.loads(pin(a.frozen_parent/"observed_fit/TERMINAL.json",bank.ORIGINAL_TERMINAL))
    source_stats = json.loads(pin(a.frozen_parent/"actual/RESULT_SAFE.json",bank.ORIGINAL_RESULT))
    need(source_done["phase"] == "prepare-edits" and source_done["actual_exit"] == 0
         and source_done["output_footer_verified"] is True and source_done["result_sha256"] == bank.ORIGINAL_RESULT
         and source_done["worker_sha256"] == bank.ORIGINAL
         and source_stats["train_audit"]["heads"]["bce"]["final_tensor_sha256"] == FROZEN_BCE
         and source_stats["train_audit"]["heads"]["bce"]["optimizer_updates"] == 1000,
         "Exact production candidate-balanced BCE authority")
    action = load(a.action_core,source.ACTION_CORE)
    occupancy = load(a.source_core,source.OCCUPANCY_CORE)
    attraction = load(a.attraction,source.ATTRACTION)
    chooser = load(a.selector,bank.SELECTOR)
    packets, counts, raw_sha, incumbent = bank.original_packets(a,source,action,attraction,source_stats)
    estimates, folds, fold_sha, estimate_sha = bank.crossfit_occupancy(packets,action,occupancy,a.source_core)
    need(counts == source_stats["counts"] == prior_stats["counts"]
         and raw_sha == source_stats["fit_packet_sha256"] == prior_stats["fit_packet_sha256"]
         and fold_sha == prior_stats["out_of_fold_group_sha256"] and estimate_sha == prior_stats["out_of_fold_probabilities_sha256"]
         and sum(v["held_candidates"] for v in folds) == 848
         and all(v["overlap_groups"] == v["overlap_images"] == 0 for v in folds),
         "Exact OOF ordered probabilities and roster without aligned-action training")
    return source, chooser, (packets,counts,estimates,incumbent), dict(
        frozen_parent_fit_sha256=bank.ORIGINAL_RESULT, frozen_parent_bce_tensor_sha256=FROZEN_BCE,
        oof_authority_fit_sha256=prior.BANK_RESULT, out_of_fold_probability_sha256=estimate_sha,
        out_of_fold_group_sha256=fold_sha, fit_packet_sha256=raw_sha,
        finite_selector_sha256=bank.SELECTOR, feature_worker_sha256=FEATURE,
        source_gate_labels_opened=False, source_gate_points_opened=False, historical_test_opened=False,
        shared_oof_updates=sum(v["train_head"]["optimizer_updates"] for v in folds))


def rows_only(packets, estimates, inc, chooser, helper):
    table, y, group, image = [], [], [], []
    count = Counter()
    audited = sha256()
    for item, p, (key,static) in zip(packets,estimates,inc,strict=True):
        q = item.k
        choices = chooser.select(p[:,0,:],p[:,1,:],*static,k=q)
        robust = choices["robust"]
        mean = choices["mean"]
        need(choices["static"] == static, "Exact frozen static-at-dynamic k candidate")
        count.update(candidates=1,robust_nonstatic=int(robust!=static),mean_nonstatic=int(mean!=static),
                     mean_differs_from_robust=int(mean!=robust))
        if robust == static:
            continue
        row, margin = helper.features(item.features,p,robust,static,q)
        need(margin >= -1e-10, "Admitted robust choice predicts nonnegative minimum gain")
        realized = helper.true_numerators(item.targets,robust,static,q)
        target = int(realized.min()) / (2*q)
        table.append(row)
        y.append(target)
        group.append(item.group_id)
        image.append(item.image_id)
        count.update(target_negative=int(target<0),target_zero=int(target==0),target_positive=int(target>0),
                     active_exchanges=q)
        audited.update((key+"\n").encode("ascii"))
        for vector in (np.asarray(robust,dtype="<i4"),np.asarray(mean,dtype="<i4"),row,
                       np.asarray([margin,target],dtype="<f8")):
            audited.update(vector.tobytes(order="C"))
    active = count["robust_nonstatic"]
    need(count["candidates"] == len(packets) == 848 and active == len(table)>0
         and count["target_negative"]+count["target_zero"]+count["target_positive"] == active,
         "Every FIT candidate explicitly considered and only nonstatic robust choices fitted")
    count.update(active_groups=len(set(group)), active_images=len(set(image)))
    return np.stack(table),np.asarray(y),group,image,dict(count),audited.hexdigest()


def main():
    p = ArgumentParser(description=__doc__,allow_abbrev=False)
    for field in ("protocol","admission-worker","feature-source","deployment-fit","capture","points","labels",
                  "prior-fit","frozen-parent","action-core","source-core","attraction","selector","out"):
        p.add_argument("--"+field,type=Path,required=True)
    a = p.parse_args()
    a.previous = a.frozen_parent
    start = perf_counter()
    need(not a.out.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and torch.get_num_threads() == 1
         and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")),
         "Known reproducible CPU runtime and new output")
    pin(a.protocol,PROTOCOL)
    helper = load(a.feature_source,FEATURE)
    source, chooser, (packets,original_counts,estimate,inc), receipt = admitted(a)
    x,y,groups,images,counts,rows_sha = rows_only(packets,estimate,inc,chooser,helper)
    a.out.mkdir()
    input_sha = source.write(a.out/"INPUT_SAFE.json",dict(status=INPUT_STATUS,fit_only=True,
                  protocol_sha256=PROTOCOL,original_counts=original_counts,counts=counts,
                  selective_fit_rows_sha256=rows_sha,**receipt))
    print(json.dumps(dict(status=INPUT_STATUS,input_receipt_sha256=input_sha)),flush=True)
    state = helper.fit(x,y,groups,images)
    checked = helper.score(state,x)
    need(checked.shape == (len(x),) and np.isfinite(checked).all(), "Private selector coefficient round-trip")
    ridge_sha = source.write(a.out/"RIDGE_PRIVATE.json",state)
    result = dict(status=STATUS, protocol_sha256=PROTOCOL, worker_sha256=digest(Path(__file__).read_bytes()),
                  admission_worker_sha256=ADMISSION_WORKER, input_receipt_sha256=input_sha,
                  original_counts=original_counts, counts=counts, selective_fit_rows_sha256=rows_sha,
                  ridge_checkpoint_sha256=ridge_sha, fit_feature_count=9, ridge_alpha=.02,
                  fitted_selective_models=1, closed_linear_solves=1, optimizer_updates=0,
                  detector_forwards=0, fit_labels_opened=True, shallow_selector_cv=False,
                  elapsed_seconds=perf_counter()-start, **receipt)
    checksum = source.write(a.out/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=STATUS,result_sha256=checksum)),flush=True)


if __name__ == "__main__":
    main()
