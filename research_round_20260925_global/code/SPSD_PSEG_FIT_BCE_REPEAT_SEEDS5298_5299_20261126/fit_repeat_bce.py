"""Repeat the original candidate-balanced BCE head on unchanged FIT packets."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
import torch


LOCK = "a0380981b57cb8c97ee740b072d323416c2e3ecd536f0ef10232104d088f6af1"
PARENT = "011914954482cd505343a6421296ae754b1ea3098f5ad3fe37cdaa14e3c05fd3"
PARENT_RESULT = "ae1b165a111a6b88939eb37b5741421259f0f39531c859a7c7663ad9116d1b05"
PARENT_TERMINAL = "b0d837ca9fdeb4b21bb6c4bb83ecda130b86696d751b4cad309a41379887991b"
ARCHIVED_BCE = "a2c169f895b1435e689480836ebf4f04fd6e05d84652f51d74e0448ed099faff"
CURRENT_BCE = "8497560cdd204309809eddbcf09fe520c8ccf5fedb94ebf861286c9c91047597"
REPEATS = (
    ("parity_probe","e8b62b3c6b207f2fc111f1820b980754813176e8c8f93d51abacafff39eb7481",
     "67ed73f3a236780b2dda91482512686c640f14313d52698afbf45f75d1c46dda"),
    ("parity_probe_repeat","3aa7d210e3cb7f1f24ff002808a89301ac50b045abd43987aca5872e08a293b0",
     "a14603fdddfdf687f3f20e82c0deba3fdd59fb504512b6c5da9ce82edb3f0137"),
)
STATUS = "SPSD_PSEG_BCE_REPEAT_SEEDS_FIT_COMPLETE"
INPUT_STATUS = "FIT_BCE_REPEAT_SEEDS_PACKETS_ADMITTED"
SEEDS = (5298,5299)


def need(flag, reason):
    if not flag:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw)==expected,"Exact artifact "+path.name)
    return raw


def load(path, expected):
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(pin(path,expected),str(path),"exec"),mod.__dict__)
    return mod


def closure(parent, result_path, terminal_path, result, terminal, expected_tensor):
    """Bind actual archived model receipt before using any source data."""
    doc = json.loads(pin(parent/result_path,result))
    end = json.loads(pin(parent/terminal_path,terminal))
    head = doc["train_audit"]["heads"]["bce"]
    need(end["phase"]=="prepare-edits" and end["actual_exit"]==0 and end["output_footer_verified"] is True
         and end["result_sha256"]==result and end["worker_sha256"]==doc["worker_sha256"]==PARENT
         and head["final_tensor_sha256"]==expected_tensor and head["optimizer_updates"]==1000
         and head["parameters"]==1346 and doc["source_gate_labels_opened"] is doc["historical_test_opened"] is False,
         "Exact prior natural pure BCE fit lineage")
    return doc


def fit_seed(core, packets, source_core, seed):
    """Keep trainer source immutable and supply only the original BCE mode and seed."""
    old_seed, old_modes = core.SEED, core.MODES
    core.SEED,core.MODES = seed,("bce",)
    try:
        return core.train_matched_action(packets,source_core=source_core,seed=seed)
    finally:
        core.SEED,core.MODES = old_seed,old_modes


def head_audit(audit, seed, expected):
    need(set(audit["heads"])=={"bce"} and audit["seed"]==seed and audit["batch_size"]==64
         and all(audit[k]==expected[k] for k in ("fit_actions","fit_candidates","fit_endpoints","fit_groups","fit_images","update_count")),
         "Only native pure BCE on the original training roster")
    head = audit["heads"]["bce"]
    need(head["last_total_loss"]==head["last_bce_loss"] and head["optimizer_updates"]==1000
         and head["parameters"]==1346 and head["observed_candidates"]==64000
         and head["initial_tensor_sha256"]==audit["initial_tensor_sha256"]
         and head["batch_sha256"]==audit["batch_sha256"],
         "Observed pure BCE, exact capacity, unique training duration and draws")
    return dict(head,seed=seed)


def main():
    p=ArgumentParser(description=__doc__,allow_abbrev=False)
    for flag in ("protocol","frozen-parent","probe-root","capture","points","labels","prior-fit",
                 "action-core","source-core","attraction","out"):
        p.add_argument("--"+flag,type=Path,required=True)
    a=p.parse_args()
    start=perf_counter()
    need(not a.out.exists() and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and torch.get_num_threads()==1
         and all(os.environ.get(key)=="1" for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")),
         "Original repeatable CPU runtime and fresh fit output")
    pin(a.protocol,LOCK)
    parent=closure(a.frozen_parent,"actual/RESULT_SAFE.json","observed_fit/TERMINAL.json",
                   PARENT_RESULT,PARENT_TERMINAL,ARCHIVED_BCE)
    proofs=[closure(a.probe_root/name,"actual/RESULT_SAFE.json","observed/TERMINAL.json",result,end,CURRENT_BCE)
            for name,result,end in REPEATS]
    old=load(a.frozen_parent/"fit_action.py",PARENT)
    prior=load(a.prior_fit,old.PRIOR_FIT)
    core=load(a.action_core,old.ACTION_CORE)
    attraction=load(a.attraction,old.ATTRACTION)
    pin(a.source_core,old.OCCUPANCY_CORE)
    packets,counts,packet_sha=old.fit_packets(a,prior,core,attraction)
    need(counts==parent["counts"] and packet_sha==parent["fit_packet_sha256"] and all(
         d["fit_packet_sha256"]==packet_sha and d["counts"]==counts for d in proofs),
         "Exact frozen original packets, labels, endpoints, order and action audit")
    a.out.mkdir()
    common=dict(source_fit_result_sha256=PARENT_RESULT,source_fit_terminal_sha256=PARENT_TERMINAL,
                archive_5297_bce_tensor_sha256=ARCHIVED_BCE,current_5297_bce_tensor_sha256=CURRENT_BCE,
                current_5297_probe_result_sha256s=[r[1] for r in REPEATS],fit_packet_sha256=packet_sha,
                counts=counts,seeds=list(SEEDS),source_gate_labels_opened=False,source_gate_points_opened=False,
                historical_test_opened=False)
    input_sha=old.write(a.out/"INPUT_SAFE.json",dict(status=INPUT_STATUS,fit_only=True,protocol_sha256=LOCK,**common))
    print(json.dumps(dict(status=INPUT_STATUS,input_receipt_sha256=input_sha)),flush=True)
    _,center,scale,check=fit_seed(core,packets,a.source_core,5297)
    control=head_audit(check,5297,parent["train_audit"])
    need(all(all(control[k]==d["train_audit"]["heads"]["bce"][k] for k in d["train_audit"]["heads"]["bce"]
                 if k!="elapsed_seconds") for d in proofs), "Exact original current 5297 anchor on pure BCE")
    heads,states={},{}
    for seed in SEEDS:
        fitted,c,s,run=fit_seed(core,packets,a.source_core,seed)
        need(np.array_equal(center,c) and np.array_equal(scale,s),"All seeds use the same exact FIT normalization")
        heads[str(seed)]=head_audit(run,seed,parent["train_audit"])
        states[str(seed)]=fitted["bce"].state_dict()
    for k in ("initial_tensor_sha256","batch_sha256","final_tensor_sha256"):
        need(len({h[k] for h in (control,*heads.values())})==3,
             "All three seeds produce independent sample draws, initial and final states")
    checkpoint=a.out/"FITTED_WEIGHTS_PRIVATE.pt"
    with checkpoint.open("xb") as stream:
        torch.save(dict(models=states,center=torch.from_numpy(center),scale=torch.from_numpy(scale),seeds=list(SEEDS)),stream)
        stream.flush()
        os.fsync(stream.fileno())
    result=dict(status=STATUS,protocol_sha256=LOCK,worker_sha256=digest(Path(__file__).read_bytes()),
                action_core_sha256=old.ACTION_CORE,training_core_sha256=old.OCCUPANCY_CORE,
                input_receipt_sha256=input_sha,output_checkpoint_sha256=digest(checkpoint.read_bytes()),
                archive_5297_used_as_historical_anchor=True,current_5297_parity=True,
                current_5297_audit=control,train_audit=heads,total_optimizer_updates=2000,
                verification_optimizer_updates=1000,detector_forwards=0,elapsed_seconds=perf_counter()-start,**common)
    checksum=old.write(a.out/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=STATUS,result_sha256=checksum)),flush=True)


if __name__=="__main__":
    main()
