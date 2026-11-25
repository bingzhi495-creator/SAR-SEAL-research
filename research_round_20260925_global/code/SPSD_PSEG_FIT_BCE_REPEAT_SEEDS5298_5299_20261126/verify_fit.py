"""Independent final state readback after natural closure of the two BCE repeat seeds."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from runpy import run_path

import numpy as np
import torch


STATUS = "SPSD_PSEG_BCE_REPEAT_SEEDS_READBACK_COMPLETE"
SEEDS = (5298,5299)
CURRENT = "8497560cdd204309809eddbcf09fe520c8ccf5fedb94ebf861286c9c91047597"
ARCHIVE = "a2c169f895b1435e689480836ebf4f04fd6e05d84652f51d74e0448ed099faff"


def need(check, reason):
    if not check:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    raw=path.read_bytes()
    need(digest(raw)==expected,"Exact independent readback "+path.name)
    return raw


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--source-core",type=Path,required=True)
    p.add_argument("--fit-sha",required=True)
    a=p.parse_args()
    t=(a.root/"observed_fit/TERMINAL.json").read_bytes()
    terminal=json.loads(t)
    start=json.loads((a.root/"observed_fit/START.json").read_bytes())
    worker=digest((a.root/"fit_repeat_bce.py").read_bytes())
    lock=digest((a.root/"PRELOCK.md").read_bytes())
    need(all(terminal[k]==start[k] for k in ("phase","worker_sha256","worker_pid","observer_pid","started_utc"))
         and terminal["phase"]=="prepare-edits" and terminal["actual_exit"]==0 and terminal["output_footer_verified"] is True
         and terminal["worker_sha256"]==worker and terminal["result_sha256"]==a.fit_sha,"Exact actual child from start to closure")
    actual=a.root/"actual"
    result=json.loads(pin(actual/"RESULT_SAFE.json",a.fit_sha))
    pre=json.loads(pin(actual/"INPUT_SAFE.json",result["input_receipt_sha256"]))
    keys=("source_fit_result_sha256","source_fit_terminal_sha256","archive_5297_bce_tensor_sha256",
          "current_5297_bce_tensor_sha256","current_5297_probe_result_sha256s","fit_packet_sha256","counts","seeds",
          "source_gate_labels_opened","source_gate_points_opened","historical_test_opened")
    need(pre["status"]=="FIT_BCE_REPEAT_SEEDS_PACKETS_ADMITTED" and pre["fit_only"] is True
         and result["status"]=="SPSD_PSEG_BCE_REPEAT_SEEDS_FIT_COMPLETE"
         and result["protocol_sha256"]==pre["protocol_sha256"]==lock and result["worker_sha256"]==worker
         and all(result[k]==pre[k] for k in keys) and result["archive_5297_used_as_historical_anchor"] is True
         and result["archive_5297_bce_tensor_sha256"]==ARCHIVE and result["current_5297_bce_tensor_sha256"]==CURRENT
         and result["current_5297_parity"] is True and result["current_5297_audit"]["final_tensor_sha256"]==CURRENT,
         "Both historical and current seed controls track exact frozen sources")
    nums=result["counts"]
    need(nums["eligible_candidate_packets"]==848 and nums["fit_groups_with_supervision"]==439
         and nums["fit_images_with_supervision"]==447 and nums["actions"]==5042 and nums["native_endpoint_pairs"]==26983
         and result["total_optimizer_updates"]==2000 and result["verification_optimizer_updates"]==1000
         and result["detector_forwards"]==0 and tuple(result["seeds"])==SEEDS
         and not any(result[k] for k in ("source_gate_labels_opened","source_gate_points_opened","historical_test_opened")),
         "Exact FIT-only roster and update accounting")
    audit=result["train_audit"]
    expected=dict(zip(map(str,SEEDS),SEEDS,strict=True))
    need(audit.keys()==expected.keys() and all(
         audit[key]["seed"]==seed and audit[key]["parameters"]==1346 and audit[key]["optimizer_updates"]==1000
         and audit[key]["observed_candidates"]==64000 and audit[key]["last_total_loss"]==audit[key]["last_bce_loss"]
         for key,seed in expected.items()),"Final pure BCE and actual observed optimizer budgets")
    data=torch.load(BytesIO(pin(actual/"FITTED_WEIGHTS_PRIVATE.pt",result["output_checkpoint_sha256"])),
                    map_location="cpu",weights_only=True)
    need(set(data)=={"models","center","scale","seeds"} and tuple(data["seeds"])==SEEDS and data["models"].keys()==audit.keys()
         and tuple(data["center"].shape)==tuple(data["scale"].shape)==(14,)
         and bool(torch.isfinite(data["center"]).all()&torch.isfinite(data["scale"]).all()&(data["scale"]>0).all()),
         "Finite exact scaled seed checkpoint")
    pin(a.source_core,result["training_core_sha256"])
    model=run_path(str(a.source_core))["PointHead"]
    endpoint={}
    for seed,state in data["models"].items():
        net=model()
        net.load_state_dict(state,strict=True)
        tensors=[v.detach().numpy().astype("<f4",copy=False) for v in net.parameters()]
        endpoint[seed]=digest(b"".join(v.tobytes() for v in tensors))
        need(all(np.isfinite(v).all() for v in tensors) and endpoint[seed]==audit[seed]["final_tensor_sha256"],
             "Direct seed tensor data readback")
    for k in ("initial_tensor_sha256","final_tensor_sha256","batch_sha256"):
        need(len({d[k] for d in (result["current_5297_audit"],*audit.values())})==3,
             "Multiple seeded draws and distinct original initialized/fitted states")
    out=dict(status=STATUS,completed=True,natural_exit=0,fit_result_sha256=a.fit_sha,terminal_sha256=digest(t),
             input_sha256=result["input_receipt_sha256"],checkpoint_sha256=result["output_checkpoint_sha256"],
             worker_sha256=worker,protocol_sha256=lock,final_tensor_sha256=endpoint,seeds=list(SEEDS),
             audit_sha256=digest(Path(__file__).read_bytes()))
    raw=(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with (a.root/"READBACK_SAFE.json").open("xb") as dest:
        dest.write(raw)
    print(json.dumps(dict(status=STATUS,result_sha256=digest(raw))),flush=True)


if __name__=="__main__":
    main()
