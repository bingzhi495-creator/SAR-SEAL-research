"""Verify an actual closed matched spatial mask-branch cue fit and read back the three final states."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from runpy import run_path

import numpy as np
import torch


REGRESSION = "d41d2538ae03ff76b385f7e25ca5288fea9930920678e50ceb0e1863047ef456"
SOURCE_FIT = "4872b63ee24c8021e09612314161c691006607dceb3ea92a2ce27690016836e4"
STATUS = "SPSD_PSEG_MATCHED_SPATIAL_CARRIER_CUE_READBACK_COMPLETE"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def required(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Exact file " + path.name)
    return raw


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--core", required=True, type=Path)
    p.add_argument("--fit-sha", required=True)
    a = p.parse_args()
    terminal_blob = (a.root / "observed_fit/TERMINAL.json").read_bytes()
    terminal = json.loads(terminal_blob)
    start = json.loads((a.root / "observed_fit/START.json").read_bytes())
    worker = digest((a.root / "fit_spatial.py").read_bytes())
    lock = digest((a.root / "PRELOCK.md").read_bytes())
    scope = ("observer_pid", "worker_pid", "worker_sha256", "started_utc", "phase")
    need(all(terminal[key] == start[key] for key in scope) and terminal["phase"] == "prepare-edits"
         and terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
         and terminal["result_sha256"] == a.fit_sha and terminal["worker_sha256"] == worker,
         "Observed natural child closure and exact result footer")
    actual = a.root / "actual"

    r = json.loads(required(actual / "RESULT_SAFE.json", a.fit_sha))
    required(a.root / "spatial_cues.py", r["feature_worker_sha256"])
    required(a.root / "cue_reader.py", r["cue_reader_sha256"])
    required(a.root / "ACQUISITION_LOCK_SAFE.json", r["cue_acquisition_lock_sha256"])
    pre = json.loads(required(actual / "INPUT_SAFE.json", r["input_receipt_sha256"]))
    need(r["status"] == "SPSD_PSEG_MATCHED_SPATIAL_CARRIER_CUE_FIT_COMPLETE"
         and pre["status"] == "FIT_MATCHED_SPATIAL_CARRIER_CUE_PACKETS_ADMITTED"
         and r["worker_sha256"] == worker and r["protocol_sha256"] == pre["protocol_sha256"] == lock
         and r["source_fit_result_sha256"] == pre["source_fit_result_sha256"] == SOURCE_FIT
         and all(r[key] == pre[key] for key in ("counts", "aligned_action_packet_sha256", "out_of_fold_probabilities_sha256"))
         and r["head_pair_baseline_tensor_parity"] is True and pre["fit_only"] is True
         and all(doc[field] is False for doc in (r, pre)
                 for field in ("source_gate_labels_opened", "source_gate_points_opened", "historical_test_opened")),
         "FIT-only source and bank parity")
    need(r["detector_forwards"] == 0 and r["total_optimizer_updates"] == 3000 and r["out_of_fold_actual_updates"] == 5000
         and r["smooth_l1_beta"] == .1 and r["all_raw_14_scalers_frozen"] is True and r["capacity_aware_replacement_width"] == 6 and r["auxiliary_weight"] == 1,
         "Fixed objective scales and accounting separation for shared proposal preparation")
    a0 = r["train_audit"]
    heads = a0["heads"]
    expected = dict(parameters=1346, optimizer_updates=1000, observed_candidates=64000, observed_endpoints=4079674,
                    observed_actions=380295, initial_tensor_sha256=a0["initial_tensor_sha256"], batch_sha256=a0["batch_sha256"])
    need(a0["fit_candidates"] == 848 and a0["fit_actions"] == 5042 and a0["fit_groups"] == 439
         and set(heads) == {"regression", "spatial", "scrambled"}
         and all(all(head[key] == value for key, value in expected.items()) for head in heads.values()),
         "Matched actual heads, capacity, initialization, sample exposures and schedule")
    fields = ("cue_capture_result_sha256", "cue_capture_terminal_sha256", "cue_capture_readback_sha256",
              "cue_fit_index_sha256", "cue_acquisition_lock_sha256", "cue_reader_sha256",
              "cue_joined_candidates", "cue_joined_images", "cue_fit_vector_payload_sha256")
    need(all(r[k] == pre[k] for k in fields) and r["cue_joined_candidates"] == 848 and r["cue_joined_images"] == 447
         and r["cue_shared_scaler"] is True, "FIT-only independently qualified capture and shared cue scaling")
    info = r["feature_audit"]
    need(0 < info["different_spatial_vs_scrambled_candidates"] <= 848
         and info["spatial"]["fit_cue_tensor_sha256"] != info["scrambled"]["fit_cue_tensor_sha256"]
         and len(info["shared_cue_center"]) == len(info["shared_cue_scale"]) == 6,
         "Different true-coordinate and scrambled cues of equal width")
    ckpt = torch.load(BytesIO(required(actual / "FITTED_WEIGHTS_PRIVATE.pt", r["output_checkpoint_sha256"])),
                      weights_only=True, map_location="cpu")
    need(set(ckpt) == {"models", "center", "scale", "seed"} and ckpt["seed"] == 5297
         and set(ckpt["models"]) == set(ckpt["center"]) == set(ckpt["scale"]) == set(heads),
         "Exact private checkpoint fields and head names")
    for name, width in (("regression",14), ("spatial",28), ("scrambled",28)):
        c, s = ckpt["center"][name], ckpt["scale"][name]
        need(tuple(c.shape) == tuple(s.shape) == (width,) and bool(torch.isfinite(c).all() & torch.isfinite(s).all()
                & (s > 0).all()), "Finite branch-specific scaling")
        if name != "regression":
            for tensor in ("center", "scale"):
                old = ckpt[tensor]["regression"]
                new = ckpt[tensor][name]
                need(torch.equal(new[:22], torch.cat((old, old[:8]))) and new[22:].tolist() == info["shared_cue_"+tensor],
                     "Frozen old feature scaling and actual saved cue scaling")
    need(all(torch.equal(ckpt[k]["spatial"], ckpt[k]["scrambled"]) for k in ("center","scale")),
         "Spatial and scrambled branch scalers must be exactly identical")
    required(a.core, r["training_core_sha256"])
    model_class = run_path(str(a.core))["PointHead"]
    final = {}
    for name, state in ckpt["models"].items():
        model = model_class()
        model.load_state_dict(state, strict=True)
        values = [p.detach().numpy().astype("<f4", copy=False) for p in model.parameters()]
        final[name] = digest(b"".join(x.tobytes() for x in values))
        need(all(np.isfinite(x).all() for x in values) and final[name] == heads[name]["final_tensor_sha256"],
             "Independent final tensor readback for " + name)
    need(final["regression"] == r["regression_baseline_tensor_sha256"] == REGRESSION and len(set(final.values())) == 3,
         "Exact previous current baseline and distinct fitted intervention")
    result = dict(status=STATUS, completed=True, natural_exit=0,
                  audit_sha256=digest(Path(__file__).read_bytes()), terminal_sha256=digest(terminal_blob),
                  worker_sha256=worker, protocol_sha256=lock, fit_result_sha256=a.fit_sha,
                  input_sha256=r["input_receipt_sha256"], checkpoint_sha256=r["output_checkpoint_sha256"], cue_capture_result_sha256=r["cue_capture_result_sha256"],
                  cue_capture_readback_sha256=r["cue_capture_readback_sha256"],
                  final_tensor_sha256=final)
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with (a.root / "READBACK_SAFE.json").open("xb") as out:
        out.write(data)
    print(json.dumps(dict(status=STATUS, result_sha256=digest(data))), flush=True)


if __name__ == "__main__":
    main()
