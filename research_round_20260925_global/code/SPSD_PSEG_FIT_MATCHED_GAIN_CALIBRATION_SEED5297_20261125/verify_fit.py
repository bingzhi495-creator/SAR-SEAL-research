"""Verify an actual closed matched component calibration fit and read back both final states."""
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
STATUS = "SPSD_PSEG_MATCHED_GAIN_CALIBRATION_READBACK_COMPLETE"


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
    worker = digest((a.root / "fit_calibration.py").read_bytes())
    lock = digest((a.root / "PRELOCK.md").read_bytes())
    scope = ("observer_pid", "worker_pid", "worker_sha256", "started_utc", "phase")
    need(all(terminal[key] == start[key] for key in scope) and terminal["phase"] == "prepare-edits"
         and terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
         and terminal["result_sha256"] == a.fit_sha and terminal["worker_sha256"] == worker,
         "Observed natural child closure and exact result footer")
    actual = a.root / "actual"
    r = json.loads(required(actual / "RESULT_SAFE.json", a.fit_sha))
    pre = json.loads(required(actual / "INPUT_SAFE.json", r["input_receipt_sha256"]))
    need(r["status"] == "SPSD_PSEG_MATCHED_GAIN_CALIBRATION_FIT_COMPLETE"
         and pre["status"] == "FIT_MATCHED_GAIN_CALIBRATION_PACKETS_ADMITTED"
         and r["worker_sha256"] == worker and r["protocol_sha256"] == pre["protocol_sha256"] == lock
         and r["source_fit_result_sha256"] == pre["source_fit_result_sha256"] == SOURCE_FIT
         and all(r[key] == pre[key] for key in ("counts", "aligned_action_packet_sha256", "out_of_fold_probabilities_sha256"))
         and r["head_pair_baseline_tensor_parity"] is True and pre["fit_only"] is True
         and all(doc[field] is False for doc in (r, pre)
                 for field in ("source_gate_labels_opened", "source_gate_points_opened", "historical_test_opened")),
         "FIT-only source and bank parity")
    need(r["detector_forwards"] == 0 and r["total_optimizer_updates"] == 3000 and r["out_of_fold_actual_updates"] == 5000
         and r["smooth_l1_beta"] == .1 and r["zero_static_components"] is True and r["auxiliary_weight"] == 1,
         "Fixed objective scales and accounting separation for shared proposal preparation")
    a0 = r["train_audit"]
    heads = a0["heads"]
    expected = dict(parameters=1346, optimizer_updates=1000, observed_candidates=64000, observed_endpoints=4079674,
                    observed_actions=380295, initial_tensor_sha256=a0["initial_tensor_sha256"], batch_sha256=a0["batch_sha256"])
    need(a0["fit_candidates"] == 848 and a0["fit_actions"] == 5042 and a0["fit_groups"] == 439
         and set(heads) == {"regression", "midpoint", "gap"}
         and all(all(head[key] == value for key, value in expected.items()) for head in heads.values()),
         "Matched actual heads, capacity, initialization, sample exposures and schedule")
    for info, expected_actions in ((r["target_geometry"], 5042), (r["exposure_geometry"], 380295)):
        need(info["real_actions"] == expected_actions and 0 <= info["exact_zero_true_midpoint"] <= expected_actions
             and info["exact_zero_true_gap"] + info["positive_true_gap"] == expected_actions,
             "Exact deployment-bank component target accounting")
    ckpt = torch.load(BytesIO(required(actual / "FITTED_WEIGHTS_PRIVATE.pt", r["output_checkpoint_sha256"])),
                      weights_only=True, map_location="cpu")
    need(set(ckpt) == {"models", "center", "scale", "seed"} and ckpt["seed"] == 5297
         and set(ckpt["models"]) == set(heads) and ckpt["center"].shape == ckpt["scale"].shape == (14,)
         and bool(torch.isfinite(ckpt["center"]).all() & torch.isfinite(ckpt["scale"]).all() & (ckpt["scale"] > 0).all()),
         "Exact private checkpoint fields and scaling")
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
                  input_sha256=r["input_receipt_sha256"], checkpoint_sha256=r["output_checkpoint_sha256"],
                  final_tensor_sha256=final)
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with (a.root / "READBACK_SAFE.json").open("xb") as out:
        out.write(data)
    print(json.dumps(dict(status=STATUS, result_sha256=digest(data))), flush=True)


if __name__ == "__main__":
    main()
