"""Independently read back closed fit receipts and raw fitted tensors."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from runpy import run_path

import numpy as np
import torch


def check(test, reason):
    if not test:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def raw_sha(path, expected):
    data = path.read_bytes()
    check(digest(data) == expected, f"Integrity mismatch for {path.name}")
    return data


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--core", required=True, type=Path)
    p.add_argument("--fit-sha", required=True)
    args = p.parse_args()
    root = args.root
    terminal_path = root / "observed_fit/TERMINAL.json"
    start_path = root / "observed_fit/START.json"
    terminal_bytes = terminal_path.read_bytes()
    terminal = json.loads(terminal_bytes)
    start = json.loads(start_path.read_bytes())
    worker = digest((root / "fit_banks.py").read_bytes())
    lock = digest((root / "PRELOCK.md").read_bytes())
    for field in ("phase", "observer_pid", "worker_pid", "worker_sha256", "started_utc"):
        check(terminal[field] == start[field], "Actual child start/terminal identity")
    check(terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
          and terminal["phase"] == "prepare-edits" and terminal["result_sha256"] == args.fit_sha
          and terminal["worker_sha256"] == worker, "Unambiguous closed subprocess and final footer")
    actual = root / "actual"
    result = json.loads(raw_sha(actual / "RESULT_SAFE.json", args.fit_sha))
    fit_input = json.loads(raw_sha(actual / "INPUT_SAFE.json", result["input_receipt_sha256"]))
    check(result["status"] == "SPSD_PSEG_MATCHED_DEPLOYMENT_ACTION_BANK_FIT_COMPLETE"
          and fit_input["status"] == "FIT_MATCHED_ACTION_BANK_PACKETS_ADMITTED"
          and result["protocol_sha256"] == fit_input["protocol_sha256"] == lock
          and result["worker_sha256"] == worker and fit_input["fit_only"]
          and fit_input["counts"] == result["counts"] and fit_input["packet_tensor_sha256"] == result["fit_packet_sha256"]
          and result["detector_forwards"] == 0 and result["paired_previous_fit_tensor_parity"]
          and result["archived_previous_fit_tensor_parity"] is False
          and result["archived_previous_fit_tensor_sha256"] == "5d50b31dc9f7733bd2aa4a5c8b078a97d4def679afb57e81a32426fbf9293822"
          and result["runtime"]["torch_threads"] == result["runtime"]["omp"] == result["runtime"]["mkl"] == result["runtime"]["openblas"] == 1
          and all(doc[key] is False for doc in (result, fit_input)
                  for key in ("source_gate_labels_opened", "source_gate_points_opened", "historical_test_opened")),
          "FIT-only experiment and unchanged checkable raw baseline")
    audited = result["train_audit"]
    audit = audited["heads"]
    arms = ("previous_bank", "deployment_bank")
    check(audited["seed"] == 5297 and audited["batch_size"] == 64
          and audited["smooth_l1_beta"] == .1 and audited["signed_action_weight"] == 1.0
          and audited["fit_candidates"] == 848 and audited["fit_actions"] == 5042
          and audited["fit_groups"] == 439 and audited["fit_endpoints"] == 53966
          and audit.keys() == set(arms)
          and sum(v["optimizer_updates"] for v in audit.values()) == result["total_optimizer_updates"] == 2000
          and all(v["optimizer_updates"] == 1000 and v["parameters"] == 1346
                  and v["observed_candidates"] == 64000 and v["observed_endpoints"] == 4079674
                  and v["observed_actions"] == 380295 and v["batch_sha256"] == audited["batch_sha256"]
                  and v["initial_tensor_sha256"] == audited["initial_tensor_sha256"] for v in audit.values()),
          "Actual matched final capacity, endpoint, proposal and update exposure")
    folds = result["out_of_fold_audit"]
    check(len(folds) == 5 and [v["fold"] for v in folds] == list(range(5))
          and sum(v["train_head"]["optimizer_updates"] for v in folds) == result["oof_optimizer_updates"] == 5000
          and sum(v["held_candidates"] for v in folds) == 848
          and sum(v["held_images"] for v in folds) == 447 and sum(v["held_groups"] for v in folds) == 439
          and all(v["overlap_groups"] == v["overlap_images"] == 0
                  and v["train_groups"] + v["held_groups"] == 439
                  and v["train_candidates"] + v["held_candidates"] == 848 for v in folds),
          "Disjoint complement fits with complete out-of-fold prediction coverage")
    ckpt = torch.load(BytesIO(raw_sha(actual / "FITTED_WEIGHTS_PRIVATE.pt", result["output_checkpoint_sha256"])),
                      map_location="cpu", weights_only=True)
    check(set(ckpt) == {"models", "center", "scale", "seed"} and ckpt["seed"] == 5297
          and set(ckpt["models"]) == set(arms) and tuple(ckpt["center"].shape) == tuple(ckpt["scale"].shape) == (14,)
          and bool(torch.isfinite(ckpt["center"]).all() & torch.isfinite(ckpt["scale"]).all()
                   & (ckpt["scale"] > 0).all()),
          "Strict private checkpoint format and valid fitted scaling")
    raw_sha(args.core, result["training_core_sha256"])
    point_head = run_path(str(args.core))["PointHead"]
    final_hashes = {}
    for arm in arms:
        model = point_head()
        model.load_state_dict(ckpt["models"][arm], strict=True)
        flat = [p.detach().numpy().astype("<f4", copy=False) for p in model.parameters()]
        actual_sha = digest(b"".join(x.tobytes() for x in flat))
        check(all(np.isfinite(x).all() for x in flat) and actual_sha == audit[arm]["final_tensor_sha256"],
              "Direct readback hash of original final " + arm + " parameters")
        final_hashes[arm] = actual_sha
    check(final_hashes["previous_bank"] == "c0f7d0f84e242d5e6549cb38d91e28cdf44fee0f379605d9b7fdcfe456425d93"
          and result["current_runtime_control_tensor_sha256"] == final_hashes["previous_bank"]
          and len(set(final_hashes.values())) == 2, "Exact control and genuine fitted intervention")
    result = dict(status="SPSD_PSEG_MATCHED_DEPLOYMENT_ACTION_BANK_READBACK_COMPLETE",
                  completed=True, natural_exit=0, audit_sha256=digest(Path(__file__).read_bytes()),
                  terminal_sha256=digest(terminal_bytes), worker_sha256=worker, protocol_sha256=lock,
                  fit_result_sha256=args.fit_sha, input_sha256=result["input_receipt_sha256"],
                  checkpoint_sha256=result["output_checkpoint_sha256"], final_tensor_sha256=final_hashes,
                  archived_previous_fit_tensor_parity=False, current_runtime_previous_fit_tensor_parity=True)
    summary = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with (root / "READBACK_SAFE.json").open("xb") as stream:
        stream.write(summary)
    print(json.dumps(dict(status=result["status"], result_sha256=digest(summary))), flush=True)


if __name__ == "__main__":
    main()
