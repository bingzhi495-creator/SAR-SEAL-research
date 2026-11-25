"""Independently bind selective calibration outputs after natural worker closure."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path

import numpy as np


STATUS = "SPSD_PSEG_SELECTIVE_WRITER_READBACK_COMPLETE"
BCE = "a2c169f895b1435e689480836ebf4f04fd6e05d84652f51d74e0448ed099faff"
FROZEN_FIT = "ae1b165a111a6b88939eb37b5741421259f0f39531c859a7c7663ad9116d1b05"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def required(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Expected exact closed " + path.name)
    return raw


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--fit-sha", required=True)
    a = p.parse_args()
    root = a.root
    terminal_raw = (root/"observed_fit/TERMINAL.json").read_bytes()
    terminal = json.loads(terminal_raw)
    start = json.loads((root/"observed_fit/START.json").read_bytes())
    worker = digest((root/"fit_selective.py").read_bytes())
    protocol = digest((root/"PRELOCK.md").read_bytes())
    need(all(terminal[k] == start[k] for k in ("phase","worker_sha256","worker_pid","observer_pid","started_utc"))
         and terminal["phase"] == "prepare-edits" and terminal["actual_exit"] == 0
         and terminal["output_footer_verified"] is True and terminal["result_sha256"] == a.fit_sha
         and terminal["worker_sha256"] == worker, "Observed real natural terminal and actual output footer")
    actual = root/"actual"
    result = json.loads(required(actual/"RESULT_SAFE.json",a.fit_sha))
    pre = json.loads(required(actual/"INPUT_SAFE.json",result["input_receipt_sha256"]))
    fields = ("counts","original_counts","selective_fit_rows_sha256","frozen_parent_bce_tensor_sha256",
              "frozen_parent_fit_sha256","oof_authority_fit_sha256","out_of_fold_probability_sha256",
              "out_of_fold_group_sha256","fit_packet_sha256","finite_selector_sha256","feature_worker_sha256",
              "shared_oof_updates")
    need(result["status"] == "SPSD_PSEG_SELECTIVE_WRITER_FIT_COMPLETE"
         and pre["status"] == "FIT_SELECTIVE_WRITER_OOF_PACKETS_ADMITTED"
         and pre["fit_only"] is True and all(pre[k] == result[k] for k in fields)
         and pre["protocol_sha256"] == result["protocol_sha256"] == protocol
         and result["worker_sha256"] == worker and result["frozen_parent_fit_sha256"] == FROZEN_FIT
         and result["frozen_parent_bce_tensor_sha256"] == BCE, "Exact prefit provenance and original BCE parent")
    need(all(r[k] is False for r in (pre,result)
             for k in ("source_gate_labels_opened","source_gate_points_opened","historical_test_opened"))
         and result["fit_labels_opened"] is True and result["shallow_selector_cv"] is False
         and result["detector_forwards"] == result["optimizer_updates"] == 0
         and result["shared_oof_updates"] == 5000 and result["fitted_selective_models"] == result["closed_linear_solves"] == 1,
         "FIT-only linear regression with no gate fit, new detector or shallow CV")
    counts = result["counts"]
    active = counts["robust_nonstatic"]
    need(0 < active <= counts["candidates"] == result["original_counts"]["eligible_candidate_packets"] == 848
         and 0 < counts["active_groups"] <= 439 and 0 < counts["active_images"] <= 447
         and counts["target_negative"]+counts["target_zero"]+counts["target_positive"] == active
         and result["ridge_alpha"] == .02 and result["fit_feature_count"] == 9, "Exact calibration sizes and target partition")
    helper = root/"selective_writer.py"
    required(helper,result["feature_worker_sha256"])
    ridge = json.loads(required(actual/"RIDGE_PRIVATE.json",result["ridge_checkpoint_sha256"]))
    m = run_path(str(helper))
    center = np.asarray(ridge["center"])
    scale = np.asarray(ridge["scale"])
    weights = np.asarray(ridge["weights"])
    probe = np.vstack((center,center+scale,center-scale))
    estimate = m["score"](ridge,probe)
    slope_sum = float(weights.sum())
    expected = ridge["intercept"]+np.asarray((0,slope_sum,-slope_sum))
    need(set(ridge) == {"status","feature_names","center","scale","weights","intercept","alpha","target"}
         and np.allclose(estimate,expected,rtol=0,atol=1e-12)
         and -1 <= ridge["intercept"] <= 1, "Independent finite private learned parameters and score replay")
    doc = dict(status=STATUS,completed=True,natural_exit=0,fit_result_sha256=a.fit_sha,
               terminal_sha256=digest(terminal_raw),worker_sha256=worker,protocol_sha256=protocol,
               input_sha256=result["input_receipt_sha256"],ridge_checkpoint_sha256=result["ridge_checkpoint_sha256"],
               feature_worker_sha256=result["feature_worker_sha256"],frozen_parent_fit_sha256=FROZEN_FIT,
               frozen_parent_bce_tensor_sha256=BCE,audit_sha256=digest(Path(__file__).read_bytes()))
    raw = (json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with (root/"READBACK_SAFE.json").open("xb") as output:
        output.write(raw)
    print(json.dumps(dict(status=STATUS,result_sha256=digest(raw))),flush=True)


if __name__ == "__main__":
    main()
