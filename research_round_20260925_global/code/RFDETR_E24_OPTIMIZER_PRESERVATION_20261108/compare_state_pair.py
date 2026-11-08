"""Read only closed pair artifacts and report exact or numeric first-gradient agreement."""
import argparse
import json
import math
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from state_preservation import (START_STEP, END_STEP, UPDATES, GRADIENT_LIMIT, compare_first, file_ref, need,
                                tensor_dict_sha256)


def load_gradient(arm_dir, report):
    path = arm_dir / "FIRST_CLIPPED_GRAD_PRIVATE.pt"
    need(path.resolve(strict=True) == path and not path.is_symlink() and
         file_ref(path, GRADIENT_LIMIT) == report["gradient_ref"], "Exact closed private first gradient archive")
    with path.open("rb") as stream:
        row = torch.load(stream, map_location="cpu", weights_only=True)
    need(set(row) == {"schema", "tensors"} and row["schema"] == "RFDETR_FIRST_CLIPPED_GRAD_V1"
         and len(row["tensors"]) == report["clipped_grad_tensors"]
         and tensor_dict_sha256(row["tensors"]) == report["clipped_grad_sha256"],
         "Exact first clipped gradients and key set")
    return row["tensors"]


def numeric_comparison(left, right):
    need(list(left) == list(right), "First gradient tensor identities differ")
    diff_sum = []
    l2 = []
    r2 = []
    dots = []
    max_abs = 0.0
    changed = 0
    for name, tensor in left.items():
        other = right[name]
        need(tensor.shape == other.shape and tensor.dtype == other.dtype and tensor.is_floating_point()
             and bool(torch.isfinite(tensor).all()) and bool(torch.isfinite(other).all()),
             "Expected finite equal-shape first gradients")
        a = tensor.double()
        b = other.double()
        delta = a - b
        diff_sum.append(float(delta.square().sum()))
        l2.append(float(a.square().sum()))
        r2.append(float(b.square().sum()))
        dots.append(float((a * b).sum()))
        max_abs = max(max_abs, float(delta.abs().max()) if delta.numel() else 0.0)
        changed += int(torch.count_nonzero(delta))
    nleft, nright = math.sqrt(math.fsum(l2)), math.sqrt(math.fsum(r2))
    norm = math.sqrt(math.fsum(diff_sum))
    need(nleft > 0 and nright > 0, "Expected nonzero first gradients")
    return dict(left_l2=nleft, right_l2=nright, diff_l2=norm, relative_l2_to_S=norm/nleft,
                max_abs_difference=max_abs, changed_elements=changed, cosine=math.fsum(dots)/(nleft*nright))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = HERE.parent
    pair = args.pair_dir.resolve(strict=True)
    out = args.output.resolve()
    need(pair.parent == root and root in out.parents and out.parent.is_dir() and not out.exists(),
         "Expected closed private pair and fresh private result")
    torch.set_num_threads(2)
    need(not torch.cuda.is_available(), "CPU-only first gradient comparison")
    reports = {}
    for arm in ("S", "F"):
        arm_dir = pair / arm
        report_path = arm_dir / "RESULT.json"
        need(report_path.resolve(strict=True) == report_path and report_path.stat().st_size <= 1024**2,
             "Exact bounded reported arm result")
        report = json.loads(report_path.read_bytes())
        meta = report["meta"]
        need(meta["arm"] == arm and meta["seed"] == 5201 and meta["updates"] == meta["ema_updates"] == UPDATES
             and meta["microbatches"] == 1488 and meta["starting_ema_clock"] == START_STEP
             and meta["ema_absolute_updates"] == END_STEP and meta["DEV"] is meta["TEST"] is False,
             "Expected completed fixed stock native optimizer comparison arm")
        need(file_ref(arm_dir / "ENDPOINT.pt", 256*1024**2) == report["endpoint_ref"],
             "Exact reported endpoint artifact")
        reports[arm] = report
    s, f = reports["S"], reports["F"]
    exact_meta = ("protocol_commit", "seed", "archive_start_sha256", "starting_ema_clock",
                  "initial_online_sha256", "initial_ema_sha256", "parameter_group_order_sha256",
                  "applied_group_lrs", "planned_visit_sha256", "visit_sha256")
    same = all(s["meta"][k] == f["meta"][k] for k in exact_meta)
    first = compare_first(s["first_step"], f["first_step"])
    need(same and first["same_initial_forward_conditions"], "Unequal initial weights, clocks, FIT tape or applied LR")
    a = load_gradient(pair / "S", s["first_step"])
    b = load_gradient(pair / "F", f["first_step"])
    diff = numeric_comparison(a, b)
    need(not first["same_clipped_gradient"] or diff["changed_elements"] == 0,
         "Gradient digest and numeric tensor identity are inconsistent")
    summary = dict(status="ACTUAL_FIRST_STEP_TENSOR_COMPARISON_PENDING_ROOT_ADMISSION",
                   same_initial_state_data_lr=True, **first, numerical_first_gradient=diff,
                   raw_update_l2={arm: r["first_step"]["raw_update_l2"] for arm, r in reports.items()},
                   mask_head_update_l2={arm: r["first_step"]["segmentation_head_update_l2"] for arm, r in reports.items()},
                   formal_training_accepted=False, AP=False, DEV=False, TEST=False)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
