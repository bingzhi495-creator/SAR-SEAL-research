"""Refit one PointRend-style head on the historically accepted HRSID FIT109 pool."""

import argparse
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from pycocotools import mask as codec


SOURCE_PROTOCOL = "648db29c9f841f33eb54fdcb6d883c743442940dcd7cea475ad1a91b424c1aa4"
CARRIER_READBACK = "19393564375cde464d6ab3a51966b8861f28f02f2dc33a08fde14de38f13dd38"
CHECKPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
RECIPE = dict(seed=5217, schedule_seed=152170, point_seed=252170, images=109,
              annotations=496, eligible_matches=392, optimizer_updates=200,
              targets_per_update=196, head_scaler=False)
MIB = 1024**2
CARRIER = "RFDETR_POINTREND_FIT109_CARRIER_ACTUAL_20261121_01"
SUPPORT = "RFDETR_FIT109_QUERY_BASIS_CAPTURE_ACTUAL_20261121_01"
BUDGET = "RFDETR_QUERY_FEATURE_FIT109_FOUR_ARM_ACTUAL_20261121_01"
FIT_ROSTER = "HRSID_FIT_NATIVE_QUERY_RETENTION_ACTUAL_PRIVATE_20261109_01/COHORT_PRIVATE.json"
FIT_GT = "HRSID_FIT_NATIVE_COCO_LAYOUT_ACTUAL_PRIVATE_20261109_01/train/_annotations.coco.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, digest, limit=MIB):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and sha256(raw).hexdigest() == digest,
            "Pinned input " + path.name)
    return raw


def load(code, relative, digest):
    path = code / relative
    module = ModuleType("_pointrend_full_fit_" + path.stem)
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(pinned(path, digest), str(path), "exec"), module.__dict__)
    return module


def fit_full_recipe(legacy, adapter, rows, archived_folds, load_roi, *,
                    steps=RECIPE["optimizer_updates"], cpu_threads=4):
    """Keep all source groups eligible while preserving the archived fold-zero RNGs."""
    by_group = legacy._index_positives(rows, archived_folds)
    included = {group: fold + 1 for group, fold in archived_folds.items()}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(cpu_threads)
        with torch.enable_grad():
            return legacy._fit_one_fold(0, by_group, included, load_roi, adapter, RECIPE["seed"], steps)
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "private-root", "capture-readback", "raw-capture", "raw-acceptance",
                 "shapely-site", "prelock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prelock-sha256", required=True)
    args = parser.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and sys.version_info[:3] == (3, 10, 13) and np.__version__ == "1.26.4"
            and torch.__version__ == "2.10.0+cu130" and not torch.cuda.is_initialized(),
            "Previously admitted CPU runtime")
    require(args.output.is_absolute() and args.output.parent.is_dir() and not args.output.exists(),
            "Fresh private execution directory")
    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("Closed local inputs only")
    sys.addaudithook(offline)
    prelock = json.loads(pinned(args.prelock, args.prelock_sha256, 64 * 1024))
    worker_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    require(prelock["source_refit"] == dict(RECIPE, worker_sha256=worker_sha),
            "Locked full-FIT source recipe and worker")
    code = args.code.resolve(strict=True)
    root = args.private_root.resolve(strict=True)
    phase = "RFDETR_POINTREND_FIT109_20261121/"
    query = "RFDETR_QUERY_FEATURE_FIT109_20261121/"
    matched = "RFDETR_MATCHED_POINT_FORMAL_20261120/"
    archive = load(code, phase + "run_point_fit109.py", "69007842f9afd5e1ab2bc86c78ce793f30efc5ebb220b1304fd7f949b3bd2e86")
    legacy = load(code, phase + "fit_point_heads.py", "0f34439648088d20d86ff40cabda8550d6973f4aeb8a22b6d58443a41e3e0918")
    adapter = load(code, "RFDETR_POINTREND_SUBDIVISION_20261121/roi_point_head.py",
                   "672bdd2aed8c0056f29c816f829bae9ceae5f9a685309beebf107aa59e547274")
    old = load(code, query + "run_fit109_query.py", "97fdc3987462aabbcd7d4fbe672c9788a3b969a221243e13d82473093170a15a")
    spot = old.load(code / "RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/run_spot.py")
    prior = old.load(code / (matched + "run_fit109.py"))
    plan = old.load(code / (matched + "fit_plan.py"))
    scoring = old.load(code / (query + "fit109_score.py"))
    protocol = json.loads(pinned(code / (phase + "POINTREND_FIT109_PRELOCK.json"), SOURCE_PROTOCOL))
    require(protocol["FIT"] and not protocol["DEV"] and not protocol["TEST"]
            and protocol["frozen_checkpoint_sha256"] == CHECKPOINT, "Fixed detector and FIT-only source")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    joined = SimpleNamespace(capture=root / CARRIER, capture_readback=args.capture_readback,
                             capture_readback_sha=CARRIER_READBACK, old_support=root / SUPPORT,
                             fit_gt=root / FIT_GT)
    accepted, capture_index = archive.admit_capture(joined, protocol)
    raw = SimpleNamespace(capture=args.raw_capture, acceptance=args.raw_acceptance,
                          acceptance_sha256=old.OLD_RAW_ACCEPTANCE, source_cohort=root / FIT_ROSTER)
    roster, _ = prior.read_population(raw, spot)
    require(set(capture_index) == {item["image_id"] for item, _, _ in roster}, "Exact FIT capture roster")
    helpers = scoring.load_helpers(code, args.shapely_site)
    pools, rows = archive.collect_old_support(joined, SimpleNamespace(old=old, spot=spot), roster, protocol)
    budget, folds = archive.read_budgets(root / BUDGET, protocol, rows, plan, roster, helpers.geometry)
    args.output.mkdir()
    prelabel = archive.write(args.output / "PRELABEL_SAFE.json", dict(
        status="POINTREND_FULL_FIT109_SOURCE_BOUND", prelock_sha256=args.prelock_sha256,
        source_protocol_sha256=SOURCE_PROTOCOL, carrier_readback_sha256=CARRIER_READBACK,
        carrier_index_sha256=accepted["index_sha256"], raw_acceptance_sha256=old.OLD_RAW_ACCEPTANCE,
        support_sha256=protocol["prior_oof_support_sha256"], budget_sha256=protocol["prior_oof_budget_sha256"],
        FIT=True, DEV=False, TEST=False, GT_opened=False, optimizer_updates=0, images=len(roster)))
    print(json.dumps(dict(event="POINTREND_FULL_FIT109_SOURCE_BOUND", sha256=prelabel)), flush=True)

    gt, attached = old.attach_fit(joined, roster, rows, pools, helpers, spot)
    truth = {ann["id"]: ann for ann in gt["annotations"]}
    positive = [dict(row, k=budget[helpers.geometry.identity(row)]["k"]) for row in attached
                if row["role"] == "valid" and row["truth"] is not None and len(row["old14"]) > 0]
    require(len(truth) == RECIPE["annotations"] and len(positive) == RECIPE["eligible_matches"],
            "Exact FIT native masks and stock 0.50 matched training rows")

    @lru_cache(maxsize=12)
    def image(iid):
        return archive.captured_arrays(joined.capture, capture_index[iid])

    @lru_cache(maxsize=96)
    def label(aid):
        ann = truth[aid]
        return codec.decode(codec.merge(codec.frPyObjects(ann["segmentation"], 800, 800))).astype(bool)

    def load_roi(row):
        spatial, values = image(row["image_id"])
        query_id, _, _, box, logits = values[row["rank"]]
        require(query_id == row["query_id"], "Verified stock training ROI identity")
        return spatial, logits, box, label(row["matched_gt_id"])

    started = perf_counter()
    head, audit = fit_full_recipe(legacy, adapter, positive, folds, load_roi)
    fit_seconds = perf_counter() - started
    points = RECIPE["optimizer_updates"] * RECIPE["targets_per_update"]
    require(audit["updates"] == RECIPE["optimizer_updates"] and audit["supervised_points"] == points
            and audit["schedule_seed"] == RECIPE["schedule_seed"] and audit["point_seed"] == RECIPE["point_seed"]
            and audit["train_positive_rows"] == len(positive) and audit["held_positive_rows"] == audit["held_sources"] == 0,
            "The single archived recipe operated over the complete eligible source pool")
    audit_sha = archive.write(args.output / "HEAD_AUDIT_PRIVATE.json", audit)
    weight_path = args.output / "HEAD_PRIVATE.npz"
    state = {name: param.detach().cpu().numpy() for name, param in head.state_dict().items()}
    with weight_path.open("xb") as stream:
        np.savez_compressed(stream, **state)
        stream.flush()
        os.fsync(stream.fileno())
    head_sha = sha256(weight_path.read_bytes()).hexdigest()
    result = dict(status="POINTREND_FULL_FIT109_SOURCE_REFIT_COMPLETE", worker_sha256=worker_sha,
                  prelock_sha256=args.prelock_sha256, source_prelabel_sha256=prelabel,
                  head_sha256=head_sha, audit_sha256=audit_sha, scaler=False, FIT=True, DEV=False, TEST=False,
                  source_images=len(roster), source_annotations=len(truth), carrier_checkpoint_sha256=CHECKPOINT,
                  supervising_rows=audit["train_positive_rows"], supervising_sources=audit["train_positive_sources"],
                  sampled_rows=audit["used_positive_rows"], sampled_sources=audit["used_sources"],
                  initialization_seed=audit["initialization_seed"], schedule_seed=audit["schedule_seed"],
                  point_seed=audit["point_seed"], optimizer=audit["optimizer"], lr=audit["lr"],
                  weight_decay=audit["weight_decay"], optimizer_updates=audit["updates"], native_targets=points,
                  trainable_parameters=audit["trainable_parameters"], parameter_update_l2=audit["parameter_update_l2"],
                  loss_first=audit["loss_first"], loss_last=audit["loss_last"], loss_mean=audit["loss_mean"],
                  fit_seconds=fit_seconds)
    result_sha = archive.write(args.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=result["status"], result_sha256=result_sha)), flush=True)


if __name__ == "__main__":
    main()
