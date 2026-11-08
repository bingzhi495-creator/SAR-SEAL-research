"""Score the fixed DEV187 Windows optimizer-history comparison after pair closure."""
import argparse
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import ModuleType


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "RFDETR_E24_WARM_BD_EVAL_CANDIDATE_20261108_01/evaluate_warm.py"
BASE_SHA = "48c8b53563906a96f8a1067330825111bd11320255b4780501c9a965270dfc48"
GEOMETRY = ROOT / (
    "github_release_20261011/research_round_20260925_global/code/"
    "RFDETR_DGX_CLOSED_ENDPOINT_GEOMETRY_20261108/measure_geometry.py"
)
GEOMETRY_SHA = "a1fc4445d9345bd143c54dcae40d375a3d49cf42d68226eb6435fd9a1e9a8fa7"
PAIR = ROOT / "RFDETR_E24_STRICT_OPTIMIZER_ACTUAL_PRIVATE_20261109_01"
ACCEPTANCE = PAIR / "ROOT_CLOSED_PAIR_ACCEPTANCE_PRIVATE.json"
SCHEMA = "RFDETR_E24_OPTIMIZER_STATE_PAIR_CLOSED_V1"
ARMS = ("F", "S")
CASES = ("E24", *ARMS)
MIB = 1024**2


def load_base():
    raw = BASE.read_bytes()
    if sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError("Exact prior native Windows evaluation helper required")
    module = ModuleType("_optimizer_pair_native_windows_eval")
    module.__file__ = str(BASE)
    exec(compile(raw, str(BASE), "exec"), module.__dict__)
    return module


def validate_acceptance(base, pair, receipt):
    base.require(type(receipt) is dict and receipt.get("schema") == SCHEMA,
                 "Exact root closed pair acceptance schema required")
    protocol = receipt.get("protocol_commit")
    base.require(type(protocol) is str and re.fullmatch(r"[0-9a-f]{40}", protocol),
                 "Exact frozen protocol commit required")
    arms = receipt.get("arms")
    base.exact_dict(arms, ARMS, "Both original closed training arms are required")
    refs = {}
    for arm in ARMS:
        row = arms[arm]
        base.exact_dict(row, ("closed", "checkpoint_ref"), "Exact closed arm receipt required")
        base.require(row["closed"] is True, "Both original training arms must be closed")
        ref = row["checkpoint_ref"]
        base.require(base.path_from_ref(ref, 256*MIB) == pair / arm / "ENDPOINT.pt",
                     "Wrong original arm endpoint path")
        refs[arm] = ref
    return protocol, refs



def endpoint_state(base, checkpoint, arm, protocol):
    base.exact_dict(checkpoint, ("meta", "ema_state"), "Exact model-only warm endpoint required")
    meta = checkpoint["meta"]
    state = checkpoint["ema_state"]
    base.require(type(meta) is dict and isinstance(state, dict) and state,
                 "Warm endpoint meta and EMA tensor mapping required")
    expected = dict(arm=arm, seed=5201, epochs=2, updates=372, microbatches=1488,
                    ema_updates=372, ema_absolute_updates=4836, starting_ema_clock=4464,
                    archive_start_sha256=base.E24_SHA, protocol_commit=protocol,
                    DEV=False, TEST=False)
    base.require(all(type(meta.get(key)) is type(value) and meta[key] == value
                     for key, value in expected.items()), "Warm endpoint meta differs")
    base.require(arm in ARMS and meta.get("adam_step_counts") == ({"372": 543} if arm == "F" else {"4836": 543}),
                 "Original arm AdamW update counts differ")
    return state



def load_geometry(base):
    return base.source_module(GEOMETRY, GEOMETRY_SHA, "_optimizer_pair_closed_geometry")


def summarize_retained(base, geometry, dataset, case_rows, mask, np):
    """Measure score-free coverage on the same kept masks as COCO."""
    images = [(row["id"], row["height"], row["width"]) for row in dataset["images"]]
    dimensions = {image: (height, width) for image, height, width in images}
    base.require(len(images) == len(dimensions), "Unique original geometry image IDs required")
    truth = defaultdict(list)
    for row in dataset["annotations"]:
        image = row["image_id"]
        height, width = dimensions[image]
        base.require(row["category_id"] == 1 and row["area"] > 0 and row.get("iscrowd", 0) == 0,
                     "Expected single-category noncrowd GT masks")
        truth[image].append((row["area"] < 1024, geometry.as_rle(row["segmentation"], height, width, mask)))
    small_flags = [small for image, _, _ in images for small, _ in truth.get(image, ())]
    total = len(small_flags)
    coverage, case_flags = {}, {}
    for expected, (arm, rows) in zip(CASES, case_rows, strict=True):
        base.require(expected == arm, "Exact ordered evaluation cases required")
        predictions = defaultdict(list)
        for row in rows:
            base.require(row["image_id"] in dimensions and row["category_id"] == 1,
                         "Mapped retained predictions must belong to fixed DEV images")
            predictions[row["image_id"]].append(row["segmentation"])
        base.require(all(len(items) <= 100 for items in predictions.values()), "Retained native top100 bound")
        coverage[arm], case_flags[arm] = {}, {}
        for label, threshold in geometry.THRESHOLDS.items():
            flags, unique, unique_small, unique_other = geometry.aggregate(
                truth, predictions, threshold, images, mask, np)
            base.require(len(flags) == total and unique <= sum(flags), "Invalid injective coverage")
            covered_small = sum(c and small for c, small in zip(flags, small_flags, strict=True))
            case_flags[arm][label] = flags
            coverage[arm][label] = dict(
                per_gt_max=sum(flags), max_injective=unique,
                small_per_gt_max=covered_small, other_per_gt_max=sum(flags) - covered_small,
                small_only_max_injective=unique_small, other_only_max_injective=unique_other,
            )
    transitions = {}
    for label in geometry.THRESHOLDS:
        paired = list(zip(case_flags["F"][label], case_flags["S"][label], small_flags, strict=True))
        groups = dict(all=paired, small=[row for row in paired if row[2]], other=[row for row in paired if not row[2]])
        transitions[label] = {
            group: dict(gained=sum(not f and s for f, s, _ in rows),
                        lost=sum(f and not s for f, s, _ in rows))
            for group, rows in groups.items()
        }
        counts = transitions[label]["all"]
        base.require(counts["gained"] - counts["lost"] ==
                     coverage["S"][label]["per_gt_max"] - coverage["F"][label]["per_gt_max"],
                     "Paired coverage transitions must close")
    return dict(scope="MAPPED_RETAINED_MASKS_ONLY_IGNORE_SCORES",
                small_definition="recorded annotation area < 1024; subset matching excludes other GT",
                thresholds="exact inclusive rational mask IoU, ordinary noncrowd",
                GT_instances=total, small_GT=sum(small_flags), other_GT=total-sum(small_flags),
                coverage=coverage, S_vs_F_per_GT=transitions)



E100_CONTEXT = ROOT / (
    "github_release_20261011/research_round_20260925_global/results/"
    "RFDETR_NATIVE_E100_DGX_FOUR_ENDPOINTS_ACTUAL_SAFE_20261108.json"
)
E100_CONTEXT_SHA = "6f58eefa92ab85c3dfb0c802b6fb0f322704124373d7ce69ee7e15080cd6adb5"
METRIC_ORDER = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl")


def load_endpoint(base, model, arm, ref, protocol, torch, device):
    with base.verified_file(ref, 256*MIB) as stream:
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    state = endpoint_state(base, checkpoint, arm, protocol)
    model.to("cpu")
    expected = model.state_dict()
    base.require(set(state) == set(expected), "Native EMA tensor keys differ")
    for name, value in state.items():
        target = expected[name]
        base.require(torch.is_tensor(value) and value.device.type == "cpu" and value.shape == target.shape
                     and value.dtype == target.dtype
                     and (not value.is_floating_point() or bool(torch.isfinite(value).all())),
                     "Native EMA tensor differs at " + name)
    loaded = model.load_state_dict(state, strict=True)
    base.require(not loaded.missing_keys and not loaded.unexpected_keys
                 and all(torch.equal(value, state[name]) for name, value in model.state_dict().items()),
                 "Exact EMA tensor readback failed")
    model.to(device).eval()
    return len(state)


def closed_training_costs(base, receipt):
    rows = receipt.get("train_metrics")
    if rows is None:
        return None
    base.exact_dict(rows, ARMS, "Exact root-accepted training-cost arms")
    for arm, row in rows.items():
        base.require(type(row) is dict and row.get("updates") == 372 and row.get("microbatches") == 1488
                     and row.get("absolute_ema") == 4836
                     and type(row.get("train_seconds")) in (int, float)
                     and math.isfinite(row["train_seconds"]) and row["train_seconds"] > 0
                     and type(row.get("peak_reserved_bytes")) is int and row["peak_reserved_bytes"] > 0,
                     "Root-accepted per-arm training costs differ")
    return rows


def prior_dgx_context(base):
    ref = base.pinned_ref(E100_CONTEXT, E100_CONTEXT_SHA)
    prior = json.loads(base.read_bytes(ref, MIB))
    base.require(prior["scientific_status"] == "SAME_DGX_FIXED_AP_ACCEPTED"
                 and prior["population"] == base.DATA_ROLE and prior["images"] == 187
                 and prior["GT_instances"] == 380 and prior["TEST"] is False,
                 "Exact closed same-DGX E100 reference required")
    case = prior["cases"]["E100"]
    base.require(case["epoch"] == 100 and case["optimizer_updates"] == 18600,
                 "Unexpected historical DGX reference budget")
    return dict(scope="CROSS_RUNTIME_CONTEXT_ONLY", backend=prior["backend"],
                accepted_summary_sha256=ref["sha256"], optimizer_updates=case["optimizer_updates"],
                metrics={key: case[key] for key in ("AP", "AP75", "APs", "AR100", "all12")})


def run(acceptance, acceptance_sha256, out):
    base = load_base()
    base.require(sys.platform == "win32" and Path(sys.executable).resolve(strict=True) == base.HOST.resolve(strict=True)
                 and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
                 "Exact offline pinned Windows evaluation interpreter required")
    base.require(acceptance == ACCEPTANCE and PAIR.resolve(strict=True) == PAIR
                 and re.fullmatch(r"[0-9a-f]{64}", acceptance_sha256),
                 "Exact canonical root-accepted original pair required")
    acceptance_ref = base.pinned_ref(acceptance, acceptance_sha256)
    receipt = json.loads(base.read_bytes(acceptance_ref, MIB))
    protocol, refs = validate_acceptance(base, PAIR, receipt)
    costs = closed_training_costs(base, receipt)
    dgx = prior_dgx_context(base)
    items, manifest_ref = base.admitted_images()
    base.require(out.is_absolute() and out.resolve() == out and out.parent == PAIR and not out.exists(),
                 "Fresh canonical output directly under the closed pair required")
    geometry = load_geometry(base)

    import numpy as np
    import torch
    from pycocotools import mask as codec
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models.postprocess import PostProcess, _MASK_CHUNK
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    base.require(version("pycocotools") == "2.0.11", "Exact pinned official COCO implementation required")
    base.require(torch.cuda.is_available() and torch.cuda.device_count() == 1
                 and not torch.is_autocast_enabled("cuda") and not torch.is_autocast_enabled("cpu"),
                 "One admitted Windows CUDA GPU in eager FP32 required")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    base.require(free >= 6*1024**3, "Local CUDA memory floor")
    torch.cuda.set_per_process_memory_fraction(min(.35, 6*1024**3/total), device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    base.read_bytes(base.pinned_ref(base.SETUP_PRIOR, base.SETUP_PRIOR_SHA), MIB)
    setup = base.source_module(base.SETUP, base.SETUP_SHA, "_optimizer_pair_e24_setup")
    core = base.source_module(base.FIT_CORE, base.FIT_CORE_SHA, "_optimizer_pair_native_candidates")
    official = base.source_module(base.OFFICIAL, base.OFFICIAL_SHA, "_optimizer_pair_coco")
    native = setup.read_pinned(setup.NATIVE / "SPEC.json", setup.NATIVE_SHA)
    base.require(native["archive_ref"]["sha256"] == base.E24_SHA, "Exact independent E24 reference")
    transform = make_coco_transforms_square_div_64(
        "val", 576, multi_scale=False, expanded_scales=False, skip_random_resize=True,
        patch_size=12, num_windows=1, aug_config=None, scale_jitter=True, gpu_postprocess=False,
    )
    runtime = (torch, np, codec, PostProcess, _MASK_CHUNK, decode_pil_image_bytes,
               transform, nested_tensor_from_tensor_list)
    out.mkdir()
    reports = {}
    started = time.monotonic()
    model, criterion = setup.setup(torch, device, native)
    del criterion
    model.eval()
    for arm in CASES:
        torch.cuda.reset_peak_memory_stats(device)
        if arm == "E24":
            state_ref, tensors = native["archive_ref"], len(model.state_dict())
        else:
            started = time.monotonic()
            state_ref = refs[arm]
            tensors = load_endpoint(base, model, arm, state_ref, protocol, torch, device)
        base.require(tensors == 545, "Exact RF-DETR Seg-Nano EMA state tensor count")
        torch.cuda.synchronize(device)
        reports[arm] = dict(state_ref=state_ref, strict_tensors=tensors, setup_seconds=time.monotonic()-started,
                            **base.export_arm(arm, model, items, out, core, runtime, device),
                            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    del model

    # Integer GT is opened only after all prediction artifacts have finalized.
    gt_ref = base.score_arms(items, reports, official)
    dataset = json.loads(base.read_bytes(gt_ref, MIB))
    started = time.monotonic()
    case_rows = ((arm, json.loads(base.read_bytes(reports[arm]["predictions_ref"], 256*MIB))) for arm in CASES)
    retained = summarize_retained(base, geometry, dataset, case_rows, codec, np)
    retained["seconds"] = time.monotonic()-started
    retained["source_sha256"] = GEOMETRY_SHA
    retained["prediction_sha256"] = {arm: reports[arm]["predictions_ref"]["sha256"] for arm in CASES}
    for report in reports.values():
        stats = report["metrics"]["stats"]
        base.require(len(stats) == len(METRIC_ORDER) and all(math.isfinite(x) and -1 <= x <= 1 for x in stats),
                     "Expected twelve official COCO scalars")
    e24, f, s = (reports[arm]["metrics"]["AP"] for arm in CASES)
    result = dict(status="OPTIMIZER_STATE_FS_DEV187_COMPLETE_PENDING_ROOT_ACCEPTANCE",
                  protocol_commit=protocol, population_role=base.DATA_ROLE, images=base.POPULATION,
                  GT_instances=380, category_ids=[1], iou_type="segm", max_dets=[1,10,100],
                  device=torch.cuda.get_device_name(0), inference_precision="eager FP32",
                  fp32_matmul_precision=torch.get_float32_matmul_precision(),
                  cudnn_tf32=torch.backends.cudnn.allow_tf32, official_metric_order=METRIC_ORDER,
                  pycocotools_version=version("pycocotools"), acceptance_ref=acceptance_ref,
                  image_manifest_ref=manifest_ref, integer_GT_ref=gt_ref, arms=reports,
                  root_accepted_training_costs=costs, dgx_E100_context=dgx,
                  retained_score_free_geometry=retained,
                  AP_point_deltas=dict(S_minus_F=100*(s-f), S_minus_E24=100*(s-e24),
                                       S_minus_max_F_E24=100*(s-max(f,e24))),
                  new_training=False, TEST=False, terminal_self_certified=False)
    return base.json_file(out / "RESULT.json", result)


def offline(event, args):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise RuntimeError("Fixed DEV187 evaluation is offline")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--acceptance", type=Path, required=True)
    cli.add_argument("--acceptance-sha256", required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args()
    sys.addaudithook(offline)
    result = run(args.acceptance, args.acceptance_sha256, args.output_dir)
    print(json.dumps(dict(result_ref=result)), flush=True)


if __name__ == "__main__":
    main()
