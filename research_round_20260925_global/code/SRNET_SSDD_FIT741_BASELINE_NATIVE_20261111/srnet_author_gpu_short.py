"""Two native SRNet CUDA SGD updates on preselected real FIT images."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import sys
from time import perf_counter
import traceback
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile


P336_SHA = "186d848c277467bae1809612b6ef8f8f4f0c61802df06e6d11e7c773c3486a48"
P335_REPORT_SHA = "81c6f27647f19cdad1013f36620be457887b49713fe9fec425911df005102f68"
TRAIN_PINS = {
    "mmrotate/models/dense_heads/SRNet_head.py": "144552cf27b5970c46ff4e3af1b4e0f59b1b273daa0aadf89cdcc513c8e2dba1",
    "mmrotate/models/dense_heads/SRNet_utils.py": "7d9210f7619fa31ad4bfacfc30e380cb1b4fce5d77e7217de15c4c14b5632c7f",
    "mmrotate/models/detectors/SRNet.py": "7fad564a8c2fb6f34f9e7dc197c7cafc643fe72b16817846fe173e52fe74a40a",
    "mmrotate/core/bbox/assigners/satss_obb_assigner.py": "72a69bb608e0c33becf6033f48eb7ad13bf9b4112892a63029ba0246dd5d6f56",
}
PACKAGES = {"torch": "1.12.0+cu116", "torchvision": "0.13.0+cu116", "mmcv-full": "1.6.0",
            "mmdet": "2.28.2", "numpy": "1.26.4", "pycocotools": "2.0.11", "opencv-python": "4.11.0.86"}
LOSS_KEYS = ("coarse.loss_cls", "coarse.loss_bbox", "refine.loss_cls", "refine.loss_bbox", "mask.masks_loss")
HEADS = ("coarse_head", "refine_head", "mask_head")
GIB = 1024**3
LIMITS = dict(host_available_floor_bytes=16*GIB, process_rss_cap_bytes=10*GIB,
              cuda_free_floor_bytes=6*GIB, cuda_allocator_cap_bytes=6*GIB, worker_seconds=300)
STATUS = "P344_SRNET_AUTHOR_FULL_LOSS_CUDA_TWO_UPDATE_COMPLETE"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def load_source(path, expected):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "Canonical ordinary source required")
    raw = path.read_bytes()
    require(0 < len(raw) <= 1024**2 and sha256(raw).hexdigest() == expected, "Pinned ordinary source differs")
    module = ModuleType("_p344_p336")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def resources():
    """Read Windows memory without adding a runtime dependency."""
    import ctypes
    from ctypes import wintypes

    class Memory(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                    *[(name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_page",
                        "avail_page", "total_virtual", "avail_virtual", "avail_extended")]]

    class ProcessMemory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                    *[(name, ctypes.c_size_t) for name in ("peak_rss", "rss", "peak_paged", "paged",
                        "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile", "private")]]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    system = Memory()
    system.length = ctypes.sizeof(system)
    require(kernel.GlobalMemoryStatusEx(ctypes.byref(system)), "Cannot read host physical memory")
    process = ProcessMemory()
    process.cb = ctypes.sizeof(process)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    api = ctypes.WinDLL("psapi", use_last_error=True)
    api.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
    require(api.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(process), process.cb),
            "Cannot read original worker memory")
    return dict(available_bytes=int(system.avail_phys), rss_bytes=int(process.rss), peak_rss_bytes=int(process.peak_rss))


def check_limits(started, state):
    observed = resources()
    state["observed_memory"] = observed
    require(observed["available_bytes"] >= LIMITS["host_available_floor_bytes"]
            and observed["peak_rss_bytes"] <= LIMITS["process_rss_cap_bytes"]
            and perf_counter()-started <= LIMITS["worker_seconds"], "Bounded SRNet worker resource guard triggered")


def describe(value, torch):
    if torch.is_tensor(value):
        return dict(type=type(value).__name__, shape=list(value.shape), dtype=str(value.dtype), device=str(value.device))
    if isinstance(value, (list, tuple)):
        return dict(type=type(value).__name__, length=len(value), first=[describe(v, torch) for v in value[:2]])
    return dict(type=type(value).__module__+"."+type(value).__name__)


def observe_call(stack, obj, name, label, torch, state):
    original = getattr(obj, name)

    def observed(*args, **kwargs):
        state["stage"] = label
        entry = dict(stage=label, positional=[describe(v, torch) for v in args],
                     keyword={key: describe(value, torch) for key, value in kwargs.items()})
        state.setdefault("native_calls", []).append(entry)
        result = original(*args, **kwargs)
        entry["returned"] = describe(result, torch)
        return result

    stack.enter_context(patch.object(obj, name, observed))


def training_update(model, optimizer, args, kwargs, torch, state):
    optimizer.zero_grad()
    state["stage"] = "author_forward_train"
    state["model_forward_calls"] = state.get("model_forward_calls", 0)+1
    losses = model.forward_train(*args, **kwargs)
    state["completed_forward_calls"] = state.get("completed_forward_calls", 0)+1
    require(type(losses) is dict and set(losses) == set(LOSS_KEYS), "Full original coarse/refine/mask loss set required")
    total, logs = model._parse_losses(losses)
    require(torch.is_tensor(total) and total.ndim == 0 and total.requires_grad and bool(torch.isfinite(total)),
            "Finite connected native total loss required")
    require(all(key in logs and math.isfinite(float(logs[key])) for key in LOSS_KEYS), "Finite native loss components required")
    state["stage"] = "full_loss_backward"
    total.backward()
    state["completed_backward_calls"] = state.get("completed_backward_calls", 0)+1
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    require(gradients and all(bool(torch.isfinite(v).all()) for v in gradients), "All observed native gradients must be finite")
    group_norms = {}
    for name in HEADS:
        parts = [p.grad.detach().double().square().sum() for p in getattr(model, name).parameters() if p.grad is not None]
        group_norms[name] = float(torch.stack(parts).sum().sqrt().item()) if parts else 0.
    require(any(value > 0 for value in group_norms.values()), "Real full-loss training needs a nonzero task gradient")
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    preclip = torch.nn.utils.clip_grad_norm_(parameters, max_norm=35, norm_type=2)
    require(bool(torch.isfinite(preclip)), "Finite original global gradient clip norm required")
    state["stage"] = "sgd_optimizer_step"
    optimizer.step()
    state["optimizer_updates"] = state.get("optimizer_updates", 0)+1
    state["stage"] = "optimizer_step_complete"
    return dict(losses={key: float(logs[key]) for key in LOSS_KEYS}, total_loss=float(total.detach()),
                gradient_tensors=len(gradients), gradient_norms=group_norms, gradient_norm_before_clip=float(preclip),
                learning_rates=[float(group["lr"]) for group in optimizer.param_groups])


def run(a, state, started):
    from importlib.metadata import version

    require(os.name == "nt" and sys.version_info[:2] == (3, 10)
            and Path(sys.executable).resolve(strict=True) == a.expected_python.resolve(strict=True)
            and sys.flags.isolated == sys.flags.utf8_mode == 1 and sys.dont_write_bytecode
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Exact isolated Windows CUDA Python required")
    check_limits(started, state)
    state["stage"] = "frozen_input_validation"
    p336 = load_source(a.p336_source, P336_SHA)
    p335 = p336.load_prior(a.p335_source)
    report = json.loads(p336.read_exact(a.prior_report, P335_REPORT_SHA, 64*1024))
    require(report["status"] == p335.STATUS and report["author_parsed_targets"] == 1661
            and report["FIT"] is True and report["DEV"] is report["TEST"] is report["AP"] is False,
            "Closed original author FIT qualification required")
    pins = {**p335.SOURCE_PINS, **TRAIN_PINS}
    for name, expected in pins.items():
        p336.read_exact(a.source/name, expected, 1024**2)
    index_path = a.fit / "_annotations.coco.json"
    index = json.loads(p336.read_exact(index_path, p335.FIT_INDEX_SHA, 2*1024**2))
    require(len(index["images"]) == 741 and len(index["annotations"]) == 1661, "Exact real FIT741 index required")
    selected = p335.choose(index["images"])[:2]
    p336.read_exact(a.archive, p335.FIT_SHA, 34*1024**2)
    with ZipFile(a.archive) as archive:
        roster = {row["image_id"]: row for row in json.loads(archive.read("manifest.json"))["rows"]}
    require(len(roster) == 741 and len({r["id"] for r in selected}) == 2, "Two preselected original FIT images required")
    for row in selected:
        item = roster[row["id"]]
        require(row["file_name"] == item["original_basename"], "Exact original FIT basename")
        p336.read_exact(a.fit/row["file_name"], item["original_image_ref"]["sha256"], 4*1024**2)
    require(len(p336.read_exact(a.r50, p336.R50_SHA, p336.R50_BYTES)) == p336.R50_BYTES,
            "Exact supplied author ImageNet initializer required")
    sys.path.insert(0, str(a.source))
    import cv2
    import mmcv
    import mmrotate
    import numpy as np
    import torch
    from mmcv.runner import build_optimizer
    from mmcv.runner.hooks import StepLrUpdaterHook
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector

    versions = {key: version(key) for key in PACKAGES}
    require(versions == PACKAGES and Path(mmrotate.__file__).resolve() == a.source/"mmrotate/__init__.py",
            "Exact author CUDA packages and source registry required")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One original CUDA GPU required")
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info(0)
    require(free >= LIMITS["cuda_free_floor_bytes"], "CUDA memory below prospective six-GiB floor")
    torch.cuda.set_per_process_memory_fraction(min(.35, LIMITS["cuda_allocator_cap_bytes"]/total), 0)
    torch.cuda.reset_peak_memory_stats(0)
    random.seed(5201)
    np.random.seed(5201)
    torch.manual_seed(5201)
    torch.cuda.manual_seed_all(5201)
    cfg = mmcv.Config.fromfile(str(a.source/p335.AUTHOR_CONFIG))
    require(dict(cfg.optimizer) == dict(type="SGD", lr=.0025, momentum=.9, weight_decay=.0001)
            and cfg.optimizer_config.grad_clip.max_norm == 35 and cfg.optimizer_config.grad_clip.norm_type == 2
            and cfg.lr_config.policy == "step" and cfg.lr_config.warmup_iters == 500
            and cfg.lr_config.warmup == "linear" and cfg.lr_config.warmup_ratio == 1/3,
            "Original SGD, warmup and gradient clip config required")
    settings = cfg.data.train.dataset.copy()
    settings.ann_file, settings.img_prefix = str(index_path), str(a.fit)+os.sep
    dataset = build_dataset(settings)
    require(type(dataset).__module__ == "mmrotate.datasets.SSDD" and dataset.CLASSES == ("ship",),
            "Original author SSDD training dataset required")
    lookup = {row["id"]: i for i, row in enumerate(dataset.data_infos)}
    state["stage"] = "author_model_initialization"
    cfg.model.backbone.init_cfg.checkpoint = str(a.r50)
    model = build_detector(cfg.model)
    model.init_weights()
    expected = torch.load(str(a.r50), map_location="cpu")
    require(torch.equal(model.backbone.conv1.weight.detach(), expected["conv1.weight"]), "Original R50 initialization differs")
    del expected
    model = model.cuda().train()
    optimizer = build_optimizer(model, cfg.optimizer)
    require(type(optimizer) is torch.optim.SGD, "Actual author SGD optimizer required")
    runner = SimpleNamespace(optimizer=optimizer, epoch=0, iter=0)
    learning_rate = dict(cfg.lr_config)
    learning_rate.pop("policy")
    schedule = StepLrUpdaterHook(**learning_rate)
    schedule.before_run(runner)
    schedule.before_train_epoch(runner)
    watched = {}
    for head in HEADS:
        name, parameter = next((name, p) for name, p in getattr(model, head).named_parameters() if p.requires_grad)
        watched[head] = (head+"."+name, parameter, parameter.detach().cpu().clone())
    state["watch_parameter_names"] = {head: value[0] for head, value in watched.items()}
    with ExitStack() as hooks:
        for head in HEADS:
            observe_call(hooks, getattr(model, head), "loss", head+".loss", torch, state)
        observe_call(hooks, model.coarse_head.assigner, "assign", "coarse_satss_assign", torch, state)
        for update, row in enumerate(selected):
            check_limits(started, state)
            state["stage"] = "author_cpu_pipeline"
            seed = 5201+update
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            data = dataset.prepare_train_img(lookup[row["id"]])
            require(data is not None, "Author augmentation returned no target sample")
            image, meta = data["img"].data, data["img_metas"].data
            boxes, labels, masks = data["gt_bboxes"].data, data["gt_labels"].data, data["gt_masks"].data
            require(image.device.type == boxes.device.type == labels.device.type == "cpu"
                    and tuple(image.shape) == (3, 608, 608) and type(masks).__name__ == "BitmapMasks"
                    and len(labels) == len(boxes) == len(masks) > 0 and masks.height == masks.width == 608,
                    "Native CPU sample tensor, rotated box, label and BitmapMasks alignment required")
            runner.iter = update
            schedule.before_train_iter(runner)
            kwargs = dict(img_metas=[meta], gt_bboxes=[boxes.cuda()], gt_labels=[labels.cuda()], gt_masks=[masks])
            values = training_update(model, optimizer, (image[None].cuda(),), kwargs, torch, state)
            torch.cuda.synchronize()
            state["updates"].append(dict(update=update+1, augmentation_seed=seed, actual_batch_size=1,
                input_hw=[608, 608], targets=len(masks), gt_mask_container=type(masks).__name__, **values))
            print(json.dumps(dict(event="P344_SRNET_AUTHOR_SGD_UPDATE", **state["updates"][-1]), allow_nan=False), flush=True)
    changes = {head: float((parameter.detach().cpu()-before).abs().max()) for head, (_, parameter, before) in watched.items()}
    require(all(math.isfinite(value) for value in changes.values()) and any(value > 0 for value in changes.values()),
            "Two real SGD updates must change an observed author head parameter")
    check_limits(started, state)
    for name, expected in pins.items():
        p336.read_exact(a.source/name, expected, 1024**2)
    state.update(author_commit=p335.AUTHOR_COMMIT, source_pins=pins,
        package_versions=versions, python_version=sys.version.split()[0],
        source_initializer_sha256=p336.R50_SHA, fit_archive_sha256=p335.FIT_SHA, fit_index_sha256=p335.FIT_INDEX_SHA,
        selected_fit_ids_sha256=sha256(json.dumps([r["id"] for r in selected]).encode("ascii")).hexdigest(),
        selected_images=2, actual_batch_size=1, author_config_batch_size=cfg.data.samples_per_gpu,
        initialization_seed=5201, complete_loss_keys=list(LOSS_KEYS), optimizer="SGD",
        original_coarse_satss_loss_executed=True, head_parameter_max_abs_changes=changes,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
        cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        FIT=True, DEV=False, TEST=False, AP=False, trained_detection_checkpoint_loaded=False)
    require(state["cuda_peak_allocated_bytes"] <= LIMITS["cuda_allocator_cap_bytes"], "Observed CUDA peak exceeded cap")
    require(state["model_forward_calls"] == state["completed_forward_calls"]
            == state["completed_backward_calls"] == state["optimizer_updates"] == 2,
            "Exactly two complete native forward/backward/SGD updates required")
    state["status"] = STATUS
    return state


def deny_network(event, _):
    if event.startswith("socket."):
        raise PermissionError("SRNet short uses the locked offline source, inputs and initializer")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("p336-source", "p335-source", "prior-report", "source", "fit", "archive", "r50", "expected-python", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    a = parser.parse_args()
    require(a.output.is_absolute() and a.output.parent.resolve(strict=True) == a.output.parent
            and not a.output.exists() and not a.output.is_symlink(), "Fresh canonical short output directory required")
    require(all(path.is_absolute() and path.resolve(strict=True) == path for path in
            (a.p336_source, a.p335_source, a.prior_report, a.source, a.fit, a.archive, a.r50, a.expected_python)),
            "Canonical exact original inputs required")
    a.output.mkdir()
    state = dict(status="P344_SRNET_AUTHOR_FULL_LOSS_CUDA_SHORT_FAILED", limits=LIMITS, updates=[], native_calls=[],
                 optimizer_updates=0, model_forward_calls=0, completed_forward_calls=0, completed_backward_calls=0,
                 FIT=True, DEV=False, TEST=False, AP=False, stock_S0=True)
    started = perf_counter()
    sys.addaudithook(deny_network)
    try:
        run(a, state, started)
    except BaseException as error:
        state["status"] = "P344_SRNET_AUTHOR_FULL_LOSS_CUDA_SHORT_FAILED"
        state["error"] = dict(type=type(error).__name__, message=str(error),
            frames=[dict(file=Path(frame.filename).name, line=frame.lineno, function=frame.name)
                    for frame in traceback.extract_tb(error.__traceback__)])
        raise
    finally:
        state["elapsed_seconds"] = perf_counter()-started
        raw = (json.dumps(state, sort_keys=True, indent=2, allow_nan=False)+"\n").encode("utf-8")
        with (a.output/"RESULT_SAFE.json").open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(dict(status=state["status"], result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
