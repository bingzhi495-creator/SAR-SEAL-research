"""Run one prospectively locked equal-budget native RF-DETR optimizer-state arm."""
from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import time
from types import ModuleType


HERE = Path(__file__).resolve().parent
D = HERE.parent
P77 = D / "RFDETR_E24_WARM_NATIVE_A_SCOUT_20261108_01"
P68 = D / "RFDETR_FIT_TRAIN_STYLE_POINT_DIAGNOSTIC_20261108_01"
P67 = D / "RFDETR_FIT_POINT_SAMPLING_DIAGNOSTIC_20261108_01"
P77_SOURCE = {
    "run_warm_arm.py": "a4cb608f80d4cd5780f0fb2271d12a99187e5b0c53657a766edad8adc378d3d6",
    "warm_training.py": "bb3b124e09ec0bb622762127f2ef7be216269c4d02460cf3da1ca29abc81d7bb",
    "risk_criterion.py": "65e5a6508ceaf161cd09177cc8f258c7b92f2963c99600b5fd43fd2ea152f035",
    "point_allocation.py": "21b98995b068b7beb09eeB78b0449872784f89aa50784e60b3a5df89b9475c88".lower(),
}
ARCHIVE = D / "RFDETR_NATIVE_E24_FIT_LOCAL_INPUT_TRANSFER_PRIVATE_20261108_01/checkpoint_23.ckpt"
ARCHIVE_SHA = "30d6b22104122e6a38f8ec7127e30ab1ae46117d898a1131c202bc5b673530b5"
RUNTIME = D / "RFDETR_WINDOWS_112_RUNTIME_PRIVATE_20261108_01/PYTHON_HOST_31013/python.exe"

sys.path.insert(0, str(HERE))
from state_preservation import (
    START_STEP, END_STEP, UPDATES, ContinuationEMA, FirstStepTrace, file_ref,
    install_optimizer_history, need, tensor_dict_sha256,
)


def offline(event, args):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise RuntimeError("FIT-only optimizer comparison is offline")


def p77_source():
    refs = [(P77 / key, digest) for key, digest in P77_SOURCE.items()] + [
        (P68 / "train_style_point_probe.py", "9a28e745959974d1bc3ebe81c1e8ed5ea0afdb2a76ed0915ffbf8ea5a6933c77"),
        (P67 / "run_sampling.py", "79d8342ee2037ff1bef3a4e5b98988356cce893c3ce53cfd28ed34c1f0be16d8"),
    ]
    for path, digest in refs:
        need(sha256(path.read_bytes()).hexdigest() == digest, "Exact unchanged P77/P68 dependency " + path.name)
    source = P77 / "run_warm_arm.py"
    original = ModuleType("_rf_optimizer_p77")
    original.__file__ = str(source)
    sys.modules[original.__name__] = original
    exec(compile(source.read_bytes(), str(source), "exec"), original.__dict__)
    return original


def initialize(arm, device, original):
    import torch
    from rfdetr import config
    from rfdetr._namespace import _namespace_from_configs
    from rfdetr.models._defaults import MODEL_DEFAULTS
    from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config
    from torch.optim.swa_utils import AveragedModel

    native = original.read_pinned(original.NATIVE / "SPEC.json", original.NATIVE_SHA)
    q = native["archive_ref"]
    need(Path(q["path"]) == ARCHIVE and q["sha256"] == ARCHIVE_SHA, "Use the independently closed E24 only")
    path = original.exact_file(q, 1024**3)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        checksum = sha256()
        for block in iter(lambda: stream.read(1024**2), b""):
            checksum.update(block)
        need(checksum.hexdigest() == ARCHIVE_SHA and before.st_size == q["bytes"], "Exact closed E24 archive")
        stream.seek(0)
        archive = torch.load(stream, map_location="cpu", weights_only=True)
        after = os.fstat(stream.fileno())
        need((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
             "Original E24 archive changed during reading")
    mods = {key: original.module(native["sources"][key], "_preserve_" + key)
            for key in ("archive", "boundary", "configuration", "e24_export", "pilot")}
    from importlib.metadata import version
    need(all(version(name) == v for name, v in mods["pilot"].VERSIONS.items()), "Exact pinned native numeric runtime")
    mods["archive"].validate_archive_progress(archive, 24, boundary_api=mods["boundary"],
                                              recipe_sha256=native["recipe_sha256"])
    recipe = original.exact_file(native["recipe"], 1024**2)
    original.exact_bytes(native["recipe"], 1024**2)
    m, t = mods["configuration"]._load_configs(recipe, HERE)
    mods["pilot"].match_model_config(archive, m, t, config)
    need(t.optimizer == "adamw" and not m.fused_optimizer and t.grad_accum_steps == 4 and t.clip_max_norm == 0.1
         and t.weight_decay == 0.0001 and t.ema_decay == 0.993 and t.ema_tau == 100
         and (m.patch_size, m.num_windows, t.mask_point_sample_ratio) == (12, 1, 16) and t.scale_jitter,
         "Exact original native stock loss and optimizer recipe")
    model = build_model_from_config(m, t, defaults=replace(MODEL_DEFAULTS, force_no_pretrain=True))
    eager = model.state_dict()
    raw = {k.removeprefix("model."): val for k, val in archive["state_dict"].items() if k.startswith("model.")}
    need(len(raw) == 545 and list(raw) == list(eager) and all(
        a.shape == b.shape and a.dtype == b.dtype and (not a.is_floating_point() or bool(torch.isfinite(a).all()))
        for a, b in zip(raw.values(), eager.values(), strict=True)
    ), "Exact native ordered E24 online weights and buffers")
    named = dict(model.named_parameters())
    need([k for k in raw if k in named] == list(named), "Exact archived native parameter traversal")
    model.load_state_dict(raw, strict=True)
    need(all(torch.equal(v, raw[k]) for k, v in model.state_dict().items()), "Original online state readback differs")
    online_digest = tensor_dict_sha256(model.state_dict())
    native_callback, align, groups = original.native_training_bits()
    ema_callback = native_callback(decay=t.ema_decay, tau=t.ema_tau, use_buffers=True)
    ema = AveragedModel(model, use_buffers=True, multi_avg_fn=ema_callback._multi_avg_fn).eval()
    mods["e24_export"].strict_ema_load(ema.module, archive, torch)
    callback = archive["callbacks"][mods["boundary"].EMA_KEY]
    avg = callback["average_model_state_dict"]
    need(int(avg["n_averaged"].item()) == callback["latest_update_step"] == START_STEP, "Exact archived E24 EMA clock")
    ema.n_averaged.copy_(avg["n_averaged"])
    ema_digest = tensor_dict_sha256(ema.module.state_dict())
    need(online_digest != ema_digest, "Archived distinct online and EMA states must stay distinct")
    model.to(device).train()
    ema.to(device)
    cfg = _namespace_from_configs(m, t)
    optimizer = torch.optim.AdamW(groups(cfg, model), lr=t.lr, weight_decay=t.weight_decay,
                                  fused=False, **t.optimizer_kwargs)
    binding = install_optimizer_history(model, optimizer, archive, arm)
    nominal = [g["initial_lr"] * original.LR_SCALE for g in optimizer.param_groups]
    need(binding["applied_group_lrs"] == nominal, "Archived fixed E24 applied LR differs from P77 scout LR")
    criterion, _ = build_criterion_from_config(m, t)
    criterion.to(device).train()
    need(model.num_queries == 100 and model.group_detr == criterion.group_detr == 13,
         "Exact native RF-DETR training query configuration")
    info = dict(archive_start_sha256=ARCHIVE_SHA, starting_ema_clock=START_STEP,
                initial_online_sha256=online_digest, initial_ema_sha256=ema_digest, **binding)
    return native, model, criterion, optimizer, ema, align, info


def main():
    import torch
    from rfdetr.datasets.coco import ConvertCoco, make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models import criterion as native_api
    from rfdetr.models.heads.segmentation import point_sample
    from rfdetr.utilities.tensors import make_collate_fn

    need(Path(sys.executable).resolve(strict=True) == RUNTIME.resolve(strict=True)
         and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
         "Exact offline pinned Windows Python runtime")
    original = p77_source()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if sys.argv[1:] == ["--dependency-check"]:
        need(not torch.cuda.is_available(), "Dependency route is CPU only")
        profiles = {}
        for arm in ("S", "F"):
            initialized = initialize(arm, torch.device("cpu"), original)
            profiles[arm] = initialized[-1]
            del initialized
        need(all(profiles["S"][k] == profiles["F"][k] for k in (
            "initial_online_sha256", "initial_ema_sha256", "starting_ema_clock",
            "parameter_group_order_sha256", "applied_group_lrs")), "Equal initial conditions in CPU dependency check")
        print(json.dumps(dict(status="E24_OPTIMIZER_STATE_DEPENDENCIES_AND_INITIALIZATION_CPU_VERIFIED",
                              arms=profiles, GPU=False, FIT=False, training=False)), flush=True)
        return

    need(len(sys.argv) == 4 and sys.argv[1] in ("S", "F") and sys.argv[2] == "5201"
         and bool(re.fullmatch(r"[0-9a-f]{40}", sys.argv[3])), "Expected fixed arm, seed and prospective protocol")
    arm, seed, commit = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    out = Path.cwd()
    need(out.name == arm and out.parent.parent == D and not any(out.iterdir()),
         "Expected fresh private S/F arm directory under a dedicated pair")
    device = torch.device("cuda:0")
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(),
         "Expected isolated BF16 CUDA")
    free, total = torch.cuda.mem_get_info(device)
    need(free >= 6 * 1024**3, "Local CUDA resource availability")
    torch.cuda.set_per_process_memory_fraction(min(.35, 6 * 1024**3 / total), device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(seed)

    started = time.perf_counter()
    native, model, criterion, optimizer, ema, align, init = initialize(arm, device, original)
    relative_ema = ContinuationEMA(ema, START_STEP)
    material = original.read_pinned(original.MATERIAL / "SPEC.json", original.MATERIAL_SHA)
    need(native["fit_image_manifest"] == material["image_manifest"] and material["DEV"] is material["TEST"] is False,
         "Exact admitted FIT-only training roles")
    transform = make_coco_transforms_square_div_64("train", 576, multi_scale=False, expanded_scales=False,
        skip_random_resize=True, patch_size=12, num_windows=1, aug_config=None, scale_jitter=True,
        gpu_postprocess=False, keypoint_flip_pairs=None)
    convert = ConvertCoco(include_masks=True, include_keypoints=False, cat2label={1: 0}, num_keypoints_per_class=[])
    collate = make_collate_fn(block_size=12, pack=True, pad_targets_to=None)
    fit = original.doc(material["image_manifest"])
    items = fit["items"]
    need(len(items) == 741 and [x["dataset_index"] for x in items] == list(range(741)), "Full FIT order")
    gt = original.original_gt(material, items)
    annotations = defaultdict(list)
    for row in gt["annotations"]:
        annotations[row["image_id"]].append(row)
    aligned = align(items, effective_batch_size=4, world_size=1)
    need(len(aligned) == 744, "Exact native full accumulation tape alignment")
    tape = original.training_tape(seed, len(aligned))
    trace = FirstStepTrace(model, optimizer, arm, START_STEP, out / "FIRST_CLIPPED_GRAD_PRIVATE.pt")
    plan = sha256()
    for e, epoch in enumerate(tape):
        for pos, (slot, micro_seed) in enumerate(epoch):
            plan.update(f"{e}/{pos}/{slot}/{aligned[slot]['dataset_index']}/{micro_seed}\n".encode())

    def fetch(slot):
        item = aligned[slot]
        image_id = item["image_id"]
        image, scale = decode_pil_image_bytes(original.exact_bytes(item["image_ref"], 16 * 1024**2), draft_size=None)
        h, w = item["original_hw"]
        need(scale == (1., 1.) and image.mode == "RGB" and image.size == (w, h), "Exact FIT input bytes and geometry")
        image, target = convert(image, dict(image_id=image_id, annotations=annotations[image_id]))
        image, target = transform(image, target)
        need(target["masks"].dtype == torch.bool and target["masks"].shape[-2:] == (576, 576), "Native square target")
        samples, packed = collate([(image, target)])
        samples = samples.to(device, non_blocking=True)
        targets = packed.to_list(device, non_blocking=True)
        trace.observe_input(samples, targets, item["dataset_index"])
        return samples, targets, item["dataset_index"]

    meta = dict(arm=arm, seed=seed, epochs=2, protocol_commit=commit, planned_visit_sha256=plan.hexdigest(),
                original_native_criterion=True, original_native_point_sampler=True, fixed_archived_E24_lr=True,
                FIT=741, aligned_per_epoch=len(aligned), optimizer_accumulation=4, precision="BF16 autocast",
                DEV=False, TEST=False, **init)
    original.put(out / "STARTED_PRIVATE.json", meta)
    print(json.dumps(dict(event="E24_OPTIMIZER_STATE_ARM_STARTED", arm=arm, seed=seed, planned_updates=UPDATES,
                         planned_microbatches=1488)), flush=True)
    fit_result = original.train_epochs(model, criterion, native_api, point_sample, optimizer, relative_ema,
                                       fetch, tape, "A", seed, device)
    first_step = trace.result()
    history = Counter(int(v["step"].item()) for v in optimizer.state.values())
    expected_step = END_STEP if arm == "S" else UPDATES
    need(history == {expected_step: 543} and int(ema.n_averaged.item()) == END_STEP
         and fit_result["visit_sha256"] == meta["planned_visit_sha256"]
         and fit_result["updates"] == fit_result["ema_updates"] == UPDATES and fit_result["microbatches"] == 1488
         and [g["lr"] for g in optimizer.param_groups] == meta["applied_group_lrs"],
         "Exact completed tape, actual AdamW counts, historical EMA and fixed LR")
    meta.update(updates=UPDATES, microbatches=1488, ema_updates=UPDATES, ema_absolute_updates=END_STEP,
                visit_sha256=fit_result["visit_sha256"], adam_step_counts={str(k): v for k, v in history.items()})
    endpoint = out / "ENDPOINT.pt"
    with endpoint.open("xb") as stream:
        torch.save(dict(meta=meta, ema_state={k: v.detach().cpu() for k, v in ema.module.state_dict().items()}), stream)
        stream.flush()
        os.fsync(stream.fileno())
    result = dict(status="ACTUAL_E24_OPTIMIZER_STATE_ARM_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE",
                  meta=meta, result=fit_result, first_step=first_step, setup_and_train_seconds=time.perf_counter() - started,
                  endpoint_ref=file_ref(endpoint, 256*1024**2), AP=False, DEV=False, TEST=False)
    original.put(out / "RESULT.json", result)
    print(json.dumps(dict(event="E24_OPTIMIZER_STATE_ARM_TERMINAL", arm=arm, updates=UPDATES,
                         microbatches=1488, train_seconds=fit_result["train_seconds"])), flush=True)


if __name__ == "__main__":
    sys.addaudithook(offline)
    main()
