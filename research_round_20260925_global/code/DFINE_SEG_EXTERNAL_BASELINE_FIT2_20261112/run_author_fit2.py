"""Qualify two CPU updates through D-FINE-seg's native FIT-only author components."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
from sysconfig import get_path
from time import perf_counter
from unittest.mock import patch


STAGE = Path(__file__).resolve().parent
INTAKE = STAGE.parent
FIT = INTAKE.parent / "HRSID_FIT_NATIVE_COCO_LAYOUT_ACTUAL_PRIVATE_20261109_01"
INDEX = FIT / "train/_annotations.coco.json"
MANIFEST = FIT / "NATIVE_FIT_LAYOUT_MANIFEST_PRIVATE.json"
AUTHOR = INTAKE / "AUTHOR_SOURCE_VERIFIED"
WEIGHTS = INTAKE / "dfine_seg_s_coco.pt"
PRELOCK_SHA = "55605cb27e08068d8700112bacd0741631ea8aeecd1212ffd4921190e9f8838c"
SCOPE = dict(FIT=True, DEV=False, TEST=False, AP=False)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, expected, size=None):
    need(path.is_file() and path.resolve(strict=True) == path, "Exact canonical immutable input")
    data = path.read_bytes()
    need(sha256(data).hexdigest() == expected and (size is None or len(data) == size), "Exact original input digest")
    return data


def emit(data):
    print(json.dumps(data, sort_keys=True, allow_nan=False), flush=True)


def prepare():
    lock = json.loads(pinned(STAGE / "PRELOCK_SAFE.json", PRELOCK_SHA))
    need(all(lock[key] is value for key, value in SCOPE.items()) and lock["ordered_training_visits"] == 2,
         "Exact scope and bounded prospective FIT2 budget")
    cfg = AUTHOR / "config.yaml"
    pinned(cfg, lock["author_config_sha256"])
    pinned(WEIGHTS, lock["coco_checkpoint_sha256"], 48119167)
    image_index = json.loads(pinned(INDEX, lock["native_fit_index_sha256"], 30076663))
    provenance = json.loads(pinned(MANIFEST, lock["native_fit_manifest_sha256"]))
    ordered = sorted(image_index["images"], key=lambda row: row["id"])[:2]
    expected = lock["selected"]
    need(len(image_index["images"]) == 2937 and len(image_index["annotations"]) == 9260
         and [row["id"] for row in ordered] == [row["image_id"] for row in expected] == [0, 1],
         "Exact first two author FIT population IDs")
    images = []
    for image, target in zip(ordered, expected, strict=True):
        origin = next(row for row in provenance["images"] if row["image_id"] == image["id"])
        name = image["file_name"]
        need(Path(name).name == name == origin["native_basename"] and name.endswith(".png")
             and origin["encoded_sha256"] == target["png_sha256"] and origin["decoded_rgb_sha256"] == target["decoded_rgb_sha256"],
             "Exact original PNG provenance")
        path = FIT / "train" / name
        pinned(path, target["png_sha256"], target["png_bytes"])
        images.append(path)
    installed = Path(get_path("purelib")) / "dfine_seg"
    verified = 0
    for file in (AUTHOR / "dfine_seg").rglob("*"):
        if file.is_file() and file.suffix in {".py", ".yaml", ".yml"}:
            need((installed / file.relative_to(AUTHOR / "dfine_seg")).read_bytes() == file.read_bytes(),
                 "Exact installed author source and config bytes")
            verified += 1
    need(verified == 70, "Exact installed author source census")
    versions = {name: version(name) for name in ("dfine-seg", "torch", "torchvision", "albumentations")}
    need(tuple(versions[k] for k in ("dfine-seg", "torch", "torchvision")) ==
         ("0.7.0", "2.13.0+cpu", "0.28.0+cpu"), "Exact isolated CPU versions")
    return lock, cfg, images, versions, verified


def tensor_sha(tensor):
    return sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def state_sha(state):
    hashed = sha256()
    for name in sorted(state):
        value = state[name]
        meta = json.dumps([name, list(value.shape), str(value.dtype)], separators=(",", ":")).encode()
        hashed.update(meta + b"\n" + bytes.fromhex(tensor_sha(value)))
    return hashed.hexdigest()


def same_tensor(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and tensor_sha(a) == tensor_sha(b)


def train(lock, config, images, versions, verified):
    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "urllib.Request"}:
            raise PermissionError("Offline author FIT-only model qualification")
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0]))
            if path.is_absolute() and path.is_relative_to(FIT) and path not in (INDEX, *images):
                raise PermissionError("Only the verified train index and two named FIT images are admitted")

    sys.addaudithook(audit)
    import cv2
    from dfine_seg import load_model, read_image
    from dfine_seg.dl.dataset import CustomDataset, Loader, load_coco_split
    from dfine_seg.dl.train import ModelEMA, ckpt_meta, training_forward, training_optimizer
    from dfine_seg.dl.utils import set_seeds
    from dfine_seg.infer.torch_model import TorchModel
    from dfine_seg.model.dfine import build_loss, build_model
    from dfine_seg.model.utils import save_checkpoint, unwrap_checkpoint
    import numpy as np
    from omegaconf import OmegaConf
    import pandas as pd
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    need(torch.version.cuda is None and not torch.cuda.is_initialized(), "CPU-only torch")
    cfg = OmegaConf.load(config)
    cfg.model_name, cfg.task = "s", "segment"
    cfg.train.label_to_name = {0: "ship"}
    cfg.train.device, cfg.train.batch_size, cfg.train.num_workers = "cpu", 1, 0
    cfg.train.use_one_class, cfg.train.coco_dataset = True, True
    cfg.train.amp_enabled = False
    cfg.train.mosaic_augs.mosaic_prob = lock["mosaic_probability"]
    need(list(cfg.train.img_size) == lock["input_hw"] and not cfg.train.keep_ratio
         and cfg.train.augs.multiscale_prob == lock["multiscale_probability"]
         and cfg.train.use_muon and cfg.train.aux_optimizer == "adan" and cfg.train.use_scheduler
         and cfg.train.use_ema and cfg.train.seed == lock["seed"]
         and cfg.train.epochs == lock["scheduler"]["planned_epochs"], "Exact native stock recipe with prelocked FIT-only overrides")
    set_seeds(cfg.train.seed, cfg.train.cudnn_fixed)
    raw_entries, _ = load_coco_split(INDEX, use_one_class=True)
    by_name = {entry["file_name"]: entry for entry in raw_entries}
    entries = []
    for path, row in zip(images, lock["selected"], strict=True):
        entry = by_name[path.name]
        need(len(entry["targets"]) == len(entry["polys_abs"]) == row["annotation_count"], "Exact native FIT polygon counts")
        entries.append(dict(entry, file_name=str(path)))
    split = pd.DataFrame([entry["file_name"] for entry in entries])
    dataset = CustomDataset(tuple(cfg.train.img_size), STAGE, split, False, "train", cfg, coco_annotations=entries)
    dataset.transform.set_random_seed(cfg.train.seed)
    collate = Loader.__new__(Loader)
    collate.multiscale_prob = cfg.train.augs.multiscale_prob
    need(len(dataset) == 2 and dataset.mosaic_prob == 0, "Exactly the prelocked two visits without mosaic")

    fresh = build_model("s", 1, True, "cpu", img_size=(640, 640), task="segment")
    fresh_state = fresh.state_dict()
    set_seeds(cfg.train.seed, cfg.train.cudnn_fixed)
    model = build_model("s", 1, True, "cpu", img_size=(640, 640), task="segment",
                        pretrained_model_path=str(WEIGHTS))
    initial_state = model.state_dict()
    official, _ = unwrap_checkpoint(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    reused, fresh_only = [], []
    for name, value in initial_state.items():
        from_coco = name in official and official[name].shape == value.shape
        if from_coco:
            need(same_tensor(value, official[name]), "Author matching tensor must equal actual COCO source")
            reused.append(name)
        else:
            need(same_tensor(value, fresh_state[name]), "Unmatched class tensor must equal seeded fresh initialization")
            fresh_only.append(name)
    need(bool(reused) and bool(fresh_only)
         and all("score_head" in name or "denoising_class_embed" in name for name in fresh_only)
         and any("mask_decoder" in name for name in reused)
         and any("backbone" in name for name in reused), "Native COCO mask/backbone reuse and class-only reinit")
    initial_digest = state_sha(initial_state)
    del fresh, fresh_state, official, initial_state, value

    model.train()
    criterion = build_loss("s", 1, cfg.train.label_smoothing, True, task="segment", img_size=(640,640))
    optimizer = training_optimizer(model, cfg)
    ema = ModelEMA(model, cfg.train.ema_momentum)
    need(state_sha(ema.model.state_dict()) == initial_digest and len(optimizer.param_groups) == 5
         and all(g.get("aux_optimizer") == "adan" for g in optimizer.param_groups[:4])
         and optimizer.param_groups[4]["use_muon"], "Exact native optimizer groups and EMA initial state")
    peak = cfg.train.base_lr * 2 * cfg.train.adan_lr_mult
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[peak] * 4 + [cfg.train.base_lr * 20],
        epochs=cfg.train.epochs, steps_per_epoch=lock["scheduler"]["nominal_fit_steps_per_epoch"],
        pct_start=cfg.train.cycler_pct_start, cycle_momentum=False,
    )
    need(scheduler.total_steps == lock["scheduler"]["total_steps"], "Exact prelocked native scheduler clock")
    forward_returns, optimizer_returns, step_rows = [], [], []
    model.register_forward_hook(lambda *_: forward_returns.append(1))
    optimizer.register_step_post_hook(lambda *_: optimizer_returns.append(1))
    mask_params = [(name, value) for name, value in model.named_parameters()
                   if "mask_decoder" in name and value.requires_grad and value.ndim >= 2]
    tracked_name, tracked = mask_params[0]
    ema_weight = dict(ema.model.named_parameters())[tracked_name]
    need(tracked.dtype == ema_weight.dtype == torch.float32, "FP32 tracked original mask weight")
    emit(dict(status="DFINE_HRSID_FIT2_REAL_TRAIN_START", initial_reused=len(reused), initial_seeded=len(fresh_only),
              native_train_records=[len(e["targets"]) for e in entries], **SCOPE))

    optimizer.zero_grad()
    for step, path in enumerate(images, start=1):
        inputs, targets, paths = collate.train_collate_fn([dataset[step-1]])
        target = targets[0]
        need(inputs.shape == (1, 3, 640, 640) and inputs.dtype == torch.float32 and paths == [path]
             and target["labels"].numel() > 0 and target["labels"].eq(0).all().item()
             and target["masks"].shape == (target["labels"].numel(), 640, 640)
             and target["masks"].flatten(1).any(1).all().item(), "Native class, mask and ordered FIT targets")
        before = tensor_sha(tracked)
        before_ema = ema_weight.detach().clone()
        lrs = [group["lr"] for group in optimizer.param_groups]
        losses = training_forward(model, criterion, None, inputs, targets, enabled=False, dtype=torch.bfloat16)
        need("loss_mask_bce" in losses and "loss_mask_dice" in losses
             and all(v.ndim == 0 and torch.isfinite(v).item() for v in losses.values()),
             "Finite author detection, segmentation, auxiliary and denoising losses")
        loss = sum(losses.values())
        loss.backward()
        need(tracked.grad is not None and torch.isfinite(tracked.grad).all().item()
             and bool(tracked.grad.abs().max().item()), "Actual nonzero finite mask-head gradient")
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_max_norm).item())
        need(math.isfinite(grad) and grad > 0, "Finite native gradient clip witness")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        after = tensor_sha(tracked)
        need(len(optimizer_returns) == len(forward_returns) == step and optimizer.state[tracked]["step"] == step
             and scheduler.last_epoch == step and before != after, "Actual original optimizer return and per-mask parameter clock")
        momentum = ema.ema_scheduler(step)
        ema.update(step, model)
        expected_ema = before_ema.mul_(momentum).add_((1.0-momentum)*tracked.detach())
        ema_sha = tensor_sha(ema_weight)
        need(same_tensor(expected_ema, ema_weight), "Exact native EMA weight arithmetic")
        row = dict(step=step, image_id=lock["selected"][step-1]["image_id"], retained_targets=int(target["labels"].numel()),
                   native_adam_aux_clock=optimizer.state[tracked]["step"], scheduler_return=scheduler.last_epoch,
                   optimizer_returns=len(optimizer_returns), ema_update=step, loss=float(loss.item()),
                   mask_bce=float(losses["loss_mask_bce"].item()), mask_dice=float(losses["loss_mask_dice"].item()),
                   gradient_norm_before_clip=grad, lrs=lrs, tracked_mask_before_sha256=before,
                   tracked_mask_after_sha256=after, tracked_ema_after_sha256=ema_sha)
        step_rows.append(row)
        emit(dict(status="DFINE_HRSID_FIT2_REAL_UPDATE_RETURN", **row))
        del losses, loss, inputs, targets, target, before_ema, expected_ema

    optimizer_class = type(optimizer).__name__
    archive = STAGE / "FIT2_AFTER_STEP2_EMA_PRIVATE.pt"
    need(not archive.exists(), "Fresh native archive only")
    final_online = state_sha(model.state_dict())
    final_ema = state_sha(ema.model.state_dict())
    need(final_online != initial_digest and final_ema != initial_digest and final_ema != final_online,
         "Trained native online and EMA states changed")
    save_checkpoint(archive, ema.model.state_dict(), ckpt_meta(cfg))
    archive_raw = archive.read_bytes()
    archive_sha = sha256(archive_raw).hexdigest()
    del archive_raw, model, optimizer, scheduler, criterion, ema, tracked, ema_weight, mask_params

    bgr = read_image(images[0])
    need(bgr.shape == (800, 800, 3) and sha256(np.ascontiguousarray(bgr[..., ::-1]).tobytes()).hexdigest()
         == lock["selected"][0]["decoded_rgb_sha256"], "Exact original post-training FIT image")
    native_predict = TorchModel._predict
    inference_returns = []

    def observe_predict(self, inputs):
        value = native_predict(self, inputs)
        inference_returns.append(list(inputs.shape))
        return value

    with patch.object(TorchModel, "_predict", observe_predict):
        wrapper = load_model(archive, task="segment", device="cpu")
        need(wrapper.n_outputs == 1 and wrapper.names == {0:"ship"} and len(inference_returns) == 1,
             "Exact updated one-class archive and one author synthetic warmup")
        need(tensor_sha(dict(wrapper.model.named_parameters())[tracked_name]) == step_rows[-1]["tracked_ema_after_sha256"],
             "Native inference actually used the twice-updated EMA mask tensor")
        output = wrapper(bgr, conf_thresh=lock["posttrain_confidence"])[0]
    masks, scores, boxes, labels = (output[name] for name in ("masks","scores","boxes","labels"))
    rows = len(scores)
    need(inference_returns == [[1,3,640,640]]*2 and len(forward_returns) == len(optimizer_returns) == 2
         and rows > 0 and masks.shape == (rows, 800, 800) and boxes.shape == (rows, 4)
         and masks.dtype == torch.uint8 and int(masks.max()) <= 1 and labels.eq(0).all().item()
         and torch.isfinite(scores).all().item() and torch.isfinite(boxes).all().item()
         and not torch.cuda.is_initialized(), "Native one-class original-size binary masks from exact updated archive")
    report = dict(status="DFINE_SEG_S_HRSID_FIT2_NATIVE_CPU_QUAL_COMPLETE_V1", prelock_sha256=PRELOCK_SHA,
                  source_commit=lock["source_commit"], installed_exact_author_files=verified, versions=versions,
                  seed=cfg.train.seed, source_checkpoint_sha256=lock["coco_checkpoint_sha256"], initial_state_sha256=initial_digest,
                  initialized_from_coco_tensors=len(reused), seeded_new_tensors=len(fresh_only),
                  seeded_new_tensor_names=fresh_only, reused_roots=dict(Counter(name.split(".")[0] for name in reused)),
                  initial_mask_tensor_count=sum("mask_decoder" in name or "mask_head" in name for name in reused),
                  optimizer_class=optimizer_class,
                  actual_training_forwards=len(forward_returns), optimizer_updates=len(optimizer_returns), ema_updates=len(step_rows),
                  steps=step_rows, tracked_mask_name=tracked_name, online_state_sha256=final_online, ema_state_sha256=final_ema,
                  updated_checkpoint_sha256=archive_sha, updated_checkpoint_bytes=archive.stat().st_size,
                  posttrain_native_masks_shape=list(masks.shape), posttrain_mask_dtype="uint8",
                  posttrain_nonempty_masks=int(masks.flatten(1).any(1).sum().item()),
                  posttrain_score_range=[scores.min().item(), scores.max().item()],
                  posttrain_real_fit_forward_calls=1, author_synthetic_warmup_calls=1, posttrain_classes=1,
                  posttrain_confidence=lock["posttrain_confidence"], mask_threshold=wrapper.mask_threshold,
                  nms_iou=wrapper.nms_iou_thresh, training_visits=2, fit_annotations_opened=True,
                  no_validation_loader=True, optimizer_updated=True, **SCOPE)
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with (STAGE/"RESULT_SAFE.json").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    emit(dict(status=report["status"], result_sha256=sha256(raw).hexdigest(), mask_shape=report["posttrain_native_masks_shape"],
              nonempty=report["posttrain_nonempty_masks"], **SCOPE))


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--preflight", action="store_true")
    args = cli.parse_args()
    need(sys.version_info[:3] == (3,11,15) and sys.flags.isolated == sys.flags.utf8_mode == 1
         and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "Exact isolated CPU process")
    items = prepare()
    if args.preflight:
        need("torch" not in sys.modules, "Preflight imports no Torch")
        emit(dict(status="DFINE_SEG_S_HRSID_FIT2_CPU_PREFLIGHT_PASSED", prelock_sha256=PRELOCK_SHA,
                  exact_fit_images=2, selected_annotated_targets=18, installed_exact_author_files=items[-1], **SCOPE))
    else:
        train(*items)


if __name__ == "__main__":
    main()
