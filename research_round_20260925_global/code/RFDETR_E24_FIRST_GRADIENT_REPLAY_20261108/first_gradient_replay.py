"""Trace one native F/S accumulation window and interrupt it before the first AdamW update."""
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from hashlib import sha256
import argparse
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType
from unittest.mock import patch

import torch


HERE = Path(__file__).resolve().parent
D = HERE.parent
PAIR = D / "RFDETR_E24_OPTIMIZER_PRESERVATION_TRAIN_CANDIDATE_20261108_01"
PINNED_PAIR = {
    "run_state_arm.py": "f870227ecce5da591c1bdc9b975591fff720b36540e0f01818277105d3c2176c",
    "state_preservation.py": "8dda41131a2930aacf00c51dbe8cfa4817c0a3cd4b02c6bb606948db77341678",
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


class BeforeUpdate(Exception):
    """Private sentinel raised before the optimizer's first actual step."""


@contextmanager
def stop_before_update(optimizer):
    def interrupt(*_):
        raise BeforeUpdate

    handle = optimizer.register_step_pre_hook(interrupt)
    try:
        try:
            yield
        except BeforeUpdate:
            pass
        else:
            raise ValueError("Native loop did not reach the first optimizer pre-step")
    finally:
        handle.remove()


def load_pair():
    for file, expected in PINNED_PAIR.items():
        need(sha256((PAIR / file).read_bytes()).hexdigest() == expected, "Exact original F/S source required")
    path = PAIR / "run_state_arm.py"
    m = ModuleType("_pinned_fs_replay_source")
    m.__file__ = str(path)
    sys.modules[m.__name__] = m
    exec(compile(path.read_bytes(), str(path), "exec"), m.__dict__)
    return m


def tree_hash(value):
    """Hash tensor content, structure, and dtypes without executing model operations."""
    digest = sha256()

    def visit(key, item):
        if torch.is_tensor(item):
            cpu = item.detach().cpu().contiguous()
            digest.update(json.dumps([key, list(cpu.shape), str(cpu.dtype)]).encode() + b"\n")
            digest.update(memoryview(cpu.reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(item, dict):
            digest.update(json.dumps([key, "dict", sorted(item)]).encode() + b"\n")
            for k in sorted(item):
                visit(k, item[k])
        elif isinstance(item, (tuple, list)):
            digest.update(json.dumps([key, type(item).__name__, len(item)]).encode() + b"\n")
            for i, sub in enumerate(item):
                visit(i, sub)
        elif hasattr(item, "tensors") and hasattr(item, "mask"):
            visit("nested_tensor", (item.tensors, item.mask))
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(json.dumps([key, item], allow_nan=False).encode() + b"\n")
        else:
            raise TypeError("Unexpected native trace object: " + type(item).__name__)

    visit("root", value)
    return digest.hexdigest()


def rng_digest(device):
    return dict(cpu=tree_hash(torch.random.get_rng_state()), cuda=tree_hash(torch.cuda.get_rng_state(device)))


def gradient_parts(model):
    gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    result = {}
    for key in ("backbone", "rest"):
        selected = {name: value for name, value in gradients.items() if name.startswith("backbone.") == (key == "backbone")}
        result[key] = dict(tensors=len(selected), sha256=tree_hash(selected))
    patch = {name: value for name, value in gradients.items()
             if name.endswith("patch_embeddings.projection.weight")}
    result["patch_projection_weight"] = dict(tensors=len(patch), sha256=tree_hash(patch))
    return result


def trace_first_window(model, criterion, optimizer, native_api, device, original, ema, fetch, tape):
    rows = []
    clip = {}
    stock_sampler = native_api.get_uncertain_point_coords_with_randomness
    stock_masks = criterion.loss_masks
    stock_clip = torch.nn.utils.clip_grad_norm_
    projections = [module for name, module in model.named_modules()
                   if name.endswith("patch_embeddings.projection") and isinstance(module, torch.nn.Conv2d)]
    need(len(projections) == 1 and tuple(projections[0].weight.shape) == (384, 3, 12, 12),
         "Exact native DINO patch projection")
    projection, = projections

    def load(slot):
        row = dict(slot=slot, rng_before_fetch=rng_digest(device), matcher=[], point_coords=[])
        if rows:
            rows[-1]["cumulative_gradient"] = gradient_parts(model)
        samples, targets, index = fetch(slot)
        row.update(index=index, input_sha256=tree_hash((samples, targets)), rng_after_fetch=rng_digest(device))
        rows.append(row)
        return samples, targets, index

    def record_output_gradient(row, key, grad):
        row[key] = tree_hash(grad)

    def patch_output(_module, _args, output):
        row = rows[-1]
        row["patch_projection_output_sha256"] = tree_hash(output)
        need(output.requires_grad, "Trainable patch projection must expose its output gradient")
        output.register_hook(lambda grad: record_output_gradient(row, "patch_projection_grad_out_sha256", grad))

    def backbone(_module, _args, output):
        row = rows[-1]
        row["backbone_output_sha256"] = tree_hash(output)
        feature = output[0][0].tensors
        row["backbone_first_feature_has_grad"] = feature.requires_grad
        if feature.requires_grad:
            feature.register_hook(lambda grad: record_output_gradient(row, "backbone_first_feature_grad_sha256", grad))

    def model_output(_module, _args, output):
        rows[-1].update(forward_outputs={k: tree_hash(v) for k, v in sorted(output.items())},
                        rng_after_forward=rng_digest(device))

    def masks(outputs, targets, indices, num_boxes):
        # Trace indices where the stock loss consumes them. A matcher forward hook changes its fast path.
        rows[-1]["matcher"].append(dict(sha256=tree_hash(indices), pairs=sum(len(x) for x, _ in indices)))
        return stock_masks(outputs, targets, indices, num_boxes)

    def sampler(*args, **kwargs):
        coords = stock_sampler(*args, **kwargs)
        rows[-1]["point_coords"].append(dict(sha256=tree_hash(coords), shape=list(coords.shape)))
        return coords

    def losses(_module, _args, output):
        rows[-1].update(losses={k: float(v.detach().cpu()) for k, v in sorted(output.items())},
                        rng_after_loss=rng_digest(device))

    def clip_gradients(*args, **kwargs):
        need(len(rows) == 4 and not clip, "Exactly one original native clipping boundary")
        rows[-1]["cumulative_gradient"] = gradient_parts(model)
        clip["preclip_gradient"] = rows[-1]["cumulative_gradient"]
        norm = stock_clip(*args, **kwargs)
        clip["preclip_global_norm"] = float(norm.detach().cpu())
        clip["postclip_gradient"] = gradient_parts(model)
        clip["rng_after_backward_and_clip"] = rng_digest(device)
        return norm

    with ExitStack() as stack:
        for layer, observer in (
            (projection, patch_output), (model.backbone, backbone), (model, model_output), (criterion, losses)
        ):
            stack.callback(layer.register_forward_hook(observer).remove)
        stack.enter_context(patch.object(criterion, "loss_masks", masks))
        stack.enter_context(patch.object(native_api, "get_uncertain_point_coords_with_randomness", sampler))
        stack.enter_context(patch.object(torch.nn.utils, "clip_grad_norm_", clip_gradients))
        stack.enter_context(stop_before_update(optimizer))
        original.train_epochs(model, criterion, native_api, None, optimizer, ema, load, [tape], "A", 5201, device)
    need(len(rows) == 4 and clip and all(
        set(("backbone_output_sha256", "forward_outputs", "losses", "cumulative_gradient")) <= set(r)
        and "patch_projection_output_sha256" in r and "patch_projection_grad_out_sha256" in r
        and len(r["matcher"]) == 5 for r in rows
    ), "Incomplete native trace: four forwards, five layer matches per microbatch, and one clip required")
    return rows, clip


def adam_steps(optimizer):
    return dict(sorted(Counter(int(x["step"].item()) for x in optimizer.state.values()).items()))


def run(args):
    source = load_pair()
    sys.addaudithook(source.offline)
    from rfdetr.datasets.coco import ConvertCoco, make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models import criterion as native_api
    from rfdetr.utilities.tensors import make_collate_fn

    need(Path(sys.executable).resolve(strict=True) == source.RUNTIME.resolve(strict=True)
         and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
         "Exact offline pinned Windows Python runtime")
    out = args.output_dir
    need(out.resolve() == out and out.parent.parent == D and out.name == args.attempt and not out.exists(),
         "Exact fresh direct child for this attempt")
    device = torch.device("cuda:0")
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(),
         "Expected isolated BF16 CUDA")
    free, total = torch.cuda.mem_get_info(device)
    need(free >= 6 * 1024**3, "Local CUDA resource availability")
    torch.cuda.set_per_process_memory_fraction(min(.35, 6 * 1024**3 / total), device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(5201)
    original = source.p77_source()
    arm = args.attempt[0]
    native, model, criterion, optimizer, ema, align, init = source.initialize(arm, device, original)
    params_before = source.tensor_dict_sha256(dict(model.named_parameters()))
    steps_before = adam_steps(optimizer)

    material = original.read_pinned(original.MATERIAL / "SPEC.json", original.MATERIAL_SHA)
    need(native["fit_image_manifest"] == material["image_manifest"] and material["DEV"] is material["TEST"] is False,
         "Exact admitted FIT-only roles")
    transform = make_coco_transforms_square_div_64("train", 576, multi_scale=False, expanded_scales=False,
        skip_random_resize=True, patch_size=12, num_windows=1, aug_config=None, scale_jitter=True,
        gpu_postprocess=False, keypoint_flip_pairs=None)
    convert = ConvertCoco(include_masks=True, include_keypoints=False, cat2label={1: 0}, num_keypoints_per_class=[])
    collate = make_collate_fn(block_size=12, pack=True, pad_targets_to=None)
    items = original.doc(material["image_manifest"])["items"]
    need(len(items) == 741 and [x["dataset_index"] for x in items] == list(range(741)), "Full FIT order")
    annotations = defaultdict(list)
    for row in original.original_gt(material, items)["annotations"]:
        annotations[row["image_id"]].append(row)
    aligned = align(items, effective_batch_size=4, world_size=1)
    need(len(aligned) == 744, "Exact native full accumulation tape alignment")
    tape = original.training_tape(5201, len(aligned))[0][:4]

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
        return samples.to(device, non_blocking=True), packed.to_list(device, non_blocking=True), item["dataset_index"]

    out.mkdir()
    rows, clip = trace_first_window(model, criterion, optimizer, native_api, device, original,
        source.ContinuationEMA(ema, source.START_STEP), fetch, tape)
    torch.cuda.synchronize(device)
    need(int(ema.n_averaged.item()) == source.START_STEP and adam_steps(optimizer) == steps_before
         and source.tensor_dict_sha256(dict(model.named_parameters())) == params_before,
         "Replay mutated parameters, AdamW clock, or EMA clock")
    gradients = {name: p.grad.detach().cpu() for name, p in model.named_parameters() if p.grad is not None}
    need(gradients and all(bool(torch.isfinite(value).all()) for value in gradients.values()), "Finite clipped gradient")
    grad_path = out / "FIRST_CLIPPED_GRAD_PRIVATE.pt"
    with grad_path.open("xb") as stream:
        torch.save(dict(schema="RFDETR_FIRST_CLIPPED_GRAD_V1", tensors=gradients), stream)
        stream.flush()
        os.fsync(stream.fileno())
    result = dict(status="RFDETR_FIRST_GRADIENT_REPLAY_COMPLETE_NO_UPDATE", attempt=args.attempt, arm=arm, seed=5201,
        protocol_commit=args.protocol_commit, init=init, tape=tape, microbatches=rows, clip=clip,
        original_training_mode=model.training, float32_matmul_precision=torch.get_float32_matmul_precision(),
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        optimizer_updates=0, ema_updates=0, DEV=False, TEST=False, AP=False,
        clipped_gradient_sha256=source.tensor_dict_sha256(gradients), gradient_ref=source.file_ref(grad_path, 256*1024**2),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    original.put(out / "RESULT.json", result)
    print(json.dumps(dict(status=result["status"], attempt=args.attempt, microbatches=len(rows),
                          preclip_global_norm=clip["preclip_global_norm"], optimizer_updates=0, ema_updates=0)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", choices=("F0", "S0", "F1", "S1"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-commit", required=True)
    args = parser.parse_args()
    need(bool(re.fullmatch(r"[0-9a-f]{40}", args.protocol_commit)), "Fixed protocol commit required")
    run(args)


if __name__ == "__main__":
    main()
