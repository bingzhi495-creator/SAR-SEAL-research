"""Capture the native final-head stock matching path on a locked SSDD FIT tranche."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import replace
from hashlib import sha256
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
import json
import os
from pathlib import Path
import random
import re
import sys
from time import perf_counter
from unittest.mock import patch
from zipfile import ZipFile


ARCHIVE_SHA = "30d6b22104122e6a38f8ec7127e30ab1ae46117d898a1131c202bc5b673530b5"
RAW_SHA = "5da981aade9d676a601c1b6da5c362c2855438e45d8d7deb5c41368d7008f4de"
RECIPE_SHA = "ba9508d9acab5ff68b10db8a1aff5d9001868d6dda00e39f299d6f5cc1b8011b"
FIT_INDEX_SHA = "6dd6dd35f8418285696a224eab0ada6538224a0603c9eeb14c5ae98ea9fbcf95"
FIT_BUNDLE_SHA = "1c8f9b75e79fbc02c983ccd283a70082fcd5ab0d6538c777e8b286dbfdb78100"
FIT_GT_SHA = "62966857c78cea1e3b060700c62a7b896942686c2e674d3eb09f516a92250bc1"
WIDTH, GROUPS, RESOLUTION, K = 100, 13, 576, 1296
CAP = 6 * 1024**3
NATIVE_SOURCE = {
    "rfdetr.models.lwdetr": "248faf15d673bc18f51f60bb55245f64eee40eb9c8adacea398aebdadad2243b",
    "rfdetr.models.transformer": "279c0af7e999ff4ce58845c94c87bd9211356d62cb1eeaeb626fe2680fbb8f4d",
    "rfdetr.models.matcher": "2521d58fd5c3270f8f352c14ae299364347bd27792e18696de8bb20c637765f6",
    "rfdetr.models.heads.segmentation": "5de847861e2bd2fab8853f442e263f4e9c0eb75c5b3fbc6a03bb25e9c7115e1e",
    "rfdetr.utilities.box_ops": "6e2eff801fdbefdd8676aad3f05721888304c8b18288b01c7f78a382adc35cdd",
    "rfdetr.datasets.coco": "be64b20a178d9cde41db69e01e17e1296e36eee791e837294c3f930ae56f03b9",
    "rfdetr.datasets.io_utils": "0e8d54c2867f8eec28aa25f650501a91bac8116ca2f7e40e9ddd4cd1332bcd65",
    "rfdetr.datasets.transforms": "a4070434e1cfcb50040f1c84abddf13e37fb8f93953c76c035c32058f8b6d2a1",
    "rfdetr.datasets._torchvision": "03c12e881f733d4f6b469f784187e461753b8775b7e35e57c1d8a6c35c46ef56",
    "rfdetr.utilities.tensors": "122f44a151db786ef4845c5d89e5cb2209624651d3638f007e3c0867f7312827",
}
VERSIONS = {"rfdetr": "1.12.0", "torch": "2.10.0+cu130", "torchvision": "0.25.0+cu130",
            "pycocotools": "2.0.11", "numpy": "1.26.4"}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(data):
    return sha256(data).hexdigest()


def offline(event, args):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise RuntimeError("The bounded FIT capture is offline")


def identity(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def exact_bytes(path, size, expected):
    need(path.stat().st_size == size, f"Exact input size for {path.name}")
    data = path.read_bytes()
    need(len(data) == size and sha(data) == expected, f"Exact closed input for {path.name}")
    return data


def file_digest(stream):
    checksum = sha256()
    for block in iter(lambda: stream.read(1024**2), b""):
        checksum.update(block)
    return checksum.hexdigest()


def state_digest(tensors):
    import torch
    out = sha256()
    for name, value in tensors.items():
        cpu = value.detach().cpu().contiguous()
        out.update(json.dumps([name, list(cpu.shape), str(cpu.dtype)]).encode() + b"\n")
        out.update(memoryview(cpu.reshape(-1).view(torch.uint8).numpy()))
    return out.hexdigest()


def view_plan(items, tranche):
    distinct = {}
    for item in sorted(items, key=lambda row: row["dataset_index"]):
        if item["partition"] == "head_fit":
            distinct.setdefault(item["image_sha256"], item)
    ordered = sorted(distinct, key=lambda digest: sha(("sampling-coverage-v1:" + digest).encode()))
    need(len(ordered) >= 96, "Exact P68 head-FIT content order availability")
    if tranche == "smoke":
        return [(distinct[key], repeat) for key in ordered[:12] for repeat in range(2)]
    need(tranche == "cohort32", "Predeclared mechanism tranche")
    return [(distinct[key], 0) for key in ordered[:32]]


def capture_final(matcher, api, outputs, targets, width=WIDTH, groups=GROUPS):
    import torch

    n = targets[0]["labels"].numel()
    need(len(targets) == 1 and 0 < n <= width and "valid" not in targets[0]
         and outputs["pred_logits"].shape[:2] == (1, width * groups)
         and isinstance(outputs["pred_masks"], dict), "Single-image sparse final segmentation head")
    trace = {}

    def observe(key, function):
        def call(*args, **kwargs):
            value = function(*args, **kwargs)
            need(key not in trace, "Each exact native component must be used once")
            trace[key] = value
            return value
        return call

    stock_sample = api.point_sample
    stock_mask_ce = api.batch_sigmoid_ce_loss

    def point_sample(values, coordinates, **kwargs):
        if kwargs.get("mode") == "nearest":
            need("draw" in trace and torch.equal(coordinates, trace["draw"].repeat(n, 1, 1)),
                 "Native GT uses the exact shared prediction draw")
        else:
            need("draw" not in trace and coordinates.shape[0] == 1, "One shared stock uniform draw")
            trace["draw"] = coordinates
        return stock_sample(values, coordinates, **kwargs)

    def mask_ce(logits, labels):
        trace["native_pred_at_draw"] = logits
        trace["native_target_at_draw"] = labels
        return stock_mask_ce(logits, labels)

    stock_assign = matcher._assign_compact_cost_matrix

    def assign(cost, sizes, group_detr):
        need(sizes == [n] and group_detr == groups and "native_total" not in trace,
             "Actual native grouped final solver call")
        trace["native_total"] = cost
        return stock_assign(cost, sizes, group_detr)

    with ExitStack() as stack:
        def wrap(obj, name, key):
            stack.enter_context(patch.object(obj, name, observe(key, getattr(obj, name))))

        wrap(matcher, "_focal_classification_cost", "class")
        wrap(api, "generalized_box_iou", "giou")
        wrap(matcher, "_compute_mask_costs", "mask")
        if outputs["pred_boxes"].dtype == targets[0]["boxes"].dtype:
            resolve = matcher._resolve_l1_cost
            stack.enter_context(patch.object(matcher, "_resolve_l1_cost", lambda: observe("bbox", resolve())))
        else:
            wrap(api, "pairwise_box_l1_cost", "bbox")
        stack.enter_context(patch.object(api, "point_sample", point_sample))
        stack.enter_context(patch.object(api, "batch_sigmoid_ce_loss", mask_ce))
        stack.enter_context(patch.object(matcher, "_assign_compact_cost_matrix", assign))
        (src, dst), = matcher(outputs, targets, group_detr=groups)
    need(set(trace) == {"class", "giou", "bbox", "mask", "draw", "native_total",
                        "native_pred_at_draw", "native_target_at_draw"}, "Complete stock native trace")
    mask_ce_cost, mask_dice_cost = trace["mask"]
    nonmask = matcher.cost_bbox * trace["bbox"] + matcher.cost_class * trace["class"] + matcher.cost_giou * -trace["giou"]
    reconstructed = nonmask + matcher.cost_mask_ce * mask_ce_cost + matcher.cost_mask_dice * mask_dice_cost
    native_total = trace["native_total"]
    need(reconstructed.shape == (groups * width, n) and bool(torch.isfinite(reconstructed).all())
         and torch.equal(reconstructed.float().cpu(), native_total), "Exact stock additive order and native solver matrix")
    need(src.shape == dst.shape == (n * groups,) and src.unique().numel() == n * groups
         and bool(((0 <= src) & (src < groups * width)).all()) and bool(((0 <= dst) & (dst < n)).all()),
         "Every actual matched query and target is bounded")
    need(all(dst[src // width == group].unique().numel() == n for group in range(groups)),
         "Every group solved a distinct one-to-one stock target assignment")
    masks = targets[0]["masks"]
    area = masks.flatten(1).sum(1)
    need(bool((area > 0).all()), "Only real nonempty augmented GT masks")
    return dict(
        nonmask=nonmask.float().cpu(), native_total=native_total,
        native_mask_ce=mask_ce_cost.float().cpu(), native_mask_dice=mask_dice_cost.float().cpu(),
        native_src=src.cpu(), native_dst=dst.cpu(), uniform_draw=trace["draw"][0].cpu(),
        native_pred_at_draw=trace["native_pred_at_draw"].float().cpu(),
        native_target_at_draw=trace["native_target_at_draw"].float().cpu(),
        sparse={key: tensor.detach().cpu() for key, tensor in outputs["pred_masks"].items()},
        target_masks=masks.cpu(), target_boxes=targets[0]["boxes"].cpu(), target_labels=targets[0]["labels"].cpu(),
        is_small=(area < 1024).cpu(), target_area_pixels=area.cpu(),
        mask_weights=(float(matcher.cost_mask_ce), float(matcher.cost_mask_dice)),
        native_cost_dtypes=dict(nonmask=str(nonmask.dtype), ce=str(mask_ce_cost.dtype), dice=str(mask_dice_cost.dtype)),
        width=width, groups=groups)


def make_sampler(packet, device, *, native_cost=False):
    """Replay native sparse point samples, or native BCE and Dice at those coordinates."""
    import torch
    from rfdetr.models.heads.segmentation import point_sample
    from rfdetr.utilities.box_ops import batch_dice_loss, batch_sigmoid_ce_loss

    device = torch.device(device)
    sparse = {key: value.to(device) for key, value in packet["sparse"].items()}
    masks = packet["target_masks"].to(device)
    n = masks.shape[0]

    def sample(points):
        xy = torch.as_tensor(points, dtype=torch.float32, device=device)
        need(xy.ndim == 2 and xy.shape[1] == 2 and xy.shape[0] > 0, "Nonempty normalized [x, y] points")
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            features = point_sample(sparse["spatial_features"], xy.unsqueeze(0), align_corners=False)
            logits = torch.einsum("bcp,bnc->bnp", features, sparse["query_features"]) + sparse["bias"]
            labels = point_sample(masks.to(logits.dtype).unsqueeze(1), xy.unsqueeze(0).repeat(n, 1, 1),
                                  align_corners=False, mode="nearest").squeeze(1)
            if native_cost:
                ce = batch_sigmoid_ce_loss(logits[0], labels)
                dice = batch_dice_loss(logits[0], labels)
                return ce.float().cpu().numpy(), dice.float().cpu().numpy()
        return logits[0].float().cpu().numpy(), labels.float().cpu().numpy()

    return sample


def initialize(args, torch, device):
    import yaml
    from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig
    from rfdetr.models._defaults import MODEL_DEFAULTS
    from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config

    recipe = yaml.safe_load(exact_bytes(args.recipe, 1815, RECIPE_SHA))
    m = RFDETRSegNanoConfig(**recipe["model"]["model_config"]["init_args"])
    t = SegmentationTrainConfig(**recipe["model"]["train_config"]["init_args"])
    need(m.num_queries == WIDTH and m.group_detr == GROUPS and m.resolution == RESOLUTION
         and m.patch_size == 12 and m.num_windows == 1 and t.mask_point_sample_ratio == 16 and t.scale_jitter,
         "Exact stock E24 configuration")
    path = args.archive
    start = path.stat()
    with path.open("rb") as stream:
        need(identity(start) == identity(os.fstat(stream.fileno())) and start.st_size == 543050422
             and file_digest(stream) == ARCHIVE_SHA, "Exact independently closed native raw E24 archive")
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
        need(identity(start) == identity(os.fstat(stream.fileno())) == identity(path.stat()), "Closed E24 archive unchanged")
    need(checkpoint["epoch"] == 23 and checkpoint["global_step"] == 4464, "Exact SSDD E24 raw clock")
    model = build_model_from_config(m, t, defaults=replace(MODEL_DEFAULTS, force_no_pretrain=True))
    raw = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items() if key.startswith("model.")}
    need(len(raw) == 545 and list(raw) == list(model.state_dict()), "Exact native raw online state layout")
    model.load_state_dict(raw, strict=True)
    need(state_digest(model.state_dict()) == RAW_SHA, "Exact independent archived E24 online weights")
    del checkpoint, raw
    criterion, _ = build_criterion_from_config(m, t)
    matcher = criterion.matcher.to(device).train()
    need(criterion.group_detr == GROUPS and (matcher.cost_bbox, matcher.cost_class, matcher.cost_giou,
         matcher.cost_mask_ce, matcher.cost_mask_dice) == (5, 2, 2, 5, 5), "Exact five stock native matcher weights")
    return model.to(device).train(), matcher


def run(args):
    import numpy as np
    import torch
    from rfdetr.datasets.coco import ConvertCoco, make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models import matcher as native
    from rfdetr.utilities.tensors import make_collate_fn

    need({name: version(name) for name in VERSIONS} == VERSIONS
         and all(sha(Path(import_module(name).__file__).read_bytes()) == digest
                 for name, digest in NATIVE_SOURCE.items()), "Exact native numeric runtime and source")
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1
         and torch.cuda.is_bf16_supported(including_emulation=False), "One visible native BF16 GPU")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    need(free >= CAP, "CUDA free memory guard")
    torch.cuda.set_per_process_memory_fraction(min(.35, CAP / total), device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    items = json.loads(exact_bytes(args.fit_index, 348469, FIT_INDEX_SHA))["items"]
    need(len(items) == 741 and [row["dataset_index"] for row in items] == list(range(741)), "Exact SSDD FIT census")
    bundle_bytes = exact_bytes(args.fit_bundle, 33079338, FIT_BUNDLE_SHA)
    model, matcher = initialize(args, torch, device)
    saved_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    convert = ConvertCoco(include_masks=True, include_keypoints=False, cat2label={1: 0}, num_keypoints_per_class=[])
    transform = make_coco_transforms_square_div_64(
        "train", RESOLUTION, multi_scale=False, expanded_scales=False, skip_random_resize=True,
        patch_size=12, num_windows=1, aug_config=None, scale_jitter=True, gpu_postprocess=False,
        keypoint_flip_pairs=None)
    collate = make_collate_fn(block_size=12, pack=True, pad_targets_to=None)
    args.output.mkdir()
    rows = []
    with ZipFile(BytesIO(bundle_bytes)) as bundle:
        annotation = bundle.read("annotations/fit.json")
        need(len(annotation) == 498760 and sha(annotation) == FIT_GT_SHA, "Exact closed original FIT labels")
        gt = json.loads(annotation)
        ids = {row["image_id"] for row in items}
        need({row["id"] for row in gt["images"]} == ids and len(ids) == 741, "Exact FIT-only image membership")
        by_image = defaultdict(list)
        for row in gt["annotations"]:
            need(row["image_id"] in ids and row["category_id"] == 1, "Only original FIT ship labels")
            by_image[row["image_id"]].append(row)
        for ordinal, (item, repeat) in enumerate(view_plan(items, args.tranche)):
            seed = 5301000 + 2 * item["dataset_index"] + repeat
            iid, digest = item["image_id"], item["image_sha256"]
            raw = bundle.read(f"images/{digest}.jpg")
            need(len(raw) == item["image_bytes"] and sha(raw) == digest, "Exact original FIT JPEG bytes")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            image, scale = decode_pil_image_bytes(raw, draft_size=None)
            h, w = item["original_hw"]
            need(image.mode == "RGB" and image.size == (w, h) and scale == (1., 1.), "Exact native FIT image decode geometry")
            image, source = convert(image, dict(image_id=iid, annotations=by_image[iid]))
            image, target = transform(image, source)
            count = len(target["labels"])
            row = dict(ordinal=ordinal, target_rows=count)
            if count == 0:
                row["status"] = "ZERO_TARGET_AUGMENTED"
            elif count > WIDTH or not bool(target["masks"].flatten(1).any(1).all()):
                row["status"] = "INVALID_TARGET_ROWS"
            else:
                samples, packed = collate([(image, target)])
                need(samples.tensors.shape == (1, 3, RESOLUTION, RESOLUTION)
                     and not bool(samples.mask.any()), "Exact packed unpadded native 576 FIT batch")
                targets = packed.to_list(device, non_blocking=True)
                try:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        output = model(samples.to(device, non_blocking=True), targets)
                        final = {key: output[key] for key in ("pred_logits", "pred_boxes", "pred_masks")}
                        packet = capture_final(matcher, native, final, targets)
                    need(packet["uniform_draw"].shape == (K, 2), "Exact stock K=1296 native matcher draw")
                    sample = make_sampler(packet, device)
                    pred, binary = sample(packet["uniform_draw"].numpy())
                    need(np.array_equal(pred, packet["native_pred_at_draw"].numpy())
                         and np.array_equal(binary, packet["native_target_at_draw"].numpy()),
                         "Saved sparse factors replay the exact stock draw")
                finally:
                    with torch.no_grad():
                        for name, value in model.named_buffers():
                            value.copy_(saved_buffers[name])
                packet.update(protocol=args.protocol, fit_index_sha256=FIT_INDEX_SHA, raw_model_sha256=RAW_SHA,
                              input_sha256=state_digest({"pixels": samples.tensors}),
                              ordinal=ordinal, dataset_index=item["dataset_index"], image_id=iid, augmentation_seed=seed)
                leaf = f"VIEW_{ordinal:02d}_PRIVATE.pt"
                with (args.output / leaf).open("xb") as stream:
                    torch.save(packet, stream)
                with (args.output / leaf).open("rb") as stream:
                    packet_sha = file_digest(stream)
                row.update(status="STOCK_TRACE_EXACT", small_targets=int(packet["is_small"].sum()),
                           private_packet_name=leaf, private_packet_sha256=packet_sha,
                           private_packet_bytes=(args.output / leaf).stat().st_size)
                del packet, sample, final, output, targets
            rows.append(row)
            print(json.dumps(dict(event="SSDD_FIT_MATCHER_CAPTURE_VIEW", **row), sort_keys=True), flush=True)
            if args.tranche == "smoke" and row["status"] == "STOCK_TRACE_EXACT":
                break
    traces = sum(row["status"] == "STOCK_TRACE_EXACT" for row in rows)
    need(state_digest(model.state_dict()) == RAW_SHA, "No online parameter or buffer update")
    need((args.tranche == "smoke" and traces == 1) or (args.tranche == "cohort32" and len(rows) == 32),
         "Exact declared tranche coverage")
    decoder = {}
    for name in ("Pillow", "simplejpeg"):
        try:
            decoder[name] = version(name)
        except PackageNotFoundError:
            decoder[name] = None
    result = dict(status="NATIVE_SSDD_FIT_FINAL_MATCHER_CAPTURE_COMPLETE", protocol=args.protocol,
                  tranche=args.tranche, visited_views=len(rows), exact_stock_traces=traces, rows=rows,
                  predeclared_mechanism_trace_count_met=args.tranche == "cohort32" and traces >= 24,
                  resolution=RESOLUTION, groups=GROUPS, width=WIDTH, stock_points=K,
                  original_archive_sha256=ARCHIVE_SHA, raw_model_sha256=RAW_SHA,
                  fit_index_sha256=FIT_INDEX_SHA, fit_bundle_sha256=FIT_BUNDLE_SHA, fit_gt_sha256=FIT_GT_SHA,
                  native_versions=VERSIONS, decoder_versions=decoder, cuda_device=torch.cuda.get_device_name(0),
                  gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device), elapsed_seconds=perf_counter() - started,
                  FIT_only=True, DEV=False, TEST=False, AP=False, optimizer_created=False, backward=False, updates=0)
    with (args.output / "CAPTURE_SAFE.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "fit-bundle", "fit-index", "recipe", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--tranche", choices=("smoke", "cohort32"), required=True)
    args = parser.parse_args()
    need(bool(re.fullmatch(r"[0-9a-f]{40}", args.protocol))
         and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
         and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
         "Prospective protocol and one visible offline GPU")
    inputs = args.archive, args.fit_bundle, args.fit_index, args.recipe
    need(all(path.is_absolute() and path.resolve(strict=True) == path and path.is_file() for path in inputs)
         and args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
         and not args.output.exists() and not args.output.is_symlink(), "Explicit canonical closed input and fresh output paths")
    sys.addaudithook(offline)
    run(args)


if __name__ == "__main__":
    main()
