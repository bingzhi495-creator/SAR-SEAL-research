"""Export frozen HRSID native E24 EMA masks and score the independently audited DEV705."""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import io
import json
import math
from os import fstat
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile


BASE = Path(__file__).resolve().parent.parent
RELEASE = BASE / "github_release_20261011/research_round_20260925_global"
DEV = BASE / "HRSID_OFFICIAL_PNG_DEV705_AUDIT_ACTUAL_PRIVATE_20261109_01"
FILES = {
    "DEV_INPUT_AUDIT_SAFE.json": "2a02c89e84da572ebf3f1fec859609376a56c57f0fdf5efd84cfb5abcfc41439",
    "ROOT_INDEPENDENT_DEV705_SAFE.json": "a650238185dfe32227c720f31d6ba2439d115902e5a4037be965626ac0180cfe",
    "DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json": "39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f",
    "DEV_ORIGINAL_GT_PRIVATE.json": "393b3da233991dc5fafc98b4ddeaa341b042d9571434ef8ce53c3f2fdada5628",
    "DEV_EMPTY_RASTERS_PRIVATE.json": "f3f2d644b2e0ce475b4a46b382ec7ec51f2efa4609c8f85b4f8f552ff57d5889",
}
MODEL_SOURCES = {
    "rfdetr.datasets.coco": "be64b20a178d9cde41db69e01e17e1296e36eee791e837294c3f930ae56f03b9",
    "rfdetr.datasets.io_utils": "0e8d54c2867f8eec28aa25f650501a91bac8116ca2f7e40e9ddd4cd1332bcd65",
    "rfdetr.models.lwdetr": "248faf15d673bc18f51f60bb55245f64eee40eb9c8adacea398aebdadad2243b",
    "rfdetr.models.transformer": "279c0af7e999ff4ce58845c94c87bd9211356d62cb1eeaeb626fe2680fbb8f4d",
    "rfdetr.models.postprocess": "0e9a13190e559e517d84b2239ab034fa7bed5e1f6372b0a608c2a095c1b67b82",
    "rfdetr.utilities.tensors": "122f44a151db786ef4845c5d89e5cb2209624651d3638f007e3c0867f7312827",
}
OFFICIAL = RELEASE / "code/DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/worker.py"
OFFICIAL_SHA = "f9fc79403d12d8840c3269b04ba22d7584b8eb5e3c592ead2b6359ad7df65748"
FIT_AUDIT = BASE / "HRSID_OFFICIAL_PNG_FIT_AUDIT_CANDIDATE_20261109_01/hrsid_png_fit_audit.py"
FIT_AUDIT_SHA = "eb5f91a333ea7c7a1c12a87574d0c192f678aa80a742107a510654e3d5512df5"
RECIPE = RELEASE / "configs/RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml"
RECIPE_SHA = "ba9508d9acab5ff68b10db8a1aff5d9001868d6dda00e39f299d6f5cc1b8011b"
FIT_INDEX_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
COCO_SHA = "a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b"
INITIAL_SHA = "63ff93afdd977ba79b5b5a46ba8b4839ef6c40eb0722b29e4667d4932404fbc2"
ENDPOINT_KEY = "sar_seal_hrsid_coco_fit_native_endpoint_v1"
DEV_IMAGES, GT_COUNT, EMPTY_COUNT = 705, 1782, 4
WIDTH, NATIVE_ROWS, STEPS = 164, 100, 24 * 735
CAP = 6 * 1024**3


def need(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def pinned_bytes(path, digest, limit):
    need(path.is_absolute() and path.resolve(strict=True) == path and 0 < path.stat().st_size <= limit,
         "Expected canonical, bounded input: " + path.name)
    data = path.read_bytes()
    need(sha256(data).hexdigest() == digest, "Changed pinned input: " + path.name)
    return data


def pinned_module(path, digest):
    module = ModuleType("_closed_" + path.stem)
    module.__file__ = str(path)
    exec(compile(pinned_bytes(path, digest, 1024**2), str(path), "exec"), module.__dict__)
    return module


def dev_file(name, limit=4 * 1024**2):
    return json.loads(pinned_bytes(DEV / name, FILES[name], limit))


def qualified_manifest():
    audit = dev_file("DEV_INPUT_AUDIT_SAFE.json")
    independent = dev_file("ROOT_INDEPENDENT_DEV705_SAFE.json")
    need(audit["decoded_dev_images"] == independent["images"] == DEV_IMAGES
         and audit["attempted_dev_rasters"] == independent["unchanged_annotations"] == GT_COUNT
         and audit["empty_dev_rasters"] == independent["independent_empty_rasters"] == EMPTY_COUNT
         and audit["fractional_coordinate_annotations"] == 213
         and independent["no_exact_FIT_encoded_or_RGB_overlap"] is True
         and independent["official_TEST_opened"] is False
         and audit["image_manifest_sha256"] == FILES["DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json"]
         and audit["original_dev_gt_sha256"] == FILES["DEV_ORIGINAL_GT_PRIVATE.json"]
         and audit["empty_raster_findings_sha256"] == FILES["DEV_EMPTY_RASTERS_PRIVATE.json"],
         "Exact independent DEV-only input qualification")
    manifest = dev_file("DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json")
    need(len(manifest) == len({item["image_id"] for item in manifest}) == DEV_IMAGES,
         "Exact ordered DEV705 image-only population")
    return manifest


def native_rows(raw, *, image_id, hw, torch, postprocess, np, codec):
    logits, boxes, masks = (raw[key] for key in ("pred_logits", "pred_boxes", "pred_masks"))
    need(tuple(logits.shape) == (1, WIDTH, 2) and tuple(boxes.shape) == (1, WIDTH, 4)
         and masks.ndim == 4 and tuple(masks.shape[:2]) == (1, WIDTH)
         and all(value.dtype == torch.float32 and bool(torch.isfinite(value).all())
                 for value in (logits, boxes, masks)), "Exact finite FP32 native eval output")
    sizes = torch.tensor([hw], dtype=torch.int64, device=logits.device)
    result, = postprocess(
        {"pred_logits": logits, "pred_boxes": boxes, "pred_masks": masks}, sizes, score_threshold=None,
    )
    scores, labels = result["scores"].tolist(), result["labels"].tolist()
    mask_batch = result["masks"].detach().cpu()
    need(mask_batch.dtype == torch.bool and tuple(mask_batch.shape) == (NATIVE_ROWS, 1, *hw)
         and len(labels) == len(scores) == NATIVE_ROWS and all(label in (0, 1) for label in labels)
         and all(math.isfinite(score) and 0 <= score <= 1 for score in scores)
         and all(a >= b for a, b in zip(scores, scores[1:])), "Native original-size stable top100")
    ranks, rows = [], []
    for index, (label, score) in enumerate(zip(labels, scores)):
        if label == 1:
            continue
        compressed = codec.encode(np.asfortranarray(mask_batch[index, 0].numpy(), dtype=np.uint8))
        counts = compressed["counts"].decode("ascii")
        need(compressed["size"] == list(hw) and bool(counts), "Original-canvas compressed COCO RLE")
        ranks.append(index + 1)
        rows.append(dict(image_id=image_id, category_id=1, score=score,
                         segmentation=dict(size=list(hw), counts=counts)))
    return dict(image_id=image_id, raw_query_count=WIDTH, native_count=NATIVE_ROWS,
                unmapped_slot1_count=NATIVE_ROWS - len(rows), native_ranks=ranks, coco=rows)


def score_variants(gt, predictions, empty_ids):
    """Use the same canonical official COCO adapter for both GT populations."""
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    adapter = pinned_module(OFFICIAL, OFFICIAL_SHA)
    image_ids = [image["id"] for image in gt["images"]]
    need(empty_ids <= {ann["id"] for ann in gt["annotations"]}, "Prelocked empty IDs belong to original GT")

    def score(excluded):
        local_gt = deepcopy(gt)
        local_gt["annotations"] = [row for row in local_gt["annotations"] if row["id"] not in excluded]
        coco = COCO()
        coco.dataset = local_gt
        captured = []

        def evaluator(*args):
            instance = COCOeval(*args)
            captured.append(instance)
            return instance

        with redirect_stdout(io.StringIO()):
            coco.createIndex()
            values = adapter.evaluate_population(dict(COCO=COCO, COCOeval=evaluator), coco, predictions, image_ids)
        evaluation, = captured
        iou, = np.flatnonzero(np.isclose(evaluation.params.iouThrs, .90))
        precision = evaluation.eval["precision"][iou, :, :, 0, -1]
        valid = precision[precision > -1]
        return dict(annotations=len(local_gt["annotations"]), AP=values[0], AP75=values[2],
                    AP90=float(valid.mean()) if valid.size else -1.0, APs=values[3], AR100=values[8],
                    all12=values)

    return dict(primary=score(set()), sensitivity=score(empty_ids))


def check_endpoint(checkpoint, training_protocol):
    need(checkpoint["pytorch-lightning_version"] == "2.6.6" and checkpoint["epoch"] == 23
         and checkpoint["global_step"] == STEPS, "Exact HRSID E24 fixed-budget archive")
    marker = checkpoint[ENDPOINT_KEY]
    expected = dict(protocol=training_protocol, epoch=24, actual_updates=STEPS,
                    native_index_sha256=FIT_INDEX_SHA, coco_sha256=COCO_SHA,
                    original_recipe_sha256=RECIPE_SHA, initial_expanded_sha256=INITIAL_SHA,
                    FIT_only=True, exact_restart_trajectory_verified=False)
    need(set(marker) == expected.keys() | {"rng_state"}
         and all(marker[key] == value for key, value in expected.items())
         and marker["FIT_only"] is True and marker["exact_restart_trajectory_verified"] is False,
         "Exact HRSID FIT-only COCO-native E24 endpoint lineage")


def load_model(path, expected_sha, torch, evaluation_root, training_protocol):
    from dataclasses import replace
    import yaml
    from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig
    from rfdetr.models._defaults import MODEL_DEFAULTS
    from rfdetr.models.lwdetr import build_model_from_config

    need(path.is_absolute() and path.resolve(strict=True) == path and 0 < path.stat().st_size <= 1024**3,
         "Expected exact closed native E24 archive")
    fit = pinned_module(FIT_AUDIT, FIT_AUDIT_SHA)
    with path.open("rb") as stream:
        identity = fit.archive_identity(fstat(stream.fileno()))
        digest = sha256()
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
        need(digest.hexdigest() == expected_sha, "Exact admitted native E24 checkpoint bytes")
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
        need(fit.archive_identity(fstat(stream.fileno())) == fit.archive_identity(path.stat()) == identity,
             "Closed E24 checkpoint remained unchanged")
    check_endpoint(checkpoint, training_protocol)
    config = RFDETRSegNanoConfig(**checkpoint["model_config"])
    need((config.resolution, config.num_queries, config.group_detr, config.num_select, config.num_classes) ==
         (576, WIDTH, 13, NATIVE_ROWS, 1) and config.pretrain_weights is None and config.segmentation_head,
         "Exact saved HRSID query configuration")
    train_args = yaml.safe_load(pinned_bytes(RECIPE, RECIPE_SHA, 1024**2))["model"]["train_config"]["init_args"]
    train = SegmentationTrainConfig(**dict(train_args, dataset_dir=str(evaluation_root)))
    model = build_model_from_config(config, train, defaults=replace(MODEL_DEFAULTS, force_no_pretrain=True))
    callback = checkpoint["callbacks"]["RFDETREMACallback"]
    average = callback["average_model_state_dict"]
    raw = checkpoint["state_dict"]
    need(set(average) == {"module." + name for name in raw} | {"n_averaged"}
         and callback["latest_update_step"] == int(average["n_averaged"]) == STEPS, "Exact EMA state and clock")
    selected = {key.removeprefix("module.model."): value for key, value in average.items()
                if key.startswith("module.model.")}
    target = model.state_dict()
    need(selected.keys() == target.keys() and all(
        value.shape == target[name].shape and value.dtype == target[name].dtype
        and (not value.is_floating_point() or bool(torch.isfinite(value).all())) for name, value in selected.items()
    ), "Full native EMA state keys, shapes, dtype and finiteness")
    model.load_state_dict(selected, strict=True)
    need(all(torch.equal(tensor, selected[name]) for name, tensor in model.state_dict().items()),
         "Strict native EMA readback")
    return model, config, train, fit


def run_export(args):
    import numpy as np
    import torch
    from pycocotools import mask as codec
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    need(sys.version_info[:3] == (3, 10, 13) and torch.__version__ == "2.10.0+cu130"
         and version("rfdetr") == "1.12.0" and version("pycocotools") == "2.0.11"
         and version("Pillow") == "12.3.0" and np.__version__ == "1.26.4", "Exact native evaluation runtime")
    for name, digest in MODEL_SOURCES.items():
        pinned_bytes(Path(import_module(name).__file__), digest, 1024**2)
    manifest = qualified_manifest()
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One isolated CUDA device")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    need(free >= CAP, "GPU admission floor")
    torch.cuda.set_per_process_memory_fraction(min(.35, CAP / total), device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    started = perf_counter()
    model, config, train, fit = load_model(args.checkpoint, args.checkpoint_sha256, torch, args.output, args.training_protocol)
    model = model.to(device).eval()
    transform = make_coco_transforms_square_div_64(
        "val", 576, multi_scale=False, expanded_scales=False, skip_random_resize=True,
        patch_size=config.patch_size, num_windows=config.num_windows, aug_config=None,
        scale_jitter=train.scale_jitter, gpu_postprocess=False,
    )
    postprocess = PostProcess(num_select=NATIVE_ROWS, upsample_masks_to_image_size=True)
    args.output.mkdir()
    digest = sha256()
    mapped = slot1 = 0
    identity = fit.archive_identity(args.archive.stat())
    need(args.archive.is_absolute() and args.archive.resolve(strict=True) == args.archive
         and identity[2] == fit.ARCHIVE_BYTES, "Exact canonical official archive")
    torch.cuda.reset_peak_memory_stats(device)
    with (args.output / "DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl").open("xb") as predictions, args.archive.open("rb") as stream:
        need(fit.archive_identity(fstat(stream.fileno())) == identity, "Closed archive handle")
        stream.seek(fit.CENTRAL_OFFSET)
        need(sha256(stream.read(fit.CENTRAL_BYTES)).hexdigest() == fit.CENTRAL_SHA, "Exact central directory")
        with ZipFile(stream) as archive, torch.inference_mode():
            for ordinal, item in enumerate(manifest, 1):
                hw = (item["height"], item["width"])
                member = item["member"]
                info = archive.getinfo(member)
                need(member == "images/" + fit.image_stem(item["file_name"]) + ".png"
                     and hw == (800, 800) and info.file_size == item["bytes"]
                     and f"{info.CRC:08x}" == item["crc32"], "Exact admitted DEV member")
                raw = fit.checked_member(archive, member, fit.PNG_LIMIT)
                need(sha256(raw).hexdigest() == item["encoded_sha256"], "Exact qualified DEV PNG bytes")
                image, scales = decode_pil_image_bytes(raw, draft_size=None)
                need(image.mode == "RGB" and image.size == (800, 800) and scales == (1.0, 1.0)
                     and sha256(np.asarray(image).tobytes(order="C")).hexdigest() == item["decoded_rgb_sha256"],
                     "Exact qualified DEV decoded RGB pixels")
                tensor, target = transform(image, None)
                need(target is None and tensor.shape == (3, 576, 576) and tensor.dtype == torch.float32,
                     "Native GT-free resize and normalization")
                sample = nested_tensor_from_tensor_list([tensor], block_size=config.patch_size * config.num_windows)
                need(sample.mask is not None and not bool(sample.mask.any()), "Native full image without padding")
                raw_outputs = model(sample.to(device))
                need(tuple(raw_outputs["pred_masks"].shape) == (1, WIDTH, 144, 144), "Exact mask canvas")
                result = native_rows(raw_outputs, image_id=item["image_id"], hw=hw, torch=torch,
                                     postprocess=postprocess, np=np, codec=codec)
                mapped += len(result["coco"])
                slot1 += result["unmapped_slot1_count"]
                line = encoded(result)
                predictions.write(line)
                digest.update(line)
                if ordinal % 100 == 0:
                    print(json.dumps(dict(event="HRSID_DEV_NATIVE_EXPORT_PROGRESS", images=ordinal)), flush=True)
        need(fit.archive_identity(fstat(stream.fileno())) == fit.archive_identity(args.archive.stat()) == identity,
             "Original archive unchanged during DEV-only export")
    torch.cuda.synchronize(device)
    need(mapped + slot1 == DEV_IMAGES * NATIVE_ROWS, "Exact full DEV native row accounting")
    result = dict(status="HRSID_DEV705_E24_NATIVE_EXPORT_COMPLETE", protocol=args.protocol,
                  checkpoint_sha256=args.checkpoint_sha256, checkpoint_ema_updates=STEPS,
                  training_protocol=args.training_protocol, initialization="official_COCO_only",
                  image_manifest_sha256=FILES["DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json"],
                  predictions_sha256=digest.hexdigest(), images=DEV_IMAGES, raw_queries=DEV_IMAGES * WIDTH,
                  native_top_rows=DEV_IMAGES * NATIVE_ROWS, mapped_rows=mapped, unmapped_slot1_rows=slot1,
                  num_select=NATIVE_ROWS, query_width=WIDTH, precision="FP32_TF32_disabled",
                  peak_allocated=torch.cuda.max_memory_allocated(device), elapsed_seconds=perf_counter() - started,
                  GT_opened=False, official_TEST_opened=False, AP=False)
    with (args.output / "DEV_NATIVE_EXPORT_SAFE.json").open("xb") as stream:
        stream.write(encoded(result))


def run_score(args):
    need(version("pycocotools") == "2.0.11" and version("numpy") == "1.26.4", "Pinned canonical CPU COCO runtime")
    manifest = qualified_manifest()
    report = json.loads(pinned_bytes(args.export / "DEV_NATIVE_EXPORT_SAFE.json", args.export_report_sha256, 1024**2))
    need(report["status"] == "HRSID_DEV705_E24_NATIVE_EXPORT_COMPLETE"
         and report["checkpoint_sha256"] == args.checkpoint_sha256 and report["checkpoint_ema_updates"] == STEPS
         and report["images"] == DEV_IMAGES and report["protocol"] == args.protocol
         and report["training_protocol"] == args.training_protocol
         and report["image_manifest_sha256"] == FILES["DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json"]
         and report["GT_opened"] is report["official_TEST_opened"] is False, "Exact closed native export")
    raw = pinned_bytes(args.export / "DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl", report["predictions_sha256"], 512 * 1024**2)
    blocks = [json.loads(line) for line in raw.splitlines()]
    del raw
    gt = dev_file("DEV_ORIGINAL_GT_PRIVATE.json")
    findings = dev_file("DEV_EMPTY_RASTERS_PRIVATE.json")
    empty_ids = {row["annotation_id"] for row in findings if row["kind"] == "EMPTY_COCO_RASTER"}
    need(len(empty_ids) == len(findings) == EMPTY_COUNT and len(gt["annotations"]) == GT_COUNT
         and [row["id"] for row in gt["images"]] == [row["image_id"] for row in manifest]
         and [row["id"] for row in gt["categories"]] == [1], "Exact original DEV and fixed empty-only sensitivity")
    flat = []
    need(len(blocks) == DEV_IMAGES, "Complete native DEV prediction order")
    for block, item in zip(blocks, manifest, strict=True):
        image_id = item["image_id"]
        ranks, rows = block["native_ranks"], block["coco"]
        need(block["image_id"] == image_id and block["raw_query_count"] == WIDTH
             and block["native_count"] == NATIVE_ROWS and block["unmapped_slot1_count"] + len(rows) == NATIVE_ROWS
             and len(ranks) == len(rows) and ranks == sorted(set(ranks))
             and all(type(rank) is int and 1 <= rank <= NATIVE_ROWS for rank in ranks)
             and all(row["image_id"] == image_id and row["category_id"] == 1
                     and row["segmentation"]["size"] == [800, 800] and "bbox" not in row for row in rows)
             and all(a["score"] >= b["score"] for a, b in zip(rows, rows[1:])), "Exact mapped native rows and order")
        flat.extend(rows)
    need(len(flat) == report["mapped_rows"], "Exact closed native prediction count")
    result = dict(status="HRSID_DEV705_E24_NATIVE_COCO_COMPLETE", protocol=args.protocol,
                  checkpoint_sha256=args.checkpoint_sha256, predictions_sha256=report["predictions_sha256"],
                  training_protocol=args.training_protocol, initialization="official_COCO_only",
                  original_dev_gt_sha256=FILES["DEV_ORIGINAL_GT_PRIVATE.json"], images=DEV_IMAGES,
                  excluded_empty_gt_sensitivity=EMPTY_COUNT, max_dets=[1, 10, 100], units="fraction",
                  official_TEST_opened=False, official_TEST_AP=False, **score_variants(gt, flat, empty_ids))
    with args.output.open("xb") as stream:
        stream.write(encoded(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(required=True, dest="mode")
    for mode, fn in (("export", run_export), ("score", run_score)):
        sub = modes.add_parser(mode)
        sub.add_argument("--protocol", required=True)
        sub.add_argument("--training-protocol", required=True)
        sub.add_argument("--checkpoint-sha256", required=True)
        sub.add_argument("--output", required=True, type=Path)
        sub.set_defaults(run=fn)
    modes.choices["export"].add_argument("--archive", required=True, type=Path)
    modes.choices["export"].add_argument("--checkpoint", required=True, type=Path)
    modes.choices["score"].add_argument("--export", required=True, type=Path)
    modes.choices["score"].add_argument("--export-report-sha256", required=True)
    args = parser.parse_args()
    need(all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (args.protocol, args.training_protocol))
         and bool(re.fullmatch(r"[0-9a-f]{64}", args.checkpoint_sha256))
         and args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
         and not args.output.exists() and not args.output.is_symlink(), "Pinned protocol and fresh isolated output")

    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise RuntimeError("DEV export and scoring are offline")

    sys.addaudithook(offline)
    args.run(args)


if __name__ == "__main__":
    main()

