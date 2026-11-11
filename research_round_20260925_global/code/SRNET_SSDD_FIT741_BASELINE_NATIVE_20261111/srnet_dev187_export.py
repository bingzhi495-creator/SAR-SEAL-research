"""Export one fixed-threshold SRNet E24 arm through stock CPU inference."""

from argparse import ArgumentParser
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from time import perf_counter
from types import ModuleType


MIB = 1024**2
AUTHOR_COMMIT = "2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6"
AUTHOR_CONFIG = "configs/SRNet/SRNet_r50_fpn_3x_SSDD_le90.py"
FIT_INDEX_SHA = "62966857c78cea1e3b060700c62a7b896942686c2e674d3eb09f516a92250bc1"
TRAIN_SOURCE_SHA = "0794b921ef6e7f4686d3a223cbd565588d8196435a7666dddf76e3d785f5f709"
MAPPING_SHA = "c2391246030018f041f6bedda36344145df7512595df6ecc4067a151f0e097a2"
IMAGE_STAGE_SHA = "8765c4cf67d6a89020670d343bfb1d31b9c37cfa9c86b7739a6934748e3b7f52"
INFERENCE_SOURCE_PINS = {
    "mmrotate/core/post_processing/bbox_nms_rotated.py":
        "5dbe45648f1825a670710d6dc4aa49f53386667170bc59246927940e1ea6d092",
    "mmrotate/models/dense_heads/rotated_anchor_head.py":
        "1c1f244efc89481bde422b4f15de1e1e1d3f70edac9676efec8d3f257608e773",
}
CPU_PACKAGES = {
    "torch": "1.12.0+cpu", "torchvision": "0.13.0+cpu", "mmcv-full": "1.6.0",
    "mmdet": "2.28.2", "numpy": "1.26.4", "opencv-python": "4.11.0.86",
    "pycocotools": "2.0.11", "Pillow": "9.5.0", "e2cnn": "0.2.3",
}
TRAIN_STATUS = "P346_SRNET_AUTHOR_BATCH4_MATCHED_TRAIN_COMPLETE_V1"
WAIT_STATUS = "SRNET_CUDA_E24_ORIGINAL_WAIT"
STATUS = "P352_SRNET_DEV187_NATIVE_EXPORT_COMPLETE"
THRESHOLDS = {"stock": .3, "low_threshold": .001}
STOCK_TEST = dict(nms_pre=2000, min_bbox_size=0, score_thr=.3,
                  nms=dict(iou_thr=.1), max_per_img=2000)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def matching(value, fields):
    return all(key in value and type(value[key]) is type(expected) and value[key] == expected for key, expected in fields.items())


def digest_text(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def exact_bytes(path, digest, limit):
    require(digest_text(digest) and path.is_absolute() and path.resolve(strict=True) == path
            and path.is_file() and 0 < path.stat().st_size <= limit,
            "Expected a canonical bounded file and exact SHA-256")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit and sha256(raw).hexdigest() == digest,
            "Exact original file bytes differ: " + path.name)
    return raw


def load_source(path, digest, name):
    source = ModuleType(name)
    source.__file__ = str(path)
    exec(compile(exact_bytes(path, digest, MIB), str(path), "exec"), source.__dict__)
    return source


def json_bytes(value):
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def make_image_only_index(manifest):
    images = []
    for image_id, item in zip(manifest["expected_ids"], manifest["items"], strict=True):
        require(image_id == item["image_id"], "Image-only manifest order differs")
        height, width = item["original_hw"]
        name = PurePosixPath(item["image_ref"]["path"]).name
        images.append(dict(id=image_id, file_name=name, height=height, width=width))
    return dict(images=images, categories=[dict(id=1, name="ship")], annotations=[])


def export_test_config(stock, arm):
    require(stock == STOCK_TEST and arm in THRESHOLDS, "Exact author test settings and fixed export arm required")
    return {**stock, "score_thr": THRESHOLDS[arm]}


def validate_closed_training(report, wait, stdout, report_sha, stdout_sha, checkpoint, checkpoint_sha):
    require(matching(wait, dict(status=WAIT_STATUS, original_os_wait=True, exit_code=0,
                                job_assigned=True, active_before_drain=0, active_after_wait=0,
                                resource_stop=None, source_sha256=TRAIN_SOURCE_SHA))
            and wait.get("log_sha256", {}).get("ORIGINAL.stdout") == stdout_sha,
            "Successful original P349 OS wait and linked stdout required")
    require(json.loads(stdout.rstrip().splitlines()[-1]) == dict(status=TRAIN_STATUS, result_sha256=report_sha),
            "Original P349 stdout must name the exact completed training report")
    require(matching(report, dict(status=TRAIN_STATUS, stage="e24", worker_source_sha256=TRAIN_SOURCE_SHA,
                                  author_commit=AUTHOR_COMMIT, fit_index_sha256=FIT_INDEX_SHA,
                                  actual_batch_size=4, fit_images=741, visits_per_full_epoch=744,
                                  completed_epochs=24, optimizer_updates=4464, model_forward_calls=4464,
                                  consumed_visits=17856, stock_author_source=True,
                                  FIT=True, DEV=False, TEST=False, AP=False)),
            "Fixed closed stock E24 training clocks and scope required")
    reference = report.get("checkpoint")
    require(isinstance(reference, dict) and reference.keys() == {"path", "bytes", "sha256"}
            and Path(reference["path"]) == checkpoint and checkpoint.name == "epoch_24.pth"
            and reference["sha256"] == checkpoint_sha and digest_text(checkpoint_sha)
            and type(reference["bytes"]) is int and 0 < reference["bytes"] <= 1024*MIB,
            "Exact original stock E24 checkpoint identity required")
    return reference


def load_checkpoint(torch, model, path, reference):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "Canonical original E24 checkpoint required")
    with path.open("rb") as stream:
        stat = os.fstat(stream.fileno())
        require(stat.st_size == reference["bytes"], "Original checkpoint length differs")
        digest = sha256()
        for block in iter(lambda: stream.read(MIB), b""):
            digest.update(block)
        require(digest.hexdigest() == reference["sha256"], "Original checkpoint SHA-256 differs")
        stream.seek(0)
        # Legacy Torch loads only this exact, locally produced and accepted artifact.
        payload = torch.load(stream, map_location="cpu")
        after = os.fstat(stream.fileno())
        require((stat.st_size, stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                "Original checkpoint changed during read")
    require(matching(payload["meta"], dict(epoch=24, iter=4464)),
            "Original checkpoint metadata must match the closed E24 endpoint")
    model.load_state_dict(payload["state_dict"], strict=True)


def write_export(dataset, items, native_infer, map_native_image, predictions_path, boxes_path):
    hashes = sha256(), sha256()
    candidates = []
    with predictions_path.open("xb") as predictions, boxes_path.open("xb") as boxes:
        for i, item in enumerate(items):
            native_boxes, native_masks = native_infer(dataset, i, item)
            rows, rotated = map_native_image(item, native_boxes, native_masks)
            candidates.append(len(rows))
            records = (dict(image_id=item["image_id"], predictions=rows),
                       dict(image_id=item["image_id"], rotated_boxes=rotated))
            for stream, digest, record in zip((predictions, boxes), hashes, records, strict=True):
                raw = json_bytes(record)
                stream.write(raw)
                digest.update(raw)
    return dict(image_count=len(candidates), candidate_count=sum(candidates),
                zero_candidate_images=candidates.count(0), max_candidates_per_image=max(candidates),
                predictions_sha256=hashes[0].hexdigest(), rotated_boxes_sha256=hashes[1].hexdigest())


def run(args):
    start = perf_counter()
    report = json.loads(exact_bytes(args.e24_report, args.e24_report_sha256, MIB))
    wait = json.loads(exact_bytes(args.e24_wait, args.e24_wait_sha256, MIB))
    stdout = exact_bytes(args.e24_stdout, args.e24_stdout_sha256, 16*MIB)
    checkpoint = validate_closed_training(report, wait, stdout, args.e24_report_sha256,
                                          args.e24_stdout_sha256, args.checkpoint, args.checkpoint_sha256)
    for name, digest in ({**report["author_source_pins"], **INFERENCE_SOURCE_PINS}).items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Invalid author source pin path")
        exact_bytes(args.source/relative, digest, MIB)

    stage = load_source(args.image_stage_source, IMAGE_STAGE_SHA, "_p352_fixed_images")
    manifest_ref = dict(path=args.manifest.as_posix(), bytes=stage.MANIFEST_BYTES, sha256=stage.MANIFEST_SHA)
    stage.manifest_objects(manifest_ref)
    manifest = json.loads(stage.checked_bytes(args.manifest, stage.MANIFEST_BYTES, stage.MANIFEST_SHA))
    index = make_image_only_index(manifest)

    require(args.images.is_absolute() and args.images.resolve(strict=True) == args.images
            and args.images.is_dir(), "Canonical materialized DEV187 image-only directory required")
    require({item.name for item in args.images.iterdir()} == {row["file_name"] for row in index["images"]},
            "DEV187 image-only directory differs from the fixed logical roster")
    require(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink()
            and not args.output.is_relative_to(args.images),
            "Fresh output outside the image-only input directory required")

    sys.path.insert(0, str(args.source))
    import cv2
    import mmcv
    import mmrotate
    import torch
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector

    require(Path(mmrotate.__file__).resolve() == args.source/"mmrotate/__init__.py",
            "Exact pinned author source must supply SRNet")
    require({name: version(name) for name in CPU_PACKAGES} == CPU_PACKAGES,
            "Exact isolated P336 CPU package versions required")
    mapping = load_source(args.mapping_source, MAPPING_SHA, "_p352_native_mapping")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    args.output.mkdir()
    index_path = args.output/"IMAGE_ONLY_COCO_PRIVATE.json"
    index_path.write_bytes(json_bytes(index))

    cfg = mmcv.Config.fromfile(str(args.source/AUTHOR_CONFIG))
    cfg.model.test_cfg = mmcv.ConfigDict(export_test_config(cfg.model.test_cfg, args.arm))
    settings = cfg.data.test.copy()
    settings.ann_file = str(index_path)
    settings.img_prefix = str(args.images) + os.sep
    settings.test_mode = True
    dataset = build_dataset(settings)
    require(dataset.CLASSES == ("ship",) and dataset.cat_ids == [1]
            and list(dataset.img_ids) == manifest["expected_ids"] and len(dataset) == stage.DEV_IMAGES,
            "Author test pipeline did not preserve the exact fixed DEV187 image/category roster")

    model = build_detector(cfg.model).cpu().eval()
    load_checkpoint(torch, model, args.checkpoint, checkpoint)

    def native_infer(dataset, i, item):
        expected = index["images"][i]
        reference = item["image_ref"]
        stage.checked_bytes(args.images/expected["file_name"], reference["bytes"], reference["sha256"])
        image = dataset.prepare_test_img(i)
        tensor, = image["img"]
        meta, = image["img_metas"]
        if not torch.is_tensor(tensor):
            tensor = tensor.data
        meta = meta.data
        require(tuple(meta["ori_shape"][:2]) == tuple(item["original_hw"])
                and tuple(tensor.shape) == (3, 608, 608) and tensor.device.type == "cpu",
                "Author test tensor and original image dimensions differ")
        with torch.inference_mode():
            result, = model.simple_test(tensor[None], [meta], rescale=True, thr=.5)
        return result

    summary = write_export(dataset, manifest["items"], native_infer, mapping.map_native_image,
                           args.output/"PREDICTIONS_PRIVATE.jsonl", args.output/"ROTATED_BOXES_PRIVATE.jsonl")
    require(not torch.cuda.is_initialized(), "Stock CPU export unexpectedly initialized CUDA")
    summary.update(status=STATUS, arm=args.arm, author_commit=AUTHOR_COMMIT,
                   checkpoint_sha256=args.checkpoint_sha256, e24_report_sha256=args.e24_report_sha256,
                   e24_original_wait_sha256=args.e24_wait_sha256, e24_stdout_sha256=args.e24_stdout_sha256,
                   image_manifest_sha256=stage.MANIFEST_SHA, mapping_source_sha256=MAPPING_SHA,
                   test_cfg=dict(cfg.model.test_cfg), mask_threshold=.5, device="cpu", precision="float32",
                   package_versions=CPU_PACKAGES, native_forward_calls=summary["image_count"],
                   FIT=False, DEV=True, GT_opened=False, TEST=False, AP=False, training=False,
                   elapsed_seconds=perf_counter()-start)
    raw = json_bytes(summary)
    (args.output/"EXPORT_SAFE.json").write_bytes(raw)
    print(json_bytes(dict(status=STATUS, report_sha256=sha256(raw).hexdigest())).decode().strip(), flush=True)


def offline(event, _):
    if event.startswith("socket.") and event != "socket.gethostname":
        raise PermissionError("Offline fixed image-only SRNet export")


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--arm", choices=tuple(THRESHOLDS), required=True)
    for name in ("source", "image-stage-source", "mapping-source", "manifest", "images", "checkpoint",
                 "e24-report", "e24-wait", "e24-stdout", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("checkpoint", "e24-report", "e24-wait", "e24-stdout"):
        parser.add_argument("--"+name+"-sha256", required=True)
    args = parser.parse_args()
    sys.addaudithook(offline)
    run(args)


if __name__ == "__main__":
    main()
