"""Exercise both fixed SRNet export thresholds on the preselected FIT image."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from unittest.mock import patch
from zipfile import ZipFile


SMOKE_SHA = "186d848c277467bae1809612b6ef8f8f4f0c61802df06e6d11e7c773c3486a48"
EXPORT_SHA = "294cd36f809695561d769010614d964f9d52ad0f68271ec817622571725e9c6a"
MAPPING_SHA = "c2391246030018f041f6bedda36344145df7512595df6ecc4067a151f0e097a2"
STATUS = "SRNET_FIT_AUTHOR_THRESHOLDS_CPU_PREFLIGHT_COMPLETE"


def load_exact(path, expected, name):
    if path.resolve(strict=True) != path or path.stat().st_size > 64 * 1024:
        raise ValueError("Bounded canonical source required")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Exact original source SHA-256 required")
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def emit(value):
    print(compact(value).decode("utf-8"), flush=True)


def run(args):
    smoke = load_exact(args.smoke, SMOKE_SHA, "_p355_smoke")
    source = load_exact(args.export_source, EXPORT_SHA, "_p355_export")
    mapper = load_exact(args.mapping_source, MAPPING_SHA, "_p355_mapping")
    sys.addaudithook(smoke.deny_network)
    original = smoke.load_prior(args.prior)
    proof = json.loads(smoke.read_exact(args.prior_report, smoke.P335_REPORT_SHA, 64 * 1024))
    smoke.require(proof["status"] == original.STATUS and proof["TEST"] is False, "Original FIT proof required")
    for name, pin in {**original.SOURCE_PINS, **source.INFERENCE_SOURCE_PINS}.items():
        smoke.read_exact(args.source / name, pin, 1024**2)
    index = args.fit / "_annotations.coco.json"
    data = json.loads(smoke.read_exact(index, original.FIT_INDEX_SHA, 2 * 1024**2))
    row = original.choose(data["images"])[0]
    smoke.read_exact(args.archive, original.FIT_SHA, 34 * 1024**2)
    with ZipFile(args.archive) as archive:
        roster = json.loads(archive.read("manifest.json"))["rows"]
    item, = (entry for entry in roster if entry["image_id"] == row["id"])
    image_ref = item["original_image_ref"]
    smoke.require(row["file_name"] == item["original_basename"], "Preselected FIT image differs")
    smoke.read_exact(args.fit / row["file_name"], image_ref["sha256"], 4 * 1024**2)
    smoke.require(len(smoke.read_exact(args.r50, smoke.R50_SHA, smoke.R50_BYTES)) == smoke.R50_BYTES,
                  "Exact R50 initialization required")
    sys.path.insert(0, str(args.source))
    import cv2
    import mmcv
    import numpy as np
    import torch
    import mmrotate
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    from pycocotools import mask as codec

    smoke.require(Path(mmrotate.__file__).resolve() == args.source / "mmrotate" / "__init__.py",
                  "Author package differs")
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.manual_seed(5201)
    np.random.seed(5201)
    cfg = mmcv.Config.fromfile(str(args.source / original.AUTHOR_CONFIG))
    stock = dict(cfg.model.test_cfg)
    arms = {arm: source.export_test_config(stock, arm) for arm in source.THRESHOLDS}
    settings = cfg.data.test.copy()
    settings.ann_file = str(index)
    settings.img_prefix = str(args.fit) + os.sep
    settings.test_mode = True
    dataset = build_dataset(settings)
    position, = (i for i, item in enumerate(dataset.data_infos) if item["id"] == row["id"])
    prepared = dataset.prepare_test_img(position)
    tensor, = prepared["img"]
    meta, = prepared["img_metas"]
    if not torch.is_tensor(tensor):
        tensor = tensor.data
    meta = meta.data
    hw = [row["height"], row["width"]]
    smoke.require(tuple(tensor.shape) == (3, 608, 608) and list(meta["ori_shape"][:2]) == hw,
                  "Author image shape differs")
    cfg.model.backbone.init_cfg.checkpoint = str(args.r50)
    model = build_detector(cfg.model)
    model.init_weights()
    model.cpu().eval()
    weights = torch.load(str(args.r50), map_location="cpu")
    smoke.require(torch.equal(model.backbone.conv1.weight.detach(), weights["conv1.weight"]),
                  "Author backbone did not load pinned initialization")
    del weights

    observed = {}
    for arm, parameters in arms.items():
        model.test_cfg.score_thr = parameters["score_thr"]
        smoke.require(dict(model.test_cfg) == parameters, "Author detector test config differs")
        native_masks = model.mask_head.get_masks
        pre_paste = []

        def monitor(features, rois, img_metas, rescale=True, thr=.5):
            counts = [int(pair[0].shape[0]) for pair in rois]
            pre_paste.extend(counts)
            emit(dict(event="SRNET_FIT_THRESHOLD_PRE_PASTE", arm=arm, candidates=counts))
            return native_masks(features, rois, img_metas, rescale, thr)

        started = perf_counter()
        with torch.inference_mode(), patch.object(model.mask_head, "get_masks", monitor):
            (boxes, masks), = model.simple_test(tensor[None], [meta], rescale=True, thr=.5)
        elapsed = perf_counter() - started
        mapped, rotated = mapper.map_native_image(dict(image_id=row["id"], original_hw=hw), boxes, masks)
        smoke.require(len(pre_paste) == 1 and pre_paste[0] == len(mapped) == len(rotated),
                      "Author post-NMS and export candidate counts differ")
        for output, bitmap in zip(mapped, masks[0], strict=True):
            rle = output["segmentation"]
            native = codec.decode(dict(size=rle["size"], counts=rle["counts"].encode("ascii")))
            smoke.require(np.array_equal(native, bitmap), "RLE differs from original author mask")
        observed[arm] = dict(score_threshold=parameters["score_thr"], candidates=len(mapped),
                             nonempty=sum(bool(mask.any()) for mask in masks[0]),
                             original_rle_roundtrips=len(mapped), elapsed_seconds=elapsed)
        emit(dict(event="SRNET_FIT_THRESHOLD_ARM_CLOSED", arm=arm, **observed[arm]))
        del boxes, masks, mapped, rotated
    smoke.require(not torch.cuda.is_initialized(), "CPU preflight initialized CUDA")
    return dict(status=STATUS, author_commit=original.AUTHOR_COMMIT, FIT=True, DEV=False, TEST=False, AP=False,
                selection="first preselected FIT image under original P335 selection",
                original_hw=hw, model_input_hw=[608, 608], r50_sha256=smoke.R50_SHA,
                task_heads="untrained author initialization", initialization_seed=5201,
                nms_pre=stock["nms_pre"], max_per_img=stock["max_per_img"],
                nms_iou=stock["nms"]["iou_thr"], mask_threshold=.5,
                inference_device="cpu", model_forward_calls=2, optimizer_updates=0, arms=observed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("smoke", "export-source", "mapping-source", "prior", "prior-report",
                 "source", "fit", "archive", "r50", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or args.output.exists():
        raise ValueError("Fresh isolated CPU preflight output required")
    payload = compact(run(args)) + b"\n"
    with args.output.open("xb") as out:
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
    emit(dict(status=STATUS, result_sha256=sha256(payload).hexdigest()))


if __name__ == "__main__":
    main()

