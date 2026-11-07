"""Exercise the official RF-DETR CLI and selected codecs on synthetic CPU inputs."""
import contextlib
import io
import math
import os
import sys
from pathlib import Path
from unittest.mock import patch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_printed_config(parsed):
    model = parsed["model"]
    training = model["train_config"]
    architecture = model["model_config"]
    require(parsed["seed_everything"] == 5201, "candidate CLI seed")
    require(architecture["class_path"] == "rfdetr.config.RFDETRSegNanoConfig", "Nano mask configuration class")
    require(training["class_path"] == "rfdetr.config.SegmentationTrainConfig", "mask training configuration class")
    for key, expected in {
        "num_classes": 1, "resolution": 576, "positional_encoding_size": 48,
        "num_queries": 100, "num_select": 100, "group_detr": 13,
        "mask_downsample_ratio": 4, "compile": False,
    }.items():
        require(architecture["init_args"][key] == expected, f"architecture field {key}")
    for key, expected in {
        "dataset_file": "roboflow", "epochs": 100, "batch_size": 1,
        "grad_accum_steps": 4, "seed": 5201, "eval_max_dets": 100,
        "eval_backend": "faster_coco_eval", "use_ema": True, "run_test": False,
        "multi_scale": "OFF", "augmentation_backend": "TV",
    }.items():
        require(training["init_args"][key] == expected, f"training field {key}")
    require(parsed["trainer"]["max_epochs"] == 100, "printed trainer epoch count")


def check_cli(config_path):
    import yaml
    from rfdetr.cli import main as cli_main
    from rfdetr.training.cli import RFDETRCli

    def prevent_instantiation(*args, **kwargs):
        raise RuntimeError("parse-only check reached class instantiation")

    stdout, stderr = io.StringIO(), io.StringIO()
    argv = ["rfdetr", "fit", "--config", str(config_path), "--print_config"]
    with patch.object(sys, "argv", argv), patch.object(
        RFDETRCli, "instantiate_classes", prevent_instantiation
    ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            cli_main()
        except SystemExit as exit_result:
            if exit_result.code not in (0, None):
                raise ValueError(f"official CLI exit {exit_result.code}: {stderr.getvalue()[-1600:]}") from exit_result
        else:
            raise ValueError("official --print_config should exit before instantiation")
    parsed = yaml.safe_load(stdout.getvalue())
    check_printed_config(parsed)
    return {
        "actual_cli_exit": 0,
        "parsed_model_resolution": parsed["model"]["model_config"]["init_args"]["resolution"],
        "classes_instantiated": False,
        "printed_bytes": len(stdout.getvalue().encode("utf-8")),
    }


def check_image_codec():
    import cv2
    import numpy as np
    from PIL import Image
    from rfdetr.datasets import io_utils

    original = np.zeros((8, 13, 3), dtype=np.uint8)
    original[:, :, 0] = 72
    original[:, :, 1] = 128
    original[:, :, 2] = 184
    require(Path(cv2.__file__).resolve().is_relative_to(Path(sys.prefix)), "selected headless cv2 provider")
    bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    require(np.array_equal(bgr, original[:, :, ::-1]), "native headless cv2 color conversion")
    jpeg, png = io.BytesIO(), io.BytesIO()
    Image.fromarray(original).save(jpeg, format="JPEG", quality=95)
    Image.fromarray(original).save(png, format="PNG")
    jpeg_data = jpeg.getvalue()
    fast = io_utils._decode_with_simplejpeg(jpeg_data, None)
    require(fast is not None, "simplejpeg path must decode the image without Pillow fallback")
    fast_pixels, fast_scales = fast
    decoded_jpeg, jpeg_scales = io_utils.decode_pil_image_bytes(jpeg_data)
    decoded_png, png_scales = io_utils.decode_pil_image_bytes(png.getvalue())
    require(fast_pixels.dtype == np.uint8 and fast_pixels.shape == (8, 13, 3), "non-square JPEG geometry")
    require(fast_scales == jpeg_scales == png_scales == (1.0, 1.0), "original decode scale")
    require(decoded_jpeg.mode == decoded_png.mode == "RGB", "RF-DETR RGB image inputs")
    require(np.array_equal(np.asarray(decoded_jpeg), fast_pixels), "public JPEG route agrees with simplejpeg")
    require(np.array_equal(np.asarray(decoded_png), original), "PNG preserves all decoded pixels")
    return {
        "size_hw": [8, 13], "simplejpeg_actual": True, "pillow_png_pixels_equal": True,
        "headless_cv2_color_conversion_passed": True, "opencv_runtime_version": cv2.__version__,
    }


def check_mask_codec_and_metric():
    import numpy as np
    import torch
    from pycocotools import mask as mask_api
    from rfdetr.datasets.coco import convert_coco_poly_to_mask
    from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision

    height, width = 8, 13
    reference = np.zeros((height, width), dtype=np.uint8)
    reference[2:5, 3:9] = 1
    encoded = mask_api.encode(np.asfortranarray(reference))
    require(int(mask_api.area(encoded)) == 18, "compressed RLE area")
    rle = {"size": list(encoded["size"]), "counts": encoded["counts"].decode("ascii")}
    polygon = [[3, 2, 9, 2, 9, 5, 3, 5]]
    decoded = convert_coco_poly_to_mask([rle, polygon], height, width)
    expected = torch.from_numpy(reference)
    require(tuple(decoded.shape) == (2, height, width) and decoded.dtype == torch.uint8, "native mask decode contract")
    require(torch.equal(decoded[0], expected) and torch.equal(decoded[1], expected), "RLE and polygon pixel equality")

    metric = OnePassCocoMeanAveragePrecision(
        iou_type=("bbox", "segm"), backend="faster_coco_eval",
        max_detection_thresholds=[1, 10, 100], sync_on_compute=False,
        num_classes=1,
    )
    backend = metric._coco_backend
    evaluator_module = backend.cocoeval.__module__
    require(type(backend).__name__ == "_FasterCocoEvalBackend"
            and evaluator_module.startswith("faster_coco_eval."), "actual selected evaluator implementation")
    instance = {
        "boxes": torch.tensor([[3, 2, 9, 5]], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.int64),
        "masks": expected[None].to(torch.bool),
    }
    predicted = dict(instance, scores=torch.tensor([0.9], dtype=torch.float32))
    metric.update([predicted], [instance])
    result = metric.compute()
    bbox_map, segm_map = float(result["bbox_map"]), float(result["segm_map"])
    require(result["classes"].tolist() == [0], "metric class index")
    require(all(math.isclose(v, 1.0, rel_tol=0.0, abs_tol=1e-6) for v in (bbox_map, segm_map)), "perfect fixture COCO metric")
    return {
        "rle_polygon_pixels_equal": True, "mask_area": 18,
        "actual_backend": "faster_coco_eval", "evaluator_module": evaluator_module, "fixture_bbox_map": bbox_map,
        "fixture_segm_map": segm_map,
    }


def run_probe(config_path):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "hide CUDA before any scientific imports")
    require("torch" not in sys.modules, "run in a fresh scientific worker")
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(not torch.cuda.is_initialized(), "CUDA untouched before CPU probe")
    cli = check_cli(Path(config_path))
    image = check_image_codec()
    mask = check_mask_codec_and_metric()
    require(not torch.cuda.is_initialized(), "CUDA untouched after CPU probe")
    return {"cli": cli, "image_codec": image, "mask_and_metric": mask, "real_sar_data": False}
