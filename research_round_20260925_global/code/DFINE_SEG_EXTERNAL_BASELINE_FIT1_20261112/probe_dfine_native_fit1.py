"""One closed HRSID FIT PNG through the fixed author D-FINE-seg-S CPU API."""
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from sysconfig import get_path
from time import perf_counter
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
AUTHOR = ROOT / "AUTHOR_SOURCE_VERIFIED"
CHECKPOINT = ROOT / "dfine_seg_s_coco.pt"
CLOSED = PROJECT / "P382_MASK_RTDETR_HRSID_FIT_ONE_CPU_EXPORT_ACTUAL_PRIVATE_20261111_01"
IMAGE_LOCK = PROJECT / "P382_MASK_RTDETR_HRSID_FIT_ONE_CPU_EXPORT_DRAFT_PRIVATE_20261111_01/IMAGE_ONLY_LOCK_PRIVATE.json"
FIT_ROOT = PROJECT / "HRSID_FIT_NATIVE_COCO_LAYOUT_ACTUAL_PRIVATE_20261109_01"
WEIGHT_SHA = "07500e13824b82c9ee7422fd98736259e3bafcad65bf63f48d0c386a4ba987c1"
IMAGE_LOCK_SHA = "7fa942c8197b3ca225609e3d91250def81d917ef7d14507d340244e6e87010cc"
P382_WAIT_SHA = "89c1b6477a732a95e6e9371a8971041ca760fa053c616361c56106a5a7a32b63"
P382_RESULT_SHA = "b6eb38f6390c29f2787246dcdb031a05eb727694b44873299096a72131f1cc4f"
AUTHOR_COMMIT = "b782ece206487792f36c165ca65548154cd12094"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def pin(path, sha, size=None):
    need(path.resolve(strict=True) == path and path.is_file(), "Expected exact canonical file")
    raw = path.read_bytes()
    need((size is None or len(raw) == size) and sha256(raw).hexdigest() == sha, "Pinned file identity differs")
    return raw


def main():
    need(sys.version_info[:3] == (3, 11, 15) and sys.flags.isolated == sys.flags.utf8_mode == 1
         and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
         "Exact isolated Python 3.11.15 CPU-only invocation")
    dependencies = {name: version(name) for name in ("dfine-seg", "torch", "torchvision", "numpy", "opencv-python-headless")}
    need(tuple(dependencies[k] for k in ("dfine-seg", "torch", "torchvision")) ==
         ("0.7.0", "2.13.0+cpu", "0.28.0+cpu"), "Exact admitted author CPU runtime")

    install = Path(get_path("purelib")) / "dfine_seg"
    verified = 0
    for author_file in (AUTHOR / "dfine_seg").rglob("*"):
        if author_file.is_file() and author_file.suffix in {".py", ".yaml", ".yml"}:
            installed = install / author_file.relative_to(AUTHOR / "dfine_seg")
            need(installed.read_bytes() == author_file.read_bytes(), "Installed author Python/config bytes differ")
            verified += 1
    need(verified >= 55, "Expected complete original author Python/config roster")

    lock = json.loads(pin(IMAGE_LOCK, IMAGE_LOCK_SHA))
    wait = json.loads(pin(CLOSED / "ORIGINAL_WAIT_PRIVATE.json", P382_WAIT_SHA))
    prior = json.loads(pin(CLOSED / "RESULT_SAFE.json", P382_RESULT_SHA))
    need(lock["FIT"] is True and lock["DEV"] is lock["TEST"] is lock["annotations_opened"] is False
         and lock["fit_image_count"] == 2937 and wait["original_os_wait"] is True
         and wait["original_os_exit"] == 0 and wait["process_group_closed"] is True
         and wait["active_before_drain"] == 0 and wait["stop_reason"] is None
         and prior["FIT"] is True and prior["DEV"] is prior["TEST"] is prior["GT_opened"] is False
         and prior["original_hw"] == [800, 800], "Exact closed prior FIT1 image selection")
    selected = lock["selected"]
    leaf = selected["native_basename"]
    need(Path(leaf).name == leaf and leaf.endswith(".png") and selected["image_id"] == 0, "One original PNG")
    image = FIT_ROOT / "train" / leaf

    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "urllib.Request"}:
            raise PermissionError("Offline model inference")
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0]))
            if path.is_absolute() and path.is_relative_to(FIT_ROOT) and path != image:
                raise PermissionError("Only exact prior closed FIT1 PNG is admitted")

    sys.addaudithook(audit)
    pin(image, selected["encoded_sha256"], selected["encoded_bytes"])
    pin(CHECKPOINT, WEIGHT_SHA, 48119167)

    import cv2
    import numpy as np
    import torch
    from dfine_seg import load_model, read_image
    from dfine_seg.infer.torch_model import TorchModel

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(20261112)
    np.random.seed(20261112)
    need(torch.version.cuda is None and not torch.cuda.is_initialized(), "Official CPU build only")

    bgr = read_image(image)
    need(bgr.shape == (800, 800, 3) and bgr.dtype == np.uint8
         and sha256(np.ascontiguousarray(bgr[..., ::-1]).tobytes()).hexdigest() == selected["decoded_rgb_sha256"],
         "Native author image read matches closed original RGB byte identity")
    actual_calls = []
    native_predict = TorchModel._predict

    def witnessed_predict(self, inputs):
        result = native_predict(self, inputs)
        actual_calls.append(dict(shape=list(inputs.shape), device=str(inputs.device), dtype=str(inputs.dtype)))
        return result

    total_start = perf_counter()
    with patch.object(TorchModel, "_predict", witnessed_predict):
        stock = load_model(CHECKPOINT, task="segment", device="cpu")
        need(len(actual_calls) == 1, "Exactly one built-in author synthetic warmup")
        need(stock.model_name == "s" and stock.task == "segment" and stock.enable_mask_head
             and stock.device == "cpu" and not stock.model.training
             and stock.binarize_masks and stock.mask_threshold == 0.5
             and stock.apply_nms and stock.nms_iou_thresh == 0.7 and stock.conf_thresh == 0.5,
             "Exact native wrapper and postprocessor settings")
        real_start = perf_counter()
        output = stock(bgr, conf_thresh=0.0)
        real_seconds = perf_counter() - real_start
    need(len(actual_calls) == 2 and all(row == dict(shape=[1, 3, 640, 640], device="cpu", dtype="torch.float32")
                                         for row in actual_calls), "Exactly one synthetic plus one real CPU forward")
    need(len(output) == 1 and set(output[0]) == {"labels", "boxes", "scores", "masks"}, "Author native mask API")
    result = output[0]
    masks, boxes, scores, labels = (result[k] for k in ("masks", "boxes", "scores", "labels"))
    rows = len(scores)
    need(0 < rows <= 300 and boxes.shape == (rows, 4) and labels.shape == (rows,)
         and masks.shape == (rows, 800, 800) and masks.dtype == torch.uint8
         and int(masks.max()) <= 1 and torch.isfinite(boxes).all().item() and torch.isfinite(scores).all().item()
         and scores.min().item() >= 0 and scores.max().item() <= 1
         and all(tensor.device.type == "cpu" for tensor in result.values()), "Native original-size finite binary mask outputs")
    pin(image, selected["encoded_sha256"], selected["encoded_bytes"])
    pin(CHECKPOINT, WEIGHT_SHA, 48119167)
    streams = {key: sha256(value.detach().contiguous().numpy().tobytes()).hexdigest() for key, value in result.items()}
    report = dict(status="DFINE_SEG_AUTHOR_CPU_HRSID_FIT1_NATIVE_MASK_PROBE_COMPLETE_V1",
                  author_commit=AUTHOR_COMMIT, author_code_config_files_exact=verified, checkpoint_sha256=WEIGHT_SHA,
                  checkpoint_bytes=48119167, model_size="s", stock_classes=stock.n_outputs, runtime=dependencies,
                  torch_cpu_threads=torch.get_num_threads(), input_image_lock_sha256=IMAGE_LOCK_SHA,
                  prior_closed_wait_sha256=P382_WAIT_SHA, encoded_image_sha256=selected["encoded_sha256"],
                  decoded_rgb_sha256=selected["decoded_rgb_sha256"], original_hw=[800, 800], network_input_hw=[640, 640],
                  masks_shape=list(masks.shape), boxes_shape=list(boxes.shape), mask_dtype="uint8",
                  nonempty_masks=int(masks.flatten(1).any(dim=1).sum().item()),
                  score_min=scores.min().item(), score_max=scores.max().item(),
                  scores_nonincreasing=bool((scores[1:] <= scores[:-1]).all().item()),
                  author_postprocessing=dict(top_queries=300, wrapper_default_confidence=0.5,
                                             real_probe_confidence=0.0, nms=True, nms_iou=0.7,
                                             mask_threshold=0.5, keep_ratio=stock.keep_ratio),
                  actual_native_forward_calls=2, synthetic_author_warmup_forwards=1, fit_png_forward_calls=1,
                  real_fit_forward_and_postprocess_seconds=real_seconds, total_model_and_real_seconds=perf_counter()-total_start,
                  prediction_tensor_sha256=streams, cuda_compiled=False, cuda_initialized=False, optimizer_updates=0,
                  FIT=True, GT_opened=False, DEV=False, TEST=False, training=False, AP=False)
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with (ROOT / "CPU_HRSID_FIT1_NATIVE_MASK_PROBE_SAFE.json").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=report["status"], result_sha256=sha256(raw).hexdigest(),
                          masks_shape=report["masks_shape"], nonempty_masks=report["nonempty_masks"],
                          real_seconds=report["real_fit_forward_and_postprocess_seconds"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
