"""Single-stream, batch-one detector test; no tracking and no frame skipping.

The pipeline timer includes read/decode, prediction, and CPU box transfer.
Rendering, JSON serialization, and video writing have a separate wall timer.
Unlabelled videos do not produce accuracy or recall claims.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("frames must be positive and warmup nonnegative")
    if not Path(args.model).is_file():
        parser.error("Download the explicit model file first; implicit downloads are disabled")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    names = dict(model.names)
    desired = {"person", "bicycle", "car", "motorcycle", "bus", "truck",
               "ebike", "e-bike", "electric bicycle", "electric_bicycle",
               "vehicle", "pedestrian", "engine", "tricycle", "obstacle"}
    classes = [index for index, name in names.items() if name.lower() in desired]
    if not classes:
        raise ValueError("Model has no supported traffic class names")
    ebike_supported = any(names[index].lower() in {"ebike", "e-bike", "electric bicycle", "electric_bicycle"} for index in classes)
    gpu = args.device != "cpu"
    hardware = torch.cuda.get_device_name(0) if gpu else platform.processor()
    board_file = Path("/proc/device-tree/model")
    board = board_file.read_text().strip("\x00") if board_file.is_file() else ""
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if 0 < fps <= 240 else 30.0
    ok, first = capture.read()
    if not ok:
        raise RuntimeError("Source contains no readable frames")
    height, width = first.shape[:2]
    options = dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                   device=args.device, half=args.half, classes=classes,
                   rect=False, augment=False, verbose=False, max_det=300)

    def synchronize():
        if gpu:
            torch.cuda.synchronize()

    for _ in range(args.warmup):
        model.predict(first, **options)
    synchronize()
    # Reopen so the first reported frame is decoded inside the measured loop.
    capture.release()
    capture = cv2.VideoCapture(source)
    writer = cv2.VideoWriter(str(output / "annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot initialize annotated video writer")
    counts = Counter()
    frames_with_class = Counter()
    latencies = []
    sdk_speeds = []
    snapshots = {0, args.frames // 3, 2 * args.frames // 3, args.frames - 1}
    started = time.perf_counter()
    records_file = output / "detections.jsonl"
    with records_file.open("w", encoding="utf-8") as records:
        for frame_index in range(args.frames):
            synchronize()
            tick = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(frame, **options)[0]
            boxes = result.boxes.data.detach().cpu().numpy()
            synchronize()
            elapsed_ms = (time.perf_counter() - tick) * 1000
            latencies.append(elapsed_ms)
            sdk_speeds.append(result.speed)
            frame_counts = Counter(names[int(box[5])] for box in boxes)
            counts.update(frame_counts)
            frames_with_class.update(frame_counts.keys())
            records.write(json.dumps({
                "frame_index": frame_index, "source_time_seconds": frame_index / fps,
                "pipeline_ms_without_render_encoding": elapsed_ms,
                "detections": [{"class_id": int(b[5]), "class_name": names[int(b[5])],
                                "confidence": float(b[4]), "xyxy": b[:4].tolist()}
                               for b in boxes],
            }) + "\n")
            annotated = result.plot(line_width=2, font_size=12)
            cv2.rectangle(annotated, (0, 0), (width, 64), (20, 20, 20), -1)
            cv2.putText(annotated, f"{Path(args.model).stem} | {hardware} | frame {frame_index}",
                        (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, "Model traffic labels | E-BIKE NOT SEPARATELY VERIFIED" if not ebike_supported
                        else "Model contains an E-bike class; accuracy still requires ground truth",
                        (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 1, cv2.LINE_AA)
            writer.write(annotated)
            if frame_index in snapshots:
                cv2.imwrite(str(output / f"frame_{frame_index:05d}.jpg"), annotated)
            if args.show:
                cv2.imshow("Traffic detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    capture.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    wall_seconds = time.perf_counter() - started
    if not latencies:
        raise RuntimeError("No frames were inferred")
    frame_count = len(latencies)
    summary = {
        "status": "executed", "model": Path(args.model).name,
        "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        "source": str(args.source), "source_resolution": [width, height],
        "source_fps": fps, "frames_inferred": frame_count,
        "source_duration_tested_seconds": frame_count / fps,
        "skipped_frames": 0, "batch_size": 1, "network_input": [args.imgsz, args.imgsz],
        "precision": "FP16" if args.half else "FP32", "confidence_threshold": args.conf,
        "nms_iou_threshold": args.iou, "max_det": 300, "warmup_frames": args.warmup,
        "hardware": hardware, "board": board,
        "runtime": "TensorRT" if Path(args.model).suffix == ".engine" else "PyTorch (not TensorRT)",
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "ultralytics_version": ultralytics.__version__, "opencv_version": cv2.__version__,
        "platform": platform.platform(), "python": platform.python_version(),
        "selected_class_names": {str(c): names[c] for c in classes},
        "ebike_supported": ebike_supported,
        "accuracy_evaluation": "not_evaluated_no_ground_truth",
        "count_note": "Detection boxes summed over frames; not unique objects, not true positives",
        "box_counts_over_frames": dict(counts), "frames_with_each_class": dict(frames_with_class),
        "pipeline_fps_without_encoding": frame_count / (sum(latencies) / 1000),
        "pipeline_latency_ms_p50": float(np.percentile(latencies, 50)),
        "pipeline_latency_ms_p95": float(np.percentile(latencies, 95)),
        "sdk_mean_ms": {key: float(np.mean([s[key] for s in sdk_speeds]))
                        for key in ("preprocess", "inference", "postprocess")},
        "wall_seconds_with_render_encoding": wall_seconds,
        "wall_fps_with_render_encoding": frame_count / wall_seconds,
        "timer_note": "Pipeline includes decode, preprocess, inference, NMS and CPU box transfer; excludes rendering/JSON/video encoding. Wall includes these, excludes model loading and warmup. No live-network/display latency claim.",
        "xavier_nx_tested": "xavier nx" in board.lower(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
