"""Evaluate locked best checkpoint on the untouched official-split pilot test."""
from pathlib import Path
import json
from ultralytics import YOLO
from common_metrics import evaluate_checkpoint

ROOT = Path(__file__).resolve().parent


def main():
    completion = json.loads((ROOT / "training_completion.json").read_text())
    if completion["status"] != "training_returned" or completion["epoch_index"] != 29:
        raise RuntimeError("Fixed 30-epoch run has not completed")
    best = Path(completion["best"])
    # Same untouched test list and evaluator on both sides; no tuning follows.
    before = evaluate_checkpoint(ROOT / "yolo11n.pt", "test", "coco80", ROOT / "evaluation/before_common4")
    after = evaluate_checkpoint(best, "test", "native8", ROOT / "evaluation/after_common4")
    model = YOLO(str(best))
    native = model.val(data=str(ROOT / "pilot.yaml"), split="test", imgsz=640, batch=16,
                       device=0, workers=4, half=True, rect=False, conf=.001, iou=.7,
                       max_det=300, plots=True, project=str(ROOT / "evaluation"),
                       name="after_native8", exist_ok=False, verbose=False)
    class_indices = native.box.ap_class_index.tolist()
    native_per_class = {}
    for index, cls in enumerate(class_indices):
        native_per_class[model.names[int(cls)]] = {
            "map50_95": float(native.box.ap[index]), "map50": float(native.box.ap50[index]),
            "precision_at_sdk_selected_f1": float(native.box.p[index]),
            "recall_at_sdk_selected_f1": float(native.box.r[index])}
    summary = {"status": "evaluated", "hardware": "RTX3090Ti", "xavier_nx_tested": False,
               "before_common4": before, "after_common4": after,
               "delta_common4_map50_95_percentage_points": (after["map50_95"] - before["map50_95"]) * 100,
               "delta_common4_map50_percentage_points": (after["map50"] - before["map50"]) * 100,
               "after_native8_sdk": dict(native.results_dict), "native8_per_class": native_per_class,
               "scope": "single-seed, 1022/128/128 official-split pilot after two pre-metric training-only exclusions, 30 epochs; scene independence unverified",
               "gt": "real official dataset annotations; no pseudo labels used for evaluation",
               "notes": "Common4 uses official COCO maxDets100; native8 SDK max_det300 metrics reported separately. Native SDK P/R are at its selected F1 point, not fixed confidence .25."}
    (ROOT / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
