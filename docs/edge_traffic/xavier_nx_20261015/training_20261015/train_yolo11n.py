"""Fixed-budget native-eight-class fine-tuning. See TRAINING_PLAN.md."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runtime_settings"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("COMET_MODE", "DISABLED")
import json
import torch
from ultralytics import YOLO, settings

def main():
    settings.update({key: False for key in ["sync", "hub", "wandb", "comet", "clearml", "dvc",
                                          "mlflow", "neptune", "raytune", "tensorboard"] if key in settings})
    audit = json.loads((ROOT / "dataset_audit_summary.json").read_text())
    if audit["status"] != "audited_ready" or audit["exact_cross_split_duplicates"]:
        raise RuntimeError("Dataset has not passed pretraining audit")
    weights = ROOT / "yolo11n.pt"
    config = dict(data=str(ROOT / "pilot.yaml"), epochs=30, imgsz=640, batch=16,
                  device=0, workers=4, cache=False, seed=42, deterministic=True,
                  optimizer="SGD", lr0=.01, lrf=.01, momentum=.937, weight_decay=.0005,
                  warmup_epochs=3, patience=31, close_mosaic=5, amp=True,
                  project=str(ROOT / "runs"), name="yolo11n_pilot30", exist_ok=False,
                  save=True, save_period=10, plots=True, val=True, verbose=False)
    (ROOT / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = YOLO(str(weights))
    results = model.train(**config)
    summary = {"status": "training_returned", "result_directory": str(model.trainer.save_dir),
               "best": str(model.trainer.best), "last": str(model.trainer.last),
               "epoch_index": model.trainer.epoch, "metrics": dict(results.results_dict),
               "hardware": torch.cuda.get_device_name(0), "training_seed": 42,
               "split_counts": {s: x["images"] for s, x in audit["splits"].items()},
               "xavier_nx_tested": False, "electric_bicycle_independently_defined": False}
    (ROOT / "training_completion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
