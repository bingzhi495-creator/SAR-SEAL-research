"""Comparable four-class AP using real dataset labels and official pycocotools.

COCO pretrained category indices differ from native dataset indices. Both sides
are mapped explicitly to the SAME four evaluation categories, never normalized
by model predictions. Native 8-class SDK metrics are reported separately.
"""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent
COMMON = ["vehicle", "bus", "bicycle", "pedestrian"]


def evaluate_common(gt, raw_predictions, kind):
    mapping = {2: 1, 5: 2, 1: 3, 0: 4} if kind == "coco80" else {i: i + 1 for i in range(4)}
    predictions = [{"image_id": d["image_id"], "category_id": mapping[d["class_id"]],
                    "bbox": d["bbox"], "score": d["score"]}
                   for d in raw_predictions if d["class_id"] in mapping]
    coco = COCO()
    coco.dataset = gt
    coco.createIndex()
    if predictions:
        result = coco.loadRes(predictions)
    else:
        result = COCO()
        result.dataset = {"images": gt["images"], "categories": gt["categories"], "annotations": []}
        result.createIndex()
    evaluation = COCOeval(coco, result, "bbox")
    evaluation.params.catIds = [1, 2, 3, 4]
    evaluation.params.imgIds = [d["id"] for d in gt["images"]]
    evaluation.params.maxDets = [1, 10, 100]
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    per_class = {}
    for index, name in enumerate(COMMON):
        precision = evaluation.eval["precision"][:, :, index, 0, -1]
        at50 = precision[0]
        valid = precision[precision >= 0]
        valid50 = at50[at50 >= 0]
        tp = fp = total = 0
        for image in evaluation.evalImgs:
            if image is None or image["category_id"] != index + 1 or image["aRng"] != evaluation.params.areaRng[0]:
                continue
            keep = np.asarray(image["dtScores"]) >= 0.25
            ignore = np.asarray(image["dtIgnore"])[0].astype(bool)
            matched = np.asarray(image["dtMatches"])[0] > 0
            tp += int(np.sum(keep & ~ignore & matched))
            fp += int(np.sum(keep & ~ignore & ~matched))
            total += int(np.sum(~np.asarray(image["gtIgnore"]).astype(bool)))
        per_class[name] = {"map50_95": float(valid.mean()) if valid.size else None,
                           "map50": float(valid50.mean()) if valid50.size else None,
                           "precision_conf025_iou05": tp / (tp + fp) if tp + fp else 0.0,
                           "recall_conf025_iou05": tp / total if total else None,
                           "true_positives": tp, "false_positives": fp, "gt_objects": total}
    return {"map50": float(evaluation.stats[1]), "map50_95": float(evaluation.stats[0]),
            "per_class": per_class, "official_coco_stats": evaluation.stats.tolist(),
            "metric": "official pycocotools bbox AP, IoU .50:.95, 101 recall points, maxDets=100",
            "pr_operating_point": "confidence .25, IoU .50, class-wise COCO matching, maxDets=100",
            "evaluation_type": "real_dataset_gt", "images": len(gt["images"]),
            "gt_objects": len(gt["annotations"]), "prediction_count": len(predictions)}


def evaluate_checkpoint(weights, split, kind, output):
    from ultralytics import YOLO
    dataset = json.loads((ROOT / "prepared_dataset.json").read_text())[split]
    gt = {"info": {"source": "Intersection-Flow-5K official v4 YOLO labels, fixed pilot"},
          "images": [], "annotations": [], "categories": [{"id": i + 1, "name": n} for i, n in enumerate(COMMON)]}
    for image_id, entry in enumerate(dataset, 1):
        w, h = entry["width"], entry["height"]
        gt["images"].append({"id": image_id, "width": w, "height": h})
        for line in Path(entry["label"]).read_text().splitlines():
            if not line.strip():
                continue
            cls, x, y, bw, bh = map(float, line.split())
            if int(cls) >= 4:
                continue
            bbox = [(x - bw / 2) * w, (y - bh / 2) * h, bw * w, bh * h]
            gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": image_id,
                                       "category_id": int(cls) + 1, "bbox": bbox,
                                       "area": bbox[2] * bbox[3], "iscrowd": 0})
    model = YOLO(str(weights))
    classes = [0, 1, 2, 5] if kind == "coco80" else [0, 1, 2, 3]
    raw = []
    for offset in range(0, len(dataset), 16):
        batch = dataset[offset:offset + 16]
        results = model.predict([d["path"] for d in batch], imgsz=640, batch=16, rect=False,
                                conf=0.001, iou=0.7, max_det=300, classes=classes,
                                half=True, device=0, augment=False, verbose=False)
        for j, result in enumerate(results):
            for box in result.boxes.data.detach().cpu().tolist():
                raw.append({"image_id": offset + j + 1, "class_id": int(box[5]),
                            "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]], "score": box[4]})
    metrics = evaluate_common(gt, raw, kind)
    metrics.update({"checkpoint": str(weights), "split": split, "mapping_kind": kind,
                    "inference": {"input": 640, "batch": 16, "precision": "FP16", "conf": .001,
                                  "nms_iou": .7, "max_candidates": 300, "tta": False},
                    "mapping_scope": "vehicle~COCO car, bus, bicycle, pedestrian~COCO person; semantic equivalence not fully verified"})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "ground_truth_local_only.json").write_text(json.dumps(gt), encoding="utf-8")
    (output / "predictions_local_only.json").write_text(json.dumps(raw), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--kind", choices=["coco80", "native8"], required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    evaluate_checkpoint(a.weights, a.split, a.kind, a.output)
