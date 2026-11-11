from argparse import ArgumentParser
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
from pathlib import Path
from time import perf_counter
import json
import os

MODES = ("stock", "low_threshold")
INTEGER = "5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55"
FLOAT = "8d98e6f49200d5390c17bee66697de1f582133d08063259a561ed5f710be7b6c"

def require(condition, detail):
    if not condition:
        raise RuntimeError(detail)

def fixed(path, digest):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == digest, "Closed input SHA differs: " + path.name)
    return data

def score(coco, eval_type, np, labels, rows):
    with redirect_stdout(StringIO()):
        gt = coco()
        gt.dataset = {**labels, "info": labels.get("info", {})}
        gt.createIndex()
        result = gt.loadRes([dict(row) for row in rows])
        e = eval_type(gt, result, "segm")
        e.params.imgIds = sorted(gt.getImgIds())
        e.params.catIds = [1]
        e.params.maxDets = [1, 10, 100]
        e.evaluate()
        e.accumulate()
        e.summarize()
    index = np.flatnonzero(np.isclose(e.params.iouThrs, 0.90))
    require(len(index) == 1, "One fixed AP90 threshold")
    precision = e.eval["precision"][index.item(), :, :, 0, -1]
    p90 = float(precision[precision >= 0].mean())
    return {"AP": float(e.stats[0]), "AP50": float(e.stats[1]), "AP75": float(e.stats[2]),
            "AP90": p90, "APs": float(e.stats[3]), "AR100": float(e.stats[8]), "all12": e.stats.tolist()}

def main(argv=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np

    parser = ArgumentParser(description="Replay fixed SRNet DEV187 predictions under both admitted label conventions")
    for name in ("original-result", "integer-labels", "float-labels", "stock-predictions", "low-threshold-predictions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    start = perf_counter()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only")
    closed = json.loads(args.original_result.read_bytes())
    require(closed["primary_arm"] == "stock" and closed["sensitivity_arm"] == "low_threshold"
            and closed["integer_gt_sha256"] == INTEGER and closed["images"] == 187
            and closed["annotations"] == 380 and closed["max_dets"] == [1, 10, 100]
            and closed["TEST"] is False, "Original endpoint qualification")
    labels = [
        json.loads(fixed(args.integer_labels, INTEGER)),
        json.loads(fixed(args.float_labels, FLOAT)),
    ]
    require(labels[0]["images"] == labels[1]["images"] and labels[0]["categories"] == labels[1]["categories"]
            and len(labels[0]["images"]) == 187 and len(labels[0]["annotations"]) == len(labels[1]["annotations"]) == 380,
            "Exact common frozen GT images/category and counts required")
    for a, b in zip(labels[0]["annotations"], labels[1]["annotations"], strict=True):
        require(a.keys() == b.keys() and all(a[k] == b[k] for k in a if k not in ("segmentation", "bbox", "area")),
                "Only official restored GT geometry may differ")
    ids = {row["id"] for row in labels[0]["images"]}
    arms = {}
    for mode in MODES:
        prior = closed["arms"][mode]
        raw = fixed(getattr(args, mode + "_predictions"), prior["predictions_sha256"])
        records = [json.loads(line) for line in raw.splitlines()]
        require(len(records) == 187 and {row["image_id"] for row in records} == ids, "Exact DEV prediction roster")
        rows = [item for record in records for item in record["predictions"]]
        require(len(rows) == prior["candidates"] and all(row["image_id"] in ids and row["category_id"] == 1
                for row in rows), "Exact candidate population")
        a = score(COCO, COCOeval, np, labels[0], rows)
        require(a == prior["metrics"], "All fixed historical integer metrics must replay exactly")
        b = score(COCO, COCOeval, np, labels[1], rows)
        arms[mode] = {"score_threshold": prior["score_threshold"], "predictions_sha256": prior["predictions_sha256"],
                      "candidates": len(rows), "integer": a, "official_train_float_dev": b,
                      "float_minus_integer": {key: b[key] - a[key] for key in a if key != "all12"}}
    report = {"status": "SRNET_FIXED_DEV187_DUAL_GT_REPLAY_ACTUAL", "dev": True, "test": False, "fit": False,
              "checkpoint_sha256": closed["checkpoint_sha256"], "integer_gt_sha256": INTEGER, "float_gt_sha256": FLOAT,
              "images": 187, "annotations": 380, "max_dets": [1, 10, 100], "primary_arm": "stock", "arms": arms,
              "model_forward_calls": 0, "training_updates": 0, "seconds": perf_counter() - start}
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as file:
        file.write(payload)
    print(json.dumps({"result_sha256": sha256(payload).hexdigest(), "seconds": report["seconds"],
                      "arms": {k: v["float_minus_integer"] for k, v in arms.items()}}))

if __name__ == "__main__":
    main()