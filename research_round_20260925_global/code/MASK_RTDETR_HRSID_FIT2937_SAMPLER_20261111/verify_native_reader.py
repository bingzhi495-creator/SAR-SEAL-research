"""Verify the fixed FIT2937 padding through the official PaddleDetection reader."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import chain
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import paddle

from fit2937_batch_sampler import Fit2937BatchSampler


def demand(condition, detail):
    if not condition:
        raise ValueError(detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "fit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--index-sha", required=True)
    parser.add_argument("--reader-sha", required=True)
    args = parser.parse_args()
    index_path = args.fit / "train/_annotations.coco.json"
    index_bytes = index_path.read_bytes()
    reader_path = args.source / "ppdet/data/reader.py"
    demand(sha256(index_bytes).hexdigest() == args.index_sha, "Unexpected FIT index")
    demand(sha256(reader_path.read_bytes()).hexdigest() == args.reader_sha, "Unexpected author reader")
    source_ids = {row["id"] for row in json.loads(index_bytes)["images"]}
    demand(len(source_ids) == 2937, "Expected FIT2937 only")
    demand(paddle.__version__ == "2.6.2" and not paddle.is_compiled_with_cuda(), "Exact CPU Paddle required")
    paddle.set_device("cpu")
    sys.path.insert(0, str(args.source))
    from ppdet.core.workspace import create, load_config
    from ppdet.data import reader

    demand(Path(reader.__file__).resolve() == reader_path.resolve(), "Expected pinned reader import")
    config = args.source / "configs/mask_rtdetr/mask_rtdetr_hgnetv2_s_6x_coco.yml"
    cfg = load_config(str(config))
    cfg.update(use_gpu=False, num_classes=1, worker_num=0)
    cfg["TrainDataset"].update(dataset_dir=str(args.fit), image_dir="train", anno_path="train/_annotations.coco.json")
    cfg["TrainReader"]["use_shared_memory"] = False
    dataset = create("TrainDataset")()

    def forbid_decode(*args, **kwargs):
        raise ValueError("Image decoding is out of scope")

    with patch.object(cv2, "imread", forbid_decode), patch.object(cv2, "imdecode", forbid_decode):
        with patch.object(reader, "DistributedBatchSampler", Fit2937BatchSampler):
            loader = create("TrainReader")(dataset, 0)
    sampler = loader._batch_sampler
    native = paddle.io.DistributedBatchSampler(dataset, batch_size=4, shuffle=True, drop_last=False)
    ordered_ids = [int(np.asarray(row["im_id"]).reshape(-1)[0]) for row in dataset.roidbs]
    demand(len(ordered_ids) == len(set(ordered_ids)) == 2937 and set(ordered_ids) == source_ids, "Native COCO identity differs")
    demand(type(sampler) is Fit2937BatchSampler and len(loader) == 735, "Unexpected real loader count")
    epoch_digests = []
    for epoch in range(24):
        sampler.set_epoch(epoch)
        native.set_epoch(epoch)
        native_batches = list(native)
        batches = list(sampler)
        demand(len(batches) == 735 and all(len(batch) == 4 for batch in batches), "Incomplete fixed batch plan")
        demand(batches[:-1] == native_batches[:-1] and batches[-1] == native_batches[-1] + native_batches[0][:3],
               "Stock permutation was changed")
        visits = [ordered_ids[index] for index in chain.from_iterable(batches)]
        counts = Counter(visits)
        demand(len(counts) == 2937 and len(visits) == 2940 and counts.most_common(3) == [(i, 2) for i in visits[:3]],
               "Unexpected padded population")
        epoch_digests.append(sha256(json.dumps(visits, separators=(",", ":")).encode()).hexdigest())
    demand(len(set(epoch_digests)) == 24, "Epoch order did not advance")
    report = {
        "status": "MASK_RTDETR_HRSID_FIT2937_NATIVE_READER_PAD_PLAN_VERIFIED",
        "images": len(ordered_ids), "reader_batches_per_epoch": len(loader), "visits_per_epoch": 2940,
        "padding_per_epoch": 3, "epochs_planned": 24, "planned_updates": 17640, "planned_visits": 70560,
        "planned_padding": 72, "reader_source_sha256": args.reader_sha, "fit_index_sha256": args.index_sha,
        "aggregate_planned_order_sha256": sha256("".join(epoch_digests).encode()).hexdigest(),
        "paddle": paddle.__version__, "png_decodes": 0, "model_forward_calls": 0,
        "optimizer_updates": 0, "DEV": False, "TEST": False,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
