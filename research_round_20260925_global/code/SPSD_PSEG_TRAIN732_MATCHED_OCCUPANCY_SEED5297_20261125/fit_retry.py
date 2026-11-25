"""Fit the four fixed-capacity occupancy controls on the admitted FIT partition."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
import torch

PROTOCOL = "4030a3cb5684a6dac0ea1e7765a48d05caec113c5c6e0e6de2c82b991e9d5f15"
CAPTURE = "e461828b04f166d98030210fe42df0d590d24e2452ea5da8630fb0fc3219d795"
POINTS = "87693897f873a2129764781611b0f7c35cadde51c0c4e830d7f8187c3b684049"
POINT_WORKER = "926f87b220c4ba9f1a7074088a52641cd66c17d4016bac91fba20c84f450f766"
LABELS = "bd4eb3add418d876f1defee88352aeee26c4f1b3135700436851fe73ad9b2796"
LABEL_READBACK = "de9455ecd6ac4db9fd3968c66e3039fb410efe3c6d5cc3f42f51c7eed893ff5e"
CORE = "0f1537f12ed4890bb60f71957e5d81926aeaab4760c8b01291383e4b5c53ee0d"
STATUS = "SPSD_PSEG_TRAIN732_MATCHED_OCCUPANCY_FIT_COMPLETE"
SPLIT = "91095aacbbbc89003ffd6eb8f25506ce99a64eee5ac369a7753203bbf962bcf9"


def require(flag, details):
    if not flag:
        raise ValueError(details)


def digest(blob):
    return sha256(blob).hexdigest()


def pinned(path, expected):
    data = path.read_bytes()
    require(digest(data) == expected, "Bound artifact changed: " + path.name)
    return data


def read_inputs(args):
    acceptance = json.loads(pinned(args.capture / "READBACK_SAFE.json", CAPTURE))
    before = json.loads(pinned(args.points / "actual_retry/RESULT_SAFE.json", POINTS))
    exit_record = json.loads((args.points / "observed_points_retry/TERMINAL.json").read_bytes())
    require(exit_record["actual_exit"] == 0 and exit_record["output_footer_verified"]
            and exit_record["worker_sha256"] == POINT_WORKER and exit_record["result_sha256"] == POINTS,
            "Complete observed point preparation")
    require(before["worker_sha256"] == POINT_WORKER and before["capture_readback_sha256"] == CAPTURE
            and before["group_partition_sha256"] == SPLIT
            and before["train_labels_opened"] is False and before["test_labels_opened"] is False
            and before["detector_forwards"] == before["optimizer_updates"] == 0
            and acceptance["identical_closed_candidate_streams"] and acceptance["original_os_exit"] == 0
            and acceptance["FIT"] and not acceptance["TEST"], "Exact independent frozen TRAIN candidate acceptance")
    labels = json.loads(pinned(args.labels / "export/RESULT_SAFE.json", LABELS))
    label_readback = json.loads(pinned(args.labels / "READBACK_SAFE.json", LABEL_READBACK))
    require(label_readback["complete_readback"] and label_readback["result_sha256"] == LABELS
            and labels["partition_sha256"] == SPLIT and labels["test_annotation_members_opened"] == 0
            and labels["partitions"] == label_readback["partitions"], "Accepted exact train-only label projection")
    fit = args.labels / "export/fit"
    reference = labels["partitions"]["fit"]
    require(reference["images"] == 732 and reference["decoded_groups"] == 721,
            "Only fit partition labels can feed the worker")
    hashes = reference["artifact_hashes"]
    pseg = json.loads(pinned(fit / "PSeg_SSDD_COCO_PRIVATE.json", hashes["pseg"]))
    spsd = json.loads(pinned(fit / "SPSD_COCO_PRIVATE.json", hashes["spsd"]))
    binding = json.loads(pinned(fit / "IMAGE_BINDINGS_PRIVATE.json", hashes["binding"]))
    require(binding["partition"] == "fit" and binding["source_partition_sha256"] == SPLIT,
            "Exact FIT label binding")
    cfg = before["partitions"]["fit"]
    index = [json.loads(line) for line in pinned(
        args.points / "actual_retry/fit/POINT_INDEX_PRIVATE.jsonl", cfg["point_index_sha256"]).splitlines()]
    require(len(index) == 732 and len({r["image_id"] for r in index}) == 732, "Exact FIT point support index")
    raw = [json.loads(line) for line in pinned(
        args.capture / "capture/DEV_MASK_PREDICTIONS_PRIVATE.jsonl", acceptance["predictions_sha256"]).splitlines()]
    require(len(raw) == 915, "Complete pre-label raw source prediction rows")
    native = {}
    for row in index:
        stock = raw[row["native_archive_ordinal"]-1]
        require(stock["image_id"] == row["image_id"] and len(stock["coco"]) >= 100,
                "Exact source native score order")
        native[row["image_id"]] = stock["coco"][:100]
    require(len(native) == 732 and sum(map(len, native.values())) == 73200, "Complete FIT-only candidate stream")
    images = {row["id"]: row for row in pseg["images"]}
    require(len(images) == 732 and images.keys() == native.keys() and pseg["images"] == spsd["images"]
            and len(pseg["annotations"]) == 1579 and len(spsd["annotations"]) == 1580, "Same FIT reference roster")
    ids = {k: binding["image_bindings"][im["file_name"]] for k, im in images.items()}
    for row in index:
        im = ids[row["image_id"]]
        require(im["image_id"] == row["image_id"] and [im["height"], im["width"]] == row["original_hw"],
                "Fit reference and point frame alignment")
    return (pseg, spsd), ids, index, native


def assignment(gt, native):
    records, lookup = [], {}
    for image_id, stream in native.items():
        for rank, row in enumerate(stream, 1):
            require(row["image_id"] == image_id and row["category_id"] == 1, "Exact native score entry")
            records.append(dict(image_id=image_id, category_id=0, score=row["score"],
                                segmentation=deepcopy(row["segmentation"])))
            lookup[len(records)] = (image_id, rank)
    with redirect_stdout(StringIO()):
        coco = COCO()
        coco.dataset = deepcopy(gt)
        coco.createIndex()
        ev = COCOeval(coco, coco.loadRes(records), "segm")
        ev.params.catIds = [0]
        ev.params.imgIds = sorted(native)
        ev.params.iouThrs = np.array([.50])
        ev.evaluate()
    large = ev.params.areaRng[0]
    matched = {}
    for img in ev.evalImgs:
        if not img or img["aRng"] != large or img["maxDet"] != 100:
            continue
        for gt_id, pred_id, ignored in zip(img["gtIds"], img["gtMatches"][0], img["gtIgnore"], strict=True):
            if pred_id and not ignored:
                require(int(gt_id) not in matched, "Native matching is one-to-one")
                matched[int(gt_id)] = lookup[int(pred_id)]
    return matched


def eligible_rows(args, refs, bindings, index, native):
    matches = [assignment(gt, native) for gt in refs]
    authored = [defaultdict(list), defaultdict(list)]
    for side, gt in zip(authored, refs, strict=True):
        for label in gt["annotations"]:
            side[label["image_id"]].append(label)
    features, targets, groups, images = [], [], [], []
    counts = Counter(pseg_native_match=len(matches[0]), spsd_native_match=len(matches[1]))
    for meta in index:
        iid = meta["image_id"]
        h, w = meta["original_hw"]
        lists = [side[iid] for side in authored]
        require(all(lists), "All admitted FIT images have both references")
        rasters = [[item["segmentation"] for item in side] for side in lists]
        overlaps = mask.iou(*rasters, [0] * len(rasters[1]))
        p_matches, s_matches = linear_sum_assignment(-overlaps)
        cases = Counter()
        by_rank = {}
        for i, j in zip(p_matches, s_matches, strict=True):
            cases["reference_pairs"] += 1
            if overlaps[i, j] < .50:
                continue
            cases["reference_pairs_ge050"] += 1
            p, s = lists[0][i], lists[1][j]
            left, right = matches[0].get(p["id"]), matches[1].get(s["id"])
            if left is None or left != right:
                continue
            require(left[0] == iid and left[1] not in by_rank, "Same-image shared native rank")
            by_rank[left[1]] = (p["segmentation"], s["segmentation"])
            cases["shared_native_match"] += 1

        expected_file = f"POINTS_PRIVATE/{meta['native_archive_ordinal']:04}.npz"
        require(meta["filename"] == expected_file, "Only the closed FIT point file for its original ordinal")
        data = pinned(args.points / "actual_retry/fit" / expected_file, meta["sha256"])
        require(len(data) == meta["bytes"], "Exact FIT point byte count")
        with np.load(BytesIO(data), allow_pickle=False) as batch:
            ranks = batch["native_ranks"]
            offsets = batch["point_offsets"]
            coordinates = batch["signed_coordinates"]
            values = batch["features"]
        require(ranks.ndim == 1 and len(ranks) == meta["supported_candidates"]
                and len(set(ranks.tolist())) == len(ranks) and all(1 <= rank <= 100 for rank in ranks)
                and offsets.shape == (len(ranks)+1,) and offsets[0] == 0 and np.all(np.diff(offsets)>0)
                and offsets[-1] == meta["paired_sites"]
                and coordinates.shape == (offsets[-1],2) and values.shape == (offsets[-1],2,14)
                and values.dtype == np.float32 and np.isfinite(values).all()
                and np.all((0 <= coordinates) & (coordinates < h*w)), "Admitted exact FIT endpoint arrays")
        for slot, rank in enumerate(ranks):
            if rank not in by_rank:
                continue
            start, end = map(int, offsets[slot:slot+2])
            p, s = by_rank[rank]
            occ = np.stack([(mask.decode(rle).ravel(order="C")[coordinates[start:end]]) for rle in (p,s)],axis=-1)
            require(occ.shape == (end-start,2,2) and np.all((occ==0)|(occ==1)), "Binary signed reference features")
            features.append(values[start:end].copy())
            targets.append(occ)
            groups.extend([bindings[iid]["decoded_group_sha256"]] * (end-start))
            images.extend([iid] * (end-start))
            cases["supported_training_hulls"] += 1
            cases["training_pairs"] += end-start
            cases["reference_disagreeing_points"] += int(np.count_nonzero(occ[:,:,0]!=occ[:,:,1]))
        counts.update(cases)
    require(len(features) == counts["supported_training_hulls"] and len(groups) == counts["training_pairs"] > 0,
            "Nonempty exact FIT-only training point population")
    counts["training_image_records"] = len(set(images))
    counts["training_decoded_groups"] = len(set(groups))
    return np.concatenate(features),np.concatenate(targets),groups,images,dict(counts)


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("protocol","capture","points","labels","core","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","CPU-only matched fitting")
    start=perf_counter()
    pinned(args.protocol,PROTOCOL)
    pinned(args.core,CORE)
    train = run_path(str(args.core))["train_matched"]
    refs, bindings, index, native = read_inputs(args)
    features, targets, groups, images, exposure = eligible_rows(args,refs,bindings,index,native)
    args.output.mkdir()
    seal=dict(status="DUAL_REFERENCE_FIT_ONLY_SUPERVISION_ACCEPTED", exposure=exposure,
              features_sha256=digest(features.tobytes()), labels_sha256=digest(targets.tobytes()),
              partition="fit", source_gate_labels_opened=False, historical_test_opened=False)
    raw=(json.dumps(seal,sort_keys=True,allow_nan=False)+"\n").encode()
    with (args.output/"FIT_INPUT_SAFE.json").open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    models,center,scale,audit=train(features,targets,groups,images,seed=5297)
    require(set(models)=={"dual","pseg_repeat","spsd_repeat","mean_repeat"} and len(audit["heads"])==4
            and all(v["optimizer_updates"]==1000 and v["parameters"]==1346 for v in audit["heads"].values()),
            "Real matched optimizer step and architecture parity")
    weight=args.output/"FITTED_WEIGHTS_PRIVATE.pt"
    with weight.open("xb") as out:
        torch.save(dict(models={k:v.state_dict() for k,v in models.items()},center=torch.from_numpy(center),
                        scale=torch.from_numpy(scale),seed=5297),out)
        out.flush(); os.fsync(out.fileno())
    doc=dict(status=STATUS, worker_sha256=digest(Path(__file__).read_bytes()),
             capture_readback_sha256=CAPTURE, points_result_sha256=POINTS, fit_labels_readback_sha256=LABEL_READBACK,
             protocol_sha256=PROTOCOL, training_core_sha256=CORE, output_checkpoint_sha256=digest(weight.read_bytes()),
             admission_sha256=digest(raw), exposure=exposure, source_gate_labels_opened=False,
             historical_test_opened=False, model_forwards_detector=0, total_optimizer_updates=4000,
             source_gate_scored=False, elapsed_seconds=perf_counter()-start, train_audit=audit)
    payload=(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with (args.output/"RESULT_SAFE.json").open("xb") as out:
        out.write(payload);out.flush();os.fsync(out.fileno())
    print(json.dumps(dict(status=STATUS,result_sha256=digest(payload))),flush=True)


if __name__=="__main__":
    main()
