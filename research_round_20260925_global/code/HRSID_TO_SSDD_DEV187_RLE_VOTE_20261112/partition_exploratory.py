"""Post-primary exact original-match partition and independent stock P250 metric replay."""
import argparse
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import sys
from types import ModuleType

WORKER_SHA = "40bc85ef6fc6612e23674d2cba82cd69dce35865320c40175df5c187fcc2afcc"
RESULT_SHA = "8d8c11965433551ade41501adcd0a2af77cc7c5f7de81e731f6da1f2d787497a"
PRELABEL_SHA = "1c8f06e33d74a4c97d9861ddf21da0342f8cc510d6fe81be2b7075dcc47ef49f"
READBACK_SHA = "44a4c65137d5ff9e89a50e0f11f9158d9d7c3ef1f0e8fcf6f483b526781a0d8a"
SCORE_TERMINAL = "b8c57c145f3ce891204b46147c5d9bdb5cb115e913fcc29787f0399df356b1ba"


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def pinned(path, digest):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, "Exact closed digest: " + path.name)
    return raw


def frozen_status(data, rows, ids):
    from copy import deepcopy
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with redirect_stdout(StringIO()):
        gt = COCO()
        gt.dataset = deepcopy(data)
        gt.createIndex()
        dt = gt.loadRes(deepcopy(rows))
        ev = COCOeval(gt, dt, "segm")
        ev.params.imgIds = ids
        ev.params.catIds = [1]
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
    status = {}
    for block in ev.evalImgs[:len(ids)]:
        if block is None:
            continue
        for rank, identity, ignored in zip(block["dtIds"], block["dtMatches"][0], block["dtIgnore"][0], strict=True):
            status[int(rank)-1] = "ignored" if ignored else "matched" if identity else "unmatched"
    require(len(status) == len(rows), "Exact raw candidate match scope")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("worker", "prelabel", "readback", "score", "score-terminal", "plan", "independent", "code",
                  "protocol", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--protocol-sha", required=True)
    args = parser.parse_args()
    pinned(args.protocol, args.protocol_sha)
    actual = json.loads(pinned(args.score, RESULT_SHA))
    terminal = json.loads(pinned(args.score_terminal, SCORE_TERMINAL))
    require(terminal["original_os_exit"] == 0 and terminal["output_sha256"] == RESULT_SHA
            and terminal["child_closed"] and not terminal["timed_out"] and actual["advance"] is False,
            "Closed original primary decision")
    worker = ModuleType("_frozen_vote_worker")
    worker.__file__ = str(args.worker)
    exec(compile(pinned(args.worker, WORKER_SHA), str(args.worker), "exec"), worker.__dict__)
    args.prelabel_sha = PRELABEL_SHA
    receipt, variants = worker.read_variants(args)
    readback = json.loads(pinned(args.readback, READBACK_SHA))
    require(receipt["GT_opened"] is readback["GT_opened"] is False
            and readback["all_edit_invariants"] and readback["variants_sha256"] == receipt["variants_sha256"],
            "Exact independent prelabel closure")
    original = worker.load_score(args.code)
    plan, independent, ids, items, arms, scorer, official = worker.read_inputs(args, original, (5202,))
    raw, changes = [], []
    for image, block, edited in zip(ids, arms[5202], variants, strict=True):
        require(image == block["image_id"] == edited["image_id"], "Exact ordered image identity")
        raw.extend(block["coco"])
        changes.extend(edited["variants"].get(str(r), {}) for r in block["native_ranks"])
    require(len(raw) == len(changes) == receipt["totals"]["mapped_candidates"], "Exact original candidate list")
    gt = original.document(original.read_ref(plan["integer_gt"], original.MIB))
    scorer.validate_gt(gt, ids, items)
    membership = frozen_status(gt, raw, ids)
    masks = dict(vote=["matched", "unmatched", "ignored"], anchor_distance=["matched", "unmatched", "ignored"],
                 vote_matched=["matched"], vote_unmatched=["unmatched"])
    counts = {name: sum(bool(change) and membership[i] == name for i, change in enumerate(changes))
              for name in ("matched", "unmatched", "ignored")}
    require(sum(counts.values()) == receipt["totals"]["edited"], "Exact fixed stock edit partition")
    metrics = {"raw": scorer.score_one(official, gt, [dict(row) for row in raw], ids)}
    for mode, scope in masks.items():
        key = "anchor_distance" if mode == "anchor_distance" else "vote"
        rows = [dict(row, segmentation=change[key]) if change and membership[i] in scope else dict(row)
                for i, (row, change) in enumerate(zip(raw, changes, strict=True))]
        metrics[mode] = scorer.score_one(official, gt, rows, ids)
    for mode in ("raw", "vote", "anchor_distance"):
        for key in original.METRICS:
            require(abs(metrics[mode][key] - actual["metrics"][mode][key]) <= 1e-12,
                    "Exact independent original stock endpoint parity: " + mode + "/" + key)
    result = dict(status="HRSID_TO_SSDD_DEV187_RLE_VOTE_MATCH_PARTITION_EXPLORATORY", DEV=True, TEST=False,
                  AP=True, model_forwards=0, optimizer_updates=0, prior_advance=False, independently_reproduced=True,
                  prior_result_sha256=RESULT_SHA, protocol_sha256=args.protocol_sha, original_match_edit_counts=counts,
                  metrics=metrics, AP_delta_vs_raw_points={m: 100*(v["AP"]-metrics["raw"]["AP"])
                                                          for m,v in metrics.items() if m != "raw"})
    digest = worker.write_new(args.output, worker.stable_json(result))
    print(json.dumps(dict(status=result["status"], output_sha256=digest, edits=counts,
                          AP_delta_vs_raw_points=result["AP_delta_vs_raw_points"]), sort_keys=True))


if __name__ == "__main__":
    main()