"""Score the five frozen D-FINE arms on the full and predeclared exact-overlap sensitivity PSeg cohorts."""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import math
import os
from pathlib import Path
import platform
import re
import sys

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ARMS = ("raw", "fixed", "free_matched_count", "free_natural", "native_matched_count")
METRICS = ("AP","AP50","AP75","APs","APm","APl","AR1","AR10","AR100","ARs","ARm","ARl")
GT_SHA = "021d52858ffa3fda9a2d28ee21f1ca114800554edf08007704f6d06a412d50bf"
INVENTORY_SHA = "ace3e722369657d9dd6a552eeac13244b3fa88772701ae0769e676816b46152e"
OVERLAP_AUDIT_SHA = "c17d1546afbc8b595f73b926e4968a67c5512b2672fc3443ba85f3fba426c1e7"
EXCLUDED = tuple(n+".jpg" for n in ("000001","000011","000229","000389","000501","000631","000639","000641","000739","001121"))
EXCLUSION_SHA = "22d034a3acf04dcc01eeb19469b0552e292206a4b61c42a3ff275a19be30ab67"
ACCEPTED = "DFINE_PSEG_TEST232_FROZEN_POINT_REPLAY_INDEPENDENTLY_ACCEPTED_V1"
EMA = "82ca80e2cb9a0b9673a136fad3d7de3097b7327f4f76e3a2d3e7184ab07e7fd8"
HEAD = "dca00310ba7398790a874e2f32c7fa26324fed1b8e071372831a85e7a83ecd3a"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def valid_sha(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def pinned(path, expected, limit):
    require(valid_sha(expected), "A closed artifact SHA-256 is required")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    require(0 < len(data) <= limit and digest(data) == expected, "Bounded exact bytes for " + path.name)
    return data


def check_acceptance(receipt):
    require(receipt.get("status") == ACCEPTED and receipt.get("TEST") is True
            and all(receipt.get(k) is False for k in ("FIT","DEV","GT_opened","AP"))
            and receipt.get("exact_native_support") is True and receipt.get("images") == 232
            and receipt.get("ema_sha256") == EMA and receipt.get("refit_accepted_sha256") == HEAD
            and receipt.get("modes") == list(ARMS) and type(receipt.get("optimizer_updates")) is int and receipt["optimizer_updates"] == 0
            and all(valid_sha(receipt.get(k)) for k in ("manifest_sha256","acceptance_sha256","replay_sha256",
                    "edits_sha256","input_predictions_sha256","candidate_audit_sha256","probability_index_sha256","original_tool_wait_sha256")),
            "Independent frozen TEST replay")


def validate_manifest(manifest):
    rows = manifest["items"]
    names = ["%06d.jpg" % i for i in range(1,1161) if i % 10 in (1,9)]
    require(manifest.get("population") == "official_TEST232" and manifest.get("TEST") is True
            and manifest.get("GT_opened") is False and manifest.get("AP") is False
            and [r["image_ref"]["path"] for r in rows] == names and len(set(manifest["expected_ids"])) == len(names)
            and [r["image_id"] for r in rows] == manifest["expected_ids"], "Frozen image-only manifest and native image order")
    lines = "".join("%s\t%d\t%s\n" % (r["image_ref"]["path"], r["image_ref"]["bytes"], r["image_ref"]["sha256"]) for r in rows)
    require(digest(lines.encode("ascii")) == INVENTORY_SHA, "Audited 232 encoded source identities")
    return rows


def join_blocks(roster, predictions, audits, changes):
    require(len(roster) == len(predictions) == len(audits) == len(changes), "Aligned complete native streams")
    blocks = {}
    for item, pred, audit, change in zip(roster, predictions, audits, changes):
        image_id = item["image_id"]
        require(pred["image_id"] == audit["image_id"] == change["image_id"] == image_id
                and len(pred["coco"]) == audit["post_nms"] == len(audit["candidates"]), "Original native image and post-NMS population")
        native = pred["coco"][:100]
        last = 1.
        for rank, (row, detail) in enumerate(zip(native, audit["candidates"][:100]), 1):
            score = row["score"]
            require(set(row) == {"image_id","category_id","score","segmentation"}
                    and row["image_id"] == image_id and row["category_id"] == 1 and type(score) in (int,float)
                    and math.isfinite(score) and 0 <= score <= last and detail["native_rank"] == rank and detail["score"] == score
                    and row["segmentation"]["size"] == item["original_hw"], "Rank, unmodified native ship confidence, canvas and source identity")
            last = score
        edits = change["changes"]
        for key, arm_edits in edits.items():
            rank = int(key)
            require(str(rank) == key and 1 <= rank <= len(native) and arm_edits and set(arm_edits) <= set(ARMS[1:]),
                    "Native top100 sparse edit rank and mode")
            for edit in arm_edits.values():
                require(set(edit) == {"swaps","rle"} and type(edit["swaps"]) is int and 0 < edit["swaps"] <= 8
                        and edit["rle"]["size"] == item["original_hw"], "Independently verified native edit")
            for a,b in (("fixed","free_matched_count"),("free_natural","native_matched_count")):
                require(arm_edits.get(a,{}).get("swaps",0) == arm_edits.get(b,{}).get("swaps",0), "Per-candidate matched budget")
        name = item["image_ref"]["path"]
        require(name not in blocks, "Unique native filename")
        blocks[name] = (native, edits, item["original_hw"])
    return blocks


def evaluate(gt, rows):
    with redirect_stdout(StringIO()):
        ref = COCO()
        ref.dataset = deepcopy(gt)
        ref.createIndex()
        if rows:
            dt = ref.loadRes(deepcopy(rows))
        else:
            dt = COCO()
            dt.dataset = dict(images=ref.dataset["images"], categories=ref.dataset["categories"], annotations=[])
            dt.createIndex()
        ev = COCOeval(ref, dt, "segm")
        ev.params.imgIds = [im["id"] for im in gt["images"]]
        ev.params.catIds = [0]
        ev.params.maxDets = [1,10,100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    values = dict(zip(METRICS, (float(v) for v in ev.stats)))
    i, = np.flatnonzero(np.isclose(ev.params.iouThrs, .90))
    precision = ev.eval["precision"][i,:,:,ev.params.areaRngLbl.index("all"),ev.params.maxDets.index(100)]
    valid = precision[precision > -1]
    values["AP90"] = float(valid.mean()) if valid.size else -1.
    return values


def score_populations(gt, blocks, excluded):
    names = [im["file_name"] for im in gt["images"]]
    ids = {im["id"] for im in gt["images"]}
    require(len(ids) == len(names) == len(set(names)) and set(names) == set(blocks) and set(excluded) <= set(names)
            and [c["id"] for c in gt["categories"]] == [0]
            and all(a["category_id"] == 0 and a["image_id"] in ids for a in gt["annotations"]), "Unique original GT roster and category zero")
    output = {}
    for label, removed in (("official", set()), ("exclude_exact_fit_overlap", set(excluded))):
        images = [im for im in gt["images"] if im["file_name"] not in removed]
        retained = {im["id"] for im in images}
        anns = [ann for ann in gt["annotations"] if ann["image_id"] in retained]
        subset = dict(gt, images=images, annotations=anns)
        metrics = {}
        for arm in ARMS:
            flat = []
            for image in images:
                native, edits, hw = blocks[image["file_name"]]
                require(hw == [image["height"],image["width"]], "Native source and original GT canvas")
                for rank, row in enumerate(native, 1):
                    edit = edits.get(str(rank), {}).get(arm)
                    flat.append(dict(image_id=image["id"], category_id=0, score=row["score"],
                                     segmentation=edit["rle"] if edit else row["segmentation"]))
            metrics[arm] = evaluate(subset, flat)
        contrasts = {}
        for a,b in (("free_natural","raw"),("free_matched_count","fixed"),("free_natural","free_matched_count"),
                    ("free_natural","native_matched_count")):
            contrasts[a+" minus "+b] = {k:100*(v-metrics[b][k]) if min(v,metrics[b][k]) >= 0 else None for k,v in metrics[a].items()}
        output[label] = dict(images=len(images), annotations=len(anns), candidates=len(flat), metrics_fraction=metrics,
                             paired_contrasts_points=contrasts)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("accepted","manifest","predictions","candidate-audit","edits","gt","output"):
        parser.add_argument("--"+k, type=Path, required=True)
    parser.add_argument("--accepted-sha256", required=True)
    args = parser.parse_args()
    runtime = dict(python=platform.python_version(),numpy=np.__version__,pycocotools=version("pycocotools"))
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and runtime == dict(python="3.10.13",numpy="1.26.4",pycocotools="2.0.11"), "Fixed isolated D-FINE CPU scoring runtime")
    require(not args.output.exists() and args.output.parent.is_dir(), "Fresh score output")
    receipt = json.loads(pinned(args.accepted,args.accepted_sha256,1024**2))
    check_acceptance(receipt)
    roster = validate_manifest(json.loads(pinned(args.manifest,receipt["manifest_sha256"],4*1024**2)))
    streams = []
    for path, key in ((args.predictions,"input_predictions_sha256"),(args.candidate_audit,"candidate_audit_sha256"),(args.edits,"edits_sha256")):
        data = pinned(path,receipt[key],256*1024**2)
        require(data.endswith(b"\n"), "Closed newline-delimited native stream")
        streams.append([json.loads(line) for line in data.splitlines()])
        del data
    blocks = join_blocks(roster,*streams)
    del streams
    gt = json.loads(pinned(args.gt,GT_SHA,2*1024**2))
    require(len(gt["images"]) == 232 and len(gt["annotations"]) == 546
            and digest(("\n".join(EXCLUDED)+"\n").encode("ascii")) == EXCLUSION_SHA, "Frozen full population and ten exclusions")
    views = score_populations(gt,blocks,EXCLUDED)
    require(views["exclude_exact_fit_overlap"]["images"] == 222, "Frozen 222-image sensitivity")
    result = dict(status="PSEG_TEST232_DFINE_FROZEN_POINT_SCORE_COMPLETE", TEST=True, GT_opened=True, AP=True,
                  historical_target_diagnostic=True, source_only_frozen_head=True, gt_sha256=GT_SHA,
                  overlap_audit_sha256=OVERLAP_AUDIT_SHA, excluded_filenames_sha256=EXCLUSION_SHA, excluded_images=len(EXCLUDED),
                  input_replay_receipt_sha256=args.accepted_sha256, ema_sha256=EMA, refit_accepted_sha256=HEAD,
                  native_category=1, original_scored_category=0,
                  views=views, arms=list(ARMS), runtime=runtime, optimizer_updates=0, model_forward_calls=0,
                  cohorted_coco_evaluations=2*len(ARMS), score_source_sha256=digest(Path(__file__).read_bytes()),
                  **{k:receipt[k] for k in ("manifest_sha256","acceptance_sha256","replay_sha256","edits_sha256",
                                            "input_predictions_sha256","candidate_audit_sha256")})
    data = (json.dumps(result,allow_nan=False,sort_keys=True,indent=2)+"\n").encode()
    with args.output.open("xb") as dest:
        require(dest.write(data) == len(data), "Complete score")
        dest.flush()
        os.fsync(dest.fileno())
    print(json.dumps(dict(status=result["status"],result_sha256=digest(data))),flush=True)


if __name__ == "__main__":
    main()
