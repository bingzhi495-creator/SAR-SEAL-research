"""Read the independently closed core36 FIT capture and verify native logit-to-mask linkage."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import sys


MANIFEST_SHA = "af43530e7f8f118261b5523cdde738cca467ead813d38f1d4c989c8f4102e184"
PROTOCOL_SHA = "62b9d59c14628e7166b6b38dbbd4ee59ce6d08c3f3ead56e096b96f2cba7cbab"
WORKER_SHA = "f927435607d83088dcb4bb2420eccdd298ae4118e62747fe15bc09746a35db17"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, expected, maximum):
    need(path.resolve(strict=True) == path and path.is_file() and 0 < path.stat().st_size <= maximum,
         "Expected exact closed canonical local artifact")
    data = path.read_bytes()
    need(sha256(data).hexdigest() == expected, "Original artifact digest differs: " + path.name)
    return data


def check(args):
    import numpy as np
    import torch
    from pycocotools import mask as codec
    from torch.nn.functional import interpolate

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    need(not torch.cuda.is_initialized() and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
         "CPU-only independent readback")
    accepted = json.loads(pinned(args.wait, args.wait_sha, 64*1024))
    outer = accepted["exact_tool_wait"]
    owner_marker = json.loads(outer["output"].splitlines()[-1])
    need(accepted["original_session_id"] > 0 and outer["exit_code"] == 0 and outer.get("session_id") is None
         and owner_marker["status"] == "FIT114_CORE36_CENTER5202_Q164_CPU_OWNER_CLOSED_V1"
         and owner_marker["images_per_arm"] == 36, "Exact natural original tool termination required")
    root = args.actual
    report = json.loads(pinned(root/"PAIR_SAFE.json", owner_marker["result_sha256"], 64*1024))
    need(report["worker_sha256"] == WORKER_SHA and report["image_only_manifest_sha256"] == MANIFEST_SHA
         and report["model_forward_calls"] == 36 and report["optimizer_updates"] == 0
         and report["FIT"] and not any(report[key] for key in ("DEV","TEST","AP")),
         "Exact closed FIT original owner scope")
    arm, = report["arm_results"]
    need(arm["arm"] == "center", "Exact center arm required")
    child = root/"center"
    wait = json.loads(pinned(child/"ORIGINAL_WAIT_PRIVATE.json", arm["wait_sha256"], 64*1024))
    need(wait["original_os_wait"] and wait["original_os_exit"] == 0 and wait["job_assigned"]
         and wait["process_group_closed"] and wait["active_before_drain"] == 0 and wait["stop_reason"] is None,
         "Exact naturally closed Windows JobObject worker")
    need(pinned(child/"ORIGINAL.stderr", wait["original_stderr_sha256"], 4*1024**2)
         if (child/"ORIGINAL.stderr").stat().st_size else wait["original_stderr_sha256"] == sha256(b"").hexdigest(),
         "Original stderr evidence must match")
    out = child/"capture"
    result = json.loads(pinned(out/"RESULT_SAFE.json", arm["report_sha256"], 64*1024))
    log = pinned(child/"ORIGINAL.stdout", wait["original_stdout_sha256"], 4*1024**2)
    footer = json.loads(log.splitlines()[-1])
    need(footer["result_sha256"] == arm["report_sha256"]
         and footer["predictions_sha256"] == arm["prediction_sha256"] == result["prediction_sha256"]
         and result["manifest_sha256"] == MANIFEST_SHA and not result["GT_opened"],
         "Exact joined worker footer")
    manifest = json.loads(pinned(args.manifest, MANIFEST_SHA, 64*1024))
    need(manifest["closed_roster_design_sha256"] == PROTOCOL_SHA, "Exact prelocked image selection")
    lines = pinned(out/"QUERIES_PRIVATE.jsonl", arm["prediction_sha256"], 64*1024**2).splitlines()
    rows = [json.loads(line) for line in lines]
    raw_count = raw_bytes = mapped = 0
    for ordinal, (item, block) in enumerate(zip(manifest["images"], rows, strict=True), 1):
        need((block["image_id"],block["source_prefix"],block["native_basename"],block["image_content_sha256"],
              block["encoded_sha256"]) == (item["image_id"],item["group"],item["native_basename"],
                                         item["decoded_rgb_sha256"],item["encoded_sha256"]), "Image binding differs")
        areas = codec.area(block["all_rles"])
        chosen = [(qid, rank, block["ship_scores"][qid], int(areas[qid]))
                  for qid,rank in zip(block["native_ids"],block["native_ranks"],strict=True)
                  if 32 <= areas[qid] < 1024]
        mapped += len(block["native_ids"])
        capture = block["logit_capture"]
        expected = [(r["query_id"],r["native_rank"],r["score"],r["predicted_area"])
                    for r in capture["candidates"]]
        path = out/"RAW_LOGITS"/f"{ordinal:03d}.npy"
        need(chosen == expected and capture["relative_path"] == f"RAW_LOGITS/{ordinal:03d}.npy"
             and [r["logit_row"] for r in capture["candidates"]] == list(range(len(chosen))),
             "Exact predicted-area/logit-to-query linkage")
        npy = pinned(path, capture["sha256"], 9*1024**2)
        from io import BytesIO
        values = np.load(BytesIO(npy), allow_pickle=False)
        need(values.dtype == np.dtype("<f4") and values.shape == (len(chosen),144,144)
             and list(values.shape) == capture["shape"] and len(npy) == capture["bytes"]
             and bool(np.isfinite(values).all()), "Exact finite native raw logit payload")
        for start in range(0,len(chosen),16):
            planes = (interpolate(torch.from_numpy(values[start:start+16,None]), size=(800,800),
                                  mode="bilinear",align_corners=False)>0)[:,0].numpy()
            for target, (qid,_,_,_) in zip(planes, chosen[start:start+16], strict=True):
                rle = codec.encode(np.asfortranarray(target,dtype=np.uint8))
                need(dict(size=rle["size"],counts=rle["counts"].decode("ascii")) == block["all_rles"][qid],
                     "Independent bilinear and zero-threshold mask bytes differ")
        raw_count += len(chosen)
        raw_bytes += len(npy)
    need(len(rows) == result["images"] == 36 and mapped == result["native_mapped_rows"]
         and raw_count == result["raw_logit_rows"] and raw_bytes == result["raw_logit_bytes"]
         and len(list((out/"RAW_LOGITS").glob("*.npy"))) == 36 and not torch.cuda.is_initialized(),
         "Complete all-rows and files recount")
    return dict(status="FIT114_CORE36_CENTER5202_Q164_INDEPENDENT_READBACK_V1",
                images=len(rows),mapped=mapped,raw_logit_rows=raw_count,raw_logit_bytes=raw_bytes,
                raw_logit_raster_RLE_exact=raw_count,original_prediction_sha256=arm["prediction_sha256"],
                original_wait_sha256=args.wait_sha,original_owner_receipt_sha256=owner_marker["result_sha256"],
                FIT=True,GT_opened=False,DEV=False,TEST=False,model_forward_calls=0)


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("actual","manifest","wait","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--wait-sha",required=True)
    args = parser.parse_args()
    result = check(args)
    raw = (json.dumps(result,sort_keys=True,allow_nan=False)+"\n").encode()
    with args.output.open("xb") as out:
        out.write(raw)
    print(json.dumps(dict(result_sha256=sha256(raw).hexdigest(),**result)),flush=True)


if __name__ == "__main__":
    main()

