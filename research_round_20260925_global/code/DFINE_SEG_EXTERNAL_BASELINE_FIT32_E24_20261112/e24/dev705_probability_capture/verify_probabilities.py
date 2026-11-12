"""Independently read back the closed DEV705 native probability capture on CPU."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter

PRED = "80382f6f7318ac7a033e4d8169cf278322a9d51731f2d86e11b5f426d318d0ed"
AUDIT = "29258878c41ec3d0cb2b173066d846a440197e2a0f45cd1277455dba0f786389"
MANIFEST = "39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f"
EMA = "82ca80e2cb9a0b9673a136fad3d7de3097b7327f4f76e3a2d3e7184ab07e7fd8"
ACCEPT = "7401f031c996298caff1761c6ef691fae0c301e65c87c15f51d186dcf81f25d8"
SIDECAR = "53dd5867d4353e3569552627d252877e4222bfcb2e6c7f60ca5bb53f3eabc940"
PROTOCOL = "51e2bd4e4b2bd4d116c3a31f010f6675e3df95bd1f88c2aa9017add2900f63a2"
WORKER = "58d4eb3d9fcb7caaf03947adec7fe2c6551e751deb1d5d5d56f3bcfa56501d81"
STATUS = "HRSID_DFINE_E24_DEV705_PROBABILITY_INDEPENDENTLY_ACCEPTED_V1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, expected, limit=2**20):
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= limit,
            "Bounded canonical readback input")
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Exact readback digest " + path.name)
    return raw


def consume(handle, digest, limit):
    line = handle.readline(limit)
    require(line.endswith(b"\n"), "Exact complete bounded row")
    digest.update(line)
    return json.loads(line)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("export", "manifest", "terminal", "output", "sidecar-source"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("terminal-sha", "result-sha"):
        parser.add_argument("--" + name, required=True)
    a = parser.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ["CUDA_VISIBLE_DEVICES"] == "-1",
            "Exact isolated CPU readback")
    start = perf_counter()
    terminal = json.loads(pinned(a.terminal, a.terminal_sha))
    result = json.loads(pinned(a.export / "RESULT_SAFE.json", a.result_sha))
    require(terminal["status"] == "HRSID_DFINE_E24_DEV705_EXPORT_OS_CLOSED_V1"
            and terminal["original_os_exit"] == 0 and terminal["original_os_wait"] is True
            and terminal["job_drained"] is True and terminal["resource_stop"] is False
            and terminal["export_result_sha256"] == a.result_sha and terminal["protocol_sha256"] == PROTOCOL
            and result["status"] == "HRSID_DFINE_E24_DEV705_EXPORT_COMPLETE_V1"
            and result["protocol_sha256"] == PROTOCOL and result["export_source_sha256"] == WORKER
            and result["predictions_sha256"] == PRED and result["candidate_audit_sha256"] == AUDIT
            and result["manifest_sha256"] == MANIFEST and result["ema_sha256"] == EMA
            and result["e24_acceptance_sha256"] == ACCEPT and result["historical_stream_byte_parity"] is True
            and result["images"] == result["model_forwards"] == result["probability_files"] == 705
            and result["probability_rows"] == result["exact_author_binary_replay"] == 70500
            and result["DEV"] is True and result["FIT"] is result["TEST"] is result["GT_opened"] is result["AP"] is False,
            "Exact naturally closed candidate and probability export")
    sidecar = {"__name__": "_sidecar_readback", "__file__": str(a.sidecar_source)}
    exec(compile(pinned(a.sidecar_source, SIDECAR), str(a.sidecar_source), "exec"), sidecar)
    roster = json.loads(pinned(a.manifest, MANIFEST))
    require(len(roster) == len({item["image_id"] for item in roster}) == 705, "Exact image-only roster")
    all_rows = probability_rows = probability_bytes = 0
    names = set()
    pred_hash, audit_hash, index_hash = sha256(), sha256(), sha256()
    with (a.export / "DEV_MASK_PREDICTIONS_PRIVATE.jsonl").open("rb") as predictions, (
            a.export / "DEV_CANDIDATE_AUDIT_PRIVATE.jsonl").open("rb") as audit, (
            a.export / "PROBABILITIES_PRIVATE.jsonl").open("rb") as index:
        for ordinal, image in enumerate(roster, 1):
            pred = consume(predictions, pred_hash, 4 * 2**20)
            detail = consume(audit, audit_hash, 2**20)
            block = consume(index, index_hash, 2**15)
            records = detail["candidates"]
            query_ids = block["query_ids"]
            ref = block["probability"]
            require(set(block) == {"image_id", "query_ids", "probability"}
                    and pred["image_id"] == detail["image_id"] == block["image_id"] == image["image_id"]
                    and image["height"] == image["width"] == 800
                    and len(pred["coco"]) == len(records) == len(query_ids) == len(set(query_ids)) == detail["post_nms"]
                    and 100 <= len(query_ids) <= 300
                    and all(type(q) is int and 0 <= q < 300 for q in query_ids)
                    and all(r["native_rank"] == i and r["score"] == p["score"]
                            and p["image_id"] == image["image_id"] and p["segmentation"]["size"] == [800, 800]
                            for i, (r, p) in enumerate(zip(records, pred["coco"], strict=True), 1)),
                    "Exact image, score, rank, shape and query lineage")
            plane = sidecar["verify_probability_file"](a.export, ordinal, ref, 100)
            names.add(Path(ref["file"]).name)
            all_rows += len(records)
            probability_rows += len(plane)
            probability_bytes += ref["bytes"]
        require(not predictions.read(1) and not audit.read(1) and not index.read(1), "No trailing image records")
    require({p.name for p in (a.export / "PROBABILITY_PRIVATE").iterdir()} == names
            and len(names) == 705 and all_rows == result["post_nms_rows"] == 184687
            and probability_rows == result["probability_rows"] == 70500
            and probability_bytes == result["probability_bytes"]
            and pred_hash.hexdigest() == PRED and audit_hash.hexdigest() == AUDIT
            and index_hash.hexdigest() == result["probability_index_sha256"], "Exact closed full-population digests")
    payload = dict(
        status=STATUS, protocol_sha256=PROTOCOL, worker_sha256=WORKER,
        predictions_sha256=PRED, candidate_audit_sha256=AUDIT, manifest_sha256=MANIFEST, ema_sha256=EMA,
        e24_acceptance_sha256=ACCEPT, export_result_sha256=a.result_sha, export_terminal_sha256=a.terminal_sha,
        original_tool_wait_sha256=terminal["original_tool_wait_sha256"], original_os_exit=0, job_drained=True,
        historical_stream_byte_parity=True, exact_author_binary_replay=result["exact_author_binary_replay"],
        probability_index_sha256=index_hash.hexdigest(), probability_rows=probability_rows, probability_files=len(names),
        probability_bytes=probability_bytes, post_nms_rows=all_rows, images=len(roster),
        model_forwards=0, GT_opened=False, DEV=True, FIT=False, TEST=False, AP=False,
        elapsed_seconds=perf_counter()-start)
    raw = sidecar["canonical_line"](payload)
    with a.output.open("xb") as stream:
        require(stream.write(raw) == len(raw), "Complete independent result")
        sidecar["flush"](stream)
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

