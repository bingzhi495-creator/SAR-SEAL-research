"""Independently verify the closed native-size SSDD DEV187 probability capture on CPU."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter

MANIFEST = "18df007e15e3e838e607f178f14e7a26f2b37b992ee39bf091e0b74cf084fb13"
EMA = "82ca80e2cb9a0b9673a136fad3d7de3097b7327f4f76e3a2d3e7184ab07e7fd8"
ACCEPT = "7401f031c996298caff1761c6ef691fae0c301e65c87c15f51d186dcf81f25d8"
PROTOCOL = "cbf4e488cf711e2d44d6dafc1deb969a0402bc99bef2de2a50ffa72e9b523e5e"
WORKER = "1efbbe2a3fdbe58e2eb4e73ddb6e80a10e011120fc96e8f8ce758e04f89619cf"
SIDECAR = "53dd5867d4353e3569552627d252877e4222bfcb2e6c7f60ca5bb53f3eabc940"
BASE = "5be622cdec5c889ca8520aab5547f6f15f01a54568a0263423a816dca7f96626"
STATUS = "DFINE_HRSID_E24_SSDD_DEV187_PROBABILITY_INDEPENDENTLY_ACCEPTED_V1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def exact(path, digest, limit=1024**2):
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= limit, "Bounded closed input")
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, "Exact closed input digest " + path.name)
    return raw


def source(path, digest):
    scope = {"__name__": "_closed_" + path.stem, "__file__": str(path)}
    exec(compile(exact(path, digest), str(path), "exec"), scope)
    return scope


def line(stream, digest, limit):
    raw = stream.readline(limit)
    require(raw.endswith(b"\n"), "Exact complete bounded image row")
    digest.update(raw)
    return json.loads(raw)


def main():
    parser = ArgumentParser(allow_abbrev=False)
    for name in ("export", "manifest", "terminal", "output", "sidecar-source", "base", "codec-wheel"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("terminal-sha", "result-sha"):
        parser.add_argument("--" + name, required=True)
    a = parser.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ["CUDA_VISIBLE_DEVICES"] == "-1",
            "Isolated fresh CPU-only closed readback")
    before = perf_counter()
    terminal = json.loads(exact(a.terminal, a.terminal_sha))
    result = json.loads(exact(a.export / "RESULT_SAFE.json", a.result_sha))
    require(terminal["status"] == "DFINE_HRSID_E24_SSDD_DEV187_PROBABILITY_EXPORT_OS_CLOSED_V1"
            and terminal["original_os_wait"] is True and terminal["original_os_exit"] == 0
            and terminal["resource_stop"] is False and terminal["job_drained"] is True
            and terminal["export_result_sha256"] == a.result_sha and terminal["protocol_sha256"] == PROTOCOL
            and result["status"] == "DFINE_HRSID_E24_SSDD_DEV187_PROBABILITY_EXPORT_COMPLETE_V1"
            and result["protocol_sha256"] == PROTOCOL and result["export_source_sha256"] == WORKER
            and result["ema_sha256"] == EMA and result["e24_acceptance_sha256"] == ACCEPT
            and result["manifest_sha256"] == MANIFEST
            and result["images"] == result["model_forwards"] == result["native_dimensions_verified"] == 187
            and result["probability_files"] == 187 and result["raw_queries"] == result["internal_top_rows"] == 56100
            and result["exact_author_binary_replay"] == result["post_nms_rows"]
            and result["exact_top100_binary_replay"] == result["probability_rows"] == result["rows_at_100"]
            and result["original_author_postprocess_parity"] is True
            and result["confidence_threshold"] == 0. and result["box_nms_iou"] == .70
            and result["mask_threshold"] == .50 and result["keep_ratio"] is result["rect"] is False
            and result["model_precision"] == "FP32_TF32_disabled"
            and result["mask_resize_precision"] == "author_CUDA_FP16"
            and result["DEV"] is True and result["FIT"] is result["TEST"] is result["GT_opened"] is result["AP"] is False,
            "Exact original closed author inference and no-label scope")
    base, sidecar = source(a.base, BASE), source(a.sidecar_source, SIDECAR)
    base["prepare_export_codec"](a.codec_wheel, a.output.parent / "READBACK_CODEC_PRIVATE")
    roster = json.loads(exact(a.manifest, MANIFEST))
    require(roster["schema"] == "Root-RFDETR-DEV187-image-only-manifest-v1"
            and len(roster["items"]) == len(roster["expected_ids"]) == len(set(roster["expected_ids"])) == 187,
            "Exact fixed original image-only population")
    ph, ah, ih = sha256(), sha256(), sha256()
    counts = dict(post_nms_rows=0, rows_at_100=0, empty_post_nms=0, empty_at_100=0, zero_prediction_images=0)
    nbytes = below = 0
    names = set()
    with (a.export / "DEV_MASK_PREDICTIONS_PRIVATE.jsonl").open("rb") as predictions, (
            a.export / "DEV_CANDIDATE_AUDIT_PRIVATE.jsonl").open("rb") as audits, (
            a.export / "PROBABILITIES_PRIVATE.jsonl").open("rb") as index:
        for ordinal, (identity, image) in enumerate(zip(roster["expected_ids"], roster["items"], strict=True), 1):
            pred, audit, block = line(predictions,ph,4*1024**2), line(audits,ah,1024**2), line(index,ih,2**15)
            require(image["image_id"] == identity == block["image_id"] and set(block) == {"image_id","query_ids","probability"},
                    "Exact complete original image-order join")
            n, empty = base["validate_rows"](identity, image["original_hw"], pred, audit)
            qids = block["query_ids"]
            limit = min(n, 100)
            require(len(qids) == len(set(qids)) == n and all(type(q) is int and 0 <= q < 300 for q in qids),
                    "Exact all-query ordered post-NMS lineage and no filler rows")
            array = sidecar["verify_probability_file"](a.export, ordinal, block["probability"], limit)
            require(len(array) == limit, "Exact top-minimum original probability rows")
            counts["post_nms_rows"] += n
            counts["rows_at_100"] += limit
            counts["empty_post_nms"] += empty
            counts["empty_at_100"] += sum(r["mask_area"] == 0 for r in audit["candidates"][:100])
            counts["zero_prediction_images"] += n == 0
            below += n < 100
            nbytes += block["probability"]["bytes"]
            names.add(Path(block["probability"]["file"]).name)
        require(not predictions.read(1) and not audits.read(1) and not index.read(1), "No extra candidate or image data")
    require({p.name for p in (a.export / "PROBABILITY_PRIVATE").iterdir()} == names and len(names) == 187
            and ph.hexdigest() == result["predictions_sha256"] and ah.hexdigest() == result["candidate_audit_sha256"]
            and ih.hexdigest() == result["probability_index_sha256"] and nbytes == result["probability_bytes"]
            and all(counts[k] == result[k] for k in counts), "Exact complete closed artifact census and stream hashes")
    receipt = dict(status=STATUS, FIT=False, DEV=True, TEST=False, GT_opened=False, AP=False,
        original_os_exit=0, original_os_wait=True, job_drained=True, images=187,
        model_forwards=0, optimizer_updates=0, protocol_sha256=PROTOCOL, worker_sha256=WORKER,
        manifest_sha256=MANIFEST, ema_sha256=EMA, e24_acceptance_sha256=ACCEPT,
        export_result_sha256=a.result_sha, export_terminal_sha256=a.terminal_sha,
        original_tool_wait_sha256=terminal["original_tool_wait_sha256"],
        predictions_sha256=ph.hexdigest(), candidate_audit_sha256=ah.hexdigest(),
        probability_index_sha256=ih.hexdigest(), probability_files=187, probability_rows=counts["rows_at_100"],
        probability_bytes=nbytes, post_nms_below100_images=below,
        exact_author_binary_replay=result["exact_author_binary_replay"],
        exact_top100_binary_replay=result["exact_top100_binary_replay"], original_author_postprocess_parity=True,
        elapsed_seconds=perf_counter()-before, **counts)
    raw = sidecar["canonical_line"](receipt)
    with a.output.open("xb") as stream:
        require(stream.write(raw) == len(raw), "Complete independent acceptance")
        sidecar["flush"](stream)
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

