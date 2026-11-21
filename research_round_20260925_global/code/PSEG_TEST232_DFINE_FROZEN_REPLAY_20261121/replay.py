"""PSeg-SSDD official TEST232, HRSID D-FINE E24 and frozen FIT109 head.

prepare seals mask deltas for the five existing modes before target labels.
verify checks the closed streams, original CPU exit, rank, area and native support.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec


LEGACY_SHA = "ee1db45898c42262ba263e15a14ee24c72ef1ee00d5015b5bd243daaeeee01d3"
EMA = "82ca80e2cb9a0b9673a136fad3d7de3097b7327f4f76e3a2d3e7184ab07e7fd8"
CAPTURE_ACCEPTED = "DFINE_HRSID_E24_SSDD_TEST232_PROBABILITY_INDEPENDENTLY_ACCEPTED_V1"
PREPARED = "DFINE_PSEG_TEST232_FROZEN_POINT_REPLAY_COMPLETE_V1"
VERIFIED = "DFINE_PSEG_TEST232_FROZEN_POINT_REPLAY_INDEPENDENTLY_ACCEPTED_V1"
MODES = ("raw", "fixed", "free_matched_count", "free_natural", "native_matched_count")
FLAGS = dict(TEST=True, DEV=False, FIT=False, GT_opened=False, AP=False)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def modules(code, a):
    code = code.resolve(strict=True)
    old_transfer = code / "DFINE_HRSID_TO_SSDD_POINT_TRANSFER_20261120"
    source = old_transfer / "ssdd_replay.py"
    raw = source.read_bytes()
    need(sha256(raw).hexdigest() == LEGACY_SHA, "Published DEV187 replay implementation")
    legacy = ModuleType("_original_transfer")
    legacy.__file__ = str(source)
    exec(compile(raw, str(source), "exec"), legacy.__dict__)
    e24 = code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24"
    a.refit = e24 / "fit109_final_refit/frozen_refit.py"
    a.common = e24 / "fit109_gray_equal_count/equal_count.py"
    a.gray = e24 / "fit109_gray_ablation/gray_ablation.py"
    a.pair_math = e24 / "fit109_paired_point/pair_math.py"
    a.rank_math = e24 / "fit109_probability_rank/rank_math.py"
    a.free = e24 / "fit109_free_pairing/free_pairing.py"
    a.training = e24 / "fit109_paired_point/train_point.py"
    native = legacy.load(old_transfer / "native_points.py", legacy.NATIVE, "_original_native_points")
    old = legacy.load(e24 / "dev705_free_endpoint/dev_endpoint.py", legacy.OLD, "_original_endpoint")
    author = code / "DFINE_HRSID_TO_SSDD_DEV187_CAPTURE_20261120/author_export_reference.py"
    dev = legacy.load(author, legacy.DEV, "_author_output")
    need(tuple(legacy.MODES) == tuple(old.MODES) == MODES, "Exact five original arms")
    return SimpleNamespace(legacy=legacy, old=old, native=native, author=dev)


def admit(m, a):
    receipt = json.loads(m.legacy.pin(a.accepted, a.accepted_sha))
    need(receipt["status"] == CAPTURE_ACCEPTED and receipt["manifest_sha256"] == a.manifest_sha
         and receipt["ema_sha256"] == EMA and receipt["images"] == receipt["probability_files"] == 232
         and receipt["exact_top100_binary_replay"] == receipt["probability_rows"] <= 23200
         and receipt["exact_author_binary_replay"] == receipt["post_nms_rows"]
         and receipt["original_author_postprocess_parity"] is True and receipt["job_drained"] is True
         and type(receipt["original_os_exit"]) is int and receipt["original_os_exit"] == 0
         and all(receipt[k] is v for k, v in FLAGS.items())
         and all(m.author.digest(receipt[k]) for k in ("predictions_sha256", "candidate_audit_sha256",
                 "probability_index_sha256", "export_result_sha256", "export_terminal_sha256",
                 "original_tool_wait_sha256")), "Independently closed official TEST232 capture")
    manifest = json.loads(m.legacy.pin(a.manifest, a.manifest_sha))
    roster = manifest["items"]
    ids = [i["image_id"] for i in roster]
    need(manifest["TEST"] is True and manifest.get("GT_opened", False) is False
         and manifest.get("AP", False) is False and len(roster) == len(set(ids)) == 232
         and ids == manifest["expected_ids"], "Complete ordered image-only official roster")
    for item in roster:
        ref = item["image_ref"]
        need(len(item["original_hw"]) == 2 and all(type(v) is int and v > 0 for v in item["original_hw"])
             and m.author.digest(ref["sha256"]) and type(ref["bytes"]) is int and ref["bytes"] > 0,
             "Native target shape and encoded image identity")
    return roster, receipt


def output_record(m, a, receipt):
    return dict(**FLAGS, manifest_sha256=a.manifest_sha, acceptance_sha256=a.accepted_sha,
                input_predictions_sha256=receipt["predictions_sha256"],
                candidate_audit_sha256=receipt["candidate_audit_sha256"],
                probability_index_sha256=receipt["probability_index_sha256"], ema_sha256=EMA,
                refit_accepted_sha256=m.old.REFIT_ACCEPTED,
                worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                modes=list(MODES), images=232, detector_forwards=0, optimizer_updates=0)


def emit(m, a, item, coco, audit, planes, image, models, counts):
    pair, rank, free, learner, head, center, scale = models
    adapted, probability = m.native.make_native(pair, rank, *item["original_hw"])
    changes = m.old.image_edits(coco, audit, planes, pair.image_features(image),
                                (adapted, probability, free, learner, head, center, scale), counts)
    return dict(image_id=item["image_id"], rgb_sha256=sha256(image.tobytes(order="C")).hexdigest(), changes=changes)


def prepare(m, a, roster, receipt):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    models = m.old.restore(a)
    models[0].raw = m.legacy.pin(a.pair_math, m.native.PAIR)
    counts, groups, digest = Counter(), set(), sha256()
    a.output.mkdir()
    start = perf_counter()
    with (a.output / m.legacy.EDITS).open("xb") as stream:
        for n, (item, coco, audit, planes) in enumerate(m.legacy.blocks(a, m.author, roster, receipt, True), 1):
            image = m.legacy.rgb(a, item)
            row = emit(m, a, item, coco, audit, planes, image, models, counts)
            groups.add(row["rgb_sha256"])
            counts["candidates"] += len(coco)
            raw = m.author.encoded(row)
            need(stream.write(raw) == len(raw), "Complete target edit line")
            digest.update(raw)
            if n % 25 == 0 or n == 232:
                print(json.dumps(dict(event="PSEG_TEST232_POINT_REPLAY_PROGRESS", images=n)), flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    need(counts["candidates"] == receipt["probability_rows"] and
         counts["area_eligible"] == counts["zero_support"] + counts["supported"] and
         counts["fixed_swaps"] == counts["free_matched_count_swaps"] and
         counts["free_natural_swaps"] == counts["native_matched_count_swaps"], "Full replay and paired budgets")
    return dict(output_record(m, a, receipt), status=PREPARED, counts=dict(counts), rgb_groups=len(groups),
                edits_sha256=digest.hexdigest(), seconds=perf_counter()-start)


def check_support(rank, coco, audit, changes, counts):
    for key, arms in changes.items():
        i = int(key) - 1
        stock = codec.decode(coco[i]["segmentation"]).astype(bool, order="C")
        pools, _ = rank.editable_pools(stock, audit[i]["box_xyxy"])
        adds = np.concatenate([pools[name][0] for name in ("add1", "add2")])
        drops = np.concatenate([pools[name][0] for name in ("drop1", "drop2")])
        for mode, data in arms.items():
            edited = codec.decode(data["rle"]).astype(bool, order="C")
            added = np.flatnonzero(edited.ravel() & ~stock.ravel())
            dropped = np.flatnonzero(stock.ravel() & ~edited.ravel())
            need(len(added) == len(dropped) == data["swaps"] and np.isin(added, adds).all() and
                 np.isin(dropped, drops).all(), "Edited masks retain original box and two-pixel support")
            counts.update({mode + "_edited": 1, mode + "_swaps": len(added)})


def check_terminal(terminal, digest):
    need(terminal["status"] == "PSEG_TEST232_REPLAY_TERMINAL_OBSERVED"
         and terminal["original_process_terminal"] is True and type(terminal["original_exit"]) is int
         and terminal["original_exit"] == 0 and terminal["result_sha256"] == digest
         and type(terminal["tool_session_handle"]) is int and terminal["tool_session_handle"] > 0,
         "Directly observed replay exit and matching result footer")


def verify(m, a, roster, receipt):
    terminal = json.loads(m.legacy.pin(a.wait, a.wait_sha))
    check_terminal(terminal, a.replay_sha)
    prepared = json.loads(m.legacy.pin(a.replay / "RESULT_SAFE.json", a.replay_sha))
    identity = output_record(m, a, receipt)
    need(prepared["status"] == PREPARED and all(prepared[k] == v for k, v in identity.items()),
         "Exact originating TEST232 population and frozen source bindings")
    rank = m.legacy.load(a.rank_math, m.old.RANK_MATH, "_support_rank")
    counts, groups = Counter(), set()
    for item, coco, audit, row in m.legacy.edit_blocks(a, m.author, m.old, roster, receipt, prepared):
        image = m.legacy.rgb(a, item)
        need(row["rgb_sha256"] == sha256(image.tobytes(order="C")).hexdigest(), "Decoded target image identity")
        groups.add(row["rgb_sha256"])
        counts["candidates"] += len(coco)
        counts["area_eligible"] += sum(32 <= d["mask_area"] < 1024 for d in audit)
        check_support(rank, coco, audit, row["changes"], counts)
    checked = ["candidates", "area_eligible"] + [f"{mode}_{stat}" for mode in MODES[1:] for stat in ("edited","swaps")]
    need(len(groups) == prepared["rgb_groups"] and counts["candidates"] == receipt["probability_rows"]
         and all(prepared["counts"].get(k, 0) == counts[k] for k in checked), "Independent frozen budget totals")
    a.output.mkdir()
    return dict(identity, status=VERIFIED, counts=dict(counts), rgb_groups=len(groups), exact_native_support=True,
                replay_sha256=a.replay_sha, original_tool_wait_sha256=a.wait_sha, edits_sha256=prepared["edits_sha256"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "verify"))
    for key in ("code", "manifest", "capture", "accepted", "images", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("manifest-sha", "accepted-sha"):
        p.add_argument("--" + key, required=True)
    for key in ("refit-result", "refit-acceptance", "weights", "replay", "wait"):
        p.add_argument("--" + key, type=Path)
    for key in ("replay-sha", "wait-sha"):
        p.add_argument("--" + key)
    a = p.parse_args()
    required = ("refit_result", "refit_acceptance", "weights") if a.mode == "prepare" else (
        "replay", "replay_sha", "wait", "wait_sha")
    for name in required:
        if getattr(a, name) is None:
            p.error("--" + name.replace("_", "-") + " required in " + a.mode)
    for name, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, name, value.resolve(strict=name != "output"))
    m = modules(a.code, a)
    m.author.cpu_runtime()
    roster, receipt = admit(m, a)
    result = (prepare if a.mode == "prepare" else verify)(m, a, roster, receipt)
    digest = m.author.write_new(a.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=result["status"], result_sha256=digest)), flush=True)


if __name__ == "__main__":
    main()
