"""Frozen label-free DEV705 replay with a same-budget spatial-attraction control."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

import numpy as np
from pycocotools import mask as codec

PROTOCOL_SHA = "fd11e933a9f5d89699c7b614cf1d3176a036ed61d2ed8f194cb4329923a4da80"
ORIGINAL_SHA = "6681d3903199214cd109efd9a8bf92681d838901e6c9c8780e4f19f7e16703b7"
ADAPTER_SHA = "782d6b12c5d22d933827d05e78feb7bdc9864224bfe55a190fed117f706d94bd"
SIDECAR_RECEIPT_SHA = "277965bef4230ec7fadac30e103095af3ce68f99d1b69f74a8e348c391cef035"
PRIOR_REPLAY_SHA = "878d2a791da39d0941a9ea57067c8c3141c524fd0d690a0cadf01888946d46a3"
PRIOR_READBACK_SHA = "c1753af526c63b7b3f37825127e8a3cb82baf90bb3d584b51a9b2bdeee5d1f7d"
PRIOR_EXIT_SHA = "33f0a3c814bf146be2028a3e7179feae64e0b4cc9447dae298f41d53face6d75"
PREPARED = "DFINE_DEV705_CLASSICAL_ATTRACTION_PRELABEL_COMPLETE"
VERIFIED = "DFINE_DEV705_CLASSICAL_ATTRACTION_PRELABEL_INDEPENDENT_ACCEPTED"
NEW_ARMS = ("attraction_dynamic", "attraction_static_at_dynamic_count", "learned_at_attraction_count")
SCOPE = dict(DEV=True, FIT=False, TEST=False, GT_opened=False, AP=False, detector_forwards=0, optimizer_updates=0)
DEPS = {"refit": "fit109_final_refit/frozen_refit.py",
        "common": "fit109_gray_equal_count/equal_count.py",
        "gray": "fit109_gray_ablation/gray_ablation.py",
        "pair_math": "fit109_paired_point/pair_math.py",
        "rank_math": "fit109_probability_rank/rank_math.py",
        "free": "fit109_free_pairing/free_pairing.py",
        "training": "fit109_paired_point/train_point.py"}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(path, expected):
    data = path.read_bytes()
    need(sha256(data).hexdigest() == expected, "Pinned bytes: " + path.name)
    return data


def load_module(path, expected):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), module.__dict__)
    return module


def check_projection(changes, original):
    projection = {}
    edited = swaps = 0
    for rank, arms in changes.items():
        added = [arms[name]["swaps"] for name in NEW_ARMS if name in arms]
        if added:
            need(len(added) == 3 and all(type(n) is int and 0 < n <= 8 for n in added)
                 and len(set(added)) == 1, "Same realized new-arm swap count on each edited candidate")
            edited += 1
            swaps += added[0]
        original_arms = {name: arm for name, arm in arms.items() if name not in NEW_ARMS}
        if original_arms:
            projection[rank] = original_arms
    need(projection == original, "Exact old five-arm projection")
    return edited, swaps


def load_inputs(a):
    pin(a.protocol, PROTOCOL_SHA)
    e24 = a.code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24"
    old = load_module(e24 / "dev705_free_endpoint/dev_endpoint.py", ORIGINAL_SHA)
    dev = load_module(a.dev_source, old.DEV_SOURCE)
    dev.cpu_runtime()
    for name, path in DEPS.items():
        setattr(a, name, e24 / path)
    a.refit_result = a.refit_root / "ACTUAL/RESULT_SAFE.json"
    a.refit_acceptance = a.refit_root / "INDEPENDENT/RESULT_SAFE.json"
    a.weights = a.refit_root / "ACTUAL/POINT_HEAD_PRIVATE.npz"
    a.sidecar_accepted_sha = SIDECAR_RECEIPT_SHA
    receipt = old.probability_receipt(a, dev)
    manifest = dev.read_manifest(a.dev_root)
    prior = old.require_prelabel(SimpleNamespace(replay=a.prior_root / "PRELABEL_ACTUAL",
                                                replay_sha=PRIOR_REPLAY_SHA,
                                                sidecar_accepted_sha=SIDECAR_RECEIPT_SHA))
    verified = json.loads(pin(a.prior_root / "PRELABEL_INDEPENDENT/RESULT_SAFE.json", PRIOR_READBACK_SHA))
    terminal = json.loads(pin(a.prior_root / "PRELABEL_ORIGINAL_WAIT_PRIVATE.json", PRIOR_EXIT_SHA))
    need(verified["status"] == old.VERIFIED and verified["replay_sha256"] == PRIOR_REPLAY_SHA
         and verified["original_wait_sha256"] == PRIOR_EXIT_SHA and verified["original_exit"] == 0
         and terminal["terminal"]["exit_code"] == 0 and PRIOR_REPLAY_SHA in terminal["terminal"]["output"]
         and verified["edits_sha256"] == prior["edits_sha256"], "Original replay and independent closure")
    return old, dev, receipt, manifest, verified


def prior_blocks(a, original):
    digest = sha256()
    with (a.prior_root / "PRELABEL_ACTUAL/EDITS_PRIVATE.jsonl").open("rb") as stream:
        for _ in range(705):
            line = stream.readline(4 * 1024 ** 2)
            need(line.endswith(b"\n"), "Exact prior complete edit stream")
            digest.update(line)
            yield json.loads(line)
        need(not stream.read(1) and digest.hexdigest() == original["edits_sha256"], "Exact prior edit stream")


def base_summary(a, old, receipt, counts, edit_sha, started):
    return dict(**SCOPE, protocol_sha256=PROTOCOL_SHA, worker_sha256=a.worker_sha, images=705,
                candidates=counts["candidates"], counts=dict(counts), edits_sha256=edit_sha,
                predictions_sha256=old.PREDICTIONS, candidate_audit_sha256=old.AUDIT,
                sidecar_accepted_sha256=SIDECAR_RECEIPT_SHA, probability_index_sha256=receipt["probability_index_sha256"],
                refit_accepted_sha256=old.REFIT_ACCEPTED, prior_readback_sha256=PRIOR_READBACK_SHA,
                seconds=perf_counter() - started)


def prepare(a, old, dev, receipt, manifest, original):
    import torch

    adapter = load_module(a.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/ueikap_transfer.py", ADAPTER_SHA)
    adapter.extended_edits(a, SimpleNamespace(old=old))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    models = old.restore(a)
    started = perf_counter()
    digest, counts = sha256(), Counter()
    a.output.mkdir()
    with a.archive.open("rb") as file, (a.output / old.EDITS).open("xb") as out:
        need(a.archive.stat().st_size == dev.ARCHIVE_BYTES, "Original DEV PNG archive extent")
        file.seek(dev.CENTRAL_OFFSET)
        need(sha256(file.read(dev.CENTRAL_BYTES)).hexdigest() == dev.CENTRAL_SHA, "Exact original PNG archive directory")
        with ZipFile(file) as archive:
            native = old.source_blocks(a, dev, manifest, receipt, True)
            for ordinal, ((item, coco, audit, probs), prior) in enumerate(zip(native, prior_blocks(a, original), strict=True), 1):
                rgb, hw = dev.read_png(archive, item)
                need(hw == (800, 800) and item["image_id"] == prior["image_id"], "Original DEV raster identity")
                changes = old.image_edits(coco, audit, probs, models[0].image_features(rgb), models, counts)
                check_projection(changes, prior["changes"])
                counts["candidates"] += len(coco)
                line = dev.encoded(dict(image_id=item["image_id"], changes=changes))
                need(out.write(line) == len(line), "Complete edit block")
                digest.update(line)
                if ordinal % 100 == 0 or ordinal == 705:
                    print(json.dumps(dict(phase="prepare-edits", images=ordinal)), flush=True)
        out.flush()
        os.fsync(out.fileno())
    need(counts["candidates"] == 70500 and all(counts[key] == value for key, value in original["counts"].items()),
         "Complete native replay and all original verified edit totals")
    return dict(base_summary(a, old, receipt, counts, digest.hexdigest(), started), status=PREPARED,
                exact_original_policy_projection=True)


def validate_native_support(stock, box, prob, afters, area, rank_math, old):
    before = codec.decode(stock).astype(bool, order="C")
    pools, _ = rank_math.editable_pools(before, box)
    sides = [np.concatenate([pools[k][0] for k in keys]) for keys in (("add1", "add2"), ("drop1", "drop2"))]
    n = min(32, len(sides[0]), len(sides[1]))
    cap = min(8, area // 10, n)
    required = next(iter(afters.values()))["swaps"]
    need(32 <= area < 1024 and 1 <= required <= cap, "Native editable area, nonempty support and cap")
    support = []
    for coordinates in sides:
        scores = rank_math.sample_probability(prob, coordinates, 800)
        ordering = np.lexsort((coordinates, np.abs(scores.astype(np.float32) - .5)))
        support.append(coordinates[ordering[:n]])
    for edit in afters.values():
        need(set(edit) == {"rle", "swaps"} and edit["swaps"] == required, "Same audited serialized arm budget")
        old.validate_edit(stock, edit["rle"], area, required)
        after = codec.decode(edit["rle"]).astype(bool, order="C")
        add = np.flatnonzero((after & ~before).ravel())
        drop = np.flatnonzero((before & ~after).ravel())
        need(len(add) == len(drop) == required and np.isin(add, support[0]).all() and np.isin(drop, support[1]).all(),
             "Exact XOR and original selected endpoints")


def verify(a, old, dev, receipt, manifest, original):
    started = perf_counter()
    terminal = json.loads((a.phase_root / "observed_prepare-edits/TERMINAL.json").read_bytes())
    need(terminal["phase"] == "prepare-edits" and terminal["actual_exit"] == 0
         and terminal["output_footer_verified"] is True and terminal["worker_sha256"] == a.worker_sha,
         "Actual successful predecessor with original source")
    prepared = json.loads(pin(a.phase_root / "prepare-edits/RESULT_SAFE.json", terminal["result_sha256"]))
    need(prepared["status"] == PREPARED and prepared["protocol_sha256"] == PROTOCOL_SHA
         and prepared["exact_original_policy_projection"] is True and prepared["sidecar_accepted_sha256"] == SIDECAR_RECEIPT_SHA,
         "Exact prepared label-free replay")
    rank_math = old.load(a.rank_math, old.RANK_MATH, "_dev_classical_independent_support")
    digest, counts = sha256(), Counter()
    with (a.phase_root / "prepare-edits" / old.EDITS).open("rb") as stream:
        native = old.source_blocks(a, dev, manifest, receipt, True)
        for ordinal, ((item, coco, audit, probs), prior) in enumerate(zip(native, prior_blocks(a, original), strict=True), 1):
            line = stream.readline(4 * 1024 ** 2)
            need(line.endswith(b"\n"), "Complete prepared edit row")
            digest.update(line)
            edited = json.loads(line)
            need(set(edited) == {"image_id", "changes"} and edited["image_id"] == item["image_id"] == prior["image_id"],
                 "Exact original candidate image")
            check_projection(edited["changes"], prior["changes"])
            counts["candidates"] += len(coco)
            for rank, arms in edited["changes"].items():
                number = int(rank)
                need(1 <= number <= 100 and str(number) == rank and arms, "Nonempty native rank edit")
                extra = {name: arms[name] for name in NEW_ARMS if name in arms}
                if extra:
                    stock = coco[number - 1]["segmentation"]
                    detail = audit[number - 1]
                    validate_native_support(stock, detail["box_xyxy"], probs[number - 1], extra,
                                            detail["mask_area"], rank_math, old)
                for name, edit in arms.items():
                    counts.update({name + "_edited": 1, name + "_swaps": edit["swaps"]})
            if ordinal % 100 == 0 or ordinal == 705:
                print(json.dumps(dict(phase="verify-edits", images=ordinal)), flush=True)
        need(not stream.read(1) and digest.hexdigest() == prepared["edits_sha256"], "Exact prepared edit stream")
    need(counts["candidates"] == 70500 and all(counts[key] == prepared["counts"].get(key, 0) for key in counts)
         and all(counts[key] == n for key, n in original["counts"].items()), "Exact independent replay census")
    a.output.mkdir()
    return dict(base_summary(a, old, receipt, counts, digest.hexdigest(), started), status=VERIFIED,
                prepare_result_sha256=terminal["result_sha256"], exact_original_policy_projection=True,
                exact_new_arm_area_xor_support=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare-edits", "verify-edits"))
    for name in ("code", "dev-source", "dev-root", "sidecar", "sidecar-accepted", "refit-root",
                 "prior-root", "protocol", "phase-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    a = parser.parse_args()
    need((a.archive is not None) == (a.phase == "prepare-edits"), "Archive only for preparing edits")
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve(strict=(key != "output")))
    a.worker_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    inputs = load_inputs(a)
    operation = prepare if a.phase == "prepare-edits" else verify
    result = operation(a, *inputs)
    digest = inputs[1].write_new(a.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=result["status"], result_sha256=digest)), flush=True)


if __name__ == "__main__":
    main()

