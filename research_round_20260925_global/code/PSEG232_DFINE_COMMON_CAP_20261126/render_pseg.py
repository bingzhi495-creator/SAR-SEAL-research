"""Render native, fixed, exactly bounded and independent signed writes on closed PSeg TEST232."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

REPLAY = "f1b6e83e98851bf867bd72f3ecb082ca2997632bdd1aedaaf64110cf06b04fd5"
COMMON = "df8494fa2275076b6eb8d3c054e641074400f4bea510c6263bac1f8db1eb0d5a"
MANIFEST = "9c0b9bc07eade8b9de4349f187a80837f6a1c8449e596a2c1234f3827356b65e"
CAPTURE = "6107389d36848d8d0e24db2cc82518f73494cd990a9f71fe18e473da0f1201c9"
CAPTURE_READBACK = "dc48e18c0b988fda3eada68dc06b1404fd5c3806502e32d455a08a2e96d0121c"
CAPTURE_TERMINAL = "194e1a52763773dee4c369fd5952cc3332148bf67ef208d6ef55cd36c37942f7"
CAPTURE_WAIT = "257937d1a1505c01fe8de55a62180965b8fc2656f3096fba5b58b5abdfe143a6"
OLD_RENDER = "531b979c316c58f608e48205a2f86e66e666eae61c1029de59c8229a941c937e"
OLD_READBACK = "e99d0f2bbd9e1839e260a88075f6af187e05ffc2b86abaf6cc33b788372b7b71"
OLD_WAIT = "1f11f07bfe772ba27dfc35387ce5592ab635405abf8bf311bc78d94aac6b0b2c"
OLD_EDITS = "4c9638e9b502cf7048e0cb6e52d88b14ff71d78b1d6ec728dc533728ae1220bc"
RENDERED = "PSEG232_DFINE_COMMON_CAP_RENDER_COMPLETE"
VERIFIED = "PSEG232_DFINE_COMMON_CAP_VERIFY_COMPLETE"
ARMS = ("fixed", "bounded", "free_matched_count")


def need(condition, detail):
    if not condition:
        raise ValueError(detail)


def native_distances(coords, width):
    ay, ax = divmod(coords[:, 0], width)
    dy, dx = divmod(coords[:, 1], width)
    return (ay[:, None] - dy[None, :])**2 + (ax[:, None] - dx[None, :])**2


def signed_record(original, coords, q, stock, area, fixed, free, old, solver, np):
    width = stock.shape[1]
    quota = min(8, area // 10, len(q))
    f = np.asarray(fixed.choose_pairs(q[:, 0] - q[:, 1], quota), dtype=np.int64)
    k = len(f)
    need(k > 0 and k == original["fixed"]["swaps"] == original["free_matched_count"]["swaps"],
         "Exact original positive per-candidate fixed quota")
    i = np.asarray(free.choose_free(q, quota, k), dtype=np.int64)
    distances = native_distances(coords, width)
    cap = int(distances[f, f].max())
    ai, di, receipts = solver.bounded_endpoints(coords, q.astype(np.float64), k, width, f, f, i[:, 0], i[:, 1])
    l = np.column_stack((ai, di))
    f_pairs = np.column_stack((f, f))
    for name, selected in (("fixed", f_pairs), ("free_matched_count", i)):
        need(old.edit(stock, coords, selected.tolist()) == original[name]["rle"],
             "Archived signed choice exactly recovers the original mask")
    need(receipts["cap_squared_sum"] == cap and np.all(distances[ai, di] <= cap),
         "Exact inherited native-coordinate squared displacement cap")
    bounded = dict(swaps=k, rle=old.edit(stock, coords, l.tolist()))
    entry = dict(coordinates=coords.tolist(), fixed_pairs=f_pairs.tolist(), bounded_pairs=l.tolist(),
                 independent_pairs=i.tolist(), cap_squared=cap, fixed=original["fixed"],
                 free_matched_count=original["free_matched_count"], bounded=bounded)
    return entry, receipts


def verify_geometry(sparse, stock, entry, np):
    shape = stock["size"]
    h, w = shape
    coords = np.asarray(entry["coordinates"])
    k = entry["fixed"]["swaps"]
    need(coords.shape == (len(coords), 2) and 0 < k <= min(8, len(coords))
         and len(coords) <= 32 and coords.dtype.kind in "iu"
         and len(np.unique(coords)) == coords.size and np.all((0 <= coords) & (coords < h*w)),
         "Native disjoint two-sign edit support")
    ends = np.cumsum(sparse.decode_runs(stock["counts"]), dtype=np.int64)
    membership = np.searchsorted(ends, (coords % w)*h + coords//w, side="right") % 2
    need(not membership[:, 0].any() and membership[:, 1].all(), "Signed native occupancy support")
    distances = native_distances(coords, w)
    for arm, key in zip(ARMS, ("fixed_pairs", "bounded_pairs", "independent_pairs"), strict=True):
        selected = np.asarray(entry[key])
        record = entry[arm]
        need(selected.shape == (k, 2) and selected.dtype.kind in "iu"
             and np.all((0 <= selected) & (selected < len(coords)))
             and all(np.unique(selected[:, side]).size == k for side in range(2))
             and record["swaps"] == k and record["rle"]["size"] == shape
             and sparse.xor_rle(stock, coords[selected, np.arange(2)].ravel()) == record["rle"],
             "Independent native RLE reconstruction and exact same-area signed XOR")
        lengths = distances[selected[:, 0], selected[:, 1]]
        if arm == "fixed":
            need(np.array_equal(selected[:, 0], selected[:, 1]) and lengths.max() == entry["cap_squared"],
                 "Actual fixed rows define the cap")
        if arm == "bounded":
            need(np.all(lengths <= entry["cap_squared"]), "Every bounded edge respects the cap")


def original_inputs(a, c):
    a.manifest = a.owner / "TEST232_IMAGE_ONLY_MANIFEST_PRIVATE.json"
    a.manifest_sha = MANIFEST
    a.accepted = a.owner / "CAPTURE_READBACK/INDEPENDENT_SAFE.json"
    a.accepted_sha = CAPTURE_READBACK
    a.images = a.owner / "IMAGE_ONLY"
    replay = c.module(a.code / "PSEG_TEST232_DFINE_FROZEN_REPLAY_20261121/replay.py", REPLAY, "_pseg_frozen_replay")
    m = replay.modules(a.code, a)
    m.author.cpu_runtime()
    roster, accepted = replay.admit(m, a)
    capture = json.loads(c.pin(a.capture / "RESULT_SAFE.json", CAPTURE))
    terminal = json.loads(c.pin(a.owner / "CAPTURE_READBACK/TERMINAL_PRIVATE.json", CAPTURE_TERMINAL))
    c.pin(a.owner / "CAPTURE_READBACK/ORIGINAL_TOOL_WAIT_PRIVATE.json", CAPTURE_WAIT)
    need(accepted["export_result_sha256"] == terminal["export_result_sha256"] == CAPTURE
         and accepted["export_terminal_sha256"] == CAPTURE_TERMINAL
         and accepted["original_tool_wait_sha256"] == CAPTURE_WAIT
         and terminal["original_os_exit"] == accepted["original_os_exit"] == 0
         and capture["status"] == "DFINE_HRSID_E24_SSDD_TEST232_PROBABILITY_EXPORT_COMPLETE_V1"
         and capture["GT_opened"] is capture["AP"] is False
         and capture["images"] == capture["model_forwards"] == 232
         and all(capture[k] == accepted[k] for k in
                 ("manifest_sha256", "ema_sha256", "post_nms_rows", "probability_rows",
                  "predictions_sha256", "candidate_audit_sha256", "probability_index_sha256"))
         and accepted["probability_rows"] == 23200, "Prior natural D-FINE capture closure and native readback")
    before = json.loads(c.pin(a.old_replay / "RESULT_SAFE.json", OLD_RENDER))
    readback = json.loads(c.pin(a.old_readback / "RESULT_SAFE.json", OLD_READBACK))
    replay.check_terminal(json.loads(c.pin(a.owner / "REPLAY_TERMINAL_PRIVATE.json", OLD_WAIT)), OLD_RENDER)
    identity = replay.output_record(m, a, accepted)
    need(before["status"] == replay.PREPARED and readback["status"] == replay.VERIFIED
         and readback["replay_sha256"] == OLD_RENDER and readback["original_tool_wait_sha256"] == OLD_WAIT
         and readback["exact_native_support"] is True and before["edits_sha256"] == readback["edits_sha256"] == OLD_EDITS
         and all(before[k] == readback[k] == v for k, v in identity.items())
         and all(r["counts"]["fixed_edited"] == r["counts"]["free_matched_count_edited"] == 6404
                 and r["counts"]["fixed_swaps"] == r["counts"]["free_matched_count_swaps"] == 29304
                 for r in (before, readback)), "Original fully closed five-arm FIT109 replay and exact old signed budget")
    return m, roster, accepted, before


def replay_phase(a, c, np):
    import torch
    from pycocotools import mask as codec

    started = perf_counter()
    worker = c.digest(Path(__file__).read_bytes())
    m, roster, capture, before = original_inputs(a, c)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    need(not torch.cuda.is_initialized(), "CPU-only exact point replay")
    solver = c.module(a.code / "PHOENIX_BOUNDED_MATCH_20261123/matching.py", c.MATCH, "_exact_matching")
    sparse = c.module(a.code / "RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py", c.SPARSE, "_native_sparse")
    a.refit_result = a.refit_root / "ACTUAL/RESULT_SAFE.json"
    a.refit_acceptance = a.refit_root / "INDEPENDENT/RESULT_SAFE.json"
    a.weights = a.refit_root / "ACTUAL/POINT_HEAD_PRIVATE.npz"
    c.pin(a.refit_acceptance, c.REFIT_ACCEPT)
    c.pin(a.weights, c.HEAD)
    models = m.old.restore(a)
    models[0].raw = m.legacy.pin(a.pair_math, m.native.PAIR)

    verify = a.phase == "verify"
    if verify:
        earlier = c.closed(a.work_root / "render/RESULT_SAFE.json", a.work_root / "observed_render/TERMINAL.json",
                           a.render_sha, a.render_terminal_sha, "prepare-edits", RENDERED)
        need(earlier["worker_sha256"] == worker and earlier["protocol_sha256"] == a.protocol_sha
             and earlier["prelock_commit"] == a.prelock_commit and earlier["label_files_opened"] == 0
             and earlier["detector_forwards"] == earlier["optimizer_updates"] == 0,
             "Same witnessed frozen renderer for independent full replay")
        output = a.work_root / "render/EDITS_PRIVATE.jsonl"
    else:
        a.output.mkdir()
        output = a.output / "EDITS_PRIVATE.jsonl"
    counts, solver_totals = Counter(), Counter()
    digest, original = sha256(), sha256()
    with output.open("rb" if verify else "xb") as tape, (a.old_replay / "EDITS_PRIVATE.jsonl").open("rb") as old:
        for item, coco, audit, planes in m.legacy.blocks(a, m.author, roster, capture, True):
            image = m.legacy.rgb(a, item)
            rgb = c.digest(image.tobytes(order="C"))
            native, probability = m.native.make_native(models[0], models[1], *item["original_hw"])
            baseline, trace = c.old_masks_and_points(m, native, probability, models, coco, audit, planes, image, counts)
            baseline_bytes = m.author.encoded(dict(image_id=item["image_id"], rgb_sha256=rgb, changes=baseline))
            need(old.readline() == baseline_bytes, "Every original five-arm edit row must replay byte-for-byte")
            original.update(baseline_bytes)
            changes = {}
            for pos, arms in baseline.items():
                if "fixed" not in arms:
                    need("free_matched_count" not in arms, "Zero-budget matched I passes through")
                    continue
                rank = int(pos)
                stock_rle = coco[rank-1]["segmentation"]
                stock = codec.decode(stock_rle).astype(bool, order="C")
                coords, q = trace[rank]
                entry, measures = signed_record(arms, coords, q, stock, audit[rank-1]["mask_area"], native,
                                                models[2], m.old, solver, np)
                verify_geometry(sparse, stock_rle, entry, np)
                changes[pos] = entry
                solver_totals.update(measures)
                counts.update(positive=1, swaps=entry["fixed"]["swaps"])
            counts.update(images=1, candidates=len(coco))
            written = c.encoded(dict(image_id=item["image_id"], rgb_sha256=rgb, changes=changes))
            digest.update(written)
            if verify:
                need(tape.readline() == written, "Independent replay must match the complete new edit row")
            else:
                need(tape.write(written) == len(written), "Complete output row")
        need(not old.read(1) and original.hexdigest() == OLD_EDITS == before["edits_sha256"],
             "Closed archived edit stream identity")
        if verify:
            need(not tape.read(1) and earlier["edits_sha256"] == digest.hexdigest()
                 and earlier["counts"] == dict(counts) and earlier["solver_totals"] == dict(solver_totals),
                 "Whole independent replay archive, counters and actions agree exactly")
    need((counts["images"], counts["candidates"], counts["positive"], counts["swaps"]) == (232,23200,6404,29304),
         "Original full image and exact signed swap populations")
    if verify:
        a.output.mkdir()
    result = dict(status=VERIFIED if verify else RENDERED, worker_sha256=worker,
                  protocol_sha256=a.protocol_sha, prelock_commit=a.prelock_commit, manifest_sha256=MANIFEST,
                  capture_result_sha256=CAPTURE, capture_terminal_sha256=CAPTURE_TERMINAL,
                  capture_readback_sha256=CAPTURE_READBACK, old_replay_result_sha256=OLD_RENDER,
                  old_readback_sha256=OLD_READBACK, old_replay_terminal_sha256=OLD_WAIT, old_edits_sha256=OLD_EDITS,
                  eof_complete=True, exact_archived_five_arm_bytes=True, original_f_i_reconstructed=True,
                  exact_signed_cap_readback=True, independent_replay=verify, weight_sha256=c.HEAD,
                  render_result_sha256=a.render_sha if verify else None,
                  render_terminal_sha256=a.render_terminal_sha if verify else None,
                  solver_sha256=c.MATCH, label_files_opened=0, detector_forwards=0, optimizer_updates=0,
                  edits_sha256=digest.hexdigest(), counts=dict(counts), solver_totals=dict(solver_totals),
                  elapsed_seconds=perf_counter()-started)
    result_sha = m.author.write_new(a.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=result["status"], result_sha256=result_sha)), flush=True)


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("render","verify"))
    for key in ("code","work-root","output","protocol","owner","capture","old-replay","old-readback","refit-root",
                "numpy-site","codec-site"):
        parser.add_argument("--"+key, type=Path, required=True)
    for key in ("protocol-sha", "prelock-commit"):
        parser.add_argument("--"+key, required=True)
    parser.add_argument("--render-sha")
    parser.add_argument("--render-terminal-sha")
    a = parser.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and a.work_root.is_dir() and not a.output.exists() and a.output.resolve() == a.work_root.resolve() / a.phase
         and len(a.prelock_commit) == 40 and all(c in "0123456789abcdef" for c in a.prelock_commit)
         and (a.phase != "verify" or (a.render_sha and a.render_terminal_sha)),
         "Fresh isolated CPU process under observed published prelock")
    for path in (a.numpy_site, a.codec_site):
        sys.path.insert(0, str(path.resolve(strict=True)))
    import numpy as np

    need(np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11",
         "Frozen native replay numerical runtime")
    from runpy import run_path

    common_path = a.code / "DFINE_RSDD74_COMMON_CAP_20261126/run_dfine.py"
    source = common_path.read_bytes()
    need(sha256(source).hexdigest() == COMMON, "Exact released original PointHead trace worker")
    bindings = run_path(str(common_path))
    bindings["old_masks_and_points"].__globals__["np"] = np
    common = SimpleNamespace(**bindings)
    common.pin(a.protocol, a.protocol_sha)
    replay_phase(a, common, np)


if __name__ == "__main__":
    main()
