"""Frozen, same-count comparison with spatial attraction on PSeg-SSDD."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask

REPLAY_SHA = "f1b6e83e98851bf867bd72f3ecb082ca2997632bdd1aedaaf64110cf06b04fd5"
SCORE_SHA = "dd8e435976d2331146c8f83d70f368ee4e0954c05520c54fa9a3b1205838d40f"
GEOMETRY_SHA = "107f31864bbb7e807a4194e552c7a1778d410f0e564f80437c5717fe6b47098d"
ADAPTER_SHA = "782d6b12c5d22d933827d05e78feb7bdc9864224bfe55a190fed117f706d94bd"
READBACK_SHA = "e99d0f2bbd9e1839e260a88075f6af187e05ffc2b86abaf6cc33b788372b7b71"
PROTOCOL_SHA = "a9ffef048cd77d5019e78e93d5db93dd2f8fec017457173a3aae98af2318302d"
NEW_ARMS = ("attraction_dynamic", "attraction_static_at_dynamic_count", "learned_at_attraction_count")
CONTRASTS = ((NEW_ARMS[2], NEW_ARMS[0]), (NEW_ARMS[2], NEW_ARMS[1]), (NEW_ARMS[1], NEW_ARMS[0]))
PREPARED = "PSEG_CLASSICAL_EDITS_PREPARED"
VERIFIED = "PSEG_CLASSICAL_EDITS_VERIFIED"
SCORED = "PSEG_CLASSICAL_MATCHED_FROZEN_SCORE_COMPLETE"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pinned(path, expected):
    data = path.read_bytes()
    need(digest(data) == expected, "Pinned bytes: " + path.name)
    return data


def document(path, expected):
    return json.loads(pinned(path, expected))


def block_stream(path, expected):
    return [json.loads(line) for line in pinned(path, expected).splitlines()]


def load_module(path, expected):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(pinned(path, expected), str(path), "exec"), module.__dict__)
    return module


def validate_extra_budget(arms):
    swaps = tuple(arms.get(name, {}).get("swaps", 0) for name in NEW_ARMS)
    need(all(type(n) is int and 0 <= n <= 8 for n in swaps) and len(set(swaps)) == 1,
         "All classical and learned arms must have the same actual per-candidate swaps")
    return swaps[0]


def inputs(a):
    a.manifest = a.owner / "TEST232_IMAGE_ONLY_MANIFEST_PRIVATE.json"
    a.accepted = a.owner / "CAPTURE_READBACK/INDEPENDENT_SAFE.json"
    a.images = a.owner / "IMAGE_ONLY"
    a.refit_result = a.refit / "ACTUAL/RESULT_SAFE.json"
    a.refit_acceptance = a.refit / "INDEPENDENT/RESULT_SAFE.json"
    a.weights = a.refit / "ACTUAL/POINT_HEAD_PRIVATE.npz"
    replay = load_module(a.code / "PSEG_TEST232_DFINE_FROZEN_REPLAY_20261121/replay.py", REPLAY_SHA)
    verified = document(a.original_readback, READBACK_SHA)
    need(verified["status"] == replay.VERIFIED and verified["exact_native_support"] is True,
         "Original five-arm readback")
    terminal = document(a.owner / "REPLAY_TERMINAL_PRIVATE.json", verified["original_tool_wait_sha256"])
    replay.check_terminal(terminal, verified["replay_sha256"])
    original = document(a.original_replay / "RESULT_SAFE.json", verified["replay_sha256"])
    need(original["edits_sha256"] == verified["edits_sha256"], "Original five-arm replay binding")
    a.manifest_sha = verified["manifest_sha256"]
    a.accepted_sha = verified["acceptance_sha256"]
    m = replay.modules(a.code, a)
    m.author.cpu_runtime()
    roster, receipt = replay.admit(m, a)
    need(receipt["probability_rows"] == 23200, "Exact native TEST232 population")
    old_edits = block_stream(a.original_replay / m.legacy.EDITS, verified["edits_sha256"])
    need(len(old_edits) == len(roster) == 232, "Original aligned test population")
    return replay, m, roster, receipt, old_edits


def crosscheck(row, original):
    projected = {}
    counts = Counter()
    need(row["image_id"] == original["image_id"] and row["rgb_sha256"] == original["rgb_sha256"],
         "Original labeled-off image identity and decoded RGB")
    for rank, arms in row["changes"].items():
        budget = validate_extra_budget(arms)
        counts["extra_candidate_edits"] += int(budget > 0)
        counts["extra_swaps_per_arm"] += budget
        kept = {name: edit for name, edit in arms.items() if name not in NEW_ARMS}
        if kept:
            projected[rank] = kept
    need(projected == original["changes"], "All five original decision streams must reproduce")
    return counts


def closed(a, phase, status):
    root = a.phase_root
    terminal = json.loads((root / ("observed_" + phase) / "TERMINAL.json").read_bytes())
    need(terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
         and terminal["phase"] == phase and terminal["worker_sha256"] == a.source_sha,
         "Observed original exit and worker identity")
    result = document(root / phase / "RESULT_SAFE.json", terminal["result_sha256"])
    need(result["status"] == status and result["protocol_sha256"] == PROTOCOL_SHA,
         "Exact stage qualification")
    return result


def generate(a, replay, m, roster, receipt, old_edits):
    import torch

    adapter = load_module(a.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/ueikap_transfer.py", ADAPTER_SHA)
    adapter.extended_edits(a, m)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    frozen = m.old.restore(a)
    frozen[0].raw = m.legacy.pin(a.pair_math, m.native.PAIR)
    counts, edit_digest = Counter(), sha256()
    with (a.output / m.legacy.EDITS).open("xb") as handle:
        blocks = m.legacy.blocks(a, m.author, roster, receipt, True)
        for (item, coco, audit, planes), old in zip(blocks, old_edits, strict=True):
            rgb = m.legacy.rgb(a, item)
            row = replay.emit(m, a, item, coco, audit, planes, rgb, frozen, Counter())
            counts.update(crosscheck(row, old))
            counts["candidates"] += len(coco)
            data = m.author.encoded(row)
            need(handle.write(data) == len(data), "Complete edit block")
            edit_digest.update(data)
        handle.flush()
        os.fsync(handle.fileno())
    need(counts["candidates"] == 23200 and counts["extra_swaps_per_arm"] > 0, "Complete replay")
    return dict(status=PREPARED, counts=dict(counts), edits_sha256=edit_digest.hexdigest(), GT_opened=False)


def accepted_streams(a, m, receipt, old_edits):
    prepared = closed(a, "prepare-edits", PREPARED)
    extended = block_stream(a.phase_root / "prepare-edits" / m.legacy.EDITS, prepared["edits_sha256"])
    predictions = block_stream(a.capture / "DEV_MASK_PREDICTIONS_PRIVATE.jsonl", receipt["predictions_sha256"])
    audits = block_stream(a.capture / "DEV_CANDIDATE_AUDIT_PRIVATE.jsonl", receipt["candidate_audit_sha256"])
    need(len(extended) == len(predictions) == len(audits) == len(old_edits), "Full edit and candidate streams")
    return prepared, extended, predictions, audits


def verify(a, replay, m, roster, receipt, old_edits):
    prepared, extended, predictions, audits = accepted_streams(a, m, receipt, old_edits)
    rank_math = m.legacy.load(a.rank_math, m.old.RANK_MATH, "_verified_support")
    counts, checked = Counter(), Counter()
    for r, p, k, edits, original in zip(roster, predictions, audits, extended, old_edits, strict=True):
        need(r["image_id"] == p["image_id"] == k["image_id"] == edits["image_id"], "Native candidate join")
        native, audit = p["coco"][:100], k["candidates"][:100]
        need(len(native) == len(audit) == 100 and all(
            d["native_rank"] == i and d["score"] == n["score"] and int(mask.area(n["segmentation"])) == d["mask_area"]
            for i, (n, d) in enumerate(zip(native, audit, strict=True), 1)), "Native ranking and area")
        counts.update(crosscheck(edits, original))
        counts["candidates"] += len(native)
        replay.check_support(rank_math, native, audit, edits["changes"], checked)
    need(dict(counts) == prepared["counts"] and counts["candidates"] == 23200, "Independent replay census")
    for arm in NEW_ARMS:
        need(checked[arm + "_edited"] == counts["extra_candidate_edits"]
             and checked[arm + "_swaps"] == counts["extra_swaps_per_arm"], "Independent exact XOR census")
    return dict(status=VERIFIED, counts=dict(counts), checked=dict(checked),
                edits_sha256=prepared["edits_sha256"], GT_opened=False)


def score(a, replay, m, roster, receipt, old_edits):
    verified = closed(a, "verify-edits", VERIFIED)
    prepared, extended, predictions, audits = accepted_streams(a, m, receipt, old_edits)
    need(prepared["edits_sha256"] == verified["edits_sha256"] and prepared["counts"] == verified["counts"],
         "Independent native support verification before target labels")
    scorer = load_module(a.code / "PSEG_TEST232_DFINE_FROZEN_SCORE_20261121/score_test232.py", SCORE_SHA)
    geom = load_module(a.code / "PSEG_TEST232_PAIRED_GEOMETRY_20261121/paired_geometry.py", GEOMETRY_SHA)
    earlier = document(a.old_score, geom.SCORE)
    scorer.ARMS += NEW_ARMS
    blocks = scorer.join_blocks(roster, predictions, audits, extended)
    gt = document(a.gt, scorer.GT_SHA)
    need(len(gt["images"]) == 232 and len(gt["annotations"]) == 546, "Frozen official population")
    views = scorer.score_populations(gt, blocks, scorer.EXCLUDED)
    for view, item in views.items():
        actual, expected = item["metrics_fraction"], earlier["views"][view]["metrics_fraction"]
        need(all(actual[arm] == expected[arm] for arm in replay.MODES), "Exact old COCO arm parity")
        item["paired_contrasts_points"] = {}
        for arm, baseline in CONTRASTS:
            item["paired_contrasts_points"][arm + " minus " + baseline] = {
                key: 100 * (val - actual[baseline][key]) if min(val, actual[baseline][key]) >= 0 else None
                for key, val in actual[arm].items()}
    del blocks
    return dict(status=SCORED, counts=verified["counts"], views=views, GT_opened=True,
                geometry=score_geometry(a, geom, gt, roster, predictions, audits, extended, earlier),
                edits_sha256=verified["edits_sha256"], prior_score_sha256=geom.SCORE)


def score_geometry(a, g, gt, roster, preds, audits, extended, earlier):
    sys.path.append(str(a.shapely_site))
    import shapely

    need(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Frozen geometry runtime")
    base = g.load(a.code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/score_endpoint.py", g.OLD)
    shared = g.load(a.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", g.GEOM)
    modules = shared.geometry_modules(a.code)
    truth, neutral, _ = modules.geometry.polygon_support(gt, modules.vector)
    need(len(truth) == 546 and not neutral, "All original polygons")
    ids_by_name = {row["file_name"]: row["id"] for row in gt["images"]}
    records = {}
    for r, p, k, e in zip(roster, preds, audits, extended, strict=True):
        name = r["image_ref"]["path"]
        records[ids_by_name[name]] = dict(name=name, rgb=e["rgb_sha256"], native=p["coco"][:100],
                                          audit=k["candidates"][:100], edit=e["changes"])
    g.ARMS, g.PAIRS = ("raw", *NEW_ARMS), CONTRASTS
    indices = {"official": sorted(records),
               "exclude_exact_fit_overlap": sorted(i for i, r in records.items() if int(Path(r["name"]).stem) not in g.OMIT)}
    output, measured = {}, {}
    original_matches = None
    for view, ids in indices.items():
        matches, _ = g.evaluate_raw(gt, records, ids, earlier["views"][view]["metrics_fraction"]["raw"])
        if original_matches is None:
            original_matches = matches
        else:
            need(matches == {gt_id: pair for gt_id, pair in original_matches.items() if pair[0] in ids},
                 "Fixed match identity in the common population")
        rows, edited = [], 0
        for gt_id, (image_id, rank) in matches.items():
            record = records[image_id]
            area = record["audit"][rank - 1]["mask_area"]
            if not 32 <= area < 1024:
                continue
            original = record["native"][rank - 1]["segmentation"]
            arm_edits = record["edit"].get(str(rank), {})
            edited += int(validate_extra_budget(arm_edits) > 0)
            if gt_id not in measured:
                masks = [original] + [arm_edits.get(arm, {}).get("rle", original) for arm in NEW_ARMS]
                need(all(int(mask.area(rle)) == area for rle in masks), "Exact equal-area geometry")
                measured[gt_id] = np.asarray([base.continuous(rle, area, truth[gt_id][0], modules) for rle in masks])
            rows.append((record["rgb"], measured[gt_id]))
        output[view] = g.summarize(rows)
        output[view].update(original_native_matches=len(matches), edited_original_matches=edited)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare-edits", "verify-edits", "score"))
    for name in ("code", "owner", "capture", "original-replay", "original-readback", "refit",
                 "protocol", "phase-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("gt", "old-score", "shapely-site"):
        p.add_argument("--" + name, type=Path)
    a = p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ["CUDA_VISIBLE_DEVICES"] == "-1",
         "Isolated CPU-only execution")
    if a.phase == "score":
        need(a.gt is not None and a.old_score is not None and a.shapely_site is not None, "Score inputs required")
    else:
        need(a.gt is None and a.old_score is None and a.shapely_site is None, "Mask generation receives no target labels")
    pinned(a.protocol, PROTOCOL_SHA)
    a.source_sha = digest(Path(__file__).read_bytes())
    a.output.mkdir()
    start = perf_counter()
    args = inputs(a)
    operation = {"prepare-edits": generate, "verify-edits": verify, "score": score}[a.phase]
    result = operation(a, *args)
    result.update(protocol_sha256=PROTOCOL_SHA, source_sha256=a.source_sha, original_readback_sha256=READBACK_SHA,
                  detector_forwards=0, optimizer_updates=0, elapsed_worker_seconds=perf_counter() - start)
    data = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with (a.output / "RESULT_SAFE.json").open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(dict(status=result["status"], result_sha256=digest(data))), flush=True)


if __name__ == "__main__":
    main()

