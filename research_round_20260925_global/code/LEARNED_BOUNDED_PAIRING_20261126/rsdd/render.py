"""Prelabel replay of distance-capped learned edits on closed Instance-RSDD inputs."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np


SOLVER_SHA = "99817f192798f52f03efe5e3ac422a453019bf458d33096a03844cb60360436c"
GF_INPUTS_SHA = "8d5d6b28279bd492de9585f570ca0a8f23919cdfadc0e4214602681ebc94e554"
RESERVE_AUTHORITY_SHA = "75c7f732a1cc4ad53d9d25280e698255ea6a7c31df818fd985b01ccd984b0389"
RESERVE_VERIFIED = "0f0aa047630c87653754152b8a47b8c570c9bff04928f06c6fd4b1031bcdcf97"
RESERVE_VERIFIED_TERMINAL = "540e733a074b53bd10583c2dc77921b594280c559de626a500dd692fd81609d2"
RESERVE_PREPARED = "6d0adf4a5d9043a31c05c12f26f193bf6a61213cea672ceaf5c0367d00108d35"
RESERVE_PREPARED_TERMINAL = "58765ca69a529919cce85d447785c854d017af2340f25c7b71d9d39db8479b1e"
RESERVE_REPLAY_SHA = "7238e6c5ca52133d51173388b01a28c48ec2a81dccd66139f6b7b2c4cbd8d072"
RESERVE_REPLAY_PATH = "INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/replay_reserve.py"
POLICIES = ("fixed", "free_matched_count")
STATUS = "LEARNED_BOUNDED_PAIRING_RSDD_RENDER_COMPLETE"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    data = Path(path).read_bytes()
    need(digest(data) == expected, f"Unexpected SHA-256 for {Path(path).name}")
    return data


def module(path, expected):
    path = Path(path)
    key = "bounded_rsdd_" + path.stem + "_" + expected[:12]
    spec = importlib.util.spec_from_file_location(key, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[key] = loaded
    exec(compile(pin(path, expected), str(path), "exec"), loaded.__dict__)
    return loaded


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def append_bounded(solver, actions, coords, posterior, width):
    fixed, free = (actions[key] for key in POLICIES)
    count = len(fixed)
    need(len(free) == count, "Original same-signed-count comparator parity")
    if count == 0:
        return dict(actions, bounded=[]), {}
    old_add, old_drop = (np.asarray(side, dtype=int) for side in zip(*fixed, strict=True))
    free_add, free_drop = (np.asarray(side, dtype=int) for side in zip(*free, strict=True))
    need(np.array_equal(old_add, old_drop), "Historical SF is an original row pairing")
    add, drop, stats = solver.bounded_endpoints(coords, posterior.astype(np.float64), count, width,
                                                old_add, old_drop, free_add, free_drop)
    stats["bounded_mask_differs_fixed"] = int(set(add) != set(old_add) or set(drop) != set(old_drop))
    stats["bounded_mask_differs_independent"] = int(set(add) != set(free_add) or set(drop) != set(free_drop))
    stats["ties_bounded_fixed"] = 1-stats["bounded_gain_gt_fixed"]
    stats["ties_independent_bounded"] = 1-stats["independent_gain_gt_bounded"]
    return dict(actions, bounded=list(zip(add.tolist(), drop.tolist(), strict=True))), stats


def verify_original(row, historical, all_arms):
    need(row["image_id"] == historical["image_id"], "Preserved original image identity")
    old = historical["changes"]
    original_arms = None if all_arms else set((*POLICIES, "free_natural"))
    stripped = {}
    bounded = {}
    for rank, edits in row["changes"].items():
        original = {k: v for k, v in edits.items() if k != "bounded"
                    and (original_arms is None or k in original_arms)}
        if original:
            stripped[rank] = original
        if "bounded" in edits:
            need(edits["bounded"]["swaps"] == original["fixed"]["swaps"] == original[POLICIES[1]]["swaps"],
                 "Exact historical per-candidate signed budget")
            bounded[rank] = edits["bounded"]
    need(stripped == old, "Historical masks, scores' positional identity and quotas must exactly replay")
    need(set(bounded) == {rank for rank, edits in old.items() if "fixed" in edits}, "Every historical positive-budget slot rendered")
    return dict(image_id=row["image_id"], changes=bounded)


def verify_edit(old_validate, stock, coords, edits, codec):
    old_validate(stock, coords, {k: v for k, v in edits.items() if k != "bounded"})
    if "bounded" in edits:
        row = edits["bounded"]
        mask = codec.decode(row["rle"]).astype(bool)
        additions, removals = np.flatnonzero(mask & ~stock), np.flatnonzero(stock & ~mask)
        need(row["swaps"] == edits["fixed"]["swaps"] == edits["free_matched_count"]["swaps"]
             == len(additions) == len(removals) and np.isin(additions, coords[:, 0]).all()
             and np.isin(removals, coords[:, 1]).all(), "Equal-area bounded writes on original native endpoints")


def reserve_archive(a):
    prior = module(a.code / RESERVE_REPLAY_PATH, RESERVE_REPLAY_SHA)
    spec = json.loads(pin(a.archive / "AUTHORITY_prepare-edits_PRIVATE.json", RESERVE_AUTHORITY_SHA))
    paths = {"images", "capture_root", "manifest", "code", "fit_result", "fit_accepted", "weights"}
    source = SimpleNamespace(**{k: Path(v) if k in paths else v for k, v in spec.items()})
    source.code = a.code
    need(source.capture_root == a.archive and source.weights == a.point_head,
         "Exact reserve capture root and original source head")
    prepared, prep_term = prior.closed(a.archive / "observed_prepare-edits/TERMINAL.json", RESERVE_PREPARED_TERMINAL,
                                       a.archive / "prepare-edits/RESULT_SAFE.json", RESERVE_PREPARED, "prepare-edits")
    accepted, accept_term = prior.closed(a.archive / "observed_verify-edits/TERMINAL.json", RESERVE_VERIFIED_TERMINAL,
                                        a.archive / "verify-edits/RESULT_SAFE.json", RESERVE_VERIFIED, "verify-edits")
    need(prep_term["worker_sha256"] == accept_term["worker_sha256"] == RESERVE_REPLAY_SHA
         and prepared["status"] == prior.PREPARED and accepted["status"] == prior.ACCEPTED
         and prepared["GT_opened"] is accepted["GT_opened"] is False
         and accepted["exact_replayed_policy_choices"] is True and accepted["exact_support_xor_area"] is True
         and accepted["edits_sha256"] == prepared["edits_sha256"], "Observed independent reserve edits")
    report, rows = prior.read_capture(source)
    old = [json.loads(line) for line in pin(a.archive / "prepare-edits/EDITS_PRIVATE.jsonl",
                                         prepared["edits_sha256"]).splitlines()]
    need(len(old) == len(rows) == 74 and len({r["acquisition_group"] for r, _ in rows}) == 13,
         "Exact 74-chip reserve population")
    source.workers = a.workers
    lineage = dict(prior_candidate_sha256=report["predictions_sha256"],
                   prior_candidate_audit_sha256=report["audit_sha256"],
                   prior_edit_sha256=prepared["edits_sha256"], prior_verified_result_sha256=RESERVE_VERIFIED)
    return rows, old, source, lineage


def gf_archive(a):
    inputs = module(a.code / "RSDD_STRICT_CONTINUOUS_20261123/inputs.py", GF_INPUTS_SHA)
    original = module(a.code / "SAR_B5_MAIN_TABLES_20261123/portable_rsdd740.py", inputs.B5)
    shared, native = original.native_modules(a.code)
    source = SimpleNamespace(code=a.code, intake=a.archive / "intake", capture=a.archive / "capture",
                             point_head=a.point_head, seed=a.seed, workers=a.workers)
    accepted = json.loads(pin(a.archive / "VERIFIED_SAFE.json", inputs.B5_ACCEPT))
    need(accepted["passed"] is True and accepted["difference_count"] == 0, "Exact published GF3 reproduction")
    stamps = accepted["fresh_summary_sha256"]
    report = inputs.observed(a.archive, "capture", inputs.B5, "capture", "capture",
                             stamps["capture"], inputs.B5_TERMINALS["capture"])
    items = original.load_items(source, shared, native)
    measured, cap_sha = original.capture_info(source, shared, items)
    need(measured == report and cap_sha == stamps["capture"], "Exact frozen GF3 capture")
    stage = f"replay{a.seed}"
    receipt = inputs.observed(a.archive, stage, inputs.B5, "replay", "verify-edits",
                             stamps[f"edits_{a.seed}"], inputs.B5_TERMINALS[stage])
    need(receipt["status"] == original.REPLAYED and receipt["target_labels_opened"] is False
         and receipt["exact_replayed_policy_choices"] is True and receipt["exact_support_xor_area"] is True
         and receipt["point_head_sha256"] == original.HEADS[a.seed] and receipt["capture_sha256"] == cap_sha,
         "Old GF3 replay closure and frozen seed")
    rows = original.capture_rows(source, shared, items, report)
    old = original.records(a.archive / stage / shared.EDITS, receipt["edits_sha256"])
    for key, expected in {"predictions_sha256": report["predictions_sha256"], "audit_sha256": report["audit_sha256"],
                          f"edits_{a.seed}": receipt["edits_sha256"]}.items():
        need(accepted["canonical_streams"][key] == dict(equal=True, observed_sha256=expected, expected_sha256=expected),
             "Original canonical input/evaluation seal")
    selected = [(row, pred, prior) for (row, pred), prior in zip(rows, old, strict=True)
                if row["sensor"] == "GF-3"]
    need(len(selected) == 583 and len({row["acquisition_group"] for row, _, _ in selected}) == 76
         and len({row["rgb_sha256"] for row, _, _ in selected}) == 583,
         "Exact Gaofen-3 cohort and grouping")
    lineage = dict(prior_candidate_sha256=report["predictions_sha256"],
                   prior_candidate_audit_sha256=report["audit_sha256"], prior_edit_sha256=receipt["edits_sha256"],
                   prior_verified_result_sha256=inputs.B5_ACCEPT)
    return [(r, p) for r, p, _ in selected], [s for _, _, s in selected], (source, items, inputs.B5), lineage


def initialize(a, source):
    global _worker
    if a.view == "gf3":
        src, items, expected = source
        old = module(a.code / "SAR_B5_MAIN_TABLES_20261123/portable_rsdd740.py", expected)
        old.initialize_replay(src, items)
        edit, base, methods, predict, request = old.worker
    else:
        old = module(a.code / RESERVE_REPLAY_PATH, RESERVE_REPLAY_SHA)
        old.initialize(source, False)
        edit, base, methods, predict, request, _ = old.worker
    solver = module(a.code / "PHOENIX_BOUNDED_MATCH_20261123/matching.py", SOLVER_SHA)
    original_policy = edit.policies
    original_validate = edit.validate_edits
    counts = Counter()

    def extend(m, stock, coords, posterior, native, *, independent):
        arms = original_policy(m, stock, coords, posterior, native, independent=independent)
        result, measures = append_bounded(solver, arms, coords, posterior, stock.shape[1])
        counts.update(measures)
        return result

    edit.policies = extend
    edit.validate_edits = lambda stock, coords, edits: verify_edit(original_validate, stock, coords, edits, edit.mask)
    _worker = (edit, base, methods, predict, request, counts)


def replay(frame):
    edit, base, methods, predict, request, counts = _worker
    row, pred, prior, all_arms = frame
    counts.clear()
    scored, census = edit.image_edits(base, methods, predict, request, row, pred, independent=True)
    bounded = verify_original(scored, prior, all_arms)
    counts.update(images=1, mapped_candidates=len(pred["coco"]), positive_candidates=census["bounded_candidates"],
                  swaps=census["bounded_swaps"], original_positive_candidates=census["fixed_candidates"])
    return bounded, dict(counts)


def run(a):
    started = perf_counter()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and not a.output.exists(), "Fresh isolated CPU process")
    need(a.view == "gf3" or a.seed == 5217, "Reserve uses its sole original frozen head")
    pin(a.protocol, a.protocol_sha)
    module(a.code / "PHOENIX_BOUNDED_MATCH_20261123/matching.py", SOLVER_SHA)
    if a.view == "gf3":
        rows, old, source, lineage = gf_archive(a)
    else:
        rows, old, source, lineage = reserve_archive(a)
    print("Previously closed native capture, image roster and frozen edit sources verified.", flush=True)
    a.output.mkdir()
    checksum = sha256()
    census = Counter()
    jobs = ((row, pred, previous, a.view == "gf3") for (row, pred), previous in zip(rows, old, strict=True))
    with (a.output / "BOUNDED_PRIVATE.jsonl").open("xb") as stream:
        with ProcessPoolExecutor(max_workers=a.workers, initializer=initialize, initargs=(a, source)) as pool:
            for data, counts in pool.map(replay, jobs, chunksize=4):
                raw = canonical(data)
                checksum.update(raw)
                stream.write(raw)
                census.update(counts)
        stream.flush()
        os.fsync(stream.fileno())
    need(census["images"] == len(rows) and census["positive_candidates"] == census["original_positive_candidates"] > 0,
         "Original positive-budget population renders completely")
    result = dict(status=STATUS, phase="prepare-edits", view=a.view, head_seed=a.seed,
                  protocol_sha256=a.protocol_sha, solver_sha256=SOLVER_SHA, worker_sha256=digest(Path(__file__).read_bytes()),
                  actual_bounded_stream_sha256=checksum.hexdigest(), counts=dict(census), prior=lineage,
                  label_files_opened=0, detector_forwards=0, optimizer_updates=0,
                  exact_historical_baseline_masks=True, exact_per_candidate_signed_counts=True,
                  elapsed_seconds=perf_counter() - started)
    raw = canonical(result)
    with (a.output / "RESULT_SAFE.json").open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(dict(status=STATUS, result_sha256=digest(raw))), flush=True)


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "archive", "point-head", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--view", choices=("gf3", "reserve74"), required=True)
    parser.add_argument("--seed", choices=(5217, 5218), type=int, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--protocol-sha", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
