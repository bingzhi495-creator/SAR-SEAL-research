"""Label-free PointRend crossover with the same archived signed quota and distance cap."""
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

LEARNED = "34d4e588dc757dd60c975ebe9498d479a80265bf568422d57b3fa0fa1f6975f5"
LEARNED_RESULT = "4a015eef90344a4d06bd77fedb5fd5a1ea9d5742811f7de01a96dcb6a24ac994"
LEARNED_TERMINAL = "92fea4ab053d319397b823bafe3cab7991c793c55a17802df0bd6bb64b11bcee"
PR = "f8a0ab392bba3134a9a100731d0cbfdcefae0c590c5d0aa64536eeea6efb55fe"
PR_RESULT = "dbc6e2dbce37cdd8fea34ccefbce16af0ed1ce58db2495de682e278921626fc8"
PR_TERMINAL = "62b9eecb1411169db4e3620822ff957e555a66981b5af8ee570582044ce04131"
BASIS_RESULT = "0d788541d84cf75989941fb6d8242a05a830f5753a46ee4067c598baaae5a830"
BASIS_TERMINAL = "99758990e53154446e4795394da913ce5e98e566ee68cf214b5ffa96a6d8f75b"
STATUS = "RSDD74_POINTREND_COMMON_CAP_RENDER_COMPLETE"
LEARNED_ARMS = ("fixed", "bounded", "free_matched_count")
PR_ARMS = ("pr_fixed", "pr_bounded", "pr_independent")


def require(condition, error):
    if not condition:
        raise ValueError(error)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    raw = Path(path).read_bytes()
    require(digest(raw) == expected, f"Changed frozen input {Path(path).name}")
    return raw


def load_module(path, expected):
    path = Path(path)
    name = "_common_cap_" + path.stem + "_" + expected[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(pin(path, expected), str(path), "exec"), module.__dict__)
    return module


def learned_module(code):
    return load_module(code / "LEARNED_BOUNDED_PAIRING_20261126/rsdd/render.py", LEARNED)


def closure(result, result_sha, terminal, terminal_sha, phase, worker=None):
    summary = json.loads(pin(result, result_sha))
    observed = json.loads(pin(terminal, terminal_sha))
    require(observed["phase"] == phase and observed["actual_exit"] == 0
            and observed["output_footer_verified"] is True and observed["result_sha256"] == result_sha
            and observed["worker_sha256"] == summary["worker_sha256"]
            and (worker is None or observed["worker_sha256"] == worker),
            "Archived phase must have closed by observed natural exit and exact receipt")
    return summary


def select_pr(solver, coords, prob, sf_rows, width):
    """Optimize PointRend writes on a cap fixed by archived occupancy SF."""
    require(prob.dtype == np.float32 and coords.shape == prob.shape and coords.ndim == 2 and coords.shape[1] == 2,
            "Coincident signed frozen PointRend probabilities and endpoint rows")
    require(len(sf_rows) > 0 and len(np.unique(sf_rows)) == len(sf_rows)
            and np.all((sf_rows >= 0) & (sf_rows < len(coords))), "Original positive SF row choices")
    k = len(sf_rows)
    add_row, add_col = np.divmod(coords[:, 0], width)
    drop_row, drop_col = np.divmod(coords[:, 1], width)
    diagonal = (add_row - drop_row)**2 + (add_col - drop_col)**2
    cap = int(diagonal[sf_rows].max())
    pa = [int(float(value) * solver.BINARY32_SCALE) for value in prob[:, 0]]
    pd = [int(float(value) * solver.BINARY32_SCALE) for value in prob[:, 1]]
    fixed = sorted(np.flatnonzero(diagonal <= cap), key=lambda i: (pd[i] - pa[i], i))[:k]
    add = np.argsort(-prob[:, 0], kind="stable")[:k]
    drop = np.argsort(prob[:, 1], kind="stable")[:k]
    bounded_add, bounded_drop, audit = solver.bounded_endpoints(
        coords, prob.astype(np.float64), k, width, sf_rows, sf_rows, add, drop
    )
    selected = dict(pr_fixed=[(int(i), int(i)) for i in fixed],
                    pr_bounded=list(zip(bounded_add.tolist(), bounded_drop.tolist(), strict=True)),
                    pr_independent=list(zip(add.tolist(), drop.tolist(), strict=True)))
    objective = {arm: sum(pa[i] - pd[j] for i, j in pairs) for arm, pairs in selected.items()}
    require(len(fixed) == k and cap == audit["cap_squared_sum"]
            and objective["pr_fixed"] <= objective["pr_bounded"] <= objective["pr_independent"],
            "Exact common cap and nested PointRend objective")
    return selected, cap, audit


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_rows(path, expected):
    return [json.loads(line) for line in pin(path, expected).splitlines()]


def read_archives(a, pr, learned):
    prepared = closure(a.reserve / "prepare-edits/RESULT_SAFE.json", learned.RESERVE_PREPARED,
                       a.reserve / "observed_prepare-edits/TERMINAL.json", learned.RESERVE_PREPARED_TERMINAL,
                       "prepare-edits", learned.RESERVE_REPLAY_SHA)
    verified = closure(a.reserve / "verify-edits/RESULT_SAFE.json", learned.RESERVE_VERIFIED,
                       a.reserve / "observed_verify-edits/TERMINAL.json", learned.RESERVE_VERIFIED_TERMINAL,
                       "verify-edits", learned.RESERVE_REPLAY_SHA)
    bounded = closure(a.learned / "reserve74_5217_render/RESULT_SAFE.json", LEARNED_RESULT,
                      a.learned / "observed_reserve74_5217_render/TERMINAL.json", LEARNED_TERMINAL,
                      "prepare-edits", LEARNED)
    proof = closure(a.pr / "replay/RESULT_SAFE.json", PR_RESULT, a.pr / "observed_replay/TERMINAL.json",
                    PR_TERMINAL, "replay", PR)
    basis = closure(a.pr / "basis/RESULT_SAFE.json", BASIS_RESULT, a.pr / "observed_basis/TERMINAL.json",
                    BASIS_TERMINAL, "capture")
    require(prepared["GT_opened"] is verified["GT_opened"] is False
            and verified["exact_replayed_policy_choices"] is verified["exact_support_xor_area"] is True
            and prepared["edits_sha256"] == verified["edits_sha256"] == pr.OCCUPANCY,
            "Closed and independently replayed original occupancy F and I")
    require(bounded["status"] == learned.STATUS and bounded["view"] == "reserve74" and bounded["head_seed"] == 5217
            and bounded["exact_historical_baseline_masks"] is bounded["exact_per_candidate_signed_counts"] is True
            and bounded["label_files_opened"] == bounded["detector_forwards"] == bounded["optimizer_updates"] == 0
            and bounded["prior"]["prior_verified_result_sha256"] == learned.RESERVE_VERIFIED
            and bounded["prior"]["prior_edit_sha256"] == prepared["edits_sha256"]
            and bounded["solver_sha256"] == learned.SOLVER_SHA, "Closed learned same-count bounded branch")
    require(proof["status"] == pr.STATUS and proof["GT_opened"] is False and proof["exact_independent_render"] is True
            and proof["same_endpoints_exact_realized_adds_drops"] is True and proof["fit_weight_sha256"] == pr.HEAD
            and proof["basis_result_sha256"] == BASIS_RESULT and proof["basis_terminal_sha256"] == BASIS_TERMINAL
            and proof["occupancy_edits_sha256"] == prepared["edits_sha256"]
            and proof["predictions_sha256"] == prepared["predictions_sha256"] == basis["predictions_sha256"]
            == bounded["prior"]["prior_candidate_sha256"]
            and proof["audit_sha256"] == prepared["candidate_audit_sha256"] == basis["audit_sha256"]
            == bounded["prior"]["prior_candidate_audit_sha256"], "Exact common native capture and original PointRend fit")

    source = SimpleNamespace(code=a.code, intake=a.b5 / "conversion", roster=a.b5 / "reserve.csv",
                             capture=a.b5 / "capture", occupancy=a.b5 / "replay", basis=a.pr / "basis",
                             basis_terminal=a.pr / "observed_basis/TERMINAL.json", fit_weights=a.fit_weights,
                             protocol=a.code.parent / "INSTANCE_RSDD_RESERVE_POINTREND_PRELOCK_20261123.md",
                             basis_worker=a.code / pr.TRANSFER / "capture_basis.py",
                             basis_worker_sha=pr.ORIGINAL_BASIS)
    original_population, native, original_replay, lineage = pr.read_population(source)
    require(original_replay["edits_sha256"] == prepared["edits_sha256"] and proof["capture_result_sha256"]
            == lineage["capture_result_sha256"] and lineage["basis_result_sha256"] == BASIS_RESULT
            and lineage["basis_terminal_sha256"] == BASIS_TERMINAL, "Exact reproduced input for PointRend")
    paired = read_rows(a.reserve / "prepare-edits/EDITS_PRIVATE.jsonl", prepared["edits_sha256"])
    capped = read_rows(a.learned / "reserve74_5217_render/BOUNDED_PRIVATE.jsonl", bounded["actual_bounded_stream_sha256"])
    population = list(zip(original_population, paired, capped, strict=True))
    for work, old, local in population:
        require(work[0][0]["image_id"] == old["image_id"] == local["image_id"] and work[2] == old,
                "Exact ordered match between native PointRend input and learned archives")
    require(len(population) == 74 and native["counts"]["mapped"] == 7346, "Exact frozen 74-image carrier")
    return population, source, proof, bounded, lineage


def initialize(source):
    global _old_pr, _solver, _tape
    pr = load_module(source.code / "INSTANCE_RSDD_RESERVE_POINTREND_20261123/replay_reserve_pointrend.py", PR)
    _old_pr = pr.source(source.code / pr.TRANSFER / "replay_pointrend.py", pr.REPLAY_OP)
    _old_pr.initialize(source, pr.HEAD)
    _solver = load_module(source.code / "PHOENIX_BOUNDED_MATCH_20261123/matching.py",
                          learned_module(source.code).SOLVER_SHA)
    _tape = []
    original = _old_pr.RecordedBudget

    class PreserveObserved(original):
        def project_matched_budget(self, adds, drops, k):
            _tape.append((adds, drops, k))
            return super().project_matched_budget(adds, drops, k)

    _old_pr.RecordedBudget = PreserveObserved


def pixel_indices(stock, edits, coords, codec):
    mask = codec.decode(edits["rle"]).astype(bool)
    k = edits["swaps"]
    added, dropped = np.flatnonzero(mask & ~stock), np.flatnonzero(stock & ~mask)
    require(mask.shape == stock.shape and len(added) == len(dropped) == k
            and np.isin(added, coords[:, 0]).all() and np.isin(dropped, coords[:, 1]).all(),
            "Exact signed XOR on the PointRend-tested native endpoint support")
    return np.flatnonzero(np.isin(coords[:, 0], added)), np.flatnonzero(np.isin(coords[:, 1], dropped))


def write_mask(stock, coords, pairs, codec):
    k = len(pairs)
    indices = np.asarray(pairs)
    updated = stock.copy(order="C")
    updated.flat[coords[indices[:, 0], 0]] = True
    updated.flat[coords[indices[:, 1], 1]] = False
    require(np.count_nonzero(updated & ~stock) == np.count_nonzero(stock & ~updated) == k,
            "Exact same area and k native writes per sign")
    rle = codec.encode(np.asfortranarray(updated, dtype=np.uint8))
    rle["counts"] = rle["counts"].decode("ascii")
    return dict(swaps=k, rle=rle)


def replay(frame):
    from pycocotools import mask as codec

    work, prior, bounded = frame
    row, stock_stream = work[0]
    _tape.clear()
    archive_row, old_counts = _old_pr.replay_image(work)
    require(len(_tape) == old_counts["rendered"], "Original PointRend replay endpoint observations")
    counts = Counter(images=1, mapped_candidates=len(stock_stream["coco"]),
                     point_head_evaluations=old_counts["point_evaluations"],
                     observed_roi_slots=old_counts["rendered"])
    edits = {}
    observed = iter(_tape)
    for pos, (info, native) in enumerate(zip(row["candidates"], stock_stream["coco"], strict=True), 1):
        position = str(pos)
        before = prior["changes"].get(position, {})
        if not (32 <= info["mask_area"] < 1024) or (info["box_xyxy"][2] == info["box_xyxy"][0]
                                                   or info["box_xyxy"][3] == info["box_xyxy"][1]):
            require("fixed" not in before, "An original positive SF slot cannot lack PointRend inputs")
            continue
        additions, removals, old_k = next(observed)
        require(old_k == before.get("free_natural", {}).get("swaps", 0), "Old PointRend actual native quota")
        if "fixed" not in before:
            continue
        coords = np.column_stack((np.asarray([p[0] for p in additions], dtype=np.int64),
                                  np.asarray([p[0] for p in removals], dtype=np.int64)))
        prob = np.column_stack(([p[1] for p in additions], [p[1] for p in removals]))
        require(len(coords) > 0 and np.isfinite(prob).all() and np.array_equal(prob, prob.astype(np.float32)),
                "The archived PointRend paste provides exact binary32 endpoint probabilities")
        learned = dict(fixed=before["fixed"], bounded=bounded["changes"][position],
                       free_matched_count=before["free_matched_count"])
        stock = codec.decode(native["segmentation"]).astype(bool)
        sf_add, sf_drop = pixel_indices(stock, learned["fixed"], coords, codec)
        require(np.array_equal(sf_add, sf_drop), "Exact archived SF paired-row identity in PointRend order")
        k = len(sf_add)
        for arm in ("bounded", "free_matched_count"):
            require(learned[arm]["swaps"] == k, "The closed learned arms must share original SF count")
            pixel_indices(stock, learned[arm], coords, codec)
        selected, cap, audit = select_pr(_solver, coords, prob.astype(np.float32), sf_add, stock.shape[1])
        point_edits = {arm: write_mask(stock, coords, pairs, codec) for arm, pairs in selected.items()}
        edits[position] = dict(learned, **point_edits, cap_squared=cap,
                               coordinates=coords.tolist(), sf_pair_indices=sf_add.tolist())
        counts.update(positive_candidates=1, swaps=k, pr_capped_independent_feasible=audit["independent_feasible"],
                      pr_bounded_differs_fixed=int(point_edits["pr_bounded"] != point_edits["pr_fixed"]),
                      pr_independent_differs_bounded=int(point_edits["pr_independent"] != point_edits["pr_bounded"]),
                      pr_boundary_add_score_ties=int(k < len(prob) and np.sort(prob[:, 0])[-k] == np.sort(prob[:, 0])[-k-1]),
                      pr_boundary_drop_score_ties=int(k < len(prob) and np.sort(prob[:, 1])[k-1] == np.sort(prob[:, 1])[k]))
    require(next(observed, None) is None and set(edits) == set(bounded["changes"]),
            "Whole archived positive cohort and no novel slots")
    return dict(image_id=row["image_id"], changes=edits), archive_row, counts


def run(a):
    started = perf_counter()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and not a.output.exists(), "Use qualified isolated CPU runtime and fresh output")
    require(len(a.prelock_commit) == 40 and all(c in "0123456789abcdef" for c in a.prelock_commit),
            "Require a verified published prelock commit")
    pin(a.protocol, a.protocol_sha256)
    pr = load_module(a.code / "INSTANCE_RSDD_RESERVE_POINTREND_20261123/replay_reserve_pointrend.py", PR)
    learned = learned_module(a.code)
    population, source, proof, bounded, lineage = read_archives(a, pr, learned)
    rows, old_rows, totals = [], [], Counter()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=initialize, initargs=(source,)) as pool:
        for row, old, counts in pool.map(replay, population, chunksize=2):
            rows.append(encode(row))
            old_rows.append(pr.encode(old))
            totals.update(counts)
    require(digest(b"".join(old_rows)) == proof["edits_sha256"],
            "Old PointRend complete replay bytes must match frozen original at its own quotas")
    require((totals["images"], totals["mapped_candidates"], totals["positive_candidates"], totals["swaps"])
            == (74, 7346, 4092, 30841) and totals["point_head_evaluations"] == proof["counts"]["point_evaluations"]
            and totals["positive_candidates"] == bounded["counts"]["positive_candidates"]
            and totals["swaps"] == bounded["counts"]["swaps"], "Frozen whole-cohort candidate and write ledger")
    data = b"".join(rows)
    a.output.mkdir()
    (a.output / "EDITS_PRIVATE.jsonl").write_bytes(data)
    summary = dict(lineage, status=STATUS, protocol_sha256=a.protocol_sha256, prelock_commit=a.prelock_commit,
                   worker_sha256=digest(Path(__file__).read_bytes()), edits_sha256=digest(data),
                   original_pointrend_result_sha256=PR_RESULT, original_pointrend_terminal_sha256=PR_TERMINAL,
                   original_pointrend_edits_sha256=proof["edits_sha256"], pointrend_head_sha256=pr.HEAD,
                   learned_bounded_result_sha256=LEARNED_RESULT, learned_bounded_terminal_sha256=LEARNED_TERMINAL,
                   learned_bounded_edits_sha256=bounded["actual_bounded_stream_sha256"],
                   prior_prepared_result_sha256=learned.RESERVE_PREPARED,
                   prior_verified_result_sha256=learned.RESERVE_VERIFIED,
                   solver_sha256=learned.SOLVER_SHA, counts=dict(totals), elapsed_seconds=perf_counter()-started,
                   exact_original_native_masks_and_scores=True, exact_archived_pointrend_replay_bytes=True,
                   exact_original_learned_fixed_bounded_independent=True,
                   exact_signed_counts_and_common_cap=True, label_files_opened=0, detector_forwards=0,
                   optimizer_updates=0)
    result = encode(summary)
    (a.output / "RESULT_SAFE.json").write_bytes(result)
    print(json.dumps(dict(status=STATUS, result_sha256=digest(result))), flush=True)


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "b5", "reserve", "pr", "learned", "fit_weights", "protocol", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--prelock-commit", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve(strict=name != "output"))
    run(args)


if __name__ == "__main__":
    main()
