"""Label-free coverage of the DEV705 frozen cumulative-distance witness."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment


COMMON_SHA = "45a78abb9e5162c8d6e19a9c6f5149a18fdef89457b6b8ece8234e5f16263a90"
READ_ARCHIVE_SHA = "6b20d05c28cf268e2d52289ce77394b8c6eba60869ec47f86c391e4ffcf8ec3f"
BOUNDED_RENDER_SHA = "dce9e33b4506d2bca7a0eec9ba1889f8450e123f817248fbfcdc124f10a9c2c7"
BOUNDED_TERMINAL_SHA = "6027a30bea3bc08aec00608abd8896099725ce13c300566645d810d080d3f134"
BOUNDED_PRELABEL_SHA = "02698b6bb815ba8b14a7f8be0e15437d14bf97b8c7a489c209724bab6c785069"
EXPECTED_RUNTIME = {"numpy": "1.26.4", "torch": "2.10.0+cu130", "pycocotools": "2.0.11",
                    "shapely": "2.0.2", "scipy": "1.15.3"}


def pin(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Frozen input digest mismatch: {path.name}")
    return raw


def minimum(cost):
    rows, columns = linear_sum_assignment(cost)
    return int(cost[rows, columns].sum(dtype=np.int64))


def distance(flips):
    pixels = np.asarray(flips, dtype=np.int64).reshape(-1, 2)
    rows, cols = np.divmod(pixels, 800)
    return ((rows[:, 0, None] - rows[None, :, 1]) ** 2
            + (cols[:, 0, None] - cols[None, :, 1]) ** 2)


def evaluate(fixed, independent, bounded):
    f, i, l = map(distance, (fixed, independent, bounded))
    # Matching on a binary violation matrix finds a cap-feasible assignment
    # exactly when the minimum number of violating edges is zero.
    cap = int(f.diagonal().max())
    eligible = minimum(i > cap) == 0
    cf, cl, ci = map(minimum, (f, l, i))
    if cf != int(f.diagonal().sum()):
        raise ValueError("The archived F subset is not an unrestricted minimum")
    return eligible, cf, cl, ci


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for option in ("code", "source", "bounded", "protocol", "output"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    a = parser.parse_args()
    start = perf_counter()
    if not (sys.flags.isolated and sys.dont_write_bytecode
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and not a.output.exists() and sys.version_info[:3] == (3, 10, 13)
            and all(version(dep) == v for dep, v in EXPECTED_RUNTIME.items())):
        raise ValueError("Requires the isolated frozen CPU runtime and a fresh output directory")
    pin(a.protocol, a.protocol_sha256)
    deps = a.code / "LEARNED_BOUNDED_PAIRING_20261126" / "hrsid"
    common_src = pin(deps / "bound_common.py", COMMON_SHA)
    spec = spec_from_file_location("bound_common", str(deps / "bound_common.py"))
    common = module_from_spec(spec)
    sys.modules[spec.name] = common
    exec(compile(common_src, str(deps / "bound_common.py"), "exec"), vars(common))
    load, closed, require = common.load, common.closed, common.require
    source = a.source
    a.intake, a.capture, a.replay = [source / stem for stem in ("intake", "capture", "replay")]
    a.render = a.bounded / "render"
    capture = closed(source, "capture", common.CAPTURE, common.CAP_TERMINAL, common.CAPTURE_CODE)
    replay = closed(source, "replay", common.REPLAY, common.REP_TERMINAL, common.REPLAY_CODE)
    # Render files use the original replay phase's observed receipt.
    render = json.loads(pin(a.render / "RESULT_SAFE.json", BOUNDED_RENDER_SHA))
    terminal = json.loads(pin(a.bounded / "observed_replay" / "TERMINAL.json", BOUNDED_TERMINAL_SHA))
    prelabel = json.loads(pin(a.bounded / "score" / "PRELABEL_SAFE.json", BOUNDED_PRELABEL_SHA))
    require(terminal["actual_exit"] == 0 and terminal["phase"] == "replay"
            and terminal["output_footer_verified"] is True
            and terminal["result_sha256"] == BOUNDED_RENDER_SHA
            and terminal["worker_sha256"] == render["worker_sha256"]
            and replay["edits_sha256"] == render["original_edits_sha256"] == common.OLD_EDITS
            and replay["GT_opened"] is render["labels_opened"] is prelabel["labels_opened"] is False
            and render["prior_capture_sha256"] == common.CAPTURE
            and render["prior_replay_sha256"] == common.REPLAY
            and capture["images"] == 705
            and prelabel["render_result_sha256"] == BOUNDED_RENDER_SHA
            and prelabel["render_terminal_sha256"] == BOUNDED_TERMINAL_SHA
            and prelabel["native_scores_masks_ranks_exact"] is prelabel["signed_budget_and_cap_all_rows_exact"] is True,
            "Frozen original label-free lineage or exact parity differs")
    archived = load(deps / "score_dev.py", READ_ARCHIVE_SHA, "_witness_archived")
    old, ops, sparse = archived.score_operators(a)
    old.admission(a, ops)
    prepared, totals = archived.read_archive(a, old, ops, sparse, render)
    require(dict(totals) == prelabel["totals"], "The original complete native census differs")

    a.output.mkdir()
    head_census = {}
    union = set()
    union_groups = set()
    groups_seen = set()
    for seed in common.SEEDS:
        counts = Counter()
        groups = set()
        by_budget = {k: Counter() for k in range(9)}
        for native, pool, points, group, all_edits in prepared:
            group = str(group)
            groups_seen.add(group)
            edits = all_edits[seed]
            counts["ineligible"] += len(pool) - len(points)
            counts["unsupported"] += sum(not len(x) for x in points.values())
            counts["supported_zero_budget"] += sum(bool(len(x)) and rank not in edits for rank, x in points.items())
            by_budget[0]["fallback"] += len(pool) - len(edits)
            for rank, (f, i, l) in edits.items():
                k = len(f) // 2
                counts["positive"] += 1
                counts["swaps_per_arm"] += k
                can_cap, cf, cl, ci = evaluate(f, i, l)
                reason = f"cap_feasible:{int(can_cap)},le_f:{int(ci <= cf)},le_l:{int(ci <= cl)}"
                by_budget[k][reason] += 1
                changed = set(i) != set(l)
                require(can_cap == (not changed), "Frozen bounded independent selection parity differs")
                if not can_cap and ci <= cf and ci <= cl:
                    by_budget[k]["witness"] += 1
                    counts["changed"] += 1
                    groups.add(group)
                    union.add((native["image_id"], rank))
                    union_groups.add(group)
        require((counts["positive"], counts["swaps_per_arm"]) == common.TOTAL[seed]
                and sum(counts[k] for k in ("ineligible", "unsupported", "supported_zero_budget", "positive"))
                == totals["mapped"], "Per-head denominator partition differs")
        head_census[str(seed)] = {
            "positive": counts["positive"], "swaps_per_arm": counts["swaps_per_arm"],
            "ineligible": counts["ineligible"], "unsupported": counts["unsupported"],
            "supported_zero_budget": counts["supported_zero_budget"],
            "changed": counts["changed"], "changed_groups": len(groups),
            "by_budget": {str(k): dict(v) for k, v in by_budget.items()},
        }
    passes = len(union) >= 20 and len(union_groups) >= 8
    common.write_result(a.output / "RESULT_SAFE.json", {
        "status": "DEV705_CUMULATIVE_WITNESS_COVERAGE_COMPLETE",
        "worker_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": a.protocol_sha256,
        "bounded_render_sha256": BOUNDED_RENDER_SHA,
        "bounded_render_terminal_sha256": BOUNDED_TERMINAL_SHA,
        "head_census": head_census,
        "population": dict(totals), "image_roster_groups": len(groups_seen),
        "changed_candidates_union_across_heads": len(union),
        "changed_prefix_groups_union_across_heads": len(union_groups),
        "minimum_changed_candidates": 20, "minimum_changed_prefix_groups": 8,
        "necessary_coverage_gate_passed": passes,
        "scoring_status": "ELIGIBLE_NOT_EXECUTED" if passes else "NOT_ENTERED_UTILITY_SCORING",
        "labels_opened": False, "mask_predictions_modified": False,
        "detector_forwards": 0, "optimizer_updates": 0,
        "elapsed_seconds": perf_counter() - start,
    })


if __name__ == "__main__":
    main()
