"""Read only the closed B5 image manifest, native predictions and occupancy edits."""
from hashlib import sha256
from pathlib import Path
from types import ModuleType


PRIOR_INPUTS = "2c3b84f852bb4f8234d935347708a2e71eba4466d0eb111471bd254aa2f45e58"
SECOND_REPLAY = (
    "489eb5591b39a12db9c6bbdf2e3c0cc661ef60f7a0ed019d3e6c9614bc380f6f",
    "c3579aba79476781a4ffc7dea08f66ec98ef364f09ae1dba77cf7821c9b5c9ac",
)
SEEDS = {74: (5217,), 740: (5217, 5218)}
HEADS = {5217: "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1",
         5218: "08b39b9911f77c039b26f8bb9bc34e940f5a350390911586f47290b2c3d88869"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_module(path, expected):
    source = path.read_bytes()
    require(sha256(source).hexdigest() == expected, "Source digest mismatch for " + path.name)
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def add_input_arguments(parser):
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--cohort", type=int, choices=tuple(SEEDS), required=True)
    parser.add_argument("--stock-root", type=Path, required=True)


def load_population(args):
    """Return (item, audit, native row, {str(seed): original edit row}) tuples and safe lineage."""
    cohort = args.cohort
    reserve = cohort == 74
    old = load_module(args.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123/inputs.py", PRIOR_INPUTS)
    shared = old.module(args.code / "INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py", old.PORTABLE)
    # This function opens only the image manifest, captured native streams and their receipts.
    _, items, predictions, capture = old.closed_captures(args, cohort, shared)
    cap_id, observer_id = old.CAPTURE[cohort]
    lineage = dict(cohort=cohort, images=cohort, capture_result_sha256=cap_id, capture_terminal_sha256=observer_id,
                   mapped_candidates=capture["counts"]["mapped"], eligible=capture["counts"]["eligible"],
                   roster_sha256=capture.get("roster_sha256"))
    for field in ("manifest_sha256", "predictions_sha256", "audit_sha256", "checkpoint_sha256"):
        lineage[field] = capture[field]
    for field in ("occupancy_result_sha256", "occupancy_terminal_sha256", "occupancy_edits_sha256",
                  "point_head_sha256", "expected_swaps", "expected_positive_budget_candidates"):
        lineage[field] = {}
    ledgers = []
    keys = []
    for seed in SEEDS[cohort]:
        key = str(seed)
        keys.append(key)
        pins = old.OCCUPANCY[cohort] if seed == 5217 else SECOND_REPLAY
        folder = "replay" if reserve else "replay" + key
        receipt = old.closed(args.stock_root, folder, "observed_" + folder, pins, "verify-edits")
        require(receipt["status"] == ("B5_RESERVE74_REPLAY_COMPLETE" if reserve else "B5_RSDD740_REPLAY_COMPLETE")
                and receipt["target_labels_opened"] is False and receipt["optimizer_updates"] == 0
                and receipt["point_head_sha256"] == HEADS[seed]
                and all(receipt[name] == capture[name] for name in
                        ("manifest_sha256", "predictions_sha256", "audit_sha256", "checkpoint_sha256", "images")),
                "Archived occupancy result must bind this exact carrier")
        if reserve:
            require(receipt["independent_decisions_equal"] is True and receipt["roster_sha256"] == shared.ROSTER,
                    "Independent reserve replay required")
        else:
            require(receipt["capture_sha256"] == cap_id and receipt["point_head_seed"] == seed
                    and receipt["model_forward_calls"] == 0 and receipt["exact_replayed_policy_choices"] is True
                    and receipt["exact_support_xor_area"] is True, "Original B5 seed replay required")
        counts = receipt["counts"]
        require(counts["images"] == cohort and counts["mapped_candidates"] == lineage["mapped_candidates"]
                and counts["fixed_swaps"] == counts["free_matched_count_swaps"]
                and counts["fixed_candidates"] == counts["free_matched_count_candidates"],
                "Archived fixed and independent counts differ")
        lineage["occupancy_result_sha256"][key], lineage["occupancy_terminal_sha256"][key] = pins
        lineage["occupancy_edits_sha256"][key] = receipt["edits_sha256"]
        lineage["point_head_sha256"][key] = receipt["point_head_sha256"]
        lineage["expected_swaps"][key] = counts["fixed_swaps"]
        lineage["expected_positive_budget_candidates"][key] = counts["fixed_candidates"]
        ledgers.append(old.checked_lines(args.stock_root / folder / shared.EDITS, receipt["edits_sha256"]))

    rows = []
    for item, (audit, pred), *edits in zip(items, predictions, *ledgers, strict=True):
        image = item["image_id"]
        require(image == audit["image_id"] == pred["image_id"] and item["original_hw"] == audit["original_hw"] == [512, 512]
                and audit["native_ranks"] == pred["native_ranks"] == sorted(set(audit["native_ranks"])),
                "Native row order, grid or rank mismatch")
        native = pred["coco"]
        for pos, (rank, query, cand, det) in enumerate(
                zip(audit["native_ranks"], audit["query_ids"], audit["candidates"], native, strict=True), 1):
            require((pos, rank, query, det["score"]) == (cand["position"], cand["native_rank"], cand["query_id"], cand["score"])
                    and 1 <= rank <= 100 and det["image_id"] == image and det["category_id"] == 1
                    and det["segmentation"]["size"] == [512, 512], "Native compact position, query or grid differs")
        for row in edits:
            require(set(row) == {"image_id", "changes"} and row["image_id"] == image, "Old occupancy row order differs")
            for k, options in row["changes"].items():
                require(k.isdecimal() and str(int(k)) == k and 1 <= int(k) <= len(native) and options,
                        "Unknown original compact position")
                require(("fixed" in options) == ("free_matched_count" in options),
                        "Archived same-count arms have unequal support")
        rows.append((item, audit, pred, dict(zip(keys, edits, strict=True))))
    require(len(rows) == cohort and len({r[0]["image_id"] for r in rows}) == cohort
            and sum(len(r[2]["coco"]) for r in rows) == lineage["mapped_candidates"]
            and sum(len(r[1]["logit_ranks"]) for r in rows) == lineage["eligible"],
            "Native population census differs")
    return rows, lineage
