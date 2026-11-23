"""Score the preregistered whole-mask Phoenix control after both review returns are frozen."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import shapely


PROTOCOL_NAME = "UMBRA12_PHOENIX_WHOLE_PRELOCK_20261123.md"
PROTOCOL_SHA = "49fb049a6986d1f3d34adb7469b3239f5742701723e84f4dce8e60a4425210cb"
PRIMARY_SOURCE_SHA = "f89f0d50a194e25acbd825fe5a94fa79e639f805fc6965fe9614cf7cc3180051"
READBACK_WORKER_SHA = "bb25563f4cb643664ab03832818b8fbda632b3d37307234c624420940de14469"
READBACK_RESULT_SHA = "98424aca7df661012d9a8c0a36b6fbc69c8464365af3b4400e9decc56ddbd11d"
READBACK_TERMINAL_SHA = "aa3db38459da10e6be6a86d7619988db11f203d8dd77f5652d992a4645f9428b"
INFERENCE_RESULT_SHA = "2d3edfa85b8c7b73027e06c0ef92e3365d26e3f0f32ed114fec38661a1eebaee"
REQUEST_STATUS = "PHOENIX_UMBRA12_WHOLE_REQUEST_COMPLETE"
INFERENCE_STATUS = "PHOENIX_UMBRA12_WHOLE_INFERENCE_COMPLETE"
READBACK_STATUS = "PHOENIX_UMBRA12_WHOLE_GEOMETRY_READBACK_COMPLETE"
STATUS = "PHOENIX_UMBRA12_WHOLE_SCORE_COMPLETE"
WHOLE_ARM = "phoenix_whole"
SECONDARY_CONTRASTS = ((WHOLE_ARM, "raw"), ("free_matched_count", WHOLE_ARM))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def load_primary(code):
    path = code / "UMBRA12_DUAL_REVIEW_SCORE_20261123/score.py"
    raw = path.read_bytes()
    require(digest(raw) == PRIMARY_SOURCE_SHA, "Expected the exact released primary scorer")
    primary = ModuleType("_umbra12_primary_score")
    primary.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), primary.__dict__)
    return primary


def observed(primary, run, phase, status, *, result_sha=None, terminal_sha=None):
    terminal_bytes = (run / ("observed_" + phase) / "TERMINAL.json").read_bytes()
    if terminal_sha is not None:
        require(digest(terminal_bytes) == terminal_sha, "Observed whole-mask terminal identity differs")
    terminal = json.loads(terminal_bytes)
    require(terminal["phase"] == phase and type(terminal["actual_exit"]) is int and terminal["actual_exit"] == 0
            and terminal["output_footer_verified"] is True, "Successful observed whole-mask stage required")
    result = terminal["result_sha256"]
    require(result_sha is None or result == result_sha, "Observed whole-mask result identity differs")
    receipt = json.loads(primary.pin(run / phase / "RESULT_SAFE.json", result))
    require(receipt["phase"] == phase and receipt["status"] == status and receipt["protocol_sha256"] == PROTOCOL_SHA
            and receipt["worker_sha256"] == terminal["worker_sha256"] and receipt["target_labels_opened"] is False,
            "Expected the bound target-blind Phoenix execution")
    return receipt, result, digest(terminal_bytes)


def read_whole_closure(primary, args):
    primary.pin(args.code / "UMBRA12_PHOENIX_WHOLE_20261123/verify_whole.py", READBACK_WORKER_SHA)
    receipt, receipt_sha, terminal_sha = observed(primary, args.whole_run, "edits", READBACK_STATUS,
                                                   result_sha=READBACK_RESULT_SHA,
                                                   terminal_sha=READBACK_TERMINAL_SHA)
    require(receipt["worker_sha256"] == READBACK_WORKER_SHA and receipt["images"] == 12
            and receipt["exact_original_compact_positions"] is receipt["exact_native_binary_pixel_counts"] is True
            and receipt["optimizer_updates"] == receipt["detector_forwards"] == 0
            and receipt["inference_result_sha256"] == INFERENCE_RESULT_SHA,
            "Independent whole-mask readback must verify exact positions and native binary pixels")
    requested, _, _ = observed(primary, args.whole_run, "request", REQUEST_STATUS,
                                result_sha=receipt["request_result_sha256"], terminal_sha=receipt["request_terminal_sha256"])
    inferred, _, _ = observed(primary, args.whole_run, "inference", INFERENCE_STATUS,
                               result_sha=receipt["inference_result_sha256"], terminal_sha=receipt["inference_terminal_sha256"])
    require(requested["lineage"] == inferred["lineage"] == receipt["capture_lineage"]
            and requested["request_sha256"] == inferred["request_sha256"]
            and requested["worker_sha256"] == inferred["request_worker_sha256"]
            and inferred["masks_sha256"] == receipt["whole_mask_ledger_sha256"]
            and inferred["request_result_sha256"] == receipt["request_result_sha256"]
            and inferred["request_terminal_sha256"] == receipt["request_terminal_sha256"]
            and inferred["counts"]["lvis"] == receipt["coverage_counts"],
            "Whole-mask return, request and independently verified branch disagree")
    return receipt, receipt_sha, terminal_sha


def require_shared_carrier(primary, blind, whole):
    expected = {
        "carrier_closure_sha256": primary.BLIND_RECEIPT_SHA,
        "carrier_protocol_sha256": primary.PROTOCOL_SHA,
        "anonymous_roster_sha256": primary.ROSTER_SHA,
        "native_intake_manifest_sha256": blind["intake_sha256"],
        "native_prediction_ledger_sha256": blind["prediction_ledger_sha256"],
        "native_candidate_audit_sha256": blind["candidate_audit_sha256"],
        "native_edit_ledger_sha256": blind["edit_ledger_sha256"],
        "checkpoint_sha256": blind["checkpoint_sha256"],
    }
    require(all(whole["capture_lineage"][key] == value for key, value in expected.items()),
            "Whole-mask control and original fixed-match score must share the exact native carrier")


def core_for_whole(primary, code):
    """Use a distinct module so the pinned primary matching and contrasts remain unmodified."""
    core = primary.source_module(code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py", primary.CORE_SHA)
    core.ARMS += (WHOLE_ARM,)
    core.CONTRASTS += SECONDARY_CONTRASTS
    return core


def overlay_phoenix(core, frames, records, ids, hw):
    """Join observed whole writes by original compact slot. A zero-area return remains a write."""
    require(len(frames) == len(records) == len(ids) and set(frames) == set(ids),
            "Whole-mask control requires the complete original window roster")
    augmented = {}
    counts = Counter(decoded_candidates=0, pass_through=0, decoder_batches=0, empty_outputs=0,
                     unchanged_outputs=0, adds=0, drops=0)
    for ordinal, (key, record) in enumerate(zip(ids, records, strict=True), 1):
        native, local_updates = frames[key]
        areas = [int(core.codec.area(mask)) for mask in native]
        positions = [p for p, area in enumerate(areas, 1) if area > 0]
        require(set(record) == {"ordinal", "image_id", "lvis"} and
                type(record["ordinal"]) is type(record["image_id"]) is int and
                record["ordinal"] == record["image_id"] == ordinal and isinstance(record["lvis"], list)
                and all(set(row) == {"position", "segmentation"} and type(row["position"]) is int for row in record["lvis"])
                and [row["position"] for row in record["lvis"]] == positions,
                "Expected a whole-mask return for each and only each original nonempty compact slot")
        updates = {position: dict(variants) for position, variants in local_updates.items()}
        counts.update(decoded_candidates=len(positions), pass_through=len(native)-len(positions),
                      decoder_batches=(len(positions)+1)//2)
        for position, row in zip(positions, record["lvis"], strict=True):
            after = core.checked_rle(row["segmentation"], *hw)
            after_area = int(core.codec.area(after))
            common = int(core.codec.area(core.codec.merge([native[position-1], after], intersect=True)))
            adds, drops = after_area - common, areas[position-1] - common
            require(min(adds, drops) >= 0, "Invalid observed whole-mask pixel overlap")
            # Metrics need a positive edit indicator. For this branch it is the XOR size.
            updates.setdefault(position, {})[WHOLE_ARM] = (adds+drops, after)
            counts.update(adds=adds, drops=drops, empty_outputs=int(after_area == 0), unchanged_outputs=int(adds+drops == 0))
        augmented[key] = (native, updates)
    counts["changed_outputs"] = counts["decoded_candidates"] - counts["unchanged_outputs"]
    counts["all_unchanged_outputs"] = counts["pass_through"] + counts["unchanged_outputs"]
    counts["signed_area_delta"] = counts["adds"] - counts["drops"]
    counts["candidate_refinement_iterations"] = counts["decoded_candidates"] * 5
    return augmented, dict(counts)


def execute(args):
    start = perf_counter()
    require(args.protocol.name == PROTOCOL_NAME and not args.output.exists(),
            "Use the published Phoenix protocol and a fresh destination")
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2"
            and sys.version_info[:3] == (3, 10, 13)
            and {p: version(p) for p in ("numpy", "pycocotools", "scipy")} ==
            {"numpy": "1.26.4", "pycocotools": "2.0.11", "scipy": "1.15.3"},
            "Use the verified primary geometry, agreement and bootstrap runtime")
    primary = load_primary(args.code)
    primary.pin(args.protocol, PROTOCOL_SHA)
    refs, excluded, agreement, admission = primary.read_frozen_reviews(args)
    closed, closed_sha, closed_terminal_sha = read_whole_closure(primary, args)
    local_core = primary.source_module(args.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py", primary.CORE_SHA)
    frames, blind = primary.read_blind_frames(local_core, args)
    require_shared_carrier(primary, blind, closed)
    raw = primary.pin(args.whole_run / "inference" / "WHOLE_MASKS_PRIVATE.jsonl", closed["whole_mask_ledger_sha256"])
    rows = [json.loads(line) for line in raw.splitlines()]
    core = core_for_whole(primary, args.code)
    augmented, counts = overlay_phoenix(core, frames, rows, primary.IDS, primary.HW)
    require(counts == closed["coverage_counts"], "Refined and original measured native masks differ from readback")
    metrics = primary.source_module(primary.HERE / "metrics.py", primary.METRICS_SHA)
    references, decision = primary.evaluate_pair(metrics, core, refs, augmented, primary.HW)
    for report in references.values():
        coverage = report["labeled_hulls_and_coverage"]
        report["whole_refinement_exposure"] = dict(
            originally_matched_hulls_prompted=coverage["native_matched"],
            originally_matched_source_groups_prompted=coverage["originally_matched_source_groups"],
            originally_matched_hulls_with_changed_mask=report["edited_among_original_matches_by_arm"][WHOLE_ARM],
        )
    return dict(status=STATUS, phase="score", worker_sha256=digest(Path(__file__).read_bytes()),
                protocol_sha256=PROTOCOL_SHA, primary_source_sha256=PRIMARY_SOURCE_SHA,
                metrics_source_sha256=primary.METRICS_SHA, blind_receipt_sha256=primary.BLIND_RECEIPT_SHA,
                anonymous_roster_sha256=primary.ROSTER_SHA, prediction_ledger_sha256=blind["prediction_ledger_sha256"],
                edit_ledger_sha256=blind["edit_ledger_sha256"], whole_readback_worker_sha256=READBACK_WORKER_SHA,
                whole_readback_result_sha256=closed_sha, whole_readback_terminal_sha256=closed_terminal_sha,
                whole_mask_ledger_sha256=closed["whole_mask_ledger_sha256"],
                whole_request_result_sha256=closed["request_result_sha256"],
                whole_inference_result_sha256=closed["inference_result_sha256"],
                whole_verified_candidate_counts=counts, admitted_images=len(frames),
                jointly_completed_windows=len(refs["A"]), detector_forwards=0, optimizer_updates=0,
                annotations_opened=True, whole_scene_ap_reported=False, agreement=agreement["agreement"],
                status_pairs=agreement["status_pairs"], excluded_annotation_counts_in_joint_windows=excluded,
                reviewer_annotation_census={role: agreement[role.lower()+"_review"] for role in ("A", "B")},
                references=references, joint_primary_decision=decision,
                secondary_estimands=[left + " minus " + right for left, right in SECONDARY_CONTRASTS],
                bootstrap_draws=4096, bootstrap_seed=5217, seconds=perf_counter()-start, **admission)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "protocol", "review-run", "blind-run", "blind-receipt", "whole-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode, "Use python -I -B for reproducible scoring")
    result = execute(args)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.mkdir()
    with (args.output / "RESULT_SAFE.json").open("xb") as stream:
        require(stream.write(raw) == len(raw), "Incomplete secondary score receipt")
    print(json.dumps(dict(status=STATUS, result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()
