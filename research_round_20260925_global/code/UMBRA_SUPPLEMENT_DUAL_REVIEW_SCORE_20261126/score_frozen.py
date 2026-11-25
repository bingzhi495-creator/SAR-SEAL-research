"""Score already sealed supplemental Umbra reviews against independently verified masks."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from types import ModuleType

HERE = Path(__file__).resolve().parent
ADMISSION_WORKER_SHA = "ff454b3f7d83171156d236220f9a8f6fbb596f72552ff93076595dccd5531c18"
HW = (1024, 1024)
CORE_SHA = "d541d6dffc9a7d49317dea34f53153c4121ba4d7523e5c904e5f7024fa895f20"
METRICS_SHA = "ad62bcb4fa6b5bfbcaacb9f6ccc989b2588173e07b18b60bcc3aca46a04c9a05"
CHECKPOINT_SHA = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
POINT_HEAD_SHA = "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1"
CARRIER_PROTOCOL_SHA = "e4fa11e77feb7578139f00cf8bd0b54b7b101662de04a49dd915bcd08049407c"
SOURCE_METADATA_SHA = "007836bf407e1e6a04d75fb45044adceabaa7ae81ee0cbc4f7054fc0e832b5fc"
PHASES = ("capture", "verify-capture", "prepare-edits", "verify-edits")
STATUS = "UMBRA_SUPPLEMENT_DUAL_REVIEW_SCORE_COMPLETE"


def load_admission_worker():
    path = HERE / "admit_reviews.py"
    source = path.read_bytes()
    if sha256(source).hexdigest() != ADMISSION_WORKER_SHA:
        raise ValueError("Exact independent review admission worker is required")
    module = ModuleType("umbra_supplement_review_admission")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


admission = load_admission_worker()


def load_module(path, expected):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(admission.pin(path, expected), str(path), "exec"), module.__dict__)
    return module


def evaluators(code):
    core = load_module(code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py", CORE_SHA)
    metrics = load_module(code / "UMBRA12_DUAL_REVIEW_SCORE_20261123/metrics.py", METRICS_SHA)
    core.MIN_EXPOSED_HULLS = 30
    core.MIN_EXPOSED_GROUPS = 12
    return core, metrics


def evaluate(core, metrics, refs, frames, hw):
    results = {role: metrics.score_reference(core, refs[role], frames, hw) for role in admission.ROLES}
    return results, metrics.combine(results["A"], results["B"])


def closed_result(terminal_path, result_path, expected_terminal, phase, worker):
    if expected_terminal is None:
        terminal_bytes = terminal_path.read_bytes()
    else:
        terminal_bytes = admission.pin(terminal_path, expected_terminal)
    terminal = json.loads(terminal_bytes)
    admission.require(terminal["phase"] == phase and type(terminal["actual_exit"]) is int
            and terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True,
            f"Observed successful exit is required for {phase}")
    result = json.loads(admission.pin(result_path, terminal["result_sha256"]))
    admission.require(result["phase"] == phase and terminal["worker_sha256"] == result["worker_sha256"] == worker,
            f"Worker identity differs for {phase}")
    return terminal, result, admission.digest(terminal_bytes)


def read_reviews(args):
    # The label-only admission must have exited before any stock/edit ledger is accessed.
    run = args.review_run
    request = run / "request"
    worker = ADMISSION_WORKER_SHA
    _, result, terminal_sha = closed_result(run / "observed_request/TERMINAL.json",
                                           request / "RESULT_SAFE.json", None, "request", worker)
    admission.require(result["status"] == admission.STATUS and result["protocol_sha256"] == admission.PRELOCK_SHA256
            and result["predictions_opened"] is False, "Review-only admission differs from the locked cohort")
    ledger = json.loads(admission.pin(request / admission.LEDGER_FILE, result["admission_sha256"]))
    compare = ("protocol_sha256", "packet_terminal_sha256", "packet_result_sha256", "roster_sha256",
               "review_sha256", "agreement", "status_pairs", "a_review", "b_review", "scope_windows")
    admission.require(ledger["schema"] == "umbra-supplement-dual-review-admission-v1"
            and ledger["predictions_opened"] is False
            and all(ledger[key] == result[key] for key in compare), "Exact A/B label admission differs")
    raw_roster = admission.pin(request / admission.ROSTER_FILE, result["roster_sha256"])
    reviews = {role: admission.pin(request / f"REVIEWER_{role}_ORIGINAL.json", result["review_sha256"][role]) for role in admission.ROLES}
    validator, roster, parsed = admission.parse_originals(
        args.code / "UMBRA_PROSPECTIVE_GEC_ANNOTATION_20261122/validate_reviews.py", raw_roster, reviews)
    report = validator.agreement_report(roster, parsed["A"], parsed["B"])
    complete = [key for key in roster if all(parsed[role][key][0] in validator.COMPLETE for role in admission.ROLES)]
    admission.require(complete == ledger["jointly_completed_ids"] and len(roster) == result["scope_windows"]
            and all(report[key] == ledger[key] for key in ("agreement", "status_pairs", "windows",
                                                          "a_review", "b_review")),
            "The sealed review consensus statistics differ")
    refs = {
        role: {key: [polygon for _, (label, polygon) in sorted(parsed[role][key][1].items())
                     if label == "ship_hull"] for key in complete} for role in admission.ROLES
    }
    excluded = {role: {tag: sum(label == tag for key in complete for label, _ in parsed[role][key][1].values())
                       for tag in ("uncertain", "truncated")} for role in admission.ROLES}
    return refs, excluded, roster, report, result, terminal_sha


def build_frames(core, roster, streams):
    admission.require(all(len(stream) == len(roster) for stream in streams), "Every blind stream must have the entire admitted roster")
    frames = {}
    decoded = set()
    for key, (prediction, audit, edit) in zip(roster, zip(*streams, strict=True), strict=True):
        image_id = int(key[4:])
        rows = audit["candidates"]
        ranks = [row["native_rank"] for row in rows]
        admission.require(audit["acquisition_group"] == key and audit["source_image_name"] == key + ".png"
                and audit["original_hw"] == list(HW) and audit["sensor"] == "Umbra"
                and prediction["image_id"] == audit["image_id"] == edit["image_id"] == image_id
                and prediction["raw_query_count"] == 164 and prediction["native_count"] == 100
                and prediction["unmapped_slot1_count"] == 100 - len(rows)
                and ranks == audit["native_ranks"] == prediction["native_ranks"]
                and all(type(rank) is int and 1 <= rank <= 100 for rank in ranks)
                and ranks == sorted(set(ranks))
                and all(row["position"] == pos for pos, row in enumerate(rows, 1)),
                "Frozen candidate identity, grid, ranks or positions differ")
        compact = dict(audit, candidates=[dict(row, native_rank=row["position"]) for row in rows])
        masks = core.verify_native(prediction, compact, image_id, *HW)
        updates = core.verify_edits(edit["changes"], masks, *HW)
        frames[key] = masks, updates
        decoded.add(audit["rgb_sha256"])
    admission.require(len(decoded) == len(roster), "Decoded native windows are not unique")
    return frames


def read_blind(args, core, roster, admitted):
    run = args.blind_run
    carrier = json.loads(admission.pin(run / "RESULT_SAFE.json", args.blind_receipt_sha256))
    admission.require(carrier["status"] == "UMBRA_SUPPLEMENT_FROZEN_CARRIER_AND_EDITS_VERIFIED"
            and carrier["protocol_sha256"] == CARRIER_PROTOCOL_SHA
            and carrier["target_labels_opened"] is False and carrier["optimizer_updates"] == 0
            and carrier["exact_native_parity"] is carrier["exact_support_and_xor"] is carrier["exact_independent_replay"] is True
            and carrier["model_forward_calls"] == len(roster)
            and carrier["manifest_sha256"] == SOURCE_METADATA_SHA
            and carrier["anonymous_roster_sha256"] == admitted["roster_sha256"]
            and carrier["reviewer_receipt_sha256"] == admitted["packet_result_sha256"]
            and carrier["reviewer_terminal_sha256"] == admitted["packet_terminal_sha256"]
            and carrier["checkpoint_sha256"] == CHECKPOINT_SHA and carrier["point_head_sha256"] == POINT_HEAD_SHA
            and carrier["images"] == len(roster) and set(carrier["phases"]) == set(PHASES),
            "Frozen carrier and exact independently sealed supplemental roster are required")
    results = {}
    for phase in PHASES:
        observed = carrier["phases"][phase]
        terminal, outcome, _ = closed_result(run / f"observed_{phase}/TERMINAL.json",
                                            run / phase / "RESULT_SAFE.json",
                                            observed["terminal_sha256"], phase, carrier["worker_sha256"])
        admission.require(terminal["result_sha256"] == observed["result_sha256"]
                and outcome["target_labels_opened"] is False and outcome["optimizer_updates"] == 0
                and outcome["admission_sha256"] == carrier["admission_sha256"]
                and outcome["protocol_sha256"] == carrier["protocol_sha256"],
                "All closed blind phases must belong to the same cohort and frozen carrier")
        results[phase] = outcome
    admission.require(results["capture"]["predictions_sha256"] == carrier["predictions_sha256"]
            and results["capture"]["audit_sha256"] == carrier["audit_sha256"]
            and results["prepare-edits"]["edits_sha256"] == results["verify-edits"]["edits_sha256"] == carrier["edits_sha256"]
            and results["verify-edits"]["exact_support_and_xor"] is True
            and results["verify-edits"]["exact_independent_replay"] is True,
            "The exact native capture and independently verified edit masks are required")
    streams = [
        tuple(json.loads(line) for line in admission.pin(run / name, carrier[key]).splitlines())
        for name, key in (("capture/PREDICTIONS_PRIVATE.jsonl", "predictions_sha256"),
                          ("capture/CANDIDATE_AUDIT_PRIVATE.jsonl", "audit_sha256"),
                          ("prepare-edits/EDITS_PRIVATE.jsonl", "edits_sha256"))
    ]
    frames = build_frames(core, roster, streams)
    admission.require(sum(len(masks) for masks, _ in frames.values()) == results["capture"]["counts"]["mapped"],
            "Frozen candidate denominator differs")
    return frames


def main():
    import shapely

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "protocol", "review-run", "blind-run", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--blind-receipt-sha256", required=True)
    args = parser.parse_args()
    admission.require(not args.output.exists(), "Use a fresh score directory")
    admission.require(sys.version_info[:3] == (3, 10, 13)
            and shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2"
            and {name: version(name) for name in ("numpy", "pycocotools", "scipy")}
            == {"numpy": "1.26.4", "pycocotools": "2.0.11", "scipy": "1.15.3"},
            "Use the verified geometry and bootstrap runtime")
    admission.pin(args.protocol, admission.PRELOCK_SHA256)
    refs, excluded, roster, agreement, admitted, terminal_sha = read_reviews(args)
    core, metrics = evaluators(args.code)
    frames = read_blind(args, core, roster, admitted)
    results, decision = evaluate(core, metrics, refs, frames, HW)
    receipt = dict(status=STATUS, phase="score", worker_sha256=admission.digest(Path(__file__).read_bytes()),
                   protocol_sha256=admission.PRELOCK_SHA256, annotation_admission_sha256=admitted["admission_sha256"],
                   annotation_terminal_sha256=terminal_sha, roster_sha256=admitted["roster_sha256"],
                   review_sha256=admitted["review_sha256"], blind_receipt_sha256=args.blind_receipt_sha256,
                   windows_admitted=len(roster), jointly_completed_windows=len(refs["A"]), agreement=agreement["agreement"],
                   status_pairs=agreement["status_pairs"], excluded_annotations=excluded,
                   references=results, joint_decision=decision,
                   original_u12_pooled=False, threshold_hulls=30, threshold_acquisition_groups=12,
                   bootstrap_draws=4096, bootstrap_seed=5217, detector_forwards=0, optimizer_updates=0,
                   whole_scene_ap_reported=False)
    raw = admission.encode(receipt)
    args.output.mkdir()
    admission.write_exact(args.output / "RESULT_SAFE.json", raw)
    print(json.dumps(dict(status=STATUS, result_sha256=admission.digest(raw))), flush=True)


if __name__ == "__main__":
    main()
