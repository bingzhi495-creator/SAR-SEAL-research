"""Evaluate both sealed independent Umbra polygon references against the prior blind mask ledger."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import shapely


HERE = Path(__file__).resolve().parent
PROTOCOL_NAME = "UMBRA12_FROZEN_CARRIER_PRELOCK_20261123.md"
PROTOCOL_SHA = "ff70161af804adc9f52c044771b504884ea80b7b2242ffc38fab51b55f2839ad"
ROSTER_SHA = "e626c38ea53727a2d235a80e675735898bbce5defce698f04fb602d674a87a56"
VALIDATOR_SHA = "e49bc030433ef833db6574fa75426774d65f4abfd2d7f7c5c8eedf848e34d9db"
CORE_SHA = "d541d6dffc9a7d49317dea34f53153c4121ba4d7523e5c904e5f7024fa895f20"
BLIND_RECEIPT_SHA = "29f6f58a12cf72cb0da444c24be9939d0dacbab56d6a664f289a1a5f5b88a665"
ADMISSION_WORKER_SHA = "fd46d28313950c94804ec61f054f90101fcec2acb9692f449b4aa3505778ddce"
METRICS_SHA = "ad62bcb4fa6b5bfbcaacb9f6ccc989b2588173e07b18b60bcc3aca46a04c9a05"
ADMITTED = "UMBRA12_DUAL_REVIEW_ADMISSION_COMPLETE"
STATUS = "UMBRA12_DUAL_REVIEW_SCORE_COMPLETE"
HW = (1024, 1024)
IDS = tuple(f"UMB-{i:03d}" for i in range(1, 13))
PHASES = ("capture", "verify-capture", "prepare-edits", "verify-edits")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(payload):
    return sha256(payload).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, f"Frozen input differs for {path.name}")
    return raw


def source_module(path, expected):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), module.__dict__)
    return module


def check_annotation_agreement(validator, roster, parsed, freeze):
    report = validator.agreement_report(roster, parsed["A"], parsed["B"])
    joint_ids = tuple(
        key for key in IDS if all(parsed[role][key][0] in validator.COMPLETE for role in ("A", "B"))
    )
    require(tuple(freeze["jointly_completed_ids"]) == joint_ids and
            all(freeze[key] == report[key] for key in ("agreement", "status_pairs", "windows")),
            "Reference agreement or jointly completed roster differs from admission")
    refs = {
        role: {
            key: [polygon for _, (tag, polygon) in sorted(parsed[role][key][1].items()) if tag == "ship_hull"]
            for key in joint_ids
        }
        for role in ("A", "B")
    }
    excluded = {
        role: {tag: sum(label == tag for key in joint_ids for label, _ in parsed[role][key][1].values())
               for tag in ("uncertain", "truncated")}
        for role in ("A", "B")
    }
    return refs, excluded, report


def read_frozen_reviews(args):
    run = args.review_run
    terminal_bytes = (run / "observed_request" / "TERMINAL.json").read_bytes()
    terminal = json.loads(terminal_bytes)
    require(terminal["phase"] == "request" and type(terminal["actual_exit"]) is int and
            terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True,
            "Successful observed independent review admission required")
    root = run / "request"
    receipt = json.loads(pin(root / "RESULT_SAFE.json", terminal["result_sha256"]))
    pin(HERE / "admit_reviews.py", ADMISSION_WORKER_SHA)
    require(receipt["status"] == ADMITTED and receipt["phase"] == "request" and
            receipt["worker_sha256"] == terminal["worker_sha256"] == ADMISSION_WORKER_SHA and
            receipt["protocol_sha256"] == PROTOCOL_SHA and receipt["roster_sha256"] == ROSTER_SHA and
            receipt["validator_sha256"] == VALIDATOR_SHA and receipt["predictions_opened"] is False,
            "Review identities and protocol must belong to the observed label-only admission")
    freeze = json.loads(pin(root / "ANNOTATION_ADMISSION_PRIVATE.json", receipt["admission_sha256"]))
    role_hashes = receipt["review_sha256"]
    require(set(role_hashes) == {"A", "B"} and role_hashes["A"] != role_hashes["B"]
            and freeze["schema"] == "umbra12-dual-review-admission-v1"
            and freeze["review_sha256"] == role_hashes and freeze["roster_sha256"] == ROSTER_SHA
            and freeze["protocol_sha256"] == PROTOCOL_SHA and freeze["predictions_opened"] is False,
            "Two distinct role-bound original reviews are required")
    validator = source_module(args.code / "UMBRA_PROSPECTIVE_GEC_ANNOTATION_20261122/validate_reviews.py", VALIDATOR_SHA)
    canon = root / "ROSTER_PINNED.json"
    file_bytes = {canon: pin(canon, ROSTER_SHA)}
    for role in ("A", "B"):
        file_path = root / f"REVIEWER_{role}_ORIGINAL.json"
        file_bytes[file_path] = pin(file_path, role_hashes[role])

    def read_exact(path):
        return json.loads(file_bytes[path].decode("utf-8-sig"), object_pairs_hook=validator.unique_object,
                          parse_constant=validator.reject_constant)

    # The released validator consumes the exact admitted bytes without reopening the source filenames.
    validator.read_json = read_exact
    roster = validator.read_roster(canon)
    require(tuple(roster) == IDS and set(roster.values()) == {HW}, "The original 12 native windows are required")
    parsed = {}
    for role in ("A", "B"):
        actual_role, entries = validator.read_review(root / f"REVIEWER_{role}_ORIGINAL.json", roster)
        require(actual_role == role, "The reviewer files cannot be interchanged")
        parsed[role] = entries
    refs, excluded, report = check_annotation_agreement(validator, roster, parsed, freeze)
    require(all(receipt[key] == report[key] for key in ("a_review", "b_review", "status_pairs", "agreement")),
            "Admission aggregates disagree with the same original review bytes")
    lineage = dict(admission_terminal_sha256=digest(terminal_bytes), admission_result_sha256=terminal["result_sha256"],
                   admission_sha256=receipt["admission_sha256"], admitted_worker_sha256=ADMISSION_WORKER_SHA,
                   review_sha256=role_hashes)
    return refs, excluded, report, lineage


def read_blind_frames(core, args):
    receipt = json.loads(pin(args.blind_receipt, BLIND_RECEIPT_SHA))
    require(receipt["status"] == "FROZEN_BLIND_CANDIDATES_AND_EDIT_LEDGER_VERIFIED"
            and receipt["protocol_sha256"] == PROTOCOL_SHA and receipt["anonymous_roster_sha256"] == ROSTER_SHA
            and receipt["images"] == 12 and receipt["detector_forwards"] == 12 and receipt["optimizer_updates"] == 0
            and receipt["target_labels_opened"] is receipt["scoring_executed"] is False
            and receipt["exact_native_parity"] is receipt["exact_support_and_xor"]
            is receipt["exact_independent_replay"] is True,
            "The previously observed closed blind capture and edit replay are required")
    results = {}
    for phase in PHASES:
        item = receipt["observed"][phase]
        terminal = json.loads(pin(args.blind_run / f"observed_{phase}" / "TERMINAL.json", item["terminal_sha256"]))
        results[phase] = json.loads(pin(args.blind_run / phase / "RESULT_SAFE.json", item["result_sha256"]))
        require(terminal["phase"] == phase and terminal["actual_exit"] == item["actual_exit"] == 0
                and terminal["output_footer_verified"] is item["output_footer_verified"] is True
                and terminal["worker_sha256"] == item["worker_sha256"] == results[phase]["worker_sha256"]
                and terminal["result_sha256"] == item["result_sha256"],
                "Blind process exit, identity or exact receipt differs")
    require(results["verify-capture"]["capture_sha256"] == receipt["observed"]["capture"]["result_sha256"]
            and results["verify-edits"]["prepared_result_sha256"] == receipt["observed"]["prepare-edits"]["result_sha256"]
            and results["verify-edits"]["capture_readback_result_sha256"] == receipt["observed"]["verify-capture"]["result_sha256"]
            and results["verify-edits"]["edits_sha256"] == receipt["edit_ledger_sha256"],
            "Capture, readback and edit receipt chain differs")
    files = (("capture/PREDICTIONS_PRIVATE.jsonl", "prediction_ledger_sha256"),
             ("capture/CANDIDATE_AUDIT_PRIVATE.jsonl", "candidate_audit_sha256"),
             ("prepare-edits/EDITS_PRIVATE.jsonl", "edit_ledger_sha256"))
    streams = [
        tuple(json.loads(line) for line in pin(args.blind_run / name, receipt[key]).splitlines())
        for name, key in files
    ]
    frames, decoded_ids = build_frames(core, streams)
    require(len(decoded_ids) == 12 and sum(len(masks) for masks, _ in frames.values()) ==
            results["verify-capture"]["counts"]["mapped"], "Frozen detector population differs")
    return frames, receipt


def build_frames(core, streams):
    require(all(len(stream) == 12 for stream in streams), "Every blind stream must contain all 12 original windows")
    frames = {}
    decoded_ids = set()
    for ordinal, (prediction, audit, edit) in enumerate(zip(*streams, strict=True), 1):
        key = IDS[ordinal - 1]
        rows = audit["candidates"]
        ranks = [item["native_rank"] for item in rows]
        require(audit["acquisition_group"] == key and audit["source_image_name"] == key + ".png"
                and audit["original_hw"] == list(HW) and audit["sensor"] == "Umbra"
                and edit["image_id"] == audit["image_id"] == prediction["image_id"] == ordinal
                and prediction["raw_query_count"] == 164 and prediction["native_count"] == 100
                and prediction["unmapped_slot1_count"] == 100 - len(rows)
                and prediction["native_ranks"] == audit["native_ranks"] == ranks
                and all(type(k) is int and 1 <= k <= 100 for k in ranks) and ranks == sorted(set(ranks))
                and all(item["position"] == pos for pos, item in enumerate(rows, 1)),
                "Captured ship-only positions, ranks, image identity or original grid differ")
        compact = dict(audit, candidates=[dict(item, native_rank=item["position"]) for item in rows])
        masks = core.verify_native(prediction, compact, ordinal, *HW)
        updates = core.verify_edits(edit["changes"], masks, *HW)
        frames[key] = (masks, updates)
        decoded_ids.add(audit["rgb_sha256"])
    return frames, decoded_ids


def evaluate_pair(metrics, core, refs, frames, hw):
    reference_results = {role: metrics.score_reference(core, refs[role], frames, hw) for role in ("A", "B")}
    return reference_results, metrics.combine(reference_results["A"], reference_results["B"])


def execute(args):
    start = perf_counter()
    require(args.protocol.name == PROTOCOL_NAME and not args.output.exists(),
            "Use the published protocol name and a new score directory")
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
            "Use the verified geometry runtime")
    require(sys.version_info[:3] == (3, 10, 13) and
            {p: version(p) for p in ("numpy", "pycocotools", "scipy")} ==
            {"numpy": "1.26.4", "pycocotools": "2.0.11", "scipy": "1.15.3"},
            "Use the verified matching, agreement and bootstrap runtime")
    pin(args.protocol, PROTOCOL_SHA)
    refs, excluded, agreement, admission = read_frozen_reviews(args)
    core = source_module(args.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py", CORE_SHA)
    metrics = source_module(HERE / "metrics.py", METRICS_SHA)
    frames, blind = read_blind_frames(core, args)
    reference_results, decision = evaluate_pair(metrics, core, refs, frames, HW)
    return dict(status=STATUS, phase="score", worker_sha256=digest(Path(__file__).read_bytes()),
                metrics_source_sha256=METRICS_SHA, protocol_sha256=PROTOCOL_SHA,
                blind_receipt_sha256=BLIND_RECEIPT_SHA, anonymous_roster_sha256=ROSTER_SHA,
                prediction_ledger_sha256=blind["prediction_ledger_sha256"],
                edit_ledger_sha256=blind["edit_ledger_sha256"], candidate_audit_sha256=blind["candidate_audit_sha256"],
                admitted_images=len(frames), distinct_decoded_images=len(frames),
                ship_candidates_within_original_top100=sum(len(masks) for masks, _ in frames.values()),
                images_without_ship_candidate=sum(not masks for masks, _ in frames.values()),
                jointly_completed_windows=len(refs["A"]), detector_forwards=0, optimizer_updates=0,
                annotations_opened=True, whole_scene_ap_reported=False,
                agreement=agreement["agreement"], status_pairs=agreement["status_pairs"],
                reviewer_annotation_census={role: agreement[role.lower() + "_review"] for role in ("A", "B")},
                excluded_annotation_counts_in_joint_windows=excluded, references=reference_results, joint_decision=decision,
                bootstrap_draws=4096, bootstrap_seed=5217, seconds=perf_counter()-start, **admission)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "protocol", "review-run", "blind-run", "blind-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode, "Use python -I -B for reproducible scoring")
    result = execute(args)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.mkdir()
    with (args.output / "RESULT_SAFE.json").open("xb") as stream:
        require(stream.write(raw) == len(raw), "Incomplete score receipt")
    print(json.dumps(dict(status=STATUS, result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()
