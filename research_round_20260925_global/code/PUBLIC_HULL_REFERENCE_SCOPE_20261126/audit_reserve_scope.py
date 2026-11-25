"""Audit a closed reserve chronology and joint metadata-screen coverage."""

from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path


PRELOCK_COMMIT = "8ea9023ffb48abbd71e683d223a44087f3021d5b"
COMMIT_UTC = "2026-11-22T18:33:19+00:00"
PROTOCOL = "3cc00d35d65318c3ae15c041c0f0b71f0c0dc743e0286c70955b9c3740009727"
CENSUS = "9454f1de69dc0d9450c30005647333f048dcea184b0561acf5dac518759e7bc6"
NATIVE_CHECK = "f54ce69df68bfbd74f857d05df63fe6670d5086174c60541fc6e660d32493845"
PHASES = {
    "capture": "de81c108a909a85b50a3fb05b842dc1f06c225b13e4f29ccab6ddaaa0341bddc",
    "verify-capture": "5a452510dd8e9594cba8fdf4185458c809fe43e5a213471482aac8883549e8d8",
    "prepare-edits": "6d0adf4a5d9043a31c05c12f26f193bf6a61213cea672ceaf5c0367d00108d35",
    "verify-edits": "0f0aa047630c87653754152b8a47b8c570c9bff04928f06c6fd4b1031bcdcf97",
    "score": "b7f46a2332866f8898ed27cd027b21019c10d9d056f3010b256a67bd8887feda",
}


def pin(path, digest):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError(f"Unexpected previously admitted bytes in {path.name}")
    return raw


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(run, reported_phase, output, observed, expected):
    terminal_bytes = (run / observed / "TERMINAL.json").read_bytes()
    terminal = json.loads(terminal_bytes)
    result = json.loads(pin(run / output / "RESULT_SAFE.json", expected))
    require(terminal["phase"] == reported_phase and terminal["actual_exit"] == 0
            and terminal["output_footer_verified"] is True and terminal["result_sha256"] == expected
            and terminal["worker_sha256"] == result["worker_sha256"], "Missing accepted natural closure")
    require(datetime.fromisoformat(terminal["started_utc"]) < datetime.fromisoformat(terminal["ended_utc"]),
            "Invalid actual execution interval")
    proof = {key: terminal[key] for key in ("started_utc", "ended_utc", "result_sha256")}
    proof["terminal_sha256"] = sha256(terminal_bytes).hexdigest()
    return proof, result


def audit(args):
    pin(args.protocol, PROTOCOL)
    census = json.loads(pin(args.census, CENSUS))
    chronology, score = {}, None
    for phase, expected in PHASES.items():
        proof, result = close(args.reserve_root, phase, phase, "observed_" + phase, expected)
        require(result["protocol_sha256"] == PROTOCOL and result["prelock_commit"] == PRELOCK_COMMIT,
                "Mismatched registered reserve method")
        chronology[phase] = proof
        if phase == "score":
            score = result
    source_proof, source = close(args.source_root, "verify-capture", "source", "observed_source", NATIVE_CHECK)
    previous = datetime.fromisoformat(COMMIT_UTC)
    for record in chronology.values():
        require(previous < datetime.fromisoformat(record["started_utc"]), "Out-of-order phase")
        previous = datetime.fromisoformat(record["ended_utc"])
    require(datetime.fromisoformat(source_proof["ended_utc"])
            < datetime.fromisoformat(chronology["prepare-edits"]["started_utc"]),
            "Source check must close before mask replay")
    require(score["census_sha256"] == CENSUS and score["source_exposure_qualified"] is True
            and source["passed"] is True and source["label_records_read"] == 0
            and source["reserve_rgb_matches_historical_test"] == source["reserve_rgb_matches_hrsid_fit"] == 0
            and source["reserve_parent_matches_historical_test"] == 0, "Original source qualification changed")

    date_overlap = set(census["parents_sharing_sensor_date_with_test"])
    center_overlap = set(census["parents_sharing_rounded_center_with_same_sensor_test"])
    parents = {item["paper_scene"]: item for item in census["per_parent"]}
    require(len(parents) == source["reserve_parents"] == 13 and sum(item["images"] for item in parents.values()) == 74
            and (date_overlap | center_overlap) <= parents.keys() and set(map(str, parents)) == set(score["by_parent"]),
            "Incorrect published parent census and fixed-match counts")
    keep = parents.keys() - date_overlap - center_overlap
    positive_matches = [score["by_parent"][str(k)]["counts"]["positive_fixed_edits"] for k in keep]
    coverage = dict(parents=len(keep), images=sum(parents[k]["images"] for k in keep),
                    positive_fixed_matches=sum(positive_matches),
                    positive_fixed_parents=sum(n > 0 for n in positive_matches))
    return dict(status="INSTANCE_RSDD_EXTERNAL_REFERENCE_SCOPE_AUDIT_COMPLETE",
                publication_witness=dict(commit=PRELOCK_COMMIT, commit_utc=COMMIT_UTC,
                                         protocol_sha256=PROTOCOL, protocol_git_blob="a0386837d785bbc18b4a81709128d2dfabdd309c"),
                originally_verified=True, closed_phases=chronology, closed_source_check=source_proof,
                exact_source_duplicate_images=0, exact_historical_duplicate_images=0,
                physical_acquisition_separation_verified=source["hrsid_physical_cross_scene_disjointness_verified"],
                original_reserve=dict(images=74, parents=len(parents), source_training_images=source["source_images"],
                                      source_head_images=source["source_fit109"], historical_images=source["historical_test_images"]),
                metadata_census_sha256=CENSUS,
                overlaps=dict(same_sensor_date=len(date_overlap), same_sensor_rounded_center=len(center_overlap),
                              either=len(date_overlap | center_overlap), both=len(date_overlap & center_overlap)),
                joint_screen_coverage=coverage, original_min_positive_matches=20, original_min_positive_parents=8,
                joint_screen_pass=coverage["positive_fixed_matches"] >= 20 and coverage["positive_fixed_parents"] >= 8,
                new_mask_predictions=False, new_reference_geometry_opened=False, new_group_effect_computed=False)


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for arg in ("reserve-root", "source-root", "protocol", "census", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    args = parser.parse_args()
    result = audit(args)
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
