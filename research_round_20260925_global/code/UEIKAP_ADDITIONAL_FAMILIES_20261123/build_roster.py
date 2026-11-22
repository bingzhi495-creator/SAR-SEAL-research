"""Select the remaining strictly mapped UEIKAP PNG members from frozen metadata."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import ceil, floor, isfinite
from pathlib import Path, PurePosixPath
import re


RECORD = "https://zenodo.org/records/18391027"
ARCHIVE = "https://zenodo.org/api/records/18391027/files/UEIKAP_SAR.zip/content"
DIRECTORY_SHA = "e7a969a39b4f6723d9293acdc9297e917d2830fcba54eeac3dd22b26b4f56610"
PREFLIGHT = {
    "all_labels_audit.json": "0d7aeb869f0bccda238992b61d1b57212e573d8479da53dc123520d510e02125",
    "full_scene_mapping.json": "51285811a9098711cc8b9e7dfa3460ab71027bc7b70f7d7453b048e3e7545261",
    "FROZEN_HUMAN_QA50.json": "405f149aaa30808ecf2a495cb4dfaf0e1ea57b3aba860c5280dcdc9e5beafb1a",
    "PROVISIONAL_EXTERNAL_RESERVATION.json": "1f3b4d3bf0ff8f0ed1767cfdd796b541e60f2a4c7d76b03c316f0b1e81dbded1",
    "selection_locked.json": "bce9f651ab2b04c0e41f08b96d4f06680a49ee3e6fef6b34406b86fb90b11d7f",
}
RECOVERY = {
    "RECOVERED_FOOTPRINTS_PRIVATE.json": "fb58d7a0c4b7ac1f81751c2c8955c61cf8ce5bb0e6e2bccd738cec9ee1def732",
    "RECOVERY_AGGREGATE_SAFE.json": "c2195a679688590284c1f269b843dd88d7986a29607502bc0d5225aa47cc9b01",
    "CATALOGUE_READBACK_SAFE.json": "7a72d54dc5b67031a02474089442cde7f3b57c1a20ba28664971b4978648793d",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_pinned(directory, pins):
    result = {}
    for name, expected in pins.items():
        raw = (directory / name).read_bytes()
        require(sha256(raw).hexdigest() == expected, "Unexpected metadata digest for " + name)
        result[name] = json.loads(raw)
    return result


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def acquisition_map(mapping, recovery):
    assignment = {r["name"]: r["acquisition_groups"][0] for r in mapping["per_image"]
                  if len(r["acquisition_groups"]) == 1}
    unknown = {r["name"] for r in mapping["unresolved_images"]}
    missing = {r["name"] for r in mapping["tiff_headers_unread_due_to_range_errors"]}
    receipt = recovery["RECOVERY_AGGREGATE_SAFE.json"]
    readback = recovery["CATALOGUE_READBACK_SAFE.json"]
    recovered = recovery["RECOVERED_FOOTPRINTS_PRIVATE.json"]
    require(len(assignment) == 493 and len(unknown) == 10 and len(missing) == 12
            and not (set(assignment) & unknown or set(assignment) & missing or unknown & missing),
            "Original mapping census differs")
    require(receipt["original_snapshot_proof_sha256"] == PREFLIGHT["full_scene_mapping.json"]
            and receipt["archive_directory_sha256"] == DIRECTORY_SHA
            and readback["safe_recovery_sha256"] == RECOVERY["RECOVERY_AGGREGATE_SAFE.json"]
            and readback["group_assignment_mismatches"] == 0 and readback["full_recovered_headers_checked"] == 12
            and not recovered["failed"] and {r["header"]["name"] for r in recovered["rows"]} == missing,
            "Recovered assignments lack the original catalogue readback")
    for row in recovered["rows"]:
        name, family = row["header"]["name"], row["unique"]
        require(name not in assignment and row["groups"] == [family], "Duplicate or ambiguous recovery")
        assignment[name] = family
    require(len(assignment) == 505 and len(set(assignment.values())) == 32, "Merged assignment census differs")
    return assignment, unknown


def inventory(audit):
    require(audit["zip_central_sha256"] == DIRECTORY_SHA and audit["all_tif_ids_paired_to_txt"] is True
            and audit["all_txt_paired_to_png_by_filename"] is True, "Missing original three-way member pairing")
    rows = {}
    for row in audit["all_files"]:
        label = PurePosixPath(row["label"])
        require(len(label.parts) == 4 and label.parts[:2] == ("UEIKAP_SAR", "labels")
                and label.parts[2] == row["split"] and row["split"] in {"train", "val", "test"}
                and label.suffix == ".txt" and row["tif_present"] and row["png_present"], "Invalid original label member")
        tiff = str(PurePosixPath("UEIKAP_SAR", "images_tifs", row["split"], label.with_suffix(".tif").name))
        png = str(PurePosixPath("UEIKAP_SAR", "images", row["split"], label.with_suffix(".png").name))
        require(tiff not in rows and re.fullmatch(r"[0-9a-f]{8}", row["tif_zip_crc32"]), "Invalid TIFF member identity")
        rows[tiff] = dict(tiff_zip_name=tiff, tiff_zip_crc32=row["tif_zip_crc32"], image_zip_name=png,
                          label_zip_name=str(label), official_split=row["split"])
    require(len(rows) == 515 and len({r["image_zip_name"] for r in rows.values()}) == 515, "Published census differs")
    return rows


def build(preflight_dir, recovery_dir):
    pre = load_pinned(preflight_dir, PREFLIGHT)
    recovery = load_pinned(recovery_dir, RECOVERY)
    mapping = pre["full_scene_mapping.json"]
    groups, unresolved = acquisition_map(mapping, recovery)
    audit = pre["all_labels_audit.json"]
    rows = inventory(audit)
    require(rows.keys() == groups.keys() | unresolved, "Incomplete inventory or acquisition mapping")

    held = pre["PROVISIONAL_EXTERNAL_RESERVATION.json"]
    families = set(held["held_groups"])
    held_names = {n for n,g in groups.items() if g in families}
    qa = pre["FROZEN_HUMAN_QA50.json"]
    qa_names = {r["tiff_zip_path"] for r in qa["items"]}
    preview = {r["tiff"]["name"] for r in pre["selection_locked.json"]["selected"]}
    require(len(held_names) == 108 and len(families) == 18
            and held_names == {r["tiff_zip_name"] for r in held["items"]}
            and len(qa_names) == 50 and len(preview) == 5 and not (qa_names & preview)
            and (qa_names | preview) <= (groups.keys() - held_names), "Historical populations disagree")
    pre_set = set(groups) - held_names
    selected = sorted(pre_set - qa_names - preview)
    require(len(pre_set) == 397 and len(selected) == 342, "Unexpected new population")
    items = [dict(image_id=i, acquisition_group=groups[n], **rows[n]) for i,n in enumerate(selected, 1)]
    roster = dict(source_record=RECORD, archive_url=ARCHIVE, archive_bytes=1571288093,
                  central_directory_sha256=DIRECTORY_SHA, expected_ids=list(range(1, len(items)+1)), items=items)

    # The old class-count audit is used only after every member has been selected and ordered.
    counts = {r["label"]: r["rows_by_class"].get("0", 0) for r in audit["all_files"]}
    def hull_count(names):
        return sum(counts[rows[n]["label_zip_name"]] for n in names)

    per_family = defaultdict(Counter)
    splits = Counter()
    for item in items:
        family = per_family[item["acquisition_group"]]
        family.update(images=1, released_class0_polygons=counts[item["label_zip_name"]])
        splits[item["official_split"]] += 1
    require(len(per_family) == 13 and splits == {"train":217, "val":60, "test":65}
            and [hull_count(s) for s in (rows, held_names, unresolved, pre_set, qa_names, preview, selected)]
            == [760, 139, 12, 609, 75, 6, 528], "Published label-count or family census differs")
    bounds = {r["name"]: r["bounds_lonlat"] for r in mapping["per_image"]}
    for row in recovery["RECOVERED_FOOTPRINTS_PRIVATE.json"]["rows"]:
        header = row["header"]
        bounds[header["name"]] = header["bounds_lonlat"]
    selected_bounds = [bounds[n] for n in selected]
    require(all(len(b) == 4 and all(isfinite(x) for x in b) and -180 <= b[0] < b[2] <= 180
                and -90 <= b[1] < b[3] <= 90 for b in selected_bounds), "Invalid catalogue footprint")
    outer = [floor(10*min(b[i] for b in selected_bounds))/10 for i in (0, 1)]
    outer += [ceil(10*max(b[i] for b in selected_bounds))/10 for i in (2, 3)]
    summary = dict(status="UEIKAP_ADDITIONAL_FAMILIES_ROSTER_COMPLETE", source_record=RECORD,
                   archive_url=ARCHIVE, archive_bytes=1571288093, central_directory_sha256=DIRECTORY_SHA,
                   selection_rule="All strict satellite/orbit/datatake assignments outside the 18 previously evaluated families, excluding the five preview and 50 QA chips. Sort by original TIFF member name.",
                   source_metadata_sha256={**PREFLIGHT, **RECOVERY}, images=342, released_class0_polygons=528,
                   families=len(per_family), per_family=dict(per_family), publisher_split=dict(splits),
                   tiff_footprints=dict(images=len(selected_bounds), crs="EPSG:4326",
                                        outward_0p1_degree_bounds_west_south_east_north=outer),
                   prior_evaluated_images=len(held_names), prior_evaluated_families=len(families),
                   unresolved_excluded=len(unresolved), qa_excluded=len(qa_names), preview_excluded=len(preview),
                   native_image_bytes_read=0, polygon_files_opened=0, model_forwards=0)
    return roster, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--recovery", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Supply a fresh output directory")
    roster, summary = build(args.preflight, args.recovery)
    roster_raw = encode(roster)
    summary.update(roster_sha256=sha256(roster_raw).hexdigest(), worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    result_raw = encode(summary)
    args.output.mkdir()
    with (args.output / "ROSTER_PRIVATE.json").open("xb") as output:
        output.write(roster_raw)
    with (args.output / "RESULT_SAFE.json").open("xb") as output:
        output.write(result_raw)
    print(json.dumps({"status": summary["status"], "result_sha256": sha256(result_raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
