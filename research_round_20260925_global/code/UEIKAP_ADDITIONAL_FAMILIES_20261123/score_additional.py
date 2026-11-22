"""Score the independently admitted UEIKAP acquisition panel against released hulls."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType


PROTOCOL = "fbcc4d067351a7a7d2db226e6261a13c2f399eed59be414226e6ae29b6553bf2"
ROSTER = "0d78b4611e7db88a146f5bad489b6301e64e1d1eb499d536fd414e7b134c456d"
RF_SHA = "9ecbac508e1506b6cee2706a31f497eb4a46dd9bb62fbd56340e9e64ec981fdc"
IMAGES, HULLS, GROUPS = 342, 528, 13


def require(value, detail):
    if not value:
        raise ValueError(detail)


def read(path, expected):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == expected, "Digest differs for " + path.name)
    return data


def observed(path, expected, result_sha, phase):
    event = json.loads(read(path, expected))
    require(event["phase"] == phase and event["actual_exit"] == 0 and event["output_footer_verified"] is True
            and event["result_sha256"] == result_sha, "Unverified " + phase + " process")
    return event


def methods(code):
    path = code / "UEIKAP_RFDETR_CARRIER_20261122" / "score_rf.py"
    rf = ModuleType("rf_reference")
    rf.__file__ = str(path)
    exec(compile(read(path, RF_SHA), str(path), "exec"), rf.__dict__)
    return rf, rf.load_geometry(code / "UEIKAP_RESERVED_HULL_FROZEN_20261122" / "score.py")


def admitted_rows(a, core, rf):
    read(a.protocol, PROTOCOL)
    observed(a.intake_terminal, a.intake_terminal_sha, a.intake_result_sha, "intake")
    intake = core.json_pinned(a.intake / "RESULT_SAFE.json", a.intake_result_sha)
    require(intake["status"] == "UEIKAP_ADDITIONAL_AUTHOR_INTAKE_COMPLETE" and
            (intake["images"], intake["labeled_hulls"], intake["acquisition_groups"]) == (IMAGES, HULLS, GROUPS)
            and intake["cohort_roster_sha256"] == ROSTER and intake["protocol_sha256"] == PROTOCOL,
            "Image and label intake must have closed on the locked additional acquisitions")
    event = observed(a.readback_terminal, a.readback_terminal_sha, a.readback_sha, "verify-edits")
    edits = core.json_pinned(a.readback, a.readback_sha)
    require(edits["status"] == "UEIKAP_ADDITIONAL_RFDETR_EDITS_VERIFIED" and
            event["worker_sha256"] == edits["worker_sha256"] and edits["exact_replayed_policy_choices"] is True
            and edits["exact_support_xor_area"] is True and edits["target_labels_opened"] is False
            and edits["GT_opened"] is False and edits["optimizer_updates"] == edits["model_forwards"] == 0
            and edits["cohort_roster_sha256"] == ROSTER and edits["protocol_sha256"] == PROTOCOL
            and edits["manifest_sha256"] == intake["images_roster_sha256"]
            and edits["intake_terminal_sha256"] == a.intake_terminal_sha, "Original verified label-free edits")
    predictions = core.stream_pinned(a.capture / "PREDICTIONS_PRIVATE.jsonl", edits["predictions_sha256"])
    audits = core.stream_pinned(a.capture / "CANDIDATE_AUDIT_PRIVATE.jsonl", edits["candidate_audit_sha256"])
    modified = core.stream_pinned(a.prepared / "EDITS_PRIVATE.jsonl", edits["edits_sha256"])
    require(len(predictions) == len(audits) == len(modified) == IMAGES, "Incomplete capture or edit stream")
    population = []
    for i, (pred, audit, item) in enumerate(zip(predictions, audits, modified), 1):
        require(i == item["image_id"], "Edits are out of sequence")
        native = rf.native_masks(core, pred, audit, i)
        population.append((native, core.verify_edits(item["changes"], native, core.HEIGHT, core.WIDTH)))

    # Open the label roster and original member provenance only after every prediction and edit passes.
    ordered = core.json_pinned(a.roster, ROSTER)
    images = core.json_pinned(a.intake / "images_roster.json", intake["images_roster_sha256"])
    labels = core.json_pinned(a.intake / "labels_roster.json", intake["labels_roster_sha256"])
    provenance = core.json_pinned(a.intake / "member_provenance.json", intake["member_provenance_sha256"])
    ids = list(range(1, IMAGES + 1))
    require(all(item["expected_ids"] == ids and len(item["items"]) == IMAGES for item in (ordered, images, labels))
            and len(provenance) == 2 * IMAGES, "Exact admitted PNG/TXT provenance and roster")
    members = {(v["image_id"], v["zip_member"]): v for v in provenance}
    require(len(members) == 2 * IMAGES, "Duplicate author member provenance")
    for i, (old, img, label, audit) in enumerate(zip(ordered["items"], images["items"], labels["items"], audits), 1):
        require(old["image_id"] == img["image_id"] == label["image_id"] == i and img["original_hw"] == [512, 512]
                and img["acquisition_group"] == old["acquisition_group"] == label["acquisition_group"]
                and audit["image_sha256"] == img["image_ref"]["sha256"], "Frozen image or group linkage")
        for ref, member, local in ((img["image_ref"], old["image_zip_name"], f"images/{i:06d}.png"),
                                   (label["label_ref"], old["label_zip_name"], f"labels/{i:06d}.txt")):
            source = members[i, member]
            require(ref["path"] == local and source["bytes"] == ref["bytes"] and source["sha256"] == ref["sha256"]
                    and re.fullmatch("[0-9a-fA-F]{8}", source["zip_crc32"]), "Original member reference differs")
    require(len({row["acquisition_group"] for row in labels["items"]}) == GROUPS
            and sum(row["count_by_class"].get("0", 0) for row in labels["items"]) == HULLS,
            "Admitted polygon and acquisition census")
    return population, labels["items"], intake, edits


def geometry(core, shapely, root, population, labels):
    counts = Counter(dict.fromkeys(("labeled_hulls", "native_matched", "native_unmatched", "native_area_below_32",
                                    "native_area_at_least_1024", "eligible_area_sf_zero", "sf_positive",
                                    "wake_rows_excluded"), 0))
    groups, matched_groups = defaultdict(list), defaultdict(list)
    exposed, areas = set(), []
    edited = Counter(dict.fromkeys(core.ARMS, 0))
    transitions = {str(t): {arm: Counter(retained=0, lost=0, gained=0, both_below=0) for arm in core.ARMS}
                   for t in (0.75, 0.90)}
    errors = {arm: dict(false_positive_area=0.0, false_negative_area=0.0) for arm in core.ARMS}
    require(len(population) == len(labels), "Complete paired label population")
    for (masks, updates), label in zip(population, labels):
        group = label["acquisition_group"]
        truth, class_counts = core.parse_yolo(core.label_path(root, label["label_ref"]), core.HEIGHT, core.WIDTH, shapely)
        require(truth and class_counts == label["count_by_class"], "Original released polygon population differs")
        matched = core.match_native(masks, [raster for _, raster in truth])
        counts.update(labeled_hulls=len(truth), native_matched=len(matched), native_unmatched=len(truth)-len(matched),
                      wake_rows_excluded=class_counts.get("1", 0))
        for index, (polygon, _) in enumerate(truth):
            if index not in matched:
                groups[group].append([0.0] * len(core.ARMS))
                continue
            position = matched[index]
            native, changes = masks[position - 1], updates.get(position, {})
            area = int(core.codec.area(native))
            areas.append(area)
            if area < 32:
                counts["native_area_below_32"] += 1
            elif area >= 1024:
                counts["native_area_at_least_1024"] += 1
            elif changes.get("fixed", (0,))[0]:
                counts["sf_positive"] += 1
                exposed.add(group)
            else:
                counts["eligible_area_sf_zero"] += 1
            cache, values = {}, []
            for arm in core.ARMS:
                swaps, mask = changes.get(arm, (0, native))
                edited[arm] += int(swaps > 0)
                key = mask["counts"]
                if key not in cache:
                    cache[key] = core.geometric_iou(mask, polygon, shapely)
                iou = cache[key]
                values.append(iou)
                # Every verified edit preserves native area. J = I/(P+T-I).
                intersection = iou * (area + polygon.area) / (1 + iou)
                errors[arm]["false_positive_area"] += max(0., area - intersection)
                errors[arm]["false_negative_area"] += max(0., polygon.area - intersection)
            for threshold, tally in transitions.items():
                original = values[core.ARMS.index("raw")] >= float(threshold)
                for arm, value in zip(core.ARMS, values):
                    positive = value >= float(threshold)
                    event = ("retained" if positive else "lost") if original else ("gained" if positive else "both_below")
                    tally[arm][event] += 1
            groups[group].append(values)
            matched_groups[group].append(values)
    require(counts["labeled_hulls"] == HULLS == counts["native_matched"] + counts["native_unmatched"]
            and len(groups) == GROUPS and sum(counts[k] for k in ("native_area_below_32", "native_area_at_least_1024",
                "eligible_area_sf_zero", "sf_positive")) == counts["native_matched"], "Hull or exposure census differs")
    main = core.paired(groups)
    coverage = counts["sf_positive"] >= core.MIN_EXPOSED_HULLS and len(exposed) >= core.MIN_EXPOSED_GROUPS
    return dict(labeled_hulls_and_coverage=dict(counts, sf_positive_source_groups=len(exposed),
                                                originally_matched_source_groups=len(matched_groups)),
                primary_estimand=core.PRIMARY, primary_all_labeled_hulls=main,
                originally_matched_hulls_sensitivity=core.paired(matched_groups) if matched_groups else None,
                native_matched_mask_area_px_five_number_summary=core.area_summary(areas),
                originally_matched_hulls_continuous_error_area_sum_square_pixels=errors,
                complete_scene_false_positive_count_identifiable=False,
                continuous_iou_threshold_transitions_on_original_matches=transitions,
                edited_among_original_matches_by_arm=dict(edited), coverage_gate_met=coverage,
                frozen_geometry_advance=coverage and main["contrasts"][core.PRIMARY]["paired_95_interval_points"][0] > 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("code", "protocol", "roster", "intake", "intake-terminal", "capture", "prepared",
                "readback", "readback-terminal", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("intake-terminal-sha", "intake-result-sha", "readback-sha", "readback-terminal-sha"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--shapely-site", type=Path)
    a = p.parse_args()
    require(not a.output.exists(), "Use a fresh score destination")
    if a.shapely_site:
        sys.path.append(str(a.shapely_site.resolve(strict=True)))
    import shapely
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Original geometry runtime")
    started = perf_counter()
    rf, core = methods(a.code)
    population, labels, intake, edits = admitted_rows(a, core, rf)
    result = geometry(core, shapely, a.intake, population, labels)
    result.update(status="UEIKAP_ADDITIONAL_RFDETR_RELEASED_HULL_SCORE_COMPLETE", images=IMAGES, groups=GROUPS,
                  whole_scene_AP=False, previously_scored_acquisition_panel=False, geographic_holdout=False,
                  optimizer_updates=0, model_forwards=0, seconds=perf_counter()-started,
                  ship_candidates_within_joint_top100=sum(len(masks) for masks, _ in population),
                  images_without_ship_candidate=sum(not masks for masks, _ in population),
                  geometry_backend=dict(shapely=shapely.__version__, geos=shapely.geos_version_string),
                  inputs=dict(protocol_sha256=PROTOCOL, cohort_roster_sha256=ROSTER,
                              worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                              historical_rf_scorer_sha256=RF_SHA, historical_geometry_lf_sha256=rf.CORE_SHA,
                              intake_result_sha256=a.intake_result_sha, intake_terminal_sha256=a.intake_terminal_sha,
                              accepted_edit_receipt_sha256=a.readback_sha, accepted_edit_terminal_sha256=a.readback_terminal_sha,
                              **{k: intake[k] for k in ("images_roster_sha256","labels_roster_sha256","member_provenance_sha256")},
                              **{k: edits[k] for k in ("predictions_sha256","candidate_audit_sha256","edits_sha256")}))
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    a.output.mkdir()
    with (a.output / "RESULT_SAFE.json").open("xb") as out:
        out.write(data)
    print(json.dumps({"status": result["status"], "result_sha256": sha256(data).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
