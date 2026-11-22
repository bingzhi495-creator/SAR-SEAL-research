"""Score frozen RF-DETR edits on previously inspected UEIKAP hulls."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys


CORE_SHA = "d541d6dffc9a7d49317dea34f53153c4121ba4d7523e5c904e5f7024fa895f20"
REPLAY_STATUS = "UEIKAP_RFDETR_RESERVED_EDIT_REPLAY_ACCEPTED_V1"


def load_geometry(path):
    source = path.read_bytes().replace(b"\r\n", b"\n")
    if sha256(source).hexdigest() != CORE_SHA:
        raise ValueError("Unexpected UEIKAP reference geometry source")
    spec = spec_from_file_location("ueikap_reference_geometry", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_masks(core, prediction, audit, image_id):
    """Match in compact ship-only positions. Preserve joint-class ranks and gaps."""
    items = audit["candidates"]
    ranks = [x["native_rank"] for x in items]
    core.check(all(type(rank) is int and 1 <= rank <= 100 for rank in ranks)
               and ranks == sorted(set(ranks)), "Invalid original top-100 ranks")
    core.check(all(type(x["position"]) is int and x["position"] == pos
                   for pos, x in enumerate(items, 1)), "Invalid compact ship positions")
    compact_audit = dict(audit, candidates=[dict(x, native_rank=x["position"]) for x in items])
    return core.verify_native(prediction, compact_audit, image_id, core.HEIGHT, core.WIDTH)


def verified_streams(core, intake_terminal, receipt, paths):
    core.check(receipt["status"] == REPLAY_STATUS and receipt["exact_replayed_policy_choices"] is True
               and receipt["exact_support_xor_area"] is True and receipt["author_TEST"] is False
               and receipt["model_forwards"] == receipt["optimizer_updates"] == 0
               and receipt["manifest_sha256"] == intake_terminal["images_roster_sha256"],
               "An independently accepted label-free RF-DETR replay is required")
    streams = [core.stream_pinned(path, receipt[field]) for path, field in zip(
        paths, ("predictions_sha256", "candidate_audit_sha256", "edits_sha256"))]
    core.check(all(len(x) == core.IMAGES for x in streams), "Incomplete RF-DETR per-image streams")
    verified = []
    for image_id, (pred, audit, edit) in enumerate(zip(*streams), 1):
        core.check(edit["image_id"] == image_id, "RF-DETR edit stream order differs")
        masks = native_masks(core, pred, audit, image_id)
        updates = core.verify_edits(edit["changes"], masks, core.HEIGHT, core.WIDTH)
        verified.append((masks, updates))
    return verified


def execute(core, args, shapely):
    terminal = core.json_pinned(args.intake / "intake_terminal.json", args.intake_terminal_sha)
    core.check(terminal["status"] == "completed_verified" and terminal["actual_exit"] == 0,
               "Unverified UEIKAP image intake")
    receipt = core.json_pinned(args.readback, args.readback_sha)
    predictions = verified_streams(core, terminal, receipt, (args.predictions, args.audit, args.edits))
    # No target label or label manifest is opened until the full prediction population passes.
    images = core.json_pinned(args.intake / "images_roster.json", terminal["images_roster_sha256"])
    labels = core.json_pinned(args.intake / "labels_roster.json", terminal["labels_roster_sha256"])
    provenance = core.json_pinned(args.intake / "member_provenance.json", terminal["member_provenance_sha256"])
    reserved = core.json_pinned(args.reservation, core.RESERVATION_SHA)
    core.verify_reservation(reserved, labels["items"], provenance)
    ids = list(range(1, core.IMAGES + 1))
    core.check(labels["expected_ids"] == images["expected_ids"] == ids and len(images["items"]) == core.IMAGES,
               "Unexpected native image and label population")
    counts = Counter(dict.fromkeys(("labeled_hulls", "native_matched", "native_unmatched", "native_area_below_32",
                                    "native_area_at_least_1024", "eligible_area_sf_zero", "sf_positive"), 0))
    all_groups, matched_groups = defaultdict(list), defaultdict(list)
    exposed_groups, native_areas = set(), []
    edits_by_arm = Counter(dict.fromkeys(core.ARMS, 0))
    transitions = {str(t): {arm: Counter(retained=0, lost=0, gained=0, both_below=0) for arm in core.ARMS}
                   for t in (0.75, 0.90)}
    for image_id, image, label, (masks, updates) in zip(ids, images["items"], labels["items"], predictions):
        core.check(image_id == image["image_id"] == label["image_id"]
                   and image["original_hw"] == [core.HEIGHT, core.WIDTH], "Image dimensions or identity differ")
        group = label["acquisition_group"]
        truth, class_counts = core.parse_yolo(core.label_path(args.intake, label["label_ref"]),
                                             core.HEIGHT, core.WIDTH, shapely)
        core.check(truth and class_counts == label["count_by_class"], "Original hull or wake count differs")
        matched = core.match_native(masks, [raster for _, raster in truth])
        counts.update(labeled_hulls=len(truth), native_matched=len(matched), native_unmatched=len(truth)-len(matched),
                      wake_rows_excluded=class_counts.get("1", 0))
        for hull, (polygon, _) in enumerate(truth):
            if hull not in matched:
                all_groups[group].append([0.0] * len(core.ARMS))
                continue
            pos = matched[hull]
            original = masks[pos - 1]
            area = int(core.codec.area(original))
            native_areas.append(area)
            variants = updates.get(pos, {})
            if area < 32:
                counts["native_area_below_32"] += 1
            elif area >= 1024:
                counts["native_area_at_least_1024"] += 1
            elif variants.get("fixed", (0,))[0] > 0:
                counts["sf_positive"] += 1
                exposed_groups.add(group)
            else:
                counts["eligible_area_sf_zero"] += 1
            cache, values = {}, []
            for arm in core.ARMS:
                swaps, mask = variants.get(arm, (0, original))
                edits_by_arm[arm] += int(swaps > 0)
                if mask["counts"] not in cache:
                    cache[mask["counts"]] = core.geometric_iou(mask, polygon, shapely)
                values.append(cache[mask["counts"]])
            for name, arms in transitions.items():
                threshold = float(name)
                old_positive = values[core.ARMS.index("raw")] >= threshold
                for arm, value in zip(core.ARMS, values):
                    new_positive = value >= threshold
                    event = (("retained" if new_positive else "lost") if old_positive
                             else ("gained" if new_positive else "both_below"))
                    arms[arm][event] += 1
            all_groups[group].append(values)
            matched_groups[group].append(values)
    core.check(counts["labeled_hulls"] == core.LABELS == counts["native_matched"] + counts["native_unmatched"]
               and len(all_groups) == core.GROUPS
               and counts["native_matched"] == sum(counts[k] for k in ("native_area_below_32",
                   "native_area_at_least_1024", "eligible_area_sf_zero", "sf_positive")), "Hull or exposure totals differ")
    main = core.paired(all_groups)
    coverage = counts["sf_positive"] >= core.MIN_EXPOSED_HULLS and len(exposed_groups) >= core.MIN_EXPOSED_GROUPS
    return {
        "status": "UEIKAP_RFDETR_RESERVED_ANNOTATED_HULL_SCORE_COMPLETE_V1",
        "images": core.IMAGES, "whole_scene_AP": False, "previously_scored_acquisition_panel": True,
        "model_forwards": 0, "optimizer_updates": 0,
        "ship_candidates_within_joint_top100": sum(len(masks) for masks, _ in predictions),
        "images_without_ship_candidate": sum(not masks for masks, _ in predictions),
        "labeled_hulls_and_coverage": dict(counts, sf_positive_source_groups=len(exposed_groups),
                                           originally_matched_source_groups=len(matched_groups)),
        "native_matched_mask_area_px_five_number_summary": core.area_summary(native_areas),
        "primary_estimand": core.PRIMARY, "primary_all_labeled_hulls": main,
        "originally_matched_hulls_sensitivity": core.paired(matched_groups) if matched_groups else None,
        "edited_among_original_matches_by_arm": dict(edits_by_arm),
        "continuous_iou_threshold_transitions_on_original_matches": transitions,
        "coverage_gate_met": coverage,
        "frozen_geometry_advance": coverage and main["contrasts"][core.PRIMARY]["paired_95_interval_points"][0] > 0,
        "inputs": {"original_reference_geometry_lf_sha256": CORE_SHA,
                   "rf_scorer_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                   "intake_terminal_sha256": args.intake_terminal_sha, "image_roster_sha256": receipt["manifest_sha256"],
                   "labels_roster_sha256": terminal["labels_roster_sha256"],
                   "member_provenance_sha256": terminal["member_provenance_sha256"],
                   "old_reservation_sha256": core.RESERVATION_SHA, "accepted_image_only_replay_sha256": args.readback_sha,
                   **{name: receipt[name] for name in ("predictions_sha256", "candidate_audit_sha256", "edits_sha256")}},
        "geometry_backend": {"shapely": shapely.__version__, "geos": shapely.geos_version_string},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("intake", "reservation", "readback", "predictions", "audit", "edits", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("intake-terminal-sha", "readback-sha"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--reference-scorer", type=Path, default=Path(__file__).resolve().parent.parent /
                        "UEIKAP_RESERVED_HULL_FROZEN_20261122" / "score.py")
    parser.add_argument("--shapely-site", type=Path)
    args = parser.parse_args()
    if args.shapely_site:
        sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    core = load_geometry(args.reference_scorer)
    core.check(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
               "Use the same geometry runtime as the original UEIKAP evaluation")
    core.check(not args.output.exists(), "Use a fresh output directory")
    result = execute(core, args, shapely)
    data = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    args.output.mkdir()
    with (args.output / "RESULT_SAFE.json").open("xb") as out:
        out.write(data)
    print(json.dumps({"status": result["status"], "result_sha256": sha256(data).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
