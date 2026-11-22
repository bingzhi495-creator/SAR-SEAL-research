"""Diagnose fixed UEIKAP edits after associating the original boxes to author hulls."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from time import perf_counter


PROTOCOL_SHA = "8e5e3eb4cca22a2a0371f78dd4e279de2d79469d31321348147fefffb99f32b5"
ADAPTER_SHA = "9ecbac508e1506b6cee2706a31f497eb4a46dd9bb62fbd56340e9e64ec981fdc"
INTAKE_SHA = "d19bad46ea19ac4b01c6b462eb7470a45d3d0ed498015675cd7778a0e83752d8"
SCORE_SHA = "79c7a83524fec183bbf66fd89cf018f4501c2b91b38cccbbff5330f4ae602fc6"
READBACK_SHA = "cfd1128eb8e13dee1c68af339ad13afebe44e94a47d77d687672a24fcc787850"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(payload):
    return sha256(payload).hexdigest()


def pinned(path, expected):
    data = path.read_bytes()
    require(digest(data) == expected, "Input SHA-256 changed: " + path.name)
    return data


def box_iou(left, right):
    ax, ay, bx, by = left
    cx, cy, dx, dy = right
    a = max(0.0, bx - ax) * max(0.0, by - ay)
    b = max(0.0, dx - cx) * max(0.0, dy - cy)
    overlap = max(0.0, min(bx, dx) - max(ax, cx)) * max(0.0, min(by, dy) - max(ay, cy))
    union = a + b - overlap
    return overlap / union if union else 0.0


def match_box(candidates, hull_boxes):
    """Greedy descending original score; earlier author row wins exact box IoU ties."""
    assignments = {}
    for pos, candidate in enumerate(candidates, 1):
        best_iou, best_idx = 0.5, None
        for idx, hull_box in enumerate(hull_boxes):
            if idx in assignments:
                continue
            overlap = box_iou(candidate, hull_box)
            if overlap > best_iou or overlap == best_iou and best_idx is None:
                best_iou, best_idx = overlap, idx
        if best_idx is not None:
            assignments[best_idx] = pos
    return assignments


def weak_hulls(assignments, raster_iou):
    return {hull for hull, pos in assignments.items() if raster_iou[pos-1, hull] < .50}


def load_adapter(path):
    pinned(path, ADAPTER_SHA)
    spec = spec_from_file_location("pinned_rf_ueikap_score", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(core, shapely, polygon, stock, edits):
    values, cache = [], {}
    for name in core.ARMS:
        rle = edits.get(name, (0, stock))[1]
        key = rle["counts"]
        if key not in cache:
            cache[key] = core.geometric_iou(rle, polygon, shapely)
        values.append(cache[key])
    return values


def transition_summary(arms):
    return {f"{t:.2f}": {name: Counter(gained=0,lost=0,retained=0,both_below=0) for name in arms}
            for t in (.50,.75,.90)}


def tally_thresholds(accum, arms, values):
    original = values[arms.index("raw")]
    for str_t, records in accum.items():
        t = float(str_t)
        for name, value in zip(arms, values):
            event = ("retained" if value >= t else "lost") if original >= t else ("gained" if value >= t else "both_below")
            records[name][event] += 1


def run(a, adapter, core, shapely):
    previous = json.loads(pinned(a.prior_score, SCORE_SHA))
    intake = core.json_pinned(a.intake / "intake_terminal.json", INTAKE_SHA)
    require(intake["status"] == "completed_verified" and intake["actual_exit"] == 0,
            "Closed approved intake")
    receipt = core.json_pinned(a.readback, READBACK_SHA)
    require(receipt["GT_opened"] is False and receipt["exact_support_xor_area"] is True and
            previous["inputs"]["accepted_image_only_replay_sha256"] == READBACK_SHA,
            "Same accepted label-free point-head edits")
    admitted = adapter.verified_streams(core, intake, receipt, (a.predictions,a.audit,a.edits))
    audit = core.stream_pinned(a.audit, receipt["candidate_audit_sha256"])
    images = core.json_pinned(a.intake / "images_roster.json", intake["images_roster_sha256"])
    labels = core.json_pinned(a.intake / "labels_roster.json", intake["labels_roster_sha256"])
    members = core.json_pinned(a.intake / "member_provenance.json", intake["member_provenance_sha256"])
    reservation = core.json_pinned(a.reservation, core.RESERVATION_SHA)
    core.verify_reservation(reservation, labels["items"], members)
    expected = list(range(1,core.IMAGES+1))
    require(labels["expected_ids"] == images["expected_ids"] == expected and
            len(admitted) == len(audit) == len(images["items"]) == len(labels["items"]) == core.IMAGES,
            "Complete original hull cohort")

    old_population, box_population, weak_population = defaultdict(list), defaultdict(list), defaultdict(list)
    box_only_population, weak_only_population = defaultdict(list), defaultdict(list)
    bins = Counter(dict.fromkeys(("both", "box_only", "mask_only", "neither", "shared_same_candidate",
                                 "shared_different_candidate", "weak_box_associated", "weak_also_mask_matched",
                                 "box_with_positive_fixed_budget", "weak_with_positive_fixed_budget", "weak_zero_fixed_budget"),0))
    active_box, active_weak, weak_groups = set(), set(), set()
    joint_transitions, weak_transitions = transition_summary(core.ARMS),transition_summary(core.ARMS)
    no_effect = [0.0] * len(core.ARMS)
    for image_id, img, ann, entry, inspected in zip(expected,images["items"],labels["items"],admitted,audit):
        masks, changes = entry
        require(image_id == img["image_id"] == ann["image_id"] == inspected["image_id"] and
                img["original_hw"] == [core.HEIGHT,core.WIDTH], "Same source image and native canvas")
        group = ann["acquisition_group"]
        truth, classes = core.parse_yolo(core.label_path(a.intake,ann["label_ref"]),core.HEIGHT,core.WIDTH,shapely)
        require(truth and classes == ann["count_by_class"], "Retain every author hull and ignore epoch wakes")
        polygons = [poly for poly,_ in truth]
        rasters = [rle for _,rle in truth]
        mask_assignment = core.match_native(masks,rasters)
        candidates = inspected["candidates"]
        geom = [item["box_xyxy"] for item in candidates]
        require(len(geom) == len(masks) and all(len(x) == 4 and all(0 <= v <= core.WIDTH for v in x) for x in geom),
                "Native clipped XYXY box of each preserved prediction")
        box_assignment = match_box(geom,[p.bounds for p in polygons])
        weak = weak_hulls(box_assignment,core.codec.iou(masks,rasters,[False]*len(rasters))) if box_assignment else set()
        for hull, polygon in enumerate(polygons):
            pos = box_assignment.get(hull)
            old = mask_assignment.get(hull)
            category = ("both" if old else "box_only") if pos else ("mask_only" if old else "neither")
            bins[category] += 1
            if category == "both":
                bins["shared_same_candidate" if old == pos else "shared_different_candidate"] += 1
            stock = masks[pos-1] if pos else None
            edits = changes.get(pos,{})
            values = measure(core,shapely,polygon,stock,edits) if pos else no_effect
            if old == pos:
                old_values = values
            elif old:
                old_values = measure(core,shapely,polygon,masks[old-1],changes.get(old,{}))
            else:
                old_values = no_effect
            old_population[group].append(old_values)
            box_population[group].append(values)
            weak_population[group].append(values if hull in weak else no_effect)
            positive = edits.get("fixed",(0,))[0] > 0
            if pos:
                box_only_population[group].append(values)
                tally_thresholds(joint_transitions,core.ARMS,values)
                if positive:
                    bins["box_with_positive_fixed_budget"] += 1
                    active_box.add(group)
            if hull in weak:
                weak_only_population[group].append(values)
                tally_thresholds(weak_transitions,core.ARMS,values)
                weak_groups.add(group)
                bins["weak_box_associated"] += 1
                bins["weak_also_mask_matched"] += int(bool(old))
                name = "weak_with_positive_fixed_budget" if positive else "weak_zero_fixed_budget"
                bins[name] += 1
                if positive:
                    active_weak.add(group)

    require(sum(bins[key] for key in ("both","box_only","mask_only","neither")) == core.LABELS
            and len(old_population) == len(box_population) == len(weak_population) == core.GROUPS,
            "Complete labeled-hull and acquisition denominators")
    require(bins["both"] + bins["mask_only"] == previous["labeled_hulls_and_coverage"]["native_matched"]
            and core.paired(old_population) == previous["primary_all_labeled_hulls"],
            "Exact fixed-mask result reproduced before interpreting the new association")
    primary = core.paired(weak_population)
    coverage = bins["weak_with_positive_fixed_budget"] >= core.MIN_EXPOSED_HULLS and \
        len(active_weak) >= core.MIN_EXPOSED_GROUPS
    lower = primary["contrasts"][core.PRIMARY]["paired_95_interval_points"][0]
    return dict(status="UEIKAP_RFDETR_BOX_ASSOCIATION_DIAGNOSTIC_COMPLETE_V1", GT_opened=True,
                previously_evaluated_cohort=True, label_assisted_assignment=True, images=core.IMAGES,
                labeled_hulls=core.LABELS, acquisition_groups=core.GROUPS, detector_forwards=0, optimizer_updates=0,
                whole_scene_AP=False, original_mask_stats_exact=True, assignment_counts=dict(bins),
                box_associated=bins["both"]+bins["box_only"], native_mask_associated=bins["both"]+bins["mask_only"],
                box_associated_source_groups=len(box_only_population), weak_source_groups=len(weak_groups),
                active_box_source_groups=len(active_box), active_weak_source_groups=len(active_weak),
                weak_full_cohort_primary=primary, all_box_full_cohort=core.paired(box_population),
                box_subset_sensitivity=core.paired(box_only_population) if box_only_population else None,
                weak_subset_sensitivity=core.paired(weak_only_population) if weak_only_population else None,
                all_box_thresholds=joint_transitions, weak_box_thresholds=weak_transitions,
                weak_coverage_met=coverage, weak_positive_lower_bound=lower>0,
                weak_coverage_and_effect_met=coverage and lower>0,
                protocol_sha256=PROTOCOL_SHA, old_score_sha256=SCORE_SHA, readback_sha256=READBACK_SHA)


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("protocol","adapter","reference-scorer","intake","reservation","readback",
                 "predictions","edits","audit","prior-score","output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(),"Fresh independent scoring output")
    pinned(a.protocol, PROTOCOL_SHA)
    import shapely
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Historic geometry runtime")
    adapter = load_adapter(a.adapter)
    core = adapter.load_geometry(a.reference_scorer)
    start = perf_counter()
    result = run(a,adapter,core,shapely)
    result.update(source_sha256=digest(Path(__file__).read_bytes()),seconds=perf_counter()-start)
    data = (json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    a.output.mkdir()
    (a.output/"RESULT_SAFE.json").write_bytes(data)
    print(json.dumps(dict(status=result["status"],result_sha256=digest(data))),flush=True)


if __name__ == "__main__":
    main()
