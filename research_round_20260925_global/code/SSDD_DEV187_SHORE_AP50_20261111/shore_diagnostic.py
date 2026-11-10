"""Compare AP50 candidate quota, cross-image scores and discovery in fixed DEV187 strata."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean
import sys
from types import ModuleType


SHORE_SHA = "f4e212294578c616e05166c0b078673c1acd73878ab60894e94bb7f773a9b6e5"
SHORE_RESULT_SHA = "fae38ddc7a61556bd6548ac59a772124948c2847691d599a443a0421586783b7"
ARMS = ("native", "center")
STRATA = ("all_water", "land_containing")
EXPECTED = dict(all_water=(156, 294), land_containing=(31, 86))


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def load_exact(path, digest, name, limit=65536):
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "Exact canonical source required")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit and sha256(raw).hexdigest() == digest, "Pinned ordinary source changed")
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def discover(records, image_count, eligible):
    """Original .50 all-area, maxDet=100 greedy match identities, excluding ignored rows."""
    need(len(records) >= image_count > 0, "Complete all-area COCO records required")
    found = set()
    for row in records[:image_count]:
        if row is None:
            continue
        matches, ignores = row["dtMatches"][0], row["dtIgnore"][0]
        for match, ignore in zip(matches, ignores, strict=True):
            if not ignore and match:
                aid = int(match)
                need(aid == match and aid in eligible and aid not in found,
                     "Foreign, nonintegral or duplicate discovery")
                found.add(aid)
    return found


def partition(eligible, first, second):
    """Partition a fixed GT census without conditioning AP on discoveries."""
    need(first <= eligible and second <= eligible, "No foreign discoveries permitted")
    return dict(both=len(first & second), first_only=len(first - second), second_only=len(second - first),
                neither=len(eligible - (first | second)))


def stratum_analysis(gt, raw, selected, support, rank, previous):
    _, template, coverage, _, official = support
    images, annotations = gt["images"], gt["annotations"]
    ids = [item["id"] for item in images]
    eligible = {row["id"] for row in annotations}
    need(len(eligible) == len(annotations) and all(i > 0 for i in eligible), "Exact positive distinct COCO GT IDs required")
    populations, discoveries, ap50, witness_cells = dict(raw=raw, equal_prefix=selected), {}, {}, {}
    for stage, rows in populations.items():
        blind = rank.blind_coverage(gt, rows, coverage)
        need(blind["eligible_gt"] == len(eligible) and not any(blind["excluded_gt"].values()),
             "Fixed eligible .50 matching census must equal original GT")
        metrics, seen = {}, {}
        for arm in ARMS:
            value, records, _ = template.evaluate_cell(gt, rows[arm], ids, official)
            seen[arm] = discover(records, len(ids), eligible)
            graph = blind["arms"][arm]["0.50"]
            need(len(seen[arm]) <= graph["maximum_one_to_one"] <= graph["any_candidate"] <= len(eligible),
                 "Greedy, score-blind maximum and independent discovery accounting differs")
            metrics[arm] = dict(AP50=value["AP50"], greedy_gt=len(seen[arm]),
                                greedy_recall=len(seen[arm]) / len(eligible),
                                score_blind_maximum_gt=graph["maximum_one_to_one"],
                                score_blind_maximum_recall=graph["maximum_recall"],
                                targets_with_any_candidate=graph["any_candidate"], candidates=len(rows[arm]))
            if stage == "raw":
                need(abs(value["AP50"] - previous["raw_arm_metrics"][arm]["AP50"]) < 1e-12,
                     "Recomputed raw stratum AP50 differs from accepted prior shore result")
            else:
                other = "center" if arm == "native" else "native"
                donated = template.score_template(rows[arm], rows[other])
                alternate, alternate_records, _ = template.evaluate_cell(gt, donated, ids, official)
                template.require_same_matching(records, alternate_records)
                need(discover(alternate_records, len(ids), eligible) == seen[arm],
                     "Donor changes only cross-image ordering and preserves .50 greedy discovery")
                ap50[arm] = {arm: value["AP50"], other: alternate["AP50"]}
                witness_cells[arm] = sum(row["dtMatches"].size for row in records if row is not None)
        discoveries[stage] = seen
        populations[stage] = dict(arms=metrics, discovery_native_then_center=partition(eligible, seen["native"], seen["center"]))

    transitions = {arm: partition(eligible, discoveries["raw"][arm], discoveries["equal_prefix"][arm]) for arm in ARMS}
    need(all(row["second_only"] == 0 for row in transitions.values()),
         "Equal nonempty prefixes must not add discoveries to the same native rank")
    n, c = ap50["native"], ap50["center"]
    raw_a, raw_b = (populations["raw"]["arms"][arm]["AP50"] for arm in ARMS)
    return dict(images=len(images), annotations=len(eligible), populations=populations,
                discovery_raw_then_equal_prefix_by_arm=transitions, equal_prefix_AP50_cells=ap50,
                exact_coco_match_ignore_witness_cells=witness_cells,
                contrasts_AP50_points=dict(raw_own=100 * (raw_b - raw_a),
                    equal_own=100 * (c["center"] - n["native"]),
                    center_score_donor_on_native_pool=100 * (n["center"] - n["native"]),
                    center_score_donor_on_center_pool=100 * (c["center"] - c["native"]),
                    center_pool_with_native_scores=100 * (c["native"] - n["native"]),
                    center_pool_with_center_scores=100 * (c["center"] - n["center"])))


def run(args):
    shore = load_exact(args.output.parent / "prior_shore_source.py", SHORE_SHA, "_exact_prior_shore")
    rank = load_exact(args.code / "SSDD_DEV187_POST_PRIMARY_RANK_20261110/p250_rank_diagnostic.py",
                      shore.RANK_SHA, "_exact_p255_AP50_only")
    rank.THRESHOLDS = (.50,)
    support = rank.load_support(args.code)
    p250, template, _, scorer, _ = support
    plan = p250.document(shore.stable(args.plan, rank.PLAN_SHA, p250.MIB))
    accepted = p250.document(shore.stable(args.accepted_score, rank.ACCEPTED_SHA, p250.MIB))
    p255 = p250.document(shore.stable(args.rank_score, shore.PRIOR_RANK_SHA, p250.MIB))
    prior = p250.document(shore.stable(args.output.parent / "PRIOR_SHORE_SAFE.json", SHORE_RESULT_SHA, p250.MIB))
    need(plan["status"] == p250.PLAN_STATUS and plan["evaluation_protocol"] == rank.PROTOCOL
         and accepted["closed_plan_sha256"] == prior["p250_plan_sha256"] == p255["closed_plan_sha256"] == rank.PLAN_SHA
         and prior["p250_accepted_sha256"] == rank.ACCEPTED_SHA
         and prior["p255_result_sha256"] == shore.PRIOR_RANK_SHA,
         "Exact independently closed P250, P255 and prior shore scope required")
    manifest_ref = plan["image_manifest"]
    manifest = shore.stable(args.image_manifest, shore.IMAGE_SHA, p250.MIB)
    need(p250.read_ref(manifest_ref, p250.MIB) == manifest, "Same closed DEV187 image manifest bytes required")
    _, ids, items = scorer.fixed_images(scorer.Artifact(manifest_ref, manifest))
    grids, census = shore.mask_population(args.source_zip, items)
    need(list(grids) == ids and census == prior["strata_census"] == dict(all_water=156, land_containing=31, all_land=0),
         "Exact previously closed sea–land split required")
    need(plan["integer_gt"]["sha256"] == p250.INTEGER_GT_SHA, "Exact integer primary GT required")
    gt = p250.document(p250.read_ref(plan["integer_gt"], p250.MIB))
    scorer.validate_gt(gt, ids, items)
    refs, arms = rank.bound_prediction_refs(p250, plan, accepted)
    by_seed = {}
    for seed in p250.TRAIN_PROTOCOLS:
        raw = {}
        for arm in ARMS:
            key = seed, arm
            record = arms[key]
            report = dict(mapped_rows=record["mapped_rows"], unmapped_slot1_rows=18700-record["mapped_rows"],
                          empty_image_predictions=record["empty_image_predictions"])
            raw[arm] = p250.score_rows(p250.read_ref(refs[key], 64 * p250.MIB), items, report)
        selected, counts = template.prefix_rows(*(template.nonempty_by_image(raw[arm], gt["images"]) for arm in ARMS))
        need(sum(counts) == p255["by_seed"][str(seed)]["prefix_counts"]["retained_per_arm"],
             "Exact prior P255 equal-count retained population required")
        by_seed[str(seed)] = {}
        for label in STRATA:
            local = {iid for iid, row in grids.items() if row[0] == label}
            subset = dict(gt, images=[row for row in gt["images"] if row["id"] in local],
                          annotations=[row for row in gt["annotations"] if row["image_id"] in local])
            need((len(local), len(subset["annotations"])) == EXPECTED[label], "Prior fixed stratum denominators differ")
            inputs = [{arm: [row for row in pool[arm] if row["image_id"] in local] for arm in ARMS}
                      for pool in (raw, selected)]
            previous = prior["by_seed"][str(seed)][label]
            need(len(inputs[1]["native"]) == len(inputs[1]["center"])
                 == previous["false_positive_burden"]["native"]["evaluated_predictions"],
                 "Exact prior shore equal-count selected per-stratum rows required")
            by_seed[str(seed)][label] = stratum_analysis(subset, *inputs, support, rank, previous)
    contrasts = {}
    for label in STRATA:
        rows = [values[label]["contrasts_AP50_points"] for values in by_seed.values()]
        contrasts[label] = {key: mean(row[key] for row in rows) for key in rows[0]}
    return dict(status="SLSSDD_P250_DEV187_AP50_DISCOVERY_COMPLETE_V1", by_seed=by_seed, mean_contrasts_AP50_points=contrasts,
                prior_shore_result_sha256=SHORE_RESULT_SHA, author_zip_sha256=shore.SOURCE_SHA,
                p250_accepted_sha256=rank.ACCEPTED_SHA, p250_plan_sha256=rank.PLAN_SHA,
                p255_result_sha256=shore.PRIOR_RANK_SHA, integer_gt_sha256=p250.INTEGER_GT_SHA,
                author_TRAIN_mask_members_read=187, author_TEST_mask_members_read=0, AP=True, **shore.SAFE_SCOPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for flag in ("source-zip", "image-manifest", "output", "code", "plan", "accepted-score", "rank-score"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.flags.isolated == sys.flags.utf8_mode == 1
         and sys.dont_write_bytecode, "Offline isolated CPU runtime required")
    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise RuntimeError("Offline CPU diagnostic required")
    sys.addaudithook(offline)
    result = run(args)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
