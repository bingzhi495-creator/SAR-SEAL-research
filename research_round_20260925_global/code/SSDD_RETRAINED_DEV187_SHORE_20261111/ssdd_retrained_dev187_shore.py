"""Post-hoc fixed sea-land diagnostics on the closed SSDD-retrained DEV187 pair."""

from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from types import ModuleType


P299_SOURCE = ("SSDD_RETRAINED_DEV187_DIAGNOSTIC_20261111/ssdd_retrained_dev187_diagnostic.py",
               "ebefd0eab1394484219cda2991b45f3c3b6a93462938e3e4b0c8297b16785930")
SHORE_SOURCE = ("SSDD_DEV187_SHORE_DIAGNOSTIC_20261110/shore_diagnostic.py",
                "f4e212294578c616e05166c0b078673c1acd73878ab60894e94bb7f773a9b6e5")
P299_SAFE_SHA = "c96087f644e13f8199fc956f52af503bb726da0c003c064385c732dc074dc98d"
STATUS = "P302_SSDD_RETRAINED_DEV187_SHORE_DIAGNOSTIC_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE"
STRATA = ("all_water", "land_containing")
ARMS = ("native", "center")
COUNTS = {"all_water": (156, 294), "land_containing": (31, 86)}
SCOPE = dict(FIT=False, DEV=True, TEST=False, official_TEST_opened=False, training=False, model_forward_calls=0)
MIB = 1024**2


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def source_module(code, name, pin):
    relative, digest = pin
    path = code / relative
    require(path.resolve(strict=True) == path and path.is_file(), "Exact canonical ordinary source required")
    with path.open("rb") as stream:
        raw = stream.read(64 * 1024 + 1)
    require(0 < len(raw) <= 64 * 1024 and sha256(raw).hexdigest() == digest,
            "Exact ordinary diagnostic source required: " + path.name)
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def split_population(gt, rows, grids):
    require(list(grids) == [image["id"] for image in gt["images"]]
            and {value[0] for value in grids.values()} == set(STRATA), "Exact DEV ordered two-stratum population required")
    result = {}
    for name in STRATA:
        ids = {iid for iid, value in grids.items() if value[0] == name}
        subset = dict(gt, images=[row for row in gt["images"] if row["id"] in ids],
                      annotations=[row for row in gt["annotations"] if row["image_id"] in ids])
        result[name] = subset, {mode: [row for row in rows[mode] if row["image_id"] in ids] for mode in ARMS}
    return result


def duplicate_census(items, grids):
    groups = defaultdict(list)
    for item in items:
        groups[item["image_ref"]["sha256"]].append(grids[item["image_id"]][0])
    return dict(distinct_jpeg=len(groups), same_byte_alias_groups=sum(len(names) > 1 for names in groups.values()),
                cross_stratum_same_byte_alias_groups=sum(len(set(names)) > 1 for names in groups.values()))


def add_back(strata, prior):
    require(sum(group["equal_nonempty"]["prefix_counts"]["retained_per_arm"] for group in strata.values())
            == prior["prefix_counts"]["retained_per_arm"], "Both strata must recover the P299 equal-prefix census")
    expected = prior["score_blind_coverage"]
    require(sum(group["equal_nonempty"]["score_blind_coverage"]["eligible_gt"] for group in strata.values())
            == expected["eligible_gt"], "Both strata must recover the P299 eligible GT denominator")
    for mode in ARMS:
        for threshold, counts in expected["arms"][mode].items():
            for name in ("maximum_one_to_one", "any_candidate"):
                require(sum(group["equal_nonempty"]["score_blind_coverage"]["arms"][mode][threshold][name]
                            for group in strata.values()) == counts[name],
                        "Both strata must recover exact P299 score-blind coverage")


def analyze_subsets(gt, rows, grids, rank, support):
    _, template, _, _, official = support
    strata = {}
    for name, (subset, local) in split_population(gt, rows, grids).items():
        ids = [row["id"] for row in subset["images"]]
        raw = {mode: template.evaluate_cell(subset, local[mode], ids, official)[0] for mode in ARMS}
        strata[name] = dict(images=len(ids), annotations=len(subset["annotations"]),
                            raw_arm_metrics=raw, raw_center_minus_native_points=template.differences(
                                raw["native"], raw["center"]),
                            equal_nonempty=rank.analyze_pair(subset, local["native"], local["center"], support))
    return strata


def diagnose(args):
    p299 = source_module(args.code, "_p302_exact_p299", P299_SOURCE)
    shore = source_module(args.code, "_p302_exact_shore", SHORE_SOURCE)
    rank = source_module(args.code, "_p302_exact_rank",
                         ("SSDD_DEV187_POST_PRIMARY_RANK_20261110/p250_rank_diagnostic.py", shore.RANK_SHA))
    packages = {name: version(name) for name in ("numpy", "pycocotools", "scipy")}
    require(packages == dict(numpy="1.26.4", pycocotools="2.0.11", scipy="1.15.3"),
            "Exact closed P299 original CPU scoring packages required")
    accepted = p299.decode(p299.pinned_bytes(args.accepted, p299.ACCEPTED_SHA256, MIB, "P295 receipt"), dict)
    prior = p299.decode(p299.pinned_bytes(args.prior, P299_SAFE_SHA, MIB, "P299 independent readback"), dict)
    require(accepted["status"] == "P286_SSDD_DEV187_INDEPENDENTLY_ACCEPTED" and accepted["DEV"] is True
            and accepted["TEST"] is accepted["FIT"] is accepted["training"] is False
            and accepted["primary_GT"] == "INTEGER_PRIMARY" and accepted["floating_gt_sha256"] is None
            and accepted["coordinator_closed"] is accepted["coordinator_original_os_wait"] is True
            and type(accepted["coordinator_original_os_exit"]) is int and accepted["coordinator_original_os_exit"] == 0
            and prior["status"] == "SSDD_RETRAINED_DEV187_CPU_DIAGNOSTIC_INDEPENDENTLY_READBACK"
            and prior["original_launcher_wait"] is True and type(prior["original_launcher_exit"]) is int
            and prior["original_launcher_exit"] == 0 and prior["original_stderr_empty"] is True,
            "Exact independently closed original P295 and P299 DEV evidence required")
    _, template, _, scoring, official = support = rank.load_support(args.code)
    manifest_raw = shore.stable(args.image_manifest, shore.IMAGE_SHA, MIB)
    manifest_ref = dict(path=args.image_manifest.as_posix(), sha256=shore.IMAGE_SHA, bytes=len(manifest_raw))
    _, ids, items = scoring.fixed_images(scoring.Artifact(manifest_ref, manifest_raw))
    require(accepted["image_manifest_sha256"] == shore.IMAGE_SHA
            and accepted["integer_gt_sha256"] == scoring.INTEGER_GT_SHA
            and prior["report"]["integer_gt_sha256"] == scoring.INTEGER_GT_SHA
            and prior["report"]["accepted_p295_sha256"] == p299.ACCEPTED_SHA256
            and prior["report"]["package_versions"] == packages and prior["report"]["status"] == p299.STATUS
            and prior["report"]["DEV"] is True and prior["report"]["TEST"] is False
            and prior["report"]["model_forward_calls"] == 0 and prior["report"]["training"] is False,
            "Exact common DEV scope and P299 independently read-back source provenance required")
    gt_raw = p299.pinned_bytes(args.integer_gt, scoring.INTEGER_GT_SHA, MIB, "original integer GT",
                               exact_size=scoring.INTEGER_GT_BYTES)
    gt = p299.decode(gt_raw, dict)
    scoring.validate_gt(gt, ids, items)
    require(args.native != args.center and args.native.name == args.center.name == "PREDICTIONS.json",
            "Distinct independently accepted original P295 predictions required")
    rows, prediction_sha = {}, {}
    for mode in ARMS:
        arm = accepted["arms"][mode]
        digest = prediction_sha[mode] = arm["prediction_sha256"]
        require(digest == prior["report"]["prediction_sha256"][mode], "Exact P295/P299 arm prediction join")
        rows[mode] = p299.decode(p299.pinned_bytes(getattr(args, mode), digest, 64 * MIB, mode + " predictions"), list)
        nonempty = sum(map(len, template.nonempty_by_image(rows[mode], gt["images"])))
        require(len(rows[mode]) == arm["prediction_count"] and nonempty == arm["nonempty_masks"]
                and len(rows[mode]) - nonempty == arm["all_zero_masks"], "Exact accepted prediction census")
    require(scoring.paired_scores(official, gt, rows, ids) == accepted["primary"] == prior["report"]["primary"],
            "Recomputed official full DEV187 result must reproduce both accepted receipts")
    grids, mask_census = shore.mask_population(args.source_zip, items)
    require(mask_census == dict(all_water=156, land_containing=31, all_land=0),
            "Exact historical fixed author DEV187 land census")
    duplicates = duplicate_census(items, grids)
    require(duplicates["distinct_jpeg"] == accepted["distinct_jpeg"] == 183
            and duplicates["same_byte_alias_groups"] == 4, "Exact historical DEV187 alias census")
    strata = analyze_subsets(gt, rows, grids, rank, support)
    require({name: (value["images"], value["annotations"]) for name, value in strata.items()} == COUNTS,
            "Exact historical two-stratum image and integer GT denominators")
    add_back(strata, prior["report"]["equal_nonempty_candidate_analysis"])
    return dict(status=STATUS, **SCOPE, device="CPU", AP=True, posthoc_development_diagnostic=True,
                training_seed=prior["report"]["training_seed"], training_protocol=accepted["training_protocol"],
                evaluation_protocol=accepted["evaluation_protocol"], primary_GT="INTEGER_PRIMARY",
                integer_gt_sha256=scoring.INTEGER_GT_SHA, image_manifest_sha256=shore.IMAGE_SHA,
                accepted_p295_sha256=p299.ACCEPTED_SHA256, prior_p299_sha256=P299_SAFE_SHA,
                prediction_sha256=prediction_sha, full_primary_scores_exact=True, images=187, annotations=380,
                jpeg_alias_census=duplicates, source_commit=shore.SOURCE_COMMIT, source_git_blob=shore.SOURCE_BLOB,
                source_zip_sha256=shore.SOURCE_SHA, author_TRAIN_mask_members_read=187, author_TEST_mask_members_read=0,
                mask_census=mask_census, package_versions=packages, strata=strata,
                group_rule="any original land pixel versus all original pixels water; no prediction used",
                group_label_limit="external author sea-land labels consult SSDD ship GT during annotation",
                candidate_effect_scope="selected mask content and within-image order at equal candidate counts",
                donor_effect_scope="cross-image score ordering at fixed masks, within-image order and COCO matches")


def deny_network(event, _):
    if event.startswith("socket."):
        raise PermissionError("P302 runs offline on closed CPU inputs")


def main(argv=None):
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for flag in ("code", "accepted", "prior", "integer-gt", "native", "center", "image-manifest", "source-zip", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.flags.isolated == sys.flags.utf8_mode == 1
            and sys.dont_write_bytecode, "Exact isolated offline CPU scoring required")
    require(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink(), "Fresh canonical output required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.addaudithook(deny_network)
    report = diagnose(args)
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
