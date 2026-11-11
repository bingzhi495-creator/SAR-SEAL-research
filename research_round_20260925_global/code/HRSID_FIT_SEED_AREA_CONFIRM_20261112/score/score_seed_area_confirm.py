"""Measure held-out HRSID FIT area-estimate error on exact original stock matches."""

from collections import Counter, defaultdict
from math import isfinite
from statistics import mean


MODES = ("zero", "median", "mean", "soft", "size")
CONTROLS = ("zero", "mean", "soft", "size")
INTERCEPTS = (0, -.5, .5, -1, 1)
SLOPES = tuple(i / 4 for i in range(9))
COEFFICIENTS = tuple((a, b) for a in INTERCEPTS for b in SLOPES)
MATCH_STATES = ("valid", "invalid_or_neutral", "unmatched")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def by_source(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    return groups


def group_mean(groups, fn):
    return mean(mean(fn(row) for row in group) for group in groups.values()) if groups else None


def action(row, mode, a, b):
    if mode == "zero" or not row["covered"]:
        return 0
    cap = row["cap"]
    return min(cap, max(-cap, cap*a+b*row[f"{mode}_delta"]))


def error(row, delta):
    return 100*abs(row["area"]+delta-row["truth"])/row["truth"]


def fit_coefficients(groups, mode):
    if not groups or mode == "zero":
        return 0, 0
    return min(COEFFICIENTS, key=lambda ab: group_mean(groups, lambda row: error(row, action(row, mode, *ab))))


def count_cases(rows):
    states = Counter(row["match_status"] for row in rows)
    covered_states = Counter(row["match_status"] for row in rows if row["covered"])
    valid = [r for r in rows if r["match_status"] == "valid"]
    covered = [r for r in valid if r["covered"]]
    return dict(
        eligible_native_candidates=len(rows),
        covered_eligible_native_candidates=sum(covered_states.values()),
        original_stock_matched_eligible=states["valid"]+states["invalid_or_neutral"],
        invalid_or_neutral_matches=states["invalid_or_neutral"],
        unmatched_eligible=states["unmatched"],
        valid_matches=len(valid),
        valid_match_source_groups=len(by_source(valid)),
        covered_valid_matches=len(covered),
        covered_valid_groups=len(by_source(covered)),
        covered_invalid_or_neutral_matches=covered_states["invalid_or_neutral"],
        covered_unmatched=covered_states["unmatched"],
    )


def summarize(rows):
    valid = [r for r in rows if r["match_status"] == "valid"]
    covered = [r for r in valid if r["covered"]]
    valid_groups = by_source(valid)
    covered_groups = by_source(covered)
    metrics = {}
    for mode in MODES:
        fn = lambda row: error(row, row["actions"][mode])
        actions = Counter(dict.fromkeys(("positive", "negative", "unmatched", "invalid_or_neutral",
                                        "uncovered", "toward", "away", "flat_target"), 0))
        for row in rows:
            delta = row["actions"][mode]
            if not delta:
                continue
            actions["positive" if delta > 0 else "negative"] += 1
            if not row["covered"]:
                actions["uncovered"] += 1
            state = row["match_status"]
            if state == "valid":
                target = row["truth"]-row["area"]
                actions["toward" if delta*target > 0 else "away" if delta*target < 0 else "flat_target"] += 1
            else:
                actions[state] += 1
        require(actions["uncovered"] == 0, "Every arm is unchanged outside exact common coverage")
        metrics[mode] = dict(
            equal_source_error=group_mean(valid_groups, fn),
            pooled_error=mean(map(fn, valid)) if valid else None,
            covered_equal_source_error=group_mean(covered_groups, fn),
            covered_pooled_error=mean(map(fn, covered)) if covered else None,
            actions=dict(actions),
        )
    return metrics


def gains_from(metrics):
    target = metrics["median"]["equal_source_error"]
    return {m: metrics[m]["equal_source_error"]-target if target is not None else None for m in CONTROLS}


def score(rows):
    require(bool(rows), "Nonempty fixed eligible native roster")
    group_folds = {}
    for row in rows:
        fold = row["ordinal"] % 3
        require(group_folds.setdefault(row["group"], fold) == fold, "A source group crosses held-out folds")
        state, truth = row["match_status"], row["truth"]
        require(state in MATCH_STATES and (truth is not None) == (state == "valid")
                and (truth is None or isfinite(truth) and truth > 0), "Exact valid stock match area")
    folds, scored = [], []
    for fold in range(3):
        fit = [r for r in rows if r["match_status"] == "valid" and r["ordinal"] % 3 != fold]
        groups = by_source(fit)
        params = {mode: fit_coefficients(groups, mode) for mode in MODES}
        held = [r for r in rows if r["ordinal"] % 3 == fold]
        held_scored = [dict(row, actions={mode: action(row, mode, *params[mode]) for mode in MODES}) for row in held]
        metrics = summarize(held_scored)
        folds.append(dict(fold=fold, calibration={mode: dict(a=a,b=b) for mode,(a,b) in params.items()},
                          train_valid_matches=len(fit), train_valid_groups=len(groups),
                          held_counts=count_cases(held), metrics=metrics, gains=gains_from(metrics)))
        scored.extend(held_scored)
    metrics = summarize(scored)
    gains = gains_from(metrics)
    positive = {m: sum(fold["gains"][m] is not None and fold["gains"][m] > 0 for fold in folds) for m in CONTROLS}
    counts = count_cases(scored)
    pass_ = (counts["covered_valid_matches"] >= 30 and counts["covered_valid_groups"] >= 15
             and all(gains[m] is not None and gains[m] >= (.50 if m == "zero" else .25) for m in CONTROLS)
             and min(positive.values()) >= 2)
    return dict(gate="MEDIAN_AREA_SIGNAL_LEAD" if pass_ else "NO_MEDIAN_AREA_SIGNAL_LEAD",
                counts=counts, gains=gains, positive_folds=positive, metrics=metrics, folds=folds)



CAPTURE_PROTOCOL_SHA = "4ac12dcc198bfafc424084f613e8ca6a69fdb4ecef3ee9b62e3d8bc0ab64b185"
MATCH_SHA = "8741bebb448120e37e0c10b58467a83d724f2294d20fddbc93720301285084aa"
GEOMETRY_SHA = "7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"
LABEL_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
ACCEPTED_STATUS = "HRSID_FIT_SEED_AREA_CONFIRM_INDEPENDENT_ACCEPTED"
DELTAS = ("mean_delta", "median_delta", "soft_delta", "size_delta")


def validate_input(anchor_stream, signal_stream):
    from pycocotools import mask

    rows = []
    groups, ordinals, names = {}, set(), set()
    counts = Counter(dict.fromkeys(("mapped_native_candidates", "eligible_native_candidates",
                                    "covered_eligible_native_candidates", "images_with_eligible",
                                    "images_with_coverage", "covered_nonzero_median"), 0))
    for anchor, side in zip(anchor_stream, signal_stream, strict=True):
        ordinal, group, basename = (anchor[key] for key in ("ordinal", "group", "basename"))
        require(type(ordinal) is int and ordinal >= 0 and ordinal not in ordinals
                and type(group) is str and group and type(basename) is str and basename and basename not in names,
                "Exact unique original ordinal and basename")
        require((ordinal, group, basename) == tuple(side[key] for key in ("ordinal", "group", "basename")),
                "Signal and original image source join")
        require(groups.setdefault(group, ordinal%3) == ordinal%3, "A source group crosses held-out folds")
        ordinals.add(ordinal)
        names.add(basename)
        candidates = anchor["candidates"]
        require(type(candidates) is list and len(candidates) <= 100, "Bounded original top-100 candidate roster")
        eligible = {}
        previous_rank, previous_score = 0, 1.
        for row in candidates:
            rank, area, raw_score = (row[key] for key in ("rank", "area", "score"))
            require(type(rank) is int and previous_rank < rank <= 100, "Exact increasing original candidate rank")
            require(type(raw_score) in (float, int) and isfinite(raw_score) and 0 <= raw_score <= previous_score,
                    "Exact nonincreasing original score order")
            require(type(area) is int and area > 0 and int(mask.area(row["rle"])) == area,
                    "Exact original nonempty native raster area")
            previous_rank, previous_score = rank, raw_score
            if 32 <= area < 1024:
                eligible[rank] = area
        signals = {}
        donors_used = {seed: set() for seed in ("5201", "5203")}
        require(type(side["signals"]) is list, "Covered signal rows")
        for row in side["signals"]:
            rank = row["rank"]
            require(type(rank) is int and rank in eligible and rank not in signals,
                    "Every covered anchor must bind one eligible native candidate")
            require(type(row["cap"]) is int and row["cap"] == 2*min(8,eligible[rank]//10), "Exact edit-free area cap")
            require(all(type(row[k]) in (float, int) and isfinite(row[k]) for k in DELTAS),
                    "Original finite predicted deltas")
            donors = row["donor_ranks"]
            require(type(donors) is dict and set(donors) == set(donors_used), "Exact two donor seeds")
            for seed, donor in donors.items():
                require(type(donor) is int and 1 <= donor <= 100 and donor not in donors_used[seed],
                        "Frozen exclusive donor rank assignment")
                donors_used[seed].add(donor)
            signals[rank] = row
        counts.update(dict(mapped_native_candidates=len(candidates), eligible_native_candidates=len(eligible),
                           covered_eligible_native_candidates=len(signals), images_with_eligible=int(bool(eligible)),
                           images_with_coverage=int(bool(signals)),
                           covered_nonzero_median=sum(row["median_delta"] != 0 for row in signals.values())))
        rows.append(dict(ordinal=ordinal, group=group, basename=basename, candidates=candidates,
                         eligible=eligible, signals=signals))
    counts["source_groups"] = len(groups)
    counts["images"] = len(rows)
    counts["covered_source_groups"] = len({r["group"] for r in rows if r["signals"]})
    return rows, dict(counts)


def labeled_cases(inputs, labels, matching, geometry):
    from contextlib import redirect_stdout
    from io import StringIO
    from pathlib import PurePosixPath

    require(len(labels["categories"]) == 1 and labels["categories"][0]["id"] == 1
            and labels["categories"][0]["name"].casefold() == "ship", "Exact native stock ship category")
    required = {row["basename"] for row in inputs}
    images = {}
    for row in labels["images"]:
        basename = PurePosixPath(row["file_name"].replace(chr(92), "/")).name
        if basename in required:
            require(basename not in images, "Exact author FIT basename must be unique")
            images[basename] = row
    require(set(images) == required, "Every closed chip joined to author FIT labels")
    image_ids = {r["id"] for r in images.values()}
    require(len(image_ids) == len(images), "Author FIT image IDs must be unique")
    annotations = defaultdict(list)
    for row in labels["annotations"]:
        if row["image_id"] in image_ids:
            annotations[row["image_id"]].append(row)
    subset = dict(images=list(images.values()), annotations=[a for values in annotations.values() for a in values],
                  categories=labels["categories"])
    targets, neutral, _ = geometry.geometry.polygon_support(subset, geometry.vector)
    rows = []
    with redirect_stdout(StringIO()):
        for row in inputs:
            image = images[row["basename"]]
            grid = (image["height"], image["width"])
            require(all(tuple(c["rle"]["size"]) == grid for c in row["candidates"]),
                    "Original native prediction and FIT label grids must match")
            image_annotations = annotations[image["id"]]
            original = {}
            if row["eligible"] and image_annotations:
                original, _, _ = matching.freeze_coco(image, image_annotations, row["candidates"])
            by_rank = {rank: key for key, rank in original.items()}
            require(len(by_rank) == len(original) and set(by_rank) <= {c["rank"] for c in row["candidates"]},
                    "Original stock matches contain each GT and native rank at most once")
            for rank, area in row["eligible"].items():
                truth_id = by_rank.get(rank)
                if truth_id is None:
                    state, target = "unmatched", None
                elif truth_id in neutral:
                    state, target = "invalid_or_neutral", None
                else:
                    require(truth_id in targets, "Every nonneutral fixed original match has source polygon support")
                    state, target = "valid", float(targets[truth_id][0].area)
                signal = row["signals"].get(rank)
                rows.append(dict(ordinal=row["ordinal"], group=row["group"], area=area,
                                 match_status=state, truth=target, covered=signal is not None,
                                 cap=2*min(8,area//10), **{k: signal[k] if signal else 0 for k in DELTAS}))
    return rows


def pinned(path, digest, max_bytes=32*1024**2):
    from hashlib import sha256
    from re import fullmatch

    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
            and 0 < path.stat().st_size <= max_bytes and fullmatch("[0-9a-f]{64}",digest) is not None,
            "Canonical bounded input required: "+path.name)
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, "Exact pinned input digest required: "+path.name)
    return raw


def json_value(raw):
    import json

    def unique(pairs):
        obj = dict(pairs)
        require(len(obj) == len(pairs), "Duplicate JSON keys are prohibited")
        return obj

    def reject(value):
        raise ValueError("Nonfinite JSON value: "+value)

    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject)


def stream(path, digest):
    raw = pinned(path, digest)
    lines = raw.splitlines()
    require(raw.endswith(b"\n") and all(lines), "Exact complete nonblank JSONL stream")
    return [json_value(line) for line in lines]


def fit_only(record):
    return record["FIT"] is True and record["DEV"] is record["TEST"] is record["GT_opened"] is False


def read_admitted_capture(args):
    capture = json_value(pinned(args.capture/"RESULT_SAFE.json", args.capture_sha))
    require(fit_only(capture) and capture["images"] == 36 and capture["model_forwards"] == 108
            and capture["protocol_sha256"] == CAPTURE_PROTOCOL_SHA and capture["coverage"]["gate"] == "COVERAGE_PASS",
            "Closed exact FIT capture with prelocked signal coverage pass required")
    witness = json_value(pinned(args.witness,args.witness_sha))
    require(witness["status"] == ACCEPTED_STATUS and fit_only(witness) and witness["signals_verified"] is True
            and witness["result_sha256"] == args.capture_sha, "Independent exact GT-free signal acceptance required")
    arms = {a["seed"]: a for a in capture["arms"]}
    require(set(arms) == {5201,5202,5203} and len(capture["arms"]) == 3 and all(a["images"] == 36 for a in arms.values()),
            "Exact three seed and 36 image capture")
    anchor_sha = arms[5202]["prediction_sha256"]
    signal_sha = capture["coverage"]["signals_sha256"]
    original = stream(args.capture/"5202"/"CANDIDATES_PRIVATE.jsonl", anchor_sha)
    signals = stream(args.capture/"SIGNALS_PRIVATE.jsonl", signal_sha)
    require(len(original) == len(signals) == 36, "Exact original 36 closed anchor and signal rows")
    rows, counts = validate_input(original, signals)
    return rows, counts, anchor_sha, signal_sha


def module(name, path, digest):
    import sys
    from types import ModuleType

    result = ModuleType(name)
    result.__file__ = str(path)
    sys.modules[name] = result
    exec(compile(pinned(path, digest, 1024**2), str(path), "exec"), result.__dict__)
    return result


def write_new(path, value):
    from hashlib import sha256
    import json
    import os

    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    return sha256(raw).hexdigest()


def run(args):
    from hashlib import sha256
    from importlib.metadata import version
    import json
    import os
    from pathlib import Path
    import sys

    require(sys.flags.isolated and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and version("pycocotools") == "2.0.11", "Pinned isolated CPU stock COCO runtime")
    require(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists(), "Fresh exact output directory")
    scorer = Path(__file__).resolve()
    pinned(scorer.with_name("PROTOCOL.md"),args.protocol_sha,65536)
    rows, counts, anchor_sha, signal_sha = read_admitted_capture(args)
    match = module("_area_confirm_original_stock", args.match_source, MATCH_SHA)
    prior = module("_area_confirm_continuous", args.code_root/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112"/"continuous_endpoint.py",
                   GEOMETRY_SHA)
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Pinned author polygon geometry runtime")
    geom = prior.geometry_modules(args.code_root)

    args.output.mkdir(exist_ok=False)
    receipt = dict(status="HRSID_FIT_SEED_AREA_CONFIRM_SCORE_PRELABEL_V1",FIT=True,DEV=False,TEST=False,
                   GT_opened=False,AP=False,anchor_sha256=anchor_sha,signals_sha256=signal_sha,
                   capture_sha256=args.capture_sha,witness_sha256=args.witness_sha,
                   capture_protocol_sha256=CAPTURE_PROTOCOL_SHA,protocol_sha256=args.protocol_sha,
                   source_sha256=sha256(scorer.read_bytes()).hexdigest(),counts=counts)
    receipt_sha = write_new(args.output/"PRELABEL_RECEIPT_SAFE.json",receipt)
    print(json.dumps(dict(status=receipt["status"],receipt_sha256=receipt_sha),sort_keys=True),flush=True)

    labels = json_value(pinned(args.fit_index,LABEL_SHA))
    require(len(labels["images"]) == 2937 and len(labels["annotations"]) == 9260, "Exact author FIT annotation release")
    result = score(labeled_cases(rows,labels,match,geom))
    require(result["counts"]["eligible_native_candidates"] == counts["eligible_native_candidates"]
            and result["counts"]["covered_eligible_native_candidates"] == counts["covered_eligible_native_candidates"],
            "Exact eligible and common coverage denominators retained after labels")
    result.update(status="HRSID_FIT_SEED_AREA_CONFIRM_SCORE_ACTUAL_V1",FIT=True,DEV=False,TEST=False,
                  GT_opened=True,AP=False,mask_edits=False,model_forwards=0,optimizer_updates=0,
                  capture_sha256=args.capture_sha,witness_sha256=args.witness_sha,protocol_sha256=args.protocol_sha,
                  fit_index_sha256=LABEL_SHA,prelabel_receipt_sha256=receipt_sha,native_counts=counts)
    result_sha = write_new(args.output/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=result["status"],gate=result["gate"],result_sha256=result_sha),sort_keys=True),flush=True)


def main():
    from argparse import ArgumentParser
    from pathlib import Path
    import sys

    parser = ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("capture","witness","fit-index","match-source","code-root","shapely-site","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("capture-sha","witness-sha","protocol-sha"):
        parser.add_argument("--"+name,required=True)

    def offline(event, _):
        if event.startswith("socket."):
            raise PermissionError("No network in the offline FIT scorer")

    sys.addaudithook(offline)
    run(parser.parse_args())


if __name__ == "__main__":
    main()


