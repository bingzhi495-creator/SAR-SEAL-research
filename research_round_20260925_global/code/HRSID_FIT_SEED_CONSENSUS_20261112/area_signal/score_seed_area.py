"""Score a locked, already closed FIT candidate-area signal without changing masks."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
from types import ModuleType

MATCH_SHA = "8741bebb448120e37e0c10b58467a83d724f2294d20fddbc93720301285084aa"
GEOMETRY_SHA = "7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"
LABEL_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
SEEDS = (5201, 5202, 5203)
MODES = ("zero", "mean_raw", "median_raw", "mean_fit", "median_fit")
SLOPES = tuple(i / 4 for i in range(9))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact(path, expected, cap=32*2**20):
    require(path.resolve(strict=True) == path and 0 < path.stat().st_size <= cap, "Canonical bounded input")
    data = path.read_bytes()
    require(sha256(data).hexdigest() == expected, "Pinned bytes: " + path.name)
    return data


def load_module(name, path, digest):
    m = ModuleType(name)
    m.__file__ = str(path)
    sys.modules[name] = m
    exec(compile(exact(path, digest, 2**20), str(path), "exec"), m.__dict__)
    return m


def lines(path, digest):
    data = exact(path, digest)
    return [json.loads(line) for line in data.splitlines()]


def input_rows(args):
    r = json.loads(exact(args.capture / "RESULT_SAFE.json", args.capture_sha))
    require(r["FIT"] is True and r["DEV"] is r["TEST"] is r["GT_opened"] is False
            and r["model_forwards"] == 108 and r["coverage"]["gate"] == "COVERAGE_PASS",
            "Accepted closed phase 1 coverage pass")
    v = json.loads(exact(args.witness, args.witness_sha))
    require(v["result_sha256"] == args.capture_sha and v["coverage_verified"] is True,
            "Independent coverage acceptance required")
    arms = {a["seed"]: lines(args.capture / str(a["seed"]) / "CANDIDATES_PRIVATE.jsonl", a["prediction_sha256"])
            for a in r["arms"]}
    associations = lines(args.capture / "ASSOCIATIONS_PRIVATE.jsonl", r["coverage"]["associations_sha256"])
    require(len(associations) == 36 and all(len(arms[s]) == 36 for s in SEEDS), "Exact 36 images")
    return arms, associations


def cases(args, matching, geom):
    arms, associations = input_rows(args)
    labels = json.loads(exact(args.fit_index, LABEL_SHA))
    require(len(labels["images"]) == 2937 and len(labels["annotations"]) == 9260, "Exact FIT annotation release")
    wanted = {r["basename"] for r in arms[5202]}
    images = {Path(row["file_name"]).name: row for row in labels["images"] if Path(row["file_name"]).name in wanted}
    require(set(images) == wanted, "All FIT chips joined by exact native basename")
    image_ids = {x["id"] for x in images.values()}
    annotations = defaultdict(list)
    for a in labels["annotations"]:
        if a["image_id"] in image_ids:
            annotations[a["image_id"]].append(a)
    subset = dict(images=list(images.values()), annotations=[a for rows in annotations.values() for a in rows],
                  categories=labels["categories"])
    targets, neutral, _ = geom.geometry.polygon_support(subset, geom.vector)
    out = []
    with redirect_stdout(StringIO()):
        for left, anchor, right, map_row in zip(*(arms[s] for s in SEEDS), associations, strict=True):
            require(left["ordinal"] == anchor["ordinal"] == right["ordinal"] == map_row["ordinal"],
                    "Exact common ordinal")
            candidates = anchor["candidates"]
            image = images[anchor["basename"]]
            original, _, _ = matching.freeze_coco(image, annotations[image["id"]], candidates)
            target_by_rank = {rank: key for key, rank in original.items()}
            donor_area = {s: {r["rank"]: r["area"] for r in b["candidates"]} for s,b in ((5201,left),(5203,right))}
            covered = {a["anchor_rank"]:a for a in map_row["matches"]}
            for point in candidates:
                area, rank = point["area"], point["rank"]
                if not 32 <= area < 1024:
                    continue
                truth_id = target_by_rank.get(rank)
                target = None if truth_id is None or truth_id in neutral else float(targets[truth_id][0].area)
                donor = covered.get(rank)
                delta_mean = delta_median = 0.
                if donor is not None:
                    aa = [area] + [donor_area[s][donor["donor_ranks"][str(s)]] for s in (5201,5203)]
                    delta_mean, delta_median = mean(aa)-area, median(aa)-area
                    require(delta_median == donor["median_area_offset"], "Exact admitted donor area offset")
                out.append(dict(ordinal=anchor["ordinal"], area=area, truth=target, covered=donor is not None,
                                mean=delta_mean, median=delta_median, cap=2*min(8,area//10)))
    return out


def change(row, kind, slope):
    return max(-row["cap"], min(row["cap"], row[kind] * slope))


def abs_error(row, delta):
    return 100 * abs(row["area"]+delta-row["truth"]) / row["truth"]


def by_group(rows, fn):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["ordinal"]].append(fn(row))
    return mean(mean(values) for values in grouped.values())


def score(rows):
    matched = [r for r in rows if r["truth"] is not None]
    require(matched, "Nonempty fixed match cohort")
    covered = [r for r in matched if r["covered"]]
    parts, pooled = [], []
    for fold in range(3):
        train = [r for r in matched if r["ordinal"] % 3 != fold]
        require(train, "Three nonempty calibration folds")
        slopes = {kind: min(SLOPES, key=lambda slope: by_group(train,lambda r: abs_error(r,change(r,kind,slope))))
                  for kind in ("mean","median")}
        held = [r for r in rows if r["ordinal"] % 3 == fold]
        scored = []
        for row in held:
            actions = {"zero":0., "mean_raw":change(row,"mean",1.), "median_raw":change(row,"median",1.),
                       "mean_fit":change(row,"mean",slopes["mean"]),
                       "median_fit":change(row,"median",slopes["median"])}
            scored.append(dict(**row, actions=actions))
        parts.append(dict(slopes=slopes, eligible=len(held), metrics=summarize(scored)))
        pooled.extend(scored)
    result = summarize(pooled)
    gains = {m: result[m]["equal_source_error"] - result["median_fit"]["equal_source_error"] for m in ("zero","mean_fit")}
    positive = {m:sum(p["metrics"][m]["equal_source_error"] > p["metrics"]["median_fit"]["equal_source_error"]
                      for p in parts) for m in ("zero","mean_fit")}
    groups = len({r["ordinal"] for r in covered})
    pass_ = len(covered)>=30 and groups>=15 and gains["zero"]>=.50 and gains["mean_fit"]>=.25 and min(positive.values())>=2
    return dict(gate="MEDIAN_AREA_SIGNAL_LEAD" if pass_ else "NO_MEDIAN_AREA_SIGNAL_LEAD",
                eligible=len(rows), matched=len(matched), covered_matches=len(covered), covered_groups=groups,
                gains=gains, positive_folds=positive, metrics=result, folds=parts)


def summarize(rows):
    matched = [r for r in rows if r["truth"] is not None]
    require(matched, "Nonempty fixed held match cohort")
    covered = [r for r in matched if r["covered"]]
    metrics = {}
    for mode in MODES:
        fn = lambda r: abs_error(r,r["actions"][mode])
        actions = Counter()
        for row in rows:
            d = row["actions"][mode]
            if not d:
                continue
            actions["positive" if d>0 else "negative"] += 1
            truth = row["truth"]
            if truth is None:
                actions["unmatched_or_neutral"] += 1
            elif d * (truth-row["area"])>0:
                actions["toward"] += 1
        metrics[mode] = dict(equal_source_error=by_group(matched,fn), pooled_error=mean(map(fn,matched)),
                             covered_pooled_error=mean(map(fn,covered)) if covered else None, actions=actions)
    return metrics


def run(args):
    require(sys.flags.isolated and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "Isolated CPU only")
    exact(Path(__file__).resolve().with_name("PROTOCOL.md"), args.protocol_sha, 65536)
    matching = load_module("_seed_area_matching", args.match_source, MATCH_SHA)
    admission = load_module("_seed_area_geometry", args.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                            GEOMETRY_SHA)
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Same continuous geometry")
    result = score(cases(args,matching,admission.geometry_modules(args.code)))
    result.update(status="HRSID_FIT_SEED_AREA_SIGNAL_ACTUAL", FIT=True, DEV=False, TEST=False, AP=False,
                  capture_sha256=args.capture_sha, protocol_sha256=args.protocol_sha, model_forwards=0, optimizer_updates=0)
    data = (json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with args.output.open("xb") as f:
        f.write(data)
    print(json.dumps(dict(status=result["status"],gate=result["gate"],result_sha256=sha256(data).hexdigest())))


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    for n in ("capture","witness","fit-index","match-source","code","shapely-site","output"):
        p.add_argument("--"+n,type=Path,required=True)
    for n in ("capture-sha","witness-sha","protocol-sha"):
        p.add_argument("--"+n,required=True)
    def offline(event, _args):
        if event.startswith("socket."):
            raise PermissionError("No network in FIT scorer")
    sys.addaudithook(offline)
    run(p.parse_args())
