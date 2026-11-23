"""Score a signed-budget endpoint replay using the accepted PSeg-SSDD probability archive."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
import shapely


NAME = "PSEG232_PHOENIX_BALANCED_RANK_PRELOCK_20261123.md"
LOCK = "bbe0bb66def842d0e07f73bb7fc2c1cee60ef30c78f6518217cc19e656c93578"
SOURCE = "5a638df3638a8647dc49b81a1f63ee9d72bee2d41dc1e94e276b5a0d2016fdc1"
PARENT = "e96c7fe8dc857fa85e0fd6051cf267068ae4a027ea8299a19678f5facfcfbcff"
PARENT_EXIT = "9f5642348225db7dc32c845ba69b2325ffeb34f299b415acc63dc403c33ddb55"
ARM = "phoenix_balanced"
STATUS = "PSEG232_PHOENIX_BALANCED_RANK_SCORE_COMPLETE"


def load_bound(factorial_source):
    data = factorial_source.read_bytes()
    if sha256(data).hexdigest() != SOURCE:
        raise ValueError("Expected the immutable prior PSeg score worker")
    source = ModuleType(factorial_source.stem)
    source.__file__ = str(factorial_source)
    exec(compile(data, str(factorial_source), "exec"), source.__dict__)
    return source


def balanced(source, request, posteriors, blocks, roster):
    totals = Counter()
    for image, tape, item in zip(request["images"], posteriors, roster, strict=True):
        source.require(image["image_id"] == tape["image_id"] == item["image_id"], "Expected the same image block")
        _, edits, hw = blocks[item["image_ref"]["path"]]
        for payload, probability in zip(image["candidates"], tape["candidates"], strict=True):
            pos, k = payload["position"], payload["k"]
            source.require(pos == probability["position"], "Endpoint slot changed")
            coords = np.asarray(payload["coords"])
            q = np.asarray(probability["probabilities"])
            add_order = np.argsort(-q[:, 0], kind="stable")
            drop_order = np.argsort(q[:, 1], kind="stable")
            left, right = coords[add_order[:k], 0], coords[drop_order[:k], 1]
            stock = source.decoded(payload["segmentation"], hw)
            revised = stock.copy(order="C")
            revised.flat[left], revised.flat[right] = True, False
            rle = source.encode(revised)
            check = source.decoded(rle, hw)
            added, dropped = check & ~stock, stock & ~check
            source.require(np.array_equal(np.flatnonzero(added), np.sort(left))
                           and np.array_equal(np.flatnonzero(dropped), np.sort(right))
                           and len(left) == len(right) == k
                           and check.sum() == stock.sum(), "Signed cut or serialized native geometry changed")
            free = edits[str(pos)]["nc"]["rle"]
            edits[str(pos)][ARM] = {"rle": rle}
            totals.update(candidates=1, additions=int(added.sum()), removals=int(dropped.sum()),
                          different_from_free=int(rle != free))
    source.require(tuple(totals[k] for k in ("candidates", "additions", "removals")) == (11054, 73770, 73770),
                   "Frozen per-side budget census differs")
    return dict(totals)


def score(args):
    started = perf_counter()
    factorial = load_bound(args.factorial_source)
    factorial.pin(args.protocol, LOCK)
    factorial.require(args.protocol.name == NAME and not args.output.exists()
                      and shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
                      "Expected a fresh output, the locked protocol and original geometry engine")
    parent, parent_sha, exit_sha = factorial.observed(
        args.parent_score, args.parent_terminal, "score", factorial.SCORE_STATUS, SOURCE, (PARENT, PARENT_EXIT))
    history = factorial.historical(args)
    comp, scorer, geom, prior, _, roster, audits, original, blocks = history
    _, chain, inference = factorial.build_inputs(args, history)
    factorial.require(all(parent[k] == v for k, v in chain.items())
                      and inference["threshold_projection_exact"] and parent["protocol_sha256"] == factorial.LOCK,
                      "Prior factorial archive lineage differs")

    request = json.loads(factorial.pin(args.request / "REQUEST_PRIVATE.json", chain["request_sha256"]))
    tape = factorial.pin(args.inference / "NARROW_POSTERIORS_PRIVATE.jsonl", chain["narrow_posteriors_sha256"])
    totals = balanced(factorial, request, map(json.loads, tape.splitlines()), blocks, roster)
    scorer.ARMS += (ARM,)
    print(json.dumps({"phase": "qualified_balanced_narrow_masks", **totals}), flush=True)

    gt = json.loads(factorial.pin(args.author_labels, scorer.GT_SHA))
    factorial.require(len(gt["images"]) == 232 and len(gt["annotations"]) == 546, "Exact author polygons required")
    scored = scorer.score_populations(gt, blocks, scorer.EXCLUDED)
    base = geom.load(args.code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/score_endpoint.py", geom.OLD)
    continuous = geom.load(args.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", geom.GEOM)
    modules = continuous.geometry_modules(args.code)
    truth, neutral, _ = modules.geometry.polygon_support(gt, modules.vector)
    factorial.require(len(truth) == 546 and not neutral, "Exact floating polygons required")
    ids = {image["file_name"]: image["id"] for image in gt["images"]}
    annotations = {ann["id"]: ann for ann in gt["annotations"]}
    records = {}
    for row, audit, edit in zip(roster, audits, original, strict=True):
        name = row["image_ref"]["path"]
        stock, changes, _ = blocks[name]
        records[ids[name]] = dict(name=name, rgb=edit["rgb_sha256"], native=stock,
                                  audit=audit["candidates"][:100], edit=changes)

    dynamic, static, learned = comp.NEW_ARMS
    factorial.PAIRS = ((ARM, static), (ARM, learned), (ARM, "nc"))
    reference_arms = ("raw", dynamic, static, learned, ARM)
    cached, anchors, views = {}, None, {}
    for view, result in scored.items():
        metrics = result["metrics_fraction"]
        earlier = parent["views"][view]
        factorial.require(all(metrics[k] == value for k, value in earlier["metrics_fraction"].items()),
                          "Prior all-candidate COCO vectors fail parity")
        expected_geo = prior["geometry"][view]
        expected_geo = dict(expected_geo, arms={**expected_geo["arms"], ARM: {}})
        ids_view = sorted(i for i, row in records.items()
                          if view == "official" or row["name"] not in scorer.EXCLUDED)
        geometry, exposures, _, coverage, pairs = factorial.fixed_view(
            gt, ids_view, records, metrics, expected_geo, geom, base, modules, truth,
            cached, reference_arms, anchors)
        if anchors is None:
            anchors = pairs
        for arm, expected in earlier["geometry"]["arms"].items():
            factorial.require(geometry["arms"][arm] == expected, "Historical continuous geometry differs")
        modified = {}
        retained = 0
        for iid in ids_view:
            row = records[iid]
            modified[iid] = dict(native=[
                dict(item, segmentation=row["edit"].get(str(rank), {}).get(ARM, {}).get("rle", item["segmentation"]))
                for rank, item in enumerate(row["native"], 1)])
        for aid, (iid, rank) in pairs.items():
            predicted = modified[iid]["native"][rank - 1]["segmentation"]
            ref = factorial.codec.merge(factorial.codec.frPyObjects(annotations[aid]["segmentation"], *predicted["size"]))
            retained += int(factorial.codec.iou([predicted], [ref], [False])[0, 0] >= .50)
        matched, audit = geom.evaluate_raw(gt, modified, ids_view, metrics[ARM])
        coverage[ARM] = dict(original_pairs_retained_50=retained, original_pairs=len(pairs),
                             rematched_hulls_50=len(matched), ground_truth_hulls=audit["gt_instances"])
        contrasts = {}
        for a, b in factorial.PAIRS:
            delta = factorial.difference(metrics[a], metrics[b])
            exposure = exposures[a + "_minus_" + b]
            interval = geometry["pairwise"][a + " minus " + b]["continuous_iou_points"]["percentile95"]
            enough = exposure["different_hulls"] >= 20 and exposure["different_decoded_groups"] >= 8
            contrasts[b] = dict(COCO_points=delta, **exposure, coverage_gate=enough,
                                favorable=bool(enough and delta["AP"] >= .30 and interval[0] > 0),
                                adverse=bool(enough and delta["AP"] <= -.30 and interval[1] < 0))
        views[view] = dict(**result, geometry=geometry, coverage=coverage, contrasts=contrasts)
        print(json.dumps(dict(phase="scored_view", view=view, images=len(ids_view))), flush=True)
    return dict(status=STATUS, phase="score", worker_sha256=factorial.digest(Path(__file__).read_bytes()),
                protocol_sha256=LOCK, parent_score_sha256=parent_sha, parent_terminal_sha256=exit_sha,
                detector_forwards=0, phoenix_forwards=0, optimizer_updates=0, counts=totals, views=views,
                primary_comparator=static, prior_COCO_and_geometry_parity=True, score_seconds=perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("factorial-source", "code", "owner", "capture", "classical-root", "request", "request-terminal",
                 "inference", "inference-terminal", "author-labels", "parent-score", "parent-terminal", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.original_readback = args.masks = args.posteriors = None
    result = score(args)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.mkdir()
    with (args.output / "RESULT_SAFE.json").open("xb") as file:
        if file.write(raw) != len(raw):
            raise OSError("Incomplete safe score")
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
