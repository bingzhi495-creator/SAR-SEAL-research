"""Frozen HRSID center masks, native-grid RLE-only majority and matched geometry control."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
from statistics import mean
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
from scipy.ndimage import binary_dilation, binary_erosion

PROTOCOL_SHA = "da1ca84f1659688ffcd8be4dd98de8cf010cef0e39b10c1eff76dfc847573400"
PLAN_SHA = "fc7ac31ba4d121f4329a6536ebca2be4f22552c140cf9af47b8f5ae60d8a4256"
INDEPENDENT_SHA = "5c703c813647b6f7d2f3e21d653500a46d26db4b46b59022b2b642ebeb99d15d"
SCORER_SHA = "dc6af86e4c38fdd79499f98f96476e398e376bf5891343a66aa7901412381cac"
EVALUATION_PROTOCOL = "ad2cb6182b37f9db1dc24908220dd5c923a1257c"
CHECKPOINTS = {
    5201: "b95f1a41ab599b7eb27ed3f25029300838cf1b9da74d7bd0c3ccaaef449485db",
    5202: "84402a6cf29a6ae428853d3049a7759f97e342defa31ab719b74667ce932ef3a",
    5203: "e6bdece9f2b8b613eb0963d923aa6bd27d119fe739dc4d44ed85c1934652bb5a",
}
PRELABEL = "HRSID_TO_SSDD_DEV187_RLE_VOTE_PRELABEL"
READBACK = "HRSID_TO_SSDD_DEV187_RLE_VOTE_PRELABEL_READBACK"
RESULT = "HRSID_TO_SSDD_DEV187_RLE_VOTE_ACTUAL"
MODES = ("raw", "anchor_distance", "vote")
STRUCTURE = np.ones((3, 3), dtype=bool)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(path, digest, cap=1024**2):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "Canonical read input")
    size = path.stat().st_size
    require(0 < size <= cap, "Bounded read input")
    raw = path.read_bytes()
    require(len(raw) == size and sha256(raw).hexdigest() == digest, "Frozen digest mismatch: " + path.name)
    return raw


def load_score(code):
    path = code / "HRSID_TO_SSDD_FROZEN_DEV187_20261110/score_frozen_hrsid_ssdd.py"
    raw = pin(path, SCORER_SHA)
    module = ModuleType("_frozen_transfer_score")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def stable_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_new(path, raw):
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    return sha256(raw).hexdigest()


def iou(left, right):
    if not left or not right:
        return np.empty((len(left), len(right)), dtype=np.float64)
    return codec.iou([r[2] for r in left], [r[2] for r in right], [0] * len(right))


def representatives(pool):
    matrix = iou(pool, pool)
    kept = []
    for i in range(len(pool)):
        if not any(matrix[previous, i] >= .90 for previous in kept):
            kept.append(i)
    return kept


def exclusive_pairs(pools):
    reps = {seed: representatives(pool) for seed, pool in pools.items()}
    donors = (5201, 5203)
    used = {s: set() for s in donors}
    matrices = {s: iou(pools[5202], pools[s]) for s in donors}
    for i in reps[5202]:
        if not 32 <= pools[5202][i][1] < 1024:
            continue
        picks = []
        for s in donors:
            choices = [j for j in reps[s] if j not in used[s] and matrices[s][i, j] >= .50]
            if not choices:
                break
            picks.append(max(choices, key=lambda j: matrices[s][i, j]))
        if len(picks) == len(donors):
            for s, j in zip(donors, picks, strict=True):
                used[s].add(j)
            yield i, *picks


def layers(anchor):
    outer_one = binary_dilation(anchor, structure=STRUCTURE, iterations=1, border_value=0)
    inner_one = binary_erosion(anchor, structure=STRUCTURE, iterations=1, border_value=0)
    outer_two = binary_dilation(outer_one, structure=STRUCTURE, iterations=1, border_value=0)
    inner_two = binary_erosion(inner_one, structure=STRUCTURE, iterations=1, border_value=0)
    return outer_two & ~inner_two, outer_one & ~inner_one


def choose_pixels(anchor, donor1, donor2):
    band, near = layers(anchor)
    disagreement = band & (donor1 == donor2) & (donor1 != anchor)
    cap = 2 * min(8, int(anchor.sum()) // 10)
    selected = np.flatnonzero(disagreement)[:cap]
    if not len(selected):
        return None
    drops = int(anchor.ravel()[selected].sum())
    adds = len(selected) - drops
    additions = np.concatenate((np.flatnonzero(band & near & ~anchor), np.flatnonzero(band & ~near & ~anchor)))[:adds]
    deletions = np.concatenate((np.flatnonzero(band & near & anchor), np.flatnonzero(band & ~near & anchor)))[:drops]
    require(len(additions) == adds and len(deletions) == drops, "Same-count anchor support")
    voted, controlled = anchor.copy(order="C"), anchor.copy(order="C")
    voted.ravel()[selected] ^= True
    controlled.ravel()[additions] = True
    controlled.ravel()[deletions] = False
    return voted, controlled, adds, drops


def binary_masks(pools, chosen):
    return [codec.decode(pools[s][r][2]).astype(bool) for s, r in zip((5202, 5201, 5203), chosen, strict=True)]


def encode(mask):
    rle = codec.encode(np.asfortranarray(mask, dtype=np.uint8))
    return dict(size=rle["size"], counts=rle["counts"].decode("ascii"))


def pools_for(blocks):
    pools = {}
    for seed, block in blocks.items():
        rows = [(rank, int(codec.area(rle)), rle)
                for rank, rle in zip(block["native_ranks"], (row["segmentation"] for row in block["coco"]), strict=True)]
        pools[seed] = [row for row in rows if row[1]]
    return pools


def read_inputs(args, original, seeds):
    plan = original.document(pin(args.plan, PLAN_SHA))
    independent = original.document(pin(args.independent, INDEPENDENT_SHA))
    require(plan["status"] == original.PLAN_STATUS and plan["evaluation_protocol"] == EVALUATION_PROTOCOL
            and independent["status"] == "FROZEN_HRSID_TO_SSDD_DEV187_INDEPENDENTLY_ACCEPTED_V1"
            and independent["closed_plan_sha256"] == PLAN_SHA and independent["DEV"] is True
            and independent["official_TEST_opened"] is False and independent["training"] is False
            and independent["target_adaptation"] is False, "Original independent fixed transfer closure")
    manifest = plan["image_manifest"]
    require(manifest["sha256"] == original.IMAGE_MANIFEST_SHA, "Exact original DEV187 image-only manifest")
    scorer, official = original.load_scoring()
    _, ids, items = scorer.fixed_images(scorer.Artifact(manifest, original.read_ref(manifest, original.MIB)))
    require(len(ids) == len(items) == 187 and len({item["image_ref"]["sha256"] for item in items}) == 183,
            "Fixed 187 logical IDs and 183 JPEG identities")
    arms = {}
    for seed in seeds:
        entry, = [row for row in plan["arms"] if row["seed"] == seed and row["mode"] == "center"]
        verified, = [row for row in independent["arms"] if row["seed"] == seed and row["mode"] == "center"]
        _, _, _, _, report = original.admit_entry(entry, EVALUATION_PROTOCOL, manifest["sha256"])
        require(report["device"] == "cpu" and verified["checkpoint_sha256"] == entry["checkpoint_sha256"] ==
                CHECKPOINTS[seed] and all(verified[k + "_sha256"] == entry[k]["sha256"] for k in ("predictions",)),
                "Exact independently accepted source checkpoint and native prediction")
        raw = original.read_ref(entry["predictions"], 64 * original.MIB)
        original.score_rows(raw, items, report)
        arms[seed] = [original.document(line) for line in raw.splitlines()]
    return plan, independent, ids, items, arms, scorer, official


def prepare(args, arms, ids):
    private = []
    totals = Counter()
    for i, image in enumerate(ids):
        pools = pools_for({s: blocks[i] for s, blocks in arms.items()})
        totals["mapped_candidates"] += len(arms[5202][i]["coco"])
        totals["nonempty_candidates"] += len(pools[5202])
        totals["raw_eligible"] += sum(32 <= row[1] < 1024 for row in pools[5202])
        edits = {}
        triplets = []
        for indices in exclusive_pairs(pools):
            totals["associated"] += 1
            a, l, r = indices
            rank, area, _ = pools[5202][a]
            triplets.append([rank, pools[5201][l][0], pools[5203][r][0]])
            output = choose_pixels(*binary_masks(pools, indices))
            if output is None:
                continue
            voted, controlled, adds, drops = output
            require(int(voted.sum()) == int(controlled.sum()) == area + adds - drops, "Exactly matched after-edit area")
            edits[str(rank)] = dict(vote=encode(voted), anchor_distance=encode(controlled), additions=adds, deletions=drops)
            totals["edited"] += 1
            totals["additions"] += adds
            totals["deletions"] += drops
        totals["associated_images"] += bool(triplets)
        totals["edited_images"] += bool(edits)
        private.append(dict(image_id=image, triplets=triplets, variants=edits))
    require(totals["mapped_candidates"] == 18684 and totals["nonempty_candidates"] == 16139, "Exact original anchor counts")
    args.output.mkdir()
    digest = write_new(args.output / "VARIANTS_PRIVATE.jsonl", b"".join(map(stable_json, private)))
    receipt = dict(status=PRELABEL, GT_opened=False, DEV=True, TEST=False, AP=False, model_forwards=0, training=False,
                   plan_sha256=PLAN_SHA, independent_sha256=INDEPENDENT_SHA, protocol_sha256=PROTOCOL_SHA,
                   source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), variants_sha256=digest,
                   images=len(ids), jpeg_identities=183, totals=dict(totals))
    return write_new(args.output / "PRELABEL_SAFE.json", stable_json(receipt))


def read_variants(args):
    raw = pin(args.prelabel / "PRELABEL_SAFE.json", args.prelabel_sha)
    prelabel = json.loads(raw)
    require(prelabel["status"] == PRELABEL and prelabel["GT_opened"] is False and prelabel["AP"] is False
            and prelabel["TEST"] is False and prelabel["plan_sha256"] == PLAN_SHA
            and prelabel["source_sha256"] == sha256(Path(__file__).read_bytes()).hexdigest(), "Exact prelabel receipt")
    stream = pin(args.prelabel / "VARIANTS_PRIVATE.jsonl", prelabel["variants_sha256"], 64 * 1024**2)
    return prelabel, [json.loads(line) for line in stream.splitlines()]


def read_terminal(path, digest, stage, result_sha):
    receipt = json.loads(pin(path, digest))
    require(receipt["stage"] == stage and receipt["original_os_exit"] == 0 and receipt["timed_out"] is False
            and receipt["output_sha256"] == result_sha and receipt["child_closed"] is True,
            "Exact natural prelabel worker terminal")


def verify(args, arms, ids):
    prelabel, variants = read_variants(args)
    read_terminal(args.prelabel_terminal, args.prelabel_terminal_sha, "prepare", args.prelabel_sha)
    totals = Counter()
    for ordinal, (block, variants_block, image) in enumerate(zip(arms[5202], variants, ids, strict=True)):
        require(block["image_id"] == variants_block["image_id"] == image, "Exact image ID sequence")
        original = {str(rank): row["segmentation"] for rank, row in zip(block["native_ranks"], block["coco"], strict=True)}
        totals["mapped_candidates"] += len(original)
        areas = {rank: int(codec.area(rle)) for rank, rle in original.items()}
        totals["nonempty_candidates"] += sum(area > 0 for area in areas.values())
        totals["raw_eligible"] += sum(32 <= area < 1024 for area in areas.values())
        pairs = variants_block["triplets"]
        pools = pools_for({s: blocks[ordinal] for s, blocks in arms.items()})
        rederived = [[pools[s][i][0] for s, i in zip((5202, 5201, 5203), triplet, strict=True)]
                     for triplet in exclusive_pairs(pools)]
        require(pairs == rederived, "Exact prediction-only donor association readback")
        variants = variants_block["variants"]
        require(all(len(set(row[i] for row in pairs)) == len(pairs) for i in range(3)), "No reused seed ranks")
        require(set(variants) <= {str(x[0]) for x in pairs} <= original.keys(), "Exact anchor support and variant ranks")
        totals["associated"] += len(pairs)
        totals["associated_images"] += bool(pairs)
        totals["edited"] += len(variants)
        totals["edited_images"] += bool(variants)
        for rank, changes in variants.items():
            before = original[rank]
            hard = codec.decode(before).astype(bool)
            band, _ = layers(hard)
            area = areas[rank]
            for mode in MODES[1:]:
                after = changes[mode]
                common = int(codec.area(codec.merge([before, after], intersect=True)))
                require(32 <= area < 1024 and before["size"] == after["size"]
                        and (int(codec.area(after)) - common, area - common) ==
                        (changes["additions"], changes["deletions"])
                        and not np.any((codec.decode(after).astype(bool) ^ hard) & ~band), "Exact same-count RLE geometry")
            require(0 < changes["additions"] + changes["deletions"] <= 2 * min(8, area//10), "Exact edit cap")
            totals["additions"] += changes["additions"]
            totals["deletions"] += changes["deletions"]
    require(dict(totals) == prelabel["totals"], "Full prelabel aggregate closure")
    args.output.mkdir()
    receipt = dict(status=READBACK, DEV=True, GT_opened=False, TEST=False, AP=False, model_forwards=0,
                   prelabel_sha256=args.prelabel_sha, prelabel_terminal_sha256=args.prelabel_terminal_sha,
                   variants_sha256=prelabel["variants_sha256"], totals=dict(totals), all_edit_invariants=True)
    return write_new(args.output / "READBACK_SAFE.json", stable_json(receipt))


def score_metrics(data, rows, ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with redirect_stdout(StringIO()):
        gt = COCO()
        gt.dataset = deepcopy(data)
        gt.createIndex()
        dt = gt.loadRes(rows)
        evaluator = COCOeval(gt, dt, "segm")
        evaluator.params.imgIds = list(ids)
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    index = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, .90)).item())
    precision = evaluator.eval["precision"][index, :, 0, 0, 2]
    ix = dict(AP=0, AP50=1, AP75=2, APs=3, APm=4, APl=5, AR100=8)
    metrics = {k: float(evaluator.stats[v]) for k, v in ix.items()}
    metrics["AP90"] = float(precision[precision >= 0].mean())
    return gt, dt, evaluator, metrics


def fixed_geometry(gt, dt, evaluator, rows, variants):
    matches = {}
    neutral = set()
    for image in evaluator.evalImgs[:len(evaluator.params.imgIds)]:
        if image is None:
            continue
        for det, identity, ignored in zip(image["dtIds"], image["dtMatches"][0], image["dtIgnore"][0], strict=True):
            if ignored:
                neutral.add(int(det) - 1)
            elif identity:
                matches[int(det) - 1] = int(identity)
    grouped = Counter()
    overlaps = {m: [] for m in MODES}
    for i, (row, changes) in enumerate(zip(rows, variants, strict=True)):
        if changes:
            grouped["edited_neutral" if i in neutral else "edited_match" if i in matches else "edited_unmatched"] += 1
        if i not in matches or not 32 <= int(codec.area(row["segmentation"])) < 1024:
            continue
        truth = gt.annToRLE(gt.anns[matches[i]])
        for m in MODES:
            predicted = changes.get(m, row["segmentation"])
            overlaps[m].append(float(codec.iou([predicted], [truth], [0])[0, 0]))
    return dict(valid_fixed_matches=len(overlaps["raw"]), iou_points={k: 100*mean(v) for k, v in overlaps.items()},
                iou_90_matches={k: sum(x >= .90 for x in v) for k, v in overlaps.items()}, edited_scope=dict(grouped))


def score(args, original, plan, independent, ids, items, arms, scorer):
    prelabel, variants = read_variants(args)
    read_terminal(args.prelabel_terminal, args.prelabel_terminal_sha, "prepare", args.prelabel_sha)
    checked = json.loads(pin(args.readback, args.readback_sha))
    read_terminal(args.readback_terminal, args.readback_terminal_sha, "verify", args.readback_sha)
    require(checked["status"] == READBACK and checked["GT_opened"] is False and checked["AP"] is False
            and checked["prelabel_sha256"] == args.prelabel_sha and checked["totals"] == prelabel["totals"]
            and checked["variants_sha256"] == prelabel["variants_sha256"] and checked["all_edit_invariants"] is True,
            "Closed independent prediction-only verification before GT")
    rows = {m: [] for m in MODES}
    changed = []
    for raw, block, image in zip(arms[5202], variants, ids, strict=True):
        require(raw["image_id"] == block["image_id"] == image, "Exact native score image join")
        for rank, row in zip(raw["native_ranks"], raw["coco"], strict=True):
            change = block["variants"].get(str(rank), {})
            changed.append(change)
            for mode in MODES:
                rows[mode].append(dict(row, segmentation=change.get(mode, row["segmentation"])))
    require(all(len(v) == 18684 for v in rows.values()), "Exact original raw mapped COCO rows")
    require(sum(bool(c) for c in changed) == prelabel["totals"]["edited"], "Exact immutable edit count")
    gt_ref = plan["integer_gt"]
    require(gt_ref["sha256"] == original.INTEGER_GT_SHA and gt_ref["bytes"] == original.INTEGER_GT_BYTES,
            "Exact P250 integer GT authority")
    data = original.document(original.read_ref(gt_ref, original.MIB))
    scorer.validate_gt(data, ids, items)
    outputs = {m: score_metrics(data, rows[m], ids) for m in MODES}
    raw_metric = outputs["raw"][-1]
    reference, = [arm["metrics"] for arm in independent["arms"] if arm["seed"] == 5202 and arm["mode"] == "center"]
    require(all(abs(raw_metric[key] - reference[key]) <= 1e-12 for key in original.METRICS),
            "Raw AP metrics reproduce independent P250 5202 center")
    metrics = {m: out[-1] for m, out in outputs.items()}
    geometry = fixed_geometry(*outputs["raw"][:3], rows["raw"], changed)
    require(sum(geometry["edited_scope"].values()) == prelabel["totals"]["edited"], "Exact fixed edit/match partition")
    delta = {m: {k: 100*(metrics["vote"][k]-metrics[m][k]) for k in original.METRICS} for m in MODES[:2]}
    gates = dict(ap_vs_raw=delta["raw"]["AP"] >= .50, ap_vs_anchor_distance=delta["anchor_distance"]["AP"] >= .30,
                 ap75_retained=delta["raw"]["AP75"] >= -1e-10)
    result = dict(prelabel, status=RESULT, GT_opened=True, AP=True, max_dets=[1,10,100], units="fraction",
                  ground_truth="P250_INTEGER_PRIMARY", gt_sha256=gt_ref["sha256"], float_sensitivity=False,
                  prelabel_sha256=args.prelabel_sha, readback_sha256=args.readback_sha,
                  raw_witness_reproduced=True, metrics=metrics, vote_deltas_points=delta, gates=gates,
                  advance=all(gates.values()), fixed_binary_geometry=geometry)
    args.output.mkdir()
    return write_new(args.output / "RESULT_SAFE.json", stable_json(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    common = ("protocol", "plan", "independent", "code", "output")
    prelabel = ("prelabel", "prelabel-terminal")
    for name, extra in (("prepare", ()), ("verify", prelabel),
                        ("score", prelabel + ("readback", "readback-terminal"))):
        stage = stages.add_parser(name)
        for field in common + extra:
            stage.add_argument("--" + field, type=Path, required=True)
        for field in extra:
            stage.add_argument("--" + field + "-sha", required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and os.environ.get("OMP_NUM_THREADS") == "1",
            "CPU-only single-thread entry")
    require(version("pycocotools") == "2.0.11", "Exact original stock COCO 2.0.11")
    pin(args.protocol, PROTOCOL_SHA)
    original = load_score(args.code)
    plan, independent, ids, items, arms, scorer, official = read_inputs(
        args, original, (5202,) if args.stage == "score" else CHECKPOINTS)
    if args.stage == "prepare":
        digest = prepare(args, arms, ids)
    elif args.stage == "verify":
        digest = verify(args, arms, ids)
    else:
        digest = score(args, original, plan, independent, ids, items, arms, scorer)
    print(json.dumps(dict(stage=args.stage, output_sha256=digest), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()