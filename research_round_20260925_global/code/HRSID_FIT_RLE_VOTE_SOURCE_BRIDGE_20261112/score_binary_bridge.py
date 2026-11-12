"""Measure the exact RLE-only cross-dataset rule on its frozen HRSID source cohort."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np

PROTOCOL = "4c9db9e57fdf8dad545438526662715ec6c74bc26a28ee90b4690fcead16be0b"
PRIOR = "65bf443df93eda98d6772358ecf88e2ab815c5a2ef4fb41718ed554c2328ba85"
STRUCTURE = "30ace2a0fa00f4353a66dd7b10b271f95c6521d94b6aa6699bd7a8e1b2ecee4c"
LEGACY_REPORT = "943ed0c91ed106cf47c51127d5261f2a268cf68ce04449710e48e3367b5d74d7"
LEGACY_EDITS = "ce52f211f0efb7bcddd0582e240ef3f2095cb51fd0297a72f67644777b44dc9d"
EDIT_MODES = ("anchor_count", "vote", "same_sign_counts", "legacy_vote")


def require(ok, why):
    if not ok:
        raise ValueError(why)


def load(path, digest, name):
    content = path.read_bytes()
    require(sha256(content).hexdigest() == digest, "Exact source: " + path.name)
    m = ModuleType(name)
    m.__file__ = str(path)
    sys.modules[name] = m
    exec(compile(content, str(path), "exec"), m.__dict__)
    return m


def two_layers(anchor):
    from scipy.ndimage import binary_dilation, binary_erosion
    neighbors = np.ones((3, 3), dtype=bool)
    outer1 = binary_dilation(anchor, structure=neighbors, border_value=0)
    core1 = binary_erosion(anchor, structure=neighbors, border_value=0)
    outer2 = binary_dilation(outer1, structure=neighbors, border_value=0)
    core2 = binary_erosion(core1, structure=neighbors, border_value=0)
    return outer2 & ~core2, outer1 & ~core1, core2


def choose_binary_edits(anchor, donor1, donor2, cap, legacy=None):
    band, near, core = two_layers(anchor)
    # A strict majority disagrees with the anchor iff both donors disagree with it.
    disagree = band & (donor1 != anchor) & (donor2 != anchor)
    all_ids = np.flatnonzero(disagree)
    selected = all_ids[:cap]
    a = anchor.ravel()
    adds = int((~a[selected]).sum())
    drops = len(selected) - adds

    def per_sign(priority, available, count):
        ids = np.flatnonzero(available)
        return ids[np.lexsort((ids, ~priority.ravel()[ids]))[:count]]

    def flipped(indices):
        out = anchor.copy()
        out.ravel()[indices] ^= True
        return out

    simple = np.concatenate((per_sign(near, band & ~anchor, adds),
                             per_sign(near, band & anchor, drops)))
    result = dict(vote=flipped(selected), anchor_count=flipped(simple))
    require(len(simple) == len(selected), "Anchor control must realize the same counts")
    if legacy is None:
        return result

    used_legacy = legacy ^ anchor
    require(not (used_legacy & ~disagree).any() and int(used_legacy.sum()) == len(selected),
            "Legacy vote used the same cap and majority disagreement support")
    original_adds = int((used_legacy & ~anchor).sum())
    original_drops = len(selected) - original_adds
    mix = np.concatenate((np.flatnonzero(disagree & ~anchor)[:original_adds],
                          np.flatnonzero(disagree & anchor)[:original_drops]))
    result.update(same_sign_counts=flipped(mix), legacy_vote=legacy)
    require(len(mix) == len(selected), "Fixed original sign counts must be feasible")
    for key, expected in (("anchor_count", (adds, drops)), ("same_sign_counts", (original_adds, original_drops))):
        out = result[key]
        require((int((out & ~anchor).sum()), int((~out & anchor).sum())) == expected, "Exact edit counts")
    cap_bound = len(all_ids) > cap
    same = np.array_equal(result["vote"], legacy)
    require(cap_bound or same, "Uncapped source RLE vote must replay legacy hard vote exactly")
    summary = dict(edited_candidates=int(len(selected) > 0), cap_bound=int(cap_bound),
                   cap_bound_mixed_sign=int(cap_bound and np.any(a[all_ids]) and np.any(~a[all_ids])),
                   identical_raw_rule_to_legacy=int(same),
                   different_location_at_same_sign=int(not np.array_equal(result["same_sign_counts"], legacy)),
                   zero_two_pixel_erosion_core=int(not core.any()),
                   add_per_arm=adds, drop_per_arm=drops,
                   legacy_add_per_arm=original_adds, legacy_drop_per_arm=original_drops)
    return result, summary


def encode(codec, mask):
    rle = codec.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def prepare(args, prior, area, structure, codec):
    cap = area.json_value(area.pinned(args.capture / "RESULT_SAFE.json", prior.CAPTURE))
    params = argparse.Namespace(capture=args.capture, capture_sha=prior.CAPTURE, witness=args.witness,
                                witness_sha=prior.WITNESS)
    cases, census, anchor_sha, signal_sha = area.read_admitted_capture(params)
    require(census["mapped_native_candidates"] == 2620 and census["covered_eligible_native_candidates"] == 468,
            "Exact accepted original source population")
    seed_rows = {row["seed"]: area.stream(args.capture / str(row["seed"]) / "CANDIDATES_PRIVATE.jsonl",
                                       row["prediction_sha256"]) for row in cap["arms"]}
    legacy_rows = area.stream(args.legacy_edits, LEGACY_EDITS)
    require(len(legacy_rows) == len(cases) == 36, "Exact native and reference image rosters")
    counts = Counter(dict(associated_candidates=0))
    private = []
    for image_index, (case, old) in enumerate(zip(cases, legacy_rows, strict=True)):
        image_rows = {seed: seed_rows[seed][image_index] for seed in prior.SEEDS}
        keys = ("ordinal", "group", "basename", "rgb_sha256")
        require(len({tuple(item[key] for key in keys) for item in image_rows.values()}) == 1
                and case["basename"] == old["basename"] == image_rows[5202]["basename"],
                "Exact original image identity")
        candidates = {seed: row["candidates"] for seed, row in image_rows.items()}
        reps = {}
        for seed, rows in candidates.items():
            selected, _ = structure.greedy_representatives(structure.iou(rows, rows, codec), .90)
            reps[seed] = [rows[i] for i in selected]
        anchors = [r for r in reps[5202] if 32 <= r["area"] < 1024]
        left = structure.iou(anchors, reps[5201], codec)
        right = structure.iou(anchors, reps[5203], codec)
        matches = structure.first_exclusive(left, right)
        observed = {anchors[i]["rank"]: (reps[5201][j]["rank"], reps[5203][k]["rank"])
                    for i, (j, k) in matches.items()}
        expected = {rank: tuple(row["donor_ranks"][str(s)] for s in (5201, 5203))
                    for rank, row in case["signals"].items()}
        require(observed == expected, "Native cross-domain association must replay exact source triplets")
        variants = {}
        counts["associated_candidates"] += len(matches)
        for i, (j, k) in matches.items():
            a, d1, d2 = anchors[i], reps[5201][j], reps[5203][k]
            anchor, donor1, donor2 = [codec.decode(row["rle"]).astype(bool) for row in (a, d1, d2)]
            key = str(a["rank"])
            old_rle = old["variants"].get(key, {}).get("vote", a["rle"])
            legacy = codec.decode(old_rle).astype(bool)
            require(anchor.shape == donor1.shape == donor2.shape == legacy.shape and anchor.ndim == 2,
                    "Same original mask grid")
            yy, xx = np.nonzero(anchor)
            require(len(yy) == a["area"], "Exact original area")
            h, w = anchor.shape
            window = np.s_[max(0, yy.min()-2):min(h, yy.max()+3),
                           max(0, xx.min()-2):min(w, xx.max()+3)]
            modes, summary = choose_binary_edits(anchor[window], donor1[window], donor2[window],
                                                  case["signals"][a["rank"]]["cap"], legacy[window])
            require(int(np.count_nonzero(legacy ^ anchor)) ==
                    summary["legacy_add_per_arm"] + summary["legacy_drop_per_arm"],
                    "Legacy hard vote had no edits outside common band")
            counts.update(summary)
            if summary["edited_candidates"]:
                replacements = {}
                for mode, mask in modes.items():
                    output = anchor.copy()
                    output[window] = mask
                    replacements[mode] = encode(codec, output)
                replacements["soft"] = replacements["vote"]  # Common scorer's primary slot.
                variants[a["rank"]] = replacements
        private.append(dict(basename=case["basename"], variants=variants))
    require(counts["associated_candidates"] == 468, "Exact source assignment denominator")
    digest = sha256()
    with (args.output / "MASK_EDITS_PRIVATE.jsonl").open("xb") as file:
        for row in private:
            data = (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
            file.write(data)
            digest.update(data)
    receipt = dict(status="HRSID_FIT_RLE_VOTE_SOURCE_BRIDGE_PRELABEL", FIT=True, DEV=False, TEST=False,
                   GT_opened=False, logit_planes_opened=False, original_candidates=2620, original_images=36,
                   anchor_sha256=anchor_sha, associations_sha256=signal_sha, legacy_edits_sha256=LEGACY_EDITS,
                   protocol_sha256=PROTOCOL, variants_sha256=digest.hexdigest(), **counts)
    area.write_new(args.output / "PRELABEL_SAFE.json", receipt)
    return cases, private, receipt


def run(args):
    started = perf_counter()
    require(sha256(args.protocol.read_bytes()).hexdigest() == PROTOCOL, "Exact protocol")
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    from pycocotools import mask as codec
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Exact geometry")

    code = args.code
    prior = load(code / "HRSID_FIT_CONSENSUS_CONTOUR_20261112/score_consensus.py", PRIOR, "_rle_bridge_prior")
    area = load(code / "HRSID_FIT_SEED_AREA_CONFIRM_20261112/score/score_seed_area_confirm.py",
                prior.AREA, "_rle_bridge_area")
    structure = load(code / "HRSID_FIT_ASSOC_STRUCTURE_20261112/score_structure.py", STRUCTURE,
                     "_rle_bridge_structure")
    stock = area.module("_rle_bridge_match", code / "HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py", prior.MATCH)
    endpoint = area.module("_rle_bridge_geometry", code /
                           "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", area.GEOMETRY_SHA)
    geom = endpoint.geometry_modules(code)
    args.output.mkdir()
    cases, private, receipt = prepare(args, prior, area, structure, codec)
    prior.MODES = ("raw", "soft", *EDIT_MODES)
    scored = prior.run_score(args, area, stock, geom, codec, cases, private)
    for field in ("mask_metrics", "fixed_geometry"):
        require(scored[field].pop("soft") == scored[field]["vote"], "Exact primary slot alias")
    legacy = area.json_value(area.pinned(args.legacy_report, LEGACY_REPORT))
    require(scored["mask_metrics"]["raw"] == legacy["mask_metrics"]["raw"] and
            scored["mask_metrics"]["legacy_vote"] == legacy["mask_metrics"]["vote"] and
            scored["fixed_geometry"]["legacy_vote"] == legacy["fixed_geometry"]["vote"],
            "Exact original source raw and legacy reference replay")
    scores = scored["mask_metrics"]
    gates = dict(ap_vs_raw=scores["vote"]["mask_ap"] - scores["raw"]["mask_ap"] >= .50,
                 ap_vs_anchor_control=scores["vote"]["mask_ap"] - scores["anchor_count"]["mask_ap"] >= .30,
                 ap75_retained=scores["vote"]["ap75"] >= scores["raw"]["ap75"] - 1e-12)
    scored.pop("soft_fixed_fold_delta")
    scored.update(gates=gates, advance=all(gates.values()))
    result = dict(receipt, **scored, status="HRSID_FIT_RLE_VOTE_SOURCE_BRIDGE_ACTUAL", GT_opened=True,
                  worker_seconds=perf_counter()-started, model_forwards=0, optimizer_updates=0,
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    area.write_new(args.output / "RESULT_SAFE.json", result)
    print(json.dumps({k: result[k] for k in ("status", "cap_bound", "cap_bound_mixed_sign", "gates", "advance")},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "capture", "witness", "fit-index", "code", "shapely-site",
                 "legacy-edits", "legacy-report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    def offline(event, _args):
        if event.startswith(("socket.", "subprocess.")):
            raise RuntimeError("Offline scientific execution")
    sys.addaudithook(offline)
    run(parser.parse_args())
