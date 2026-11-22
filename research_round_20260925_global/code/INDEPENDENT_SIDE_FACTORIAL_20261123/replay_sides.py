"""Cross the archived addition and deletion locations at the original per-instance counts."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from types import ModuleType

import numpy as np
from pycocotools import mask as codec


STATUS = "INDEPENDENT_SIDE_FACTORIAL_REPLAY_COMPLETE"
PROTOCOL = "fa939f911890ec204a1e94018e3e92066871098d9b2b0745aee6348c2a62620d"
INPUTS = "ae600c98bcb353a1a2b4dce4a4b5b2aa9edca46b15a7ddb68fe50ec5b549ff0f"
ARMS = ("FF", "IF", "FI", "II")
COUNT_FIELDS = ("mapped_candidates", "positive_budget_candidates", "swaps_per_arm",
                "adds_differ_candidates", "drops_differ_candidates", "both_sides_differ_candidates",
                "adds_displaced_pixels", "drops_displaced_pixels")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def inputs_module():
    path = Path(__file__).with_name("side_inputs.py")
    source = path.read_bytes()
    require(sha256(source).hexdigest() == INPUTS, "Input reader digest mismatch")
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def as_json(rle):
    return {"size": rle["size"], "counts": rle["counts"].decode("ascii")}


def intersection(left, right):
    return codec.merge([left, right], intersect=True)


def cross_masks(native, fixed, independent, area):
    """Render all four arms, verifying exact per-sign membership in native coordinates."""
    require(set(fixed) == set(independent) == {"swaps", "rle"}, "Old edit fields differ")
    k = fixed["swaps"]
    require(type(k) is int and type(independent["swaps"]) is int and k == independent["swaps"]
            and 32 <= area < 1024 and 1 <= k <= min(8, area // 10)
            and int(codec.area(native)) == area, "Original area or matched edit budget differs")
    # Keep all production set operations in compressed COCO masks except this one complement.
    outside = codec.encode(np.asfortranarray(1 - codec.decode(native), dtype=np.uint8))
    before = {"F": fixed["rle"], "I": independent["rle"]}
    additions, retained = {}, {}
    for letter, source in before.items():
        require(set(source) == {"size", "counts"} and source["size"] == native["size"]
                and isinstance(source["counts"], str), "Original RLE grid or representation differs")
        additions[letter] = intersection(outside, source)
        retained[letter] = intersection(native, source)
        require(int(codec.area(additions[letter])) == k and int(codec.area(retained[letter])) == area - k,
                "Actual additions or removals differ from the original matched count")

    arms = {}
    for key in ARMS:
        plus, keep = additions[key[0]], retained[key[1]]
        combined = codec.merge([plus, keep])
        require(intersection(outside, combined) == plus and intersection(native, combined) == keep
                and int(codec.area(combined)) == area, "Crossed mask has incorrect signed locations or area")
        mask = as_json(combined)
        if key[0] == key[1]:
            require(mask == before[key[0]], "Reconstructed original arm differs in canonical RLE")
        arms[key] = {"rle": mask, "adds": k, "drops": k}
    # Equal-size sides: the number displaced is half their symmetric difference.
    displaced_adds = k - int(codec.area(intersection(additions["F"], additions["I"])))
    displaced_drops = area - k - int(codec.area(intersection(retained["F"], retained["I"])))
    require(0 <= displaced_adds <= k and 0 <= displaced_drops <= k, "Displacement accounting differs")
    return arms, displaced_adds, displaced_drops


def image_edits(row, counts):
    item, audit, prediction, history = row
    changes = {}
    for seed, original in history.items():
        tally = counts[seed]
        tally["mapped_candidates"] += len(prediction["coco"])
        for key, old_options in original["changes"].items():
            fixed = old_options.get("fixed")
            independent = old_options.get("free_matched_count")
            require((fixed is None) == (independent is None), "Matched arms differ at a zero budget")
            if fixed is None:
                continue
            position = int(key) - 1
            native = prediction["coco"][position]["segmentation"]
            area = audit["candidates"][position]["mask_area"]
            arms, da, dd = cross_masks(native, fixed, independent, area)
            choices = changes.setdefault(key, {})
            for arm, choice in arms.items():
                choices[seed + "/" + arm] = choice
            tally.update(positive_budget_candidates=1, swaps_per_arm=fixed["swaps"], adds_displaced_pixels=da,
                         drops_displaced_pixels=dd, adds_differ_candidates=int(da > 0),
                         drops_differ_candidates=int(dd > 0), both_sides_differ_candidates=int(da > 0 and dd > 0))
    return {"image_id": item["image_id"], "changes": changes}


def main():
    source = inputs_module()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    source.add_input_arguments(parser)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sha256(args.protocol.read_bytes()).hexdigest() == args.protocol_sha256 == PROTOCOL,
            "Prelocked protocol fingerprint differs")
    runtime = {k: version(k) for k in ("numpy", "pycocotools")}
    require(runtime == {"numpy": "1.26.4", "pycocotools": "2.0.11"}, "Use the qualified B5 environment")
    require(not args.output.exists(), "Supply a fresh output directory")
    rows, lineage = source.load_population(args)
    seeds = source.SEEDS[args.cohort]
    counts = {str(seed): Counter(dict.fromkeys(COUNT_FIELDS, 0)) for seed in seeds}
    args.output.mkdir()
    digest = sha256()
    with (args.output / "EDITS_PRIVATE.jsonl").open("xb") as stream:
        for ordinal, row in enumerate(rows, 1):
            line = encoded(image_edits(row, counts))
            stream.write(line)
            digest.update(line)
            if ordinal % 100 == 0:
                print(json.dumps({"phase": "replay", "images": ordinal, "total": args.cohort}), flush=True)
    for key, tally in counts.items():
        require(tally["swaps_per_arm"] == lineage["expected_swaps"][key]
                and tally["positive_budget_candidates"] == lineage["expected_positive_budget_candidates"][key]
                and tally["mapped_candidates"] == lineage["mapped_candidates"], "Full-population count mismatch")
    report = dict(status=STATUS, phase="replay", cohort=args.cohort, images=len(rows), arms=list(ARMS), seeds=list(seeds),
                  input_lineage=lineage, protocol_sha256=PROTOCOL, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_reader_sha256=INPUTS, edits_sha256=digest.hexdigest(), counts_by_seed=counts,
                  original_arms_exact=True, per_candidate_signed_counts_exact=True, cross_context_signed_xor_exact=True,
                  model_forwards=0, optimizer_updates=0, target_labels_opened=False, GT_opened=False, operator_versions=runtime)
    payload = encoded(report)
    with (args.output / "RESULT_SAFE.json").open("xb") as stream:
        stream.write(payload)
    print(json.dumps({"status": STATUS, "result_sha256": sha256(payload).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
