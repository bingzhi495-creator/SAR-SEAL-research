"""Cross frozen Phoenix and learned addition/removal locations on Instance-RSDD's 74-image reserve."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec


LOCK = "f0a154154a18f05d3ff45f42fc6dddfc233c4ef595a9a86534e18d9b15e29ed9"
PHX_WORKER = "675d84109e4b7c643be1d3724c65d8d088a44c5d56de60c51cf063f0ed6ee76f"
PHX_RESULT = "fe94442d5e58c5b2568269f9cefdc5dbb86e307da43cde6b5368db09298786ec"
PHX_TERM = "6684f7d858e40fbe78bc631baa153cf5a011461104df4348bdba4b48b8c89770"
PHX_EDITS = "0cdeabbaa13c8f215be22591ca3ab878013fa54694113d9e4f83285bf5c0292b"
BUILDER = "da5ff070df83b3f4ab190baf494a71daee1f1fb9eff335aa255c2a20984b5021"
INPUTS = "2c3b84f852bb4f8234d935347708a2e71eba4466d0eb111471bd254aa2f45e58"
SIDE = "aa590649fb2fa0a70d7fd239b26e9f865efac2d8c0dfe86c4525933f6720f214"
SEEDS = (5217, 5218, 5219)
LEARNED = tuple(f"{s}/learned" for s in SEEDS)
CFA_ARMS = ("dynamic", "static", *LEARNED)
PHX_ARMS = (*CFA_ARMS, "phoenix")
HYBRIDS = tuple(f"{s}/{arm}" for s in SEEDS for arm in ("LP", "PL"))
STATUS = "RSDD74_PHOENIX_SIGNED_SIDE_EDITS_COMPLETE"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Pinned source differs for " + path.name)
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), mod.__dict__)
    return mod


def bitmap(rle):
    require(isinstance(rle, dict) and set(rle) == {"counts", "size"} and rle["size"] == [512, 512]
            and isinstance(rle["counts"], str) and rle["counts"], "Original 512-square COCO mask required")
    mask = codec.decode(rle).astype(bool)
    encoded = codec.encode(np.asfortranarray(mask, dtype=np.uint8))
    require(encoded["counts"].decode("ascii") == rle["counts"] and encoded["size"] == rle["size"],
            "Canonical RLE reconstruction differs")
    return mask


def cross_candidate(native, area, archived, side, seed_counts):
    require(set(archived) == set(PHX_ARMS), "Missing or unexpected inherited arm")
    stock = bitmap(native)
    k = archived["phoenix"]["swaps"]
    require(type(k) is int and 32 <= area < 1024 and int(stock.sum()) == area
            and 0 < k <= min(8, area // 10), "Invalid shared signed budget or native area")
    locations = {}
    for arm, entry in archived.items():
        require(set(entry) == {"rle", "swaps"} and type(entry["swaps"]) is int and entry["swaps"] == k,
                "All six source arms require identical signed counts")
        mask = bitmap(entry["rle"])
        adds = np.flatnonzero(mask & ~stock)
        drops = np.flatnonzero(stock & ~mask)
        require(len(adds) == len(drops) == k, "Source mask violates its native-relative quota")
        locations[arm] = (adds, drops)

    expanded = dict(archived)
    pa, pd = locations["phoenix"]
    for seed, learned_arm in zip(SEEDS, LEARNED, strict=True):
        la, ld = locations[learned_arm]
        # The established crossing function takes its first source as F (here P)
        # and its second as I (here L). IF is learned additions and Phoenix removals.
        crossed, displaced_a, displaced_d = side.cross_masks(native, archived["phoenix"], archived[learned_arm], area)
        require(crossed["FF"]["rle"] == archived["phoenix"]["rle"] and crossed["II"]["rle"] == archived[learned_arm]["rle"],
                "Historical pure Phoenix or learned mask changed")
        for label, original_name, add, drop in (("LP", "IF", la, pd), ("PL", "FI", pa, ld)):
            result = crossed[original_name]
            mask = bitmap(result["rle"])
            # Verify the compressed set algebra independently on native pixel locations.
            require(result["adds"] == result["drops"] == k and np.array_equal(np.flatnonzero(mask & ~stock), add)
                    and np.array_equal(np.flatnonzero(stock & ~mask), drop), "Hybrid changed the prescribed signed positions")
            expanded[f"{seed}/{label}"] = dict(rle=result["rle"], swaps=k)
        require(k - np.intersect1d(pa, la).size == displaced_a and k - np.intersect1d(pd, ld).size == displaced_d,
                "Signed displacement counts differ")
        seed_counts[str(seed)].update(positive_budget_candidates=1, swaps_per_arm=k, adds_displaced_pixels=displaced_a,
                drops_displaced_pixels=displaced_d, adds_differ_candidates=int(displaced_a > 0),
                drops_differ_candidates=int(displaced_d > 0), both_sides_differ_candidates=int(displaced_a > 0 and displaced_d > 0))
    require(set(expanded) == set((*PHX_ARMS, *HYBRIDS)), "Incomplete factorial candidate")
    return expanded


def projected(row, arms):
    return dict(image_id=row["image_id"], changes={
        pos: {arm: choices[arm] for arm in arms} for pos, choices in row["changes"].items()})


def expand_image(row, native_row, side, seed_counts, census):
    item, audit, pred, _, support = native_row
    image_id = item["image_id"]
    require(set(row) == {"image_id", "changes"} and isinstance(row["changes"], dict)
            and row["image_id"] == pred["image_id"] == audit["image_id"] == image_id and support is None,
            "Unexpected image identity, historical support or ledger order")
    require(all(key.isascii() and key.isdecimal() and str(int(key)) == key and 1 <= int(key) <= len(pred["coco"])
                for key in row["changes"]), "Unknown compact candidate position")
    changes = {}
    for pos, (cand, det) in enumerate(zip(audit["candidates"], pred["coco"], strict=True), 1):
        native = det["segmentation"]
        area = int(codec.area(native))
        require(area == cand["mask_area"] and cand["position"] == pos and det["image_id"] == image_id
                and det["score"] == cand["score"] and native["size"] == [512, 512], "Native identity, confidence or area differs")
        census["mapped_candidates"] += 1
        census["eligible_planes"] += int(32 <= area < 1024)
        key = str(pos)
        if key not in row["changes"]:
            census["zero_budget_candidates"] += 1
            continue
        original = row["changes"][key]
        changes[key] = cross_candidate(native, area, original, side, seed_counts)
        census.update(positive_budget_candidates=1, swaps_per_arm=original["phoenix"]["swaps"])
    for tally in seed_counts.values():
        tally["mapped_candidates"] += len(pred["coco"])
    census["images"] += 1
    return dict(image_id=image_id, changes=changes)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("code", "stock-root", "point-root", "phx-edits", "phx-edits-terminal", "protocol", "output"):
        p.add_argument("--"+key, required=True, type=Path)
    p.add_argument("--protocol-sha256", required=True)
    a = p.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve(strict=key != "output"))
    require(not a.output.exists() and sys.flags.isolated and sys.dont_write_bytecode
            and sha256(a.protocol.read_bytes()).hexdigest() == a.protocol_sha256 == LOCK,
            "Fresh isolated output and exact experiment lock required")
    runtime = {k: version(k) for k in ("numpy", "pycocotools")}
    require(runtime == {"numpy": "1.26.4", "pycocotools": "2.0.11"}, "Expected original native mask runtime")
    start = perf_counter()
    a.cohort, a.support_root = 74, None
    api = source(a.code / "INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/build_request.py", BUILDER)
    side = source(a.code / "INDEPENDENT_SIDE_FACTORIAL_20261123/replay_sides.py", SIDE)
    receipt, previous_sha, terminal_sha = api.observed(a.phx_edits, a.phx_edits_terminal, "edits",
             "PHOENIX_RSDD74_EQUAL_COUNT_EDITS_COMPLETE", PHX_RESULT, PHX_TERM, PHX_WORKER)
    require(receipt["phase"] == "edits" and receipt["protocol_sha256"] == api.LOCK and receipt["edits_sha256"] == PHX_EDITS
            and receipt["source_replay"]["edits_sha256"] == api.CFA_EDITS and receipt["target_labels_opened"] is False
            and receipt["independent_ranking_exact"] is receipt["archived_arms_exact"] is True
            and receipt["detector_forwards"] == receipt["optimizer_updates"] == 0
            and all(receipt["counts"][k] == v for k, v in api.TOTALS.items()), "Required previous complete endpoint readback")
    loader = source(a.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123/inputs.py", INPUTS)
    rows, lineage = loader.load_population(a)
    require(lineage == receipt["input_lineage"], "Current and previous native population lineage differ")
    original = api.pin(a.phx_edits / "EDITS_PRIVATE.jsonl", PHX_EDITS)
    seed_counts = {str(seed): Counter(dict.fromkeys(side.COUNT_FIELDS, 0)) for seed in SEEDS}
    census, edited_sha, pure_sha, cfa_sha = Counter(), sha256(), sha256(), sha256()
    a.output.mkdir()
    with (a.output / "EDITS_PRIVATE.jsonl").open("xb") as output:
        for ordinal, (line, native) in enumerate(zip(original.splitlines(keepends=True), rows, strict=True), 1):
            inherited = json.loads(line)
            new = expand_image(inherited, native, side, seed_counts, census)
            legacy = api.encode(projected(new, PHX_ARMS))
            require(legacy == line, "Hybrid projection did not reproduce the complete original image edit bytes")
            pure_sha.update(legacy)
            cfa_sha.update(api.encode(projected(new, CFA_ARMS)))
            data = api.encode(new)
            require(output.write(data) == len(data), "Incomplete output row")
            edited_sha.update(data)
            if ordinal in (1, 25, 50, 74):
                print(json.dumps(dict(phase="edits", images=ordinal, positive_budget_candidates=census["positive_budget_candidates"])), flush=True)
    require(all(census[k] == v for k, v in api.TOTALS.items()) and census["zero_budget_candidates"] == 3239
            and pure_sha.hexdigest() == PHX_EDITS and cfa_sha.hexdigest() == api.CFA_EDITS, "Historical population or byte projection differs")
    require(all(all(tally[k] == census[k] for k in ("mapped_candidates", "positive_budget_candidates", "swaps_per_arm"))
                for tally in seed_counts.values()), "Per-seed budget census differs")
    report = dict(status=STATUS, phase="edits", cohort=74, images=census["images"], seeds=list(SEEDS), arms=list((*PHX_ARMS, *HYBRIDS)),
                  counts=census, counts_by_seed=seed_counts, input_lineage=lineage, protocol_sha256=LOCK,
                  phx_edits_result_sha256=previous_sha, phx_edits_terminal_sha256=terminal_sha, phx_edits_sha256=PHX_EDITS,
                  cfa_edits_sha256=cfa_sha.hexdigest(), edits_sha256=edited_sha.hexdigest(), original_arms_exact=True,
                  per_candidate_signed_counts_exact=True, cross_context_signed_xor_exact=True, exact_native_fallback=True,
                  detector_forwards=0, point_head_forwards=0, phoenix_forwards=0, model_forwards=0, optimizer_updates=0,
                  target_labels_opened=False, GT_opened=False, operator_versions=runtime, seconds=perf_counter()-start,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    sha = api.write(a.output / "RESULT_SAFE.json", api.encode(report))
    print(json.dumps(dict(status=STATUS, result_sha256=sha)), flush=True)


if __name__ == "__main__":
    main()
