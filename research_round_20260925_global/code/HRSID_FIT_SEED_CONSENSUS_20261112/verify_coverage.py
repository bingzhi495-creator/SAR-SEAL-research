"""Independently reconstruct closed FIT seed-association coverage from RLE."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


SEEDS = (5201, 5202, 5203)
DONORS = (5201, 5203)
SHAPE = (800, 800)
CATEGORIES = ("no_donor", "below_iou", "not_reciprocal", "ambiguous", "accepted")


def require(valid, message):
    if not valid:
        raise ValueError(message)


def foreground_indices(rle, expected_shape=SHAPE):
    """Decode COCO compressed counts into ascending Fortran-order indices."""
    require(tuple(rle["size"]) == expected_shape, "Unexpected RLE dimensions")
    counts = []
    bit = 0
    run = 0
    for character in rle["counts"]:
        symbol = ord(character) - 48
        require(0 <= symbol <= 63, "Invalid compressed RLE symbol")
        run |= (symbol & 31) << bit
        bit += 5
        require(bit <= 40, "Compressed RLE integer is too wide")
        if symbol & 32:
            continue
        if symbol & 16:
            run -= 1 << bit
        if len(counts) >= 3:
            run += counts[-2]
        require(run >= 0, "Negative RLE run")
        counts.append(run)
        run, bit = 0, 0
    require(bit == 0 and sum(counts) == expected_shape[0] * expected_shape[1],
            "RLE does not cover exactly one image")
    bounds = np.cumsum([0, *counts], dtype=np.int64)
    starts = bounds[1:-1:2]
    ends = bounds[2::2]
    if not len(starts):
        return np.empty(0, dtype=np.int64)
    return np.concatenate([np.arange(start, end, dtype=np.int64) for start, end in zip(starts, ends, strict=True)])


def mask_matrix(candidates, shape=SHAPE):
    pixels = [foreground_indices(c["rle"], shape) for c in candidates]
    areas = np.array([len(p) for p in pixels], dtype=np.int64)
    require(all(a == c["area"] and a > 0 for a, c in zip(areas, candidates, strict=True)),
            "RLE area does not match its nonempty candidate")
    columns = np.concatenate(pixels) if pixels else np.empty(0, dtype=np.int64)
    offsets = np.cumsum(np.r_[0, areas])
    masks = csr_matrix((np.ones(len(columns), dtype=np.int64), columns, offsets),
                       shape=(len(candidates), shape[0] * shape[1]))
    return masks, areas


def association(anchor, donor, shape=SHAPE):
    """Return per-anchor decisions and donor indices before area eligibility."""
    if not donor:
        return [("no_donor", None)] * len(anchor)
    if not anchor:
        return []
    a_mask, a_area = mask_matrix(anchor, shape)
    d_mask, d_area = mask_matrix(donor, shape)
    overlap = (a_mask @ d_mask.T).toarray()
    union = a_area[:, None] + d_area[None, :] - overlap
    iou = overlap / union

    donor_for_anchor = np.argsort(-iou, axis=1, kind="stable")
    anchor_for_donor = np.argsort(-iou, axis=0, kind="stable")
    decisions = []
    for i, (j, *rest) in enumerate(donor_for_anchor):
        if 2 * overlap[i, j] < union[i, j]:
            decisions.append(("below_iou", None))
            continue
        if i != anchor_for_donor[0, j]:
            decisions.append(("not_reciprocal", None))
            continue
        row_second = iou[i, rest[0]] if len(donor) > 1 else 0.
        col_second = iou[anchor_for_donor[1, j], j] if len(anchor) > 1 else 0.
        if min(iou[i, j] - row_second, iou[i, j] - col_second) + 1e-12 < .10:
            decisions.append(("ambiguous", None))
            continue
        decisions.append(("accepted", int(j)))
    return decisions


def read_artifact(path, expected_sha=None, limit=128 * 2**20):
    require(path.is_file() and 0 < path.stat().st_size <= limit, "Invalid artifact size or path")
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    require(expected_sha is None or digest == expected_sha, "Unexpected artifact digest: " + path.name)
    return raw, digest


def json_lines(raw):
    return [json.loads(line) for line in raw.splitlines()]


def check_arm(seed_dir, arm, check_logits):
    raw, _ = read_artifact(seed_dir / "CANDIDATES_PRIVATE.jsonl", arm["prediction_sha256"])
    blocks = json_lines(raw)
    require(len(blocks) == arm["images"] == arm["model_forwards"] == 36, "Image/forward census changed")
    require(arm["raw_queries"] == 164 * len(blocks) and arm["native_rows"] == 100 * len(blocks),
            "Query or native-row census changed")
    counts = Counter()
    for ordinal, block in enumerate(blocks, start=1):
        require(block["ordinal"] == ordinal and 0 <= block["mapped"] <= 100, "Image order or mapped count changed")
        candidates = block["candidates"]
        require(len(candidates) <= block["mapped"], "Nonempty count exceeds mapped count")
        ranks = [c["rank"] for c in candidates]
        require(ranks == sorted(set(ranks)) and len({c["query"] for c in candidates}) == len(candidates),
                "Candidate rank or query is not unique and stable")
        for candidate in candidates:
            foreground = foreground_indices(candidate["rle"])
            require(len(foreground) == candidate["area"] and len(foreground) > 0,
                    "Candidate area differs from its RLE")
        logits = [c["logit_row"] for c in candidates if c["logit_row"] is not None]
        require(logits == list(range(len(logits))), "Logit row indexing differs from candidate order")
        require(all((c["logit_row"] is not None) == (16 <= c["area"] < 2048) for c in candidates),
                "Logit support census changed")
        logit_file = block["logit"]
        require(logit_file["rows"] == len(logits), "Logit row count mismatch")
        if check_logits:
            filename = logit_file["filename"]
            require(Path(filename).name == filename, "Logit filename escapes seed directory")
            raw_logit, _ = read_artifact(seed_dir / filename, logit_file["sha256"])
            require(len(raw_logit) == logit_file["bytes"], "Logit byte count mismatch")
        counts.update(mapped=block["mapped"], nonempty=len(candidates),
                      raw_logit_planes=len(logits), logit_bytes=logit_file["bytes"])
    require(all(counts[key] == arm[key] for key in ("mapped", "nonempty", "raw_logit_planes", "logit_bytes")),
            "Arm aggregate differs from saved candidates")
    require(len({block["group"] for block in blocks}) == len(blocks), "Duplicated source group")
    return blocks


def reconstruct(arms):
    per_donor = {seed: Counter({key: 0 for key in CATEGORIES}) for seed in DONORS}
    offsets = Counter({"add": 0, "drop": 0, "zero": 0})
    eligible = 0
    covered_groups, nonzero_groups = set(), set()
    output = []
    for blocks in zip(*(arms[seed] for seed in SEEDS), strict=True):
        group, anchor_block = blocks[1]["group"], blocks[1]
        require(all(all(b[key] == anchor_block[key] for key in ("ordinal", "group", "basename", "rgb_sha256"))
                    for b in blocks), "Seed image identities differ")
        candidates = {seed: b["candidates"] for seed, b in zip(SEEDS, blocks, strict=True)}
        anchor = candidates[5202]
        associations = {seed: association(anchor, candidates[seed]) for seed in DONORS}
        matches = []
        for i, a in enumerate(anchor):
            if not 32 <= a["area"] < 1024:
                continue
            eligible += 1
            for seed in DONORS:
                per_donor[seed][associations[seed][i][0]] += 1
            donor_ids = {seed: associations[seed][i][1] for seed in DONORS}
            if None in donor_ids.values():
                continue
            donor_matches = {seed: candidates[seed][j] for seed, j in donor_ids.items()}
            area_change = sorted([a["area"], *(d["area"] for d in donor_matches.values())])[1] - a["area"]
            offsets["add" if area_change > 0 else "drop" if area_change < 0 else "zero"] += 1
            covered_groups.add(group)
            if area_change:
                nonzero_groups.add(group)
            matches.append({
                "anchor_rank": a["rank"],
                "donor_ranks": {str(seed): d["rank"] for seed, d in donor_matches.items()},
                "median_area_offset": area_change,
            })
        output.append({"ordinal": anchor_block["ordinal"], "matches": matches})

    covered = sum(offsets.values())
    nonzero = offsets["add"] + offsets["drop"]
    passed = covered >= 60 and len(covered_groups) >= 20 and nonzero >= 30 and len(nonzero_groups) >= 15
    coverage = {
        "eligible": eligible,
        "covered_both": covered,
        "covered_groups": len(covered_groups),
        "nonzero_median_area_offset": nonzero,
        "nonzero_groups": len(nonzero_groups),
        "offsets": dict(offsets),
        "per_donor": {str(seed): dict(counts) for seed, counts in per_donor.items()},
        "gate": "COVERAGE_PASS" if passed else "COVERAGE_STOP",
    }
    return output, coverage


def main(args):
    raw, digest = read_artifact(args.closed_stage / "RESULT_SAFE.json", args.result_sha256)
    safe = json.loads(raw)
    arm_records = {int(arm["seed"]): arm for arm in safe["arms"]}
    require(len(safe["arms"]) == 3 and set(arm_records) == set(SEEDS), "Unexpected seed cohort")
    arms = {seed: check_arm(args.closed_stage / str(seed), arm_records[seed], args.check_logit_hashes)
            for seed in SEEDS}
    assoc_raw, assoc_digest = read_artifact(args.closed_stage / "ASSOCIATIONS_PRIVATE.jsonl",
                                            safe["coverage"]["associations_sha256"])
    associations, coverage = reconstruct(arms)
    require(associations == json_lines(assoc_raw), "Independent association identities or area offsets differ")
    recorded = safe["coverage"]
    require(all(recorded[key] == value for key, value in coverage.items() if key != "per_donor"),
            "Independent coverage aggregate or gate differs")
    require(set(recorded["per_donor"]) == set(coverage["per_donor"]), "Unexpected donor keys")
    for seed, counts in coverage["per_donor"].items():
        original_counts = recorded["per_donor"][seed]
        require(set(original_counts) <= set(CATEGORIES)
                and all(original_counts.get(key, 0) == value for key, value in counts.items()),
                "Independent per-donor category counts differ")
    encoded = (json.dumps({
        "status": "CLOSED_SEED_CONSENSUS_COVERAGE_INDEPENDENT_READBACK",
        "result_sha256": digest,
        "coverage_verified": True,
        "associations_sha256": assoc_digest,
        "coverage": coverage,
        "images": len(associations),
        "logit_file_hashes_checked": args.check_logit_hashes,
        "labels_opened": False,
        "models_executed": False,
    }, sort_keys=True) + "\n").encode("utf-8")
    if args.safe_output is not None:
        with args.safe_output.open("xb") as f:
            f.write(encoded)
    print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--closed-stage", type=Path, required=True)
    parser.add_argument("--result-sha256", required=True, help="Hash from independently accepted original completion")
    parser.add_argument("--check-logit-hashes", action="store_true")
    parser.add_argument("--safe-output", type=Path)
    main(parser.parse_args())


