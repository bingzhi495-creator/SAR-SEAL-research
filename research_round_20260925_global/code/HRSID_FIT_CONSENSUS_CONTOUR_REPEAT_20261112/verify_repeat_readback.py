"""Independently check sealed repeat invariants without FIT labels or model calls."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from pycocotools import mask as codec

RESULT_SHA = "9b92a8c594eb7f950b4b06959a62637ee9b683ea9808cf4df9ab5c0e4f906784"
PRELABEL_SHA = "21bb47b3ff03448a00d926ec08b8cc11d968bff617bc435dc209bf3874f22855"
SCORE_TERMINAL = "230534d39d5de83de9c3591104654daeb87d6d44ebb4022fcf38e23e9a09cf42"
EDIT_TERMINAL = "fdd1bf8447667fe720b957f6b0df010ec7095000b0e9f56de900c106d135a164"
MODES = ("anchor_count", "vote", "soft")


def load(path, digest, stream=False):
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError("Mismatch reading " + path.name)
    return [json.loads(line) for line in data.splitlines()] if stream else json.loads(data)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def run(args):
    result = load(args.score / "RESULT_SAFE.json", RESULT_SHA)
    prelabel = load(args.prelabel / "PRELABEL_SAFE.json", PRELABEL_SHA)
    for path, expected in ((args.score, SCORE_TERMINAL), (args.prelabel, EDIT_TERMINAL)):
        terminal = load(path / "ORIGINAL_TOOL_TERMINAL_PRIVATE.json", expected)
        check(terminal["exit_code"] == 0 and bool(terminal["chunk_id"]), "Original natural terminal")
    check(prelabel["GT_opened"] is False and prelabel["AP"] is False
          and result["GT_opened"] is True and result["AP"] is True, "Separate label stage")
    check(all(rec["FIT"] and not rec["DEV"] and not rec["TEST"] and not rec["model_forwards"]
              and not rec["optimizer_updates"] for rec in (result, prelabel)), "FIT-only static replay")
    check(result["prelabel_sha256"] == PRELABEL_SHA and result["prelabel_terminal_sha256"] == EDIT_TERMINAL,
          "Result depends on exact prelabel closure")
    anchors = load(args.anchor, prelabel["anchor_sha256"], stream=True)
    edited = load(args.prelabel / "MASK_EDITS_PRIVATE.jsonl", prelabel["variants_sha256"], stream=True)
    original_count = edit_count = added = dropped = 0
    check(len(anchors) == len(edited) == 36, "Exact image roster")
    for base, variants in zip(anchors, edited, strict=True):
        check(base["basename"] == variants["basename"], "Exact image ordering")
        originals = {str(c["rank"]): c for c in base["candidates"]}
        check(len(originals) == len(base["candidates"]) and set(variants["variants"]) <= originals.keys(),
              "Masks apply solely to existing ranks")
        original_count += len(originals)
        for rank, modes in variants["variants"].items():
            original = originals[rank]
            area = int(codec.area(original["rle"]))
            cap = 2 * min(8, area // 10)
            check(32 <= area < 1024 and set(modes) == set(MODES), "Exact eligible edited rank")
            actions = set()
            for current in modes.values():
                check(current["size"] == original["rle"]["size"] == [800, 800], "Exact native grid")
                overlap = int(codec.area(codec.merge([original["rle"], current], intersect=True)))
                actions.add((int(codec.area(current)) - overlap, area - overlap))
            check(len(actions) == 1, "Equal per-candidate budget and area")
            up, down = actions.pop()
            check(0 < up + down <= cap, "Bounded nonzero edit")
            edit_count += 1
            added += up
            dropped += down
    check((original_count, edit_count, added, dropped) ==
          tuple(prelabel[k] for k in ("original_candidates", "edited_candidates", "add_per_arm", "drop_per_arm")),
          "Aggregate count/area closure")
    scope = result["edited_candidate_scope"]
    check(sum(scope[key] for key in ("edited_matches", "edited_unmatched", "edited_neutral")) == edit_count,
          "Scoring candidate scope closure")
    ap = result["mask_metrics"]
    soft = ap["soft"]
    delta = {m: soft["mask_ap"] - ap[m]["mask_ap"] for m in ("raw", *MODES[:-1])}
    positive = {m: sum(val > 0 for val in values) for m, values in result["soft_fixed_fold_delta"].items()}
    gates = dict(ap_vs_raw=delta["raw"] >= .50, ap_vs_anchor_control=delta["anchor_count"] >= .30,
                 ap75_retained=soft["ap75"] >= ap["raw"]["ap75"] - 1e-12,
                 fixed_geometry_fold_signs=all(positive[m] >= 2 for m in ("raw", "anchor_count")))
    check(gates == result["gates"] and all(gates.values()) == result["advance"], "Exact frozen gate recomputation")
    return dict(status="HRSID_FIT_CONTOUR_REPEAT_INDEPENDENT_READBACK", result_sha256=RESULT_SHA,
                prelabel_sha256=PRELABEL_SHA, original_candidates=original_count, edited_candidates=edit_count,
                add_per_arm=added, drop_per_arm=dropped, gates=gates, soft_mask_ap_delta=delta,
                positive_geometry_folds=positive, FIT=True, DEV=False, TEST=False, GT_opened=False,
                model_forwards=0, optimizer_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("anchor", "prelabel", "score", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    result = run(args)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as out:
        out.write(raw)
    print(json.dumps(dict(status=result["status"], readback_sha256=sha256(raw).hexdigest()), sort_keys=True))