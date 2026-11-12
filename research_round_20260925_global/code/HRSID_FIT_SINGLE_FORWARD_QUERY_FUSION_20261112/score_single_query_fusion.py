"""Evaluate ordinary same-forward query fusion on the fixed FIT36 candidate roster."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np

PROTOCOL = "756d9d5101d6aab12bf898ac36ef3655149cf30ef5dfaa54f2be17096ada29ea"
PREVIOUS_SOURCE = "65bf443df93eda98d6772358ecf88e2ab815c5a2ef4fb41718ed554c2328ba85"
THREE_FORWARD = "943ed0c91ed106cf47c51127d5261f2a268cf68ce04449710e48e3367b5d74d7"


def require(ok, why):
    if not ok:
        raise ValueError(why)


def module(path, digest, name):
    text = path.read_bytes()
    require(sha256(text).hexdigest() == digest, "Exact source digest: " + path.name)
    instance = ModuleType(name)
    instance.__file__ = str(path)
    sys.modules[name] = instance
    exec(compile(text, str(path), "exec"), instance.__dict__)
    return instance


def suppression_groups(overlaps):
    used = np.zeros(len(overlaps), dtype=bool)
    groups = []
    for index, row in enumerate(overlaps):
        if used[index]:
            continue
        used[index] = True
        children = np.flatnonzero((row >= .85) & ~used)
        used[children] = True
        groups.append((index, children.tolist()))
    return groups


def read_carrier(args, prior, area):
    captured = area.json_value(area.pinned(args.capture / "RESULT_SAFE.json", prior.CAPTURE))
    accepted = area.json_value(area.pinned(args.witness, prior.WITNESS))
    require(area.fit_only(captured) and area.fit_only(accepted)
            and captured["protocol_sha256"] == area.CAPTURE_PROTOCOL_SHA
            and captured["images"] == 36 and accepted["status"] == area.ACCEPTED_STATUS
            and accepted["signals_verified"] is True and accepted["result_sha256"] == prior.CAPTURE,
            "Exact closed and independently accepted carrier")
    anchor = next(arm for arm in captured["arms"] if arm["seed"] == 5202)
    blocks = area.stream(args.capture / "5202/CANDIDATES_PRIVATE.jsonl", anchor["prediction_sha256"])
    require(len(blocks) == 36 and len({b["basename"] for b in blocks}) == 36
            and len({b["group"] for b in blocks}) == 36
            and sum(len(b["candidates"]) for b in blocks) == 2620,
            "Fixed FIT36 image and candidate rosters")
    return blocks, anchor["prediction_sha256"]


def produce_modes(anchor, donors, planes, prior, verify, codec, expit):
    records = [anchor, *donors]
    hard = [codec.decode(r["rle"]).astype(bool) for r in records]
    require(all(h.shape == (800, 800) for h in hard), "Original 800x800 masks")
    yy, xx, inside = verify.band_points(np.flatnonzero(hard[0].ravel(order="F")))
    native = [h[yy, xx] for h in hard]
    require(int(native[0].sum()) == inside, "Original anchor band population")
    values = [prior.original_values(planes[r["logit_row"]], mask, yy, xx, verify)
              for r, mask in zip(records, hard, strict=True)]
    score = np.asarray([r["score"] for r in records], dtype=float)
    require(np.isfinite(score).all() and score.min() >= 0 and score.sum() > 0, "Native weights")
    probability = np.stack([expit(logits) for logits, _ in values])
    weighted = np.average(probability, axis=0, weights=score)
    cap = 2 * min(8, anchor["area"] // 10)
    edits = prior.action_indices(native[0], np.sum(native, axis=0) >= 2, probability[0], weighted, cap)
    add = int((~native[0][edits["vote"]]).sum())
    drop = len(edits["vote"]) - add
    result = {}
    for name, positions in edits.items():
        if len(positions) == 0:
            continue
        require(len(positions) == add + drop <= cap and int(native[0][positions].sum()) == drop,
                "Same native band, add/drop counts and after-edit area")
        mask = hard[0].copy()
        mask[yy[positions], xx[positions]] ^= True
        rle = codec.encode(np.asfortranarray(mask, dtype=np.uint8))
        rle["counts"] = rle["counts"].decode("ascii")
        require(int(codec.area(rle)) == anchor["area"] + add - drop, "Exact after-edit area")
        result[name] = rle
    return result, add, drop, sum(b for _, b in values)


def prepare(args, blocks, anchor_sha, area, prior, verify, codec, expit):
    counts = Counter(dict(original_candidates=2620, original_images=36, edited_candidates=0,
                          associated_candidates=0, source_groups_with_support=0,
                          add_per_arm=0, drop_per_arm=0, author_torch_fallback_planes=0,
                          repeated_query_donor_exclusions=0))
    rows, private = [], []
    for block in blocks:
        candidates = block["candidates"]
        planes = verify.array_from(args.capture / "5202", block)
        masks = [r["rle"] for r in candidates]
        overlap = codec.iou(masks, masks, np.zeros(len(masks), dtype=np.uint8).tolist())
        groups = suppression_groups(overlap)
        counts["representatives"] += len(groups)
        counts["suppressed"] += sum(len(members) for _, members in groups)
        variants = {}
        eligible_in_image = 0
        for i, members in groups:
            anchor = candidates[i]
            if not 32 <= anchor["area"] < 1024 or anchor["logit_row"] is None:
                continue
            donors = []
            seen = {anchor["query"]}
            for j in members:
                record = candidates[j]
                if record["query"] in seen:
                    counts["repeated_query_donor_exclusions"] += 1
                    continue
                if record["logit_row"] is None:
                    continue
                donors.append(record)
                seen.add(record["query"])
                if len(donors) == 2:
                    break
            if len(donors) != 2:
                continue
            eligible_in_image += 1
            counts["associated_candidates"] += 1
            edits, add, drop, fallback = produce_modes(anchor, donors, planes, prior, verify, codec, expit)
            counts["author_torch_fallback_planes"] += fallback
            if edits:
                variants[anchor["rank"]] = edits
                counts["edited_candidates"] += 1
                counts["add_per_arm"] += add
                counts["drop_per_arm"] += drop
        counts["source_groups_with_support"] += bool(eligible_in_image)
        rows.append(dict(candidates=candidates, **{key: block[key] for key in ("ordinal", "group", "basename")}))
        private.append(dict(basename=block["basename"], variants=variants))
    require(counts["representatives"] + counts["suppressed"] == counts["original_candidates"],
            "Each original candidate assigned at most once")
    digest = sha256()
    with (args.output / "MASK_EDITS_PRIVATE.jsonl").open("xb") as file:
        for row in private:
            line = (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            file.write(line)
            digest.update(line)
    receipt = dict(status="HRSID_FIT_SINGLE_FORWARD_QUERY_PRELABEL", FIT=True, DEV=False, TEST=False,
                   GT_opened=False, only_seed=5202, variants_sha256=digest.hexdigest(),
                   anchor_sha256=anchor_sha, protocol_sha256=PROTOCOL, **counts)
    area.write_new(args.output / "PRELABEL_SAFE.json", receipt)
    return rows, private, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "capture", "witness", "fit-index", "code", "shapely-site",
                 "three-forward-summary", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    start = perf_counter()
    require(sha256(args.protocol.read_bytes()).hexdigest() == PROTOCOL, "Locked protocol")
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    from scipy.special import expit
    import shapely
    from pycocotools import mask as codec
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Exact geometry")

    code = args.code
    prior = module(code / "HRSID_FIT_CONSENSUS_CONTOUR_20261112/score_consensus.py",
                   PREVIOUS_SOURCE, "_single_forward_prior")
    area = module(code / "HRSID_FIT_SEED_AREA_CONFIRM_20261112/score/score_seed_area_confirm.py",
                  prior.AREA, "_single_forward_area")
    verify = area.module("_single_forward_verify", code /
                         "HRSID_FIT_SEED_AREA_CONFIRM_20261112/verify_area_signals.py", prior.VERIFY)
    stock = area.module("_single_forward_match", code / "HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py", prior.MATCH)
    endpoint = area.module("_single_forward_geometry", code /
                           "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", area.GEOMETRY_SHA)
    geom = endpoint.geometry_modules(code)
    args.output.mkdir()
    blocks, anchor_sha = read_carrier(args, prior, area)
    rows, private, receipt = prepare(args, blocks, anchor_sha, area, prior, verify, codec, expit)
    scored = prior.run_score(args, area, stock, geom, codec, rows, private)
    counts = scored["edited_candidate_scope"]
    require(sum(counts[k] for k in ("edited_matches", "edited_unmatched", "edited_neutral"))
            == receipt["edited_candidates"], "Exact edit support")
    external = area.json_value(area.pinned(args.three_forward_summary, THREE_FORWARD))
    require(external["anchor_sha256"] == anchor_sha and
            external["mask_metrics"]["raw"] == scored["mask_metrics"]["raw"] and
            external["fixed_geometry"]["raw"] == scored["fixed_geometry"]["raw"],
            "Exact raw endpoint replay for higher-compute reference")
    reference = dict(mask_metrics=external["mask_metrics"]["soft"], fixed_geometry=external["fixed_geometry"]["soft"],
                     reference_sha256=THREE_FORWARD)
    result = dict(receipt, **scored, status="HRSID_FIT_SINGLE_FORWARD_QUERY_ACTUAL", GT_opened=True,
                  soft_mode="classification_score_weighted_probability",
                  higher_compute_reference=reference, model_forwards=0, optimizer_updates=0,
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  worker_seconds=perf_counter()-start)
    area.write_new(args.output / "RESULT_SAFE.json", result)
    print(json.dumps({k: result[k] for k in ("status", "associated_candidates", "edited_candidates",
                                             "gates", "advance")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    import os
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only execution")
    import torch
    torch.set_num_threads(1)

    def offline(event, _args):
        if event.startswith(("socket.", "subprocess.")):
            raise RuntimeError("Offline scientific execution")
    sys.addaudithook(offline)
    main()
