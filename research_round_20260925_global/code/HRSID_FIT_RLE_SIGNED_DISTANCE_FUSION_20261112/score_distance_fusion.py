"""Evaluate fixed-count shape-based averaging on nested HRSID donor supports."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np

PROTOCOL = "953ba5d3c7647e2a03229e3d745d490155e1989981e0636dc9d6ef3681f88123"
HEADROOM_SOURCE = "fd89f4808138f98c2103e6aa89ec10ae30ca8ac4114dee4cc11954e97c008278"
MODES = ("raw", "anchor_count", "vote", "sdf_both", "sdf_any", "anchor_any")
COMPARATORS = ("vote", "sdf_both", "anchor_any")
COEFFICIENTS = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0),
                         (2, 0, 0), (0, 0, 1), (0, 0, 0), (0, 0, 0),
                         (0, 2, 0), (3, 0, 0)], dtype=np.int16)


def require(ok, why):
    if not ok:
        raise ValueError(why)


def module(path, digest, name):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == digest, "Exact source " + path.name)
    result = ModuleType(name)
    result.__file__ = str(path)
    sys.modules[name] = result
    exec(compile(data, str(path), "exec"), result.__dict__)
    return result


def signed_distance(mask):
    from scipy.ndimage import distance_transform_edt
    require(mask.ndim == 2 and mask.any(), "Distance requires nonempty 2D native mask")
    border = np.pad(mask, 1, constant_values=False)
    distance = distance_transform_edt(border) - distance_transform_edt(~border)
    return np.clip(distance[1:-1, 1:-1], -3.0, 3.0)


def signed_distance_sum(distances):
    # The six possible clipped center distances have exact coefficient keys.
    coefficients = np.zeros((*distances[0].shape, 3), dtype=np.int16)
    for distance in distances:
        squares = np.rint(distance ** 2).astype(np.int8)
        require(np.isin(squares, (1, 2, 4, 5, 8, 9)).all(), "Exact native distance magnitudes")
        coefficients += np.sign(distance).astype(np.int16)[..., None] * COEFFICIENTS[squares]
    return coefficients[..., 0] + coefficients[..., 1] * np.sqrt(2) + coefficients[..., 2] * np.sqrt(5)


def choose_by_score(anchor, support, score, adds, drops):
    flat = score.ravel()
    off = np.flatnonzero(support & ~anchor)
    on = np.flatnonzero(support & anchor)
    require(0 <= adds <= len(off) and 0 <= drops <= len(on), "Exact inherited counts fit support")
    selected_add = off[np.lexsort((off, -flat[off]))[:adds]]
    selected_drop = on[np.lexsort((on, flat[on]))[:drops]]
    result = anchor.copy(order="C")
    result.ravel()[selected_add] = True
    result.ravel()[selected_drop] = False
    return result


def tight_bounds(masks):
    h, w = masks[0].shape
    yy, xx = np.nonzero(np.logical_or.reduce(masks))
    return np.s_[max(0, yy.min() - 3):min(h, yy.max() + 4),
                 max(0, xx.min() - 3):min(w, xx.max() + 4)]


def cutoff_ties(anchor, support, score, selected):
    ties = 0
    for original_value in (True, False):
        points = support & (anchor == original_value)
        edited = selected & points
        other = points & ~edited
        if edited.any() and other.any():
            boundary = (score[edited].max() == score[other].min() if original_value else
                        score[edited].min() == score[other].max())
            ties += int(boundary)
    return ties


def select_triplet(anchor, donor1, donor2, voted, bridge):
    require(anchor.shape == donor1.shape == donor2.shape == voted.shape, "Exact original grid")
    window = tight_bounds((anchor, donor1, donor2))
    a, d1, d2, v = (mask[window] for mask in (anchor, donor1, donor2, voted))
    band, _, core = bridge.two_layers(a)
    both = band & (d1 != a) & (d2 != a)
    any_donor = band & ((d1 != a) | (d2 != a))
    old = a ^ v
    adds, drops = int((old & ~a).sum()), int((old & a).sum())
    require(int((anchor ^ voted).sum()) == adds + drops and not (old & ~both).any(),
            "Exact closed majority edit support")
    cap = 2 * min(8, int(anchor.sum()) // 10)
    require(np.array_equal(np.flatnonzero(old), np.flatnonzero(both)[:cap]), "Exact closed row-major vote")
    fields = tuple(signed_distance(mask) for mask in (a, d1, d2))
    score = signed_distance_sum(fields)
    outputs = dict(sdf_both=choose_by_score(a, both, score, adds, drops),
                   sdf_any=choose_by_score(a, any_donor, score, adds, drops),
                   anchor_any=choose_by_score(a, any_donor, fields[0], adds, drops))
    full = choose_by_score(a, band, score, adds, drops)
    require(np.array_equal(full, outputs["sdf_any"]), "Predicted full-band parity at fixed counts")
    new = a ^ outputs["sdf_any"]
    single = new & ~both
    summary = dict(add_per_arm=adds, drop_per_arm=drops, eligible_single_sites=int((any_donor & ~both).sum()),
                   selected_single_adds=int((single & ~a).sum()), selected_single_drops=int((single & a).sum()),
                   edited_candidates_using_single=int(single.any()), cutoff_tie_sides=cutoff_ties(a, any_donor, score, new),
                   coreless_edited_anchors=int(not core.any()), full_band_parity_candidates=1,
                   different_from_vote=int(not np.array_equal(outputs["sdf_any"], v)))
    for name, mask in outputs.items():
        require((int((mask & ~a).sum()), int((~mask & a).sum())) == (adds, drops), "Equal edits and area")
        lifted = anchor.copy()
        lifted[window] = mask
        outputs[name] = lifted
    return outputs, summary


def prepare(args, base, prior, bridge, area, codec):
    admitted = args.output / "CLOSED_SOURCE_ADMISSION"
    admitted.mkdir()
    cases, donors, closed, old = base.prepare(argparse.Namespace(**dict(vars(args), output=admitted)), prior, area)
    read_mask = lambda item: codec.decode(item).astype(bool)
    private, counts = [], Counter()
    for ordinal, (case, row) in enumerate(zip(cases, closed, strict=True)):
        require(case["basename"] == row["basename"] and
                all(donors[s][ordinal]["basename"] == row["basename"] and
                    donors[s][ordinal]["group"] == case["group"] and
                    donors[s][ordinal]["ordinal"] == case["ordinal"] for s in (5201, 5203)),
                "Exact original case and donor identity")
        anchors = {p["rank"]: p for p in case["candidates"]}
        pools = {s: {p["rank"]: p for p in donors[s][ordinal]["candidates"]} for s in (5201, 5203)}
        variants = {}
        for rank_text, fixed in row["variants"].items():
            rank = int(rank_text)
            signal = case["signals"][rank]
            anchor = read_mask(anchors[rank]["rle"])
            pair = [read_mask(pools[s][signal["donor_ranks"][str(s)]]["rle"]) for s in (5201, 5203)]
            modes, summary = select_triplet(anchor, *pair, read_mask(fixed["vote"]), bridge)
            counts.update(summary)
            masks = {name: bridge.encode(codec, mask) for name, mask in modes.items()}
            variants[rank] = dict(vote=fixed["vote"], anchor_count=fixed["anchor_count"], soft=masks["sdf_any"], **masks)
        private.append(dict(basename=row["basename"], variants=variants))
    require(counts["full_band_parity_candidates"] == 464 and counts["add_per_arm"] == 2932 and
            counts["drop_per_arm"] == 3677, "Exact earlier vote counts and parity")
    digest = sha256()
    with (args.output / "VARIANTS_PRIVATE.jsonl").open("xb") as file:
        for row in private:
            packed = (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            file.write(packed)
            digest.update(packed)
    receipt = dict(status="HRSID_FIT_SIGNED_DISTANCE_FUSION_PRELABEL", FIT=True, DEV=False, TEST=False,
                   GT_opened=False, logit_planes_opened=False, images=36, original_candidates=2620,
                   edited_candidates=464, source_variant_sha256=old["variants_sha256"],
                   protocol_sha256=PROTOCOL, variants_sha256=digest.hexdigest(), **counts)
    area.write_new(args.output / "PRELABEL_SAFE.json", receipt)
    return cases, private, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("protocol", "bridge", "capture", "witness", "code", "fit-index", "shapely-site", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    args = parser.parse_args()
    started = perf_counter()
    require(sha256(args.protocol.read_bytes()).hexdigest() == PROTOCOL, "Exact locked protocol")
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    from pycocotools import mask as codec
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Exact geometry")
    code = args.code
    base = module(code / "HRSID_FIT_RLE_VOTE_HEADROOM_20261112/score_headroom.py", HEADROOM_SOURCE, "_sdf_headroom")
    bridge = module(code / "HRSID_FIT_RLE_VOTE_SOURCE_BRIDGE_20261112/score_binary_bridge.py", base.BRIDGE_CODE, "_sdf_bridge")
    prior = module(code / "HRSID_FIT_CONSENSUS_CONTOUR_20261112/score_consensus.py", bridge.PRIOR, "_sdf_prior")
    area = module(code / "HRSID_FIT_SEED_AREA_CONFIRM_20261112/score/score_seed_area_confirm.py", prior.AREA, "_sdf_area")
    stock = area.module("_sdf_match", code / "HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py", prior.MATCH)
    geom = area.module("_sdf_geometry", code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                       area.GEOMETRY_SHA).geometry_modules(code)
    args.output.mkdir()
    cases, private, receipt = prepare(args, base, prior, bridge, area, codec)
    prior.MODES = (*MODES, "soft")
    scored = prior.run_score(args, area, stock, geom, codec, cases, private)
    for key in ("mask_metrics", "fixed_geometry"):
        require(scored[key].pop("soft") == scored[key]["sdf_any"], "Exact common score alias")
    scores, geometry = scored["mask_metrics"], scored["fixed_geometry"]
    past = area.json_value(area.pinned(args.bridge / "RESULT_SAFE.json", base.BRIDGE_REPORT))
    for mode in ("raw", "anchor_count", "vote"):
        require(scores[mode] == past["mask_metrics"][mode] and geometry[mode] == past["fixed_geometry"][mode],
                "Exact closed raw and control replay")
    margin = {m: dict(AP=scores["sdf_any"]["mask_ap"] - scores[m]["mask_ap"],
                      IoU=geometry["sdf_any"]["iou_points"] - geometry[m]["iou_points"]) for m in COMPARATORS}
    gates = dict(ap_margin=all(x["AP"] >= .30 for x in margin.values()),
                 continuous_iou_margin=all(x["IoU"] >= .20 for x in margin.values()),
                 ap75_retained=scores["sdf_any"]["ap75"] >= scores["vote"]["ap75"] - 1e-12,
                 strict_retained=geometry["sdf_any"]["strict_hits"] >= geometry["vote"]["strict_hits"])
    scored.pop("soft_fixed_fold_delta")
    scored.update(gates=gates, advance=all(gates.values()), lead_margins=margin)
    result = dict(receipt, **scored, status="HRSID_FIT_SIGNED_DISTANCE_FUSION_ACTUAL", GT_opened=True,
                  model_forwards=0, optimizer_updates=0, worker_seconds=perf_counter()-started,
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    area.write_new(args.output / "RESULT_SAFE.json", result)
    print(json.dumps({key: result[key] for key in ("status", "lead_margins", "gates", "advance")},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    def offline(event, _args):
        if event.startswith(("socket.", "subprocess.")):
            raise RuntimeError("Offline CPU diagnostic")
    sys.addaudithook(offline)
    main()
