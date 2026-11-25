"""Retrospective equal-area endpoint ranking envelope on the closed SRNet reserve."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
from math import fsum
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np


STATUS = "SRNET_RSDD74_MATCHED_RANKING_ORACLE_COMPLETE"
PROTOCOL_SHA = "23e98ad86fae80a9fbfed17f22f9654f66cec33342cb4226fb60d931f411a4c9"
PRELOCK_COMMIT = "c831ae7eedb731f1a7e9bab321105be836b92763"
SCORE_SHA = "8f26062eed2f56501f1a50a656961535b9fff3aa7ed493b7d4fe758f44b1f77a"
SCORE_TERMINAL_SHA = "799047e9123044aff40ef1bead0698ba6a7d103aa744d9d4934d766d562012e9"
SCORE_WORKER_SHA = "6ba17a0bcaa897f6f56bca354c22b2db57b79e38b18f60f50cc4ae3498613889"
SCORE_AUTHORITY_SHA = "7121fd1b7c1c98ee87ce6cf23963c09fabc437af458f57fffa43fb99a75c6339"
PAST = "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/oracle.py"
PAST_SHA = "ec90d3c4f7682fd0e64c689fc2a73231f0daedba02d3e982275a6518ee55d815"

ARMS = ("raw", "fixed", "free_matched_count", "oracle", "oracle_at_most_k")
PAIRS = (("oracle", "free_matched_count"), ("oracle", "fixed"), ("oracle", "raw"),
         ("free_matched_count", "fixed"), ("oracle_at_most_k", "oracle"))
FIELDS = ("continuous_iou_points", "fp_per_truth_area_points", "fn_per_truth_area_points",
          "intersection_per_truth_area_points", "added_true_mass_per_truth_area_points",
          "removed_true_mass_per_truth_area_points")
POSITIVE_FIELDS = ("true_mass_per_added_cell", "true_mass_per_removed_cell", "background_mass_per_removed_cell")


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Exact input: " + path.name)
    return raw


def load(path, expected):
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), mod.__dict__)
    return mod


def best_choices(coords, occupancy, k):
    require(coords.shape == occupancy.shape and coords.ndim == 2 and coords.shape[1] == 2
            and 0 <= k <= len(coords) and np.isfinite(occupancy).all()
            and np.all((occupancy >= 0) & (occupancy <= 1 + 1e-12)), "Frozen fractional signed endpoints")
    adds = np.argsort(-occupancy[:, 0], kind="stable")[:k]
    drops = np.argsort(occupancy[:, 1], kind="stable")[:k]
    return list(zip(adds.tolist(), drops.tolist(), strict=True))


def selected_mass(occupancy, choices):
    return fsum(occupancy[i, 0] for i, _ in choices), fsum(occupancy[j, 1] for _, j in choices)


def native_anchors(evaluator, gt, blocks):
    """Stop the original raster matching algorithm before AP accumulation."""
    pairs = [(i, rank, row) for i, block in blocks.items()
             for rank, row in zip(block["ranks"], block["stock"], strict=True)]
    with redirect_stdout(StringIO()):
        ref = evaluator.COCO()
        ref.dataset = deepcopy(gt)
        ref.createIndex()
        det = ref.loadRes(deepcopy([row for _, _, row in pairs]))
        ev = evaluator.COCOeval(ref, det, "segm")
        ev.params.catIds = [1]
        ev.params.imgIds = [im["id"] for im in gt["images"]]
        ev.params.maxDets = [1, 10, 100]
        ev.params.iouThrs = np.asarray([.50])
        ev.evaluate()
    anchors = {}
    for row in ev.evalImgs:
        if row is None or row["maxDet"] != 100 or list(row["aRng"]) != list(ev.params.areaRng[0]):
            continue
        for ident, det, ignore in zip(row["gtIds"], row["gtMatches"][0], row["gtIgnore"], strict=True):
            if det and not ignore:
                record = pairs[int(det) - 1]
                require(ident not in anchors and record[0] == row["image_id"], "Unique original .50 match")
                anchors[int(ident)] = record
    return anchors


def recover_choices(codec, stock, coord, edit, fixed):
    if not edit:
        return []
    rendered = codec.decode(edit["rle"]).astype(bool)
    added = np.flatnonzero(rendered & ~stock)
    dropped = np.flatnonzero(stock & ~rendered)
    k = edit["swaps"]
    require(len(added) == len(dropped) == k, "Exact archived signed operations")
    a_ids = np.flatnonzero(np.isin(coord[:, 0], added)).tolist()
    d_ids = np.flatnonzero(np.isin(coord[:, 1], dropped)).tolist()
    require(len(a_ids) == len(d_ids) == k and (not fixed or a_ids == d_ids),
            "All archived sites are from the original support and F preserves fixed pairs")
    return list(zip(a_ids, d_ids, strict=True))


def exact_points(a, replay, stream, blocks):
    base, methods = replay.edit_modules(a.code)
    point, sampler = base.canvas(methods, (512, 512))
    entries, counts = {}, Counter()
    for row in stream:
        counts["images"] += 1
        rgb = base.load_rgb(a.manifest.parent / "images", row)
        cells = methods.pair.image_features(rgb)
        probs = replay.native_probabilities(a, row)
        frame = blocks[row["image_id"]]
        for cand, orig, p in zip(row["candidates"], frame["stock"], probs, strict=True):
            counts["native_candidates"] += 1
            stock = replay.codec.decode(orig["segmentation"]).astype(bool)
            require(np.array_equal(p > .5, stock), "Exact strict mask/point posterior parity")
            area = int(stock.sum())
            rank = cand["rank"]
            if not 32 <= area < 1024:
                continue
            counts["area_eligible_candidates"] += 1
            box = np.asarray(cand["clipped_corner_xyxy"], dtype=np.float32)
            data = None if np.any(box[2:] == box[:2]) else point.paired_points(
                stock, box, area, cand["score"], p, cells, sampler)
            coords = data[0] if data is not None else np.empty((0, 2), dtype=np.int32)
            base.check_support(methods, stock, box, p, coords)
            observed = frame["changes"].get(rank, {})
            base.check_edits(stock, coords, observed)
            fixed = recover_choices(replay.codec, stock, coords, observed.get(replay.F), fixed=True)
            free = recover_choices(replay.codec, stock, coords, observed.get(replay.I), fixed=False)
            reencoded = {name:base.mask_edit(stock, coords, choices)
                         for name, choices in ((replay.F, fixed), (replay.I, free)) if choices}
            require(reencoded == observed, "Exact archived F and I mask re-encoding")
            k = len(fixed)
            cap = min(8, area // 10, len(coords))
            require(len(free) == k <= cap, "Original realized budget and cap")
            counts["positive_candidates"] += int(k > 0)
            entries[(row["image_id"], rank)] = (coords, fixed, free)
    require(counts["images"] == 74, "Whole closed point support reconstructed")
    return entries, counts


def policy_values(geom, geom_iou, codec, stock, options, points, target, shapely):
    coords, fixed, independent = points
    k = len(fixed)
    area = int(codec.area(stock))
    target_area = float(target.area)
    base_iou = geom(stock, target, shapely)
    base_hit = base_iou * (area + target_area) / (1 + base_iou)
    if len(coords):
        y, x = np.divmod(coords, stock["size"][1])
        occ = shapely.area(shapely.intersection(shapely.box(x, y, x + 1, y + 1), target))
    else:
        occ = np.empty((0, 2))
    oracle = best_choices(coords, occ, k)
    gain_curve = [fsum((a, -d)) for a, d in (selected_mass(occ, oracle[:j]) for j in range(k + 1))]
    fewer = max(range(k + 1), key=gain_curve.__getitem__)
    all_choices = ([], fixed, independent, oracle, oracle[:fewer])
    values, per_unit = [], []
    for arm, chosen in zip(ARMS, all_choices, strict=True):
        a_mass, d_mass = selected_mass(occ, chosen)
        hit = fsum((base_hit, a_mass, -d_mass))
        ji = float(geom_iou(hit, area, target_area))
        if arm in ARMS[:3]:
            ref = options.get(arm, {}).get("rle", stock)
            require(abs(ji - 100 * geom(ref, target, shapely)) < 1e-8,
                    "Unit-cell signed mass reproduces prior raster polygon geometry")
        values.append((ji, 100 * (area - hit) / target_area, 100 * (target_area - hit) / target_area,
                       100 * hit / target_area, 100 * a_mass / target_area, 100 * d_mass / target_area))
        if k:
            per_unit.append((a_mass / k, d_mass / k, 1 - d_mass / k) if arm != "raw" else (0.0, 0.0, 0.0))
    require(values[4][0] + 1e-8 >= values[3][0] >= max(values[1][0], values[2][0]) - 1e-8,
            "Exact attainable ranking upper bound for both old policies")
    counters = Counter(eligible_matches=1, positive_matches=int(k > 0), zero_budget_matches=int(k == 0),
                       add_budget=k, drop_budget=k, signed_endpoints_per_side=len(coords),
                       oracle_below_native=int(values[3][0] < values[0][0] - 1e-8),
                       secondary_less_than_k=int(fewer < k), secondary_balanced_budget=fewer)
    if k:
        minimum = min(8, area // 10, len(coords))
        counters.update(policy_max_cap_active=int(minimum == 8), area_cap_active=int(minimum == area // 10),
                        endpoint_cap_active=int(minimum == len(coords)), positive_gain_limited=int(k < minimum),
                        learned_oracle_add_mass_shortfall=int(values[3][4] > values[2][4] + 1e-8),
                        learned_oracle_drop_mass_shortfall=int(values[3][5] < values[2][5] - 1e-8))
    return np.asarray(values), np.asarray(per_unit), counters


def summaries(reserve, groups, units, dates, counts, prior):
    parents = sorted(groups)
    require(len(parents) == prior["geometry"]["represented_parents"] and
            len({dates[p] for p in parents}) == prior["geometry"]["represented_sensor_date_blocks"],
            "Original parent and sensor-date geometry denominator")
    mu = np.stack([np.mean(groups[p], axis=0) for p in parents])
    means = mu.mean(axis=0)
    for n, arm in enumerate(ARMS[:3]):
        for j, key in enumerate(FIELDS[:3]):
            require(abs(means[n, j] - prior["geometry"]["group_equal_arms"][arm][key]) < 1e-8,
                    "Original N/F/I aggregate geometry parity")
    require(counts["eligible_matches"] == prior["geometry"]["counts"]["eligible_matches"] and
            counts["original_matches"] == prior["geometry"]["counts"]["original_matches"] and
            counts["positive_matches"] == prior["geometry"]["counts"]["positive_fixed_edits"], "Original counts parity")
    contrasts = {}
    for left, right in PAIRS:
        delta = mu[:, ARMS.index(left)] - mu[:, ARMS.index(right)]
        sd = reserve.bootstrap(delta, [dates[p] for p in parents])
        parent = reserve.bootstrap(delta, parents)
        contrasts[left + " minus " + right] = {
            field: dict(delta=float(m), sensor_date_interval_95=s, parent_interval_95=p)
            for field, m, s, p in zip(FIELDS, delta.mean(axis=0), sd, parent, strict=True)}
    unit_summary = {}
    if units:
        per_parent = np.stack([np.mean(units[p], axis=0) for p in sorted(units)])
        unit_summary = {arm:dict(zip(POSITIVE_FIELDS, row.tolist(), strict=True))
                        for arm, row in zip(ARMS[1:4], per_parent.mean(axis=0)[1:4], strict=True)}
    regret = contrasts["oracle minus free_matched_count"]
    return dict(counts=dict(counts), represented_parents=len(parents),
                represented_sensor_date_blocks=len({dates[p] for p in parents}), positive_parents=len(units),
                geometry_points={arm: dict(zip(FIELDS, row.tolist(), strict=True))
                                 for arm, row in zip(ARMS, means, strict=True)},
                contrasts_points=contrasts,
                oracle_minus_learned_signed_parts_points_per_truth_area=dict(
                    extra_occupied_add_mass=regret[FIELDS[4]]["delta"],
                    avoided_occupied_removal_mass=-regret[FIELDS[5]]["delta"]),
                parent_equal_mean_on_positive_budgets_only=unit_summary,
                material_fixed_count_headroom=(regret[FIELDS[0]]["delta"] >= .30
                                               and regret[FIELDS[0]]["sensor_date_interval_95"][0] > 0))


def run(args, auth):
    started = perf_counter()
    root = Path(auth["paired_replay"]).resolve(strict=True)
    score_worker = load(root / "score.py", SCORE_WORKER_SHA)
    orig = json.loads(pin(root / "AUTHORITY_score.json", SCORE_AUTHORITY_SHA))
    keys = ("code", "protocol", "capture", "manifest", "prepare", "source_result",
            "source_terminal", "hrsid_result", "hrsid_terminal", "census", "author_train", "shapely_site")
    site = SimpleNamespace(**{k: Path(v) if k in keys else v for k, v in orig.items()})
    replay = load(root / "replay.py", site.replay_worker_sha)
    prior = replay.closed(root / "observed_score/TERMINAL.json", SCORE_TERMINAL_SHA,
                          root / "score/RESULT_SAFE.json", SCORE_SHA, "score", SCORE_WORKER_SHA)
    require(prior["status"] == score_worker.STATUS and prior["authority_sha256"] == SCORE_AUTHORITY_SHA
            and prior["protocol_sha256"] == site.protocol_sha and prior["prelock_commit"] == site.prelock_commit,
            "Scope of original closed comparison")
    score_worker.audit_source(site, replay)
    proof, blocks = score_worker.read_closed_stream(site, replay)
    cap, rows = replay.checked_capture(site)
    points, counts = exact_points(site, replay, rows, blocks)
    require(counts["native_candidates"] == proof["counts"]["native_candidates"] == cap["counts"]["candidates"]
            and counts["positive_candidates"] == proof["counts"]["positive_matched_budget_candidates"],
            "Exact candidate flow before opening admitted author reference")

    reserve = load(site.code / score_worker.RESERVE, score_worker.RESERVE_SHA)
    old = load(site.code / score_worker.HISTORIC, reserve.HISTORIC)
    evaluator = old.module(site.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", old.SCORE_SOURCE)
    adapter = old.module(site.code / "UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", old.GEOMETRY_SOURCE)
    core = adapter.load_geometry(site.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    historical_oracle = load(site.code / PAST, PAST_SHA)
    sys.path.append(str(site.shapely_site))
    import shapely
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Original geometry runtime")
    census = replay.json_pin(site.census, reserve.CENSUS)
    dates = {p["paper_scene"]: (p["sensor"], p["calendar_date"]) for p in census["per_parent"]}
    nested = set(census["date_separated_sensitivity"]["parent_scene_ids"])
    gt, _ = old.coco_annotation_ids(reserve.select_labels(pin(site.author_train, reserve.TRAIN), blocks))
    targets = {v["id"]: shapely.Polygon(np.asarray(v["segmentation"][0]).reshape(-1, 2)) for v in gt["annotations"]}
    anchors = native_anchors(evaluator, gt, blocks)
    flow = {"full74": (defaultdict(list), defaultdict(list), Counter()),
            "nested42_sensor_date": (defaultdict(list), defaultdict(list), Counter())}
    for gid, (image, rank, original) in anchors.items():
        frame = blocks[image]
        group = frame["group"]
        use = ["full74", "nested42_sensor_date"] if group in nested else ["full74"]
        for view in use:
            flow[view][2]["original_matches"] += 1
        stock = original["segmentation"]
        poly = targets[gid]
        if not poly.is_valid or poly.area <= 0 or not 32 <= int(core.codec.area(stock)) < 1024:
            continue
        options = frame["changes"].get(rank, {})
        v, unit, counter = policy_values(core.geometric_iou, historical_oracle.iou, core.codec, stock, options,
                                         points[(image, rank)], poly, shapely)
        for view in use:
            geo, units, totals = flow[view]
            geo[group].append(v)
            if unit.size:
                units[group].append(unit)
            totals.update(counter)
    views = {name: summaries(reserve, geo, units, dates, totals, prior["views"][name])
             for name, (geo, units, totals) in flow.items()}
    result = dict(status=STATUS, retrospective_author_train_only=True, protocol_sha256=auth["protocol_sha"],
                  prelock_commit=auth["prelock_commit"], authority_sha256=args.authority_sha,
                  worker_sha256=digest(Path(__file__).read_bytes()), frozen_score_sha256=SCORE_SHA,
                  frozen_score_terminal_sha256=SCORE_TERMINAL_SHA, original_candidate_flow=dict(counts),
                  fixed_mask_area=True, exact_signed_quota=True, endpoint_write_policy_selected=False,
                  original_score_order_retained=True, AP_computed=False, optimizer_updates=0,
                  detector_forwards=0, point_head_forward_calls=0, author_test_opened=False,
                  bootstrap_draws=4096, bootstrap_seed=5217,
                  original_group_metrics_reproduced=True, views=views, elapsed_seconds=perf_counter()-started)
    payload = replay.encoded(result)
    args.output.mkdir()
    (args.output / "RESULT_SAFE.json").write_bytes(payload)
    print(json.dumps(dict(status=STATUS, result_sha256=digest(payload))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(pin(args.authority, args.authority_sha))
    require(set(raw) == {"protocol", "protocol_sha", "prelock_commit", "paired_replay"}
            and all(Path(raw[key]).is_absolute() for key in ("protocol", "paired_replay"))
            and raw["protocol_sha"] == PROTOCOL_SHA
            and raw["prelock_commit"] == PRELOCK_COMMIT
            and args.output.name == "actual" and not args.output.exists()
            and sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11", "Fresh admitted CPU diagnostic")
    pin(Path(raw["protocol"]), raw["protocol_sha"])
    run(args, raw)


if __name__ == "__main__":
    main()

