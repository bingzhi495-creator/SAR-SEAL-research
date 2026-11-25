"""Split fixed-area oracle regret by the clipped native SRNet prior strata."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
from math import fsum
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np


STATUS = "SRNET_RSDD74_PRIOR_CLIP_STRATA_COMPLETE"
PROTOCOL_SHA = "d14c88a154fa2c3e436f0763e70c995372063cd62603fb1c9d94b3c20f8367e2"
PRELOCK_COMMIT = "17948d219a19c7cee62639456f328524dddec398"
ORACLE_WORKER = "2273d873a639ee53487269bdf2e110fd029fb03e1a4f76dc9a6f240ad0e8a4a1"
ORACLE_RESULT = "6035a39e675b02025154f87fc8e0d07390b7927c86b1dc04d1c07704da3c2b95"
ORACLE_TERMINAL = "54b3589765046ab323626c49ea9da1dc2255cd835a40a710876d62babadc0b12"
ORACLE_AUTHORITY = "02dad48b2bdd4b54bf71790364c879a6021977784ee4d8e5762e5fc499df3c89"
TRAINER = "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_paired_point/train_point.py"
TRAINER_SHA = "3111c34afd791cb15987ed94c5e898de4c3ff2cad61677b616420651335d812c"
CLIP_LOW = np.float32(1e-4)
CLIP_HIGH = np.float32(1-1e-4)
ARMS = ("free_matched_count", "stratified_oracle", "oracle")
VIEWS = ("full74", "nested42_sensor_date")
STRATUM_FIELDS = ("added_tail_mass_per_truth_area_points", "added_interior_mass_per_truth_area_points",
                  "removed_tail_mass_per_truth_area_points", "removed_interior_mass_per_truth_area_points")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    data = path.read_bytes()
    require(digest(data) == expected, "Exact input: " + path.name)
    return data


def load(path, expected):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), module.__dict__)
    return module


def clip_strata(p):
    require(p.ndim == 2 and p.shape[1] == 2 and p.dtype == np.float32 and np.isfinite(p).all(),
            "Native per-sign FP32 prior")
    require(np.all(p[:, 0] <= .5) and np.all(p[:, 1] > .5), "Author strict foreground threshold and sign")
    return (np.column_stack((p[:, 0] <= CLIP_LOW, p[:, 1] >= CLIP_HIGH)),
            np.column_stack((p[:, 0] == 0, p[:, 1] == 1)))


def optimal_in_strata(occ, tail, incumbent):
    indices = np.asarray(incumbent, dtype=np.intp).reshape(-1, 2)
    k = len(indices)
    require(occ.shape == tail.shape and occ.shape[1] == 2 and 0 <= k <= len(occ),
            "Exact masks and quota")
    sides = []
    for side in range(2):
        chosen = []
        for t in (True, False):
            rows = np.flatnonzero(tail[:, side] == t)
            amount = int(np.count_nonzero(tail[indices[:, side], side] == t))
            values = occ[rows, side] * (-1 if side == 0 else 1)
            chosen.extend(rows[np.argsort(values, kind="stable")[:amount]].tolist())
        require(len(chosen) == k and len(set(chosen)) == k, "Exact incumbent tail and interior quotas")
        sides.append(chosen)
    return list(zip(*sides, strict=True))


def point_priors(site, replay, rows, points):
    native = {}
    for row in rows:
        probs = replay.native_probabilities(site, row)
        for cand, plane in zip(row["candidates"], probs, strict=True):
            key = row["image_id"], cand["rank"]
            if key in points:
                coords = points[key][0]
                native[key] = clip_strata(plane.ravel()[coords])
    require(set(native) == set(points), "Every admitted pair from the original native projected probability")
    return native


def counts_for_policy(tail, extreme, actions):
    idx = np.asarray(actions, dtype=np.intp).reshape(-1, 2)
    counts = {}
    for i, name in enumerate(("add", "drop")):
        sat, ex = tail[idx[:, i], i], extreme[idx[:, i], i]
        counts[name + "_clip_tail"] = int(np.count_nonzero(sat))
        counts[name + "_interior"] = int(np.count_nonzero(~sat))
        counts[name + "_exact_zero_or_one"] = int(np.count_nonzero(ex))
        counts[name + "_clip_tail_other_than_exact_zero_or_one"] = int(np.count_nonzero(sat & ~ex))
    return counts


def oracle_policy_values(last, core, base, historical_iou, stock, polygon, geom_values, coords, chosen, occ, shapely):
    fixed_area = int(core.codec.area(stock))
    T = float(polygon.area)
    iou_N = geom_values[0, 0] / 100
    intersection_N = iou_N * (fixed_area + T) / (1 + iou_N)
    add, drop = last.selected_mass(occ, chosen)
    hit = fsum((intersection_N, add, -drop))
    ji = float(historical_iou(hit, fixed_area, T))
    bitmap = core.codec.decode(stock).astype(bool)
    if chosen:
        written = base.mask_edit(bitmap, coords, chosen)
        require(int(core.codec.area(written["rle"])) == fixed_area, "Exact area under constrained oracle")
        require(abs(ji - 100 * core.geometric_iou(written["rle"], polygon, shapely)) < 1e-8,
                "Direct unit-cell overlap matches stratified signed intersection")
    return np.asarray((ji, 100 * (fixed_area - hit) / T, 100 * (T - hit) / T,
                       100 * hit / T, 100 * add / T, 100 * drop / T))


def anchor_geometry(last, core, base, historic_iou, stock, options, points, prior_p, polygon, shapely):
    old_values, _, counts = last.policy_values(core.geometric_iou, historic_iou, core.codec, stock, options,
                                              points, polygon, shapely)
    coords, _, incumbent = points
    tail, extreme = prior_p
    y, x = np.divmod(coords, stock["size"][1])
    occ = np.asarray(shapely.area(shapely.intersection(shapely.box(x, y, x + 1, y + 1), polygon))).reshape(-1, 2)
    within = optimal_in_strata(occ, tail, incumbent)
    freely = last.best_choices(coords, occ, len(incumbent))
    frames = dict(support=[(i, i) for i in range(len(coords))], incumbent=incumbent,
                  stratified_oracle=within, oracle=freely)
    strata = {arm: counts_for_policy(tail, extreme, selected) for arm, selected in frames.items()}
    require(all(strata["incumbent"][key] == strata["stratified_oracle"][key]
                for key in strata["incumbent"] if "_exact" not in key), "Original clipping quotas by both signed sides")
    counts["positive_clipped_support_matches"] = int(len(incumbent) > 0 and tail.any())
    for i, name in enumerate(("add", "drop")):
        if incumbent:
            key = name + "_clip_tail"
            counts[name + "_clip_tail_eligible_matches"] = int(tail[:, i].any())
            counts[name + "_mixed_support_matches"] = int(tail[:, i].any() and not tail[:, i].all())
            counts[name + "_oracle_tail_quota_differs"] = int(strata["incumbent"][key] != strata["oracle"][key])
    middle = oracle_policy_values(last, core, base, historic_iou, stock, polygon, old_values, coords, within, occ, shapely)
    outer = old_values[[2, 3]]
    require(outer[0, 0] - 1e-8 <= middle[0] <= outer[1, 0] + 1e-8, "Nested geometry optima")
    geometry = np.stack((outer[0], middle, outer[1]))
    extra = []
    for actions in (incumbent, within, freely):
        idx = np.asarray(actions, dtype=np.intp).reshape(-1, 2)
        fields = []
        for side in range(2):
            ix = idx[:, side]
            selected = occ[ix, side]
            sat = tail[ix, side]
            fields.extend((100*fsum(selected[sat])/polygon.area, 100*fsum(selected[~sat])/polygon.area))
        extra.append(fields)
    extra = np.asarray(extra)
    require(np.allclose(extra.reshape(3, 2, 2).sum(axis=2), geometry[:, 4:6], rtol=0., atol=1e-10),
            "Four signed stratum mass components exactly partition the original two sides")
    return np.hstack((geometry, extra)), strata, counts


def describe(reserve, last, groups, dates, counts, strata, prior):
    parents = sorted(groups)
    date_keys = [dates[p] for p in parents]
    require(len(parents) == prior["represented_parents"] and len(set(date_keys)) == prior["represented_sensor_date_blocks"],
            "Original clustering and eligibility")
    for name in ("original_matches", "eligible_matches", "positive_matches", "zero_budget_matches", "add_budget", "drop_budget"):
        require(counts.get(name, 0) == prior["counts"].get(name, 0), "Exact previous denominator: " + name)
    matrix = np.stack([np.mean(groups[p], axis=0) for p in parents])
    average = matrix.mean(axis=0)
    for idx in (0, 2):
        old = prior["geometry_points"][ARMS[idx]]
        require(all(abs(average[idx, i] - old[name]) < 1e-8 for i, name in enumerate(last.FIELDS)),
                "Previous geometry reproduced without any alternate support")
    within = matrix[:, 1] - matrix[:, 0]
    between = matrix[:, 2] - matrix[:, 1]
    pool = matrix[:, 2] - matrix[:, 0]
    require(np.allclose(within + between, pool, rtol=0., atol=1e-12), "Additive regret decomposition")
    estimates = {}
    fields = last.FIELDS + STRATUM_FIELDS
    for name, contrast in (("within_bin", within), ("cross_bin", between), ("total", pool),
                           ("cross_minus_within", between - within)):
        sd = reserve.bootstrap(contrast, date_keys)
        pi = reserve.bootstrap(contrast, parents)
        estimates[name] = {
            f: dict(delta=float(mu), sensor_date_interval_95=ci, parent_interval_95=pc)
            for f, mu, ci, pc in zip(fields, contrast.mean(axis=0), sd, pi, strict=True)}
    field = last.FIELDS[0]
    total_iou = estimates["total"][field]["delta"]
    old_iou = prior["contrasts_points"]["oracle minus free_matched_count"][field]
    require(abs(old_iou["delta"] - total_iou) < 1e-8
            and np.allclose(old_iou["sensor_date_interval_95"], estimates["total"][field]["sensor_date_interval_95"],
                            rtol=0., atol=1e-8), "Prior O minus I mean and paired blocked CI")
    cross_iou = estimates["cross_bin"][field]
    difference = estimates["cross_minus_within"][field]["sensor_date_interval_95"]
    return dict(counts=dict(counts), endpoint_quota_counts={key: dict(value) for key, value in strata.items()},
                represented_parents=len(parents), represented_sensor_date_blocks=len(set(date_keys)),
                geometry_points={a:dict(zip(fields, row.tolist(), strict=True))
                                 for a, row in zip(ARMS, average, strict=True)},
                decomposition_points=estimates,
                cross_share_of_total_iou=cross_iou["delta"] / total_iou if total_iou > 1e-12 else None,
                cross_bin_material=(cross_iou["delta"] >= .30 and cross_iou["sensor_date_interval_95"][0] > 0),
                cross_bin_dominance=("cross_bin" if difference[0] > 0 else "within_bin" if difference[1] < 0 else "unresolved"))


def run(a, auth):
    started = perf_counter()
    prior_root = Path(auth["geometry_oracle"])
    last = load(prior_root / "oracle.py", ORACLE_WORKER)
    old_auth = json.loads(pin(prior_root / "AUTHORITY_PRIVATE.json", ORACLE_AUTHORITY))
    root = Path(old_auth["paired_replay"]).resolve(strict=True)
    score = load(root / "score.py", last.SCORE_WORKER_SHA)
    authority = json.loads(pin(root / "AUTHORITY_score.json", last.SCORE_AUTHORITY_SHA))
    path_keys = {"code", "protocol", "capture", "manifest", "prepare", "source_result",
                 "source_terminal", "hrsid_result", "hrsid_terminal", "census", "author_train", "shapely_site"}
    site = SimpleNamespace(**{k: Path(v) if k in path_keys else v for k, v in authority.items()})
    replay = load(root / "replay.py", site.replay_worker_sha)
    prior = replay.closed(prior_root / "observed_oracle/TERMINAL.json", ORACLE_TERMINAL,
                          prior_root / "actual/RESULT_SAFE.json", ORACLE_RESULT, "score", ORACLE_WORKER)
    previous = replay.closed(root / "observed_score/TERMINAL.json", last.SCORE_TERMINAL_SHA,
                             root / "score/RESULT_SAFE.json", last.SCORE_SHA, "score", last.SCORE_WORKER_SHA)
    require(prior["status"] == last.STATUS and prior["authority_sha256"] == ORACLE_AUTHORITY
            and prior["protocol_sha256"] == old_auth["protocol_sha"] == last.PROTOCOL_SHA
            and prior["prelock_commit"] == old_auth["prelock_commit"] == last.PRELOCK_COMMIT
            and prior["frozen_score_sha256"] == last.SCORE_SHA and previous["authority_sha256"] == last.SCORE_AUTHORITY_SHA,
            "Exact naturally closed previous same-support oracle and F/I score")
    require(pin(site.code / TRAINER, TRAINER_SHA).count(b"probabilities.clamp(1e-4,1-1e-4)") == 1,
            "Point-head forward used exactly the native prior FP32 clip constants")
    score.audit_source(site, replay)
    proven, blocks = score.read_closed_stream(site, replay)
    capture, rows = replay.checked_capture(site)
    points, flow = last.exact_points(site, replay, rows, blocks)
    require(dict(flow) == prior["original_candidate_flow"] and flow["native_candidates"]
            == proven["counts"]["native_candidates"] == capture["counts"]["candidates"], "Full closed candidate parity")
    priors = point_priors(site, replay, rows, points)

    reserve = load(site.code / score.RESERVE, score.RESERVE_SHA)
    old = load(site.code / score.HISTORIC, reserve.HISTORIC)
    evaluator = old.module(site.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", old.SCORE_SOURCE)
    adapter = old.module(site.code / "UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", old.GEOMETRY_SOURCE)
    core = adapter.load_geometry(site.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    base = load(site.code / replay.BASE, replay.BASE_SHA)
    historical_oracle = load(site.code / last.PAST, last.PAST_SHA)
    sys.path.append(str(site.shapely_site))
    import shapely
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Original geometry runtime")
    census = replay.json_pin(site.census, reserve.CENSUS)
    dates = {p["paper_scene"]:(p["sensor"], p["calendar_date"]) for p in census["per_parent"]}
    nested = set(census["date_separated_sensitivity"]["parent_scene_ids"])
    # All prior, roster, stock, and policy checks precede the scoped author-reference read.
    gt, _ = old.coco_annotation_ids(reserve.select_labels(pin(site.author_train, reserve.TRAIN), blocks))
    polygons = {v["id"]:shapely.Polygon(np.asarray(v["segmentation"][0]).reshape(-1,2)) for v in gt["annotations"]}
    anchors = last.native_anchors(evaluator, gt, blocks)
    accum = {name:(defaultdict(list), Counter(), defaultdict(Counter)) for name in VIEWS}
    for gid, (i, rank, original) in anchors.items():
        group = blocks[i]["group"]
        views = VIEWS if group in nested else VIEWS[:1]
        for name in views:
            accum[name][1]["original_matches"] += 1
        poly = polygons[gid]
        stock = original["segmentation"]
        if not poly.is_valid or poly.area <= 0 or not 32 <= int(core.codec.area(stock)) < 1024:
            continue
        opts = blocks[i]["changes"].get(rank, {})
        values, bins, counts = anchor_geometry(last, core, base, historical_oracle.iou,
                                              stock, opts, points[(i, rank)], priors[(i, rank)], poly, shapely)
        for name in views:
            geometry, totals, strata = accum[name]
            geometry[group].append(values)
            totals.update(counts)
            for arm, counter in bins.items():
                strata[arm].update(counter)
    report = {name: describe(reserve, last, groups, dates, totals, strata, prior["views"][name])
              for name, (groups, totals, strata) in accum.items()}
    result = dict(status=STATUS, prelock_commit=auth["prelock_commit"], protocol_sha256=auth["protocol_sha"],
                  input_authority_sha256=a.authority_sha, worker_sha256=digest(Path(__file__).read_bytes()),
                  original_oracle_result_sha256=ORACLE_RESULT, original_oracle_terminal_sha256=ORACLE_TERMINAL,
                  frozen_score_sha256=last.SCORE_SHA, native_prior_fp32_clip_extrema=[float(CLIP_LOW), float(CLIP_HIGH)],
                  original_candidate_flow=dict(flow), old_oracle_and_i_geometry_exact=True,
                  fixed_k_and_area_exact=True, model_forwards=0, optimizer_updates=0, AP_computed=False,
                  retrospective=True, author_test_opened=False, bootstrap_draws=4096, bootstrap_seed=5217,
                  views=report, elapsed_seconds=perf_counter()-started)
    out = replay.encoded(result)
    a.output.mkdir()
    (a.output / "RESULT_SAFE.json").write_bytes(out)
    print(json.dumps(dict(status=STATUS, result_sha256=digest(out))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--authority", type=Path, required=True)
    p.add_argument("--authority-sha", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    auth = json.loads(pin(a.authority, a.authority_sha))
    require(set(auth) == {"protocol", "protocol_sha", "prelock_commit", "geometry_oracle"}
            and all(Path(auth[k]).is_absolute() for k in ("protocol", "geometry_oracle"))
            and auth["prelock_commit"] == PRELOCK_COMMIT
            and auth["protocol_sha"] == PROTOCOL_SHA
            and sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11"
            and a.output.name == "actual" and not a.output.exists(), "Fresh qualified no-forward diagnostic")
    pin(Path(auth["protocol"]), auth["protocol_sha"])
    run(a, auth)


if __name__ == "__main__":
    main()

