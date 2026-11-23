"""Score crossed Phoenix/occupancy signed edits on the closed 74-image Instance-RSDD reserve."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
import shapely


LOCK = "f0a154154a18f05d3ff45f42fc6dddfc233c4ef595a9a86534e18d9b15e29ed9"
REPLAY_WORKER = "928092b85ed68cecfc307349d0e0809e7931ca34e67ac75f011bc35fbf8c4172"
PHX_WORKER = "675d84109e4b7c643be1d3724c65d8d088a44c5d56de60c51cf063f0ed6ee76f"
PHX_SCORE = "bdc5553031428a995474aab45c447f4bbf4723cf51b863470f13d1db97709c6b"
PHX_TERM = "fb8f9f5b9eb0c5aa45beab34b26f6e4cbb0d66de3d6df2fdd0af25c014b4579b"
PREPARED = "RSDD74_PHOENIX_SIGNED_SIDE_EDITS_COMPLETE"
STATUS = "RSDD74_PHOENIX_SIGNED_SIDE_SCORE_COMPLETE"
SEEDS = (5217, 5218, 5219)
OLD = ("raw", "dynamic", "static", "phoenix", *(f"{s}/learned" for s in SEEDS))
CROSSED = tuple(f"{seed}/{side}" for seed in SEEDS for side in ("LP", "PL"))
ARMS = (*OLD, *CROSSED)
SIDE = {"addition": "LP", "deletion": "PL"}
GEOMETRY = ("continuous_iou_points", "fp_per_truth_area_points", "fn_per_truth_area_points",
            "intersection_per_truth_area_points", "intersection_px2")
INDEX = {a: i for i, a in enumerate(ARMS)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def load_module(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Frozen module differs: " + path.name)
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def sources(a):
    phx = load_module(a.code / "INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/score.py", PHX_WORKER)
    api = phx.builder_api()
    cfa = api.module(a.code / "INSTANCE_RSDD_CLASSICAL_FAIR_MATCH_20261123/score.py", phx.CFA_SCORER)
    h = cfa.source(a.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123/score_quota_swap.py", cfa.HISTORY_SHA)
    reserve = h.published_module(a.code / "INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/score_reserve.py", h.RESERVE_SCORE)
    old = h.published_module(a.code / "INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py", reserve.HISTORIC)
    evaluator = old.module(a.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", old.SCORE_SOURCE)
    adapter = old.module(a.code / "UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", old.GEOMETRY_SOURCE)
    core = adapter.load_geometry(a.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    return phx, api, cfa, h, reserve, old, evaluator, adapter, core


def checked_inputs(a, phx, api, cfa, adapter, core):
    reference, rsha, rtsha = api.observed(a.phx_score, a.phx_score_terminal, "score", phx.SCORE,
                                          PHX_SCORE, PHX_TERM, PHX_WORKER)
    original, osha, otsha = api.observed(a.phx_edits, a.phx_edits_terminal, "edits", phx.EDIT,
                                         reference["edits_result_sha256"], reference["edits_terminal_sha256"], PHX_WORKER)
    replay, esha, etsha = api.observed(a.replay, a.replay_terminal, "edits", PREPARED, worker=REPLAY_WORKER)
    require(replay["phase"] == "edits" and replay["protocol_sha256"] == LOCK
            and replay["phx_edits_result_sha256"] == osha and replay["phx_edits_terminal_sha256"] == otsha
            and replay["phx_edits_sha256"] == original["edits_sha256"]
            and replay["original_arms_exact"] is replay["per_candidate_signed_counts_exact"] is True
            and replay["cross_context_signed_xor_exact"] is replay["exact_native_fallback"] is True
            and replay["target_labels_opened"] is original["target_labels_opened"] is False
            and replay["GT_opened"] is False
            and replay["detector_forwards"] == replay["optimizer_updates"] == 0
            and original["edits_sha256"] == reference["edits_sha256"] and reference["historical_arms_exact"] is True,
            "Successful label-free replay of the original Phoenix edits required")
    inputs = cfa.source(a.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123/inputs.py", cfa.INPUTS_SHA)
    rows, lineage = inputs.load_population(a)
    require(lineage == replay["input_lineage"] == reference["input_lineage"] == original["input_lineage"],
            "Native capture/roster lineage differs")
    archive = api.pin(a.phx_edits / "EDITS_PRIVATE.jsonl", original["edits_sha256"])
    crossed = api.pin(a.replay / "EDITS_PRIVATE.jsonl", replay["edits_sha256"])
    cfa.EDITED = tuple(arm for arm in ARMS if arm != "raw")

    def check_rows():
        for record, orig, new in zip(rows, map(json.loads, archive.splitlines()),
                                     map(json.loads, crossed.splitlines()), strict=True):
            require(orig["image_id"] == new["image_id"] == record[0]["image_id"]
                    and set(orig["changes"]) == set(new["changes"]), "Historical edit roster differs")
            for position, saved in orig["changes"].items():
                entries = new["changes"][position]
                require(set(saved) == set(OLD) - {"raw"} and all(saved[key] == entries[key] for key in saved)
                        and set(entries) == set(cfa.EDITED), "Original pure control arms differ")
                native = record[2]["coco"][int(position)-1]["segmentation"]
                def signatures(mask):
                    # Intersect/union with native isolate drops/adds and canonicalize the RLE.
                    return tuple(core.codec.merge([native, mask], intersect=flag)["counts"] for flag in (False, True))
                pp_add, pp_drop = signatures(saved["phoenix"]["rle"])
                for seed in SEEDS:
                    ll_add, ll_drop = signatures(saved[f"{seed}/learned"]["rle"])
                    require(signatures(entries[f"{seed}/LP"]["rle"]) == (ll_add, pp_drop)
                            and signatures(entries[f"{seed}/PL"]["rle"]) == (pp_add, ll_drop),
                            "Hybrid signed pixel sets disagree with archived parent masks")
            yield new
    blocks, counts, _ = cfa.checked_blocks(rows, check_rows(), adapter, core)
    require(counts == dict(mapped_candidates=7346, eligible_candidates=4137, positive_budget_candidates=4107,
                           swaps_per_arm=29900), "Complete historical candidate counts required")
    require(replay["counts"] == dict(images=74, mapped_candidates=7346, eligible_planes=4137,
                zero_budget_candidates=7346-4107, positive_budget_candidates=4107, swaps_per_arm=29900),
            "Replay and independent signed-side census disagree")
    chain = dict(phx_score_result_sha256=rsha, phx_score_terminal_sha256=rtsha, phx_edits_result_sha256=osha,
                 phx_edits_terminal_sha256=otsha, phx_edits_sha256=original["edits_sha256"], input_lineage=lineage,
                 replay_result_sha256=esha, replay_terminal_sha256=etsha, replay_worker_sha256=REPLAY_WORKER,
                 edits_sha256=replay["edits_sha256"])
    return blocks, counts, reference, chain


def geometry_values(rle, truth, area, core):
    iou = core.geometric_iou(rle, truth, shapely)
    intersection = iou * (area + truth.area) / (1 + iou)
    return (100*iou, 100*(area-intersection)/truth.area, 100*(truth.area-intersection)/truth.area,
            100*intersection/truth.area, intersection)


def fixed_matches(anchors, scope, polygons, dates, core, cfa):
    groups = defaultdict(list)
    exposures = {side: {k: Counter() for k in (*map(str, SEEDS), "all_seeds")} for side in SIDE}
    counts = Counter(native_matches=len(anchors), eligible_matches=0, invalid_polygon_matches=0, area_excluded_matches=0)
    max_residual = 0.
    for gid, (image, pos, det) in anchors.items():
        truth = polygons[gid]
        if not truth.is_valid or truth.area <= 0:
            counts["invalid_polygon_matches"] += 1
            continue
        native = det["segmentation"]
        area = int(core.codec.area(native))
        if not 32 <= area < 1024:
            counts["area_excluded_matches"] += 1
            continue
        counts["eligible_matches"] += 1
        group = scope[image]["group"]
        edits = scope[image]["changes"].get(pos, {})
        masks = {a: edits[a]["rle"] if a in edits else native for a in ARMS}
        values, cache = {}, {}
        for a, mask in masks.items():
            key = mask["counts"]
            if key not in cache:
                cache[key] = geometry_values(mask, truth, area, core)
            values[a] = cache[key]
        for seed in SEEDS:
            residual = abs(values[f"{seed}/learned"][4] + values["phoenix"][4]
                           - values[f"{seed}/LP"][4] - values[f"{seed}/PL"][4])
            require(residual <= 1e-7, "True intersection failed signed additivity on a fixed hull")
            max_residual = max(max_residual, residual)
        for side, arm in SIDE.items():
            different = [cfa.different_pixels(masks[f"{s}/{arm}"], masks["phoenix"], area, core.codec) for s in SEEDS]
            for seed, changed in zip(SEEDS, different, strict=True):
                if changed:
                    exposures[side][str(seed)][group] += 1
            if all(different):
                exposures[side]["all_seeds"][group] += 1
        groups[group].append([values[a] for a in ARMS])
    exposure = {side: cfa.exposure_summary(counters, dates) for side, counters in exposures.items()}
    return groups, exposure, dict(counts), max_residual


class PairedBootstrap:
    """Resample whole dates while giving each represented parent its original weight."""
    def __init__(self, keys):
        labels = sorted(set(keys))
        self.members = [np.asarray([v == key for v in keys]) for key in labels]
        draws = np.random.Generator(np.random.PCG64(5217)).integers(len(labels), size=(4096, len(labels)))
        self.weights = np.stack([(draws == i).sum(axis=1) for i in range(len(labels))], axis=1)
        self.sizes = self.weights @ np.asarray([m.sum() for m in self.members])

    def effect(self, values, percentiles=(2.5, 97.5)):
        require(values.ndim == 1 and np.isfinite(values).all(), "Finite paired parent contrasts required")
        sums = np.array([values[m].sum() for m in self.members])
        sample = (self.weights @ sums) / self.sizes
        return dict(delta=float(values.mean()), interval=np.percentile(sample, percentiles).tolist(),
                    percentile_limits=list(percentiles))


def arm_pairs(seed):
    lp, pl, ll = (f"{seed}/{v}" for v in ("LP", "PL", "learned"))
    return {"LP_minus_PP": (lp, "phoenix"), "PL_minus_PP": (pl, "phoenix"),
            "LL_minus_PL": (ll, pl), "LL_minus_LP": (ll, lp), "LL_minus_PP": (ll, "phoenix")}


def numeric_contrast(values, seed, label):
    if label == "LL_minus_LP_minus_PL_plus_PP":
        return (values[:, INDEX[f"{seed}/learned"]] - values[:, INDEX[f"{seed}/LP"]]
                - values[:, INDEX[f"{seed}/PL"]] + values[:, INDEX["phoenix"]])
    a, b = arm_pairs(seed)[label]
    return values[:, INDEX[a]] - values[:, INDEX[b]]


def sign_counts(values):
    tolerance = 1e-10
    return dict(improved=int((values > tolerance).sum()), worsened=int((values < -tolerance).sum()),
                tied=int((np.abs(values) <= tolerance).sum()))


def summarize_geometry(groups, exposures, dates):
    parents = sorted(groups)
    require(bool(parents), "At least one fixed eligible hull required in this prequalified reserve view")
    means = np.stack([np.mean(groups[k], axis=0) for k in parents])
    keys = [dates[k] for k in parents]
    bootstrap = PairedBootstrap(keys)
    parent_bootstrap = PairedBootstrap(parents)
    hulls = np.concatenate([np.asarray(groups[k]) for k in parents], axis=0)
    primary, contrasts = {}, {}
    for side, arm in SIDE.items():
        side_values = np.stack([means[:, INDEX[f"{s}/{arm}"], 3] - means[:, INDEX["phoenix"], 3] for s in SEEDS])
        pooled = bootstrap.effect(side_values.mean(axis=0), (1.25, 98.75))
        lo, hi = pooled["interval"]
        per_seed = {str(s): bootstrap.effect(d, (1.25, 98.75)) for s, d in zip(SEEDS, side_values, strict=True)}
        signs = side_values.mean(axis=1)
        exposure = exposures[side]["all_seeds"]
        covered = exposure["matches"] >= 20 and exposure["sensor_date_blocks"] >= 8
        primary[side] = dict(seed_mean=pooled, per_seed=per_seed, exposure=exposures[side], common_coverage=covered,
                             parent_only_sensitivity=parent_bootstrap.effect(side_values.mean(axis=0), (1.25, 98.75)),
                             parent_direction=sign_counts(side_values.mean(axis=0)),
                             all_seed_deltas_positive=bool((signs > 0).all()), all_seed_deltas_negative=bool((signs < 0).all()),
                             occupancy_advantage=bool(covered and lo > 0 and (signs > 0).all()),
                             phoenix_advantage=bool(covered and hi < 0 and (signs < 0).all()))
    directions = {}
    for label in (*arm_pairs(SEEDS[0]), "LL_minus_LP_minus_PL_plus_PP"):
        diffs = np.stack([numeric_contrast(means, s, label) for s in SEEDS])
        hull_diffs = np.stack([numeric_contrast(hulls, s, label) for s in SEEDS])
        directions[label] = {field: dict(seed_mean=sign_counts(hull_diffs[:, :, j].mean(axis=0)),
                               per_seed={str(s): sign_counts(d[:, j]) for s,d in zip(SEEDS, hull_diffs, strict=True)})
                             for j, field in enumerate(GEOMETRY) if j in (0, 3)}
        contrasts[label] = {field: dict(seed_mean=bootstrap.effect(diffs[:, :, j].mean(axis=0)),
                            per_seed={str(s): bootstrap.effect(d[:, j]) for s, d in zip(SEEDS, diffs, strict=True)})
                            for j, field in enumerate(GEOMETRY)}
    avg = means.mean(axis=0)
    return dict(primary=primary, paired_contrasts=contrasts, fixed_hull_direction=directions, direction_tolerance_points=1e-10,
                represented_groups=len(parents), sensor_date_blocks=len(set(keys)),
                group_equal_arms={a: dict(zip(GEOMETRY, avg[INDEX[a]].tolist(), strict=True)) for a in ARMS})


def score_view(ids, gt, blocks, polygons, dates, evaluator, h, core, cfa, reference):
    scope = {i:b for i,b in blocks.items() if i in ids}
    part = dict(gt, images=[r for r in gt["images"] if r["id"] in scope],
                annotations=[r for r in gt["annotations"] if r["image_id"] in scope])
    metrics = {}
    for arm in ARMS:
        output, matched = evaluator.score_one(part, scope, arm, anchors=arm == "raw")
        if arm == "raw":
            anchors = matched
        metrics[arm] = {field: h.available(value) for field, value in output.items()}
    groups, exposures, counts, residual = fixed_matches(anchors, scope, polygons, dates, core, cfa)
    geometry = summarize_geometry(groups, exposures, dates)
    geometry.update(counts=counts, max_intersection_additivity_error_px2=residual)
    require(counts == reference["geometry"]["counts"] and reference["images"] == len(scope)
            and reference["annotations"] == len(part["annotations"])
            and geometry["represented_groups"] == reference["geometry"]["represented_groups"]
            and geometry["sensor_date_blocks"] == reference["geometry"]["sensor_date_blocks"],
            "Native matches or original view census differs")
    for arm in OLD:
        require(metrics[arm] == reference["metrics_fraction"][arm], "Original COCO metrics differ: " + arm)
        expected = reference["geometry"]["group_equal_arms"][arm]
        require(all(abs(v-expected[k]) < 1e-9 for k,v in geometry["group_equal_arms"][arm].items()),
                "Original continuous geometry differs: " + arm)
    contrasts = {}
    for label in arm_pairs(SEEDS[0]):
        per_seed = {}
        for s in SEEDS:
            a, b = arm_pairs(s)[label]
            per_seed[str(s)] = h.difference(metrics[a], metrics[b])
        contrasts[label] = dict(per_seed=per_seed, across_seeds=h.stats(list(per_seed.values())))
    interaction = {}
    for seed in SEEDS:
        pp, lp, pl, ll = (metrics[a] for a in ("phoenix", f"{seed}/LP", f"{seed}/PL", f"{seed}/learned"))
        interaction[str(seed)] = {k: (None if any(row[k] is None for row in (pp, lp, pl, ll)) else
                                      100*(ll[k]-lp[k]-pl[k]+pp[k])) for k in pp}
    contrasts["LL_minus_LP_minus_PL_plus_PP"] = dict(per_seed=interaction, across_seeds=h.stats(list(interaction.values())))
    return dict(images=len(scope), annotations=len(part["annotations"]), source_groups=len({b["group"] for b in scope.values()}),
                metrics_fraction=metrics, coco_contrasts_points=contrasts, geometry=geometry)


def run(a):
    start = perf_counter()
    require(LOCK == a.protocol_sha256 == digest(a.protocol.read_bytes()) and not a.output.exists(),
            "Exact frozen protocol and fresh output required")
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Frozen geometry runtime required")
    phx, api, cfa, h, reserve, old, evaluator, adapter, core = sources(a)
    blocks, counts, reference, chain = checked_inputs(a, phx, api, cfa, adapter, core)
    gt, rekeyed, polygons, sets, dates, labels_sha = h.views_and_labels(a, old, reserve, blocks)
    views, cache = {}, {}
    for name, ids in sets.items():
        key = frozenset(ids)
        if key not in cache:
            cache[key] = score_view(ids, gt, blocks, polygons, dates, evaluator, h, core, cfa, reference["views"][name])
        views[name] = cache[key]
        print(json.dumps(dict(view=name, images=len(ids))), flush=True)
    require(views["full"]["geometry"]["counts"]["eligible_matches"] == 47
            and views["date_separated"]["geometry"]["counts"]["eligible_matches"] == 32, "Archived fixed denominators required")
    result = dict(phase="score", status=STATUS, worker_sha256=digest(Path(__file__).read_bytes()), protocol_sha256=LOCK,
                  **chain, counts=counts, arms=list(ARMS), seeds=list(SEEDS), bootstrap_draws=4096, bootstrap_seed=5217,
                  cluster_unit="sensor/date; equal parent weight", primary_view="full",
                  primary=views["full"]["geometry"]["primary"], simultaneous_percentiles=[1.25,98.75],
                  additivity_tolerance_px2=1e-7, historical_arms_exact=True, source_side_sets_exact=True,
                  target_labels_opened=True, detector_forwards=0, optimizer_updates=0, author_labels_sha256=labels_sha,
                  author_ids_rekeyed=rekeyed, views=views, seconds=perf_counter()-start)
    a.output.mkdir()
    stamp = api.write(a.output / "RESULT_SAFE.json", api.encode(result))
    print(json.dumps(dict(status=STATUS, result_sha256=stamp)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code","stock-root","point-root","phx-edits","phx-edits-terminal","phx-score","phx-score-terminal",
                 "replay","replay-terminal","author-labels","protocol","output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    for k, v in vars(args).items():
        if isinstance(v, Path):
            setattr(args, k, v.resolve(strict=k != "output"))
    args.cohort, args.support_root = 74, None
    run(args)


if __name__ == "__main__":
    main()
