"""A source-held-out FIT109 development pilot using the verified historical point fits."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec
from scipy import ndimage


PRIVATE = Path(os.environ["SEAL_FIT109_SOURCE_ROOT"]).resolve(strict=True)
CODE = PRIVATE / "github_release_20261011/research_round_20260925_global/code"
E24 = "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24"
CAPTURE = "HRSID_DFINE_E24_FIT114_AREA_CAPTURE_DRAFT_20261112_01"
POINTS = "HRSID_DFINE_E24_FIT109_PAIRED_POINT_TRAIN_DRAFT_20261112_01"
REFERENCES = {
    "protocol": ("fit109_free_pairing/PROTOCOL.md", "4a55c85ae9e0710846a4a6c8a1ed69acc9b49d8280aefcf3a0219e01b1abd55b"),
    "free": ("fit109_free_pairing/free_pairing.py", "d24b7050be95786fffdee386e1e8a6f87ddcb8df8b69c2171eb752e6b6a35fef"),
    "common": ("fit109_gray_equal_count/equal_count.py", "f26da339c97bb4e22044d85e1cc4abb78e117a6bc3fde69500c08e2fca155ae5"),
    "gray": ("fit109_gray_ablation/gray_ablation.py", "b5ebca62babc8c131e4f7cbc64f9b9b8e0c838d25a61cb450556b9aa4648498a"),
    "pair": ("fit109_paired_point/fit_pair.py", "6a2d1d14c8359f5aed18c54879bed5bf1e10bde575e59e47d3ce4583507f4a11"),
    "pair_math": ("fit109_paired_point/pair_math.py", "8fe7a2c146c08c89f377b1f20b321276a60356de4bdbb27be2a00544dfc77962"),
    "training": ("fit109_paired_point/train_point.py", "3111c34afd791cb15987ed94c5e898de4c3ff2cad61677b616420651335d812c"),
    "base": ("fit109_probability_rank/fit_rank.py", "9ce987e68f996c3ef4146727867de8b56c08bc56b008665128c36854366d6d23"),
    "official": ("fit109_selective_deletion/fit_selective_deletion.py", "aaa08f598877b4043de3b1d09bc107994566506eee6f5a8c9a43002aaa3d0ff0"),
}
INPUTS = {
    "capture_receipt": (CAPTURE + "/CONTROL/EXPORT/RESULT_SAFE.json", "7d9b19175bc445fef6fb1387b287b1c32002d7c74054cc16ae4431963e25f3ad"),
    "accepted": (CAPTURE + "/INDEPENDENT_READBACK_SAFE.json", "cd4c2a6a7859319429546f8b9a2ab31f6cfbf7076569ad44611c51f77a5f1d35"),
    "candidates": (CAPTURE + "/CONTROL/EXPORT/core/CANDIDATES_PRIVATE.jsonl", "c1d4d0dcdebbe47b083e24ef96be5db57d4655caea919703d183ea5544e0c0fa"),
    "prelabel_receipt": (POINTS + "/PRELABEL_ACTUAL/PRELABEL_SAFE.json", "7c763fd75a8870a84426b2e8a30f8b1e4be6eefedd68dd2e2c88b8c617992519"),
    "points_manifest": (POINTS + "/PRELABEL_ACTUAL/POINTS_MANIFEST_PRIVATE.json", "9a0082f5b9bd7288b11b917a35c00b6baf1e4a2acdd95176351a0c57e7ecb275"),
    "wait": (POINTS + "/PREPARE_ORIGINAL_TOOL_WAIT_PRIVATE.json", "dc254ce8ff521a4508d0e68aff8e82371f7047c41121a249dc21c8e3b45bc07a"),
    "fit_gt": ("HRSID_FIT_NATIVE_COCO_LAYOUT_ACTUAL_PRIVATE_20261109_01/train/_annotations.coco.json", "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"),
    "previous": ("HRSID_DFINE_E24_FIT109_INDEPENDENT_GRAY_ABLATION_DRAFT_20261112_01/ACTUAL/RESULT_SAFE.json", "6334218f6e2c93ac6555fc81e9bad9afbff06aea07ca5a3b025c8e2dc0c5a066"),
    "direct_result": ("HRSID_DFINE_E24_FIT109_DIRECT_PAIR_UTILITY_DRAFT_20261112_01/ACTUAL/RESULT_SAFE.json", "bd8c8067eeea9e7efdc9c8f6f442cf1ab3e239daccbda3cb7c860ea70c368586"),
    "historical": ("HRSID_DFINE_E24_FIT109_FREE_ENDPOINT_PAIRING_DRAFT_20261112_01/ACTUAL/RESULT_SAFE.json", "66867e13d7697ac18aff843433d5f2d171773e002a2cf8a1d9bf63da4ccddd7f"),
}
VERSIONS = {"numpy": "1.26.4", "torch": "2.10.0+cu130", "scipy": "1.15.3", "pycocotools": "2.0.11"}
SIMPLE = "simple"
FREE = "free_at_simple_count"
LAYER = "layer1_at_simple_count"
NEW = (SIMPLE, FREE, LAYER)
CONTROLS = (FREE, LAYER)
MODES = ("raw", "fixed", "free_natural", SIMPLE, FREE, LAYER, "free_matched_count")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, digest):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, f"SHA-256 mismatch for {path}")
    return raw


def module(path, digest):
    obj = ModuleType(path.stem)
    obj.__file__ = str(path)
    exec(compile(pinned(path, digest), str(path), "exec"), obj.__dict__)
    return obj


def choose_controls(free, topology, stock, coords, p, drop_layers, natural_count):
    """Only the mask, frozen point support, scores and layer enter the new policies."""
    require(np.isin(drop_layers, (1, 2)).all(), "Original removal layers")
    simple = topology.constrained_pairs(stock, coords, p, natural_count)
    n = len(simple)
    require(n <= natural_count and all(0 <= a < len(p) and 0 <= b < len(p) and p[a, 0] > p[b, 1]
                                       for a, b in simple), "Bounded positive simple exchanges")
    require(all(len({pair[j] for pair in simple}) == n for j in (0, 1)), "Distinct endpoints")
    equal = free.choose_free(p, natural_count, fixed_count=n)
    indices = np.arange(len(p))
    drop_order = np.lexsort((indices, p[:, 1], drop_layers))
    layer = [(a, int(b)) for (a, _), b in zip(equal, drop_order[:n], strict=True)]
    require(len(equal) == len(layer) == n, "Same per-mask count for both controls")
    return {SIMPLE: simple, FREE: equal, LAYER: layer}


def topology_changes(stock, coords, actions):
    """Audit the full foreground within its bounding box plus one exterior background cell."""
    ys, xs = np.nonzero(stock)
    ey, ex = np.divmod(coords, stock.shape[1])
    y0, y1 = min(ys.min(), ey.min()), max(ys.max(), ey.max()) + 1
    x0, x1 = min(xs.min(), ex.min()), max(xs.max(), ex.max()) + 1
    original = np.pad(stock[y0:y1, x0:x1].astype(bool), 1)
    positions = np.stack((ey-y0+1, ex-x0+1), axis=-1)

    def counts(mask):
        return ndimage.label(mask, np.ones((3, 3), dtype=bool))[1], ndimage.label(~mask)[1] - 1

    old_components, old_holes = counts(original)
    cache = {((), ()): (0, 0)}
    result = {}
    for name, pairs in actions.items():
        key = tuple(tuple(sorted(p[j] for p in pairs)) for j in (0, 1))
        if key not in cache:
            edited = original.copy()
            for j, value in ((0, True), (1, False)):
                y, x = positions[list(key[j]), j].T
                require(np.all(edited[y, x] != value), "Declared native endpoint sign")
                edited[y, x] = value
            cf, ch = counts(edited)
            cache[key] = (cf - old_components, ch - old_holes)
        result[name] = cache[key]
    return result


def observe_fit(slots, free, topology, learner, math, previous):
    by_features = {id(s["data"]["features"]): i for i, s in enumerate(slots) if s["data"] is not None}
    observed = set()
    extras = {m: [[] for _ in slots] for m in NEW}
    totals = Counter()
    hist = Counter()
    changes = {m: Counter() for m in MODES}
    original = learner.probabilities

    def observer(model, f, adj, center, scale):
        p = original(model, f, adj, center, scale)
        i = by_features[id(f)]
        require(i not in observed and adj is f, "Single original held-out score per supported candidate")
        observed.add(i)
        s = slots[i]
        cap = min(8, s["area"] // 10, len(p))
        natural = free.choose_free(p, cap)
        fixed = [(v, v) for v in math.choose_pairs(p[:, 0] - p[:, 1], cap)]
        same = free.choose_free(p, cap, fixed_count=len(fixed))
        require(not fixed or natural, "Positive fixed pairs imply a feasible natural pair")
        k = len(natural)
        if not k:
            hist[0, 0] += 1
            return p
        stock = codec.decode(s["row"]["segmentation"])
        coords = s["data"]["coordinates"]
        chosen = choose_controls(free, topology, stock, coords, p, f[:, 1, 2], k)
        for arm in NEW:
            extras[arm][i] = chosen[arm]
        t = len(chosen[SIMPLE])
        hist[k, t] += 1
        totals.update(partially_feasible=int(0 < t < k), no_feasible_exchange=int(t == 0),
                      full_natural_count=int(t == k), unfilled_swaps=k-t)
        row = dict(raw=[], fixed=fixed, free_matched_count=same, free_natural=natural, **chosen)
        top = topology_changes(stock, coords, row)
        require(top[SIMPLE] == (0, 0), "Simple arm must preserve full-mask topology")
        for arm, (components, holes) in top.items():
            changes[arm].update(components_increased=int(components > 0), components_decreased=int(components < 0),
                                holes_increased=int(holes > 0), holes_decreased=int(holes < 0),
                                components_delta=components, holes_delta=holes,
                                any_topology_change=int(bool(components or holes)))
        return p

    learner.probabilities = observer
    try:
        actions, training, usage = free.fit(slots, learner, math, previous)
    finally:
        learner.probabilities = original
    require(len(observed) == len(by_features) == sum(s["data"] is not None for s in slots), "Complete score coverage")
    hist[0, 0] += len(slots) - len(observed)
    actions.update(extras)
    totals.update(original_candidates=len(slots), supported_candidates=len(observed),
                  unsupported_noops=len(slots)-len(observed), natural_positive=sum(n for (k, _), n in hist.items() if k))
    histogram = [{"natural_count": k, "simple_count": t, "candidates": n} for (k, t), n in sorted(hist.items())]
    require(sum(hist.values()) == len(slots), "Every candidate represented")
    return actions, training, usage, dict(totals), histogram, changes


def additional_geometry(slots, actions):
    overlap = {m: Counter() for m in MODES}
    group_ious = defaultdict(list)
    for i, s in enumerate(slots):
        if s["raw"] is None:
            continue
        area, old = s["raw"]
        ious = []
        for m in MODES:
            selected = actions[m][i]
            add = drop = 0.
            if selected:
                indices = np.asarray(selected)
                add = float(s["truth"][indices[:, 0], 0].sum())
                drop = float(s["truth"][indices[:, 1], 1].sum())
            overlap[m].update(added_overlap=add, removed_overlap=drop, swapped_pairs=len(selected))
            intersection = float(np.clip(old + add - drop, 0, min(area, s["area"])))
            ious.append(100*intersection/(area+s["area"]-intersection))
        if actions[SIMPLE][i]:
            group_ious[s["group"]].append(ious)
    for m in MODES:
        overlap[m]["net_overlap"] = overlap[m]["added_overlap"] - overlap[m]["removed_overlap"]
    conditional = {"pairs": sum(map(len, group_ious.values())), "source_groups": len(group_ious), "simple_minus": {}}
    if group_ious:
        grouped = np.array([np.mean(group_ious[g], axis=0) for g in sorted(group_ious)])
        draws = np.random.Generator(np.random.PCG64(5217)).integers(len(grouped), size=(4096, len(grouped)))
        for arm in CONTROLS:
            difference = grouped[:, MODES.index(SIMPLE)] - grouped[:, MODES.index(arm)]
            conditional["simple_minus"][arm] = dict(delta_points=float(difference.mean()),
                interval95_points=np.percentile(difference[draws].mean(axis=1), [2.5, 97.5]).tolist())
    return overlap, conditional


def sources(a):
    for name, (relative, digest) in REFERENCES.items():
        path = a.code / E24 / relative
        pinned(path, digest)
        setattr(a, name, path)
    for name, (relative, _) in INPUTS.items():
        setattr(a, name, a.private_root / relative)
    a.capture = a.capture_receipt.parent
    a.prelabel = a.prelabel_receipt.parent
    a.shapely_site = a.private_root / "HRSID_FIT_FROZEN_VECTOR_OCCUPANCY_DRAFT_PRIVATE_20261111_01/shapely_cp310_runtime"
    free = module(a.free, REFERENCES["free"][1])
    common = free.load(a.common, free.COMMON, "_pilot_common")
    gray = common.load(a.gray, common.GRAY, "_pilot_gray")
    pair = common.load_pair(a.pair, gray.PAIR)
    base = common.load(a.base, pair.BASE, "_pilot_base")
    math = common.load(a.pair_math, gray.PAIR_MATH, "_pilot_math")
    learner = gray.ablation_training(a.training)
    official = common.load(a.official, pair.OFFICIAL, "_pilot_official")
    free.adapt_function(a.training, gray.TRAIN, learner, "gains", None,
                        (("def gains(", "def probabilities("), ("return points[:,0]-points[:,1]", "return points")))
    free.adapted_readouts(a, common, gray, pair)
    common.MODES = gray.MODES = pair.MODES = MODES
    pair.GEOMETRY = (*MODES, "oracle")
    a.prelabel_sha = gray.OLD_PRELABEL
    a.wait_sha = gray.OLD_WAIT
    topology = module(a.helper, a.helper_sha256)
    return SimpleNamespace(free=free, common=common, gray=gray, pair=pair, base=base, math=math,
                           learner=learner, official=official, topology=topology)


def evaluate(a, modules):
    import torch
    m = modules
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only FIT scope")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    historical, previous, direct = [json.loads(pinned(getattr(a, n), INPUTS[n][1]))
                                     for n in ("historical", "previous", "direct_result")]
    require(previous["full_exact_replay"] and direct["exact_natural_bce_replay"] and
            historical["exact_fixed_bce_replay"] and not any(previous[k] for k in ("DEV", "TEST", "advance")),
            "Historical development scope")
    _, blocks = m.base.read_core(a)
    receipt, planned = m.pair.sealed_points(a, m.base, m.math, blocks)
    require(receipt["points_manifest_sha256"] == INPUTS["points_manifest"][1], "Exact frozen feature inventory")
    a.output.mkdir()
    def seal(name, value):
        return m.base.seal(a.output / name, m.base.encoded(value))
    pre = seal("PRELABEL_SAFE.json", dict(status="DFINE_FIT109_SIMPLE_POINT_PRELABEL_V1", FIT=True, DEV=False,
               TEST=False, GT_opened=False, prelock_sha256=a.prelock_sha256, helper_sha256=a.helper_sha256,
               worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
               source_sha256={k: v[1] for k, v in REFERENCES.items()},
               input_sha256={k: v[1] for k, v in INPUTS.items()}, counts=receipt["counts"]))
    gt, annotations = m.pair.read_labels(a, m.base, blocks)
    slots, scope = m.pair.fixed_slots(a, m.base, blocks, planned, gt, annotations)
    actions, training, usage, realization, histogram, top = observe_fit(slots, m.free, m.topology, m.learner,
                                                                         m.math, previous)
    require(training == historical["training"] and scope == historical["match_scope"] and usage == historical["usage"],
            "Historical five-fold optimizer, scope and policy replay")
    coco, budgets = m.gray.materialize(slots, actions, m.base, m.official, gt)
    geometry = m.pair.fixed_geometry(slots, actions, m.math)
    geometry["simple_minus"] = geometry.pop("free_natural_minus")
    transitions, witnesses = m.common.threshold(slots, actions)
    m.common.validate_transitions(transitions, geometry)
    for arm in m.free.MODES:
        require(coco[arm] == historical["cohorts"][arm] and budgets[arm] == historical["budgets"][arm]
                and transitions[arm] == historical["strict90_transitions"][arm] and
                geometry["metrics"][arm] == historical["geometry"]["metrics"][arm],
                f"Historical AP, budgets, original geometry and strict transitions for {arm}")
    require((geometry["pairs"], geometry["source_groups"]) == (412, 93) and
            budgets[SIMPLE] == budgets[FREE] == budgets[LAYER], "Fixed denominator and identical candidate budget")
    overlap, conditional = additional_geometry(slots, actions)
    witness_digest = m.base.seal(a.output / "STRICT_TRANSITIONS_PRIVATE.jsonl",
                                 b"".join(m.base.encoded(r) for r in witnesses))
    contrasts, gates = {}, {}
    for arm in CONTROLS:
        delta = geometry["simple_minus"][arm]
        additional = sum(row["arms"][SIMPLE]["state"] == "lost" and row["arms"][arm]["state"] != "lost"
                         for row in witnesses)
        contrasts[arm] = {k: coco[SIMPLE][k] - coco[arm][k] for k in coco[SIMPLE]}
        contrasts[arm].update(additional_original_strict90_identities_lost=additional,
                              strict90_loss_count_difference=transitions[SIMPLE]["lost"]-transitions[arm]["lost"])
        gates[arm] = dict(AP=contrasts[arm]["AP"] >= .30, continuous_IoU=delta["delta_points"] >= .20,
                          positive_lower95=delta["interval95_points"][0] > 0,
                          AP75_retained=contrasts[arm]["AP75"] >= 0., original_strict90_losses=additional == 0)
    result = dict(status="DFINE_FIT109_SIMPLE_POINT_HISTORICAL_OOF_ACTUAL_V1", FIT=True, DEV=False, TEST=False,
                  detector_forwards=0, detector_optimizer_updates=0, point_optimizer_updates=1000,
                  prelabel_sha256=pre, transition_witness_sha256=witness_digest, versions=VERSIONS,
                  historical_exact_replay=True, fold_training=training, original_usage=usage, match_scope=scope,
                  cohorts=coco, budgets=budgets, realized_counts=realization, realized_count_histogram=histogram,
                  topology=top, fixed_geometry=geometry, simple_active_geometry=conditional,
                  fixed_match_overlap_cells=overlap, strict90_transitions=transitions, simple_minus_controls=contrasts,
                  gates=gates, followup_gate_met=all(all(g.values()) for g in gates.values()))
    digest = seal("RESULT_SAFE.json", result)
    print(json.dumps(dict(status=result["status"], result_sha256=digest, followup_gate_met=result["followup_gate_met"])),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bindings", "check", "run"))
    parser.add_argument("--private-root", type=Path, default=PRIVATE)
    parser.add_argument("--code", type=Path, default=CODE)
    parser.add_argument("--helper", required=True, type=Path)
    parser.add_argument("--helper-sha256", required=True)
    parser.add_argument("--prelock", required=True, type=Path)
    parser.add_argument("--prelock-sha256", required=True)
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    pinned(a.prelock, a.prelock_sha256)
    require(all(importlib.metadata.version(k) == v for k, v in VERSIONS.items()), "Pinned CPU libraries")
    resolved = sources(a)
    if a.mode == "bindings":
        table = {k: dict(path=str(getattr(a, k)), sha256=v[1]) for k, v in {**REFERENCES, **INPUTS}.items()}
        table.update(prelock=dict(path=str(a.prelock), sha256=a.prelock_sha256),
                     helper=dict(path=str(a.helper), sha256=a.helper_sha256),
                     shapely_runtime=dict(path=str(a.shapely_site), version="2.0.2", geos="3.11.2"),
                     versions=VERSIONS)
        print(json.dumps(table, indent=2, sort_keys=True))
    elif a.mode == "check":
        print(json.dumps(dict(status="DFINE_FIT109_SIMPLE_POINT_SOURCE_IMPORT_CHECK_V1", private_files_opened=0,
                              references=len(REFERENCES), versions=VERSIONS)))
    else:
        require(a.output is not None and not a.output.exists(), "Fresh result directory required")
        evaluate(a, resolved)


if __name__ == "__main__":
    main()
