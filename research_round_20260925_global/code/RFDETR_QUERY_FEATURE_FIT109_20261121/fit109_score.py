"""Four-arm FIT source scoring on an already admitted in-memory population.

The caller supplies the original ordered mapped pool and OOF replay count. COCO
sees every mapped stock score and original annotation. Geometry uses stock
nonignored .50 identities among the complete area-eligible ledger.
"""
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pycocotools import mask as codec
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ARMS = ("Z", "F", "Q", "QP")
CONTROLS = ("Z", "F", "QP")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load_helpers(code_root, shapely_site=None):
    """Load ordinary published sources only. Caller supplies original runtime site when needed."""
    from hashlib import sha256
    from types import ModuleType
    code = Path(code_root).resolve(strict=True)
    fit_path = code / "RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/fit_step.py"
    source = fit_path.read_bytes()
    need(sha256(source).hexdigest() == "94e60f5b5557787687d1f228c33c249a1d7889172a225373b97bd789ec8b72b1",
         "Published point truth helper digest")
    fit = ModuleType("_fit109_truth_source")
    fit.__file__ = str(fit_path)
    exec(compile(source, str(fit_path), "exec"), fit.__dict__)
    if shapely_site is None:
        import shapely
        shapely_site = Path(shapely.__file__).parent.parent
    truth = fit.load_geometry(code, Path(shapely_site))
    def published(relative, digest):
        return fit.load(code / relative, digest, "_fit109_" + Path(relative).stem)
    return SimpleNamespace(
        fit=fit, truth=truth,
        geometry=published("RFDETR_MATCHED_POINT_FORMAL_20261120/evaluate_geometry.py",
                           "fdee52489ba2df0e9db147f26d41432a3276638360f2858fb5b8c5b739953a44"),
        shape=published("MASK_RTDETR_DEV705_FIXED_MATCH_GEOMETRY_20261112/fixed_match_geometry.py",
                        "dedd47d5453b2442ff02585c9b4d216917e5ed175e01eae07a354a22710f3a94"),
        sparse=published("RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py",
                         "cb9b8e839de85588fdf116ce5554ed6cadbc20bdbe76e38180561528bfd870d5"),
        official=published("DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/worker.py",
                           "f9fc79403d12d8840c3269b04ba22d7584b8eb5e3c592ead2b6359ad7df65748"))


def same_count_edit(coordinates, probabilities, k):
    """Independent add/drop ranking; ties retain their original pair ordinal."""
    coords = np.asarray(coordinates)
    p = np.asarray(probabilities)
    need(coords.ndim == 2 and coords.shape[1] == 2 and coords.dtype.kind in "iu"
         and p.shape == coords.shape and p.dtype == np.float32 and np.isfinite(p).all()
         and ((p >= 0) & (p <= 1)).all(), "Aligned original endpoint probabilities")
    need(isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_))
         and 0 <= k <= len(coords), "Original replay count in native support")
    k = int(k)
    add = np.argsort(-p[:, 0], kind="stable")[:k]
    drop = np.argsort(p[:, 1], kind="stable")[:k]
    flipped = np.column_stack((coords[add, 0], coords[drop, 1])).reshape(-1)
    return dict(flipped=flipped, adds=k, drops=k, area_delta=0, edits=2*k)


def check_coordinates(coords, source, helpers):
    if not len(coords):
        return
    height, width = source["size"]
    need((coords < height*width).all(), "Original native pixel bounds")
    # Alternating foreground/background RLE; rows use native row-major indices.
    boundaries = np.cumsum(helpers.sparse.decode_runs(source["counts"]), dtype=np.int64)
    need(boundaries[-1] == height*width, "Complete native canvas")
    y, x = np.divmod(coords.astype(np.int64), width)
    values = np.searchsorted(boundaries, x*height+y, side="right") % 2
    need((values[:, 0] == 0).all() and (values[:, 1] == 1).all(), "Native add/drop orientation")


def pooled(gt, stream, official):
    coco = COCO()
    # COCOeval replaces GT polygons with RLE and adds ignore flags in place.
    coco.dataset = deepcopy(gt)
    with redirect_stdout(StringIO()):
        coco.createIndex()
        values = official.evaluate_population(dict(COCO=COCO, COCOeval=COCOeval), coco, stream,
                                              [im["id"] for im in gt["images"]])
    return dict(AP=values[0], AP75=values[2], all12=values)


def paired_geometry(groups, values, helpers):
    names = sorted(set(groups))
    by_group = defaultdict(list)
    for name, value in zip(groups, values, strict=True):
        by_group[name].append(value)
    means = None
    draws = None
    if names:
        means = 100*np.stack([np.mean(by_group[name], axis=0) for name in names])
        rng = np.random.Generator(np.random.PCG64(helpers.geometry.BOOTSTRAP_SEED))
        draws = rng.integers(len(names), size=(helpers.geometry.DRAW_COUNT, len(names)))
    estimates = {arm: None if means is None else float(means[:, i].mean()) for i, arm in enumerate(ARMS)}
    contrasts = {}
    for arm in CONTROLS:
        effect, interval = None, None
        if means is not None:
            delta = means[:, ARMS.index("Q")]-means[:, ARMS.index(arm)]
            effect = float(delta.mean())
            interval = np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()
        contrasts["Q_minus_"+arm] = dict(geometry_delta_points=effect, geometry_ci95_points=interval)
    return estimates, contrasts


def gate(arms, comparisons):
    tests = {}
    for arm in CONTROLS:
        row = comparisons["Q_minus_"+arm]
        ap, geo, interval = (row[k] for k in ("AP_delta_points","geometry_delta_points","geometry_ci95_points"))
        tests["AP_vs_"+arm] = ap is not None and ap >= .30
        tests["geometry_vs_"+arm] = geo is not None and geo >= .20 and interval is not None and interval[0] > 0
    q, f = arms["Q"], arms["F"]
    tests["AP75_vs_F"] = 0 <= f["AP75"] <= q["AP75"] <= 1
    tests["original_strict90_loss_vs_F"] = q["strict90"]["lost"] <= f["strict90"]["lost"]
    return dict(advance=all(tests.values()), checks=tests)


def evaluate_fit109(gt, candidates_by_image, records, predictions, replay_k, helpers, *,
                    expected_images=109, expected_flipped=None):
    """Return aggregate JSON-safe score. Mask AP/AP75 are fractions, contrasts use points.

    records key = (image_id, original 1-based rank, original query_id).
    predictions[key][arm] = aligned FP32 (K,2) occupancy; replay_k[key] =
    admitted old OOF SU swap count. Include all eligible K=0 rows. Optional
    expected_flipped[key][arm] independently verifies ordered head-side choices.
    """
    images = gt["images"]
    ids = [im["id"] for im in images]
    need(len(ids) == len(set(ids)) == expected_images and set(candidates_by_image) == set(ids),
         "Exact original image and mapped candidate populations")
    need([cat["id"] for cat in gt["categories"]] == [1], "Original ship category")
    anns = defaultdict(list)
    for ann in gt["annotations"]:
        need(ann["image_id"] in candidates_by_image and ann["category_id"] == 1, "Original FIT annotations")
        anns[ann["image_id"]].append(ann)
    need(len({ann["id"] for ann in gt["annotations"]}) == len(gt["annotations"]), "Unique annotation IDs")
    points = {helpers.geometry.identity(r): r for r in records}
    need(len(points) == len(records) and set(points) == set(predictions) == set(replay_k)
         and (expected_flipped is None or set(points) == set(expected_flipped)),
         "Exact complete support, OOF probabilities and original replay count identities")
    streams = {arm: [] for arm in ARMS}
    totals = Counter()
    strict = {arm: Counter({t: 0 for t in helpers.geometry.TRANSITIONS}) for arm in ARMS}
    edited = Counter()
    errors = {arm: [] for arm in (*ARMS, "stock")}
    valid_groups, ious = [], []
    seen = set()
    image_for_group = {}
    for image in images:
        iid = image["id"]
        pool = candidates_by_image[iid]
        ranks = [cand["rank"] for cand in pool]
        scores = [cand["score"] for cand in pool]
        need(all(type(rank) is int and 1 <= rank <= 100 for rank in ranks)
             and ranks == sorted(set(ranks)) and len({c["query_id"] for c in pool}) == len(pool)
             and all(type(c["query_id"]) is int and 0 <= c["query_id"] < 164 for c in pool)
             and all(np.isfinite(s) and 0 <= s <= 1 for s in scores) and scores == sorted(scores, reverse=True),
             "Stable original mapped rank, query and score order")
        eligible = []
        for cand in pool:
            need(cand.get("image_id", iid) == iid and cand.get("category_id",1) == 1
                 and cand["rle"]["size"] == [image["height"], image["width"]]
                 and type(cand["area"]) is int and int(codec.area(cand["rle"])) == cand["area"],
                 "Original mapped category, canvas and codec area")
            if 32 <= cand["area"] < 1024:
                key = (iid, cand["rank"], cand["query_id"])
                need(key in points, "Every eligible identity, including unsupported candidates")
                row = points[key]
                group = row["group"]
                need(image_for_group.setdefault(group, iid) == iid, "Original one FIT chip per source prefix")
                eligible.append(row)
                seen.add(key)
        need(len({row["group"] for row in eligible}) <= 1, "One source prefix per FIT image")
        attached, _ = helpers.fit.attach_truth(image, anns[iid], pool, eligible, helpers.truth)
        fixed = {row["rank"]: row for row in attached}
        need(len(fixed) == len(eligible), "Complete fixed-match ledger")
        subgt = dict(categories=gt["categories"], images=[image], annotations=anns[iid])
        modules = helpers.truth.modules
        polygons, _, _ = modules.geometry.polygon_support(subgt, modules.vector)
        annotation_by_id = {a["id"]: a for a in anns[iid]}
        for cand in pool:
            rank, source = cand["rank"], cand["rle"]
            totals["mapped_candidates"] += 1
            row = fixed.get(rank)
            key = (iid, rank, cand["query_id"])
            shape = None
            outputs = {}
            if row is not None:
                need(helpers.geometry.identity(row) == key, "Fixed identity preserved")
                coords = helpers.geometry.point_geometry(row)
                check_coordinates(coords, source, helpers)
                k = replay_k[key]
                need(isinstance(k, (int,np.integer)) and not isinstance(k, (bool,np.bool_))
                     and 0 <= k <= min(8, cand["area"]//10, len(coords)), "Old OOF k within original cap")
                need(set(predictions[key]) == set(ARMS)
                     and (expected_flipped is None or set(expected_flipped[key]) == set(ARMS)),
                     "Exact four equal-count arms")
                role = row["role"]
                totals[role] += 1
                totals["zero_budget"] += int(k == 0)
                totals["unsupported"] += int(len(coords) == 0)
                if role == "valid":
                    totals["valid_unsupported_noop"] += int(len(coords) == 0)
                    valid_groups.append(row["group"])
                    aid = row["matched_gt_id"]
                    if modules.geometry.stratum(annotation_by_id[aid]["area"]) == "small":
                        ref = polygons[aid][0]
                        axes = helpers.shape.hull(ref)
                        if axes[0] >= 2:
                            shape = ref, axes
                            stock_value, _ = helpers.shape.measure(ref, source, axes, modules)
                            errors["stock"].append(stock_value["centroid_abs_short_px"])
            for arm in ARMS:
                rle = source
                if row is not None:
                    edit = same_count_edit(coords, predictions[key][arm], k)
                    if expected_flipped is not None:
                        supplied = np.asarray(expected_flipped[key][arm])
                        need(supplied.ndim == 1 and (not supplied.size or supplied.dtype.kind in "iu")
                             and np.array_equal(edit["flipped"], supplied), "Ordered OOF edit witness")
                    rle = helpers.sparse.xor_rle(source, edit["flipped"])
                    if k:
                        common = int(codec.area(codec.merge([source, rle], intersect=True)))
                        need(int(codec.area(rle)) == cand["area"] and cand["area"]-common == k,
                             "Exact codec area and independently verified binary add/drop counts")
                    else:
                        need(rle is source, "Zero-budget original RLE identity")
                    edited[arm] += edit["edits"]
                    if row["role"] == "valid":
                        geo = helpers.geometry.candidate_geometry(row, edit)
                        outputs[arm] = geo["iou"]
                        transition = helpers.geometry.strict_transition(geo["raw_iou"], geo["iou"], edit["edits"], .90)
                        strict[arm][transition] += 1
                        if shape is not None:
                            measured, _ = helpers.shape.measure(shape[0], rle, shape[1], modules)
                            need(abs(measured["continuous_iou_points"]-100*geo["iou"]) <= 1e-7,
                                 "Sparse fixed-match IoU agrees with continuous footprint")
                            errors[arm].append(measured["centroid_abs_short_px"])
                streams[arm].append(dict(image_id=iid, category_id=1, score=cand["score"], segmentation=rle))
            if outputs:
                ious.append([outputs[a] for a in ARMS])
    need(seen == set(points) and all(len(s) == totals["mapped_candidates"] for s in streams.values())
         and len(ious) == len(valid_groups) == totals["valid"], "Complete unfiltered population and fixed matches")
    means, comparisons = paired_geometry(valid_groups, ious, helpers)
    arms = {}
    reference_errors = np.asarray(errors["stock"])
    for arm in ARMS:
        values = np.asarray(errors[arm])
        need(values.shape == reference_errors.shape, "Same fixed small-slender support")
        # Descriptive failure = absolute short-axis displacement increased vs original raster.
        worse = int((values > reference_errors+helpers.shape.TIE).sum())
        worse_f = int((values > np.asarray(errors["F"])+helpers.shape.TIE).sum())
        arms[arm] = dict(**pooled(gt, streams.pop(arm), helpers.official),
                         source_equal_iou_points=means[arm], strict90=dict(strict[arm]), edits=edited[arm],
                         small_slender=dict(valid_instances=len(values),
                             mean_abs_short_px=float(values.mean()) if len(values) else None,
                             stock_mean_abs_short_px=float(reference_errors.mean()) if len(values) else None,
                             worse_than_stock=worse, worse_than_stock_fraction=worse/len(values) if len(values) else None,
                             worse_than_F=worse_f, worse_than_F_fraction=worse_f/len(values) if len(values) else None))
    for arm in CONTROLS:
        q, control = arms["Q"]["AP"], arms[arm]["AP"]
        comparisons["Q_minus_"+arm]["AP_delta_points"] = 100*(q-control) if q >= 0 and control >= 0 else None
    counts = {k: totals[k] for k in ("mapped_candidates","valid","unmatched","neutral","unsupported",
                                       "valid_unsupported_noop","zero_budget")}
    counts.update(images=len(images), annotations=len(gt["annotations"]), eligible=len(seen),
                  source_groups_with_valid=len(set(valid_groups)))
    return dict(schema="RFDETR-FIT109-query-position-score-v1", counts=counts, arms=arms, comparisons=comparisons,
                bootstrap={"seed": helpers.geometry.BOOTSTRAP_SEED, "draws": helpers.geometry.DRAW_COUNT,
                           "unit": "original contributing source prefix"},
                gate=gate(arms, comparisons))
