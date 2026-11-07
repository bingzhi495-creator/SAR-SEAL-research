"""Original fixed-pair geometry adapter and predeclared cluster aggregation."""

from copy import deepcopy
from math import fsum


METHODS = ("native", "PLAIN", "FULL")
GT_KINDS = ("integer", "float")
SHARED = ("shared_foreground_FN", "shared_background_FP")
MAIN_CONTRAST = "FULL_minus_PLAIN"
BOOTSTRAP_SEED = 20261107
BOOTSTRAP_DRAWS = 10000
MARGIN_PP = 1.0


def need(ok, message):
    if not ok:
        raise ValueError(message)


def common_delta(geometry):
    errors = geometry["contrasts"][MAIN_CONTRAST]["errors"]
    return sum(errors[key] for key in SHARED) / geometry["regions"]["GT_union"]


def cluster_interval(np, control, image_ids, rows, verified_scene_by_image):
    if verified_scene_by_image is None:
        cluster_by_image = {image: image for image in image_ids}
        unit = "image"
    else:
        need(type(verified_scene_by_image) is dict and set(verified_scene_by_image) == set(image_ids)
             and all(type(value) is str and value and value.strip() == value
                     for value in verified_scene_by_image.values()), "Complete externally verified scene mapping")
        cluster_by_image = verified_scene_by_image
        unit = "scene"
    cluster_ids = tuple(dict.fromkeys(cluster_by_image[image] for image in image_ids))
    grouped = {cluster: [] for cluster in cluster_ids}
    for row in rows:
        grouped[cluster_by_image[row["image_id"]]].append(common_delta(row["geometry"]))
    numerators = np.array([fsum(grouped[cluster]) for cluster in cluster_ids], dtype=np.float64)
    counts = np.array([len(grouped[cluster]) for cluster in cluster_ids], dtype=np.int64)
    nonempty = int(np.count_nonzero(counts))
    result = dict(point_pp=100 * fsum(common_delta(row["geometry"]) for row in rows) / len(rows),
                  sampling_unit=unit, clusters=len(cluster_ids), clusters_with_pairs=nonempty,
                  seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS, draws_executed=0, valid_draws=0,
                  interval_method="percentile_2.5_97.5_linear", CI95_pp=None, margin_pp=MARGIN_PP,
                  margin_decision="UNRESOLVED_INSUFFICIENT_CLUSTERS")
    if nonempty < 2:
        return result
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    values = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    valid_draws = 0
    for first in range(0, BOOTSTRAP_DRAWS, 500):
        control.check()
        stop = min(first + 500, BOOTSTRAP_DRAWS)
        sampled = rng.integers(0, len(cluster_ids), size=(stop - first, len(cluster_ids)))
        total = numerators[sampled].sum(axis=1)
        sampled_count = counts[sampled].sum(axis=1)
        valid = sampled_count > 0
        valid_draws += int(np.count_nonzero(valid))
        values[first:stop] = np.divide(total, sampled_count, out=np.full(stop - first, np.nan), where=valid)
    result.update(draws_executed=BOOTSTRAP_DRAWS, valid_draws=valid_draws,
                  margin_decision="UNRESOLVED_UNDEFINED_REPLICATES")
    if valid_draws == BOOTSTRAP_DRAWS:
        low, high = (float(value) for value in np.quantile(values * 100, [0.025, 0.975], method="linear"))
        decision = "DOES_NOT_CLEAR_MARGIN"
        if high < -MARGIN_PP:
            decision = "FEWER_COMMON_ERRORS_BEYOND_MARGIN"
        elif low > MARGIN_PP:
            decision = "MORE_COMMON_ERRORS_BEYOND_MARGIN"
        result.update(CI95_pp=[low, high], margin_decision=decision)
    return result


def measure(values, areas):
    pixels = sum(values)
    return dict(pixels=pixels, instance_mean_pp=100 * fsum(v/a for v, a in zip(values, areas)) / len(areas),
                pixel_pooled_pp=100 * pixels / sum(areas))


def describe(rows):
    if not rows:
        return None
    geometries = [row["geometry"] for row in rows]
    areas = [geometry["regions"]["GT_union"] for geometry in geometries]
    regions = {key: sum(g["regions"][key] for g in geometries) for key in geometries[0]["regions"]}
    methods, contrasts = {}, {}
    for name in METHODS:
        items = [g["methods"][name] for g in geometries]
        methods[name] = dict(
            errors={key: measure([item["errors"][key] for item in items], areas) for key in items[0]["errors"]},
            band_coverage_pixels=dict(integer_only=sum(x["errors"]["float_band_FP"] for x in items),
                                      float_only=sum(x["errors"]["integer_band_FP"] for x in items)),
            iou75_pass={gt: sum(item["iou"][gt]["at_least_0_75"] for item in items) for gt in GT_KINDS})
    for name in geometries[0]["contrasts"]:
        items = [g["contrasts"][name] for g in geometries]
        crossings = {gt: [item["iou75_crossing"][gt] for item in items] for gt in GT_KINDS}
        contrasts[name] = dict(
            errors={key: measure([item["errors"][key] for item in items], areas) for key in items[0]["errors"]},
            shared_error_total=measure([sum(item["errors"][key] for key in SHARED) for item in items], areas),
            iou75_crossing={gt: dict(up=values.count(1), down=values.count(-1), unchanged=values.count(0))
                            for gt, values in crossings.items()})
    return dict(regions=regions, methods=methods, contrasts=contrasts)


def summarize_pairs(np, control, *, image_ids, all_annotation_ids, rows,
                    verified_scene_by_image=None, predeclared_subgroups=None):
    """Aggregate complete fixed pairs, retaining image clusters with no pairs."""
    images, all_gt = set(image_ids), set(all_annotation_ids)
    paired = [row["GT_annotation_id"] for row in rows]
    need(len(images) == len(image_ids) and len(all_gt) == len(all_annotation_ids) and rows
         and len(set(paired)) == len(paired) and set(paired) <= all_gt
         and all(row["image_id"] in images and row["geometry"]["regions"]["GT_union"] > 0 for row in rows),
         "Unique original image and annotation coverage")
    for row in rows:
        geometry = row["geometry"]
        region = geometry["regions"]
        for method in geometry["methods"].values():
            errors = method["errors"]
            need(errors["integer_band_FN"] + errors["float_band_FP"] == region["integer_only"]
                 and errors["float_band_FN"] + errors["integer_band_FP"] == region["float_only"],
                 "Both GT error directions must exactly partition the disagreement band")
    scope = dict(images=len(images), total_GT=len(all_gt), paired_GT=len(rows), unpaired_GT=len(all_gt)-len(rows))
    primary = cluster_interval(np, control, image_ids, rows, verified_scene_by_image)
    subgroups = {}
    if predeclared_subgroups is not None:
        need(type(predeclared_subgroups) is dict, "Explicit predeclared subgroup membership")
        for name, members in predeclared_subgroups.items():
            need(type(name) is str and name and name.strip() == name
                 and type(members) in (list, tuple) and all(type(aid) is int for aid in members)
                 and len(set(members)) == len(members) and set(members) <= all_gt,
                 "Predeclared subgroup must name exact original annotation IDs")
            selected = set(members)
            included = [row for row in rows if row["GT_annotation_id"] in selected]
            subgroups[name] = dict(scope=dict(total_GT=len(selected), paired_GT=len(included),
                                              unpaired_GT=len(selected)-len(included)),
                                   descriptive=describe(included))
    return dict(scope=scope, primary_FULL_minus_PLAIN=primary, descriptive=describe(rows), subgroups=subgroups)


def select_fixed_pairs(np, evaluation, native, image_ids, all_annotation_ids, accepted_pairs, original_selector):
    """Run the original COCO .50 selector and verify the accepted pair identities."""
    area_ranges = []
    for params in (evaluation.params, evaluation._paramsEval):
        labels = list(params.areaRngLbl)
        need(params.iouType == "segm" and set(params.imgIds) == set(image_ids)
             and len(params.imgIds) == len(image_ids) and list(params.catIds) == [1]
             and list(params.maxDets) == [1, 10, 100] and np.array_equal(params.iouThrs, np.linspace(.5, .95, 10))
             and labels.count("all") == 1, "Original actual native COCO settings")
        area_ranges.append(list(params.areaRng[labels.index("all")]))
    area = area_ranges[0]
    need(area == area_ranges[1], "Same actual all-area COCO evaluation")
    annotations = evaluation.cocoDt.anns
    need(len(native) == 100 * len(image_ids) and set(annotations) == set(range(1, len(native) + 1)),
         "Exact original COCO loadRes ordinal identities")
    for ordinal, record in enumerate(native, 1):
        annotation = annotations[ordinal]
        need(annotation["id"] == ordinal and annotation["image_id"] == record["image_id"]
             and annotation["query_id"] == record["query_id"] and annotation["category_id"] == record["category_id"] == 1
             and float(annotation["score"]).hex() == float(record["score"]).hex(),
             "Exact original native query/category/score at each COCO ordinal")
    considered, ignored = [], 0
    for row in evaluation.evalImgs:
        if (row is not None and list(row["aRng"]) == area and row["maxDet"] == 100 and row["category_id"] == 1):
            considered.extend(int(value) for value in row["gtIds"])
            ignored += sum(int(value) for value in row["gtIgnore"])
    need(len(considered) == len(all_annotation_ids) and set(considered) == set(all_annotation_ids),
         "All original GT identities considered by native matching")
    pairs = original_selector(evaluation.evalImgs, annotations, area)
    need(type(accepted_pairs) is list and len(accepted_pairs) == len(pairs), "Complete accepted native pair identities")
    image_index = {image: index for index, image in enumerate(image_ids)}
    for pair, saved in zip(pairs, accepted_pairs):
        need(all(type(saved[key]) is int for key in
                 ("image_id", "GT_annotation_id", "native_dt_id", "query_id", "dataset_index"))
             and pair == dict(image_id=saved["image_id"], GT_annotation_id=saved["GT_annotation_id"],
                              dt_id=saved["native_dt_id"], query_id=saved["query_id"])
             and saved["dataset_index"] == image_index[pair["image_id"]],
             "Exact ordered accepted native pair identities")
    return pairs, ignored


def make_coco(loader, ground_truth):
    coco = loader.COCO()
    coco.dataset = deepcopy(ground_truth)
    coco.createIndex()
    return coco


def evaluate_fixed_pairs(np, control, *, shards, native, integer_gt, floating_gt, accepted_integer_summary,
                         accepted_integer_pairs, modules, verified_scene_by_image=None, predeclared_subgroups=None):
    """Consume only admitted E50 docs. Root owns input bytes, process and result acceptance."""
    expected_modules = {"population_core", "population_contract", "eval_body", "official",
                        "coco_loader", "cocoeval", "mask_api", "gt_sensitivity",
                        "geometry_diagnostic", "fixed_pair_core"}
    need(type(modules) is dict and set(modules) == expected_modules, "Exact original geometry dependency roles")
    control.check()
    core = modules["population_core"]
    core._shards(shards)
    original = accepted_integer_summary
    need(original["schema"] == "U67-DGX-fixed50-population-metrics-pending-v1"
         and original["status"] == "METRICS_COMPLETE_PENDING_ROOT_ACCEPTANCE"
         and core.signature(original["endpoint"]) == core.signature(shards[0]["endpoint"])
         and original["AP"] is original["DEV"] is True
         and original["TEST"] is original["new_training"] is original["qualified"] is False
         and (original["images"], original["GT"], original["queries"], original["fixed_epoch"],
              original["updates_per_arm"]) == (187, 380, 18700, 50, 37050),
         "Original accepted E50 summary document and endpoint")
    modules["gt_sensitivity"].validate_GT_pair(integer_gt, floating_gt)
    ci, cf = (make_coco(modules["coco_loader"], ground_truth) for ground_truth in (integer_gt, floating_gt))
    ids, all_gt = list(ci.imgs), list(ci.anns)
    need(len(ids) == 187 and ids == list(cf.imgs) and len(all_gt) == 380 and all_gt == list(cf.anns)
         and set(ci.cats) == set(cf.cats) == {1} and type(native) is list and len(native) == 18700
         and [native[i*100]["image_id"] for i in range(187)] == ids,
         "Exact original ordered DEV187, both 380-GT variants and all native queries")
    ranges = core.RANGES
    merge = modules["population_contract"].merge_predictions

    def joined(groups):
        return merge([dict(dataset_range=span, predictions=rows) for span, rows in zip(ranges, groups)], native, ranges)

    need(core.signature(joined([shard["identity_predictions"] for shard in shards])) == core.signature(native),
         "Whole original zero-residual native prediction population")
    predictions = dict(native=native)
    for arm in ("PLAIN", "FULL"):
        predictions[arm] = joined([shard["predictions"][arm] for shard in shards])
    ordinals = {(record["image_id"], record["query_id"]): index for index, record in enumerate(native)}
    need(len(ordinals) == 18700, "Unique original native image/query identity")
    control.check()
    metrics, _, native_eval = modules["eval_body"].official_metrics(
        np, control, modules["official"], modules["cocoeval"].COCOeval, ci, native, ids)
    need(core.signature(metrics) == core.signature(original["results"]["native_official"]["COCO"]),
         "Exact accepted native integer COCO12 before selecting geometry pairs")
    geometry = modules["geometry_diagnostic"]
    pairs, ignored = select_fixed_pairs(np, native_eval, native, ids, all_gt, accepted_integer_pairs,
                                       geometry.select_native_pairs)
    need(len(pairs) == 374 and ignored == 0, "Original 374 fixed pairs, zero ignored and six unmatched")
    del native_eval
    masks, analyze = modules["mask_api"], modules["fixed_pair_core"].analyze_pair
    rows = []
    for pair in pairs:
        control.check()
        image, aid = pair["image_id"], pair["GT_annotation_id"]
        original_image = ci.imgs[image]
        shape = (original_image["height"], original_image["width"])
        ordinal = ordinals[image, pair["query_id"]]
        need(pair["dt_id"] == ordinal + 1 and ci.anns[aid]["image_id"] == cf.anns[aid]["image_id"] == image
             and ci.anns[aid]["category_id"] == cf.anns[aid]["category_id"] == 1,
             "Unchanged annotation/query pair across both original grids")
        gt_arrays = [coco.annToMask(coco.anns[aid]) != 0 for coco in (ci, cf)]
        pred_arrays = [geometry._decode(np, masks, predictions[method][ordinal]["segmentation"], shape)
                       for method in METHODS]
        arrays = gt_arrays + pred_arrays
        need(all(array.shape == shape for array in arrays), "All five masks use the common original pixel grid")
        result = analyze(*[(shape, array.tobytes(order="C")) for array in arrays])
        native_iou = result["methods"]["native"]["iou"]["integer"]
        need(2 * native_iou["intersection"] >= native_iou["union"], "Original native raster pair reproduces IoU .50")
        rows.append(dict(pair, geometry=result))
    summary = summarize_pairs(np, control, image_ids=ids, all_annotation_ids=all_gt, rows=rows,
                              verified_scene_by_image=verified_scene_by_image, predeclared_subgroups=predeclared_subgroups)
    summary["scope"].update(ignored_GT=ignored, unmatched_GT=380-374-ignored)
    summary.update(schema="Root-E50-DEV187-dual-GT-fixed-pair-geometry-v1",
                   status="GEOMETRY_COMPLETE_PENDING_ROOT_ACCEPTANCE", native_integer_COCO12_exact=True,
                   original_pair_identities_exact=True, paired_geometry_is_not_AP=True,
                   new_model_AP=False, qualified=False, TEST=False, new_training=False,
                   formal_training_budget_credit=0, shared_background_relative_to_each_instance=True)
    paired_gt = {pair["GT_annotation_id"] for pair in pairs}
    control.check()
    return dict(summary=summary, private_pair_records=rows,
                private_unmatched_annotation_ids=[aid for aid in all_gt if aid not in paired_gt])
