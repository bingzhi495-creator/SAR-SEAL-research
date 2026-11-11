"""Compare fixed common-support HRSID masks against raster and source polygons."""

import numpy as np


ARMS = ('native', 'center')


def measure_pair(raster, predictions, rings, footprint):
    raster_area = int(np.count_nonzero(raster))
    if not raster_area or tuple(predictions) != ARMS:
        raise ValueError('A nonempty original raster and ordered fixed pair are required')
    measured = footprint(rings, raster.shape, dict(predictions, source_raster=raster))
    polygon_area = measured['visible_polygon_area']
    if not polygon_area > 0:
        raise ValueError('A valid visible source polygon is required')
    result = {'self_polygon_iou': measured['by_arm']['source_raster']['iou']}
    overlap = {rep: {} for rep in ('raster', 'polygon')}
    for arm, predicted in predictions.items():
        intersection = int(np.count_nonzero(predicted & raster))
        union = int(np.count_nonzero(predicted | raster))
        if not 2 * intersection >= union > 0:
            raise ValueError('The fixed original raster match must meet IoU .50')
        continuous = measured['by_arm'][arm]
        pred_area = int(np.count_nonzero(predicted))
        polygon_intersection = polygon_area - continuous['fn_area']
        overlap['raster'][arm] = intersection
        overlap['polygon'][arm] = polygon_intersection
        result[arm] = dict(raster_iou=intersection / union, polygon_iou=continuous['iou'],
                           raster_fp_rate=np.count_nonzero(predicted & ~raster) / raster_area,
                           raster_fn_rate=np.count_nonzero(raster & ~predicted) / raster_area,
                           polygon_fp_rate=continuous['fp_area'] / polygon_area,
                           polygon_fn_rate=continuous['fn_area'] / polygon_area,
                           raster_margin90=19*intersection - 9*(pred_area+raster_area),
                           polygon_margin90=19*polygon_intersection - 9*(pred_area+polygon_area))
    polygon_delta = overlap['polygon']['center'] - overlap['polygon']['native']
    raster_delta = overlap['raster']['center'] - overlap['raster']['native']
    margin_difference = (result['center']['polygon_margin90'] - result['native']['polygon_margin90']
                         - result['center']['raster_margin90'] + result['native']['raster_margin90'])
    result['margin_attribution_error'] = margin_difference - 19*(polygon_delta-raster_delta)
    if abs(result['margin_attribution_error']) > 1e-7:
        raise ArithmeticError('Paired strict-margin attribution must conserve exact area terms')
    return result


THRESHOLDS = (.50, .75, .90)
TOLERANCE = 1e-10
BOOTSTRAP_SEED = 363705
BOOTSTRAP_DRAWS = 4096
PINS = {
    'p360': ('HRSID_COMMON_DEV705_LARGE_FAILURE_20261111/large_core_fn.py',
             '224dd916d4d5672d8aaa9707f8b9233926d4590043ab754ff151b25ce5546acc'),
    'vector_label_geometry': ('HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110/vector_label_geometry.py',
                              '4877c48d38a03e58e97163381ba544df3c90a78dde85c9141085fa0b92da3b09'),
    'vector_population_math': ('HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110/vector_population_math.py',
                               '0821d2cc3b6ca793d33fdfe35b634f3678aad072fc55b86352fa2501e969fc7e'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired_cluster(values, groups):
    names = ('raster', 'polygon', 'polygon_minus_raster')
    require(values.shape == (len(groups), 2) and np.isfinite(values).all(), 'Finite paired representation changes required')
    if not len(values):
        return {}
    group_keys, codes, counts = np.unique(groups, return_inverse=True, return_counts=True)
    values = np.column_stack((values, values[:, 1] - values[:, 0]))
    sums = np.zeros((len(group_keys), len(names)))
    np.add.at(sums, codes, values)
    means = sums / counts[:, None]
    result = {}
    if len(group_keys) >= 2:
        generator = np.random.default_rng(BOOTSTRAP_SEED)
        choices = generator.integers(0, len(group_keys), size=(BOOTSTRAP_DRAWS, len(group_keys)))
        object_draws = sums[choices].sum(axis=1) / counts[choices].sum(axis=1)[:, None]
        group_draws = means[choices].mean(axis=1)
        object_ci = np.quantile(object_draws, (.025, .975), axis=0)
        group_ci = np.quantile(group_draws, (.025, .975), axis=0)
        leave_one = (sums.sum(axis=0) - sums) / (counts.sum() - counts)[:, None]
    for col, name in enumerate(names):
        direction = means[:, col]
        result[name] = dict(object_mean=float(values[:, col].mean()), group_equal_mean=float(direction.mean()),
                            group_signs=[int((direction > TOLERANCE).sum()), int((np.abs(direction) <= TOLERANCE).sum()),
                                         int((direction < -TOLERANCE).sum())],
                            object_bootstrap_95=object_ci[:, col].tolist() if len(group_keys) >= 2 else None,
                            group_equal_bootstrap_95=group_ci[:, col].tolist() if len(group_keys) >= 2 else None,
                            object_leave_one_group_range=[float(leave_one[:, col].min()), float(leave_one[:, col].max())]
                            if len(group_keys) >= 2 else None)
    return result


def summarize(rows):
    if not rows:
        return dict(pairs=0, source_proxy_groups=0)
    keys = ('raster_iou', 'polygon_iou', 'raster_fp_rate', 'polygon_fp_rate', 'raster_fn_rate', 'polygon_fn_rate',
            'raster_margin90', 'polygon_margin90')
    values = {arm: {key: np.array([row[2][arm][key] for row in rows]) for key in keys} for arm in ARMS}
    groups = [row[0] for row in rows]
    pairs = {}
    for metric in ('iou', 'fp_rate', 'fn_rate', 'margin90'):
        matrix = np.column_stack([values['center'][f'{rep}_{metric}'] - values['native'][f'{rep}_{metric}']
                                  for rep in ('raster', 'polygon')])
        pairs[metric] = paired_cluster(matrix, groups)
    self_iou = np.array([row[2]['self_polygon_iou'] for row in rows])
    threshold_summary = {}
    for threshold in THRESHOLDS:
        reps = {}
        for rep in ('raster', 'polygon'):
            a, b = (values[arm][f'{rep}_iou'] for arm in ARMS)
            a_pass, b_pass = a >= threshold, b >= threshold
            reps[rep] = dict(native_at=int(a_pass.sum()), center_at=int(b_pass.sum()),
                             both=int((a_pass & b_pass).sum()), gain=int((~a_pass & b_pass).sum()),
                             loss=int((a_pass & ~b_pass).sum()), neither=int((~a_pass & ~b_pass).sum()),
                             numerically_ambiguous={arm: int((np.abs(v - threshold) <= TOLERANCE).sum())
                                                    for arm, v in zip(ARMS, (a, b), strict=True)}
                             if rep == 'polygon' else {})
        threshold_summary[str(threshold)] = dict(reps, source_raster_vs_polygon_at=int((self_iou >= threshold).sum()))
    return dict(pairs=len(rows), source_proxy_groups=len(set(groups)),
                arms={arm: {key: float(vals.mean()) for key, vals in values[arm].items()} for arm in ARMS},
                paired_center_minus_native=pairs, thresholds=threshold_summary,
                raster_against_source_polygon=dict(mean_iou=float(self_iou.mean()), median_iou=float(np.median(self_iou))),
                margin_attribution_max_residual=max(abs(row[2]['margin_attribution_error']) for row in rows))


def run(args):
    from collections import Counter
    from contextlib import redirect_stdout
    from hashlib import sha256
    from importlib.metadata import version
    from io import StringIO
    import json
    import os
    import sys
    from pathlib import Path
    from types import ModuleType
    from pycocotools import mask
    from pycocotools.coco import COCO

    require(sys.platform == 'win32' and sys.version_info[:3] == (3, 10, 13)
            and sys.flags.isolated == sys.flags.utf8_mode == 1 and sys.dont_write_bytecode
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'Original isolated Windows CPU runtime required')
    require({name: version(name) for name in ('numpy', 'pycocotools', 'scipy')}
            == dict(numpy='1.26.4', pycocotools='2.0.11', scipy='1.15.3'), 'Expected original numeric runtime')
    for path in (args.code_root, args.pair, args.native_arm_receipt, args.center_arm_receipt, args.accepted_result,
                 args.prior_result, args.image_manifest, args.shapely_site):
        require(path.is_absolute() and path.resolve(strict=True) == path, 'Canonical predeclared closed inputs required')
    require(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink(), 'Fresh canonical aggregate path required')

    def deny_network(event, _):
        if event.startswith('socket.'):
            raise PermissionError('Offline closed prediction diagnostic required')

    sys.addaudithook(deny_network)
    sys.path.append(str(args.shapely_site))
    import shapely
    require(Path(shapely.__file__).resolve().is_relative_to(args.shapely_site)
            and (shapely.__version__, shapely.geos_version_string) == ('2.0.2', '3.11.2'),
            'Same prior GEOS/Shapely geometry runtime required')
    mods = {}
    for key, (relative, digest) in PINS.items():
        path = args.code_root / relative
        raw = path.read_bytes()
        require(path.resolve(strict=True) == path and sha256(raw).hexdigest() == digest, 'Pinned ordinary source required')
        mod = ModuleType(key)
        mod.__file__ = str(path)
        exec(compile(raw, str(path), 'exec'), mod.__dict__)
        mods[key] = mod
    p360, vector, math_api = (mods[key] for key in ('p360', 'vector_label_geometry', 'vector_population_math'))
    require((args.pair_sha256, args.training_protocol, args.accepted_result_sha256)
            == (p360.PAIR_SHA, p360.TRAIN_PROTOCOL, p360.PRIMARY_SHA), 'Exact prior accepted A/B lineage required')
    prior_raw, accepted_raw = args.prior_result.read_bytes(), args.accepted_result.read_bytes()
    require(sha256(prior_raw).hexdigest() == p360.BASELINE_SHA
            and sha256(accepted_raw).hexdigest() == p360.PRIMARY_SHA, 'Exact prior score and diagnostic required')
    prior, accepted = json.loads(prior_raw), json.loads(accepted_raw)
    require(accepted['status'] == 'HRSID_MASK_COMMON_FIT_E24_DEV705_INDEPENDENTLY_ACCEPTED_V1'
            and accepted['official_TEST_opened'] is False and accepted['training_seed'] == 5202,
            'Exact accepted source protocol required')
    root = args.code_root
    closed = p360.load(root / p360.P282_PATH, p360.P282_SHA, '_p363_closed_pair')
    grouping = p360.load(root / p360.GROUP_PATH, p360.GROUP_SHA, '_p363_groups')
    gt, native, center, provenance = closed.verified_inputs(root, args.pair, p360.PAIR_SHA,
        native_arm_receipt=args.native_arm_receipt, center_arm_receipt=args.center_arm_receipt,
        training_protocol=p360.TRAIN_PROTOCOL)
    require(provenance == prior['provenance']
            and provenance['training_pair_receipt_sha256'] == accepted['independent_training_pair_sha256']
            and provenance['original_dev_gt_sha256'] == accepted['original_dev_gt_sha256'],
            'Exact closed prediction, GT and prior diagnostic lineage required')
    driver = grouping.load_driver()
    pair = driver.read_json(args.pair, p360.PAIR_SHA)
    group_gt, _, groups = grouping.load_population(driver, dict(path=args.image_manifest.as_posix(),
                                                                 sha256=p360.MANIFEST_SHA), pair['gt'])
    require(group_gt == gt, 'Identical original group and GT population required')
    modules = closed.load_diagnostics(root)
    failure = modules['dev705_paired_mask_failure_scope']
    score = modules['dev705_score_template']
    ordered, counts = score.prefix_rows(score.nonempty_by_image(native, gt['images']),
                                       score.nonempty_by_image(center, gt['images']))
    require(sum(counts) == 41482, 'Exact original equal nonempty prefixes required')
    adapter = failure.load_official_adapter()
    matches = {arm: p360.fixed_matches(gt, ordered[arm], adapter, failure.match_official) for arm in ARMS}
    common = matches['native'].keys() & matches['center'].keys()
    require(len(common) == prior['diagnostics']['paired_mask_failure_scope']['fn_concentration']['all']['paired_gt'] == 1620,
            'Exact previously published common .50 match count required')

    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(StringIO()):
        coco.createIndex()
    rows, skipped, representations, matched_by_stratum = [], Counter(), Counter(), Counter()
    raw_groups = Counter()
    for ann in gt['annotations']:
        aid = ann['id']
        if aid not in common:
            continue
        rle = coco.annToRLE(ann)
        require(int(mask.area(rle)) > 0 and not ann.get('iscrowd', 0) and not ann.get('ignore', 0),
                'Original common matches must be eligible')
        stratum = failure.stratum(rle, ann['area'])
        matched_by_stratum[stratum] += 1
        group = groups[ann['image_id']]
        raw_groups[group] += 1
        segmentation = ann.get('segmentation')
        representation = 'polygon_list' if type(segmentation) is list else 'rle' if type(segmentation) is dict else 'other'
        representations[representation] += 1
        try:
            rings = vector.polygon_rings(segmentation)
        except vector.UnsupportedPolygon as error:
            skipped[f'{representation}: {error}'] += 1
            continue
        raster = mask.decode(rle).astype(bool, copy=False)
        require(np.array_equal(vector.polygon_raster(rings, raster.shape), raster),
                'Original polygon must re-rasterize to exact official association mask')
        predictions = {arm: mask.decode(matches[arm][aid]).astype(bool, copy=False) for arm in ARMS}
        row = measure_pair(raster, predictions, rings, math_api.run_footprint_metrics)
        rows.append((group, stratum, row))
    require(sum(matched_by_stratum.values()) == 1620 == len(rows) + sum(skipped.values()),
            'All original matches must be represented or explicitly excluded')
    scopes = {'all': rows}
    scopes.update({size: [row for row in rows if row[1].split('_', 1)[0] == size] for size in ('small', 'medium', 'large')})
    scopes.update({key: [row for row in rows if row[1] == key] for key in sorted(matched_by_stratum)})
    report = dict(status='HRSID_COMMON_DEV705_CONTINUOUS_POLYGON_FIXED_PAIR_ACTUAL_V1', FIT=False, DEV=True,
                  official_TEST_opened=False, COCO_AP=False, model_forward_calls=0, optimizer_updates=0,
                  geometry='union of original source polygon components clipped to canvas versus predicted unit pixel squares',
                  association='unchanged official raster IoU .50 all-area matches at equal nonempty prefix, both arms',
                  source_annotation_shared_across_representations=True, predictions_edited=False,
                  training_seed=5202, paired_gt=1620, equal_nonempty_prefix_rows_per_arm=sum(counts),
                  eligible_pairs=len(rows), re_rasterized_exact_pairs=len(rows), stock_common_matches_by_stratum=matched_by_stratum,
                  original_source_proxy_groups=len(raw_groups), source_representation_counts=representations,
                  excluded_pair_reasons=skipped, provenance=provenance, primary_accepted_sha256=p360.PRIMARY_SHA,
                  prior_diagnostic_sha256=p360.BASELINE_SHA, image_manifest_sha256=p360.MANIFEST_SHA,
                  geometry_runtime=dict(shapely=shapely.__version__, geos=shapely.geos_version_string),
                  source_sha256={k: sha for k, (_, sha) in PINS.items()},
                  clustered_bootstrap=dict(seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_DRAWS,
                                           cluster='historical source proxy prefix', units='fraction for IoU and error rates'),
                  strata={key: summarize(values) for key, values in scopes.items()})
    result = (json.dumps(report, sort_keys=True, allow_nan=False, indent=2) + '\n').encode('utf8')
    with args.output.open('xb') as stream:
        stream.write(result)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=report['status'], result_sha256=sha256(result).hexdigest(),
                          eligible_pairs=len(rows), excluded_pairs=sum(skipped.values()))), flush=True)


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('code-root', 'pair', 'native-arm-receipt', 'center-arm-receipt', 'accepted-result',
                 'prior-result', 'image-manifest', 'shapely-site', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('pair-sha256', 'training-protocol', 'accepted-result-sha256'):
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
