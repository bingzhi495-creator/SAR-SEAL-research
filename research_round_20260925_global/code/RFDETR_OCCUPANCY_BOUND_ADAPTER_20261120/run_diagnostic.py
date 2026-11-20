"""Geometry-only diagnostic after the exact closed RF-DETR DEV705 score."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from time import perf_counter
from types import ModuleType

ORIGINAL_LOADER_SHA = '3a77e660fd31570cf1f4b44e818b28cc6743d414591216a5f274a11139955b05'
ORIGINAL_SCORE_SHA = 'e0d9f9f8e6d1878f9e7d61278a54ddd9bfef88314796ac558b4bb675abbd225c'
ORIGINAL_RECEIPT_SHA = 'd3659561c411709aa32842f58416eb59db4925242d1c402c4e1b09a95dc10e2e'
ADAPTER_SHA = 'e094367dd03dae02b98406a74d5700f30c44574369cf61ea3d0646ef2e80ade8'
ORACLE_SHA = '16e5929035d2e95f23b240d755bf9c9a80b3c53fd5eb6570467ad975b454edcf'
STATUS = 'RFDETR_CENTER5202_DEV705_OCCUPANCY_BOUND_DIAGNOSTIC_COMPLETE'
MIB = 1024**2


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def pinned(path, expected, limit):
    need(digest(expected), 'Explicit expected SHA256')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit and sha256(raw).hexdigest() == expected, 'Exact bounded input ' + path.name)
    return raw


def validate_score_closure(score, receipt, score_sha, receipt_sha):
    need(score_sha == ORIGINAL_SCORE_SHA and receipt_sha == ORIGINAL_RECEIPT_SHA,
         'Exact original score and sole-terminal receipt pins')
    need(set(receipt) == {'launch', 'terminal', 'observed_score_footer', 'scope', 'verified_report_sha256'}
         and receipt['scope'] == 'exact sole original score natural exit'
         and type(receipt['launch']['session_id']) is int and receipt['launch']['session_id'] == 74246
         and type(receipt['terminal']['exit_code']) is int and receipt['terminal']['exit_code'] == 0,
         'Exact sole original score session and natural exit0')
    footer = receipt['observed_score_footer']
    need(score['status'] == footer['status'] == 'RFDETR_CENTER5202_DEV705_MATCHED_SCORE_ACTUAL'
         and receipt['verified_report_sha256'] == footer['result_sha256'] == score_sha
         and footer['gate'] == score['gate'], 'Actual observed footer equals original score and complete gate')
    gate = score['gate']
    need(gate['schema'] == 'RFDETR-fixed-point-refinement-gate-v1' and gate['primary'] == 'SU'
         and gate['seeds'] == [5217, 5218, 5219] and gate['advance'] is False
         and gate['gates']['AP_vs_LW=S'] is False and gate['gates']['geometry_vs_LW=S'] is False,
         'Precisely the two observed prelocked LW blockers')
    need(score['DEV_GT_opened'] is True and digest(score['prelabel_sha256'])
         and digest(score['private_shape_witness_sha256']), 'Closed DEV score and original private geometry identity')
    return dict(original_score_closed=True, lw_ap_blocker=True, lw_geometry_blocker=True)


def load_original(path):
    raw = pinned(path, ORIGINAL_LOADER_SHA, MIB)
    module = ModuleType('_unchanged_admitted_score_loader')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def validate_score_prelabel(prelabel, args, approved, original):
    need(prelabel['status'] == 'RFDETR_CENTER5202_DEV705_SCORE_PRELABEL'
         and prelabel['verified_readout_sha256'] == args.readout_accepted_sha
         and prelabel['sparse_sha256'] == approved['edits_sha256']
         and prelabel['target_native_sha256'] == original.STOCK
         and prelabel['previous_native_score_sha256'] == original.OLD_SCORE
         and prelabel['images'] == 705 and prelabel['DEV'] is True
         and all(prelabel[key] is False for key in ('FIT', 'TEST', 'GT_opened', 'AP')),
         'Exact closed score prelabel links to the same admitted readout')


def join_prepared(np, capture_rows, sparse_rows, readout):
    """Join already byte-admitted metadata without reconstructing any variant."""
    need(len(capture_rows) == len(sparse_rows), 'Complete equal-length input streams')
    result, seen = [], set()
    for ordinal, (capture, sparse) in enumerate(zip(capture_rows, sparse_rows, strict=True), 1):
        image_id = capture['native_block']['image_id']
        candidates = capture['pool']
        hw = sparse['original_hw']
        need(type(image_id) is int and image_id not in seen and sparse['image_id'] == image_id
             and sparse['ordinal'] == ordinal and sparse['group'] == capture['source_prefix']
             and sparse['mapped_identity_sha256'] == readout.mapped_identity_sha256(candidates)
             and len(hw) == 2 and all(type(n) is int and n > 0 for n in hw)
             and set(sparse['seeds']) == {str(s) for s in readout.SEEDS},
             'Same admitted original image, group, candidate digest and seeds')
        seen.add(image_id)
        eligible = [row for row in candidates if 32 <= row['area'] < 1024]
        points = sparse['points']
        need(len(points) == len({p['rank'] for p in points})
             and {p['rank']:p['query_id'] for p in points} == {c['rank']:c['query_id'] for c in eligible},
             'Complete admitted area-eligible roster, including unsupported no-ops')
        coordinates = {}
        for point in points:
            raw = np.asarray(point['coordinates'])
            if raw.size == 0:
                raw = np.empty((0, 2), dtype=np.int64)
            need(raw.ndim == 2 and raw.shape[1] == 2 and raw.dtype.kind in 'iu' and len(raw) <= 32
                 and np.all((raw >= 0) & (raw < hw[0]*hw[1])) and np.unique(raw).size == raw.size,
                 'Original disjoint bounded point indices')
            need(point['rank'] not in coordinates, 'Unique original point candidate')
            coordinates[point['rank']] = raw
        result.append(dict(native=capture['native_block'], pool=candidates, sparse=sparse, points=coordinates))
    return result


def load_dev_geometry_inputs(native):
    gt = native.dev_file('DEV_ORIGINAL_GT_PRIVATE.json')
    findings = native.dev_file('DEV_EMPTY_RASTERS_PRIVATE.json')
    empty_ids = {row['annotation_id'] for row in findings if row['kind'] == 'EMPTY_COCO_RASTER'}
    need(len(empty_ids) == len(findings) == 4, 'Original four empty raster IDs remain in primary GT')
    return gt, findings


def compute_geometry_only(np, gt, prepared, original_score, *, eligibility, scorer,
                          fit_step, helpers, geometry, adapter, oracle):
    result = adapter.from_prepared(np, gt, prepared, eligibility=eligibility, score_module=scorer,
                                   fit_step=fit_step, geometry_helpers=helpers,
                                   geometry_module=geometry, oracle_module=oracle)
    observed = json.loads(json.dumps(result['observed_geometry_summary'], allow_nan=False))
    counts = json.loads(json.dumps(result['stock_matching_counts'], allow_nan=False))
    need(observed == original_score['geometry']['summary']
         and counts == original_score['geometry']['stock_matching_counts'],
         'Recomputed original observed geometry and stock matching must match exactly')
    need(result['AP_computed'] is False and result['model_executed'] is False,
         'Geometry-only adapter output')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'capture-accepted', 'image-manifest', 'readout-root', 'readout-accepted',
                 'historical-native', 'historical-score', 'original-dev-dir', 'code', 'spot-source',
                 'shapely-site', 'output', 'formal-source', 'original-score', 'original-score-receipt',
                 'adapter-source', 'oracle-source'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('capture-accepted-sha', 'readout-accepted-sha', 'original-score-sha', 'original-score-receipt-sha'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--probability-repair', type=Path, required=True)
    parser.add_argument('--probability-repair-sha', required=True)
    args = parser.parse_args(argv)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'CPU-only geometry diagnostic')
    original_score = json.loads(pinned(args.original_score, args.original_score_sha, 8*MIB))
    original_receipt = json.loads(pinned(args.original_score_receipt, args.original_score_receipt_sha, MIB))
    eligibility = validate_score_closure(original_score, original_receipt,
                                         args.original_score_sha, args.original_score_receipt_sha)
    need(args.output.is_absolute() and not args.output.exists() and args.output.parent.is_dir(),
         'New exclusive diagnostic output under an existing parent')
    original = load_original(args.formal_source / 'run_score_dev.py')
    fit_step = original.load_fit_step(args.spot_source.resolve(strict=True) / 'fit_step.py')
    spot = fit_step.load(args.spot_source / 'run_spot.py', original.SOURCE_SPOT, '_rf_original_slots')
    import numpy as np
    import torch
    from pycocotools import mask as codec
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start = perf_counter()
    readout = fit_step.load(args.formal_source / 'run_dev_readout.py', original.SOURCE_READOUT, '_rf_label_free_reader')
    scorer = fit_step.load(args.formal_source / 'score_dev.py', original.MATHEMATICS, '_rf_fixed_match_interface')
    geometry = fit_step.load(args.formal_source / 'evaluate_geometry.py', original.GEOMETRY, '_rf_matched_geometry')
    adapter = fit_step.load(args.adapter_source, ADAPTER_SHA, '_rf_occupancy_adapter')
    oracle = fit_step.load(args.oracle_source, ORACLE_SHA, '_rf_occupancy_oracle')
    native = fit_step.load(args.code / 'HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py',
                           original.NATIVE_SOURCE, '_original_native_rfd')
    native.DEV = args.original_dev_dir

    # The following original prelabel admission and capture-loader calls retain
    # their exact source pins and fields. No score/reconstruction call follows.
    approved = json.loads(spot.pinned(args.readout_accepted, args.readout_accepted_sha, MIB))
    spot.need(approved['status'] == 'RFDETR_CENTER5202_DEV705_MATCHED_READOUT_INDEPENDENT_ACCEPTED'
              and approved['DEV'] and not any(approved[k] for k in ('FIT', 'TEST', 'AP', 'GT_opened'))
              and approved['images'] == 705 and approved['original_exit'] == 0
              and approved['capture_accepted_sha256'] == args.capture_accepted_sha
              and all(approved[key] for key in ('exact_candidate_identity', 'exact_native_support',
                                                'exact_per_candidate_budgets', 'actual_XOR_area_RLE_verified')),
              'Only the independently verified label-free 705-image readout')
    stage = json.loads(spot.pinned(args.readout_root / 'RESULT_SAFE.json', approved['readout_result_sha256'], MIB))
    spot.need(stage['status'] == readout.STATUS and stage['capture_accepted_sha256'] == args.capture_accepted_sha
              and stage['edits_sha256'] == approved['edits_sha256']
              and not any(stage[k] for k in ('FIT', 'TEST', 'GT_opened', 'AP')),
              'Exact original prelabel sparse materialization')
    original.verify_readout_pair(approved, stage, readout)
    original_prelabel = json.loads(spot.pinned(args.readout_root / 'PRELABEL_SAFE.json', stage['prelabel_sha256'], 64*1024))
    spot.need(original_prelabel['stock_predictions_sha256'] == original.STOCK
              and original_prelabel['capture_accepted_sha256'] == args.capture_accepted_sha
              and not original_prelabel['GT_opened'], 'Same original native stock before labels')
    score_prelabel = json.loads(spot.pinned(args.original_score.parent / 'PRELABEL_SAFE.json',
                                           original_score['prelabel_sha256'], 64*1024))
    validate_score_prelabel(score_prelabel, args, approved, original)

    images, blocks, capacity = readout.original_population(args, spot)
    original_rows = capacity['independent_CPU_raster_disagree_rows']
    original_pixels = capacity['independent_CPU_raster_disagree_pixels']
    for receipt in (approved, stage):
        spot.need(receipt['probability_repair_sha256'] == args.probability_repair_sha
                  and receipt['probability_repair_consumed'] == 1
                  and receipt['original_CPU_disagree_rows'] == original_rows == 1
                  and receipt['original_CPU_disagree_pixels'] == original_pixels == 1
                  and receipt['effective_unresolved_CPU_disagree_rows'] == 0
                  and receipt['effective_unresolved_CPU_disagree_pixels'] == 0,
                  'Same verified single CUDA correction already consumed before labels')
    need(original_score['probability_repair_sha256'] == args.probability_repair_sha
         and original_score['original_CPU_disagree_rows'] == original_rows
         and original_score['original_CPU_disagree_pixels'] == original_pixels,
         'Same original score renderer qualification')
    sparse = [json.loads(line) for line in spot.pinned(args.readout_root / 'SPARSE_EDITS_PRIVATE.jsonl',
                                                       approved['edits_sha256'], 512*MIB).splitlines()]
    old_stream = spot.pinned(args.historical_native, original.STOCK, 512*MIB)
    old_score = json.loads(spot.pinned(args.historical_score, original.OLD_SCORE, 64*1024))
    need(old_score['status'] == 'HRSID_DEV705_E24_NATIVE_COCO_COMPLETE'
         and old_score['predictions_sha256'] == original.STOCK
         and old_score['original_dev_gt_sha256'] == native.FILES['DEV_ORIGINAL_GT_PRIVATE.json'],
         'Original stock and exact DEV GT identity, without rescoring')
    pool = []
    legacy_digest = sha256()
    for item, block in zip(images, blocks, strict=True):
        rows = spot.native_candidate_rows(block)
        xyxy = spot.author_boxes(block['raw_boxes_cxcywh'], torch)
        for row in rows:
            row['area'] = int(codec.area(row['rle']))
            row['box_xyxy'] = xyxy[row['query_id']].tolist()
        legacy_digest.update(readout.legacy_stock_bytes(item['image_id'], rows))
        pool.append(dict(native_block=readout.stock_block(item['image_id'], rows),
                         pool=rows, source_prefix=block['source_prefix']))
    need(legacy_digest.hexdigest() == sha256(old_stream).hexdigest() == original.STOCK,
         'Same canonical historical native bytes')
    need(len(pool) == len(sparse) == 705, 'Complete original DEV705 population')
    prepared = join_prepared(np, pool, sparse, readout)
    args.output.mkdir()
    seal = spot.write(args.output / 'PRELABEL_SAFE.json', dict(
        status='RFDETR_CENTER5202_DEV705_OCCUPANCY_BOUND_PRELABEL',
        original_score_sha256=args.original_score_sha, original_score_receipt_sha256=args.original_score_receipt_sha,
        original_score_prelabel_sha256=original_score['prelabel_sha256'],
        verified_readout_sha256=args.readout_accepted_sha, capture_accepted_sha256=args.capture_accepted_sha,
        sparse_sha256=approved['edits_sha256'], target_native_sha256=original.STOCK,
        original_dev_gt_sha256=native.FILES['DEV_ORIGINAL_GT_PRIVATE.json'], images=705,
        trigger=eligibility, DEV=True, FIT=False, TEST=False, GT_opened=False, AP=False,
        stored_original_score_read=True,
        original_score_previously_opened_DEV_GT=True))
    gt, findings = load_dev_geometry_inputs(native)
    helpers = fit_step.load_geometry(args.code, args.shapely_site)
    result = compute_geometry_only(np, gt, prepared, original_score, eligibility=eligibility,
        scorer=scorer, fit_step=fit_step, helpers=helpers, geometry=geometry, adapter=adapter, oracle=oracle)
    private = result.pop('private_witness')
    private_digest = spot.write(args.output / 'OCCUPANCY_BOUND_WITNESS_PRIVATE.json', private)
    need(not torch.cuda.is_initialized(), 'No accelerator context in occupancy diagnostic')
    result.update(status=STATUS, original_score_sha256=args.original_score_sha,
                  original_score_receipt_sha256=args.original_score_receipt_sha,
                  original_private_shape_witness_sha256=original_score['private_shape_witness_sha256'],
                  private_witness_sha256=private_digest, prelabel_sha256=seal,
                  original_observed_geometry_exact=True, original_stock_sha256=original.STOCK,
                  readout_accepted_sha256=args.readout_accepted_sha, capture_accepted_sha256=args.capture_accepted_sha,
                  sparse_sha256=approved['edits_sha256'], probability_repair_sha256=args.probability_repair_sha,
                  probability_repair_reapplied=False, original_empty_rasters_preserved=len(findings),
                  DEV=True, FIT=False, TEST=False, DEV_GT_opened=True, AP=False,
                  AP_recomputed=False, stored_original_score_read=True,
                  model_forward_calls=0, optimizer_updates=0, wall_seconds=perf_counter()-start,
                  sources=dict(original_loader=ORIGINAL_LOADER_SHA, adapter=ADAPTER_SHA, oracle=ORACLE_SHA,
                               score_math=original.MATHEMATICS, geometry=original.GEOMETRY,
                               readout=original.SOURCE_READOUT, fit_step=original.SOURCE_FIT_STEP,
                               entry=sha256(Path(__file__).read_bytes()).hexdigest()))
    result_sha = spot.write(args.output / 'RESULT_SAFE.json', result)
    print(json.dumps(dict(status=STATUS, result_sha256=result_sha, AP=False, TEST=False,
                          original_observed_geometry_exact=True)), flush=True)


if __name__ == '__main__':
    main()
