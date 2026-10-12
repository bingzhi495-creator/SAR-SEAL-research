"""Inert, standalone CPU diagnostic of already qualified full-FIT24 exports.

ROOT supplies a fresh consumer manifest only after actual GPU terminal0 and
CPU qualification0. Current plus six prior raw terminals precede every hash,
qualification/data read and scientific import. No training or pixel-logit
replay occurs; crossed masks/scores remain conditional, nonadditive diagnostics.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import types

MANIFEST_SCHEMA = 'ROOT-crossed-geometry-score-fullFIT24-evaluation-CPU-consumer-v1'
REPORT_SCHEMA = 'ROOT-crossed-geometry-score-fullFIT24-evaluation-CPU-report-v1'
STATUS = 'ROOT_CROSSED_GEOMETRY_SCORE_FULLFIT24_EXPOSED_DEV_CPU_EVALUATED'
INPUT_SPEC_SCHEMA = 'ROOT-shared-readout-objective-control-fullFIT24-spec-v1'
INPUT_QUAL_SCHEMA = 'ROOT-shared-readout-objective-control-fullFIT24-CPU-qualification-v1'
INPUT_QUAL_STATUS = 'ROOT_SHARED_READOUT_OBJECTIVE_CONTROL_FULLFIT24_DEV_CPU_QUALIFIED'
INPUT_REPORT_SCHEMA = 'shared-readout-objective-control-fullFIT24-report-v1'
INPUT_REPORT_STATUS = 'SHARED_READOUT_OBJECTIVE_CONTROL_FULLFIT24_DEV_GPU_COMPLETE_NO_SOTA'
ARMS = ('raw_uniform_BCE', 'raw_full_BCE', 'pair_pooled_full_BCE', 'mask_pooled_full_BCE', 'positive_global_gain')
ENDPOINTS = ('native0',) + ARMS
EXECUTION = dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2, torch_interop_threads=2, cuda_visible_devices='')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
            size += len(block)
    return dict(path=str(path), bytes=size, sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'} and
        Path(row['path']).is_absolute() and type(row['bytes']) is int and row['bytes'] > 0 and
        isinstance(row['sha256'], str) and len(row['sha256']) == 64 and set(row['sha256']) <= set('0123456789abcdef'),
        'Exact absolute pinned reference required')
    require(reference(row['path']) == row, 'Pinned bytes changed: ' + row['path'])
    return Path(row['path']).resolve(strict=True)


def read_ref(row):
    return read(bound(row))


def load_source(row, name):
    path = bound(row)
    require(path.suffix == '.py', 'Explicit Python SOURCE required')
    value = path.read_bytes()
    require(len(value) == row['bytes'] and hashlib.sha256(value).hexdigest() == row['sha256'], 'SOURCE changed before import')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(value, str(path), 'exec'), module.__dict__)
    return module


def terminal(receipt, phase):
    require(receipt.get('status') == 'ACTUAL_OWNED_TERMINAL' and receipt.get('original_identity') is not None and
        type(receipt.get('returncode')) is int and receipt['returncode'] == 0 and receipt.get('root_wait_reaped') is True and
        receipt.get('held_handle_exit') == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        receipt.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
        all(key in receipt and receipt[key] is None for key in ('abort', 'error')) and receipt.get('wait_errors') is False and
        receipt.get('GPU_observation_errors') == [], 'STOP before hashes/imports: actual held0/reap/birth absence/noerrors required')
    peak = receipt.get('whole_GPU_observed_peak_MiB')
    require(receipt.get('phase', phase) == phase and type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
        receipt.get('whole_GPU_used_limit_MiB') == 8192 and receipt.get('whole_leaf_VRAM_certified') is False and
        receipt.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation', 'Actual prior whole-GPU scope differs')


def canonical_digest(predictions):
    value = json.dumps(predictions, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return dict(sha256=hashlib.sha256(value).hexdigest(), serialization='UTF8 compact sorted-key JSON; export list order retained', count=len(predictions))


def prediction_map(predictions, ids):
    require(isinstance(predictions, list) and len(predictions) == 18700, 'All187 images/all100 query rows required')
    result = {}
    for row in predictions:
        image, query = row['image_id'], row['query_id']
        require(type(image) is int and image in ids and type(query) is int and 0 <= query < 100 and row['category_id'] == 1,
            'Actual image/category/query identity differs')
        require((image, query) not in result, 'Duplicate actual query identity')
        score, rle = row['score'], row['segmentation']
        require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1 and
            isinstance(rle, dict) and set(rle) == {'size', 'counts'} and isinstance(rle['size'], list) and len(rle['size']) == 2 and
            all(type(v) is int and v > 0 for v in rle['size']) and isinstance(rle['counts'], str), 'Qualified score/compressed RLE schema differs')
        result[image, query] = row
    require(set(result) == {(image, query) for image in ids for query in range(100)}, 'Complete original query population differs')
    return result


def cross_predictions(native, altered, ids):
    """Pure identity join; preserve native order and all non-mask/score fields."""
    native_map, altered_map = prediction_map(native, ids), prediction_map(altered, ids)
    require(set(native_map) == set(altered_map), 'Crossed endpoints differ in actual query identity')
    geometry, score = [], []
    for original in native:
        changed = altered_map[original['image_id'], original['query_id']]
        require({k: v for k, v in original.items() if k not in ('segmentation', 'score')} ==
            {k: v for k, v in changed.items() if k not in ('segmentation', 'score')}, 'Crossed endpoint metadata differs')
        require(original['segmentation']['size'] == changed['segmentation']['size'], 'Crossed original mask dimensions differ')
        geometry.append(dict(original, segmentation=changed['segmentation']))
        score.append(dict(original, score=changed['score']))
    require(all(row['segmentation'] == original['segmentation'] for row, original in zip(score, native)) and
        [(r['image_id'], r['query_id']) for r in geometry] == [(r['image_id'], r['query_id']) for r in native],
        'Score crossing changed RLE or native export order')
    return geometry, score


def write_report(path, report):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    require(path.is_absolute() and path.parent.is_dir() and not path.exists() and not temporary.exists(), 'Fresh absolute atomic report required')
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def run(manifest_file):
    started = time.monotonic()
    manifest_path = Path(manifest_file)
    require(manifest_path.is_absolute(), 'Absolute --manifest required')
    manifest = read(manifest_path)  # ROOT consumer metadata only.
    require(set(manifest) == {'schema', 'inputs', 'original_spec', 'original_CPU_qualification', 'original_GPU_receipt',
        'qualified_CPU_reader', 'qualified_CPU_tool_terminal', 'consumer_source', 'execution', 'output_report'} and
        manifest['schema'] == MANIFEST_SCHEMA and manifest['inputs'] == 'QUALIFIED_FULLFIT24_EXPORTS_READ_ONLY' and
        manifest['execution'] == EXECUTION, 'Exact separate ROOT CPU consumer manifest required')
    observed = manifest['qualified_CPU_tool_terminal']
    require(isinstance(observed, dict) and set(observed) == {'session_id', 'exit_code'} and
        type(observed['session_id']) is int and observed['session_id'] > 0 and type(observed['exit_code']) is int and observed['exit_code'] == 0,
        'ROOT actual preceding CPU qualification tool terminal0 required')
    spec_path = Path(manifest['original_spec']['path'])
    require(spec_path.is_absolute() and Path(manifest['original_GPU_receipt']['path']).is_absolute(), 'Absolute original metadata/raw paths required')
    spec = read(spec_path)  # Execution metadata only, not yet hashed.
    current_raw = read(manifest['original_GPU_receipt']['path'])
    terminal(current_raw, spec['phase'])
    raw_refs = dict(current=manifest['original_GPU_receipt'])
    for role, phase_role in (('teacher_receipt', 'teacher_phase'), ('preflight_receipt', 'preflight_phase'),
        ('short2_receipt', 'short2_phase'), ('prior_DEV_receipt', 'prior_DEV_phase'),
        ('prior_diag_receipt', 'prior_diag_phase'), ('previous_control_receipt', 'previous_control_phase')):
        require(Path(spec[role]['path']).is_absolute(), 'Absolute prior raw receipt required')
        terminal(read(spec[role]['path']), spec[phase_role])
        raw_refs[role] = spec[role]
    # All seven raw terminal checks above precede every hash, Q/data read or scientific import.
    source = reference(__file__)
    require(source == manifest['consumer_source'] and manifest['qualified_CPU_reader'] == spec['CPU_reader'], 'Separate consumer/current reader SOURCE differs')
    for role in ('original_spec', 'original_CPU_qualification', 'original_GPU_receipt', 'qualified_CPU_reader', 'consumer_source'):
        bound(manifest[role])
    canonical_receipt = spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    require(Path(manifest['original_GPU_receipt']['path']).resolve(strict=True) == canonical_receipt.resolve(strict=True), 'Original current raw receipt differs')
    reader = load_source(manifest['qualified_CPU_reader'], 'qualified_fullFIT_source_gate_only')
    require(reader.SPEC_SCHEMA == INPUT_SPEC_SCHEMA and reader.QUAL_SCHEMA == INPUT_QUAL_SCHEMA and
        reader.QUAL_STATUS == INPUT_QUAL_STATUS and reader.REPORT_SCHEMA == INPUT_REPORT_SCHEMA and
        reader.REPORT_STATUS == INPUT_REPORT_STATUS and spec['schema'] == INPUT_SPEC_SCHEMA,
        'Only the actual fullFIT24 producer/qualifier schema is accepted')
    execution, contract = reader.source_gate(spec_path.resolve(strict=True), spec, current_raw)
    q = read_ref(manifest['original_CPU_qualification'])
    reader.expect(q, dict(schema=reader.QUAL_SCHEMA, status=reader.QUAL_STATUS, phase=spec['phase'],
        source=execution['source'], reader_source=execution['reader'], spec=execution['spec'],
        parent_receipt=manifest['original_GPU_receipt'], GPU_admission=execution['GPU_admission'], owner_driver=execution['owner_driver'],
        SOURCE_contract=spec['SOURCE_contract'], readout_seed=spec['readout_seed'], arms=list(ARMS), endpoints=list(ENDPOINTS),
        current_reader_owner_admitted=True, fixed_endpoint_epoch=24, DEV_selection=False, LR_schedule=False,
        fresh_readout_experiment=True, previous_readout_checkpoint_loaded=False,
        previous_readout_optimizer_or_order_RNG_loaded=False, raw_current_and_six_original_terminal_gates_before_scientific_access=True,
        complete_original101_and_new_SOURCE_closure_owner_bound_before_imports=True,
        native610_and_typed_BN106_affine159_buffers_CPU_exact=True, all187_DEV380_GT_and_all100_queries_preserved=True,
        all_selected_raw_GT_including_empty_crowd_ignore_preserved=True, official_COCO12_maxDets=[1, 10, 100],
        CPU_affinity=[0, 1], torch_threads=2, torch_interop_threads=2, CUDA_initialized=False,
        AP_computed=True, AP_scope='full187_exposed_DEV_at_fixedE24_one_readout_seed', no_TEST_access=True), 'Actual completed fullFIT CPU qualification')
    reader.expect(q['training'], dict(microsteps=17784, updates_per_arm=8892, raw_GT_visits_per_arm=39864),
        'Actual qualified completed24 training counters')
    require(q['lineage']['DEV_annotation'] == spec['DEV_annotation'] and q['teacher_checkpoint'] == spec['teacher_checkpoint'] and
        q['teacher_qualification'] == spec['teacher_qualification'], 'Qualified original GT/teacher lineage differs')
    report = read_ref(q['report'])
    reader.expect(report, dict(schema=reader.REPORT_SCHEMA, status=reader.REPORT_STATUS, source=execution['source'], spec=execution['spec'],
        phase=spec['phase'], readout_seed=spec['readout_seed'], current_endpoints=list(ENDPOINTS), error=None,
        shared_teacher_forward=True, full_native610_and_BN_exact=True, DEV_historically_exposed=True, no_TEST_access=True), 'Qualified producer report')
    require(q['artifacts'] == report['result'] and set(q['artifacts']['mode_rows']) == set(q['endpoint_results']) == set(ENDPOINTS),
        'Qualified endpoint primary refs/report differ')
    primary = q['artifacts']['mode_rows']
    for mode in ENDPOINTS:
        row, result = primary[mode], q['endpoint_results'][mode]
        require(result['predictions'] == row['predictions'] and result['image_rows'] == row['image_rows'] and
            result['query_score_factors'] == row['score_factors'] and row['checkpoint'] ==
            (spec['teacher_checkpoint'] if mode == 'native0' else q['artifacts']['checkpoints'][mode]), 'Qualified endpoint primary/checkpoint join differs')
        for role in ('predictions', 'image_rows', 'score_factors'):
            bound(row[role])
    official_ref = spec['official_evaluation_CPU_reader']
    require(official_ref in spec['sources'], 'Official COCO evaluator absent from original owner closure')
    official = load_source(official_ref, 'sealed_official_COCO12_only')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(psutil.Process().cpu_affinity() == [0, 1] and torch.get_num_threads() == torch.get_num_interop_threads() == 2 and
        not torch.cuda.is_initialized(), 'Fresh CPU01/threads2/CUDAempty required')
    RNG_before = torch.get_rng_state().clone()
    ids = contract['split']['evaluation_ids']
    coco = COCO(str(bound(spec['DEV_annotation'])))
    require(len(ids) == len(coco.imgs) == 187 and set(ids) == set(coco.imgs) and len(coco.anns) == 380,
        'Original DEV187/all380 rawGT required')
    native = read_ref(primary['native0']['predictions'])
    prediction_map(native, ids)
    require(list(dict.fromkeys(row['image_id'] for row in native)) == ids, 'Native original image export order differs')
    for row in native:
        image = coco.imgs[row['image_id']]
        require(row['segmentation']['size'] == [image['height'], image['width']], 'Native RLE original dimensions differ')
    native_metrics = official.coco_metrics(coco, native, ids, COCOeval)
    require(native_metrics == q['endpoint_results']['native0']['official_COCO'], 'Current native COCO12 differs from actual qualified metrics')
    results = {}
    for arm in ARMS:
        altered = read_ref(primary[arm]['predictions'])
        geometry, score = cross_predictions(native, altered, ids)
        geometry_metrics = official.coco_metrics(coco, geometry, ids, COCOeval)
        score_metrics = official.coco_metrics(coco, score, ids, COCOeval)
        if arm == 'positive_global_gain':
            require(geometry == native and geometry_metrics == native_metrics, 'Actual gain identical-mask geometry crossing must equal native COCO12')
        results[arm] = dict(qualified_actual_COCO=q['endpoint_results'][arm]['official_COCO'],
            geometry_only_COCO=geometry_metrics, score_only_COCO=score_metrics,
            altered_predictions=primary[arm]['predictions'], qualified_actual_prediction_digest=canonical_digest(altered),
            geometry_only_prediction_digest=canonical_digest(geometry), score_only_prediction_digest=canonical_digest(score),
            geometry_only_masks_from=arm, geometry_only_scores_from='native0', score_only_masks_from='native0', score_only_scores_from=arm,
            actual_image_query_join=True, all18700_retained=True, native_export_order_preserved=True,
            all_prediction_metadata_preserved=True, score_only_RLE_exactly_native=True,
            gain_geometry_only_exactly_native=True if arm == 'positive_global_gain' else None)
        del altered, geometry, score
    require(torch.equal(RNG_before, torch.get_rng_state()) and not torch.cuda.is_initialized() and
        os.environ['CUDA_VISIBLE_DEVICES'] == '' and psutil.Process().cpu_affinity() == [0, 1] and
        torch.get_num_threads() == torch.get_num_interop_threads() == 2, 'CPU resources/CUDA/RNG scope changed')
    historical = q['historical_native0_baseline']
    drift = historical['superiority_interpretation_stopped_due_to_baseline_drift']
    require(type(drift) is bool and drift is (not historical['exact_historical_native0_reproduction']), 'Historical baseline drift interpretation flag differs')
    result = dict(schema=REPORT_SCHEMA, status=STATUS, manifest=reference(manifest_path), source=source,
        inputs={key: manifest[key] for key in ('original_spec', 'original_CPU_qualification', 'original_GPU_receipt', 'qualified_CPU_reader')},
        ROOT_observed_qualified_CPU_tool_terminal=observed, tool_terminal_independently_proved_by_file=False,
        original_owner=execution, raw_terminal_refs=raw_refs, current_and_six_prior_raw_gates_before_any_hash_or_scientific_access=True,
        qualified_producer_report=q['report'], qualified_endpoint_primary_refs=primary, official_evaluation_CPU_reader=official_ref,
        DEV_annotation=spec['DEV_annotation'], phase=spec['phase'], readout_seed=spec['readout_seed'], endpoints=list(ENDPOINTS),
        native0_qualified_COCO=q['endpoint_results']['native0']['official_COCO'], native0_recomputed_COCO=native_metrics,
        fixed_endpoint_epoch=24, qualified_training_microsteps=17784, qualified_training_updates_per_arm=8892,
        qualified_training_raw_GT_visits_per_arm=39864,
        native_prediction_digest=canonical_digest(native), crossed_results=results, original_raw_GT_flags_preserved=True,
        evaluation_images=187, evaluation_raw_GT=380, queries_per_image=100, predictions_per_cross=18700,
        score_filter=None, additional_NMS=False, official_COCO12_maxDets=[1, 10, 100], AP_computed=True,
        execution=EXECUTION, CUDA_initialized=False, external_CPU_RNG_unchanged=True, elapsed_seconds=time.monotonic() - started,
        shared_fixed_teacher=True, one_readout_seed=True, DEV_historically_exposed=True, blind_holdout_claimed=False, no_TEST_access=True,
        retraining_performed=False, full_numerical_qualification_repeated=False, checkpoint_tensors_loaded=False, pixel_logits_loaded=False,
        historical_native0_baseline=historical, superiority_interpretation_stopped_due_to_baseline_drift=drift,
        interpretation=('Crossed metrics describe only these current qualified exports; historical superiority interpretation remains stopped.' if drift else
            'Crossed metrics diagnose the fixed E24 exports conditional on this teacher and readout seed; they are not an additive causal decomposition.'),
        additive_decomposition_claimed=False, efficacy_claimed=False, convergence_claimed=False, novelty_claimed=False, SOTA_claimed=False,
        evidence_limits=['Masks and scores are crossed by actual image/query identity, preserving all100 native candidates and native export order.',
            'Altered scores include original mask-quality factors. Recombining them with different masks is a diagnostic counterfactual, not a deployable head.',
            'COCO ranking, matching and interpolation make geometry/score effects nonadditive; these metrics cannot identify training-mechanism causation.',
            'Already qualified endpoint exports and their original GT flags are reused; no training, neural forward, tensor checkpoint or pixel-logit replay occurs.',
            'One fixed teacher, one readout seed, fixed E24 historically exposed DEV; no blind TEST, convergence or SOTA verdict.',
            'The preceding CPU tool terminal0 is ROOT-observed manifest metadata; a JSON file cannot independently establish the tool exit.'])
    write_report(manifest['output_report'], result)
    print(json.dumps(dict(status=STATUS, report=reference(manifest['output_report']), baseline_drift=drift), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    run(parser.parse_args().manifest)


if __name__ == '__main__':
    main()
