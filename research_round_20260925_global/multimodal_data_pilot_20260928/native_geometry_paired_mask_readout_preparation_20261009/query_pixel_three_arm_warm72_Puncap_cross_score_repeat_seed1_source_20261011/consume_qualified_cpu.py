"""ROOT-only CPU crossing of qualified phase48/effective-readout72 exports.

Current plus eight prior raw terminals precede all hashes, qualification/data
reads and scientific imports. No model, checkpoint tensors or pixel fields are
loaded. Crosses are nondeployable, nonadditive diagnostics on exposed DEV.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types

MANIFEST_SCHEMA = 'ROOT-query-pixel-three-arm-warm72-Puncap-cross-score-CPU-consumer-v1'
REPORT_SCHEMA = 'ROOT-query-pixel-three-arm-warm72-Puncap-cross-score-CPU-report-v1'
STATUS = 'ROOT_QUERY_PIXEL_THREE_ARM_WARM72_PUNCAP_CROSS_SCORE_EXPOSED_DEV_CPU_EVALUATED'
INPUT_SPEC = 'ROOT-query-pixel-posterior-scalar-control-E24warm-additional48-Puncap-eval-spec-v1'
INPUT_PREFIX = 'query-pixel-posterior-scalar-control-E24warm-additional48-Puncap-eval'
INPUT_Q = 'ROOT-query-pixel-posterior-scalar-control-E24warm-additional48-Puncap-eval-CPU-qualification-v1'
INPUT_STATUS = 'ROOT_QUERY_PIXEL_POSTERIOR_SCALAR_CONTROL_E24WARM_ADDITIONAL48_PUNCAP_EVAL_CPU_QUALIFIED'
ARMS = ('posterior_BCE', 'bounded_BCE', 'query_scalar_IoU')
CONTROLS = ('native_official', 'native_zero_budget', 'P_native_posterior_budget',
    'P_corrected_official', 'P_corrected_posterior_budget', 'B_native_posterior_budget',
    'B_corrected_official', 'B_corrected_posterior_budget', 'scalar_native_IoU',
    'P_uncapped_official', 'P_uncapped_posterior_budget')
CROSSES = (('P_bounded_geometry_native_score', 'P_corrected_posterior_budget', 'native_official'),
    ('P_unbounded_geometry_native_score', 'P_uncapped_posterior_budget', 'native_official'),
    ('B_bounded_geometry_native_score', 'B_corrected_posterior_budget', 'native_official'),
    ('P_bounded_score_native_geometry', 'native_official', 'P_corrected_posterior_budget'),
    ('P_unbounded_score_native_geometry', 'native_official', 'P_uncapped_posterior_budget'),
    ('B_bounded_score_native_geometry', 'native_official', 'B_corrected_posterior_budget'))
PRIOR = (('teacher_receipt', 'teacher_phase'), ('preflight_receipt', 'preflight_phase'),
    ('short2_receipt', 'short2_phase'), ('prior_DEV_receipt', 'prior_DEV_phase'),
    ('prior_diag_receipt', 'prior_diag_phase'), ('previous_control_receipt', 'previous_control_phase'))
EXECUTION = dict(device='cpu', cpu_affinity=[0, 1], cuda_visible_devices='',
    numeric_thread_environment=2, torch_imported=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
            size += len(block)
    return dict(path=str(path), bytes=size, sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and Path(row['path']).is_absolute() and reference(row['path']) == row,
        'Exact absolute pinned reference required')
    return Path(row['path']).resolve(strict=True)


def read_ref(row):
    return read(bound(row))


def load_source(row, name):
    path = bound(row)
    require(path.suffix == '.py', 'Only pinned Python SOURCE may be loaded')
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == row['sha256'], 'SOURCE changed before compilation')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def terminal(row, phase):
    require(row.get('status') == 'ACTUAL_OWNED_TERMINAL' and row.get('original_identity') is not None and
        type(row.get('returncode')) is int and row['returncode'] == 0 and row.get('root_wait_reaped') is True and
        row.get('held_handle_exit') == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        row.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
        all(key in row and row[key] is None for key in ('abort', 'error')) and row.get('wait_errors') is False and
        row.get('GPU_observation_errors') == [], 'STOP before hashes/imports: actual terminal held0/reap/birth/noerrors required')
    peak = row.get('whole_GPU_observed_peak_MiB')
    require(row.get('phase', phase) == phase and type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
        row.get('whole_GPU_used_limit_MiB') == 8192 and row.get('whole_leaf_VRAM_certified') is False and
        row.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation', 'Actual whole-GPU observed scope differs')


def canonical_digest(rows):
    value = json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return dict(sha256=hashlib.sha256(value).hexdigest(), count=len(rows),
        serialization='UTF8 compact sorted-key JSON; geometry export list order retained')


def prediction_map(rows, ids, coco):
    require(isinstance(rows, list) and len(rows) == 18700, 'All187 images/all100 query rows required')
    result = {}
    for row in rows:
        image, query = row['image_id'], row['query_id']
        require(type(image) is int and image in ids and type(query) is int and 0 <= query < 100 and
            type(row['category_id']) is int and row['category_id'] == 1, 'Actual image/category/query identity differs')
        require((image, query) not in result, 'Repeated actual query identity')
        score, rle = row['score'], row['segmentation']
        require(type(score) in (int, float) and 0 <= score <= 1 and math.isfinite(score) and
            isinstance(rle, dict) and set(rle) == {'size', 'counts'} and isinstance(rle['counts'], str) and
            rle['size'] == [coco.imgs[image]['height'], coco.imgs[image]['width']], 'Qualified score/original-size compressed RLE differs')
        result[image, query] = row
    require(set(result) == {(image, query) for image in ids for query in range(100)} and
        list(dict.fromkeys(row['image_id'] for row in rows)) == ids, 'Complete original image/query population or order differs')
    return result


def class_factors(rows, predictions):
    require(isinstance(rows, list) and len(rows) == 18700, 'All18700 qualified score factors required')
    result = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {'image_id', 'query_id', 'class_probability', 'mask_quality', 'official_score'},
            'Exact exported factor fields required')
        key = (row['image_id'], row['query_id'])
        require(type(key[0]) is int and type(key[1]) is int and key in predictions and key not in result,
            'Complete unique actual image/query factor identity required')
        for name in ('class_probability', 'mask_quality', 'official_score'):
            value = row[name]
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                'Finite exported class/quality/score in [0,1] required')
        require(row['official_score'] == predictions[key]['score'], 'Factor score differs from actual exported prediction')
        result[key] = row['class_probability']
    require(set(result) == set(predictions), 'Score-factor population differs from actual all100 predictions')
    return result


def write_report(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    require(path.is_absolute() and path.parent.is_dir() and not path.exists() and not temporary.exists(), 'Fresh absolute atomic report required')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def run(manifest_file):
    started = time.monotonic()
    manifest_path = Path(manifest_file)
    require(manifest_path.is_absolute(), 'Absolute --manifest required')
    manifest = read(manifest_path)  # ROOT consumer metadata; not yet hashed.
    require(set(manifest) == {'schema', 'inputs', 'original_spec', 'original_CPU_qualification', 'original_GPU_receipt',
        'qualified_CPU_reader', 'qualified_CPU_tool_terminal', 'consumer_source', 'cross_score_helper_source', 'execution', 'output_report'} and
        manifest['schema'] == MANIFEST_SCHEMA and manifest['inputs'] == 'QUALIFIED_QUERY_PIXEL_THREE_ARM_WARM72_PUNCAP_EXPORTS_READ_ONLY' and
        manifest['execution'] == EXECUTION, 'Exact separate ROOT CPU consumer manifest required')
    observed = manifest['qualified_CPU_tool_terminal']
    require(isinstance(observed, dict) and set(observed) == {'session_id', 'exit_code'} and type(observed['session_id']) is int and
        observed['session_id'] > 0 and type(observed['exit_code']) is int and observed['exit_code'] == 0,
        'ROOT-observed actual preceding CPU qualification session/terminal0 required; metadata is not independent tool proof')
    spec_path = Path(manifest['original_spec']['path'])
    require(spec_path.is_absolute() and Path(manifest['original_GPU_receipt']['path']).is_absolute(), 'Absolute original SPEC/raw paths required')
    spec = read(spec_path)  # Original execution metadata only.
    current_raw = read(manifest['original_GPU_receipt']['path'])
    terminal(current_raw, spec['phase'])
    ctx_metadata = read(spec['context']['path'])
    raw_refs = dict(current=manifest['original_GPU_receipt'])
    for receipt, phase in PRIOR:
        terminal(read(ctx_metadata[receipt]['path']), ctx_metadata[phase])
        raw_refs[receipt] = ctx_metadata[receipt]
    terminal(read(spec['native_preflight_receipt']['path']), spec['native_preflight_phase'])
    raw_refs['native_preflight_receipt'] = spec['native_preflight_receipt']
    terminal(read(spec['first24_receipt']['path']), spec['first24_phase'])
    raw_refs['first24_receipt'] = spec['first24_receipt']
    # All nine raw terminals above precede every hash, Q/data read or scientific import.
    manifest_ref = reference(manifest_path)
    require(read_ref(manifest_ref) == manifest, 'Consumer manifest changed after raw bootstrap')
    target = Path(manifest['output_report'])
    require(target.is_absolute() and target.parent.is_dir() and not target.exists() and
        not target.with_name(target.name + '.tmp').exists(), 'Fresh absolute output required before scientific access')
    source = reference(__file__)
    require(source == manifest['consumer_source'] and manifest['qualified_CPU_reader'] == spec['CPU_reader'] and
        Path(manifest['cross_score_helper_source']['path']).resolve() == Path(__file__).with_name('cross_score_rows.py').resolve(),
        'Separate consumer/helper/original qualified reader bindings differ')
    pinned = ('original_spec', 'original_CPU_qualification', 'original_GPU_receipt', 'qualified_CPU_reader',
        'consumer_source', 'cross_score_helper_source')
    for role in pinned:
        bound(manifest[role])
    canonical_raw = spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    require(Path(manifest['original_GPU_receipt']['path']).resolve(strict=True) == canonical_raw.resolve(strict=True), 'Original current raw path differs')
    reader = load_source(manifest['qualified_CPU_reader'], 'qualified_warm72_provenance_only')
    require(reader.SPEC_SCHEMA == INPUT_SPEC and reader.QUAL_SCHEMA == INPUT_Q and reader.QUAL_STATUS == INPUT_STATUS and
        spec['schema'] == INPUT_SPEC, 'Only the sealed warm72 Puncap qualification interface is accepted')
    execution, ctx, prior_raw, first24 = reader.source_gate(spec_path.resolve(strict=True), spec, current_raw)
    previous_q, previous_state, previous_matching = first24
    require(manifest['consumer_source'] not in spec['sources'] and manifest['cross_score_helper_source'] not in spec['sources'],
        'New consumer/helper must remain separate-manifest-bound, not retrospectively original GPU owner-bound')
    # The reader/writer module top levels are stdlib only; call no numerical workflow.
    require('torch' not in sys.modules, 'This consumer must not import Torch or reconstruct a model')
    q = read_ref(manifest['original_CPU_qualification'])
    output = Path(spec['output_directory'])
    report_ref, start_ref, aggregate_ref = [reference(output / name) for name in ('REPORT.json', 'START.json', 'QUALIFIED.json')]
    reader.expect(q, dict(schema=INPUT_Q, status=INPUT_STATUS, phase=spec['phase'], source=execution['source'], reader_source=execution['reader'],
        spec=execution['spec'], report=report_ref, start=start_ref, GPU_qualification=aggregate_ref, parent_receipt=manifest['original_GPU_receipt'],
        GPU_admission=execution['GPU_admission'], owner_driver=execution['owner_driver'], context=spec['context'], SOURCE_proposal=spec['SOURCE_proposal'],
        arms=list(ARMS), controls=list(CONTROLS), completed_epochs=48, fixed_endpoint_epoch=48, readout_seed=2026100912,
        phase_epochs=48, effective_readout_epochs=72, teacher_epoch=24, total_data_exposure_epochs=96,
        first24_CPU_qualification=spec['first24_CPU_qualification'], warm_checkpoints=spec['warm_checkpoints'],
        weights_warm_start=True, previous_optimizer_or_order_RNG_loaded=False,
        microsteps=35568, optimizer_updates=17784, raw_GT_visits=79728, native610_and_typed_BN106_affine159_buffers_CPU_exact=True,
        current_and_eight_prior_raw_terminals_before_hashes_imports=True, complete_current_SOURCE_closure_GPU_owner_bound=True,
        all187_DEV380_raw_GT_and_all100_queries_retained=True, DEV_historically_exposed=True, DEV_selection=False, no_TEST_access=True,
        AP_computed=True, official_COCO12_maxDets=[1, 10, 100], CPU_affinity=[0, 1], torch_threads=2, torch_interop_threads=2, CUDA_initialized=False),
        'Actual completed warm72 CPU qualification')
    require(previous_q['spec'] == spec['first24_training_spec'] and previous_q['source'] == spec['first24_entry_source'] and
        previous_q['reader_source'] == spec['first24_CPU_reader_source'] and previous_q['parent_receipt'] == spec['first24_receipt'] and
        previous_q['artifacts']['checkpoints'] == spec['warm_checkpoints'], 'First24 ancestor must not substitute current warm72 qualification')
    del previous_q, previous_state, previous_matching, first24, prior_raw
    report, start, aggregate = read_ref(report_ref), read_ref(start_ref), read_ref(aggregate_ref)
    reader.expect(report, dict(schema=INPUT_PREFIX + '-report-v1', status='QUERY_PIXEL_POSTERIOR_SCALAR_CONTROL_E24WARM_ADDITIONAL48_PUNCAP_EVAL_GPU_COMPLETE_PENDING_CPU',
        source=execution['source'], reader=execution['reader'], spec=execution['spec'], phase=spec['phase'], context=spec['context'],
        GPU_admission=execution['GPU_admission'], arms=list(ARMS), controls=list(CONTROLS), fixed_endpoint_epoch=48,
        phase_epochs=48, effective_readout_epochs=72, teacher_epoch=24, total_data_exposure_epochs=96,
        epochs=48, microsteps=35568, optimizer_updates=17784, raw_GT_visits=79728,
        first24_CPU_qualification=spec['first24_CPU_qualification'], warm_checkpoints=spec['warm_checkpoints'],
        weights_warm_start=True, previous_optimizer_or_order_RNG_loaded=False,
        shared_teacher_forward=True, all_native610_frozen=True, DEV_historically_exposed=True, DEV_selection=False, no_TEST_access=True, error=None), 'Original qualified producer report')
    reader.expect(start, dict(schema=INPUT_PREFIX + '-start-v1', source=execution['source'], reader=execution['reader'],
        spec=execution['spec'], phase=spec['phase'], context=spec['context'], arms=list(ARMS), controls=list(CONTROLS),
        phase_epochs=48, effective_readout_epochs=72, teacher_epoch=24, total_data_exposure_epochs=96,
        first24_CPU_qualification=spec['first24_CPU_qualification'], warm_checkpoints=spec['warm_checkpoints'],
        weights_warm_start=True, previous_optimizer_or_order_RNG_loaded=False,
        fixed_endpoint_epoch=48, DEV_selection=False), 'Original qualified START')
    reader.expect(aggregate, dict(schema=INPUT_PREFIX + '-aggregate-v1', status='QUERY_PIXEL_POSTERIOR_SCALAR_CONTROL_E24WARM_ADDITIONAL48_PUNCAP_EVAL_PENDING_CPU_READER',
        phase=spec['phase'], source=execution['source'], spec=execution['spec'], report=report_ref, independently_CPU_qualified=False), 'Original pending aggregate')
    require(q['artifacts'] == report['result'] == aggregate['result'] and q['teacher_checkpoint'] == ctx['teacher_checkpoint'] and
        q['teacher_qualification'] == ctx['teacher_qualification'], 'Original result/teacher qualification joins differ')
    primary, endpoints = q['artifacts']['mode_rows'], q['endpoint_results']['controls']
    require(set(primary) == set(endpoints) == set(CONTROLS), 'All eleven qualified control records required')
    for control in CONTROLS:
        require(set(primary[control]) == {'predictions', 'score_factors', 'image_rows', 'checkpoint'} and
            all(endpoints[control][key] == primary[control][key] for key in ('predictions', 'score_factors', 'image_rows', 'checkpoint')) and
            endpoints[control]['queries'] == 18700 and len(endpoints[control]['GPU_export_COCO12']['stats']) == 12,
            'Qualified endpoint primary refs/COCO12 differ: ' + control)
    official_ref = ctx['official_evaluation_CPU_reader']
    require(official_ref in spec['sources'] and ctx['DEV_annotation'] in spec['sources'], 'Official evaluator/DEV absent from original owner closure')
    official = load_source(official_ref, 'sealed_official_COCO12_only')
    helper = load_source(manifest['cross_score_helper_source'], 'sealed_cross_score_rows_only')
    contract = read_ref(ctx['SOURCE_contract'])
    ids = contract['split']['evaluation_ids']
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    require(psutil.Process().cpu_affinity() == [0, 1] and 'torch' not in sys.modules, 'Fresh CPU01/no Torch required')
    # Official helper has stdlib-only top-level imports. Scientific imports start here.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO(str(bound(ctx['DEV_annotation'])))
    require(len(ids) == len(set(ids)) == len(coco.imgs) == 187 and set(ids) == set(coco.imgs) and
        len(coco.anns) == 380 and set(coco.cats) == {1}, 'Original exposed DEV187/all380 raw GT required')
    selected = ('native_official', 'P_corrected_posterior_budget', 'P_uncapped_posterior_budget', 'B_corrected_posterior_budget')
    predictions = {control: read_ref(primary[control]['predictions']) for control in selected}
    native_class_factors = None
    for control, rows in predictions.items():
        by_query = prediction_map(rows, ids, coco)
        classes = class_factors(read_ref(primary[control]['score_factors']), by_query)
        if control == 'native_official':
            native_class_factors = classes
        else:
            require(classes == native_class_factors, 'All selected controls must retain exact exported native class probabilities')
    native = predictions['native_official']
    native_metrics = official.coco_metrics(coco, native, ids, COCOeval)  # One original sanity check only.
    require(native_metrics == endpoints['native_official']['GPU_export_COCO12'], 'Native official COCO12 differs from qualified GPU export metrics')
    crossed_results = {}
    for name, geometry_control, score_control in CROSSES:
        geometry_rows, score_rows = predictions[geometry_control], predictions[score_control]
        crossed = helper.cross_score_rows(geometry_rows, score_rows)
        require(len(crossed) == 18700 and all({k: v for k, v in before.items() if k != 'score'} ==
            {k: v for k, v in after.items() if k != 'score'} for before, after in zip(geometry_rows, crossed)), 'Cross modified geometry or metadata')
        metrics = official.coco_metrics(coco, crossed, ids, COCOeval)
        crossed_results[name] = dict(geometry_control=geometry_control, score_control=score_control,
            geometry_predictions=primary[geometry_control]['predictions'], score_predictions=primary[score_control]['predictions'],
            COCO12=metrics, prediction_digest=canonical_digest(crossed), all18700_retained=True,
            only_score_replaced=True, geometry_export_order_and_empty_RLE_preserved=True)
        del crossed
    require(psutil.Process().cpu_affinity() == [0, 1] and os.environ['CUDA_VISIBLE_DEVICES'] == '' and
        'torch' not in sys.modules and all(os.environ[key] == '2' for key in
        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'CPU resource scope changed')
    for role in pinned:
        bound(manifest[role])
    for row in (report_ref, start_ref, aggregate_ref, official_ref, ctx['DEV_annotation']):
        bound(row)
    for control in selected:
        bound(primary[control]['predictions'])
        bound(primary[control]['score_factors'])
    bound(manifest_ref)
    result = dict(schema=REPORT_SCHEMA, status=STATUS, manifest=manifest_ref, source=source,
        inputs={key: manifest[key] for key in pinned}, ROOT_observed_qualified_CPU_tool_terminal=observed,
        tool_terminal_independently_proved_by_file=False, original_owner=execution, raw_terminal_refs=raw_refs,
        current_and_eight_prior_raw_gates_before_hashes_scientific_access=True, original_SPEC_or_GPU_closure_modified=False,
        consumer_and_helper_separately_manifest_bound=True, consumer_or_helper_original_GPU_owner_bound=False,
        first24_lineage=dict(training_spec=spec['first24_training_spec'], CPU_qualification=spec['first24_CPU_qualification'],
            receipt=spec['first24_receipt'], phase=spec['first24_phase'], warm_checkpoints=spec['warm_checkpoints']),
        first24_metrics_substituted_for_current=False,
        qualified_producer_report=report_ref, qualified_endpoint_primary_refs=primary, official_evaluation_CPU_reader=official_ref,
        DEV_annotation=ctx['DEV_annotation'], phase=spec['phase'], fixed_endpoint_epoch=48, readout_seed=2026100912,
        phase_epochs=48, effective_readout_epochs=72, teacher_epoch=24, total_data_exposure_epochs=96,
        qualified_original_GPU_COCO12={c: endpoints[c]['GPU_export_COCO12'] for c in CONTROLS},
        original_eleven_metrics_recomputed=False, native_official_sanity_recomputed_once=True, native_recomputed_COCO12=native_metrics,
        cross_metric_calls=6, exported_class_probability_equality_checked=True,
        crossed_results=crossed_results, original_raw_GT_flags_preserved=True, evaluation_images=187, evaluation_raw_GT=380,
        queries_per_image=100, predictions_per_cross=18700, score_filter=None, additional_NMS=False, official_COCO12_maxDets=[1, 10, 100],
        execution=EXECUTION, elapsed_seconds=time.monotonic() - started, AP_computed=True, shared_fixed_teacher=True,
        one_readout_seed=True, DEV_historically_exposed=True, blind_holdout_claimed=False, DEV_selection=False, no_TEST_access=True,
        retraining_performed=False, full_numerical_qualification_repeated=False, checkpoint_tensors_loaded=False, pixel_fields_loaded=False,
        additive_decomposition_claimed=False, causal_attribution_claimed=False, deployment_claimed=False, efficacy_claimed=False, novelty_claimed=False,
        evidence_limits=['Only the actual warm72 Puncap qualification supplies current controls; first24 supplies warm-start ancestry only.',
            'No training, model loading or dense numerical replay is repeated. Actual GPU RLE and scores are used; CPU-reconstructed metrics do not substitute them.',
            'The three geometry crossings freeze actual native official scores. Recomputed official mask-quality on changed masks is not geometry-only.',
            'The three score crossings transfer corrected-support budget scores to native masks; they do not recompute a budget on native support and are not deployable quality heads.',
            'Native-support budget and scalar comparisons already exist in the copied qualified eleven-control metrics, separately from transplanted scores.',
            'All crossings are conditional on this teacher, seed, phase48/effective-readout72 budget and exposed DEV; AP differences are not additive causal effects.',
            'The prior qualification permits practical near-zero CPU/GPU differences, without measured GPU pixel-logit allclose or a certified numerical bound; this consumer does not render pixels.',
            'Teacher24 plus readout72 data exposure96 is not native96 compute, objective or optimizer-trajectory matching.',
            'The preceding CPU session terminal0 is ROOT-observed manifest metadata; a file does not independently prove that tool exit.'])
    write_report(manifest['output_report'], result)
    print(json.dumps(dict(status=STATUS, report=reference(manifest['output_report'])), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--owner-execute', action='store_true', required=True)
    run(parser.parse_args().manifest)


if __name__ == '__main__':
    main()
