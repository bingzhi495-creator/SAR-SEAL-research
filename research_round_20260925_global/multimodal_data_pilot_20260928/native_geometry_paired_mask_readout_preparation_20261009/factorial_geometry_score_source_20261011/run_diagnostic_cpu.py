"""Four-cell score/geometry diagnostic on one qualified fixed joint endpoint.

Inert SOURCE at authoring. No model, CP tensor, pixel replay, training or selection.
S1 was computed on G1 support: transplanting it onto G0 is counterfactual,
not a deployable native-mask quality estimator or additive causal decomposition.
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

SCHEMA = 'ROOT-joint-native-factorial-geometry-score-CPU-manifest-v1'
REPORT_SCHEMA = 'ROOT-joint-native-factorial-geometry-score-CPU-report-v1'
STATUS = 'ROOT_JOINT_NATIVE_FACTORIAL_GEOMETRY_SCORE_CPU_DIAGNOSTIC_COMPLETE'
INPUTS = 'QUALIFIED_FIXED_JOINT_ENDPOINT_B_EXPORTS_READ_ONLY'
READER_SHA = '7eb9955bec54717ad0b9dc78ce85b4d0b3fb8ccea2564f2f5e9cd607171d0fe8'
ENTRY_SHA = '4859b6bafe41c8f7e5e5cfab1813c3d428f4574d6a91aedaf077ff386919232d'
OFFICIAL_SHA = '2357929fcd38ffc3020c5b8d204ee371da8da3972ff2e2e03398c8225c55e916'
CONTROLS = ('B_native_official', 'B_posterior_official', 'B_posterior_budget')
MODES = ('native0', 'global', 'spatial_halves', 'spatial_reversed')
EXECUTION = dict(device='cpu', cpu_affinity=[0, 1], cuda_visible_devices='',
    numeric_thread_environment=2, torch_imported=False, deadline_seconds=1800)
COCO_CONFIGURATION = dict(iouType='segm', catIds=[1], maxDets=[1, 10, 100],
    images=187, raw_GT=380, queries_per_image=100, score_filter=None,
    added_NMS=False, original_crowd_ignore_empty_GT_preserved=True)
KEYS = {'schema', 'inputs', 'evaluation_spec', 'evaluation_CPU_qualification',
    'evaluation_GPU_receipt', 'evaluation_GPU_admission', 'qualified_CPU_reader',
    'qualified_CPU_tool_terminal', 'consumer_source', 'crossed_predictions_source',
    'manifest_builder_source', 'official_evaluation_CPU_reader', 'DEV', 'exports',
    'execution', 'COCO_configuration', 'output_report'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    require(Path(path).is_absolute(), 'Absolute JSON locator required')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True); sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            sha.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha.hexdigest())


def bound(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'bytes', 'sha256'} and
        Path(ref['path']).is_absolute() and type(ref['bytes']) is int and ref['bytes'] > 0 and
        isinstance(ref['sha256'], str) and len(ref['sha256']) == 64 and
        set(ref['sha256']) <= set('0123456789abcdef') and reference(ref['path']) == ref,
        'Exact absolute pinned reference differs')
    return Path(ref['path']).resolve(strict=True)


def load(ref, name):
    path = bound(ref); source = path.read_bytes()
    require(path.suffix == '.py' and len(source) == ref['bytes'] and
        hashlib.sha256(source).hexdigest() == ref['sha256'], 'Verified SOURCE changed')
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def terminal(raw, phase):
    require(raw.get('status') == 'ACTUAL_OWNED_TERMINAL' and raw.get('original_identity') is not None and
        type(raw.get('returncode')) is int and raw['returncode'] == 0 and raw.get('root_wait_reaped') is True and
        raw.get('held_handle_exit') == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        raw.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
        all(k in raw and raw[k] is None for k in ('abort', 'error')) and raw.get('wait_errors') is False and
        raw.get('GPU_observation_errors') == [], 'STOP before hashes/Q/imports: actual original raw0 required')
    peak = raw.get('whole_GPU_observed_peak_MiB')
    require(raw.get('phase', phase) == phase and type(peak) in (int, float) and math.isfinite(peak) and
        0 <= peak <= 8192 and raw.get('whole_GPU_used_limit_MiB') == 8192 and
        raw.get('whole_leaf_VRAM_certified') is False and
        raw.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation', 'Raw resource scope differs')


def bootstrap(spec_path):
    """Only untrusted JSON raw locators are read before ALL terminal gates."""
    def gated(raw_path, phase):
        raw_path = Path(raw_path)
        require(raw_path.is_absolute() and raw_path.parent.name == phase, 'Absolute raw locator/phase directory differs')
        terminal(read(raw_path), phase)
    spec_path = Path(spec_path)
    require(spec_path.is_absolute(), 'Absolute original evaluation SPEC required')
    spec = read(spec_path)
    require(spec.get('endpoint_kind') in ('joint24', 'warm48'), 'One explicit fixed joint endpoint required')
    current = spec_path.parent/'finite_point_cache_execution_20261005_v2'/'runs'/spec['phase']/'PARENT_RECEIPT.json'
    gated(current, spec['phase']); raw_refs = [dict(path=str(current), phase=spec['phase'])]
    train = read(spec['joint_training_spec']['path']); tiny = read(train['joint_preflight_spec']['path'])
    require(set(tiny['training_receiptmap']) == set(tiny['training_phasemap']) == set(MODES), 'Original four raw locators required')
    priors = [(tiny['training_receiptmap'][m], tiny['training_phasemap'][m]) for m in MODES]
    for row in tiny['prior_receipts']:
        require(set(row) == {'receipt', 'phase'}, 'Explicit restoration/technical prior raw required')
        priors.append((row['receipt'], row['phase']))
    priors.append((train['joint_preflight_receipt'], train['joint_preflight_phase']))
    if spec['endpoint_kind'] == 'warm48':
        priors.append((train['first24_receipt'], train['first24_phase']))
    priors.append((spec['joint_training_receipt'], spec['joint_training_phase']))
    for ref, phase in priors:
        gated(ref['path'], phase); raw_refs.append(dict(path=ref['path'], phase=phase))
    return spec, current, raw_refs


def qualified_inputs(spec_path, spec, q_ref):
    """Called only after bootstrap; reuse sealed provenance, never numerical run."""
    require(spec['CPU_reader']['sha256'] == READER_SHA and spec['entry_source']['sha256'] == ENTRY_SHA and
        spec['official_evaluation_CPU_reader']['sha256'] == OFFICIAL_SHA, 'Sealed fixed-endpoint interfaces required')
    reader = load(spec['CPU_reader'], '_factorial_sealed_joint_reader_provenance_ONLY')
    producer, refs, lineage = reader.source_gate(Path(spec_path).resolve(strict=True), spec)
    stage = reader.STAGES[spec['endpoint_kind']]
    report, report_ref, start_ref, aggregate_ref = reader.report_check(spec, refs, stage)
    q = read(bound(q_ref))
    reader.expect(q, dict(schema=reader.QUAL_SCHEMA, status=reader.QUAL_STATUS, **refs,
        report=report_ref, start=start_ref, GPU_qualification=aggregate_ref, endpoint_kind=spec['endpoint_kind'],
        effective_native_epoch=stage['native_epoch'], effective_B_readout_epoch=stage['head_epoch'],
        joint_training_CPU_qualification=spec['joint_training_CPU_qualification'], checkpoints=spec['joint_checkpoints'],
        artifacts=report['result'], controls=list(reader.CONTROLS), official_COCO_maxDets=[1, 10, 100],
        all187_images_all380_raw_GT_preserved=True, all100_queries_per_image_preserved=True,
        own_native610_and_B_readout8_CPU_checkpoint_verified=True, within_B_class_packet_and_probability_identity_verified=True,
        B_posterior_geometry='L_plus_u', B_bounded_geometry_used=False, historically_exposed_DEV=True,
        no_TEST=True, endpoint_selection=False, GT_used_for_scores=False,
        CPU_execution=dict(device='cpu', affinity=[0,1], CUDA_VISIBLE_DEVICES='', torch_num_threads=2,
            torch_num_interop_threads=2, CUDA_initialized=False)), 'Actual complete fixed endpoint CPU qualification differs')
    require('torch' not in sys.modules and q['arms']['B']['image_rows'] == report['result']['B']['image_rows'],
        'Provenance-only workflow or B image-row join differs')
    exports = {}
    for name in CONTROLS:
        primary, result = report['controls'][name], q['control_results'][name]
        require(primary == report['result']['B']['controls'][name] and
            all(result[k] == primary[k] for k in ('predictions', 'score_factors')) and
            result['all18700_queries_retained'] is True and len(result['GPU_export_COCO12']['stats']) == 12,
            'Qualified own-B all100 export refs differ: ' + name)
        exports[name] = {k: primary[k] for k in ('predictions', 'score_factors')}
    return reader, refs, q, report, exports, (report_ref, start_ref, aggregate_ref)


def population(rows, factors, ids, coco, image_rows):
    expected_order = [(r['image_id'], q) for r in image_rows for q in r['prediction_query_ids']]
    require(len(rows) == len(factors) == len(expected_order) == 18700 and
        [(r['image_id'], r['query_id']) for r in rows] == expected_order, 'Original native export order differs')
    result, classes = {}, {}
    for row, factor in zip(rows, factors):
        image, query = row['image_id'], row['query_id']; key = (image, query)
        require(type(image) is int and image in ids and type(query) is int and 0 <= query < 100 and key not in result and
            type(row['category_id']) is int and row['category_id'] == 1, 'Unique all100 image/query/category identity required')
        score, rle = row['score'], row['segmentation']
        require(type(score) in (int,float) and math.isfinite(score) and 0 <= score <= 1 and
            set(rle) == {'size','counts'} and isinstance(rle['counts'], str) and
            rle['size'] == [coco.imgs[image]['height'], coco.imgs[image]['width']], 'Original-size compressed RLE/score differs')
        require((factor['image_id'], factor['query_id']) == key and factor['score'] == score and
            type(factor['class_probability']) in (int,float) and math.isfinite(factor['class_probability']) and
            0 <= factor['class_probability'] <= 1, 'Actual exported class/score factors differ')
        result[key], classes[key] = row, factor['class_probability']
    require(set(result) == {(i,q) for i in ids for q in range(100)}, 'Complete query population required')
    return result, classes


def write_fresh(path, value):
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    require(path.is_absolute() and path.parent.is_dir() and not path.exists() and not temporary.exists(), 'Fresh absolute report required')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    # Exclusive publication refuses a concurrent destination; same-directory link is atomic.
    try:
        os.link(str(temporary), str(path))
    finally:
        temporary.unlink()


def run(manifest_file):
    started = time.monotonic(); manifest_path = Path(manifest_file)
    manifest = read(manifest_path)
    require(set(manifest) == KEYS and manifest['schema'] == SCHEMA and manifest['inputs'] == INPUTS and
        manifest['execution'] == EXECUTION and manifest['COCO_configuration'] == COCO_CONFIGURATION, 'Exact separate consumer manifest required')
    observed = manifest['qualified_CPU_tool_terminal']
    require(set(observed) == {'session_id','exit_code'} and type(observed['session_id']) is int and observed['session_id'] > 0 and
        type(observed['exit_code']) is int and observed['exit_code'] == 0, 'ROOT must confirm actual preceding CPU tool terminal0')
    spec_path = Path(manifest['evaluation_spec']['path']); spec, current, raw_locators = bootstrap(spec_path)
    # ALL current/ancestry raw gates above precede the first hash, Q or prediction read, or SOURCE import.
    manifest_ref = reference(manifest_path); bound(manifest_ref)
    require(read(bound(manifest['evaluation_spec'])) == spec and reference(current) == manifest['evaluation_GPU_receipt'] and
        reference(__file__) == manifest['consumer_source'] and manifest['qualified_CPU_reader'] == spec['CPU_reader'] and
        manifest['official_evaluation_CPU_reader'] == spec['official_evaluation_CPU_reader'] and manifest['DEV'] == spec['DEV'] and
        Path(manifest['crossed_predictions_source']['path']).resolve() == Path(__file__).with_name('crossed_predictions.py').resolve(),
        'Separate consumer and qualified original refs differ')
    target = Path(manifest['output_report'])
    require(target.is_absolute() and target.parent.is_dir() and not target.exists() and
        not target.with_name(target.name+'.tmp').exists(), 'Fresh absolute output required before scientific access')
    for key in ('consumer_source','crossed_predictions_source','manifest_builder_source','evaluation_GPU_admission','DEV'):
        bound(manifest[key])
    reader, refs, q, report, exports, metadata_refs = qualified_inputs(spec_path, spec, manifest['evaluation_CPU_qualification'])
    require(exports == manifest['exports'] and refs['GPU_admission'] == manifest['evaluation_GPU_admission'] and
        all(manifest[k] not in spec['sources'] for k in ('consumer_source','crossed_predictions_source','manifest_builder_source')),
        'Exact exports/separate post-qualification consumer bindings required')
    helper = load(manifest['crossed_predictions_source'], '_factorial_crossed_predictions_ONLY')
    official = load(manifest['official_evaluation_CPU_reader'], '_factorial_official_COCO12_ONLY')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    import psutil
    psutil.Process().cpu_affinity([0,1])
    require(psutil.Process().cpu_affinity() == [0,1] and 'torch' not in sys.modules, 'Actual CPU01/no Torch before scientific imports required')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO(str(bound(manifest['DEV']))); ids = list(coco.imgs)
    require(len(ids) == len(set(ids)) == 187 and len(coco.anns) == 380 and set(coco.cats) == {1}, 'All original DEV187/380 GT/flags required')
    image_rows_ref = q['arms']['B']['image_rows']; image_rows = read(bound(image_rows_ref))
    require(len(image_rows) == 187 and [r['image_id'] for r in image_rows] == ids and
        all(len(r['prediction_query_ids']) == 100 and set(r['prediction_query_ids']) == set(range(100)) for r in image_rows),
        'Qualified B original all100 image/query order required')
    rows, maps, native_classes = {}, {}, None
    for name in CONTROLS:
        rows[name] = read(bound(exports[name]['predictions']))
        maps[name], classes = population(rows[name], read(bound(exports[name]['score_factors'])), ids, coco, image_rows)
        if native_classes is None:
            native_classes = classes
        require(classes == native_classes, 'Same own-B actual class probabilities required')
    cells = helper.assemble_crossed_predictions(*(rows[n] for n in CONTROLS), image_ids=ids, query_count=100)
    require(set(cells) == {'F00','F10','F01','F11'}, 'Exactly four fixed factorial cells required')
    baseline_order = [(r['image_id'],r['query_id']) for r in rows[CONTROLS[0]]]
    metrics = {}
    for cell in ('F00','F10','F01','F11'):
        geometry = maps[CONTROLS[0] if cell in ('F00','F01') else CONTROLS[2]]
        scores = maps[CONTROLS[0] if cell in ('F00','F10') else CONTROLS[2]]
        require([(r['image_id'],r['query_id']) for r in cells[cell]] == baseline_order and
            all({k:v for k,v in r.items() if k != 'score'} == {k:v for k,v in geometry[key].items() if k != 'score'} and
                r['score'] == scores[key]['score'] for r,key in zip(cells[cell],baseline_order)),
            'Cell must preserve native order/geometry metadata and replace only the keyed score')
        metrics[cell] = official.coco_metrics(coco, cells[cell], ids, COCOeval)
        require(len(metrics[cell]['stats']) == 12 and all(type(v) in (int,float) and math.isfinite(v) and -1 <= v <= 1
            for v in metrics[cell]['stats']), 'COCO12 finite metrics including genuine undefined -1 required')
    require(metrics['F00'] == q['control_results'][CONTROLS[0]]['GPU_export_COCO12'] and
        metrics['F11'] == q['control_results'][CONTROLS[2]]['GPU_export_COCO12'], 'Qualified factual endpoint COCO12 sanity differs')
    main_AP = {key:value['stats'][0] for key,value in metrics.items()}
    try:
        summary = dict(available=True, values=helper.summarize_factorial_AP(main_AP))
    except ValueError as exc:
        require(any(v == -1 for v in main_AP.values()) and str(exc) == 'undefined AP, decomposition unavailable',
            'Unexpected factorial summary error')
        summary = dict(available=False, reason=str(exc), values=None)
    require(time.monotonic()-started < EXECUTION['deadline_seconds'] and psutil.Process().cpu_affinity() == [0,1] and
        os.environ['CUDA_VISIBLE_DEVICES'] == '' and 'torch' not in sys.modules and
        all(os.environ[k] == '2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
        'Diagnostic CPU resource/deadline scope changed')
    for key in ('evaluation_spec','evaluation_CPU_qualification','evaluation_GPU_receipt','evaluation_GPU_admission',
        'qualified_CPU_reader','consumer_source','crossed_predictions_source','manifest_builder_source','official_evaluation_CPU_reader','DEV'):
        bound(manifest[key])
    for ref in metadata_refs + (image_rows_ref,):
        bound(ref)
    for value in exports.values():
        for ref in value.values(): bound(ref)
    raw_refs = [dict(receipt=reference(r['path']),phase=r['phase']) for r in raw_locators]
    bound(manifest_ref)
    result = dict(schema=REPORT_SCHEMA, status=STATUS, manifest=manifest_ref, source=manifest['consumer_source'],
        original_evaluation=refs, endpoint_kind=spec['endpoint_kind'], phase=spec['phase'],
        effective_native_epoch=q['effective_native_epoch'], effective_B_readout_epoch=q['effective_B_readout_epoch'],
        own_B_checkpoint=q['checkpoints']['B'], evaluation_CPU_qualification=manifest['evaluation_CPU_qualification'],
        ROOT_observed_preceding_CPU_tool_terminal=observed, tool_terminal_independently_proved_by_file=False,
        all_current_and_ancestry_raw_gates_before_hashes_Q_predictions_imports=True, raw_terminal_refs=raw_refs,
        new_consumer_original_GPU_owner_bound=False, original_SPEC_or_GPU_closure_modified=False,
        exports=exports, DEV=manifest['DEV'], COCO_configuration=COCO_CONFIGURATION, execution=EXECUTION,
        cells=metrics, main_AP_fraction_units=main_AP, factorial_AP_diagnostic=summary,
        factual_endpoint_COCO12_exact=True, metric_calls=4, all18700_rows_retained=True,
        native_export_order_and_empty_RLE_preserved=True, transplanted_empty_mask_scores_not_zeroed=True,
        posterior_official_and_budget_geometry_equal=True, within_B_class_probability_exact=True,
        original_GPU_RLE_and_scores_used=True, CPU_reconstructed_geometry_substituted=False,
        checkpoint_tensors_loaded=False, pixel_fields_loaded=False, training_performed=False,
        GPU_forward_performed=False, AP_computed=True, endpoint_selection=False, historically_exposed_DEV=True,
        no_TEST=True, deployment_claimed=False, causal_attribution_claimed=False, additive_decomposition_claimed=False,
        novelty_claimed=False, efficacy_claimed=False, elapsed_seconds=time.monotonic()-started,
        evidence_limits=['G0 is own-B native L geometry; G1 is own-B direct L+u posterior geometry.',
            'S0 is actual native official score; S1 is actual posterior budget score computed on G1 support.',
            'F01 transplants S1 onto G0 and is counterfactual; it does not recompute a native-support budget.',
            'Posterior official quality is not the geometry-only score; F10 freezes S0 exactly.',
            'Conditional AP contrasts and interaction are algebra on four paired exported packets, not additive causal effects.',
            'No model or CPU/GPU numerical replay is repeated; the original qualified practical rendering limits remain.',
            'This fixed endpoint/exposed DEV diagnostic supplies no blind-test, convergence or external superiority claim.'])
    write_fresh(manifest['output_report'], result)
    print(json.dumps(dict(status=STATUS, report=reference(manifest['output_report'])), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True); parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args(); require(args.owner_execute, 'Explicit future ROOT CPU execution required')
    run(args.manifest)


if __name__ == '__main__':
    main()
