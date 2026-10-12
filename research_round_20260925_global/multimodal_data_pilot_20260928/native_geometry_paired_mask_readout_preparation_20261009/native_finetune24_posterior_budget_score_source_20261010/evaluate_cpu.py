"""Score-only CPU control of already qualified nativeE48 all100 exports.

ROOT binds this consumer and the frozen scorer separately from the original
GPU owner. Seven actual raw terminal gates precede every hash/SOURCE import or
scientific access. The unchanged V2 reader binds its original CPU supplement;
its successful qualification supplies model-state evidence without a model
reload. Saved M200 is CPU-rendered, but support and class probabilities are
the actual exported GPU RLE and score factors. Only prediction.score changes.
No training, forward replay, mask thresholding, filtering, or selection occurs.
"""
import argparse
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import types

SCHEMA = 'ROOT-nativeE48-posterior-budget-score-consumer-v1'
REPORT_SCHEMA = 'nativeE48-posterior-budget-score-control-report-v1'
STATUS = 'NATIVE_E48_POSTERIOR_BUDGET_SCORE_CONTROL_CPU_COMPLETE'
V2_SCHEMA = 'ROOT-native-finetune24-finalE48-evaluation-CPU-SUPPLEMENT-V2-qualification-v1'
V2_STATUS = 'ROOT_NATIVE_FINETUNE24_FINAL_E48_EVALUATION_CPU_SUPPLEMENT_V2_QUALIFIED'
EXECUTION = dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2,
                 torch_interop_threads=2, cuda_visible_devices='')
LEGACY = ('native0', 'global', 'spatial_halves', 'spatial_reversed')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def terminal(raw, phase):
    # Duplicated tiny bootstrap only: the verified V2 SOURCE cannot be imported
    # until current + four original + technical + formal raw terminals pass.
    expected = dict(status='ACTUAL_OWNED_TERMINAL', returncode=0, root_wait_reaped=True,
        held_handle_exit=dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0), abort=None,
        error=None, wait_errors=False, GPU_observation_errors=[], whole_GPU_used_limit_MiB=8192,
        whole_leaf_VRAM_certified=False, GPU_scope='whole GPU observed, not per-process WDDM allocation')
    require(isinstance(raw, dict) and all(k in raw and raw[k] == v and
        (type(v) not in (int, bool) or type(raw[k]) is type(v)) for k, v in expected.items()),
        'STOP before hashes/imports: actual owned terminal differs')
    peak = raw.get('whole_GPU_observed_peak_MiB')
    require(raw.get('original_identity') is not None and raw.get('phase', phase) == phase and
        raw.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
        type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192,
        'STOP: actual identity/birth absence/whole GPU peak differs')


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'bytes', 'sha256'} and
        Path(ref['path']).is_absolute() and type(ref['bytes']) is int and ref['bytes'] > 0 and
        isinstance(ref['sha256'], str) and len(ref['sha256']) == 64 and
        set(ref['sha256']) <= set('0123456789abcdef'), 'Exact absolute pinned reference required')
    require(reference(ref['path']) == ref, 'Pinned input changed: ' + ref['path'])
    return Path(ref['path'])


def load_source(ref, name):
    path = bound(ref)
    source = path.read_bytes()
    require(path.suffix == '.py' and len(source) == ref['bytes'] and
        hashlib.sha256(source).hexdigest() == ref['sha256'], 'SOURCE changed before import')
    module = types.ModuleType(name)
    module.__file__ = str(path)  # V2.source_gate must retain the actual V2 path.
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def inputs(manifest_path, manifest):
    require(set(manifest) == {'schema', 'inputs', 'original_GPU_spec', 'original_GPU_admission',
        'original_GPU_receipt', 'V2_CPU_qualification', 'V2_CPU_consumer_manifest', 'sources',
        'execution', 'deadline_seconds'} and manifest['schema'] == SCHEMA and
        manifest['inputs'] == 'CPU_QUALIFIED_NATIVE_E48_EXPORTS_READ_ONLY' and
        manifest['execution'] == EXECUTION and type(manifest['deadline_seconds']) is int and
        manifest['deadline_seconds'] == 1800 and set(manifest['sources']) == {'reader', 'scorer', 'consumer'},
        'Exact thin score-control manifest required')
    spec_path = Path(manifest['original_GPU_spec']['path'])
    require(spec_path.is_absolute(), 'Absolute original GPU SPEC required')
    spec = read(spec_path)  # Metadata needed solely to locate the seven raws.
    receipt_path = spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    terminal(read(receipt_path), spec['phase'])
    require(set(spec['training_receiptmap']) == set(spec['training_phasemap']) == set(LEGACY), 'Four original raw identities required')
    for mode in LEGACY:
        require(Path(spec['training_receiptmap'][mode]['path']).is_absolute(), 'Absolute raw path required')
        terminal(read(spec['training_receiptmap'][mode]['path']), spec['training_phasemap'][mode])
    for receipt, phase in (('restoration_receipt', 'restoration_phase'), ('native_phase_receipt', 'native_phase_phase')):
        require(Path(spec[receipt]['path']).is_absolute(), 'Absolute native raw path required')
        terminal(read(spec[receipt]['path']), spec[phase])
    # No hash or imported SOURCE precedes the complete raw7 gate above.
    manifest_ref = reference(manifest_path)
    for ref in list(manifest['sources'].values()) + [manifest[k] for k in
            ('original_GPU_spec', 'original_GPU_admission', 'original_GPU_receipt',
             'V2_CPU_qualification', 'V2_CPU_consumer_manifest')]:
        bound(ref)
    require(manifest['sources']['consumer'] == reference(__file__) and
        Path(manifest['sources']['scorer']['path']).resolve() == Path(__file__).resolve().parent / 'score_control.py',
        'Actual consumer/frozen local scorer SOURCE differs')
    reader = load_source(manifest['sources']['reader'], 'bound_unchanged_nativeE48_V2_reader')
    consumer_path = bound(manifest['V2_CPU_consumer_manifest'])
    refs = reader.source_gate(spec_path, spec, consumer_path, read(consumer_path))
    require(refs['reader'] == manifest['sources']['reader'] and refs['spec'] == manifest['original_GPU_spec'] and
        refs['admission'] == manifest['original_GPU_admission'] and refs['receipt'] == manifest['original_GPU_receipt'] and
        refs['consumer'] == manifest['V2_CPU_consumer_manifest'], 'Actual V2/original GPU provenance differs')
    q = read(bound(manifest['V2_CPU_qualification']))
    reader.expect(q, dict(schema=V2_SCHEMA, status=V2_STATUS, phase=spec['phase'], endpoint='nativeE48',
        source=refs['source'], reader_source=refs['reader'], spec=refs['spec'], SOURCE_proposal=spec['SOURCE_proposal'],
        consumer_manifest=refs['consumer'], original_CPU_reader=refs['original_reader'], original_CPU_failure=refs['failure'],
        parent_receipt=refs['receipt'], GPU_admission=refs['admission'], owner_driver=refs['owner'],
        CPU_bound_sources=refs['CPU_sources'], original_GPU_sources=spec['sources'], sources=refs['CPU_sources'],
        original_GPU_DEV_owner_bound=False, supplementalDEVCPUbound=True, new_reader_GPUowner_bound=False, preserved_failed_v1=True,
        checkpoint=spec['native_phase_checkpoint'], completed_phase_epoch=24, effective_native_epoch=48,
        images=187, raw_GT=380, predictions_count=18700, AP_computed=True, AP_computed_on_GPU=False,
        independent_CPU_metric_reconstruction=True, training_executed=False, native_forward_replayed=False,
        optimizer_or_RNG_restored=False, no_TEST=True, blind_DEV_claimed=False,
        actual_current_and_six_prior_raw_terminals_passed_before_hashes_imports=True,
        original_declared_GPU_sources_bound_before_imports=True, complete_CPU_supplement_sources_bound_before_imports=True,
        new_formal_nativeE48_qualification_required=True, fixed_phase24_only=True, intermediate_nodes_evaluated=False), 'Successful actual V2 qualification')
    reader.expect(q['state_check'], dict(full_native610_CPU_exact=True, registered_parameters=451,
        registered_adapter_tensors=0, typed_BN106_affine159_buffers_original_exact=True,
        evaluation_before_after610_exact_to_E48=True, CPU_native_forward_replayed=False), 'Qualified E48 state evidence')
    reader.expect(q['CPU_execution'], dict(EXECUTION, CUDA_initialized=False), 'Qualified CPU execution scope')
    record, report_ref, start_ref = reader.report_check(spec, refs)
    require(q['report'] == report_ref and q['start'] == start_ref and all(q[k] == record[k] for k in
        ('image_rows', 'predictions', 'query_score_factors', 'state_proof')), 'Qualified exports/report joins differ')
    scorer = load_source(manifest['sources']['scorer'], 'bound_frozen_posterior_budget_scorer')
    official = reader.load_source(spec['official_evaluation_CPU_reader'], 'bound_generic_official_COCO_helpers')
    return reader, scorer, official, spec, refs, q, record, manifest_ref, report_ref, start_ref


def score_rows(torch, np, masks, reader, scorer, spec, refs, rows, predictions, grouped, factors, started, deadline):
    factor_map = {(f['image_id'], f['query_id']): f for f in factors}
    require(len(factor_map) == len(factors) == 18700, 'Unique actual score factors required')
    scored, new_scores, summary = copy.deepcopy(predictions), {}, []
    packet_keys = {'schema', 'source', 'spec', 'checkpoint', 'endpoint', 'dataset_index', 'image_id', 'original_shape',
        'actual_img_shape', 'final_class_logits', 'final_mask_logits200', 'final_field_digests', 'prediction_query_ids',
        'GPU_original_logits_saved', 'all100_fields_saved'}
    with torch.no_grad():
        for row in rows:
            require(time.monotonic() - started <= deadline, 'CPU score-control deadline exceeded')
            packet = torch.load(str(reader.bound(row['final_field_packet'])), map_location='cpu')
            require(isinstance(packet, dict) and set(packet) == packet_keys, 'Exact qualified final-field packet required')
            reader.expect(packet, dict(schema='native-finetune24-finalE48-final-fields-v1', source=refs['source'], spec=refs['spec'],
                checkpoint=spec['native_phase_checkpoint'], endpoint='nativeE48', dataset_index=row['dataset_index'], image_id=row['image_id'],
                original_shape=row['original_shape'], actual_img_shape=row['actual_img_shape'], final_field_digests=row['final_fields'],
                prediction_query_ids=row['prediction_query_ids'], GPU_original_logits_saved=False, all100_fields_saved=True), 'Qualified final packet')
            for key, shape, digest_name in (('final_class_logits', (1, 100, 2), 'classes'), ('final_mask_logits200', (1, 100, 200, 200), 'masks')):
                value = packet[key]
                require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.float32 and
                    tuple(value.shape) == shape and not value.requires_grad and bool(torch.isfinite(value).all()) and
                    reader.tensor_digest(torch, value) == row['final_fields'][digest_name], 'Exact finite qualified tensor differs: ' + key)
            population = {p['query_id']: p for p in grouped[row['image_id']]}
            budgets, empty_count = [], 0
            for first in range(0, 100, 10):
                queries = list(range(first, first + 10))
                logits = reader.render_original(torch, packet['final_mask_logits200'][:, first:first + 10], row['actual_img_shape'], row['original_shape'])
                binary = np.stack([masks.decode(dict(size=population[q]['segmentation']['size'],
                    counts=population[q]['segmentation']['counts'].encode('ascii'))).astype(bool) for q in queries])
                support = torch.from_numpy(np.ascontiguousarray(binary))
                classes = torch.tensor([factor_map[(row['image_id'], q)]['class_probability'] for q in queries], dtype=torch.float32)
                result = scorer.posterior_budget_scores(torch, logits, support, classes, eps=1e-6)
                for local, query in enumerate(queries):
                    budget, score = float(result['posterior_budget'][local]), float(result['class_times_posterior_budget'][local])
                    require(math.isfinite(budget) and math.isfinite(score) and 0 <= budget <= 1 and 0 <= score <= 1,
                        'Nonfinite/unbounded posterior-budget score')
                    empty = not bool(support[local].any())
                    require(not empty or budget == score == 0., 'Empty exported mask must retain zero score')
                    new_scores[(row['image_id'], query)] = score
                    budgets.append(budget)
                    empty_count += int(empty)
                del logits, binary, support, classes, result
            summary.append(dict(image_id=row['image_id'], queries=100, empty_exported_masks=empty_count,
                posterior_budget_min=min(budgets), posterior_budget_max=max(budgets), posterior_budget_mean=math.fsum(budgets) / 100))
            del packet, population
    require(len(new_scores) == len(predictions) == 18700, 'All100 score population must survive')
    for before, after in zip(predictions, scored):
        after['score'] = new_scores[(before['image_id'], before['query_id'])]
        require({k: v for k, v in before.items() if k != 'score'} == {k: v for k, v in after.items() if k != 'score'},
            'Only score may change; geometry/category/query/export order must remain exact')
    return scored, summary


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def run(manifest_file, output_directory):
    started = time.monotonic()
    manifest_path, output = Path(manifest_file), Path(output_directory)
    require(manifest_path.is_absolute() and output.is_absolute() and not output.exists() and output.parent.is_dir() and
        os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Absolute manifest/fresh output directory/CUDA-empty required')
    manifest_path = manifest_path.resolve(strict=True)
    manifest = read(manifest_path)
    reader, scorer, official, spec, refs, q, record, manifest_ref, report_ref, start_ref = inputs(manifest_path, manifest)
    # Every SOURCE/provenance/Q input above is bound before scientific imports.
    import psutil
    import torch
    import numpy as np
    from pycocotools import mask as masks
    from pycocotools.cocoeval import COCOeval
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'ROOT CPU01/noCUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 2 and torch.__version__ == '2.0.1+cu118' and
        (Path(torch.__file__).resolve().parent / 'version.py').resolve() == reader.bound(spec['torch_version_source']).resolve() and
        Path(inspect.getsourcefile(torch.nn.functional.interpolate)).resolve() == reader.bound(spec['interpolation_SOURCE']).resolve(),
        'Bound Torch version/interpolation/threads differ')
    coco = COCO(str(reader.bound(spec['DEV'])))
    ids, rows = reader.image_population(spec, refs, record, coco)
    predictions, grouped = official.prediction_population(record, ids, coco, np, masks, rows)
    factors_proof = official.score_factors(record, predictions, rows, np)
    factors = reader.read_ref(record['query_score_factors'])
    scored, summaries = score_rows(torch, np, masks, reader, scorer, spec, refs, rows, predictions, grouped,
        factors, started, manifest['deadline_seconds'])
    original_metrics = official.coco_metrics(coco, predictions, ids, COCOeval)
    require(original_metrics == q['COCO'], 'Unchanged native exports must reproduce their qualified official COCO12')
    new_metrics = official.coco_metrics(coco, scored, ids, COCOeval)
    for ref in (refs['CPU_sources'] + list(manifest['sources'].values()) + [manifest_ref, manifest['V2_CPU_qualification'],
            refs['consumer'], refs['failure'], refs['spec'], refs['admission'], refs['receipt'], refs['owner'], report_ref, start_ref] +
            [record[k] for k in ('image_rows', 'predictions', 'query_score_factors', 'state_proof')]):
        reader.bound(ref)
    for row in rows:
        reader.bound(row['image'])
        reader.bound(row['final_field_packet'])
    require(time.monotonic() - started <= manifest['deadline_seconds'] and not torch.cuda.is_initialized() and
        psutil.Process().cpu_affinity() == [0, 1] and torch.get_num_threads() == torch.get_num_interop_threads() == 2,
        'Final CPU deadline/resources/noCUDA proof differs')
    output.mkdir()
    prediction_path = output / 'ALL_PREDICTIONS_POSTERIOR_BUDGET.json'
    write(prediction_path, scored)
    report = dict(schema=REPORT_SCHEMA, status=STATUS, manifest=manifest_ref, sources=manifest['sources'],
        V2_CPU_qualification=manifest['V2_CPU_qualification'], V2_CPU_consumer_manifest=refs['consumer'],
        original_GPU_spec=refs['spec'], original_GPU_admission=refs['admission'], original_GPU_receipt=refs['receipt'], owner_driver=refs['owner'],
        original_GPU_DEV_owner_bound=False, V2_reader_GPUowner_bound=False, score_consumer_GPUowner_bound=False,
        score_consumer_and_scorer_separately_manifest_bound_before_imports=True,
        actual_current_and_six_prior_raw_terminals_passed_before_hashes_imports=True,
        original_CPU_bound_source_count=len(refs['CPU_sources']), model_state_evidence_from_qualified_V2_not_reloaded=True,
        original_report=report_ref, original_start=start_ref, image_rows=record['image_rows'],
        original_predictions=record['predictions'], query_score_factors=record['query_score_factors'],
        final_field_packets=[r['final_field_packet'] for r in rows], scored_predictions=reference(prediction_path),
        endpoint='nativeE48', images=187, raw_GT=380, predictions_count=18700, all100_query_ordinals_retained=True,
        non_score_prediction_fields_exact=True, actual_GPU_RLE_support_used=True, actual_exported_class_probability_used=True,
        score='class_probability * sum(B*sigmoid(L))/(sum(B)+sum((1-B)*sigmoid(L))+1e-6)',
        score_only=True, geometry_changed=False, masks_rethresholded=False, added_NMS=False, added_score_cutoff=False,
        native_official_COCO=original_metrics, posterior_budget_COCO=new_metrics, native_COCO_exact_qualified_V2=True,
        original_factor_arithmetic=factors_proof, per_image=summaries, CPU_execution=dict(EXECUTION, CUDA_initialized=False),
        AP_computed=True, AP_computed_on_GPU=False, independent_CPU_metric_reconstruction=True,
        training_executed=False, model_reloaded=False, native_forward_replayed=False, GPU_forward_replayed=False,
        no_TEST=True, blind_DEV_claimed=False, no_endpoint_selection=True, no_score_threshold_selection=True,
        novelty_claimed=False, calibration_claimed=False, expected_GT_IoU_claimed=False, efficacy_claimed=False, SOTA_claimed=False,
        elapsed_seconds=time.monotonic() - started, evidence_limits=[
            'V2 qualification supplies nativeE48 model/state/forward evidence; this score-only consumer does not independently reload that model or reap prior CPU tool sessions.',
            'Original DEV and V2 reader were separately CPU-supplement-bound, not original GPU-owner-bound. The scorer and this consumer are newly manifest-bound; no retroactive GPU closure claim.',
            'CPU two-stage bilinear logits use saved M200. GPU original pixel logits were not saved: no measured GPU pixel allclose, certified interpolation bound, or bitwise GPU renderer replay.',
            'Actual exported GPU RLE remains the fixed binary support even near zero; CPU logits are never thresholded to change geometry. Class probabilities are actual exported factors.',
            'The smoothed soft-IoU confidence formula is a known SoftDiceConfidence transform, not an IoU guarantee or calibrated learned quality head; a ratio of expected budgets is not expected IoU.',
            'Only fixed nativeE48 scores are changed. Exposed DEV COCO differences describe this untrained scoring control; no novelty, convergence, SOTA or broader efficacy conclusion follows.'
        ])
    write(output / 'REPORT.json', report)
    print(json.dumps(dict(status=STATUS, output_directory=str(output), native_AP=original_metrics['stats'][0],
        posterior_budget_AP=new_metrics['stats'][0], images=187, predictions=18700)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, help='Fresh ROOT score-control manifest')
    parser.add_argument('--out', required=True, help='Fresh absolute output directory')
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'SOURCE_ONLY: explicit ROOT-owned CPU execution required')
    run(args.spec, args.out)


if __name__ == '__main__':
    main()
