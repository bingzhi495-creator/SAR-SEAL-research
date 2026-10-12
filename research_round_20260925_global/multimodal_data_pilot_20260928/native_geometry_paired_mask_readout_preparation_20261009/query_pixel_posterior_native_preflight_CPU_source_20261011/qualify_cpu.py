"""Small, tensor-free CPU consumer of the actual native posterior preflight.

ROOT alone executes after current and six prior GPU owners actually terminate.
This checks independent provenance, interface and report consistency. Native
forward/backward/state equality and gradient values remain observations of the
pinned GPU helper; no Torch import, tensor reload or numerical replay occurs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

SPEC_SCHEMA = 'ROOT-query-pixel-posterior-native-preflight-entry-spec-v1'
QUAL_SCHEMA = 'ROOT-query-pixel-posterior-native-preflight-CPU-qualification-v1'
QUAL_STATUS = 'ROOT_QUERY_PIXEL_POSTERIOR_NATIVE_PREFLIGHT_CPU_QUALIFIED'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
PRIOR = (('teacher_receipt', 'teacher_phase'), ('preflight_receipt', 'preflight_phase'),
    ('short2_receipt', 'short2_phase'), ('prior_DEV_receipt', 'prior_DEV_phase'),
    ('prior_diag_receipt', 'prior_diag_phase'), ('previous_control_receipt', 'previous_control_phase'))
PINS = {
    'entry_source': (13295, '6da8bbec0f1760b9af29386361ab44fada5c457bb0e3cf43fba68fccfca57780'),
    'preflight_helper_source': (10977, '24322e4a16d38c3331c5e8dcd1ee142c1393a174cfd986fa231c98f14f76ad95'),
    'posterior_readout_source': (11555, '5c309b4d1a2318e8c4d673d343dbe956f1387d4b01180de8cd796d0726457305'),
    'native_adapter_source': (7873, 'b87ba100eefd560779fe45787534ca817436ad9a08cfb8d029e1c8762d74ede2')}
PARAMETERS = {'feature_projection.weight': 4096, 'feature_projection.bias': 16,
    'query_projection.weight': 4096, 'query_projection.bias': 16,
    'hidden.weight': 1152, 'hidden.bias': 32, 'final.weight': 32, 'final.bias': 1}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def expect(row, values, label):
    require(isinstance(row, dict), label + ' must be an object')
    for key, value in values.items():
        require(key in row and row[key] == value and
            (type(value) not in (bool, int) or type(row[key]) is type(value)), label + ': ' + key)


def finite(value, label, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value) and
        (not nonnegative or value >= 0), 'Finite scalar required: ' + label)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


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
        set(ref['sha256']) <= set('0123456789abcdef') and reference(ref['path']) == ref,
        'Exact pinned reference differs')
    return Path(ref['path'])


def terminal(raw, phase):
    expect(raw, dict(status='ACTUAL_OWNED_TERMINAL', returncode=0, root_wait_reaped=True,
        held_handle_exit=dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0), abort=None, error=None,
        wait_errors=False, GPU_observation_errors=[], whole_GPU_used_limit_MiB=8192,
        whole_leaf_VRAM_certified=False, GPU_scope='whole GPU observed, not per-process WDDM allocation'),
        'STOP before hashes/report/imports: actual owned terminal')
    require(raw.get('original_identity') is not None and raw.get('phase', phase) == phase and
        raw.get('numeric_birth_absence', {}).get('original_birth_absent') is True, 'Actual identity/birth absence differs')
    finite(raw.get('whole_GPU_observed_peak_MiB'), 'Whole GPU peak', True)
    require(raw['whole_GPU_observed_peak_MiB'] <= 8192, 'Whole GPU observation exceeds 8192MiB')


def gate(spec_path, spec):
    # Only the two JSON metadata objects needed to locate raw paths precede
    # the gate. No reference is hashed and no report/payload/SOURCE is loaded.
    expect(spec, dict(schema=SPEC_SCHEMA, CPU_affinity=[0, 1], allocator_limit_MiB=4096,
        whole_GPU_owner_limit_MiB=8192), 'Original preflight SPEC')
    require(type(spec['deadline_seconds']) is int and spec['deadline_seconds'] > 0 and
        Path(spec['context']['path']).is_absolute(), 'Original resource/context metadata differs')
    owner_root = spec_path.parent / 'finite_point_cache_execution_20261005_v2'
    receipt_path = owner_root / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    raw = {'current': read(receipt_path)}
    terminal(raw['current'], spec['phase'])
    ctx = read(spec['context']['path'])
    for receipt, phase in PRIOR:
        require(Path(ctx[receipt]['path']).is_absolute(), 'Absolute prior raw path required')
        raw[receipt] = read(ctx[receipt]['path'])
        terminal(raw[receipt], ctx[phase])
    expect(ctx, dict(schema='ROOT-shared-readout-objective-control-fullFIT24-spec-v1', expected_epoch=24), 'Construction context')
    require(type(ctx['readout_seed']['seed_id']) is int and ctx['readout_seed']['seed_id'] == 0,
        'Original seed0 construction context required')
    refs = dict(source=reference(spec['entry_source']['path']), reader=reference(__file__), spec=reference(spec_path),
        receipt=reference(receipt_path), owner=reference(owner_root / 'run_owned_gpu_limited.py'))
    require(Path(spec['GPU_admission_file']).is_absolute(), 'Absolute current admission required')
    refs['admission'] = reference(spec['GPU_admission_file'])
    admission = read(spec['GPU_admission_file'])
    expect(admission, dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE', automatic_retry=False,
        phase=spec['phase'], allowed_cpus=[0, 1], cuda_visible_devices='0', whole_GPU_used_limit_MiB=8192,
        qualified_leaf_deadline_seconds=spec['deadline_seconds'], driver_sha256=OWNER_SHA,
        entry_args=['--spec', str(spec_path), '--owner-execute']), 'Actual current owner admission')
    require(refs['owner']['sha256'] == OWNER_SHA and raw['current']['admission_sha256'] == refs['admission']['sha256'] and
        Path(admission['entry']).resolve(strict=True) == Path(refs['source']['path']) and
        spec['entry_source'] == refs['source'], 'Actual raw/admission/SPEC/entry/owner differs')
    sources = spec['sources']
    require(len({r['path'] for r in sources}) == len(sources) and refs['reader'] in sources and
        all(ref in sources for ref in ctx['sources']) and spec['context'] in sources,
        'Reader and complete original context must be in the original GPU closure')
    hashes = admission['bound_source_sha256']
    require(hashes.get(str(spec_path)) == refs['spec']['sha256'] and
        admission.get('scientific_spec', refs['spec']) == refs['spec'], 'Original SPEC not owner-bound')
    for ref in sources:
        bound(ref)
        require(hashes.get(ref['path']) == ref['sha256'], 'Complete original closure not owner-bound')
    for role, (size, sha) in PINS.items():
        require(spec[role] in sources and spec[role]['bytes'] == size and spec[role]['sha256'] == sha,
            'Canonical immutable component differs: ' + role)
    require(ctx == read(bound(spec['context'])), 'Context changed after raw-location read')
    return ctx, raw, refs


def teacher_join(ctx, raw, sources):
    # Metadata lineage only: do not load the scientific checkpoint or rebuild a model.
    q = read(bound(ctx['teacher_qualification']))
    expect(q, dict(schema='ROOT-native-spatial-position-preserving-common24-CPU-qualification-v1',
        status='ROOT_NATIVE_SPATIAL_POSITION_PRESERVING_COMMON24_CPU_QUALIFIED', kind='mask2former', mode='native0',
        phase=ctx['teacher_phase'], source=ctx['training_source'], reader_source=ctx['training_CPU_reader'],
        spec=ctx['teacher_training_spec'], parent_receipt=ctx['teacher_receipt'], checkpoint=ctx['teacher_checkpoint'],
        all_six_nodes_CPU_qualified=True, completed_24_epoch_schedule=True, completed_epoch=24,
        global_microsteps=17784, global_updates=8892, consumed_raw_GT=39864,
        automatic_resume_eligible=False, inference_executed=False, AP_computed=False), 'Original qualified teacher')
    require(ctx['checkpointmap']['native0'] == q['checkpoint'] and ctx['qualificationmap']['native0'] == ctx['teacher_qualification'] and
        q['node_checkpointmap']['24'] == q['checkpoint'], 'Original teacher maps differ')
    train, adm = read(bound(q['spec'])), read(bound(q['GPU_admission']))
    require(raw['teacher_receipt']['admission_sha256'] == q['GPU_admission']['sha256'] and
        q['owner_driver']['sha256'] == OWNER_SHA and adm['phase'] == q['phase'] and
        Path(adm['entry']).resolve(strict=True) == Path(q['source']['path']) and
        adm['bound_source_sha256'].get(q['spec']['path']) == q['spec']['sha256'], 'Teacher original owner provenance differs')
    for ref in train['sources']:
        bound(ref)
        require(ref in sources and adm['bound_source_sha256'].get(ref['path']) == ref['sha256'], 'Teacher complete closure differs')
    for role in ('source', 'reader_source', 'report', 'start', 'owner_driver'):
        bound(q[role])
    return q


def rng_pair(rows):
    require(isinstance(rows, list) and len(rows) == 2, 'CPU/CUDA RNG digest pair required')
    for row in rows:
        require(set(row) == {'shape', 'dtype', 'sha256'} and row['dtype'] == 'uint8' and
            isinstance(row['shape'], list) and len(row['shape']) == 1 and type(row['shape'][0]) is int and
            row['shape'][0] > 0 and len(row['sha256']) == 64 and set(row['sha256']) <= set('0123456789abcdef'), 'RNG digest contract differs')


def matching_check(matching, expected, spec):
    anns, count = expected['annotation_ids'], len(expected['annotation_ids'])
    expect(matching, dict(image_id=expected['image_id'], raw_ann_ids=anns, seed=spec['matching_seed'], points=12544,
        weights=[2.0, 5.0, 5.0], native_get_targets_calls=1, external_RNG_restored=True,
        actual_total_is_native_GPU_stack_sum_then_CPU=True, CPU_cost_derivation_claimed=False), 'Actual native matching')
    rows, queries, gt = matching['rows'], matching['positive_query_indices'], matching['assigned_GT_indices']
    require(len(rows) == len(queries) == len(gt) == count and all(type(q) is int and 0 <= q < 100 for q in queries) and
        len(set(queries)) == count and all(type(j) is int for j in gt) and set(gt) == set(range(count)), 'Complete one-to-one rawGT assignment differs')
    require(rows == [dict(annotation_id=ann, GT_index=j, query_index=queries[gt.index(j)]) for j, ann in enumerate(anns)] and
        matching['query_by_GT'] == [row['query_index'] for row in rows], 'Actual query-to-original GT order differs')
    require(len(matching['cost_parts']) == len(matching['cost_digests']) == 3, 'Three actual native cost parts required')
    for matrix, digest in zip(matching['cost_parts'] + [matching['actual_total']],
                              matching['cost_digests'] + [matching['actual_total_digest']]):
        require(isinstance(matrix, list) and len(matrix) == 100 and all(isinstance(row, list) and len(row) == count for row in matrix), 'Native cost100xGT dimensions differ')
        for row in matrix:
            for value in row:
                finite(value, 'Exported native cost')
        require(set(digest) == {'shape', 'dtype', 'sha256'} and digest['shape'] == [100, count] and digest['dtype'] == 'float32' and
            len(digest['sha256']) == 64 and set(digest['sha256']) <= set('0123456789abcdef'), 'Native cost digest metadata differs')
    rng_pair(matching['private_RNG_before'])
    rng_pair(matching['private_RNG_after'])


def report_check(spec, ctx, refs):
    expected = spec['expected_row']
    require(type(expected['dataset_index']) is int and expected['dataset_index'] == 0 and type(expected['image_id']) is int and
        expected['image_id'] > 0 and isinstance(expected['annotation_ids'], list) and 0 < len(expected['annotation_ids']) <= 100 and
        len(set(expected['annotation_ids'])) == len(expected['annotation_ids']) and
        all(type(i) is int and i > 0 for i in expected['annotation_ids']), 'Prelocked canonical first FIT row differs')
    require(all(type(spec[k]) is int and 0 <= spec[k] <= 2**64 - 1 for k in ('readout_seed', 'point_seed')) and
        spec['matching_seed'] == 2800000 + expected['image_id'], 'Immutable explicit seed protocol differs')
    report_ref = reference(Path(spec['output_directory']) / 'REPORT.json')
    report = read(bound(report_ref))
    expect(report, dict(schema='query-pixel-posterior-native-preflight-entry-report-v1',
        status='QUERY_PIXEL_POSTERIOR_NATIVE_PREFLIGHT_ENTRY_BACKWARD_PASS', source=refs['source'], spec=refs['spec'],
        phase=spec['phase'], context=spec['context'], GPU_admission=refs['admission'], prior_six_raw_gates_before_hashes_imports=True,
        context_used_for_construction_metadata_only=True, old_readout_checkpoint_loaded=False,
        optimizer_steps=0, AP_computed=False, efficacy_claimed=False), 'Actual completed preflight report')
    expect(report['teacher_loading'], dict(checkpoint=ctx['teacher_checkpoint'], qualification=ctx['teacher_qualification'], mode='native0',
        strict_fullstate_loaded=True, native_state_tensors=610, registered_parameters=451, original_frozen_BN_exact=True,
        inference_backbone_with_cp=False, no_random_missing_keys=True), 'SOURCE-observed strict CPU teacher load')
    expect(report['cleanup'], dict(allocated_after=0), 'Actual cleanup')
    finite(report['cleanup']['allocated_before'], 'Allocation before cleanup', True)
    finite(report['elapsed_seconds'], 'Elapsed', True)
    require(report['elapsed_seconds'] < spec['deadline_seconds'] and type(report['torch_peak_allocated_bytes']) is int and
        0 <= report['torch_peak_allocated_bytes'] <= 4096 * 1024**2, 'Actual deadline/allocator report differs')
    result = report['result']
    expect(result, dict(schema='query-pixel-posterior-native-injected-preflight-v1',
        status='QUERY_PIXEL_POSTERIOR_NATIVE_INJECTED_PREFLIGHT_BACKWARD_PASS', image_id=expected['image_id'], raw_ann_ids=expected['annotation_ids'],
        native_forward_calls=1, backward_calls=2, readout_seed=spec['readout_seed'], point_seed=spec['point_seed'], points_per_GT=12544,
        native610_exact=True, native451_frozen_gradNone=True, BN106_affine159_buffers_exact=True, all_native_modules_eval=True,
        native_all10_tensor_versions_unchanged=True, external_RNG_restored=True, hooks_removed=True,
        raw_processed_GT_source_row_validation='ROOT caller qualified shared.original_targets', no_optimizer_step=True,
        warm_perturbation_is_structural_only=True, all100_new_head_inference_qualified=False, AP_computed=False,
        calibration_qualified=False, efficacy_claimed=False, GPU_backward_bitwise_determinism_claimed=False), 'Pinned helper observations')
    forward = result['forward']
    expect(forward, dict(head_calls=1, backbone_calls=1, pixel_decoder_calls=1, mask_embed_calls=10,
        seed=100000 + expected['image_id'], external_RNG_restored=True, hooks_removed=True, all4_GT_fields_absent=True,
        classes_all10_detached=True, first9_masks_detached=True, native_all10_tensor_versions_unchanged=None,
        classes10=None, native_masks10=None), 'One actual shared native forward')
    rng_pair(forward['private_RNG_before'])
    rng_pair(forward['private_RNG_after'])
    matching_check(result['matching'], expected, spec)
    require(set(result['stages']) == {'cold', 'warm'}, 'Exactly two structural backward stages required')
    for stage, row in result['stages'].items():
        finite(row['loss'], stage + ' BCE', True)
        require(set(row['gradients']) == set(PARAMETERS), 'Exactly eight named parameter gradients required')
        require(row['cold_native_point_BCE_exact'] is (True if stage == 'cold' else None), 'Cold native BCE identity scope differs')
        for name, gradient in row['gradients'].items():
            require(set(gradient) == {'norm', 'nonzero'} and type(gradient['nonzero']) is int and
                0 <= gradient['nonzero'] <= PARAMETERS[name], 'Actual gradient scalar metadata differs')
            finite(gradient['norm'], stage + ':' + name, True)
            active = stage == 'warm' or name.startswith('final.')
            require((gradient['nonzero'] > 0 and gradient['norm'] > 0) if active else
                (gradient['nonzero'] == 0 and gradient['norm'] == 0), 'Actual cold/warm gradient activity differs')
    return report_ref, report, result


def run(spec_file, out_file, owner_execute):
    started = time.monotonic()
    require(owner_execute, 'Explicit --owner-execute required; ROOT alone executes')
    path, out = Path(spec_file), Path(out_file)
    require(path.is_absolute() and out.is_absolute() and not out.exists() and out.parent.is_dir() and
        os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules,
        'Absolute original SPEC/fresh ABS output/CUDA-empty/no Torch required')
    path = path.resolve(strict=True)
    spec = read(path)
    ctx, raw, refs = gate(path, spec)
    teacher_join(ctx, raw, spec['sources'])
    report_ref, report, result = report_check(spec, ctx, refs)
    import psutil
    require(psutil.Process().cpu_affinity() == [0, 1] and 'torch' not in sys.modules, 'ROOT CPU01/no scientific runtime required')
    for ref in spec['sources']:
        bound(ref)
    for ref in (refs['spec'], refs['receipt'], refs['admission'], refs['owner'], report_ref):
        bound(ref)
    require(time.monotonic() - started < spec['deadline_seconds'], 'CPU consistency deadline exceeded')
    q = dict(schema=QUAL_SCHEMA, status=QUAL_STATUS, phase=spec['phase'], source=refs['source'], reader_source=refs['reader'],
        spec=refs['spec'], report=report_ref, parent_receipt=refs['receipt'], context=spec['context'], GPU_admission=refs['admission'],
        owner_driver=refs['owner'], preflight_helper_source=spec['preflight_helper_source'],
        posterior_readout_source=spec['posterior_readout_source'], native_adapter_source=spec['native_adapter_source'],
        teacher_checkpoint=ctx['teacher_checkpoint'], teacher_qualification=ctx['teacher_qualification'],
        optimizer_steps=0, native_forward_calls=1, backward_calls=2, points_per_GT=12544,
        image_id=result['image_id'], raw_ann_ids=result['raw_ann_ids'], readout_seed=result['readout_seed'], point_seed=result['point_seed'],
        forward_seed=result['forward']['seed'], matching_seed=result['matching']['seed'],
        native610_exact=True, native451_frozen_gradNone=True, BN106_affine159_buffers_exact=True,
        cold_native_point_BCE_exact=True, cold_final_only_gradient_activity=True, warm_all8_finite_positive_gradient_activity=True,
        native_all10_tensor_versions_unchanged=True, hooks_removed=True, external_RNG_restored=True,
        stages=result['stages'], fixed_matching_query_by_GT=result['matching']['query_by_GT'],
        current_and_six_prior_actual_raw_terminal0_before_hashes_report_imports=True,
        reader_in_original_GPU_owner_closure=True, complete_current_and_context_closure_bound=True,
        strict_CPU_teacher610_load_SOURCE_observed=True, GPU_forward_backward_state_checks_SOURCE_observed=True,
        independent_lineage_interface_report_consistency=True, independent_numerical_forward_backward_replay=False,
        bounded_delta_formula_SOURCE_bound=True, bounded_delta_GPU_numerically_verified=False,
        all100_new_head_inference_qualified=False, AP_computed=False, efficacy_claimed=False, calibration_qualified=False,
        formal_training_or_resume_qualified=False, GPU_backward_bitwise_determinism_claimed=False,
        CPU_execution=dict(device='cpu', cpu_affinity=[0, 1], cuda_visible_devices='', torch_imported=False,
            CUDA_initialized_by_reader=False, requested_torch_threads=2, requested_torch_interop_threads=2,
            Torch_thread_counts_measured=False), elapsed_seconds=time.monotonic() - started,
        evidence_limits=[
            'CPU independently checks pinned lineage, original GPU closure, interfaces and report consistency only; no model, tensor, GT or gradient replay.',
            'Strict CPU teacher load, frozen610/451/BN106+159, raw/processed GT alignment, native matching and cold/warm gradient checks are pinned SOURCE-observed GPU observations.',
            'Cost arrays/digest metadata are checked for dimensions and finite values; native costs or Hungarian optimality are not independently derived.',
            'The bound readout SOURCE implements kappa*tanh(u) with default kappa=1; no bounded-delta numerical values were recorded in this preflight report.',
            'One image, one teacher forward and two constructed structural backward checks with no optimizer do not qualify training, all100 inference, AP, calibration or efficacy.',
            'No Torch is imported by this CPU consumer; requested 2+2 Torch threads are metadata and no scientific thread count is measured.',
            'The ROOT actual CPU tool terminal is separate execution evidence, not proved by this JSON file.'])
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(q, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=QUAL_STATUS, output=str(out))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    run(args.spec, args.out, args.owner_execute)


if __name__ == '__main__':
    main()
