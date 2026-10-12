"""ROOT-only seed1 warm48 manifest construction; no training or CPU replay.

Future ROOT execution checks eight original raw terminals before any SHA or
qualification read. Only sealed stdlib provenance gates are called afterward;
producer.run/execute and scientific imports are never called by this builder.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import types

H = Path(r'E:\SAR\seal\research_round_20260925_global\multimodal_data_pilot_20260928\goal_continuation_20261005')
PHASE = 'query_pixel_posterior_scalar_control_E24warm_additional48_Puncap_eval_seed1_20261011_01'
BASE = 'ROOT_QUERY_PIXEL_POSTERIOR_SCALAR_CONTROL_E24WARM_ADDITIONAL48_PUNCAP_EVAL_SEED1'
TEMPLATE = H / 'ROOT_QUERY_PIXEL_POSTERIOR_SCALAR_CONTROL_E24WARM_ADDITIONAL48_PUNCAP_EVAL_SEED0_SPEC_20261011_01.json'
FIRST_Q = H / 'ROOT_QUERY_PIXEL_POSTERIOR_SCALAR_CONTROL_FULLFIT24_SEED1_CPU_QUALIFICATION_20261011_01.json'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
PRIOR = (('teacher_receipt', 'teacher_phase'), ('preflight_receipt', 'preflight_phase'),
    ('short2_receipt', 'short2_phase'), ('prior_DEV_receipt', 'prior_DEV_phase'),
    ('prior_diag_receipt', 'prior_diag_phase'), ('previous_control_receipt', 'previous_control_phase'))
COMPONENTS = ('posterior_readout_source', 'native_adapter_source', 'preflight_helper_source',
    'native_preflight_entry_source', 'native_preflight_CPU_reader_source', 'native_preflight_spec',
    'native_preflight_CPU_qualification', 'native_preflight_report', 'native_preflight_receipt',
    'query_iou_control_source', 'scalar_CPU_fixture', 'scalar_fixture_source')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    require(Path(path).is_absolute(), 'Absolute input required')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest, count = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
            count += len(block)
    return dict(path=str(path), bytes=count, sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'} and
        Path(row['path']).is_absolute() and reference(row['path']) == row, 'Exact pinned reference differs')
    return Path(row['path'])


def terminal(row, phase):
    require(row['status'] == 'ACTUAL_OWNED_TERMINAL' and row['original_identity'] is not None and
        type(row['returncode']) is int and row['returncode'] == 0 and row['root_wait_reaped'] is True and
        row['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        row['numeric_birth_absence']['original_birth_absent'] is True and row['abort'] is None and
        row['error'] is None and row['wait_errors'] is False and row['GPU_observation_errors'] == [],
        'STOP before hashes: actual held0/reap/original-birth absence/noerror required')
    peak = row['whole_GPU_observed_peak_MiB']
    require(row.get('phase', phase) == phase and row['whole_GPU_used_limit_MiB'] == 8192 and
        row['whole_leaf_VRAM_certified'] is False and row['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation' and
        type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192, 'Raw GPU scope differs')


def closure(rows):
    result, by_path = [], {}
    for row in rows:
        bound(row)
        if row['path'] in by_path:
            require(by_path[row['path']] == row, 'Conflicting pins for one closure path')
        else:
            by_path[row['path']] = row
            result.append(row)
    return result


def load_checks(row):
    source = bound(row).read_bytes()
    require(hashlib.sha256(source).hexdigest() == row['sha256'], 'Producer SOURCE changed before compilation')
    module = types.ModuleType('sealed_warm48_manifest_provenance_only')
    module.__file__ = row['path']
    exec(compile(source, row['path'], 'exec'), module.__dict__)
    return module


def rewrite(value, ref_changes, strings, old_closure, new_closure):
    """Clone current-job metadata, replacing complete owner source lists."""
    for old, new in ref_changes:
        if value == old:
            return deepcopy(new)
    if isinstance(value, list):
        if value == old_closure:
            return deepcopy(new_closure)
        return [rewrite(item, ref_changes, strings, old_closure, new_closure) for item in value]
    if isinstance(value, dict):
        return {key: rewrite(item, ref_changes, strings, old_closure, new_closure) for key, item in value.items()}
    if isinstance(value, str):
        for old, new in strings:
            value = value.replace(old, new)
    return value


def write_fresh(path, value):
    temp = path.with_name(path.name + '.tmp')
    require(not path.exists() and not temp.exists(), 'Fresh atomic destination required')
    with temp.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    temp.replace(path)


def build(first24_spec_path, deadline_seconds):
    require(type(deadline_seconds) is int and deadline_seconds > 0, 'ROOT must prelock a positive deadline')
    previous_path = Path(first24_spec_path)
    require(previous_path.is_absolute() and previous_path.parent.resolve() == H.resolve(), 'Own first24 SPEC must be in canonical H')
    # Bootstrap JSON reads obtain raw paths only; no file SHA occurs here.
    template, previous = read(TEMPLATE), read(previous_path)
    require(template['context'] == previous['context'], 'Same original teacher context required')
    ctx = read(previous['context']['path'])
    raw = {}
    for receipt, phase in PRIOR:
        raw[receipt] = read(ctx[receipt]['path'])
        terminal(raw[receipt], ctx[phase])
    raw['native_preflight_receipt'] = read(previous['native_preflight_receipt']['path'])
    terminal(raw['native_preflight_receipt'], previous['native_preflight_phase'])
    require(Path(previous['phase']).name == previous['phase'], 'Safe own first24 phase required')
    first_receipt_path = H / 'finite_point_cache_execution_20261005_v2' / 'runs' / previous['phase'] / 'PARENT_RECEIPT.json'
    raw['first24_receipt'] = read(first_receipt_path)
    terminal(raw['first24_receipt'], previous['phase'])
    # ALL eight original held0 gates have passed before the first SHA/Q/admission access.
    previous_ref, q_ref, template_ref = reference(previous_path), reference(FIRST_Q), reference(TEMPLATE)
    q = read(bound(q_ref))
    require(q['spec'] == previous_ref and q['context'] == previous['context'] and
        q['phase'] == previous['phase'] and q['parent_receipt'] == reference(first_receipt_path), 'Own first24 Q/SPEC/raw/context join differs')
    require((previous['readout_seed'], previous['order_seed'], previous['point_seed_base']) == (2026100912, 2026101022, 3900000) and
        (q['readout_seed'], q['order_seed'], q['point_seed_base']) == (2026100912, 2026101022, 3900000), 'Only own seed1 may warm seed1')
    require(ctx['schema'] == 'ROOT-shared-readout-objective-control-fullFIT24-spec-v1' and
        ctx['expected_epoch'] == 24 and ctx['readout_seed']['seed_id'] == 0, 'Original seed0 teacher construction context must remain unchanged')
    here = H.parent / 'native_geometry_paired_mask_readout_preparation_20261009' / 'query_pixel_posterior_scalar_control_E24warm_additional48_Puncap_eval_repeat_seed1_source_20261011'
    fresh = {role: reference(here / name) for role, name in
        (('entry_source', 'train.py'), ('CPU_reader', 'qualify_cpu.py'), ('SOURCE_proposal', 'SOURCE_PROPOSAL.json'))}
    writer = load_checks(fresh['entry_source'])
    proposal = read(bound(fresh['SOURCE_proposal']))
    require(proposal['schema'] == writer.PREFIX + '-SOURCE-proposal-v1' and proposal['producer_SOURCE'] == fresh['entry_source'] and
        proposal['SPEC_schema'] == writer.SCHEMA and proposal['CPU_numeric_protocol'] == writer.CPU_NUMERIC_PROTOCOL and
        proposal['controls'] == list(writer.CONTROLS) and proposal['first24_controls'] == list(writer.FIRST24_CONTROLS) and
        (proposal['readout_seed'], proposal['order_seed'], proposal['point_seed_base']) == (2026100912, 2026101022, 3900000),
        'Fresh SOURCE proposal/seed/control binding differs')
    spec_path, admission_path = H / (BASE + '_SPEC_20261011_01.json'), H / (BASE + '_ADMISSION_20261011_01.json')
    output = H / 'QUERY_PIXEL_POSTERIOR_SCALAR_CONTROL_E24WARM_ADDITIONAL48_PUNCAP_EVAL_SEED1_01'
    acceptance = H / (BASE + '_CPU_QUALIFICATION_20261011_01.json')
    for path in (spec_path, admission_path, output, acceptance):
        require(not path.exists() and not path.with_name(path.name + '.tmp').exists() and path.parent.is_dir(), 'Fresh planned destination required: ' + str(path))
    template_admission_ref = reference(template['GPU_admission_file'])
    old_admission = read(bound(template_admission_ref))
    require(old_admission['driver_sha256'] == writer.OWNER_SHA == OWNER_SHA and
        old_admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        old_admission['automatic_retry'] is False and old_admission['phase'] == template['phase'] and
        old_admission['entry_args'] == ['--spec', str(TEMPLATE), '--owner-execute'] and
        Path(old_admission['entry']).resolve() == Path(template['entry_source']['path']).resolve() and
        old_admission['bound_source_sha256'].get(str(TEMPLATE)) == template_ref['sha256'], 'Original warm48 owner template differs')
    spec = deepcopy(template)
    spec.update(fresh)
    spec.update(schema=writer.SCHEMA, phase=PHASE, context=previous['context'], GPU_admission_file=str(admission_path),
        output_directory=str(output), arms=list(writer.ARMS), controls=list(writer.CONTROLS), epochs=48,
        microsteps=35568, optimizer_updates=17784, raw_GT_visits=79728, accumulation=2, readout_seed=2026100912,
        order_seed=2026101022, point_seed_base=3900000, kappa=1., phase_epochs=48, effective_readout_epochs=72,
        teacher_epoch=24, total_data_exposure_epochs=96, LR_schedule=True,
        schedule=dict(milestones=[15808, 17126], gamma=.1, warmup_updates=0, warmup_start_factor=1),
        CPU_affinity=[0, 1], allocator_limit_MiB=4096, whole_GPU_owner_limit_MiB=8192,
        deadline_seconds=deadline_seconds, no_feature_cache=True, first24_training_spec=previous_ref,
        first24_CPU_qualification=q_ref, first24_report=q['report'], first24_receipt=q['parent_receipt'],
        first24_phase=previous['phase'], first24_entry_source=q['source'], first24_CPU_reader_source=q['reader_source'],
        first24_SOURCE_proposal=q['SOURCE_proposal'], warm_checkpoints=q['artifacts']['checkpoints'])
    for role in COMPONENTS + ('native_preflight_phase',):
        require(template[role] == previous[role], 'Teacher/integration component changed: ' + role)
        spec[role] = deepcopy(previous[role])
    spec.update(manifest_builder_source=reference(__file__), construction_template_spec=template_ref,
        construction_template_GPU_admission=template_admission_ref)
    evidence = [q[role] for role in ('GPU_admission', 'start', 'GPU_qualification')]
    evidence += [q['artifacts'][role] for role in ('state_proof', 'matching')]
    spec['sources'] = closure(ctx['sources'] + previous['sources'] + list(fresh.values()) +
        [spec[role] for role in writer.NEW_ROLES] + list(spec['warm_checkpoints'].values()) + evidence +
        [spec['context'], spec['manifest_builder_source'], template_ref, template_admission_ref])
    require(all(key in spec for key in proposal['required_SPEC_keys']), 'Required current SPEC field missing')
    writer.first24_gate(spec, raw['first24_receipt'])  # Only stdlib metadata; no CP load/run/execute.
    ref_changes = [(template[role], spec[role]) for role in
        ('entry_source', 'CPU_reader', 'SOURCE_proposal', 'first24_training_spec', 'first24_CPU_qualification',
        'first24_report', 'first24_receipt', 'first24_entry_source', 'first24_CPU_reader_source', 'first24_SOURCE_proposal')]
    ref_changes += [(template['warm_checkpoints'][arm], spec['warm_checkpoints'][arm]) for arm in writer.ARMS]
    strings = [(template['phase'], PHASE), (template['first24_phase'], previous['phase']),
        (str(TEMPLATE), str(spec_path)), (template['output_directory'], str(output)), (template['GPU_admission_file'], str(admission_path))]
    for old, new in ref_changes:
        strings += [(old['path'], new['path']), (old['sha256'], new['sha256'])]
    admission = rewrite(old_admission, ref_changes, strings, template['sources'], spec['sources'])
    admission.update(phase=PHASE, entry=fresh['entry_source']['path'], entry_args=['--spec', str(spec_path), '--owner-execute'],
        qualified_leaf_deadline_seconds=deadline_seconds, automatic_retry=False, driver_sha256=OWNER_SHA,
        allowed_cpus=[0, 1], cuda_visible_devices='0', whole_GPU_used_limit_MiB=8192)
    for key in ('deadline_seconds', 'leaf_deadline_seconds'):
        if key in admission:
            admission[key] = deadline_seconds
    write_fresh(spec_path, spec)
    spec_ref = reference(spec_path)
    admission = rewrite(admission, [(template_ref, spec_ref)], [(template_ref['sha256'], spec_ref['sha256'])], template['sources'], spec['sources'])
    admission['scientific_spec'] = spec_ref
    admission['bound_source_sha256'] = {r['path']: r['sha256'] for r in spec['sources']}
    admission['bound_source_sha256'][str(spec_path)] = spec_ref['sha256']
    for key in ('source_count', 'source_closure_count'):
        if key in admission:
            admission[key] = len(spec['sources'])
    if 'bound_source_count' in admission:
        admission['bound_source_count'] = len(admission['bound_source_sha256'])
    write_fresh(admission_path, admission)
    writer.gate(spec_path, spec)  # Sealed provenance only, still no scientific execution.
    for row in spec['sources']:
        bound(row)
    print(json.dumps(dict(status='ROOT_SEED1_WARM48_MANIFESTED_NOT_LAUNCHED', phase=PHASE,
        spec=spec_ref, GPU_admission=reference(admission_path), entry=fresh['entry_source'], CPU_reader=fresh['CPU_reader'],
        output_directory=str(output), CPU_acceptance=str(acceptance), source_count=len(spec['sources']),
        first24_CPU_qualification=q_ref, warm_checkpoints=spec['warm_checkpoints'],
        readout_seed=2026100912, order_seed=2026101022, point_seed_base=3900000,
        phase_epochs=48, effective_readout_epochs=72, total_data_exposure_epochs=96,
        scientific_execution_performed=False, AUTH_written=False, admission_requires_ROOT_review=True), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first24-spec', required=True)
    parser.add_argument('--deadline-seconds', type=int, required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Only ROOT may construct these fresh manifests')
    build(args.first24_spec, args.deadline_seconds)


if __name__ == '__main__':
    main()

