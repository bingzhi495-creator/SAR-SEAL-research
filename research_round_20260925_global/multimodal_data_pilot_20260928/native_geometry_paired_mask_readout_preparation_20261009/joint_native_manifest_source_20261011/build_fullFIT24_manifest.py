"""Inert ROOT authoring of fresh formal A/B joint additional24 manifests.

Future explicit --owner-author reads only bound lineage metadata and SOURCE.
Original four raws, native-restoration raw, every tiny prior and tiny actual raw
precede all hashes/SOURCE execution. No model, CP loading, GT decode, scientific
library, launch, AUTH, retry or performance selection is performed.
Actual tiny CPU tool exit0 must also be confirmed by ROOT outside these files.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import types

PHASE = 'query_pixel_joint_native_fullFIT24_20261011_01'
PREFIX = 'ROOT_QUERY_PIXEL_JOINT_NATIVE_FULLFIT24'
TINY_PHASE = 'query_pixel_joint_native_preflight_20261011_01'
MODES = ('native0', 'global', 'spatial_halves', 'spatial_reversed')
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
ENTRY_SHA = 'e503fa980c4516defd7c58ee046d899d3af08143840f224b5f3b2b2debdd2397'
READER_SHA = '0a377d8d0093a5a2f3916b2f0e138fd3030c0610fb7420d54f403547e23b8cfa'
TINY_ENTRY_SHA = 'fedd1a1f913019de1c4c7d1960131574f06c650850732f4fb3754e29f0a677a4'
TINY_READER_SHA = '5a5d5158cacaa5b7905f960e866c5e9d56eb2bf4ac714da8f5770f91c16754e0'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    require(Path(path).is_absolute(), 'Absolute JSON locator required')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'bytes', 'sha256'} and
        Path(ref['path']).is_absolute() and type(ref['bytes']) is int and ref['bytes'] > 0 and
        isinstance(ref['sha256'], str) and len(ref['sha256']) == 64 and
        set(ref['sha256']) <= set('0123456789abcdef') and reference(ref['path']) == ref, 'Pinned input differs')
    return Path(ref['path'])


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
        raw.get('GPU_observation_errors') == [], 'STOP before hashes/imports: actual owned raw0 required')
    peak = raw.get('whole_GPU_observed_peak_MiB')
    require(raw.get('phase', phase) == phase and type(peak) in (int, float) and math.isfinite(peak) and
        0 <= peak <= 8192 and raw.get('whole_GPU_used_limit_MiB') == 8192 and
        raw.get('whole_leaf_VRAM_certified') is False and
        raw.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation', 'Raw scope differs')


def union_refs(rows):
    refs = {}
    for ref in rows:
        require(isinstance(ref, dict) and set(ref) == {'path', 'bytes', 'sha256'} and
            (ref['path'] not in refs or refs[ref['path']] == ref), 'Conflicting closure reference')
        refs[ref['path']] = ref
    return [refs[path] for path in sorted(refs)]


def write_fresh(path, value):
    require(path.is_absolute() and path.parent.is_dir() and not path.exists(), 'Fresh absolute output required')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')


def build(args):
    c = Path(__file__).resolve(strict=True).parent.parent
    h = Path(args.artifact_root)
    require(h.is_absolute() and h.resolve(strict=True) == c.parent / 'goal_continuation_20261005',
        'Canonical absolute artifact root required')
    h = h.resolve(strict=True)
    paths = {k: h / (PREFIX + '_' + k + '_20261011_01.json') for k in ('SPEC', 'ADMISSION', 'CPU_QUALIFICATION')}
    output = h / 'QUERY_PIXEL_JOINT_NATIVE_FULLFIT24_01'
    runtime = h / 'finite_point_cache_execution_20261005_v2' / 'runs' / PHASE
    require(all(not p.exists() for p in list(paths.values()) + [output, runtime]), 'Fresh outputs/runtime required; no overwrite/retry')
    template_path = h / 'ROOT_NATIVE_FINETUNE24_FORMAL_SPEC_20261010_01.json'
    tiny_path = h / 'ROOT_QUERY_PIXEL_JOINT_NATIVE_PREFLIGHT_SPEC_20261011_01.json'
    q_path = h / 'ROOT_QUERY_PIXEL_JOINT_NATIVE_PREFLIGHT_CPU_QUALIFICATION_20261011_01.json'
    template, tiny = read(template_path), read(tiny_path)  # Raw locators only until all gates pass.
    require(tiny['phase'] == TINY_PHASE and
        set(template['training_receiptmap']) == set(template['training_phasemap']) == set(MODES) and
        tiny['training_receiptmap'] == template['training_receiptmap'] and
        tiny['training_phasemap'] == template['training_phasemap'], 'Original four raw maps differ')
    raw_rows = [(tiny['training_receiptmap'][k], tiny['training_phasemap'][k]) for k in MODES]
    require(all(set(row) == {'receipt', 'phase'} for row in tiny['prior_receipts']), 'Explicit tiny prior identities required')
    raw_rows += [(row['receipt'], row['phase']) for row in tiny['prior_receipts']]
    restoration = (template['restoration_receipt'], template['restoration_phase'])
    require(restoration in raw_rows, 'Actual native restoration raw must be retained by tiny prior list')
    raw_path = h / 'finite_point_cache_execution_20261005_v2' / 'runs' / TINY_PHASE / 'PARENT_RECEIPT.json'
    require(Path(tiny['GPU_admission_file']).is_absolute(), 'Absolute tiny owner locator required')
    for ref, phase in raw_rows:
        terminal(read(ref['path']), phase)
    tiny_raw = read(raw_path); terminal(tiny_raw, TINY_PHASE)
    # No hash, qualification read, compiled SOURCE or scientific access precedes every raw gate.

    template_ref, tiny_ref, q_ref, raw_ref = [reference(p) for p in (template_path, tiny_path, q_path, raw_path)]
    require(tiny['entry_source']['sha256'] == TINY_ENTRY_SHA and tiny['CPU_reader']['sha256'] == TINY_READER_SHA,
        'Sealed tiny SOURCE identities differ')
    tiny_entry = load(tiny['entry_source'], '_ROOT_actual_tiny_provenance_ONLY')
    actual_entry, actual_spec, tiny_admission = tiny_entry.source_gate(tiny_path, tiny)
    require(actual_entry == tiny['entry_source'] and actual_spec == tiny_ref and
        tiny_raw['admission_sha256'] == tiny_admission['sha256'], 'Actual tiny owner/SPEC/raw differs')
    q = read(bound(q_ref)); report_ref = q['report']; report = read(bound(report_ref))
    require(q['schema'] == 'ROOT-query-pixel-joint-native-preflight-CPU-qualification-v1' and
        q['status'] == 'ROOT_QUERY_PIXEL_JOINT_NATIVE_PREFLIGHT_CPU_QUALIFIED' and q['source'] == actual_entry and
        q['reader_source'] == tiny['CPU_reader'] and q['spec'] == tiny_ref and q['parent_receipt'] == raw_ref and
        q['GPU_admission'] == tiny_admission and q['phase'] == TINY_PHASE and
        q['technical_microsteps_per_arm'] == 4 and q['native_updates_per_arm'] == q['B_readout_updates'] == 2 and
        q['formal_training_or_initializer_qualified'] is False and q['no_AP'] is q['no_DEV'] is q['no_TEST'] is True and
        report_ref == reference(Path(tiny['output_directory']) / 'REPORT.json'), 'Actual qualified tiny metadata differs')
    require(report['source'] == actual_entry and report['spec'] == tiny_ref and report['GPU_admission'] == tiny_admission and
        report['phase'] == TINY_PHASE and report['error'] is None and report['cleanup']['allocated_after'] == 0,
        'Actual tiny report/cleanup differs')
    for ref in tiny['sources']:
        bound(ref)
    require(tiny['native_loader_source'] == template['entry_source'] and
        all(tiny[k] == template[k] for k in ('FIT', 'image_root', 'checkpointmap', 'qualificationmap', 'initializers', 'schedule')),
        'Original teacher/data/paramwise construction differs')
    native = load(template['entry_source'], '_ROOT_native_E24_template_provenance_ONLY')
    _, template_source, actual_template, template_admission = native.source_gate(template_path, template)
    require(actual_template == template_ref and template_source == tiny['native_loader_source'] and
        template['kind'] == 'mask2former' and template['teacher_epoch'] == template['expected_epoch'] == 24 and
        template['execution_kind'] == 'formal_additional24' and template['backbone_with_cp'] is True,
        'Canonical native24 template differs')
    restoration_q = native.restoration_gate(template, template_source)
    require(restoration_q is not None and template_admission['path'] == template['GPU_admission_file'],
        'Original restoration qualification/template owner differs')

    d = c / 'query_pixel_joint_native_fullFIT24_source_20261011'
    sources = dict(entry_source=reference(d / 'train.py'), CPU_reader=reference(d / 'qualify_cpu.py'),
        manifest_builder_SOURCE=reference(__file__), joint_preflight_entry_source=actual_entry,
        joint_preflight_CPU_reader=tiny['CPU_reader'], joint_preflight_spec=tiny_ref,
        joint_preflight_CPU_qualification=q_ref, joint_preflight_report=report_ref, joint_preflight_receipt=raw_ref)
    require(sources['entry_source']['sha256'] == ENTRY_SHA and sources['CPU_reader']['sha256'] == READER_SHA,
        'Sealed formal producer/reader differs')
    writer = load(sources['entry_source'], '_ROOT_formal_joint_provenance_ONLY')
    spec = copy.deepcopy(tiny)
    spec.update(sources, schema=writer.SCHEMA, phase=PHASE, joint_preflight_phase=TINY_PHASE, arms=list(writer.ARMS),
        **writer.COUNTERS, accumulation=2, auxiliary_weight=1., schedule=writer.SCHEDULE,
        B_readout_optimizer=writer.HEAD_RECIPE, B_readout_LR_schedule=False,
        CPU_affinity=[0, 1], allocator_limit_MiB=4096, whole_GPU_owner_limit_MiB=8192, deadline_seconds=21600,
        CPU_numeric_protocol=writer.NUMERIC, kappa=1., eps=1e-6, readout_seed=2026100911,
        order_seed=2026101021, loss_seed_base=1000000, point_seed_base=3800000,
        backbone_with_cp=True, no_feature_cache=True, no_DEV_selection=True,
        GPU_admission_file=str(paths['ADMISSION']), output_directory=str(output),
        CPU_qualification_output=str(paths['CPU_QUALIFICATION']),
        serial_arms=True, technical_weights_loaded=False, old_optimizer_restored=False, old_RNG_restored=False,
        manifest_builder_scope='STDLIB_AUTHORING_ONLY_NO_SCIENCE_OR_LAUNCH',
        tiny_CPU_tool_gate_scope='Actual CPU tool exit0 requires ROOT confirmation; qualification file alone is not OS evidence')
    require(writer.COUNTERS == dict(epochs=24, microsteps=17784, optimizer_updates=8892, raw_GT_visits_per_arm=39864) and
        writer.SCHEDULE == dict(gamma=.1, milestones=[7904,8563], warmup_start_factor=1, warmup_updates=0) and
        writer.HEAD_RECIPE == dict(type='AdamW', lr=1e-4, weight_decay=.05, betas=[.9,.999], eps=1e-8),
        'Authoritative formal recipe changed')
    legacy = load(spec['evaluation_source'], '_ROOT_original_teacher_lineage_ONLY')
    teachers = legacy.qualified_training(spec); native.validate_phase_recipe(spec, teachers['native0']['training_spec'])
    extras = [template_ref, template_admission, tiny_admission] + list(sources.values())
    for row in teachers.values():
        qualification = row['qualification']
        extras += row['training_spec']['sources'] + [qualification[k] for k in
            ('source', 'reader_source', 'spec', 'report', 'start', 'GPU_admission', 'owner_driver')]
        extras += list(qualification['model_proof']['proof_refs'].values())
    spec['sources'] = union_refs(template['sources'] + tiny['sources'] + extras)
    require(all(spec[k] in spec['sources'] for k in writer.MODULES + writer.LIBRARIES + writer.PREFLIGHT_ROLES +
        ('entry_source','CPU_reader','training_CPU_reader','reused_CPU_reader','readout_source','geometry_source','hooks_source','FIT')),
        'Complete original/tiny/formal role closure required')
    for ref in spec['sources']:
        bound(ref)
    driver = h / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py'
    require(reference(driver)['sha256'] == OWNER_SHA, 'Original owner driver differs')
    admission = dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE', phase=PHASE, driver_sha256=OWNER_SHA,
        entry=sources['entry_source']['path'], entry_args=['--spec', str(paths['SPEC']), '--owner-execute'],
        allowed_cpus=[0,1], cuda_visible_devices='0', qualified_leaf_deadline_seconds=21600,
        whole_GPU_used_limit_MiB=8192, automatic_retry=False, manifest_builder_SOURCE=sources['manifest_builder_SOURCE'],
        bound_source_sha256={ref['path']:ref['sha256'] for ref in spec['sources']})
    write_fresh(paths['SPEC'], spec); spec_ref = reference(paths['SPEC'])
    admission['scientific_spec'] = spec_ref; admission['bound_source_sha256'][spec_ref['path']] = spec_ref['sha256']
    write_fresh(paths['ADMISSION'], admission)
    actual_source, actual_spec, actual_admission, _ = writer.source_gate(paths['SPEC'], spec)
    require(actual_source == sources['entry_source'] and actual_spec == spec_ref and
        actual_admission == reference(paths['ADMISSION']), 'Final SOURCE-only compatibility gate differs')
    print(json.dumps(dict(status='ROOT_JOINT_NATIVE_FULLFIT24_MANIFESTS_BUILT_NOT_EXECUTED', spec=spec_ref,
        admission=actual_admission, CPU_qualification_output=str(paths['CPU_QUALIFICATION']),
        output_directory=str(output), owner_runtime=str(runtime), source_count=len(spec['sources']),
        counters_per_arm=writer.COUNTERS, science_executed=False, training_launched=False), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', required=True); parser.add_argument('--owner-author', action='store_true')
    args = parser.parse_args(); require(args.owner_author, 'Inert SOURCE: explicit future ROOT authoring required')
    build(args)


if __name__ == '__main__':
    main()
