"""Inert future ROOT authoring of one fixed joint24 or own warm48 evaluation job.

Every original/restoration/tiny/own training raw precedes hashes, CPU
qualification reads and verified provenance SOURCE execution. No model/CP load,
GT decode, scientific import, launch, AUTH, retry or endpoint selection.
ROOT must separately confirm the selected training CPU tool actually exited0;
a qualification file is not operating-system evidence.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import types

MODES = ('native0', 'global', 'spatial_halves', 'spatial_reversed')
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
ENTRY_SHA = '4859b6bafe41c8f7e5e5cfab1813c3d428f4574d6a91aedaf077ff386919232d'
READER_SHA = '7eb9955bec54717ad0b9dc78ce85b4d0b3fb8ccea2564f2f5e9cd607171d0fe8'
TRAIN = {
    'joint24': dict(prefix='ROOT_QUERY_PIXEL_JOINT_NATIVE_FULLFIT24',
        phase='query_pixel_joint_native_fullFIT24_20261011_02',
        source_sha='6c9103a2d9507332d6396897f564474ca0cc9ba371c45b76a6a775eceed47e86',
        reader_sha='154a320a6ea6b00f6f0aa883f3274911e8f9e44797fb4d3001467c02282226ac'),
    'warm48': dict(prefix='ROOT_QUERY_PIXEL_JOINT_NATIVE_E24WARM_ADDITIONAL48',
        phase='query_pixel_joint_native_E24warm_additional48_20261011_02',
        source_sha='65d72eecff1e7614cfdbc3d67fc2b2272e2661185ebe06e205c9ee616efa8783',
        reader_sha='c946b4265139ee6e7d18e9287bec0c96447d1a6062074672e45cf749b4400afe')}


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
        raw.get('GPU_observation_errors') == [], 'STOP before hashes/imports: strict actual original raw0 required')
    peak = raw.get('whole_GPU_observed_peak_MiB')
    require(raw.get('phase', phase) == phase and type(peak) in (int, float) and math.isfinite(peak) and
        0 <= peak <= 8192 and raw.get('whole_GPU_used_limit_MiB') == 8192 and
        raw.get('whole_leaf_VRAM_certified') is False and
        raw.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation', 'Raw resource scope differs')


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
    c = Path(__file__).resolve(strict=True).parent.parent; h = Path(args.artifact_root)
    require(h.is_absolute() and h.resolve(strict=True) == c.parent / 'goal_continuation_20261005',
        'Canonical absolute artifact root required')
    h = h.resolve(strict=True); kind = args.endpoint_kind; selected = TRAIN[kind]
    prefix = selected['prefix'] + '_EVALUATION'
    phase = selected['phase'].replace('_20261011_02', '_evaluation_20261011_02')
    paths = {key: h / (prefix + '_' + key + '_20261011_02.json') for key in ('SPEC','ADMISSION','CPU_QUALIFICATION')}
    output = h / (prefix[5:] + '_02')
    runtime = h / 'finite_point_cache_execution_20261005_v2' / 'runs' / phase
    require(all(not path.exists() for path in list(paths.values()) + [output,runtime]),
        'Fresh stage-specific GPU/CPU outputs/runtime required; no overwrite or automatic retry')
    train_path = h / (selected['prefix'] + '_SPEC_20261011_02.json')
    q_path = h / (selected['prefix'] + '_CPU_QUALIFICATION_20261011_02.json')
    train = read(train_path); tiny = read(train['joint_preflight_spec']['path'])  # Raw locators only.
    require(train['phase'] == selected['phase'] and
        set(tiny['training_receiptmap']) == set(tiny['training_phasemap']) == set(MODES) and
        train['training_receiptmap'] == tiny['training_receiptmap'] and train['training_phasemap'] == tiny['training_phasemap'],
        'Selected stage/original four raw identities differ')
    raws = [(tiny['training_receiptmap'][mode],tiny['training_phasemap'][mode]) for mode in MODES]
    require(all(set(row) == {'receipt','phase'} for row in tiny['prior_receipts']), 'Explicit tiny prior raw identities required')
    raws += [(row['receipt'],row['phase']) for row in tiny['prior_receipts']]
    require((train['restoration_receipt'],train['restoration_phase']) in raws, 'Original restoration raw must be retained')
    raws.append((train['joint_preflight_receipt'],train['joint_preflight_phase']))
    if kind == 'warm48':
        raws.append((train['first24_receipt'],train['first24_phase']))
    raw_path = h / 'finite_point_cache_execution_20261005_v2' / 'runs' / selected['phase'] / 'PARENT_RECEIPT.json'
    for ref, raw_phase in raws:
        terminal(read(ref['path']), raw_phase)
    raw = read(raw_path); terminal(raw, selected['phase'])
    # No hash, Q read, compiled SOURCE or scientific access precedes all actual prerequisite raws.

    train_ref,q_ref,raw_ref = [reference(path) for path in (train_path,q_path,raw_path)]
    require(train['entry_source']['sha256'] == selected['source_sha'] and train['CPU_reader']['sha256'] == selected['reader_sha'] and
        Path(train['CPU_qualification_output']).resolve(strict=True) == q_path.resolve(strict=True), 'Sealed own training SOURCE/Q locator differs')
    training_reader = load(train['CPU_reader'], '_ROOT_selected_joint_training_provenance_ONLY')
    training, refs = training_reader.source_gate(train_path,train)
    require(refs['source'] == train['entry_source'] and refs['reader_source'] == train['CPU_reader'] and refs['spec'] == train_ref and
        refs['parent_receipt'] == raw_ref and refs['phase'] == selected['phase'] and raw['admission_sha256'] == refs['GPU_admission']['sha256'],
        'Actual selected training raw/owner/SPEC/SOURCE differs')
    q = read(bound(q_ref)); report_ref = q['report']; report = read(bound(report_ref))
    stage = dict(prefix=training.PREFIX, phase_epochs=training.COUNTERS['epochs'], microsteps=training.COUNTERS['microsteps'],
        updates=training.COUNTERS['optimizer_updates'], GT_visits=training.COUNTERS['raw_GT_visits_per_arm'],
        native_epoch=48 if kind == 'joint24' else 96, head_epoch=24 if kind == 'joint24' else 72)
    require(q['schema'] == training_reader.QUAL_SCHEMA and all(q[key] == refs[key] for key in refs) and
        q['completed_phase_epochs'] == stage['phase_epochs'] and q['effective_native_epoch'] == stage['native_epoch'] and
        q['phase_microsteps_per_arm'] == stage['microsteps'] and q['native_updates_per_arm'] == q['B_readout_updates'] == stage['updates'] and
        q['phase_raw_GT_visits_per_arm'] == stage['GT_visits'] and q['counters_per_arm'] == training.COUNTERS and
        q['CPU_execution'] == dict(device='cpu',affinity=[0,1],CUDA_VISIBLE_DEVICES='',torch_num_threads=2,torch_num_interop_threads=2,CUDA_initialized=False) and
        q['no_AP'] is q['no_DEV'] is q['no_TEST'] is True and q['formal_resume_eligible'] is False and
        report_ref == reference(Path(train['output_directory'])/'REPORT.json') and q['result'] == report['result'] and
        set(q['result']) == set(q['arms']) == {'A','B'}, 'Actual complete selected training CPU qualification differs')
    require(report['source'] == refs['source'] and report['spec'] == train_ref and report['GPU_admission'] == refs['GPU_admission'] and
        report['phase'] == selected['phase'] and report['error'] is None and report['cleanup']['allocated_after'] == 0,
        'Actual selected training report/cleanup differs')
    primary = [q[key] for key in ('report','start','GPU_qualification','GPU_admission')]; checkpoints = {}
    for arm in ('A','B'):
        row = q['result'][arm]; summary = q['arms'][arm]; checkpoints[arm] = row['checkpoint']
        require(row['checkpoint'] == summary['checkpoint'] and summary['typed_BN106_affine159_buffers_CPU_exact'] is True and
            row['completed_phase_epochs'] == stage['phase_epochs'] and row['effective_native_epoch'] == stage['native_epoch'] and
            row['phase_microsteps'] == stage['microsteps'] and row['phase_updates'] == stage['updates'] and
            row['phase_raw_GT_visits'] == stage['GT_visits'] and len(row['audits']) == 4, 'Own arm endpoint/primary counters differ')
        primary += [row[key] for key in ('checkpoint','endpoint_proof','trace','order','epochs','start')] + row['audits']
    if kind == 'warm48':
        require(q['completed_joint_training_epochs'] == q['effective_B_readout_epoch'] == 72 and q['weights_warm_start'] is True and
            q['first24_CPU_qualification'] == train['first24_CPU_qualification'], 'Own first24->warm48 qualification differs')

    d = c / 'query_pixel_joint_native_evaluation_source_20261011_first24_02'
    sources = dict(entry_source=reference(d/'evaluate.py'), CPU_reader=reference(d/'qualify_cpu.py'),
        manifest_builder_SOURCE=reference(__file__), joint_training_entry_source=refs['source'],
        joint_training_CPU_reader_source=refs['reader_source'], joint_training_spec=train_ref,
        joint_training_CPU_qualification=q_ref, joint_training_report=report_ref, joint_training_receipt=raw_ref,
        posterior_dense_source=reference(c/'query_pixel_posterior_scalar_control_E24warm_additional48_Puncap_eval_source_20261011'/'train.py'),
        official_export_source=reference(c/'shared_readout_objective_control_source_20261010'/'train.py'),
        native_adapter_source=reference(c/'query_pixel_posterior_coupled_readout_source_20261010'/'native_adapter.py'),
        official_evaluation_CPU_reader=reference(c/'spatial_position_preserving_common24_evaluation_source_20261009'/'qualify_evaluation_cpu.py'))
    require(sources['entry_source']['bytes'] == 41278 and sources['entry_source']['sha256'] == ENTRY_SHA and
        sources['CPU_reader']['sha256'] == READER_SHA and sources['official_evaluation_CPU_reader']['bytes'] == 49923 and
        sources['official_evaluation_CPU_reader']['sha256'] == '2357929fcd38ffc3020c5b8d204ee371da8da3972ff2e2e03398c8225c55e916',
        'Sealed fixed evaluation producer/reader/official helper differs')
    pure_pins = dict(posterior_dense_source='a6db1111c4afa99c734150d116313121ca35d38500b3533caaee884f5a087996',
        official_export_source='114af52ba6f64e4fed3e93f74177d8920bccca0b5a84aaef692a7b1854f611d6',
        native_adapter_source='b87ba100eefd560779fe45787534ca817436ad9a08cfb8d029e1c8762d74ede2')
    require(all(sources[key]['sha256'] == value for key,value in pure_pins.items()), 'Sealed pure evaluation helper differs')
    writer = load(sources['entry_source'], '_ROOT_fixed_joint_evaluation_provenance_ONLY')
    require(all(writer.STAGES[kind][key] == value for key,value in stage.items()) and
        q['status'] == writer.STAGES[kind]['qual_status'] and report['schema'] == stage['prefix']+'-report-v1' and
        report['status'] == writer.STAGES[kind]['report_status'], 'Prelocked stage/SOURCE/report status differs')
    spec = copy.deepcopy(train)
    spec.update(sources, schema=writer.SCHEMA, phase=phase, endpoint_kind=kind, joint_training_phase=selected['phase'],
        joint_checkpoints=checkpoints, arms=list(writer.ARMS), controls=list(writer.CONTROLS), kind='mask2former',
        evaluation_images=187, evaluation_raw_GT=380, head_seed_base=100000, CPU_affinity=[0,1],
        allocator_limit_MiB=4096, whole_GPU_owner_limit_MiB=8192, deadline_seconds=21600,
        CPU_numeric_protocol=writer.NUMERIC, kappa=1., eps=1e-6, endpoint_selection=False,
        historically_exposed_DEV=True, no_TEST=True, GPU_admission_file=str(paths['ADMISSION']),
        output_directory=str(output), CPU_qualification_output=str(paths['CPU_QUALIFICATION']),
        manifest_builder_scope='STDLIB_AUTHORING_ONLY_NO_SCIENCE_OR_LAUNCH',
        training_CPU_tool_gate_scope='Actual CPU tool exit0 requires ROOT confirmation; qualification file alone is not OS evidence')
    spec['DEV'] = reference(c.parent.parent/'experiments'/'public_baseline_20260925'/'protocol'/'pseg_ssdd'/'dev.json')
    require(spec['DEV']['bytes'] == 118319 and spec['DEV']['sha256'] == writer.DEV_SHA and len(writer.CONTROLS) == 6 and
        spec['joint_checkpoints'] == {arm:q['arms'][arm]['checkpoint'] for arm in writer.ARMS}, 'Fixed DEV/six own endpoint controls differ')
    legacy = load(spec['evaluation_source'], '_ROOT_original_joint_initializer_lineage_ONLY'); teachers = legacy.qualified_training(spec)
    extras = list(sources.values()) + [spec['DEV']] + primary
    for row in teachers.values():
        original = row['qualification']; extras += row['training_spec']['sources'] + [original[key] for key in
            ('source','reader_source','spec','report','start','GPU_admission','owner_driver')] + list(original['model_proof']['proof_refs'].values())
    spec['sources'] = union_refs(train['sources'] + extras)
    require(all(spec[key] in spec['sources'] for key in writer.TRAIN_ROLES + writer.PURE_ROLES + training.LIBRARIES +
        ('entry_source','CPU_reader','official_evaluation_CPU_reader','DEV')) and all(ref in spec['sources'] for ref in primary),
        'Complete own training/original/library/DEV/CPU helper closure required')
    for ref in spec['sources']:
        bound(ref)
    driver = h/'finite_point_cache_execution_20261005_v2'/'run_owned_gpu_limited.py'
    require(reference(driver)['sha256'] == OWNER_SHA, 'Original owner driver differs')
    admission = dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE',phase=phase,driver_sha256=OWNER_SHA,
        entry=sources['entry_source']['path'],entry_args=['--spec',str(paths['SPEC']),'--owner-execute'],allowed_cpus=[0,1],cuda_visible_devices='0',
        qualified_leaf_deadline_seconds=21600,whole_GPU_used_limit_MiB=8192,automatic_retry=False,manifest_builder_SOURCE=sources['manifest_builder_SOURCE'],
        bound_source_sha256={ref['path']:ref['sha256'] for ref in spec['sources']})
    write_fresh(paths['SPEC'],spec); spec_ref = reference(paths['SPEC'])
    admission['scientific_spec'] = spec_ref; admission['bound_source_sha256'][spec_ref['path']] = spec_ref['sha256']
    write_fresh(paths['ADMISSION'],admission)
    actual_source,actual_spec,actual_admission,lineage = writer.source_gate(paths['SPEC'],spec)
    require(actual_source == sources['entry_source'] and actual_spec == spec_ref and actual_admission == reference(paths['ADMISSION']) and
        lineage['qualification'] == q, 'Final metadata-only SOURCE compatibility gate differs')
    print(json.dumps(dict(status='ROOT_JOINT_NATIVE_FIXED_ENDPOINT_EVALUATION_MANIFESTS_BUILT_NOT_EXECUTED',
        endpoint_kind=kind,effective_native_epoch=stage['native_epoch'],effective_B_readout_epoch=stage['head_epoch'],
        spec=spec_ref,admission=actual_admission,CPU_qualification_output=str(paths['CPU_QUALIFICATION']),
        output_directory=str(output),owner_runtime=str(runtime),source_count=len(spec['sources']),controls=spec['controls'],
        science_executed=False,evaluation_launched=False,endpoint_selection=False),sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint-kind',choices=tuple(TRAIN),required=True); parser.add_argument('--artifact-root',required=True)
    parser.add_argument('--owner-author',action='store_true'); args = parser.parse_args()
    require(args.owner_author, 'Inert SOURCE: explicit future ROOT authoring required'); build(args)


if __name__ == '__main__':
    main()

