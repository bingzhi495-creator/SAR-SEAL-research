"""Inert ROOT-only authoring of own C technical GPU and independent CPU jobs.

No launch, scientific import, model, qualification or formal initializer is run.
All mandatory owned RAWs precede SHA/SOURCE loading and fresh manifest writes.
Mandatory absolute receipt paths carry phase identity; an existing RAW phase
must agree. No receipt is enriched and no A/B technical acceptance is reused.
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
GPU_PHASE = 'query_pixel_PointRend_style_C_technical_20261012_01'
CPU_PHASE = 'query_pixel_PointRend_style_C_technical_CPU_20261012_01'
PREFIX = 'ROOT_QUERY_PIXEL_POINTREND_STYLE_C_TECHNICAL'
PINS = {
 'train.py': (55965, '301cf6a696136ae24541964da084b77bd32e2eba35ebdb2a7355b4bfe75962be'),
 'point_adapter.py': (12105, 'bfb72891cc3245cdada574433905f9963d3a7f83c978e533078b86df2d3eb5a3'),
 'qualify_cpu.py': (60630, '39ae467f68908caca269b0fb23f59efd17c9f5100a65ca33567a9f07231d639f'),
 'evaluate.py': (41816, '1c970d39e18d92e4a3075f08aaca56c3f68854d800991324d75b3a04fd3001f3'),
 'qualify_evaluation_cpu.py': (46946, 'f9d7382721db50f897906b98fd4f938380dcd91d4a1e04ca4cb2ceeeb264ae77')}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    require(Path(path).is_absolute(), 'Absolute JSON locator required')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ref_arg(values):
    require(values is not None, 'Explicit ROOT reference path/bytes/SHA required')
    path, size, sha = values
    require(Path(path).is_absolute() and int(size) > 0 and len(sha) == 64 and
        set(sha) <= set('0123456789abcdef'), 'Malformed ROOT pinned reference')
    return dict(path=str(Path(path)), bytes=int(size), sha256=sha)


def reference(path):
    path = Path(path).resolve(strict=True); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
        Path(ref['path']).is_absolute() and type(ref['bytes']) is int and ref['bytes'] > 0 and
        type(ref['sha256']) is str and len(ref['sha256']) == 64 and
        set(ref['sha256']) <= set('0123456789abcdef') and reference(ref['path']) == ref,
        'Pinned input changed or is malformed')
    return Path(ref['path'])


def raw_gate(ref, phase):
    path = Path(ref['path'])
    require(type(phase) is str and bool(phase) and path.is_absolute() and
        path.name == 'PARENT_RECEIPT.json' and path.parent.name == phase,
        'Absolute phase-bound PARENT_RECEIPT required BEFORE read')
    raw = read(path)
    require(raw.get('status') == 'ACTUAL_OWNED_TERMINAL' and
        type(raw.get('original_identity')) is dict and bool(raw['original_identity']) and
        type(raw.get('returncode')) is int and raw['returncode'] == 0 and raw.get('root_wait_reaped') is True and
        raw.get('held_handle_exit') == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        type(raw['held_handle_exit']['actual_exit_code']) is int and
        raw.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
        all(k in raw and raw[k] is None for k in ('abort', 'error')) and raw.get('wait_errors') is False and
        raw.get('GPU_observation_errors') == [], 'STOP: actual strict owned terminal0 required BEFORE hashes')
    peak = raw.get('whole_GPU_observed_peak_MiB')
    digest = raw.get('admission_sha256')
    require(type(digest) is str and len(digest) == 64 and set(digest) <= set('0123456789abcdef') and
        ('phase' not in raw or (type(raw['phase']) is str and raw['phase'] == phase)) and
        type(peak) in (int, float) and math.isfinite(peak) and
        0 <= peak <= 8192 and raw.get('whole_GPU_used_limit_MiB') == 8192 and
        raw.get('whole_leaf_VRAM_certified') is False and
        raw.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation', 'Actual RAW phase/resource scope differs')
    return raw


def bootstrap(spec, current=None):
    rows = []
    if current is not None:
        rows.append((current, spec['phase']))
    template = read(spec['native_template_spec']['path'])  # RAW-locator metadata only.
    require(template['execution_kind'] == 'formal_additional24' and
        set(template['training_receiptmap']) == set(template['training_phasemap']) == set(MODES),
        'Canonical original native24 template/four RAW identities required')
    rows += [(template['training_receiptmap'][m], template['training_phasemap'][m]) for m in MODES]
    rows += [(template['restoration_receipt'], template['restoration_phase']),
        (spec['point_synthetic_CPU_receipt'], spec['point_synthetic_CPU_phase'])]
    for extra in spec.get('prior_receipts', []):
        require(set(extra) == {'receipt', 'phase'}, 'Explicit additional RAW identity required')
        rows.append((extra['receipt'], extra['phase']))
    require(len({ref['path'] for ref, phase in rows}) == len(rows), 'Duplicate RAW identity is not independent evidence')
    raws = [raw_gate(ref, phase) for ref, phase in rows]
    return template, rows, raws


def union_refs(rows):
    unique = {}
    for ref in rows:
        require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
            (ref['path'] not in unique or unique[ref['path']] == ref), 'Conflicting closure reference')
        unique[ref['path']] = ref
    return [unique[p] for p in sorted(unique)]


def ancestor_records(args, rows, fixture):
    explicit = {}
    for raw_path, admission_path, size, digest in args.ancestor_admission or []:
        require(Path(raw_path).is_absolute() and raw_path not in explicit, 'Unique absolute ancestor RAW locator required')
        explicit[raw_path] = ref_arg((admission_path,size,digest))
    fixture_path = fixture['point_synthetic_CPU_receipt']['path']
    require(set(explicit) == {ref['path'] for ref,phase in rows if ref['path'] != fixture_path},
        'Every original/restoration/additional RAW needs its explicit actual admission reference')
    explicit[fixture_path] = fixture['point_synthetic_CPU_admission']
    return [dict(receipt=ref,phase=phase,admission=explicit[ref['path']]) for ref,phase in rows]


def admission_inputs(records, rows, raws, inherited, fixture):
    require(len(records) == len(rows) and all(set(r) == {'receipt','phase','admission'} for r in records),
        'Exact inherited-RAW/added-actual-admission identities required')
    added = []
    for record, (receipt,phase), raw in zip(records,rows,raws):
        require(record['receipt'] == receipt and record['phase'] == phase, 'Ancestor admission mapping/order differs')
        require(receipt == fixture or receipt in inherited, 'Ancestor RAW hash must retain its qualified inherited pin')
        bound(receipt); admission = record['admission']; actual = read(bound(admission))
        require(raw['admission_sha256'] == admission['sha256'] and
            actual['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and actual['automatic_retry'] is False and
            type(actual['phase']) is str and actual['phase'] == phase and actual['driver_sha256'] == OWNER_SHA and
            actual['whole_GPU_used_limit_MiB'] == 8192 and type(actual['qualified_leaf_deadline_seconds']) is int and
            actual['qualified_leaf_deadline_seconds'] > 0, 'Actual ancestor RAW/admission/phase/driver identity differs')
        entry = reference(actual['entry']); hashes = actual['bound_source_sha256']
        require(entry in inherited and hashes.get(entry['path']) == entry['sha256'] and type(hashes) is dict and
            type(actual['entry_args']) is list and all(type(s) is str for s in actual['entry_args']),
            'Original entry SOURCE must remain qualified and admission-bound')
        for path,digest in hashes.items():
            require(Path(path).is_absolute() and type(digest) is str and len(digest) == 64 and
                set(digest) <= set('0123456789abcdef'), 'Original admission input identity malformed')
            ref = reference(path); require(ref['sha256'] == digest, 'Actual original admission input changed'); added.append(ref)
        if receipt != fixture:
            require(actual['entry_args'][:1] == ['--spec'] and len(actual['entry_args']) >= 2,
                'Explicit original --spec entry input required; no guessed CLI adaptation')
            original_spec = reference(actual['entry_args'][1])
            require(original_spec in inherited and hashes.get(original_spec['path']) == original_spec['sha256'],
                'Original executed SPEC must remain qualified and admission-bound')
            if 'scientific_spec' in actual:
                require(actual['scientific_spec'] == original_spec, 'Original scientific SPEC differs')
        added.append(admission)
    return union_refs(added)


def load_source(ref, name):
    path = bound(ref); code = path.read_bytes()
    require(path.suffix == '.py' and len(code) == ref['bytes'] and hashlib.sha256(code).hexdigest() == ref['sha256'],
        'Verified SOURCE bytes changed')
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(code, str(path), 'exec'), module.__dict__)
    return module


def locations(root):
    require(root.is_absolute() and root.is_dir(), 'Existing absolute artifact root required')
    paths = {k: root / (PREFIX + '_' + k + '_20261012_01.json') for k in
        ('SPEC', 'GPU_ADMISSION', 'CPU_ADMISSION', 'CPU_QUALIFICATION')}
    paths['output'] = root / 'QUERY_PIXEL_POINTREND_STYLE_C_TECHNICAL_20261012_01'
    runs = root / 'finite_point_cache_execution_20261005_v2' / 'runs'
    paths['gpu_runtime'], paths['cpu_runtime'] = runs / GPU_PHASE, runs / CPU_PHASE
    return paths


def fresh(paths):
    require(all(p.is_absolute() and p.parent.is_dir() and not p.exists() for p in paths),
        'Fresh absolute manifests/output/runtime required; no overwrite, retry or launch')


def write_fresh(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')


def owner(phase, entry, argv, deadline, refs, cuda):
    return dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE', phase=phase,
        driver_sha256=OWNER_SHA, entry=entry['path'], entry_args=argv, allowed_cpus=[0,1],
        cuda_visible_devices=cuda, qualified_leaf_deadline_seconds=deadline, whole_GPU_used_limit_MiB=8192,
        automatic_retry=False, bound_source_sha256={r['path']:r['sha256'] for r in refs},
        budget_basis='ROOT_PRESET_TECHNICAL_UPPER_LIMIT_NOT_MEASURED_THROUGHPUT',
        manifest_scope='AUTHORING_ONLY_NO_LAUNCH_NO_NUMERICAL_QUALIFICATION')


def frozen_sources(c):
    d = c / 'query_pixel_PointRend_style_native_integration_source_20261012'
    refs = {name:reference(d/name) for name in PINS}
    require(all((ref['bytes'],ref['sha256']) == PINS[name] for name,ref in refs.items()),
        'Frozen C five-source versions differ; no automatic successor substitution')
    return refs


def build_gpu(args, c, h, paths):
    require(0 < args.deadline_seconds <= 10800, 'Technical GPU preset deadline must be within frozen SOURCE cap10800')
    fresh(list(paths.values()))
    template_ref = ref_arg(args.native_template_spec)
    require(Path(template_ref['path']).parent == h, 'Template/SPEC/owned driver must share canonical artifact root')
    template_locator = read(template_ref['path'])  # Only locators used before all RAWs.
    fixture = dict(point_synthetic_CPU_qualification=ref_arg(args.point_synthetic_qualification),
        point_synthetic_CPU_receipt=ref_arg(args.point_synthetic_CPU_receipt),
        point_synthetic_CPU_phase=args.point_synthetic_CPU_phase,
        point_synthetic_CPU_admission=ref_arg(args.point_synthetic_CPU_admission))
    extra = [dict(receipt=ref_arg(row[:3]),phase=row[3]) for row in (args.prior_receipt or [])]
    provisional = dict(native_template_spec=template_ref, prior_receipts=extra, **fixture)
    template, raw_rows, raws = bootstrap(provisional)
    require(template == template_locator and read(bound(template_ref)) == template, 'Template changed after RAW bootstrap')
    original_owner = read(template['GPU_admission_file'])
    require(original_owner['driver_sha256'] == OWNER_SHA and original_owner['automatic_retry'] is False and
        original_owner['bound_source_sha256'].get(template_ref['path']) == template_ref['sha256'], 'Canonical template owner differs')
    for ref in template['sources']:
        bound(ref); require(original_owner['bound_source_sha256'].get(ref['path']) == ref['sha256'], 'Original closure not owner-bound')
    versions = frozen_sources(c)
    writer = load_source(versions['train.py'], '_ROOT_C_technical_provenance_ONLY')
    head, head_fixture = ref_arg(args.point_head_source), ref_arg(args.point_fixture_source)
    control = c / 'query_pixel_PointRend_style_control_source_20261011'
    require(Path(head['path']) == control/'pointrend_style_point_head.py' and
        Path(head_fixture['path']) == control/'synthetic_fixture.py', 'Explicit original head/fixture SOURCE identities required')
    d = c / 'query_pixel_joint_native_source_20261011'
    roles = dict(entry_source=versions['train.py'],CPU_reader=versions['qualify_cpu.py'],point_adapter_source=versions['point_adapter.py'],
        native_loader_source=template['entry_source'],native_template_spec=template_ref,
        renderer_source=reference(c/'frozen_feature_linear_readout_diagnostic_source_20261010'/'diagnose.py'),
        joint_capture_source=reference(d/'live_native_capture.py'),joint_losses_source=reference(d/'joint_native_losses.py'),
        point_head_source=head,point_fixture_source=head_fixture,manifest_builder_SOURCE=reference(__file__))
    if 'reused_CPU_reader' not in template:
        roles['reused_CPU_reader'] = reference(c.parent/'native_coco_comparator_micro_preflight_preparation_20261008'/'qualify_cpu.py')
    spec = copy.deepcopy(template); stage = writer.STAGES['technical']
    spec.update(roles, **fixture, schema=writer.SCHEMA,phase=GPU_PHASE,stage_kind='technical',
        execution_kind='C_technical_four_microsteps',arms=['C'],
        **{k:stage[k] for k in ('epochs','microsteps','optimizer_updates','raw_GT_visits_per_arm')},
        **writer.SEEDS,accumulation=2,auxiliary_weight=1.,CPU_affinity=[0,1],allocator_limit_MiB=4096,whole_GPU_owner_limit_MiB=8192,
        deadline_seconds=args.deadline_seconds,CPU_numeric_protocol=writer.NUMERIC,C_readout_optimizer=writer.HEAD_RECIPE,
        C_readout_LR_schedule=False,schedule=dict(gamma=.1,milestones=[7904,8563],warmup_start_factor=1,warmup_updates=0),
        no_feature_cache=True,no_DEV_selection=True,direct_point_logits=True,native_L_added_to_point_logits=False,
        prior_receipts=extra,GPU_admission_file=str(paths['GPU_ADMISSION']),output_directory=str(paths['output']),
        CPU_qualification_output=str(paths['CPU_QUALIFICATION']),technical_weights_must_not_initialize_formal=True,
        technical_budget_basis='PRESET_UPPER_LIMIT_UNMEASURED_NO_RESOURCE_FIT_CLAIM')
    inherited = list(template['sources']) + [template_ref]
    legacy = load_source(spec['evaluation_source'], '_ROOT_original_teacher_lineage_ONLY')
    for row in legacy.qualified_training(spec).values():
        q = row['qualification']
        inherited += row['training_spec']['sources'] + [q[k] for k in ('source','reader_source','spec','report','start','GPU_admission','owner_driver')]
        inherited += list(q['model_proof']['proof_refs'].values())
    # Preserve inherited pins; add actual historical admissions rather than inventing old roles.
    original_closure = union_refs(inherited)
    records = ancestor_records(args,raw_rows,fixture)
    fixture_inputs = [head,head_fixture,fixture['point_synthetic_CPU_qualification']]
    added_admissions = admission_inputs(records,raw_rows,raws,original_closure+fixture_inputs,fixture['point_synthetic_CPU_receipt'])
    spec['ancestor_admissions'] = records
    spec['sources'] = union_refs(original_closure + list(roles.values()) + list(versions.values()) + fixture_inputs +
        added_admissions + [ref for ref,phase in raw_rows])
    for ref in spec['sources']: bound(ref)
    for role in writer.MODULES + writer.LIBRARIES + ('entry_source','CPU_reader','native_template_spec','training_CPU_reader','reused_CPU_reader',
        'readout_source','geometry_source','hooks_source','FIT','point_fixture_source','point_synthetic_CPU_qualification',
        'point_synthetic_CPU_receipt','point_synthetic_CPU_admission'):
        require(spec[role] in spec['sources'], 'Required C role not closure-bound: '+role)
    driver = reference(h/'finite_point_cache_execution_20261005_v2'/'run_owned_gpu_limited.py')
    require(driver['sha256'] == OWNER_SHA, 'Frozen owned driver SOURCE differs')
    admission = owner(GPU_PHASE, spec['entry_source'], ['--spec',str(paths['SPEC']),'--owner-execute'],
        args.deadline_seconds, spec['sources'], '0')
    write_fresh(paths['SPEC'],spec); spec_ref = reference(paths['SPEC'])
    admission.update(scientific_spec=spec_ref,manifest_builder_SOURCE=roles['manifest_builder_SOURCE'])
    admission['bound_source_sha256'][spec_ref['path']] = spec_ref['sha256']
    write_fresh(paths['GPU_ADMISSION'],admission)
    writer.source_gate(paths['SPEC'],spec)  # Verified pure provenance only; never run/model/fixture.
    return dict(spec=spec_ref,admission=reference(paths['GPU_ADMISSION']),output_directory=str(paths['output']),
        future_CPU_qualification=str(paths['CPU_QUALIFICATION']),phase=GPU_PHASE)


def build_cpu(args, c, h, paths):
    fresh([paths['CPU_ADMISSION'],paths['CPU_QUALIFICATION'],paths['cpu_runtime']])
    spec_ref = ref_arg(args.technical_spec)
    require(Path(spec_ref['path']) == paths['SPEC'], 'Own fixed C technical SPEC required, not A/B or another stage')
    spec = read(spec_ref['path'])  # Only RAW locators used before every required gate.
    require(spec['stage_kind'] == 'technical' and spec['phase'] == GPU_PHASE, 'Own C technical stage identity required')
    current = dict(path=str(paths['gpu_runtime']/'PARENT_RECEIPT.json'))
    template, raw_rows, raws = bootstrap(spec,current)
    require(read(bound(spec_ref)) == spec, 'C technical SPEC changed after RAW bootstrap')
    for ref, phase in raw_rows[1:]: bound(ref)
    versions = frozen_sources(c)
    require(spec['entry_source'] == versions['train.py'] and spec['CPU_reader'] == versions['qualify_cpu.py'], 'Actual C SOURCE versions differ')
    reader = load_source(versions['qualify_cpu.py'], '_ROOT_C_CPU_manifest_provenance_ONLY')
    producer, refs, context = reader.source_gate(paths['SPEC'],spec)
    require(refs['spec'] == spec_ref and context['stage'] == producer.STAGES['technical'], 'Actual C GPU owner/closure differs')
    ancestry_inputs = admission_inputs(spec['ancestor_admissions'],raw_rows[1:],raws[1:],spec['sources'],spec['point_synthetic_CPU_receipt'])
    require(all(ref in spec['sources'] for ref in ancestry_inputs), 'Current C SPEC must bind added actual original-admission inputs')
    report_ref = reference(paths['output']/'REPORT.json'); report = read(bound(report_ref))
    start_ref = reference(paths['output']/'START.json'); start = read(bound(start_ref))
    aggregate_ref = reference(paths['output']/'QUALIFIED.json'); aggregate = read(bound(aggregate_ref))
    require(report['schema'] == producer.PREFIX+'-report-v1' and report['status'] == 'POINTREND_STYLE_NATIVE_INTEGRATION_GPU_COMPLETE_PENDING_CPU' and
        report['stage_kind'] == 'technical' and report['phase'] == GPU_PHASE and report['error'] is None and set(report['result']) == {'C'} and
        report['source'] == refs['source'] and report['spec'] == spec_ref and report['GPU_admission'] == refs['GPU_admission'] and
        report['counters_per_arm'] == dict(epochs=0,microsteps=4,optimizer_updates=2,raw_GT_visits_per_arm=None) and
        report['formal_initializer_qualified'] is False and report['technical_weights_loaded'] is False and report['cleanup']['allocated_after'] == 0,
        'Actual own C technical REPORT identity/completion differs')
    require(start['schema'] == producer.PREFIX+'-start-v1' and start['source'] == refs['source'] and start['spec'] == spec_ref and
        start['GPU_admission'] == refs['GPU_admission'] and start['phase'] == GPU_PHASE and start['stage_kind'] == 'technical' and
        aggregate['schema'] == producer.PREFIX+'-aggregate-v1' and aggregate['status'] == 'ROOT_POINTREND_STYLE_NATIVE_INTEGRATION_PENDING_CPU_READER' and
        aggregate['report'] == report_ref and aggregate['result'] == report['result'] and aggregate['independently_CPU_qualified'] is False,
        'Own C START/aggregate interface differs')
    row = report['result']['C']
    require(row['phase_microsteps'] == 4 and row['phase_updates'] == 2 and row['readout_parameter_tensors'] == 8 and
        row['readout_parameters'] == 198402 and len(row['audits']) == 4 and row['cleanup']['allocated_after'] == 0,
        'Actual technical4/2/C8/four audits required')
    inputs = union_refs(spec['sources'] + [refs[k] for k in ('spec','parent_receipt','GPU_admission')] +
        [report_ref,start_ref,aggregate_ref,reference(__file__)] + [row[k] for k in ('checkpoint','endpoint_proof','trace','order','epochs','start')] + row['audits'])
    for ref in inputs: bound(ref)
    require(reference(h/'finite_point_cache_execution_20261005_v2'/'run_owned_gpu_limited.py')['sha256'] == OWNER_SHA, 'Frozen driver differs')
    argv = ['--spec',str(paths['SPEC']),'--acceptance',str(paths['CPU_QUALIFICATION']),'--CPU-admission',str(paths['CPU_ADMISSION']),'--owner-execute']
    admission = owner(CPU_PHASE,versions['qualify_cpu.py'],argv,args.deadline_seconds,inputs,'')
    admission.update(scientific_spec=spec_ref,manifest_builder_SOURCE=reference(__file__),GPU_parent_receipt=refs['parent_receipt'],
        CPU_scope='CPU01/2threads/2interop/CUDA empty; RAW GPU fields observe whole card, not CPU CUDA allocation')
    write_fresh(paths['CPU_ADMISSION'],admission)
    reader.CPU_owner_gate(paths['SPEC'],paths['CPU_QUALIFICATION'],paths['CPU_ADMISSION'],spec,refs)
    return dict(spec=spec_ref,admission=reference(paths['CPU_ADMISSION']),qualification_output=str(paths['CPU_QUALIFICATION']),phase=CPU_PHASE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',choices=('gpu','cpu'),required=True); parser.add_argument('--artifact-root',required=True)
    parser.add_argument('--deadline-seconds',type=int,required=True); parser.add_argument('--owner-author',action='store_true')
    for name in ('native-template-spec','point-head-source','point-fixture-source','point-synthetic-qualification',
        'point-synthetic-CPU-receipt','point-synthetic-CPU-admission','technical-spec'):
        parser.add_argument('--'+name,nargs=3,metavar=('ABS_PATH','BYTES','SHA256'))
    parser.add_argument('--point-synthetic-CPU-phase'); parser.add_argument('--prior-receipt',nargs=4,action='append')
    parser.add_argument('--ancestor-admission',nargs=4,action='append',metavar=('RAW_ABS','ADMISSION_ABS','BYTES','SHA256'))
    args = parser.parse_args(); require(args.owner_author, 'Inert SOURCE: explicit future ROOT --owner-author required')
    require(args.deadline_seconds > 0, 'Explicit preset positive upper deadline, not measured throughput, required')
    c = Path(__file__).resolve(strict=True).parent.parent; h = Path(args.artifact_root)
    require(h == c.parent/'goal_continuation_20261005', 'Canonical artifact root/owner runtime required')
    paths = locations(h)
    result = build_gpu(args,c,h,paths) if args.job == 'gpu' else build_cpu(args,c,h,paths)
    print(json.dumps(dict(status='ROOT_C_TECHNICAL_MANIFEST_BUILT_NOT_LAUNCHED_NOT_QUALIFIED',job=args.job,
        science_executed=False,qualification_generated=False,formal_initializer_qualified=False,**result),sort_keys=True))


if __name__ == '__main__':
    main()
