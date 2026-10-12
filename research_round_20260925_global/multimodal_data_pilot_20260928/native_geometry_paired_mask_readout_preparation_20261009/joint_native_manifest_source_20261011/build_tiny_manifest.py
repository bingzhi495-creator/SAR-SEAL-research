"""Inert stdlib authoring of ROOT's tiny native A/B integration manifests.

Only explicit future --owner-author reads lineage/fixture/FIT metadata. Original
four training raws and the actual restoration raw precede every hash or SOURCE
load. No scientific library, model, GPU, qualification, AUTH or retry is run.
The supplied fixture acceptance is not an independent OS/tool-exit attestation.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import types

PHASE = 'query_pixel_joint_native_preflight_20261011_01'
PREFIX = 'ROOT_QUERY_PIXEL_JOINT_NATIVE_PREFLIGHT'
MODES = ('native0', 'global', 'spatial_halves', 'spatial_reversed')
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
ENTRY_SHA = 'fedd1a1f913019de1c4c7d1960131574f06c650850732f4fb3754e29f0a677a4'
READER_SHA = '5a5d5158cacaa5b7905f960e866c5e9d56eb2bf4ac714da8f5770f91c16754e0'
READOUT_SHA = '5c309b4d1a2318e8c4d673d343dbe956f1387d4b01180de8cd796d0726457305'
RENDERER_SHA = 'dd20f440e11b6890f0f797b8f048f6953adf2a8ee8c33e74cfe91874df78018b'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    require(Path(path).is_absolute(), 'Absolute JSON locator required')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True); digest = hashlib.sha256()
    require(path.is_absolute(), 'Absolute reference required')
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


def load_source(ref, name):
    path = bound(ref); code = path.read_bytes()
    require(path.suffix == '.py' and len(code) == ref['bytes'] and
        hashlib.sha256(code).hexdigest() == ref['sha256'], 'Verified SOURCE bytes changed')
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(code, str(path), 'exec'), module.__dict__)
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
        raw.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation', 'Raw resource scope differs')


def union_refs(rows):
    unique = {}
    for ref in rows:
        require(set(ref) == {'path', 'bytes', 'sha256'} and
            (ref['path'] not in unique or unique[ref['path']] == ref), 'Conflicting closure reference')
        unique[ref['path']] = ref
    return [unique[p] for p in sorted(unique)]


def write_fresh(path, value):
    require(path.is_absolute() and path.parent.is_dir() and not path.exists(), 'Fresh absolute output required')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')


def fixture_gate(ref, roles):
    fixture = read(bound(ref))
    expected = dict(parameter_scalars=9441, parameter_tensors=8, queries_retained=100,
        cold_native_value_and_M_gradient_exact=True, manual_two_stage_renderer_and_repeat_counts=True,
        cold_F_e0_and_upstream_head_gradient_zero=True, warm_three_native_and_all8_head_gradients=True,
        live_contrast_gradient=True, permutations_and_duplicate_points_preserved=True,
        invalid_packets_rejected=True, empty_native_and_joint_connected_zero=True,
        native_inputs_versions_values_unchanged=True, external_CPU_RNG_unchanged=True)
    require(fixture['status'] == 'JOINT_NATIVE_POINT_SYNTHETIC_CPU_PASS' and fixture['sources'] == roles and
        fixture['checks'] == expected and fixture['execution'] == dict(device='cpu', cpu_affinity=[0, 1],
        torch_threads=2, torch_interop_threads=2, cuda_visible_devices='', CUDA_initialized=False) and
        fixture['synthetic_execution_performed'] is True and fixture['optimizer_steps'] == 0 and
        all(fixture[k] is False for k in ('native_model_integration_qualified', 'assignment_generation_qualified',
            'scientific_data_accessed', 'AP_computed', 'efficacy_claimed')), 'Actual synthetic acceptance/interface differs')
    for source in roles.values():
        bound(source)
    return fixture


def build(args):
    c = Path(__file__).resolve(strict=True).parent.parent; m = c.parent
    h = m / 'goal_continuation_20261005'
    paths = {k: h / (PREFIX + '_' + k + '_20261011_01.json') for k in ('SPEC', 'ADMISSION', 'CPU_QUALIFICATION')}
    output = h / 'QUERY_PIXEL_JOINT_NATIVE_PREFLIGHT_01'
    runtime = h / 'finite_point_cache_execution_20261005_v2' / 'runs' / PHASE
    require(h.is_dir() and all(not p.exists() for p in list(paths.values()) + [output, runtime]),
        'Fresh fixed SPEC/admission/CPU-Q/output/owner runtime required; no overwrite or retry')
    template_path = h / 'ROOT_NATIVE_FINETUNE24_FORMAL_SPEC_20261010_01.json'
    template = read(template_path)  # Locator metadata only until every required raw passes.
    require(set(template['training_receiptmap']) == set(template['training_phasemap']) == set(MODES), 'Four original raws required')
    prior = [dict(receipt=template['restoration_receipt'], phase=template['restoration_phase'])]
    for path, size, digest, phase in args.prior_receipt or []:
        prior.append(dict(receipt=dict(path=path, bytes=int(size), sha256=digest), phase=phase))
    raw_rows = [(template['training_receiptmap'][k], template['training_phasemap'][k]) for k in MODES]
    raw_rows += [(r['receipt'], r['phase']) for r in prior]
    require(len({r['path'] for r, phase in raw_rows}) == len(raw_rows), 'Duplicate raw is not independent evidence')
    for ref, phase in raw_rows:
        terminal(read(ref['path']), phase)  # No hash/SOURCE import precedes all raw gates.

    template_ref = reference(template_path); owner = read(template['GPU_admission_file'])
    require(template['kind'] == 'mask2former' and template['teacher_epoch'] == template['expected_epoch'] == 24 and
        template['backbone_with_cp'] is True and owner['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        owner['automatic_retry'] is False and owner['driver_sha256'] == OWNER_SHA and
        owner['bound_source_sha256'].get(str(template_path)) == template_ref['sha256'], 'Canonical native template owner differs')
    for ref in template['sources']:
        bound(ref); require(owner['bound_source_sha256'].get(ref['path']) == ref['sha256'], 'Template closure not owner-bound')
    for ref, phase in raw_rows:
        bound(ref)
    if 'scientific_spec' in owner:
        require(owner['scientific_spec'] == template_ref, 'Template scientific SPEC binding differs')
    d = c / 'query_pixel_joint_native_source_20261011'; tiny = c / 'query_pixel_joint_native_preflight_source_20261011'
    sources = dict(entry_source=reference(tiny / 'run.py'), CPU_reader=reference(tiny / 'qualify_cpu.py'),
        native_loader_source=template['entry_source'], manifest_builder_SOURCE=reference(__file__),
        joint_capture_source=reference(d / 'live_native_capture.py'), joint_losses_source=reference(d / 'joint_native_losses.py'),
        joint_adapter_source=reference(d / 'joint_adapter.py'), native_objective_source=reference(d / 'native_point_objective.py'),
        synthetic_fixture_source=reference(d / 'synthetic_fixture.py'),
        posterior_readout_source=reference(c / 'query_pixel_posterior_coupled_readout_source_20261010' / 'query_pixel_readout.py'),
        renderer_source=reference(c / 'frozen_feature_linear_readout_diagnostic_source_20261010' / 'diagnose.py'),
        reused_CPU_reader=reference(m / 'native_coco_comparator_micro_preflight_preparation_20261008' / 'qualify_cpu.py'))
    require(sources['entry_source']['sha256'] == ENTRY_SHA and sources['CPU_reader']['sha256'] == READER_SHA and
        sources['posterior_readout_source']['sha256'] == READOUT_SHA and sources['renderer_source']['sha256'] == RENDERER_SHA and
        Path(sources['native_loader_source']['path']) == c / 'native_finetune24_fresh_optimizer_source_20261010' / 'train.py',
        'Frozen tiny/original teacher/renderer/readout SOURCE identity differs')
    acceptance = dict(path=args.fixture_acceptance[0], bytes=int(args.fixture_acceptance[1]), sha256=args.fixture_acceptance[2])
    roles = dict(fixture=sources['synthetic_fixture_source'], native_objective=sources['native_objective_source'],
        joint_adapter=sources['joint_adapter_source'], readout=sources['posterior_readout_source'], renderer=sources['renderer_source'])
    fixture_gate(acceptance, roles)
    writer = load_source(sources['entry_source'], '_ROOT_tiny_manifest_provenance_ONLY')
    legacy = load_source(template['evaluation_source'], '_ROOT_original_E24_lineage_ONLY')
    spec = copy.deepcopy(template)
    spec.update(sources, schema=writer.SCHEMA, phase=PHASE, prior_receipts=prior, teacher_epoch=24,
        CPU_affinity=[0, 1], allocator_limit_MiB=4096, whole_GPU_owner_limit_MiB=8192, deadline_seconds=10800,
        CPU_numeric_protocol=writer.NUMERIC, kappa=1., eps=1e-6, auxiliary_weight=1., order_seed=2026101021,
        loss_seed_base=1000000, point_seed_base=3800000, readout_seed=2026100911,
        GPU_admission_file=str(paths['ADMISSION']), output_directory=str(output),
        CPU_qualification_output=str(paths['CPU_QUALIFICATION']), synthetic_CPU_fixture=acceptance,
        synthetic_fixture_scope='ROOT-supplied actual CPU acceptance; file metadata is not OS/tool-exit attestation',
        manifest_builder_scope='STDLIB_AUTHORING_ONLY_NO_SCIENCE_OR_LAUNCH')
    teachers = legacy.qualified_training(spec)  # Existing stdlib lineage only, not model preparation.
    extras = [template_ref, acceptance] + list(sources.values()) + [r['receipt'] for r in prior]
    for row in teachers.values():
        q = row['qualification']
        extras += row['training_spec']['sources'] + [q[k] for k in
            ('source', 'reader_source', 'spec', 'report', 'start', 'GPU_admission', 'owner_driver')]
        extras += list(q['model_proof']['proof_refs'].values())
    spec['sources'] = union_refs(template['sources'] + extras)
    require(all(spec[k] in spec['sources'] for k in writer.MODULES + writer.LIBRARIES +
        ('entry_source', 'CPU_reader', 'training_CPU_reader', 'reused_CPU_reader', 'readout_source', 'geometry_source', 'hooks_source', 'FIT')),
        'Every tiny component/library/data role must be closure-bound')
    fit = read(bound(spec['FIT'])); ids = [row['id'] for row in fit['images']]
    require(len(ids) == len(set(ids)) == 741 and len(fit['annotations']) == 1661, 'Canonical FIT741/1661 metadata differs')
    spec['image_ids'] = ids[:2]
    require(all(type(i) is int and i > 0 and 1 <= sum(a['image_id'] == i for a in fit['annotations']) <= 100
        for i in spec['image_ids']), 'First two canonical FIT images must retain positive all-GT population')
    for ref in spec['sources']:
        bound(ref)
    admission = dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE', phase=PHASE, driver_sha256=OWNER_SHA,
        entry=sources['entry_source']['path'], entry_args=['--spec', str(paths['SPEC']), '--owner-execute'],
        allowed_cpus=[0, 1], cuda_visible_devices='0', qualified_leaf_deadline_seconds=10800,
        whole_GPU_used_limit_MiB=8192, automatic_retry=False,
        bound_source_sha256={r['path']: r['sha256'] for r in spec['sources']})
    driver = h / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py'
    require(reference(driver)['sha256'] == OWNER_SHA, 'Original owner driver SOURCE differs')
    write_fresh(paths['SPEC'], spec); spec_ref = reference(paths['SPEC'])
    admission.update(scientific_spec=spec_ref, manifest_builder_SOURCE=sources['manifest_builder_SOURCE'])
    admission['bound_source_sha256'][str(paths['SPEC'])] = spec_ref['sha256']
    write_fresh(paths['ADMISSION'], admission)
    writer.source_gate(paths['SPEC'], spec)  # Pure provenance compatibility, never run()/model/fixture.
    print(json.dumps(dict(status='ROOT_JOINT_NATIVE_TINY_MANIFESTS_BUILT_NOT_EXECUTED', spec=spec_ref,
        admission=reference(paths['ADMISSION']), CPU_qualification_output=str(paths['CPU_QUALIFICATION']),
        output_directory=str(output), image_ids=spec['image_ids'], source_count=len(spec['sources']), science_executed=False), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-acceptance', nargs=3, required=True, metavar=('ABS_JSON', 'BYTES', 'SHA256'))
    parser.add_argument('--prior-receipt', nargs=4, action='append', metavar=('ABS_JSON', 'BYTES', 'SHA256', 'PHASE'))
    parser.add_argument('--owner-author', action='store_true')
    args = parser.parse_args(); require(args.owner_author, 'Inert SOURCE: explicit future ROOT authoring required')
    build(args)


if __name__ == '__main__':
    main()
