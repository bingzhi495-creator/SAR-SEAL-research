"""Inert stdlib builder for ROOT's fixed final nativeE96 evaluation manifests.

Authoring this SOURCE does not inspect any scientific payload or admit execution.
At explicit future ROOT execution, read the qualified nativeE96 lineage, gate all
eight original owned raw terminals BEFORE any SHA or SOURCE import, and preserve
the complete original closure. Create fresh evaluation SPEC/admission only; no
model/data import, GPU launch, CPU qualification, AUTH edit or retry occurs here.
The CPU session/exit pair is ROOT-supplied terminal metadata, not OS attestation.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import types


PHASE = 'native_finetune48_finalE96_evaluation_20261011_01'
Q_NAME = 'ROOT_NATIVE_FINETUNE48_E48WARM_FORMAL_CPU_QUALIFICATION_20261011_01.json'
SPEC_NAME = 'ROOT_NATIVE_FINETUNE48_FINAL_E96_EVALUATION_SPEC_20261011_01.json'
ADMISSION_NAME = 'ROOT_NATIVE_FINETUNE48_FINAL_E96_EVALUATION_ADMISSION_20261011_01.json'
CPU_NAME = 'ROOT_NATIVE_FINETUNE48_FINAL_E96_EVALUATION_CPU_QUALIFICATION_20261011_01.json'
OUTPUT_NAME = 'NATIVE_FINETUNE48_FINAL_E96_EVALUATION_01'
MODES = ('native0', 'global', 'spatial_halves', 'spatial_reversed')
Q_SCHEMA = 'ROOT-native-finetune48-E48warm-weights-fresh-optimizer-CPU-qualification-v1'
Q_STATUS = 'ROOT_NATIVE_FINETUNE48_E48WARM_WEIGHTS_FRESH_OPTIMIZER_CPU_QUALIFIED'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    path = Path(path)
    require(path.is_absolute(), 'Absolute JSON path required')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path)
    require(path.is_absolute(), 'Absolute reference path required')
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'bytes', 'sha256'} and
        Path(ref['path']).is_absolute() and type(ref['bytes']) is int and ref['bytes'] > 0 and
        isinstance(ref['sha256'], str) and len(ref['sha256']) == 64 and
        set(ref['sha256']) <= set('0123456789abcdef'), 'Exact pinned reference required')
    require(reference(ref['path']) == ref, 'Pinned input differs: ' + ref['path'])
    return Path(ref['path'])


def terminal(raw, phase):
    # Kept identical in meaning to the immutable finalE96 evaluator gate.
    require(raw.get('status') == 'ACTUAL_OWNED_TERMINAL' and raw.get('original_identity') is not None and
        type(raw.get('returncode')) is int and raw['returncode'] == 0 and raw.get('root_wait_reaped') is True and
        raw.get('held_handle_exit') == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        raw.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
        all(k in raw and raw[k] is None for k in ('abort', 'error')) and raw.get('wait_errors') is False and
        raw.get('GPU_observation_errors') == [], 'STOP before hashes/imports: strict original raw terminal0 required')
    peak = raw.get('whole_GPU_observed_peak_MiB')
    require(raw.get('phase', phase) == phase and type(peak) in (int, float) and math.isfinite(peak) and
        0 <= peak <= 8192 and raw.get('whole_GPU_used_limit_MiB') == 8192 and
        raw.get('whole_leaf_VRAM_certified') is False and
        raw.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation', 'Owned raw resource scope differs')


def raw_gate(train, q):
    require(set(train['training_receiptmap']) == set(train['training_phasemap']) == set(MODES),
        'All four original raw identities required')
    require(q['phase'] == train['phase'], 'Formal raw/phase locator differs')
    rows = {m: (train['training_receiptmap'][m], train['training_phasemap'][m]) for m in MODES}
    rows.update(native48_restoration=(train['native48_restoration_receipt'], train['native48_restoration_phase']),
        native48=(train['native48_receipt'], train['native48_phase']),
        restoration=(train['restoration_receipt'], train['restoration_phase']),
        native_phase=(q['parent_receipt'], q['phase']))
    raw = {}
    for label, (ref, phase) in rows.items():
        require(Path(ref['path']).is_absolute(), 'Absolute raw receipt locator required')
        raw[label] = read(ref['path'])
        terminal(raw[label], phase)
    return raw


def load_source(ref):
    path = bound(ref)
    require(path.suffix == '.py', 'Python SOURCE required')
    source = path.read_bytes()
    require(len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'],
        'SOURCE changed before compilation')
    module = types.ModuleType('_ROOT_nativeE96_manifest_provenance_ONLY')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def union_refs(rows):
    unique = {}
    for ref in rows:
        require(isinstance(ref, dict) and set(ref) == {'path', 'bytes', 'sha256'}, 'Pinned closure row required')
        require(ref['path'] not in unique or unique[ref['path']] == ref, 'Conflicting closure pin: ' + ref['path'])
        unique[ref['path']] = ref
    return [unique[path] for path in sorted(unique)]


def write_fresh(path, value):
    require(path.is_absolute() and path.parent.is_dir() and not path.exists(), 'Fresh absolute output required')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def build(artifact_root):
    h = Path(artifact_root)
    require(h.is_absolute() and h.is_dir(), 'Existing absolute artifact root H required')
    h = h.resolve(strict=True)
    spec_path, admission_path, cpu_path = h / SPEC_NAME, h / ADMISSION_NAME, h / CPU_NAME
    output = h / OUTPUT_NAME
    require(all(not p.exists() for p in (spec_path, admission_path, cpu_path, output)),
        'Fresh fixed evaluation SPEC/admission/CPU-Q/output required; no overwrites or retries')

    # Locator JSON reads only: no reference/hash/import until ALL eight raws pass.
    q_path = h / Q_NAME
    q = read(q_path)
    require(q['schema'] == Q_SCHEMA and q['status'] == Q_STATUS and
        q['execution_kind'] == 'formal_additional48' and q['formal_additional48_completed'] is True and
        q['formal_training_completion_CPU_state_qualified'] is True and q['formal_resume_qualified'] is False and
        q['completed_phase_epochs'] == 48 and q['effective_native_epoch'] == 96 and
        q['phase_microsteps'] == 35568 and q['phase_updates'] == 17784 and q['phase_raw_GT_visits'] == 79728,
        'Exact completed nativeE96 qualification required')
    train = read(q['spec']['path'])
    raw = raw_gate(train, q)

    # Everything below is FUTURE ROOT-authorized provenance I/O, never ML execution.
    q_ref = reference(q_path)
    bound(q['spec'])
    owner = read(bound(q['GPU_admission']))
    require(train['schema'] == 'ROOT-native-finetune48-E48warm-weights-fresh-optimizer-spec-v1' and
        train['execution_kind'] == 'formal_additional48' and train['phase'] == q['phase'] and
        train['entry_source'] == q['source'] and train['CPU_reader'] == q['reader_source'] and
        train['SOURCE_proposal'] == q['SOURCE_proposal'], 'Qualified formal train SOURCE/SPEC identity differs')
    require(owner['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        owner['automatic_retry'] is False and owner['phase'] == q['phase'] and
        owner['driver_sha256'] == q['owner_driver']['sha256'] == OWNER_SHA and
        raw['native_phase']['admission_sha256'] == q['GPU_admission']['sha256'] and
        owner['bound_source_sha256'].get(q['spec']['path']) == q['spec']['sha256'],
        'Qualified original owner/raw/SPEC binding differs')
    for ref in train['sources']:
        bound(ref)
        require(owner['bound_source_sha256'].get(ref['path']) == ref['sha256'], 'Original train closure was not owner-bound')
    if 'scientific_spec' in owner:
        require(owner['scientific_spec'] == q['spec'], 'Original owner scientific SPEC differs')
    require(set(q['nodes']) == {'4', '8', '16', '24', '36', '48'} and
        q['checkpoint'] == q['nodes']['48']['checkpoint'], 'Fixed final node48 checkpoint required')

    here = Path(__file__).resolve(strict=True).parent
    proposal_ref = reference(here / 'SOURCE_PROPOSAL.json')
    proposal = read(proposal_ref['path'])
    entry, reader, builder = reference(here / 'evaluate.py'), reference(here / 'qualify_cpu.py'), reference(__file__)
    require(proposal['producer_SOURCE'] == entry and proposal['endpoint'] == 'nativeE96', 'Canonical evaluator SOURCE differs')
    evaluation = load_source(entry)  # Module is stdlib-only; no run()/model/data imports.
    require(proposal['schema'] == evaluation.SCHEMA and proposal['kind'] == evaluation.PROPOSAL_KIND and
        proposal['required_source_roles'] == list(evaluation.REQUIRED_SOURCE_ROLES), 'Canonical SOURCE proposal API differs')

    spec = copy.deepcopy(train)
    spec.update(schema=evaluation.SCHEMA, kind='mask2former', endpoint='nativeE96',
        completed_phase_epoch=48, effective_native_epoch=96, phase=PHASE,
        entry_source=entry, CPU_reader=reader, SOURCE_proposal=proposal_ref,
        GPU_admission_file=str(admission_path), output_directory=str(output),
        CPU_qualification_output=str(cpu_path), CPU_affinity=[0, 1], allocator_limit_MiB=4096,
        whole_GPU_owner_limit_MiB=8192, deadline_seconds=10800,
        DEV=proposal['canonical_DEV'], native_CPU_tool_terminal=dict(session_id=63080, exit_code=0),
        native_phase_source=q['source'], native_phase_CPU_reader=q['reader_source'],
        native_phase_SOURCE_proposal=q['SOURCE_proposal'], native_phase_spec=q['spec'],
        native_phase_qualification=q_ref, native_phase_receipt=q['parent_receipt'],
        native_phase_GPU_admission=q['GPU_admission'], native_phase_checkpoint=q['checkpoint'],
        native_phase_phase=q['phase'], manifest_builder_SOURCE=builder,
        manifest_builder_scope='STDLIB_AUTHORING_ONLY_NO_SCIENCE_OR_LAUNCH')
    canonical = dict(proposal['immutable_SOURCE_refs'], **proposal['library_SOURCE_refs'])
    for role, ref in canonical.items():
        if role in spec:
            require(spec[role] == ref, 'Original/canonical immutable SOURCE pin conflict: ' + role)
        spec[role] = ref
    technical = read(bound(train['restoration_CPU_qualification']))
    extras = [q_ref, proposal_ref, entry, reader, builder, spec['DEV']]
    extras += [q[k] for k in ('source', 'reader_source', 'spec', 'report', 'start', 'parent_receipt',
        'GPU_admission', 'owner_driver', 'SOURCE_proposal', 'checkpoint')]
    extras += list(q['proof_refs'].values()) + [q['nodes']['48']['state_proof']]
    extras += [technical[k] for k in ('source', 'reader_source', 'spec', 'report', 'start',
        'parent_receipt', 'GPU_admission', 'owner_driver')]
    extras += list(canonical.values()) + [spec[k] for k in evaluation.REQUIRED_SOURCE_ROLES]
    spec['sources'] = union_refs(train['sources'] + extras)
    require(set(proposal['required_SPEC_fields']) <= set(spec), 'Required evaluation SPEC field missing')
    require(all(ref in spec['sources'] for ref in train['sources']), 'Original complete sources must be preserved')
    for ref in spec['sources']:
        bound(ref)
    evaluation.qualified_native_phase(spec, raw)  # Existing JSON/SOURCE lineage checks only.

    admission = copy.deepcopy(owner)
    admission.update(phase=PHASE, entry=entry['path'], entry_args=['--spec', str(spec_path), '--owner-execute'],
        allowed_cpus=[0, 1], cuda_visible_devices='0', qualified_leaf_deadline_seconds=10800,
        whole_GPU_used_limit_MiB=8192, automatic_retry=False,
        bound_source_sha256={ref['path']: ref['sha256'] for ref in spec['sources']},
        manifest_builder_SOURCE=builder, qualified_native_phase_CPU_qualification=q_ref)
    runtime = Path(q['owner_driver']['path']).parent / 'runs' / PHASE
    require(runtime == h / 'finite_point_cache_execution_20261005_v2' / 'runs' / PHASE,
        'Owner runtime must equal the immutable CPU reader receipt path derived from H')
    require(not runtime.exists(), 'Fresh owner runtime phase required')
    # Old evaluation failure/supplement narrative is not introduced or reused.
    write_fresh(spec_path, spec)
    spec_ref = reference(spec_path)
    admission['scientific_spec'] = spec_ref
    admission['bound_source_sha256'][str(spec_path)] = spec_ref['sha256']
    write_fresh(admission_path, admission)
    evaluation.source_gate(spec_path, spec)  # Assert exact current closure/owner compatibility.
    print(json.dumps(dict(status='ROOT_NATIVE_E96_EVALUATION_MANIFESTS_BUILT_NOT_EXECUTED',
        spec=spec_ref, admission=reference(admission_path), CPU_qualification_output=str(cpu_path),
        output_directory=str(output), phase=PHASE, prior_CPU_tool_terminal=spec['native_CPU_tool_terminal'],
        native_phase_checkpoint=spec['native_phase_checkpoint'], DEV_GPU_owner_bound=True,
        source_count=len(spec['sources']), science_executed=False), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', required=True, help='Absolute ROOT artifact directory H')
    parser.add_argument('--owner-author', action='store_true')
    args = parser.parse_args()
    require(args.owner_author, 'Inert SOURCE: explicit future ROOT manifest authoring required')
    build(args.artifact_root)


if __name__ == '__main__':
    main()
