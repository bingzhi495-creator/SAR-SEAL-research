"""Unbound SOURCE: ROOT-owned independent N10 fixed-eight training endpoint.

Thirty normal original RAWs and the three preserved failed1 original RAWs must
be terminal before any input hash, SOURCE import or numerical operation. The
four saved C/PR states, optimizer endpoints, private image orders and cold
factory proofs are checked. Only the final full1661-instance cached-point BCE
is independently recomputed. Intermediate epoch losses and gradient activity
are bound records; no complete gradient/optimizer trajectory is replayed.

Different C/PR architectures share seeds, image orders and cold native-logit
degeneration, not identical weights. No teacher, native model, JPEG/GT decoder,
optimizer step, DEV/TEST or AP evaluation is executed by this reader. Passing
qualifies this limited endpoint evidence, not efficacy or scientific novelty.
ROOT separately qualifies this reader's actual owned terminal after exit.
"""
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types


SCHEMA = 'ROOT-N10-C-PR-FL91-fixed8-training-CPU-spec-v1'
Q_SCHEMA = 'ROOT-N10-C-PR-FL91-fixed8-training-CPU-qualification-v1'
STATUS = 'ROOT_N10_C_PR_FL91_FIXED8_TRAINING_CPU_QUALIFIED'
TRAIN_SCHEMA = 'ROOT-N10-C-PR-FL91-point-training-spec-v1'
TRAIN_REPORT_SCHEMA = 'N10-C-PR-FL91-point-training-report-v1'
TRAIN_STATUS = 'N10_C_PR_FL91_POINT_TRAINING_COMPLETE_PENDING_ROOT_QUALIFICATION'
CP_SCHEMA = 'N10-C-PR-FL91-point-head-checkpoint-v1'
START_SCHEMA = 'N10-C-PR-FL91-point-training-start-v1'
HEAD_SPEC_SCHEMA = 'ROOT-N10-C-PR-FL91-synthetic-head-CPU-spec-v1'
HEAD_Q_SCHEMA = 'ROOT-N10-C-PR-FL91-synthetic-head-CPU-qualification-v1'
HEAD_Q_STATUS = 'ROOT_N10_C_PR_FL91_SYNTHETIC_HEAD_CPU_QUALIFIED'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
SEEDS = (202610120701, 202610120702)
ARMS = ('C', 'PR')
KEYS = tuple('%s_seed%d' % (arm, seed) for seed in SEEDS for arm in ARMS)
NAMES = dict(C=('hidden1.weight', 'hidden1.bias', 'hidden2.weight', 'hidden2.bias',
                'final.weight', 'final.bias'),
    PR=('hidden1.weight', 'hidden1.bias', 'hidden2.weight', 'hidden2.bias',
        'hidden3.weight', 'hidden3.bias', 'final.weight', 'final.bias'))
SHAPES = dict(C=((64,562), (64,), (64,64), (64,), (1,64), (1,)),
    PR=((91,257), (91,), (91,92), (91,), (91,92), (91,), (1,92), (1,)))
PARAMETERS = dict(C=40257, PR=40497)
FACTORY_SPEC_KEYS = dict(C='point_heads_source', PR='PR_point_heads_source')
RAW_KEYS = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
REF_KEYS = ('entry_source', 'training_entry_source', 'training_spec', 'training_report',
    'training_GPU_RAW', 'training_GPU_admission', 'point_heads_source',
    'PR_point_heads_source', 'head_CPU_qualification')
REQUIRED_KEYS = ('schema', 'phase', 'sources', 'RAW_inputs', 'prior_failed_CPU_receipt',
    'additional_failed_RAW_inputs', 'CPU_admission_file', 'Q_path', 'deadline_seconds',
    'recipe', 'numeric_protocol') + REF_KEYS
NUMERIC = dict(endpoint_BCE_rtol=1e-4, endpoint_BCE_atol=1e-6,
    instance_BCE_arithmetic_rtol=1e-4, instance_BCE_arithmetic_atol=1e-6)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
        type(ref['path']) is str and Path(ref['path']).is_absolute() and
        type(ref['bytes']) is int and ref['bytes'] >= 0 and
        type(ref['sha256']) is str and len(ref['sha256']) == 64 and
        set(ref['sha256']) <= set('0123456789abcdef'), 'Malformed absolute byte/SHA reference')


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    ref_shape(ref)
    require(reference(ref['path']) == ref, 'Actual bound input changed: ' + ref['path'])
    return Path(ref['path'])


def deadline(started, seconds):
    require(time.monotonic() - started < seconds,
        'Prelocked fixed8 CPU endpoint deadline exceeded; no retry or fallback')


def terminal(ref, phase, code=0):
    """Original RAW JSON only, before every hash/import."""
    ref_shape(ref)
    path = Path(ref['path'])
    require(path.name == 'PARENT_RECEIPT.json' and type(phase) is str and phase and
        path.parent.name == phase, 'Original absolute RAW/phase locator required')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and
        raw['returncode'] == code and type(raw['original_identity']) is dict and raw['original_identity'] and
        raw['root_wait_reaped'] is True and
        raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=code) and
        type(raw['held_handle_exit']['actual_exit_code']) is int and
        raw['numeric_birth_absence']['original_birth_absent'] is True and
        raw.get('error') is None and raw.get('abort') is None and
        raw['wait_errors'] is False and raw['GPU_observation_errors'] == [],
        'All30 normal and three preserved failed1 terminals required before any SHA/import')
    if 'phase' in raw:
        require(type(raw['phase']) is str and raw['phase'] == phase, 'Original RAW phase differs')
    peak, sha = raw['whole_GPU_observed_peak_MiB'], raw['admission_sha256']
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef') and
        type(peak) in (int,float) and math.isfinite(peak) and 0 <= peak <= 8192 and
        type(raw['whole_GPU_used_limit_MiB']) is int and raw['whole_GPU_used_limit_MiB'] == 8192 and
        raw['whole_leaf_VRAM_certified'] is False and
        raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
        'Original owner/resource observation differs')
    return raw


def load(ref, name):
    path = bound(ref)
    source = path.read_bytes()
    require(path.suffix == '.py' and len(source) == ref['bytes'] and
        hashlib.sha256(source).hexdigest() == ref['sha256'], 'Bound Python SOURCE bytes changed')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def cache_lineage(train, training, closure):
    """CPU metadata counterpart, without calling GPU-only source_gate."""
    keys = ('cache_entry_source', 'cache_spec', 'cache_report', 'cache_CPU_qualification',
        'cache_CPU_reader_source', 'cache_GPU_receipt', 'cache_GPU_admission',
        'cache_CPU_receipt', 'cache_CPU_admission', 'point_heads_source',
        'PR_point_heads_source', 'head_CPU_qualification', 'head_CPU_spec',
        'head_CPU_entry_source', 'head_CPU_receipt', 'head_CPU_admission')
    require(all(training[key] in closure.values() for key in keys),
        'Actual original cache and C/PR head primary lineage absent')
    cache = load(training['cache_entry_source'], '_N10_CPU_original_cache_provenance')
    cached = read(bound(training['cache_spec']))
    report = read(bound(training['cache_report']))
    q = read(bound(training['cache_CPU_qualification']))
    require(cache.OWNER_SHA == OWNER_SHA and cached['schema'] == cache.SCHEMA and
        cached['entry_source'] == training['cache_entry_source'] and
        cached['point_heads_source'] == training['point_heads_source'] and
        cached['recipe'] == cache.RECIPE and cached['RAW_inputs'] == training['RAW_inputs'][:17] and
        all(ref in closure.values() for ref in cached['sources']),
        'Exact original qualified cache SOURCE/recipe/closure differs')
    require(q['schema'] == train.CACHE_Q_SCHEMA and q['status'] == train.CACHE_Q_STATUS and
        q['source'] == training['cache_entry_source'] and q['reader_source'] == training['cache_CPU_reader_source'] and
        q['cache_spec'] == training['cache_spec'] and q['cache_report'] == training['cache_report'] and
        q['cache_GPU_admission'] == training['cache_GPU_admission'] and q['cache_GPU_RAW'] == training['cache_GPU_receipt'] and
        q['CPU_admission'] == training['cache_CPU_admission'] and q['point_heads_source'] == training['point_heads_source'] and
        q['phase'] == training['cache_CPU_phase'] and q['cache_phase'] == training['cache_GPU_phase'] and
        q['recipe'] == cache.RECIPE and q['FIT'] == cached['FIT'] and q['image_root'] == cached['image_root'] and
        q['images'] == 741 and q['raw_GT'] == 1661 and q['point_count'] == 1700864 and
        q['hard_targets_CPU_checked'] is True and q['indices_JPEG_patches_CPU_checked'] is True and
        q['point_head_synthetic_CPU_qualified'] is True,
        'Actual original independent CPU cache/head qualification differs')
    require(q['spec'] in closure.values(), 'Actual original cache CPU SPEC absent')
    cpu = read(bound(q['spec']))
    require(cpu['RAW_inputs'] == training['RAW_inputs'][:18] and
        all(ref in closure.values() for ref in cpu['sources']), 'Complete original cache CPU closure differs')
    require(report['schema'] == 'N07-frozen-E24-point-cache-report-v1' and report['status'] == cache.STATUS and
        report['source'] == training['cache_entry_source'] and report['spec'] == training['cache_spec'] and
        report['GPU_admission'] == training['cache_GPU_admission'] and report['phase'] == training['cache_GPU_phase'] and
        report['packet_refs'] == q['packet_refs'] and len(report['packet_refs']) == len(report['rows']) == 741 and
        report['images'] == 741 and report['raw_GT'] == 1661 and report['point_count'] == 1700864 and
        report['native610_unchanged_SOURCE_observed'] is True and report['all_raw_GT_retained'] is True and
        report['optimizer_created'] is False and report['weights_updated'] is False and
        report['cleanup']['allocated_after'] == 0, 'Actual immutable frozen cache report differs')
    require(all(ref in closure.values() for ref in report['packet_refs']) and
        report['start'] in closure.values() and report['trace'] in closure.values() and
        cached['context_GPU_spec'] in closure.values(), 'All741 packet/START/TRACE/context inputs absent')
    context = read(bound(cached['context_GPU_spec']))
    require(all(ref in closure.values() for ref in context['sources']), 'Original E24 closure absent')
    return cache, cached, report, q, context


def head_lineage(training, spec, closure, rows, owners):
    hq = read(bound(training['head_CPU_qualification']))
    hs = read(bound(training['head_CPU_spec']))
    require(hq['schema'] == HEAD_Q_SCHEMA and hq['status'] == HEAD_Q_STATUS and
        hq['source'] == training['head_CPU_entry_source'] and hq['spec'] == training['head_CPU_spec'] and
        hq['CPU_admission'] == training['head_CPU_admission'] and
        hq['original_point_heads_source'] == spec['point_heads_source'] and
        hq['PR_point_heads_source'] == spec['PR_point_heads_source'] and
        hq['C_parameters'] == 40257 and hq['PR_parameters'] == 40497 and
        hq['synthetic_cold_identity_and_band_checked'] is True and
        hq['synthetic_backward_update_checked'] is True and hq['synthetic_CPU_only'] is True and
        hq['no_task_efficacy_claim'] is True and hq['CPU_affinity'] == [0,1] and
        hq['threads'] == 2 and hq['CUDA_initialized'] is False and
        hq['RAW_inputs'] == rows[:28] and hq['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'] and
        hq['additional_failed_RAW_inputs'] == spec['additional_failed_RAW_inputs'],
        'Actual independent synthetic C/PR CPU qualification differs')
    require(hs['schema'] == HEAD_SPEC_SCHEMA and hs['entry_source'] == training['head_CPU_entry_source'] and
        hs['original_point_heads_source'] == spec['point_heads_source'] and
        hs['PR_point_heads_source'] == spec['PR_point_heads_source'] and hs['RAW_inputs'] == rows[:28] and
        hs['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'] and
        hs['additional_failed_RAW_inputs'] == spec['additional_failed_RAW_inputs'] and
        all(ref in closure.values() for ref in hs['sources']), 'Complete synthetic head CPU input closure differs')
    row, owner = rows[28], owners[28]
    require(row['receipt'] == training['head_CPU_receipt'] and row['phase'] == training['head_CPU_phase'] and
        row['admission'] == training['head_CPU_admission'] and owner['cuda_visible_devices'] == '' and
        owner['entry'] == training['head_CPU_entry_source']['path'] and
        owner['entry_args'] == ['--spec',training['head_CPU_spec']['path'],'--owner-execute'] and
        owner['scientific_spec'] == training['head_CPU_spec'] and
        hs['phase'] == hq['phase'] == training['head_CPU_phase'],
        'Original held-terminal0 head CPU owner/provenance differs')


def source_gate(path, spec, started):
    require(all(key in spec for key in REQUIRED_KEYS) and spec['schema'] == SCHEMA and
        type(spec['RAW_inputs']) is list and len(spec['RAW_inputs']) == 30 and
        all(type(row) is dict and set(RAW_KEYS) <= set(row) for row in spec['RAW_inputs']),
        'Exactly30 normal ancestors, including the actual N10 training terminal, required')
    rows = spec['RAW_inputs']
    for row in rows:
        ref_shape(row['receipt'])
    paths = {row['receipt']['path'] for row in rows}
    require(len(paths) == 30, 'Thirty distinct successful original RAWs required')
    raws = [terminal(row['receipt'],row['phase']) for row in rows]
    failed_refs = [spec['prior_failed_CPU_receipt']]
    ref_shape(failed_refs[0])
    terminal(failed_refs[0], Path(failed_refs[0]['path']).parent.name, 1)
    extra = spec['additional_failed_RAW_inputs']
    require(type(extra) is list and len(extra) == 2, 'Two additional historical failed1 rows required')
    for row in extra:
        require(type(row) is dict and set(row) == {'receipt','phase','returncode'} and
            type(row['returncode']) is int and row['returncode'] == 1, 'Explicit preserved failed1 row required')
        terminal(row['receipt'],row['phase'],1)
        failed_refs.append(row['receipt'])
    require(len({ref['path'] for ref in failed_refs}) == 3 and
        not paths.intersection(ref['path'] for ref in failed_refs), 'Failed1 cannot qualify a normal ancestor')
    # All30 successful and three preserved failed1 terminals precede SHA/import.
    require(type(spec['sources']) is list and spec['sources'], 'Complete CPU input closure required')
    for ref in spec['sources']:
        ref_shape(ref)
    closure = {ref['path']:ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique complete CPU closure required')
    required = [spec[key] for key in REF_KEYS] + failed_refs + [row[key]
        for row in rows for key in RAW_KEYS if key != 'phase']
    require(all(ref in closure.values() for ref in required), 'All actual CPU primary inputs must be bound')
    require(type(spec['deadline_seconds']) is int and spec['deadline_seconds'] > 0 and
        os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
        all(os.environ.get(key) == '2' for key in THREAD_ENV), 'Prelocked CPU deadline/environment differs')
    for ref in spec['sources']:
        deadline(started,spec['deadline_seconds'])
        bound(ref)
    train = load(spec['training_entry_source'], '_N10_CPU_training_endpoint_provenance')
    training = read(bound(spec['training_spec']))
    train_source, train_spec, train_admission = (spec[key] for key in
        ('training_entry_source','training_spec','training_GPU_admission'))
    require(train.SCHEMA == TRAIN_SCHEMA and train.STATUS == TRAIN_STATUS and train.CP_SCHEMA == CP_SCHEMA and
        train.ARMS == ARMS and train.SEEDS == SEEDS and train.NAMES == NAMES and train.SHAPES == SHAPES and
        train.PARAMETERS == PARAMETERS and train.FACTORY_SPEC_KEYS == FACTORY_SPEC_KEYS and
        training['schema'] == TRAIN_SCHEMA and training['entry_source'] == train_source and
        training['recipe'] == train.RECIPE and training['RAW_inputs'] == rows[:29] and
        training['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'] and
        training['additional_failed_RAW_inputs'] == extra and
        all(ref in closure.values() for ref in training['sources']) and
        training['point_heads_source'] == spec['point_heads_source'] and
        training['PR_point_heads_source'] == spec['PR_point_heads_source'] and
        training['head_CPU_qualification'] == spec['head_CPU_qualification'],
        'Complete actual N10 fixed8 training/source/head closure differs')
    cache,cached,cache_report,cache_q,context = cache_lineage(train,training,closure)
    owners = [cache.owner_join(row,raw,closure) for row,raw in zip(rows,raws)]
    for index,key,phase_key,admission_key,entry,args in (
        (17,'cache_GPU_receipt','cache_GPU_phase','cache_GPU_admission',training['cache_entry_source']['path'],
         ['--spec',training['cache_spec']['path'],'--owner-execute']),
        (18,'cache_CPU_receipt','cache_CPU_phase','cache_CPU_admission',training['cache_CPU_reader_source']['path'],
         ['--spec',cache_q['spec']['path'],'--owner-execute'])):
        row,owner = rows[index],owners[index]
        require(row['receipt'] == training[key] and row['phase'] == training[phase_key] and
            row['admission'] == training[admission_key] and owner['entry'] == entry and owner['entry_args'] == args,
            'Original cache GPU/CPU owned lineage differs')
    head_lineage(training,spec,closure,rows,owners)
    row,owner = rows[-1],owners[-1]
    require(row['receipt'] == spec['training_GPU_RAW'] and row['phase'] == training['phase'] and
        row['admission'] == train_admission and owner['entry'] == train_source['path'] and
        owner['entry_args'] == ['--spec',train_spec['path'],'--owner-execute'] and
        owner['scientific_spec'] == train_spec and owner['cuda_visible_devices'] == '0' and
        owner['qualified_leaf_deadline_seconds'] == training['deadline_seconds'] and
        all(owner['bound_source_sha256'].get(ref['path']) == ref['sha256'] for ref in training['sources']) and
        training['allocator_limit_MiB'] == 4096 and training['whole_GPU_limit_MiB'] == 8192,
        'Actual N10 training held0 owner and complete original training closure differs')
    source,spec_ref = reference(__file__),reference(path)
    require(source == spec['entry_source'], 'Actual CPU SOURCE differs')
    admission_path = Path(spec['CPU_admission_file'])
    require(admission_path.is_absolute(), 'Absolute current CPU admission required')
    admit_ref = reference(admission_path)
    admission = read(admission_path)
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        admission['driver_sha256'] == OWNER_SHA and type(spec['phase']) is str and spec['phase'] and
        admission['phase'] == spec['phase'] and admission['scientific_spec'] == spec_ref and
        admission['entry'] == source['path'] and admission['entry_args'] == ['--spec',str(path),'--owner-execute'] and
        admission['allowed_cpus'] == [0,1] and admission['cuda_visible_devices'] == '' and
        admission['automatic_retry'] is False and admission['whole_GPU_used_limit_MiB'] == 8192 and
        admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
        admission['bound_source_sha256'].get(str(path)) == spec_ref['sha256'] and
        all(admission['bound_source_sha256'].get(ref['path']) == ref['sha256'] for ref in spec['sources']),
        'Actual CPU owner/complete own closure differs')
    report = read(bound(spec['training_report']))
    require(report['schema'] == TRAIN_REPORT_SCHEMA and report['status'] == TRAIN_STATUS and
        report['source'] == train_source and report['spec'] == train_spec and
        report['GPU_admission'] == train_admission and report['phase'] == training['phase'] and
        report['recipe'] == train.RECIPE and report['cache_spec'] == training['cache_spec'] and
        report['cache_report'] == training['cache_report'] and
        report['cache_CPU_qualification'] == training['cache_CPU_qualification'] and
        report['teacher_checkpoint'] == context['checkpointmap']['native0'] and
        report['point_heads_source'] == spec['point_heads_source'] and
        report['PR_point_heads_source'] == spec['PR_point_heads_source'] and
        report['head_CPU_qualification'] == spec['head_CPU_qualification'] and
        report['native_model_loaded'] is False and report['native_forward_calls'] == 0 and
        report['native_weights_updated'] is False and report['all4_fixed_same_epochs'] is True and
        report['paired_order_and_initialization'] is False and report['paired_order'] is True and
        report['paired_seed'] is True and report['paired_cold_native_identity'] is True and
        report['no_DEV'] is True and report['no_TEST'] is True and report['AP_computed'] is False and
        report['cleanup']['allocated_after'] == 0, 'Complete actual N10 training report differs')
    require(type(report['completed_head_epochs']) is int and report['completed_head_epochs'] == 8 and
        report['continued_all4_to24'] is False and
        set(report['checkpoints']) == set(report['epoch8_checkpoints']) == set(report['endpoint_BCE_histories']) == set(KEYS) and
        report['checkpoints'] == report['epoch8_checkpoints'], 'Only fixed8 final checkpoint population is allowed')
    fresh = report['fresh_optimizer_state_entries_per_model']
    require(set(fresh) == set(KEYS) and all(type(value) is int and value == 0 for value in fresh.values()),
        'Actual four fresh AdamW state populations differ')
    outputs = [report[key] for key in ('start','trace','decision')]
    outputs += [row[key] for row in report['checkpoints'].values() for key in ('checkpoint','proof')]
    require(all(ref in closure.values() for ref in outputs),
        'All four checkpoints/proofs and START/TRACE/fixed decision must be CPU inputs')
    require(spec['recipe'] == train.RECIPE and train.RECIPE['fixed_epochs'] == 8 and
        train.RECIPE['paired_initialization'] is False and train.RECIPE['paired_cold_native_identity'] is True and
        spec['numeric_protocol'] == NUMERIC, 'Prelocked fixed8 CPU recipe/math differs')
    target = Path(spec['Q_path'])
    require(target.is_absolute() and target.parent.is_dir() and not target.exists() and
        not target.with_suffix(target.suffix+'.tmp').exists(), 'Fresh absolute Q path required')
    return dict(train=train,cache=cache,cached=cached,cache_report=cache_report,cache_q=cache_q,
        context=context,training=training,report=report,closure=closure,source=source,
        spec_ref=spec_ref,admit_ref=admit_ref,admission=admission)


def digest(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest())


def resources(psutil, torch):
    require(psutil.Process().cpu_affinity() == [0,1] and
        torch.get_num_threads() == torch.get_num_interop_threads() == 2 and
        os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
        all(os.environ.get(key) == '2' for key in THREAD_ENV) and
        not torch.cuda.is_initialized(), 'Actual CPU01/threads2/cold CUDA differs')


def imported_files(psutil, torch, np, gate):
    modules = dict(psutil=psutil,psutil_platform=psutil._psplatform,
        psutil_binary=psutil._psplatform.cext,torch=torch,torch_binary=torch._C,
        torch_tensor=sys.modules['torch._tensor'],torch_autograd=sys.modules['torch.autograd'],
        torch_random=torch.random,numpy=np,numpy_binary=np.core._multiarray_umath)
    result = {name:reference(module.__file__) for name,module in modules.items()}
    objects = dict(torch_Module=torch.nn.Module,torch_Linear=torch.nn.Linear,
        AdamW=torch.optim.AdamW,Optimizer=torch.optim.Optimizer,
        point_math=torch.nn.functional.binary_cross_entropy_with_logits,
        torch_serialization=torch.load)
    result.update({name:reference(inspect.getsourcefile(obj)) for name,obj in objects.items()})
    require(all(ref in gate['closure'].values() and
        gate['admission']['bound_source_sha256'].get(ref['path']) == ref['sha256']
        for ref in result.values()), 'Actual CPU library/binary/serialization SOURCE absent or unbound')
    require(result['AdamW'] == gate['context']['adamw_SOURCE'] and
        result['point_math'] == gate['context']['interpolation_SOURCE'], 'Original math SOURCE differs')
    return result


def finite_state(torch, state, arm):
    require(arm in ARMS and type(state) is dict and tuple(state) == NAMES[arm],
        'Exact per-arm state names differ')
    for name,shape in zip(NAMES[arm],SHAPES[arm]):
        value = state[name]
        require(isinstance(value,torch.Tensor) and value.device.type == 'cpu' and
            value.dtype == torch.float32 and tuple(value.shape) == shape and not value.requires_grad and
            bool(torch.isfinite(value).all()), 'Finite exact per-arm CPU state differs: '+arm+'/'+name)
    require(sum(state[name].numel() for name in NAMES[arm]) == PARAMETERS[arm],
        'Per-arm parameter count differs')


def records_check(report,training,start,decision,trace_rows):
    fresh = report['fresh_optimizer_state_entries_per_model']
    require(start['schema'] == START_SCHEMA and start['source'] == report['source'] and
        start['spec'] == report['spec'] and start['GPU_admission'] == report['GPU_admission'] and
        start['phase'] == training['phase'] and start['recipe'] == training['recipe'] and
        start['cache_CPU_qualification'] == report['cache_CPU_qualification'] and
        start['cache_spec'] == report['cache_spec'] and start['cache_report'] == report['cache_report'] and
        start['teacher_checkpoint'] == report['teacher_checkpoint'] and
        start['point_heads_source'] == report['point_heads_source'] and
        start['PR_point_heads_source'] == report['PR_point_heads_source'] and
        start['head_CPU_qualification'] == report['head_CPU_qualification'] and
        start['fresh_optimizer_state_entries_per_model'] == fresh and
        set(start['initial_head_state_proofs']) == set(KEYS) and
        start['paired_seed'] is True and start['paired_initialization'] is False and
        start['paired_cold_native_identity'] is True and start['paired_image_order'] is True and
        start['native_model_loaded'] is False and start['no_DEV'] is True and
        start['no_TEST'] is True and start['no_AP'] is True, 'Actual per-arm cold/fresh START differs')
    expected_rows = {(seed,epoch) for seed in SEEDS for epoch in range(1,9)}
    require(len(trace_rows) == 16 and {(row['seed'],row['epoch']) for row in trace_rows} == expected_rows,
        'Complete unique16 fixed8 epoch traces required')
    trace = {(row['seed'],row['epoch']):row for row in trace_rows}
    for row in trace_rows:
        require(row['images'] == 741 and row['GT_instances'] == 1661 and row['updates_per_model'] == 1661 and
            row['point_visits_per_model'] == 1700864 and row['paired_initialization'] is False and
            row['paired_seed'] is True and row['paired_cold_native_identity'] is True and
            row['paired_image_order'] is True and row['native_forward_calls'] == 0 and
            row['native_weights_updated'] is False and
            set(row['online_BCE']) == set(row['full_FIT_endpoint_BCE']) == set(ARMS) and
            all(type(value) in (int,float) and math.isfinite(value) and value >= 0
                for field in ('online_BCE','full_FIT_endpoint_BCE') for value in row[field].values()),
            'Actual1661-instance epoch counters and finite losses differ')
    for key in KEYS:
        arm,seed_text = key.split('_seed',1)
        seed = int(seed_text)
        history = report['endpoint_BCE_histories'][key]
        require(len(history) == 8 and [row['epoch'] for row in history] == list(range(1,9)) and
            all(type(row['BCE']) in (int,float) and math.isfinite(row['BCE']) and row['BCE'] >= 0 for row in history) and
            all(trace[(seed,row['epoch'])]['full_FIT_endpoint_BCE'][arm] == row['BCE'] for row in history),
            'Finite bound full8 history/TRACE differs')
    require(decision == dict(rule=training['recipe']['extension_rule'],fixed_final_epoch=8,
        continue_all4=False,DEV_accessed=False,TEST_accessed=False,best_epoch_selection=False,
        automatic_extension=False), 'Exact fixed8 decision differs; extension branches are forbidden')
    return trace


def checkpoint_check(torch,ref,proof_ref,key,training,report,spec,start,trace,factory):
    arm,seed_text = key.split('_seed',1)
    require(arm in ARMS, 'Known C/PR checkpoint arm required')
    seed = int(seed_text)
    cp = torch.load(str(bound(ref)),map_location='cpu')
    proof = read(bound(proof_ref))
    require(type(cp) is dict and set(cp) == {'state_dict','optimizer_state_dict','private_order_state','meta'},
        'Own complete checkpoint fields differ')
    state,meta = cp['state_dict'],cp['meta']
    finite_state(torch,state,arm)
    updates,epoch,count = 13288,8,len(NAMES[arm])
    require(meta['schema'] == CP_SCHEMA and meta['source'] == spec['training_entry_source'] and
        meta['spec'] == spec['training_spec'] and meta['GPU_admission'] == spec['training_GPU_admission'] and
        meta['phase'] == training['phase'] and meta['arm'] == arm and meta['seed'] == seed and
        meta['point_heads_source'] == spec['point_heads_source'] and
        meta['PR_point_heads_source'] == spec['PR_point_heads_source'] and
        meta['head_factory_source'] == spec[FACTORY_SPEC_KEYS[arm]] and
        meta['head_CPU_qualification'] == spec['head_CPU_qualification'] and
        meta['cache_spec'] == report['cache_spec'] and meta['cache_report'] == report['cache_report'] and
        meta['cache_CPU_qualification'] == report['cache_CPU_qualification'] and
        meta['teacher_checkpoint'] == report['teacher_checkpoint'] and meta['native_weights_epoch'] == 24 and
        meta['native_weights_updated'] is False and meta['completed_head_epochs'] == epoch and
        meta['updates'] == meta['GT_visits'] == updates and meta['point_visits'] == updates*1024 and
        meta['image_visits'] == epoch*741 and meta['points_fixed_across_epochs'] is True and
        meta['recipe'] == training['recipe'] and meta['parameter_tensors'] == count and
        meta['parameter_scalars'] == PARAMETERS[arm] and meta['no_pending_gradients'] is True and
        meta['formal_native_initializer'] is False and meta['training_resume_eligible'] is False and
        meta['paired_initialization'] is False and meta['paired_seed'] is True and
        meta['paired_cold_native_identity'] is True and meta['paired_image_order'] is True,
        'Own exact fixed8 per-arm checkpoint metadata differs')
    require(proof['checkpoint'] == ref and proof['updates'] == updates and
        proof['parameter_tensors'] == count and proof['parameter_scalars'] == PARAMETERS[arm] and
        proof['full_readback_exact'] is True and proof['state'] == {name:digest(value) for name,value in state.items()},
        'Own checkpoint finite state proof differs')
    opt = cp['optimizer_state_dict']
    require(type(opt) is dict and set(opt) == {'state','param_groups'} and
        len(opt['param_groups']) == 1 and len(opt['state']) == count,
        'Per-arm6/8 AdamW endpoint moment populations required')
    group = opt['param_groups'][0]
    identifiers = group['params']
    require(group['lr'] == 3e-4 and group['weight_decay'] == 1e-4 and group['betas'] == (.9,.999) and
        group['eps'] == 1e-8 and group['amsgrad'] is False and
        len(identifiers) == len(set(identifiers)) == count and set(identifiers) == set(opt['state']) and
        set(proof['optimizer']) == set(NAMES[arm]), 'Own constant AdamW group/proof population differs')
    for name,shape,identifier in zip(NAMES[arm],SHAPES[arm],identifiers):
        entry = opt['state'][identifier]
        step = entry['step']
        require(set(entry) == {'step','exp_avg','exp_avg_sq'} and isinstance(step,torch.Tensor) and
            step.device.type == 'cpu' and step.numel() == 1 and step.dtype == torch.float32 and
            bool(torch.isfinite(step).all()) and float(step) == updates and
            all(isinstance(entry[field],torch.Tensor) and entry[field].device.type == 'cpu' and
                entry[field].dtype == torch.float32 and tuple(entry[field].shape) == shape and
                bool(torch.isfinite(entry[field]).all()) for field in ('exp_avg','exp_avg_sq')) and
            proof['optimizer'][name] == dict({field:digest(value) for field,value in entry.items()},updates=updates),
            'Own6/8 endpoint moment values/counts/proof differ: '+key+'/'+name)
    history = meta['endpoint_BCE_history']
    require(history == report['endpoint_BCE_histories'][key] and len(history) == 8,
        'Own fixed8 endpoint history differs')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    orders = meta['epoch_orders']
    require(len(orders) == 8, 'Complete private fixed8 image orders required')
    for number,row in enumerate(orders,1):
        order = torch.randperm(741,generator=generator).tolist()
        sha = hashlib.sha256(json.dumps(order,separators=(',',':')).encode('ascii')).hexdigest()
        require(row == dict(epoch=number,indices=order,sha256=sha) and
            trace[(seed,number)]['image_order_sha256'] == sha and
            trace[(seed,number)]['full_FIT_endpoint_BCE'][arm] == history[number-1]['BCE'],
            'Own private image order/TRACE/loss history differs')
    private = cp['private_order_state']
    require(isinstance(private,torch.Tensor) and private.device.type == 'cpu' and
        private.dtype == torch.uint8 and torch.equal(generator.get_state(),private) and
        proof['private_order_state'] == digest(private), 'Own private RNG endpoint differs')
    activity = meta['gradient_activity']
    require(set(activity) == set(NAMES[arm]) and all(set(value) == {'present','nonzero'} and
        type(value['present']) is int and value['present'] == updates and
        type(value['nonzero']) is int and 0 <= value['nonzero'] <= updates for value in activity.values()),
        'Bound per-arm gradient activity records differ')
    cold = factory.make_point_head(torch,seed)
    cold_state = dict(cold.state_dict())
    finite_state(torch,{name:value.detach() for name,value in cold_state.items()},arm)
    require({name:digest(value) for name,value in cold_state.items()} == start['initial_head_state_proofs'][key] and
        not bool((cold.final.weight != 0).any()) and not bool((cold.final.bias != 0).any()),
        'Own arm_seed cold initializer proof differs; C/PR weights are not asserted identical')
    cold.load_state_dict(state,strict=True)
    require(not tuple(cold.buffers()), 'Unexpected head buffers')
    del cold,cp,opt
    return state,dict(checkpoint=ref,proof=proof_ref,epoch=8,updates=updates,
        parameter_tensors=count,parameter_scalars=PARAMETERS[arm],strict_CPU_loaded=True,
        AdamW_endpoint_step=updates,private_order_reconstructed=True,
        gradient_activity_record=activity,head_factory_source=spec[FACTORY_SPEC_KEYS[arm]],
        own_cold_factory_proof_CPU_reconstructed=True)


def packet_check(torch,value,row,index,cache_report,cache):
    require(value['schema'] == cache.PACKET_SCHEMA and value['dataset_index'] == row['dataset_index'] == index and
        value['identity'] == row['identity'] and value['source'] == cache_report['source'] and
        value['spec'] == cache_report['spec'] and value['phase'] == cache_report['phase'],
        'Own qualified cache row differs')
    G = len(value['identity']['ann_ids'])
    require(1 <= G <= 100 and G == row['raw_GT'] and len(set(value['identity']['ann_ids'])) == G,
        'Complete positive raw-GT instance order required')
    shapes = dict(F_samples=(G,1024,256),query_embeddings=(G,256),native_logits=(G,1024),
        patches=(G,1024,49),native_class_logits=(1,100,2))
    for name,shape in shapes.items():
        tensor = value[name]
        require(isinstance(tensor,torch.Tensor) and tensor.device.type == 'cpu' and tensor.dtype == torch.float32 and
            tuple(tensor.shape) == shape and not tensor.requires_grad and bool(torch.isfinite(tensor).all()),
            'Actual detached FP32 cache operand differs: '+name)
    for name,shape,dtype in (('targets',(G,1024),torch.bool),('point_indices',(G,1024),torch.int64),
        ('query_ids',(G,),torch.int64)):
        tensor = value[name]
        require(isinstance(tensor,torch.Tensor) and tensor.device.type == 'cpu' and tensor.dtype == dtype and
            tuple(tensor.shape) == shape and not tensor.requires_grad, 'Cache discrete operand differs: '+name)
    H,W = value['identity']['original_shape']
    require(bool(((value['point_indices'] >= 0) & (value['point_indices'] < H*W)).all()) and
        len(set(value['query_ids'].tolist())) == G and
        bool(((value['query_ids'] >= 0) & (value['query_ids'] < 100)).all()) and
        bool(((value['patches'] >= 0) & (value['patches'] <= 1)).all()) and
        all(digest(value[name]) == expected for name,expected in row['cache_digests'].items()),
        'Qualified point/query/value digests differ')
    return G


def independent_logits(torch,state,F,e,L,patch,arm):
    """Explicit per-arm functional arithmetic; never trained head.forward."""
    require(arm in ARMS, 'Known C/PR arm required')
    normalized_F = torch.nn.functional.layer_norm(F,(256,),weight=None,bias=None,eps=1e-5)
    coarse = L.clamp(-8,8).unsqueeze(1)/8
    if arm == 'C':
        normalized_e = torch.nn.functional.layer_norm(e,(256,),weight=None,bias=None,eps=1e-5)
        inputs = torch.cat((normalized_F,normalized_e,coarse,torch.zeros_like(patch)),dim=1)
        hidden = torch.relu(torch.nn.functional.linear(inputs,state['hidden1.weight'],state['hidden1.bias']))
        hidden = torch.relu(torch.nn.functional.linear(hidden,state['hidden2.weight'],state['hidden2.bias']))
        residual = torch.nn.functional.linear(hidden,state['final.weight'],state['final.bias']).squeeze(1)
    else:
        hidden = normalized_F
        for name in ('hidden1','hidden2','hidden3'):
            hidden = torch.relu(torch.nn.functional.linear(torch.cat((hidden,coarse),dim=1),
                state[name+'.weight'],state[name+'.bias']))
        residual = torch.nn.functional.linear(torch.cat((hidden,coarse),dim=1),
            state['final.weight'],state['final.bias']).squeeze(1)
    result = torch.where(L.abs() <= 2,L+residual,L)
    require(bool(torch.isfinite(residual).all()) and bool(torch.isfinite(result).all()) and
        torch.equal(result[L.abs() > 2],L[L.abs() > 2]), 'Independent original-band finite logits differ')
    return result


def endpoint_check(torch,states,gate,spec,started):
    losses = {key:[] for key in KEYS}
    instances,arithmetic_max = [],0.
    report,q,cache = gate['cache_report'],gate['cache_q'],gate['cache']
    with torch.no_grad():
        for index,(ref,row) in enumerate(zip(report['packet_refs'],report['rows'])):
            deadline(started,spec['deadline_seconds'])
            require(row['packet'] == ref == q['packet_refs'][index], 'Qualified741 packet order differs')
            value = torch.load(str(bound(ref)),map_location='cpu')
            G = packet_check(torch,value,row,index,report,cache)
            for g in range(G):
                F,e,L = value['F_samples'][g],value['query_embeddings'][g].expand(1024,256),value['native_logits'][g]
                patch,y = value['patches'][g],value['targets'][g].to(dtype=torch.float32)
                instances.append(dict(dataset_index=index,image_id=value['identity']['image_id'],
                    annotation_id=value['identity']['ann_ids'][g],query_id=int(value['query_ids'][g])))
                for key in KEYS:
                    arm = key.split('_seed',1)[0]
                    logits = independent_logits(torch,states[key],F,e,L,patch,arm)
                    per_point = logits.clamp(min=0)-logits*y+torch.log1p(torch.exp(-logits.abs()))
                    functional = torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
                    require(torch.allclose(per_point,functional,rtol=NUMERIC['instance_BCE_arithmetic_rtol'],
                        atol=NUMERIC['instance_BCE_arithmetic_atol']) and bool(torch.isfinite(per_point).all()),
                        'Independent stable point BCE arithmetic differs')
                    arithmetic_max = max(arithmetic_max,float((per_point-functional).abs().max()))
                    losses[key].append(float(per_point.mean()))
            del value
            print(json.dumps(dict(N10_CPU_training_endpoint_images=index+1,instances=len(instances))),flush=True)
    require(len(instances) == 1661 and all(len(values) == 1661 for values in losses.values()),
        'Full1661 fixed-cache endpoint population differs')
    return losses,instances,arithmetic_max


def atomic_write(path,value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    require(path.parent.is_dir() and not path.exists() and not temporary.exists(), 'Fresh Q output required')
    with temporary.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def qualify(spec_path):
    started = time.monotonic()
    path = Path(spec_path)
    require(path.is_absolute(), 'Absolute ROOT CPU SPEC required')
    path = path.resolve(strict=True)
    spec = read(path)
    gate = source_gate(path,spec,started)
    import psutil
    require(psutil.Process().cpu_affinity() == [0,1], 'Actual CPU01 required before numerical imports')
    import torch
    import numpy as np
    require(torch.__version__ == '2.0.1+cu118' and np.__version__ == '1.23.5' and
        not torch.cuda.is_initialized(), 'Original CPU Torch/NumPy and cold CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    resources(psutil,torch)
    libraries = imported_files(psutil,torch,np,gate)
    external = torch.random.get_rng_state().clone()
    factories = {arm:load(spec[key],'_N10_CPU_'+arm+'_cold_constructor')
        for arm,key in FACTORY_SPEC_KEYS.items()}
    training,report = gate['training'],gate['report']
    start,decision = read(bound(report['start'])),read(bound(report['decision']))
    trace_rows = []
    def reject(value):
        raise ValueError('Nonfinite TRACE constant: '+value)
    for line in bound(report['trace']).read_text(encoding='utf-8').splitlines():
        if line.strip():
            trace_rows.append(json.loads(line,parse_constant=reject))
    trace = records_check(report,training,start,decision,trace_rows)
    states,results = {},{}
    for key in KEYS:
        final = report['checkpoints'][key]
        require(final['completed_head_epochs'] == 8 and final['updates'] == 13288 and
            final['mean_endpoint_BCE'] == report['endpoint_BCE_histories'][key][-1]['BCE'] and
            final['head_state_finite'] is True and final['no_pending_gradients'] is True,
            'Own exact fixed8 final checkpoint result differs')
        arm = key.split('_seed',1)[0]
        states[key],results[key] = checkpoint_check(torch,final['checkpoint'],final['proof'],key,
            training,report,spec,start,trace,factories[arm])
    per_instance,instances,arithmetic_max = endpoint_check(torch,states,gate,spec,started)
    endpoints = {}
    for key in KEYS:
        mean = math.fsum(per_instance[key])/1661
        exported = report['endpoint_BCE_histories'][key][-1]['BCE']
        require(math.isclose(mean,exported,rel_tol=NUMERIC['endpoint_BCE_rtol'],
            abs_tol=NUMERIC['endpoint_BCE_atol']), 'Independent final fullFIT mean BCE differs: '+key)
        endpoints[key] = dict(CPU_mean_BCE=mean,GPU_recorded_mean_BCE=exported,
            absolute_difference=abs(mean-exported),instances=1661,points=1700864)
    require(torch.equal(external,torch.random.get_rng_state()) and not torch.cuda.is_initialized() and
        'mmdet' not in sys.modules, 'CPU endpoint changed external RNG/CUDA or loaded native model')
    for ref in spec['sources']:
        deadline(started,spec['deadline_seconds'])
        bound(ref)
    require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source'] and
        reference(spec['CPU_admission_file']) == gate['admit_ref'] and
        libraries == imported_files(psutil,torch,np,gate), 'Final inputs/libraries changed')
    resources(psutil,torch)
    deadline(started,spec['deadline_seconds'])
    q = dict(schema=Q_SCHEMA,status=STATUS,phase=spec['phase'],source=spec['training_entry_source'],
        reader_source=gate['source'],spec=gate['spec_ref'],training_spec=spec['training_spec'],
        training_report=spec['training_report'],training_GPU_RAW=spec['training_GPU_RAW'],
        training_GPU_admission=spec['training_GPU_admission'],CPU_admission=gate['admit_ref'],
        training_phase=training['phase'],cache_CPU_qualification=training['cache_CPU_qualification'],
        point_heads_source=spec['point_heads_source'],PR_point_heads_source=spec['PR_point_heads_source'],
        head_CPU_qualification=spec['head_CPU_qualification'],
        head_CPU_spec=training['head_CPU_spec'],head_CPU_entry_source=training['head_CPU_entry_source'],
        head_CPU_receipt=training['head_CPU_receipt'],head_CPU_admission=training['head_CPU_admission'],
        RAW_inputs=spec['RAW_inputs'],prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'],
        recipe=training['recipe'],numeric_protocol=NUMERIC,completed_head_epochs=8,
        images_per_epoch=741,instances_per_epoch=1661,points_per_endpoint=1700864,updates_per_model=13288,
        C_parameters=40257,PR_parameters=40497,parameter_tensors_by_arm=dict(C=6,PR=8),
        checkpoints=report['checkpoints'],checkpoint_results=results,epoch8_checkpoint_results=results,
        per_arm_cold_initializer_proofs_CPU_checked=True,C_PR_identical_weight_initialization_claimed=False,
        paired_seed_and_native_cold_identity_provenance_CPU_checked=True,
        private_image_order_and_endpoint_RNG_CPU_reconstructed=True,
        AdamW_endpoint_states_and_stepcounts_CPU_checked=True,all16_epoch_records_CPU_checked=True,
        fixed8_no_extension_decision_CPU_checked=True,final_fullFIT_endpoint_BCE_CPU_recomputed=True,
        final_endpoints=endpoints,instance_order=instances,per_instance_CPU_BCE=per_instance,
        stable_instance_BCE_arithmetic_max_abs_difference=arithmetic_max,
        intermediate_epoch_BCE_numerically_replayed=False,every_gradient_or_AdamW_update_CPU_replayed=False,
        optimizer_continuation_trajectory_CPU_replayed=False,
        original_native_teacher_loaded_or_replayed=False,GT_or_JPEG_decode_repeated=False,
        qualified_cache_targets_and_patches_reused=True,
        native610_unchanged_SOURCE_observed=gate['cache_q']['native610_unchanged_SOURCE_observed'],
        native610_unchanged_independent_CPU_verified=False,optimizer_created=False,weights_updated=False,
        no_DEV=True,no_TEST=True,no_AP=True,task_efficacy_established=False,innovation_established=False,
        generalization_established=False,convergence_established=False,
        actual_imported_files=libraries,CPU_execution=dict(affinity=[0,1],threads=2,
            CUDA_VISIBLE_DEVICES='',CUDA_initialized=False),CPU_affinity=[0,1],threads=2,CUDA_initialized=False,
        elapsed_seconds=time.monotonic()-started,deadline_seconds=spec['deadline_seconds'],automatic_retry=False,
        current_owned_CPU_terminal_self_certified=False,whole_leaf_VRAM_certified=False,
        evidence_limits=[
            'Independent final1024-point-per-instance FIT endpoint arithmetic, not full-frame risk, AP or unseen-data convergence.',
            'Only fixed8 final states exist; intermediate epoch BCE and gradient activity are checked bound records, not replayed trajectories.',
            'Four fresh-zero optimizer records and final6/8 moment populations are SOURCE/record-consistent; endpoint moments do not prove every update.',
            'Each arm_seed cold factory proof is reconstructed; C/PR share seeds/image orders/cold native identity, not identical architecture weights.',
            'C and PR differ in feature information, width, depth and condition paths; nearby parameter counts do not prove matched capacity or compute.',
            'GT/JPEG identity relies on the independently qualified immutable cache; no native backbone/matching/decoding is replayed.',
            'The final CPU/GPU BCE comparison uses a prelocked practical tolerance, not bitwise equality or a certified GPU error bound.',
            'No optimizer step, native teacher, DEV/TEST or AP is executed by this reader; no task benefit or scientific novelty follows from a pass.',
            'ROOT separately closes this current CPU actual owned terminal; the Q cannot self-certify its final RAW.'
        ])
    target = Path(spec['Q_path'])
    atomic_write(target,q)
    del states
    gc.collect()
    print(json.dumps(dict(status=STATUS,Q=reference(target)),allow_nan=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Explicit single ROOT-owned CPU execution required')
    qualify(args.spec)
