"""SOURCE-ONLY UNEXECUTED U57 DGX fixed-E50 CPU inference model restoration.

Standard-library imports only. Actual Root acceptance, source admission, original
CPU model construction and all resource enforcement remain caller obligations.
This module never starts training/evaluation, grants qualification or retries.
"""
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import random
import time
import traceback
from types import ModuleType

ARMS = ('PLAIN', 'FULL')
TOTAL = 37050
FINAL_CURSOR = dict(global_updates=37050, completed_epochs=50, within_epoch=0)
BODY_SHA256 = 'd80ab26ad7092431f4001535705e7afcc4fda7cbf66bacb4dc2cb66edd3d6724'
BODY_BYTES = 43423
ENTRY_SHA256 = 'cd62a640e63210bcb63e58e09bb1e429f1a68ea66dcfec240831a5f1db31aca9'
ENVIRONMENT = dict(python='3.12.14', torch='2.10.0+cu130', numpy='1.26.4', system='Linux', machine='aarch64')
CONSTRUCTOR_ROLES = ('shared_initialization', 'full_legacy', 'full_dsconv', 'full_head', 'full_feature',
                     'plain_legacy', 'plain_feature', 'instance_adapter')
RESTORE_PINS = dict(
    training_state=BODY_SHA256, training_entry=ENTRY_SHA256,
    shared_initialization='4ba968db7688a03d5975c8546607791a5e09e82ef6c5c7d096cd6a368d1b0292',
    full_legacy='c210b318be29aca7d26516e9f47c8849e2c6a76642d4049fe7b133726ee809b9',
    full_dsconv='b4b612edf9f87cf29dc48bb675d501658fdec5dbee1ea363a7f02d86d197e5f0',
    full_head='6caba966ac7601b611fd1b36fdbd1d758875c5d029758fefe586738c8009f16e',
    full_feature='f7953780580449dd246cbdf1803652183c1dd178f2fd60b37adb238a6d2b6a5a',
    plain_legacy='902e13e5c60b1579459854c6bf4393ad8f268092a73a1d851f0fa640c810f48d',
    plain_feature='d93ff973a57ec3abfde98bb7b9ab4fcf5a3443445a512c2acd16460fce5bfbad',
    instance_adapter='9ad221170f69c028474f76700c991cb76dfecf19b11f5198767843b8e378d8b3',
    u36_helpers='1e726092a19d80a06f58d6a56f85f4c15bff2b744f0468a7e0a35fa2c85999e5',
    u39_training_ops='378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def strict_ref(value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}
         and type(value['path']) is str and '\\' not in value['path']
         and PurePosixPath(value['path']).anchor == '/'
         and str(PurePosixPath(value['path'])) == value['path']
         and '..' not in PurePosixPath(value['path']).parts
         and type(value['bytes']) is int and value['bytes'] > 0
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'), 'Strict canonical DGX POSIX file reference')
    return value


def digest_string(value):
    need(type(value) is str and len(value) == 64 and set(value) <= set('0123456789abcdef'), 'SHA256 metadata')
    return value


def _validate_endpoint(training, entry, metadata, mode):
    """Pure metadata gate only; genuine documents must be byte-closed by restore_arm."""
    need(type(metadata) is dict and set(metadata) == {'refs', 'qualified', 'spec', 'pair_commit', 'order'},
         'Four explicitly referenced U57 documents, no inferred endpoint path')
    refs = metadata['refs']
    need(type(refs) is dict and set(refs) == {'qualified', 'spec', 'pair_commit', 'order'}, 'Four endpoint references')
    for value in refs.values():
        strict_ref(value)
    q, spec, pair = metadata['qualified'], metadata['spec'], metadata['pair_commit']
    final_cursor = FINAL_CURSOR if mode == 'FORMAL' else training.cursor(2)
    final_stop = TOTAL if mode == 'FORMAL' else 2
    need(q['schema'] == 'Root-U57-DGX-original-update-training-accepted-v1'
         and q['root_accepted'] is True and q['paired_prefix_advanced'] is True
         and q['mode'] == spec['prelock']['mode'] == pair['identity']['mode'] == mode
         and q['result']['paired_checkpoint_prefix'] == pair['cursor'] == final_cursor,
         'Only the actually Root-accepted fixed paired endpoint for this separate route')
    admitted = entry.source_table(spec)
    for value in admitted.values():
        strict_ref(value)
    training.validate_prelock(spec['prelock'])
    training.validate_order(metadata['order'])
    start, stop = training.validate_span(spec['segment'], spec['prelock'])
    need(stop == final_stop and q['accepted_span'] == pair['span'] == spec['segment'], 'Same actual final segment; no fixed last-start assumption')
    for value in (q['result']['paired_checkpoint_prefix'], pair['cursor']):
        need(type(value) is dict and set(value) == set(final_cursor)
             and all(type(value[k]) is int and value[k] == v for k, v in final_cursor.items()), 'Integer fixed paired cursor')
    terminal = dict(root_observed_tool_exit_code=0, SSH_returncode=0, worker_returncode=0, reaped=True,
                    owner_original_birth_absent=True, child_original_birth_absent=True,
                    remaining_owned_group=[], all_owned_processes_absent=True)
    observed = q['original_terminal']
    need(type(observed) is dict and set(observed) == set(terminal)
         and all(type(observed[k]) is type(v) and observed[k] == v for k, v in terminal.items()),
         'Actual Root/SSH/worker zero and completed original ownership closure')
    need(all(q[k] is False for k in ('AP', 'DEV', 'TEST')), 'Training acceptance itself contains no evaluation claim')
    need(set(q['evidence_refs']) == {'spec_ref', 'worker_report_ref', 'owner_receipt_ref'}
         and q['evidence_refs']['spec_ref'] == refs['spec'], 'Accepted actual source spec reference')
    for value in q['evidence_refs'].values():
        strict_ref(value)
    need(spec['body_source']['sha256'] == BODY_SHA256 and spec['body_source']['bytes'] == BODY_BYTES
         and spec['entry_source']['sha256'] == ENTRY_SHA256, 'Independent reviewed U57 body/entry, never U40')
    need(training.ENVIRONMENT == ENVIRONMENT and spec['prelock']['environment'] == ENVIRONMENT,
         'DGX Python3.12/Torch2.10/NumPy1.26.4 source recipe only')
    context, order = spec['context'], metadata['order']
    identity = dict(mode=mode, run_id=spec['prelock']['run_id'], recipe=deepcopy(training.RECIPE),
                    environment=deepcopy(ENVIRONMENT), backend_policy=deepcopy(spec['prelock']['backend_policy']),
                    sources=dict(body_source=spec['body_source'], component_sources=spec['source_refs']),
                    context={k: v for k, v in context.items() if k != 'parent_pair'},
                    order_sha256=order['order_sha256'], execution_order_sha256=training.metadata_sha256(
                        order['order'] if mode == 'FORMAL' else [54, 710]))
    need(q['run_id'] == identity['run_id']
         and training.metadata_sha256(pair['identity']) == training.metadata_sha256(identity),
         'Both arms use the same formal run, original sources, recipe, runtime, context and complete stored order')
    result = q['result']
    need(type(result) is dict and set(result) == {'pair_commit_ref', 'checkpoint_refs', 'trace_refs', 'order_ref', 'paired_checkpoint_prefix'}
         and result['pair_commit_ref'] == refs['pair_commit']
         and result['order_ref'] == pair['order_ref'] == context['order_ref'] == refs['order']
         and pair['previous_pair'] == (None if start == 0 else context['parent_pair']['pair_commit_ref'])
         and result['checkpoint_refs'] == pair['checkpoint_refs']
         and result['trace_refs'] == pair['trace_refs'] == pair['traces'], 'One accepted pair/order/two-arm output chain')
    need(pair['schema'] == 'U57-DGX-paired-checkpoint-commit-v1'
         and pair['status'] == 'PAIRED_CHECKPOINTS_WRITTEN_PENDING_ROOT_ACCEPTANCE'
         and set(pair['checkpoints']) == set(pair['checkpoint_refs']) == set(pair['trace_refs']) == set(ARMS),
         'Complete U57 paired commit, not a partial arm or pending worker report alone')
    digest_string(pair['actual_trace_sha256'])
    output = PurePosixPath(spec['output_directory']) / 'SEGMENT'
    need(refs['pair_commit']['path'] == str(output / 'PAIR_COMMIT.json'), 'Pair location derives from this exact spec')
    output_refs = [refs['pair_commit']]
    for arm in ARMS:
        record = pair['checkpoints'][arm]
        need(type(record) is dict and set(record) == {'reference', 'CPU_state_signature_sha256', 'serialization_roundtrip_verified'}
             and record['reference'] == pair['checkpoint_refs'][arm]
             and record['serialization_roundtrip_verified'] is True, 'Exact complete-state checkpoint record: ' + arm)
        digest_string(record['CPU_state_signature_sha256'])
        strict_ref(record['reference'])
        strict_ref(pair['trace_refs'][arm])
        need(record['reference']['path'] == str(output / (arm + '.pth'))
             and pair['trace_refs'][arm]['path'] == str(output / (arm + '.trace.jsonl')), 'Actual arm-specific output paths')
        output_refs += [record['reference'], pair['trace_refs'][arm]]
    need(len({v['path'] for v in output_refs}) == 5 and len({v['path'] for v in refs.values()}) == 4,
         'No aliased arms or endpoint documents')
    return dict(schema='U58-U57-DGX-fixed-endpoint-metadata-v1', mode=mode, formal_endpoint=mode == 'FORMAL', run_id=q['run_id'],
                cursor=deepcopy(final_cursor), checkpoint_refs=deepcopy(pair['checkpoint_refs']),
                pair_commit_ref=deepcopy(refs['pair_commit']), order_ref=deepcopy(refs['order']),
                training_qualified=deepcopy(refs['qualified']), endpoint_refs=deepcopy(refs),
                restore_source_refs=dict(training_state=deepcopy(spec['body_source']),
                                         training_entry=deepcopy(spec['entry_source']), **deepcopy(spec['source_refs'])),
                metadata_only=True, numerical_runtime_qualified=False)


def validate_final_endpoint(training, entry, metadata):
    """Production gate: only Root-accepted independent FORMAL E50, never midpoint."""
    return _validate_endpoint(training, entry, metadata, 'FORMAL')


def validate_smoke_endpoint(training, entry, metadata):
    """Technical gate: only separately accepted SMOKE2; no FORMAL/AP admission."""
    return _validate_endpoint(training, entry, metadata, 'SMOKE')


def validate_restore_sources(endpoint, source_refs):
    """Exact twelve restore roles, including eight original constructor roles."""
    need(type(source_refs) is dict and set(source_refs) == set(RESTORE_PINS), 'Only original restore roles')
    for role, pin in RESTORE_PINS.items():
        strict_ref(source_refs[role])
        need(source_refs[role]['sha256'] == pin and source_refs[role] == endpoint['restore_source_refs'][role],
             'Exact accepted original source reference: ' + role)
    return deepcopy(source_refs)


def _validate_loaded_checkpoint(torch, training, entry, helpers, ops, adapter, arm, metadata, saved, fresh_receipt, mode):
    """Validate an already CPU-loaded container; performs no load, mutation or qualification.

    This public technical check still enforces the FORMAL E50 endpoint. It is
    not a SMOKE route or an alternative production loader. Caller-side model
    construction is not proven merely by an injected initialization receipt.
    """
    need(arm in ARMS, 'One original arm')
    endpoint = _validate_endpoint(training, entry, metadata, mode)
    pair = metadata['pair_commit']
    keys = {'schema', 'arm', 'identity', 'cursor', 'model', 'optimizer', 'effective_parameter_names',
            'inactive_parameter_names', 'optimizer_state_devices', 'module_training_modes', 'rng', 'order_ref',
            'requested_trace_sha256', 'actual_trace_sha256', 'initialization_receipt', 'previous_checkpoint', 'previous_pair'}
    need(type(saved) is dict and set(saved) == keys and saved['schema'] == 'U57-DGX-original-update-arm-state-v1'
         and saved['arm'] == arm and type(saved['optimizer']) is dict, 'Complete original U57 arm container, not model-only or U40')
    optimizer, rng = saved['optimizer'], saved['rng']
    need(set(optimizer) == {'state', 'param_groups'} and type(optimizer['state']) is dict
         and type(optimizer['param_groups']) is list and len(optimizer['param_groups']) == 1
         and all(type(value) is dict for value in optimizer['param_groups'])
         and type(saved['optimizer_state_devices']) is dict,
         'Original complete optimizer container is loaded and signature-verified, never restored')
    need(type(rng) is dict and set(rng) == {'python', 'numpy', 'torch_CPU', 'torch_CUDA'}
         and type(rng['python']) is tuple and len(rng['python']) == 3
         and type(rng['numpy']) is dict and set(rng['numpy']) == {'name', 'keys', 'position', 'has_gauss', 'cached_gaussian'}
         and isinstance(rng['torch_CPU'], torch.Tensor) and isinstance(rng['torch_CUDA'], torch.Tensor),
         'Original complete four-RNG container is loaded and signature-verified, never restored')
    signature = training.metadata_sha256(training._signature(torch, helpers, saved))
    need(signature == pair['checkpoints'][arm]['CPU_state_signature_sha256'],
         'Complete checkpoint CPU state signature, including optimizer/RNG; never substitute a model-only digest')
    expected_cursor = endpoint['cursor']
    active_plan = deepcopy(metadata['order'])
    if mode == 'SMOKE':
        active_plan['order'] = [54, 710]
    need(type(saved['cursor']) is dict and set(saved['cursor']) == set(expected_cursor)
         and all(type(saved['cursor'][k]) is int and saved['cursor'][k] == v for k, v in expected_cursor.items())
         and training.metadata_sha256(saved['identity']) == training.metadata_sha256(pair['identity'])
         and saved['order_ref'] == endpoint['order_ref']
         and saved['actual_trace_sha256'] == pair['actual_trace_sha256']
         and saved['requested_trace_sha256'] == training.expected_trace(active_plan, expected_cursor['global_updates'])
         and saved['previous_pair'] == pair['previous_pair']
         and saved['previous_checkpoint'] == (None if metadata['spec']['segment']['start'] == 0
             else metadata['spec']['context']['parent_pair']['checkpoint_refs'][arm]),
         'Exact final arm identity, complete actual/requested order and previous accepted pair/arm')
    need(isinstance(adapter, torch.nn.Module) and not torch.cuda.is_initialized(), 'Caller-owned original cold CPU model')
    current = adapter.state_dict()
    need(isinstance(current, dict) and type(saved['model']) is dict and set(current) == set(saved['model']),
         'Entire original model state, including trained head, BN and inactive states')
    for name, value in saved['model'].items():
        expected = current[name]
        need(isinstance(value, torch.Tensor) and isinstance(expected, torch.Tensor)
             and value.device.type == expected.device.type == 'cpu'
             and value.dtype == expected.dtype and tuple(value.shape) == tuple(expected.shape)
             and (not value.is_floating_point() or value.dtype == torch.float32)
             and bool(torch.isfinite(value).all()), 'Exact finite CPU model dtype/shape: ' + name)
    params = dict(adapter.named_parameters())
    effective, inactive, parameter_report = ops._partition(adapter)
    need(set(params) == effective | inactive and not effective & inactive
         and saved['effective_parameter_names'] == sorted(effective)
         and saved['inactive_parameter_names'] == sorted(inactive)
         and all(p.device.type == 'cpu' and p.dtype == torch.float32 and p.grad is None for p in params.values()),
         'Exact original effective/inactive parameter names; no retained gradients or GPU parameter')
    modes = saved['module_training_modes']
    need(type(modes) is dict and set(modes) == set(dict(adapter.named_modules()))
         and all(type(value) is bool for value in modes.values()), 'Exact full module tree and recorded Boolean training modes')
    need(type(fresh_receipt) is dict and fresh_receipt == saved['initialization_receipt']
         and fresh_receipt['seed'] == fresh_receipt['adapter_seed'] == training.SEED
         and len(fresh_receipt['common_state']) == 104 and fresh_receipt['common_parameter_tensors'] == 68
         and fresh_receipt['common_BN_buffers'] == 36, 'Match caller-supplied original initialization receipt, not proof of construction by itself')
    model_signature = training.metadata_sha256(training._signature(torch, helpers, saved['model']))
    return dict(complete_checkpoint_CPU_state_signature_sha256=signature,
                model_CPU_state_signature_sha256=model_signature,
                endpoint_refs=deepcopy(endpoint['endpoint_refs']), model_mutated=False,
                effective_parameter_names=sorted(effective), inactive_parameter_names=sorted(inactive),
                module_names=sorted(modes), parameter_report=deepcopy(parameter_report),
                numerical_runtime_qualified=False)


def validate_loaded_checkpoint(torch, training, entry, helpers, ops, adapter, arm, metadata, saved, fresh_receipt):
    """Technical validation only; production FORMAL E50 gate, no model mutation."""
    return _validate_loaded_checkpoint(torch, training, entry, helpers, ops, adapter, arm, metadata, saved, fresh_receipt, 'FORMAL')


def _restore_loaded_model(torch, training, entry, helpers, ops, adapter, arm, metadata, saved, fresh_receipt, mode):
    proof = _validate_loaded_checkpoint(torch, training, entry, helpers, ops, adapter, arm, metadata, saved, fresh_receipt, mode)
    adapter.load_state_dict(saved['model'], strict=True)
    training._equal(torch, adapter.state_dict(), saved['model'], 'U58 restored complete model/BN/inactive')
    need(training.metadata_sha256(training._signature(torch, helpers, adapter.state_dict()))
         == proof['model_CPU_state_signature_sha256'], 'Restored model signature, distinct from complete checkpoint signature')
    adapter.eval().requires_grad_(False)
    need(all(not module.training for module in adapter.modules())
         and all(p.device.type == 'cpu' and p.dtype == torch.float32 and not p.requires_grad and p.grad is None
                 for p in adapter.parameters()) and not torch.cuda.is_initialized()
         and training.metadata_sha256(training._signature(torch, helpers, adapter.state_dict()))
                 == proof['model_CPU_state_signature_sha256'], 'Exact trained CPU model retained in frozen eval')
    proof.update(restored_model_state_exact=True, optimizer_restored=False, RNG_restored=False,
                 all_modules_eval=True, all_parameters_frozen=True, all_parameter_grads_none=True,
                 CUDA_initialized=False, model_forward_executed=False, adapter_object_replaced=False,
                 trained_head_reset=False, formal_endpoint=mode == 'FORMAL', mode=mode)
    return proof


def restore_loaded_model(torch, training, entry, helpers, ops, adapter, arm, metadata, saved, fresh_receipt):
    """Injected-container technical helper; caller still owes byte/source/runtime closure.

    Only FORMAL E50 is accepted. This entry cannot qualify a model or establish
    genuine build_pair construction merely from an initialization receipt.
    """
    return _restore_loaded_model(torch, training, entry, helpers, ops, adapter, arm, metadata, saved, fresh_receipt, 'FORMAL')


def _closed(c, reference):
    strict_ref(reference)
    c.check()
    path = Path(c.closed(reference))
    need(str(path) == reference['path'] and path.is_absolute() and not path.is_symlink()
         and path.resolve(strict=True) == path, 'Canonical admitted file, no path or symlink substitution')
    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
            size += len(block)
            c.check()
    after = path.stat()
    stamp = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    need(stamp(before) == stamp(after) and size == reference['bytes'] and digest.hexdigest() == reference['sha256'],
         'Exact stable admitted bytes/SHA')
    return path


def _json(c, reference):
    path = _closed(c, reference)
    raw = path.read_bytes()
    need(len(raw) == reference['bytes'] and hashlib.sha256(raw).hexdigest() == reference['sha256'],
         'Actual metadata byte identity at parse')
    return json.loads(raw)


def _module(c, module, reference, pin):
    path = _closed(c, reference)
    need(reference['sha256'] == pin and hasattr(module, '__file__')
         and Path(module.__file__).resolve(strict=True) == path, 'Actual injected original module file/source')


def _numpy_rng(np):
    value = np.random.get_state()
    return dict(name=value[0], keys=value[1].tolist(), position=int(value[2]),
                has_gauss=int(value[3]), cached_gaussian=float(value[4]))


def _restore_arm(torch, np, c, adapter, arm, metadata, fresh_receipt, *, modules, source_refs, report, mode):
    need(type(report) is dict and not report, 'Caller-owned fresh partial restore report')
    report.update(schema='U58-U57-DGX-CPU-arm-restore-pending-v1', status='STARTED', arm=arm, mode=mode,
                  qualified=False, numerical_runtime_qualified=False, AP=False, DEV=False, TEST=False,
                  new_training=False, optimizer_created=False, optimizer_restored=False, RNG_restored=False,
                  caller_build_pair_construction_proven=False, CPU_checkpoint_optimizer_payload_loaded=False,
                  CPU_checkpoint_RNG_payload_loaded=False, CPU_checkpoint_payload_released=False)
    saved = outside_cpu = None
    started = time.perf_counter()
    try:
        need(type(modules) is dict and set(modules) == set(RESTORE_PINS), 'Inject exact original restore modules')
        training, entry = modules['training_state'], modules['training_entry']
        helpers, ops = modules['u36_helpers'], modules['u39_training_ops']
        endpoint = _validate_endpoint(training, entry, metadata, mode)
        validate_restore_sources(endpoint, source_refs)
        for role, pin in RESTORE_PINS.items():
            _module(c, modules[role], source_refs[role], pin)
        actual = dict(refs=deepcopy(metadata['refs']))
        for key, reference in actual['refs'].items():
            actual[key] = _json(c, reference)
        need(actual == metadata, 'Caller metadata must equal the four actual closed endpoint documents')
        # These are code refs only. No population, image, annotation, cached
        # teacher output, prior checkpoint or trace is read by the helper.
        for reference in endpoint['restore_source_refs'].values():
            _closed(c, reference)
        actual_environment = dict(python=platform.python_version(), torch=str(torch.__version__),
                                  numpy=str(np.__version__), system=platform.system(), machine=platform.machine())
        need(type(torch) is ModuleType and torch.__name__ == 'torch'
             and type(np) is ModuleType and np.__name__ == 'numpy'
             and actual_environment == ENVIRONMENT and torch.get_default_dtype() == torch.float32
             and not torch.cuda.is_initialized(), 'Caller-admitted original DGX libraries and cold CPU FP32 runtime')
        need(modules['full_legacy'].DSConv is modules['full_dsconv'].DSConv
             and modules['full_legacy'].CoordConv is modules['full_head'].CoordConv, 'Original constructor dependency objects')
        need(isinstance(adapter, torch.nn.Module) and all(p.grad is None for p in adapter.parameters())
             and all(value.device.type == 'cpu' for value in adapter.state_dict().values()), 'Existing caller-owned cold CPU adapter')
        outside_cpu = torch.get_rng_state().clone()
        outside_numpy, outside_python = _numpy_rng(np), random.getstate()
        c.check()
        saved = torch.load(str(_closed(c, endpoint['checkpoint_refs'][arm])), map_location='cpu', weights_only=True)
        report['checkpoints_loaded'] = 1
        report['CPU_checkpoint_optimizer_payload_loaded'] = type(saved) is dict and type(saved.get('optimizer')) is dict
        report['CPU_checkpoint_RNG_payload_loaded'] = type(saved) is dict and type(saved.get('rng')) is dict
        proof = _restore_loaded_model(torch, training, entry, helpers, ops, adapter, arm, actual, saved, fresh_receipt, mode)
        report.update(proof)
        need(torch.equal(torch.get_rng_state(), outside_cpu) and _numpy_rng(np) == outside_numpy
             and random.getstate() == outside_python and not torch.cuda.is_initialized(), 'Caller RNG untouched; no CUDA initialization')
        for role, pin in RESTORE_PINS.items():
            _module(c, modules[role], source_refs[role], pin)
        for reference in actual['refs'].values():
            _closed(c, reference)
        _closed(c, endpoint['checkpoint_refs'][arm])
        c.check()
        report.update(status='RESTORED_CPU_ONLY', actual_environment=actual_environment,
                      seed=training.SEED, cursor=deepcopy(endpoint['cursor']),
                      fixed_epoch=50 if mode == 'FORMAL' else None, formal_endpoint=mode == 'FORMAL',
                      checkpoint_ref=deepcopy(endpoint['checkpoint_refs'][arm]),
                      pair_commit_ref=deepcopy(endpoint['pair_commit_ref']), training_qualified=deepcopy(endpoint['training_qualified']),
                      order_ref=deepcopy(endpoint['order_ref']), restore_sources=deepcopy(source_refs),
                      initialization_receipt_sha256=training.metadata_sha256(fresh_receipt),
                      external_CPU_RNG_unchanged=True, external_NumPy_RNG_unchanged=True, external_Python_RNG_unchanged=True,
                      source_and_endpoint_bytes_rechecked=True, model_forward_executed=False,
                      receipt_alone_proves_build_pair=False, population_qualification=False)
    except BaseException as error:
        report.update(status='FAILED', error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
        raise
    finally:
        saved = outside_cpu = None
        gc.collect()
        report['CPU_checkpoint_payload_released'] = True
        report['elapsed_seconds'] = time.perf_counter() - started
    return report


def restore_arm(torch, np, c, adapter, arm, endpoint_metadata, fresh_initialization_receipt, *, modules, source_refs, report):
    """Production FORMAL E50 CPU restore; no construction, optimizer/RNG restore or AP.

    Caller must supply genuine admitted libraries, original build_pair adapter,
    original fresh receipt and live owner Control. Failure may mutate adapter:
    discard it, never silently fall back or retry. Receipt is pending evidence,
    never scientific qualification or proof of genuine model construction.
    """
    return _restore_arm(torch, np, c, adapter, arm, endpoint_metadata, fresh_initialization_receipt,
                        modules=modules, source_refs=source_refs, report=report, mode='FORMAL')


def restore_smoke_arm(torch, np, c, adapter, arm, endpoint_metadata, fresh_initialization_receipt, *, modules, source_refs, report):
    """Separate technical SMOKE2 restore; rejects every FORMAL endpoint, no AP."""
    return _restore_arm(torch, np, c, adapter, arm, endpoint_metadata, fresh_initialization_receipt,
                        modules=modules, source_refs=source_refs, report=report, mode='SMOKE')
