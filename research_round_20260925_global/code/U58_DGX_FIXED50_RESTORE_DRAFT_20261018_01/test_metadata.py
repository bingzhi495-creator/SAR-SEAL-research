"""Fabricated metadata and tensor-like stdlib stubs only; no torch/numpy import."""
from copy import deepcopy
import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest.mock import patch


HERE = Path(__file__).parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


training = module(HERE.parent / 'U57_DGX_ORIGINAL_UPDATE_TRAINING_DRAFT_20261018_01' / 'training_body.py', '_U58_TEST_ORDINARY_BODY')
entry = module(HERE.parent / 'U57_DGX_ORIGINAL_UPDATE_TRAINING_DRAFT_20261018_01' / 'entry.py', '_U58_TEST_ORDINARY_ENTRY')


def ref(name, sha=None, size=1):
    return dict(path='/synthetic/' + name, bytes=size,
                sha256=sha or hashlib.sha256(name.encode()).hexdigest())


def fixture(mode='FORMAL', stop=37050):
    start = 37017 if mode == 'FORMAL' else 1
    run = 'U57_SYNTHETIC_' + mode
    pre = training.training_contract(mode, run)
    order = dict(schema='U40-private-epoch-order-v1', seed=training.SEED, images=741, epochs=50,
                 updates=37050, rule=training.ORDER_RULE, python_implementation='CPython',
                 python_version='3.8.17', python_major_minor=[3, 8],
                 generator_initial_state=[3, [0] * 625, None], generator_final_state=[3, [0] * 625, None],
                 order=list(range(741)) * 50)
    order['order_sha256'] = training.metadata_sha256(order)
    sources = {role: ref('sources/' + role + '.py', sha) for role, sha in training.SOURCE_PINS.items()}
    parent = dict(qualified_ref=ref('prior/ROOT.json'), pair_commit_ref=ref('prior/PAIR.json'),
                  checkpoint_refs={arm: ref('prior/' + arm + '.pth') for arm in training.ARMS},
                  trace_refs={arm: ref('prior/' + arm + '.trace.jsonl') for arm in training.ARMS})
    context = dict(manifest_ref=ref('MANIFEST.json'), order_ref=ref('ORDER.json'),
                   order_generation_ref=ref('ORDER_GENERATION.json'), U54_validation_ref=ref('U54_ROOT.json'),
                   U54_validation_sources=dict(worker_report_ref=ref('U54_WORKER.json'), owner_receipt_ref=ref('U54_OWNER.json')),
                   parent_pair=parent)
    spec = dict(schema=entry.SCHEMA, operation=entry.OPERATION, phase='SYNTHETIC_FINAL',
                entry_source=ref('sources/entry.py', 'cd62a640e63210bcb63e58e09bb1e429f1a68ea66dcfec240831a5f1db31aca9'),
                body_source=ref('sources/body.py', entry.BODY_SHA256, entry.BODY_BYTES),
                owner_source=ref('sources/owner.py'), source_refs=sources,
                library_sources={name: ref('libraries/' + name + '.py') for name in ('psutil', 'torch', 'numpy')},
                context=context, package_refs=[ref('data/' + str(i) + '.pt') for i in range(741)],
                segment=dict(start=start, stop=stop), prelock=pre,
                output_directory='/synthetic/current', owner_output_directory='/synthetic/owner')
    all_sources = [spec[k] for k in ('entry_source', 'body_source', 'owner_source')]
    all_sources += list(sources.values()) + list(spec['library_sources'].values()) + spec['package_refs']
    all_sources += [context[k] for k in ('manifest_ref', 'order_ref', 'order_generation_ref', 'U54_validation_ref')]
    all_sources += list(context['U54_validation_sources'].values()) + entry.parent_sources(parent)
    spec['sources'] = entry.canonical_sources(all_sources)
    identity = dict(mode=mode, run_id=run, recipe=deepcopy(training.RECIPE), environment=deepcopy(training.ENVIRONMENT),
                    backend_policy=deepcopy(pre['backend_policy']),
                    sources=dict(body_source=spec['body_source'], component_sources=sources),
                    context={k: v for k, v in context.items() if k != 'parent_pair'},
                    order_sha256=order['order_sha256'],
                    execution_order_sha256=training.metadata_sha256(order['order'] if mode == 'FORMAL' else [54, 710]))
    checkpoint_refs = {arm: ref('current/SEGMENT/' + arm + '.pth') for arm in training.ARMS}
    trace_refs = {arm: ref('current/SEGMENT/' + arm + '.trace.jsonl') for arm in training.ARMS}
    pair = dict(schema='U57-DGX-paired-checkpoint-commit-v1', status='PAIRED_CHECKPOINTS_WRITTEN_PENDING_ROOT_ACCEPTANCE',
                identity=identity, cursor=training.cursor(stop), span=deepcopy(spec['segment']), order_ref=context['order_ref'],
                previous_pair=parent['pair_commit_ref'], checkpoints={arm: dict(reference=checkpoint_refs[arm],
                CPU_state_signature_sha256='f' * 64, serialization_roundtrip_verified=True) for arm in training.ARMS},
                traces=trace_refs, checkpoint_refs=checkpoint_refs, trace_refs=trace_refs, actual_trace_sha256='d' * 64)
    refs = dict(qualified=ref('ROOT_ACCEPTED.json'), spec=ref('SPEC.json'),
                pair_commit=ref('current/SEGMENT/PAIR_COMMIT.json'), order=context['order_ref'])
    terminal = dict(root_observed_tool_exit_code=0, SSH_returncode=0, worker_returncode=0, reaped=True,
                    owner_original_birth_absent=True, child_original_birth_absent=True,
                    remaining_owned_group=[], all_owned_processes_absent=True)
    q = dict(schema='Root-U57-DGX-original-update-training-accepted-v1', root_accepted=True,
             paired_prefix_advanced=True, mode=mode, run_id=run,
             result=dict(pair_commit_ref=refs['pair_commit'], checkpoint_refs=checkpoint_refs,
                         trace_refs=trace_refs, order_ref=refs['order'], paired_checkpoint_prefix=training.cursor(stop)),
             accepted_span=deepcopy(spec['segment']), original_terminal=terminal,
             evidence_refs=dict(spec_ref=refs['spec'], worker_report_ref=ref('WORKER.json'), owner_receipt_ref=ref('OWNER.json')),
             AP=False, DEV=False, TEST=False)
    return deepcopy(dict(refs=refs, qualified=q, spec=spec, pair_commit=pair, order=order))


class TensorStub:
    def __init__(self, value, dtype='float32', device='cpu'):
        self.value, self.dtype, self.shape = value, dtype, (1,)
        self.device = SimpleNamespace(type=device)
        self.grad, self.requires_grad, self.finite = None, True, True

    def is_floating_point(self):
        return self.dtype in ('float32', 'float64')

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return deepcopy(self)


class ModelStub:
    def __init__(self):
        self.state = {name: TensorStub(value) for name, value in
                      (('backbone.weight', 1), ('readout.weight', 2), ('inactive.weight', 3))}
        self.state['BN.num_batches_tracked'] = TensorStub(4, 'int64')
        self.nodes = {name: SimpleNamespace(training=True) for name in ('', 'backbone', 'readout', 'inactive', 'BN')}
        self.strict_loads = 0

    def state_dict(self):
        return self.state

    def named_parameters(self):
        return [(k, v) for k, v in self.state.items() if k.endswith('.weight')]

    def parameters(self):
        return [v for _, v in self.named_parameters()]

    def named_modules(self):
        return list(self.nodes.items())

    def modules(self):
        return list(self.nodes.values())

    def load_state_dict(self, state, strict=False):
        if not strict:
            raise AssertionError('Strict model restoration required')
        self.strict_loads += 1
        self.state = deepcopy(state)

    def eval(self):
        for node in self.nodes.values():
            node.training = False
        return self

    def requires_grad_(self, enabled):
        for value in self.parameters():
            value.requires_grad = enabled
        return self


torch_stub = SimpleNamespace(Tensor=TensorStub, nn=SimpleNamespace(Module=ModelStub), float32='float32',
                             cuda=SimpleNamespace(is_initialized=lambda: False),
                             isfinite=lambda value: SimpleNamespace(all=lambda: value.finite),
                             equal=lambda a, b: (a.value, a.dtype, a.shape) == (b.value, b.dtype, b.shape))
helpers_stub = SimpleNamespace(_digest=lambda tensor: training.metadata_sha256(
    dict(value=tensor.value, dtype=tensor.dtype, shape=list(tensor.shape))))
ops_stub = SimpleNamespace(_partition=lambda model: ({'backbone.weight', 'readout.weight'}, {'inactive.weight'}, dict(synthetic=True)))


def checkpoint_fixture(mode='FORMAL'):
    metadata, model = fixture(mode, 37050 if mode == 'FORMAL' else 2), ModelStub()
    pair = metadata['pair_commit']
    receipt = dict(seed=training.SEED, adapter_seed=training.SEED,
                   common_state={str(i): 'synthetic' for i in range(104)}, common_parameter_tensors=68, common_BN_buffers=36)
    saved = dict(schema='U57-DGX-original-update-arm-state-v1', arm='FULL', identity=deepcopy(pair['identity']),
                 cursor=deepcopy(pair['cursor']), model=deepcopy(model.state_dict()),
                 optimizer=dict(state={0: dict(step=TensorStub(37050))}, param_groups=[dict(params=[0])]),
                 effective_parameter_names=['backbone.weight', 'readout.weight'], inactive_parameter_names=['inactive.weight'],
                 optimizer_state_devices=dict(synthetic='cuda:0'), module_training_modes={k: True for k in model.nodes},
                 rng=dict(python=(3, (0,) * 625, None), numpy=dict(name='MT19937', keys=[0] * 624,
                     position=0, has_gauss=0, cached_gaussian=0.0), torch_CPU=TensorStub(0, 'uint8'),
                     torch_CUDA=TensorStub(1, 'uint8')), order_ref=pair['order_ref'],
                 requested_trace_sha256=training.expected_trace(
                     metadata['order'] if mode == 'FORMAL' else dict(order=[54, 710]), pair['cursor']['global_updates']),
                 actual_trace_sha256=pair['actual_trace_sha256'],
                 initialization_receipt=receipt, previous_checkpoint=metadata['spec']['context']['parent_pair']['checkpoint_refs']['FULL'],
                 previous_pair=pair['previous_pair'])
    pair['checkpoints']['FULL']['CPU_state_signature_sha256'] = training.metadata_sha256(training._signature(torch_stub, helpers_stub, saved))
    return metadata, model, saved, receipt


class EndpointTests(unittest.TestCase):
    def test_production_wrapper_loads_one_cpu_container_and_reports_no_qualification(self):
        self.check_stubbed_wrapper('FORMAL')

    def test_separate_smoke_wrapper_loads_smoke_container_without_formal_credit(self):
        self.check_stubbed_wrapper('SMOKE')

    def check_stubbed_wrapper(self, mode):
        restore = module(HERE / 'restore_state.py', '_U58_RESTORE')
        metadata, model, saved, receipt = checkpoint_fixture(mode)
        endpoint = (restore.validate_final_endpoint if mode == 'FORMAL' else restore.validate_smoke_endpoint)(training, entry, metadata)
        sources = {role: endpoint['restore_source_refs'][role] for role in restore.RESTORE_PINS}
        modules = {role: SimpleNamespace() for role in sources}
        modules.update(training_state=training, training_entry=entry, u36_helpers=helpers_stub, u39_training_ops=ops_stub)
        token = object()
        modules['full_legacy'] = SimpleNamespace(DSConv=token, CoordConv=token)
        modules['full_dsconv'] = SimpleNamespace(DSConv=token)
        modules['full_head'] = SimpleNamespace(CoordConv=token)
        actual_torch = ModuleType('torch')
        actual_torch.__dict__.update(vars(torch_stub), __version__='2.10.0+cu130',
                                     get_default_dtype=lambda: 'float32', get_rng_state=lambda: TensorStub(999, 'int64'))
        loads = []
        def load(path, **kwargs):
            loads.append((path, kwargs))
            return deepcopy(saved)
        actual_torch.load = load
        actual_numpy = ModuleType('numpy')
        actual_numpy.__version__ = '1.26.4'
        actual_numpy.random = SimpleNamespace(get_state=lambda: ('MT19937', SimpleNamespace(tolist=lambda: [1, 2]), 0, 0, 0.0))
        control = SimpleNamespace(check=lambda: None)
        report = {}
        # Source and byte admission are deliberately stubbed here; this test
        # proves orchestration only, never real files, modules or Torch.
        with patch.object(restore, '_module'), patch.object(restore, '_closed', side_effect=lambda c, value: value['path']), \
             patch.object(restore, '_json', side_effect=lambda c, value: next(metadata[k] for k, r in metadata['refs'].items() if r == value)), \
             patch.object(restore.platform, 'python_version', return_value='3.12.14'), \
             patch.object(restore.platform, 'system', return_value='Linux'), patch.object(restore.platform, 'machine', return_value='aarch64'):
            target = restore.restore_arm if mode == 'FORMAL' else restore.restore_smoke_arm
            result = target(actual_torch, actual_numpy, control, model, 'FULL', metadata, receipt,
                            modules=modules, source_refs=sources, report=report)
        self.assertIs(result, report)
        self.assertEqual(loads, [(endpoint['checkpoint_refs']['FULL']['path'], dict(map_location='cpu', weights_only=True))])
        self.assertEqual(report['status'], 'RESTORED_CPU_ONLY')
        self.assertFalse(report['qualified'])
        self.assertFalse(report['AP'])
        self.assertTrue(report['CPU_checkpoint_payload_released'])
        self.assertTrue(report['external_CPU_RNG_unchanged'])
        self.assertTrue(report['external_NumPy_RNG_unchanged'])
        self.assertTrue(report['external_Python_RNG_unchanged'])
        self.assertFalse(report['optimizer_created'])
        self.assertFalse(report['optimizer_restored'])
        self.assertEqual(report['formal_endpoint'], mode == 'FORMAL')
        self.assertEqual(report['cursor'], training.cursor(37050 if mode == 'FORMAL' else 2))

    def test_restore_sources_are_exact_original_constructor_and_validation_roles(self):
        restore = module(HERE / 'restore_state.py', '_U58_RESTORE')
        endpoint = restore.validate_final_endpoint(training, entry, fixture())
        sources = {role: endpoint['restore_source_refs'][role] for role in restore.RESTORE_PINS}
        self.assertEqual(len(restore.CONSTRUCTOR_ROLES), 8)
        self.assertEqual(restore.validate_restore_sources(endpoint, sources), sources)
        for bad in (dict(sources, portable_reader=ref('reader.py')), {k: v for k, v in sources.items() if k != 'instance_adapter'}):
            with self.assertRaises(ValueError):
                restore.validate_restore_sources(endpoint, bad)
        for role in sources:
            bad = deepcopy(sources)
            bad[role] = ref('substituted/' + role, bad[role]['sha256'])
            with self.subTest(role=role), self.assertRaises(ValueError):
                restore.validate_restore_sources(endpoint, bad)

    def test_loaded_model_strict_restore_preserves_trained_head_and_freezes_cpu_eval(self):
        restore = module(HERE / 'restore_state.py', '_U58_RESTORE')
        metadata, model, saved, receipt = checkpoint_fixture()
        saved['model']['readout.weight'].value = 921
        metadata['pair_commit']['checkpoints']['FULL']['CPU_state_signature_sha256'] = training.metadata_sha256(
            training._signature(torch_stub, helpers_stub, saved))
        model_identity = id(model)
        proof = restore.restore_loaded_model(torch_stub, training, entry, helpers_stub, ops_stub,
                                             model, 'FULL', metadata, saved, receipt)
        self.assertEqual(id(model), model_identity)
        self.assertEqual(model.state['readout.weight'].value, 921)
        self.assertEqual(model.strict_loads, 1)
        self.assertTrue(all(not value.training for value in model.modules()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        self.assertTrue(proof['restored_model_state_exact'])
        self.assertFalse(proof['numerical_runtime_qualified'])
        self.assertFalse(proof['optimizer_restored'])

    def test_invalid_checkpoint_or_warm_cuda_is_rejected_before_model_write(self):
        restore = module(HERE / 'restore_state.py', '_U58_RESTORE')
        for label in ('signature', 'CUDA', 'retained grad', 'current GPU state'):
            metadata, model, saved, receipt = checkpoint_fixture()
            fake_torch = SimpleNamespace(**vars(torch_stub))
            if label == 'signature':
                metadata['pair_commit']['checkpoints']['FULL']['CPU_state_signature_sha256'] = '0' * 64
            elif label == 'CUDA':
                fake_torch.cuda = SimpleNamespace(is_initialized=lambda: True)
            elif label == 'retained grad':
                model.state['backbone.weight'].grad = TensorStub(1)
            else:
                model.state['BN.num_batches_tracked'].device.type = 'cuda'
            with self.subTest(label=label), self.assertRaises(ValueError):
                restore.restore_loaded_model(fake_torch, training, entry, helpers_stub, ops_stub,
                                             model, 'FULL', metadata, saved, receipt)
            self.assertEqual(model.strict_loads, 0)

    def test_smoke_is_a_separate_fixed_two_update_technical_endpoint(self):
        restore = module(HERE / 'restore_state.py', '_U58_RESTORE')
        metadata = fixture('SMOKE', 2)
        result = restore.validate_smoke_endpoint(training, entry, metadata)
        self.assertEqual(result['cursor'], training.cursor(2))
        self.assertEqual(result['mode'], 'SMOKE')
        self.assertFalse(result['formal_endpoint'])
        for bad in (fixture('FORMAL', 2), fixture('FORMAL', 37050), fixture('SMOKE', 1)):
            with self.assertRaises(ValueError):
                restore.validate_smoke_endpoint(training, entry, bad)

    def test_final_endpoint_accepts_actual_schema_without_fixing_the_last_segment_start(self):
        restore = module(HERE / 'restore_state.py', '_U58_RESTORE')
        metadata = fixture()
        before = deepcopy(metadata)
        result = restore.validate_final_endpoint(training, entry, metadata)
        self.assertEqual(metadata, before)
        self.assertEqual(result['cursor'], training.cursor(37050))
        self.assertEqual(result['checkpoint_refs'], metadata['pair_commit']['checkpoint_refs'])
        self.assertEqual(result['training_qualified'], metadata['refs']['qualified'])

    def test_endpoint_rejects_incomplete_training_and_cross_document_or_source_substitution(self):
        restore = module(HERE / 'restore_state.py', '_U58_RESTORE')
        for bad in (fixture('FORMAL', 37049), fixture('SMOKE', 2)):
            with self.assertRaises(ValueError):
                restore.validate_final_endpoint(training, entry, bad)

        mutations = [
            ('not accepted', lambda m: m['qualified'].update(root_accepted=False)),
            ('run mismatch', lambda m: m['qualified'].update(run_id='OTHER')),
            ('wrong spec', lambda m: m['qualified']['evidence_refs'].update(spec_ref=ref('OTHER_SPEC.json'))),
            ('wrong arm refs', lambda m: m['qualified']['result'].update(checkpoint_refs={a: ref('OTHER_' + a) for a in training.ARMS})),
            ('wrong span', lambda m: m['pair_commit'].update(span=dict(start=37018, stop=37050))),
            ('wrong CP record', lambda m: m['pair_commit']['checkpoints']['FULL'].update(reference=ref('OTHER_FULL.pth'))),
            ('wrong signature', lambda m: m['pair_commit']['checkpoints']['FULL'].update(CPU_state_signature_sha256='short')),
            ('wrong order', lambda m: m['pair_commit'].update(order_ref=ref('OTHER_ORDER.json'))),
            ('wrong runtime', lambda m: m['pair_commit']['identity']['environment'].update(torch='2.0.1+cu118')),
            ('wrong body', lambda m: m['pair_commit']['identity']['sources'].update(body_source=ref('OTHER_BODY.py'))),
            ('not actual0', lambda m: m['qualified']['original_terminal'].update(SSH_returncode=1)),
            ('AP claim', lambda m: m['qualified'].update(AP=True)),
            ('noninteger cursor', lambda m: m['pair_commit']['cursor'].update(completed_epochs=50.0)),
            ('Windows path', lambda m: m['refs']['qualified'].update(path='E:/ROOT.json')),
            ('POSIX backslash', lambda m: m['refs']['qualified'].update(path='/home/false\\ROOT.json')),
        ]
        for label, mutate in mutations:
            bad = fixture()
            mutate(bad)
            with self.subTest(label=label), self.assertRaises(ValueError):
                restore.validate_final_endpoint(training, entry, bad)

    def test_complete_checkpoint_signature_cannot_be_replaced_by_the_model_signature(self):
        restore = module(HERE / 'restore_state.py', '_U58_RESTORE')
        metadata, model, saved, receipt = checkpoint_fixture()
        result = restore.validate_loaded_checkpoint(torch_stub, training, entry, helpers_stub, ops_stub,
                                                     model, 'FULL', metadata, saved, receipt)
        complete = metadata['pair_commit']['checkpoints']['FULL']['CPU_state_signature_sha256']
        self.assertEqual(result['complete_checkpoint_CPU_state_signature_sha256'], complete)
        self.assertNotEqual(result['model_CPU_state_signature_sha256'], complete)
        metadata['pair_commit']['checkpoints']['FULL']['CPU_state_signature_sha256'] = result['model_CPU_state_signature_sha256']
        with self.assertRaises(ValueError):
            restore.validate_loaded_checkpoint(torch_stub, training, entry, helpers_stub, ops_stub,
                                                model, 'FULL', metadata, saved, receipt)

    def test_signed_container_still_requires_exact_finite_model_names_tree_and_lineage(self):
        restore = module(HERE / 'restore_state.py', '_U58_RESTORE')
        mutations = [
            ('old schema', lambda s: s.update(schema='U40-continuous-arm-state-v1')),
            ('wrong arm', lambda s: s.update(arm='PLAIN')),
            ('early cursor', lambda s: s.update(cursor=training.cursor(37049))),
            ('wrong identity', lambda s: s['identity'].update(run_id='OTHER')),
            ('wrong order', lambda s: s.update(order_ref=ref('OTHER_ORDER.json'))),
            ('wrong parent', lambda s: s.update(previous_checkpoint=ref('OTHER_PARENT.pth'))),
            ('wrong effective names', lambda s: s.update(effective_parameter_names=['readout.weight'])),
            ('missing inactive names', lambda s: s.update(inactive_parameter_names=[])),
            ('missing head', lambda s: s['model'].pop('readout.weight')),
            ('wrong dtype', lambda s: setattr(s['model']['readout.weight'], 'dtype', 'float64')),
            ('wrong shape', lambda s: setattr(s['model']['readout.weight'], 'shape', (2,))),
            ('nonfinite', lambda s: setattr(s['model']['readout.weight'], 'finite', False)),
            ('wrong module tree', lambda s: s['module_training_modes'].pop('BN')),
            ('invalid module mode', lambda s: s['module_training_modes'].update(BN=1)),
            ('wrong init receipt', lambda s: s.update(initialization_receipt=dict(other=True))),
            ('missing optimizer container', lambda s: s.pop('optimizer')),
            ('model-only optimizer', lambda s: s.update(optimizer={})),
            ('missing RNG payload', lambda s: s.update(rng={})),
        ]
        for label, mutate in mutations:
            metadata, model, saved, receipt = checkpoint_fixture()
            mutate(saved)
            metadata['pair_commit']['checkpoints']['FULL']['CPU_state_signature_sha256'] = training.metadata_sha256(
                training._signature(torch_stub, helpers_stub, saved))
            with self.subTest(label=label), self.assertRaises(ValueError):
                restore.validate_loaded_checkpoint(torch_stub, training, entry, helpers_stub, ops_stub,
                                                    model, 'FULL', metadata, saved, receipt)



if __name__ == '__main__':
    unittest.main()
