"""UNBOUND real-full size64 CPU feature/adapter contract; not executed.

The original source-closed convolutional full modules run on synthetic inputs in
eval mode, with autograd only in the extra adapter test. No old weights, real
images/GT, optimizer, renderer, AP or size400/train-mode qualification occurs.
"""

import gc
import hashlib
import io
from pathlib import Path
import sys
import time
import traceback
import types
import unittest


ROLES = ('entry_source', 'body_source', 'runtime_source', 'legacy_source', 'dsconv_source',
         'head_source', 'feature_source', 'test_source', 'adapter_source')
HEAD_NAMES = ('final_super_0_1', 'final_super_0_2', 'final_super_0_3', 'final_super_0_4')
EXPECTED_REAL_ID = ('U30_FULL_FEATURE_PUBLIC_CONTRACT.make_real_legacy_suite.<locals>.'
    'RealLegacySize64Contract.test_feature_forward_matches_real_legacy_with_temporary_identity_semantic_head')
EXPECTED_BACKWARD_ID = ('U30_FULL_FEATURE_CPU_PROBE_BODY.run.<locals>.RealFullAdapterBackwardContract.'
    'test_size64_real_feature_adapter_finite_backward_and_unused_heads_unchanged')


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def load(c, reference, name):
    path = Path(c.closed(reference)); raw = path.read_bytes()
    need(len(raw) == reference['bytes'] and hashlib.sha256(raw).hexdigest() == reference['sha256'],
         'Exact source-closed real component')
    module = types.ModuleType(name); module.__file__ = str(path); sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def run(torch, spec, c, pre):
    started = time.monotonic(); c.check()
    need(not torch.cuda.is_initialized() and torch.get_num_threads() == torch.get_num_interop_threads() == 2,
         'Cold CUDA and two actual CPU threads')
    need(pre['expected_real_legacy_tests'] == pre['expected_additional_tests'] == 1
         and pre['input_size'] == 64 and pre['batch_size'] == 1 and pre['query_count'] == 2
         and pre['mode'] == 'eval', 'Exactly the locked size64 eval contracts')
    output = Path(spec['output_directory']); output.mkdir()
    original_rng = torch.get_rng_state().clone(); original_lazy_init = torch.cuda._lazy_init
    checkpoints = []; attempts = []; checks = {}; started_tests = []; successful_tests = []
    result = None; suite = None; fatal = None; public_loaded = False
    text = io.StringIO()
    # Literal names are the original dependency imports, not replacement source.
    # These names are restored even when an import or test fails.
    aliases = ('dsconv', 'head', 'model_full', 'feature_backbone',
               'U30_FULL_FEATURE_PUBLIC_CONTRACT', 'full_instance_adapter')
    missing = object(); previous = {name:sys.modules.get(name, missing) for name in aliases}
    loaded_modules = {}; import_records = []; aliases_restored = False

    class Tee:
        def write(self, value):
            sys.stdout.write(value); text.write(value)
            return len(value)
        def flush(self):
            sys.stdout.flush()

    def CPU_checkpoint(label):
        initialized = bool(torch.cuda.is_initialized())
        checkpoints.append(dict(label=label, CUDA_initialized=initialized))
        need(not initialized, 'CUDA must remain uninitialized throughout the real CPU contracts')
        c.check()

    def forbidden_cuda_init(*args, **kwargs):
        attempts.append('torch.cuda._lazy_init attempted')
        raise RuntimeError('CUDA initialization is forbidden in this owned CPU contract')

    class ObservedResult(unittest.TextTestResult):
        def startTest(self, test):
            CPU_checkpoint('before:'+test.id()); started_tests.append(test.id())
            super().startTest(test)
        def addSuccess(self, test):
            successful_tests.append(test.id()); super().addSuccess(test)
        def stopTest(self, test):
            CPU_checkpoint('after:'+test.id()); super().stopTest(test)

    class RealFullAdapterBackwardContract(unittest.TestCase):
        def test_size64_real_feature_adapter_finite_backward_and_unused_heads_unchanged(self):
            torch.manual_seed(202610173202)
            legacy = legacy_module.UnetPlusPlus(num_classes=1, deep_supervision=False,
                use_x2_transformer=False, input_size=64, instantiate_x2_transformer=False,
                include_legacy_x4_transformer=False)
            legacy.eval()
            wrapper = feature_module.make_feature_backbone(torch, legacy)
            adapter = adapter_module.make_adapter(torch, wrapper, 'legacy.CONV0_4', 64,
                                                   query_channels=256, hidden=32)
            adapter.eval()
            self.assertIs(wrapper.legacy, legacy)
            self.assertFalse(any(module.training for module in adapter.modules()))
            self.assertIsNone(legacy.encoder_x2); self.assertFalse(hasattr(legacy, 'encoder_x4'))
            # A disposable nonzero readout allows gradients through the actual
            # source-closed original modules; original backbone weights stay at
            # their ordinary random initialization and no optimizer is created.
            with torch.no_grad():
                adapter.head[-1].weight.fill_(.05); adapter.head[-1].bias.fill_(.01)
            params = dict(adapter.named_parameters())
            inactive_wrapper = wrapper.inactive_parameter_names()
            self.assertIsInstance(inactive_wrapper, tuple)
            self.assertEqual(len(inactive_wrapper), len(set(inactive_wrapper)))
            self.assertTrue(inactive_wrapper)
            self.assertTrue(all(type(name) is str and name.startswith('legacy.')
                                for name in inactive_wrapper))
            inactive = {'backbone.'+name for name in inactive_wrapper}
            self.assertTrue(inactive <= set(params))
            effective = {name for name, parameter in params.items()
                         if parameter.requires_grad and name not in inactive}
            self.assertTrue(effective)
            manifest = [dict(name=name, shape=list(parameter.shape), dtype=str(parameter.dtype),
                numel=parameter.numel(), requires_grad=parameter.requires_grad,
                inactive_prelocked=name in inactive) for name, parameter in params.items()]
            modes_before = {name:module.training for name, module in adapter.named_modules()}
            flags_before = {name:parameter.requires_grad for name, parameter in params.items()}
            head_calls = {name:0 for name in HEAD_NAMES}; counts = {'wrapper':0, 'feature_exit':0, 'legacy_forward':0}
            handles = []; head_bn_before = {}; head_bn_modules = []
            head_bn_buffer_counts = {name:0 for name in HEAD_NAMES}
            def increment(key):
                def hook(module, arguments, value):
                    counts[key] += 1
                return hook
            def head_increment(key):
                def hook(module, arguments, value):
                    head_calls[key] += 1
                return hook
            for head_name in HEAD_NAMES:
                head = getattr(legacy, head_name)
                for local_name, module in head.named_modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        prefix = head_name+('.'+local_name if local_name else '')
                        head_bn_modules.append(prefix)
                        for buffer_name, buffer in module.named_buffers(recurse=False):
                            if buffer is not None:
                                head_bn_before[prefix+'.'+buffer_name] = buffer.detach().clone()
                                head_bn_buffer_counts[head_name] += 1
            self.assertTrue(head_bn_before, 'The real head BN-buffer comparison must be nonvacuous')
            all_legacy_buffers = dict(legacy.named_buffers())
            self.assertTrue(set(head_bn_before) <= set(all_legacy_buffers))
            for head_name in HEAD_NAMES:
                handles.append(getattr(legacy, head_name).register_forward_hook(head_increment(head_name)))
            handles.extend([wrapper.register_forward_hook(increment('wrapper')),
                legacy.CONV0_4.register_forward_hook(increment('feature_exit')),
                legacy.register_forward_hook(increment('legacy_forward'))])
            try:
                image = torch.randn(1, 3, 64, 64, requires_grad=True)
                embeddings = torch.randn(2, 256, requires_grad=True)
                coarse = torch.randn(2, 200, 200, requires_grad=True)
                value = adapter(image, embeddings, coarse, query_batch_size=1)
                self.assertEqual(tuple(value.shape), (2, 1, 64, 64))
                self.assertTrue(bool(torch.isfinite(value).all()))
                self.assertGreater(float(value.abs().sum().item()), 0.)
                loss = value.square().mean(); self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward(); CPU_checkpoint('after_real_full_backward')
                missing_gradients = []; nonfinite_gradients = []; zero_gradients = []; gradients = []
                for name in sorted(effective):
                    grad = params[name].grad
                    if grad is None:
                        missing_gradients.append(name)
                        gradients.append(dict(name=name, missing=True))
                        continue
                    is_finite = bool(torch.isfinite(grad).all())
                    is_zero = bool(torch.equal(grad, torch.zeros_like(grad)))
                    if not is_finite:
                        nonfinite_gradients.append(name)
                    if is_zero:
                        zero_gradients.append(name)
                    gradients.append(dict(name=name, missing=False, shape=list(grad.shape),
                        dtype=str(grad.dtype), finite=is_finite, exact_zero=is_zero,
                        absolute_sum=float(grad.detach().abs().sum().item()) if is_finite else None))
                unexpected_inactive = sorted(name for name in inactive if params[name].grad is not None)
                buffer_checks = []
                current_buffers = dict(legacy.named_buffers())
                for name, before in sorted(head_bn_before.items()):
                    current = current_buffers[name]
                    unchanged = (tuple(current.shape) == tuple(before.shape)
                                 and current.dtype == before.dtype and torch.equal(current, before))
                    buffer_checks.append(dict(name=name, shape=list(before.shape), dtype=str(before.dtype),
                                               bitexact_unchanged=bool(unchanged)))
                input_gradients = {}
                for name, tensor in [('image', image), ('embeddings', embeddings), ('coarse', coarse)]:
                    grad = tensor.grad
                    input_gradients[name] = dict(missing=grad is None,
                        finite=grad is not None and bool(torch.isfinite(grad).all()),
                        exact_zero=grad is not None and bool(torch.equal(grad, torch.zeros_like(grad))))
                checks['real_full_adapter_backward'] = dict(input_size=64, batch_size=1, query_count=2,
                    mode='eval', output_shape=list(value.shape), output_finite=True,
                    nonzero_readout_weight=True, real_loss=float(loss.item()), backward_calls=1,
                    allocated_parameter_count=sum(parameter.numel() for parameter in params.values()),
                    allocated_parameter_tensors=len(params),
                    requires_grad_parameter_count=sum(parameter.numel() for parameter in params.values() if parameter.requires_grad),
                    requires_grad_parameter_tensors=sum(1 for parameter in params.values() if parameter.requires_grad),
                    nontrainable_parameter_names=sorted(name for name, parameter in params.items() if not parameter.requires_grad),
                    nontrainable_parameter_count=sum(parameter.numel() for parameter in params.values() if not parameter.requires_grad),
                    effective_parameter_count=sum(params[name].numel() for name in effective),
                    effective_parameter_tensors=len(effective),
                    inactive_parameter_count=sum(params[name].numel() for name in inactive),
                    inactive_parameter_tensors=len(inactive), inactive_parameter_names=sorted(inactive),
                    effective_parameter_names=sorted(effective), parameter_manifest=manifest,
                    gradient_missing_names=missing_gradients, gradient_nonfinite_names=nonfinite_gradients,
                    gradient_zero_names=zero_gradients, effective_gradient_checks=gradients,
                    zero_gradients_allowed=True, unexpected_inactive_gradient_names=unexpected_inactive,
                    input_gradient_checks=input_gradients, actual_call_counts=dict(counts),
                    semantic_head_call_counts=dict(head_calls), semantic_head_BN_modules=head_bn_modules,
                    semantic_head_BN_buffer_counts_by_head=head_bn_buffer_counts,
                    semantic_head_BN_buffer_count=len(head_bn_before), semantic_head_BN_buffer_checks=buffer_checks,
                    semantic_head_BN_all_bitexact_unchanged=all(row['bitexact_unchanged'] for row in buffer_checks),
                    semantic_heads_executed=any(head_calls.values()), optimizer_created=False, optimizer_steps=0)
                self.assertEqual(missing_gradients, []); self.assertEqual(nonfinite_gradients, [])
                self.assertEqual(unexpected_inactive, [])
                self.assertTrue(all(row['finite'] for row in input_gradients.values()))
                self.assertEqual(counts, dict(wrapper=1, feature_exit=1, legacy_forward=0))
                self.assertTrue(all(count == 0 for count in head_calls.values()))
                self.assertTrue(all(row['bitexact_unchanged'] for row in buffer_checks))
                self.assertEqual({name:module.training for name, module in adapter.named_modules()}, modes_before)
                self.assertEqual({name:parameter.requires_grad for name, parameter in adapter.named_parameters()}, flags_before)
            finally:
                for handle in handles:
                    handle.remove()

    try:
        torch.cuda._lazy_init = forbidden_cuda_init
        CPU_checkpoint('before_source_imports')
        for role, name in [('dsconv_source', 'dsconv'), ('head_source', 'head'),
                           ('legacy_source', 'model_full'), ('feature_source', 'feature_backbone'),
                           ('test_source', 'U30_FULL_FEATURE_PUBLIC_CONTRACT'),
                           ('adapter_source', 'full_instance_adapter')]:
            loaded_modules[name] = load(c, spec[role], name)
            import_records.append(dict(role=role, module_name=name, source=spec[role]))
            CPU_checkpoint('after_import:'+name)
        legacy_module = loaded_modules['model_full']; feature_module = loaded_modules['feature_backbone']
        public = loaded_modules['U30_FULL_FEATURE_PUBLIC_CONTRACT']; adapter_module = loaded_modules['full_instance_adapter']
        need(public.factory_module is feature_module, 'Public real-legacy test uses the exact closed feature factory')
        need(legacy_module.UnetPlusPlus.__module__ == 'model_full'
             and legacy_module.UnetPlusPlus.__name__ == 'UnetPlusPlus', 'Original actual legacy class')
        need(legacy_module.DSConv is loaded_modules['dsconv'].DSConv
             and legacy_module.CoordConv is loaded_modules['head'].CoordConv,
             'Original legacy DSConv and CoordConv imports resolve to the exact closed modules')
        public_suite = public.make_real_legacy_suite(torch, legacy_module.UnetPlusPlus)
        need(public_suite.countTestCases() == 1, 'Execute the original complete real-legacy suite unchanged')
        public_loaded = True
        suite = unittest.TestSuite([public_suite,
            unittest.defaultTestLoader.loadTestsFromTestCase(RealFullAdapterBackwardContract)])
        need(suite.countTestCases() == 2, 'Exactly one real-legacy equality and one real adapter backward test')
        with torch.random.fork_rng(devices=[]):
            result = unittest.TextTestRunner(stream=Tee(), verbosity=2, resultclass=ObservedResult).run(suite)
        CPU_checkpoint('after_all_actual_tests')
    except Exception:
        fatal = traceback.format_exc(); print(fatal, file=sys.stderr, flush=True)
    finally:
        torch.cuda._lazy_init = original_lazy_init
        torch.set_rng_state(original_rng)
        for name in aliases:
            if previous[name] is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous[name]
        aliases_restored = all((name not in sys.modules if previous[name] is missing
                                else sys.modules.get(name) is previous[name]) for name in aliases)
        suite = None; loaded_modules.clear(); gc.collect()
    restored = bool(torch.equal(torch.get_rng_state(), original_rng))
    CPU_checkpoint('before_report')
    source_refs = {role:spec[role] for role in ROLES}
    for value in source_refs.values():
        need(c.ref(value['path']) == value, 'All actual size64 sources unchanged')
    errors = [] if result is None else [dict(test=str(test), traceback=trace) for test, trace in result.errors]
    failures = [] if result is None else [dict(test=str(test), traceback=trace) for test, trace in result.failures]
    skipped = [] if result is None else [dict(test=str(test), reason=reason) for test, reason in result.skipped]
    tests_run = 0 if result is None else result.testsRun
    real_ids = [name for name in successful_tests if name.startswith('U30_FULL_FEATURE_PUBLIC_CONTRACT.')]
    backward_ids = [name for name in successful_tests if name.startswith(__name__+'.run.<locals>.RealFullAdapterBackwardContract.')]
    expected_ids = [EXPECTED_REAL_ID, EXPECTED_BACKWARD_ID]
    passed = (fatal is None and result is not None and result.wasSuccessful() and tests_run == 2
              and len(started_tests) == len(successful_tests) == 2 and len(real_ids) == len(backward_ids) == 1
              and started_tests == successful_tests == expected_ids
              and not skipped and not result.expectedFailures and not result.unexpectedSuccesses
              and set(checks) == {'real_full_adapter_backward'} and not attempts and restored and aliases_restored)
    log = output/'UNITTEST_STDOUT.txt'
    with log.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text.getvalue()+('' if fatal is None else '\nFATAL\n'+fatal))
    log_ref = c.ref(log)
    report = dict(schema='U30-full-feature-size64-contract-report-pending-v1',
        status=('ACTUAL_SIZE64_FEATURE_CHECKS_PASSED_PENDING_ROOT_ACTUAL0' if passed else 'ACTUAL_SIZE64_FEATURE_CHECKS_FAILED_PRESERVED'),
        source=spec['body_source'], source_refs=source_refs, spec=c.execution_spec_ref,
        CPU_admission=c.execution_admission_ref, phase=spec['phase'], prelock=spec['prelock'],
        real_legacy_tests=1, additional_tests=1, tests_run=tests_run, all_tests_passed=passed,
        original_real_legacy_test_source_loaded_unchanged=public_loaded,
        original_real_legacy_test_source_executed_unchanged=EXPECTED_REAL_ID in started_tests,
        expected_test_ids=expected_ids,
        actual_started_tests=started_tests, actual_successful_tests=successful_tests,
        real_legacy_equality_test_success_ids=real_ids, real_adapter_backward_test_success_ids=backward_ids,
        actual_checks=checks, errors=errors, failures=failures, skipped=skipped,
        expected_failures=[] if result is None else [str(test) for test, _ in result.expectedFailures],
        unexpected_successes=[] if result is None else [str(test) for test in result.unexpectedSuccesses],
        fatal_exception=fatal, unittest_log_ref=log_ref, actual_import_records=import_records,
        dependency_module_aliases=list(aliases), dependency_module_aliases_restored_exact=aliases_restored,
        CUDA_checkpoints=checkpoints, CUDA_lazy_initialization_blocked_during_tests=True,
        CUDA_initialization_attempts=attempts, CUDA_initialized=False, external_CPU_RNG_restored_exact=restored,
        actual_CPU_threads=torch.get_num_threads(), actual_CPU_interop_threads=torch.get_num_interop_threads(),
        actual_CPU_affinity=c.psutil.Process().cpu_affinity(),
        synthetic_inputs_only=True, original_full_feature_modules_executed=True if passed else None,
        original_semantic_head_executed=False if passed else None,
        original_weights_loaded=False, real_images_loaded=False, GT_loaded=False,
        optimizer_created=False, optimizer_steps=0, disposable_synthetic_backward_only=True,
        size400_verified=False, train_mode_BN_verified=False, actual_full_architecture_verified=False,
        full_method_training_eligible=False, new_training=False, AP=False, DEV=False, TEST=False,
        qualified=False, original_actual0_self_certified=False, innovation_established=False, goal_complete=False,
        peak_RSS_bytes=c.peak_rss, minimum_available_RAM_bytes=c.minimum_available,
        limitations=['Only one synthetic size64 image in eval mode, original full convolutional feature graph and two query readouts.',
            'The public unchanged test compares the feature wrapper bitexactly with original full forward using a temporary identity semantic head.',
            'Original Transformer allocation is explicitly disabled; no old weights are loaded and no original model bug is repaired.',
            'Effective/inactive parameter names are selected before backward from the wrapper ABI; zero finite gradients are allowed.',
            'No train-mode BN update, size400 resource, actual-image/GT/teacher/renderer, optimizer, AP or efficacy evidence.'],
        elapsed_seconds=time.monotonic()-started)
    report_ref = c.write_json(output/'FULL_FEATURE_CONTRACT_REPORT.json', report)
    del original_rng; gc.collect(); c.check()
    need(passed, 'Actual real-full size64 contract failed; original report/log preserved, no success qualification')
    return dict(report_ref=report_ref, unittest_log_ref=log_ref, real_legacy_tests=1, additional_tests=1,
        tests_run=tests_run, all_tests_passed=True, original_full_feature_modules_executed=True,
        original_semantic_head_executed=False, size400_verified=False, train_mode_BN_verified=False,
        qualified=False, new_training=False, AP=False, DEV=False, TEST=False)
