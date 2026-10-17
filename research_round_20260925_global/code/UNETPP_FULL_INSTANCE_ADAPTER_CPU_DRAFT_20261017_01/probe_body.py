"""UNBOUND synthetic CPU adapter contract body; no scientific imports at import.

The real public unittest is executed unchanged, followed by two actual tiny-nn
checks. Any failure remains in the report and original stdout and raises before
entry completion. These checks do not execute the user's full UNet++ or data.
"""
import gc
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import traceback
import types
import unittest


def need(ok,message):
    if not ok:raise RuntimeError(message)


def load(c,reference,name):
    path=Path(c.closed(reference));raw=path.read_bytes()
    need(len(raw)==reference['bytes'] and hashlib.sha256(raw).hexdigest()==reference['sha256'],'Exact owned fixture source')
    module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def run(torch,spec,c,pre):
    started=time.monotonic();c.check()
    need(not torch.cuda.is_initialized() and torch.get_num_threads()==torch.get_num_interop_threads()==2,
         'Real CPU-only two-thread synthetic probe')
    need(pre['expected_public_tests']==1 and pre['expected_additional_tests']==2,'No fixture may be silently removed')
    output=Path(spec['output_directory']);output.mkdir()
    original_rng=torch.get_rng_state().clone()
    original_lazy_init=torch.cuda._lazy_init
    checkpoints=[];attempts=[];checks={};fatal=None;result=None;suite=None
    started_tests=[];successful_tests=[];public_loaded=False
    text=io.StringIO()

    class Tee:
        def write(self,value):
            sys.stdout.write(value);text.write(value)
            return len(value)
        def flush(self):sys.stdout.flush()

    def CPU_checkpoint(label):
        initialized=bool(torch.cuda.is_initialized())
        checkpoints.append(dict(label=label,CUDA_initialized=initialized))
        need(not initialized,'CUDA must stay uninitialized throughout every synthetic test')
        c.check()

    def forbidden_cuda_init(*args,**kwargs):
        attempts.append('torch.cuda._lazy_init attempted')
        raise RuntimeError('CUDA initialization is forbidden in this owned CPU synthetic contract')

    class ObservedResult(unittest.TextTestResult):
        def startTest(self,test):
            CPU_checkpoint('before:'+test.id())
            started_tests.append(test.id())
            super().startTest(test)
        def addSuccess(self,test):
            successful_tests.append(test.id())
            super().addSuccess(test)
        def stopTest(self,test):
            CPU_checkpoint('after:'+test.id())
            super().stopTest(test)

    class TinyRealBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature=torch.nn.Sequential(torch.nn.Conv2d(3,4,3,padding=1),torch.nn.ReLU())
            self.semantic_output=torch.nn.Conv2d(4,1,1)
            self.calls=0
        def forward(self,image):
            self.calls+=1
            return self.semantic_output(self.feature(image))

    class AdditionalActualContracts(unittest.TestCase):
        def test_zero_initialized_head_is_exact_zero_residual(self):
            torch.manual_seed(202610173102)
            backbone=TinyRealBackbone().eval()
            adapter=adapter_module.make_adapter(torch,backbone,'feature',4,query_channels=8,hidden=6)
            image=torch.randn(1,3,8,12);embeddings=torch.randn(5,8);coarse=torch.randn(5,200,200)
            value=adapter(image,embeddings,coarse,query_batch_size=2)
            self.assertEqual(tuple(value.shape),(5,1,8,12))
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertTrue(torch.equal(value,torch.zeros_like(value)))
            self.assertEqual(backbone.calls,1)
            checks['zero_init']=dict(exact_zero_residual=True,finite=True,shape=[5,1,8,12],backbone_calls=backbone.calls)

        def test_nonzero_readout_has_real_finite_nonzero_backward_to_shared_backbone(self):
            torch.manual_seed(202610173103)
            backbone=TinyRealBackbone().eval()
            adapter=adapter_module.make_adapter(torch,backbone,'feature',4,query_channels=8,hidden=6)
            # An explicit disposable synthetic initialization activates the
            # readout path; no optimizer or real model update is performed.
            with torch.no_grad():
                for parameter in adapter.parameters():parameter.fill_(.1)
            image=torch.full((1,3,8,12),.2,requires_grad=True)
            embeddings=torch.full((3,8),.3,requires_grad=True)
            coarse=torch.full((3,200,200),.4,requires_grad=True)
            value=adapter(image,embeddings,coarse,query_batch_size=2)
            loss=value.square().mean();self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            def actual_gradient(tensor):
                self.assertIsNotNone(tensor.grad)
                self.assertTrue(bool(torch.isfinite(tensor.grad).all()))
                total=float(tensor.grad.abs().sum().item());self.assertGreater(total,0.)
                return total
            gradient_sums=dict(image=actual_gradient(image),embeddings=actual_gradient(embeddings),
                coarse=actual_gradient(coarse),captured_backbone_weight=actual_gradient(backbone.feature[0].weight))
            readout_gradients={}
            for name,parameter in adapter.named_parameters():
                if not name.startswith('backbone.'):
                    readout_gradients[name]=actual_gradient(parameter)
            self.assertTrue(readout_gradients)
            self.assertEqual(backbone.calls,1)
            self.assertTrue(all(parameter.grad is None for parameter in backbone.semantic_output.parameters()))
            checks['backward']=dict(real_loss=float(loss.item()),backward_calls=1,backbone_calls=backbone.calls,
                input_gradient_absolute_sums=gradient_sums,readout_parameter_gradient_absolute_sums=readout_gradients,
                all_checked_gradients_finite_nonzero=True,original_unused_semantic_tail_gradients_absent=True,
                optimizer_created=False,optimizer_steps=0)

    aliases=('full_instance_adapter','UNETPP_ADAPTER_PUBLIC_CONTRACT')
    need(all(name not in sys.modules for name in aliases),'Fresh source-closed adapter/test module namespace')
    try:
        torch.cuda._lazy_init=forbidden_cuda_init
        CPU_checkpoint('before_source_imports')
        adapter_module=load(c,spec['adapter_source'],aliases[0])
        public=load(c,spec['test_source'],aliases[1])
        need(public.adapter_module is adapter_module and public.torch is torch,
             'Original public test imports the exact source-closed factory and admitted Torch')
        public_loaded=True
        public_suite=unittest.defaultTestLoader.loadTestsFromModule(public)
        need(public_suite.countTestCases()==1,'Execute the original complete public unittest unchanged')
        suite=unittest.TestSuite([public_suite,unittest.defaultTestLoader.loadTestsFromTestCase(AdditionalActualContracts)])
        need(suite.countTestCases()==3,'Original public test plus exactly both additional real checks')
        with torch.random.fork_rng(devices=[]):
            result=unittest.TextTestRunner(stream=Tee(),verbosity=2,resultclass=ObservedResult).run(suite)
        CPU_checkpoint('after_all_actual_tests')
    except Exception:
        fatal=traceback.format_exc();print(fatal,file=sys.stderr,flush=True)
    finally:
        torch.cuda._lazy_init=original_lazy_init
        torch.set_rng_state(original_rng)
        for name in aliases:sys.modules.pop(name,None)
        suite=None;gc.collect()
    restored=bool(torch.equal(torch.get_rng_state(),original_rng))
    CPU_checkpoint('before_report')
    source_refs={key:spec[key] for key in ('entry_source','body_source','adapter_source','test_source','runtime_source')}
    for value in source_refs.values():need(c.ref(value['path'])==value,'All contract sources unchanged')
    errors=[] if result is None else [dict(test=str(test),traceback=trace) for test,trace in result.errors]
    failures=[] if result is None else [dict(test=str(test),traceback=trace) for test,trace in result.failures]
    skips=[] if result is None else [dict(test=str(test),reason=reason) for test,reason in result.skipped]
    tests_run=0 if result is None else result.testsRun
    public_started=[name for name in started_tests if name.startswith(aliases[1]+'.')]
    public_successful=[name for name in successful_tests if name.startswith(aliases[1]+'.')]
    passed=(fatal is None and result is not None and result.wasSuccessful() and tests_run==3
            and not skips and not result.expectedFailures and not result.unexpectedSuccesses
            and set(checks)=={'zero_init','backward'} and not attempts and restored
            and len(public_started)==len(public_successful)==1 and len(successful_tests)==3)
    log=output/'UNITTEST_STDOUT.txt'
    log_text=text.getvalue()+('' if fatal is None else '\nFATAL\n'+fatal)
    with log.open('x',encoding='utf-8',newline='\n') as stream:stream.write(log_text)
    log_ref=c.ref(log)
    report=dict(schema='UNETPP-full-instance-adapter-synthetic-contract-report-pending-v1',
        status=('ACTUAL_SYNTHETIC_CHECKS_PASSED_PENDING_ROOT_ACTUAL0' if passed else 'ACTUAL_SYNTHETIC_CHECKS_FAILED_PRESERVED'),
        source=spec['body_source'],source_refs=source_refs,spec=c.execution_spec_ref,
        CPU_admission=c.execution_admission_ref,phase=spec['phase'],prelock=spec['prelock'],
        public_tests=1,additional_tests=2,tests_run=tests_run,all_tests_passed=passed,
        public_test_source_loaded_unchanged=public_loaded,
        public_test_source_executed_unchanged=bool(public_started),
        actual_started_tests=started_tests,actual_successful_tests=successful_tests,
        actual_checks=checks,errors=errors,failures=failures,
        skipped=skips,expected_failures=[] if result is None else [str(test) for test,_ in result.expectedFailures],
        unexpected_successes=[] if result is None else [str(test) for test in result.unexpectedSuccesses],
        fatal_exception=fatal,unittest_log_ref=log_ref,CUDA_checkpoints=checkpoints,
        CUDA_lazy_initialization_blocked_during_tests=True,CUDA_initialization_attempts=attempts,
        CUDA_initialized=False,external_CPU_RNG_restored_exact=restored,
        actual_CPU_threads=torch.get_num_threads(),actual_CPU_interop_threads=torch.get_num_interop_threads(),
        actual_CPU_affinity=c.psutil.Process().cpu_affinity(),
        synthetic_only=True,original_full_backbone_executed=False,original_weights_loaded=False,
        real_images_loaded=False,GT_loaded=False,optimizer_created=False,optimizer_steps=0,
        disposable_synthetic_backward_only=True,new_training=False,AP=False,DEV=False,TEST=False,
        qualified=False,original_actual0_self_certified=False,innovation_established=False,goal_complete=False,
        limitations=['Real tiny-nn CPU contracts only; the full user UNet++/DSConv/Transformer was neither instantiated nor executed.',
            'No real-image, frozen teacher, initialization/checkpoint, renderer, complete architecture, training efficacy, CUDA memory or AP evidence.',
            'Unused original semantic output parameters need not receive gradients when reading an earlier feature; the probe verifies the captured backbone path and adapter readout.',
            'The tests do not certify training memory bounded by query microbatch size or numerical equivalence on GPU.'],
        elapsed_seconds=time.monotonic()-started)
    report_ref=c.write_json(output/'ADAPTER_CONTRACT_REPORT.json',report)
    del original_rng
    gc.collect();c.check()
    need(passed,'Actual synthetic contract failed: original errors/report/stdout preserved; no success qualification')
    return dict(report_ref=report_ref,unittest_log_ref=log_ref,public_tests=1,additional_tests=2,
        tests_run=tests_run,all_tests_passed=True,synthetic_only=True,original_full_backbone_executed=False,
        qualified=False,new_training=False,AP=False,DEV=False,TEST=False)
