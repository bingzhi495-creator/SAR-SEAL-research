"""UNBOUND teacher-only CPU restoration draft; stdlib-only imports.

No numerical path has been executed. All genuine modules, registrations,
context references, resources and scientific admission belong to the caller.
"""
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time
import traceback


MODULE_ROLES=dict(native='native_loader_source',legacy='evaluation_source',pref='micro_preflight_source',
    initializer='initializer_source',common='training_source')
PINS=dict(native=(45363,'63ddb7783c24b06397476d178307c60eedee2cf751cc75236deb582999b0c7a2'),
    pref=(30547,'7f7e5ce0eea8f0dae6e7fc01a4b54baafdc52fb3339b039ae50ef26b60870201'))
REGISTRATION_ROLES=('register_all_modules','init_default_scope','registry_module','revert_sync_batchnorm','Config','CocoDataset')


def need(ok,message):
    if not ok:raise ValueError(message)


@contextmanager
def closed_helpers(modules,closed_ref):
    """Route original one-reference helpers through this caller; restore on error."""
    original={}
    try:
        for role,module in modules.items():
            if hasattr(module,'bound'):
                need(callable(module.bound) and len(inspect.signature(module.bound).parameters)==1,
                     'Original one-reference bound ABI: '+role)
                original[role]=module.bound;module.bound=closed_ref
        yield
    finally:
        for role,value in original.items():modules[role].bound=value


def qualified_teacher_inputs(context,legacy,native):
    """Exact original one-argument lineage call; no dataset/cache access here."""
    teachers=legacy.qualified_training(context)
    teacher=teachers['native0']
    construction=dict(teacher['training_spec'],training_source=context['training_source'],
        checkpointmap=context['checkpointmap'],qualificationmap=context['qualificationmap'],
        order_seed=native.ORDER_SEED,phase_loss_seed_base=native.LOSS_SEED_BASE)
    return dict(teacher=teacher,construction=construction)


def _strict_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'}
         and type(value['path']) is str and Path(value['path']).is_absolute()
         and type(value['bytes']) is int and value['bytes']>=0
         and type(value['sha256']) is str and len(value['sha256'])==64
         and set(value['sha256'])<=set('0123456789abcdef'),'Strict canonical caller-admitted reference')
    return value


def _source(c,obj,reference,pin=None):
    _strict_ref(reference);path=Path(c.closed(reference)).resolve(strict=True)
    need(str(path)==reference['path'],'Caller must provide canonical paths, not junction aliases')
    actual=inspect.getsourcefile(obj) if inspect.isclass(obj) or inspect.isfunction(obj) else obj.__file__
    need(Path(actual).resolve(strict=True)==path,'Injected actual dependency must equal admitted source')
    raw=path.read_bytes()
    need(len(raw)==reference['bytes'] and hashlib.sha256(raw).hexdigest()==reference['sha256'],'Actual source bytes/SHA')
    if pin is not None:need((len(raw),reference['sha256'])==pin,'Original numerical helper pin')


def restore_teacher_cpu(torch,c,context,*,modules,registration,source_refs,report):
    """Restore the original E24 teacher on cold CPU; return model plus JSON receipt.

    All objects and refs must already be admitted by the caller. No framework
    import, dataset construction, data iteration, teacher forward, optimizer,
    CUDA initialization or file output is introduced here. Original historical
    lineage helpers still hash their declared inputs and parse qualification
    JSON: this is not a claim that no old data-file bytes are accessed.
    """
    need(type(report) is dict and not report,'Fresh caller-owned partial restore report')
    report.update(schema='U47-external-teacher-CPU-restore-pending-v1',status='STARTED',qualified=False,
        new_training=False,AP=False,model_forward_executed=False,optimizer_created=False,optimizer_restored=False,
        dataset_constructed=False,image_decoded=False,annotation_records_parsed=False,
        CUDA_initialized=False,independent_anchor_released=False,external_CPU_RNG_restored=False,
        original_helper_bindings_restored=False,full_framework_binary_closure_claimed=False,
        historical_lineage_source_hashing_may_include_data_bytes=True)
    model=anchor=outside_cpu=inputs=None;started=time.perf_counter()
    try:
        try:
            need(torch.__version__=='2.0.1+cu118' and platform.python_version()=='3.8.17'
                 and torch.get_default_dtype()==torch.float32 and not torch.cuda.is_initialized(),'Exact cold-CPU FP32 teacher runtime')
            need(type(modules) is dict and set(modules)==set(MODULE_ROLES),'Five already-admitted original numerical modules')
            need(type(registration) is dict and set(registration)==set(REGISTRATION_ROLES)|{'MODELS'},'Explicit original framework registration objects')
            need(type(source_refs) is dict and set(source_refs)=={'context','modules','registration'}
                 and set(source_refs['modules'])==set(MODULE_ROLES)
                 and set(source_refs['registration'])==set(REGISTRATION_ROLES),'Explicit context/five-module/six-registration source map')
            actual=json.loads(Path(c.closed(_strict_ref(source_refs['context']))).read_text(encoding='utf-8-sig'))
            need(actual==context,'Injected parsed context equals its actual admitted original JSON')
            for value in context['sources']:
                _strict_ref(value)
                need(c.allowed.get(str(Path(value['path']).resolve()).lower())==value,'Original context closure must be pre-admitted')
            for role,key in MODULE_ROLES.items():
                need(source_refs['modules'][role]==context[key],'Exact original context source role: '+role)
                _source(c,modules[role],source_refs['modules'][role],PINS.get(role))
            for role in REGISTRATION_ROLES:_source(c,registration[role],source_refs['registration'][role])
            need(registration['MODELS'] is registration['registry_module'].MODELS,'Original mmdet MODELS registry object')
            native,legacy,pref,initializer,common=(modules[k] for k in ('native','legacy','pref','initializer','common'))
            need(len(inspect.signature(legacy.qualified_training).parameters)==1
                 and len(inspect.signature(native.prepare_training_model).parameters)==10
                 and len(inspect.signature(pref.prepare_cpu).parameters)==9,'Original one/ten/nine-argument teacher-loading ABI')
            outside_cpu=torch.get_rng_state().clone();c.check()
            try:
                with closed_helpers(modules,c.closed):
                    # The old qualifier observes all original terminal receipts
                    # before its scientific lineage. Close these refs first without
                    # replacing or bypassing its original held/identity checks.
                    for value in context['training_receiptmap'].values():c.closed(value)
                    inputs=qualified_teacher_inputs(context,legacy,native)
                    registration['register_all_modules'](init_default_scope=False)
                    registration['init_default_scope']('mmdet')
                    classes=tuple(registration['CocoDataset'].METAINFO['classes'])
                    model,anchor,policy,loading=native.prepare_training_model(torch,pref,initializer,common,
                        registration['MODELS'],registration['revert_sync_batchnorm'],registration['Config'],
                        classes,inputs['construction'],inputs['teacher'])
                    native.imported_library_gate(torch,model,context)
                    need(Path(inspect.getsourcefile(torch.nn.functional.interpolate)).resolve()==
                         Path(c.closed(context['interpolation_SOURCE'])).resolve(),'Original actual interpolation source')
                    model.requires_grad_(False);model.eval();model.backbone.with_cp=False
                    need(len(anchor)==len(model.state_dict())==610 and len(tuple(model.parameters()))==451
                         and all(p.device.type=='cpu' and p.dtype==torch.float32 and not p.requires_grad and p.grad is None
                                 for p in model.parameters()) and all(not child.training for child in model.modules()),
                         'Complete frozen eval CPU teacher610/451, no gradients')
                    need(all(value.device.type=='cpu' and torch.equal(value.detach(),anchor[name])
                         and (not value.is_floating_point() or value.dtype==torch.float32 and bool(torch.isfinite(value).all()))
                         for name,value in model.state_dict().items()),'All model/BN state exactly equals original loaded CPU anchor')
                    proof=pref.state_proof(model.state_dict())
                    need(loading['schema']=='native-finetune24-fresh-optimizer-loading-proof-v1'
                         and loading['teacher_checkpoint']==context['checkpointmap']['native0']
                         and loading['teacher_qualification']==context['qualificationmap']['native0']
                         and loading['teacher_training_source']==context['training_source'] and loading['teacher_epoch']==24
                         and loading['teacher_state']==proof and loading['strict_full_native610_loaded'] is True
                         and loading['old_optimizer_restored'] is False and loading['old_RNG_restored'] is False,
                         'Original exact checkpoint/qualification/loading/state proof; no optimizer or RNG restoration')
                    report.update(status='CPU_STATE_RESTORED_PENDING_FINAL_CHECKS',teacher_checkpoint=deepcopy(context['checkpointmap']['native0']),
                        teacher_qualification=deepcopy(context['qualificationmap']['native0']),teacher_epoch=24,
                        teacher_state_proof=proof,original_loading_proof=deepcopy(loading),original_trainability_policy=deepcopy(policy),
                        original_policy_is_provenance_not_live_trainability=True,source_refs=deepcopy(source_refs),
                        state_entries=610,parameter_tensors=451,all_modules_eval=True,all_parameters_frozen=True,
                        all_parameter_grads_none=True,restored_model_BN_state_exact=True,backbone_with_cp=False,
                        state_proof_algorithm='original pref.state_proof/digest_tensor: per-name shape/dtype/CPU contiguous numeric-byte SHA; device omitted',
                        old_checkpoint_optimizer_and_RNG_containers_deserialized=loading['old_checkpoint_optimizer_and_RNG_containers_deserialized'],
                        original_qualification_checked=True,original_library_gate_passed=True,
                        loading_receipt_file_written=False,teacher_model_returned_to_caller=False,
                        preprocessing_resolved_here=False,config_sources_exported_here=False,numerical_runtime_qualified=False)
            finally:
                report['original_helper_bindings_restored']=True
            need(not torch.cuda.is_initialized(),'Teacher construction/restoration never initializes CUDA')
        finally:
            anchor=inputs=None;gc.collect();report['independent_anchor_released']=True
            if outside_cpu is not None:
                torch.set_rng_state(outside_cpu)
                report['external_CPU_RNG_restored']=bool(torch.equal(torch.get_rng_state(),outside_cpu))
            outside_cpu=None;report['elapsed_seconds']=time.perf_counter()-started
        need(report['external_CPU_RNG_restored'],'Caller CPU RNG actually restored after teacher construction')
        json.dumps(report,allow_nan=False);c.check()
        report.update(status='RESTORED_CPU_ONLY',teacher_model_returned_to_caller=True)
        return dict(model=model,receipt=report,teacher_state_proof=report['teacher_state_proof'])
    except BaseException as error:
        report.update(status='FAILED',teacher_model_returned_to_caller=False,error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        model=None;traceback.clear_frames(error.__traceback__);error.__traceback__=None
        raise
