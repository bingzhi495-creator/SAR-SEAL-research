"""UNBOUND thin fixed-E50 dispatch; no numerical execution during authoring.

No model/optimizer/tensor is constructed here. GPU computation belongs only to
the fixed prediction core; CPU computation belongs only to the fixed metrics
core. CPU imports real Torch only for the caller's CUDA-cold checks.
"""
from copy import deepcopy
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
import types

GPU_OPERATION = 'GPU_U40_FIXED50_FULL_METHOD_DEV_SHARD'
CPU_OPERATION = 'CPU_U40_FIXED50_FULL_DEV_COCO'
CORE_PINS = dict(
    prediction_core_source='819a2350487394265980cb77f8ff7e89547dbcac8e0d3e3c175a67a2794c9680',
    metrics_source='f513d59321ffb571f6d823a74f31e3b6fd190086dc73bd77b400fe0500eb489a')


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def _bytes(c, reference):
    path=Path(c.closed(reference)); raw=path.read_bytes()
    need(len(raw)==reference['bytes'] and hashlib.sha256(raw).hexdigest()==reference['sha256'],
         'Explicit admitted source/metadata changed')
    return path,raw


def _json(c, reference):
    return json.loads(_bytes(c,reference)[1])


def _module(c, reference, name, previous):
    path,raw=_bytes(c,reference)
    if name not in previous:
        previous[name]=(name in sys.modules,sys.modules.get(name))
    module=types.ModuleType(name);module.__file__=str(path)
    sys.modules[name]=module
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def _library(c, reference, name):
    path,_=_bytes(c,reference)
    module=importlib.import_module(name)
    need(Path(module.__file__).resolve()==path.resolve(), 'Canonical official module must match admitted source: '+name)
    return module


def _prediction_modules(c, refs, previous):
    # Literal dependencies of the unchanged legacy full backbone. Their aliases
    # are temporary and restored even if loading or core prediction fails.
    modules={}
    for role,name in (('full_dsconv','dsconv'),('full_head','head'),('full_legacy','model_full')):
        modules[role]=_module(c,refs[role],name,previous)
    for role,reference in refs.items():
        if role in modules:
            continue
        if role=='mask_api':
            modules[role]=_library(c,reference,'pycocotools.mask')
        else:
            modules[role]=_module(c,reference,'U40_E50_PRED_'+role,previous)
    return modules


def _metrics_modules(c, refs, previous):
    modules={}
    libraries=dict(coco='pycocotools.coco',cocoeval='pycocotools.cocoeval',masks='pycocotools.mask')
    for role,reference in refs.items():
        modules[role]=(_library(c,reference,libraries[role]) if role in libraries else
                       _module(c,reference,'U40_E50_CPU_'+role,previous))
    return modules


def run(torch, np, spec, c, pre):
    """Return untouched core result plus a separate wrapper provenance report."""
    gpu=spec['operation']==GPU_OPERATION
    need(gpu or spec['operation']==CPU_OPERATION, 'Only fixed50 GPU prediction or CPU full COCO')
    need(not torch.cuda.is_initialized(), 'All module construction starts CUDA cold')
    started=time.monotonic(); previous={}; result=None; modules=None; core=None
    core_report={}
    audit=dict(schema='U40-fixed50-DEV-wrapper-report-pending-v1',status='STARTED',
        operation=spec['operation'],phase=spec['phase'],source=spec['body_source'],
        entry_source=spec['entry_source'],runtime_source=spec['runtime_source'],
        spec=c.execution_spec_ref,admission=c.execution_admission_ref,prelock=spec['prelock'],
        parent_ROOT_qualified=spec['parent_ROOT_qualified'],training_qualified=spec['training_qualified'],
        checkpoint_refs=spec['checkpoint_refs'],pair_commit_ref=spec['pair_commit_ref'],order_ref=spec['order_ref'],
        seed=202610173101,fixed_epoch=50,updates_per_arm=37050,
        original_image_references_metadata_sha256=pre['original_image_references_metadata_sha256'],
        qualified=False,original_actual0_self_certified=False,new_training=False,
        AP=not gpu,DEV=True,TEST=False,utility_gate_decided=False,
        wrapper_constructed_model_or_optimizer=False,wrapper_created_Torch_tensors=False,
        CPU_operation_Torch_imported_for_CUDA_cold_check=(not gpu),
        framework_scope='Explicit declared Python sources and pinned runtime versions; not a complete framework import-byte audit')
    wrapper_ref=None
    try:
        role='prediction_core_source' if gpu else 'metrics_source'
        reference=spec[role]
        need(reference['sha256']==CORE_PINS[role], 'Exact unchanged published fixed50 core')
        core=_module(c,reference,'U40_E50_ACTIVE_CORE',previous)
        if gpu:
            modules=_prediction_modules(c,spec['prediction_sources'],previous)
            limits=dict(deadline_seconds=spec['deadline_seconds'],allowed_cpus=spec['allowed_cpus'],threads=spec['thread_limit'],
                allocator_bytes=spec['allocator_limit_MiB']*1024**2,whole_GPU_bytes=spec['whole_GPU_limit_MiB']*1024**2,
                maximum_RSS_bytes=spec['maximum_process_RSS_bytes'],minimum_available_RAM_bytes=spec['minimum_available_RAM_bytes'])
            audit['invoked_function']='prediction_core.predict_shard'
            result=core.predict_shard(torch,np,c,endpoint_refs=spec['endpoint_refs'],dev_context_ref=spec['dev_context'],
                original_image_refs=spec['original_image_refs'],modules=modules,source_refs=spec['prediction_sources'],
                prediction_core_ref=spec['prediction_core_source'],shard_index=spec['shard_index'],
                dataset_range=spec['dataset_range'],output_directory=spec['output_directory'],limits=limits,report=core_report)
        else:
            modules=_metrics_modules(c,spec['metrics_module_sources'],previous)
            # The scientific core owns all full-population/COCO checks. These two
            # joins additionally retain this wrapper's one common prelock/RGB set.
            for reference in spec['accepted_GPU_shards']:
                accepted=_json(c,reference)
                need(accepted['schema']=='ROOT-U40-fixed50-full-method-DEV-shard-qualified-v1'
                     and accepted['qualified'] is True and accepted['prelock']==spec['prelock'],
                     'Only actual Root prediction shards sharing this exact prelock')
                shard=_json(c,accepted['result']['report_ref'])
                need(shard['original_image_references_metadata_sha256']==pre['original_image_references_metadata_sha256'],
                     'All four actual prediction shards must retain the fixed original RGB identities')
            need(not torch.cuda.is_initialized(), 'CPU official modules did not initialize CUDA')
            audit['invoked_function']='population_metrics.evaluate_population'
            result=core.evaluate_population(torch,np,c,spec,pre,modules,core_report)
            need(not torch.cuda.is_initialized(), 'CPU metrics never initialized CUDA')
        audit.update(status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
            result=deepcopy(result),core_source=reference if gpu else spec['metrics_source'],
            core_report_ref=result['report_ref'])
    except BaseException as error:
        audit.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error),
                                               traceback=traceback.format_exc()))
        traceback.clear_frames(error.__traceback__);error.__traceback__=None
        raise
    finally:
        modules=core=None;gc.collect()
        for name,(existed,value) in previous.items():
            if existed:sys.modules[name]=value
            else:sys.modules.pop(name,None)
        previous.clear()
        audit['elapsed_seconds']=time.monotonic()-started
        audit['CUDA_initialized_at_return']=bool(torch.cuda.is_initialized())
        out=Path(spec['output_directory'])
        if out.is_dir():
            # Each unchanged core exclusively creates its output directory.
            # Pre-output failure remains in the original owned driver's logs.
            wrapper_ref=c.write_json(out/'WRAPPER_REPORT.json',audit)
    return dict(result=result,wrapper_report_ref=wrapper_ref,
                core_peak_allocated_bytes=(core_report['torch_peak_allocated_bytes'] if gpu else 0))
