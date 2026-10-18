"""UNBOUND E50 single-arm CPU restore; numerical path has not been executed.

Only stdlib imports. Caller already owns the cold-CPU build_pair adapter and
fresh initialization receipt. No model constructor, optimizer, teacher, image,
population, runner, GPU transfer, qualification, or checkpoint writing occurs.
"""
from copy import deepcopy
import gc
import hashlib
from pathlib import Path
import platform
import time
import traceback

ROLES=('prediction_core','training_state','u39_training_ops','u36_helpers')
PINS=dict(prediction_core='819a2350487394265980cb77f8ff7e89547dbcac8e0d3e3c175a67a2794c9680',
    training_state='8f9b549a8383877aa6f0ae3b1d2a9786c44c0f486e45b447b06dea2e25f5a0c5',
    u39_training_ops='378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d',
    u36_helpers='1e726092a19d80a06f58d6a56f85f4c15bff2b744f0468a7e0a35fa2c85999e5')


def need(ok,message):
    if not ok:raise ValueError(message)


def validate_restore_request(prediction_core,metadata,arm,source_refs):
    """Public pure metadata boundary; delegates original full endpoint checks."""
    need(arm in ('PLAIN','FULL'),'One actual paired arm, no alias')
    endpoint=prediction_core.validate_final_endpoint(metadata)
    need(type(source_refs) is dict and set(source_refs)==set(ROLES),'Four explicit original restore helper sources')
    for role,value in source_refs.items():
        prediction_core.strict_ref(value);need(value['sha256']==PINS[role],'Original restore helper pin: '+role)
    sources=endpoint['training_sources']
    for role,original in (('training_state','core_source'),('u39_training_ops','u39_training_ops'),('u36_helpers','u36_helpers')):
        need(source_refs[role]==sources[original],'Reuse exact original training source ref: '+role)
    return endpoint


def _module(c,module,reference,expected_sha):
    path=Path(c.closed(reference)).resolve(strict=True)
    need(Path(module.__file__).resolve(strict=True)==path,'Actually injected module must be source-closed')
    raw=path.read_bytes()
    need(len(raw)==reference['bytes'] and reference['sha256']==hashlib.sha256(raw).hexdigest()==expected_sha,
         'Exact original module bytes/SHA')


def restore_arm(torch,c,adapter,arm,endpoint_metadata,fresh_initialization_receipt,*,modules,source_refs,report):
    """Restore one existing CPU adapter in place; return JSON-only receipt.

    Call once per arm after cold-CPU build_pair and before the teacher's first
    CUDA use. Keep that same CPU adapter for future windows. The caller owns
    Source admission, RAM/deadline enforcement and subsequent GPU lifetimes.
    A failure may leave a partially restored adapter: discard it, never retry
    or silently use it. This function does not qualify its own numerical work.
    """
    need(type(report) is dict and not report,'Caller-owned fresh partial restore report')
    report.update(schema='U44-external-fixed50-CPU-arm-restore-pending-v1',status='STARTED',arm=arm,
        qualified=False,new_training=False,AP=False,optimizer_created=False,optimizer_restored=False,
        CPU_checkpoint_optimizer_payload_loaded=False,CPU_checkpoint_payload_released=False)
    saved=outside_cpu=None;started=time.perf_counter()
    try:
        need(type(modules) is dict and set(modules)==set(ROLES),'Inject exactly four admitted original helper modules')
        core=modules['prediction_core'];training=modules['training_state'];ops=modules['u39_training_ops'];helpers=modules['u36_helpers']
        endpoint=validate_restore_request(core,endpoint_metadata,arm,source_refs)
        for role in ROLES:_module(c,modules[role],source_refs[role],PINS[role])
        # Rejoin the caller's four parsed metadata documents to their closed
        # references. No FIT image, annotation, cache, or weight is opened here.
        actual=dict(refs=deepcopy(endpoint_metadata['refs']))
        for key,value in actual['refs'].items():actual[key]=core._json(c,value)
        need(actual==endpoint_metadata,'Caller metadata must equal the four actual closed endpoint JSON documents')
        for reference in endpoint['training_sources'].values():core._closed(c,reference)
        need(torch.__version__=='2.0.1+cu118' and platform.python_version()=='3.8.17'
             and torch.get_default_dtype()==torch.float32 and not torch.cuda.is_initialized(),'Original cold-CPU FP32 restore runtime')
        need(isinstance(adapter,torch.nn.Module) and all(value.device.type=='cpu' for value in adapter.state_dict().values())
             and all(p.grad is None for p in adapter.parameters()),'Already constructed CPU adapter, no retained gradients')
        outside_cpu=torch.get_rng_state().clone();c.check()
        training.validate_order(actual['order'])
        need(actual['pair_commit']['identity']['recipe']==training.RECIPE
             and actual['pair_commit']['identity']['runtime']==dict(python=platform.python_version(),
                 torch=str(torch.__version__),numpy='1.23.5'),'Original complete order/recipe/runtime lineage')
        effective,inactive,report['parameters']=ops._partition(adapter)
        # Exactly one single-arm checkpoint load; its complete CPU container may
        # include Adam state. No optimizer object or optimizer load is performed.
        saved=torch.load(str(core._closed(c,endpoint['checkpoint_refs'][arm])),map_location='cpu')
        need(type(saved) is dict and type(saved.get('optimizer')) is dict,'Original checkpoint container includes its CPU optimizer payload')
        report['CPU_checkpoint_optimizer_payload_loaded']=True
        pair=actual['pair_commit']
        need(saved['schema']=='U40-continuous-arm-state-v1' and saved['arm']==arm
             and saved['cursor']==core.FINAL_CURSOR and saved['identity']==pair['identity']
             and saved['effective_parameter_names']==sorted(effective)
             and saved['inactive_parameter_names']==sorted(inactive)
             and saved['order_ref']==endpoint['order_ref']
             and saved['actual_trace_sha256']==pair['actual_trace_sha256']
             and saved['requested_trace_sha256']==training.expected_trace(actual['order'],37050),
             'Exact final arm/source/recipe/cursor/names/actual and requested order identity')
        need(saved['initialization_receipt']==fresh_initialization_receipt,'Original common104/adapter initialization lineage exact')
        need(all(value.device.type=='cpu' and (not value.is_floating_point() or
            (value.dtype==torch.float32 and bool(torch.isfinite(value).all()))) for value in saved['model'].values()),
             'Entire finite CPU FP32 model state, including BN and inactive states')
        adapter.load_state_dict(saved['model'],strict=True)
        training._equal(torch,adapter.state_dict(),saved['model'],'E50 external CPU model/BN/inactive')
        need(set(dict(adapter.named_modules()))==set(saved['module_training_modes']),'Same full module tree')
        proof=core._state_digest(helpers,adapter.state_dict())
        adapter.eval().requires_grad_(False)
        need(core._state_digest(helpers,adapter.state_dict())==proof and all(not m.training for m in adapter.modules())
             and all(not p.requires_grad and p.grad is None for p in adapter.parameters()),'Restored trained state unchanged; frozen eval inference')
        need(torch.equal(torch.get_rng_state(),outside_cpu) and not torch.cuda.is_initialized(),'External CPU RNG unchanged and CUDA still cold')
        report.update(status='RESTORED_CPU_ONLY',seed=core.SEED,fixed_epoch=50,updates_per_arm=37050,
            endpoint_refs=deepcopy(actual['refs']),checkpoint_ref=deepcopy(endpoint['checkpoint_refs'][arm]),
            checkpoint_refs=deepcopy(endpoint['checkpoint_refs']),training_qualified=deepcopy(endpoint['training_qualified']),
            pair_commit_ref=deepcopy(endpoint['pair_commit_ref']),order_ref=deepcopy(endpoint['order_ref']),
            training_sources=deepcopy(endpoint['training_sources']),restore_sources=deepcopy(source_refs),
            initialization_receipt_sha256=core.metadata_sha(fresh_initialization_receipt),model_state_sha256=proof,
            state_proof_algorithm='original prediction_core._state_digest with u36_helpers._digest; names/dtype/shape/numeric bytes, no device',
            effective_parameter_names=sorted(effective),inactive_parameter_names=sorted(inactive),module_names=sorted(dict(adapter.named_modules())),
            restored_model_state_exact=True,initialization_lineage_exact=True,all_modules_eval=True,
            all_parameters_frozen=True,all_parameter_grads_none=True,external_CPU_RNG_unchanged=True,
            CUDA_initialized=False,model_forward_executed=False,trained_head_reset=False,
            adapter_object_replaced=False,adapter_destroyed=False,checkpoints_loaded=1,
            numerical_runtime_qualified=False,population_qualification=False)
    except BaseException as error:
        report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        # Keep the original text above, release nested traceback tensor aliases.
        traceback.clear_frames(error.__traceback__);error.__traceback__=None
        raise
    finally:
        saved=outside_cpu=None;gc.collect()
        report['CPU_checkpoint_payload_released']=True;report['elapsed_seconds']=time.perf_counter()-started
    c.check();return report
