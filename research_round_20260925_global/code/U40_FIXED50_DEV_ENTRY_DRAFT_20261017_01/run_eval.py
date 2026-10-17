"""UNBOUND Root-owned fixed-E50 DEV entry; no numerical execution during authoring.

GPU delegates to the unchanged prediction core, CPU to the unchanged metrics
core. Root still owns binding, supervision, original termination and acceptance.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

GPU_OPERATION = 'GPU_U40_FIXED50_FULL_METHOD_DEV_SHARD'
CPU_OPERATION = 'CPU_U40_FIXED50_FULL_DEV_COCO'
GPU_SCHEMA = 'ROOT-U40-fixed50-full-method-DEV-prediction-spec-v1'
CPU_SCHEMA = 'ROOT-U40-fixed50-complete-DEV-metrics-spec-v1'
SEED = 202610173101
RANGES = [[0,47],[47,94],[94,141],[141,187]]
TRAINING_ROLES = ('shared_initialization','rgb_matched_reader','full_legacy','full_dsconv',
    'full_head','full_feature','plain_legacy','plain_feature','instance_adapter',
    'matched_reader','image_geometry','u36_helpers','u39_training_ops',
    'entry_source','body_source','runtime_source','core_source')
PREDICTION_ROLES = ('shared_initialization','full_legacy','full_dsconv','full_head','full_feature',
    'plain_legacy','plain_feature','instance_adapter','u36_helpers','u39_training_ops',
    'image_geometry','training_state','dev_reader','dev_rgb_reader','population_contract','mask_api','encoder')
METRICS_ROLES = ('legacy_eval','population','geometry','official','coco','cocoeval','masks')
CORE_PINS = dict(prediction_core_source='819a2350487394265980cb77f8ff7e89547dbcac8e0d3e3c175a67a2794c9680',
                 metrics_source='f513d59321ffb571f6d823a74f31e3b6fd190086dc73bd77b400fe0500eb489a')


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def validate_operation(spec):
    """Pure declared metadata only; never a qualification or efficacy gate."""
    need(spec['operation'] in (GPU_OPERATION, CPU_OPERATION), 'Only fixed50 prediction or full CPU COCO')
    gpu = spec['operation'] == GPU_OPERATION
    need(spec['schema'] == (GPU_SCHEMA if gpu else CPU_SCHEMA)
         and type(spec['seed']) is int and spec['seed'] == SEED
         and spec['fixed_epoch'] == 50 and spec['updates_per_arm'] == 37050,
         'Only preselected actual E50/37050, never a midpoint or old E24 endpoint')
    need(all(type(spec[key]) is int and spec[key] >= 0 for key in (
            'owner_index','normal_RAW_count','total_failed_RAW_count',
            'total_terminal_identity_pin_count','legacy_pin_count'))
         and spec['owner_index'] == spec['normal_RAW_count']
         and spec['total_terminal_identity_pin_count'] == spec['normal_RAW_count']+spec['total_failed_RAW_count'],
         'Dynamic actual account counts; legacy physical pin count stays independent')
    need(spec['allowed_cpus'] == [0,1] and spec['thread_limit'] == 2
         and spec['cuda_visible_devices'] == ('0' if gpu else '')
         and spec['whole_GPU_limit_MiB'] == 8192 and spec['allocator_limit_MiB'] == 4096
         and spec['maximum_process_RSS_bytes'] == spec['minimum_available_RAM_bytes'] == 16*1024**3
         and spec['deadline_seconds'] == 1800 and spec['automatic_retry'] is False
         and spec['no_new_training'] is True and spec['no_TEST'] is True
         and spec['no_AP'] is gpu and spec['no_DEV'] is False,
         'Original CPU/thread/device/RSS/RAM/deadline/no-retry resource scope')
    if gpu:
        need(type(spec['shard_index']) is int and 0 <= spec['shard_index'] < 4
             and spec['dataset_range'] == RANGES[spec['shard_index']], 'One fixed canonical GPU shard')
    return gpu


def strict_ref(value):
    need(type(value) is dict and set(value) == {'path','bytes','sha256'}
         and type(value['path']) is str and Path(value['path']).is_absolute()
         and type(value['bytes']) is int and value['bytes'] >= 0
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'), 'Strict closed absolute reference')
    return value


def path_key(path):
    return str(Path(path).resolve()).replace('\\','/').casefold()


def metadata_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def validate_declared_inputs(spec):
    """Validate declared membership only; owned c.closed does actual byte checks."""
    gpu = validate_operation(spec)
    need(type(spec['sources']) is list, 'Explicit direct Source references')
    sources = {}
    for value in spec['sources']:
        strict_ref(value); key=path_key(value['path'])
        need(key not in sources, 'No duplicate/path-alias Source entries')
        sources[key]=value
    required = [spec[key] for key in ('entry_source','body_source','runtime_source',
        'prediction_core_source','prelock','parent_ROOT_qualified','training_qualified',
        'pair_commit_ref','order_ref','dev_context')]
    need(type(spec['checkpoint_refs']) is dict and set(spec['checkpoint_refs']) == {'PLAIN','FULL'},
         'Both exact final paired checkpoints')
    required += list(spec['checkpoint_refs'].values())
    for key,roles in (('training_sources',TRAINING_ROLES),('prediction_sources',PREDICTION_ROLES)):
        need(type(spec[key]) is dict and set(spec[key]) == set(roles), 'Exact declared '+key+' roles')
        required += list(spec[key].values())
    images=spec['original_image_refs']
    need(type(images) is list and len(images) == 187, 'All 187 original RGB references in canonical ordinal order')
    for value in images: strict_ref(value)
    need(len({path_key(value['path']) for value in images}) == 187, 'Distinct admitted original image paths')
    if gpu:
        need(type(spec['endpoint_refs']) is dict
             and set(spec['endpoint_refs']) == {'training_qualified','pair_commit','preparation','order'}
             and spec['endpoint_refs']['training_qualified'] == spec['training_qualified']
             and spec['endpoint_refs']['pair_commit'] == spec['pair_commit_ref']
             and spec['endpoint_refs']['order'] == spec['order_ref'], 'Fixed common endpoint reference joins')
        required += list(spec['endpoint_refs'].values())
        first,stop=spec['dataset_range']; required += images[first:stop]
    else:
        need(type(spec['metrics_module_sources']) is dict
             and set(spec['metrics_module_sources']) == set(METRICS_ROLES), 'Seven injected CPU metric modules')
        required += [spec['metrics_source'],spec['coco_context']]+list(spec['metrics_module_sources'].values())
        accepted=spec['accepted_GPU_shards']
        need(type(accepted) is list and len(accepted) == 4
             and len({path_key(strict_ref(value)['path']) for value in accepted}) == 4,
             'Four distinct fixed-order Root GPU acceptances')
        required += accepted
    for value in required:
        strict_ref(value)
        need(sources.get(path_key(value['path'])) == value, 'Declared input missing/conflicting in Source map: '+value['path'])
    return sources


def validate_prelock(spec, pre):
    need(pre['schema'] == 'U40-fixed50-complete-DEV-metrics-prelock-v1'
         and pre['seed'] == SEED and pre['fixed_epoch'] == 50 and pre['updates_per_arm'] == 37050
         and pre['images'] == 187 and pre['GT'] == 380 and pre['queries'] == 18700
         and pre['GPU_ranges'] == RANGES and pre['query_batch'] == 4 and pre['binary_rule'] == '>0'
         and pre['fixed_original_S0'] is True and pre['GT_used_for_prediction'] is False
         and pre['score_update'] is False and pre['query_filter'] is None and pre['NMS'] is False
         and pre['no_new_training'] is True and pre['no_TEST'] is True,
         'One preselected E50/all100 original-score masks-only contract; no new gate')
    need(pre['original_image_references_metadata_sha256'] == metadata_sha(spec['original_image_refs']),
         'Original ordinal-ordered 187 RGB reference identities changed')


def validate_result(spec, result):
    """Only returned metadata/ownership; original terminal acceptance is external."""
    gpu=validate_operation(spec)
    need(type(result) is dict and result['seed'] == SEED and result['fixed_epoch'] == 50
         and result['updates_per_arm'] == 37050 and result['new_training'] is False
         and result['AP'] is (not gpu) and result['DEV'] is True and result['TEST'] is False
         and result['qualified'] is False, 'Only pending fixed50 result with the original operation scope')
    references=[result['report_ref']]
    if gpu:
        first,stop=spec['dataset_range']
        need(result['dataset_range'] == [first,stop] and result['shard_index'] == spec['shard_index']
             and result['images'] == stop-first and result['queries'] == (stop-first)*100
             and type(result['predictions']) is dict and set(result['predictions']) == {'PLAIN','FULL'},
             'Complete actual PLAIN/FULL fixed shard population')
        for key in ('training_qualified','pair_commit_ref','checkpoint_refs','order_ref',
                    'dev_context','training_sources','prediction_sources'):
            need(result[key] == spec[key], 'Core result substituted requested provenance: '+key)
        references += list(result['predictions'].values())+[result['identity_predictions'],result['Omega_support']]
    else:
        need(result['images'] == 187 and result['GT'] == 380 and result['queries'] == 18700
             and result['utility_gate_decided'] is False, 'Complete CPU population; no new progression gate')
    out=Path(spec['output_directory']).resolve()
    for value in references:
        strict_ref(value)
        need(out in Path(value['path']).resolve().parents, 'Owned output reference required')
    return references


def reference(path):
    path=Path(path);digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def _load(path, name):
    # No importlib bytecode cache writes into immutable source directories.
    path=Path(path);module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def _code_references(spec, gpu):
    values=[spec[key] for key in ('entry_source','body_source','runtime_source','prediction_core_source')]
    values+=list(spec['training_sources'].values())+list(spec['prediction_sources'].values())
    if not gpu:
        values+=[spec['metrics_source']]+list(spec['metrics_module_sources'].values())
    return list({path_key(value['path']):value for value in values}.values())


def run(path):
    """Future original owned execution only; not called by metadata tests."""
    path=Path(path)
    need(path.is_absolute(), 'Absolute bound scientific spec path')
    spec=json.loads(path.read_bytes());gpu=validate_operation(spec)
    source_map=validate_declared_inputs(spec)
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==spec['cuda_visible_devices'],
         'Operation-specific device visibility before scientific imports')
    need(reference(__file__)==spec['entry_source'], 'Execute the exact bound entry source')
    runtime_ref=spec['runtime_source']
    need(reference(runtime_ref['path'])==runtime_ref, 'Exact original resource-control source')
    runtime=_load(runtime_ref['path'],'U40_E50_ORIGINAL_RESOURCE_CONTROL')
    admission_path=Path(spec['admission_file'])
    admission=json.loads(admission_path.read_bytes())
    keys={'status','phase','driver_sha256','entry','entry_args','allowed_cpus','thread_limit',
          'cuda_visible_devices','whole_GPU_used_limit_MiB','qualified_leaf_deadline_seconds',
          'automatic_retry','scientific_spec','bound_source_sha256'}
    need(set(admission)==keys and len(admission)==13
         and admission['phase']==spec['phase'] and admission['scientific_spec']==reference(path)
         and admission['entry']==spec['entry_source']['path']
         and admission['entry_args']==['--spec',str(path),'--owner-execute']
         and admission['allowed_cpus']==[0,1] and admission['thread_limit']==2
         and admission['cuda_visible_devices']==spec['cuda_visible_devices']
         and admission['whole_GPU_used_limit_MiB']==8192
         and admission['qualified_leaf_deadline_seconds']==1800 and admission['automatic_retry'] is False
         and admission['bound_source_sha256']=={value['path']:value['sha256'] for value in spec['sources']+[reference(path)]},
         'Exact original thirteen-field owned admission and full declared source map')
    c=runtime.Control(spec)
    c.execution_spec_ref=reference(path);c.execution_admission_ref=reference(admission_path)
    c.check()
    pre=json.loads(Path(c.closed(spec['prelock'])).read_bytes());validate_prelock(spec,pre)
    parent=json.loads(Path(c.closed(spec['parent_ROOT_qualified'])).read_bytes())
    need(parent['qualified'] is True and parent['normal_RAW_count']==spec['normal_RAW_count']
         and parent['total_failed_RAW_count']==spec['total_failed_RAW_count']
         and parent['total_terminal_identity_pin_count']==spec['total_terminal_identity_pin_count']
         and parent['legacy_pin_count']==spec['legacy_pin_count']
         and len(parent['RAW_inputs'])==spec['normal_RAW_count']
         and len(parent['terminal_identity_pins'])==spec['legacy_pin_count'],
         'Current actual accounting parent; independent of the older E50 model parent')
    # Close all explicit Python code before Torch/NumPy imports. This does not
    # scan or claim byte admission of the entire installed framework environment.
    code_refs=_code_references(spec,gpu)
    for value in code_refs:
        source=Path(c.closed(value))
        need(reference(source)==value, 'Declared injected source bytes changed')
    core_role='prediction_core_source' if gpu else 'metrics_source'
    need(spec[core_role]['sha256']==CORE_PINS[core_role], 'Unchanged published fixed50 numerical core')
    output=Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and not output.is_symlink()
         and c.psutil.disk_usage(str(output.parent)).free>=2*1024**3, 'Fresh owned core output with storage margin')
    import torch
    import numpy as np
    need(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5'
         and not torch.cuda.is_initialized(), 'Pinned genuine scientific modules; CUDA initially cold')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    if gpu:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    body=_load(c.closed(spec['body_source']),'U40_E50_THIN_EVAL_BODY')
    packet=None
    try:
        c.check();packet=body.run(torch,np,spec,c,pre);c.check()
    finally:
        # Core owns the allocator cap and arm-by-arm aliases. This outer cleanup
        # performs a second actual check without initializing CUDA in CPU mode.
        gc.collect()
        if gpu and torch.cuda.is_initialized():
            torch.cuda.synchronize(0);torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache();torch.cuda.synchronize(0)
        elif not gpu:
            need(not torch.cuda.is_initialized(), 'CPU operation must stay CUDA cold, including failure paths')
    need(type(packet) is dict and set(packet)=={'result','wrapper_report_ref','core_peak_allocated_bytes'},
         'Exact thin-body return; original core result stays intact')
    result=packet['result'];owned=validate_result(spec,result)
    wrapper=packet['wrapper_report_ref'];strict_ref(wrapper)
    need(output.resolve() in Path(wrapper['path']).resolve().parents, 'Owned wrapper provenance report')
    for value in owned+[wrapper]:
        need(c.ref(value['path'])==value, 'All returned output reference bytes exact')
    if gpu:
        need(torch.cuda.is_initialized(), 'Successful prediction must include actual CUDA execution')
        peak=max(int(packet['core_peak_allocated_bytes']),int(torch.cuda.max_memory_allocated(0)))
        cleanup=dict(CUDA_initialized=True,allocated_after=int(torch.cuda.memory_allocated(0)),
            reserved_after=int(torch.cuda.memory_reserved(0)),peak_allocated_bytes=peak,
            peak_scope='Largest recorded per-arm measurement and final allocator interval; core resets occur')
        need(cleanup['allocated_after']==cleanup['reserved_after']==0 and 0<peak<=4096*1024**2,
             'Core and outer actual CUDA cleanup must both pass within the fixed allocator cap')
    else:
        need(packet['core_peak_allocated_bytes']==0 and not torch.cuda.is_initialized(), 'No CPU Torch tensor/model/CUDA initialization')
        cleanup=dict(CUDA_initialized=False,Torch_imported_for_cold_check=True,
                     Torch_tensor_model_optimizer_initialization=False)
    for value in code_refs:
        need(reference(value['path'])==value, 'Declared code source unchanged after evaluation')
    c.check()
    q=dict(schema='U40-owned-pending-fixed50-full-method-DEV-v1',
        status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',operation=spec['operation'],phase=spec['phase'],
        source=spec['entry_source'],body_source=spec['body_source'],runtime_source=spec['runtime_source'],
        spec=c.execution_spec_ref,admission=c.execution_admission_ref,prelock=spec['prelock'],
        parent_ROOT_qualified=spec['parent_ROOT_qualified'],training_qualified=spec['training_qualified'],
        checkpoint_refs=spec['checkpoint_refs'],pair_commit_ref=spec['pair_commit_ref'],order_ref=spec['order_ref'],
        dev_context=spec['dev_context'],training_sources=spec['training_sources'],prediction_sources=spec['prediction_sources'],
        prediction_core_source=spec['prediction_core_source'],metrics_source=spec.get('metrics_source'),
        result=result,wrapper_report_ref=wrapper,seed=SEED,fixed_epoch=50,updates_per_arm=37050,
        original_image_references_metadata_sha256=pre['original_image_references_metadata_sha256'],
        owner_index=spec['owner_index'],normal_RAW_count=spec['normal_RAW_count'],
        total_failed_RAW_count=spec['total_failed_RAW_count'],
        total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],legacy_pin_count=spec['legacy_pin_count'],
        new_training=False,DEV=True,AP=not gpu,TEST=False,utility_gate_decided=False,
        elapsed_seconds=time.monotonic()-c.started,peak_RSS_bytes=c.peak_rss,
        minimum_available_RAM_bytes=c.minimum_available,cleanup=cleanup,
        qualified=False,original_actual0_self_certified=False,innovation_established=False,goal_complete=False,
        source_scope='Declared closed Python sources and pinned runtime versions; no full-environment byte audit')
    qref=c.write_json(output/'Q.json',q)
    print(json.dumps(dict(status=q['status'],Q=qref,qualified=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();need(args.owner_execute,'Root owns the sole original leaf')
    run(args.spec)
