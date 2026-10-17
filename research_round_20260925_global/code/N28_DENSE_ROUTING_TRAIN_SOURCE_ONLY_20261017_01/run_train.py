"""Root-owned formal N28 paired dense-routing training in fixed eight-epoch stages.

Body owns CPU construction/resume before CUDA, the actual optimizer updates and
joint stage checkpoint. This entry locks admission, resources and stage scope;
it does not initialize a probe, perturb readouts or evaluate utility. Completion
remains pending the original task's actual terminal and independent Root intake.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


SEEDS = [202610172801,202610172802,202610172803]
MODES = ['SELECTIVE','WHOLE_NODE']
STAGES = ((0,8),(8,16),(16,24))


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def no_torch():
    need(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules),
         'Original admission and source closure must precede Torch import')


def reference(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def load(path, name):
    module_spec = importlib.util.spec_from_file_location(name,str(path))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def output_reference(c, value, output):
    need(type(value) is dict and set(value) == {'path','bytes','sha256'},
         'Completed output requires an exact three-field reference')
    path = Path(value['path'])
    need(path.is_absolute() and output.resolve() in path.resolve().parents
         and not path.is_symlink() and c.ref(path) == value,
         'Completed immutable artifact identity inside owned output')
    return path


def run(path):
    no_torch()
    path = Path(path)
    need(path.is_absolute(), 'Absolute original paired-training spec required')
    spec = json.loads(path.read_bytes())
    need(spec['schema'] == 'ROOT-N28-dense-routing-train-spec-v1'
         and spec['operation'] == 'GPU_DENSE_ROUTING_PAIRED_TRAIN'
         and spec['owner_index'] == spec['normal_RAW_count']
         and spec['total_terminal_identity_pin_count'] == spec['normal_RAW_count']+spec['total_failed_RAW_count']
         and all(type(spec[key]) is int and spec[key] >= 0 for key in
                 ('owner_index','normal_RAW_count','total_failed_RAW_count',
                  'total_terminal_identity_pin_count','legacy_pin_count'))
         and type(spec['seed']) is int and spec['seed'] in SEEDS
         and type(spec['start_epoch']) is type(spec['end_epoch']) is int
         and (spec['start_epoch'],spec['end_epoch']) in STAGES
         and spec['automatic_retry'] is False and spec['deadline_seconds'] == 1800
         and spec['whole_GPU_limit_MiB'] == 8192 and spec['allocator_limit_MiB'] == 4096
         and spec['allowed_cpus'] == [0,1] and spec['thread_limit'] == 2
         and spec['cuda_visible_devices'] == '0'
         and spec['maximum_process_RSS_bytes'] == spec['minimum_available_RAM_bytes'] == 16*1024**3
         and spec['no_new_training'] is False
         and all(spec[key] is True for key in ('no_AP','no_DEV','no_TEST'))
         and 'quality_source' not in spec and 'setup_source' not in spec,
         'Fixed paired-training seed/stage, original accounting and resource limits')
    need(reference(__file__) == spec['entry_source']
         and reference(spec['runtime_source']['path']) == spec['runtime_source'],
         'Actual entry and reusable resource-control source identity')
    runtime = load(spec['runtime_source']['path'],'N28_TRAIN_OWNED_RESOURCE_CONTROL')
    no_torch()
    spec_ref = reference(path)
    admission_path = Path(spec['admission_file'])
    need(admission_path.is_absolute(), 'Absolute own original admission')
    ad = json.loads(admission_path.read_bytes())
    admission_ref = reference(admission_path)
    keys = {'status','phase','driver_sha256','entry','entry_args','allowed_cpus','thread_limit',
            'cuda_visible_devices','whole_GPU_used_limit_MiB','qualified_leaf_deadline_seconds',
            'automatic_retry','scientific_spec','bound_source_sha256'}
    need(set(ad) == keys and len(ad) == 13 and ad['scientific_spec'] == spec_ref
         and ad['entry'] == spec['entry_source']['path']
         and ad['entry_args'] == ['--spec',str(path),'--owner-execute']
         and ad['phase'] == spec['phase'] and ad['allowed_cpus'] == [0,1] and ad['thread_limit'] == 2
         and ad['cuda_visible_devices'] == '0' and ad['automatic_retry'] is False
         and ad['whole_GPU_used_limit_MiB'] == 8192 and ad['qualified_leaf_deadline_seconds'] == 1800
         and ad['bound_source_sha256'] == {r['path']:r['sha256'] for r in spec['sources']+[spec_ref]},
         'Exact thirteen-field own admission and complete direct binding map')
    c = runtime.Control(spec)
    c.execution_spec_ref,c.execution_admission_ref = spec_ref,admission_ref
    c.check()
    need(len(c.allowed) == len(spec['sources']), 'Unique direct input paths')
    roles = ('entry_source','body_source','factory_source','reader_source','train_core_source','runtime_source')
    for key in roles:
        c.closed(spec[key])
    c.closed(spec['context'])
    pre = json.loads(c.closed(spec['prelock']).read_bytes())
    need(pre['schema'] == 'N28-dense-routing-paired-train-prelock-v1'
         and pre['images'] == 741 and pre['GT'] == 1661
         and pre['epochs'] == 24 and pre['stage_epochs'] == 8 and pre['microbatch'] == 4
         and pre['updates_per_epoch'] == 741 and pre['total_updates_per_arm'] == 17784
         and pre['seeds'] == SEEDS and all(type(seed) is int for seed in pre['seeds'])
         and pre['modes'] == MODES and pre['lr'] == 3e-4 and pre['weight_decay'] == 1e-4
         and pre['betas'] == [.9,.999] and pre['eps'] == 1e-8,
         'Complete FIT741/1661, fixed24 training, eight-epoch stages and paired optimizer recipe')
    parent = json.loads(c.closed(spec['parent_ROOT_qualified']).read_bytes())
    need(parent['qualified'] is True and parent['normal_RAW_count'] == spec['normal_RAW_count']
         and parent['total_failed_RAW_count'] == spec['total_failed_RAW_count'],
         'Current actually Root-qualified terminal predecessor')
    output = Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists()
         and not output.is_symlink() and c.psutil.disk_usage(str(output.parent)).free >= 2*1024**3,
         'Fresh owned training-stage output and storage margin')
    no_torch()
    import torch
    import numpy as np
    need(torch.__version__ == '2.0.1+cu118' and np.__version__ == '1.23.5'
         and not torch.cuda.is_initialized(), 'Pinned original runtime with cold CUDA')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    c.check()
    body = load(c.closed(spec['body_source']),'N28_OWNED_DENSE_ROUTING_PAIRED_TRAIN_BODY')
    need(not torch.cuda.is_initialized(), 'Body import preserves cold CUDA before CPU construction/resume')
    result = body.run(torch,np,spec,c,pre)
    c.check()
    need(type(result) is dict and output.is_dir()
         and type(result['seed']) is int and result['seed'] == spec['seed']
         and type(result['end_epoch']) is int and result['end_epoch'] == spec['end_epoch']
         and type(result['cumulative_updates_per_arm']) is int
         and result['cumulative_updates_per_arm'] == spec['end_epoch']*741
         and result['new_training'] is True
         and all(result[key] is False for key in ('qualified','AP','DEV','TEST')),
         'Completed exact training stage remains pending actual0 and Root intake')
    report_ref,checkpoint_ref = result['report_ref'],result['checkpoint_ref']
    report_path = output_reference(c,report_ref,output)
    checkpoint_path = output_reference(c,checkpoint_ref,output)
    need(report_path.resolve() != checkpoint_path.resolve(), 'Separate report and paired stage checkpoint')
    need(torch.cuda.is_initialized(), 'Formal paired training must actually execute on CUDA')
    peak = torch.cuda.max_memory_allocated(0)
    gc.collect()
    torch.cuda.synchronize(0)
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    cleanup = dict(CUDA_initialized=True,allocated_after=torch.cuda.memory_allocated(0),
                   reserved_after=torch.cuda.memory_reserved(0),peak_allocated_bytes=peak)
    need(cleanup['allocated_after'] == cleanup['reserved_after'] == 0 and peak <= 4096*1024**2,
         'Complete GPU release including cuBLAS and allocator ceiling')
    for key in roles:
        need(reference(spec[key]['path']) == spec[key], 'Owned source unchanged at completion')
    output_reference(c,report_ref,output)
    output_reference(c,checkpoint_ref,output)
    c.check()
    q = dict(schema='N28-owned-pending-dense-routing-paired-train-v1',
        status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
        operation=spec['operation'],phase=spec['phase'],source=spec['entry_source'],
        body_source=spec['body_source'],factory_source=spec['factory_source'],
        reader_source=spec['reader_source'],train_core_source=spec['train_core_source'],runtime_source=spec['runtime_source'],
        spec=spec_ref,admission=admission_ref,context=spec['context'],prelock=spec['prelock'],
        parent_ROOT_qualified=spec['parent_ROOT_qualified'],result=result,
        seed=spec['seed'],start_epoch=spec['start_epoch'],end_epoch=spec['end_epoch'],
        epochs_this_task=spec['end_epoch']-spec['start_epoch'],
        cumulative_updates_per_arm=result['cumulative_updates_per_arm'],
        full_24_epochs_complete=(spec['end_epoch'] == 24),
        owner_index=spec['owner_index'],normal_RAW_count=spec['normal_RAW_count'],
        total_failed_RAW_count=spec['total_failed_RAW_count'],
        total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],legacy_pin_count=spec['legacy_pin_count'],
        qualified=False,original_actual0_self_certified=False,new_training=True,
        FIT=True,DEV=False,TEST=False,AP=False,utility_evaluated=False,
        innovation_established=False,goal_complete=False,cleanup=cleanup,
        elapsed_seconds=time.monotonic()-c.started,peak_RSS_bytes=c.peak_rss,
        minimum_available_RAM_bytes=c.minimum_available)
    qref = c.write_json(output/'Q.json',q)
    print(json.dumps(dict(status=q['status'],Q=qref,qualified=False,new_training=True,
                          full_24_epochs_complete=q['full_24_epochs_complete'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true')
    args = parser.parse_args()
    need(args.owner_execute,'Root owns the sole scientific paired-training leaf')
    run(args.spec)
