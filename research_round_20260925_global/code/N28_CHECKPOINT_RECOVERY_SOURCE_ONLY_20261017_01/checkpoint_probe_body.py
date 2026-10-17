"""UNBOUND CPU-only recovery probe for a preserved failed N28 training task.

No scientific imports, checkpoint reads or I/O occur at import. Root alone may
execute run after independent admission. No forward/backward/optimizer update,
teacher/data access or CUDA initialization occurs. E05 is checked as an original
partial state, never as successful completion of the failed requested E08 task.
"""
from copy import deepcopy
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import time


SEED=202610172801
END=5
UPDATES=3705
MODES=['SELECTIVE','WHOLE_NODE']


def need(ok,message):
    if not ok:
        raise ValueError(message)


def read(c,reference):
    return json.loads(Path(c.closed(reference)).read_bytes())


def load(c,reference,name):
    module_spec=importlib.util.spec_from_file_location(name,str(c.closed(reference)))
    module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def finite(value):
    need(type(value) in (int,float) and math.isfinite(value),'Finite plain scalar required')
    return float(value)


def validate_prelock(pre):
    """Fixed recovery-only CPU scope; Root may add independent plan metadata."""
    fixed=dict(schema='N28-dense-routing-checkpoint-recovery-prelock-v1',seed=SEED,
        completed_epoch=END,cumulative_updates_per_arm=UPDATES,images_per_epoch=741,GT_per_epoch=1661,
        modes=MODES,lr=3e-4,weight_decay=1e-4,betas=[.9,.999],eps=1e-8,optimizer='AdamW',
        no_new_training=True,no_AP=True,no_DEV=True,no_TEST=True,CUDA_RNG_restore_deferred=True)
    need(type(pre) is dict and all(key in pre and type(pre[key]) is type(value)
         and pre[key]==value for key,value in fixed.items()),'Exact fixed CPU recovery prelock ABI')


def validate_completion_log(text,history,seed):
    """Public pure-metadata join: every complete stdout epoch, no loss selection."""
    need(type(text) is str and type(history) is list and len(history)==END and seed==SEED,
         'Exact original five-epoch completion-log scope')
    records=[]
    for line in text.splitlines():
        try:
            value=json.loads(line)
        except (TypeError,ValueError):
            need('"N28_seed"' not in line,'Incomplete or malformed training-completion JSON')
            continue
        if type(value) is dict and 'N28_seed' in value:
            records.append(value)
    need(len(records)==END,'Exactly five original completed-epoch log records')
    for epoch,(record,row) in enumerate(zip(records,history),1):
        need(set(record)=={'N28_seed','completed_epoch','cumulative_updates_per_arm','mean_image_loss'}
             and type(record['N28_seed']) is int and record['N28_seed']==seed
             and type(record['completed_epoch']) is int and record['completed_epoch']==row['epoch']==epoch
             and type(record['cumulative_updates_per_arm']) is int
             and record['cumulative_updates_per_arm']==row['cumulative_updates_per_arm']==epoch*741
             and set(record['mean_image_loss'])==set(MODES),
             'Original full-seed, contiguous completion sequence and update counts')
        for mode in MODES:
            need(finite(record['mean_image_loss'][mode]).hex()==
                 finite(row['arms'][mode]['mean_image_loss']).hex(),
                 'Every original logged mean loss must exactly join checkpoint history')
    return records


def check_history(history):
    need(type(history) is list and len(history)==END,'Original complete history1..5')
    parameter_tensors=None
    for epoch,row in enumerate(history,1):
        need(type(row['epoch']) is int and row['epoch']==epoch
             and row['images']==741 and row['GT_visits']==1661 and row['microbatch']==4
             and row['cumulative_updates_per_arm']==epoch*741
             and row['full_mask_writable'] is True and row['teacher_updated'] is False
             and row['optimizer_updates_per_image_per_arm']==1
             and row['loss_weighting']=='chunk_GT/image_GT; fixed microbatch BN semantics'
             and set(row['arms'])==set(MODES),'Original per-epoch FIT/count/loss scope')
        order=row['image_order']
        need(type(order) is list and len(order)==741
             and all(type(index) is int for index in order) and set(order)==set(range(741)),
             'Every recorded original epoch order is a complete741 permutation')
        need(type(row['shared_microbatches']) is int and row['shared_microbatches']>=741,
             'Original fixed-microbatch coverage')
        current={}
        for mode in MODES:
            arm=row['arms'][mode]
            current[mode]=arm['active_parameter_tensors']
            need(type(current[mode]) is int and current[mode]>0
                 and arm['updates']==741 and arm['GT_visits']==1661
                 and arm['microbatches']==row['shared_microbatches']
                 and arm['finite_gradient_checks']==741*current[mode],
                 'Original all-parameter finite-gradient and image/GT counters')
            for name in ('image_loss_sum','instance_loss_sum','mean_image_loss','mean_instance_loss'):
                need(finite(arm[name])>=0,'Original finite nonnegative training losses')
            need(arm['mean_image_loss']==arm['image_loss_sum']/741
                 and arm['mean_instance_loss']==arm['instance_loss_sum']/1661,
                 'Original image and instance loss denominators')
        need(current[MODES[0]]==current[MODES[1]],'Paired parameter-tensor counts')
        if parameter_tensors is None:
            parameter_tensors=current
        else:
            need(current==parameter_tensors,'Original fixed parameter tensor population across epochs')
    json.dumps(history,allow_nan=False)
    return parameter_tensors


def tensor_digest(value):
    array=value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape),dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def state_digest(state):
    rows=[dict(name=name,**tensor_digest(value)) for name,value in state.items()]
    return dict(tensors=rows,sha256=hashlib.sha256(
        json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest())


def cpu_tensor(torch,value,shape=None,dtype=None):
    need(isinstance(value,torch.Tensor) and value.device.type=='cpu' and not value.requires_grad
         and bool(torch.isfinite(value).all()) and (shape is None or tuple(value.shape)==tuple(shape))
         and (dtype is None or value.dtype==dtype),'Exact finite detached CPU tensor required')
    return value


def optimizer_recipe(group):
    need(group['lr']==3e-4 and group['weight_decay']==1e-4
         and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8
         and group['foreach'] is False and group['fused'] is False
         and group['amsgrad'] is False and group['maximize'] is False
         and group['capturable'] is False and group['differentiable'] is False,
         'Complete unchanged original AdamW recipe')


def check_raw_optimizer(torch,head,value):
    need(type(value) is dict and set(value)=={'state','param_groups'}
         and type(value['param_groups']) is list and len(value['param_groups'])==1,
         'Original complete single-group AdamW state_dict')
    group=value['param_groups'][0];optimizer_recipe(group)
    parameters=list(head.parameters());ids=group['params']
    need(type(ids) is list and len(ids)==len(parameters) and len(set(ids))==len(ids)
         and all(type(index) is int for index in ids) and set(value['state'])==set(ids),
         'Every original serialized parameter has exactly one complete optimizer state')
    for identifier,parameter in zip(ids,parameters):
        state=value['state'][identifier]
        need(type(state) is dict and set(state)=={'step','exp_avg','exp_avg_sq'},'Original AdamW moment keys')
        step=cpu_tensor(torch,state['step'])
        need(step.numel()==1 and float(step.item())==UPDATES,'Original raw per-parameter step3705')
        for name in ('exp_avg','exp_avg_sq'):
            cpu_tensor(torch,state[name],parameter.shape,parameter.dtype)
    return ids


def check_loaded_optimizer(torch,head,optimizer,raw,ids,expected_parameter_tensors):
    parameters=list(head.parameters())
    owned=[p for group in optimizer.param_groups for p in group['params']]
    need(len(owned)==len(parameters)==expected_parameter_tensors
         and len({id(p) for p in owned})==len(parameters)
         and {id(p) for p in owned}=={id(p) for p in parameters}
         and len(optimizer.state)==len(parameters),'Original CPU optimizer ownership/state completeness')
    for group in optimizer.param_groups:optimizer_recipe(group)
    rows=[]
    for identifier,parameter in zip(ids,parameters):
        state=optimizer.state[parameter]
        need(set(state)=={'step','exp_avg','exp_avg_sq'},'Loaded AdamW complete moments')
        step=cpu_tensor(torch,state['step'])
        need(step.numel()==1 and float(step.item())==UPDATES,'Every loaded parameter step3705')
        entry=dict(serialized_parameter_id=identifier)
        for name in ('step','exp_avg','exp_avg_sq'):
            value=state[name]
            if name!='step':cpu_tensor(torch,value,parameter.shape,parameter.dtype)
            need(tensor_digest(value)==tensor_digest(raw['state'][identifier][name]),
                 'Optimizer loading must preserve every original CPU state tensor byte')
            entry[name]=tensor_digest(value)
        rows.append(entry)
    return dict(parameter_tensors=len(parameters),state_entries=len(optimizer.state),expected_step=UPDATES,
        all_parameter_steps_exact=True,recipe_exact=True,raw_and_loaded_moments_shape_dtype_finite=True,
        raw_and_loaded_state_tensors_bitexact=True,
        state_tensor_digest=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest())


def run(torch,np,spec,c,pre):
    """PENDING Root-owned actual CPU execution; no scientific probe has run here."""
    started=time.monotonic();c.check()
    need(not torch.cuda.is_initialized(),'CPU recovery must never initialize CUDA')
    original_rng=torch.get_rng_state().clone()
    binding=read(c,spec['failed_training_binding'])
    failure=read(c,spec['failed_ROOT_preserved'])
    need(binding['schema']=='ROOT-N28-dense-routing-train-binding-v1'
         and binding['key']=='N28_TRAIN_202610172801_E08'
         and failure['schema']=='ROOT-N28-dense-routing-controlled-training-failure-preserved-v1'
         and failure['qualified'] is True
         and failure['qualified_scope']=='TERMINAL_IDENTITY_HISTORY_AND_ACCOUNTING_ONLY'
         and failure['status']=='RESOURCE_ABORT_RECORDED_RECOVERY_NOT_YET_QUALIFIED'
         and failure['seed']==SEED and failure['requested_start_epoch']==0
         and failure['requested_end_epoch']==8
         and failure['original_training_stage_successful'] is False
         and failure['partial_checkpoints_qualified'] is False,
         'Only the preserved failed original requested0..8 task; partial states are not already qualified')
    need(all(binding[key]==failure[key] for key in
             ('spec','admission','source','body_source','prelock','phase'))
         and binding['out']==failure['partial_output_directory'],
         'Original failure and binding exact source/spec/admission/phase/output association')
    need(type(pre) is dict and pre==read(c,spec['prelock']),'Exact independent recovery prelock')
    validate_prelock(pre)
    checkpoint_path=Path(c.closed(spec['recovery_checkpoint']))
    stdout_path=Path(c.closed(spec['original_stdout']))
    original_output=Path(binding['out'])
    need(checkpoint_path.resolve()==(original_output/'TRAIN_STATE_E05.pth').resolve()
         and stdout_path.resolve()==(Path(binding['original_driver']).parent/'runs'/binding['phase']/'stdout.log').resolve(),
         'Only original E05 and original owned stdout, no copied/relabelled checkpoint')
    # File-name metadata only: earlier checkpoints are not opened or hashed.
    inventory=[]
    for path in original_output.iterdir():
        match=re.fullmatch(r'TRAIN_STATE_E(\d{2})\.pth',path.name)
        if match:
            need(path.is_file() and not path.is_symlink(),'Original checkpoint filename must be a regular file')
            inventory.append((int(match.group(1)),path.name))
    inventory.sort()
    need(len({epoch for epoch,_ in inventory})==len(inventory),'Unique checkpoint epoch filenames')
    saved_epochs={epoch for epoch,_ in inventory}
    checkpoint=torch.load(str(checkpoint_path),map_location='cpu')
    need(checkpoint['schema']=='N28-dense-routing-paired-train-state-v1'
         and type(checkpoint['seed']) is int and checkpoint['seed']==SEED
         and type(checkpoint['completed_epoch']) is int and checkpoint['completed_epoch']==END
         and type(checkpoint['cumulative_updates_per_arm']) is int
         and checkpoint['cumulative_updates_per_arm']==UPDATES
         and checkpoint['modes']==MODES and set(checkpoint['models'])==set(checkpoint['optimizers'])==set(MODES)
         and checkpoint['factory_source']==spec['factory_source']
         and checkpoint['train_core_source']==spec['train_core_source']
         and checkpoint['reader_source']==spec['training_reader_source'],
         'Original E05 schema/full seed/source identities and complete paired state')
    history=deepcopy(checkpoint['history']);parameter_tensors=check_history(history)
    completions=validate_completion_log(stdout_path.read_text(encoding='utf-8-sig'),history,SEED)
    completed_epochs={row['completed_epoch'] for row in completions}
    contiguous=0
    while contiguous+1 in saved_epochs and contiguous+1 in completed_epochs:
        contiguous+=1
    need(contiguous==END and set(range(1,END+1))<=saved_epochs,
         'E05 is the largest continuous checkpoint-filename/completion-log endpoint, never loss-selected')
    factory=load(c,spec['factory_source'],'N28_RECOVERY_ORIGINAL_CPU_FACTORY')
    c.closed(spec['train_core_source']);c.closed(spec['training_reader_source'])
    heads={};optimizers={};model_digests={};optimizer_checks={}
    try:
        with torch.random.fork_rng(devices=[]):
            for mode in MODES:
                torch.random.default_generator.manual_seed(SEED)
                heads[mode]=factory.make_head(torch,mode)
        need(torch.equal(torch.get_rng_state(),original_rng) and not torch.cuda.is_initialized(),
             'CPU-only constructors preserve external global RNG')
        for mode in MODES:
            head=heads[mode];state=checkpoint['models'][mode]
            expected=head.state_dict()
            need(list(state)==list(expected),'Exact original parameter and BN-buffer state keys')
            for name,value in expected.items():
                cpu_tensor(torch,state[name],value.shape,value.dtype)
            head.load_state_dict(state,strict=True);head.eval()
            need(sum(p.numel() for p in head.parameters())==2978555
                 and len(list(head.parameters()))==parameter_tensors[mode]
                 and all(p.device.type=='cpu' and p.dtype==torch.float32 and p.grad is None
                         and bool(torch.isfinite(p).all()) for p in head.parameters()),
                 'Original finite CPU dense head capacity with no gradients')
            loaded=head.state_dict()
            for name,value in loaded.items():cpu_tensor(torch,value,state[name].shape,state[name].dtype)
            raw_digest=state_digest(state);loaded_digest=state_digest(loaded)
            need(raw_digest==loaded_digest,'Strict CPU load preserves every parameter and BN-buffer byte')
            model_digests[mode]=loaded_digest
            optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-4,
                betas=(.9,.999),eps=1e-8,amsgrad=False,maximize=False,capturable=False,
                differentiable=False,foreach=False,fused=False)
            raw_optimizer=checkpoint['optimizers'][mode]
            ids=check_raw_optimizer(torch,head,raw_optimizer)
            optimizer.load_state_dict(raw_optimizer)
            optimizer_checks[mode]=check_loaded_optimizer(torch,head,optimizer,raw_optimizer,ids,parameter_tensors[mode])
            optimizers[mode]=optimizer
            del head,state,expected,loaded,raw_digest,loaded_digest,optimizer,raw_optimizer,ids
        saved_order=cpu_tensor(torch,checkpoint['order_generator_state'],dtype=torch.uint8)
        saved_cpu=cpu_tensor(torch,checkpoint['torch_CPU_RNG_state'],dtype=torch.uint8)
        saved_cuda=cpu_tensor(torch,checkpoint['torch_CUDA_RNG_state'],dtype=torch.uint8)
        need(saved_order.ndim==saved_cpu.ndim==saved_cuda.ndim==1
             and min(saved_order.numel(),saved_cpu.numel(),saved_cuda.numel())>0,'Original one-dimensional CPU RNG payloads')
        private_order=torch.Generator(device='cpu')
        private_order.set_state(saved_order)
        need(torch.equal(private_order.get_state(),saved_order),'Actual private CPU order-generator restoration exact')
        torch.set_rng_state(saved_cpu)
        need(torch.equal(torch.get_rng_state(),saved_cpu),'Actual global CPU RNG restoration exact')
        rng_digests=dict(private_order=tensor_digest(saved_order),torch_CPU=tensor_digest(saved_cpu),
                         torch_CUDA_payload_only=tensor_digest(saved_cuda))
        need(not torch.cuda.is_initialized(),'No CUDA call/state restoration in CPU recovery')
        del private_order,saved_order,saved_cpu,saved_cuda
    finally:
        torch.set_rng_state(original_rng)
        optimizers.clear();heads.clear();gc.collect()
    need(torch.equal(torch.get_rng_state(),original_rng) and not torch.cuda.is_initialized(),
         'External global CPU RNG restored and CUDA remains uninitialized')
    need(c.ref(spec['recovery_checkpoint']['path'])==spec['recovery_checkpoint']
         and c.ref(spec['original_stdout']['path'])==spec['original_stdout'],
         'Original failed-task checkpoint/log bytes remain unchanged')
    output=Path(spec['output_directory']);output.mkdir()
    report=dict(schema='N28-original-E05-checkpoint-CPU-recovery-report-pending-v1',
        status='CPU_STATE_LOAD_AND_RESTORE_CHECKS_COMPLETE_PENDING_ORIGINAL_ACTUAL0_ROOT_ACCEPTANCE',
        source=c.ref(Path(__file__)),spec=c.execution_spec_ref,CPU_admission=c.execution_admission_ref,
        phase=spec['phase'],prelock=spec['prelock'],factory_source=spec['factory_source'],
        train_core_source=spec['train_core_source'],training_reader_source=spec['training_reader_source'],
        failed_training_binding=spec['failed_training_binding'],failed_ROOT_preserved=spec['failed_ROOT_preserved'],
        original_stdout=spec['original_stdout'],checkpoint_ref=spec['recovery_checkpoint'],
        seed=SEED,end_epoch=END,cumulative_updates_per_arm=UPDATES,modes=MODES,history=history,
        requested_start_epoch=0,requested_end_epoch=8,original_training_stage_successful=False,
        original_requested_endpoint_completed=False,original_failure_preserved=True,
        recovered_original_training_state=True,recovery_is_not_original_stage_success=True,
        parameter_count_per_arm=2978555,model_state_digests=model_digests,optimizer_checks=optimizer_checks,
        optimizer_recipe=dict(type='AdamW',lr=3e-4,weight_decay=1e-4,betas=[.9,.999],eps=1e-8,
            amsgrad=False,maximize=False,capturable=False,differentiable=False,foreach=False,fused=False),
        optimizer_steps_executed=0,forward_calls=0,backward_calls=0,reader_data_loads=0,
        checkpoint_filename_inventory=[dict(epoch=epoch,filename=name) for epoch,name in inventory],
        logged_completion_records=completions,largest_contiguous_saved_and_logged_epoch=contiguous,
        endpoint_selected_by_loss=False,earlier_checkpoint_contents_read=False,
        earlier_checkpoint_integrity_revalidated=False,original_recorded_order_permutations_checked=True,
        history_order_generation_replayed=False,RNG_payload_digests=rng_digests,
        private_order_generator_restored_exact=True,checkpoint_CPU_RNG_restored_exact=True,
        external_CPU_RNG_restored_exact=True,actual_CUDA_RNG_restore_verified=False,
        actual_CUDA_RNG_restore_pending_next_GPU=True,CUDA_initialized=False,
        probe_scope='Actual CPU deserialization and strict model/BN/complete AdamW-state loading plus private/global CPU RNG restoration of the preserved original E05. No optimization replay, next-update equivalence, CUDA RNG/device restoration or successful E08 completion is certified.',
        new_training=False,AP=False,DEV=False,TEST=False,qualified=False,
        original_actual0_self_certified=False,innovation_established=False,goal_complete=False,
        elapsed_seconds=time.monotonic()-started)
    report_ref=c.write_json(output/'CPU_RECOVERY_REPORT.json',report)
    del checkpoint,original_rng
    gc.collect();c.check()
    need(not torch.cuda.is_initialized(),'CPU-only recovery through final return')
    return dict(checkpoint_ref=spec['recovery_checkpoint'],seed=SEED,end_epoch=END,
        cumulative_updates_per_arm=UPDATES,history=history,report_ref=report_ref,
        new_training=False,recovered_original_training_state=True,AP=False,DEV=False,TEST=False)
