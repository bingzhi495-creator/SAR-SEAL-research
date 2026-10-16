# encoding: utf-8
"""UNBOUND_NOT_EXECUTED N26 quality-only training. No scientific imports at import."""
import gc
import hashlib
import json
from pathlib import Path
import time
import types

SEEDS = [202610172601, 202610172602, 202610172603]
MERGED_KEYS = {'FIT_features', 'FIT_targets_FP64', 'FIT_identity_order',
               'DEV_features', 'DEV_prediction_order', 'R_prediction_reference'}
IDENTITY_KEYS = {'dataset_index', 'image_id', 'GT_index', 'GT_annotation_id', 'query_id'}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(c, ref):
    return json.loads(Path(c.closed(ref)).read_text(encoding='utf-8-sig'))


def load(c, ref, name):
    path = Path(c.closed(ref)); data = path.read_bytes()
    need(len(data) == ref['bytes'] and hashlib.sha256(data).hexdigest() == ref['sha256'],
         'Closed source changed: ' + name)
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


def validate_prelock(pre, spec):
    fixed = dict(schema='N26-fixed-query-mask-quality-train-prelock-v1', images=741, GT=1661,
        epochs=24, batch_size=64, steps_per_epoch=26, total_updates=624, seeds=SEEDS,
        AdamW_lr=.0003, AdamW_weight_decay=.0001, AdamW_betas=[.9,.999], AdamW_eps=1e-8)
    need(all(pre[k] == v for k,v in fixed.items())
         and type(pre['seeds']) is list and all(type(s) is int for s in pre['seeds']),
         'Exact N26 training prelock required')
    need(spec['operation'] == 'GPU_QUERY_MASK_QUALITY_TRAIN'
         and type(spec['seed']) is int and spec['seed'] in SEEDS and spec['seed'] in pre['seeds'],
         'Only the three complete prelocked N26 seeds; execution requires separate Root authorization')
    need(spec['allocator_limit_MiB'] == 4096, 'Exact four-GiB allocator limit required')


def tensor(torch, value, shape, dtype, label):
    need(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(shape)
         and value.dtype == dtype and value.device.type == 'cpu' and not value.requires_grad
         and bool(torch.isfinite(value).all()), label)
    return value


def cpu_tree(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {k:cpu_tree(torch,v) for k,v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(torch,v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(torch,v) for v in value)
    need(value is None or type(value) in (str,int,float,bool), 'Only tensors and basic training-state values')
    return value


def state_digest(state):
    records = []
    for name,value in state.items():
        a = value.detach().cpu().contiguous().numpy()
        records.append(dict(name=name,shape=list(a.shape),dtype=str(a.dtype),
                            sha256=hashlib.sha256(a.tobytes()).hexdigest()))
    whole = hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(sha256=whole,tensors=records)


def save(torch, c, path, value):
    with path.open('xb') as stream:
        torch.save(value, stream)
    return c.ref(path)


def clear_cuda(torch):
    gc.collect(); torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    result = dict(allocated_after=int(torch.cuda.memory_allocated(0)),
                  reserved_after=int(torch.cuda.memory_reserved(0)))
    need(result['allocated_after'] == result['reserved_after'] == 0, 'Quality-head CUDA aliases must be released')
    return result


def run(torch, np, spec, c, pre):
    started = time.monotonic(); c.check(); validate_prelock(pre,spec)
    need(pre == read(c,spec['prelock']), 'Caller prelock must equal its pinned JSON')
    need(torch.__version__ == '2.0.1+cu118' and torch.get_num_threads() == 2
         and not torch.cuda.is_initialized(), 'Pinned Torch/two threads and cold CUDA required')
    need(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
         and not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic,
         'Original deterministic FP32 runtime policy required')
    accepted = read(c,spec['inputs_qualified'])
    need(accepted['qualified'] is True and accepted['quality_training_eligible'] is True
         and accepted['result']['merged_ref'] == spec['merged_inputs'],
         'Independent Root-qualified merged quality inputs must be joined before loading')
    merged = torch.load(str(c.closed(spec['merged_inputs'])),map_location='cpu',weights_only=True)
    need(type(merged) is dict and set(merged) == MERGED_KEYS, 'Exact six-key merged input container required')
    X = tensor(torch,merged['FIT_features'],(1661,1040),torch.float32,'Full detached FIT features')
    y64 = tensor(torch,merged['FIT_targets_FP64'],(1661,),torch.float64,'Full matched FP64 FIT targets')
    need(bool(((y64 >= 0) & (y64 <= 1)).all()), 'All matched targets retained within [0,1]')
    identities = merged['FIT_identity_order']
    need(type(identities) is list and len(identities) == 1661
         and all(type(r) is dict and set(r) == IDENTITY_KEYS
                 and all(type(v) is int for v in r.values())
                 and 0 <= r['dataset_index'] < 741 and 0 <= r['query_id'] < 100
                 and r['GT_index'] >= 0 for r in identities)
         and {r['dataset_index'] for r in identities} == set(range(741))
         and len({(r['image_id'],r['GT_annotation_id']) for r in identities}) == 1661
         and len({(r['image_id'],r['query_id']) for r in identities}) == 1661,
         'All741/1661 original matched identities; no filtering or duplicate query')
    # The merged file also deserializes DEV tensors. They are neither accessed nor used here.
    y = y64.to(dtype=torch.float32)
    quality = load(c,spec['quality_source'],'N26_TRAIN_QUALITY_FACTORY')
    cpu_external = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(spec['seed'])
        initialization_rng_before = torch.get_rng_state().clone()
        head = quality.build_head(torch)
        initialization_rng_after = torch.get_rng_state().clone()
    need(not torch.cuda.is_initialized() and torch.equal(cpu_external,torch.get_rng_state()),
         'CPU standard Linear initialization must restore external RNG and leave CUDA cold')
    initial_state = cpu_tree(torch,head.state_dict()); initial_digest = state_digest(initial_state)
    need(len(initial_state) == 6 and sum(p.numel() for p in head.parameters()) == 299521
         and all(p.device.type == 'cpu' and p.dtype == torch.float32 and p.requires_grad
                 for p in head.parameters()), 'Exactly299521 trainable quality parameters in six tensors')
    order_generator = torch.Generator(device='cpu').manual_seed(spec['seed'])
    order_initial = order_generator.get_state().clone()
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(4096*1024**2/torch.cuda.get_device_properties(0).total_memory,0)
    torch.cuda.reset_peak_memory_stats(0)
    cuda_external = torch.cuda.get_rng_state(0).cpu().clone()
    head.to('cuda:0'); head.train()
    optimizer = torch.optim.AdamW(head.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,
                                 weight_decay=.0001,foreach=False,fused=False)
    output = Path(spec['output_directory']); output.mkdir()
    binding = dict(source=c.ref(Path(__file__)),spec=c.execution_spec_ref,
        GPU_admission=c.execution_admission_ref,phase=spec['phase'],prelock=spec['prelock'],
        merged_inputs=spec['merged_inputs'],inputs_qualified=spec['inputs_qualified'],
        quality_source=spec['quality_source'],seed=spec['seed'])
    epochs,orders = [],[]; updates = 0
    for epoch in range(1,25):
        c.check(); rng_before = order_generator.get_state().clone()
        order = torch.randperm(1661,generator=order_generator,device='cpu')
        need(torch.equal(torch.sort(order).values,torch.arange(1661,dtype=torch.int64)),
             'Every epoch visits all1661 matched samples exactly once')
        losses,counts = [],[]
        for first in range(0,1661,64):
            c.check(); indices = order[first:first+64]
            xb = X[indices].to('cuda:0'); yb = y[indices].to('cuda:0')
            optimizer.zero_grad(set_to_none=True)
            qhat = head(xb)
            need(qhat.shape == yb.shape and qhat.dtype == torch.float32, 'FP32 scalar quality per matched sample')
            loss = ((qhat-yb)**2).mean()
            need(loss.dtype == torch.float32 and bool(torch.isfinite(loss)), 'Finite FP32 mean squared error')
            loss.backward()
            need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()),
                 'All six and only quality-head tensors receive finite gradients')
            optimizer.step(); updates += 1
            losses.append(float(loss.detach().cpu())); counts.append(len(indices))
            del indices,xb,yb,qhat,loss
        need(counts == [64]*25+[61] and len(losses) == 26 and sum(counts) == 1661,
             'Original26 updates with true61-sample tail; no dropped or repeated samples')
        epochs.append(dict(epoch=epoch,updates=26,samples=1661,batch_counts=counts,
            online_pre_update_MSE=sum(v*n for v,n in zip(losses,counts))/1661,
            per_update_MSE=losses,order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest()))
        orders.append(dict(epoch=epoch,order=order.clone(),rng_before=rng_before,
                           rng_after=order_generator.get_state().clone()))
        print(json.dumps(dict(N26_quality_training_completed_epochs=epoch,locked_epochs=24)),flush=True)
    need(updates == 624, 'Exactly624 AdamW updates at fixedE24')
    optimizer.zero_grad(set_to_none=True)
    final_state = cpu_tree(torch,head.state_dict()); final_digest = state_digest(final_state)
    optimizer_state = cpu_tree(torch,optimizer.state_dict())
    need(all(p.grad is None for p in head.parameters()) and all(bool(torch.isfinite(v).all()) for v in final_state.values()),
         'Finite final six-tensor state and cleared gradients')
    need(torch.equal(cpu_external,torch.get_rng_state())
         and torch.equal(cuda_external,torch.cuda.get_rng_state(0).cpu()), 'External CPU/CUDA RNG unchanged')
    peak = int(torch.cuda.max_memory_allocated(0)); head.cpu(); del optimizer,head
    cleanup = clear_cuda(torch)
    need(peak <= 4096*1024**2, 'Peak allocated memory within fourGiB')
    checkpoint = dict(schema='N26-query-mask-quality-fixedE24-checkpoint-v1',**binding,
        state_dict=final_state,epoch=24,updates=624,parameters=299521,tensors=6,
        initialization_digest=initial_digest,final_digest=final_digest,
        training_target='original-matched-GT-full-grid-R-mask-IoU',qualified=False)
    checkpoint_ref = save(torch,c,output/'QUALITY_HEAD_E24.pth',checkpoint)
    train_state_ref = save(torch,c,output/'TRAIN_STATE_E24.pth',dict(
        schema='N26-query-mask-quality-fixedE24-full-training-state-v1',**binding,
        epoch=24,updates=624,initial_state_dict=initial_state,state_dict=final_state,
        optimizer_state_dict=optimizer_state,initialization_rng_before=initialization_rng_before,
        initialization_rng_after=initialization_rng_after,external_CPU_RNG=cpu_external,
        external_CUDA_RNG=cuda_external,order_initial_RNG=order_initial,
        order_final_RNG=order_generator.get_state().clone(),epoch_orders=orders,epoch_metrics=epochs))
    c.check()
    report_ref = c.write_json(output/'TRAIN_REPORT.json',dict(
        schema='N26-query-mask-quality-fixedE24-training-report-v1',
        status='N26_QUALITY_FIXED_E24_TRAIN_COMPLETE_PENDING_ROOT_ACTUAL0',**binding,
        checkpoint_ref=checkpoint_ref,training_state_ref=train_state_ref,
        images=741,GT=1661,epochs=24,batch_size=64,steps_per_epoch=26,total_updates=624,
        last_batch=61,parameters=299521,tensors=6,initialization_digest=initial_digest,
        final_digest=final_digest,epoch_metrics=epochs,
        optimizer=dict(name='AdamW',lr=.0003,weight_decay=.0001,betas=[.9,.999],eps=1e-8,
                       foreach=False,fused=False),scheduler=False,mixed_precision=False,gradient_clipping=False,
        only_quality_head_trained=True,teacher_loaded=False,DEV_container_deserialized=True,
        DEV_features_accessed=False,DEV_numeric_used=False,DEV_loss_evaluated=False,AP_evaluated=False,
        external_CPU_CUDA_RNG_unchanged=True,cleanup=cleanup,torch_peak_allocated_bytes=peak,
        qualified=False,ROOT_actual_GPU0_verified=False,current_owned_terminal_self_certified=False,
        elapsed_seconds=time.monotonic()-started))
    return dict(report_ref=report_ref,checkpoint_ref=checkpoint_ref,training_state_ref=train_state_ref)
