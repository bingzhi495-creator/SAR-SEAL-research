"""N07 fixed-endpoint all100 original-detail geometry evaluation.

Future ROOT-owned execution only. The single selected endpoint is the common
epoch8 or epoch24 determined by FIT BCE, never by DEV. One frozen original E24
forward supplies all four heads. Each export keeps its own native query order,
class and S0 score exactly; only segmentation changes in abs(native L)<=2.
This entry does not qualify its own terminal or independently replay training.
"""
import argparse
import copy
from functools import partial
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import types

SCHEMA='ROOT-N07-original-detail-fixed-endpoint-evaluation-spec-v1'
STATUS='N07_ORIGINAL_DETAIL_FIXED_ENDPOINT_EVALUATION_COMPLETE_PENDING_CPU'
PACKET_SCHEMA='N07-original-detail-all100-native-fields-v1'
SEEDS=(202610120701,202610120702)
ARMS=('C','R')
KEYS=tuple('%s_seed%d'%(arm,seed) for seed in SEEDS for arm in ARMS)
CONTROLS=('native_official',)+tuple(key+'_fixed_native_score' for key in KEYS)
NAMES=('hidden1.weight','hidden1.bias','hidden2.weight','hidden2.bias','final.weight','final.bias')
SHAPES=((64,562),(64,),(64,64),(64,),(1,64),(1,))
RECIPE=dict(images=187,raw_GT=380,queries=100,controls=list(CONTROLS),native_weights_epoch=24,
            head_endpoint='common8or24 selected only by prelocked FIT-BCE rule',band_abs_logit=2.,
            original_JPEG_patch_size=7,patch_padding='replicate',pixel_chunk_size=2048,feature_channel_chunk=16,
            fixed_native_scores=True,fixed_native_class=True,fixed_native_query_order=True,
            all100_queries_retained=True,no_NMS=True,no_score_cutoff=True,no_GT_query_selection=True,
            save_all187_low_fields=True,best_DEV_selection=False,TEST_accessed=False)


def require(ok,message):
    if not ok: raise RuntimeError(message)


def read(path):
    def reject(value): raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)


def reference(path):
    path=Path(path).resolve(strict=True); digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def bound(ref):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and type(ref['path']) is str and
            Path(ref['path']).is_absolute() and type(ref['bytes']) is int and ref['bytes']>=0 and type(ref['sha256']) is str and
            len(ref['sha256'])==64 and set(ref['sha256'])<=set('0123456789abcdef') and reference(ref['path'])==ref,'Actual pinned input differs')
    return Path(ref['path'])


def terminal(ref,phase,code=0):
    path=Path(ref['path'])
    require(path.is_absolute() and path.name=='PARENT_RECEIPT.json' and type(phase) is str and phase and path.parent.name==phase,
            'Explicit original RAW/phase locator required')
    raw=read(path)
    require(raw['status']=='ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode']==code and
            type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
            raw['held_handle_exit']==dict(wait_state='WAIT_OBJECT_0',actual_exit_code=code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and raw['numeric_birth_absence']['original_birth_absent'] is True and
            raw.get('error') is None and raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors']==[],
            'All20 normal plus old failed1 owned states must precede SHA/import')
    if 'phase' in raw: require(type(raw['phase']) is str and raw['phase']==phase,'Original RAW phase differs')
    sha,peak=raw['admission_sha256'],raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha)==64 and set(sha)<=set('0123456789abcdef') and type(peak) in (int,float) and
            math.isfinite(peak) and 0<=peak<=8192 and type(raw['whole_GPU_used_limit_MiB']) is int and raw['whole_GPU_used_limit_MiB']==8192 and
            raw['whole_leaf_VRAM_certified'] is False and raw['GPU_scope']=='whole GPU observed, not per-process WDDM allocation','RAW resource scope differs')
    return raw


def load(ref,name):
    path=bound(ref); source=path.read_bytes()
    require(len(source)==ref['bytes'] and hashlib.sha256(source).hexdigest()==ref['sha256'],'SOURCE bytes changed')
    module=types.ModuleType(name); module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__); return module


def source_gate(path,spec):
    require(spec['schema']==SCHEMA and type(spec['RAW_inputs']) is list and len(spec['RAW_inputs'])==20 and
            len({row['receipt']['path'] for row in spec['RAW_inputs']})==20,'Twenty actual normal RAWs required')
    raws=[terminal(row['receipt'],row['phase']) for row in spec['RAW_inputs']]
    terminal(spec['prior_failed_CPU_receipt'],Path(spec['prior_failed_CPU_receipt']['path']).parent.name,1)
    require(spec['prior_failed_CPU_receipt']['path'] not in {row['receipt']['path'] for row in spec['RAW_inputs']},'Failed attempt cannot qualify an ancestor')
    # No input hash, SOURCE import, checkpoint or scientific file read precedes these states.
    for ref in spec['sources']: bound(ref)
    closure={ref['path']:ref for ref in spec['sources']}; require(len(closure)==len(spec['sources']),'Unique complete evaluation closure required')
    required=[spec[key] for key in ('entry_source','head_training_entry_source','head_training_spec','head_training_report',
              'head_training_receipt','head_training_GPU_admission','cache_entry_source','point_heads_source',
              'official_export_source','official_evaluation_CPU_reader','DEV','prior_failed_CPU_receipt')]
    required+=list(spec['head_checkpoints'].values())+[row[k] for row in spec['RAW_inputs'] for k in ('receipt','admission','phase_binding','bootstrap')]
    require(all(ref in closure.values() for ref in required),'Actual evaluation primary inputs absent from sources')
    train=load(spec['head_training_entry_source'],'_N07_head_training_ABI'); training=read(bound(spec['head_training_spec']))
    cache,cached,cache_report,cache_q,context,old_closure,train_source,train_spec,train_admission=train.source_gate(bound(spec['head_training_spec']),training)
    require(all(ref in closure.values() for ref in training['sources']) and training['RAW_inputs']==spec['RAW_inputs'][:19] and
            training['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'] and train_source==spec['head_training_entry_source'] and
            training['cache_entry_source']==spec['cache_entry_source'] and training['point_heads_source']==spec['point_heads_source'],
            'Complete own fixed cache/training lineage required')
    owners=[cache.owner_join(row,raw,closure) for row,raw in zip(spec['RAW_inputs'],raws)]
    row,owner=spec['RAW_inputs'][-1],owners[-1]
    require(row['receipt']==spec['head_training_receipt'] and row['phase']==spec['head_training_phase']==training['phase'] and
            row['admission']==spec['head_training_GPU_admission']==train_admission and
            owner['entry']==train_source['path'] and owner['entry_args']==['--spec',train_spec['path'],'--owner-execute'],
            'Actual own head-training GPU terminal/admission required')
    source,spec_ref=reference(__file__),reference(path); require(source==spec['entry_source'],'Actual evaluation SOURCE differs')
    owner_path=Path(spec['GPU_admission_file']); require(owner_path.is_absolute(),'Absolute own admission required')
    owner_ref=reference(owner_path); own=read(owner_path)
    require(own['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and own['driver_sha256']==cache.OWNER_SHA and
            type(spec['phase']) is str and spec['phase'] and own['phase']==spec['phase'] and own['scientific_spec']==spec_ref and
            own['entry']==source['path'] and own['entry_args']==['--spec',str(path),'--owner-execute'] and own['allowed_cpus']==[0,1] and
            own['cuda_visible_devices']=='0' and own['automatic_retry'] is False and own['whole_GPU_used_limit_MiB']==8192 and
            type(spec['deadline_seconds']) is int and spec['deadline_seconds']>0 and own['qualified_leaf_deadline_seconds']==spec['deadline_seconds'] and
            own['bound_source_sha256'].get(str(path))==spec_ref['sha256'] and
            all(own['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in spec['sources']),'Own fixed evaluation owner/closure differs')
    report=read(bound(spec['head_training_report']))
    require(report['schema']=='N07-paired-point-training-report-v1' and report['status']==train.STATUS and report['source']==train_source and
            report['spec']==train_spec and report['GPU_admission']==train_admission and report['phase']==training['phase'] and
            report['recipe']==train.RECIPE and report['cache_CPU_qualification']==training['cache_CPU_qualification'] and
            report['teacher_checkpoint']==context['checkpointmap']['native0'] and report['point_heads_source']==spec['point_heads_source'] and
            report['native_model_loaded'] is False and report['native_weights_updated'] is False and report['no_DEV'] is True and report['no_TEST'] is True and
            report['all4_fixed_same_epochs'] is True and report['paired_order_and_initialization'] is True and report['cleanup']['allocated_after']==0,
            'Complete own paired-head training report required')
    epochs=report['completed_head_epochs']; require(type(epochs) is int and epochs in (8,24) and epochs==spec['head_endpoint_epoch'] and
            report['continued_all4_to24'] is (epochs==24) and set(report['checkpoints'])==set(spec['head_checkpoints'])==set(KEYS),
            'One prelocked common FIT-selected endpoint required; no DEV/best checkpoint choice')
    require(all(report['checkpoints'][key]['checkpoint']==spec['head_checkpoints'][key] and
            report['checkpoints'][key]['completed_head_epochs']==epochs and report['checkpoints'][key]['updates']==epochs*1661 for key in KEYS),
            'All four own final head checkpoints differ')
    outputs=[report[key] for key in ('start','trace','decision')]
    outputs += [item[key] for group in ('checkpoints','epoch8_checkpoints') for item in report[group].values() for key in ('checkpoint','proof')]
    require(all(ref in closure.values() for ref in outputs),'Complete training START/TRACE/decision/own checkpoint proofs absent')
    require(spec['recipe']==RECIPE and spec['image_root']==cached['image_root'] and
            spec['allocator_limit_MiB']==4096 and spec['whole_GPU_limit_MiB']==8192 and
            os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and all(os.environ.get(key)=='2' for key in
            ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'Fixed evaluation/resource recipe differs')
    return train,cache,cached,cache_report,cache_q,context,training,report,closure,source,spec_ref,owner_ref


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def head_endpoints(torch,factory,train,spec,report,closure):
    """CPU state/moment/proof and FIT endpoint consistency, not optimizer replay."""
    heads={}; states={}; cpu_rng=torch.random.get_rng_state().clone()
    start=read(bound(report['start'])); decision=read(bound(report['decision']))
    require(start['recipe']==train.RECIPE and start['source']==spec['head_training_entry_source'] and start['spec']==spec['head_training_spec'] and
            start['GPU_admission']==spec['head_training_GPU_admission'] and start['phase']==spec['head_training_phase'] and
            start['fresh_optimizer_state_entries_per_model']==dict.fromkeys(KEYS,0) and
            decision['rule']==train.RECIPE['extension_rule'] and decision['continue_all4']==report['continued_all4_to24'] and
            decision['fixed_final_epoch']==spec['head_endpoint_epoch'] and decision['DEV_accessed'] is False and decision['best_epoch_selection'] is False,
            'Actual paired START/continuation decision differs')
    trace=[json.loads(line) for line in bound(report['trace']).read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(trace)==2*spec['head_endpoint_epoch'] and len({(r['seed'],r['epoch']) for r in trace})==len(trace),'Complete paired epoch trace required')
    histories=report['endpoint_BCE_histories']; computed=[]; cp_by_key={}
    for key in KEYS:
        arm,seed_text=key.split('_seed'); seed=int(seed_text); cp=torch.load(str(bound(spec['head_checkpoints'][key])),map_location='cpu'); meta=cp['meta']
        state=cp['state_dict']; proof=read(bound(report['checkpoints'][key]['proof']))
        require(meta['schema']==train.CP_SCHEMA and meta['source']==spec['head_training_entry_source'] and meta['spec']==spec['head_training_spec'] and
                meta['GPU_admission']==spec['head_training_GPU_admission'] and meta['phase']==spec['head_training_phase'] and meta['arm']==arm and meta['seed']==seed and
                meta['point_heads_source']==spec['point_heads_source'] and meta['cache_spec']==report['cache_spec'] and meta['cache_report']==report['cache_report'] and
                meta['cache_CPU_qualification']==report['cache_CPU_qualification'] and meta['teacher_checkpoint']==report['teacher_checkpoint'] and
                meta['native_weights_epoch']==24 and meta['native_weights_updated'] is False and meta['recipe']==train.RECIPE and
                meta['completed_head_epochs']==spec['head_endpoint_epoch'] and meta['updates']==meta['GT_visits']==spec['head_endpoint_epoch']*1661 and
                meta['image_visits']==spec['head_endpoint_epoch']*741 and meta['point_visits']==meta['updates']*1024 and
                meta['parameter_tensors']==6 and meta['parameter_scalars']==40257 and meta['no_pending_gradients'] is True,
                'Own final six-tensor trained-head metadata differs')
        require(tuple(state)==NAMES and all(isinstance(state[n],torch.Tensor) and state[n].device.type=='cpu' and state[n].dtype==torch.float32 and
                tuple(state[n].shape)==shape and bool(torch.isfinite(state[n]).all()) for n,shape in zip(NAMES,SHAPES)), 'Own finite CPU head state differs')
        require(proof['checkpoint']==spec['head_checkpoints'][key] and proof['updates']==meta['updates'] and proof['full_readback_exact'] is True and
                proof['state']=={n:train.digest(torch,v) for n,v in state.items()} and
                proof['private_order_state']==train.digest(torch,cp['private_order_state']),'Own saved state/order proof differs')
        opt=cp['optimizer_state_dict']; require(len(opt['param_groups'])==1 and len(opt['state'])==6,'Actual six moment entries required')
        group=opt['param_groups'][0]
        require(group['lr']==3e-4 and group['weight_decay']==1e-4 and group['betas']==(.9,.999) and group['eps']==1e-8 and
                group['amsgrad'] is False and len(group['params'])==6,'Own constant AdamW recipe differs')
        for name,shape,identifier in zip(NAMES,SHAPES,group['params']):
            entry=opt['state'][identifier]
            require(set(entry)=={'step','exp_avg','exp_avg_sq'} and float(entry['step'])==meta['updates'] and
                    all(tuple(entry[k].shape)==shape and entry[k].dtype==torch.float32 and bool(torch.isfinite(entry[k]).all()) for k in ('exp_avg','exp_avg_sq')) and
                    proof['optimizer'][name]==dict({k:train.digest(torch,v) for k,v in entry.items()},updates=meta['updates']), 'Own AdamW moment proof differs')
        history=meta['endpoint_BCE_history']; require(history==histories[key] and len(history)==spec['head_endpoint_epoch'] and
                [row['epoch'] for row in history]==list(range(1,spec['head_endpoint_epoch']+1)) and
                all(type(row['BCE']) in (int,float) and math.isfinite(row['BCE']) and row['BCE']>=0 for row in history),'Complete FIT BCE history differs')
        for epoch in range(1,spec['head_endpoint_epoch']+1):
            row=[r for r in trace if r['seed']==seed and r['epoch']==epoch]
            require(len(row)==1 and row[0]['full_FIT_endpoint_BCE'][arm]==history[epoch-1]['BCE'] and row[0]['GT_instances']==1661 and
                    row[0]['images']==741 and row[0]['updates_per_model']==1661 and row[0]['point_visits_per_model']==1661*1024,'Full FIT endpoint trace differs')
        losses={row['epoch']:row['BCE'] for row in history}
        decreases=[(losses[a]-losses[b])/losses[a] if losses[a]>0 else 0. for a,b in ((6,7),(7,8))]
        computed.append(dict(seed=seed,arm=arm,relative_decrease_6_to_7=decreases[0],relative_decrease_7_to_8=decreases[1],trigger=all(v>=.01 for v in decreases)))
        head=factory.make_point_head(torch,seed)
        require({n:train.digest(torch,v) for n,v in head.state_dict().items()}==start['initial_head_state_proofs'][str(seed)],'Actual paired cold initializer differs')
        head.load_state_dict(state,strict=True); train.check_head(torch,head); head.requires_grad_(False); head.eval()
        heads[key]=head; states[key]=state; cp_by_key[key]=cp
    require(computed==decision['per_model'] and any(row['trigger'] for row in computed)==decision['continue_all4'],'Exact FIT-only continuation predicate differs')
    for seed in SEEDS:
        left,right=cp_by_key['C_seed%d'%seed],cp_by_key['R_seed%d'%seed]
        require(left['meta']['epoch_orders']==right['meta']['epoch_orders'] and torch.equal(left['private_order_state'],right['private_order_state']),
                'Paired complete image-order/RNG endpoint differs')
        rng=torch.Generator(device='cpu'); rng.manual_seed(seed)
        for epoch,row in enumerate(left['meta']['epoch_orders'],1):
            order=torch.randperm(741,generator=rng).tolist()
            require(row==dict(epoch=epoch,indices=order,sha256=hashlib.sha256(json.dumps(order,separators=(',',':')).encode('ascii')).hexdigest()),
                    'Independent private image-order reconstruction differs')
        require(len(left['meta']['epoch_orders'])==spec['head_endpoint_epoch'] and torch.equal(rng.get_state(),left['private_order_state']),'Own final order state differs')
    require(torch.equal(cpu_rng,torch.random.get_rng_state()) and not torch.cuda.is_initialized(),'CPU checkpoint loading changed external RNG/CUDA')
    return heads,states


def original_feature(torch,renderer,F,identity):
    H,W=identity['original_shape']; original=F.new_empty((256,H,W))
    for first in range(0,256,16):
        values=renderer(F[:,first:first+16],identity)
        require(tuple(values.shape)==(16,H,W),'Actual original coarse feature projection differs')
        original[first:first+16]=values
    return original


def refine(torch,np,cache,head,arm,F_original,e,L,gray,spec,started):
    """All100 query identity; band comes solely from fixed native L."""
    refined=L.clone(); counts=[]
    for q in range(100):
        indices=torch.nonzero(L[q].flatten().abs()<=2,as_tuple=False).flatten(); counts.append(indices.numel())
        for first in range(0,len(indices),2048):
            require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked evaluation deadline exceeded')
            idx=indices[first:first+2048]; patch=cache.original_patches(torch,np,gray,idx.cpu()[None])[0].to(L.device)
            values=head(F_original.flatten(1)[:,idx].transpose(0,1),e[0,q].expand(len(idx),256),L[q].flatten()[idx],patch,arm=arm)
            require(bool(values['band'].all()) and bool(torch.isfinite(values['logits']).all()),'Actual band-only point-head output differs')
            refined[q].flatten()[idx]=values['logits']
        del indices
    require(torch.equal(refined[L.abs()>2],L[L.abs()>2]),'Outside native boundary band must remain exact')
    return refined,counts


def fixed_score_rows(np,masks,native,logits):
    rows=copy.deepcopy(native)
    for row in rows:
        q=row['query_id']; binary=(logits[q]>0).cpu().numpy().astype(np.uint8)
        rle=masks.encode(np.asfortranarray(binary)); counts=rle['counts'].decode('ascii') if isinstance(rle['counts'],bytes) else rle['counts']
        row['segmentation']=dict(size=list(rle['size']),counts=counts)
    require(all({k:v for k,v in left.items() if k!='segmentation'}=={k:v for k,v in right.items() if k!='segmentation'} for left,right in zip(native,rows)),
            'Only geometry may change; no empty-mask score rezero/filter/reorder')
    return rows


def run(spec_path):
    started=time.monotonic(); path=Path(spec_path); require(path.is_absolute(),'Absolute ROOT SPEC required'); path=path.resolve(strict=True)
    spec=read(path); train,cache,cached,cache_report,cache_q,context,training,report,closure,source,spec_ref,owner_ref=source_gate(path,spec)
    output=Path(spec['output_directory']); require(output.is_absolute() and not output.exists(),'Fresh absolute evaluation output required')
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1],'Actual CPU01 required before science imports')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(),'Original Torch/cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    factory=load(spec['point_heads_source'],'_N07_evaluation_point_heads')
    heads,head_states=head_endpoints(torch,factory,train,spec,report,closure)
    legacy=load(context['evaluation_source'],'_N07_E24_qualification'); teachers=legacy.qualified_training(context)
    import numpy as np
    import cv2
    require(np.__version__=='1.23.5','Qualified original NumPy version required')
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    from pycocotools import mask as masks
    from pycocotools.cocoeval import COCOeval
    libraries=cache.imported_files(torch,np,cv2,CocoDataset,COCO,closure)
    for role,obj in (('COCOeval',COCOeval),('masks',masks),('mask_binary',masks._mask),('torch_binary',torch._C)):
        ref=reference(inspect.getsourcefile(obj) if role=='COCOeval' else obj.__file__)
        require(ref in closure.values(),'Actual evaluator/library file absent: '+role); libraries[role]=ref
    modules=[load(context[key],'_N07_eval_'+key) for key in cache.MODULES]
    native,legacy,pref,initializer,common,old,diag,joint,preflight=modules; old.torch=torch
    exporter=load(spec['official_export_source'],'_N07_official_export'); official=load(spec['official_evaluation_CPU_reader'],'_N07_official_metrics')
    register_all_modules(init_default_scope=False); init_default_scope('mmdet')
    construction=dict(teachers['native0']['training_spec'],training_source=context['training_source'],checkpointmap=context['checkpointmap'],
                      qualificationmap=context['qualificationmap'],order_seed=native.ORDER_SEED,phase_loss_seed_base=native.LOSS_SEED_BASE)
    model,anchor,policy,loading=native.prepare_training_model(torch,pref,initializer,common,MODELS,revert_sync_batchnorm,Config,
                          tuple(CocoDataset.METAINFO['classes']),construction,teachers['native0'])
    native.imported_library_gate(torch,model,context)
    require(Path(inspect.getsourcefile(torch.nn.functional.interpolate)).resolve()==bound(context['interpolation_SOURCE']).resolve(),'Original interpolation SOURCE differs')
    model.requires_grad_(False); model.eval(); model.backbone.with_cp=False
    require(len(model.state_dict())==len(anchor)==610 and len(tuple(model.parameters()))==451,'Strict frozen original E24 model required')
    dataset=CocoDataset(ann_file=str(bound(spec['DEV'])),data_prefix=dict(img=spec['image_root']),metainfo=dict(classes=('ship',)),pipeline=[],filter_cfg=None,test_mode=True)
    coco=COCO(str(bound(spec['DEV']))); ids=[dataset.get_data_info(i)['img_id'] for i in range(len(dataset))]
    require(len(dataset)==len(coco.imgs)==187 and len(coco.anns)==380 and set(coco.cats)=={1} and ids==list(coco.imgs) and len(set(ids))==187,
            'Full canonical DEV187/380/category1 required')
    require(all(str(Path(dataset.get_data_info(i)['img_path']).resolve(strict=True)) in closure for i in range(187)), 'All actual DEV images must be owner-bound')
    torch.cuda.init(); torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0); model.to('cuda:0')
    for head in heads.values(): head.to('cuda:0')
    output.mkdir(); fields_dir=output/'FINAL_FIELDS'; fields_dir.mkdir(); predictions={key:[] for key in CONTROLS}; factors=[]; rows=[]; packets=[]
    write(output/'START.json',dict(schema='N07-original-detail-evaluation-start-v1',source=source,spec=spec_ref,GPU_admission=owner_ref,phase=spec['phase'],
          head_training_report=spec['head_training_report'],head_endpoint_epoch=spec['head_endpoint_epoch'],head_checkpoints=spec['head_checkpoints'],
          cache_CPU_qualification=training['cache_CPU_qualification'],teacher_checkpoint=report['teacher_checkpoint'],teacher_loading=loading,
          recipe=RECIPE,head_state_proofs={key:{name:pref.digest_tensor(value) for name,value in state.items()} for key,state in head_states.items()},
          budget_is_unmeasured_prelock=True,no_TEST=True,no_DEV_selection=True))
    renderer=partial(diag.render_original,torch); native_before=pref.state_proof(model.state_dict())
    with (output/'TRACE.jsonl').open('x',encoding='utf-8',newline='\n') as trace:
        for index in range(187):
            require(time.monotonic()-started<spec['deadline_seconds'],'Fixed evaluation deadline exceeded')
            data,samples,gray,raw,identity=old.load_image(model,dataset,coco,dataset.get_data_info(index))
            require(len(samples)==1 and not any(k in samples[0] for k in ('gt_instances','ignored_instances','gt_sem_seg','proposals')),'GT/proposals survived stripping')
            external=pref.rng_snapshot(torch); seed=100000+identity['image_id']
            with torch.random.fork_rng(devices=[0]):
                torch.random.default_generator.manual_seed(seed); torch.cuda.manual_seed(seed)
                with torch.no_grad():
                    features=model.extract_feat(data['inputs'])
                    with cache.observed_fields(torch,model.panoptic_head) as observed:
                        classes,mask_stages=model.panoptic_head(features,samples)
                    F,e,M,C=observed['F'][0],observed['e'][-1],mask_stages[-1],classes[-1]; del features
                    versions=[value._version for value in classes+mask_stages]; L=renderer(M,identity)
                    native_rows,native_factors,qids,class_packet=exporter.export_original(torch,np,masks,model.panoptic_fusion_head,C,L,identity['image_id'],pref)
                    require(qids==[row['query_id'] for row in native_rows] and sorted(qids)==list(range(100)),'Native all100 order/identity differs')
                    padded=torch.nn.functional.interpolate(M,size=(800,800),mode='bilinear',align_corners=False)
                    actual=model.panoptic_fusion_head.predict(C,padded,samples,rescale=True)[0]['ins_results']; ordered=L[qids]; binary=ordered>0
                    quality=(ordered.sigmoid()*binary).flatten(1).sum(1)/(binary.flatten(1).sum(1)+1e-6); score=C[0].softmax(-1)[:,0][qids]*quality
                    require(len(actual)==100 and torch.equal(actual.masks,binary) and torch.equal(actual.scores,score) and bool((actual.labels==0).all()),
                            'Whole native official fusion differs from original-space export')
                    del padded,actual,ordered,binary,quality,score
                    predictions['native_official']+=native_rows; factors+=native_factors
                    F_original=original_feature(torch,renderer,F,identity); band_counts=None
                    for key in KEYS:
                        logits,counts=refine(torch,np,cache,heads[key],key[0],F_original,e,L,raw,spec,started)
                        if band_counts is None: band_counts=counts
                        require(counts==band_counts,'Four models must use identical native GT-free bands')
                        predictions[key+'_fixed_native_score']+=fixed_score_rows(np,masks,native_rows,logits); del logits
                    require(versions==[value._version for value in classes+mask_stages],'Refinement changed native stage tensors')
                    values=dict(F256=F,e0=e,M200=M,final_class_logits=C)
                    field_digests={name:pref.digest_tensor(value) for name,value in values.items()}
                    packet=dict(schema=PACKET_SCHEMA,source=source,spec=spec_ref,phase=spec['phase'],dataset_index=index,identity=identity,
                                teacher_checkpoint=report['teacher_checkpoint'],head_checkpoints=spec['head_checkpoints'],head_endpoint_epoch=spec['head_endpoint_epoch'],
                                prediction_query_ids=qids,field_digests=field_digests,native_fields={name:value.cpu().clone() for name,value in values.items()},
                                GPU_original_logits_saved=False,original_JPEG_gray_digest=dict(shape=list(raw.shape),dtype=str(raw.dtype),sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
                    target=fields_dir/('IMAGE_%04d.pth'%index)
                    with target.open('xb') as stream: torch.save(packet,stream)
                    require(pref.tree_equal(torch,packet,torch.load(str(target),map_location='cpu')),'All100 native field readback differs')
                    packet_ref=reference(target); packets.append(packet_ref)
                    del packet,F_original,L,values,F,e,M,C,classes,mask_stages,observed,native_rows,native_factors
            restored=pref.rng_snapshot(torch); require(all(torch.equal(a,b) for a,b in zip(external,restored)),'Private native forward RNG not restored')
            row=dict(dataset_index=index,image_id=identity['image_id'],identity=identity,ann_ids=identity['ann_ids'],raw_GT=len(identity['ann_ids']),
                     head_seed=seed,prediction_query_ids=qids,native_class_packet=class_packet,band_counts_all100=band_counts,
                     final_fields=field_digests,packet=packet_ref,GT_removed_from_forward=True,loader_accesses_annotations=True,
                     matching_or_loss_executed=False,native610_SOURCE_unchanged=True,native_all10_versions_unchanged=True,
                     no_GT_query_selection=True,all100_queries_retained=True,no_NMS=True,no_score_cutoff=True,
                     fixed_native_score=True,new_empty_geometry_may_keep_positive_native_score=True,external_RNG_restored=True,
                     independent_CPU_evaluation_replayed=False)
            rows.append(row); trace.write(json.dumps(row,allow_nan=False)+'\n'); trace.flush()
            print(json.dumps(dict(evaluated_images=index+1,elapsed_seconds=time.monotonic()-started)),flush=True)
            del data,samples,gray,raw,identity,external,restored,qids,class_packet
    require(sum(row['raw_GT'] for row in rows)==380 and all(len(values)==18700 for values in predictions.values()),'Full all100 DEV population differs')
    require(native_before==pref.state_proof(model.state_dict()) and all(torch.equal(value.detach().cpu(),anchor[name]) for name,value in model.state_dict().items()) and
            all(torch.equal(value.detach().cpu(),head_states[key][name]) for key,head in heads.items() for name,value in head.state_dict().items()) and
            all(not p.requires_grad and p.grad is None for module in [model]+list(heads.values()) for p in module.parameters()) and
            all(not module.training for root in [model]+list(heads.values()) for module in root.modules()), 'Evaluation modified frozen native610/head states')
    peak=torch.cuda.max_memory_allocated(0); model.cpu()
    for head in heads.values(): head.cpu()
    heads.clear(); del head,heads,model,anchor; gc.collect(); cleanup=pref.clear_gpu(torch)
    require(cleanup['allocated_after']==0,'Native/point inference tensor survived GPU cleanup')
    control_results={}; control_artifacts={}
    for key in CONTROLS:
        target=output/(key+'_PREDICTIONS.json'); write(target,predictions[key]); metrics=official.coco_metrics(coco,predictions[key],ids,COCOeval)
        require(len(metrics['stats'])==12 and all(type(v) in (int,float) and math.isfinite(v) and (v==-1 or 0<=v<=1) for v in metrics['stats']),
                'Official COCO12 values differ; genuine undefined slices remain -1')
        control_artifacts[key]=reference(target); control_results[key]=metrics
    write(output/'NATIVE_QUERY_SCORE_FACTORS.json',factors); write(output/'IMAGE_ROWS.json',rows)
    for ref in spec['sources']: bound(ref)
    require(reference(path)==spec_ref and reference(__file__)==source and reference(spec['GPU_admission_file'])==owner_ref and
            psutil.Process().cpu_affinity()==[0,1] and torch.get_num_threads()==torch.get_num_interop_threads()==2,'Final provenance/resources changed')
    result=dict(schema='N07-original-detail-fixed-endpoint-evaluation-report-v1',status=STATUS,source=source,spec=spec_ref,GPU_admission=owner_ref,phase=spec['phase'],
                start=reference(output/'START.json'),trace=reference(output/'TRACE.jsonl'),image_rows=reference(output/'IMAGE_ROWS.json'),
                native_query_score_factors=reference(output/'NATIVE_QUERY_SCORE_FACTORS.json'),final_field_packets=packets,
                head_training_report=spec['head_training_report'],head_checkpoints=spec['head_checkpoints'],head_endpoint_epoch=spec['head_endpoint_epoch'],
                teacher_checkpoint=report['teacher_checkpoint'],cache_CPU_qualification=training['cache_CPU_qualification'],DEV=spec['DEV'],
                recipe=RECIPE,controls=list(CONTROLS),predictions=control_artifacts,control_results=control_results,images=187,raw_GT=380,
                predictions_per_control=18700,native_forwards=187,native_weights_updated=False,heads_updated=False,matching_or_loss_executed=False,
                original187_all100_low_fields_saved=True,GPU_original_logits_saved=False,independent_CPU_evaluation_replayed=False,
                independent_CPU_training_trajectory_replayed=False,no_TEST=True,no_best_DEV_selection=True,actual_imported_files=libraries,
                torch_peak_allocated_bytes=peak,cleanup=cleanup,elapsed_seconds=time.monotonic()-started,budget_is_unmeasured_prelock=True,
                evidence_limits=['All four heads change only abs(original native L)<=2 and keep the same native S0 score values/query order, including new empty masks.',
                'C/R test access to real7x7 JPEG grayscale patches; equal capacity and paired access do not imply equal image information.',
                'Native F/e remain coarse200 fields projected to the original raster; interpolation is not original-image detail.',
                'FIXED native scores isolate a conditional geometry diagnostic; this is not score calibration, causal identification, or a SAR-scattering claim.',
                'GPU original logits are not archived; future CPU all100 field/render replay needs its own practical numerical tolerance and actual owned qualification.',
                'Head weight/moment/order consistency checks do not independently replay every training gradient/optimizer update.',
                'Allocator and deadline are prelocked upper bounds with unmeasured throughput/suitability; ROOT separately closes actual tool terminals.'])
    write(output/'REPORT.json',result); print(json.dumps(dict(status=STATUS,report=reference(output/'REPORT.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True); parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args(); require(args.owner_execute,'Explicit single ROOT-owned fixed endpoint evaluation required'); run(args.spec)
