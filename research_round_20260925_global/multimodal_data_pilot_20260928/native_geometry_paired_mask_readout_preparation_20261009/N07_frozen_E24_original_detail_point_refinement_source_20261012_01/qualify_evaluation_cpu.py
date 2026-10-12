"""Independent N07 fixed-endpoint export/COCO/one-to-one geometry CPU reader.

Future ROOT-owned execution only. All21 successful ancestors plus the preserved
N01 failed1 precede hashes/imports. Independently check every DEV image/query,
fixed S0, low-field/JPEG identity, native M rendering, five COCO12 results and
official matching pixels. No native teacher, four-head dense inference, optimizer
or training trajectory is replayed. The band/sign tolerances are practical CPU
consistency tests, not certified CUDA error bounds. ROOT closes this CPU RAW.
"""
import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types

SCHEMA='ROOT-N07-original-detail-fixed-endpoint-evaluation-CPU-qualification-spec-v1'
Q_SCHEMA='ROOT-N07-original-detail-fixed-endpoint-evaluation-CPU-qualification-v1'
STATUS='ROOT_N07_ORIGINAL_DETAIL_FIXED_ENDPOINT_EVALUATION_CPU_QUALIFIED'
E24_SCHEMA='ROOT-fixed-E24-native0-export-COCO-reference-recompute-CPU-qualification-v1'
E24_STATUS='ROOT_FIXED_E24_NATIVE0_EXPORT_COCO_REFERENCE_RECOMPUTED_CPU'
SEEDS=(202610120701,202610120702)
KEYS=tuple('%s_seed%d'%(arm,seed) for seed in SEEDS for arm in ('C','R'))
CONTROLS=('native_official',)+tuple(key+'_fixed_native_score' for key in KEYS)
NAMES=('hidden1.weight','hidden1.bias','hidden2.weight','hidden2.bias','final.weight','final.bias')
SHAPES=((64,562),(64,),(64,64),(64,),(1,64),(1,))
METRICS=('AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL')
NUMERIC=dict(class_rtol=1e-6,class_atol=1e-6,quality_rtol=1e-4,quality_atol=1e-4,
             CPU_sign_and_band_rtol=1e-4,CPU_sign_and_band_atol=1e-4,COCO_stats_atol=1e-12,
             renderer_query_chunk=4,official_quality_eps=1e-6,certified_GPU_error_bound=False)
REQUIRED_SPEC_KEYS=('schema','phase','sources','RAW_inputs','prior_failed_CPU_receipt','entry_source',
    'evaluation_entry_source','evaluation_spec','evaluation_report','evaluation_GPU_RAW','evaluation_GPU_admission',
    'training_CPU_reader_source','renderer_source','DEV','E24_reference_DEV','E24_reference_predictions',
    'E24_reference_Q','expected_E24_COCO_stats','numeric_protocol','CPU_admission_file','deadline_seconds','Q_path')


def require(ok,message):
    if not ok: raise RuntimeError(message)


def read(path):
    def reject(value): raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)


def reference(path):
    path=Path(path).resolve(strict=True); value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): value.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=value.hexdigest())


def bound(ref):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and type(ref['path']) is str and
            Path(ref['path']).is_absolute() and type(ref['bytes']) is int and ref['bytes']>=0 and
            type(ref['sha256']) is str and len(ref['sha256'])==64 and set(ref['sha256'])<=set('0123456789abcdef') and
            reference(ref['path'])==ref,'Actual pinned input differs')
    return Path(ref['path'])


def terminal(ref,phase,code=0):
    path=Path(ref['path'])
    require(path.is_absolute() and path.name=='PARENT_RECEIPT.json' and type(phase) is str and phase and path.parent.name==phase,
            'Original absolute RAW/phase locator required')
    raw=read(path)
    require(raw['status']=='ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode']==code and
            type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
            raw['held_handle_exit']==dict(wait_state='WAIT_OBJECT_0',actual_exit_code=code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and raw['numeric_birth_absence']['original_birth_absent'] is True and
            raw.get('error') is None and raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors']==[],
            'All21 successful plus preserved failed1 terminals must precede SHA/import')
    if 'phase' in raw: require(type(raw['phase']) is str and raw['phase']==phase,'Original RAW phase differs')
    sha,peak=raw['admission_sha256'],raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha)==64 and set(sha)<=set('0123456789abcdef') and type(peak) in (int,float) and math.isfinite(peak) and
            0<=peak<=8192 and type(raw['whole_GPU_used_limit_MiB']) is int and raw['whole_GPU_used_limit_MiB']==8192 and
            raw['whole_leaf_VRAM_certified'] is False and raw['GPU_scope']=='whole GPU observed, not per-process WDDM allocation',
            'Original resource observation scope differs')
    return raw


def load(ref,name):
    path=bound(ref); data=path.read_bytes()
    require(len(data)==ref['bytes'] and hashlib.sha256(data).hexdigest()==ref['sha256'],'SOURCE bytes changed')
    module=types.ModuleType(name); module.__file__=str(path)
    exec(compile(data,str(path),'exec'),module.__dict__); return module


def stats_close(left,right,message):
    require(type(left) is list and type(right) is list and len(left)==len(right)==12,message)
    require(all(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and
                (a==-1 or 0<=a<=1) and (b==-1 or 0<=b<=1) and abs(a-b)<=NUMERIC['COCO_stats_atol']
                for a,b in zip(left,right)),message)


def source_gate(path,spec):
    require(all(key in spec for key in REQUIRED_SPEC_KEYS) and spec['schema']==SCHEMA and
            type(spec['RAW_inputs']) is list and len(spec['RAW_inputs'])==21 and
            len({row['receipt']['path'] for row in spec['RAW_inputs']})==21 and
            all({'receipt','phase','admission','phase_binding','bootstrap'}<=set(row) for row in spec['RAW_inputs']),
            'Exactly21 complete successful ancestors required')
    raws=[terminal(row['receipt'],row['phase']) for row in spec['RAW_inputs']]
    terminal(spec['prior_failed_CPU_receipt'],Path(spec['prior_failed_CPU_receipt']['path']).parent.name,1)
    require(spec['prior_failed_CPU_receipt']['path'] not in {row['receipt']['path'] for row in spec['RAW_inputs']},'Failure cannot qualify an ancestor')
    # All22 terminal reads precede every hash, SOURCE import or other payload read.
    for ref in spec['sources']: bound(ref)
    closure={ref['path']:ref for ref in spec['sources']}; require(len(closure)==len(spec['sources']),'Unique complete CPU input closure required')
    roles=('entry_source','evaluation_entry_source','evaluation_spec','evaluation_report','evaluation_GPU_RAW','evaluation_GPU_admission',
           'training_CPU_reader_source','renderer_source','DEV','E24_reference_DEV','E24_reference_predictions','E24_reference_Q','prior_failed_CPU_receipt')
    required=[spec[k] for k in roles]+[row[k] for row in spec['RAW_inputs'] for k in ('receipt','admission','phase_binding','bootstrap')]
    require(all(ref in closure.values() for ref in required),'Actual CPU primary inputs absent')
    evaluator=load(spec['evaluation_entry_source'],'_N07_CPU_evaluation_ABI'); evaluation=read(bound(spec['evaluation_spec']))
    require(evaluation['schema']==evaluator.SCHEMA and evaluator.CONTROLS==CONTROLS and evaluation['recipe']==evaluator.RECIPE and
            evaluation['entry_source']==spec['evaluation_entry_source'] and evaluation['RAW_inputs']==spec['RAW_inputs'][:20] and
            evaluation['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'] and evaluation['DEV']==spec['DEV']==spec['E24_reference_DEV'] and
            all(ref in closure.values() for ref in evaluation['sources']),'Actual fixed evaluation recipe/ancestry/DEV differs')
    require(evaluation['head_endpoint_epoch'] in (8,24) and set(evaluation['head_checkpoints'])==set(KEYS),'One fixed common head endpoint required')
    train=load(evaluation['head_training_entry_source'],'_N07_CPU_training_ABI'); training=read(bound(evaluation['head_training_spec']))
    require(training['schema']==train.SCHEMA and training['entry_source']==evaluation['head_training_entry_source'] and
            training['recipe']==train.RECIPE and training['RAW_inputs']==evaluation['RAW_inputs'][:19] and
            training['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'] and training['cache_entry_source']==evaluation['cache_entry_source'] and
            training['point_heads_source']==evaluation['point_heads_source'] and all(ref in closure.values() for ref in training['sources']),
            'Actual paired training/cache primary closure differs')
    provenance=load(spec['training_CPU_reader_source'],'_N07_CPU_cache_lineage_only')
    cache,cached,cache_report,cache_q,context=provenance.cache_lineage(train,training,closure)
    require(spec['renderer_source']==context['renderer_source'] and evaluation['image_root']==cached['image_root'],
            'Original pure renderer and JPEG image root differ')
    owners=[cache.owner_join(row,raw,closure) for row,raw in zip(spec['RAW_inputs'],raws)]
    for index,receipt,phase,admission,entry,specification in (
        (17,training['cache_GPU_receipt'],training['cache_GPU_phase'],training['cache_GPU_admission'],training['cache_entry_source'],training['cache_spec']),
        (18,training['cache_CPU_receipt'],training['cache_CPU_phase'],training['cache_CPU_admission'],training['cache_CPU_reader_source'],cache_q['spec']),
        (19,evaluation['head_training_receipt'],evaluation['head_training_phase'],evaluation['head_training_GPU_admission'],evaluation['head_training_entry_source'],evaluation['head_training_spec']),
        (20,spec['evaluation_GPU_RAW'],evaluation['phase'],spec['evaluation_GPU_admission'],spec['evaluation_entry_source'],spec['evaluation_spec'])):
        row,owner=spec['RAW_inputs'][index],owners[index]
        require(row['receipt']==receipt and row['phase']==phase and row['admission']==admission and owner['entry']==entry['path'] and
                owner['scientific_spec']==specification and owner['entry_args']==['--spec',specification['path'],'--owner-execute'],
                'Actual own cache/training/evaluation owner identity differs')
    require(owners[-1]['cuda_visible_devices']=='0' and owners[-1]['qualified_leaf_deadline_seconds']==evaluation['deadline_seconds'] and
            all(owners[-1]['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in evaluation['sources']),
            'Complete original evaluation owner differs')
    source,spec_ref=reference(__file__),reference(path); require(source==spec['entry_source'],'Actual CPU entry SOURCE differs')
    owner_path=Path(spec['CPU_admission_file']); require(owner_path.is_absolute(),'Absolute own CPU admission required')
    owner_ref=reference(owner_path); owner=read(owner_path)
    require(owner['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and owner['driver_sha256']==cache.OWNER_SHA and
            type(spec['phase']) is str and spec['phase'] and owner['phase']==spec['phase'] and owner['scientific_spec']==spec_ref and
            owner['entry']==source['path'] and owner['entry_args']==['--spec',str(path),'--owner-execute'] and owner['allowed_cpus']==[0,1] and
            owner['cuda_visible_devices']=='' and owner['automatic_retry'] is False and owner['whole_GPU_used_limit_MiB']==8192 and
            type(spec['deadline_seconds']) is int and spec['deadline_seconds']>0 and owner['qualified_leaf_deadline_seconds']==spec['deadline_seconds'] and
            owner['bound_source_sha256'].get(str(path))==spec_ref['sha256'] and
            all(owner['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in spec['sources']),'Actual CPU owner/closure differs')
    head_report=read(bound(evaluation['head_training_report'])); report=read(bound(spec['evaluation_report']))
    require(head_report['schema']=='N07-paired-point-training-report-v1' and head_report['status']==train.STATUS and
            head_report['source']==evaluation['head_training_entry_source'] and head_report['spec']==evaluation['head_training_spec'] and
            head_report['GPU_admission']==evaluation['head_training_GPU_admission'] and head_report['phase']==evaluation['head_training_phase'] and
            head_report['recipe']==train.RECIPE and head_report['completed_head_epochs']==evaluation['head_endpoint_epoch'] and
            head_report['teacher_checkpoint']==context['checkpointmap']['native0'] and head_report['cache_CPU_qualification']==training['cache_CPU_qualification'] and
            head_report['all4_fixed_same_epochs'] is True and head_report['paired_order_and_initialization'] is True and
            head_report['native_weights_updated'] is False and head_report['no_DEV'] is True and head_report['no_TEST'] is True,
            'Own fixed paired head-training report differs')
    require(report['schema']=='N07-original-detail-fixed-endpoint-evaluation-report-v1' and report['status']==evaluator.STATUS and
            report['source']==spec['evaluation_entry_source'] and report['spec']==spec['evaluation_spec'] and
            report['GPU_admission']==spec['evaluation_GPU_admission'] and report['phase']==evaluation['phase'] and
            report['head_training_report']==evaluation['head_training_report'] and report['head_checkpoints']==evaluation['head_checkpoints'] and
            report['head_endpoint_epoch']==evaluation['head_endpoint_epoch'] and report['teacher_checkpoint']==head_report['teacher_checkpoint'] and
            report['cache_CPU_qualification']==training['cache_CPU_qualification'] and report['DEV']==spec['DEV'] and
            report['recipe']==evaluator.RECIPE and report['controls']==list(CONTROLS) and
            set(report['predictions'])==set(report['control_results'])==set(CONTROLS) and
            report['images']==187 and report['raw_GT']==380 and report['predictions_per_control']==18700 and report['native_forwards']==187 and
            report['native_weights_updated'] is False and report['heads_updated'] is False and report['matching_or_loss_executed'] is False and
            report['original187_all100_low_fields_saved'] is True and report['GPU_original_logits_saved'] is False and
            report['independent_CPU_evaluation_replayed'] is False and report['no_TEST'] is True and report['no_best_DEV_selection'] is True and
            report['cleanup']['allocated_after']==0,'Actual five-control fixed GPU report differs')
    outputs=[report[k] for k in ('start','trace','image_rows','native_query_score_factors')]+report['final_field_packets']+list(report['predictions'].values())
    outputs += [head_report[k] for k in ('start','trace','decision')]
    outputs += [row[k] for group in ('checkpoints','epoch8_checkpoints') for row in head_report[group].values() for k in ('checkpoint','proof')]
    require(len(report['final_field_packets'])==187 and all(ref in closure.values() for ref in outputs),'Complete GPU/head output inputs absent')
    anchor=read(bound(spec['E24_reference_Q']))
    require(anchor['schema']==E24_SCHEMA and anchor['status']==E24_STATUS and anchor['predictions']==spec['E24_reference_predictions'] and
            anchor['DEV']==spec['E24_reference_DEV'] and anchor['images']==187 and anchor['raw_GT']==380 and anchor['predictions_count']==18700 and
            anchor['spec'] in closure.values(),'Qualified fixed-export strong E24 reference identity differs')
    stats_close(anchor['COCO']['stats'],spec['expected_E24_COCO_stats'],'Prelocked strong E24 reference stats differ')
    require(spec['numeric_protocol']==NUMERIC and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
            all(os.environ.get(key)=='2' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
            'Prelocked CPU environment/numeric scope differs')
    target=Path(spec['Q_path']); details=target.with_name(target.stem+'_DETAILS.json')
    require(target.is_absolute() and details!=target and all(not p.exists() and not p.with_suffix(p.suffix+'.tmp').exists() for p in (target,details)),
            'Fresh absolute Q and details paths required')
    return dict(evaluator=evaluator,evaluation=evaluation,train=train,training=training,cache=cache,cache_q=cache_q,context=context,
                head_report=head_report,report=report,closure=closure,source=source,spec_ref=spec_ref,owner_ref=owner_ref,owner=owner,details=details)


def digest(tensor):
    array=tensor.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest())


def imported_files(torch,np,cv2,COCO,COCOeval,masks,gate):
    modules=dict(torch=torch,torch_binary=torch._C,numpy=np,numpy_binary=np.core._multiarray_umath,
                 cv2=cv2,cv2_binary=cv2._native,COCO=sys.modules[COCO.__module__],COCOeval=sys.modules[COCOeval.__module__],
                 masks=masks,mask_binary=masks._mask)
    result={name:reference(module.__file__) for name,module in modules.items()}
    result['interpolation']=reference(inspect.getsourcefile(torch.nn.functional.interpolate))
    require(all(ref in gate['closure'].values() and gate['owner']['bound_source_sha256'].get(ref['path'])==ref['sha256']
                for ref in result.values()) and result['interpolation']==gate['context']['interpolation_SOURCE'],
            'Actual CPU imported library/binary/math input absent or unbound')
    return result


def population(coco,rows,ids):
    require(type(rows) is list and len(rows)==18700,'Complete18700 export required'); grouped={i:[] for i in ids}
    for index,row in enumerate(rows):
        require(type(row) is dict and set(row)=={'image_id','query_id','category_id','score','segmentation'} and
                type(row['image_id']) is int and row['image_id']==ids[index//100] and type(row['query_id']) is int and 0<=row['query_id']<100 and
                type(row['category_id']) is int and row['category_id']==1 and type(row['score']) in (int,float) and
                math.isfinite(row['score']) and 0<=row['score']<=1,'Original canonical image/query/order/score differs')
        rle=row['segmentation']; image=coco.imgs[row['image_id']]
        require(type(rle) is dict and set(rle)=={'size','counts'} and rle['size']==[image['height'],image['width']] and
                type(rle['counts']) is str,'Original-size compressed RLE required')
        grouped[row['image_id']].append(row)
    require(all(len(v)==100 and {r['query_id'] for r in v}==set(range(100)) for v in grouped.values()),'Every original query exactly once required')
    return grouped


def rle_bytes(rle):
    return dict(size=list(rle['size']),counts=rle['counts'].encode('ascii') if isinstance(rle['counts'],str) else rle['counts'])


def render_original(torch,value,identity):
    """Independent two bilinear operations; no original renderer invocation."""
    h,w=identity['actual_img_shape']; H,W=identity['original_shape']
    padded=torch.nn.functional.interpolate(value,size=(800,800),mode='bilinear',align_corners=False)
    crop=padded[0,:,:h,:w]
    return torch.nn.functional.interpolate(crop[:,None],size=(H,W),mode='bilinear',align_corners=False)[:,0]


def head_state_check(torch,gate,start):
    results={}; evaluation,report,train=gate['evaluation'],gate['head_report'],gate['train']
    for key in KEYS:
        ref=evaluation['head_checkpoints'][key]; row=report['checkpoints'][key]; cp=torch.load(str(bound(ref)),map_location='cpu')
        meta,state=cp['meta'],cp['state_dict']; proof=read(bound(row['proof'])); epoch=evaluation['head_endpoint_epoch']; updates=1661*epoch
        require(row['checkpoint']==ref and meta['schema']==train.CP_SCHEMA and meta['source']==evaluation['head_training_entry_source'] and
                meta['spec']==evaluation['head_training_spec'] and meta['GPU_admission']==evaluation['head_training_GPU_admission'] and
                meta['phase']==evaluation['head_training_phase'] and meta['arm']==key[0] and meta['seed']==int(key.split('_seed')[1]) and
                meta['teacher_checkpoint']==report['teacher_checkpoint'] and meta['point_heads_source']==evaluation['point_heads_source'] and
                meta['native_weights_epoch']==24 and meta['native_weights_updated'] is False and meta['completed_head_epochs']==epoch and
                meta['updates']==updates and meta['parameter_tensors']==6 and meta['parameter_scalars']==40257 and meta['no_pending_gradients'] is True,
                'Own final head metadata differs')
        require(tuple(state)==NAMES and all(isinstance(state[n],torch.Tensor) and state[n].device.type=='cpu' and state[n].dtype==torch.float32 and
                tuple(state[n].shape)==s and bool(torch.isfinite(state[n]).all()) for n,s in zip(NAMES,SHAPES)),'Own six finite state tensors differ')
        values={name:digest(value) for name,value in state.items()}
        require(proof['checkpoint']==ref and proof['state']==values and start['head_state_proofs'][key]==values and
                proof['updates']==updates and proof['full_readback_exact'] is True,'Own head/source/START state proofs differ')
        opt=cp['optimizer_state_dict']; require(len(opt['param_groups'])==1 and len(opt['state'])==6,'Six actual head moment entries required')
        group=opt['param_groups'][0]; identifiers=group['params']
        require(group['lr']==3e-4 and group['weight_decay']==1e-4 and group['betas']==(.9,.999) and group['eps']==1e-8 and
                group['amsgrad'] is False and len(identifiers)==len(set(identifiers))==6 and set(identifiers)==set(opt['state']),'Head moment recipe differs')
        for name,shape,identifier in zip(NAMES,SHAPES,identifiers):
            entry=opt['state'][identifier]; step=entry['step']
            require(set(entry)=={'step','exp_avg','exp_avg_sq'} and isinstance(step,torch.Tensor) and step.device.type=='cpu' and
                    step.dtype==torch.float32 and step.numel()==1 and float(step)==updates and
                    all(isinstance(entry[k],torch.Tensor) and entry[k].device.type=='cpu' and entry[k].dtype==torch.float32 and
                        tuple(entry[k].shape)==shape and bool(torch.isfinite(entry[k]).all()) for k in ('exp_avg','exp_avg_sq')) and
                    proof['optimizer'][name]==dict({k:digest(v) for k,v in entry.items()},updates=updates),'Head endpoint moment/count proof differs')
        results[key]=dict(checkpoint=ref,proof=row['proof'],parameter_tensors=6,parameter_scalars=40257,epoch=epoch,updates=updates,
                          CPU_state_and_moment_proofs_checked=True,head_inference_CPU_replayed=False,training_trajectory_CPU_replayed=False)
        del cp,opt,state
    return results


def packet_check(torch,np,cv2,coco,value,row,index,gate):
    evaluation,report=gate['evaluation'],gate['report']; identity=row['identity']; iid=identity['image_id']; image=coco.imgs[iid]
    require(row['dataset_index']==value['dataset_index']==index and row['image_id']==iid and value['identity']==identity and
            value['schema']==gate['evaluator'].PACKET_SCHEMA and value['source']==report['source'] and value['spec']==report['spec'] and
            value['phase']==report['phase'] and value['teacher_checkpoint']==report['teacher_checkpoint'] and
            value['head_checkpoints']==evaluation['head_checkpoints'] and value['head_endpoint_epoch']==evaluation['head_endpoint_epoch'] and
            value['GPU_original_logits_saved'] is False and value['prediction_query_ids']==row['prediction_query_ids'] and
            value['field_digests']==row['final_fields'],'Actual all100 low-field packet identity differs')
    H,W=image['height'],image['width']; h,w=identity['actual_img_shape']
    require(identity['original_shape']==[H,W] and 0<h<=800 and 0<w<=800 and identity['batch_input_shape']==[800,800] and
            identity['native_input_gamma']==1. and identity['gray_decode']=='cv2.IMREAD_GRAYSCALE' and identity['target_gray_resize']=='nearest' and
            identity['ann_ids']==row['ann_ids']==[a['id'] for a in coco.imgToAnns[iid]] and row['raw_GT']==len(identity['ann_ids']) and
            len(set(identity['ann_ids']))==len(identity['ann_ids']) and row['head_seed']==100000+iid,'Actual image/all-GT/forward seed identity differs')
    image_ref=identity['image']; expected=(Path(evaluation['image_root'])/image['file_name']).resolve(strict=True)
    require(image_ref in gate['closure'].values() and bound(image_ref).resolve()==expected,'Actual original JPEG identity absent or differs')
    gray=cv2.imread(str(bound(image_ref)),cv2.IMREAD_GRAYSCALE)
    require(gray is not None and gray.dtype==np.uint8 and gray.shape==(H,W) and value['original_JPEG_gray_digest']==
            dict(shape=[H,W],dtype='uint8',sha256=hashlib.sha256(gray.tobytes()).hexdigest()),'Actual original JPEG gray digest differs')
    gray800=np.zeros((800,800),dtype=np.uint8); gray800[:h,:w]=cv2.resize(gray,(w,h),interpolation=cv2.INTER_NEAREST)
    require(identity['gray800_sha256']==hashlib.sha256(gray800.tobytes()).hexdigest(),'Original nearest-resize/pad JPEG mapping differs')
    fields=value['native_fields']; shapes=dict(F256=(1,256,200,200),e0=(1,100,256),M200=(1,100,200,200),final_class_logits=(1,100,2))
    require(set(fields)==set(shapes)==set(value['field_digests']),'Exactly four native fields required')
    for name,shape in shapes.items():
        t=fields[name]; require(isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==torch.float32 and tuple(t.shape)==shape and
                not t.requires_grad and bool(torch.isfinite(t).all()) and digest(t)==value['field_digests'][name],'Finite CPU native field/digest differs: '+name)
    qids=row['prediction_query_ids']; require(len(qids)==100 and all(type(q) is int for q in qids) and set(qids)==set(range(100)), 'All100 query ordinals required')
    class_packet=dict(digest(fields['final_class_logits'][0]),values=fields['final_class_logits'][0].tolist())
    require(class_packet==row['native_class_packet'] and len(row['band_counts_all100'])==100 and
            all(type(n) is int and 0<=n<=H*W for n in row['band_counts_all100']),'Native class/band population packet differs')
    for name in ('GT_removed_from_forward','loader_accesses_annotations','native610_SOURCE_unchanged','native_all10_versions_unchanged',
                 'no_GT_query_selection','all100_queries_retained','no_NMS','no_score_cutoff','fixed_native_score',
                 'new_empty_geometry_may_keep_positive_native_score','external_RNG_restored'):
        require(row[name] is True,'GPU SOURCE-observed scope differs: '+name)
    require(row['matching_or_loss_executed'] is False and row['independent_CPU_evaluation_replayed'] is False,'GPU scope differs')
    return fields


def replay_fields(torch,np,cv2,masks,coco,gate,populations,factors,started,spec):
    report=gate['report']; rows=read(bound(report['image_rows'])); trace=[read_json_line(line) for line in bound(report['trace']).read_text(encoding='utf-8').splitlines() if line.strip()]
    require(rows==trace and len(rows)==187 and [r['image_id'] for r in rows]==list(coco.imgs) and sum(r['raw_GT'] for r in rows)==380,
            'Complete actual187 TRACE/image-row identity differs')
    checks=[]; native=populations['native_official']
    with torch.no_grad():
        for index,(row,ref) in enumerate(zip(rows,report['final_field_packets'])):
            require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked CPU deadline exceeded')
            require(row['packet']==ref and ref in gate['closure'].values(),'Actual packet order/ref differs')
            packet=torch.load(str(bound(ref)),map_location='cpu'); fields=packet_check(torch,np,cv2,coco,packet,row,index,gate); iid=row['image_id']
            require(row['prediction_query_ids']==[r['query_id'] for r in native[iid]],'Export query order differs from low-field identity')
            lookup={key:{r['query_id']:r for r in grouped[iid]} for key,grouped in populations.items()}
            probabilities=fields['final_class_logits'][0].softmax(-1)[:,0]; stats=dict(native_threshold_flips=0,near_sign_pixels=0,near_band_pixels=0,
                 max_abs_class_difference=0.,max_abs_quality_difference=0.,changed_pixels={key:0 for key in CONTROLS[1:]},
                 empty_predictions={key:0 for key in CONTROLS},positive_S0_on_empty_refined={key:0 for key in CONTROLS[1:]})
            for first in range(0,100,NUMERIC['renderer_query_chunk']):
                L=render_original(torch,fields['M200'][:,first:first+NUMERIC['renderer_query_chunk']],row['identity'])
                require(bool(torch.isfinite(L).all()),'Independent native original logits not finite')
                for ordinal,z in enumerate(L,first):
                    exported=lookup['native_official'][ordinal]; support=masks.decode(rle_bytes(exported['segmentation'])).astype(bool)
                    B=torch.from_numpy(np.ascontiguousarray(support)); require(tuple(B.shape)==tuple(z.shape),'Original support shape differs')
                    tau=NUMERIC['CPU_sign_and_band_atol']+NUMERIC['CPU_sign_and_band_rtol']*z.abs()
                    near0=z.abs()<=tau; flips=B!=(z>0)
                    require(not bool((flips & ~near0).any()),'Native GPU/CPU support flip outside prelocked practical near-zero band')
                    near_band=(z.abs()-2).abs()<=tau
                    cpu_count=int((z.abs()<=2).sum()); gpu_count=row['band_counts_all100'][ordinal]
                    require(abs(cpu_count-gpu_count)<=int(near_band.sum()),'Recorded native band count differs outside practical boundary allowance')
                    factor=factors[iid,ordinal]; c=float(probabilities[ordinal]); quality=float((z.sigmoid()*B).sum()/(B.sum()+1e-6))
                    require(math.isclose(c,factor['class_probability'],rel_tol=NUMERIC['class_rtol'],abs_tol=NUMERIC['class_atol']) and
                            math.isclose(quality,factor['mask_quality'],rel_tol=NUMERIC['quality_rtol'],abs_tol=NUMERIC['quality_atol']),
                            'Independent native class/quality on exported GPU support differs')
                    stats['max_abs_class_difference']=max(stats['max_abs_class_difference'],abs(c-factor['class_probability']))
                    stats['max_abs_quality_difference']=max(stats['max_abs_quality_difference'],abs(quality-factor['mask_quality']))
                    stats['native_threshold_flips']+=int(flips.sum()); stats['near_sign_pixels']+=int(near0.sum()); stats['near_band_pixels']+=int(near_band.sum())
                    stats['empty_predictions']['native_official']+=int(not support.any())
                    if not support.any(): require(factor['mask_quality']==factor['official_score']==0,'Original native empty support score differs')
                    for control in CONTROLS[1:]:
                        actual=masks.decode(rle_bytes(lookup[control][ordinal]['segmentation'])).astype(bool)
                        require(actual.shape==support.shape,'Refined raster shape differs'); changed=torch.from_numpy(np.ascontiguousarray(actual!=support))
                        require(not bool((changed & (z.abs()>2+tau)).any()),'Refined geometry changed outside practical native boundary band')
                        stats['changed_pixels'][control]+=int(changed.sum()); empty=not actual.any(); stats['empty_predictions'][control]+=int(empty)
                        stats['positive_S0_on_empty_refined'][control]+=int(empty and exported['score']>0)
                del L
            checks.append(dict(dataset_index=index,image_id=iid,packet=ref,queries=100,JPEG_identity_CPU_checked=True,
                               fields_CPU_shape_digest_finite_checked=True,F_e_to_M_CPU_replayed=False,**stats))
            del packet,fields; print(json.dumps(dict(N07_CPU_evaluation_lowfields_images=index+1)),flush=True)
    return checks


def read_json_line(line):
    return json.loads(line,parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def factor_check(np,values,native,ids):
    require(type(values) is list and len(values)==18700,'Complete native score factors required'); factors={}; exact=0; error=0.
    for row,value in zip(native,values):
        require(set(value)=={'image_id','query_id','class_probability','mask_quality','official_score'} and
                (value['image_id'],value['query_id'])==(row['image_id'],row['query_id']) and value['official_score']==row['score'],
                'Native factor/query/score order differs')
        for name in ('class_probability','mask_quality','official_score'):
            v=value[name]; require(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 and float(np.float32(v))==v,'Bounded exported FP32 factor required')
        c,m,s=value['class_probability'],value['mask_quality'],value['official_score']; product=c*m; cpu=float(np.float32(c)*np.float32(m))
        allowance=(2.**-24/(1.-2.**-24))*abs(product)+2.**-126
        require(abs(s-product)<=allowance and abs(cpu-product)<=allowance,'Native FP32 product violates rounding/underflow allowance')
        key=(row['image_id'],row['query_id']); require(key not in factors,'Duplicate native score factor'); factors[key]=value
        exact+=int(cpu==s); error=max(error,abs(cpu-s))
    require(set(factors)=={(i,q) for i in ids for q in range(100)},'Native factor population differs')
    return factors,dict(entries=18700,CPU_FP32_product_exact_count=exact,max_abs_CPU_product_minus_export=error,FP32_rounding_underflow_checked=True)


def pair_pixels(np,masks,evaluator,g,d,iou):
    a=masks.decode(rle_bytes(evaluator.cocoGt.annToRLE(evaluator.cocoGt.anns[g]))).astype(bool)
    b=masks.decode(rle_bytes(evaluator.cocoDt.annToRLE(evaluator.cocoDt.anns[d]))).astype(bool)
    require(a.shape==b.shape,'Matched original GT/prediction shape differs')
    tp=int(np.count_nonzero(a & b)); fp=int(np.count_nonzero(~a & b)); fn=int(np.count_nonzero(a & ~b)); union=tp+fp+fn; total=2*tp+fp+fn
    observed=tp/union if union else 0.; require(abs(observed-iou)<=1e-12,'Ordinary matched pixel IoU differs from cached matrix')
    return dict(gt_id=g,dt_id=d,query_id=evaluator.cocoDt.anns[d]['query_id'],IoU=observed,Dice=2*tp/total if total else 0.,
                pixel_TP=tp,pixel_FP=fp,pixel_FN=fn,empty_union_IoU_convention=0)


def matching_images(np,masks,evaluator,ids):
    images=[]; p=evaluator.params
    for iid in ids:
        cells=[v for v in evaluator.evalImgs if v is not None and v['image_id']==iid and v['category_id']==1 and v['aRng']==p.areaRng[0] and v['maxDet']==100]
        require(len(cells)==1,'Exactly one actual area-all/maxDet100 cell required'); cell=cells[0]; gids,dids=cell['gtIds'],cell['dtIds']
        require(len(dids)==100 and cell['dtMatches'].shape==cell['dtIgnore'].shape==(10,100) and
                cell['gtMatches'].shape==(10,len(gids)) and cell['gtIgnore'].shape==(len(gids),),'Actual matching array population differs')
        original=evaluator._gts[iid,1]; dts=evaluator._dts[iid,1]; original_ids=[r['id'] for r in original]
        ordered=np.argsort([-r['score'] for r in dts],kind='mergesort')[:100]
        require(dids==[dts[int(i)]['id'] for i in ordered] and set(gids)==set(original_ids),'Actual cached matrix row/column ID mapping differs')
        matrix=np.asarray(evaluator.ious[iid,1],dtype=np.float64) if original_ids else np.zeros((100,0),dtype=np.float64)
        require(matrix.shape==(100,len(original_ids)) and bool(np.isfinite(matrix).all()),'Finite original-order cached IoU matrix required')
        columns={g:i for i,g in enumerate(original_ids)}; positions={g:i for i,g in enumerate(gids)}; eligible={g for g,v in zip(gids,cell['gtIgnore']) if not bool(v)}
        require(all(not evaluator.cocoGt.anns[g].get('iscrowd',0) for g in eligible),'Nonignored all-area GT must be ordinary')
        for name in ('gtMatches','dtMatches'):
            a=cell[name]; require(bool(np.isfinite(a).all()) and bool((a==np.floor(a)).all()),'Matching IDs must be finite integral values')
        pairs={}; thresholds=[]
        for ti,t in enumerate(p.iouThrs):
            matched=set(); counts=dict(TP=0,FN=0,FP=0,ignored_DT=0,ignored_GT=len(gids)-len(eligible)); gt_rows=[]
            for di,d in enumerate(dids):
                g=int(cell['dtMatches'][ti,di]); ignored=bool(cell['dtIgnore'][ti,di])
                if ignored: counts['ignored_DT']+=1
                elif g:
                    require(g in eligible and g not in matched and int(cell['gtMatches'][ti,positions[g]])==d,'Ordinary one-to-one reciprocity differs')
                    matched.add(g); counts['TP']+=1; pairkey='%d:%d'%(d,g)
                    if pairkey not in pairs: pairs[pairkey]=pair_pixels(np,masks,evaluator,g,d,float(matrix[di,columns[g]]))
                else: counts['FP']+=1
            counts['FN']=len(eligible)-len(matched)
            for gi,g in enumerate(gids):
                d=int(cell['gtMatches'][ti,gi]); ignored=bool(cell['gtIgnore'][gi]); pairkey='%d:%d'%(d,g) if d and not ignored else None
                require(ignored or (g in matched)==bool(d),'Actual GT array and ordinary DT assignments differ')
                gt_rows.append(dict(gt_id=g,ignored=ignored,matched_DT=d,FN=not ignored and not d,TP_pair_key=pairkey))
            require(counts['TP']+counts['FP']+counts['ignored_DT']==100 and counts['TP']+counts['FN']==len(eligible) and
                    counts['FN']==sum(r['FN'] for r in gt_rows),'Actual one-to-one counts do not conserve population')
            pixel={name:sum(pairs[r['TP_pair_key']][name] for r in gt_rows if r['TP_pair_key']) for name in ('pixel_TP','pixel_FP','pixel_FN')}
            thresholds.append(dict(threshold=float(t),counts=counts,GT=gt_rows,matched_pair_pixel_totals=pixel))
        images.append(dict(image_id=iid,actual_COCO_arrays=dict(gtIds=gids,dtIds=dids,gtIgnore=cell['gtIgnore'].astype(bool).tolist(),
                           dtIgnore=cell['dtIgnore'].astype(bool).tolist(),gtMatches=cell['gtMatches'].astype(np.int64).tolist(),
                           dtMatches=cell['dtMatches'].astype(np.int64).tolist(),dtScores=[float(v) for v in cell['dtScores']],area_range=list(cell['aRng']),maxDet=100),
                           thresholds=thresholds,TP_pairs=pairs))
    require(sum(len(v['actual_COCO_arrays']['gtIds']) for v in images)==380,'Every original GT including FN/ignore must remain')
    return images


def matching_summary(images):
    result=[]
    for ti in range(10):
        cells=[image['thresholds'][ti] for image in images]
        counts={key:sum(c['counts'][key] for c in cells) for key in ('TP','FN','FP','ignored_DT','ignored_GT')}
        pixels={key:sum(c['matched_pair_pixel_totals'][key] for c in cells) for key in ('pixel_TP','pixel_FP','pixel_FN')}
        require(counts['TP']+counts['FP']+counts['ignored_DT']==18700 and counts['TP']+counts['FN']+counts['ignored_GT']==380,'Aggregate matching conservation differs')
        result.append(dict(threshold=cells[0]['threshold'],counts=counts,matched_pair_pixel_totals=pixels,
                           matched_pair_pixels_are_conditional_on_threshold_and_assignment=True))
    return result


def paired_differences(left,right):
    details=[]; summary=[]
    for ti in range(10):
        records=[]; groups=dict(both_matched=0,gained_match=0,lost_match=0,both_FN=0,ignored_GT=0,same_query_both_matched=0,switched_query_both_matched=0)
        for a,b in zip(left,right):
            require(a['image_id']==b['image_id'],'Paired original image order differs'); ar=a['thresholds'][ti]['GT']; br=b['thresholds'][ti]['GT']
            lookup={r['gt_id']:r for r in br}; require(set(lookup)=={r['gt_id'] for r in ar},'Paired original GT identities differ')
            for x in ar:
                y=lookup[x['gt_id']]; require(x['ignored']==y['ignored'],'Paired official ignore flags differ')
                xp=a['TP_pairs'][x['TP_pair_key']] if x['TP_pair_key'] else None; yp=b['TP_pairs'][y['TP_pair_key']] if y['TP_pair_key'] else None
                status='ignored_GT' if x['ignored'] else ('both_matched' if xp and yp else 'lost_match' if xp else 'gained_match' if yp else 'both_FN')
                groups[status]+=1; delta=None
                if xp and yp:
                    same=xp['query_id']==yp['query_id']; groups['same_query_both_matched' if same else 'switched_query_both_matched']+=1
                    delta={k:yp[k]-xp[k] for k in ('IoU','Dice','pixel_TP','pixel_FP','pixel_FN')}
                records.append(dict(image_id=a['image_id'],gt_id=x['gt_id'],status=status,left=xp,right=yp,right_minus_left=delta))
        both=[r['right_minus_left'] for r in records if r['right_minus_left'] is not None]
        require(sum(groups[k] for k in ('both_matched','gained_match','lost_match','both_FN','ignored_GT'))==380,'Paired GT population differs')
        summary.append(dict(threshold=left[0]['thresholds'][ti]['threshold'],counts=groups,
                            both_matched_conditional_mean_IoU_delta=math.fsum(r['IoU'] for r in both)/len(both) if both else None,
                            both_matched_conditional_pixel_deltas={k:sum(r[k] for r in both) for k in ('pixel_TP','pixel_FP','pixel_FN')},
                            conditional_subsets_not_global_or_causal=True))
        details.append(dict(threshold=summary[-1]['threshold'],GT_pairs=records))
    return summary,details


def official_metrics(coco,predictions,ids,COCOeval):
    dataset=copy.deepcopy(coco); dataset.dataset.setdefault('info',{})
    evaluator=COCOeval(dataset,dataset.loadRes(copy.deepcopy(predictions)),'segm')
    evaluator.params.imgIds=list(ids); evaluator.params.catIds=[1]; evaluator.params.maxDets=[1,10,100]
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize(); stats=evaluator.stats.tolist()
    require(len(evaluator.params.iouThrs)==10 and all(abs(float(t)-(.5+.05*i))<=1e-12 for i,t in enumerate(evaluator.params.iouThrs)),
            'Locked ten IoU thresholds differ')
    stats_close(stats,stats,'Official COCO12 contains invalid statistics')
    return evaluator,dict(stats=stats,metrics=dict(zip(METRICS,stats)),iouType='segm',maxDets=[1,10,100],original_GT_preserved=True,
                          score_filter=None,official_COCO_crowd_ignore_semantics=True)


def atomic_write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp'); path.parent.mkdir(parents=True,exist_ok=True)
    require(not path.exists() and not temporary.exists(),'Fresh output required')
    with temporary.open('x',encoding='utf-8',newline='\n') as stream: json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')
    temporary.replace(path)


def qualify(spec_path):
    started=time.monotonic(); path=Path(spec_path); require(path.is_absolute(),'Absolute ROOT CPU SPEC required'); path=path.resolve(strict=True)
    spec=read(path); gate=source_gate(path,spec); report,evaluation=gate['report'],gate['evaluation']
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1],'Actual CPU01 required before scientific imports')
    import torch
    import numpy as np
    import cv2
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    require(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and not torch.cuda.is_initialized(),'Original CPU libraries/no initialized CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2); rng=torch.random.get_rng_state().clone()
    libraries=imported_files(torch,np,cv2,COCO,COCOeval,masks,gate)
    coco=COCO(str(bound(spec['DEV']))); ids=list(coco.imgs)
    require(len(ids)==len(set(ids))==187 and ids==[r['id'] for r in coco.dataset['images']] and len(coco.anns)==380 and set(coco.cats)=={1},
            'Full original DEV187/380/category1 canonical population required')
    predictions={control:read(bound(report['predictions'][control])) for control in CONTROLS}
    populations={control:population(coco,rows,ids) for control,rows in predictions.items()}; native=predictions['native_official']
    for control in CONTROLS[1:]:
        require(all({k:v for k,v in a.items() if k!='segmentation'}=={k:v for k,v in b.items() if k!='segmentation'}
                    for a,b in zip(native,predictions[control])),'Only geometry may change; fixed S0/query/class/order including empty geometry must remain exact')
    anchor_rows=read(bound(spec['E24_reference_predictions'])); population(coco,anchor_rows,ids)
    anchor={(r['image_id'],r['query_id']):r for r in anchor_rows}
    require(all(anchor[r['image_id'],r['query_id']]==r for r in native),'Own original native export differs from bound strong E24 export by query identity')
    start=read(bound(report['start']))
    require(start['schema']=='N07-original-detail-evaluation-start-v1' and start['source']==report['source'] and start['spec']==report['spec'] and
            start['GPU_admission']==report['GPU_admission'] and start['phase']==report['phase'] and start['recipe']==report['recipe'] and
            start['head_training_report']==report['head_training_report'] and start['head_endpoint_epoch']==report['head_endpoint_epoch'] and
            start['head_checkpoints']==report['head_checkpoints'] and start['cache_CPU_qualification']==report['cache_CPU_qualification'] and
            start['teacher_checkpoint']==report['teacher_checkpoint'] and start['no_TEST'] is True and start['no_DEV_selection'] is True,
            'Actual fixed GPU START differs')
    heads=head_state_check(torch,gate,start); factors,factor_proof=factor_check(np,read(bound(report['native_query_score_factors'])),native,ids)
    packet_checks=replay_fields(torch,np,cv2,masks,coco,gate,populations,factors,started,spec)
    controls={}; matching={}; summaries={}
    for control in CONTROLS:
        require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked CPU COCO deadline exceeded')
        evaluator,metrics=official_metrics(coco,predictions[control],ids,COCOeval); expected=report['control_results'][control]
        require(all(expected[k]==metrics[k] for k in ('iouType','maxDets','original_GT_preserved','score_filter','official_COCO_crowd_ignore_semantics')),
                'Locked GPU COCO conventions differ')
        stats_close(metrics['stats'],expected['stats'],'Independent official COCO12 differs: '+control)
        require(expected['metrics']==dict(zip(METRICS,expected['stats'])),'GPU metric names/stat alignment differs')
        matching[control]=matching_images(np,masks,evaluator,ids); summaries[control]=matching_summary(matching[control])
        controls[control]=dict(predictions=report['predictions'][control],COCO=metrics,one_to_one=summaries[control],all100_identity_and_fixed_scores_CPU_checked=True)
        del evaluator; gc.collect(); print(json.dumps(dict(N07_CPU_evaluation_COCO_control=control)),flush=True)
    stats_close(controls['native_official']['COCO']['stats'],spec['expected_E24_COCO_stats'],'Strong E24 COCO12 anchor differs')
    contrasts=[('native_official',control) for control in CONTROLS[1:]]+[
        ('C_seed%d_fixed_native_score'%seed,'R_seed%d_fixed_native_score'%seed) for seed in SEEDS]
    paired={}; paired_details={}
    for left,right in contrasts:
        label=right+'_minus_'+left; paired[label],paired_details[label]=paired_differences(matching[left],matching[right])
    require(torch.equal(rng,torch.random.get_rng_state()) and not torch.cuda.is_initialized() and 'mmdet' not in sys.modules,
            'CPU reader changed RNG, initialized CUDA or loaded native model')
    for ref in spec['sources']: bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and reference(spec['CPU_admission_file'])==gate['owner_ref'] and
            libraries==imported_files(torch,np,cv2,COCO,COCOeval,masks,gate) and psutil.Process().cpu_affinity()==[0,1] and
            torch.get_num_threads()==torch.get_num_interop_threads()==2,'Final inputs/libraries/resources changed')
    atomic_write(gate['details'],dict(schema='N07-original-detail-fixed-endpoint-CPU-matching-details-v1',controls=matching,
                                     paired_differences=paired_details,population='all187 images/all380 GT, including FN and ignore',
                                     matching='actual COCO area-all/maxDet100/10 thresholds; no best-query or GT-directed inference'))
    q=dict(schema=Q_SCHEMA,status=STATUS,phase=spec['phase'],source=spec['evaluation_entry_source'],reader_source=gate['source'],spec=gate['spec_ref'],
           evaluation_spec=spec['evaluation_spec'],evaluation_report=spec['evaluation_report'],evaluation_GPU_RAW=spec['evaluation_GPU_RAW'],
           evaluation_GPU_admission=spec['evaluation_GPU_admission'],CPU_admission=gate['owner_ref'],evaluation_phase=evaluation['phase'],
           RAW_inputs=spec['RAW_inputs'],prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],head_training_report=report['head_training_report'],
           head_checkpoints=report['head_checkpoints'],head_endpoint_epoch=report['head_endpoint_epoch'],teacher_checkpoint=report['teacher_checkpoint'],
           cache_CPU_qualification=report['cache_CPU_qualification'],DEV=spec['DEV'],E24_reference_Q=spec['E24_reference_Q'],
           E24_reference_predictions=spec['E24_reference_predictions'],expected_E24_COCO_stats=spec['expected_E24_COCO_stats'],
           controls=list(CONTROLS),control_results=controls,images=187,raw_GT=380,predictions_per_control=18700,
           packet_refs=report['final_field_packets'],packet_checks=packet_checks,head_endpoint_state_checks=heads,native_factor_check=factor_proof,
           paired_one_to_one_differences=paired,matching_details=reference(gate['details']),numeric_protocol=NUMERIC,
           complete_identity_fixed_scores_and_empty_query_population_CPU_checked=True,original_JPEG_identity_CPU_checked=True,
           five_official_COCO12_CPU_recomputed=True,strong_E24_native_export_and_COCO_reference_CPU_checked=True,
           official_one_to_one_matching_and_matched_pair_pixels_CPU_checked=True,native_M200_original_render_CPU_replayed=True,
           practical_native_support_quality_and_band_consistency_CPU_checked=True,certified_GPU_original_logit_error_bound=False,
           original_native_teacher_CPU_replayed=False,F_e_to_M_CPU_replayed=False,all_four_head_dense_inference_CPU_replayed=False,
           training_gradient_or_optimizer_trajectory_CPU_replayed=False,native610_unchanged_SOURCE_observed=True,native610_unchanged_independent_CPU_verified=False,
           no_optimizer=True,no_weight_update=True,no_GT_query_selection=True,no_NMS=True,no_score_cutoff=True,no_TEST=True,best_DEV_selection=False,
           actual_imported_files=libraries,CPU_execution=dict(affinity=[0,1],threads=2,CUDA_VISIBLE_DEVICES='',CUDA_initialized=False),elapsed_seconds=time.monotonic()-started,
           resource_suitability_measured_by_this_SOURCE=False,evidence_limits=[
             'Complete five-export identity, fixed-score, official COCO and actual one-to-one matching are independently recomputed; no GT chooses a prediction.',
             'Matched-pair pixel totals and both-matched deltas condition on threshold/query assignment. Report gained/lost/FN/ignored GT; these subsets are not global pixel risk or causal attribution.',
             'Original M200 is independently CPU rendered. GPU original logits are absent: prelocked near-zero/band allowances are practical consistency tests, not certified GPU error bounds or pixel-logit allclose.',
             'F/e/M/C, all187 JPEG identities and head endpoint states are checked; native teacher, F/e-to-M projection, all four dense head outputs and full training trajectory are not independently replayed.',
             'C/R share capacity and paired training recipe but differ in real JPEG information. Fixed native S0 is a conditional geometry diagnostic, not a quality calibration or deployable routing claim.',
             'DEV is historically exposed; no blind TEST, convergence, method benefit, SOTA or scientific decision is asserted. Runtime/memory/deadline suitability remains subject to actual ROOT measurements.',
             'ROOT separately closes this current CPU actual owned terminal; this Q does not self-claim its final RAW.'])
    atomic_write(Path(spec['Q_path']),q); print(json.dumps(dict(status=STATUS,Q=reference(spec['Q_path']),details=q['matching_details'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True); parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args(); require(args.owner_execute,'Explicit single ROOT-owned CPU execution required'); qualify(args.spec)
