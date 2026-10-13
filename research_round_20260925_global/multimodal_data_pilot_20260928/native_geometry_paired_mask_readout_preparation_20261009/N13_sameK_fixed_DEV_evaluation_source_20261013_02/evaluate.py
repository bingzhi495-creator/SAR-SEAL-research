"""UNBOUND N13 full187 paired fixed-epoch GPU evaluation; ROOT owns admission.
Full natural100 native masks/class/order/S0 exact before ROI refinement.
Shared exact GPU L selector is produced once or explicitly reused. No training.
"""
import argparse
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

OWNER_SHA="ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924"
RAW_KEYS=("receipt","phase","admission","phase_binding","bootstrap")
THREAD_ENV=("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")
def require(ok,message):
    if not ok: raise RuntimeError(message)

def read(path):
    def reject(value): raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)

def ref_shape(ref):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and
        type(ref['path']) is str and Path(ref['path']).is_absolute() and
        type(ref['bytes']) is int and ref['bytes']>=0 and type(ref['sha256']) is str and
        len(ref['sha256'])==64 and set(ref['sha256'])<=set('0123456789abcdef'),'Malformed absolute pinned reference')

def reference(path):
    path=Path(path).resolve(strict=True); h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())

def bound(ref):
    ref_shape(ref); require(reference(ref['path'])==ref,'Pinned input changed: '+ref['path']); return Path(ref['path'])

def terminal(ref,phase,code=0):
    # Reads only original RAW JSON; this function never hashes or imports science.
    ref_shape(ref); path=Path(ref['path'])
    require(path.name=='PARENT_RECEIPT.json' and type(phase) is str and phase and path.parent.name==phase,'Original RAW locator required')
    raw=read(path)
    require(raw['status']=='ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode']==code and
        type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
        raw['held_handle_exit']==dict(wait_state='WAIT_OBJECT_0',actual_exit_code=code) and
        type(raw['held_handle_exit']['actual_exit_code']) is int and raw['numeric_birth_absence']['original_birth_absent'] is True and
        raw.get('error') is None and raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors']==[],
        'Actual held exit/reap/original birth absence required before every SHA/import')
    if 'phase' in raw: require(raw['phase']==phase,'RAW phase differs')
    peak,sha=raw['whole_GPU_observed_peak_MiB'],raw['admission_sha256']
    require(type(sha) is str and len(sha)==64 and set(sha)<=set('0123456789abcdef') and
        type(peak) in (int,float) and math.isfinite(peak) and 0<=peak<=8192 and
        raw['whole_GPU_used_limit_MiB']==8192 and raw['whole_leaf_VRAM_certified'] is False and
        raw['GPU_scope']=='whole GPU observed, not per-process WDDM allocation','RAW resources differ')
    return raw

def pre_gate(spec,minimum):
    rows=spec['RAW_inputs']; expected=spec['normal_RAW_count']
    require(type(expected) is int and expected>=minimum and type(rows) is list and len(rows)==expected and
        len({row['receipt']['path'] for row in rows})==expected,'All prelocked normal RAW ancestors required')
    normals=[terminal(row['receipt'],row['phase']) for row in rows]
    failed=terminal(spec['prior_failed_CPU_receipt'],Path(spec['prior_failed_CPU_receipt']['path']).parent.name,1)
    extra=spec['additional_failed_RAW_inputs']
    require(type(extra) is list and len(extra)==7 and all(type(row) is dict and set(row)=={'receipt','phase','returncode'} and
        type(row['returncode']) is int and row['returncode']==1 for row in extra),'Preserve exact three-key extra failed7 rows, eight failures total')
    failures=[failed]+[terminal(row['receipt'],row['phase'],1) for row in extra]
    failed_paths=[spec['prior_failed_CPU_receipt']['path']]+[row['receipt']['path'] for row in extra]
    require(len(set(failed_paths))==8 and not {row['receipt']['path'] for row in rows}.intersection(failed_paths),
        'Normal ancestors and eight preserved failures must be distinct')
    all_rows=[dict(row,returncode=0) for row in rows]+[dict(receipt=spec['prior_failed_CPU_receipt'],phase=Path(spec['prior_failed_CPU_receipt']['path']).parent.name,returncode=1)]+extra
    pins=spec['terminal_identity_pins']; locators=[row['receipt']['path'] for row in all_rows]
    require(type(pins) is dict and set(pins)==set(locators),'Exact all-normal-plus-eight-failure identity pins required')
    for row,raw in zip(all_rows,normals+failures):
        pin=pins[row['receipt']['path']]
        require(type(pin) is dict and set(pin)=={'phase','returncode','original_identity','admission_sha256'} and
            pin['phase']==row['phase'] and type(pin['returncode']) is int and pin['returncode']==row['returncode'] and
            pin['original_identity']==raw['original_identity'] and pin['admission_sha256']==raw['admission_sha256'],
            'Actual original identity/admission must match its prelocked pin before every SHA/import')
    # FIRST HASH may occur only after ALL above actual terminal checks.
    closure={ref['path']:ref for ref in spec['sources']}
    require(len(closure)==len(spec['sources']),'Unique SOURCE closure required')
    for ref in spec['sources']: bound(ref)
    require(all(row[k] in closure.values() for row in rows for k in RAW_KEYS if k!='phase') and
        spec['prior_failed_CPU_receipt'] in closure.values() and all(row['receipt'] in closure.values() for row in extra),
        'RAW owner metadata and failed receipts must be sources-bound')
    return normals,closure

def load(ref,name):
    path=bound(ref); source=path.read_bytes(); require(len(source)==ref['bytes'] and hashlib.sha256(source).hexdigest()==ref['sha256'],'SOURCE changed')
    module=types.ModuleType(name); module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__); return module


SCHEMA='ROOT-N13-sameK-fixed-fullDEV-GPU-spec-v1'
REPORT_SCHEMA='N13-sameK-fixed-fullDEV-GPU-report-v1'
STATUS='N13_SAMEK_FIXED_FULLDEV_GPU_COMPLETE_PENDING_CPU'
ARMS=('LOG_PR','GEO_PR')
NAMES=('hidden1.weight','hidden1.bias','hidden2.weight','hidden2.bias','hidden3.weight','hidden3.bias','final.weight','final.bias')
SHAPES=((91,257),(91,),(91,92),(91,),(91,92),(91,),(1,92),(1,))


def deadline(started,spec):
    require(time.monotonic()-started<spec['deadline_seconds'],'Locked full187 evaluation deadline exceeded; no retry')


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def digest(value):
    a=value.detach().cpu().contiguous().numpy()
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def source_gate(path,spec):
    require(spec['schema']==SCHEMA,'Exact full187 paired GPU schema required')
    raws,closure=pre_gate(spec,41)
    lineage=load(spec['endpoint_lineage_source'],'_N13_fullDEV_lineage')
    lineage.check_recipe(spec)
    gate=lineage.own_gate(path,spec,__file__,closure,raws)
    gate.update(lineage=lineage)
    gate.update(lineage.endpoint_gate(spec,closure,gate['owners']))
    prior,q=lineage.prior_gate(spec,closure,gate['owners'])
    require(prior['teacher_checkpoint']==gate['cache_report']['teacher_checkpoint'],
        'DEV packet teacher and qualified FIT teacher differ')
    gate.update(report=prior,prior=prior,prior_q=q,packets=q['packet_refs'],
        selector=lineage.selector_parent_gate(spec,closure,gate['owners']))
    return gate


def load_heads(torch,spec,gate):
    factory=load(spec['point_heads_source'],'_N13_DEV_same_PR_factory')
    parent=gate['parent']; heads,states,proofs={},{},{}
    outside=torch.get_rng_state().clone()
    for arm in ARMS:
        ref=spec['endpoint']['checkpoints'][arm]; row=parent['checkpoints'][arm]
        cp=torch.load(str(bound(ref)),map_location='cpu'); state,meta=cp['state_dict'],cp['meta']
        require(type(state) is dict and tuple(state)==NAMES and
            all(isinstance(state[n],torch.Tensor) and state[n].device.type=='cpu' and state[n].dtype==torch.float32 and
                tuple(state[n].shape)==shape and not state[n].requires_grad and bool(torch.isfinite(state[n]).all())
                for n,shape in zip(NAMES,SHAPES)) and sum(v.numel() for v in state.values())==40497,
            'Finite exact8/40497 PR endpoint state required')
        require(meta['schema']==parent['CP_schema'] and meta['source']==spec['endpoint']['training_entry_source'] and
            meta['spec']==spec['endpoint']['training_spec'] and meta['GPU_admission']==spec['endpoint']['training_GPU_admission'] and
            meta['phase']==parent['spec']['phase'] and meta['arm']==arm and meta['seed']==spec['seed'] and
            meta['completed_head_epochs']==spec['head_endpoint_epoch'] and meta['updates']==1661*spec['head_endpoint_epoch'] and
            meta['head_factory_source']==spec['point_heads_source'] and meta['cache_CPU_qualification']==spec['cache_CPU_qualification'] and
            meta['cache_spec']==spec['cache_spec'] and meta['cache_report']==spec['cache_report'] and
            meta['recipe']==spec['training_recipe'] and meta['parameter_tensors']==8 and meta['parameter_scalars']==40497 and
            meta['native_weights_updated'] is False and meta['no_pending_gradients'] is True and
            row['checkpoint']==ref and parent['Q']['checkpoints'][arm]==row,
            'Exact qualified endpoint/CPU proof identity differs')
        proof=read(bound(row['proof'])); values={name:digest(value) for name,value in state.items()}
        require(proof['checkpoint']==ref and proof['state']==values and proof['updates']==meta['updates'] and
            proof['full_readback_exact'] is True,'Actual head checkpoint state proof differs')
        head=factory.make_point_head(torch,spec['seed']); head.load_state_dict(state,strict=True); head.eval()
        for p in head.parameters(): p.requires_grad_(False)
        heads[arm]=head; states[arm]=state; proofs[arm]=values
        del cp
    require(torch.equal(outside,torch.get_rng_state()) and not torch.cuda.is_initialized(),
        'CPU endpoint loading changed RNG or initialized CUDA')
    return heads,states,proofs


def selector_packet(torch,spec,gate,index,row,ref,identity,qids,L,destination):
    lineage=gate['lineage']; expected=dict(dataset_index=index,image_id=row['image_id'],identity=identity,
        native_packet=ref,natural_query_ids=list(range(100)),native_export_query_ids=qids,
        renderer_source=spec['renderer_source'])
    actual=L.detach().cpu().contiguous(); value_digest=digest(actual)
    if spec['selector_mode']=='produce':
        packet=dict(schema=lineage.SELECTOR_PACKET_SCHEMA,source=gate['source'],spec=gate['spec_ref'],
            GPU_admission=gate['admission_ref'],phase=spec['phase'],native_L=actual,L_digest=value_digest,**expected)
        target=destination/('SELECTOR_%03d_IMAGE%d.pth'%(index,row['image_id']))
        with target.open('xb') as stream: torch.save(packet,stream)
        replay=torch.load(str(target),map_location='cpu')
        require(replay['L_digest']==value_digest and digest(replay['native_L'])==value_digest and
            torch.equal(actual,replay['native_L']),'Exact saved selector readback differs')
        carrier=reference(target)
        del packet,replay
    else:
        old=gate['selector']['manifest']['rows'][index]; carrier=old['carrier']
        require(all(old[k]==v for k,v in expected.items()) and old['L_digest']==value_digest,
            'Reused native selector row/query/renderer/shape differs')
        replay=torch.load(str(bound(carrier)),map_location='cpu')
        require(replay['schema']==lineage.SELECTOR_PACKET_SCHEMA and
            replay['source']==gate['selector']['parent']['evaluation_entry_source'] and
            replay['spec']==gate['selector']['parent']['evaluation_spec'] and
            replay['GPU_admission']==gate['selector']['parent']['evaluation_GPU_admission'] and
            all(replay[k]==v for k,v in expected.items()) and replay['L_digest']==value_digest and
            digest(replay['native_L'])==value_digest and torch.equal(actual,replay['native_L']),
            'Actual current natural100 GPU L bytes must exactly match immutable selector')
        del replay
    return dict(carrier=carrier,L_digest=value_digest,**expected)


def refine(torch,head,arm,F_original,e,L,regions,spec,started):
    result=L.clone(); feature=F_original.flatten(1); calls=0; counts=[]
    for q in range(100):
        indices=torch.from_numpy(regions[arm][q]).to(device=L.device,dtype=torch.long)
        counts.append(int(indices.numel()))
        for first in range(0,len(indices),2048):
            deadline(started,spec); idx=indices[first:first+2048]; native=L[q].flatten()[idx]
            predicted=head(feature[:,idx].T,e[0,q].expand(len(idx),256),native,L.new_zeros((len(idx),49)),arm='PR')
            residual=predicted['residual']
            require(residual.dtype==torch.float32 and residual.device==L.device and tuple(residual.shape)==tuple(native.shape) and
                bool(torch.isfinite(residual).all()),'Finite ungated original PR residual required')
            result[q].flatten()[idx]=native+residual; calls+=1
        outside=torch.ones(L[q].numel(),device=L.device,dtype=torch.bool); outside[indices]=False
        require(torch.equal(result[q].flatten()[outside],L[q].flatten()[outside]),
            'Refinement changed outside exact arm ROI')
    return result,counts,calls


def evaluate_image(torch,np,ndimage,masks,renderer,kernels,reader,n10,coco,packet,row,index,ref,native,heads,gate,spec,started,selector_dir):
    fields,identity,qids=n10.packet_check(torch,reader,coco,packet,row,index,ref,gate)
    require([v['query_id'] for v in native]==qids,'Original export ordinal/query ID alignment differs')
    F=e=M=C=L=top=order=ordered=binary=quality=scores=F_original=refined=None
    begun=time.monotonic()
    with torch.no_grad():
        F=fields['F256'].to('cuda:0'); e=fields['e0'].to('cuda:0')
        M=fields['M200'].to('cuda:0'); C=fields['final_class_logits'].to('cuda:0')
        require(torch.equal(torch.einsum('bqc,bchw->bqhw',e,F),M),'Original native M=eF exact GPU projection differs')
        top,order=kernels.order(torch,C,types.SimpleNamespace(num_classes=1))
        require(order.cpu().tolist()==qids,'Original sorted=False all100 query order differs')
        L=renderer.render_original(torch,M,identity)
        require(L.dtype==torch.float32 and L.device.type=='cuda' and tuple(L.shape)==(100,)+tuple(identity['original_shape']) and
            bool(torch.isfinite(L).all()),'Exact natural100 FP32 original renderer ABI differs')
        ordered=L[order]; binary,quality,scores=kernels.quality(torch,ordered,top)
        baseline=[]; native_masks=binary.cpu().numpy().astype(np.uint8); values=scores.cpu().tolist()
        for ordinal,qid in enumerate(qids):
            current=dict(image_id=row['image_id'],query_id=qid,category_id=1,score=float(values[ordinal]),
                segmentation=n10.segmentation(np,masks,native_masks[ordinal]))
            require(current==native[ordinal],'Original all100 query/class/order/mask/S0 must reproduce exactly')
            if not bool(native_masks[ordinal].any()): require(current['score']==0.,'Original empty query score differs')
            baseline.append(current)
        native_time=time.monotonic()-begun
        M=C=top=order=ordered=binary=quality=scores=None
        tick=time.monotonic()
        selector_row=selector_packet(torch,spec,gate,index,row,ref,identity,qids,L,selector_dir)
        selector_io=time.monotonic()-tick
        exact=L.detach().cpu().contiguous().numpy(); tick=time.monotonic()
        region_indices={arm:[] for arm in ARMS}; region_digests={arm:[] for arm in ARMS}; Ks=[]; fallbacks=[]
        for q in range(100):
            deadline(started,spec)
            P,regions,K,fallback=gate['common'].gpu_regions(np,ndimage,exact[q])
            Ks.append(K); fallbacks.append(bool(fallback))
            for arm in ARMS:
                idx=np.flatnonzero(regions[arm]).astype(np.int64)
                require(len(idx)==K,'Exact sameK per-query regions required')
                region_indices[arm].append(idx); region_digests[arm].append(
                    dict(shape=list(regions[arm].shape),dtype='bool',sha256=hashlib.sha256(regions[arm].tobytes()).hexdigest()))
        region_time=time.monotonic()-tick; del exact
        require(Ks==row['band_counts_all100'],'Natural100 original LOG K population differs')
        tick=time.monotonic(); F_original=n10.original_feature(torch,renderer,F,identity); F=None
        torch.cuda.synchronize(0)
        feature_time=time.monotonic()-tick; predictions={}; diagnostics={}
        for arm in ARMS:
            tick=time.monotonic(); refined,counts,calls=refine(torch,heads[arm],arm,F_original,e,L,region_indices,spec,started)
            torch.cuda.synchronize(0); head_time=time.monotonic()-tick
            require(counts==Ks,'Arm editing budgets differ')
            tick=time.monotonic(); exported=n10.fixed_score_rows(np,masks,baseline,refined); per_query=[]
            for ordinal,original in enumerate(baseline):
                q=original['query_id']; before=native_masks[ordinal].astype(bool); after=(refined[q]>0).cpu().numpy().astype(bool)
                added=int(np.count_nonzero(~before & after)); removed=int(np.count_nonzero(before & ~after))
                per_query.append(dict(query_id=q,native_export_ordinal=ordinal,fixed_native_score=original['score'],K=Ks[q],
                    native_foreground_pixels=int(before.sum()),refined_foreground_pixels=int(after.sum()),
                    added_pixels=added,removed_pixels=removed,changed_pixels=added+removed,native_empty=not bool(before.any()),
                    refined_empty=not bool(after.any()),native_empty_new_nonempty_keeps_zero_score=not bool(before.any()) and bool(after.any()),
                    new_empty_keeps_positive_native_score=bool(before.any()) and not bool(after.any()) and original['score']>0))
            label=gate['lineage'].controls(spec['seed'],spec['head_endpoint_epoch'])[ARMS.index(arm)+1]
            predictions[label]=exported; diagnostics[label]=dict(arm=arm,seed=spec['seed'],head_point_chunk_calls=calls,
                head_seconds=head_time,export_seconds=time.monotonic()-tick,query_diagnostics=per_query,
                **{key:sum(v[key] for v in per_query) for key in
                    ('added_pixels','removed_pixels','changed_pixels','native_empty_new_nonempty_keeps_zero_score','new_empty_keeps_positive_native_score')})
            refined=None
        summary=dict(dataset_index=index,image_id=row['image_id'],packet=ref,queries=100,native_query_order=qids,
            native_M_equals_eF_exact=True,native_masks_scores_and_classes_exact=True,fixed_native_scores_and_class_and_order_exact=True,
            external_region_gate=True,exact_GPU_L_selector=True,K_natural_query_order=Ks,fallback_natural_query_order=fallbacks,
            region_digests=region_digests,selector=selector_row,controls=diagnostics,native_seconds=native_time,
            selector_IO_seconds=selector_io,region_seconds=region_time,feature_seconds=feature_time,
            elapsed_seconds=time.monotonic()-begun,GT_used_to_choose_queries_or_residual=False)
    F=e=M=C=L=top=order=ordered=binary=quality=scores=F_original=refined=None
    return baseline,predictions,summary,selector_row


def run(spec_path):
    started=time.monotonic(); path=Path(spec_path); require(path.is_absolute(),'Absolute ROOT SPEC required')
    path=path.resolve(strict=True); spec=read(path); gate=source_gate(path,spec)
    output=Path(spec['output_directory']); require(output.is_absolute() and not output.exists(),'Fresh output directory required')
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1] and psutil.virtual_memory().available>=16*1024**3,'Actual CPU01/RAM16GiB required')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(),'Qualified cold CUDA Torch required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    heads,states,head_proofs=load_heads(torch,spec,gate)
    import numpy as np
    import cv2
    import scipy
    from scipy import ndimage
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks
    require(np.__version__=='1.23.5' and cv2.__version__=='4.8.0' and scipy.__version__=='1.10.1','Qualified numeric versions required')
    n10=load(spec['N10_evaluation_source'],'_N13_bound_original_N10_evaluation_helpers')
    reader=load(spec['reader_source'],'_N13_original_pure_CPU_reader')
    renderer=load(spec['renderer_source'],'_N13_original_renderer')
    official=load(spec['COCO_source'],'_N13_original_COCO')
    kernels=n10.original_export_kernels(spec['export_source'])
    coco=COCO(str(bound(spec['DEV']))); ids=list(coco.imgs)
    require(len(ids)==len(set(ids))==187 and ids==[r['id'] for r in coco.dataset['images']] and len(coco.anns)==380 and set(coco.cats)=={1},
        'Complete canonical DEV187/380/category1 required')
    native=read(bound(spec['native_reference_predictions'])); population=reader.population(coco,native,ids)
    rows=read(bound(gate['prior']['image_rows']))
    trace=[json.loads(line) for line in bound(gate['prior']['trace']).read_text(encoding='utf-8').splitlines() if line.strip()]
    require(rows==trace and len(rows)==187 and [r['image_id'] for r in rows]==ids and sum(r['raw_GT'] for r in rows)==380,
        'Original complete canonical image rows differ')
    pixels=sum(coco.imgs[i]['height']*coco.imgs[i]['width'] for i in ids); selector_bytes=400*pixels
    require(spec['selector_uncompressed_tensor_bytes']==selector_bytes and
        type(spec['minimum_free_output_bytes']) is int and spec['minimum_free_output_bytes']>=
            (selector_bytes+2*1024**3 if spec['selector_mode']=='produce' else 2*1024**3) and
        psutil.disk_usage(str(output.parent)).free>=spec['minimum_free_output_bytes'],
        'Actual original-raster selector bytes/free storage must match ROOT prelocked metadata')
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,gate['closure'],gate['admission'],ndimage)
    require(not torch.cuda.is_initialized(),'Complete metadata/checkpoint/library gates must precede CUDA')
    torch.cuda.init(); torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0); outside=[torch.get_rng_state().clone(),torch.cuda.get_rng_state(0).clone()]
    for head in heads.values(): head.to('cuda:0')
    output.mkdir(); selector_dir=output/'SELECTORS'
    if spec['selector_mode']=='produce': selector_dir.mkdir()
    write(output/'START.json',dict(schema='N13-sameK-fixed-fullDEV-GPU-start-v1',source=gate['source'],spec=gate['spec_ref'],
        GPU_admission=gate['admission_ref'],phase=spec['phase'],recipe=spec['recipe'],endpoint=spec['endpoint'],
        seed=spec['seed'],head_endpoint_epoch=spec['head_endpoint_epoch'],head_state_proofs=head_proofs,
        selector_mode=spec['selector_mode'],selector_parent=spec['selector_parent'],recovered_endpoint_use=spec['recovered_endpoint_use'],
        native_reference_predictions=spec['native_reference_predictions'],archived_packet_refs=gate['packets'],
        native_model_loaded=False,optimizer_created=False,no_TEST=True,best_DEV_selection=False))
    predictions={control:[] for control in spec['controls']}; diagnostics=[]; selector_rows=[]
    with (output/'TRACE.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        for index,(row,ref) in enumerate(zip(rows,gate['packets'])):
            deadline(started,spec); packet=torch.load(str(bound(ref)),map_location='cpu')
            base,refined,diagnostic,selector=evaluate_image(torch,np,ndimage,masks,renderer,kernels,reader,n10,coco,packet,row,index,ref,
                population[row['image_id']],heads,gate,spec,started,selector_dir)
            predictions['native_official']+=base
            for key,values in refined.items(): predictions[key]+=values
            diagnostics.append(diagnostic); selector_rows.append(selector); stream.write(json.dumps(diagnostic,allow_nan=False)+'\n'); stream.flush()
            del packet,base,refined,diagnostic,selector
            print(json.dumps(dict(N13_evaluated_DEV_images=index+1,exact_native_queries=100*(index+1))),flush=True)
    require(predictions['native_official']==native and all(len(v)==18700 for v in predictions.values()),'Complete exact18700 population differs')
    for values in predictions.values():
        reader.population(coco,values,ids)
        require(all({k:v for k,v in a.items() if k!='segmentation'}=={k:v for k,v in b.items() if k!='segmentation'}
            for a,b in zip(native,values)),'All original query/class/order/S0 differs')
    require(all(torch.equal(v.cpu(),states[arm][name]) for arm,head in heads.items() for name,v in head.state_dict().items()) and
        all(not p.requires_grad and p.grad is None for head in heads.values() for p in head.parameters()) and
        torch.equal(outside[0],torch.get_rng_state()) and torch.equal(outside[1],torch.cuda.get_rng_state(0)),
        'Read-only state or outside RNG changed')
    peak=torch.cuda.max_memory_allocated(0)
    for head in heads.values(): head.cpu()
    heads.clear(); del head,heads; gc.collect(); torch.cuda.synchronize(0)
    torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache(); torch.cuda.synchronize(0)
    cleanup=dict(allocated_after=torch.cuda.memory_allocated(0),reserved_after=torch.cuda.memory_reserved(0))
    require(cleanup['allocated_after']==0 and peak<=4096*1024**2,'Allocator cleanup/cap required')
    if spec['selector_mode']=='produce':
        manifest=dict(schema=gate['lineage'].SELECTOR_SCHEMA,source=gate['source'],spec=gate['spec_ref'],
            GPU_admission=gate['admission_ref'],phase=spec['phase'],renderer_source=spec['renderer_source'],
            prior_N07_report=spec['prior_N07_report'],native_reference_predictions=spec['native_reference_predictions'],
            images=187,queries=18700,uncompressed_tensor_bytes=selector_bytes,rows=selector_rows,
            shared_frozen_native_selector=True,current_owned_terminal_self_certified=False)
        write(output/'SELECTOR_MANIFEST.json',manifest); selector_manifest=reference(output/'SELECTOR_MANIFEST.json')
    else:
        require(selector_rows==gate['selector']['manifest']['rows'],'Reused full187 selector manifest differs')
        selector_manifest=gate['selector']['manifest_ref']
    refs,results={},{}
    for control,values in predictions.items():
        target=output/(control.upper()+'_PREDICTIONS.json'); write(target,values); refs[control]=reference(target)
        deadline(started,spec); metrics=official.coco_metrics(coco,values,ids,COCOeval)
        require(metrics['iouType']=='segm' and metrics['maxDets']==[1,10,100] and metrics['score_filter'] is None and
            metrics['original_GT_preserved'] is True and metrics['official_COCO_crowd_ignore_semantics'] is True,
            'Original official COCO convention differs')
        n10.stats_exact(metrics['stats'],metrics['stats'],'Finite defined COCO12/-1 required')
        results[control]=dict(predictions=refs[control],COCO=metrics,stats=metrics['stats'])
    n10.stats_exact(results['native_official']['stats'],gate['prior_q']['expected_E24_COCO_stats'],'Strong E24 COCO12 differs')
    n10.stats_exact(results['native_official']['stats'],gate['prior']['control_results']['native_official']['stats'],'Original native COCO12 differs')
    write(output/'IMAGE_DIAGNOSTICS.json',diagnostics)
    for ref in spec['sources']: deadline(started,spec); bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and
        reference(spec['GPU_admission_file'])==gate['admission_ref'] and
        gate['common'].imported_files(torch,np,cv2,masks,COCO,gate['closure'],gate['admission'],ndimage)==libraries and
        psutil.Process().cpu_affinity()==[0,1] and torch.get_num_threads()==torch.get_num_interop_threads()==2,
        'Final closure/resources changed')
    gate['lineage'].final_endpoint_check(gate['parent']); deadline(started,spec)
    result=dict(schema=REPORT_SCHEMA,status=STATUS,phase=spec['phase'],source=gate['source'],spec=gate['spec_ref'],
        GPU_admission=gate['admission_ref'],normal_RAW_count=spec['normal_RAW_count'],RAW_inputs=spec['RAW_inputs'],
        prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'],
        terminal_identity_pins=spec['terminal_identity_pins'],recipe=spec['recipe'],numeric_protocol=spec['numeric_protocol'],
        seed=spec['seed'],head_endpoint_epoch=spec['head_endpoint_epoch'],endpoint=spec['endpoint'],head_state_proofs=head_proofs,
        recovered_endpoint_use=spec['recovered_endpoint_use'],selector_mode=spec['selector_mode'],selector_parent=spec['selector_parent'],
        selector_manifest=selector_manifest,start=reference(output/'START.json'),trace=reference(output/'TRACE.jsonl'),
        image_diagnostics=reference(output/'IMAGE_DIAGNOSTICS.json'),predictions=refs,control_results=results,
        images=187,raw_GT=380,predictions_per_control=18700,controls=spec['controls'],packet_refs=gate['packets'],
        teacher_checkpoint=gate['cache_report']['teacher_checkpoint'],prior_N07_report=spec['prior_N07_report'],
        native_reference_predictions=spec['native_reference_predictions'],DEV=spec['DEV'],
        exact_native_export_and_COCO=True,exact_shared_GPU_L_selector=True,external_region_gate=True,
        native_model_loaded=False,native_forward_calls=0,original_JPEG_decoded=False,optimizer_created=False,weights_updated=False,
        no_GT_query_selection=True,no_NMS=True,no_score_cutoff=True,no_TEST=True,best_DEV_selection=False,
        current_owned_GPU_terminal_self_certified=False,independent_CPU_evaluation_replayed=False,
        task_efficacy_established=False,innovation_established=False,convergence_established=False,generalization_established=False,
        torch_peak_allocated_bytes=peak,cleanup=cleanup,actual_imported_files=libraries,elapsed_seconds=time.monotonic()-started,
        costs=dict(shared_native_seconds=sum(r['native_seconds'] for r in diagnostics),selector_IO_seconds=sum(r['selector_IO_seconds'] for r in diagnostics),
            shared_region_seconds=sum(r['region_seconds'] for r in diagnostics),shared_feature_seconds=sum(r['feature_seconds'] for r in diagnostics),
            arm_head_seconds={arm:sum(r['controls'][spec['controls'][i+1]]['head_seconds'] for r in diagnostics) for i,arm in enumerate(ARMS)}),
        evidence_limits=['Full historical DEV, fixed native scores, complete query population; no blind TEST or external SOTA claim.',
            'Support and training-point distribution changed jointly; equal K/capacity does not prove equal sorting cost or mechanism causality.',
            'Recovered original seed701 GPU remains exit1; separate actual CPU0/ROOT adoption and new explicit fixedE08 curve authorization are retained.',
            '8/24 are curves only; three qualified paired48 results, mechanisms and cross-dataset evidence are still required.',
            'The current report cannot certify its own actual held terminal or independent CPU result.'])
    write(output/'REPORT.json',result)
    print(json.dumps(dict(status=STATUS,report=reference(output/'REPORT.json')),allow_nan=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true'); args=parser.parse_args()
    require(args.owner_execute,'Explicit single ROOT-owned execution required'); run(args.spec)
