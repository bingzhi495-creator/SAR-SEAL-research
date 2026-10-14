"""UNBOUND real fixed-PR diagnostic draft; no run or efficacy claim.
Privileged fixed374+6 cohort, original701 LOG E48; ROOT owns admission.
"""
import argparse
import ast
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


SCHEMA='ROOT-N13-fixed-PR-real-diagnostic-GPU-spec-v1'
REPORT_SCHEMA='N13-fixed-PR-real-diagnostic-GPU-report-v1'
STATUS='N13_FIXED_PR_REAL_DIAGNOSTIC_COMPLETE_PENDING_INDEPENDENT_CPU'
LOCKED={'matched_count':374,'unmatched_gt_count':6,'total_gt_count':380}
DISTANCE_NAMES=('D_le_1','D_gt_1_le_2','D_gt_2_le_4','D_gt_4_or_infinite')

def write(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def deadline(started,spec):
    require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked diagnostic deadline exceeded; no automatic retry')

def source_gate(path,spec):
    require(spec['schema']==SCHEMA and spec['diagnostic_protocol']['carrier_seed']==202610120701 and
        spec['diagnostic_protocol']['carrier_epoch']==48 and spec['diagnostic_protocol']['arm']=='LOG_PR' and
        spec['diagnostic_protocol']['population']==LOCKED and spec['diagnostic_protocol']['distance_names']==list(DISTANCE_NAMES),
        'Exact preregistered first701 LOG E48 fixed diagnostic required')
    raws,closure=pre_gate(spec,93)
    require(spec['normal_RAW_count']==93 and len(spec['terminal_identity_pins'])==101,'Complete final curve ancestry required')
    original=read(bound(spec['carrier_GPU_spec']))
    cpu=read(bound(spec['carrier_CPU_spec']));q=read(bound(spec['carrier_CPU_qualification']))
    require(original['seed']==cpu['seed']==q['seed']==202610120701 and
        original['head_endpoint_epoch']==cpu['head_endpoint_epoch']==q['head_endpoint_epoch']==48 and
        q['status']=='ROOT_N13_SAMEK_FIXED_FULLDEV_CPU_QUALIFIED_V1' and
        q['evaluation_spec']==spec['carrier_GPU_spec'] and q['spec']==spec['carrier_CPU_spec'] and
        q['source']==spec['carrier_evaluation_source'] and q['reader_source']==spec['carrier_CPU_source'] and
        q['evaluation_GPU_RAW']==spec['carrier_GPU_RAW'] and q['CPU_admission']==spec['carrier_CPU_admission'] and
        q['original_native_374pair_6FN_cohort_preserved'] is True and
        q['exact_GPU_L_regions_all18700_CPU_checked'] is True and q['independent_two_PR_formula_all_selected_pixels_CPU_checked'] is True,
        'Previously qualified carrier provenance differs')
    require(all(ref in closure.values() for payload in (original,cpu) for ref in payload['sources']),
        'Complete original carrier sources required')
    context=dict(original)
    for key in ('sources','RAW_inputs','normal_RAW_count','prior_failed_CPU_receipt','additional_failed_RAW_inputs',
                'terminal_identity_pins','phase','entry_source','GPU_admission_file','output_directory'):
        context[key]=spec[key]
    lineage=load(original['endpoint_lineage_source'],'_N13_real_diagnostic_carrier_lineage')
    lineage.check_recipe(original)
    gate=lineage.own_gate(path,context,__file__,closure,raws)
    gi=lineage.owner_ref_join(context,gate['owners'],spec['carrier_GPU_RAW'],q['evaluation_GPU_admission'],q['source'],q['evaluation_spec'],'0')
    ci=lineage.owner_ref_join(context,gate['owners'],spec['carrier_CPU_RAW'],q['CPU_admission'],q['reader_source'],q['spec'],'')
    require(gi<ci and original['RAW_inputs']==spec['RAW_inputs'][:gi] and cpu['RAW_inputs']==spec['RAW_inputs'][:ci],
        'Actual carrier GPU0/CPU0 chronological joins differ')
    gate.update(lineage=lineage,context=context,carrier_Q=q,carrier_CPU=cpu)
    gate.update(lineage.endpoint_gate(context,closure,gate['owners']))
    prior,prior_q=lineage.prior_gate(context,closure,gate['owners'])
    require(prior['teacher_checkpoint']==gate['cache_report']['teacher_checkpoint'],'Exact shared teacher required')
    gate.update(prior=prior,prior_q=prior_q,report=prior,packets=prior_q['packet_refs'],
                selector=lineage.selector_parent_gate(context,closure,gate['owners']))
    fixed=read(bound(spec['fixed_population']))
    details=read(bound(q['details']));key='LOG_PR_seed202610120701_E48_fixed_native_score'
    selected=[row for row in details['fixed_native_query_GT_pairs'][key] if abs(row['threshold']-.5)<1e-12]
    require(len(selected)==1 and fixed['rows']==selected[0]['GT_rows'] and len(fixed['rows'])==380 and
        sum(row['status']=='native_matched_fixed_query' for row in fixed['rows'])==374 and
        sum(row['status']=='native_FN' for row in fixed['rows'])==6 and
        all(row['status'] in ('native_matched_fixed_query','native_FN') for row in fixed['rows']),
        'Exact existing native .50 identities must be preserved, including six unmatched')
    require(len({(r['image_id'],r['gt_id']) for r in fixed['rows']})==380 and
        len({(r['image_id'],r['query_id']) for r in fixed['rows'] if r['query_id'] is not None})==374,
        'Unique actual fixed pairing required')
    gate.update(fixed=fixed,evaluation=load(spec['carrier_evaluation_source'],'_N13_real_original_evaluation'),
                modules={k:load(v,'_N13_real_'+k) for k,v in spec['diagnostic_sources'].items() if k!='author'})
    return gate

def native_distance_bands(np,ndimage,P):
    # In-domain opposite-label chessboard distance; empty/full is infinite.
    if not P.any() or P.all():
        D=np.full(P.shape,np.inf,dtype=np.float64)
    else:
        D=np.where(P,ndimage.distance_transform_cdt(P,metric='chessboard'),
                       ndimage.distance_transform_cdt(~P,metric='chessboard'))
        require(bool((D>=1).all()),'Positive exact native opposite distances required')
    masks={DISTANCE_NAMES[0]:D<=1,DISTANCE_NAMES[1]:(D>1)&(D<=2),
           DISTANCE_NAMES[2]:(D>2)&(D<=4),DISTANCE_NAMES[3]:D>4}
    return {name:dict(mask=mask,source='native_in_domain_opposite_chessboard_D;empty_full_infinite;fixed_bands1_2_4')
            for name,mask in masks.items()}

def author_boundary(torch,ref):
    source=bound(ref);tree=ast.parse(source.read_text('utf8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate_block_target']
    require(len(nodes)==1 and not nodes[0].decorator_list,'Unique unchanged author boundary definition required')
    namespace=dict(torch=torch,F=torch.nn.functional)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    return namespace['generate_block_target']

def integer_width2(torch,np,ndimage,P):
    counts=ndimage.convolve(P.astype(np.int64),np.ones((5,5),dtype=np.int64),mode='constant',cval=0)
    return (P & (counts<=22)) | (~P & (counts>=3))

def pixels(np,A,G):
    return dict(pixel_TP=int(np.count_nonzero(A&G)),pixel_FP=int(np.count_nonzero(A&~G)),
                pixel_FN=int(np.count_nonzero(~A&G)))

def run(spec_path):
    started=time.monotonic();path=Path(spec_path).resolve(strict=True);spec=read(path);gate=source_gate(path,spec)
    output=Path(spec['output_directory']);require(output.is_absolute() and not output.exists(),'Fresh diagnostic output required')
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1] and psutil.virtual_memory().available>=16*1024**3,'Actual CPU01/RAM16GiB required')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(),'Qualified cold Torch runtime required')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    import numpy as np
    import cv2
    import scipy
    from scipy import ndimage
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    require(np.__version__=='1.23.5' and cv2.__version__=='4.8.0' and scipy.__version__=='1.10.1','Qualified numerical versions required')
    context=gate['context'];evaluation=gate['evaluation'];modules=gate['modules']
    heads,states,proofs=evaluation.load_heads(torch,context,gate)
    n10=load(context['N10_evaluation_source'],'_N13_real_N10')
    reader=load(context['reader_source'],'_N13_real_native_reader');renderer=load(context['renderer_source'],'_N13_real_renderer')
    kernels=n10.original_export_kernels(context['export_source'])
    boundary=author_boundary(torch,spec['diagnostic_sources']['author'])
    coco=COCO(str(bound(context['DEV'])));ids=list(coco.imgs)
    require(len(ids)==187 and len(coco.anns)==380 and set(coco.cats)=={1},'Actual DEV187/380 required')
    native=read(bound(context['native_reference_predictions']));native_pop=reader.population(coco,native,ids)
    original_report=read(bound(gate['carrier_Q']['evaluation_report']))
    logref=original_report['predictions']['LOG_PR_seed202610120701_E48_fixed_native_score']
    logrows=read(bound(logref));logpop=reader.population(coco,logrows,ids)
    require(len(native)==len(logrows)==18700 and all({k:v for k,v in a.items() if k!='segmentation'}==
        {k:v for k,v in b.items() if k!='segmentation'} for a,b in zip(native,logrows)),
        'Original full query/class/score/order/empty score population differs')
    rows=read(bound(gate['prior']['image_rows']));manifest=gate['selector']['manifest']
    require(len(rows)==len(manifest['rows'])==len(gate['packets'])==187 and [r['image_id'] for r in rows]==ids,
        'Full canonical packets and selectors required')
    require(psutil.disk_usage(str(output.parent)).free>=spec['minimum_free_output_bytes'],'Prelocked output free space required')
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,gate['closure'],gate['admission'],ndimage)
    require(not torch.cuda.is_initialized(),'All qualified gates precede CUDA')
    torch.cuda.init();torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0);outside=[torch.get_rng_state().clone(),torch.cuda.get_rng_state(0).clone()]
    gpu_weights={k:v.to('cuda:0') for k,v in states['LOG_PR'].items()}
    output.mkdir();(output/'CANDIDATES').mkdir()
    write(output/'START.json',dict(schema='N13-fixed-PR-real-diagnostic-start-v1',source=gate['source'],spec=gate['spec_ref'],
        GPU_admission=gate['admission_ref'],phase=spec['phase'],carrier_CPU_qualification=spec['carrier_CPU_qualification'],
        diagnostic_protocol=spec['diagnostic_protocol'],fixed_population=spec['fixed_population'],head_state_proofs=proofs,
        privileged_GT_diagnostic=True,training_started=False,no_TEST=True,AP_evaluated=False))
    paired_by_image={iid:[r for r in gate['fixed']['rows'] if r['image_id']==iid and r['query_id'] is not None] for iid in ids}
    unmatched=[dict(image_id=r['image_id'],gt_id=r['gt_id']) for r in gate['fixed']['rows'] if r['query_id'] is None]
    diagnostics=[];candidate_refs=[];image_checks=[];candidate_points=0;old_calls=halo_calls=0
    timings={k:0. for k in ('native','selector_IO','feature','domains_distance','candidate_reconstruct_statistics','serialization')}
    with torch.no_grad(),(output/'TRACE.jsonl').open('x',encoding='utf8',newline='\n') as trace:
        for index,(row,ref,selector_row) in enumerate(zip(rows,gate['packets'],manifest['rows'])):
            deadline(started,spec);tick=time.monotonic();packet=torch.load(str(bound(ref)),map_location='cpu')
            fields,identity,qids=n10.packet_check(torch,reader,coco,packet,row,index,ref,gate)
            F=fields['F256'].to('cuda:0');e=fields['e0'].to('cuda:0');M=fields['M200'].to('cuda:0');cls=fields['final_class_logits'].to('cuda:0')
            require(torch.equal(torch.einsum('bqc,bchw->bqhw',e,F),M),'Actual original M=eF differs')
            L=renderer.render_original(torch,M,identity);top,order=kernels.order(torch,cls,types.SimpleNamespace(num_classes=1))
            require(order.cpu().tolist()==qids,'Original natural100 export order differs')
            binary,quality,scores=kernels.quality(torch,L[order],top)
            for ordinal,qid in enumerate(qids):
                require(dict(image_id=row['image_id'],query_id=qid,category_id=1,score=float(scores[ordinal].item()),
                    segmentation=n10.segmentation(np,masks,binary[ordinal].cpu().numpy().astype(np.uint8)))==native_pop[row['image_id']][ordinal],
                    'Exact original all100 mask/class/order/score reproduction failed')
            torch.cuda.synchronize(0);timings['native']+=time.monotonic()-tick
            tick=time.monotonic();carrier=torch.load(str(bound(selector_row['carrier'])),map_location='cpu')
            expected=dict(dataset_index=index,image_id=row['image_id'],identity=identity,native_packet=ref,
                natural_query_ids=list(range(100)),native_export_query_ids=qids,renderer_source=context['renderer_source'])
            require(carrier['schema']==gate['lineage'].SELECTOR_PACKET_SCHEMA and all(carrier[k]==selector_row[k]==v for k,v in expected.items()) and
                carrier['source']==gate['selector']['report']['source'] and carrier['spec']==gate['selector']['report']['spec'] and
                carrier['GPU_admission']==gate['selector']['report']['GPU_admission'] and
                evaluation.digest(carrier['native_L'])==carrier['L_digest']==selector_row['L_digest'] and
                torch.equal(L.detach().cpu(),carrier['native_L']),'Exact frozen GPU L selector reproduction failed')
            timings['selector_IO']+=time.monotonic()-tick
            tick=time.monotonic();F_original=n10.original_feature(torch,renderer,F,identity);torch.cuda.synchronize(0)
            timings['feature']+=time.monotonic()-tick
            M=cls=top=order=binary=quality=scores=F=e=None
            image_records=[];loglookup={r['query_id']:r for r in logpop[row['image_id']]}
            for pair in paired_by_image[row['image_id']]:
                deadline(started,spec);qid=pair['query_id'];z=L[qid];tick=time.monotonic()
                domains=modules['domains'].assemble_fixed_pr_domains(torch,boundary,z)
                P=domains['native_P'].cpu().numpy();width=domains['native_width2'].cpu().numpy()
                require(np.array_equal(width,integer_width2(torch,np,ndimage,P)),'Actual author width2 differs from independent integer/25 definition')
                bands=native_distance_bands(np,ndimage,P)
                require(int(domains['old_LOG'].sum().item())==row['band_counts_all100'][qid],'Original natural-query K differs')
                G=masks.decode(reader.rle_bytes(coco.annToRLE(coco.anns[pair['gt_id']]))).astype(bool)
                require(pixels(np,P,G)=={k:pair['native'][k] for k in ('pixel_TP','pixel_FP','pixel_FN')},'Actual fixed native pair pixel identity differs')
                original_LOG=torch.from_numpy(masks.decode(reader.rle_bytes(loglookup[qid]['segmentation'])).astype(bool)).to('cuda:0')
                timings['domains_distance']+=time.monotonic()-tick;tick=time.monotonic()
                metadata=dict(carrier_seed=202610120701,carrier_epoch=48,carrier_arm='LOG_PR',
                    comparison='offline_privileged_fixed374_not_AP',distance_definition='native_in_domain_opposite_chessboard_D;empty_full_infinite;bands1_2_4',
                    source=gate['source'],spec=gate['spec_ref'],native_packet=ref,selector=selector_row['carrier'],
                    original_LOG_predictions=logref,actual_pair_source=spec['fixed_population'])
                result=modules['pipeline'].diagnose_fixed_pr_pair(torch,modules['adapter'],modules['statistics'],modules['oracle'],
                    weight_tensors=tuple(gpu_weights[k] for k in evaluation.NAMES),F_original=F_original,native_L=z,old_LOG=domains['old_LOG'],new_halo=domains['new_halo'],
                    G=G,image_id=pair['image_id'],gt_id=pair['gt_id'],query_id=qid,expected_original_LOG_binary=original_LOG,
                    metadata=metadata,distance_bands=bands)
                require(pixels(np,result['original_LOG_binary'].cpu().numpy(),G)==
                    {k:pair['refined'][k] for k in ('pixel_TP','pixel_FP','pixel_FN')},'Actual original LOG pair pixel identity differs')
                torch.cuda.synchronize(0);timings['candidate_reconstruct_statistics']+=time.monotonic()-tick
                candidates=result['candidate_partitions'];candidate_points+=domains['logical_count']['Omega']
                old_calls+=(domains['logical_count']['old_LOG']+2047)//2048;halo_calls+=(domains['logical_count']['new_halo']+2047)//2048
                tick=time.monotonic();target=output/'CANDIDATES'/('IMAGE%d_GT%d_QUERY%d.pth'%(pair['image_id'],pair['gt_id'],qid))
                payload=dict(schema='N13-fixed-PR-real-candidate-v1',source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],
                    phase=spec['phase'],identity={k:pair[k] for k in ('image_id','gt_id','query_id')},native_packet=ref,
                    selector=selector_row['carrier'],head_checkpoint=context['endpoint']['checkpoints']['LOG_PR'],
                    domains={k:v.detach().cpu().contiguous() for k,v in domains.items() if isinstance(v,torch.Tensor)},
                    candidates=candidates,diagnostic=result['diagnostic'])
                # Store CPU-only tensors; never serialize a live CUDA storage.
                def cpu_tree(value):
                    if isinstance(value,torch.Tensor):return value.detach().cpu().contiguous()
                    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
                    if isinstance(value,list):return [cpu_tree(v) for v in value]
                    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
                    return value
                with target.open('xb') as stream:torch.save(cpu_tree(payload),stream)
                replay=torch.load(str(target),map_location='cpu')
                require(replay['identity']==payload['identity'] and replay['diagnostic']==result['diagnostic'] and
                    torch.equal(replay['candidates']['merged_C'],result['candidate'].cpu()),'Actual candidate packet saved readback differs')
                cr=reference(target);candidate_refs.append(cr);diagnostics.append(result['diagnostic'])
                record=dict(identity=payload['identity'],candidate=cr,domain_counts=domains['logical_count'],reconstruction=result['reconstruction_check'])
                image_records.append(record);timings['serialization']+=time.monotonic()-tick
                del G,original_LOG,domains,result,candidates,payload,replay,bands,P,width,z
            image_check=dict(dataset_index=index,image_id=row['image_id'],native_all100_exact=True,selector_exact=True,pairs=image_records)
            image_checks.append(image_check);trace.write(json.dumps(image_check,allow_nan=False)+'\n');trace.flush()
            del packet,fields,carrier,L,F_original,loglookup,image_records
            print(json.dumps(dict(real_diagnostic_images=index+1,matched_pairs=len(diagnostics))),flush=True)
    require(len(diagnostics)==len(candidate_refs)==374 and len(unmatched)==6,'Complete preregistered fixed population required')
    population=modules['statistics'].diagnose_population(modules['oracle'],matched_diagnostics=diagnostics,
        unmatched_gt=unmatched,metadata=dict(protocol=spec['diagnostic_protocol'],fixed_population=spec['fixed_population']),locked_population=LOCKED)
    require(all(torch.equal(states['LOG_PR'][k],v.cpu()) for k,v in gpu_weights.items()) and
        all(torch.equal(v,head.state_dict()[k]) for arm,head in heads.items() for k,v in states[arm].items()) and
        torch.equal(outside[0],torch.get_rng_state()) and torch.equal(outside[1],torch.cuda.get_rng_state(0)),
        'Frozen head/RNG changed during diagnostic')
    peak=torch.cuda.max_memory_allocated(0);gpu_weights.clear();heads.clear();del gpu_weights,heads
    gc.collect();torch.cuda.synchronize(0);torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();torch.cuda.synchronize(0)
    cleanup=dict(allocated_after=torch.cuda.memory_allocated(0),reserved_after=torch.cuda.memory_reserved(0))
    require(cleanup['allocated_after']==0 and peak<=4096*1024**2,'Actual diagnostic allocator cap/cleanup required')
    write(output/'POPULATION.json',population);write(output/'PAIR_DIAGNOSTICS.json',diagnostics);write(output/'IMAGE_CHECKS.json',image_checks)
    report=dict(schema=REPORT_SCHEMA,status=STATUS,source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],phase=spec['phase'],
        RAW_inputs=spec['RAW_inputs'],terminal_identity_pins=spec['terminal_identity_pins'],normal_RAW_count=93,
        carrier_CPU_qualification=spec['carrier_CPU_qualification'],fixed_population=spec['fixed_population'],
        diagnostic_protocol=spec['diagnostic_protocol'],images=187,matched_pairs=374,unmatched_GT=6,raw_GT=380,
        original_all18700_native_export_exact=True,all374_original_LOG_full_binary_reconstructed=True,
        author_width2_all374_independent_integer_checked=True,candidate_generated_without_G_or_S_or_K=True,
        population=reference(output/'POPULATION.json'),pair_diagnostics=reference(output/'PAIR_DIAGNOSTICS.json'),
        image_checks=reference(output/'IMAGE_CHECKS.json'),trace=reference(output/'TRACE.jsonl'),start=reference(output/'START.json'),
        candidate_packets=candidate_refs,head_state_proofs=proofs,logical_candidate_points=candidate_points,
        actual_PR_chunk_calls=dict(old_LOG=old_calls,new_halo=halo_calls),cost_seconds=timings,
        total_elapsed_seconds=time.monotonic()-started,peak_allocator_bytes=peak,cleanup=cleanup,actual_imported_files=libraries,
        no_TEST=True,AP_evaluated=False,deployment_gain_established=False,innovation_established=False,
        independent_CPU_qualified=False,current_owned_GPU_terminal_self_certified=False,whole_leaf_VRAM_certified=False,
        evidence_limits='First701 LOG E48, fixed374 native .50 pairs plus six unmatched; privileged local IoU diagnostics, not AP, causality, deployment or cross-data.')
    write(output/'REPORT.json',report)
    print(json.dumps(dict(status=STATUS,report=reference(output/'REPORT.json'),matched_pairs=374,independent_CPU_qualified=False)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    options=parser.parse_args();require(options.owner_execute,'ROOT-owned single scientific attempt required');run(options.spec)


