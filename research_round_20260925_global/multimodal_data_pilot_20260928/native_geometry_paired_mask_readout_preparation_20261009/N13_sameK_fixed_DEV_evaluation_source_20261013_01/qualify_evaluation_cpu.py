"""UNBOUND N13 independent CPU full187 paired fixed-epoch evaluation; ROOT owns admission.
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
    require(type(extra) is list and len(extra)==6 and all(type(row) is dict and set(row)=={'receipt','phase','returncode'} and
        type(row['returncode']) is int and row['returncode']==1 for row in extra),'Preserve exact three-key extra failed6 rows, seven failures total')
    failures=[failed]+[terminal(row['receipt'],row['phase'],1) for row in extra]
    failed_paths=[spec['prior_failed_CPU_receipt']['path']]+[row['receipt']['path'] for row in extra]
    require(len(set(failed_paths))==7 and not {row['receipt']['path'] for row in rows}.intersection(failed_paths),
        'Normal ancestors and seven preserved failures must be distinct')
    all_rows=[dict(row,returncode=0) for row in rows]+[dict(receipt=spec['prior_failed_CPU_receipt'],phase=Path(spec['prior_failed_CPU_receipt']['path']).parent.name,returncode=1)]+extra
    pins=spec['terminal_identity_pins']; locators=[row['receipt']['path'] for row in all_rows]
    require(type(pins) is dict and set(pins)==set(locators),'Exact all-normal-plus-seven-failure identity pins required')
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

def fixed_native_pairs(np,masks,reader,coco,native_matching,populations):
    """Keep each native matched query fixed; preserve FN/ignored population."""
    summaries,details = {},{}
    for control in CONTROLS[1:]:
        control_summaries,control_details = [],[]
        for threshold in (.50,.75,.90):
            records = []
            for image in native_matching:
                cells = [cell for cell in image['thresholds'] if abs(cell['threshold']-threshold) <= 1e-12]
                require(len(cells) == 1,'Actual native matching threshold required')
                lookup = {row['query_id']:row for row in populations[control][image['image_id']]}
                for row in cells[0]['GT']:
                    pair = image['TP_pairs'][row['TP_pair_key']] if row['TP_pair_key'] else None
                    if pair is None:
                        records.append(dict(image_id=image['image_id'],gt_id=row['gt_id'],
                            status='ignored_GT' if row['ignored'] else 'native_FN',query_id=None,
                            native=None,refined=None,refined_minus_native=None))
                        continue
                    gt = coco.anns[row['gt_id']]
                    a = masks.decode(reader.rle_bytes(coco.annToRLE(gt))).astype(bool)
                    b = masks.decode(reader.rle_bytes(lookup[pair['query_id']]['segmentation'])).astype(bool)
                    require(a.shape == b.shape,'Fixed native query/GT raster shape differs')
                    tp,fp,fn = int(np.count_nonzero(a & b)),int(np.count_nonzero(~a & b)),int(np.count_nonzero(a & ~b))
                    union,total = tp+fp+fn,2*tp+fp+fn
                    refined = dict(IoU=tp/union if union else 0.,Dice=2*tp/total if total else 0.,
                        pixel_TP=tp,pixel_FP=fp,pixel_FN=fn)
                    native = {name:pair[name] for name in refined}
                    records.append(dict(image_id=image['image_id'],gt_id=row['gt_id'],status='native_matched_fixed_query',
                        query_id=pair['query_id'],native=native,refined=refined,
                        refined_minus_native={name:refined[name]-native[name] for name in refined}))
            require(len(records) == 380,'Fixed native pairing must retain all380 GT, including FN/ignore')
            selected = [row for row in records if row['refined_minus_native'] is not None]
            deltas = [row['refined_minus_native'] for row in selected]
            control_summaries.append(dict(threshold=threshold,GT=380,
                native_matched_pairs=len(selected),native_FN=sum(row['status'] == 'native_FN' for row in records),
                ignored_GT=sum(row['status'] == 'ignored_GT' for row in records),
                conditional_mean_IoU_delta=math.fsum(row['IoU'] for row in deltas)/len(deltas) if deltas else None,
                conditional_mean_Dice_delta=math.fsum(row['Dice'] for row in deltas)/len(deltas) if deltas else None,
                conditional_pixel_deltas={name:sum(row[name] for row in deltas) for name in ('pixel_TP','pixel_FP','pixel_FN')},
                fixed_native_query_and_assignment=True,not_global_pixel_risk_or_causal_decomposition=True))
            control_details.append(dict(threshold=threshold,GT_rows=records))
        summaries[control],details[control] = control_summaries,control_details
    return summaries,details


SCHEMA='ROOT-N13-sameK-fixed-fullDEV-CPU-spec-v1'
Q_SCHEMA='ROOT-N13-sameK-fixed-fullDEV-CPU-qualification-v1'
STATUS='ROOT_N13_SAMEK_FIXED_FULLDEV_CPU_QUALIFIED_V1'
ARMS=('LOG_PR','GEO_PR')


def deadline(started,spec):
    require(time.monotonic()-started<spec['deadline_seconds'],'Locked independent full187 CPU deadline exceeded; no retry')


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def digest(value):
    a=value.detach().cpu().contiguous().numpy()
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def source_gate(path,spec):
    require(spec['schema']==SCHEMA,'Exact independent full187 paired CPU schema required')
    raws,closure=pre_gate(spec,42)
    lineage=load(spec['endpoint_lineage_source'],'_N13_CPU_actual_endpoint_lineage')
    lineage.check_recipe(spec)
    gate=lineage.own_gate(path,spec,__file__,closure,raws,cpu=True)
    extra=('evaluation_entry_source','evaluation_spec','evaluation_report','evaluation_GPU_RAW','evaluation_GPU_admission')
    require(all(spec[k] in closure.values() for k in extra),'Complete actual new GPU evaluation primary refs required')
    evaluation=read(bound(spec['evaluation_spec'])); report=read(bound(spec['evaluation_report']))
    require(evaluation['schema']==lineage.GPU_SCHEMA and report['schema']==lineage.GPU_REPORT_SCHEMA and report['status']==lineage.GPU_STATUS and
        evaluation['entry_source']==report['source']==spec['evaluation_entry_source'] and report['spec']==spec['evaluation_spec'] and
        report['GPU_admission']==spec['evaluation_GPU_admission'] and report['phase']==evaluation['phase'] and
        all(ref in closure.values() for ref in evaluation['sources']),
        'Actual new GPU source/spec/report/admission closure differs')
    lineage.check_recipe(evaluation)
    gi=lineage.owner_ref_join(spec,gate['owners'],spec['evaluation_GPU_RAW'],spec['evaluation_GPU_admission'],
        spec['evaluation_entry_source'],spec['evaluation_spec'],'0')
    require(evaluation['normal_RAW_count']==gi and evaluation['RAW_inputs']==spec['RAW_inputs'][:gi] and
        report['RAW_inputs']==evaluation['RAW_inputs'] and
        evaluation['prior_failed_CPU_receipt']==report['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'] and
        evaluation['additional_failed_RAW_inputs']==report['additional_failed_RAW_inputs']==spec['additional_failed_RAW_inputs'] and
        all(spec['terminal_identity_pins'].get(k)==v for k,v in evaluation['terminal_identity_pins'].items()) and
        report['terminal_identity_pins']==evaluation['terminal_identity_pins'],
        'Exact current actual GPU0 prefix and seven-failure identity chain differs')
    fields=tuple(k for k in lineage.EVAL_REFS if k!='entry_source')+tuple(k for k in lineage.REF_KEYS if k!='entry_source')+(
        'endpoint','training_recipe','seed','head_endpoint_epoch','controls','recovered_endpoint_use','selector_mode','selector_parent','validation_protocol')
    require(all(evaluation[k]==spec[k] for k in fields),'Independent CPU must check the exact same endpoint/native/cache/region inputs')
    gate.update(lineage=lineage,evaluation=evaluation,evaluation_report=report)
    gate.update(lineage.endpoint_gate(spec,closure,gate['owners']))
    prior,prior_q=lineage.prior_gate(spec,closure,gate['owners'])
    selector=lineage.selector_parent_gate(spec,closure,gate['owners'])
    manifest=read(bound(report['selector_manifest']))
    require(manifest['schema']==lineage.SELECTOR_SCHEMA and len(manifest['rows'])==187 and manifest['images']==187 and manifest['queries']==18700 and
        report['recipe']==evaluation['recipe'] and report['numeric_protocol']==spec['numeric_protocol'] and
        report['endpoint']==spec['endpoint'] and report['controls']==spec['controls'] and report['seed']==spec['seed'] and
        report['head_endpoint_epoch']==spec['head_endpoint_epoch'] and report['images']==187 and report['raw_GT']==380 and
        report['predictions_per_control']==18700 and set(report['predictions'])==set(report['control_results'])==set(spec['controls']) and
        report['teacher_checkpoint']==prior['teacher_checkpoint']==gate['cache_report']['teacher_checkpoint'] and
        report['packet_refs']==prior_q['packet_refs'] and report['selector_mode']==spec['selector_mode'] and
        report['selector_parent']==spec['selector_parent'] and report['recovered_endpoint_use']==spec['recovered_endpoint_use'] and
        report['exact_native_export_and_COCO'] is True and report['exact_shared_GPU_L_selector'] is True and
        report['external_region_gate'] is True and report['cleanup']['allocated_after']==0 and
        report['current_owned_GPU_terminal_self_certified'] is False and report['no_TEST'] is True and report['best_DEV_selection'] is False,
        'Actual full187 GPU report/selector endpoint protocol differs')
    if spec['selector_mode']=='produce':
        require(manifest['source']==spec['evaluation_entry_source'] and manifest['spec']==spec['evaluation_spec'] and
            manifest['GPU_admission']==spec['evaluation_GPU_admission'],'Exact current selector producer provenance differs')
    else:
        require(report['selector_manifest']==selector['manifest_ref'] and manifest==selector['manifest'],
            'CPU must replay exactly the qualified reused selector')
    outputs=[report[k] for k in ('start','trace','image_diagnostics','selector_manifest')]+list(report['predictions'].values())
    outputs += [row['carrier'] for row in manifest['rows']]
    require(all(ref in closure.values() for ref in outputs),'All GPU exports/selector carrier bytes must be independently sources-bound')
    gate.update(prior=prior,prior_q=prior_q,report=prior,packets=prior_q['packet_refs'],selector_manifest=manifest,selector=selector)
    return gate


def dense_query(torch,formula,state,F_original,z,indices,started,spec):
    result=z.clone(); calls=0; flat=F_original.flatten(1)
    for first in range(0,len(indices),2048):
        deadline(started,spec); idx=torch.from_numpy(indices[first:first+2048])
        F=flat[:,idx].T; L=z.flatten()[idx]; flag=torch.ones(len(idx),dtype=torch.bool,device='cpu')
        predicted=formula.independent_logits(torch,state,F,L,flag)
        require(predicted.dtype==torch.float32 and predicted.device.type=='cpu' and tuple(predicted.shape)==(len(idx),) and
            bool(torch.isfinite(predicted).all()),'Independent CPU ungated PR formula differs')
        result.flatten()[idx]=predicted; calls+=1
    return result,calls


def support_check(torch,np,masks,reader,exported,z):
    actual=masks.decode(reader.rle_bytes(exported['segmentation'])).astype(bool)
    support=torch.from_numpy(np.ascontiguousarray(actual))
    require(tuple(support.shape)==tuple(z.shape),'Actual exported raster shape differs')
    tau=spec_numeric['refined_support_atol']+spec_numeric['refined_support_rtol']*z.abs()
    flips=support!=(z>0)
    require(not bool((flips & (z.abs()>tau)).any()),'GPU/CPU refined support differs outside prelocked near-zero allowance')
    return support,dict(CPU_GPU_support_flips=int(flips.sum()),practical_near_zero_pixels=int((z.abs()<=tau).sum()))


def replay(torch,np,cv2,masks,reader,renderer,n10,region_reader,formula,coco,gate,populations,factors,states,started,spec):
    prior,report=gate['prior'],gate['evaluation_report']; rows=read(bound(prior['image_rows']))
    diagnostics=read(bound(report['image_diagnostics']))
    trace=[json.loads(line) for line in bound(report['trace']).read_text(encoding='utf-8').splitlines() if line.strip()]
    manifest=gate['selector_manifest']; require(len(rows)==len(diagnostics)==len(trace)==len(manifest['rows'])==187 and diagnostics==trace,
        'All187 GPU trace/selector rows must be complete')
    checks=[]; calls={arm:0 for arm in ARMS}; totals={arm:dict(added_pixels=0,removed_pixels=0,changed_pixels=0) for arm in ARMS}
    native_allclose_failed_images=0
    with torch.no_grad():
        for index,(row,ref,observed,selector_row) in enumerate(zip(rows,gate['packets'],diagnostics,manifest['rows'])):
            deadline(started,spec)
            packet=torch.load(str(bound(ref)),map_location='cpu')
            fields,identity,qids=n10.packet_check(torch,reader,coco,packet,row,index,ref,gate)
            carrier=torch.load(str(bound(selector_row['carrier'])),map_location='cpu')
            z_all=carrier['native_L']; expected=dict(dataset_index=index,image_id=row['image_id'],identity=identity,
                native_packet=ref,natural_query_ids=list(range(100)),native_export_query_ids=qids,renderer_source=spec['renderer_source'])
            require(carrier['schema']==gate['lineage'].SELECTOR_PACKET_SCHEMA and all(carrier[k]==selector_row[k]==v for k,v in expected.items()) and
                isinstance(z_all,torch.Tensor) and z_all.device.type=='cpu' and z_all.dtype==torch.float32 and
                tuple(z_all.shape)==(100,)+tuple(identity['original_shape']) and not z_all.requires_grad and bool(torch.isfinite(z_all).all()) and
                digest(z_all)==carrier['L_digest']==selector_row['L_digest'] and observed['selector']==selector_row and
                observed['dataset_index']==index and observed['image_id']==row['image_id'] and observed['packet']==ref and
                observed['queries']==100 and observed['native_query_order']==qids and
                observed['native_M_equals_eF_exact'] is True and observed['native_masks_scores_and_classes_exact'] is True and
                observed['external_region_gate'] is True and observed['exact_GPU_L_selector'] is True and
                set(observed['controls'])==set(spec['controls'][1:]),'Exact full100 saved GPU selector identity/digest differs')
            producer=gate['evaluation'] if spec['selector_mode']=='produce' else gate['selector']['report']
            expected_source=spec['evaluation_entry_source'] if spec['selector_mode']=='produce' else producer['source']
            expected_spec=spec['evaluation_spec'] if spec['selector_mode']=='produce' else producer['spec']
            expected_admission=spec['evaluation_GPU_admission'] if spec['selector_mode']=='produce' else producer['GPU_admission']
            require(carrier['source']==expected_source and carrier['spec']==expected_spec and carrier['GPU_admission']==expected_admission,
                'Actual selector producer SOURCE/spec/admission differs')
            approx=renderer.render_original(torch,fields['M200'],identity)
            full_allclose=bool(torch.allclose(approx,z_all,rtol=1e-4,atol=1e-4))
            native_allclose_failed_images+=not full_allclose
            F_original=n10.original_feature(torch,renderer,fields['F256'],identity)
            probabilities=fields['final_class_logits'][0].softmax(-1)[:,0]
            query_checks=[]; Ks=[]; fallbacks=[]; region_digests={arm:[] for arm in ARMS}; iid=row['image_id']
            for ordinal,qid in enumerate(qids):
                deadline(started,spec); z=z_all[qid]; P,regions,K,fallback=region_reader.independent_regions(np,cv2,z.numpy())
                # Store region facts in natural IDs, not native export order.
                native_row=populations['native_official'][iid][ordinal]
                require(native_row['query_id']==qid,'Natural query/existing prediction ordinal differs')
                native_support=masks.decode(reader.rle_bytes(native_row['segmentation'])).astype(bool)
                require(np.array_equal(native_support,P),'Saved exact GPU L sign must equal original exported native support exactly')
                factor=factors[iid,qid]; probability=float(probabilities[qid])
                mask_support=torch.from_numpy(np.ascontiguousarray(native_support))
                quality=float((approx[qid].sigmoid()*mask_support).sum()/(mask_support.sum()+1e-6))
                require(math.isclose(probability,factor['class_probability'],rel_tol=spec_numeric['class_rtol'],abs_tol=spec_numeric['class_atol']) and
                    math.isclose(quality,factor['mask_quality'],rel_tol=spec_numeric['quality_rtol'],abs_tol=spec_numeric['quality_atol']),
                    'Original native class/quality practical CPU check differs')
                if not bool(native_support.any()): require(native_row['score']==factor['mask_quality']==0.,'Original empty S0 differs')
                require(observed['K_natural_query_order'][qid]==K and observed['fallback_natural_query_order'][qid]==fallback and K==row['band_counts_all100'][qid],
                    'Exact independent sameK/fallback differs')
                per_arm={}
                for arm,control in zip(ARMS,spec['controls'][1:]):
                    region=regions[arm]; region_digest=dict(shape=list(region.shape),dtype='bool',sha256=hashlib.sha256(region.tobytes()).hexdigest())
                    require(region_digest==observed['region_digests'][arm][qid] and int(region.sum())==K,'Independent OpenCV exact region/digest differs')
                    indices=np.flatnonzero(region).astype(np.int64)
                    refined,count=dense_query(torch,formula,states[arm],F_original,z,indices,started,spec)
                    calls[arm]+=count
                    after=populations[control][iid][ordinal]; require(after['query_id']==qid,'Refined query order differs')
                    support,support_stats=support_check(torch,np,masks,reader,after,refined)
                    changed=support.numpy()!=native_support
                    require(not bool((changed & ~region).any()),'Actual exported refined geometry changed outside exact arm region')
                    added=int(np.count_nonzero(~native_support & support.numpy())); removed=int(np.count_nonzero(native_support & ~support.numpy()))
                    current=dict(query_id=qid,native_export_ordinal=ordinal,fixed_native_score=native_row['score'],K=K,
                        native_foreground_pixels=int(native_support.sum()),refined_foreground_pixels=int(support.sum()),
                        added_pixels=added,removed_pixels=removed,changed_pixels=added+removed,native_empty=not bool(native_support.any()),
                        refined_empty=not bool(support.any()),native_empty_new_nonempty_keeps_zero_score=not bool(native_support.any()) and bool(support.any()),
                        new_empty_keeps_positive_native_score=bool(native_support.any()) and not bool(support.any()) and native_row['score']>0)
                    require(current==observed['controls'][control]['query_diagnostics'][ordinal],'Actual per-query pixel/empty accounting differs')
                    per_arm[arm]=dict(current,CPU_independent_chunk_calls=count,**support_stats)
                    for name in totals[arm]: totals[arm][name]+=current[name]
                    del refined,support,changed
                query_checks.append(dict(query_id=qid,native_export_ordinal=ordinal,K=K,fallback=fallback,arms=per_arm))
            for arm,control in zip(ARMS,spec['controls'][1:]):
                actual=observed['controls'][control]; values=[v['arms'][arm] for v in query_checks]
                require(actual['arm']==arm and actual['seed']==spec['seed'] and
                    actual['head_point_chunk_calls']==sum((k+2047)//2048 for k in observed['K_natural_query_order']) and
                    all(actual[name]==sum(v[name] for v in values) for name in
                        ('added_pixels','removed_pixels','changed_pixels','native_empty_new_nonempty_keeps_zero_score','new_empty_keeps_positive_native_score')),
                    'Full100 exact point-budget/chunk/pixel aggregate differs')
            checks.append(dict(dataset_index=index,image_id=iid,selector=selector_row['carrier'],queries=100,
                native_CPU_full_L_allclose=full_allclose,native_CPU_full_L_allclose_is_gate=False,query_checks=query_checks))
            del packet,fields,carrier,z_all,approx,F_original,probabilities
            gc.collect()
            print(json.dumps(dict(N13_CPU_independent_replayed_DEV_images=index+1,all100_queries=100*(index+1))),flush=True)
    return checks,calls,totals,native_allclose_failed_images


def qualify(spec_path):
    global CONTROLS,spec_numeric
    started=time.monotonic(); path=Path(spec_path); require(path.is_absolute(),'Absolute ROOT CPU SPEC required')
    path=path.resolve(strict=True); spec=read(path); gate=source_gate(path,spec)
    CONTROLS=tuple(spec['controls']); spec_numeric=spec['numeric_protocol']
    require(Path(spec['Q_path']).is_absolute() and not Path(spec['Q_path']).exists() and
        Path(spec['details_path']).is_absolute() and not Path(spec['details_path']).exists(),'Fresh absolute CPU outputs required')
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1] and psutil.virtual_memory().available>=16*1024**3,'Actual CPU01/RAM16GiB required')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(),'Qualified CPU/cold CUDA Torch required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2); rng=torch.get_rng_state().clone()
    import numpy as np
    import cv2
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks
    require(np.__version__=='1.23.5' and cv2.__version__=='4.8.0','Qualified NumPy/OpenCV required')
    evaluation=load(spec['evaluation_entry_source'],'_N13_CPU_original_GPU_entry_metadata_helpers')
    heads,states,proofs=evaluation.load_heads(torch,spec,gate); heads.clear(); del heads
    n10=load(spec['N10_evaluation_source'],'_N13_actual_bound_N10_packet_feature_helpers')
    reader=load(spec['reader_source'],'_N13_original_independent_COCO_reader')
    renderer=load(spec['renderer_source'],'_N13_original_renderer_CPU')
    region_reader=load(spec['CPU_region_source'],'_N13_independent_OpenCV_region_reader')
    formula=load(spec['independent_head_source'],'_N13_independent_PR_functional_arithmetic')
    coco=COCO(str(bound(spec['DEV']))); ids=list(coco.imgs)
    require(len(ids)==len(set(ids))==187 and ids==[r['id'] for r in coco.dataset['images']] and len(coco.anns)==380 and set(coco.cats)=={1},
        'Complete canonical DEV187/380/category1 required')
    report=gate['evaluation_report']; predictions={key:read(bound(report['predictions'][key])) for key in CONTROLS}
    populations={key:reader.population(coco,values,ids) for key,values in predictions.items()}
    native=predictions['native_official']
    require(native==read(bound(spec['native_reference_predictions'])) and
        all(len(values)==18700 and all({k:v for k,v in a.items() if k!='segmentation'}=={k:v for k,v in b.items() if k!='segmentation'}
            for a,b in zip(native,values)) for values in predictions.values()),
        'Exact full18700 identity/class/order/fixed-score population required')
    start=read(bound(report['start']))
    require(start['schema']=='N13-sameK-fixed-fullDEV-GPU-start-v1' and start['source']==spec['evaluation_entry_source'] and
        start['spec']==spec['evaluation_spec'] and start['GPU_admission']==spec['evaluation_GPU_admission'] and
        start['endpoint']==spec['endpoint'] and start['head_state_proofs']==proofs==report['head_state_proofs'] and
        start['selector_mode']==spec['selector_mode'] and start['selector_parent']==spec['selector_parent'] and
        start['recovered_endpoint_use']==spec['recovered_endpoint_use'] and start['no_TEST'] is True and start['best_DEV_selection'] is False,
        'Actual GPU START/endpoint/head state proofs differ')
    factors,factor_proof=reader.factor_check(np,read(bound(gate['prior']['native_query_score_factors'])),native,ids)
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,gate['closure'],gate['admission'])
    checks,calls,totals,failed_L=replay(torch,np,cv2,masks,reader,renderer,n10,region_reader,formula,coco,gate,populations,factors,states,started,spec)
    results,matching={},{}
    for control in CONTROLS:
        deadline(started,spec); evaluator,metrics=reader.official_metrics(coco,predictions[control],ids,COCOeval)
        expected=report['control_results'][control]
        require(metrics['iouType']=='segm' and metrics['maxDets']==[1,10,100] and metrics['score_filter'] is None and
            metrics['original_GT_preserved'] is True and metrics['official_COCO_crowd_ignore_semantics'] is True and
            len(metrics['stats'])==12 and all((a==-1 and b==-1) or math.isclose(a,b,rel_tol=0.,abs_tol=spec_numeric['COCO_stats_atol'])
                for a,b in zip(metrics['stats'],expected['stats'])),'Independent original official COCO12 differs')
        matching[control]=reader.matching_images(np,masks,evaluator,ids)
        results[control]=dict(predictions=report['predictions'][control],COCO=metrics,one_to_one=reader.matching_summary(matching[control]))
        del evaluator; gc.collect()
    require(all((a==-1 and b==-1) or math.isclose(a,b,rel_tol=0.,abs_tol=spec_numeric['COCO_stats_atol'])
        for a,b in zip(results['native_official']['COCO']['stats'],gate['prior_q']['expected_E24_COCO_stats'])),
        'Independent native E24 COCO12 anchor differs')
    paired={}; paired_details={}
    for left,right in [('native_official',control) for control in CONTROLS[1:]]+[(CONTROLS[1],CONTROLS[2])]:
        label=right+'_minus_'+left; paired[label],paired_details[label]=reader.paired_differences(matching[left],matching[right])
    fixed,fixed_details=fixed_native_pairs(np,masks,reader,coco,matching['native_official'],populations)
    at50=fixed[CONTROLS[1]][0]
    require(at50['threshold']==.50 and at50['GT']==380 and at50['native_matched_pairs']==374 and
        at50['native_FN']==6 and at50['ignored_GT']==0,'Original fixed374/6 native cohort differs')
    write(spec['details_path'],dict(schema='N13-sameK-fixed-fullDEV-independent-CPU-details-v1',source=gate['source'],spec=gate['spec_ref'],
        evaluation_report=spec['evaluation_report'],full187_all100_formula_checks=checks,
        controls_actual_COCO_matching=matching,paired_reassignment_differences=paired_details,
        fixed_native_query_GT_pairs=fixed_details,fixed_pairs_separate_from_reassignment=True))
    require(torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized() and 'mmdet' not in sys.modules,
        'Independent CPU replay changed outside RNG or initialized native model/CUDA')
    for ref in spec['sources']: deadline(started,spec); bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and
        reference(spec['CPU_admission_file'])==gate['admission_ref'] and
        gate['common'].imported_files(torch,np,cv2,masks,COCO,gate['closure'],gate['admission'])==libraries and
        psutil.Process().cpu_affinity()==[0,1] and torch.get_num_threads()==torch.get_num_interop_threads()==2,
        'Final source/SPEC/CPU admission/import/resource closure changed')
    gate['lineage'].final_endpoint_check(gate['parent']); deadline(started,spec)
    q=dict(schema=Q_SCHEMA,status=STATUS,phase=spec['phase'],source=spec['evaluation_entry_source'],reader_source=gate['source'],
        spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],evaluation_spec=spec['evaluation_spec'],
        evaluation_report=spec['evaluation_report'],evaluation_GPU_RAW=spec['evaluation_GPU_RAW'],evaluation_GPU_admission=spec['evaluation_GPU_admission'],
        normal_RAW_count=spec['normal_RAW_count'],RAW_inputs=spec['RAW_inputs'],prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'],terminal_identity_pins=spec['terminal_identity_pins'],
        seed=spec['seed'],head_endpoint_epoch=spec['head_endpoint_epoch'],endpoint=spec['endpoint'],recovered_endpoint_use=spec['recovered_endpoint_use'],
        recipe=spec['recipe'],numeric_protocol=spec['numeric_protocol'],selector_mode=spec['selector_mode'],selector_manifest=report['selector_manifest'],
        images=187,raw_GT=380,predictions_per_control=18700,controls=spec['controls'],control_results=results,
        exact_GPU_L_regions_all18700_CPU_checked=True,all18700_identity_fixed_score_and_order_CPU_checked=True,
        independent_two_PR_formula_all_selected_pixels_CPU_checked=True,all_changed_pixels_within_exact_region_CPU_checked=True,
        three_original_official_COCO12_CPU_recomputed=True,strong_E24_COCO_reference_CPU_checked=True,
        original_native_374pair_6FN_cohort_preserved=True,paired_one_to_one_differences=paired,fixed_native_query_GT_pairs=fixed,
        details=reference(spec['details_path']),head_state_proofs=proofs,native_factor_check=factor_proof,
        CPU_independent_point_chunk_calls=calls,pixel_change_totals=totals,
        native_CPU_full_L_allclose_failed_images=failed_L,native_CPU_full_L_allclose_is_qualification_gate=False,
        CPU_approximate_L_substituted_for_selection=False,CPU_approximate_masks_substituted_for_metrics=False,
        GPU_CPU_bitwise_head_equality_claimed=False,certified_GPU_error_bound=False,region_boundary_allowance=False,
        head_Module_forward_called=False,optimizer_created=False,backward_calls=0,weights_updated=False,
        per_step_gradient_or_AdamW_replayed=False,no_GT_query_selection=True,no_NMS=True,no_score_cutoff=True,no_TEST=True,best_DEV_selection=False,
        task_efficacy_established=False,innovation_established=False,convergence_established=False,generalization_established=False,
        current_owned_CPU_terminal_self_certified=False,whole_leaf_VRAM_certified=False,
        actual_imported_files=libraries,CPU_affinity=[0,1],threads=2,CUDA_initialized=False,elapsed_seconds=time.monotonic()-started,
        evidence_limits=['Exact saved original GPU L determines both regions; CPU M-to-L allclose remains diagnostic.',
            'OpenCV independently rebuilds all100 sameK regions and functional PR arithmetic checks every selected pixel.',
            'Practical near-zero refined-support allowance is prelocked, excludes region-boundary allowances and is not a certified CUDA error bound.',
            'Official COCO/matching use unchanged GPU exports; approximate CPU masks never substitute those predictions.',
            'Original fixed query/GT pairs include native FN/ignore and remain separate from actual COCO reassignment.',
            'Historical DEV curves do not establish blind TEST, mechanism causality, innovation, convergence, external SOTA or generalization.',
            'ROOT separately closes this current actual CPU held/reaped terminal.'])
    write(spec['Q_path'],q); print(json.dumps(dict(status=STATUS,Q=reference(spec['Q_path'])),allow_nan=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true'); args=parser.parse_args()
    require(args.owner_execute,'Explicit single ROOT-owned CPU evaluation qualification required'); qualify(args.spec)
