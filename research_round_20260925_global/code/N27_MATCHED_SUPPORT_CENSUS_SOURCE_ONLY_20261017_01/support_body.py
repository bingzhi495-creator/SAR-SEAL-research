# encoding: utf-8
"""UNBOUND N27 matched-FIT fixed-support census. No scientific imports at import."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics
import time

THRESHOLDS = [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]
IDENTITY_KEYS = ('dataset_index','image_id','GT_index','GT_annotation_id','query_id')


def support_iou_ceiling(GT_area, FP_out, FN_out):
    """Exact best IoU for one nonempty GT with all outside-support bits fixed.

    Choosing GT at every writable pixel removes all inside FP/FN. Outside FP
    adds to the union; outside FN removes from the intersection, not the union.
    This is an instance/mask/support oracle, never an AP or learnability bound.
    """
    if any(type(v) is not int for v in (GT_area,FP_out,FN_out)):
        raise TypeError('Integer counts only; bool and rounded floats are not counts')
    if GT_area <= 0 or FP_out < 0 or not 0 <= FN_out <= GT_area:
        raise ValueError('Nonempty true GT and valid outside FP/FN counts required')
    return Fraction(GT_area-FN_out,GT_area+FP_out)


def need(ok,message):
    if not ok:
        raise RuntimeError(message)


def read(c,ref):
    return json.loads(Path(c.closed(ref)).read_text(encoding='utf-8-sig'))


def digest(np,value):
    array=np.ascontiguousarray(value)
    return dict(shape=list(array.shape),dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def confusion(np,pred,GT,omega):
    tp=pred & GT; fp=pred & ~GT; fn=~pred & GT
    result={}
    for name,mask in (('TP',tp),('FP',fp),('FN',fn)):
        inside=int(np.count_nonzero(mask & omega)); outside=int(np.count_nonzero(mask & ~omega))
        result[name+'_inside']=inside;result[name+'_outside']=outside;result[name]=inside+outside
    denominator=result['TP']+result['FP']+result['FN']
    need(denominator > 0,'Every matched true GT must be nonempty')
    result['IoU_FP64']=float(np.float64(result['TP'])/np.float64(denominator))
    return result


def exact_iou(counts):
    return Fraction(counts['TP'],counts['TP']+counts['FP']+counts['FN'])


def validate_prelock(spec,pre):
    need(spec['operation']=='CPU_MATCHED_SUPPORT_CENSUS'
         and pre['schema']=='N27-matched-support-census-prelock-v1'
         and pre['images']==741 and pre['GT']==1661 and pre['Omega_abs_logit']==2.0
         and pre['thresholds']==THRESHOLDS
         and all(pre[k] is True for k in ('no_new_training','no_DEV','no_TEST','no_AP')),
         'Exact prelocked FIT741/1661, originalOmega, all ten thresholds and no new task')


def capture_evidence(c,context):
    """Only JSON refs/identity; all actual terminals remain Root's authority."""
    reports=context['FIT_capture_reports']
    need(type(reports) is list and len(reports)==2,'Exactly two accepted complete FIT capture scopes')
    by_instance={};by_image={};report_values=[]
    for ref,span in zip(reports,((0,371),(371,741))):
        report=read(c,ref);report_values.append(report)
        expected=list(range(*span))
        need(report['schema']=='N26-FIT-frozen-R-complete-mask-quality-capture-report-v1'
             and report['dataset_indices']==expected and report['images']==len(expected)
             and report['full_FIT_images']==741 and report['full_FIT_GT']==1661
             and report['context']==context['FIT_context']
             and all(report[k] is True for k in ('complete_Omega_R','all100_e_class_L_byteexact',
                 'all_ten_matcher_GT_correspondence_exact','RLE_roundtrip_exact')),
             'Original complete N26 matched FIT capture with exact context/calibration')
        audit=read(c,report['audit'])
        need(audit['schema']=='N26-complete-R-mask-matched-quality-audit-v1'
             and audit['dataset_indices']==expected and audit['private_RLE_not_for_publication'] is True
             and all(audit[k]==report[k] for k in ('source','spec','GPU_admission','phase',
                 'context','prelock','setup_source','quality_source','teacher_checkpoint','fixed_PR_checkpoint')),
             'Exact capture report/AUDIT binding and private mask evidence')
        need([v['dataset_index'] for v in audit['images']]==expected
             and len(audit['records'])==report['samples'],'Original scope image/sample counts')
        for image in audit['images']:
            index=image['dataset_index'];need(index not in by_image,'No duplicate image across capture scopes')
            by_image[index]=image
        for item in audit['records']:
            need(all(type(item[k]) is int for k in IDENTITY_KEYS)
                 and item['dataset_index'] in expected,'Original matched identity fields')
            key=tuple(item[k] for k in IDENTITY_KEYS)
            need(key not in by_instance,'No repeated matched capture identity')
            by_instance[key]=item
    need(set(by_image)==set(range(741)) and len(by_instance)==1661,
         'Exactly741 images and1661 original matched capture records')
    return by_instance,by_image,report_values


def aggregate(records):
    native_total=Counter();R_total=Counter();bands=Counter();ratios=[]
    coverage=[dict(IoU=t,native_reached=0,R_reached=0,support_oracle_reached=0,
        R_below_threshold=0,R_below_but_oracle_reachable=0,
        R_below_and_still_unreachable=0) for t in THRESHOLDS]
    for row in records:
        for kind,total in (('native',native_total),('R',R_total)):
            total.update({k:v for k,v in row[kind].items() if k!='IoU_FP64'})
        R=row['R'];error=R['FP']+R['FN'];outside=R['FP_outside']+R['FN_outside']
        if error:ratios.append(Fraction(outside,error))
        native_iou=exact_iou(row['native']);R_iou=exact_iou(R)
        ceiling=Fraction(row['oracle']['intersection'],row['oracle']['union'])
        band='below_0.50'
        for i,t in enumerate(THRESHOLDS):
            threshold=Fraction(str(t));entry=coverage[i]
            native_ok=native_iou>=threshold;current_ok=R_iou>=threshold;oracle_ok=ceiling>=threshold
            entry['native_reached']+=int(native_ok);entry['R_reached']+=int(current_ok)
            entry['support_oracle_reached']+=int(oracle_ok)
            entry['R_below_threshold']+=int(not current_ok)
            entry['R_below_but_oracle_reachable']+=int(not current_ok and oracle_ok)
            entry['R_below_and_still_unreachable']+=int(not current_ok and not oracle_ok)
            if current_ok:band=('at_least_0.95' if i==9 else '[%.2f,%.2f)'%(t,THRESHOLDS[i+1]))
        bands[band]+=1
    total_error=R_total['FP']+R_total['FN'];outside_error=R_total['FP_outside']+R_total['FN_outside']
    return dict(images=741,samples=len(records),native_counts=dict(native_total),R_counts=dict(R_total),
        R_error_pixels=total_error,R_outside_error_pixels=outside_error,
        R_pooled_outside_error_fraction=(float(Fraction(outside_error,total_error)) if total_error else None),
        R_nonzero_error_instances=len(ratios),R_zero_error_instances=len(records)-len(ratios),
        R_nonzero_instance_outside_error_fraction_median=(float(statistics.median(ratios)) if ratios else None),
        native_empty_instances=sum(r['native_empty'] for r in records),
        R_empty_instances=sum(r['R_empty'] for r in records),K0_instances=sum(r['Omega_points']==0 for r in records),
        R_current_IoU_bands=dict(bands),threshold_coverage=coverage,
        threshold_comparison='exact integer ratios versus exact decimal threshold; no tolerance',
        zero_error_instance_fraction='undefined; excluded from median and counted separately')


def run(torch,np,spec,c,pre):
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    started=time.monotonic();c.check();validate_prelock(spec,pre)
    need(pre==read(c,spec['prelock']) and not torch.cuda.is_initialized() and torch.get_num_threads()==2,
         'Pinned prelock/two CPU threads; no CUDA initialization')
    outside_rng=torch.get_rng_state().clone()
    context=read(c,spec['context'])
    need(context['schema']=='ROOT-N27-matched-support-census-context-v1','Exact N27 context schema')
    accepted=read(c,context['N26_inputs_qualified'])
    need(accepted['schema']=='ROOT-N26-independent-quality-inputs-qualified-v1'
         and accepted['qualified'] is True and accepted['quality_training_eligible'] is True
         and accepted['worker_numerical_verification_complete'] is True
         and accepted['verification_report']==accepted['result']['report_ref'],
         'Root-accepted independent N26 input qualification required')
    worker=read(c,accepted['verification_report'])
    need(worker['schema']=='N26-independent-quality-input-verification-report-pending-v1'
         and worker['numeric_checks_complete'] is True,'Actual qualified independent input worker report')
    fit=read(c,context['FIT_context'])
    need(fit['schema']=='ROOT-N26-quality-input-context-v1' and fit['images']==741 and fit['GT']==1661
         and fit['dataset_indices']==list(range(741))
         and all(len(fit[k])==741 for k in ('existing_packet_refs','existing_native_L_refs','identities','image_ids')),
         'Original full FIT capture context and ordered refs')
    verified_FIT=worker['input_references']['FIT']
    need(set(verified_FIT)=={'N16_report','annotation','context','old_packet_refs','shards'}
         and verified_FIT['N16_report']==fit['N16_GPU_report'] and verified_FIT['annotation']==fit['FIT']
         and verified_FIT['context']==context['FIT_context']
         and verified_FIT['old_packet_refs']==fit['existing_packet_refs']
         and type(verified_FIT['shards']) is list and len(verified_FIT['shards'])==2
         and all({'report','inputs','audit','quality_factory'}<=set(v) for v in verified_FIT['shards'])
         and [v['report'] for v in verified_FIT['shards']]==context['FIT_capture_reports'],
         'Exact Root-qualified independent FIT inputs, no DEV-side reference used')
    evidence,image_evidence,capture_reports=capture_evidence(c,context)
    need(all(shard['inputs']==report['inputs'] and shard['audit']==report['audit']
             for shard,report in zip(verified_FIT['shards'],capture_reports)),
         'Both actual capture input/AUDIT refs equal the independently verified FIT parents')
    n16=read(c,fit['N16_GPU_report'])
    need(n16['spec']==fit['N16_GPU_spec'] and n16['images']==741 and n16['raw_GT']==1661
         and n16['packet_refs']==fit['existing_packet_refs'] and n16['native_L_refs']==fit['existing_native_L_refs'],
         'Original N16 full report/packet/L carrier refs')
    coco=COCO(str(c.closed(fit['FIT'])))
    need(len(coco.imgs)==741 and len(coco.anns)==1661 and set(coco.cats)=={1}
         and len(set(fit['image_ids']))==741 and set(fit['image_ids'])==set(coco.imgs),
         'Original complete FIT741/1661/one category')
    records=[];consumed=set()
    for index in range(741):
        c.check();identity=fit['identities'][index];image_id=fit['image_ids'][index]
        row=n16['rows'][index];p_ref=fit['existing_packet_refs'][index];L_ref=fit['existing_native_L_refs'][index]
        observed=image_evidence[index]
        need(row['dataset_index']==index and row['image_id']==identity['image_id']==image_id==observed['image_id']
             and row['identity']==identity and row['packet']==observed['native_packet']==p_ref
             and row['natural_native_L_carrier']==observed['native_L_carrier']==L_ref,
             'Canonical image and N26/N16 packet/carrier association')
        packet=torch.load(str(c.closed(p_ref)),map_location='cpu')
        carrier=torch.load(str(c.closed(L_ref)),map_location='cpu')
        need(packet['schema']=='N16-all100-FIT-action-sampled-cache-body-v1'
             and carrier['schema']=='N16-actual-GPU-natural100-L-carrier-v1'
             and packet['dataset_index']==carrier['dataset_index']==index
             and packet['identity']==carrier['identity']==identity
             and packet['natural_native_L_carrier']==L_ref
             and packet['teacher_checkpoint']==fit['teacher_checkpoint']
             and packet['fixed_PR_checkpoint']==fit['fixed_PR_checkpoint']
             and all(value[k]==n16[k] for value in (packet,carrier) for k in ('source','spec','GPU_admission','phase')),
             'Exact natural100 packet/carrier/original teacher and PR metadata')
        H,W=identity['original_shape'];ann_ids=identity['ann_ids'];G=len(ann_ids)
        need(packet['raw_GT_ids']==ann_ids and len(set(ann_ids))==G and G>0
             and observed['samples']==row['raw_GT']==G and observed['matched_GT_annotation_ids']==ann_ids,
             'Complete original real GT order')
        L=carrier['native_L'];assignment=packet['assignment'];mapping=assignment['raw_GT_index_by_query']
        need(isinstance(L,torch.Tensor) and L.device.type=='cpu' and L.dtype==torch.float32
             and tuple(L.shape)==(100,H,W) and not L.requires_grad and bool(torch.isfinite(L).all())
             and isinstance(mapping,torch.Tensor) and mapping.device.type=='cpu' and mapping.dtype==torch.int64
             and mapping.shape==(100,) and assignment['natural_query_ids']==list(range(100)),
             'Finite CPUFP32 complete original L and true matcher indices')
        gt_by_q=mapping.tolist()
        need(all(type(g) is int and -2<=g<G for g in gt_by_q)
             and sorted(g for g in gt_by_q if g>=0)==list(range(G))
             and assignment['raw_GT_annotation_id_by_query']==[ann_ids[g] if g>=0 else None for g in gt_by_q],
             'Exactly one query per original GT; negative/ignore proxies excluded')
        by_gt={g:q for q,g in enumerate(gt_by_q) if g>=0}
        need(len(packet['K_counts_all100'])==len(row['K_counts_all100'])==100
             and observed['original_matching_rng_restored'] is True,'Original all100 counts and matcher audit')
        matching_seed=packet['matching_loss_seed']
        need(matching_seed==row['matching_loss_seed'],'Original packet/row matching seed')
        info=coco.imgs[image_id]
        need(info['height']==H and info['width']==W,'Original annotation image grid')
        raw_hash=hashlib.sha256()
        for g in range(G):
            c.check();q=by_gt[g];ann_id=ann_ids[g];key=(index,image_id,g,ann_id,q)
            need(key in evidence and key not in consumed,'One exact original matched N26 evidence record')
            audit=evidence[key];consumed.add(key)
            need(audit['identity']==identity and audit['original_shape']==[H,W]
                 and audit['matching_loss_seed']==matching_seed
                 and audit['complete_Omega_not_sampled_PR'] is True and audit['RLE_roundtrip_exact'] is True
                 and audit['GT_used_in_mask_features'] is False,'Original full-R matched identity/audit')
            ann=coco.anns[ann_id]
            need(ann['image_id']==image_id and ann['category_id']==1 and ann['id']==ann_id,
                 'True original annotation, never a best-IoU substitution')
            GT=np.ascontiguousarray(coco.annToMask(ann)!=0,dtype=np.bool_)
            need(GT.shape==(H,W) and bool(GT.any()) and digest(np,GT)==audit['matched_GT_digest'],
                 'Exact nonempty original matched GT raster/digest')
            raw_hash.update(GT.tobytes())
            values=L[q].numpy();P=values>0;omega=np.abs(values)<=np.float32(2)
            K=int(np.count_nonzero(omega))
            need(K==packet['K_counts_all100'][q]==row['K_counts_all100'][q]==audit['Omega_points']
                 and digest(np,P)==audit['native_mask_digest'],'Exact full-grid Omega/K/native mask')
            rle=audit['R_mask_RLE']
            need(rle['size']==[H,W] and type(rle['counts']) is str,'Private full R mask RLE shape and ASCII counts')
            R=np.ascontiguousarray(masks.decode(dict(size=rle['size'],counts=rle['counts'].encode('ascii')))!=0,
                                    dtype=np.bool_)
            need(R.shape==(H,W) and digest(np,R)==audit['R_mask_digest']
                 and int(np.count_nonzero(R))==audit['R_mask_area']
                 and bool(R.any())==(not audit['empty']) and np.array_equal(R[~omega],P[~omega]),
                 'Exact full R and outsideOmega equality, including K0')
            native_counts=confusion(np,P,GT,omega);R_counts=confusion(np,R,GT,omega)
            need(np.float64(R_counts['IoU_FP64']).tobytes()==np.float64(audit['quality_target_FP64']).tobytes(),
                 'Matched complete R IoU must restore N26 FP64 target bitexact')
            area=int(np.count_nonzero(GT))
            ceiling=support_iou_ceiling(area,R_counts['FP_outside'],R_counts['FN_outside'])
            oracle=R.copy();oracle[omega]=GT[omega]
            oracle_intersection=int(np.count_nonzero(oracle & GT));oracle_union=int(np.count_nonzero(oracle | GT))
            need(oracle_intersection==area-R_counts['FN_outside']
                 and oracle_union==area+R_counts['FP_outside']
                 and Fraction(oracle_intersection,oracle_union)==ceiling
                 and exact_iou(native_counts)<=ceiling and exact_iou(R_counts)<=ceiling,
                 'Direct full-mask oracle equals exact integer formula and bounds both fixed-support masks')
            error=R_counts['FP']+R_counts['FN'];outside_error=R_counts['FP_outside']+R_counts['FN_outside']
            records.append(dict(dataset_index=index,image_id=image_id,GT_index=g,GT_annotation_id=ann_id,
                query_id=q,original_shape=[H,W],GT_area=area,Omega_points=K,
                native_empty=not bool(P.any()),R_empty=not bool(R.any()),native=native_counts,R=R_counts,
                R_outside_error_fraction=(float(Fraction(outside_error,error)) if error else None),
                oracle=dict(intersection=oracle_intersection,union=oracle_union,IoU_FP64=float(ceiling))))
            del GT,values,P,omega,R,oracle,native_counts,R_counts
        need(dict(shape=[G,H,W],dtype='bool',sha256=raw_hash.hexdigest())==packet['raw_GT_digest'],
             'Complete original raw-GT stack digest and order exact')
        del L,mapping,assignment,packet,carrier
        c.check();print(json.dumps(dict(N27_completed_FIT_images=index+1,locked_images=741)),flush=True)
    need(len(records)==len(consumed)==1661 and consumed==set(evidence)
         and {r['GT_annotation_id'] for r in records}==set(coco.anns), 'Every true matched pair and original GT exactly once')
    need(not torch.cuda.is_initialized() and torch.equal(outside_rng,torch.get_rng_state()),
         'No CUDA initialization or CPU RNG change in pure fixed-input census')
    summary=aggregate(records);output=Path(spec['output_directory']);output.mkdir()
    binding=dict(source=c.ref(Path(__file__)),spec=c.execution_spec_ref,
        CPU_admission=c.execution_admission_ref,phase=spec['phase'],context=spec['context'],prelock=spec['prelock'])
    counts_ref=c.write_json(output/'PER_MATCHED_INSTANCE_COUNTS.json',dict(
        schema='N27-private-true-matched-fixed-support-per-instance-counts-v1',**binding,
        images=741,samples=1661,records=records,private_scalars_only=True,qualified=False))
    report_ref=c.write_json(output/'SUPPORT_REPORT.json',dict(
        schema='N27-matched-fixed-support-census-report-pending-v1',
        status='N27_FIT_MATCHED_SUPPORT_CENSUS_COMPLETE_PENDING_ROOT_ACTUAL0',**binding,
        images=741,samples=1661,counts_ref=counts_ref,summary=summary,
        FIT_context=context['FIT_context'],FIT_capture_reports=context['FIT_capture_reports'],
        N26_inputs_qualified=context['N26_inputs_qualified'],all1661_true_matcher_pairs_exact=True,
        native_R_GT_digests_exact=True,all_R_FP64_targets_bitexact=True,all_K_counts_exact=True,
        R_outside_Omega_equals_native=True,direct_oracle_and_integer_formula_exact=True,
        legacy_packet_F_e0_containers_deserialized_but_unused=True,
        original_L_carrier_shape='CPU_FP32_100_by_H_original_by_W_original',
        native_or_R_score_or_class_used=False,query_or_GT_reassigned=False,
        teacher_or_PR_forward=False,new_training=False,AP=False,DEV=False,TEST=False,
        qualified=False,ROOT_actual_CPU0_verified=False,current_owned_terminal_self_certified=False,
        no_AP_upper_bound_claim=True,no_learnability_or_unique_cause_claim=True,
        elapsed_seconds=time.monotonic()-started))
    return dict(report_ref=report_ref,counts_ref=counts_ref,images=741,samples=1661,
                qualified=False,AP=False,new_training=False,DEV=False,TEST=False)
