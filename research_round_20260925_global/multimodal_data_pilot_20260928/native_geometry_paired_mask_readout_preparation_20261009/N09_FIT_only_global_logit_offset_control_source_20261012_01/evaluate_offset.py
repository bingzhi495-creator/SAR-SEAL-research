"""ROOT-owned N09 full-DEV archived-field GPU offset evaluation.

This is SOURCE preparation, not permission to execute. N09 is a known scalar
control: one FIT-only intercept changes original FP32 logits only inside the
closed original abs(L)<=2 band. Reuse qualified N07 M200/class packets; never
load a teacher/head checkpoint, forward a model, train, or choose a query by GT.
Before using delta, the archived native fields must reproduce every original
query/order/class/FP32 S0 and compressed mask exactly on CUDA, without tolerance.
The offset export retains all 18,700 original scores, including both kinds of
empty-mask transition. ROOT owns admission, final process reaping and any later
independent CPU matching/pixel qualification.
"""
import argparse
import ast
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

SCHEMA='ROOT-N09-FIT-global-band-offset-fullDEV-GPU-v1'
REPORT_SCHEMA='N09-FIT-global-band-offset-fullDEV-GPU-report-v1'
STATUS='N09_FIT_GLOBAL_BAND_OFFSET_FULLDEV_GPU_COMPLETE_PENDING_CPU'
FIT_SPEC_SCHEMA='ROOT-N09-FIT-global-band-offset-CPU-v1'
FIT_REPORT_SCHEMA='N09-FIT-global-band-offset-report-v1'
FIT_STATUS='N09_FIT_GLOBAL_BAND_OFFSET_CPU_COMPLETE_PENDING_ROOT_QUALIFICATION'
N07_REPORT_SCHEMA='N07-original-detail-fixed-endpoint-evaluation-report-v1'
N07_STATUS='N07_ORIGINAL_DETAIL_FIXED_ENDPOINT_EVALUATION_COMPLETE_PENDING_CPU'
N07_Q_SCHEMA='ROOT-N07-original-detail-fixed-endpoint-evaluation-CPU-qualification-v1'
N07_Q_STATUS='ROOT_N07_ORIGINAL_DETAIL_FIXED_ENDPOINT_EVALUATION_CPU_QUALIFIED'
PACKET_SCHEMA='N07-original-detail-all100-native-fields-v1'
OWNER_SHA='ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
RAW_KEYS=('receipt','phase','admission','phase_binding','bootstrap')
REQUIRED_SPEC_KEYS=('schema','phase','sources','RAW_inputs','prior_failed_CPU_receipt',
    'entry_source','output_dir','FIT_offset_report','prior_N07_report','N07_cpu_qualification',
    'native_predictions','DEV','reader_source','renderer_source','export_source','COCO_source',
    'offset_source','GPU_admission_file','deadline_seconds','allocator_limit_MiB','whole_GPU_limit_MiB')
FIT_SPEC_KEYS={'schema','phase','sources','RAW_inputs','prior_failed_CPU_receipt','FIT',
    'cache_report','cache_CPU_qualification','cache_dir','offset_source','test_source',
    'output_dir','CPU_admission_file','deadline_seconds'}
RECIPE=dict(images=187,raw_GT=380,queries_per_image=100,predictions=18700,native_weights_epoch=24,
    inherited_N07_common_head_endpoint=8,renderer_query_chunk=4,render_dtype='float32',
    render_device='cuda:0',native_band_closed_abs_logit=2.,offset_parameters=1,
    offset_fitting='only FIT cached full-point equal-GT BCE; no DEV/AP selection',
    native_export_order='original softmax/topk100 sorted=False; exact archived query order',
    native_score='original class probability * original-mask sigmoid quality, eps=1e-6',
    mask_threshold='strict logit>0',fixed_native_scores=True,fixed_native_class=True,
    fixed_native_query_order=True,all100_queries_retained=True,no_NMS=True,
    no_score_cutoff=True,no_GT_query_selection=True,no_teacher_or_head_replay=True,
    baseline_masks_and_scores_exact=True,baseline_tolerance=None,no_TEST=True,
    best_DEV_selection=False,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,
    CPU_affinity=[0,1],threads=2)


def require(ok,message):
    if not ok: raise RuntimeError(message)


def read(path):
    def reject(value): raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)


def ref_shape(ref):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and
        type(ref['path']) is str and Path(ref['path']).is_absolute() and
        type(ref['bytes']) is int and ref['bytes']>=0 and type(ref['sha256']) is str and
        len(ref['sha256'])==64 and set(ref['sha256'])<=set('0123456789abcdef'),'Malformed absolute byte/SHA reference')


def reference(path):
    path=Path(path).resolve(strict=True); digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def bound(ref):
    ref_shape(ref); require(reference(ref['path'])==ref,'Actual pinned input differs: '+ref['path'])
    return Path(ref['path'])


def terminal(ref,phase,code=0):
    """Only original RAW JSON is read here: never hash/import a payload first."""
    ref_shape(ref); path=Path(ref['path'])
    require(path.name=='PARENT_RECEIPT.json' and type(phase) is str and phase and path.parent.name==phase,
        'Explicit original RAW/phase locator required')
    raw=read(path)
    require(raw['status']=='ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode']==code and
        type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
        raw['held_handle_exit']==dict(wait_state='WAIT_OBJECT_0',actual_exit_code=code) and
        type(raw['held_handle_exit']['actual_exit_code']) is int and
        raw['numeric_birth_absence']['original_birth_absent'] is True and raw.get('error') is None and
        raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors']==[],
        'Original held/reaped terminal%d required before SHA/import'%code)
    if 'phase' in raw: require(raw['phase']==phase,'Original RAW phase differs')
    sha,peak=raw['admission_sha256'],raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha)==64 and set(sha)<=set('0123456789abcdef') and
        type(peak) in (int,float) and math.isfinite(peak) and 0<=peak<=8192 and
        type(raw['whole_GPU_used_limit_MiB']) is int and raw['whole_GPU_used_limit_MiB']==8192 and
        raw['whole_leaf_VRAM_certified'] is False and
        raw['GPU_scope']=='whole GPU observed, not per-process WDDM allocation','Original RAW resource scope differs')
    return raw


def owner_join(row,raw,closure):
    """Join RAW/PB/bootstrap/owner without replaying an ancestor's payload tree."""
    owner,pb=read(bound(row['admission'])),read(bound(row['phase_binding']))
    bootstrap=bound(row['bootstrap']); runtime=Path(row['receipt']['path']).parent
    require(Path(row['phase_binding']['path'])==runtime/'PHASE_BINDING.json' and bootstrap==runtime/'bootstrap.py' and
        owner['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and owner['phase']==row['phase'] and
        owner['driver_sha256']==OWNER_SHA and owner['automatic_retry'] is False and
        owner['allowed_cpus']==[0,1] and owner['cuda_visible_devices'] in ('','0') and
        owner['whole_GPU_used_limit_MiB']==8192 and
        raw['admission_sha256']==row['admission']['sha256']==pb['admission_sha256'] and
        pb['bootstrap_sha256']==row['bootstrap']['sha256'] and pb['argv'][-1]==str(bootstrap) and
        pb['CPU_affinity']==owner['allowed_cpus'] and pb['CUDA_visible_devices']==owner['cuda_visible_devices'] and
        pb['automatic_retry'] is False,'Actual original RAW/owner/PB/bootstrap identity differs')
    entry,args=owner['entry'],owner['entry_args']
    require(entry in closure and owner['bound_source_sha256'].get(entry)==closure[entry]['sha256'],
        'Ancestor entry must remain actually sources-bound')
    tree=ast.parse(bootstrap.read_text(encoding='utf-8'))
    argv=[ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node,ast.Assign) and any(
        isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='sys' and t.attr=='argv'
        for t in node.targets)]
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and
        isinstance(node.func.value,ast.Name) and node.func.value.id=='runpy' and node.func.attr=='run_path']
    require(argv==[[entry]+args] and len(calls)==1 and ast.literal_eval(calls[0].args[0])==entry and
        {v.arg:ast.literal_eval(v.value) for v in calls[0].keywords}=={'run_name':'__main__'},'Actual ancestor bootstrap differs')
    if '--spec' in args:
        require(args.count('--spec')==1,'Ambiguous original SPEC argument')
        locator=args[args.index('--spec')+1]
        require(locator in closure and owner['bound_source_sha256'].get(locator)==closure[locator]['sha256'],
            'Actual ancestor SPEC must remain sources-bound')
    if 'scientific_spec' in owner:
        ref=owner['scientific_spec']
        require(ref in closure.values() and owner['bound_source_sha256'].get(ref['path'])==ref['sha256'],
            'Actual ancestor scientific SPEC differs')
    return owner


def load(ref,name):
    path=bound(ref); source=path.read_bytes()
    module=types.ModuleType(name); module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__); return module


def joined_phase(ref,source,spec_ref,admission,phase,cuda,rows,owners):
    require(all(v in ref.values() for v in (source,spec_ref,admission)),'Parent primary refs absent from actual closure')
    matches=[i for i,owner in enumerate(owners) if owner['entry']==source['path'] and
        owner['entry_args']==['--spec',spec_ref['path'],'--owner-execute']]
    require(len(matches)==1,'Exactly one actual successful parent execution required')
    i=matches[0]; row,owner=rows[i],owners[i]
    require(row['admission']==admission and row['phase']==phase and owner['scientific_spec']==spec_ref and
        owner['cuda_visible_devices']==cuda,'Parent report does not join its actual owned terminal')
    return i


def source_gate(path,spec):
    require(all(k in spec for k in REQUIRED_SPEC_KEYS) and spec['schema']==SCHEMA and
        type(spec['phase']) is str and spec['phase'] and type(spec['RAW_inputs']) is list and spec['RAW_inputs'] and
        all(type(row) is dict and set(RAW_KEYS)<=set(row) for row in spec['RAW_inputs']),
        'Complete ROOT-authored fullDEV GPU SPEC required')
    rows=spec['RAW_inputs']; receipt_paths=[row['receipt']['path'] for row in rows]
    require(len(set(receipt_paths))==len(rows),'Normal ancestor RAW locators must be unique')
    raws=[terminal(row['receipt'],row['phase']) for row in rows]
    terminal(spec['prior_failed_CPU_receipt'],Path(spec['prior_failed_CPU_receipt']['path']).parent.name,1)
    failed_refs=[spec['prior_failed_CPU_receipt']]
    # ROOT may preserve other failed1 attempts explicitly, never in the normal list.
    for row in spec.get('additional_failed_RAW_inputs',[]):
        require(type(row) is dict and set(row)=={'receipt','phase','returncode'} and
            type(row['returncode']) is int and row['returncode']==1,'Explicit historical failed1 locator required')
        terminal(row['receipt'],row['phase'],1); failed_refs.append(row['receipt'])
    require(len({r['path'] for r in failed_refs})==len(failed_refs) and
        not set(receipt_paths)&{r['path'] for r in failed_refs},'Failure cannot qualify a normal ancestor')
    # Every requested terminal gate is complete before the first byte hash/import.
    require(type(spec['sources']) is list and spec['sources'],'Complete byte/SHA source/input closure required')
    for ref in spec['sources']: ref_shape(ref)
    closure={ref['path']:ref for ref in spec['sources']}
    require(len(closure)==len(spec['sources']),'Unique source/input references required')
    for ref in spec['sources']: bound(ref)
    roles=('entry_source','FIT_offset_report','prior_N07_report','N07_cpu_qualification','native_predictions',
        'DEV','reader_source','renderer_source','export_source','COCO_source','offset_source')
    required=[spec[k] for k in roles]+failed_refs+[row[k] for row in rows for k in RAW_KEYS if k!='phase']
    require(all(ref in closure.values() for ref in required),'Actual primary inputs absent from sources')
    source,spec_ref=reference(__file__),reference(path)
    require(source==spec['entry_source'],'Actual N09 evaluation entry SOURCE differs')
    owners=[owner_join(row,raw,closure) for row,raw in zip(rows,raws)]
    admission_path=Path(spec['GPU_admission_file']); require(admission_path.is_absolute(),'Absolute actual own GPU admission required')
    admission_ref=reference(admission_path); admission=read(admission_path)
    require(admission['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        admission['driver_sha256']==OWNER_SHA and admission['phase']==spec['phase'] and
        admission['scientific_spec']==spec_ref and admission['entry']==source['path'] and
        admission['entry_args']==['--spec',str(path),'--owner-execute'] and admission['allowed_cpus']==[0,1] and
        admission['cuda_visible_devices']=='0' and admission['automatic_retry'] is False and
        admission['whole_GPU_used_limit_MiB']==8192 and type(spec['deadline_seconds']) is int and
        spec['deadline_seconds']>0 and admission['qualified_leaf_deadline_seconds']==spec['deadline_seconds'] and
        admission['bound_source_sha256'].get(str(path))==spec_ref['sha256'] and
        all(admission['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in spec['sources']),
        'Actual current ROOT owner/complete input closure differs')
    require(type(spec['allocator_limit_MiB']) is int and spec['allocator_limit_MiB']==4096 and
        type(spec['whole_GPU_limit_MiB']) is int and spec['whole_GPU_limit_MiB']==8192 and
        os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and all(os.environ.get(k)=='2' for k in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'Actual fixed resource policy differs')
    fit=read(bound(spec['FIT_offset_report']))
    require(fit['schema']==FIT_REPORT_SCHEMA and fit['status']==FIT_STATUS and
        type(fit['delta']) in (int,float) and math.isfinite(fit['delta']) and
        fit['delta']==fit['FIT_fitted']['delta'] and fit['offset_source']==spec['offset_source'] and
        fit['FIT']==dict(images=741,GT=1661,points_per_GT=1024,total_points=1661*1024) and
        fit['no_DEV'] is True and fit['no_TEST'] is True and fit['AP_computed'] is False and
        fit['CUDA_initialized'] is False and fit['CPU_affinity']==[0,1] and fit['threads']==2,
        'Finite single FIT-only offset with the actual CPU report ABI required')
    fit_index=joined_phase(closure,fit['source'],fit['spec'],fit['CPU_admission'],fit['phase'],'',rows,owners)
    fitted=read(bound(fit['spec']))
    require(FIT_SPEC_KEYS<=set(fitted) and fitted['schema']==FIT_SPEC_SCHEMA and fitted['phase']==fit['phase'] and
        fit['source'] in fitted['sources'] and fitted['offset_source']==spec['offset_source'] and
        fitted['cache_report']==fit['cache_report'] and fitted['cache_CPU_qualification']==fit['cache_CPU_qualification'] and
        fitted['cache_dir']==fit['cache_dir'] and fitted['FIT']!=spec['DEV'] and
        fitted['CPU_admission_file']==fit['CPU_admission']['path'] and
        all(row in rows[:fit_index] for row in fitted['RAW_inputs']) and
        fitted['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'],
        'Actual scalar fitting SPEC must remain rooted in FIT only')
    fit_math=fit['FIT_fitted']
    require(fit_math['GT_instances']==1661 and fit_math['points_per_GT']==1024 and fit_math['total_points']==1661*1024 and
        fit_math['fixture_mode'] is False and fit_math['fit_input_dtype']=='float32' and
        fit_math['native_band_closed_abs_logit']==2. and fit_math['band_renormalized'] is False and
        fit_math['regularizer_or_offset_cap'] is None and fit_math['no_DEV_selection_by_operator'] is True,
        'Actual uncapped FIT full-point BCE scalar differs')
    report,q=read(bound(spec['prior_N07_report'])),read(bound(spec['N07_cpu_qualification']))
    require(report['schema']==N07_REPORT_SCHEMA and report['status']==N07_STATUS and
        q['schema']==N07_Q_SCHEMA and q['status']==N07_Q_STATUS and
        q['evaluation_report']==spec['prior_N07_report'] and q['source']==report['source'] and
        q['evaluation_spec']==report['spec'] and q['evaluation_GPU_admission']==report['GPU_admission'] and
        q['reader_source']==spec['reader_source'] and report['DEV']==q['DEV']==spec['DEV'] and
        report['images']==q['images']==187 and report['raw_GT']==q['raw_GT']==380 and
        report['predictions_per_control']==q['predictions_per_control']==18700 and
        report['head_endpoint_epoch']==q['head_endpoint_epoch']==8 and
        report['native_forwards']==187 and report['native_weights_updated'] is False and
        report['heads_updated'] is False and report['matching_or_loss_executed'] is False and
        report['original187_all100_low_fields_saved'] is True and report['GPU_original_logits_saved'] is False and
        report['cleanup']['allocated_after']==0 and report['no_TEST'] is True and report['no_best_DEV_selection'] is True and
        q['complete_identity_fixed_scores_and_empty_query_population_CPU_checked'] is True and
        q['five_official_COCO12_CPU_recomputed'] is True and q['strong_E24_native_export_and_COCO_reference_CPU_checked'] is True and
        q['native_M200_original_render_CPU_replayed'] is True and q['no_NMS'] is True and
        q['no_score_cutoff'] is True and q['no_GT_query_selection'] is True and q['no_TEST'] is True,
        'Qualified original full187/380/all100 fixed8 N07 parents required')
    gpu_index=joined_phase(closure,report['source'],report['spec'],report['GPU_admission'],report['phase'],'0',rows,owners)
    cpu_index=joined_phase(closure,q['reader_source'],q['spec'],q['CPU_admission'],q['phase'],'',rows,owners)
    require(q['evaluation_GPU_RAW']==rows[gpu_index]['receipt'] and gpu_index<cpu_index<fit_index and
        all(row in rows[:cpu_index] for row in q['RAW_inputs']) and
        q['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'],'Actual N07 GPU/CPU/FIT chronology differs')
    n07_spec=read(bound(report['spec']))
    reader_spec=read(bound(q['spec']))
    require(n07_spec['official_export_source']==spec['export_source'] and
        n07_spec['official_evaluation_CPU_reader']==spec['COCO_source'] and
        reader_spec['renderer_source']==spec['renderer_source'],'Actual original exporter/COCO/renderer SOURCE binding differs')
    packets=report['final_field_packets']
    require(type(packets) is list and len(packets)==187 and packets==q['packet_refs'] and
        len({ref['path'] for ref in packets})==187 and all(ref in closure.values() for ref in packets) and
        report['image_rows'] in closure.values() and
        report['predictions']['native_official']==spec['native_predictions']==q['control_results']['native_official']['predictions'],
        'Actual archived packet/image-row/native-export inputs differ')
    return dict(closure=closure,source=source,spec_ref=spec_ref,admission_ref=admission_ref,admission=admission,
        fit=fit,report=report,q=q,packets=packets,owners=owners,failed_refs=failed_refs)


def original_export_kernels(ref):
    """Execute only audited original exporter expressions, without a fusion/model.

    The query-order and S0 expressions are copied as AST nodes from the actually
    bound N07 exporter, rather than inferred from a different postprocessor.
    Rendering/quality is chunked, and every resulting native value must be exact.
    """
    path=bound(ref); tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='export_original']
    require(len(functions)==1,'Actual N07 official exporter function required')
    fragments=("probabilities = torch.nn.functional.softmax(classes[0], dim=-1)[:, :-1]",
        "top, indices = probabilities.flatten(0, 1).topk(100, sorted=False)",
        "qids = indices // fusion.num_classes", "binary = ordered > 0",
        "quality = (ordered.sigmoid() * binary).flatten(1).sum(1) / (binary.flatten(1).sum(1) + 1e-6)",
        "scores = top * quality")
    assignments=[]
    for text in fragments:
        expected=ast.parse(text).body[0]; signature=ast.dump(expected,include_attributes=False)
        matches=[n for n in functions[0].body if isinstance(n,ast.Assign) and ast.dump(n,include_attributes=False)==signature]
        require(len(matches)==1,'Original exporter expression ABI differs: '+text); assignments.append(copy.deepcopy(matches[0]))
    module=ast.parse('def order(torch, classes, fusion):\n    pass\ndef quality(torch, ordered, top):\n    pass\n')
    module.body[0].body=assignments[:3]+ast.parse('return top, qids').body
    module.body[1].body=assignments[3:]+ast.parse('return binary, quality, scores').body
    ast.fix_missing_locations(module); scope={}
    exec(compile(module,str(path),'exec'),scope)
    return types.SimpleNamespace(order=scope['order'],quality=scope['quality'])


def imported_files(torch,np,COCO,COCOeval,masks,gate):
    modules=dict(torch=torch,torch_binary=torch._C,numpy=np,numpy_binary=np.core._multiarray_umath,
        COCO=sys.modules[COCO.__module__],COCOeval=sys.modules[COCOeval.__module__],masks=masks,mask_binary=masks._mask)
    result={name:reference(module.__file__) for name,module in modules.items()}
    result['interpolation']=reference(inspect.getsourcefile(torch.nn.functional.interpolate))
    require(all(ref in gate['closure'].values() and
        gate['admission']['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in result.values()),
        'Actual imported library/binary/math SOURCE absent or unbound')
    return result


def segmentation(np,masks,binary):
    require(binary.dtype==np.uint8 and binary.ndim==2,'Original uint8 binary raster required')
    value=masks.encode(np.asfortranarray(binary)); counts=value['counts']
    return dict(size=list(value['size']),counts=counts.decode('ascii') if isinstance(counts,bytes) else counts)


def packet_check(torch,reader,coco,packet,row,index,ref,gate):
    report=gate['report']; iid=row['image_id']; identity=row['identity']; image=coco.imgs[iid]
    require(row['dataset_index']==index and row['packet']==ref and packet['schema']==PACKET_SCHEMA and
        packet['dataset_index']==index and packet['identity']==identity and packet['source']==report['source'] and
        packet['spec']==report['spec'] and packet['phase']==report['phase'] and
        packet['teacher_checkpoint']==report['teacher_checkpoint'] and packet['head_checkpoints']==report['head_checkpoints'] and
        packet['head_endpoint_epoch']==8 and packet['GPU_original_logits_saved'] is False and
        packet['prediction_query_ids']==row['prediction_query_ids'] and packet['field_digests']==row['final_fields'],
        'Actual archived all100 field packet provenance differs')
    H,W=image['height'],image['width']; h,w=identity['actual_img_shape']
    require(identity['image_id']==iid and identity['original_shape']==[H,W] and type(h) is int and type(w) is int and
        0<h<=800 and 0<w<=800 and identity['batch_input_shape']==[800,800] and
        identity['ann_ids']==row['ann_ids']==[a['id'] for a in coco.imgToAnns[iid]] and
        row['raw_GT']==len(identity['ann_ids']),'Canonical DEV original/crop/annotation metadata differs')
    qids=packet['prediction_query_ids']
    require(len(qids)==100 and all(type(q) is int for q in qids) and sorted(qids)==list(range(100)),
        'All100 exact archived query identities required')
    fields=packet['native_fields']
    require(set(fields)==set(packet['field_digests'])=={'F256','e0','M200','final_class_logits'},
        'Original four-field packet ABI differs')
    for name,shape in (('M200',(1,100,200,200)),('final_class_logits',(1,100,2))):
        value=fields[name]
        require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.float32 and
            tuple(value.shape)==shape and not value.requires_grad and bool(torch.isfinite(value).all()) and
            reader.digest(value)==packet['field_digests'][name],'Finite original CPU mask/class field differs: '+name)
    require(dict(reader.digest(fields['final_class_logits'][0]),values=fields['final_class_logits'][0].tolist())==row['native_class_packet'],
        'Archived original class values/digest differ')
    return fields['M200'],fields['final_class_logits'],identity,qids


def evaluate_image(torch,np,masks,renderer,offset,kernels,reader,coco,packet,row,index,ref,native,gate,spec,started):
    M_cpu,C_cpu,identity,archived_qids=packet_check(torch,reader,coco,packet,row,index,ref,gate)
    require([r['query_id'] for r in native]==archived_qids,'Exact original native export order differs')
    M=C=top=qids=low=L=binary=quality=scores=changed=shifted=refined=band=None
    baseline=[]; predictions=[]; diagnostics=[]
    try:
        with torch.no_grad():
            M=M_cpu.to('cuda:0'); C=C_cpu.to('cuda:0')
            top,qids=kernels.order(torch,C,types.SimpleNamespace(num_classes=1))
            require(qids.cpu().tolist()==archived_qids,'Native topk100 sorted=False order changed on CUDA')
            for first in range(0,100,4):
                require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked fullDEV GPU deadline exceeded; no retry/fallback')
                low=M[:,qids[first:first+4]]; L=renderer.render_original(torch,low,identity)
                require(L.device.type=='cuda' and L.dtype==torch.float32 and
                    tuple(L.shape)==(4,)+tuple(identity['original_shape']) and bool(torch.isfinite(L).all()),
                    'Official CUDA FP32 original-space four-query renderer differs')
                binary,quality,scores=kernels.quality(torch,L,top[first:first+4])
                native_scores=scores.cpu().tolist(); native_masks=binary.cpu().numpy().astype(np.uint8)
                # This exact native gate precedes the offset application for the chunk.
                for j in range(4):
                    original=native[first+j]; rle=segmentation(np,masks,native_masks[j])
                    recomputed=dict(image_id=identity['image_id'],query_id=archived_qids[first+j],category_id=1,
                        score=float(native_scores[j]),segmentation=rle)
                    require(recomputed==original,'Native baseline query/mask/class/order/S0 must reproduce exactly; no tolerance')
                    if not native_masks[j].any(): require(original['score']==0.,'Original empty-support native S0 must remain zero')
                    baseline.append(recomputed)
                shifted=offset.apply_offset(torch,L,gate['fit']['delta']); band=L.abs()<=2.
                require(shifted.dtype==torch.float32 and shifted.device==L.device and
                    torch.equal(shifted[~band],L[~band]),'Offset changed a pixel outside the original native closed band')
                refined=shifted>0; changed=refined!=binary
                require(not bool((changed & ~band).any()),'Offset geometry changed outside the original band')
                masks_after=refined.cpu().numpy().astype(np.uint8); bands=band.flatten(1).sum(1).cpu().tolist()
                for j in range(4):
                    original=native[first+j]; before=native_masks[j].astype(bool); after=masks_after[j].astype(bool)
                    added=int(np.count_nonzero(~before & after)); removed=int(np.count_nonzero(before & ~after))
                    if gate['fit']['delta']>=0: require(removed==0,'Nonnegative global offset cannot remove foreground')
                    if gate['fit']['delta']<=0: require(added==0,'Nonpositive global offset cannot add foreground')
                    exported=copy.deepcopy(original); exported['segmentation']=segmentation(np,masks,masks_after[j]); predictions.append(exported)
                    diagnostics.append(dict(query_id=original['query_id'],native_export_ordinal=first+j,
                        fixed_native_score=original['score'],band_pixels=int(bands[j]),
                        native_foreground_pixels=int(before.sum()),offset_foreground_pixels=int(after.sum()),
                        added_pixels=added,removed_pixels=removed,changed_pixels=added+removed,
                        native_empty=not bool(before.any()),offset_empty=not bool(after.any()),
                        native_empty_new_nonempty_keeps_zero_score=not bool(before.any()) and bool(after.any()),
                        new_empty_keeps_positive_native_score=bool(before.any()) and not bool(after.any()) and original['score']>0))
                low=L=binary=quality=scores=changed=shifted=refined=band=None
        require(baseline==native and all({k:v for k,v in a.items() if k!='segmentation'}==
            {k:v for k,v in b.items() if k!='segmentation'} for a,b in zip(native,predictions)),
            'Only geometry may change; original S0/query/class/order/empty policy must stay exact')
        summary=dict(dataset_index=index,image_id=identity['image_id'],packet=ref,queries=100,
            native_query_order=archived_qids,native_masks_scores_and_classes_exact=True,
            fixed_native_scores_and_class_and_order_exact=True,original_band_only=True,
            added_pixels=sum(r['added_pixels'] for r in diagnostics),removed_pixels=sum(r['removed_pixels'] for r in diagnostics),
            changed_pixels=sum(r['changed_pixels'] for r in diagnostics),
            expanded_queries=sum(r['added_pixels']>0 for r in diagnostics),shrunk_queries=sum(r['removed_pixels']>0 for r in diagnostics),
            changed_queries=sum(r['changed_pixels']>0 for r in diagnostics),
            native_empty_queries=sum(r['native_empty'] for r in diagnostics),offset_empty_queries=sum(r['offset_empty'] for r in diagnostics),
            native_empty_new_nonempty_keeps_zero_score=sum(r['native_empty_new_nonempty_keeps_zero_score'] for r in diagnostics),
            new_empty_keeps_positive_native_score=sum(r['new_empty_keeps_positive_native_score'] for r in diagnostics),query_diagnostics=diagnostics)
        return baseline,predictions,summary
    finally:
        M=C=top=qids=low=L=binary=quality=scores=changed=shifted=refined=band=None


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def stats_exact(actual,expected,message):
    require(type(actual) is list and type(expected) is list and len(actual)==len(expected)==12 and
        all(type(v) in (int,float) and math.isfinite(v) and (v==-1 or 0<=v<=1) for v in actual+expected) and
        actual==expected,message)


def run(spec_path):
    started=time.monotonic(); path=Path(spec_path); require(path.is_absolute(),'Absolute ROOT-authored SPEC required')
    path=path.resolve(strict=True); spec=read(path); gate=source_gate(path,spec)
    output=Path(spec['output_dir']); require(output.is_absolute() and not output.exists(),'Fresh absolute fullDEV output directory required')
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1],'Actual CPU01 required before scientific imports')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(),'Qualified original Torch and cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks
    require(np.__version__=='1.23.5','Qualified original NumPy required')
    libraries=imported_files(torch,np,COCO,COCOeval,masks,gate)
    reader=load(spec['reader_source'],'_N09_old_N07_pure_reader')
    renderer=load(spec['renderer_source'],'_N09_actual_official_renderer')
    official=load(spec['COCO_source'],'_N09_actual_official_COCO')
    offset=load(spec['offset_source'],'_N09_fit_only_offset_math')
    require(reader.Q_SCHEMA==N07_Q_SCHEMA and reader.STATUS==N07_Q_STATUS and offset.BAND_LIMIT==2.,
        'Bound old reader/offset helper ABI differs')
    kernels=original_export_kernels(spec['export_source'])
    coco=COCO(str(bound(spec['DEV']))); ids=list(coco.imgs)
    require(len(ids)==len(set(ids))==187 and len(coco.anns)==380 and set(coco.cats)=={1},'Full DEV187/380/category1 required')
    native=read(bound(spec['native_predictions'])); grouped=reader.population(coco,native,ids)
    image_rows=read(bound(gate['report']['image_rows']))
    require(type(image_rows) is list and len(image_rows)==187 and [r['image_id'] for r in image_rows]==ids and
        sum(r['raw_GT'] for r in image_rows)==380,'Exact canonical archived DEV image rows required')
    require(not torch.cuda.is_initialized(),'Payload/library qualification must precede CUDA initialization')
    torch.cuda.init(); torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    require(torch.cuda.memory_allocated(0)==0,'Clean GPU allocator required before archived-field rendering')
    torch.cuda.reset_peak_memory_stats(0); output.mkdir()
    write(output/'START.json',dict(schema='N09-FIT-global-band-offset-fullDEV-GPU-start-v1',phase=spec['phase'],
        source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],recipe=RECIPE,
        FIT_offset_report=spec['FIT_offset_report'],delta=gate['fit']['delta'],prior_N07_report=spec['prior_N07_report'],
        N07_cpu_qualification=spec['N07_cpu_qualification'],native_predictions=spec['native_predictions'],
        archived_packet_refs=gate['packets'],actual_imported_files=libraries,no_teacher_or_head_loaded=True,
        no_optimizer=True,no_weight_update=True,no_TEST=True,no_DEV_offset_selection=True))
    baseline=[]; predictions=[]; diagnostics=[]
    with (output/'TRACE.jsonl').open('x',encoding='utf-8',newline='\n') as trace:
        for index,(row,ref) in enumerate(zip(image_rows,gate['packets'])):
            require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked fullDEV GPU deadline exceeded')
            packet=torch.load(str(bound(ref)),map_location='cpu')
            base,refined,diagnostic=evaluate_image(torch,np,masks,renderer,offset,kernels,reader,coco,packet,row,index,
                ref,grouped[row['image_id']],gate,spec,started)
            baseline+=base; predictions+=refined; diagnostics.append(diagnostic)
            trace.write(json.dumps(diagnostic,allow_nan=False)+'\n'); trace.flush(); del packet,base,refined,diagnostic
            print(json.dumps(dict(rendered_DEV_images=index+1,native_exact_queries=100*(index+1))),flush=True)
    require(len(baseline)==len(predictions)==18700 and baseline==native,'Full baseline exactness/all100 population required')
    reader.population(coco,predictions,ids)
    require(all({k:v for k,v in a.items() if k!='segmentation'}=={k:v for k,v in b.items() if k!='segmentation'}
        for a,b in zip(native,predictions)),'Full18700 fixed original identity/S0/class/order differs')
    if gate['fit']['delta']==0: require(predictions==native,'Zero offset must produce an exactly identical export')
    torch.cuda.synchronize(0); peak=torch.cuda.max_memory_allocated(0); gc.collect()
    before=torch.cuda.memory_allocated(0); torch.cuda.empty_cache(); torch.cuda.synchronize(0)
    cleanup=dict(allocated_before=before,allocated_after=torch.cuda.memory_allocated(0),reserved_after=torch.cuda.memory_reserved(0))
    require(cleanup['allocated_after']==0 and peak<=4096*1024**2,'Clean allocator4096 GPU completion required')
    write(output/'NATIVE0_PREDICTIONS.json',baseline); write(output/'ALL_PREDICTIONS.json',predictions)
    write(output/'IMAGE_DIAGNOSTICS.json',diagnostics)
    require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked native COCO deadline exceeded')
    baseline_COCO=official.coco_metrics(coco,baseline,ids,COCOeval)
    stats_exact(baseline_COCO['stats'],gate['report']['control_results']['native_official']['stats'],
        'Native0 official COCO12 must equal the original GPU reference exactly')
    stats_exact(baseline_COCO['stats'],gate['q']['control_results']['native_official']['COCO']['stats'],
        'Native0 official COCO12 must equal the qualified CPU reference exactly')
    stats_exact(baseline_COCO['stats'],gate['q']['expected_E24_COCO_stats'],
        'Native0 official COCO12 must equal the strong E24 reference exactly')
    require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked offset COCO deadline exceeded')
    offset_COCO=official.coco_metrics(coco,predictions,ids,COCOeval)
    stats_exact(offset_COCO['stats'],offset_COCO['stats'],'Official offset COCO12 must preserve undefined slices as -1')
    for result in (baseline_COCO,offset_COCO):
        require(result['iouType']=='segm' and result['maxDets']==[1,10,100] and result['original_GT_preserved'] is True and
            result['score_filter'] is None and result['official_COCO_crowd_ignore_semantics'] is True,'Actual original COCO protocol differs')
    totals={k:sum(row[k] for row in diagnostics) for k in ('added_pixels','removed_pixels','changed_pixels','expanded_queries',
        'shrunk_queries','changed_queries','native_empty_queries','offset_empty_queries',
        'native_empty_new_nonempty_keeps_zero_score','new_empty_keeps_positive_native_score')}
    require(totals['changed_pixels']==totals['added_pixels']+totals['removed_pixels'],'Pixel change accounting differs')
    for ref in spec['sources']: bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and
        reference(spec['GPU_admission_file'])==gate['admission_ref'] and
        imported_files(torch,np,COCO,COCOeval,masks,gate)==libraries and psutil.Process().cpu_affinity()==[0,1] and
        torch.get_num_threads()==torch.get_num_interop_threads()==2 and torch.cuda.memory_allocated(0)==0,
        'Final provenance/libraries/resources/clean GPU changed')
    result=dict(schema=REPORT_SCHEMA,status=STATUS,phase=spec['phase'],source=gate['source'],spec=gate['spec_ref'],
        GPU_admission=gate['admission_ref'],RAW_inputs=spec['RAW_inputs'],prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec.get('additional_failed_RAW_inputs',[]),start=reference(output/'START.json'),
        trace=reference(output/'TRACE.jsonl'),image_diagnostics=reference(output/'IMAGE_DIAGNOSTICS.json'),
        predictions=reference(output/'ALL_PREDICTIONS.json'),native0_predictions=reference(output/'NATIVE0_PREDICTIONS.json'),
        FIT_offset_report=spec['FIT_offset_report'],FIT_spec=gate['fit']['spec'],delta=gate['fit']['delta'],
        prior_N07_report=spec['prior_N07_report'],N07_cpu_qualification=spec['N07_cpu_qualification'],
        native_reference_predictions=spec['native_predictions'],DEV=spec['DEV'],packet_refs=gate['packets'],
        reader_source=spec['reader_source'],renderer_source=spec['renderer_source'],export_source=spec['export_source'],
        COCO_source=spec['COCO_source'],offset_source=spec['offset_source'],recipe=RECIPE,
        native0_COCO=baseline_COCO,offset_COCO=offset_COCO,native0_COCO_exact_reference=True,
        native0_export_exact_reference=True,native0_exact_queries=18700,all100_identity_and_fixed_scores_exact=True,
        images=187,raw_GT=380,predictions_count=18700,change_totals=totals,
        native_model_loaded=False,point_head_loaded=False,teacher_forwards=0,head_forwards=0,
        optimizer_created=False,backward_calls=0,weights_updated=False,training_or_matching_loss_executed=False,
        no_GT_query_selection=True,no_NMS=True,no_score_cutoff=True,no_TEST=True,best_DEV_selection=False,
        original_native_empty_score_rule_preserved=True,new_empty_geometry_may_keep_positive_native_score=True,
        fixed_native_query_GT_matched_pixel_diagnostics_deferred_to_CPU=True,
        independent_CPU_evaluation_replayed=False,actual_imported_files=libraries,
        torch_peak_allocated_bytes=peak,cleanup=cleanup,elapsed_seconds=time.monotonic()-started,
        evidence_limits=['N09 is one known FIT-only scalar control, not a novel primitive or an innovation claim.',
            'Exact CUDA reproduction is a required gate for this archived renderer/chunk/dtype path; no tolerance substitutes for an original mask.',
            'Only segmentation changes: original S0, all100 query order/classes and native empty-score semantics stay fixed.',
            'Per-image/per-query expansion and shrinkage are descriptive pixel counts; GT-matched conditional pixel comparisons require the separate CPU reader.',
            'DEV has historical exposure; this fixed-control evaluation establishes no blind TEST, convergence, SOTA, or causal mechanism claim.',
            'Inherited N07 qualification does not independently replay teacher/head training or this new scalar evaluation.',
            'The current final RAW and whole-GPU observation are closed by ROOT, not self-certified by this report.'])
    write(output/'REPORT.json',result)
    print(json.dumps(dict(status=STATUS,report=reference(output/'REPORT.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true'); args=parser.parse_args()
    require(args.owner_execute,'Explicit single ROOT-owned execution required'); run(args.spec)
