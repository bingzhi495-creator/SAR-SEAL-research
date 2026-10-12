"""ROOT-owned first-image native rendering/export batching diagnostic.

SOURCE preparation is not permission to execute or scientific qualification.
This new diagnostic preserves the validated ancestor gates and compares only
the first qualified N07 image. It reproduces the original all100 renderer and
exporter expressions, then independently renders/export-quality-reduces chunk4.
All original query identities, class probabilities, quality factors and S0 are
reported, including empty masks. Exact disagreement is a diagnostic result,
never an excuse to replace a score, introduce tolerance or proceed to N09 AP.

There is no offset application, AP computation, model/head load, GT selection,
training, threshold tuning or automatic retry. FIT/offset refs are inherited
lineage only. ROOT separately closes the actual owned terminal and interprets
the first-image findings; the prior failed GPU attempt remains a failed1.
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

SCHEMA='ROOT-N09-first-image-native-export-batch-diagnostic-GPU-v1'
REPORT_SCHEMA='N09-first-image-native-export-batch-diagnostic-report-v1'
STATUS='N09_FIRST_IMAGE_NATIVE_EXPORT_BATCH_DIAGNOSTIC_COMPLETE_PENDING_ROOT_REVIEW'
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
    'offset_source','GPU_admission_file','deadline_seconds','allocator_limit_MiB','whole_GPU_limit_MiB',
    'prior_failed_GPU01_receipt')
FIT_SPEC_KEYS={'schema','phase','sources','RAW_inputs','prior_failed_CPU_receipt','FIT',
    'cache_report','cache_CPU_qualification','cache_dir','offset_source','test_source',
    'output_dir','CPU_admission_file','deadline_seconds'}
RECIPE=dict(images=1,dataset_index=0,queries=100,
    cases=['original_all100','chunk4'],chunk_queries=4,
    input_fields=['M200','final_class_logits'],render_dtype='float32',render_device='cuda:0',
    original_all100_path='render original natural query order, then original exporter indexing/quality',
    chunk4_path='index four queries before rendering; original quality expressions per chunk',
    native_export_order='original softmax/topk100 sorted=False independently in each case',
    mask_threshold='strict logit>0',quality_eps=1e-6,
    archive_comparison='exact by query identity plus separate export order; no tolerance',
    logits_comparison='query-aligned original all100 versus chunk4, exact FP32 values and FP64 differences',
    all100_queries_retained=True,empty_masks_and_scores_retained=True,
    no_offset_application=True,no_AP=True,no_teacher_or_head=True,no_GT_query_selection=True,
    no_threshold_tuning=True,no_TEST=True,no_NMS=True,no_score_cutoff=True,
    allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,deadline_seconds=600,
    CPU_affinity=[0,1],threads=2,root_cause_not_assumed=True)

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


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def deadline(started,spec):
    require(time.monotonic()-started<spec['deadline_seconds'],
        'Prelocked first-image diagnostic deadline exceeded; no fallback/retry')


def score_factors(ref,gate,image_id,native):
    require(ref in gate['closure'].values(),'Actual original score-factor ref must be sources-bound')
    factors=read(bound(ref))
    require(type(factors) is list,'Actual original score-factor list required')
    rows=[row for row in factors if row['image_id']==image_id]
    require(len(rows)==100 and [row['query_id'] for row in rows]==[row['query_id'] for row in native],
        'First-image original all100 score-factor order differs')
    result={}
    for row,prediction in zip(rows,native):
        require(set(row)=={'image_id','query_id','class_probability','mask_quality','official_score'} and
            all(type(row[key]) in (int,float) and math.isfinite(row[key]) and 0<=row[key]<=1
                for key in ('class_probability','mask_quality','official_score')) and
            row['official_score']==prediction['score'],'First-image finite original score factors differ')
        result[row['query_id']]=row
    return result


def tensor_difference(torch,left,right):
    """Compare finite FP32 CPU values without a tolerance or GPU error claim."""
    require(left.device.type==right.device.type=='cpu' and left.dtype==right.dtype==torch.float32 and
        left.shape==right.shape and bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()),
        'Comparable finite FP32 CPU render snapshots required')
    different=left!=right
    difference=left.to(torch.float64)-right.to(torch.float64)
    return dict(exact=bool(torch.equal(left,right)),different_pixels=int(different.sum()),
        max_abs_difference=float(difference.abs().max()),
        min_difference=float(difference.min()),max_difference=float(difference.max()),
        difference_dtype='float64',tolerance=None,certified_GPU_error_bound=False)


def decode_support(masks,rle):
    counts=rle['counts'].encode('ascii') if isinstance(rle['counts'],str) else rle['counts']
    return masks.decode(dict(size=list(rle['size']),counts=counts)).astype(bool)


def export_rows(np,masks,identity,qids,binary,probabilities,quality,scores):
    require(len(qids)==len(binary)==len(probabilities)==len(quality)==len(scores)==100 and
        sorted(qids)==list(range(100)),'Every original query exactly once required')
    rows=[]; factors=[]
    for ordinal,(qid,mask,probability,q,score) in enumerate(zip(qids,binary,probabilities,quality,scores)):
        require(mask.dtype==np.uint8 and mask.shape==tuple(identity['original_shape']) and
            all(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1
                for value in (probability,q,score)),'Finite original raster/export values required')
        empty=not bool(mask.any())
        if empty: require(score==0.,'Original exporter must retain empty masks with zero S0')
        rows.append(dict(image_id=identity['image_id'],query_id=qid,category_id=1,
            score=float(score),segmentation=segmentation(np,masks,mask)))
        factors.append(dict(query_id=qid,export_ordinal=ordinal,class_probability=float(probability),
            mask_quality=float(q),official_score=float(score),foreground_pixels=int(mask.sum()),empty=empty))
    return rows,factors


def compare_archive(np,masks,predictions,factors,native,original_factors):
    archive={row['query_id']:row for row in native}; observed_factors={row['query_id']:row for row in factors}
    require(len(archive)==len(observed_factors)==100,'Complete first-image original-query mapping required')
    archived_order=[row['query_id'] for row in native]; actual_order=[row['query_id'] for row in predictions]
    rows=[]
    for ordinal,prediction in enumerate(predictions):
        qid=prediction['query_id']; original=archive[qid]; actual_factor=observed_factors[qid]; old_factor=original_factors[qid]
        a=decode_support(masks,original['segmentation']); b=decode_support(masks,prediction['segmentation'])
        require(a.shape==b.shape,'Compared archived/case masks must share original raster')
        rows.append(dict(query_id=qid,actual_export_ordinal=ordinal,archived_export_ordinal=archived_order.index(qid),
            RLE_exact=prediction['segmentation']==original['segmentation'],support_exact=bool(np.array_equal(a,b)),
            different_support_pixels=int(np.count_nonzero(a!=b)),added_support_pixels=int(np.count_nonzero(~a & b)),
            removed_support_pixels=int(np.count_nonzero(a & ~b)),
            archived_empty=not bool(a.any()),actual_empty=not bool(b.any()),
            score_exact=prediction['score']==original['score'],archived_score=original['score'],actual_score=prediction['score'],
            score_difference=prediction['score']-original['score'],
            class_probability_exact=actual_factor['class_probability']==old_factor['class_probability'],
            archived_class_probability=old_factor['class_probability'],actual_class_probability=actual_factor['class_probability'],
            class_probability_difference=actual_factor['class_probability']-old_factor['class_probability'],
            mask_quality_exact=actual_factor['mask_quality']==old_factor['mask_quality'],
            archived_mask_quality=old_factor['mask_quality'],actual_mask_quality=actual_factor['mask_quality'],
            mask_quality_difference=actual_factor['mask_quality']-old_factor['mask_quality']))
    return dict(query_order_exact=actual_order==archived_order,archived_query_order=archived_order,actual_query_order=actual_order,
        exact_prediction_dictionary=predictions==native,
        mismatch_queries={name:sum(not row[name] for row in rows) for name in
            ('RLE_exact','support_exact','score_exact','class_probability_exact','mask_quality_exact')},
        different_support_pixels=sum(row['different_support_pixels'] for row in rows),
        max_abs_score_difference=max(abs(row['score_difference']) for row in rows),
        max_abs_quality_difference=max(abs(row['mask_quality_difference']) for row in rows),
        max_abs_class_probability_difference=max(abs(row['class_probability_difference']) for row in rows),
        per_query=rows,tolerance=None)


def compare_cases(np,masks,left_rows,left_factors,right_rows,right_factors,logit_rows):
    right={row['query_id']:row for row in right_rows}; right_factor={row['query_id']:row for row in right_factors}
    logits={row['query_id']:row for row in logit_rows}; rows=[]
    for left,factor in zip(left_rows,left_factors):
        qid=left['query_id']; other=right[qid]; f=right_factor[qid]
        a=decode_support(masks,left['segmentation']); b=decode_support(masks,other['segmentation'])
        rows.append(dict(query_id=qid,rendered_logits=logits[qid],
            RLE_exact=left['segmentation']==other['segmentation'],support_exact=bool(np.array_equal(a,b)),
            different_support_pixels=int(np.count_nonzero(a!=b)),
            score_exact=left['score']==other['score'],all100_score=left['score'],chunk4_score=other['score'],
            chunk4_minus_all100_score=other['score']-left['score'],
            class_probability_exact=factor['class_probability']==f['class_probability'],
            chunk4_minus_all100_class_probability=f['class_probability']-factor['class_probability'],
            mask_quality_exact=factor['mask_quality']==f['mask_quality'],
            all100_mask_quality=factor['mask_quality'],chunk4_mask_quality=f['mask_quality'],
            chunk4_minus_all100_mask_quality=f['mask_quality']-factor['mask_quality']))
    return dict(query_order_exact=[r['query_id'] for r in left_rows]==[r['query_id'] for r in right_rows],
        mismatch_queries={name:sum(not row[name] for row in rows) for name in
            ('RLE_exact','support_exact','score_exact','class_probability_exact','mask_quality_exact')},
        logits_exact_queries=sum(row['rendered_logits']['exact'] for row in rows),
        different_logit_pixels=sum(row['rendered_logits']['different_pixels'] for row in rows),
        max_abs_logit_difference=max(row['rendered_logits']['max_abs_difference'] for row in rows),
        different_support_pixels=sum(row['different_support_pixels'] for row in rows),
        max_abs_score_difference=max(abs(row['chunk4_minus_all100_score']) for row in rows),
        max_abs_quality_difference=max(abs(row['chunk4_minus_all100_mask_quality']) for row in rows),
        per_query=rows,tolerance=None,causal_root_cause_established=False)


def diagnose_first(torch,np,masks,renderer,kernels,identity,M_cpu,C_cpu,native,original_factors,spec,started):
    M=C=natural=ordered=top=qids=binary=quality=scores=low=L=None
    try:
        with torch.no_grad():
            M=M_cpu.to('cuda:0'); C=C_cpu.to('cuda:0'); deadline(started,spec)
            top,qids=kernels.order(torch,C,types.SimpleNamespace(num_classes=1))
            all_order=qids.cpu().tolist(); require(sorted(all_order)==list(range(100)),'Original all100 order must cover all queries')
            natural=renderer.render_original(torch,M,identity)
            require(natural.device.type=='cuda' and natural.dtype==torch.float32 and
                tuple(natural.shape)==(100,)+tuple(identity['original_shape']) and bool(torch.isfinite(natural).all()),
                'Original natural-order all100 CUDA FP32 renderer ABI differs')
            # Preserve original full rendering before original exporter indexing.
            natural_cpu=natural.detach().cpu()
            ordered=natural[qids]; binary,quality,scores=kernels.quality(torch,ordered,top)
            all_rows,all_factors=export_rows(np,masks,identity,all_order,binary.cpu().numpy().astype(np.uint8),
                top.cpu().tolist(),quality.cpu().tolist(),scores.cpu().tolist())
            natural=ordered=top=qids=binary=quality=scores=None
            deadline(started,spec)
            top,qids=kernels.order(torch,C,types.SimpleNamespace(num_classes=1))
            chunk_order=qids.cpu().tolist(); require(sorted(chunk_order)==list(range(100)),'Chunk4 export order must cover all queries')
            chunk_masks=[]; chunk_probabilities=[]; chunk_quality=[]; chunk_scores=[]; logit_differences=[]
            for first in range(0,100,4):
                deadline(started,spec)
                low=M[:,qids[first:first+4]]; L=renderer.render_original(torch,low,identity)
                require(L.device.type=='cuda' and L.dtype==torch.float32 and
                    tuple(L.shape)==(4,)+tuple(identity['original_shape']) and bool(torch.isfinite(L).all()),
                    'Chunk4 CUDA FP32 renderer ABI differs')
                binary,quality,scores=kernels.quality(torch,L,top[first:first+4]); L_cpu=L.detach().cpu()
                for j,qid in enumerate(chunk_order[first:first+4]):
                    logit_differences.append(dict(query_id=qid,**tensor_difference(torch,L_cpu[j],natural_cpu[qid])))
                chunk_masks.extend(binary.cpu().numpy().astype(np.uint8)); chunk_probabilities.extend(top[first:first+4].cpu().tolist())
                chunk_quality.extend(quality.cpu().tolist()); chunk_scores.extend(scores.cpu().tolist())
                low=L=binary=quality=scores=None; del L_cpu
            chunk_rows,chunk_factors=export_rows(np,masks,identity,chunk_order,chunk_masks,
                chunk_probabilities,chunk_quality,chunk_scores)
            return dict(original_all100=dict(predictions=all_rows,score_factors=all_factors,
                    versus_archive=compare_archive(np,masks,all_rows,all_factors,native,original_factors)),
                chunk4=dict(predictions=chunk_rows,score_factors=chunk_factors,
                    versus_archive=compare_archive(np,masks,chunk_rows,chunk_factors,native,original_factors)),
                chunk4_vs_original_all100=compare_cases(np,masks,all_rows,all_factors,chunk_rows,chunk_factors,logit_differences))
    finally:
        M=C=natural=ordered=top=qids=binary=quality=scores=low=L=None


def run(spec_path):
    started=time.monotonic(); path=Path(spec_path); require(path.is_absolute(),'Absolute ROOT-authored diagnostic SPEC required')
    path=path.resolve(strict=True); spec=read(path)
    require(spec['deadline_seconds']==600 and type(spec['deadline_seconds']) is int,
        'Prelocked first-image 600-second diagnostic budget required')
    gate=source_gate(path,spec)
    failed01=spec['prior_failed_GPU01_receipt']
    require(failed01 in gate['failed_refs'] and any(row['receipt']==failed01 and row['returncode']==1
        for row in spec.get('additional_failed_RAW_inputs',[])),
        'Actual prior GPU01 failed1 must be explicitly preserved, never a normal qualification')
    output=Path(spec['output_dir']); require(output.is_absolute() and not output.exists(),'Fresh absolute diagnostic output required')
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1],'Actual CPU01 required before scientific imports')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(),'Original Torch and cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks
    require(np.__version__=='1.23.5','Original qualified NumPy version required')
    libraries=imported_files(torch,np,COCO,COCOeval,masks,gate)
    reader=load(spec['reader_source'],'_N09_diagnostic_pure_old_reader')
    renderer=load(spec['renderer_source'],'_N09_diagnostic_actual_original_renderer')
    require(reader.Q_SCHEMA==N07_Q_SCHEMA and reader.STATUS==N07_Q_STATUS,'Actual old pure reader ABI differs')
    kernels=original_export_kernels(spec['export_source'])
    coco=COCO(str(bound(spec['DEV']))); ids=list(coco.imgs)
    require(len(ids)==len(set(ids))==187 and len(coco.anns)==380 and set(coco.cats)=={1},'Qualified original DEV metadata differs')
    rows=read(bound(gate['report']['image_rows'])); native_full=read(bound(spec['native_predictions']))
    require(type(rows) is list and len(rows)==187 and [row['image_id'] for row in rows]==ids and
        type(native_full) is list and len(native_full)==18700,'Actual canonical archived parent population differs')
    row,ref=rows[0],gate['packets'][0]; iid=row['image_id']; native=native_full[:100]
    require(all(prediction['image_id']==iid for prediction in native) and
        [prediction['query_id'] for prediction in native]==row['prediction_query_ids'],
        'First canonical image must use its exact original all100 export')
    factor_ref=gate['report']['native_query_score_factors']; factors=score_factors(factor_ref,gate,iid,native)
    packet=torch.load(str(bound(ref)),map_location='cpu')
    M_cpu,C_cpu,identity,archived_qids=packet_check(torch,reader,coco,packet,row,0,ref,gate)
    require(archived_qids==[prediction['query_id'] for prediction in native],'Actual first-image packet/native query order differs')
    deadline(started,spec); require(not torch.cuda.is_initialized(),'Ancestor/packet qualification must precede CUDA initialization')
    torch.cuda.init(); torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    require(torch.cuda.memory_allocated(0)==0,'Clean CUDA allocator required before first-image diagnostic')
    torch.cuda.reset_peak_memory_stats(0); output.mkdir()
    write(output/'START.json',dict(schema='N09-first-image-native-export-batch-diagnostic-start-v1',
        source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],phase=spec['phase'],recipe=RECIPE,
        prior_N07_report=spec['prior_N07_report'],N07_cpu_qualification=spec['N07_cpu_qualification'],
        native_reference_predictions=spec['native_predictions'],native_score_factors=factor_ref,
        prior_failed_GPU01_receipt=failed01,dataset_index=0,image_id=iid,packet=ref,identity=identity,
        original_native_query_order=archived_qids,actual_imported_files=libraries,
        no_offset_application=True,no_AP_computation=True,scientific_result_present=False))
    comparison=diagnose_first(torch,np,masks,renderer,kernels,identity,M_cpu,C_cpu,native,factors,spec,started)
    torch.cuda.synchronize(0); peak=torch.cuda.max_memory_allocated(0); gc.collect()
    before=torch.cuda.memory_allocated(0); torch.cuda.empty_cache(); torch.cuda.synchronize(0)
    cleanup=dict(allocated_before=before,allocated_after=torch.cuda.memory_allocated(0),reserved_after=torch.cuda.memory_reserved(0))
    require(cleanup['allocated_after']==0 and peak<=4096*1024**2,'Clean allocator4096 diagnostic completion required')
    for reference_row in spec['sources']: bound(reference_row)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and
        reference(spec['GPU_admission_file'])==gate['admission_ref'] and
        imported_files(torch,np,COCO,COCOeval,masks,gate)==libraries and psutil.Process().cpu_affinity()==[0,1] and
        torch.get_num_threads()==torch.get_num_interop_threads()==2 and torch.cuda.memory_allocated(0)==0,
        'Final diagnostic SOURCE/provenance/library/resource closure changed')
    deadline(started,spec)
    result=dict(schema=REPORT_SCHEMA,status=STATUS,source=gate['source'],spec=gate['spec_ref'],
        GPU_admission=gate['admission_ref'],phase=spec['phase'],start=reference(output/'START.json'),
        RAW_inputs=spec['RAW_inputs'],prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec.get('additional_failed_RAW_inputs',[]),prior_failed_GPU01_receipt=failed01,
        prior_N07_report=spec['prior_N07_report'],N07_cpu_qualification=spec['N07_cpu_qualification'],
        native_reference_predictions=spec['native_predictions'],native_score_factors=factor_ref,packet=ref,
        DEV=spec['DEV'],reader_source=spec['reader_source'],renderer_source=spec['renderer_source'],export_source=spec['export_source'],
        recipe=RECIPE,scope=dict(images=1,dataset_index=0,image_id=iid,queries=100,GT_query_selection=False),
        identity=identity,archived_query_order=archived_qids,archived_predictions=native,
        archived_score_factors=[factors[q] for q in archived_qids],comparison=comparison,
        native_model_loaded=False,point_head_loaded=False,teacher_forwards=0,head_forwards=0,
        optimizer_created=False,backward_calls=0,weights_updated=False,offset_operator_loaded=False,
        offset_applied=False,AP_computed=False,threshold_tuning_performed=False,
        no_NMS=True,no_score_cutoff=True,no_GT_query_selection=True,no_TEST=True,
        exact_differences_reported_without_tolerance=True,root_cause_assumed=False,
        root_cause_established_by_this_report=False,full_DEV_export_qualified=False,
        independent_CPU_evaluation_replayed=False,actual_imported_files=libraries,
        torch_peak_allocated_bytes=peak,cleanup=cleanup,elapsed_seconds=time.monotonic()-started,
        current_owned_terminal_self_certified=False,whole_leaf_VRAM_certified=False,
        evidence_limits=['One first-image diagnostic; this does not qualify the full187/all100 export or N09 efficacy.',
            'Both case exports preserve all100 queries and original exporter expressions, including empty masks and S0.',
            'Exact numeric disagreements are recorded; no tolerance, substitution, offset or AP is used.',
            'Batching is a hypothesis tested here. Agreement or disagreement must be interpreted by ROOT, with the prior failed1 preserved.',
            'FIT and offset metadata remain ancestor-bound only; neither the scalar nor its operator is executed.',
            'No teacher, head, optimizer, training, TEST or independently replayed CPU/GPU error qualification is supplied.',
            'ROOT separately closes the actual owned terminal and observes whole GPU resources.'])
    write(output/'DIAGNOSTIC_REPORT.json',result)
    print(json.dumps(dict(status=STATUS,report=reference(output/'DIAGNOSTIC_REPORT.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true'); args=parser.parse_args()
    require(args.owner_execute,'Explicit new single ROOT-owned diagnostic execution required'); run(args.spec)
