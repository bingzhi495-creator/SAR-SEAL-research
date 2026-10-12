"""Deferred independent C direct PointHead endpoint CPU qualifier.
No backbone forward; CPU/GPU practical tolerance is not a certified numerical bound.
All187 images/all100 queries remain required. No authoring execution.
"""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import types

PREFIX='query-pixel-PointRend-style-fixed-endpoint-evaluation'
SCHEMA='ROOT-'+PREFIX+'-spec-v1'
QUAL_SCHEMA='ROOT-'+PREFIX+'-CPU-qualification-v1'
QUAL_STATUS='ROOT_QUERY_PIXEL_POINTREND_STYLE_FIXED_ENDPOINT_EVALUATION_CPU_QUALIFIED'
CONTROLS=('C_native_official','C_direct_official','C_direct_native_score')
MODES=('native0','global','spatial_halves','spatial_reversed')
OFFICIAL_CPU_BYTES=49923
OFFICIAL_CPU_SHA='2357929fcd38ffc3020c5b8d204ee371da8da3972ff2e2e03398c8225c55e916'
NUMERIC=dict(class_rtol=1e-6,class_atol=1e-6,renderer_rtol=1e-4,renderer_atol=1e-4,
    quality_rtol=1e-4,quality_atol=1e-4,threshold_near_zero_rtol=1e-4,threshold_near_zero_atol=1e-4)
STAGES={'first24':dict(initial_native_epoch=24,phase_epochs=24,native_epoch=48,head_epoch=24,microsteps=17784,updates=8892,GT_visits=39864),
    'warm48':dict(initial_native_epoch=48,phase_epochs=48,native_epoch=96,head_epoch=72,microsteps=35568,updates=17784,GT_visits=79728)}
HEAD_SHAPES={'fc1.weight':(256,257,1),'fc1.bias':(256,),'fc2.weight':(256,257,1),'fc2.bias':(256,),
    'fc3.weight':(256,257,1),'fc3.bias':(256,),'predictor.weight':(1,257,1),'predictor.bias':(1,)}

def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def expect(row, values, label):
    require(isinstance(row, dict) and all(key in row and row[key] == value for key, value in values.items()), label)


def read(path):
    require(Path(path).is_absolute(), 'Absolute JSON locator required')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'bytes', 'sha256'} and
        Path(ref['path']).is_absolute() and type(ref['bytes']) is int and ref['bytes'] > 0 and
        isinstance(ref['sha256'], str) and len(ref['sha256']) == 64 and
        set(ref['sha256']) <= set('0123456789abcdef') and reference(ref['path']) == ref, 'Pinned input differs')
    return Path(ref['path']).resolve(strict=True)


def read_ref(ref):
    return read(bound(ref))


def load(ref, name):
    path = bound(ref); source = path.read_bytes()
    require(path.suffix == '.py' and len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'], 'SOURCE changed')
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def finite(value, label, lower=None, upper=None):
    require(type(value) in (int, float) and math.isfinite(value) and
        (lower is None or value >= lower) and (upper is None or value <= upper), label + ': finite bounded scalar required')


def close(actual, expected, label, rtol, atol):
    finite(actual, label); finite(expected, label)
    require(math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), label + ': numerical mismatch')
    return abs(actual - expected)


def terminal(raw, phase, receipt_path):
    require(type(phase) is str and bool(phase), 'STOP before hashes/imports: nonempty phase required')
    receipt = Path(receipt_path)
    require(receipt.is_absolute() and receipt.name == 'PARENT_RECEIPT.json' and receipt.parent.name == phase,
        'STOP before hashes/imports: explicit original phase-bound receipt path required')
    digest = raw.get('admission_sha256')
    require(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
        'STOP before hashes/imports: lowercase SHA256 admission digest required')
    require('phase' not in raw or (type(raw['phase']) is str and raw['phase'] == phase),
        'STOP before hashes/imports: supplied RAW phase differs')
    expect(raw, dict(status='ACTUAL_OWNED_TERMINAL', returncode=0, root_wait_reaped=True,
        held_handle_exit=dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0), abort=None, error=None,
        wait_errors=False, GPU_observation_errors=[], whole_GPU_used_limit_MiB=8192,
        whole_leaf_VRAM_certified=False, GPU_scope='whole GPU observed, not per-process WDDM allocation'), 'STOP before hashes/imports: raw0 required')
    require(type(raw['returncode']) is int and
        type(raw['held_handle_exit']['actual_exit_code']) is int and type(raw.get('original_identity')) is dict and bool(raw['original_identity']) and
        raw.get('numeric_birth_absence', {}).get('original_birth_absent') is True,
        'STOP: actual held/reaped original birth identity differs')
    finite(raw.get('whole_GPU_observed_peak_MiB'), 'Observed whole GPU peak', 0, 8192)


def tensor(torch, value, shape, dtype, label):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.layout == torch.strided and
        tuple(value.shape) == tuple(shape) and value.dtype == dtype and not value.requires_grad and
        bool(torch.isfinite(value).all()), label + ': detached finite CPU shape/dtype differs')
    return value


def tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def state_digest(state):
    return {name: tensor_digest(value) for name, value in state.items()}


def digest_contract(row, shape, dtype):
    require(isinstance(row, dict) and set(row) == {'shape', 'dtype', 'sha256'} and row['shape'] == list(shape) and
        row['dtype'] == dtype and isinstance(row['sha256'], str) and len(row['sha256']) == 64 and
        set(row['sha256']) <= set('0123456789abcdef'), 'Tensor digest contract differs')


def decode_binary(np, masks, prediction, shape):
    require(set(prediction) == {'image_id', 'query_id', 'category_id', 'score', 'segmentation'} and
        prediction['category_id'] == 1 and type(prediction['query_id']) is int and 0 <= prediction['query_id'] < 100, 'Exact all100 prediction fields differ')
    finite(prediction['score'], 'Exported score', 0, 1)
    rle = prediction['segmentation']
    require(set(rle) == {'size', 'counts'} and rle['size'] == list(shape) and isinstance(rle['counts'], str), 'Original compressed RLE differs')
    encoded = dict(size=rle['size'], counts=rle['counts'].encode('ascii')); binary = masks.decode(encoded)
    require(binary.dtype == np.uint8 and list(binary.shape) == list(shape) and bool(np.isin(binary, [0, 1]).all()) and
        int(masks.area(encoded)) == int(binary.sum(dtype=np.int64)), 'RLE decode/area differs')
    return binary


def render_original(torch, field, actual_shape, original_shape):
    require(field.device.type == 'cpu' and field.dtype == torch.float32 and field.ndim == 4 and field.shape[0] == 1 and
        tuple(field.shape[-2:]) == (200,200), 'CPU FP32 low-field renderer input differs')
    h, w = actual_shape
    require(type(h) is type(w) is int and 0 < h <= 800 and 0 < w <= 800, 'Native crop geometry differs')
    padded = torch.nn.functional.interpolate(field, size=(800,800), mode='bilinear', align_corners=False)
    cropped = padded[:, :, :h, :w]
    original = torch.nn.functional.interpolate(cropped, size=tuple(original_shape), mode='bilinear', align_corners=False)[0]
    require(bool(torch.isfinite(original).all()), 'CPU original renderer is nonfinite')
    return original


def encode_prediction(np, masks, image_id, query, binary, score):
    rle = masks.encode(np.asfortranarray(binary.cpu().numpy().astype(np.uint8)))
    counts = rle['counts'].decode('ascii') if isinstance(rle['counts'], bytes) else rle['counts']
    return dict(image_id=image_id, query_id=query, category_id=1, score=float(score), segmentation=dict(size=list(rle['size']), counts=counts))


def best_geometry(torch, binary, targets, query, current, source):
    predicted = int(binary.sum())
    for index, gt in enumerate(targets):
        area, intersection = int(gt.sum()), int((binary & gt).sum()); union = predicted+area-intersection
        iou = intersection/union if union else 1.
        candidate = dict(best_query_id=query, max_mask_IoU=iou, TP=intersection, FP=predicted-intersection, FN=area-intersection,
            GT_pixels=area, prediction_pixels=predicted, empty_GT_raster=area==0, geometry_source=source)
        if current[index] is None or iou > current[index]['max_mask_IoU'] or (iou == current[index]['max_mask_IoU'] and query < current[index]['best_query_id']):
            current[index] = candidate


def raw_gate(ref,phase):
    path=Path(ref['path'])
    require(path.is_absolute() and path.name=='PARENT_RECEIPT.json' and path.parent.name==phase,'Absolute raw phase locator required before read')
    value=read(path); terminal(value,phase,path); return value


def source_gate(path,spec):
    receipt=path.parent/'finite_point_cache_execution_20261005_v2'/'runs'/spec['phase']/'PARENT_RECEIPT.json'
    raw=raw_gate(dict(path=str(receipt)),spec['phase'])
    train=read(spec['training_spec']['path']); template=read(train['native_template_spec']['path'])
    require(template['execution_kind']=='formal_additional24' and set(template['training_receiptmap'])==set(template['training_phasemap'])==set(MODES),
        'Original four raw population differs')
    for mode in MODES: raw_gate(template['training_receiptmap'][mode],template['training_phasemap'][mode])
    raw_gate(template['restoration_receipt'],template['restoration_phase'])
    raw_gate(train['point_synthetic_CPU_receipt'],train['point_synthetic_CPU_phase'])
    for row in train.get('prior_receipts',[]):
        require(set(row)=={'receipt','phase'},'Explicit ancestry raw required'); raw_gate(row['receipt'],row['phase'])
    raw_gate(train['C_tiny_receipt'],train['C_tiny_phase'])
    raw_gate(train['C_tiny_CPU_receipt'],train['C_tiny_CPU_phase'])
    if spec['endpoint_kind']=='warm48':
        raw_gate(train['first24_receipt'],train['first24_phase'])
        raw_gate(train['first24_CPU_receipt'],train['first24_CPU_phase'])
    raw_gate(spec['training_receipt'],spec['training_phase'])
    raw_gate(spec['training_CPU_receipt'],spec['training_CPU_phase'])
    # Current plus original/restoration/technical/own first/current training precede SHA or SOURCE execution.
    source,reader,spec_ref=reference(Path(__file__).with_name('evaluate.py')),reference(__file__),reference(path)
    require(spec['entry_source']==source and spec['CPU_reader']==reader,'Actual current C evaluation SOURCE/reader differs')
    producer=load(source,'_C_endpoint_provenance_ONLY')
    actual_source,actual_spec,admission,lineage=producer.source_gate(path,spec)
    require(actual_source==source and actual_spec==spec_ref and raw['admission_sha256']==admission['sha256'] and
        producer.CONTROLS==CONTROLS and producer.NUMERIC==NUMERIC,'Actual raw/owner/protocol differs')
    official=spec['official_evaluation_CPU_reader']
    require(official in spec['sources'] and official['bytes']==OFFICIAL_CPU_BYTES and official['sha256']==OFFICIAL_CPU_SHA and
        Path(official['path']).resolve()==(Path(__file__).parent.parent/'spatial_position_preserving_common24_evaluation_source_20261009'/'qualify_evaluation_cpu.py').resolve(),
        'Sealed canonical official COCO12 helper required')
    return producer,dict(source=source,reader_source=reader,spec=spec_ref,parent_receipt=reference(receipt),GPU_admission=admission,phase=spec['phase']),lineage


def CPU_owner_gate(spec_path,output,admission_path,spec,refs,report,report_ref,start_ref,aggregate_ref):
    require(admission_path.is_absolute(),'Explicit absolute evaluation CPU admission required')
    owner=read(admission_path); admission=reference(admission_path)
    expect(owner,dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE',automatic_retry=False,allowed_cpus=[0,1],cuda_visible_devices='',
        whole_GPU_used_limit_MiB=8192,driver_sha256='ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'),'Actual CPU owner policy differs')
    require(type(owner['phase'])is str and owner['phase']!=spec['phase'] and type(owner['qualified_leaf_deadline_seconds'])is int and
        owner['qualified_leaf_deadline_seconds']>0 and Path(owner['entry']).resolve(strict=True)==Path(refs['reader_source']['path']) and
        owner['entry_args']==['--spec',str(spec_path),'--acceptance',str(output),'--CPU-admission',str(admission_path),'--owner-execute'],
        'Actual evaluation CPU phase/entry/output args differ')
    row=report['result']['C']; images=read_ref(row['image_rows'])
    inputs=spec['sources']+[refs[k]for k in ('source','reader_source','spec','parent_receipt','GPU_admission')]+[
        report_ref,start_ref,aggregate_ref,row['image_rows'],row['state_proof']]+[image['final_field_packet']for image in images]+[
        row['controls'][name][key]for name in CONTROLS for key in ('predictions','score_factors')]
    require(all(owner['bound_source_sha256'].get(r['path'])==r['sha256']for r in inputs),
        'Complete actual evaluation numerical inputs/SOURCE/SPEC must be bound to actual CPU owner')
    return admission


def report_check(spec,refs,stage):
    output=Path(spec['output_directory']); report_ref=reference(output/'REPORT.json')
    report,start,aggregate=read_ref(report_ref),read(output/'START.json'),read(output/'QUALIFIED.json')
    expect(report,dict(schema=PREFIX+'-report-v1',status='POINTREND_STYLE_FIXED_ENDPOINT_DEV_EXPORT_COMPLETE_NO_AP',
        source=refs['source'],spec=refs['spec'],GPU_admission=refs['GPU_admission'],phase=spec['phase'],endpoint_kind=spec['endpoint_kind'],
        effective_native_epoch=stage['native_epoch'],effective_C_readout_epoch=stage['head_epoch'],training_CPU_qualification=spec['training_CPU_qualification'],
        checkpoint=spec['trained_checkpoint'],images_per_arm=187,raw_GT_per_arm=380,queries_per_image=100,CPU_numeric_protocol=NUMERIC,error=None,
        GT_removed_from_forward=True,loader_accesses_annotations=True,matching_or_loss_executed=False,one_native_forward_per_image=True,
        direct_point_logits=True,native_L_added_to_point_logits=False,main_official_quality_recomputed_on_direct_mask=True,
        secondary_fixed_native_scores_diagnostic_only=True,fixed_score_empty_direct_queries_retained=True,
        original_teacher_E24_only_initializer_and_BN_reference=True,trained_native610_used_for_classes_feature_masks=True,
        no_TEST=True,AP_computed_on_GPU=False,official_COCO12_pending_CPU=True,training_performed=False,optimizer_or_RNG_restored=False,
        endpoint_selection=False,formal_resume_eligible=False,efficacy_claimed=False,faithful_PointRend_reproduction_claimed=False,
        capacity_matched_claimed=False,compute_matched_claimed=False,conditioning_isolated_claimed=False,
        CPU_renderer_GPU_bitwise_claimed=False,certified_GPU_error_bound=False,GPU_original_logits_saved=False),'Complete C report differs')
    require(report['cleanup']['allocated_after']==0 and set(report['result'])=={'C'} and set(report['controls'])==set(CONTROLS) and
        report['controls']==report['result']['C']['controls'],'Complete C population/cleanup differs')
    expect(start,dict(schema=PREFIX+'-start-v1',source=refs['source'],spec=refs['spec'],GPU_admission=refs['GPU_admission'],phase=spec['phase'],
        endpoint_kind=spec['endpoint_kind'],fixed_effective_native_epoch=stage['native_epoch'],fixed_C_readout_epoch=stage['head_epoch'],
        controls=list(CONTROLS),arms=['C'],training_CPU_qualification=spec['training_CPU_qualification'],checkpoint=spec['trained_checkpoint'],
        GT_free_inference=True,loader_accesses_annotations=True,training_performed=False,optimizer_restored=False,AP_computed_on_GPU=False,
        no_TEST=True,endpoint_selection=False,CPU_numeric_protocol=NUMERIC,direct_point_logits=True,native_L_added_to_point_logits=False),'C start differs')
    expect(aggregate,dict(schema=PREFIX+'-aggregate-v1',status='ROOT_POINTREND_STYLE_FIXED_ENDPOINT_EVALUATION_PENDING_CPU_READER',
        source=refs['source'],spec=refs['spec'],report=report_ref,result=report['result'],controls=report['controls'],phase=spec['phase'],
        endpoint_kind=spec['endpoint_kind'],independently_CPU_qualified=False,AP_computed=False,formal_resume_eligible=False),'C pending aggregate differs')
    return report,report_ref,reference(output/'START.json'),reference(output/'QUALIFIED.json')


def own_states(torch,spec,refs,lineage,report,anchor,policy,original_loading):
    record=report['result']['C']; training=lineage['qualification']; stage=lineage['stage']
    expect(record,dict(arm='C',checkpoint=spec['trained_checkpoint'],qualification=spec['training_CPU_qualification'],images=187,raw_GT=380,
        all100_fields_saved=True,field_packets=187,full_native610_exact=True,readout_state_tensors=8,readout_parameters=198402,
        effective_native_epoch=stage['native_epoch'],effective_readout_epoch=stage['head_epoch'],no_GPU_AP=True,no_TEST=True,GT_free_inference=True),'C endpoint record differs')
    require(record['cleanup']['allocated_after']==0,'C CUDA cleanup differs')
    cp=torch.load(str(bound(spec['trained_checkpoint'])),map_location='cpu')
    require(set(cp)=={'schema','meta','native_state_dict','readout_state_dict','native_optimizer_state_dict',
        'readout_optimizer_state_dict','private_order_state','external_rng_state'} and
        cp['schema']=='query-pixel-PointRend-style-native-integration-checkpoint-v1','Own C container required')
    meta,state,head=cp['meta'],cp['native_state_dict'],cp['readout_state_dict']
    expect(meta,dict(source=training['source'],spec=training['spec'],phase=training['phase'],arm='C',stage_kind=spec['endpoint_kind'],
        teacher_checkpoint=spec['checkpointmap']['native0'],teacher_qualification=spec['qualificationmap']['native0'],initial_native_epoch=stage['initial_native_epoch'],
        completed_phase_epochs=stage['phase_epochs'],effective_native_epoch=stage['native_epoch'],effective_readout_epoch=stage['head_epoch'],
        phase_microsteps=stage['microsteps'],phase_updates=stage['updates'],phase_raw_GT_visits=stage['GT_visits'],auxiliary_weight=1.,
        direct_point_logits=True,native_L_added_to_point_logits=False,no_pending_accumulation_gradients=True,formal_resume_eligible=False),'Own C metadata differs')
    for key in ('readout_seed','order_seed','loss_seed_base','point_seed_base'): require(meta[key]==lineage['train'][key],'Own C seed differs')
    endpoint=read_ref(training['result']['C']['endpoint_proof']); proof=read_ref(record['state_proof']); loading=proof['loading']
    require(set(state)==set(anchor) and len(state)==610,'Own C full610 population differs')
    for name,value in state.items(): tensor(torch,value,anchor[name].shape,anchor[name].dtype,'C native '+name)
    require(state_digest(state)==endpoint['after_native610'],'Own C checkpoint/qualified native610 digest differs')
    require(tuple(head)==tuple(HEAD_SHAPES) and len(head)==8 and sum(v.numel()for v in head.values())==198402,'Actual direct Head8/198402 population differs')
    for name,shape in HEAD_SHAPES.items(): tensor(torch,head[name],shape,torch.float32,'C head '+name)
    require(state_digest(head)==endpoint['after_readout'],'Own C checkpoint/qualified direct Head8 digest differs')
    require(len(policy['frozen_names'])==106 and len(policy['BN_buffer_names'])==159 and
        all(torch.equal(state[n],anchor[n])for n in policy['frozen_names']+policy['BN_buffer_names']),'Original E24 typed BN106/159 differs')
    expect(loading,dict(checkpoint=spec['trained_checkpoint'],qualification=spec['training_CPU_qualification'],arm='C',endpoint_kind=spec['endpoint_kind'],
        effective_native_epoch=stage['native_epoch'],effective_readout_epoch=stage['head_epoch'],strict_OWN_native610_loaded=True,strict_OWN_PointHead8_loaded=True,
        trained_fullstate=state_digest(state),trained_readout=state_digest(head),BN_parameter_names=policy['frozen_names'],BN_buffer_names=policy['BN_buffer_names'],
        original_typed_BN_exact=True,active_native_equal_original_E24_required=False,inference_backbone_with_cp=False,
        training_backbone_with_cp=lineage['train']['backbone_with_cp'],optimizer_or_RNG_restored=False,
        checkpoint_optimizer_and_RNG_containers_deserialized=True,direct_point_logits=True,native_L_added_to_point_logits=False),'Actual C loading differs')
    require(loading['original_teacher_loading']==original_loading,'Original E24 constructor/loading must match qualified training anchor')
    expect(proof,dict(schema=PREFIX+'-state-proof-v1',source=refs['source'],spec=refs['spec'],arm='C',before_native610=state_digest(state),after_native610=state_digest(state),
        before_readout=state_digest(head),after_readout=state_digest(head),native610_and_readout8_CPU_identity_exact=True,params_gradNone=True,
        all_parameters_frozen=True,all_modules_eval=True,no_optimizer_or_RNG_restored=True,GT_removed_from_forward=True,
        direct_point_logits=True,native_L_added_to_point_logits=False),'Actual C evaluation state proof differs')
    return head,dict(checkpoint=spec['trained_checkpoint'],state_proof=record['state_proof'],native610_exact_qualified_endpoint=True,
        direct_head8_198402_exact_qualified_endpoint=True,typed_BN106_affine159_buffers_CPU_exact=True,optimizer_or_RNG_restored=False)


def image_population(spec,record,coco):
    rows=read_ref(record['image_rows']); ids=list(coco.imgs); packet_paths=set(); previous_rng=None
    require(len(ids)==len(rows)==187 and len(coco.anns)==380 and set(coco.cats)=={1} and [r['image_id']for r in rows]==ids,'Ordered DEV187/380 differs')
    for index,row in enumerate(rows):
        image_id=ids[index]; image=coco.imgs[image_id]; H,W=image['height'],image['width']; ann_ids=[a['id']for a in coco.imgToAnns[image_id]]
        require(type(H)is type(W)is int and H>0 and W>0 and type(image_id)is int,'Image geometry/ID differs')
        scale=800./max(H,W); actual=[int(H*scale+.5),int(W*scale+.5)]; identity=row['identity']
        expect(identity,dict(image_id=image_id,ann_ids=ann_ids,original_shape=[H,W],actual_img_shape=actual,batch_input_shape=[800,800],
            native_input_gamma=1.,gray_decode='cv2.IMREAD_GRAYSCALE',target_gray_resize='nearest'),'Native loader identity differs')
        require([r['annotation_id']for r in identity['parser_rows']]==ann_ids and all(r['native_parser_retained']is True for r in identity['parser_rows']),
            'Original annotation order/parser retention differs')
        require(Path(identity['image']['path']).resolve()==(Path(spec['image_root'])/image['file_name']).resolve(),'Original image path differs'); bound(identity['image'])
        require(len(identity['scale_factor'])==2,'Native scale factor length differs')
        for value,expected in zip(identity['scale_factor'],(actual[1]/W,actual[0]/H)): close(value,expected,'Native scale',1e-7,1e-7)
        expect(row,dict(arm='C',endpoint_kind=spec['endpoint_kind'],dataset_index=index,image_id=image_id,checkpoint=spec['trained_checkpoint'],
            qualification=spec['training_CPU_qualification'],ann_ids=ann_ids,annotation_count=len(ann_ids),head_seed=100000+image_id,
            GT_removed_from_forward=True,loader_accesses_annotations=True,matching_or_loss_executed=False,external_RNG_restored=True,predictions=100,
            passive_pixel_decoder_calls=1,passive_mask_embed_calls=10,native_all10_tensor_versions_unchanged=True,native_official_fusion_TorchEqual=True,
            all100_queries_retained=True,no_NMS=True,no_score_cutoff=True,GPU_original_logits_saved=False,direct_point_logits=True,
            native_L_added_to_point_logits=False,secondary_native_scores_exact=True,fixed_score_empty_direct_queries_retained=True),'C image observation differs')
        queries=row['prediction_query_ids']; require(isinstance(queries,list) and len(queries)==100 and all(type(q)is int for q in queries) and
            set(queries)==set(range(100)),'Original native query permutation differs')
        rng=row['external_RNG_before']; require(rng==row['external_RNG_after'] and (previous_rng is None or rng==previous_rng) and
            isinstance(rng,list) and len(rng)==2,'External RNG restoration/continuity differs')
        for digest in rng:
            require(isinstance(digest['shape'],list) and len(digest['shape'])==1 and type(digest['shape'][0])is int and digest['shape'][0]>0,'RNG shape differs')
            digest_contract(digest,digest['shape'],'uint8')
        previous_rng=row['external_RNG_after']; shapes=dict(native_masks=(1,100,200,200),classes=(1,100,2),feature=(1,256,200,200))
        require(set(row['final_fields'])==set(shapes),'C low-field digest population differs')
        for name,shape in shapes.items(): digest_contract(row['final_fields'][name],shape,'float32')
        packet=bound(row['final_field_packet']); require(packet==(Path(spec['output_directory'])/'C'/'FIELD_PACKETS'/('IMAGE_%04d.pth'%index)).resolve(strict=True) and
            str(packet)not in packet_paths,'Unique canonical C packet differs'); packet_paths.add(str(packet))
    require(sum(r['annotation_count']for r in rows)==380,'All raw GT must remain'); return ids,rows


def prediction_population(np,record,name,image_rows):
    geometry='native'if name=='C_native_official'else'direct_point_logits'; diagnostic=name=='C_direct_native_score'
    expect(record,dict(images=187,raw_GT=380,predictions_count=18700,geometry=geometry,
        score_kind='fixed_own_native_official'if diagnostic else'own_official_foreground_sigmoid_mean',diagnostic_only=diagnostic,
        AP_computed_on_GPU=False,all100_queries_retained=True),'C control metadata differs')
    predictions,factors=read_ref(record['predictions']),read_ref(record['score_factors'])
    require(isinstance(predictions,list) and isinstance(factors,list) and len(predictions)==len(factors)==18700,'All18700 predictions/factors required')
    mapped={}
    for index,(prediction,factor)in enumerate(zip(predictions,factors)):
        image=image_rows[index//100]; q=image['prediction_query_ids'][index%100]; key=(image['image_id'],q)
        require((prediction['image_id'],prediction['query_id'])==key and key not in mapped,'Original population/order/unique query differs')
        require(set(factor)=={'image_id','query_id','class_probability','quality','score','quality_kind','geometry_kind'} and
            (factor['image_id'],factor['query_id'])==key and factor['score']==prediction['score'] and factor['geometry_kind']==geometry and
            factor['quality_kind']==('fixed_own_native_official_quality'if diagnostic else'official_foreground_sigmoid_mean'),'C factors differ')
        for field in ('class_probability','quality','score'):
            finite(factor[field],'FP32 factor '+field,0,1); require(float(np.float32(factor[field]))==factor[field],'Exact stored FP32 factor required')
        product=factor['class_probability']*factor['quality']; cpu_product=float(np.float32(factor['class_probability'])*np.float32(factor['quality']))
        rounding=(2.**-24/(1.-2.**-24))*abs(product)+2.**-126
        require(abs(cpu_product-product)<=rounding and abs(factor['score']-product)<=rounding,'FP32 product arithmetic differs')
        mapped[key]=(prediction,factor)
    return predictions,mapped


def dense_direct(torch,fine,native,weights,pixel_chunk_size=2048):
    """Independent row-matmul standard point head; direct output, never native+output.

    Pixel chunking bounds intermediates, not total MACs. All original pixels are replayed.
    """
    require(fine.ndim==3 and fine.shape[0]==256 and native.shape==fine.shape[1:] and not torch.is_grad_enabled(),'CPU no-grad field shapes differ')
    rows=fine.reshape(256,-1).transpose(0,1); coarse=native.reshape(-1,1); result=torch.empty_like(native).reshape(-1)
    for start in range(0,len(rows),pixel_chunk_size):
        stop=min(start+pixel_chunk_size,len(rows)); L=coarse[start:stop]; hidden=rows[start:stop]
        for name in ('fc1','fc2','fc3'):
            hidden=torch.nn.functional.linear(torch.cat((hidden,L),dim=1),weights[name+'.weight'][:,:,0],weights[name+'.bias']).relu()
        result[start:stop]=torch.nn.functional.linear(torch.cat((hidden,L),dim=1),weights['predictor.weight'][:,:,0],weights['predictor.bias'])[:,0]
    require(bool(torch.isfinite(result).all()),'Independent CPU direct logits nonfinite'); return result.reshape_as(native)


def official_quality(torch,logits,support):
    binary=support.to(torch.float32); return (logits.sigmoid()*binary).sum()/(binary.sum()+1e-6)


def replay(torch,np,masks,coco,spec,refs,record,weights,ids,rows):
    require(set(record['controls'])==set(CONTROLS),'Exact three C controls required')
    exports,mapped={},{}
    for name in CONTROLS: exports[name],mapped[name]=prediction_population(np,record['controls'][name],name,rows)
    reconstructed={name:[]for name in CONTROLS}; kinds=('native','direct_point_logits')
    cpu_geometry={kind:[]for kind in kinds}; gpu_geometry={kind:[]for kind in kinds}; numerical=[]; packet_refs=[]
    total_flips={name:0 for name in CONTROLS}; exact={name:0 for name in CONTROLS}; empty={name:0 for name in CONTROLS}
    for index,row in enumerate(rows):
        image_id=ids[index]; identity=row['identity']; H,W=identity['original_shape']; h,w=identity['actual_img_shape']
        packet_ref=row['final_field_packet']; packet=torch.load(str(bound(packet_ref)),map_location='cpu'); packet_refs.append(packet_ref)
        require(set(packet)=={'schema','source','spec','arm','endpoint_kind','checkpoint','dataset_index','image_id','identity',
            'native_masks200','final_class_logits','feature','field_digests','prediction_query_ids','GPU_original_logits_saved',
            'all100_fields_saved','direct_point_logits','native_L_added_to_point_logits'},'C packet exact keys differ')
        expect(packet,dict(schema=PREFIX+'-fields-v1',source=refs['source'],spec=refs['spec'],arm='C',endpoint_kind=spec['endpoint_kind'],
            checkpoint=spec['trained_checkpoint'],dataset_index=index,image_id=image_id,identity=identity,field_digests=row['final_fields'],
            prediction_query_ids=row['prediction_query_ids'],GPU_original_logits_saved=False,all100_fields_saved=True,
            direct_point_logits=True,native_L_added_to_point_logits=False),'C packet provenance differs')
        M=tensor(torch,packet['native_masks200'],(1,100,200,200),torch.float32,'M200')
        C=tensor(torch,packet['final_class_logits'],(1,100,2),torch.float32,'Classes')
        F=tensor(torch,packet['feature'],(1,256,200,200),torch.float32,'F256')
        require(all(tensor_digest(value)==row['final_fields'][name]for name,value in (('native_masks',M),('classes',C),('feature',F))),'Actual lowfield bytes differ')
        class_packet=row['native_class_packet']; values=torch.tensor(class_packet['values'],dtype=torch.float32)
        tensor(torch,values,(100,2),torch.float32,'Class packet values')
        require(set(class_packet)=={'shape','dtype','sha256','values'} and tensor_digest(values)=={key:class_packet[key]for key in ('shape','dtype','sha256')} and
            torch.equal(values,C[0]),'Class packet/actual class tensor differ')
        probabilities=C[0].softmax(-1)[:,0]; fine=render_original(torch,F,[h,w],[H,W])
        anns=coco.imgToAnns[image_id]; targets=[]
        for ann in anns:
            raw=np.zeros((H,W),dtype=np.uint8)if ann.get('segmentation')==[]else coco.annToMask(ann)
            require(raw.shape==(H,W) and bool(np.isin(raw,[0,1]).all()),'Original all GT raster differs'); targets.append(torch.from_numpy(raw.astype(bool)))
        best={kind:[None]*len(anns)for kind in kinds}; gpu_best={kind:[None]*len(anns)for kind in kinds}
        cpu_by_query={name:{}for name in CONTROLS}; flips={name:0 for name in CONTROLS}; max_class_error=max_quality_error=0.
        for start in range(0,100,4):
            stop=min(start+4,100); L=render_original(torch,M[:,start:stop],[h,w],[H,W])
            for offset,query in enumerate(range(start,stop)):
                native=L[offset]; direct=dense_direct(torch,fine,native,weights,spec['pixel_chunk_size'])
                actual_native,native_factor=mapped['C_native_official'][(image_id,query)]
                actual_direct,direct_factor=mapped['C_direct_official'][(image_id,query)]
                diagnostic,diagnostic_factor=mapped['C_direct_native_score'][(image_id,query)]
                require(diagnostic['segmentation']==actual_direct['segmentation'] and
                    all(diagnostic_factor[k]==native_factor[k]for k in ('class_probability','quality','score')),
                    'Fixed score diagnostic must use own native S0 and unchanged direct RLE')
                native_binary=torch.from_numpy(decode_binary(np,masks,actual_native,[H,W]).astype(bool))
                native_cpu=native>0; cpu_native_quality=official_quality(torch,native,native_cpu); cpu_native_score=probabilities[query]*cpu_native_quality
                for name,logits in (('C_native_official',native),('C_direct_official',direct),('C_direct_native_score',direct)):
                    prediction,factor=mapped[name][(image_id,query)]; binary=torch.from_numpy(decode_binary(np,masks,prediction,[H,W]).astype(bool))
                    cpu_binary=logits>0; changed=binary!=cpu_binary
                    envelope=NUMERIC['threshold_near_zero_atol']+NUMERIC['threshold_near_zero_rtol']*logits.abs()
                    require(not bool((changed & (logits.abs()>envelope)).any()),'CPU/GPU mask flip outside practical nearzero envelope')
                    count=int(changed.sum()); flips[name]+=count; total_flips[name]+=count; exact[name]+=int(count==0); empty[name]+=int(not bool(binary.any()))
                    max_class_error=max(max_class_error,close(factor['class_probability'],float(probabilities[query]),'CPU ship probability',NUMERIC['class_rtol'],NUMERIC['class_atol']))
                    require(factor['class_probability']==native_factor['class_probability'],'Within C control class probability changed')
                    diagnostic_case=name=='C_direct_native_score'
                    actual_quality=official_quality(torch,native,native_binary)if diagnostic_case else official_quality(torch,logits,binary)
                    max_quality_error=max(max_quality_error,close(factor['quality'],float(actual_quality),'Quality on actual GPU support',NUMERIC['quality_rtol'],NUMERIC['quality_atol']))
                    if not diagnostic_case and not bool(binary.any()): require(factor['quality']==factor['score']==0.,'Own official empty support score must be exact0')
                    if diagnostic_case:
                        cpu_score=cpu_native_score
                    else: cpu_score=probabilities[query]*official_quality(torch,logits,cpu_binary)
                    finite(float(cpu_score),'Independently recomputed CPU score',0,1)
                    cpu_by_query[name][query]=encode_prediction(np,masks,image_id,query,cpu_binary,cpu_score)
                    if not diagnostic_case:
                        kind='native'if name=='C_native_official'else'direct_point_logits'
                        best_geometry(torch,cpu_binary,targets,query,best[kind],'independent_CPU_threshold')
                        best_geometry(torch,binary,targets,query,gpu_best[kind],'actual_GPU_exported_RLE')
                del native,direct,native_binary,native_cpu,binary,cpu_binary,changed,envelope
            del L
        for name in CONTROLS: reconstructed[name].extend(cpu_by_query[name][q]for q in row['prediction_query_ids'])
        for kind in kinds:
            for ann,cpu_result,gpu_result in zip(anns,best[kind],gpu_best[kind]):
                base=dict(image_id=image_id,annotation_id=ann['id'],category_id=ann['category_id'],iscrowd=int(ann.get('iscrowd',0)),
                    ignore=int(ann.get('ignore',0)),original_shape=[H,W])
                for destination,value in ((cpu_geometry[kind],cpu_result),(gpu_geometry[kind],gpu_result)):
                    require(value is not None,'Missing best100 raw GT'); value.update(FN_over_GT_area=value['FN']/max(value['GT_pixels'],1),FP_over_GT_area=value['FP']/max(value['GT_pixels'],1))
                    destination.append(dict(base,**value))
        numerical.append(dict(image_id=image_id,threshold_disagreements=flips,max_class_probability_absolute_error=max_class_error,
            max_actual_support_quality_absolute_error=max_quality_error,direct_output_not_native_residual=True,
            fixed_native_score_diagnostic_recomputed_from_CPU_native_support=True))
        if(index+1)%50==0 or index==186: print(json.dumps(dict(arm='C',CPU_reconstructed_images=index+1)),flush=True)
        del packet,M,C,F,fine,values,probabilities,targets,best,gpu_best,cpu_by_query
    require(all(len(p)==18700 for p in reconstructed.values()) and all(len(g)==380 for g in cpu_geometry.values()),'Complete independent C population differs')
    return dict(exports=exports,reconstructed=reconstructed,CPU_geometry=cpu_geometry,GPU_geometry=gpu_geometry,field_packets=packet_refs,
        numerical=numerical,threshold_disagreements=total_flips,exact_binary_queries=exact,empty_GPU_predictions=empty)


def run(spec_file,acceptance_file,CPU_admission_file):
    path,target=Path(spec_file),Path(acceptance_file)
    require(path.is_absolute() and target.is_absolute() and not target.exists(),'Absolute SPEC/fresh CPU acceptance required')
    path=path.resolve(strict=True); spec=read(path)
    expect(spec,dict(schema=SCHEMA,kind='mask2former',arms=['C'],controls=list(CONTROLS),evaluation_images=187,evaluation_raw_GT=380,
        head_seed_base=100000,CPU_affinity=[0,1],allocator_limit_MiB=4096,whole_GPU_owner_limit_MiB=8192,CPU_numeric_protocol=NUMERIC,
        pixel_chunk_size=2048,direct_point_logits=True,native_L_added_to_point_logits=False,endpoint_selection=False,historically_exposed_DEV=True,no_TEST=True),
        'Prelocked C three-control SPEC differs')
    require(spec['endpoint_kind']in STAGES and type(spec['deadline_seconds'])is int and 0<spec['deadline_seconds']<=21600,'Fixed first24/warm48 endpoint required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2'for k in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU CUDAempty/numeric environment2 required before imports')
    producer,refs,lineage=source_gate(path,spec); stage=STAGES[spec['endpoint_kind']]
    require(lineage['stage']==stage,'Independent stage contract differs')
    report,report_ref,start_ref,aggregate_ref=report_check(spec,refs,stage)
    CPU_admission_path=Path(CPU_admission_file)
    refs['CPU_admission']=CPU_owner_gate(path,target,CPU_admission_path,spec,refs,report,report_ref,start_ref,aggregate_ref)
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1],'CPU01 affinity required before ML imports')
    import torch
    require(not torch.cuda.is_initialized(),'CPU entered with initialized CUDA')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    require(torch.__version__=='2.0.1+cu118' and torch.get_num_threads()==torch.get_num_interop_threads()==2,'Pinned Torch/threads2+2 required')
    require(Path(inspect.getsourcefile(torch.nn.functional.interpolate)).resolve()==bound(spec['interpolation_SOURCE']).resolve(),'Actual interpolation SOURCE differs')
    import numpy as np
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    helper=load(spec['training_CPU_reader_source'],'_C_independent_original_teacher_CPU_helper')
    training_start=read_ref(lineage['qualification']['result']['C']['start']); loading=training_start['teacher_loading']
    original_loading=loading['original_native24_loading']if spec['endpoint_kind']=='warm48'else loading
    _,_,anchor,policy,_,_=helper.original_teacher_state(torch,spec,original_loading)
    weights,state_result=own_states(torch,spec,refs,lineage,report,anchor,policy,original_loading); del anchor
    official=load(spec['official_evaluation_CPU_reader'],'_C_sealed_official_COCO_ONLY'); coco=COCO(str(bound(spec['DEV'])))
    record=report['result']['C']; ids,rows=image_population(spec,record,coco); results={}
    with torch.no_grad():
        numerical=replay(torch,np,masks,coco,spec,refs,record,weights,ids,rows)
        for name in CONTROLS:
            exported=official.coco_metrics(coco,numerical['exports'][name],ids,COCOeval)
            reconstructed=official.coco_metrics(coco,numerical['reconstructed'][name],ids,COCOeval)
            results[name]=dict(predictions=record['controls'][name]['predictions'],score_factors=record['controls'][name]['score_factors'],
                GPU_export_COCO12=exported,independent_CPU_COCO12=reconstructed,
                CPU_minus_GPU_COCO12=[cpu-gpu for cpu,gpu in zip(reconstructed['stats'],exported['stats'])],
                threshold_disagreements=numerical['threshold_disagreements'][name],exact_binary_queries=numerical['exact_binary_queries'][name],
                empty_GPU_predictions_retained=numerical['empty_GPU_predictions'][name],all18700_queries_retained=True,
                geometry_kind=record['controls'][name]['geometry'],CPU_scores_independently_recomputed=True,GT_used_for_scores=False,
                fixed_native_score_diagnostic_only=name=='C_direct_native_score')
        arm_result=dict(state_result,image_rows=record['image_rows'],field_packets=numerical['field_packets'],per_image_numerical_checks=numerical['numerical'],
            GPU_export_best100={k:official.geometry_summary(v)for k,v in numerical['GPU_geometry'].items()},
            independent_CPU_best100={k:official.geometry_summary(v)for k,v in numerical['CPU_geometry'].items()},
            GPU_export_geometry_rows=numerical['GPU_geometry'],independent_CPU_geometry_rows=numerical['CPU_geometry'],
            lowest_actual_query_ID_wins_geometry_ties=True,both_empty_diagnostic_IoU=1.)
    require(not torch.cuda.is_initialized() and psutil.Process().cpu_affinity()==[0,1] and torch.get_num_threads()==torch.get_num_interop_threads()==2,'CPU resource scope changed')
    for ref in spec['sources']: bound(ref)
    for ref in [refs[k]for k in ('source','reader_source','spec','parent_receipt','GPU_admission','CPU_admission')]+[report_ref,start_ref,aggregate_ref,
        record['image_rows'],record['state_proof']]+numerical['field_packets']+[
        record['controls'][name][k]for name in CONTROLS for k in ('predictions','score_factors')]: bound(ref)
    acceptance=dict(schema=QUAL_SCHEMA,status=QUAL_STATUS,**refs,report=report_ref,start=start_ref,GPU_qualification=aggregate_ref,
        endpoint_kind=spec['endpoint_kind'],effective_native_epoch=stage['native_epoch'],effective_C_readout_epoch=stage['head_epoch'],
        training_CPU_qualification=spec['training_CPU_qualification'],checkpoint=spec['trained_checkpoint'],artifacts=report['result'],
        controls=list(CONTROLS),control_results=results,arms={'C':arm_result},CPU_execution=dict(device='cpu',affinity=[0,1],CUDA_VISIBLE_DEVICES='',
        torch_num_threads=2,torch_num_interop_threads=2,CUDA_initialized=False),CPU_numeric_protocol=NUMERIC,
        all187_images_all380_raw_GT_preserved=True,all100_queries_per_image_preserved=True,typed_BN106_affine159_buffers_CPU_exact=True,
        own_native610_and_direct_head8_CPU_checkpoint_verified=True,official_COCO_maxDets=[1,10,100],official_segmentation_metrics_computed=True,
        direct_point_logits=True,native_L_added_to_point_logits=False,main_official_quality_recomputed_on_direct_mask=True,
        secondary_fixed_native_scores_diagnostic_only=True,fixed_score_empty_direct_queries_retained=True,GT_used_for_scores=False,
        historically_exposed_DEV=True,no_TEST=True,endpoint_selection=False,formal_resume_eligible=False,CPU_backbone_forward_performed=False,
        CUDA_training_or_inference_replayed=False,CPU_renderer_GPU_bitwise_claimed=False,certified_GPU_error_bound=False,
        efficacy_claimed=False,convergence_claimed=False,faithful_PointRend_reproduction_claimed=False,capacity_matched_claimed=False,compute_matched_claimed=False,
        evidence_limits=[
            'CPU verifies own C trained native610 and direct Head8 against qualified checkpoint bytes; original E24 is only architecture and typed BN anchor.',
            'All187 low-field packets and all100 query direct PointHead logits, masks and scores are independently reconstructed without native backbone forward.',
            'Nearzero tolerance is a practical consistency check, not certified GPU error. Binary disagreements and both independent CPU/GPU-export COCO12 populations are reported.',
            'Main direct official quality uses its own actual GPU support. Fixed native score diagnostic uses native support and is not quality calibrated for the direct mask.',
            'A positive own native score on an empty direct diagnostic mask is retained; no query is filtered.',
            'Best100 is a GT-conditioned reusable-query geometry diagnostic with empty-union IoU1, not AP, one-to-one recall or deployed assignment.',
            'Pixel/query chunking limits intermediate memory, not total compute; full256-feature all187 replay speed and GPU fit are not established by SOURCE.',
            'This same-carrier uniform-point adaptation is not faithful author PointRend reproduction or a capacity/compute-matched control. Fixed exposed DEV is not blind TEST.'])
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8',newline='\n')as stream: json.dump(acceptance,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=QUAL_STATUS,acceptance=reference(target)),sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True); parser.add_argument('--acceptance',required=True)
    parser.add_argument('--CPU-admission',required=True)
    parser.add_argument('--owner-execute',action='store_true'); args=parser.parse_args()
    require(args.owner_execute,'Inert SOURCE: explicit ROOT-owned CPU execution required'); run(args.spec,args.acceptance,args.CPU_admission)


if __name__=='__main__':
    main()
