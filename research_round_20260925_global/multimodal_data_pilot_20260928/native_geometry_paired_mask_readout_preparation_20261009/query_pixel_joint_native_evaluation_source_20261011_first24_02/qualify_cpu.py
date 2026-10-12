"""CPU qualification of one prelocked joint24 or own warm48 DEV endpoint.

Independent low-field rendering, B8 functional L+u, RLE/factors and COCO12;
no CPU backbone forward, native training replay or GPU error certificate.
Authoring is inert. ROOT alone may execute after the actual raw terminal gates.
"""
import argparse
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import types

PREFIX = 'query-pixel-joint-native-fixed-endpoint-evaluation'
SCHEMA = 'ROOT-' + PREFIX + '-spec-v1'
QUAL_SCHEMA = 'ROOT-' + PREFIX + '-CPU-qualification-v1'
QUAL_STATUS = 'ROOT_QUERY_PIXEL_JOINT_NATIVE_FIXED_ENDPOINT_EVALUATION_CPU_QUALIFIED'
PRODUCER_BYTES = 41278
PRODUCER_SHA = '4859b6bafe41c8f7e5e5cfab1813c3d428f4574d6a91aedaf077ff386919232d'
OFFICIAL_CPU_BYTES = 49923
OFFICIAL_CPU_SHA = '2357929fcd38ffc3020c5b8d204ee371da8da3972ff2e2e03398c8225c55e916'
ARMS = ('A', 'B')
MODES = ('native0', 'global', 'spatial_halves', 'spatial_reversed')
CONTROLS = ('A_native_official', 'A_native_zero_budget', 'B_native_official',
    'B_native_posterior_budget', 'B_posterior_official', 'B_posterior_budget')
NUMERIC = dict(class_rtol=1e-6, class_atol=1e-6, renderer_rtol=1e-4, renderer_atol=1e-4,
    quality_rtol=1e-4, quality_atol=1e-4, threshold_near_zero_rtol=1e-4, threshold_near_zero_atol=1e-4)
READOUT_SHAPES = {'feature_projection.weight': (16, 256, 1, 1), 'feature_projection.bias': (16,),
    'query_projection.weight': (16, 256), 'query_projection.bias': (16,),
    'hidden.weight': (32, 36), 'hidden.bias': (32,), 'final.weight': (1, 32), 'final.bias': (1,)}
STAGES = {
    'joint24': dict(prefix='query-pixel-joint-native-fullFIT24', initial_native_epoch=24,
        phase_epochs=24, native_epoch=48, head_epoch=24, microsteps=17784, updates=8892, GT_visits=39864),
    'warm48': dict(prefix='query-pixel-joint-native-E24warm-additional48', initial_native_epoch=48,
        phase_epochs=48, native_epoch=96, head_epoch=72, microsteps=35568, updates=17784, GT_visits=79728)}


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


def terminal(raw, phase):
    expect(raw, dict(status='ACTUAL_OWNED_TERMINAL', returncode=0, root_wait_reaped=True,
        held_handle_exit=dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0), abort=None, error=None,
        wait_errors=False, GPU_observation_errors=[], whole_GPU_used_limit_MiB=8192,
        whole_leaf_VRAM_certified=False, GPU_scope='whole GPU observed, not per-process WDDM allocation'), 'STOP before hashes/imports: raw0 required')
    require(type(raw['returncode']) is int and raw.get('original_identity') is not None and
        raw.get('numeric_birth_absence', {}).get('original_birth_absent') is True and raw.get('phase', phase) == phase,
        'STOP: actual held/reaped original birth identity differs')
    finite(raw.get('whole_GPU_observed_peak_MiB'), 'Observed whole GPU peak', 0, 8192)


def source_gate(path, spec):
    receipt = path.parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    raw = read(receipt); terminal(raw, spec['phase'])
    train = read(spec['joint_training_spec']['path']); tiny = read(train['joint_preflight_spec']['path'])
    require(set(tiny['training_receiptmap']) == set(tiny['training_phasemap']) == set(MODES), 'Original four raw gates required')
    for mode in MODES:
        terminal(read(tiny['training_receiptmap'][mode]['path']), tiny['training_phasemap'][mode])
    for row in tiny['prior_receipts']:
        require(set(row) == {'receipt', 'phase'}, 'Explicit technical prior raw required')
        terminal(read(row['receipt']['path']), row['phase'])
    terminal(read(train['joint_preflight_receipt']['path']), train['joint_preflight_phase'])
    if spec['endpoint_kind'] == 'warm48':
        terminal(read(train['first24_receipt']['path']), train['first24_phase'])
    terminal(read(spec['joint_training_receipt']['path']), spec['joint_training_phase'])
    # Current evaluation and every actual training dependency precede first SHA or SOURCE execution.
    source, reader, spec_ref = reference(Path(__file__).with_name('evaluate.py')), reference(__file__), reference(path)
    require(source['bytes'] == PRODUCER_BYTES and source['sha256'] == PRODUCER_SHA and
        spec['entry_source'] == source and spec['CPU_reader'] == reader, 'Exact final producer/current CPU reader differs')
    producer = load(source, '_joint_fixed_endpoint_provenance_ONLY')
    actual_source, actual_spec, admission, lineage = producer.source_gate(path, spec)
    require(actual_source == source and actual_spec == spec_ref and raw['admission_sha256'] == admission['sha256'] and
        producer.CONTROLS == CONTROLS and producer.NUMERIC == NUMERIC, 'Current raw/SOURCE/admission/protocol differs')
    require(spec['official_evaluation_CPU_reader'] in spec['sources'] and
        train['joint_preflight_CPU_reader'] in spec['sources'] and train['joint_preflight_report'] in spec['sources'],
        'Independent official and original teacher helper refs must be current owner-bound')
    official = spec['official_evaluation_CPU_reader']
    require(official['bytes'] == OFFICIAL_CPU_BYTES and official['sha256'] == OFFICIAL_CPU_SHA and
        Path(official['path']).resolve() == (Path(__file__).parent.parent/'spatial_position_preserving_common24_evaluation_source_20261009'/'qualify_evaluation_cpu.py').resolve(),
        'Sealed canonical official COCO12 helper SOURCE required')
    return producer, dict(source=source, reader_source=reader, spec=spec_ref, parent_receipt=reference(receipt),
        GPU_admission=admission, phase=spec['phase']), lineage


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


def report_check(spec, refs, stage):
    output = Path(spec['output_directory']); report_ref = reference(output / 'REPORT.json')
    report, start, aggregate = read_ref(report_ref), read(output / 'START.json'), read(output / 'QUALIFIED.json')
    expect(report, dict(schema=PREFIX+'-report-v1', status='QUERY_PIXEL_JOINT_NATIVE_FIXED_ENDPOINT_DEV_EXPORT_COMPLETE_NO_AP',
        source=refs['source'], spec=refs['spec'], GPU_admission=refs['GPU_admission'], phase=spec['phase'],
        endpoint_kind=spec['endpoint_kind'], effective_native_epoch=stage['native_epoch'], effective_B_readout_epoch=stage['head_epoch'],
        joint_training_CPU_qualification=spec['joint_training_CPU_qualification'], checkpoints=spec['joint_checkpoints'],
        images_per_arm=187, raw_GT_per_arm=380, queries_per_image=100, CPU_numeric_protocol=NUMERIC, error=None,
        GT_removed_from_forward=True, loader_accesses_annotations=True, matching_or_loss_executed=False,
        one_native_forward_per_image_per_arm=True, two_passive_no_grad_hooks=True, serial_arms=True,
        B_posterior_geometry='L_plus_u', B_bounded_geometry_used=False,
        original_teacher_E24_only_initializer_and_BN_reference=True, trained_native610_used_for_classes_feature_coefficient_masks=True,
        historical_cross_arm_class_or_score_equality_assumed=False, no_TEST=True, AP_computed_on_GPU=False,
        official_COCO12_pending_CPU=True, training_performed=False, optimizer_or_RNG_restored=False, endpoint_selection=False,
        formal_resume_eligible=False, CPU_renderer_GPU_bitwise_claimed=False, certified_GPU_error_bound=False,
        GPU_original_logits_saved=False, efficacy_claimed=False, compute_matched_claimed=False, conditioning_isolated_claimed=False), 'Actual complete fixed-endpoint report differs')
    require(report['cleanup']['allocated_after'] == 0 and set(report['result']) == set(ARMS) and
        list(report['controls']) != [] and set(report['controls']) == set(CONTROLS), 'Complete cleanup/result/control population differs')
    expect(start, dict(schema=PREFIX+'-start-v1', source=refs['source'], spec=refs['spec'], GPU_admission=refs['GPU_admission'],
        phase=spec['phase'], endpoint_kind=spec['endpoint_kind'], fixed_effective_native_epoch=stage['native_epoch'],
        fixed_B_readout_epoch=stage['head_epoch'], controls=list(CONTROLS), arms=list(ARMS),
        joint_training_CPU_qualification=spec['joint_training_CPU_qualification'], checkpoints=spec['joint_checkpoints'],
        GT_free_inference=True, loader_accesses_annotations=True, training_performed=False, optimizer_restored=False,
        AP_computed_on_GPU=False, no_TEST=True, endpoint_selection=False, CPU_numeric_protocol=NUMERIC), 'Actual fixed start differs')
    expect(aggregate, dict(schema=PREFIX+'-aggregate-v1', status='ROOT_QUERY_PIXEL_JOINT_NATIVE_FIXED_ENDPOINT_EVALUATION_PENDING_CPU_READER',
        source=refs['source'], spec=refs['spec'], report=report_ref, result=report['result'], controls=report['controls'],
        phase=spec['phase'], endpoint_kind=spec['endpoint_kind'], independently_CPU_qualified=False,
        AP_computed=False, formal_resume_eligible=False), 'Actual pending aggregate differs')
    merged = {name: value for arm in ARMS for name, value in report['result'][arm]['controls'].items()}
    require(merged == report['controls'], 'Actual arm/control aggregate join differs')
    return report, report_ref, reference(output/'START.json'), reference(output/'QUALIFIED.json')


def own_states(torch, spec, refs, lineage, report, anchor, policy):
    stage, q = lineage['stage'], lineage['qualification']; states, summaries = {}, {}
    for arm in ARMS:
        record = report['result'][arm]; cp = torch.load(str(bound(spec['joint_checkpoints'][arm])), map_location='cpu')
        require(set(cp) == {'schema', 'meta', 'native_state_dict', 'readout_state_dict', 'native_optimizer_state_dict',
            'readout_optimizer_state_dict', 'private_order_state', 'external_rng_state'} and cp['schema'] == stage['prefix']+'-checkpoint-v1',
            'Own complete trained native610/head container differs')
        expect(cp['meta'], dict(source=q['source'], spec=q['spec'], phase=q['phase'], arm=arm,
            teacher_checkpoint=spec['checkpointmap']['native0'], teacher_qualification=spec['qualificationmap']['native0'],
            initial_native_epoch=stage['initial_native_epoch'], completed_phase_epochs=stage['phase_epochs'], effective_native_epoch=stage['native_epoch'],
            phase_microsteps=stage['microsteps'], phase_updates=stage['updates'], phase_raw_GT_visits=stage['GT_visits'],
            no_pending_accumulation_gradients=True, formal_resume_eligible=False, auxiliary_weight=1., kappa=1., eps=1e-6), 'Own endpoint CP metadata differs')
        state, head = cp['native_state_dict'], cp['readout_state_dict']; trained = read_ref(q['result'][arm]['endpoint_proof'])
        expect(trained, dict(source=q['source'], spec=q['spec'], arm=arm, checkpoint=spec['joint_checkpoints'][arm]), 'Qualified trained endpoint proof identity differs')
        require(len(state) == len(anchor) == 610 and set(state) == set(anchor), 'Own native610 key population differs')
        for name, value in state.items():
            tensor(torch, value, anchor[name].shape, anchor[name].dtype, name)
        native_digest = state_digest(state)
        require(native_digest == trained['after_native610'] and len(policy['frozen_names']) == 106 and len(policy['BN_buffer_names']) == 159 and
            all(torch.equal(state[n], anchor[n]) for n in policy['frozen_names']+policy['BN_buffer_names']), 'Own native610/original typed BN106/159 differs')
        require(set(head) == (set(READOUT_SHAPES) if arm == 'B' else set()), 'Own external eight-tensor population differs')
        for name, value in head.items():
            tensor(torch, value, READOUT_SHAPES[name], torch.float32, name)
        require(sum(v.numel() for v in head.values()) == (9441 if arm == 'B' else 0), 'Own external scalar parameter count differs')
        head_digest = state_digest(head); require(head_digest == trained['after_readout'], 'Own qualified head proof differs')
        expect(record, dict(arm=arm, checkpoint=spec['joint_checkpoints'][arm], qualification=spec['joint_training_CPU_qualification'],
            images=187, raw_GT=380, all100_fields_saved=True, field_packets=187, full_native610_exact=True,
            readout_state_tensors=8 if arm=='B' else 0, readout_parameters=9441 if arm=='B' else 0,
            effective_native_epoch=stage['native_epoch'], effective_readout_epoch=stage['head_epoch'] if arm=='B' else 0,
            no_GPU_AP=True, no_TEST=True, GT_free_inference=True), 'Evaluation arm record differs')
        require(record['cleanup']['allocated_after'] == 0, 'Serial arm allocated cleanup differs')
        proof = read_ref(record['state_proof'])
        expect(proof, dict(schema=PREFIX+'-state-proof-v1', source=refs['source'], spec=refs['spec'], arm=arm,
            before_native610=native_digest, after_native610=native_digest, before_readout=head_digest, after_readout=head_digest,
            native610_and_readout8_CPU_identity_exact=True, params_gradNone=True, all_parameters_frozen=True, all_modules_eval=True,
            no_optimizer_or_RNG_restored=True, GT_removed_from_forward=True), 'Evaluation state proof differs')
        expect(proof['loading'], dict(checkpoint=spec['joint_checkpoints'][arm], qualification=spec['joint_training_CPU_qualification'], arm=arm,
            endpoint_kind=spec['endpoint_kind'], effective_native_epoch=stage['native_epoch'], effective_readout_epoch=stage['head_epoch'] if arm=='B' else 0,
            strict_OWN_native610_loaded=True, strict_OWN_readout8_loaded=arm=='B', trained_fullstate=native_digest, trained_readout=head_digest,
            BN_parameter_names=policy['frozen_names'], BN_buffer_names=policy['BN_buffer_names'], original_typed_BN_exact=True,
            active_native_equal_original_E24_required=False, inference_backbone_with_cp=False,
            training_backbone_with_cp=lineage['train']['backbone_with_cp'], optimizer_or_RNG_restored=False,
            checkpoint_optimizer_and_RNG_containers_deserialized=True, historical_class_or_score_equality_assumed=False), 'Actual own strict-loading observation differs')
        states[arm] = head
        summaries[arm] = dict(checkpoint=spec['joint_checkpoints'][arm], qualified_endpoint=q['result'][arm]['endpoint_proof'],
            evaluation_state_proof=record['state_proof'], native610_CPU_checkpoint_population_and_digests_exact=True,
            typed_BN106_affine159_buffers_CPU_exact=True, readout_tensors=len(head), readout_parameters=sum(v.numel() for v in head.values()),
            no_CPU_native_backbone_forward=True, strict_model_loading_GPU_SOURCE_observed=True, active_native_equal_original_E24_required=False)
        del cp, state, trained, proof
    return states, summaries


def image_population(spec, refs, record, arm, coco):
    rows = read_ref(record['image_rows']); ids = list(coco.imgs); packet_paths = set(); previous_rng = None
    require(len(ids) == len(rows) == 187 and len(coco.anns) == 380 and set(coco.cats) == {1} and
        [row['image_id'] for row in rows] == ids, 'Complete ordered DEV187/380 image population differs')
    for index, row in enumerate(rows):
        image_id = ids[index]; image = coco.imgs[image_id]; H, W = image['height'], image['width']
        require(type(H) is type(W) is int and H > 0 and W > 0 and type(image_id) is int, 'Original image geometry/ID differs')
        scale = 800. / max(H, W); actual = [int(H*scale+.5), int(W*scale+.5)]; ann_ids = [a['id'] for a in coco.imgToAnns[image_id]]
        identity = row['identity']
        expect(identity, dict(image_id=image_id, ann_ids=ann_ids, original_shape=[H, W], actual_img_shape=actual,
            batch_input_shape=[800, 800], native_input_gamma=1., gray_decode='cv2.IMREAD_GRAYSCALE', target_gray_resize='nearest'), 'Original native loader identity differs')
        require([p['annotation_id'] for p in identity['parser_rows']] == ann_ids and
            all(p['native_parser_retained'] is True for p in identity['parser_rows']), 'Raw annotation order/parser retention differs')
        require(Path(identity['image']['path']).resolve() == (Path(spec['image_root'])/image['file_name']).resolve(), 'Native original image path differs')
        bound(identity['image'])
        require(len(identity['scale_factor']) == 2, 'Native scale factor length differs')
        for value, expected in zip(identity['scale_factor'], (actual[1]/W, actual[0]/H)):
            close(value, expected, 'Native image scale', 1e-7, 1e-7)
        expect(row, dict(arm=arm, endpoint_kind=spec['endpoint_kind'], dataset_index=index, image_id=image_id,
            checkpoint=spec['joint_checkpoints'][arm], qualification=spec['joint_training_CPU_qualification'], ann_ids=ann_ids,
            annotation_count=len(ann_ids), head_seed=100000+image_id, GT_removed_from_forward=True, loader_accesses_annotations=True,
            matching_or_loss_executed=False, external_RNG_restored=True, predictions=100, passive_pixel_decoder_calls=1,
            passive_mask_embed_calls=10, native_all10_tensor_versions_unchanged=True, native_official_fusion_TorchEqual=True,
            all100_queries_retained=True, no_NMS=True, no_score_cutoff=True, GPU_original_logits_saved=False,
            historical_cross_arm_class_equality_assumed=False), 'Actual DEV image observation differs')
        queries = row['prediction_query_ids']
        require(isinstance(queries, list) and len(queries) == 100 and all(type(q) is int for q in queries) and set(queries) == set(range(100)), 'Native original query permutation differs')
        rng = row['external_RNG_before']
        require(rng == row['external_RNG_after'] and (previous_rng is None or rng == previous_rng) and isinstance(rng, list) and len(rng) == 2, 'External CPU/CUDA RNG restoration/continuity differs')
        for digest in rng:
            require(isinstance(digest['shape'], list) and len(digest['shape']) == 1 and type(digest['shape'][0]) is int and digest['shape'][0] > 0, 'RNG digest dimensions differ')
            digest_contract(digest, digest['shape'], 'uint8')
        previous_rng = row['external_RNG_after']
        fields = row['final_fields']; shapes = dict(native_masks=(1,100,200,200), classes=(1,100,2))
        if arm == 'B': shapes.update(feature=(1,256,200,200), coefficient=(1,100,256))
        require(set(fields) == set(shapes), 'Actual final field digest population differs')
        for name, shape in shapes.items(): digest_contract(fields[name], shape, 'float32')
        packet = bound(row['final_field_packet'])
        require(packet == (Path(spec['output_directory'])/arm/'FINAL_FIELDS'/('IMAGE_%04d.pth'%index)).resolve(strict=True) and
            str(packet) not in packet_paths, 'Unique canonical field packet path differs')
        packet_paths.add(str(packet))
    require(sum(row['annotation_count'] for row in rows) == 380, 'All raw GT annotations must remain')
    return ids, rows


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


def prediction_population(torch, np, spec, record, name, image_rows):
    geometry = 'L_plus_u' if name.startswith('B_posterior_') else 'native'
    expect(record, dict(images=187, raw_GT=380, predictions_count=18700, geometry=geometry, AP_computed_on_GPU=False,
        all100_queries_retained=True), 'Control metadata differs')
    predictions, factors = read_ref(record['predictions']), read_ref(record['score_factors'])
    require(isinstance(predictions, list) and isinstance(factors, list) and len(predictions) == len(factors) == 18700, 'All18700 predictions/factors required')
    mapped = {}; budget = name.endswith('budget')
    for index, (prediction, factor) in enumerate(zip(predictions, factors)):
        image = image_rows[index//100]; q = image['prediction_query_ids'][index%100]; key = (image['image_id'], q)
        require((prediction['image_id'], prediction['query_id']) == key and key not in mapped, 'Prediction native order/unique query ID differs')
        expected_keys = {'image_id', 'query_id', 'class_probability', 'quality', 'score', 'quality_kind', 'geometry_kind'}
        if budget: expected_keys.update(('predicted_intersection', 'predicted_union', 'epsilon'))
        require(set(factor) == expected_keys and (factor['image_id'], factor['query_id']) == key and
            factor['score'] == prediction['score'] and factor['geometry_kind'] == geometry and
            factor['quality_kind'] == ('posterior_intersection_union_ratio' if budget else 'official_foreground_sigmoid_mean'), 'Actual factor identity/schema differs')
        for field in ('class_probability', 'quality', 'score'):
            finite(factor[field], 'FP32 factor '+field, 0, 1)
            require(float(np.float32(factor[field])) == factor[field], 'Factor must retain exact exported FP32 value')
        product = factor['class_probability']*factor['quality']
        cpu_product = float(np.float32(factor['class_probability'])*np.float32(factor['quality']))
        rounding = (2.**-24/(1.-2.**-24))*abs(product)+2.**-126
        require(abs(cpu_product-product)<=rounding and abs(factor['score']-product)<=rounding, 'FP32 score factor arithmetic differs')
        if budget:
            require(factor['epsilon'] == 1e-6, 'Budget epsilon differs')
            finite(factor['predicted_intersection'], 'Predicted intersection', 0)
            finite(factor['predicted_union'], 'Predicted union', 0)
        mapped[key] = (prediction, factor)
    return predictions, mapped


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


def dense_block(torch, pixel_hidden, queries, native, weights, start, stop):
    """Independent factorization of concat(pixel16, query16, L,p,B,contrast)."""
    L = native; p = L.sigmoid()
    contrast = p - torch.nn.functional.avg_pool2d(p[:,None], 3, stride=1, padding=1, count_include_pad=False)[:,0]
    W = weights['hidden.weight']; q = torch.nn.functional.linear(queries[start:stop], W[:,16:32])
    hidden = pixel_hidden[None] + q[:,None,None,:]
    for j, scalar in enumerate((L, p, (L>0).to(L.dtype), contrast)):
        hidden = hidden + scalar[...,None]*W[:,32+j]
    u = torch.nn.functional.linear(torch.nn.functional.silu(hidden), weights['final.weight'], weights['final.bias'])[...,0]
    logits = L + u
    require(bool(torch.isfinite(logits).all()), 'Independent direct L+u is nonfinite')
    return logits


def quality(torch, logits, posterior, support, budget):
    binary = support.to(torch.float32); area = binary.sum()
    intersection = (binary*posterior).sum()
    union = area+((1-binary)*posterior).sum()
    value = intersection/(union+1e-6) if budget else (binary*logits.sigmoid()).sum()/(area+1e-6)
    return value, intersection, union


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


def replay_arm(torch, np, masks, coco, spec, refs, arm, record, weights, ids, image_rows):
    names = [name for name in CONTROLS if name.startswith(arm+'_')]
    require(set(record['controls']) == set(names), 'Exact per-arm controls differ')
    exports, mapped = {}, {}
    for name in names:
        exports[name], mapped[name] = prediction_population(torch, np, spec, record['controls'][name], name, image_rows)
    reconstructed = {name:[] for name in names}; geometry = {'native':[], 'L_plus_u':[]} if arm=='B' else {'native':[]}
    gpu_geometry = {kind:[] for kind in geometry}; numerical = []; equality_counts = {name:0 for name in names}
    packet_refs = []; total_flips = {name:0 for name in names}; empty = {name:0 for name in names}
    for index, row in enumerate(image_rows):
        image_id = ids[index]; identity = row['identity']; H,W = identity['original_shape']; h,w = identity['actual_img_shape']
        packet_ref = row['final_field_packet']; packet = torch.load(str(bound(packet_ref)), map_location='cpu'); packet_refs.append(packet_ref)
        require(set(packet) == {'schema','source','spec','arm','endpoint_kind','checkpoint','dataset_index','image_id','identity',
            'native_masks200','final_class_logits','feature','coefficient','field_digests','prediction_query_ids','GPU_original_logits_saved',
            'all100_fields_saved','posterior_geometry'}, 'Actual field packet key population differs')
        expect(packet, dict(schema=PREFIX+'-fields-v1', source=refs['source'], spec=refs['spec'], arm=arm, endpoint_kind=spec['endpoint_kind'],
            checkpoint=spec['joint_checkpoints'][arm], dataset_index=index, image_id=image_id, identity=identity, field_digests=row['final_fields'],
            prediction_query_ids=row['prediction_query_ids'], GPU_original_logits_saved=False, all100_fields_saved=True,
            posterior_geometry='L_plus_u' if arm=='B' else None), 'Actual field packet provenance differs')
        M = tensor(torch, packet['native_masks200'], (1,100,200,200), torch.float32, 'M200')
        C = tensor(torch, packet['final_class_logits'], (1,100,2), torch.float32, 'Classes')
        require(tensor_digest(M) == row['final_fields']['native_masks'] and tensor_digest(C) == row['final_fields']['classes'], 'Actual final tensor SHA differs')
        class_packet = row['native_class_packet']; values = torch.tensor(class_packet['values'], dtype=torch.float32)
        tensor(torch, values, (100,2), torch.float32, 'Class packet values')
        require(set(class_packet) == {'shape','dtype','sha256','values'} and tensor_digest(values) == {key:class_packet[key] for key in ('shape','dtype','sha256')} and
            torch.equal(values,C[0]), 'Class packet FP32 bytes/actual tensor differ')
        probabilities = C[0].softmax(-1)[:,0]
        projection_max = None
        if arm == 'B':
            F = tensor(torch, packet['feature'], (1,256,200,200), torch.float32, 'Feature')
            e = tensor(torch, packet['coefficient'], (1,100,256), torch.float32, 'Coefficient')
            require(tensor_digest(F) == row['final_fields']['feature'] and tensor_digest(e) == row['final_fields']['coefficient'], 'Actual F/e bytes differ')
            projected = torch.einsum('bqc,bchw->bqhw',e,F); difference = (projected-M).abs()
            projection_max = float(difference.max())
            require(torch.allclose(projected,M,rtol=NUMERIC['renderer_rtol'],atol=NUMERIC['renderer_atol']), 'Independent native eF coarse projection differs')
            pixel200 = torch.nn.functional.conv2d(F, weights['feature_projection.weight'], weights['feature_projection.bias'])
            pixel = render_original(torch,pixel200,[h,w],[H,W]).permute(1,2,0)
            queries = torch.nn.functional.linear(e[0],weights['query_projection.weight'],weights['query_projection.bias'])
            pixel_hidden = torch.nn.functional.linear(pixel,weights['hidden.weight'][:,:16],weights['hidden.bias'])
            del projected,difference,pixel200,pixel,F,e
        else:
            require(packet['feature'] is packet['coefficient'] is None, 'A must omit F/e without external head')
        anns = coco.imgToAnns[image_id]; targets = []
        for ann in anns:
            raw = np.zeros((H,W),dtype=np.uint8) if ann.get('segmentation') == [] else coco.annToMask(ann)
            require(raw.shape == (H,W) and bool(np.isin(raw,[0,1]).all()), 'All raw GT rasters must retain original geometry')
            targets.append(torch.from_numpy(raw.astype(bool)))
        best = {kind:[None]*len(anns) for kind in geometry}; gpu_best = {kind:[None]*len(anns) for kind in geometry}
        cpu_by_query = {name:{} for name in names}; flips = {name:0 for name in names}; max_class_error = max_quality_error = 0.
        for start in range(0,100,4):
            stop = min(start+4,100); L = render_original(torch,M[:,start:stop],[h,w],[H,W])
            posterior_logits = dense_block(torch,pixel_hidden,queries,L,weights,start,stop) if arm=='B' else None
            for offset, query in enumerate(range(start,stop)):
                native = L[offset]; p = native.sigmoid()
                learned = posterior_logits[offset] if arm=='B' else None; posterior = learned.sigmoid() if arm=='B' else p
                cases = {arm+'_native_official':(native,p,False),
                    'A_native_zero_budget' if arm=='A' else 'B_native_posterior_budget':(native,posterior,True)}
                if arm=='B': cases.update(B_posterior_official=(learned,posterior,False), B_posterior_budget=(learned,posterior,True))
                actual_native = mapped[arm+'_native_official'][(image_id,query)][0]
                for name,(logits,posterior_value,budget) in cases.items():
                    prediction, factor = mapped[name][(image_id,query)]
                    binary = torch.from_numpy(decode_binary(np,masks,prediction,[H,W]).astype(bool)); cpu_binary = logits>0
                    changed = binary != cpu_binary; envelope = NUMERIC['threshold_near_zero_atol']+NUMERIC['threshold_near_zero_rtol']*logits.abs()
                    require(not bool((changed & (logits.abs()>envelope)).any()), 'GPU/CPU mask disagreement outside practical nearzero envelope')
                    count = int(changed.sum()); flips[name] += count; total_flips[name] += count
                    empty[name] += int(not bool(binary.any())); equality_counts[name] += int(count==0)
                    c = factor['class_probability']; max_class_error = max(max_class_error,close(c,float(probabilities[query]),'CPU ship probability',NUMERIC['class_rtol'],NUMERIC['class_atol']))
                    actual_quality,I,U = quality(torch,logits,posterior_value,binary,budget)
                    max_quality_error = max(max_quality_error,close(factor['quality'],float(actual_quality),'CPU quality on actual GPU support',NUMERIC['quality_rtol'],NUMERIC['quality_atol']))
                    if budget:
                        close(factor['predicted_intersection'],float(I),'CPU predicted intersection',NUMERIC['quality_rtol'],NUMERIC['quality_atol'])
                        close(factor['predicted_union'],float(U),'CPU predicted union',NUMERIC['quality_rtol'],NUMERIC['quality_atol'])
                    if not bool(binary.any()): require(factor['quality']==factor['score']==0., 'Empty exported mask must retain exact zero quality/score')
                    cpu_quality,_,_ = quality(torch,logits,posterior_value,cpu_binary,budget)
                    cpu_score = probabilities[query]*cpu_quality
                    finite(float(cpu_score),'Independent CPU score',0,1)
                    cpu_by_query[name][query] = encode_prediction(np,masks,image_id,query,cpu_binary,cpu_score)
                    require(c == mapped[arm+'_native_official'][(image_id,query)][1]['class_probability'], 'Within-arm class probability changed by mask/score control')
                    kind = 'L_plus_u' if name.startswith('B_posterior_') else 'native'
                    if name == arm+'_native_official' or name == 'B_posterior_official':
                        best_geometry(torch,cpu_binary,targets,query,best[kind],'independent_CPU_threshold')
                        best_geometry(torch,binary,targets,query,gpu_best[kind],'actual_GPU_exported_RLE')
                    if kind=='native': require(prediction['segmentation'] == actual_native['segmentation'], 'Score-only native geometry was altered')
                    elif name=='B_posterior_budget': require(prediction['segmentation'] == mapped['B_posterior_official'][(image_id,query)][0]['segmentation'], 'Posterior score-only geometry changed')
                del cases,native,p,learned,posterior,binary,cpu_binary,changed,envelope
            del L,posterior_logits
        for name in names: reconstructed[name].extend(cpu_by_query[name][q] for q in row['prediction_query_ids'])
        for kind in geometry:
            for ann,cpu_result,gpu_result in zip(anns,best[kind],gpu_best[kind]):
                base = dict(image_id=image_id,annotation_id=ann['id'],category_id=ann['category_id'],iscrowd=int(ann.get('iscrowd',0)),
                    ignore=int(ann.get('ignore',0)),original_shape=[H,W])
                for destination,value in ((geometry[kind],cpu_result),(gpu_geometry[kind],gpu_result)):
                    require(value is not None,'Missing best100 raw GT')
                    value.update(FN_over_GT_area=value['FN']/max(value['GT_pixels'],1),FP_over_GT_area=value['FP']/max(value['GT_pixels'],1))
                    destination.append(dict(base,**value))
        numerical.append(dict(image_id=image_id,native_projection_maxabs=projection_max,threshold_disagreements=flips,
            max_class_probability_absolute_error=max_class_error,max_actual_support_quality_absolute_error=max_quality_error))
        if(index+1)%50==0 or index==186: print(json.dumps(dict(arm=arm,CPU_reconstructed_images=index+1)),flush=True)
        if arm=='B': del pixel_hidden,queries
        del packet,M,C,values,probabilities,targets,best,gpu_best,cpu_by_query
    require(all(len(p)==18700 for p in reconstructed.values()) and all(len(g)==380 for g in geometry.values()), 'Complete independent all100/GT population differs')
    return dict(exports=exports,reconstructed=reconstructed,CPU_geometry=geometry,GPU_geometry=gpu_geometry,field_packets=packet_refs,
        numerical=numerical,threshold_disagreements=total_flips,exact_binary_queries=equality_counts,empty_GPU_predictions=empty)


def run(spec_file, acceptance_file):
    path = Path(spec_file); target = Path(acceptance_file)
    require(path.is_absolute() and target.is_absolute() and not target.exists(), 'Absolute SPEC/fresh CPU acceptance required')
    path = path.resolve(strict=True); spec = read(path)
    expect(spec, dict(schema=SCHEMA,kind='mask2former',arms=list(ARMS),controls=list(CONTROLS),evaluation_images=187,evaluation_raw_GT=380,
        head_seed_base=100000,CPU_affinity=[0,1],allocator_limit_MiB=4096,whole_GPU_owner_limit_MiB=8192,CPU_numeric_protocol=NUMERIC,
        kappa=1.,eps=1e-6,endpoint_selection=False,historically_exposed_DEV=True,no_TEST=True), 'Prelocked six-control SPEC differs')
    require(spec['endpoint_kind'] in STAGES and type(spec['deadline_seconds']) is int and 0<spec['deadline_seconds']<=21600,
        'Explicit joint24/warm48 endpoint/resource cap required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(k)=='2' for k in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')), 'CPU CUDAempty/numeric environment2 required before imports')
    producer,refs,lineage = source_gate(path,spec); stage = STAGES[spec['endpoint_kind']]
    require(all(lineage['stage'][k]==v for k,v in stage.items()),'Independent stage contract differs')
    report,report_ref,start_ref,aggregate_ref = report_check(spec,refs,stage)
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1], 'Actual CPU01 affinity required before scientific imports')
    import torch
    require(not torch.cuda.is_initialized(), 'CPU reader entered with initialized CUDA')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    require(torch.__version__=='2.0.1+cu118' and torch.get_num_threads()==torch.get_num_interop_threads()==2, 'Bound Torch2.0.1/2+2 required')
    require(Path(inspect.getsourcefile(torch.nn.functional.interpolate)).resolve()==bound(spec['interpolation_SOURCE']).resolve(), 'Actual interpolation SOURCE differs')
    import numpy as np
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    helper = load(lineage['train']['joint_preflight_CPU_reader'],'_joint_independent_original_teacher_CPU_helper')
    tiny_report = read_ref(lineage['train']['joint_preflight_report'])
    _,_,anchor,policy,_,_ = helper.teacher_state(torch,spec,tiny_report['result'])
    states,state_results = own_states(torch,spec,refs,lineage,report,anchor,policy)
    del anchor
    official = load(spec['official_evaluation_CPU_reader'],'_joint_sealed_generic_official_COCO_ONLY')
    coco = COCO(str(bound(spec['DEV']))); results,arm_results = {},{}
    with torch.no_grad():
        for arm in ARMS:
            record = report['result'][arm]; ids,rows = image_population(spec,refs,record,arm,coco)
            replay = replay_arm(torch,np,masks,coco,spec,refs,arm,record,states[arm],ids,rows)
            for name in replay['exports']:
                exported = official.coco_metrics(coco,replay['exports'][name],ids,COCOeval)
                reconstructed = official.coco_metrics(coco,replay['reconstructed'][name],ids,COCOeval)
                differences = [cpu-gpu for cpu,gpu in zip(reconstructed['stats'],exported['stats'])]
                results[name] = dict(predictions=record['controls'][name]['predictions'],score_factors=record['controls'][name]['score_factors'],
                    GPU_export_COCO12=exported,independent_CPU_COCO12=reconstructed,CPU_minus_GPU_COCO12=differences,
                    threshold_disagreements=replay['threshold_disagreements'][name],exact_binary_queries=replay['exact_binary_queries'][name],
                    empty_GPU_predictions_retained=replay['empty_GPU_predictions'][name],all18700_queries_retained=True,
                    geometry_kind=record['controls'][name]['geometry'],CPU_scores_independently_recomputed=True,GT_used_for_scores=False)
            arm_results[arm] = dict(state_results[arm],image_rows=record['image_rows'],field_packets=replay['field_packets'],
                per_image_numerical_checks=replay['numerical'],GPU_export_best100={k:official.geometry_summary(v) for k,v in replay['GPU_geometry'].items()},
                independent_CPU_best100={k:official.geometry_summary(v) for k,v in replay['CPU_geometry'].items()},
                GPU_export_geometry_rows=replay['GPU_geometry'],independent_CPU_geometry_rows=replay['CPU_geometry'],
                lowest_actual_query_ID_wins_geometry_ties=True,both_empty_diagnostic_IoU=1.)
            del replay,rows
    require(not torch.cuda.is_initialized() and psutil.Process().cpu_affinity()==[0,1] and torch.get_num_threads()==torch.get_num_interop_threads()==2,
        'CPU execution resource scope changed')
    for ref in spec['sources']: bound(ref)
    require(reference(path)==refs['spec'] and reference(__file__)==refs['reader_source'] and reference(spec['GPU_admission_file'])==refs['GPU_admission'] and
        reference(refs['parent_receipt']['path'])==refs['parent_receipt'] and reference(report_ref['path'])==report_ref and
        reference(start_ref['path'])==start_ref and reference(aggregate_ref['path'])==aggregate_ref, 'Final provenance changed')
    acceptance = dict(schema=QUAL_SCHEMA,status=QUAL_STATUS,**refs,report=report_ref,start=start_ref,GPU_qualification=aggregate_ref,
        endpoint_kind=spec['endpoint_kind'],effective_native_epoch=stage['native_epoch'],effective_B_readout_epoch=stage['head_epoch'],
        joint_training_CPU_qualification=spec['joint_training_CPU_qualification'],checkpoints=spec['joint_checkpoints'],
        artifacts=report['result'],controls=list(CONTROLS),control_results=results,arms=arm_results,
        CPU_execution=dict(device='cpu',affinity=[0,1],CUDA_VISIBLE_DEVICES='',torch_num_threads=2,torch_num_interop_threads=2,CUDA_initialized=False),
        CPU_numeric_protocol=NUMERIC,all187_images_all380_raw_GT_preserved=True,all100_queries_per_image_preserved=True,
        typed_BN106_affine159_buffers_CPU_exact=True,own_native610_and_B_readout8_CPU_checkpoint_verified=True,
        official_COCO_maxDets=[1,10,100],official_segmentation_metrics_computed=True,
        B_posterior_geometry='L_plus_u',B_bounded_geometry_used=False,GT_used_for_scores=False,GT_free_inference_GPU_SOURCE_observed=True,
        historical_cross_arm_class_equality_assumed=False,within_B_class_packet_and_probability_identity_verified=True,
        historically_exposed_DEV=True,no_TEST=True,endpoint_selection=False,formal_resume_eligible=False,
        CPU_backbone_forward_performed=False,CUDA_training_or_inference_replayed=False,CPU_renderer_GPU_bitwise_claimed=False,
        certified_GPU_error_bound=False,efficacy_claimed=False,convergence_claimed=False,compute_matched_claimed=False,
        evidence_limits=[
            'Independent CPU verifies own trained native610/B8 bytes and original typed BN, not the native training trajectory.',
            'CPU independently renders saved low fields and evaluates direct B L+u and six fixed geometry/score controls; no native backbone forward is repeated.',
            'The prelocked nearzero envelope is a practical consistency check, not a certified CUDA rounding bound. Binary disagreements and both COCO12 populations are reported.',
            'Quality checks use the actual GPU exported support; CPU COCO uses its own thresholds and recomputed probabilities and scores.',
            'A and B have independently trained native weights and classes. Class equality is enforced only among controls within the same arm.',
            'Best100 is GT-conditioned, reusable-query diagnostic geometry with empty-union IoU1; it is not deployable assignment, AP or one-to-one recall.',
            'Original image preprocessing, GT exclusion, official GPU fusion equality, passive hooks and frozen eval state remain bound SOURCE observations; CPU does not rerun them.',
            'Fixed endpoints and exposed DEV provide no blind-test, superiority, convergence or novel-method claim.'])
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(acceptance,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=QUAL_STATUS,acceptance=reference(target)),sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True); parser.add_argument('--acceptance',required=True); parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args(); require(args.owner_execute,'Inert SOURCE: explicit ROOT-owned CPU execution required')
    run(args.spec,args.acceptance)


if __name__=='__main__':
    main()
