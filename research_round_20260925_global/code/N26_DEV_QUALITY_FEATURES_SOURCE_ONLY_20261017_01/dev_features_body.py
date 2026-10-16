"""UNBOUND CPU-only, GT-free DEV features from fixed qualified R masks and e0.

This module imports neither Torch nor NumPy at module scope. The Root-owned
entry injects both libraries and owns scientific execution. CPU torch.load
necessarily loads the complete old bundle, including unrelated F/L tensors;
only e0 and identity metadata are consumed, and no old tensor is copied out.
"""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def no_cuda(torch):
    need(not torch.cuda.is_initialized(), 'DEV feature capture must never initialize CUDA')


def read(c, ref):
    return json.loads(c.closed(ref).read_bytes())


def load_quality(c, ref):
    module_spec = importlib.util.spec_from_file_location(
        'N26_DEV_FIXEDMASK_QUALITY_FEATURE_RULE', str(c.closed(ref)))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    need(callable(module.make_mask_features), 'Bound predicted-mask feature function required')
    return module


def tensor_summary(torch, value):
    need(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
         and value.dtype == torch.float32 and not value.requires_grad,
         'Only detached CPU FP32 quality inputs may be summarized')
    raw = value.detach().contiguous().numpy().tobytes(order='C')
    return dict(shape=list(value.shape), dtype='torch.float32', device='cpu',
                bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _population(prediction_context, bundle_context):
    """Validate metadata only, retaining original non-natural export ordinals."""
    need(prediction_context['schema'] == 'ROOT-N25-qualified-fixedmask-score-context-v1'
         and bundle_context['schema'] == 'ROOT-N23-qualified-cache-DEV-context-v1',
         'Exact accepted prediction/bundle context schemas')
    ids = prediction_context['image_ids']
    rows = bundle_context['rows']
    order = prediction_context['prediction_order']
    need(len(ids) == len(set(ids)) == len(rows) == 187
         and ids == bundle_context['image_ids'], 'Original complete DEV187 image population')
    need(len(order) == 18700 and all(isinstance(pair, (list, tuple)) and len(pair) == 2
                                   for pair in order), 'Original18700 prediction identities')
    order = [tuple(pair) for pair in order]
    need(len(set(order)) == 18700, 'Original prediction identities must be unique')
    reconstructed = []
    for index, row in enumerate(rows):
        qids = row['qids']
        need(type(row['dataset_index']) is int and row['dataset_index'] == index
             and row['image_id'] == ids[index]
             and len(qids) == 100 and all(type(q) is int for q in qids)
             and sorted(qids) == list(range(100)), 'Canonical image and natural100 query metadata')
        shape = row['original_shape']
        need(len(shape) == 2 and all(type(v) is int and v > 0 for v in shape),
             'Positive original-grid height and width')
        reconstructed.extend((row['image_id'], q) for q in qids)
    need(reconstructed == order, 'Bundle export identities must preserve exact prediction ordinals')
    # File identity may differ for equivalent qualified exports. Complete
    # prediction-dictionary equivalence is checked after both direct refs load.
    return ids, rows, order


def run(torch, np, spec, c, pre):
    """Root-only numerical entry; generate features without targets, scores or AP."""
    no_cuda(torch)
    need(spec['operation'] == 'CPU_DEV_QUALITY_FEATURES'
         and spec['dataset_indices'] == list(range(187))
         and pre['no_new_training'] is True
         and pre['DEV_images'] == 187 and pre['DEV_queries'] == 18700,
         'Locked CPU-only full DEV feature population, independent of FIT counts')
    need(spec['allowed_cpus'] == [0, 1] and spec['thread_limit'] == 2
         and spec['cuda_visible_devices'] == ''
         and spec['maximum_process_RSS_bytes'] == 16 * 1024**3
         and spec['minimum_available_RAM_bytes'] == 16 * 1024**3
         and torch.get_num_threads() <= 2, 'Root-controlled CPU01/two-thread/16GiB resource scope')
    c.check()
    ctx = read(c, spec['context'])
    need(ctx['schema'] == 'ROOT-N26-quality-input-context-v1', 'Exact N26 input context')
    prediction_ref = ctx['DEV_prediction_context']
    bundle_context_ref = ctx['DEV_bundle_context']
    prediction_context = read(c, prediction_ref)
    bundle_context = read(c, bundle_context_ref)
    ids, rows, order = _population(prediction_context, bundle_context)
    prediction_ref_R = prediction_context['predictions']['R']
    predictions = read(c, prediction_ref_R)
    need(len(predictions) == 18700
         and [(v['image_id'], v['query_id']) for v in predictions] == order
         and all(v['category_id'] == 1 for v in predictions),
         'Qualified R predictions retain complete original query ordinals')
    calibration_R_ref = bundle_context['R_predictions']
    calibration_R = read(c, calibration_R_ref)
    need(len(calibration_R) == 18700, 'Original bundle-context R has all18700 candidates')
    calibration_by_identity = {(v['image_id'], v['query_id']):v for v in calibration_R}
    need(len(calibration_by_identity) == 18700 and set(calibration_by_identity) == set(order)
         and all(value == calibration_by_identity[(value['image_id'], value['query_id'])]
                 for value in predictions),
         'Both qualified R exports must have identical complete dictionaries by image/query')
    # N25 remains authoritative for output ordinals. Equality includes original
    # S0, class, RLE and every other dictionary field, not only mask geometry.
    del calibration_R, calibration_by_identity
    c.check()
    quality = load_quality(c, spec['quality_source'])
    from pycocotools import mask as masks
    no_cuda(torch)
    output = Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir()
         and not output.exists() and not output.is_symlink(), 'Fresh owned DEV output required')
    output.mkdir()
    features = torch.empty((18700, 1040), dtype=torch.float32, device='cpu')
    records, image_records = [], []
    seen = set()
    empty_total = 0
    ordinal = 0
    with torch.no_grad():
        for index, row in enumerate(rows):
            c.check()
            no_cuda(torch)
            image_id = row['image_id']
            bundle_ref = row['bundle']
            packet = torch.load(str(c.closed(bundle_ref)), map_location='cpu')
            need(packet['schema'] == 'N16-actual-GPU-image-capture-bundle-v1'
                 and packet['binding'] == row['capture_binding']
                 and packet['dataset_index'] == index and packet['image_id'] == image_id,
                 'Actual original bundle identity and binding')
            shared = packet['members'][row['shared_locator']]
            need(shared['schema'] == 'N16-actual-GPU-image-shared-inputs-v1'
                 and shared['actual_GPU_F_captured'] is True
                 and shared['CPU_F_rerendered'] is False
                 and shared['dataset_index'] == index and shared['image_id'] == image_id
                 and shared['natural_query_ids'] == list(range(100))
                 and shared['native_export_query_ids'] == row['qids'],
                 'Qualified actual shared-input native100 identity')
            embedding = shared['e0']
            need(isinstance(embedding, torch.Tensor) and embedding.device.type == 'cpu'
                 and embedding.dtype == torch.float32 and tuple(embedding.shape) == (100, 256)
                 and not embedding.requires_grad and bool(torch.isfinite(embedding).all()),
                 'Unchanged finite original CPU FP32 e0[query]')
            embedding_before = tensor_summary(torch, embedding)
            # Whole-bundle deserialization was unavoidable. Do not inspect F/L
            # values, call a model, read a GT/matcher file, or retain other members.
            del packet, shared
            gc.collect()
            no_cuda(torch)
            c.check()
            image_first = ordinal
            image_empty = 0
            for query in row['qids']:
                need(ordinal < 18700 and order[ordinal] == (image_id, query)
                     and ordinal not in seen, 'Exact original R prediction ordinal')
                pred = predictions[ordinal]
                rle = pred['segmentation']
                need(isinstance(rle, dict) and list(rle['size']) == row['original_shape'],
                     'Original R RLE carries exact original-grid shape')
                decoded = masks.decode(rle)
                need(isinstance(decoded, np.ndarray) and decoded.ndim == 2
                     and list(decoded.shape) == row['original_shape']
                     and decoded.dtype == np.uint8
                     and bool(((decoded == 0) | (decoded == 1)).all()),
                     'Original R RLE decodes to one binary original-grid mask')
                binary = decoded.astype(np.bool_, copy=False).copy(order='C')
                predicted_mask = torch.from_numpy(binary)
                mask_values = quality.make_mask_features(torch, predicted_mask)
                need(isinstance(mask_values, torch.Tensor) and mask_values.device.type == 'cpu'
                     and mask_values.dtype == torch.float32 and tuple(mask_values.shape) == (784,)
                     and not mask_values.requires_grad and bool(torch.isfinite(mask_values).all()),
                     'Bound predicted-mask-only feature is finite CPU FP32[784]')
                ys, xs = np.nonzero(binary)
                area = int(len(ys))
                empty = area == 0
                bbox = None if empty else [int(ys.min()), int(ys.max())+1,
                                           int(xs.min()), int(xs.max())+1]
                need(not empty or bool((mask_values == 0).all()), 'Empty mask maps to exact zero784')
                features[ordinal, :256].copy_(embedding[query])
                features[ordinal, 256:].copy_(mask_values)
                need(torch.equal(features[ordinal, :256], embedding[query]),
                     'Embedding is copied without normalization or arithmetic')
                rle_bytes = json.dumps(rle, sort_keys=True, separators=(',', ':'),
                                       ensure_ascii=True, allow_nan=False).encode('ascii')
                records.append(dict(original_prediction_ordinal=ordinal,
                    image_id=image_id, query_id=query, dataset_index=index,
                    original_shape=list(row['original_shape']), mask_area=area, empty_mask=empty,
                    crop_y0_y1_x0_x1=bbox, crop_padding=0,
                    predicted_mask_sha256=hashlib.sha256(binary.tobytes(order='C')).hexdigest(),
                    original_RLE_dictionary_sha256=hashlib.sha256(rle_bytes).hexdigest(),
                    e0=tensor_summary(torch, embedding[query]),
                    mask_feature=tensor_summary(torch, mask_values)))
                seen.add(ordinal)
                ordinal += 1
                image_empty += int(empty)
                del pred, rle, decoded, binary, predicted_mask, mask_values, ys, xs
            need(tensor_summary(torch, embedding) == embedding_before,
                 'Original e0 tensor remains bitwise unchanged after every image')
            image_records.append(dict(dataset_index=index, image_id=image_id,
                original_shape=list(row['original_shape']), bundle_reference=bundle_ref,
                shared_locator=row['shared_locator'], original_query_ids=list(row['qids']),
                original_ordinal_range=[image_first, ordinal], e0=embedding_before,
                embedding_unchanged=True, empty_masks=image_empty))
            empty_total += image_empty
            del embedding
            gc.collect()
            c.check()
            no_cuda(torch)
            print(json.dumps(dict(stage='N26_CPU_DEV_QUALITY_FEATURES', images=index+1,
                                  queries=ordinal, CUDA_initialized=False)), flush=True)
    need(ordinal == len(records) == len(seen) == 18700 and len(image_records) == 187
         and features.device.type == 'cpu' and features.dtype == torch.float32
         and not features.requires_grad and bool(torch.isfinite(features).all()),
         'Exactly18700 original-order detached CPU FP32 quality features')
    features_summary = tensor_summary(torch, features)
    features_path = output / 'DEV_QUALITY_INPUTS.pth'
    payload = dict(schema='N26-fixed-R-DEV-quality-inputs-v1', features=features,
        prediction_order=[list(pair) for pair in order], image_ids=list(ids),
        original_R_prediction_reference=prediction_ref_R,
        calibration_R_prediction_reference=calibration_R_ref,
        full_R_prediction_dictionary_calibration=True,
        features_layout=['original_FP32_e0_256', 'predicted_mask_bbox_bilinear28x28_784'],
        quality_source=spec['quality_source'], inputs_use_GT=False, includes_targets=False,
        includes_scores=False, includes_old_F_or_L=False)
    with features_path.open('xb') as stream:
        torch.save(payload, stream)
    features_ref = c.ref(features_path)
    c.check()
    no_cuda(torch)
    audit = dict(schema='N26-fixed-R-DEV-quality-features-audit-v1', images=187, queries=18700,
        original_R_prediction_reference=prediction_ref_R,
        calibration_R_prediction_reference=calibration_R_ref,
        full_R_prediction_dictionary_calibration=True,
        DEV_prediction_context=ctx['DEV_prediction_context'],
        DEV_bundle_context=ctx['DEV_bundle_context'], quality_source=spec['quality_source'],
        prediction_records=records, image_records=image_records, empty_masks=empty_total,
        features=features_summary, original_ordinal_preserved=True, embedding_unchanged=True,
        GT_input_used=False, targets_generated=False, matcher_used=False, COCO_used=False,
        old_bundle_F_L_deserialized_with_container=True, old_F_L_used_as_numeric_inputs=False,
        old_F_L_copied_to_output=False, full_RLE_or_GT_annotations_copied=False,
        CUDA_initialized=False, new_training=False, new_scores=False, AP=False, qualified=False)
    audit_ref = c.write_json(output / 'DEV_FEATURE_AUDIT.json', audit)
    report = dict(schema='N26-fixed-R-DEV-quality-features-report-pending-v1',
        operation=spec['operation'], images=187, queries=18700, features_ref=features_ref,
        audit_ref=audit_ref, features=features_summary, empty_masks=empty_total,
        context=spec['context'], quality_source=spec['quality_source'],
        original_R_prediction_reference=prediction_ref_R,
        calibration_R_prediction_reference=calibration_R_ref,
        full_R_prediction_dictionary_calibration=True,
        original_ordinal_preserved=True, all_natural100_queries=True, embedding_unchanged=True,
        features_are_CPU_FP32=True, GT_input_used=False, targets_generated=False,
        matcher_used=False, COCO_used=False, quality_head_constructed=False, model_forward=False,
        old_bundle_F_L_deserialized_with_container=True, old_F_L_used_as_numeric_inputs=False,
        old_F_L_copied_to_output=False, CPU=True, CUDA_initialized=False,
        new_training=False, new_scores=False, AP=False, DEV=True, TEST=False,
        status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
        qualified=False, original_actual0_self_certified=False, innovation_established=False,
        goal_complete=False)
    report_ref = c.write_json(output / 'DEV_FEATURE_REPORT.json', report)
    c.check()
    no_cuda(torch)
    return dict(report_ref=report_ref, features_ref=features_ref, audit_ref=audit_ref,
                images=187, queries=18700, feature_shape=[18700, 1040],
                full_R_prediction_dictionary_calibration=True,
                original_ordinal_preserved=True, GT_input_used=False, new_training=False,
                new_scores=False, AP=False, CUDA_initialized=False)
