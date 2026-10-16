"""UNBOUND independent CPU reconstruction of matched FIT and GT-free DEV inputs.

Only Root may execute run. IoU uses independent NumPy intersection/union counts.
Mask features are independently reextracted through the same bound interpolation
implementation; this is not an independent interpolation algorithm. Loading old
CPU bundles necessarily deserializes unrelated F/L, never used as numeric inputs.
"""
import gc
import hashlib
import json
from pathlib import Path
import types


FIT_INPUT_SCHEMA = 'N26-FIT-matched-frozen-R-query-mask-quality-inputs-v1'
FIT_REPORT_SCHEMA = 'N26-FIT-frozen-R-complete-mask-quality-capture-report-v1'
DEV_INPUT_SCHEMA = 'N26-fixed-R-DEV-quality-inputs-v1'
DEV_REPORT_SCHEMA = 'N26-fixed-R-DEV-quality-features-report-pending-v1'
MERGED_KEYS = {'FIT_features', 'FIT_targets_FP64', 'FIT_identity_order',
               'DEV_features', 'DEV_prediction_order', 'R_prediction_reference'}


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def no_cuda(torch):
    need(not torch.cuda.is_initialized(), 'Independent input verification must never initialize CUDA')


def read(c, ref):
    return json.loads(Path(c.closed(ref)).read_bytes())


def load_quality(c, ref):
    path = Path(c.closed(ref))
    source = path.read_bytes()
    need(len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'],
         'Bound quality source must retain its exact identity')
    module = types.ModuleType('N26_INDEPENDENT_INPUT_MASK_FEATURE_RULE')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    need(callable(module.make_mask_features), 'Predicted-mask-only feature function required')
    return module


def check_factory_group(c, canonical, original_refs):
    """Close actual published paths; compare content identity across factories.

    Report/audit/inputs must retain one identical complete original reference.
    Only that group's relation to the canonical implementation is path-agnostic.
    Control.closed verifies the actual direct bound file, not an arbitrary alias.
    """
    need(isinstance(original_refs, (list, tuple)) and len(original_refs) == 3,
         'Exactly report/audit/inputs original factory references required')
    for ref in [canonical] + list(original_refs):
        need(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'}
             and type(ref['path']) is str and Path(ref['path']).is_absolute()
             and type(ref['bytes']) is int and ref['bytes'] > 0
             and type(ref['sha256']) is str and len(ref['sha256']) == 64
             and all(ch in '0123456789abcdef' for ch in ref['sha256']),
             'Exact absolute strict reference for every actual factory')
    original = original_refs[0]
    need(all(ref == original for ref in original_refs),
         'Each capture group must retain the same complete original factory reference')
    c.closed(canonical)
    c.closed(original)
    need((original['bytes'], original['sha256']) == (canonical['bytes'], canonical['sha256']),
         'Original and canonical factory bytes and SHA256 must match exactly')
    return dict(original_factory=original, canonical_factory=canonical,
                full_group_ref_exact=True, bound_factory_bytes_sha_exact=True)


def tensor(torch, value, dtype, shape, label):
    need(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
         and value.dtype == dtype and tuple(value.shape) == tuple(shape)
         and not value.requires_grad and bool(torch.isfinite(value).all()), label)
    return value


def array_digest(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def tensor_digest(value):
    return array_digest(value.detach().contiguous().numpy())


def dev_summary(value):
    raw = value.detach().contiguous().numpy().tobytes(order='C')
    return dict(shape=list(value.shape), dtype='torch.float32', device='cpu',
                bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def same_bits(left, right, label):
    need(tensor_digest(left) == tensor_digest(right), 'Exact tensor byte comparison: ' + label)


def binary(np, value, shape, label):
    need(isinstance(value, np.ndarray) and value.ndim == 2
         and tuple(value.shape) == tuple(shape) and bool(np.isfinite(value).all())
         and bool(((value == 0) | (value == 1)).all()), label)
    return value.astype(np.bool_, copy=False).copy(order='C')


def mask_info(np, value):
    yy, xx = np.nonzero(value)
    area = int(len(yy))
    if not area:
        return area, None, None
    y0, y1, x0, x1 = int(yy.min()), int(yy.max())+1, int(xx.min()), int(xx.max())+1
    return area, [y0, x0, y1, x1], [y0, y1, x0, x1]


def mask_features(torch, quality, predicted):
    # The sole feature-function argument is a predicted mask, never GT/target.
    mask = torch.from_numpy(predicted.copy(order='C'))
    return tensor(torch, quality.make_mask_features(torch, mask), torch.float32,
                  (784,), 'Reextracted predicted-mask-only784 must be finite detached CPU FP32')


def fit_pair_order(torch, packet, identity):
    ann_ids = identity['ann_ids']
    need(1 <= len(ann_ids) <= 100 and len(set(ann_ids)) == len(ann_ids)
         and packet['raw_GT_ids'] == ann_ids, 'Original complete distinct true GT identities')
    assignment = tensor(torch, packet['assignment']['raw_GT_index_by_query'], torch.int64,
                        (100,), 'Original frozen natural100 matcher assignment')
    raw = assignment.numpy().tolist()
    need(all(-2 <= value < len(ann_ids) for value in raw), 'Original matcher index bounds')
    result = []
    for g, ann_id in enumerate(ann_ids):
        queries = [q for q, target in enumerate(raw) if target == g]
        need(len(queries) == 1, 'Every original real GT has exactly one frozen matched query')
        result.append((g, ann_id, queries[0]))
    need(sum(v >= 0 for v in raw) == len(result), 'No duplicate/extra supervised query')
    return result


def fit_verify(torch, np, spec, c, pre, ctx, quality, masks, COCO):
    fit_context_ref = ctx['FIT_context']
    fit = read(c, fit_context_ref)
    need(fit['schema'] == 'ROOT-N26-quality-input-context-v1'
         and fit['images'] == 741 and fit['GT'] == 1661
         and fit['dataset_indices'] == list(range(741))
         and len(fit['existing_packet_refs']) == len(fit['identities']) == len(fit['image_ids']) == 741,
         'Exact original complete FIT context')
    n16 = read(c, fit['N16_GPU_report'])
    need(n16['images'] == 741 and n16['raw_GT'] == 1661
         and n16['packet_refs'] == fit['existing_packet_refs'] and len(n16['rows']) == 741,
         'Original N16 full-FIT report association')
    coco = COCO(str(c.closed(fit['FIT'])))
    need(len(coco.imgs) == 741 and len(coco.anns) == 1661
         and set(coco.imgs) == set(fit['image_ids']), 'Only official complete FIT GT is loaded')
    report_refs = ctx['FIT_capture_reports']
    need(len(report_refs) == 2 and pre['dataset_shards'] == [[0, 371], [371, 741]],
         'Exactly both fixed original capture shards')
    feature_parts, target_parts, identity_order, shard_refs = [], [], [], []
    counts = dict(images=0, samples=0, matched_GT_nonempty=0, frozen_pairs_exact=0,
                  embeddings_exact=0, predicted_mask_features_exact=0, targets_FP64_exact=0,
                  R_masks_digest_exact=0, matched_GT_digest_exact=0, crops_exact=0,
                  empty_R_masks=0, zero_quality=0, quality_below_half=0, K0=0)
    seen_GT, seen_queries = set(), set()
    binding_keys = ['source', 'spec', 'GPU_admission', 'phase', 'context', 'prelock',
                    'setup_source', 'quality_source', 'teacher_checkpoint', 'fixed_PR_checkpoint']
    with torch.no_grad():
        for shard_index, report_ref in enumerate(report_refs):
            c.check()
            report = read(c, report_ref)
            start, stop = pre['dataset_shards'][shard_index]
            indices = list(range(start, stop))
            need(report['schema'] == FIT_REPORT_SCHEMA and report['dataset_indices'] == indices
                 and report['images'] == stop-start and report['full_FIT_images'] == 741
                 and report['full_FIT_GT'] == 1661 and report['context'] == fit_context_ref
                 and report['complete_Omega_R'] is True and report['PR_chunk'] == 2048
                 and report['GT_used_in_mask_features'] is False
                 and report['unmatched_zero_target_created'] is False,
                 'Original complete-R shard, target isolation and bound feature rule')
            inputs = torch.load(str(c.closed(report['inputs'])), map_location='cpu')
            audit = read(c, report['audit'])
            expected_samples = sum(len(fit['identities'][i]['ann_ids']) for i in indices)
            need(inputs['schema'] == FIT_INPUT_SCHEMA
                 and audit['schema'] == 'N26-complete-R-mask-matched-quality-audit-v1'
                 and inputs['dataset_indices'] == audit['dataset_indices'] == indices
                 and inputs['matched_only'] is True and inputs['unmatched_zero_target_created'] is False
                 and audit['GT_only_in_targets'] is True
                 and report['samples'] == len(inputs['records']) == len(audit['records']) == expected_samples
                 and len(audit['images']) == len(indices), 'Complete retained matched shard records')
            need(all(inputs[k] == audit[k] == report[k] for k in binding_keys),
                 'Capture report/audit/small-payload direct binding metadata')
            factory_proof = check_factory_group(c, spec['quality_source'],
                [report['quality_source'], audit['quality_source'], inputs['quality_source']])
            X = tensor(torch, inputs['features'], torch.float32, (expected_samples, 1040),
                       'Captured FIT CPU FP32 features')
            y = tensor(torch, inputs['targets_FP64'], torch.float64, (expected_samples,),
                       'Captured FIT CPU FP64 targets')
            need(bool(((y >= 0) & (y <= 1)).all()), 'Matched quality targets remain in [0,1]')
            cursor = 0
            for local_index, index in enumerate(indices):
                c.check()
                no_cuda(torch)
                identity = fit['identities'][index]
                image_id = fit['image_ids'][index]
                shape = identity['original_shape']
                packet_ref = fit['existing_packet_refs'][index]
                packet = torch.load(str(c.closed(packet_ref)), map_location='cpu')
                need(packet['schema'] == 'N16-all100-FIT-action-sampled-cache-body-v1'
                     and packet['identity'] == n16['rows'][index]['identity'] == identity
                     and identity['image_id'] == image_id
                     and n16['rows'][index]['packet'] == packet_ref
                     and all(packet[k] == n16[k] for k in ('source', 'spec', 'GPU_admission', 'phase')),
                     'Original N16 packet identity/source and canonical image join')
                pairs = fit_pair_order(torch, packet, identity)
                embedding = tensor(torch, packet['native_e0'], torch.float32, (1, 100, 256),
                                   'Original frozen all100 native_e0')
                embedding_before = tensor_digest(embedding)
                image_audit = audit['images'][local_index]
                need(image_audit['dataset_index'] == index and image_audit['image_id'] == image_id
                     and image_audit['native_packet'] == packet_ref
                     and image_audit['samples'] == len(pairs)
                     and image_audit['matched_GT_annotation_ids'] == identity['ann_ids']
                     and set(coco.getAnnIds(imgIds=[image_id])) == set(identity['ann_ids']),
                     'Exact original image-level audit and all official FIT annotations')
                for g, ann_id, query in pairs:
                    record = audit['records'][cursor]
                    need(inputs['records'][cursor] == {k:v for k,v in record.items() if k != 'R_mask_RLE'}
                         and record['identity'] == identity and record['dataset_index'] == index
                         and record['image_id'] == image_id and record['GT_index'] == g
                         and record['GT_annotation_id'] == ann_id and record['query_id'] == query
                         and record['original_shape'] == shape
                         and record['matching_loss_seed'] == packet['matching_loss_seed']
                         and record['complete_Omega_not_sampled_PR'] is True
                         and record['GT_used_in_mask_features'] is False,
                         'One retained sample exactly equals the frozen true matcher pairing')
                    need((image_id, ann_id) not in seen_GT and (image_id, query) not in seen_queries,
                         'No duplicate true GT or query in FIT supervision')
                    rle = record['R_mask_RLE']
                    need(list(rle['size']) == shape, 'Captured R RLE original shape')
                    predicted = binary(np, masks.decode(rle), shape, 'Complete original-grid R binary mask')
                    annotation = coco.anns[ann_id]
                    need(annotation['image_id'] == image_id, 'Matched annotation belongs to this image')
                    GT = binary(np, coco.annToMask(annotation), shape, 'True matched original-grid FIT GT')
                    need(bool(GT.any()), 'Every actual matched FIT GT must be nonempty')
                    area, fit_crop, _ = mask_info(np, predicted)
                    need(array_digest(predicted) == record['R_mask_digest']
                         and array_digest(GT) == record['matched_GT_digest']
                         and tensor_digest(embedding[0, query]) == record['e0_digest']
                         and area == record['R_mask_area'] and record['empty'] == (area == 0)
                         and fit_crop == record['crop_yxyx_halfopen']
                         and record['RLE_roundtrip_exact'] is True, 'Independent mask/GT/e0/crop audit reconstruction')
                    # Independently derived matched IoU, with no max-over-GT or
                    # call to the feature module's matched_mask_iou function.
                    intersection = int(np.count_nonzero(predicted & GT))
                    union = int(np.count_nonzero(predicted | GT))
                    need(union > 0, 'Nonempty matched GT gives strictly positive true union')
                    target = np.float64(intersection) / np.float64(union)
                    target_bytes = np.asarray(target, dtype=np.float64).tobytes()
                    need(np.isfinite(target) and 0 <= target <= 1
                         and y[cursor].numpy().tobytes() == target_bytes
                         and np.asarray(record['quality_target_FP64'], dtype=np.float64).tobytes() == target_bytes,
                         'Independent full-grid matched FP64 IoU equals stored target and audit bits')
                    mask_values = mask_features(torch, quality, predicted)
                    same_bits(X[cursor, :256], embedding[0, query], 'FIT original e0')
                    same_bits(X[cursor, 256:], mask_values, 'FIT reextracted predicted-mask784')
                    need(area != 0 or bool((mask_values == 0).all()), 'Empty FIT mask retains all-zero784')
                    K = int(packet['K_counts_all100'][query])
                    need(K >= 0 and record['Omega_points'] == K and record['PR_point_chunk'] == 2048
                         and record['PR_chunk_calls'] == (K+2047)//2048
                         and record['last_PR_chunk_points'] == ((K % 2048 or 2048) if K else 0),
                         'Original K0 and exact2048/actual-tail metadata preserved')
                    identity_order.append(dict(dataset_index=index, image_id=image_id, GT_index=g,
                                               GT_annotation_id=ann_id, query_id=query))
                    seen_GT.add((image_id, ann_id)); seen_queries.add((image_id, query))
                    for key in ('samples', 'matched_GT_nonempty', 'frozen_pairs_exact', 'embeddings_exact',
                                'predicted_mask_features_exact', 'targets_FP64_exact', 'R_masks_digest_exact',
                                'matched_GT_digest_exact', 'crops_exact'):
                        counts[key] += 1
                    counts['empty_R_masks'] += int(area == 0)
                    counts['zero_quality'] += int(target == 0)
                    counts['quality_below_half'] += int(target < .5)
                    counts['K0'] += int(K == 0)
                    cursor += 1
                    del record, rle, predicted, annotation, GT, mask_values
                need(tensor_digest(embedding) == embedding_before, 'Original FIT e0 remains bitwise unchanged')
                counts['images'] += 1
                del packet, embedding
                gc.collect(); c.check(); no_cuda(torch)
                print(json.dumps(dict(stage='N26_VERIFY_FIT', images=counts['images'],
                                      samples=counts['samples'])), flush=True)
            need(cursor == expected_samples, 'Every shard sample verified exactly once')
            feature_parts.append(X); target_parts.append(y)
            shard_refs.append(dict(report=report_ref, inputs=report['inputs'], audit=report['audit'],
                                   quality_factory=factory_proof))
            del inputs, audit, report, X, y
            gc.collect(); c.check()
    need(counts['images'] == 741 and counts['samples'] == len(identity_order) == 1661
         and len(seen_GT) == len(seen_queries) == 1661, 'Complete FIT741/1661 independent reconstruction')
    return torch.cat(feature_parts, dim=0).contiguous(), torch.cat(target_parts, dim=0).contiguous(), \
        identity_order, counts, dict(context=fit_context_ref, annotation=fit['FIT'],
            N16_report=fit['N16_GPU_report'], old_packet_refs=fit['existing_packet_refs'], shards=shard_refs)


def dev_verify(torch, np, spec, c, ctx, quality, masks):
    prediction_context = read(c, ctx['DEV_prediction_context'])
    bundle_context = read(c, ctx['DEV_bundle_context'])
    report = read(c, ctx['DEV_feature_report'])
    need(prediction_context['schema'] == 'ROOT-N25-qualified-fixedmask-score-context-v1'
         and bundle_context['schema'] == 'ROOT-N23-qualified-cache-DEV-context-v1'
         and report['schema'] == DEV_REPORT_SCHEMA and report['images'] == 187
         and report['queries'] == 18700,
         'Original complete DEV feature and identity context schemas')
    R_ref = prediction_context['predictions']['R']
    calibration_R_ref = bundle_context['R_predictions']
    predictions = read(c, R_ref)
    old_R = read(c, calibration_R_ref)
    order = [list(pair) for pair in prediction_context['prediction_order']]
    ids, rows = prediction_context['image_ids'], bundle_context['rows']
    need(len(ids) == len(set(ids)) == len(rows) == 187 and ids == bundle_context['image_ids']
         and len(order) == 18700 and len({tuple(pair) for pair in order}) == 18700
         and len(predictions) == len(old_R) == 18700
         and [[v['image_id'], v['query_id']] for v in predictions] == order,
         'Original18700 R query ordinals and complete image population')
    old_by = {(v['image_id'], v['query_id']):v for v in old_R}
    need(len(old_by) == 18700 and set(old_by) == {tuple(pair) for pair in order}
         and all(value == old_by[(value['image_id'], value['query_id'])] for value in predictions),
         'Both qualified R exports have identical complete dictionaries by image/query')
    del old_R, old_by
    inputs = torch.load(str(c.closed(report['features_ref'])), map_location='cpu')
    audit = read(c, report['audit_ref'])
    need(inputs['schema'] == DEV_INPUT_SCHEMA
         and audit['schema'] == 'N26-fixed-R-DEV-quality-features-audit-v1'
         and inputs['prediction_order'] == order and inputs['image_ids'] == ids
         and inputs['inputs_use_GT'] is False and inputs['includes_targets'] is False
         and inputs['includes_scores'] is False and inputs['includes_old_F_or_L'] is False
         and inputs['quality_source'] == audit['quality_source'] == report['quality_source']
         and audit['DEV_prediction_context'] == ctx['DEV_prediction_context']
         and audit['DEV_bundle_context'] == ctx['DEV_bundle_context']
         and report['original_R_prediction_reference'] == audit['original_R_prediction_reference']
             == inputs['original_R_prediction_reference'] == R_ref
         and report['calibration_R_prediction_reference'] == audit['calibration_R_prediction_reference']
             == inputs['calibration_R_prediction_reference'] == calibration_R_ref
         and all(value['full_R_prediction_dictionary_calibration'] is True
                 for value in (report, audit, inputs))
         and len(audit['prediction_records']) == 18700 and len(audit['image_records']) == 187,
         'DEV GT-free small payload, audit and complete R dictionary calibration')
    factory_proof = check_factory_group(c, spec['quality_source'],
        [report['quality_source'], audit['quality_source'], inputs['quality_source']])
    X = tensor(torch, inputs['features'], torch.float32, (18700, 1040), 'Original DEV feature matrix')
    need(dev_summary(X) == report['features'] == audit['features'], 'Original DEV full feature summary exact')
    counts = dict(images=0, queries=0, embeddings_exact=0, predicted_mask_features_exact=0,
                  R_masks_digest_exact=0, crops_exact=0, empty_R_masks=0)
    ordinal = 0
    with torch.no_grad():
        for index, row in enumerate(rows):
            c.check(); no_cuda(torch)
            image_id = ids[index]
            qids, shape = row['qids'], row['original_shape']
            need(row['dataset_index'] == index and row['image_id'] == image_id
                 and len(qids) == 100 and all(type(q) is int for q in qids)
                 and sorted(qids) == list(range(100))
                 and order[ordinal:ordinal+100] == [[image_id, q] for q in qids],
                 'Original bundle image/export order and natural100 identities')
            packet = torch.load(str(c.closed(row['bundle'])), map_location='cpu')
            need(packet['schema'] == 'N16-actual-GPU-image-capture-bundle-v1'
                 and packet['binding'] == row['capture_binding'] and packet['dataset_index'] == index
                 and packet['image_id'] == image_id, 'Original DEV CPU bundle identity/binding')
            shared = packet['members'][row['shared_locator']]
            need(shared['schema'] == 'N16-actual-GPU-image-shared-inputs-v1'
                 and shared['dataset_index'] == index and shared['image_id'] == image_id
                 and shared['actual_GPU_F_captured'] is True and shared['CPU_F_rerendered'] is False
                 and shared['natural_query_ids'] == list(range(100))
                 and shared['native_export_query_ids'] == qids, 'Actual shared native100 e0 identity')
            embedding = tensor(torch, shared['e0'], torch.float32, (100, 256), 'Original DEV e0')
            embedding_before = dev_summary(embedding)
            image_audit = audit['image_records'][index]
            need(image_audit['dataset_index'] == index and image_audit['image_id'] == image_id
                 and image_audit['original_shape'] == shape and image_audit['bundle_reference'] == row['bundle']
                 and image_audit['shared_locator'] == row['shared_locator']
                 and image_audit['original_query_ids'] == qids
                 and image_audit['original_ordinal_range'] == [ordinal, ordinal+100]
                 and image_audit['e0'] == embedding_before and image_audit['embedding_unchanged'] is True,
                 'Exact original DEV image-level embedding/ordinal audit')
            del packet, shared
            gc.collect(); c.check()
            image_empty = 0
            for query in qids:
                record = audit['prediction_records'][ordinal]
                prediction = predictions[ordinal]
                need(record['original_prediction_ordinal'] == ordinal
                     and record['image_id'] == image_id and record['query_id'] == query
                     and record['dataset_index'] == index and record['original_shape'] == shape
                     and prediction['category_id'] == 1, 'DEV sample audit exact query/ordinal join')
                rle = prediction['segmentation']
                need(list(rle['size']) == shape, 'Original DEV fixed R RLE shape')
                predicted = binary(np, masks.decode(rle), shape, 'Original fixed-R DEV mask')
                area, _, crop = mask_info(np, predicted)
                rle_bytes = json.dumps(rle, sort_keys=True, separators=(',', ':'),
                                       ensure_ascii=True, allow_nan=False).encode('ascii')
                values = mask_features(torch, quality, predicted)
                need(record['mask_area'] == area and record['empty_mask'] == (area == 0)
                     and record['crop_y0_y1_x0_x1'] == crop and record['crop_padding'] == 0
                     and record['predicted_mask_sha256'] == array_digest(predicted)['sha256']
                     and record['original_RLE_dictionary_sha256'] == hashlib.sha256(rle_bytes).hexdigest()
                     and record['e0'] == dev_summary(embedding[query])
                     and record['mask_feature'] == dev_summary(values), 'DEV independent R/crop/feature audit exact')
                same_bits(X[ordinal, :256], embedding[query], 'DEV original e0')
                same_bits(X[ordinal, 256:], values, 'DEV reextracted predicted-mask784')
                need(area != 0 or bool((values == 0).all()), 'Empty DEV mask retains exact zero784')
                for key in ('queries', 'embeddings_exact', 'predicted_mask_features_exact',
                            'R_masks_digest_exact', 'crops_exact'):
                    counts[key] += 1
                image_empty += int(area == 0)
                ordinal += 1
                del record, prediction, rle, predicted, values
            need(dev_summary(embedding) == embedding_before and image_empty == image_audit['empty_masks'],
                 'Original DEV embedding unchanged and empty counts exact')
            counts['images'] += 1; counts['empty_R_masks'] += image_empty
            del embedding
            gc.collect(); c.check(); no_cuda(torch)
            print(json.dumps(dict(stage='N26_VERIFY_DEV', images=counts['images'], queries=ordinal)), flush=True)
    need(ordinal == counts['queries'] == 18700 and counts['images'] == 187
         and counts['empty_R_masks'] == audit['empty_masks'] == report['empty_masks'],
         'Every original DEV query independently reextracted with exact empty count')
    return X, order, R_ref, counts, dict(feature_report=ctx['DEV_feature_report'],
        features=report['features_ref'], audit=report['audit_ref'],
        prediction_context=ctx['DEV_prediction_context'], bundle_context=ctx['DEV_bundle_context'],
        original_R_prediction=R_ref, calibration_R_prediction=calibration_R_ref,
        quality_factory=factory_proof,
        old_bundle_refs=[row['bundle'] for row in rows])


def run(torch, np, spec, c, pre):
    no_cuda(torch)
    need(spec['operation'] == 'CPU_VERIFY_QUALITY_INPUTS'
         and pre['no_new_training'] is True and pre['images'] == 741 and pre['GT'] == 1661
         and pre['DEV_images'] == 187 and pre['DEV_queries'] == 18700,
         'Fixed full-FIT/full-DEV independent CPU input-verification scope')
    need(spec['allowed_cpus'] == [0, 1] and spec['thread_limit'] == 2
         and spec['cuda_visible_devices'] == '' and torch.get_num_threads() <= 2
         and spec['maximum_process_RSS_bytes'] == 16 * 1024**3
         and spec['minimum_available_RAM_bytes'] == 16 * 1024**3,
         'Root-controlled CPU01/two-thread/emptyCUDA/16GiB scope')
    c.check()
    ctx = read(c, spec['context'])
    need(ctx['schema'] == 'ROOT-N26-quality-verification-context-v1', 'Exact verification context')
    quality = load_quality(c, spec['quality_source'])
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    no_cuda(torch)
    output = Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists()
         and not output.is_symlink(), 'Fresh owned verification output')
    output.mkdir()
    FIT_X, FIT_y, FIT_order, fit_counts, fit_refs = fit_verify(torch, np, spec, c, pre, ctx, quality, masks, COCO)
    c.check(); no_cuda(torch)
    DEV_X, DEV_order, R_ref, dev_counts, dev_refs = dev_verify(torch, np, spec, c, ctx, quality, masks)
    tensor(torch, FIT_X, torch.float32, (1661, 1040), 'Verified merged FIT features')
    tensor(torch, FIT_y, torch.float64, (1661,), 'Verified merged FIT FP64 targets')
    tensor(torch, DEV_X, torch.float32, (18700, 1040), 'Verified merged DEV features')
    payload = dict(FIT_features=FIT_X, FIT_targets_FP64=FIT_y, FIT_identity_order=FIT_order,
                   DEV_features=DEV_X, DEV_prediction_order=DEV_order, R_prediction_reference=R_ref)
    need(set(payload) == MERGED_KEYS, 'Exactly the six frozen training/evaluation input keys')
    merged_path = output / 'MERGED_QUALITY_INPUTS.pth'
    c.check()
    with merged_path.open('xb') as stream:
        torch.save(payload, stream)
    replay = torch.load(str(merged_path), map_location='cpu', weights_only=True)
    need(set(replay) == MERGED_KEYS and replay['FIT_identity_order'] == FIT_order
         and replay['DEV_prediction_order'] == DEV_order and replay['R_prediction_reference'] == R_ref,
         'Merged small-payload identities/references read back exactly')
    for key in ('FIT_features', 'FIT_targets_FP64', 'DEV_features'):
        same_bits(replay[key], payload[key], 'Merged saved readback ' + key)
    merged_ref = c.ref(merged_path)
    del replay
    gc.collect(); c.check(); no_cuda(torch)
    report = dict(schema='N26-independent-quality-input-verification-report-pending-v1',
        operation=spec['operation'], images=741, GT=1661, DEV_images=187, DEV_queries=18700,
        FIT_checks=fit_counts, DEV_checks=dev_counts, merged_ref=merged_ref,
        FIT_features=tensor_digest(FIT_X), FIT_targets_FP64=tensor_digest(FIT_y),
        DEV_features=tensor_digest(DEV_X), context=spec['context'], quality_source=spec['quality_source'],
        input_references=dict(FIT=fit_refs, DEV=dev_refs),
        original_quality_factories_closed_and_byteidentical=True,
        numeric_checks_complete=True, fixed_masks_preserved=True,
        GT_feature_isolation=True, GT_only_for_FIT_matched_targets=True, DEV_GT_read=False,
        frozen_original_matcher_pairs_exact=True, max_any_GT_used=False,
        unmatched_zero_target_created=False, low_quality_empty_K0_retained=True,
        independently_counted_intersection_union_FP64=True,
        mask_features_independently_reextracted=True,
        same_bound_interpolation_implementation=True, independent_interpolation_algorithm=False,
        native_L_mask_digest_not_recomputed=True, R_model_forward_repeated=False,
        old_bundle_F_L_deserialized_with_container=True, old_F_L_used_as_numeric_inputs=False,
        old_F_L_copied_to_merged=False, no_model_or_weights_loaded=True,
        FIT_COCO_only_for_annToMask=True, DEV_COCO_used=False, AP=False, new_training=False,
        new_scores=False, CPU=True, CUDA_initialized=False, TEST=False,
        quality_training_eligible=False, qualified=False, original_actual0_self_certified=False,
        status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
        innovation_established=False, goal_complete=False)
    report_ref = c.write_json(output / 'VERIFY_REPORT.json', report)
    c.check(); no_cuda(torch)
    return dict(report_ref=report_ref, merged_ref=merged_ref, images=741, GT=1661,
                DEV_queries=18700, numeric_checks_complete=True, quality_training_eligible=False,
                AP=False, new_training=False, CUDA_initialized=False)
