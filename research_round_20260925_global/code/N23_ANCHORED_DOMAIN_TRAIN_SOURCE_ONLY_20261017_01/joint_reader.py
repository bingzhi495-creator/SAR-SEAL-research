# encoding: utf-8
"""UNBOUND pure reader: join original LOG_PR draws to real matched C-cache draws.

No file I/O, tensor/model imports, resampling, optimization or real execution.
Root must verify both accepted cache REPORT bindings before calling this reader.
"""
import dataset_adapter

OLD_BINDING_KEYS = ('source', 'spec', 'GPU_admission', 'phase', 'teacher_checkpoint')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def build_joint_image(np, old_packet, old_row, new_packet, new_row,
                      expected_old_binding, expected_new_binding, *, synthetic_only=False):
    require(type(expected_old_binding) is dict and set(expected_old_binding) == set(OLD_BINDING_KEYS),
            'Exact original REPORT five-role binding required')
    require(all(old_packet[k] == expected_old_binding[k] for k in OLD_BINDING_KEYS if k != 'GPU_admission'),
            'Original packet source/spec/phase/teacher binding drift')
    # GPU_admission is REPORT-level in the original writer, absent in old packets.
    # Root must close that REPORT; this pure reader does not manufacture evidence.
    require(expected_old_binding['teacher_checkpoint'] == expected_new_binding['teacher_checkpoint'],
            'Same actual frozen teacher checkpoint required')
    require(old_packet['schema'] == 'N13-sameK-region-FIT-cache-packet-v1',
            'Original LOG_PR packet schema required')
    require(old_packet['dataset_index'] == old_row['dataset_index'] ==
            new_packet['dataset_index'] == new_row['dataset_index'] and
            old_packet['identity'] == old_row['identity'] == new_packet['identity'] == new_row['identity'] and
            old_row['image_id'] == old_packet['identity']['image_id'],
            'Old/new image ordinal or original identity drift')
    shared = dataset_adapter.build_shared_batches(np, new_packet, new_row,
        expected_new_binding, synthetic_only=synthetic_only)
    ann_ids = old_packet['identity']['ann_ids']
    require(isinstance(old_packet['raw_GT_digest'],dict) and
            old_packet['raw_GT_digest'] == new_packet['raw_GT_digest'],
            'Same canonical true-GT raster identity required across cached point pools')
    G = len(ann_ids)
    require(ann_ids == old_row['raw_GT_ids'] == new_packet['raw_GT_ids'] and
            old_row['raw_GT'] == G, 'Same original real-GT order required')
    arm = old_packet['arms']['LOG_PR']
    P = len(arm['native_logits'][0])
    require(P > 0 and (synthetic_only or P == 1024) and old_row['points_per_arm'] == G*P,
            'Every original LOG_PR 1024-point draw must remain intact')
    require(old_packet['replacement'] is True and old_packet['duplicates_preserved'] is True,
            'Original with-replacement points and duplicates required')
    arr = lambda value, dtype, shape, label: dataset_adapter._array(
        np, value, dtype, shape, label, synthetic_only)
    qids = arr(old_packet['query_ids'], 'int64', (G,), 'Old original raw-GT query ids')
    require(qids.tolist() == old_row['query_ids'] and len(set(qids.tolist())) == G,
            'Original GT-ordered unique query map drift')
    embeddings = arr(old_packet['query_embeddings'], 'float32', (G,256), 'Old matched query embeddings')
    natural_e0 = arr(new_packet['native_e0'], 'float32', (1,100,256), 'New complete native e0')
    require(bool(((qids >= 0) & (qids < 100)).all()), 'Original natural query bounds required')
    require(embeddings.tobytes(order='C') == natural_e0[0,qids].tobytes(order='C'),
            'Matched query embeddings must be bitexact, including signed zero')
    F = arr(arm['F_samples'], 'float32', (G,P,256), 'Old original F samples')
    L = arr(arm['native_logits'], 'float32', (G,P), 'Old original native logits')
    targets = arr(arm['targets'], 'bool', (G,P), 'Old original true-GT point targets')
    flats = arr(arm['point_indices'], 'int64', (G,P), 'Old original point indices')
    flags = arr(arm['point_in_region'], 'bool', (G,P), 'Old original LOG_PR membership')
    H,W = old_packet['identity']['original_shape']
    require(bool(((flats >= 0) & (flats < H*W)).all()), 'Original point coordinate out of bounds')
    require(bool(np.array_equal(flags, np.abs(L) <= np.float32(2))),
            'Original LOG_PR point membership must be abs(L)<=2')
    require(len(old_packet['K']) == G and old_packet['K'] == old_row['K'] and
            all(type(k) is int and 0 <= k <= H*W for k in old_packet['K']),
            'Original per-GT Omega K metadata drift')
    new_assignment = arr(new_packet['assignment']['raw_GT_index_by_query'],
        'int64', (100,), 'New complete original real-GT assignment')
    by_gt = {b['GT_annotation_id']: b for b in shared['batches']}
    batches = []
    for g, annotation_id in enumerate(ann_ids):
        q = int(qids[g])
        require(int(new_assignment[q]) == g and old_packet['K'][g] == new_packet['K_counts_all100'][q],
                'Old/new actual raw-GT query or Omega K mismatch')
        new = by_gt.get(annotation_id)
        if old_packet['K'][g] == 0:
            require(new is None and not bool(flags[g].any()),
                    'K0 retains old no-gradient points only; no fabricated new batch or Omega membership')
        else:
            require(new is not None, 'Positive-K original GT has no original new-cache draw')
            require(new['GT_index'] == g and new['query_id'] == q,
                    'New sampled batch is not the same original matched GT/query')
        old = dict(F=F[g].copy(), L=L[g].copy(), targets=targets[g].copy(),
            point_in_region=flags[g].copy(), flat_indices=flats[g].copy(),
            e0=embeddings[g].copy(), n=P)
        batches.append(dict(GT_index=g, GT_annotation_id=annotation_id, query_id=q,
            K=int(old_packet['K'][g]), old=old, new=new))
    return dict(schema='N23-original-LOG-PR-anchored-matched-C-joint-image-v1',
        raw_GT_order=list(ann_ids), batches=batches,
        audit=dict(real_GT_instances=G, old_draws=G*P,
            new_matched_draws=shared['audit']['retained_matched_draws'],
            discarded_new_background_draws=shared['audit']['dropped_background_draws'],
            K0_real_GT_instances=sum(k == 0 for k in old_packet['K']),
            K_positive_real_GT_instances=sum(k > 0 for k in old_packet['K']),
            original_duplicate_draws_preserved=True, no_resampling=True,
            negative_empty_proxy_in_main_supervision=False,
            old_GPU_admission_binding_scope='caller-verified accepted REPORT, absent in original packet'),
        synthetic_only=synthetic_only, qualified=False, training_executed=False)
