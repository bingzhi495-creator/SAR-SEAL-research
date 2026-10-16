# encoding: utf-8
"""N23 UNBOUND shared-C batches; no file I/O, model import, sampling or optimization."""

N16_CACHE_BODY_SHA256 = '77de4c5445ac455672095b8c795e16046a4ed6822bae50f739f40cb4625ea5bd'
C_ENGINE_SHA256 = '8f0bd504ce7899225d868381d9b656402f8882bb19f6bd880060085fe269f9cc'
BINDING_KEYS = ('source', 'spec', 'GPU_admission', 'phase', 'teacher_checkpoint', 'fixed_PR_checkpoint')
ARMS = ('OMEGA_GATED', 'C_GATED')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def _array(np, value, dtype, shape, label, synthetic_only):
    if isinstance(value, np.ndarray):
        require(synthetic_only, label + ': ndarray is UNIT_FIXTURE-only')
        array = value
    else:
        require(getattr(getattr(value, 'device', None), 'type', None) == 'cpu'
                and getattr(value, 'requires_grad', None) is False,
                label + ': original CPU non-gradient tensor required')
        array = value.detach().numpy()
    require(array.dtype == np.dtype(dtype) and tuple(array.shape) == tuple(shape),
            label + ': exact cached dtype/shape required')
    if array.dtype.kind == 'f':
        require(bool(np.isfinite(array).all()), label + ': finite input required')
    return array


def build_shared_batches(np, packet, row, expected_cache_binding, *, synthetic_only=False):
    """Return real-matched GT-ordered batches without changing any draw or duplicate.

    Root must close the accepted N16 cache metadata/source before calling. The
    frozen writer samples only C. This thin reader never invents C membership
    from sparse L and never loads the full raster to change the original sample.
    """
    require(type(synthetic_only) is bool, 'Explicit synthetic flag required')
    require(set(expected_cache_binding) == set(BINDING_KEYS) | {'cache_body_source', 'C_engine_source'},
            'Accepted cache binding and exact writer/C-engine roles required')
    writer = expected_cache_binding['cache_body_source']
    require(writer['sha256'] == N16_CACHE_BODY_SHA256 and writer['bytes'] == 23893,
            'Qualified complete-C writer source identity required')
    engine = expected_cache_binding['C_engine_source']
    require(engine['sha256'] == C_ENGINE_SHA256 and engine['bytes'] == 7684,
            'Qualified physical C-domain engine identity required')
    require(packet['schema'] == 'N16-all100-FIT-action-sampled-cache-body-v1'
            and packet['natural_query_ids'] == list(range(100)), 'Original N16 natural100 packet required')
    require(all(packet[k] == expected_cache_binding[k] for k in BINDING_KEYS), 'Cache parent binding drift')
    require(packet['dataset_index'] == row['dataset_index'] and packet['identity'] == row['identity']
            and row['image_id'] == packet['identity']['image_id'], 'Image/ordinal identity drift')
    identity = packet['identity']; H, W = identity['original_shape']
    require(type(H) is type(W) is int and H > 0 and W > 0, 'Original positive image dimensions required')
    GT_ids = packet['raw_GT_ids']
    require(GT_ids == identity['ann_ids'] and len(set(GT_ids)) == len(GT_ids) > 0,
            'Distinct original real GT identities required')
    sample = packet['samples']; n = len(sample['query_ids'])
    require(n == row['points'] and (synthetic_only or n == 4096), 'Original draw count must remain unchanged')
    require(sample['replacement'] is True and sample['duplicates_preserved'] is True,
            'Original with-replacement draw multiplicity required')
    arr = lambda value, dtype, shape, label: _array(np, value, dtype, shape, label, synthetic_only)
    qids = arr(sample['query_ids'], 'int64', (n,), 'Natural sample query ids')
    flats = arr(sample['flat_indices'], 'int64', (n,), 'Original sample coordinates')
    inverse = arr(packet['sample_to_unique'], 'int64', (n,), 'Sample-to-unique map')
    U = len(packet['unique_flat_indices'])
    unique = arr(packet['unique_flat_indices'], 'int64', (U,), 'Original unique coordinates')
    F_unique = arr(packet['sampled_F_unique'], 'float32', (U, 256), 'Actual sampled F256')
    e0 = arr(packet['native_e0'], 'float32', (1, 100, 256), 'Actual e0')
    L = arr(sample['native_L'], 'float32', (n,), 'Actual sample native L')
    targets = arr(sample['targets'], 'bool', (n,), 'Actual true-GT sampled targets')
    weights = arr(sample['importance_weights'], 'float32', (n,), 'Original importance weights')
    strata = arr(sample['stratum_ids'], 'uint8', (n,), 'Original matched/background strata')
    require(bool(((qids >= 0) & (qids < 100)).all()) and bool(((flats >= 0) & (flats < H*W)).all())
            and bool(((inverse >= 0) & (inverse < U)).all()) and bool((weights > 0).all()),
            'Captured sample index/weight outside original bounds')
    require(bool(np.array_equal(unique[inverse], flats)) and bool(np.array_equal(unique, np.unique(flats))),
            'F coordinate/inverse map no longer represents original shared samples')
    assignment = packet['assignment']
    raw = arr(assignment['raw_GT_index_by_query'], 'int64', (100,), 'Original real-GT assignment')
    kinds = assignment['supervision_kind_by_query']
    annotation_ids = assignment['raw_GT_annotation_id_by_query']
    require(len(kinds) == len(annotation_ids) == 100, 'Complete original query assignment required')
    for q in range(100):
        g = int(raw[q])
        require(-2 <= g < len(GT_ids), 'Raw GT assignment index invalid')
        expected_kind = 'matched' if g >= 0 else ('background' if g == -1 else 'ignored')
        require(kinds[q] == expected_kind and annotation_ids[q] == (GT_ids[g] if g >= 0 else None),
                'Real/negative-proxy/ignored identity disagreement')
    require(sorted(int(v) for v in raw if v >= 0) == list(range(len(GT_ids))),
            'Original one-query-to-each-real-GT matching required')
    require(bool(np.all(raw[qids] >= -1)) and bool(np.array_equal(strata, (raw[qids] == -1).astype(np.uint8))),
            'Ignored draw or forged matched stratum rejected')
    require(len(packet['C_counts_all100']) == len(packet['K_counts_all100']) == 100
            and packet['C_counts_all100'] == row['C_counts_all100']
            and packet['K_counts_all100'] == row['K_counts_all100'], 'Original C/K metadata drift')
    require(all(packet['C_counts_all100'][int(q)] > 0 and packet['K_counts_all100'][int(q)] > 0 for q in qids),
            'Original N16 active-query K>0/C>0 sampling condition violated')
    keep = raw[qids] >= 0
    batches, unsampled = [], []
    for g, annotation_id in enumerate(GT_ids):
        q = int(np.flatnonzero(raw == g)[0])
        positions = np.flatnonzero(qids == q).astype(np.int64)
        if not len(positions):
            unsampled.append(dict(query_id=q, GT_index=g, GT_annotation_id=annotation_id,
                                  K=int(packet['K_counts_all100'][q])))
            continue
        require(q in packet['active_matched_queries'], 'Sampled match absent from original active population')
        native = L[positions].copy()
        batches.append(dict(dataset_index=packet['dataset_index'], image_id=identity['image_id'],
            query_id=q, GT_index=g, GT_annotation_id=annotation_id, n=int(len(positions)),
            sample_positions=positions, flat_indices=flats[positions].copy(),
            F=F_unique[inverse[positions]].copy(), e0=e0[0, q].copy(), L=native,
            targets=targets[positions].copy(), importance_weights=weights[positions].copy(),
            Omega_flag=np.abs(native) <= np.float32(2), C_flag=np.ones(len(positions), dtype=np.bool_)))
    require(sum(b['n'] for b in batches) == int(keep.sum()), 'Matched draw lost or repeated by adapter')
    audit = dict(original_draws=n, retained_matched_draws=int(keep.sum()),
        dropped_background_draws=int((~keep).sum()), ignored_draws=0,
        real_GT_instances=len(GT_ids), nonempty_matched_batches=len(batches),
        unsampled_real_GT=unsampled, matched_query_ids=[b['query_id'] for b in batches],
        Omega_editable_draws=sum(int(b['Omega_flag'].sum()) for b in batches),
        C_editable_draws=sum(b['n'] for b in batches),
        K0_queries_present_in_original_sample=False, no_resampling=True,
        duplicate_draws_preserved=True, negative_empty_proxy_in_main_supervision=False,
        weights_preserved_without_selecting_loss_reduction=True)
    return dict(schema='N23-shared-original-C-matched-query-batches-v1', batches=batches, audit=audit,
        candidate_membership_basis='accepted N16 writer samples only domain_indices[q]=C; no sparse-L inference',
        cache_binding=expected_cache_binding, synthetic_only=synthetic_only,
        qualified=False, training_executed=False)
