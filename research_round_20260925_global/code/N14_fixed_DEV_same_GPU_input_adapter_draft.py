"""UNBOUND / UNEXECUTED: fixed stage24 N14 DEV evaluation adapter helpers.

This is not an admitted runner. ROOT must supply the new stage24 training/CPU
qualification gate, actual own evaluation SOURCE/admission/RAW closure, a fresh
bounded capture sink, and the original closed renderer/export/COCO helpers.
Nothing imports Torch, reads a file, constructs a head, or launches at import.
The original N13 eight-tensor endpoint gate must NOT qualify the memory head.
"""

ARMS = ('BASE_CONTINUATION', 'MEMORY_ADDON')
PARAMETERS = dict(BASE_CONTINUATION=40497, MEMORY_ADDON=42698)
BASE_NAMES = ('hidden1.weight', 'hidden1.bias', 'hidden2.weight', 'hidden2.bias',
              'hidden3.weight', 'hidden3.bias', 'final.weight', 'final.bias')
BASE_SHAPES = ((91, 257), (91,), (91, 92), (91,),
               (91, 92), (91,), (1, 92), (1,))
MEMORY_SHAPES = dict(memory_keys=(8, 8), memory_values=(8, 8),
                     **{'query.weight': (8, 257), 'query.bias': (8,),
                        'output.weight': (1, 8), 'output.bias': (1,)})
STAGE24_UPDATES = 1661 * 24
POINT_CHUNK = 2048
GATE_SCHEMA = 'ROOT-N14-stage24-training-CPU-gate-adapter-v1'
CAPTURE_SCHEMA = 'N14-fixed-DEV-actual-GPU-point-input-capture-v1'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def is_reference(value):
    # Syntax only. The ROOT gate/bound() must verify actual closure/hash/owner.
    return (isinstance(value, dict) and set(value) == {'path', 'bytes', 'sha256'}
            and isinstance(value['path'], str) and type(value['bytes']) is int
            and value['bytes'] >= 0 and isinstance(value['sha256'], str)
            and len(value['sha256']) == 64)


def stage24_adapter_contract():
    """NEW adapter output contract; these are NOT claimed actual CPU-Q keys.

    The adapter must resolve actual closed stage24 Q/spec/report keys itself.
    Passing booleans without validating the underlying actual evidence is not
    an implementation of this contract. Separate training CPU numeric protocol
    and future DEV same-input CPU protocol remain independently bound.
    """
    return dict(schema=GATE_SCHEMA, mandatory_proofs=[
        'all_normal_and_failed_actual_RAW_terminal_before_first_SHA',
        'own_evaluation_full_SOURCE_and_admission_bound',
        'stage24_GPU_source_spec_admission_RAW_report_cp_traces_closed',
        'stage24_independent_CPU_source_spec_admission_RAW_Q_details_closed',
        'actual_GPU_and_CPU_owner_ref_joins_exact',
        'same_two_checkpoint_refs_in_GPU_report_CPU_spec_and_CPU_Q',
        'all_14_or_8_parameter_state_and_AdamW_steps_exact39864',
        'complete_stage24_history_orders_activity_and_three_trace_leaves',
        'original_E24_parent_and_original_first_stage_proofs_retained',
        'original_native_DEV_packets_teacher_scores_selector_COCO_closed',
        'evaluation_at_prelocked_stage24_without_best_DEV_selection'],
        mandatory_refs=['GPU_source', 'GPU_spec', 'GPU_admission', 'GPU_RAW',
                        'GPU_report_ref', 'CPU_source', 'CPU_spec',
                        'CPU_admission', 'CPU_RAW', 'CPU_qualification_ref'],
        checkpoint_arms=list(ARMS), completed_stage=24, updates=STAGE24_UPDATES)


def check_recipe(spec):
    """Only the invariant portion of a new ROOT recipe, not a SOURCE gate."""
    require(spec['original_parent_epochs'] == 24 and
            spec['fixed_stage_epochs'] == spec['stage_epochs_completed'] == 24,
            'Evaluate the complete prelocked new stage24 after the E24 parent')
    require(spec['selector_mode'] == 'reuse' and spec['point_chunk'] == POINT_CHUNK,
            'Reuse the qualified exact GPU selector; retain head chunk2048')
    require(spec['controls'] == ['native_official'] + list(ARMS),
            'Native plus both N14 arms are required')
    require(spec['images'] == 187 and spec['raw_GT'] == 380 and
            spec['queries_per_image'] == 100 and spec['predictions_per_control'] == 18700,
            'Historical full DEV187/GT380/all100 population required')
    require(spec['support_formula'] == 'abs(actual_original_GPU_native_L)<=2' and
            spec['fixed_native_scores_class_order'] is True and
            spec['best_DEV_selection'] is False and spec['no_TEST'] is True,
            'Original oldLOG support and natural score/order protocol required')


def load_stage24_heads(torch, spec, root_stage24_gate, bound,
                       base_factory, memory_factory):
    """CPU-only load after the REQUIRED new ROOT gate, before CUDA init.

    root_stage24_gate(spec) returns this helper's NEW canonical adapter record,
    containing actual closed report/Q refs and a decoded actual GPU_report.
    It must NOT delegate MEM qualification to old lineage.endpoint_gate().
    bound(ref) is ROOT's already-admitted complete-closure reference checker.
    Base factory ABI: make_point_head(torch, seed).
    Memory factory ABI: make_context_memory_head(torch, base, seed,tokens,dim).
    """
    import json
    check_recipe(spec)
    require(callable(root_stage24_gate), 'Actual ROOT stage24 adapter required')
    gate = root_stage24_gate(spec)  # RAW terminal gates must precede its first SHA.
    contract = stage24_adapter_contract()
    require(gate['schema'] == GATE_SCHEMA and
            all(gate['proofs'].get(k) is True for k in contract['mandatory_proofs']),
            'New stage24 training+independentCPU+own evaluation gate incomplete')
    require(all(is_reference(gate[k]) for k in contract['mandatory_refs']),
            'Canonical adapter must bind actual GPU and independent CPU refs')
    report = gate['GPU_report']
    require(report['source'] == gate['GPU_source'] and report['spec'] == gate['GPU_spec']
            and report['GPU_admission'] == gate['GPU_admission']
            and report['phase'] == gate['GPU_phase']
            and report['original_parent_epochs'] == report['fixed_stage_epochs'] == 24
            and report['stage_epochs_completed'] == report['last_stage_epoch'] == 24
            and report['first_stage_epoch'] == 17
            and report['cumulative_stage_updates_per_arm'] == STAGE24_UPDATES
            and report['leaf_updates_per_arm'] == 8 * 1661
            and report['continuous_parent_plus_stage'] is False,
            'Actual N14 stage17-24 report metadata differs')
    require(report['checkpoints'] == gate['checkpoints'] == spec['stage24_checkpoints']
            and set(report['checkpoints']) == set(ARMS),
            'The same two stage24 checkpoints must be bound everywhere')
    require(is_reference(report['stage_origin_START']) and
            len(report['trace_chain']) == 3 and
            all(is_reference(x) for x in report['trace_chain']) and
            report['trace_chain'][-1] == report['trace'] and
            len(report['history']) == len(report['epoch_orders']) == 24,
            'Complete stage24 cumulative history/orders/three-leaf traces required')
    # Check these references through the caller's closed SOURCE checker too.
    for key in contract['mandatory_refs']:
        bound(gate[key])
    require(json.loads(bound(gate['GPU_report_ref']).read_text(encoding='utf-8')) == report,
            'The decoded adapter report must equal the actual closed report file')
    for ref in report['trace_chain'] + [report['stage_origin_START']]:
        bound(ref)
    require(not torch.cuda.is_initialized(), 'Head loading must precede CUDA')
    outside = torch.get_rng_state().clone()
    heads, states, proofs = {}, {}, {}
    for arm in ARMS:
        cp_ref = report['checkpoints'][arm]
        cp = torch.load(str(bound(cp_ref)), map_location='cpu')
        meta, state = cp['meta'], cp['state_dict']
        require(meta['schema'] == 'N14-context-memory-stage-checkpoint-v2'
                and meta['arm'] == arm and meta['source'] == gate['GPU_source']
                and meta['spec'] == gate['GPU_spec']
                and meta['GPU_admission'] == gate['GPU_admission']
                and meta['phase'] == gate['GPU_phase']
                and meta['parent_head_epochs'] == meta['fixed_stage_epochs'] == 24
                and meta['stage_epochs_completed'] == 24
                and meta['stage_updates'] == meta['cumulative_stage_updates'] == STAGE24_UPDATES
                and meta['parameters'] == PARAMETERS[arm]
                and meta['stage_origin_START'] == report['stage_origin_START']
                and meta['trace_chain'] == report['trace_chain']
                and meta['history'] == report['history']
                and meta['epoch_orders'] == report['epoch_orders']
                and meta['gradient_activity'] == report['gradient_activity'][arm]
                and meta['initial_state_digests'] == report['initial_state_digests'][arm]
                and meta['original_behavior_preflight'] == report['behavior_preflight']
                and meta['continuous_parent_plus_stage'] is False,
                'The selected trained N14 state must match its own actual report')
        require(meta['checkpoint_resume_eligible'] is True and
                meta['original_preflight_inherited'] is True,
                'Stage24 must retain the original first-stage proofs')
        expected = dict(zip(BASE_NAMES, BASE_SHAPES))
        if arm == 'MEMORY_ADDON':
            expected = {'base.' + name: shape for name, shape in expected.items()}
            expected.update(MEMORY_SHAPES)
        require(set(state) == set(expected), 'Exact 8/14 named state tensors required')
        for name, shape in expected.items():
            value = state[name]
            require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                    and value.dtype == torch.float32 and tuple(value.shape) == shape
                    and bool(torch.isfinite(value).all()), 'Finite original FP32 state required')
        head = base_factory.make_point_head(torch, spec['seed'])
        if arm == 'MEMORY_ADDON':
            head = memory_factory.make_context_memory_head(
                torch, head, seed=spec['memory_seed'], tokens=8, dim=8)
        head.load_state_dict(state, strict=True)
        require(dict((name, tuple(p.shape)) for name, p in head.named_parameters()) == expected
                and sum(p.numel() for p in head.parameters()) == PARAMETERS[arm]
                and not tuple(head.buffers()), 'Factory ABI/state count differs')
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        heads[arm], states[arm] = head, state
        proofs[arm] = dict(checkpoint=cp_ref, parameters=PARAMETERS[arm],
                          stage_epochs_completed=24, cumulative_stage_updates=STAGE24_UPDATES,
                          independent_CPU_qualification=gate['CPU_qualification_ref'])
        del cp, head, parameter
    require(torch.equal(outside, torch.get_rng_state()) and not torch.cuda.is_initialized(),
            'Qualified head construction/loading must preserve CPU RNG/cold CUDA')
    return heads, states, proofs, gate


def capture_payload_budget(total_pixels, total_LOG_points, total_union_points=None,
                           images=187, maximum_image_pixels=None,
                           maximum_image_union_points=None,
                           maximum_image_LOG_points=None):
    """Exact raw tensor payload accounting; no actual data access.

    U = spatial union across ALL100 query LOG supports, T = sum of query Ks.
    FP32 F[U,256], INT64 union[U], FP32 e[1,100,256]: 1032U+102400.
    Per query: INT64 indices, FP32 L, residual+combined for each arm: 28T.
    This excludes serialization/RLE/reference metadata overhead. ROOT must
    prelock filesystem overhead/free-space margin and a separate peak RAM cap.
    The immutable full100 L selector is reused and is NOT duplicated here.
    """
    require(type(images) is int and images == 187 and type(total_pixels) is int
            and total_pixels > 0 and type(total_LOG_points) is int
            and 0 <= total_LOG_points <= 100 * total_pixels, 'Invalid full DEV count budget')
    bound_union = total_pixels if total_union_points is None else total_union_points
    require(type(bound_union) is int and 0 <= bound_union <= total_pixels,
            'Union pixel count cannot exceed original raster pixels')
    result = dict(raw_tensor_payload_bytes=1032 * bound_union + 28 * total_LOG_points
                  + images * 102400, union_count_is_upper_bound=total_union_points is None,
                  reused_selector_tensor_bytes=400 * total_pixels,
                  additional_full_raster_GPU_F_allocations=0,
                  additional_full100_selector_files=0,
                  per_point_chunk_F_tensor_bytes=4 * 256 * POINT_CHUNK)
    if maximum_image_pixels is not None:
        n = maximum_image_pixels
        require(type(n) is int and n > 0, 'Maximum original image pixels required')
        u = n if maximum_image_union_points is None else maximum_image_union_points
        require(type(u) is int and 0 <= u <= n, 'Maximum image union budget invalid')
        # Shared tensors only, NOT a total-process peak-RAM certificate.
        result['largest_image_shared_tensor_bytes_bound'] = 1032 * u + 102400
        result['largest_reused_selector_tensor_bytes'] = 400 * n
        result['one_query_tensor_vectors_bytes_bound'] = 28 * n
        result['union_boolean_working_bytes'] = n
        t = 100 * n if maximum_image_LOG_points is None else maximum_image_LOG_points
        require(type(t) is int and 0 <= t <= 100 * n, 'Maximum image total LOG points invalid')
        result['all100_LOG_index_arrays_bytes_bound'] = 8 * t
    return result


class ImageCapture:
    """Actual-GPU sample capture with a caller-provided fresh, bounded sink.

    sink.write(relative_name, value) saves once with exact readback and returns
    an actual {path,bytes,sha256} ref; sink.read(ref) verifies then decodes it.
    ROOT must implement/admit the sink; it may not overwrite existing files.
    This helper does not invent a training/evaluation owner or RAW certificate.
    """

    def __init__(self, torch, np, sink, binding, index, image_id, identity,
                 native_packet_ref, selector_row, qids, baseline,
                 F_original, e, indices):
        require(all(is_reference(binding[k]) for k in
                    ('source', 'spec', 'GPU_admission', 'stage24_GPU_report',
                     'stage24_CPU_qualification')), 'Capture must bind its own admitted stage24 evaluation')
        require(len(indices) == 100 and len(qids) == 100 and sorted(qids) == list(range(100))
                and len(baseline) == 100, 'Capture ALL100, including empty/unmatched queries')
        shape = tuple(identity['original_shape'])
        require(len(shape) == 2 and tuple(F_original.shape) == (256,) + shape
                and F_original.dtype == torch.float32 and F_original.device.type == 'cuda'
                and tuple(e.shape) == (1, 100, 256) and e.dtype == torch.float32
                and e.device == F_original.device, 'Actual original-raster GPU F/e ABI differs')
        self.torch, self.np, self.sink = torch, np, sink
        self.index, self.prefix = index, 'IMAGE_%03d_%s' % (index, image_id)
        self.indices, self.query_inputs, self.outputs = indices, {}, {arm: {} for arm in ARMS}
        pixels = shape[0] * shape[1]
        union = np.zeros(pixels, dtype=np.bool_)
        for idx in indices:
            require(idx.dtype == np.int64 and idx.ndim == 1
                    and (len(idx) == 0 or (int(idx[0]) >= 0 and int(idx[-1]) < pixels
                                          and bool(np.all(idx[1:] > idx[:-1])))),
                    'Natural-query support indices must be sorted unique flat INT64')
            union[idx] = True
        union_idx = np.flatnonzero(union).astype(np.int64)
        del union
        sampled = torch.empty((len(union_idx), 256), device='cpu', dtype=torch.float32)
        feature = F_original.flatten(1)
        # Chunked gathers avoid a full union-sized additional GPU F allocation.
        for start in range(0, len(union_idx), POINT_CHUNK):
            idx = torch.from_numpy(union_idx[start:start + POINT_CHUNK]).to(F_original.device)
            block = feature[:, idx].T.detach().cpu().contiguous()
            sampled[start:start + len(idx)].copy_(block)
            del idx, block
        require(bool(torch.isfinite(sampled).all()), 'Actual sampled GPU F must be finite')
        shared = dict(schema=CAPTURE_SCHEMA, binding=binding, dataset_index=index,
                      image_id=image_id, identity=identity, native_packet=native_packet_ref,
                      selector=selector_row, natural_query_ids=list(range(100)),
                      native_export_query_ids=qids, baseline=baseline,
                      union_flat_indices=torch.from_numpy(union_idx), sampled_F=sampled,
                      e0=e.detach().cpu().contiguous(), support_formula='abs(actual_original_GPU_native_L)<=2',
                      F_capture_source='actual GPU original_feature values, no CPU rerender',
                      point_chunk=POINT_CHUNK, head_parameters=PARAMETERS)
        self.shared_ref = sink.write(self.prefix + '/SHARED.pth', shared)
        require(is_reference(self.shared_ref), 'Shared capture must return a bound actual ref')
        self.union_count, self.total_points = len(union_idx), sum(len(x) for x in indices)
        self.tensor_payload_bytes = 1032 * self.union_count + 28 * self.total_points + 102400
        self.shared_metadata = {k: v for k, v in shared.items()
                                if k not in ('schema', 'union_flat_indices', 'sampled_F', 'e0')}
        del shared, sampled, feature, union_idx

    def record_query(self, arm, query_id, native_cpu, residual_cpu, combined_cpu):
        torch = self.torch
        require(arm in ARMS and type(query_id) is int and 0 <= query_id < 100
                and query_id not in self.outputs[arm], 'Each arm must capture each natural query exactly once')
        idx = self.indices[query_id]
        for value in (native_cpu, residual_cpu, combined_cpu):
            require(value.device.type == 'cpu' and value.dtype == torch.float32
                    and tuple(value.shape) == (len(idx),) and bool(torch.isfinite(value).all()),
                    'CPU FP32 captures must be actual finite per-point values')
        if arm == ARMS[0]:
            require(query_id not in self.query_inputs, 'Query input must be stored once')
            value = dict(schema=CAPTURE_SCHEMA, shared=self.shared_ref, query_id=query_id,
                         K=len(idx), flat_indices=torch.from_numpy(idx), native_L=native_cpu)
            self.query_inputs[query_id] = self.sink.write(
                self.prefix + '/QUERY_%03d_INPUT.pth' % query_id, value)
        else:
            require(query_id in self.query_inputs, 'BASE inputs must precede MEMORY outputs')
            original = self.sink.read(self.query_inputs[query_id])
            require(original['shared'] == self.shared_ref and original['query_id'] == query_id
                    and torch.equal(original['native_L'], native_cpu)
                    and torch.equal(original['flat_indices'], torch.from_numpy(idx)),
                    'Both arms must consume the identical actual native query inputs')
            del original
        value = dict(schema=CAPTURE_SCHEMA, shared=self.shared_ref,
                     inputs=self.query_inputs[query_id], query_id=query_id, arm=arm,
                     GPU_residual=residual_cpu, GPU_refined_logits=combined_cpu)
        self.outputs[arm][query_id] = self.sink.write(
            self.prefix + '/QUERY_%03d_%s.pth' % (query_id, arm), value)

    def finish(self, gpu_predictions):
        require(set(self.query_inputs) == set(range(100)) and
                all(set(self.outputs[arm]) == set(range(100)) for arm in ARMS)
                and set(gpu_predictions) == set(ARMS)
                and all(len(v) == 100 for v in gpu_predictions.values()),
                'Complete ALL100 input/output/export capture required')
        manifest = dict(schema=CAPTURE_SCHEMA, shared=self.shared_ref,
                        query_input_refs=[self.query_inputs[q] for q in range(100)],
                        arm_output_refs={arm: [self.outputs[arm][q] for q in range(100)] for arm in ARMS},
                        GPU_predictions_by_arm=gpu_predictions, U=self.union_count,
                        T=self.total_points, raw_tensor_payload_bytes=self.tensor_payload_bytes,
                        natural_queries=100, GT_query_selection=False,
                        independent_CPU_replayed=False, current_owned_terminal_self_certified=False,
                        **self.shared_metadata)
        return self.sink.write(self.prefix + '/MANIFEST.pth', manifest)


def refine_with_capture(torch, head, arm, F_original, e, L, indices,
                        capture, deadline):
    """Original sequential-arm/chunk2048 math; external oldLOG residual gate.

    Only one full100 refined raster is held, as in the old evaluator. Actual
    sampled F is captured once per IMAGE by ImageCapture, shared over queries.
    """
    result, feature, calls, counts = L.clone(), F_original.flatten(1), 0, []
    require(arm in ARMS and len(indices) == 100, 'Both new N14 arms/all100 required')
    for q in range(100):
        idx_all = torch.from_numpy(indices[q]).to(device=L.device, dtype=torch.long)
        count = int(idx_all.numel())
        counts.append(count)
        native_cpu = torch.empty(count, device='cpu', dtype=torch.float32)
        residual_cpu = torch.empty_like(native_cpu)
        combined_cpu = torch.empty_like(native_cpu)
        for first in range(0, count, POINT_CHUNK):
            deadline()
            idx = idx_all[first:first + POINT_CHUNK]
            native = L[q].flatten()[idx]
            predicted = head(feature[:, idx].T, e[0, q].expand(len(idx), 256), native,
                             L.new_zeros((len(idx), 49)), arm='PR')
            residual = predicted['residual']
            require(residual.dtype == torch.float32 and residual.device == L.device
                    and tuple(residual.shape) == tuple(native.shape)
                    and bool(torch.isfinite(residual).all()), 'Finite ungated FP32 residual required')
            combined = native + residual
            result[q].flatten()[idx] = combined
            native_cpu[first:first + len(idx)].copy_(native.detach().cpu())
            residual_cpu[first:first + len(idx)].copy_(residual.detach().cpu())
            combined_cpu[first:first + len(idx)].copy_(combined.detach().cpu())
            calls += 1
            del idx, native, predicted, residual, combined
        outside = torch.ones(L[q].numel(), device=L.device, dtype=torch.bool)
        outside[idx_all] = False
        require(torch.equal(result[q].flatten()[outside], L[q].flatten()[outside]),
                'Outside original oldLOG support must remain bitwise native')
        capture.record_query(arm, q, native_cpu, residual_cpu, combined_cpu)
        del idx_all, outside, native_cpu, residual_cpu, combined_cpu
    return result, counts, calls


def evaluate_image(torch, np, masks, renderer, kernels, reader, n10, coco,
                   packet, row, index, ref, native, heads, packet_selector_gate, spec,
                   original_selector_packet, capture_sink, capture_binding, deadline):
    """Adapted old image skeleton; requires a separately admitted caller.

    packet_check(), original_feature(), segmentation(), fixed_score_rows(),
    renderer.render_original(), kernels.order()/quality() remain the original
    closed helpers. original_selector_packet is the old exact reuse verifier,
    not the old full source_gate()/endpoint_gate()/load_heads().
    packet_selector_gate contains the original native-packet/selector helper
    ABI prepared by ROOT. It is distinct from the canonical N14 training gate
    returned by load_stage24_heads(); do not substitute one for the other.
    """
    import types
    check_recipe(spec)
    fields, identity, qids = n10.packet_check(torch, reader, coco, packet, row, index, ref, packet_selector_gate)
    require([v['query_id'] for v in native] == qids, 'Original export query order differs')
    with torch.no_grad():
        F, e = fields['F256'].to('cuda:0'), fields['e0'].to('cuda:0')
        M, C = fields['M200'].to('cuda:0'), fields['final_class_logits'].to('cuda:0')
        require(torch.equal(torch.einsum('bqc,bchw->bqhw', e, F), M), 'Original native M=eF must be exact')
        top, order = kernels.order(torch, C, types.SimpleNamespace(num_classes=1))
        require(order.cpu().tolist() == qids, 'Original all100 sorted=False order differs')
        L = renderer.render_original(torch, M, identity)
        require(L.dtype == torch.float32 and L.device.type == 'cuda'
                and tuple(L.shape) == (100,) + tuple(identity['original_shape'])
                and bool(torch.isfinite(L).all()), 'Original GPU natural100 L ABI differs')
        binary, quality, scores = kernels.quality(torch, L[order], top)
        native_masks, values = binary.cpu().numpy().astype(np.uint8), scores.cpu().tolist()
        baseline = []
        for ordinal, qid in enumerate(qids):
            current = dict(image_id=row['image_id'], query_id=qid, category_id=1,
                           score=float(values[ordinal]),
                           segmentation=n10.segmentation(np, masks, native_masks[ordinal]))
            require(current == native[ordinal], 'Native query/class/order/mask/natural score differs')
            if not bool(native_masks[ordinal].any()):
                require(current['score'] == 0., 'Original empty-query score differs')
            baseline.append(current)
        del M, C, top, order, binary, quality, scores
        # The existing verifier enforces current GPU L == immutable full selector.
        selector_row = original_selector_packet(torch, spec, packet_selector_gate, index, row, ref,
                                                identity, qids, L, None)
        exact = L.detach().cpu().contiguous().numpy()
        indices = [np.flatnonzero(np.abs(exact[q].reshape(-1)) <= 2).astype(np.int64)
                   for q in range(100)]
        counts = [len(x) for x in indices]
        require(counts == row['band_counts_all100'], 'Original all100 oldLOG point counts differ')
        del exact
        F_original = n10.original_feature(torch, renderer, F, identity)
        del F
        capture = ImageCapture(torch, np, capture_sink, capture_binding, index,
                               row['image_id'], identity, ref, selector_row, qids,
                               baseline, F_original, e, indices)
        predictions, diagnostics = {}, {}
        for arm in ARMS:
            refined, arm_counts, calls = refine_with_capture(
                torch, heads[arm], arm, F_original, e, L, indices, capture, deadline)
            require(arm_counts == counts, 'Both arms must use identical original oldLOG support')
            exported = n10.fixed_score_rows(np, masks, baseline, refined)
            require(len(exported) == 100 and
                    all({k: v for k, v in x.items() if k != 'segmentation'} ==
                        {k: v for k, v in y.items() if k != 'segmentation'}
                        for x, y in zip(baseline, exported)), 'Original fixed score/class/order differs')
            per_query = []
            for ordinal, original in enumerate(baseline):
                q = original['query_id']
                before, after = native_masks[ordinal].astype(bool), (refined[q] > 0).cpu().numpy().astype(bool)
                added, removed = int(np.count_nonzero(~before & after)), int(np.count_nonzero(before & ~after))
                per_query.append(dict(query_id=q, native_export_ordinal=ordinal, K=counts[q],
                                      fixed_native_score=original['score'], added_pixels=added,
                                      removed_pixels=removed, changed_pixels=added + removed))
            predictions[arm] = exported
            diagnostics[arm] = dict(parameters=PARAMETERS[arm], head_point_chunk_calls=calls,
                                    query_diagnostics=per_query)
            del refined, before, after
        capture_ref = capture.finish(predictions)
        summary = dict(dataset_index=index, image_id=row['image_id'], queries=100,
                       packet=ref, selector=selector_row, capture=capture_ref,
                       K_natural_query_order=counts, U=capture.union_count, T=capture.total_points,
                       raw_capture_tensor_bytes=capture.tensor_payload_bytes, controls=diagnostics,
                       exact_GPU_L_selector=True, oldLOG_support_unchanged=True,
                       GT_used_to_choose_queries_or_residual=False,
                       fixed_native_scores_class_order=True, stage_epochs_completed=24)
        del F_original, e, L, capture
    return baseline, predictions, summary, selector_row, capture_ref


def cpu_replay_captured_image(torch, np, masks, n10, heads, manifest,
                              sink, load_closed_selector, numeric_protocol,
                              completed_GPU_evaluation_ref):
    """CPU same-input 8/14-tensor residual/mask replay over ALL100 queries.

    ROOT must admit a separate CPU runner and lock/bind numeric_protocol before
    CPU execution. This helper performs no CPU F resize/rerender. Numeric
    residual agreement alone never waives an exact sign-mask/RLE difference.
    """
    require(numeric_protocol['same_actual_GPU_point_inputs'] is True
            and numeric_protocol['point_chunk'] == POINT_CHUNK
            and numeric_protocol['mask_RLE_exact'] is True
            and numeric_protocol['locked_before_CPU_execution'] is True
            and is_reference(completed_GPU_evaluation_ref)
            and numeric_protocol['bound_completed_GPU_evaluation_report'] == completed_GPU_evaluation_ref,
            'A prelocked actual-GPU-input CPU protocol is required')
    atol, rtol = numeric_protocol['atol'], numeric_protocol['rtol']
    require(type(atol) in (int, float) and 0 <= atol < float('inf')
            and type(rtol) in (int, float) and 0 <= rtol < float('inf'), 'Finite locked tolerance required')
    shared = sink.read(manifest['shared'])
    require(shared['schema'] == manifest['schema'] == CAPTURE_SCHEMA
            and shared['binding'] == manifest['binding']
            and shared['native_export_query_ids'] == manifest['native_export_query_ids']
            and manifest['natural_queries'] == 100
            and len(manifest['query_input_refs']) == 100
            and set(heads) == set(ARMS), 'Complete bound image capture required')
    metadata = {k: v for k, v in shared.items()
                if k not in ('schema', 'union_flat_indices', 'sampled_F', 'e0')}
    require(all(manifest[k] == value for k, value in metadata.items())
            and manifest['head_parameters'] == PARAMETERS
            and manifest['support_formula'] == 'abs(actual_original_GPU_native_L)<=2'
            and manifest['point_chunk'] == POINT_CHUNK,
            'All shared input provenance/shape/order metadata must match the image manifest')
    for arm in ARMS:
        require(not heads[arm].training and not tuple(heads[arm].buffers())
                and sum(p.numel() for p in heads[arm].parameters()) == PARAMETERS[arm]
                and all(p.device.type == 'cpu' and p.dtype == torch.float32
                        and not p.requires_grad and p.grad is None
                        and bool(torch.isfinite(p).all()) for p in heads[arm].parameters()),
                'CPU replay requires frozen qualified FP32 stage24 heads')
    F, union, e = shared['sampled_F'], shared['union_flat_indices'], shared['e0']
    require(F.device.type == union.device.type == e.device.type == 'cpu'
            and F.dtype == e.dtype == torch.float32 and union.dtype == torch.int64
            and tuple(F.shape) == (manifest['U'], 256) and tuple(union.shape) == (manifest['U'],)
            and tuple(e.shape) == (1, 100, 256) and bool(torch.isfinite(F).all())
            and bool(torch.isfinite(e).all()) and bool((union[1:] > union[:-1]).all()),
            'Finite actual captured shared F/e and sorted spatial union required')
    L = load_closed_selector(manifest['selector'])
    shape = tuple(manifest['identity']['original_shape'])
    require(L.device.type == 'cpu' and L.dtype == torch.float32
            and tuple(L.shape) == (100,) + shape and bool(torch.isfinite(L).all()),
            'Reuse qualified actual original GPU full100 L, no CPU rerender')
    union_check = np.zeros(shape[0] * shape[1], dtype=np.bool_)
    baseline = manifest['baseline']
    require(len(baseline) == 100 and [x['query_id'] for x in baseline] == manifest['native_export_query_ids'],
            'Original all100 export ordinal/natural score baseline required')
    originals = {row['query_id']: row for row in baseline}
    replay = {arm: {} for arm in ARMS}
    total_points = 0
    with torch.no_grad():
        for q in range(100):
            inputs = sink.read(manifest['query_input_refs'][q])
            idx, native = inputs['flat_indices'], inputs['native_L']
            expected_idx = torch.nonzero(L[q].flatten().abs() <= 2, as_tuple=False).flatten()
            require(inputs['shared'] == manifest['shared'] and inputs['query_id'] == q
                    and idx.dtype == torch.int64 and idx.device.type == 'cpu'
                    and native.dtype == torch.float32 and native.device.type == 'cpu'
                    and inputs['K'] == len(idx) and tuple(native.shape) == (len(idx),)
                    and torch.equal(idx, expected_idx) and torch.equal(native, L[q].flatten()[idx]),
                    'Capture must contain ALL oldLOG points from the exact selector')
            union_check[idx.numpy()] = True
            positions = torch.searchsorted(union, idx)
            require(len(idx) == 0 or (int(positions.max()) < len(union)
                                     and torch.equal(union[positions], idx)), 'Shared F mapping differs')
            total_points += len(idx)
            before = masks.decode(originals[q]['segmentation']).astype(np.uint8)
            require(tuple(before.shape) == shape and np.array_equal(before.astype(bool), (L[q] > 0).numpy()),
                    'Original native RLE must be exact from the qualified selector')
            for arm in ARMS:
                actual = sink.read(manifest['arm_output_refs'][arm][q])
                require(actual['shared'] == manifest['shared'] and actual['query_id'] == q
                        and actual['arm'] == arm and actual['inputs'] == manifest['query_input_refs'][q],
                        'Actual GPU query outputs must bind the same sampled inputs')
                residual = torch.empty_like(native)
                combined = torch.empty_like(native)
                for first in range(0, len(idx), POINT_CHUNK):
                    last = min(first + POINT_CHUNK, len(idx))
                    result = heads[arm](F[positions[first:last]], e[0, q].expand(last - first, 256),
                                        native[first:last], native.new_zeros((last - first, 49)), arm='PR')
                    value = result['residual']
                    require(value.dtype == torch.float32 and value.device.type == 'cpu'
                            and tuple(value.shape) == (last - first,)
                            and bool(torch.isfinite(value).all()), 'Finite CPU point residual required')
                    residual[first:last] = value
                    combined[first:last] = native[first:last] + value
                    del result, value
                for name, replayed in (('GPU_residual', residual), ('GPU_refined_logits', combined)):
                    target = actual[name]
                    require(target.dtype == torch.float32 and target.device.type == 'cpu'
                            and target.shape == replayed.shape and bool(torch.isfinite(target).all())
                            and torch.allclose(replayed, target, atol=atol, rtol=rtol),
                            'CPU same-input residual/combined tolerance failed')
                require(torch.equal(combined > 0, actual['GPU_refined_logits'] > 0),
                        'Even within numeric tolerance, exact point sign-mask is mandatory')
                after = before.copy()
                after.reshape(-1)[idx.numpy()] = (combined > 0).numpy().astype(np.uint8)
                predicted = dict(originals[q])
                predicted['segmentation'] = n10.segmentation(np, masks, after)
                ordinal = manifest['native_export_query_ids'].index(q)
                require(predicted == manifest['GPU_predictions_by_arm'][arm][ordinal],
                        'Independent CPU full-query fixed-score RLE must exactly match actual GPU export')
                replay[arm][q] = predicted
                del actual, residual, combined, after, predicted
            del inputs, idx, native, expected_idx, positions, before
    require(total_points == manifest['T'] and
            torch.equal(torch.from_numpy(np.flatnonzero(union_check).astype(np.int64)), union)
            and manifest['raw_tensor_payload_bytes'] == 1032 * len(union) + 28 * total_points + 102400,
            'Shared union/ALL100 payload accounting must match exactly')
    return {arm: [replay[arm][q] for q in manifest['native_export_query_ids']] for arm in ARMS}


def official_COCO_readback(coco, ids, native, GPU_predictions, CPU_predictions,
                           official, COCOeval, n10, GPU_control_results):
    """Original official adapter, after separate full187 CPU source/RAW gate."""
    require(len(ids) == len(set(ids)) == 187 and len(coco.anns) == 380
            and set(coco.cats) == {1} and len(native) == 18700
            and set(GPU_predictions) == set(CPU_predictions) == set(ARMS), 'Original full DEV population required')
    results = {}
    for control in ('native_official',) + ARMS:
        values = native if control == 'native_official' else CPU_predictions[control]
        require(len(values) == 18700 and
                (control == 'native_official' or values == GPU_predictions[control]),
                'Exact full18700 CPU/GPU query population required')
        require(all({k: v for k, v in a.items() if k != 'segmentation'} ==
                    {k: v for k, v in b.items() if k != 'segmentation'}
                    for a, b in zip(native, values)), 'Natural scores/class/order must remain exact')
        metrics = official.coco_metrics(coco, values, ids, COCOeval)
        require(metrics['iouType'] == 'segm' and metrics['maxDets'] == [1, 10, 100]
                and metrics['score_filter'] is None and metrics['original_GT_preserved'] is True
                and metrics['official_COCO_crowd_ignore_semantics'] is True, 'Original official COCO convention differs')
        n10.stats_exact(metrics['stats'], GPU_control_results[control]['stats'],
                        'Independent CPU original official COCO12 must match actual GPU')
        results[control] = metrics
    return results


# Intentionally no run(), argparse entrypoint, auto-IO, or owner execution path.
# This ordinary helper is not an actual SOURCE, qualification, or science run.
