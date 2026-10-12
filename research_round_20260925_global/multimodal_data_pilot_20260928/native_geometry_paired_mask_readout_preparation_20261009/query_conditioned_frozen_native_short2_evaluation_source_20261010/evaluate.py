"""Inert ROOT-owned official DEV evaluation of five frozen-native endpoints.

Native0 is the full E24 teacher; four query modes strictly load independently
CPU-qualified full612 SHORT2 E2 endpoints whose native610 equals that teacher.
The three original RAW teacher/preflight/SHORT2 terminal gates precede every
SOURCE/SPEC/Q/CP/GT hash and scientific import. All five CPU models precede CUDA.
Every original DEV187 image/380 raw annotation is retained. GT is stripped
from forward samples; no matching, training, TEST or AP runs in this entry.

The unchanged official single-class/all100 instance fusion determines query
order, masks and class-probability times predicted-mask-quality scores. No NMS,
cutoff or query filtering is added. Frozen class/native field equality does
not imply equal mask quality or official scores. Small class packets, native
field digests, query-hook support/RMS, mask RLE and source-observed TorchEqual
fusion proofs support an independent CPU COCO12 and score-free best100 audit.
Omitted pixel logits are not independently reconstructed by that audit.
SOURCE preparation performs no scientific execution or payload access.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import types
from contextlib import nullcontext


SCHEMA = 'ROOT-query-conditioned-frozen-native-short2-evaluation-spec-v1'
REPORT_SCHEMA = 'query-conditioned-frozen-native-short2-evaluation-report-v1'
STATE_SCHEMA = 'query-conditioned-frozen-native-short2-evaluation-state-proof-v1'
MODES = ('raw', 'raw_rms', 'query_rms', 'permuted_query_rms')
ENDPOINTS = ('native0',) + MODES
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
SHORT2_SHA = '16f838776edf3a6d24730dc283a7c98239b2c365e4a513d731ec4ead05d68548'
SHORT2_READER_SHA = '1d2c2c06c358db731556094f68a35ee7d70a92bb43464350200b396412bb0cfc'
OFFICIAL_SOURCE_SHA = 'b4ca717bd2018ea8404808c03510f830081a6660c893eda04fce72d7adb43a48'
OFFICIAL_READER_SHA = '2357929fcd38ffc3020c5b8d204ee371da8da3972ff2e2e03398c8225c55e916'
PREFIX = 'panoptic_head.query_pair_readout.'
ACTIVE = [PREFIX + name for name in ('hidden.weight', 'final.weight')]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for value in iter(lambda: stream.read(1048576), b''):
            digest.update(value)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'} and
            Path(row['path']).is_absolute() and reference(row['path']) == row, 'Absolute pinned reference differs')
    return Path(row['path'])


def load_source(row, name):
    path = bound(row)
    value = path.read_bytes()
    require(len(value) == row['bytes'] and hashlib.sha256(value).hexdigest() == row['sha256'], 'SOURCE changed before compile')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(value, str(path), 'exec'), module.__dict__)
    return module


def raw_terminal(receipt, phase):
    require(receipt['status'] == 'ACTUAL_OWNED_TERMINAL' and receipt['original_identity'] is not None and
            type(receipt['returncode']) is int and receipt['returncode'] == 0 and
            receipt['root_wait_reaped'] is True and receipt['held_handle_exit'] ==
            dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
            receipt['numeric_birth_absence']['original_birth_absent'] is True and
            receipt['abort'] is None and receipt['error'] is None and receipt['wait_errors'] is False and
            receipt['GPU_observation_errors'] == [], 'STOP before scientific access: original held0/reap/birth absence/noerror required')
    if 'phase' in receipt:
        require(receipt['phase'] == phase, 'Original terminal phase differs')
    peak = receipt['whole_GPU_observed_peak_MiB']
    require(type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            receipt['whole_GPU_used_limit_MiB'] == 8192 and receipt['whole_leaf_VRAM_certified'] is False and
            receipt['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation', 'Original whole-GPU observed scope differs')


def owner_gate(spec_path, spec):
    admission = read(Path(spec['GPU_admission_file']).resolve(strict=True))
    entry, spec_ref = reference(__file__), reference(spec_path)
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            admission['automatic_retry'] is False and admission['phase'] == spec['phase'] and
            admission['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and admission['cuda_visible_devices'] == '0' and
            admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] == 1800 and
            admission['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192 and
            Path(admission['entry']).resolve(strict=True) == Path(entry['path']) and
            admission['entry_args'] == ['--spec', str(spec_path), '--owner-execute'] and admission['driver_sha256'] == OWNER_SHA,
            'Actual current evaluation owner/entry/resource metadata differs')
    hashes = admission['bound_source_sha256']
    require(spec['entry_source'] == entry and entry in spec['sources'] and spec['CPU_reader'] in spec['sources'] and
            hashes.get(str(spec_path)) == spec_ref['sha256'], 'Current evaluation entry/reader/SPEC not owner-bound')
    for source in spec['sources']:
        require(hashes.get(source['path']) == source['sha256'], 'Evaluation complete SOURCE closure not owner-bound')
        bound(source)
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Current scientific SPEC differs')
    return spec_ref


def qualified_short2(spec, raw):
    """Only the actual original all-FIT E2 qualification is eligible."""
    bound(spec['short2_receipt'])
    train = read(bound(spec['short2_training_spec']))
    q = read(bound(spec['short2_CPU_qualification']))
    require(train['schema'] == 'ROOT-query-conditioned-frozen-native-short2-spec-v1' and
            train['entry_source'] == q['source'] == spec['short2_source'] and
            train['CPU_reader'] == q['reader_source'] == spec['short2_CPU_reader'] and
            train['phase'] == q['phase'] == spec['short2_phase'] and q['spec'] == spec['short2_training_spec'] and
            q['parent_receipt'] == spec['short2_receipt'] and
            q['schema'] == 'ROOT-query-conditioned-frozen-native-short2-CPU-qualification-v1' and
            q['status'] == 'ROOT_QUERY_CONDITIONED_FROZEN_NATIVE_SHORT2_CPU_QUALIFIED' and
            q['modes'] == list(MODES) and q['actual_epochs_per_arm'] == 2 and
            q['actual_microsteps_per_arm'] == 1482 and q['actual_optimizer_updates_per_arm'] == 741 and
            q['actual_raw_GT_visits_per_arm'] == 3322 and q['total_trace_records'] == 5928 and
            q['AP_computed'] is False and q['efficacy_claimed'] is False and q['eligible_for_formal_resume'] is False,
            'Actual independently qualified complete frozen-native SHORT2 lineage differs')
    for key in ('all_teacher610_states_and_typed_BN_exact',
                'only_fresh2W_changed_and_all_AdamW_moments_step741_verified',
                'native30_and_all10_assignment_GT_coverage_verified', 'private_CPU_generator_order_independently_reproduced',
                'odd741_epoch_boundary_actual_pending_gradient_digest_identity_verified', 'no_epoch_gradient_flush'):
        require(q[key] is True, 'Actual SHORT2 CPU assertion absent: ' + key)
    require(q['teacher_optimizer_order_RNG_or_preflight_CP_restored'] is False and
            train['training_all_FIT'] is True and train['epochs'] == 2 and train['FIT_images'] == 741 and
            train['FIT_GT'] == 1661 and train['total_microsteps'] == 1482 and train['total_updates'] == 741,
            'Only complete fresh SHORT2 state endpoints are accepted')
    for key in ('teacher_checkpoint', 'teacher_qualification', 'teacher_training_spec', 'teacher_receipt',
                'preflight_CPU_qualification', 'preflight_source', 'preflight_CPU_reader',
                'preflight_training_spec', 'preflight_receipt', 'FIT_annotation'):
        require(q[key] == train[key] == spec[key], 'Actual SHORT2 teacher/preflight input differs: ' + key)
    for key in ('initializer_source', 'micro_preflight_source', 'native_helpers', 'native_preflight_helpers',
                'reused_CPU_reader', 'evaluation_source', 'training_source', 'training_CPU_reader',
                'readout_source', 'geometry_source', 'hooks_source', 'operator_CPU_fixture',
                'operator_CPU_fixture_source', 'image_root', 'teacher_phase', 'preflight_phase'):
        require(train[key] == spec[key], 'Actual SHORT2 native construction/input role differs: ' + key)
    require(isinstance(train['sources'], list) and len(train['sources']) > 0 and
            all(source in spec['sources'] for source in train['sources']) and
            spec['checkpointmap']['native0'] == spec['teacher_checkpoint'] and
            spec['qualificationmap']['native0'] == spec['teacher_qualification'] and
            all(spec['checkpointmap'][mode] == q['checkpointmap'][mode] == q['training_proofs'][mode]['checkpoint'] and
                spec['qualificationmap'][mode] == spec['short2_CPU_qualification'] for mode in MODES),
            'Actual SHORT2 closure and five endpoint CP/qualification joins differ')
    admission_path = bound(q['GPU_admission'])
    admission = read(admission_path)
    require(raw['admission_sha256'] == q['GPU_admission']['sha256'] and
            Path(train['GPU_admission_file']).resolve(strict=True) == admission_path.resolve(strict=True) and
            admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            admission['automatic_retry'] is False and admission['phase'] == q['phase'] and
            admission['entry_args'] == ['--spec', spec['short2_training_spec']['path'], '--owner-execute'] and
            Path(admission['entry']).resolve(strict=True) == Path(spec['short2_source']['path']).resolve(strict=True) and
            admission['driver_sha256'] == OWNER_SHA and admission['allowed_cpus'] == [0, 1] and
            admission['cuda_visible_devices'] == '0' and admission['whole_GPU_used_limit_MiB'] == 8192 and
            admission['qualified_leaf_deadline_seconds'] == train['deadline_seconds'] == 10800,
            'Actual SHORT2 raw receipt/admission/source/SPEC join differs')
    hashes = admission['bound_source_sha256']
    require(hashes.get(q['spec']['path']) == q['spec']['sha256'] and hashes.get(q['source']['path']) == q['source']['sha256'] and
            all(hashes.get(source['path']) == source['sha256'] for source in train['sources']), 'Original SHORT2 bindings differ')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == q['spec'], 'Original SHORT2 scientific SPEC differs')
    for key in ('source', 'reader_source', 'report', 'start', 'GPU_qualification', 'plan', 'owner_driver'):
        bound(q[key])
    return train, q


def prepare_cpu(torch, core, pref, initializer, official, readout, geometry, hooks,
                registry, revert, Config, classes, spec, teacher_row, short_q, mode, teacher):
    helper_spec = dict(initializer_source=spec['initializer_source'], training_source=spec['training_source'],
        checkpointmap={'native0': spec['teacher_checkpoint']}, qualificationmap={'native0': spec['teacher_qualification']}, expected_epoch=24)
    model, native, teacher_meta, teacher_loading = official.prepare_model(torch, pref, initializer, None, None, None,
        registry, revert, Config, classes, helper_spec, teacher_row, 'native0')
    if teacher is not None:
        require(pref.tree_equal(torch, teacher, native), 'All five architectures must load the same exact610 teacher')
    if mode == 'native0':
        return (model, native, teacher_meta, teacher_loading), native
    hooks.register_query_pair_readout(torch, readout, geometry, model, mode,
        spec['readout_source'], spec['geometry_source'], hidden_seed=2026100911, permutation_seed=2026101013)
    cp = torch.load(str(bound(spec['checkpointmap'][mode])), map_location='cpu')
    state, meta = cp['state_dict'], cp['meta']
    require(meta['schema'] == 'query-conditioned-frozen-native-short2-checkpoint-v1' and
            meta['kind'] == 'mask2former' and meta['mode'] == mode and meta['source'] == spec['short2_source'] and
            meta['spec'] == spec['short2_training_spec'] and meta['phase'] == spec['short2_phase'] and
            meta['teacher_checkpoint'] == spec['teacher_checkpoint'] and meta['teacher_mode'] == 'native0' and
            meta['teacher_epoch'] == 24 and meta['completed_epoch'] == meta['completed_epochs'] == 2 and
            meta['global_microsteps'] == 1482 and meta['global_updates'] == 741 and meta['consumed_raw_GT'] == 3322 and
            meta['native610_exact'] is True and meta['frozen_native'] is True and meta['native_loss_only'] is True and
            meta['native_loss_atoms'] == 30 and meta['no_pending_accumulation_gradients'] is True and
            meta['preflight_checkpoint_loaded'] is False and meta['resume_checkpoint'] is None and
            meta['resume_qualification'] is None and meta['eligible_for_formal_resume'] is False,
            'Only actual full612 qualified all-FIT SHORT2 E2 inference endpoints are accepted')
    current = model.state_dict()
    require(set(state) == set(current) == set(native) | set(ACTIVE) and len(state) == 612 and
            all(torch.is_tensor(value) and value.device.type == 'cpu' and value.shape == current[name].shape and
                value.dtype == current[name].dtype and bool(torch.isfinite(value).all()) for name, value in state.items()) and
            all(torch.equal(state[name], value) for name, value in native.items()),
            'Trained full612 schema/finite values/native610 preservation differs')
    loaded = model.load_state_dict(state, strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys and
            all(torch.equal(value.detach().cpu(), state[name]) for name, value in model.state_dict().items()),
            'Strict full612 endpoint load differs')
    proof = read(bound(short_q['training_proofs'][mode]['state_proof']))
    require(pref.state_proof(state) == proof['after'] and pref.state_proof(native) == proof['native_teacher610'] and
            proof['native610_exact'] is True and proof['frozen_BN_all_exact'] is True,
            'Loaded endpoint differs from independently qualified actual fullstate proof')
    model.requires_grad_(False).eval()
    require(len(list(model.parameters())) == 453 and all(parameter.grad is None for parameter in model.parameters()) and
            all(not module.training for module in model.modules()), 'Fully frozen eval endpoint required')
    loading = dict(mode=mode, checkpoint=spec['checkpointmap'][mode], qualification=spec['short2_CPU_qualification'],
        teacher_loading=teacher_loading, teacher_checkpoint=spec['teacher_checkpoint'], completed_epoch=2,
        CP_qualified=True, strict_fullstate_loaded=True, trained_fullstate=pref.state_proof(state), native_state_tensors=610,
        adapter_state_tensors=2, registered_parameters=453, native610_exact_teacher=True,
        BN_parameter_names=teacher_loading['BN_parameter_names'], BN_buffer_names=teacher_loading['BN_buffer_names'],
        original_frozen_BN_exact=True, inference_backbone_with_cp=False, optimizer_or_RNG_restored=False,
        no_random_missing_keys=True)
    del cp, current, proof
    return (model, state, meta, loading), native


def official_predictions(torch, np, masks_api, model, classes, masks, samples, identity, pref):
    """Exact official200->800/crop/raw instance fusion; no added filtering."""
    fusion = model.panoptic_fusion_head
    require(fusion.num_classes == fusion.num_things_classes == 1 and fusion.num_stuff_classes == 0 and
            fusion.test_cfg['max_per_image'] == 100 and fusion.test_cfg.get('instance_on') is True and
            fusion.test_cfg.get('panoptic_on') is False and not fusion.test_cfg.get('semantic_on', False),
            'Official single-class instance-only all100 fusion required')
    padded = torch.nn.functional.interpolate(masks, size=(800, 800), mode='bilinear', align_corners=False)
    instances = fusion.predict(classes, padded, samples, rescale=True)[0]['ins_results']
    ih, iw = identity['actual_img_shape']
    oh, ow = identity['original_shape']
    rendered = torch.nn.functional.interpolate(padded[0, :, :ih, :iw][:, None], size=(oh, ow),
        mode='bilinear', align_corners=False)[:, 0]
    probabilities = torch.nn.functional.softmax(classes[0], dim=-1)[:, :-1]
    top_scores, indices = probabilities.flatten(0, 1).topk(100, sorted=False)
    qids = indices // fusion.num_classes
    require(torch.equal(qids.sort().values, torch.arange(100, device=qids.device)) and
            torch.equal(top_scores, probabilities[qids, 0]), 'Official all100 query mapping differs')
    ordered = rendered[qids]
    binary = (ordered > 0).float()
    qualities = (ordered.sigmoid() * binary).flatten(1).sum(1) / (binary.flatten(1).sum(1) + 1e-6)
    scores = probabilities[qids, 0] * qualities
    require(len(instances) == 100 and torch.equal(instances.masks, binary.bool()) and torch.equal(instances.scores, scores) and
            bool((instances.labels == 0).all()) and bool(torch.isfinite(scores).all()) and bool(((scores >= 0) & (scores <= 1)).all()),
            'Official fusion/query/mask/score SOURCE-observed TorchEqual check failed')
    ids = qids.detach().cpu().tolist()
    binary_cpu = instances.masks.detach().cpu().numpy().astype(np.uint8)
    score_cpu, class_cpu, quality_cpu = scores.cpu().tolist(), probabilities[qids, 0].cpu().tolist(), qualities.cpu().tolist()
    predictions, factors = [], []
    for query, mask, score, probability, quality in zip(ids, binary_cpu, score_cpu, class_cpu, quality_cpu):
        rle = masks_api.encode(np.asfortranarray(mask))
        counts = rle['counts'].decode('ascii') if isinstance(rle['counts'], bytes) else rle['counts']
        predictions.append(dict(image_id=int(identity['image_id']), query_id=int(query), category_id=1,
            score=float(score), segmentation=dict(size=list(rle['size']), counts=counts)))
        factors.append(dict(image_id=int(identity['image_id']), query_id=int(query), class_probability=float(probability),
            mask_quality=float(quality), official_score=float(score)))
    packet = dict(pref.digest_tensor(classes[0]), values=classes[0].detach().cpu().tolist())
    return predictions, factors, ids, packet, dict(classes=pref.digest_tensor(classes), masks=pref.digest_tensor(masks))


def evaluate_endpoint(torch, np, masks_api, core, pref, old, native, hooks, spec, prepared,
                      teacher, dataset, coco, output, mode, shared, started):
    model, state, meta, loading = prepared
    directory = output / mode
    directory.mkdir()
    model.to('cuda:0').requires_grad_(False).eval()
    before = pref.state_proof(model.state_dict())
    require(before == loading['trained_fullstate'] and all(parameter.grad is None for parameter in model.parameters()),
            'Evaluation start differs from actual qualified endpoint')
    baseline_hooks = core.hook_inventory(model)
    rows, predictions, factors = [], [], []
    try:
        for index in range(187):
            pref.check_deadline(started, spec['deadline_seconds'])
            image_id = int(dataset.get_data_info(index)['img_id'])
            data, samples, identity = core.image_data(native, old, model, dataset, coco, image_id)
            require(identity['dataset_index'] == index and len(samples) == 1 and
                    all(not any(key in sample for key in core.GT_FIELDS) for sample in samples), 'All four GT fields must be absent from head samples')
            external = pref.rng_snapshot(torch)
            cache, counts, original, fields = {}, dict(pixel=0, coefficient=0), [], []
            head = model.panoptic_head

            def capture_feature(module, inputs, output):
                counts['pixel'] += 1
                value = output[0]
                require(torch.is_tensor(value) and tuple(value.shape) == (1, 256, 200, 200) and
                        value.dtype == torch.float32 and not value.requires_grad, 'Actual frozen FP32 F200 required')
                cache['feature'] = value

            def capture_coefficient(module, inputs, output):
                counts['coefficient'] += 1
                require(tuple(output.shape) == (1, 100, 256) and output.dtype == torch.float32 and not output.requires_grad,
                        'Actual frozen FP32 e0 readout required')
                if counts['coefficient'] == 10:
                    cache['coefficient'] = output

            def capture_original(module, inputs, output):
                require(len(output) == 2 and len(output[0]) == len(output[1]) == 10 and
                        all(tuple(value.shape) == (1, 100, 2) and value.dtype == torch.float32 and not value.requires_grad for value in output[0]) and
                        all(tuple(value.shape) == (1, 100, 200, 200) and value.dtype == torch.float32 and not value.requires_grad for value in output[1]),
                        'All ten native class/mask interfaces must remain frozen FP32')
                original.append(output)
                fields.append(dict(classes=[pref.digest_tensor(value) for value in output[0]],
                    native_masks=[pref.digest_tensor(value) for value in output[1]]))

            feature_handle = head.pixel_decoder.register_forward_hook(capture_feature)
            coefficient_handle = head.mask_embed.register_forward_hook(capture_coefficient)
            original_handle = head.register_forward_hook(capture_original)
            seed = 100000 + image_id
            try:
                with torch.random.fork_rng(devices=[0]):
                    torch.random.default_generator.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    private_before = pref.rng_snapshot(torch)
                    with torch.no_grad():
                        features = model.extract_feat(data['inputs'])
                        scope = hooks.query_pair_readout_context(torch, head, [sample.metainfo for sample in samples]) \
                            if mode != 'native0' else nullcontext(None)
                        with scope as hook_record:
                            classes, masks = head(features, samples)
                        require(counts == dict(pixel=1, coefficient=10) and len(original) == len(fields) == 1 and
                                classes is original[0][0] and all(masks[i] is original[0][1][i] for i in range(9)),
                                'Only final native mask may be replaced; field capture counts differ')
                        valid = hooks.metadata_valid_fraction(torch, [sample.metainfo for sample in samples], cache['feature'])
                        native_fields = dict(fields[0], feature=pref.digest_tensor(cache['feature']),
                            coefficient=pref.digest_tensor(cache['coefficient']), valid_fraction=pref.digest_tensor(valid))
                        if mode != 'native0':
                            require(hook_record['mode'] == mode and hook_record['apply'] is True and
                                    hook_record['forward_calls'] == hook_record['completed_forwards'] == 1 and
                                    hook_record['adapter_calls'] == hook_record['final_mask_replacements'] == 1 and
                                    hook_record['class_list_object_preserved'] is True and
                                    hook_record['first_nine_mask_objects_preserved'] is True and
                                    hook_record['hooks_removed'] is True and hook_record['cache_cleared_after_forward'] is True and
                                    hook_record['readout_source'] == spec['readout_source'] and
                                    hook_record['geometry_source'] == spec['geometry_source'], 'Actual query whole-head bridge scope differs')
                        image_predictions, image_factors, query_ids, packet, final_fields = official_predictions(
                            torch, np, masks_api, model, classes[-1], masks[-1], samples, identity, pref)
                        common = dict(native_fields=native_fields, native_class_packet=packet, prediction_query_ids=query_ids,
                            class_probabilities=[row['class_probability'] for row in image_factors])
                        if mode == 'native0':
                            shared[image_id] = common
                        require(common == shared[image_id], 'Observed frozen native fields/classes/query order differ across five endpoints')
                    private_after = pref.rng_snapshot(torch)
            finally:
                feature_handle.remove()
                coefficient_handle.remove()
                original_handle.remove()
                cache.clear()
                original.clear()
                fields.clear()
            after_rng = pref.rng_snapshot(torch)
            require(core.equal_rng(torch, external, after_rng) and core.hook_inventory(model) == baseline_hooks and
                    hooks._contexts_inactive(head) and all(parameter.grad is None for parameter in model.parameters()) and
                    all(not module.training for module in model.modules()), 'Evaluation RNG/hooks/gradNone/eval cleanup differs')
            predictions.extend(image_predictions)
            factors.extend(image_factors)
            rows.append(dict(mode=mode, checkpoint_mode=mode, dataset_index=index, image_id=image_id, identity=identity,
                image=identity['image'], ann_ids=identity['ann_ids'], annotation_count=len(identity['ann_ids']),
                original_shape=identity['original_shape'], actual_img_shape=identity['actual_img_shape'],
                batch_input_shape=identity['batch_input_shape'], scale_factor=identity['scale_factor'],
                input=pref.digest_tensor(data['inputs']), GT_removed_from_forward=True, loader_accesses_annotations=True,
                matching_or_loss_executed=False, checkpoint=spec['checkpointmap'][mode], CP_qualified=True, head_seed=seed,
                external_RNG_before=[pref.digest_tensor(value) for value in external],
                external_RNG_after=[pref.digest_tensor(value) for value in after_rng], external_RNG_restored=True,
                private_RNG_before=[pref.digest_tensor(value) for value in private_before],
                private_RNG_after=[pref.digest_tensor(value) for value in private_after], final_fields=final_fields,
                native_fields=native_fields, native_class_packet=packet, native_hook=hook_record,
                official_instance_postprocess=True, source_native_query_ID_mapping_verified=True,
                official_masks_query_ID_bitwise_equal=True, official_scores_query_ID_bitwise_equal=True,
                prediction_query_ids=query_ids, predictions=100, query_score_factors_verified=True,
                class_probability_mask_quality_product_TorchEqual_official_score=True, all_native_state610_exact=True,
                native_gradNone=True, all_modules_eval=True, hook_cache_clear=True, added_NMS=False,
                added_score_cutoff=False, query_filter=False, inference_ablation_not_training_arm=False,
                class_or_query_identity_does_not_imply_equal_mask_quality_or_scores=True))
            if (index + 1) % 25 == 0 or index == 186:
                print(json.dumps(dict(status='DEV_ENDPOINT_PROGRESS', mode=mode, evaluated_images=index + 1,
                    elapsed_seconds=time.monotonic() - started)), flush=True)
            del data, samples, features, classes, masks, identity, external, after_rng, valid, native_fields
            del image_predictions, image_factors, query_ids, packet, final_fields, common, hook_record, scope
        after = pref.state_proof(model.state_dict())
        require(before == after and all(torch.equal(value.detach().cpu(), state[name]) for name, value in model.state_dict().items()) and
                all(torch.equal(model.state_dict()[name].detach().cpu(), value) for name, value in teacher.items()) and
                all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()) and
                all(not module.training for module in model.modules()) and core.hook_inventory(model) == baseline_hooks,
                'Evaluation changed trained endpoint/native610/parameter grad/eval/hook state')
        require(len(rows) == 187 and len(predictions) == len(factors) == 18700 and
                len({row['image_id'] for row in rows}) == 187 and sum(row['annotation_count'] for row in rows) == 380,
                'Complete DEV187/380/all100 population required')
        core.write(directory / 'IMAGE_ROWS.json', rows)
        core.write(directory / 'ALL_PREDICTIONS.json', predictions)
        core.write(directory / 'QUERY_SCORE_FACTORS.json', factors)
        core.write(directory / 'STATE_PROOF.json', dict(schema=STATE_SCHEMA, mode=mode, checkpoint_mode=mode,
            checkpoint=spec['checkpointmap'][mode], qualification=spec['qualificationmap'][mode], loading_proof=loading,
            native_teacher610=pref.state_proof(teacher), before=before, after=after, full_native_and_adapter_state_exact=True,
            params_gradNone=True, all_parameters_frozen=True, all_modules_eval=True, hook_cache_clear=True,
            frozen_teacher610_exact=True, registered_state_not_pruned=mode != 'native0',
            optimizer_or_RNG_restored=False, same_CP_inference_ablation=False))
        return dict(mode=mode, checkpoint_mode=mode, checkpoint=spec['checkpointmap'][mode], qualification=spec['qualificationmap'][mode],
            completed_epoch=meta['completed_epoch'], CP_qualified=True, image_rows=reference(directory / 'IMAGE_ROWS.json'),
            predictions=reference(directory / 'ALL_PREDICTIONS.json'), query_score_factors=reference(directory / 'QUERY_SCORE_FACTORS.json'),
            state_proof=reference(directory / 'STATE_PROOF.json'), images=187, raw_GT=380, predictions_count=18700,
            query_score_factors_count=18700, fullstate_tensors=len(state), full_native_state_exact=True, adapter_state_exact=True,
            params_gradNone=True, frozen_teacher610_exact=True, loader_accesses_annotations=True, GT_removed_from_forward=True,
            native_official_fusion=True, query_score_factors_verified=True, source_native_query_ID_mapping_verified=True,
            class_probability_mask_quality_product_TorchEqual_official_score=True, cross_model_query_ID_object_identity_claimed=False,
            cross_model_factor_exchange=False, inference_ablation_not_training_arm=False, registered_state_not_pruned=mode != 'native0',
            added_NMS=False, added_score_cutoff=False, peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
    finally:
        model.to('cpu')


def run(spec_file):
    started = time.monotonic()
    require(Path(spec_file).is_absolute(), 'Absolute --spec required')
    spec_path = Path(spec_file).resolve(strict=True)
    spec = read(spec_path)  # Metadata only; not hashed yet.
    receipts = {}
    for role, phase in (('teacher_receipt', 'teacher_phase'), ('preflight_receipt', 'preflight_phase'),
                        ('short2_receipt', 'short2_phase')):
        require(Path(spec[role]['path']).is_absolute(), 'Absolute original raw receipt required')
        receipts[role] = read(spec[role]['path'])
        raw_terminal(receipts[role], spec[phase])
    # All original held owners terminal0 before FIRST SOURCE/SPEC/Q/CP/GT hash/import.
    require(spec['schema'] == SCHEMA and spec['phase'].startswith('query_conditioned_frozen_native_short2_evaluation_') and
            spec['modes'] == list(MODES) and spec['endpoints'] == list(ENDPOINTS) and
            set(spec['checkpointmap']) == set(spec['qualificationmap']) == set(ENDPOINTS) and
            spec['expected_epoch'] == 24 and spec['short2_epoch'] == 2 and spec['head_seed_base'] == 100000 and
            spec['evaluation_images'] == 187 and spec['evaluation_GT'] == 380 and spec['CPU_affinity'] == [0, 1] and
            spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_owner_limit_MiB'] == 8192 and
            spec['deadline_seconds'] == 1800 and os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Exact five-endpoint DEV/resource scope required')
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and Path(spec['image_root']).is_absolute() and
            Path(spec['GPU_admission_file']).is_absolute(), 'Fresh absolute output/current admission/image root required')
    spec_ref = owner_gate(spec_path, spec)
    for role, size, sha in (('short2_source', 48366, SHORT2_SHA), ('short2_CPU_reader', 56265, SHORT2_READER_SHA),
                           ('official_evaluation_source', 40816, OFFICIAL_SOURCE_SHA),
                           ('official_evaluation_CPU_reader', 49923, OFFICIAL_READER_SHA)):
        require(spec[role] in spec['sources'] and spec[role]['bytes'] == size and spec[role]['sha256'] == sha,
                'Exact immutable qualified SOURCE role differs: ' + role)
    require(spec['official_evaluation_source'] == spec['evaluation_source'], 'Reuse original official E loader SOURCE alias')
    short_source = load_source(spec['short2_source'], 'sealed_actual_SHORT2_lineage_helpers')
    short_source.preflight_gate(spec, receipts['preflight_receipt'])
    short_train, short_q = qualified_short2(spec, receipts['short2_receipt'])
    core = load_source(spec['preflight_source'], 'sealed_actual_query_preflight_helpers')
    teacher_row = core.qualified_teacher(spec, receipts['teacher_receipt'])
    core.fixture_gate(spec)
    output.mkdir(parents=True)
    core.write(output / 'START.json', dict(schema='query-conditioned-frozen-native-short2-evaluation-start-v1',
        phase=spec['phase'], source=spec['entry_source'], spec=spec_ref, endpoints=list(ENDPOINTS), modes=list(MODES),
        expected_epoch=24, short2_epoch=2, evaluation_images=187, evaluation_GT=380,
        checkpointmap=spec['checkpointmap'], qualificationmap=spec['qualificationmap'],
        teacher_qualification=spec['teacher_qualification'], short2_CPU_qualification=spec['short2_CPU_qualification'],
        short2_training_spec=spec['short2_training_spec'], original_three_terminal_gates_before_scientific_access=True,
        all_five_CPU_models_before_CUDA_required=True, no_TEST=True, no_AP=True, no_retry=True,
        optimizer_or_RNG_restored=False, official_metrics_pending_independent_CPU_reader=True))
    import psutil
    import numpy as np
    import torch
    from pycocotools import mask as masks_api
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'Actual CPU01/cold CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 2, 'Actual torch2/2 threads required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    pref = load_source(spec['micro_preflight_source'], 'query_DEV_native_pref')
    initializer = load_source(spec['initializer_source'], 'query_DEV_native_initializer')
    old = load_source(spec['native_helpers'], 'query_DEV_native_images'); old.torch = torch
    native = load_source(spec['native_preflight_helpers'], 'query_DEV_native_image_shim')
    official = load_source(spec['official_evaluation_source'], 'query_DEV_official_E24_loader')
    readout = load_source(spec['readout_source'], 'query_DEV_readout')
    geometry = load_source(spec['geometry_source'], 'query_DEV_geometry')
    hooks = load_source(spec['hooks_source'], 'query_DEV_whole_head_bridge')
    classes = tuple(CocoDataset.METAINFO['classes'])
    require(len(classes) == 80 and classes[8] == 'boat' and tuple(readout.MODES) == tuple(hooks.MODES) == MODES,
            'Original COCO80 order and actual query readout modes required')
    prepared, teacher = {}, None
    for mode in ENDPOINTS:
        pref.check_deadline(started, spec['deadline_seconds'])
        prepared[mode], native_state = prepare_cpu(torch, core, pref, initializer, official, readout, geometry, hooks,
            MODELS, revert_sync_batchnorm, Config, classes, spec, teacher_row, short_q, mode, teacher)
        if teacher is None:
            teacher = native_state
        require(not torch.cuda.is_initialized(), 'All five CPU preparations must precede CUDA')
        del native_state
    dataset = CocoDataset(ann_file=str(bound(spec['DEV_annotation'])), data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
    coco = COCO(str(bound(spec['DEV_annotation'])))
    require(len(dataset) == len(coco.imgs) == 187 and len(coco.anns) == 380 and len(prepared) == 5 and
            not torch.cuda.is_initialized(), 'Complete DEV187/380 and all five CPU models required before CUDA')
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(4096 * 1024 ** 2 / torch.cuda.get_device_properties(0).total_memory, 0)
    results, cleanup, shared = {}, {}, {}
    for mode in ENDPOINTS:
        current = prepared.pop(mode)
        results[mode] = evaluate_endpoint(torch, np, masks_api, core, pref, old, native, hooks, spec, current,
            teacher, dataset, coco, output, mode, shared, started)
        del current
        cleanup[mode] = pref.clear_gpu(torch)
    report = dict(schema=REPORT_SCHEMA, phase=spec['phase'], source=spec['entry_source'], spec=spec_ref,
        status='QUERY_CONDITIONED_FROZEN_NATIVE_SHORT2_DEV_EXPORT_COMPLETE_NO_AP', error=None,
        endpoints=list(ENDPOINTS), modes=list(MODES), mode_rows=results, checkpointmap=spec['checkpointmap'],
        qualificationmap=spec['qualificationmap'], teacher_qualification=spec['teacher_qualification'],
        short2_CPU_qualification=spec['short2_CPU_qualification'], short2_training_spec=spec['short2_training_spec'],
        evaluation_images=187, evaluation_GT=380, full_teacher610_exact_all_endpoints=True,
        all_five_CPU_models_before_CUDA=True, only_one_model_on_CUDA_at_a_time=True,
        native_official_fusion=True, all100_predictions_per_image=True, GT_removed_from_forward=True,
        no_TEST=True, no_AP=True, AP_computed=False, official_metrics_pending_independent_CPU_reader=True,
        mask_quality_not_independently_CPU_recomputed_from_omitted_logits=True,
        class_or_query_identity_does_not_imply_equal_mask_quality_or_scores=True,
        no_added_NMS_or_score_cutoff=True, optimizer_or_RNG_restored=False,
        efficacy_claimed=False, scientific_method_decision_computed=False, cleanup=cleanup,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), elapsed_seconds=time.monotonic() - started,
        allocator_limit_MiB=4096, GPU_scope='whole GPU must be verified from actual owner receipt')
    report_path = output / 'REPORT.json'
    core.write(report_path, report)
    core.write(output / 'QUALIFIED.json', dict(schema='query-conditioned-frozen-native-short2-evaluation-aggregate-v1',
        status='ROOT_QUERY_CONDITIONED_FROZEN_NATIVE_SHORT2_EVALUATION_PENDING_CPU_READER', source=spec['entry_source'],
        spec=spec_ref, report=reference(report_path), mode_rows=results, short2_CPU_qualification=spec['short2_CPU_qualification'],
        independently_CPU_qualified=False, AP_computed=False, efficacy_claimed=False))
    print(json.dumps(dict(status='DEV_EXPORT_COMPLETE_CPU_QUALIFICATION_PENDING', endpoints=list(ENDPOINTS))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Inert SOURCE requires explicit ROOT --owner-execute')
    try:
        run(args.spec)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
