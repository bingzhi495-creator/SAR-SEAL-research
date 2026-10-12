"""Inert evaluation of the prelocked additional24/nativeE48 final endpoint.

This loader accepts the NEW native fresh-optimizer qualification/checkpoint
schemas. It does not route nativeE48 through the old E24 endpoint loader.
Only DEV187 is evaluated, once, with the unchanged all100 native fusion and
renderer. No training, optimizer/RNG restore, node selection, score cutoff,
NMS addition or GPU AP is performed. ROOT alone executes after admission.
Per-image final100 class/M200 packets support independent bounded CPU checks;
original GPU raster logits are not retained, so full pixelwise CUDA replay
or a certified interpolation error bound is not claimed.
"""
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import traceback
import types

SCHEMA = 'ROOT-native-finetune24-finalE48-evaluation-spec-v1'
REPORT_SCHEMA = 'native-finetune24-finalE48-evaluation-report-v1'
REPORT_STATUS = 'NATIVE_FINETUNE24_FINAL_E48_EVALUATION_COMPLETE_NO_GPU_AP'
PROPOSAL_KIND = 'ROOT_INERT_NATIVE_FINETUNE24_FINAL_E48_EVALUATION_SOURCE_PROPOSAL_NOT_SCIENCE_ADMISSION'
NATIVE_SCHEMA = 'ROOT-native-finetune24-fresh-optimizer-spec-v1'
NATIVE_Q_SCHEMA = 'ROOT-native-finetune24-fresh-optimizer-CPU-qualification-v1'
NATIVE_Q_STATUS = 'ROOT_NATIVE_FINETUNE24_FRESH_OPTIMIZER_CPU_QUALIFIED'
CP_SCHEMA = 'native-finetune24-fresh-optimizer-checkpoint-v1'
ENDPOINT = 'nativeE48'
LEGACY_MODES = ('native0', 'global', 'spatial_halves', 'spatial_reversed')
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
GT_FIELDS = ('gt_instances', 'ignored_instances', 'gt_sem_seg', 'proposals')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'} and
        Path(row['path']).is_absolute() and type(row['bytes']) is int and row['bytes'] > 0 and
        isinstance(row['sha256'], str) and len(row['sha256']) == 64 and
        set(row['sha256']) <= set('0123456789abcdef'), 'Exact absolute pinned reference required')
    require(reference(row['path']) == row, 'Pinned input differs: ' + row['path'])
    return Path(row['path'])


def read_ref(row):
    return read(bound(row))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def load_source(row, name):
    path = bound(row)
    require(path.suffix == '.py', 'Bound Python SOURCE required')
    source = path.read_bytes()
    require(len(source) == row['bytes'] and hashlib.sha256(source).hexdigest() == row['sha256'],
        'SOURCE changed immediately before import')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def terminal(raw, phase):
    require(raw.get('status') == 'ACTUAL_OWNED_TERMINAL' and raw.get('original_identity') is not None and
        type(raw.get('returncode')) is int and raw['returncode'] == 0 and raw.get('root_wait_reaped') is True and
        raw.get('held_handle_exit') == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        raw.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
        all(key in raw and raw[key] is None for key in ('abort', 'error')) and raw.get('wait_errors') is False and
        raw.get('GPU_observation_errors') == [], 'STOP before hashes/imports: owned raw terminal differs')
    peak = raw.get('whole_GPU_observed_peak_MiB')
    require(raw.get('phase', phase) == phase and type(peak) in (int, float) and math.isfinite(peak) and
        0 <= peak <= 8192 and raw.get('whole_GPU_used_limit_MiB') == 8192 and
        raw.get('whole_leaf_VRAM_certified') is False and
        raw.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation', 'Owner resource scope differs')


def source_gate(spec_path, spec):
    # Four original raws + new technical raw + new formal raw precede ALL hashes.
    require(set(spec['training_receiptmap']) == set(spec['training_phasemap']) == set(LEGACY_MODES),
        'Original four raw identities required')
    raw = {}
    for mode in LEGACY_MODES:
        require(Path(spec['training_receiptmap'][mode]['path']).is_absolute(), 'Absolute original raw required')
        raw[mode] = read(spec['training_receiptmap'][mode]['path'])
        terminal(raw[mode], spec['training_phasemap'][mode])
    for label, receipt_key, phase_key in (('restoration', 'restoration_receipt', 'restoration_phase'),
            ('native_phase', 'native_phase_receipt', 'native_phase_phase')):
        require(Path(spec[receipt_key]['path']).is_absolute(), 'Absolute new native raw required')
        raw[label] = read(spec[receipt_key]['path'])
        terminal(raw[label], spec[phase_key])
    require(Path(spec['GPU_admission_file']).is_absolute(), 'Absolute current owner admission required')
    admission = read(spec['GPU_admission_file'])
    source, spec_ref = reference(__file__), reference(spec_path)
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        admission['automatic_retry'] is False and admission['phase'] == spec['phase'] and
        admission['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and admission['cuda_visible_devices'] == '0' and
        admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
        admission['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192 and
        Path(admission['entry']).resolve(strict=True) == Path(source['path']) and
        admission['entry_args'] == ['--spec', str(spec_path), '--owner-execute'] and
        admission['driver_sha256'] == OWNER_SHA, 'Current entry/admission/owner/resources differ')
    require(spec['entry_source'] == source and source in spec['sources'] and spec['CPU_reader'] in spec['sources'] and
        len({r['path'] for r in spec['sources']}) == len(spec['sources']), 'Exact fresh entry/reader closure required')
    hashes = admission['bound_source_sha256']
    require(hashes.get(str(spec_path)) == spec_ref['sha256'], 'Evaluation SPEC is not owner-bound')
    for ref in spec['sources']:
        bound(ref)
        require(hashes.get(ref['path']) == ref['sha256'], 'Evaluation complete closure is not owner-bound')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Current admitted SPEC differs')
    proposal = read_ref(spec['SOURCE_proposal'])
    require(proposal['kind'] == PROPOSAL_KIND and proposal['schema'] == SCHEMA and
        proposal['producer_SOURCE'] == source and spec['SOURCE_proposal'] in spec['sources'] and
        proposal['endpoint'] == ENDPOINT and spec['DEV'] == proposal['canonical_DEV'], 'Canonical evaluation proposal differs')
    for role, ref in dict(proposal['immutable_SOURCE_refs'], **proposal['library_SOURCE_refs']).items():
        require(spec[role] == ref and ref in spec['sources'], 'Canonical immutable SOURCE role differs: ' + role)
    return raw, source, spec_ref, reference(spec['GPU_admission_file'])


def qualified_native_phase(spec, raw):
    q = read_ref(spec['native_phase_qualification'])
    require(q['schema'] == NATIVE_Q_SCHEMA and q['status'] == NATIVE_Q_STATUS and
        q['execution_kind'] == 'formal_additional24' and q['formal_additional24_completed'] is True and
        q['formal_training_completion_CPU_state_qualified'] is True and q['formal_resume_qualified'] is False and
        q['completed_phase_epochs'] == 24 and q['effective_native_epoch'] == 48 and
        q['phase_microsteps'] == 17784 and q['phase_updates'] == 8892 and q['phase_raw_GT_visits'] == 39864 and
        q['source'] == spec['native_phase_source'] and q['reader_source'] == spec['native_phase_CPU_reader'] and
        q['spec'] == spec['native_phase_spec'] and q['parent_receipt'] == spec['native_phase_receipt'] and
        q['phase'] == spec['native_phase_phase'] and q['checkpoint'] == spec['native_phase_checkpoint'] and
        q['technical_CPU_snapshot'] is None and q['formal_resume_eligible'] is False and
        q['old_optimizer_restored'] is False and q['old_RNG_restored'] is False and
        q['initial_native610_CPU_exact'] is True and q['typed_BN106_affine159_buffers_exact'] is True and
        q['native345_active'] is True and q['fresh_optimizer_state_entries'] == 0 and q['native_loss_atoms'] == 30 and
        q['full_native_CPU_states_and_AdamW_moments_verified'] is True and q['inference_executed'] is False and
        q['AP_computed'] is False, 'Exact NEW formal nativeE48 CPU qualification required; technical/oldE24 refused')
    require(q['CPU_execution'] == dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2,
        torch_interop_threads=2, cuda_visible_devices='', CUDA_initialized=False), 'Actual native CPU resource scope differs')
    cpu_terminal = spec['native_CPU_tool_terminal']
    require(set(cpu_terminal) == {'session_id', 'exit_code'} and type(cpu_terminal['session_id']) is int and
        cpu_terminal['session_id'] > 0 and type(cpu_terminal['exit_code']) is int and cpu_terminal['exit_code'] == 0,
        'ROOT actual native CPU tool terminal0 metadata required')
    for role in ('source', 'reader_source', 'spec', 'report', 'start', 'parent_receipt', 'GPU_admission', 'owner_driver'):
        bound(q[role])
    train, admission = read_ref(q['spec']), read_ref(q['GPU_admission'])
    require(train['schema'] == NATIVE_SCHEMA and train['execution_kind'] == 'formal_additional24' and
        train['entry_source'] == q['source'] and train['CPU_reader'] == q['reader_source'] and train['phase'] == q['phase'] and
        train['restoration_receipt'] == spec['restoration_receipt'] and train['restoration_phase'] == spec['restoration_phase'] and
        train['checkpointmap'] == spec['checkpointmap'] and train['qualificationmap'] == spec['qualificationmap'] and
        train['training_receiptmap'] == spec['training_receiptmap'] and train['training_phasemap'] == spec['training_phasemap'] and
        q['teacher_checkpoint'] == spec['checkpointmap']['native0'] and q['teacher_qualification'] == spec['qualificationmap']['native0'],
        'New formal phase and original E24/technical lineage differ')
    require(raw['native_phase']['admission_sha256'] == q['GPU_admission']['sha256'] and
        admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and admission['automatic_retry'] is False and
        admission['phase'] == q['phase'] and admission['allowed_cpus'] == train['CPU_affinity'] == [0, 1] and
        admission['cuda_visible_devices'] == '0' and admission['qualified_leaf_deadline_seconds'] == train['deadline_seconds'] and
        admission['whole_GPU_used_limit_MiB'] == train['whole_GPU_owner_limit_MiB'] == 8192 and
        Path(admission['entry']).resolve(strict=True) == Path(q['source']['path']) and
        admission['entry_args'] == ['--spec', q['spec']['path'], '--owner-execute'] and
        admission['driver_sha256'] == q['owner_driver']['sha256'] == OWNER_SHA,
        'Actual new native formal raw/admission/entry/SPEC/owner join differs')
    hashes = admission['bound_source_sha256']
    require(hashes.get(q['spec']['path']) == q['spec']['sha256'], 'New native formal SPEC is not owner-bound')
    for ref in train['sources']:
        bound(ref)
        require(ref in spec['sources'] and hashes.get(ref['path']) == ref['sha256'], 'Complete native formal closure not preserved/bound')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == q['spec'], 'New native formal admitted SPEC differs')
    require(set(q['nodes']) == {'2', '4', '8', '12', '18', '24'} and q['nodes']['24']['checkpoint'] == q['checkpoint'] and
        q['nodes']['24']['completed_phase_epoch'] == 24 and q['nodes']['24']['effective_native_epoch'] == 48 and
        q['nodes']['24']['full_native610_CPU_verified'] is True and
        q['nodes']['24']['typed_BN106_affine159_buffers_exact'] is True, 'Only fixed final native phase24 node admitted')
    report, start = read_ref(q['report']), read_ref(q['start'])
    require(report['schema'] == 'native-finetune24-fresh-optimizer-report-v1' and
        report['status'] == 'NATIVE_FINETUNE24_FRESH_OPTIMIZER_OWNED_COMPLETE_NO_AP' and report['error'] is None and
        report['source'] == q['source'] and report['spec'] == q['spec'] and report['phase'] == q['phase'] and
        report['execution_kind'] == 'formal_additional24' and report['GPU_admission'] == q['GPU_admission'] and
        report['cleanup']['allocated_after'] == 0 and report['result']['checkpoint'] == q['checkpoint'] and
        start['source'] == q['source'] and start['spec'] == q['spec'] and start['phase'] == q['phase'],
        'New native actual completed report/START/checkpoint join differs')
    return dict(qualification=q, training_spec=train)


def prepare_model(torch, pref, initializer, registry, revert, Config, classes, spec, row):
    train, q = row['training_spec'], row['qualification']
    original_train = row['original_training_spec']
    initialized = read_ref(original_train['initializer_qualification'])
    require(initialized['schema'] == 'ROOT-native-coco-comparator-initializer-CPU-inspection-v1' and
        initialized['all_initializers_strictly_compatible'] is True and initialized['source'] == spec['initializer_source'],
        'Original strict native initializer qualification differs')
    construction = dict(original_train, backbone_with_cp=False)
    model, initial, initialization = pref.prepare_cpu(torch, initializer, registry, revert, Config, classes,
        construction, 'mask2former', initialized['initializers']['mask2former'])
    cp = torch.load(str(bound(spec['native_phase_checkpoint'])), map_location='cpu')
    require(set(cp) == {'state_dict', 'optimizer_state_dict', 'external_rng_state', 'private_order_state', 'meta'},
        'Complete NEW native phase24 checkpoint required')
    state, meta = cp['state_dict'], cp['meta']
    require(meta['schema'] == CP_SCHEMA and meta['kind'] == 'mask2former' and meta['mode'] == 'native0' and
        meta['execution_kind'] == 'formal_additional24' and meta['source'] == q['source'] and meta['spec'] == q['spec'] and
        meta['phase'] == q['phase'] and meta['teacher_checkpoint'] == q['teacher_checkpoint'] and meta['teacher_epoch'] == 24 and
        meta['phase_initial_epoch'] == 0 and meta['completed_phase_epoch'] == 24 and meta['effective_native_epoch'] == 48 and
        meta['phase_microsteps'] == 17784 and meta['phase_updates'] == 8892 and meta['phase_raw_GT_visits'] == 39864 and
        meta['effective_native_microsteps'] == 35568 and meta['effective_native_updates'] == 17784 and
        meta['effective_native_raw_GT_visits'] == 79728 and meta['phase_recipe'] == train['initializers']['mask2former'] and
        meta['schedule'] == train['schedule'] and meta['fresh_optimizer_at_phase_start'] is True and
        meta['old_optimizer_restored'] is False and meta['old_RNG_restored'] is False and
        meta['no_pending_accumulation_gradients'] is True and meta['formal_resume_eligible'] is False,
        'Only NEW full formal native phase24/E48 metadata eligible')
    current = model.state_dict()
    require(len(state) == len(current) == 610 and len(list(model.parameters())) == 451 and state.keys() == current.keys() and
        all(isinstance(v, torch.Tensor) and v.device.type == 'cpu' and v.shape == current[n].shape and v.dtype == current[n].dtype and
            bool(torch.isfinite(v).all()) for n, v in state.items()), 'Exact finite full native610 shapes/dtypes required')
    node_proof = read_ref(q['nodes']['24']['state_proof'])
    require(pref.state_proof(state) == node_proof['after'], 'Loaded nativeE48 state differs from independent CPU qualified proof')
    loaded = model.load_state_dict(state, strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys and
        all(torch.equal(v.detach().cpu(), state[n]) for n, v in model.state_dict().items()), 'Strict full nativeE48 load differs')
    bn_params, bn_buffers = [], []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            bn_params.extend(name + '.' + local for local, _ in module.named_parameters(recurse=False))
            bn_buffers.extend(name + '.' + local for local, _ in module.named_buffers(recurse=False))
    require(len(bn_params) == 106 and len(bn_buffers) == 159 and
        all(torch.equal(state[n], initial[n]) for n in bn_params + bn_buffers), 'TypedBN initializer state differs')
    model.requires_grad_(False).eval()
    head, fusion = model.panoptic_head, model.panoptic_fusion_head
    require(head.num_queries == 100 and head.num_classes == 1 and
        fusion.num_classes == fusion.num_things_classes == 1 and fusion.num_stuff_classes == 0 and
        fusion.test_cfg.get('max_per_image', 100) == 100 and fusion.test_cfg.get('instance_on', False) is True and
        fusion.test_cfg.get('panoptic_on', True) is False and fusion.test_cfg.get('semantic_on', False) is False,
        'Original all100 single-class instance fusion required')
    for role, obj in dict(native_mask_head_SOURCE=type(head), native_parent_head_SOURCE=type(head).__mro__[1],
            native_fusion_SOURCE=type(fusion), interpolation_SOURCE=torch.nn.functional.interpolate).items():
        require(Path(inspect.getsourcefile(obj)).resolve() == bound(spec[role]).resolve(), 'Actual native inference SOURCE differs: ' + role)
    require(torch.__version__ == '2.0.1+cu118' and
        (Path(torch.__file__).resolve().parent / 'version.py').resolve() == bound(spec['torch_version_source']).resolve(),
        'Bound Torch version differs')
    proof = dict(schema='native-finetune24-finalE48-loading-proof-v1', initialization=initialization,
        checkpoint=spec['native_phase_checkpoint'], qualification=spec['native_phase_qualification'], endpoint=ENDPOINT,
        strict_fullstate_loaded=True, native_state_tensors=610, registered_parameters=451, registered_adapter_tensors=0,
        trained_fullstate=pref.state_proof(state), initial_native_state=pref.state_proof(initial),
        BN_parameter_names=bn_params, BN_buffer_names=bn_buffers, original_frozen_BN_exact=True,
        completed_phase_epoch=24, effective_native_epoch=48, inference_backbone_with_cp=False,
        training_backbone_with_cp=train['backbone_with_cp'], optimizer_or_RNG_restored_for_inference=False,
        checkpoint_optimizer_and_RNG_containers_deserialized=True, no_random_missing_keys=True)
    del cp, current, initial, initialized, node_proof
    return model, state, proof


def evaluate_endpoint(torch, np, masks_api, pref, old, spec, spec_ref, prepared, dataset, coco, output, started):
    model, state, loading = prepared
    directory = output / ENDPOINT
    directory.mkdir()
    packet_dir = directory / 'FINAL_FIELDS'
    packet_dir.mkdir()
    model.to('cuda:0').eval()
    before = pref.state_proof(model.state_dict())
    require(before == loading['trained_fullstate'], 'Evaluation actual START native610 differs')
    image_rows, predictions, factors = [], [], []
    try:
        for index in range(187):
            pref.check_deadline(started, spec['deadline_seconds'])
            data, samples, gray, raw, identity = old.load_image(model, dataset, coco, dataset.get_data_info(index))
            require(len(samples) == 1 and not any(key in samples[0] for key in GT_FIELDS), 'GT/proposals survived forward stripping')
            external = pref.rng_snapshot(torch)
            seed = 100000 + identity['image_id']
            with torch.random.fork_rng(devices=[0]):
                torch.random.default_generator.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                with torch.no_grad():
                    features = model.extract_feat(data['inputs'])
                    classes, mask_fields = model.panoptic_head(features, samples)
                    require(len(classes) == len(mask_fields) == 10 and
                        all(tuple(v.shape) == (1,100,2) and v.dtype == torch.float32 and not v.requires_grad and bool(torch.isfinite(v).all())
                            for v in classes) and
                        all(tuple(v.shape) == (1,100,200,200) and v.dtype == torch.float32 and not v.requires_grad and bool(torch.isfinite(v).all())
                            for v in mask_fields), 'Expected ten finite FP32 native readouts')
                    final_class, final_masks = classes[-1], mask_fields[-1]
                    final_digests = dict(classes=pref.digest_tensor(final_class), masks=pref.digest_tensor(final_masks))
                    del classes, mask_fields, features
                    ih, iw = identity['actual_img_shape']
                    oh, ow = identity['original_shape']
                    require(0 < ih <= 800 and 0 < iw <= 800 and identity['batch_input_shape'] == [800,800], 'Native fixed800 geometry differs')
                    padded = torch.nn.functional.interpolate(final_masks, size=(800,800), mode='bilinear', align_corners=False)
                    instances = model.panoptic_fusion_head.predict(final_class, padded, samples, rescale=True)[0]['ins_results']
                    rendered = torch.nn.functional.interpolate(padded[0,:,:ih,:iw][:,None], size=(oh,ow),
                        mode='bilinear', align_corners=False)[:,0]
                    probabilities = final_class[0].softmax(-1)[:, :-1]
                    top_scores, top_indices = probabilities.flatten(0,1).topk(100, sorted=False)
                    query_indices = top_indices // model.panoptic_fusion_head.num_classes
                    require(torch.equal(torch.sort(query_indices).values, torch.arange(100,device=query_indices.device)) and
                        torch.equal(top_scores, probabilities[query_indices,0]), 'All100 original query ordinal mapping differs')
                    ordered_logits = rendered[query_indices]
                    binary = (ordered_logits > 0).float()
                    quality = (ordered_logits.sigmoid() * binary).flatten(1).sum(1) / (binary.flatten(1).sum(1) + 1e-6)
                    scores = probabilities[query_indices,0] * quality
                    require(len(instances) == 100 and torch.equal(instances.masks, binary.bool()) and
                        torch.equal(instances.scores, scores) and bool((instances.labels == 0).all()) and
                        bool(torch.isfinite(scores).all()) and bool(((scores >= 0) & (scores <= 1)).all()),
                        'Unchanged official masks/scores/labels differ from independently derived native query mapping')
                    query_ids = query_indices.cpu().tolist()
                    binary_cpu = instances.masks.cpu().numpy().astype(np.uint8)
                    score_values = scores.cpu().tolist()
                    class_values = probabilities[query_indices,0].cpu().tolist()
                    quality_values = quality.cpu().tolist()
                    fields = dict(schema='native-finetune24-finalE48-final-fields-v1', source=reference(__file__), spec=spec_ref,
                        checkpoint=spec['native_phase_checkpoint'], endpoint=ENDPOINT, dataset_index=index,
                        image_id=identity['image_id'], original_shape=identity['original_shape'], actual_img_shape=identity['actual_img_shape'],
                        final_class_logits=final_class.detach().cpu().clone(), final_mask_logits200=final_masks.detach().cpu().clone(),
                        final_field_digests=final_digests, prediction_query_ids=query_ids,
                        GPU_original_logits_saved=False, all100_fields_saved=True)
                    packet_path = packet_dir / ('IMAGE_%04d.pth' % index)
                    with packet_path.open('xb') as stream:
                        torch.save(fields, stream)
                    readback = torch.load(packet_path,map_location='cpu')
                    require(pref.tree_equal(torch, fields, readback), 'Final100 CPU field packet readback differs')
                    packet_ref = reference(packet_path)
                    del fields, readback, final_class, final_masks, padded, rendered, probabilities
                    del top_scores, top_indices, query_indices, ordered_logits, binary, quality, scores, instances
            after_rng = pref.rng_snapshot(torch)
            require(all(torch.equal(a,b) for a,b in zip(external,after_rng)), 'Head private RNG did not restore external states')
            for query_id, mask, score, cp, mq in zip(query_ids,binary_cpu,score_values,class_values,quality_values):
                rle = masks_api.encode(np.asfortranarray(mask))
                counts = rle['counts'].decode('ascii') if isinstance(rle['counts'],bytes) else rle['counts']
                predictions.append(dict(image_id=int(identity['image_id']),query_id=int(query_id),category_id=1,
                    score=float(score),segmentation=dict(size=list(rle['size']),counts=counts)))
                factors.append(dict(image_id=int(identity['image_id']),query_id=int(query_id),class_probability=float(cp),
                    mask_quality=float(mq),official_score=float(score)))
            image_rows.append(dict(endpoint=ENDPOINT,dataset_index=index,image_id=identity['image_id'],image=identity['image'],
                ann_ids=identity['ann_ids'],annotation_count=len(identity['ann_ids']),original_shape=identity['original_shape'],
                actual_img_shape=identity['actual_img_shape'],batch_input_shape=[800,800],scale_factor=identity['scale_factor'],
                checkpoint=spec['native_phase_checkpoint'],qualification=spec['native_phase_qualification'],CP_qualified=True,
                GT_removed_from_forward=True,loader_accesses_annotations=True,matching_or_loss_executed=False,
                head_seed=seed,external_RNG_before=[pref.digest_tensor(v) for v in external],
                external_RNG_after=[pref.digest_tensor(v) for v in after_rng],external_RNG_restored=True,
                final_fields=final_digests,final_field_packet=packet_ref,prediction_query_ids=query_ids,predictions=100,
                official_instance_postprocess=True,source_native_query_ID_mapping_verified=True,
                official_masks_query_ID_bitwise_equal=True,official_scores_query_ID_bitwise_equal=True,
                query_score_factors_verified=True,class_probability_mask_quality_product_TorchEqual_official_score=True,
                added_NMS=False,added_score_cutoff=False,query_filter=False,all100_fields_saved=True,
                GPU_original_logits_saved=False,historical_class_or_score_equality_assumed=False))
            if (index+1)%50 == 0 or index == 186:
                print(json.dumps(dict(endpoint=ENDPOINT,evaluated_images=index+1,elapsed_seconds=time.monotonic()-started)),flush=True)
            del data,samples,gray,raw,identity,external,after_rng,binary_cpu,score_values,class_values,quality_values
            del query_ids,mask
        after = pref.state_proof(model.state_dict())
        require(before == after and all(torch.equal(v.detach().cpu(),state[n]) for n,v in model.state_dict().items()) and
            all(not p.requires_grad and p.grad is None for p in model.parameters()) and
            all(not m.training for m in model.modules()), 'Evaluation changed native610 state/gradients/eval policy')
        require(len(image_rows) == 187 and len(predictions) == len(factors) == 18700 and
            len({r['image_id'] for r in image_rows}) == 187 and sum(r['annotation_count'] for r in image_rows) == 380,
            'Complete DEV187/380/all100 population differs')
        write(directory/'IMAGE_ROWS.json',image_rows)
        write(directory/'ALL_PREDICTIONS.json',predictions)
        write(directory/'QUERY_SCORE_FACTORS.json',factors)
        write(directory/'STATE_PROOF.json',dict(schema='native-finetune24-finalE48-evaluation-state-proof-v1',
            source=reference(__file__),spec=spec_ref,endpoint=ENDPOINT,loading_proof=loading,before=before,after=after,
            full_native610_state_exact=True,params_gradNone=True,all_parameters_frozen=True,all_modules_eval=True,
            no_adapter_registered=True,optimizer_or_RNG_restored=False))
        return dict(endpoint=ENDPOINT,checkpoint=spec['native_phase_checkpoint'],qualification=spec['native_phase_qualification'],
            completed_phase_epoch=24,effective_native_epoch=48,CP_qualified=True,images=187,raw_GT=380,predictions_count=18700,
            query_score_factors_count=18700,image_rows=reference(directory/'IMAGE_ROWS.json'),
            predictions=reference(directory/'ALL_PREDICTIONS.json'),query_score_factors=reference(directory/'QUERY_SCORE_FACTORS.json'),
            state_proof=reference(directory/'STATE_PROOF.json'),all100_fields_saved=True,final_field_packets=187,
            GPU_original_logits_saved=False,full_native610_state_exact=True,registered_adapter_tensors=0,params_gradNone=True,
            native_official_fusion=True,GT_removed_from_forward=True,loader_accesses_annotations=True,
            query_score_factors_verified=True,class_probability_mask_quality_product_TorchEqual_official_score=True,
            source_native_query_ID_mapping_verified=True,added_NMS=False,added_score_cutoff=False,query_filter=False,
            AP_computed_on_GPU=False,historical_class_or_score_equality_assumed=False)
    finally:
        model.to('cpu')


def run(spec_file):
    started = time.monotonic()
    require(Path(spec_file).is_absolute(), 'Absolute --spec required')
    spec_path = Path(spec_file).resolve(strict=True)
    spec = read(spec_path)
    require(spec['schema'] == SCHEMA and spec['kind'] == 'mask2former' and spec['endpoint'] == ENDPOINT and
        spec['completed_phase_epoch'] == 24 and spec['effective_native_epoch'] == 48 and
        spec['CPU_affinity'] == [0,1] and spec['allocator_limit_MiB'] == 4096 and
        spec['whole_GPU_owner_limit_MiB'] == 8192 and type(spec['deadline_seconds']) is int and
        0 < spec['deadline_seconds'] <= 10800 and os.environ.get('CUDA_VISIBLE_DEVICES') == '0',
        'Only prelocked nativeE48/CPU01/CUDA0/fresh owned resource scope supported')
    raw, source, spec_ref, admission_ref = source_gate(spec_path,spec)
    row = qualified_native_phase(spec,raw)
    legacy = load_source(spec['evaluation_source'],'original_native_teacher_lineage_ONLY')
    original = legacy.qualified_training(spec)['native0']
    row['original_training_spec'] = original['training_spec']
    train = row['training_spec']
    require(train['initializers'] == original['training_spec']['initializers'] and
        all(spec[k] == train[k] for k in ('image_root','native_helpers','initializer_source','micro_preflight_source',
            'training_source','training_CPU_reader','readout_source','geometry_source','hooks_source')),
        'Original/nativeE48 architecture/initializer/helper provenance differs')
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists(), 'Fresh absolute evaluation output required')
    output.mkdir(parents=True)
    write(output/'START.json',dict(schema='native-finetune24-finalE48-evaluation-start-v1',source=source,spec=spec_ref,
        phase=spec['phase'],endpoint=ENDPOINT,native_phase_checkpoint=spec['native_phase_checkpoint'],
        native_phase_qualification=spec['native_phase_qualification'],native_CPU_tool_terminal=spec['native_CPU_tool_terminal'],
        completed_phase_epoch=24,effective_native_epoch=48,no_training=True,no_GPU_AP=True,no_TEST=True,
        no_node_selection=True,no_score_threshold_selection=True,no_retry=True))
    import psutil
    import torch
    import numpy as np
    from pycocotools import mask as masks_api
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity() == [0,1] and not torch.cuda.is_initialized(), 'ROOT CPU01/coldCUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    require(len(CocoDataset.METAINFO['classes']) == 80 and CocoDataset.METAINFO['classes'][8] == 'boat','Original boat8/null80 mapping required')
    pref=load_source(spec['micro_preflight_source'],'original_native_preprocessing_AND_tensor_proofs')
    initializer=load_source(spec['initializer_source'],'original_strict_COCO_initializer')
    old=load_source(spec['native_helpers'],'original_native_image_helpers')
    old.torch=torch
    prepared=prepare_model(torch,pref,initializer,MODELS,revert_sync_batchnorm,Config,
        tuple(CocoDataset.METAINFO['classes']),spec,row)
    dataset=CocoDataset(ann_file=str(bound(spec['DEV'])),data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)),pipeline=[],filter_cfg=None,test_mode=True)
    coco=COCO(str(bound(spec['DEV'])))
    require(len(dataset)==len(coco.imgs)==187 and len(coco.anns)==380 and
        [int(dataset.get_data_info(i)['img_id']) for i in range(187)]==list(coco.imgs),'Official ordered DEV187/380 required')
    torch.cuda.init()
    capacity=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/capacity),0)
    torch.cuda.reset_peak_memory_stats(0)
    error,result,cleanup=None,None,None
    try:
        result=evaluate_endpoint(torch,np,masks_api,pref,old,spec,spec_ref,prepared,dataset,coco,output,started)
        del prepared
        cleanup=pref.clear_gpu(torch)
        for ref in spec['sources']:
            bound(ref)
        pref.check_deadline(started,spec['deadline_seconds'])
    except Exception as exception:
        error=dict(type=type(exception).__name__,message=str(exception),no_retry=True,no_fallback=True,
            OOM_technical_stop=isinstance(exception,torch.cuda.OutOfMemoryError))
        traceback.clear_frames(exception.__traceback__)
        if 'prepared' in locals():
            prepared[0].cpu()
            del prepared
        gc.collect()
        try:
            cleanup=pref.clear_gpu(torch)
        except Exception as cleanup_error:
            error['cleanup_error']=str(cleanup_error)
    write(output/'REPORT.json',dict(schema=REPORT_SCHEMA,status=REPORT_STATUS if error is None else
        'NATIVE_FINETUNE24_FINAL_E48_EVALUATION_STOP_NO_METHOD_RESULT',phase=spec['phase'],source=source,spec=spec_ref,
        GPU_admission=admission_ref,SOURCE_proposal=spec['SOURCE_proposal'],endpoint=ENDPOINT,result=result,error=error,cleanup=cleanup,
        native_phase_checkpoint=spec['native_phase_checkpoint'],native_phase_qualification=spec['native_phase_qualification'],
        native_CPU_tool_terminal=spec['native_CPU_tool_terminal'],elapsed_seconds=time.monotonic()-started,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),no_training=True,no_GPU_AP=True,no_TEST=True,
        no_node_selection=True,no_score_threshold_selection=True,no_automatic_retry=True,
        full_pixelwise_GPU_original_logit_replay_claimed=False,blind_DEV_claimed=False,
        historical_class_or_score_equality_assumed=False,matched_compute_claimed=False,matched_objective_claimed=False,
        efficacy_claimed=False,convergence_claimed=False,novelty_claimed=False,SOTA_claimed=False))
    if error is not None:
        raise RuntimeError('TECHNICAL_STOP: '+error['type']+': '+error['message'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args()
    require(args.owner_execute,'SOURCE_ONLY: explicit ROOT-owned execution required')
    run(args.spec)


if __name__=='__main__':
    main()
