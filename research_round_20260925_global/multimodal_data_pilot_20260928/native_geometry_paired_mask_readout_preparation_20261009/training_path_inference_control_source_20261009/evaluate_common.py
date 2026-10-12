"""Inert fresh five-endpoint residual-off evaluation; no mechanism verdict.

This entry loads each full native checkpoint, not a frozen warm-start adapter
checkpoint. It exports official original-resolution predictions and score
factors without computing AP, matching GT, filtering queries or exchanging
factors between models. Query IDs identify ordinals within one model only.
Both paired/fg_fine residual-off endpoints retain their own trained CP and
mode; they are inference ablations, never new training arms. apply=False skips
geometry/phi but leaves the adapter registered: neither compute-matched to on
nor a qualified pruned610 deployment, and no zero-inference-cost claim is made. This successor
does not replace the original four-endpoint evaluation or its qualification.
SOURCE preparation has not executed numerical/model qualification. SPEC must
bind training_CPU_reader to the original60171-byte/5f1eac59 source ref. Scientific imports occur only
inside the owner-executed run, after every training terminal gate passes.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import traceback
from contextlib import nullcontext

SPEC_SCHEMA = 'ROOT-native-geometry-paired-residual-off-controls-evaluation-spec-v1'
REPORT_SCHEMA = 'native-geometry-paired-residual-off-controls-evaluation-report-v1'
CP_SCHEMA = 'native-geometry-paired-common-schedule-checkpoint-v1'
QUAL_SCHEMA = 'ROOT-native-geometry-paired-common-schedule-CPU-qualification-v1'
QUAL_STATUS = 'ROOT_NATIVE_GEOMETRY_PAIRED_COMMON_SEGMENT_CPU_QUALIFIED'
MODES = ('native0', 'paired', 'fg_fine')
ENDPOINTS = ('native0', 'paired', 'paired_residual_off', 'fg_fine', 'fg_fine_residual_off')
OFF_ENDPOINTS = ('paired_residual_off', 'fg_fine_residual_off')
ENDPOINT_MODE = {label: label.replace('_residual_off', '') for label in ENDPOINTS}
TRAIN_READER_BYTES = 60171
TRAIN_READER_SHA = '5f1eac5900e314556b5041b9c2446b9a872ab0f6875d87f544945102e0d1a559'
ADAPTER_PREFIX = 'panoptic_head.paired_mask_readout.'
GT_FIELDS = ('gt_instances', 'ignored_instances', 'gt_sem_seg', 'gt_panoptic_seg')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(row):
    require(Path(row['path']).is_absolute(), 'Absolute bound path required')
    actual = reference(row['path'])
    require(actual == {key: row[key] for key in ('path', 'bytes', 'sha256')}, 'Pinned input differs')
    return Path(actual['path'])


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def load_source(row, name):
    loader = importlib.util.spec_from_file_location(name, bound(row))
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def terminal_guard(receipt, phase):
    require(receipt['status'] == 'ACTUAL_OWNED_TERMINAL' and receipt['original_identity'] is not None and
        type(receipt['returncode']) is int and receipt['returncode'] == 0 and
        receipt['root_wait_reaped'] is True and receipt['held_handle_exit'] ==
        dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        receipt['numeric_birth_absence']['original_birth_absent'] is True and
        receipt['abort'] is None and receipt['error'] is None and receipt['wait_errors'] is False and
        receipt['GPU_observation_errors'] == [], 'STOP: actual training exit0/held reap/birth absence/no errors required')
    if 'phase' in receipt:
        require(receipt['phase'] == phase, 'Actual training receipt phase differs')
    peak = receipt['whole_GPU_observed_peak_MiB']
    require(type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
        receipt['whole_GPU_used_limit_MiB'] == 8192 and receipt['whole_leaf_VRAM_certified'] is False and
        receipt['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
        'Actual whole-GPU observation exceeds its qualified scope')


def qualified_training(spec):
    """First inspect all actual terminal metadata; only then hash/load inputs."""
    rows = {}
    for mode in MODES:
        qualification = read(spec['qualificationmap'][mode]['path'])
        require(qualification['schema'] == QUAL_SCHEMA and qualification['status'] == QUAL_STATUS and
            qualification['kind'] == 'mask2former' and qualification['mode'] == mode,
            'Actual mode-specific independent CPU qualification required')
        receipt = read(qualification['parent_receipt']['path'])
        terminal_guard(receipt, qualification['phase'])
        rows[mode] = dict(qualification=qualification, receipt=receipt)
    # No scientific file has been read or hashed above this line.
    common = None
    for mode, row in rows.items():
        q, receipt = row['qualification'], row['receipt']
        bound(spec['qualificationmap'][mode]); bound(q['parent_receipt'])
        bound(q['reader_source']); bound(q['source']); bound(q['report']); bound(q['start'])
        require(q['reader_source'] == spec['training_CPU_reader'] and spec['training_CPU_reader'] in spec['sources'] and
            q['reader_source']['bytes'] == TRAIN_READER_BYTES and q['reader_source']['sha256'] == TRAIN_READER_SHA,
            'Original independently qualified training CPU reader role differs')
        train = read(bound(q['spec']))
        admission = read(bound(q['GPU_admission']))
        require(receipt['admission_sha256'] == q['GPU_admission']['sha256'] and
            admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            admission['automatic_retry'] is False and admission['phase'] == q['phase'] and
            admission['allowed_cpus'] == train['CPU_affinity'] == [0, 1] and
            admission['cuda_visible_devices'] == '0' and
            admission['qualified_leaf_deadline_seconds'] == train['deadline_seconds'] and
            admission['whole_GPU_used_limit_MiB'] == train['whole_GPU_owner_limit_MiB'] == 8192 and
            Path(admission['entry']).resolve(strict=True) == Path(spec['training_source']['path']).resolve(strict=True) and
            admission['entry_args'] == ['--spec', q['spec']['path']], 'Actual training admission/source/SPEC join differs')
        hashes = admission['bound_source_sha256']
        require(hashes.get(q['spec']['path']) == q['spec']['sha256'] and
            hashes.get(q['source']['path']) == q['source']['sha256'] and
            admission['driver_sha256'] == q['owner_driver']['sha256'], 'Actual training admission hashes differ')
        bound(q['owner_driver'])
        if 'scientific_spec' in admission:
            require(admission['scientific_spec'] == q['spec'], 'Training admission scientific SPEC differs')
        for source in train['sources']:
            bound(source)
            require(hashes.get(source['path']) == source['sha256'] and source in spec['sources'],
                'Current executing training source absent from evaluation/admission bindings')
        epoch = spec['expected_epoch']
        require(q['source'] == spec['training_source'] and q['checkpoint'] == spec['checkpointmap'][mode] and
            q['checkpointmap'] == {mode: q['checkpoint']} and q['technical_training_checkpoint_eligible'] is True and
            q['completed_epoch'] == epoch and q['global_microsteps'] == epoch * 741 and
            q['global_updates'] == epoch * 741 // 2 and q['consumed_raw_GT'] == epoch * 1661 and
            train['schema'] == 'ROOT-native-geometry-paired-common-schedule-spec-v1' and
            train['kind'] == 'mask2former' and train['mode'] == mode and train['phase'] == q['phase'] and
            train['initial_epoch'] + train['epochs_this_segment'] == epoch,
            'Only the actual independently qualified complete even-epoch full checkpoint is eligible')
        for key in ('native_helpers', 'initializer_source', 'micro_preflight_source', 'adapter_source', 'hooks_source', 'image_root'):
            require(train[key] == spec[key], 'Native construction/adapter/input role differs: ' + key)
        protocol = q['training_protocol']
        require(protocol['training_source'] == spec['training_source'] and protocol['mode'] == mode and
            protocol['native_loss_only'] is True and protocol['native_loss_atoms'] == 30 and protocol['Sobel'] is False,
            'Qualified full-native training protocol differs')
        signature = {key: value for key, value in protocol.items() if key not in
            ('mode', 'adapter_factory_hidden_seed', 'final_coefficient_adapter')}
        if common is None:
            common = signature
        require(signature == common, 'Modes do not share the same initialization/data/optimizer/schedule protocol')
        row['training_spec'] = train
    return rows


def prepare_model(torch, pref, initializer, am, hooks, registry, revert, Config, classes, spec, row, mode):
    """Construct qualified native architecture on CPU, then load its FULL endpoint."""
    train, q = row['training_spec'], row['qualification']
    initialized = read(bound(train['initializer_qualification']))
    require(initialized['schema'] == 'ROOT-native-coco-comparator-initializer-CPU-inspection-v1' and
        initialized['all_initializers_strictly_compatible'] is True and initialized['source'] == spec['initializer_source'],
        'Actual independent canonical COCO/boat initialization qualification differs')
    construction = dict(train, backbone_with_cp=False)
    model, initial, initialization = pref.prepare_cpu(torch, initializer, registry, revert, Config, classes,
        construction, 'mask2former', initialized['initializers']['mask2former'])
    require(len(model.state_dict()) == 610 and len(list(model.parameters())) == 451 and
        all(torch.equal(value.detach().cpu(), initial[name]) for name, value in model.state_dict().items()),
        'Fresh qualified native610 architecture/strict initialization differs')
    if mode != 'native0':
        hooks.register_paired_mask_readout(torch, am, model, mode, spec['adapter_source'])
    cp = torch.load(str(bound(spec['checkpointmap'][mode])), map_location='cpu')
    meta, state = cp['meta'], cp['state_dict']
    require(meta['schema'] == CP_SCHEMA and meta['kind'] == 'mask2former' and meta['mode'] == mode and
        meta['source'] == spec['training_source'] and meta['spec'] == q['spec'] and meta['phase'] == q['phase'] and
        meta['training_protocol'] == q['training_protocol'] and meta['completed_epoch'] == spec['expected_epoch'] and
        meta['global_microsteps'] == q['global_microsteps'] and meta['global_updates'] == q['global_updates'] and
        meta['consumed_raw_GT'] == q['consumed_raw_GT'] and meta['no_pending_accumulation_gradients'] is True and
        meta['not_technical_preflight_checkpoint'] is True, 'Full trained endpoint metadata differs; technical CP is ineligible')
    current = model.state_dict()
    require(state.keys() == current.keys() and len(state) == (610 if mode == 'native0' else 612) and
        all(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.shape == current[name].shape and
            value.dtype == current[name].dtype and bool(torch.isfinite(value).all()) for name, value in state.items()),
        'Trained fullstate keys/shapes/dtypes/finite differ')
    loaded = model.load_state_dict(state, strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys and all(torch.equal(value.detach().cpu(), state[name])
        for name, value in model.state_dict().items()), 'Full native/adapter checkpoint load is not tensor-exact')
    require(pref.state_proof(state) == read(bound(q['model_proof']['proof_refs']['state_proof']))['after'],
        'Loaded trained state differs from independently qualified endpoint proof')
    BN_params, BN_buffers = [], []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            BN_params.extend(name + '.' + local for local, value in module.named_parameters(recurse=False))
            BN_buffers.extend(name + '.' + local for local, value in module.named_buffers(recurse=False))
    require(len(BN_params) == 106 and len(BN_buffers) == 159 and
        all(torch.equal(state[name], initial[name]) for name in BN_params + BN_buffers), 'Original frozen BN state changed')
    model.requires_grad_(False).eval()
    fusion = model.panoptic_fusion_head
    require(model.panoptic_head.num_queries == 100 and model.panoptic_head.num_classes == 1 and
        fusion.num_classes == fusion.num_things_classes == 1 and fusion.num_stuff_classes == 0 and
        fusion.test_cfg.get('max_per_image', 100) == 100 and fusion.test_cfg.get('instance_on', False) is True and
        fusion.test_cfg.get('panoptic_on', True) is False and fusion.test_cfg.get('semantic_on', False) is False,
        'Native single-class/all100 official instance fusion required')
    proof = dict(initialization=initialization, checkpoint=spec['checkpointmap'][mode], qualification=spec['qualificationmap'][mode],
        mode=mode, CP_qualified=True, strict_fullstate_loaded=True, native_state_tensors=610,
        adapter_state_tensors=0 if mode == 'native0' else 2, registered_parameters=451 if mode == 'native0' else 453,
        initial_native_state=pref.state_proof(initial), trained_fullstate=pref.state_proof(state),
        native_changed_from_COCO_ship_initializer={name: not torch.equal(state[name], value) for name, value in initial.items()},
        trained_segment_changed=q['model_proof']['changed'], BN_parameter_names=BN_params, BN_buffer_names=BN_buffers,
        original_frozen_BN_exact=True, inference_backbone_with_cp=False,
        training_backbone_with_cp=train['backbone_with_cp'], no_random_missing_keys=True)
    del cp, current, initial, initialized
    return model, state, meta, proof


def evaluate_endpoint(torch, np, mask_utils, pref, old, hooks, spec, row, prepared, dataset, coco, output, label, started):
    mode = ENDPOINT_MODE[label]
    model, state, meta, loading_proof = prepared
    directory = output / label
    directory.mkdir()
    model.to('cuda').eval()
    before = pref.state_proof({name: value.detach().cpu() for name, value in model.state_dict().items()})
    require(before == loading_proof['trained_fullstate'] and all(parameter.grad is None for parameter in model.parameters()),
        'Evaluation START differs from qualified trained checkpoint or has parameter gradients')
    image_rows, predictions, factors = [], [], []
    try:
        for index in range(187):
            pref.check_deadline(started, spec['deadline_seconds'])
            data, samples, gray, raw, identity = old.load_image(model, dataset, coco, dataset.get_data_info(index))
            require(len(samples) == 1 and all(not any(key in sample for key in GT_FIELDS) for sample in samples),
                'GT survived forwarded metainfo sample stripping')
            external = pref.rng_snapshot(torch)
            external_digest = [pref.digest_tensor(value) for value in external]
            seed = 100000 + identity['image_id']
            with torch.random.fork_rng(devices=[0]):
                torch.random.default_generator.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with torch.no_grad():
                    features = model.extract_feat(data['inputs'])
                    scope = hooks.paired_readout_context(torch, model.panoptic_head, [sample.metainfo for sample in samples],
                        apply=label not in OFF_ENDPOINTS) if mode != 'native0' else nullcontext(None)
                    with scope as hook_record:
                        classes, masks = model.panoptic_head(features, samples)
                    require(len(classes) == len(masks) == 10 and
                        all(tuple(value.shape) == (1, 100, 2) and value.dtype == torch.float32 and not value.requires_grad
                            for value in classes) and
                        all(tuple(value.shape) == (1, 100, 200, 200) and value.dtype == torch.float32 and not value.requires_grad
                            for value in masks), 'Expected all ten original-shaped FP32 readouts')
                    final_class, final_masks = classes[-1], masks[-1]
                    del classes, masks, features
                    ih, iw = identity['actual_img_shape']
                    oh, ow = identity['original_shape']
                    require(0 < ih <= 800 and 0 < iw <= 800 and identity['batch_input_shape'] == [800, 800],
                        'Native keep-ratio geometry differs from fixed800 canvas')
                    # Match head.predict upsampling, then call the unchanged official fusion.
                    padded = torch.nn.functional.interpolate(final_masks, size=(800, 800), mode='bilinear', align_corners=False)
                    instances = model.panoptic_fusion_head.predict(final_class, padded, samples, rescale=True)[0]['ins_results']
                    rendered = torch.nn.functional.interpolate(padded[0, :, :ih, :iw][:, None], size=(oh, ow),
                        mode='bilinear', align_corners=False)[:, 0]
                    probabilities = torch.nn.functional.softmax(final_class[0], dim=-1)[:, :-1]
                    top_scores, top_indices = probabilities.flatten(0, 1).topk(100, sorted=False)
                    query_indices = top_indices // model.panoptic_fusion_head.num_classes
                    require(torch.equal(torch.sort(query_indices).values, torch.arange(100, device=query_indices.device)) and
                        torch.equal(top_scores, probabilities[query_indices, 0]), 'Official ranking did not preserve all native query ordinals')
                    ordered_logits = rendered[query_indices]
                    ordered_binary = (ordered_logits > 0).float()
                    qualities = (ordered_logits.sigmoid() * ordered_binary).flatten(1).sum(1) / (
                        ordered_binary.flatten(1).sum(1) + 1e-6)
                    official_scores = probabilities[query_indices, 0] * qualities
                    require(len(instances) == 100 and torch.equal(instances.masks, ordered_binary.bool()) and
                        torch.equal(instances.scores, official_scores) and bool((instances.labels == 0).all()) and
                        bool(torch.isfinite(instances.scores).all()) and
                        bool(((instances.scores >= 0) & (instances.scores <= 1)).all()),
                        'Official masks/scores/labels are not exactly aligned with native query IDs')
                    query_ids = query_indices.cpu().tolist()
                    binary = instances.masks.cpu().numpy().astype(np.uint8)
                    scores = instances.scores.cpu().tolist()
                    class_probabilities = probabilities[query_indices, 0].cpu().tolist()
                    mask_qualities = qualities.cpu().tolist()
                    final_digests = dict(classes=pref.digest_tensor(final_class), masks=pref.digest_tensor(final_masks))
                    del final_class, final_masks, padded, rendered, probabilities, top_scores, top_indices, query_indices
                    del ordered_logits, ordered_binary, qualities, official_scores, instances
            after_rng = pref.rng_snapshot(torch)
            require(all(torch.equal(a, b) for a, b in zip(external, after_rng)), 'External CPU/CUDA RNG did not restore')
            for query_id, mask, score, class_probability, mask_quality in zip(query_ids, binary, scores, class_probabilities, mask_qualities):
                rle = mask_utils.encode(np.asfortranarray(mask))
                counts = rle['counts'].decode('ascii') if isinstance(rle['counts'], bytes) else rle['counts']
                predictions.append(dict(image_id=int(identity['image_id']), query_id=int(query_id), category_id=1,
                    score=float(score), segmentation=dict(size=list(rle['size']), counts=counts)))
                factors.append(dict(image_id=int(identity['image_id']), query_id=int(query_id),
                    class_probability=float(class_probability), mask_quality=float(mask_quality), official_score=float(score)))
            image_rows.append(dict(mode=label, checkpoint_mode=mode, dataset_index=index, image_id=identity['image_id'],
                image=identity['image'], ann_ids=identity['ann_ids'], annotation_count=len(identity['ann_ids']),
                original_shape=identity['original_shape'], actual_img_shape=identity['actual_img_shape'],
                batch_input_shape=identity['batch_input_shape'], scale_factor=identity['scale_factor'],
                GT_removed_from_forward=True, loader_accesses_annotations=True, matching_or_loss_executed=False,
                checkpoint=spec['checkpointmap'][mode], CP_qualified=True, head_seed=seed,
                external_RNG_before=external_digest, external_RNG_after=[pref.digest_tensor(value) for value in after_rng],
                external_RNG_restored=True, final_fields=final_digests, paired_hook=hook_record,
                official_instance_postprocess=True, source_native_query_ID_mapping_verified=True,
                official_masks_query_ID_bitwise_equal=True, official_scores_query_ID_bitwise_equal=True,
                prediction_query_ids=query_ids, predictions=100, query_score_factors_verified=True,
                class_probability_mask_quality_product_TorchEqual_official_score=True,
                added_NMS=False, added_score_cutoff=False, query_filter=False,
                inference_ablation_not_training_arm=label in OFF_ENDPOINTS))
            if (index + 1) % 50 == 0 or index == 186:
                print(json.dumps(dict(mode=label, evaluated_images=index + 1, elapsed_seconds=time.monotonic() - started)), flush=True)
            del data, samples, gray, raw, identity, external, after_rng, binary, scores, class_probabilities, mask_qualities
            del query_ids, mask, hook_record, scope
        after = pref.state_proof({name: value.detach().cpu() for name, value in model.state_dict().items()})
        require(before == after and all(torch.equal(value.detach().cpu(), state[name]) and value.shape == state[name].shape and
            value.dtype == state[name].dtype for name, value in model.state_dict().items()) and
            all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()) and
            all(not module.training for module in model.modules()),
            'Evaluation changed full trained native/adapter state or parameter gradients')
        require(len(image_rows) == 187 and len(predictions) == len(factors) == 18700 and
            len({entry['image_id'] for entry in image_rows}) == 187 and
            sum(entry['annotation_count'] for entry in image_rows) == 380, 'Complete DEV187/380/all100 population differs')
        write(directory / 'IMAGE_ROWS.json', image_rows)
        write(directory / 'ALL_PREDICTIONS.json', predictions)
        write(directory / 'QUERY_SCORE_FACTORS.json', factors)
        write(directory / 'STATE_PROOF.json', dict(schema='native-geometry-paired-residual-off-controls-evaluation-state-proof-v1',
            mode=label, checkpoint_mode=mode, loading_proof=loading_proof, before=before, after=after,
            full_native_and_adapter_state_exact=True, params_gradNone=True, all_parameters_frozen=True,
            all_modules_eval=all(not module.training for module in model.modules()),
            same_CP_inference_ablation=label in OFF_ENDPOINTS))
        return dict(mode=label, checkpoint_mode=mode, checkpoint=spec['checkpointmap'][mode],
            qualification=spec['qualificationmap'][mode], completed_epoch=meta['completed_epoch'], CP_qualified=True,
            image_rows=reference(directory / 'IMAGE_ROWS.json'), predictions=reference(directory / 'ALL_PREDICTIONS.json'),
            query_score_factors=reference(directory / 'QUERY_SCORE_FACTORS.json'), state_proof=reference(directory / 'STATE_PROOF.json'),
            images=187, raw_GT=380, predictions_count=18700, query_score_factors_count=18700,
            fullstate_tensors=len(state), full_native_state_exact=True, adapter_state_exact=True, params_gradNone=True,
            loader_accesses_annotations=True, GT_removed_from_forward=True, native_official_fusion=True,
            query_score_factors_verified=True, source_native_query_ID_mapping_verified=True,
            class_probability_mask_quality_product_TorchEqual_official_score=True,
            cross_model_query_ID_object_identity_claimed=False, cross_model_factor_exchange=False,
            inference_ablation_not_training_arm=label in OFF_ENDPOINTS,
            adapter_forward_executed=mode != 'native0' and label not in OFF_ENDPOINTS,
            retained_adapter_state_tensors=0 if mode == 'native0' else 2,
            residual_off_is_compute_matched=False if label in OFF_ENDPOINTS else None,
            pruned_deployment_qualified=False, added_NMS=False, added_score_cutoff=False,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
    finally:
        model.to('cpu')


def run(spec_file):
    started = time.monotonic()
    require(Path(spec_file).is_absolute(), 'Absolute --spec required')
    spec_path = Path(spec_file).resolve(strict=True)
    spec = read(spec_path)
    require(spec['schema'] == SPEC_SCHEMA and spec['phase'].startswith('native_geometry_paired_residual_off_controls_evaluation_') and
        spec['modes'] == list(MODES) and set(spec['checkpointmap']) == set(spec['qualificationmap']) == set(MODES) and
        type(spec['expected_epoch']) is int and spec['expected_epoch'] in range(2, 25, 2) and spec['evaluation_images'] == 187 and
        spec['paired_residual_off_probe'] is True and spec['fg_fine_residual_off_probe'] is True and
        spec['endpoints'] == list(ENDPOINTS), 'Explicit three trained modes/five inference endpoints/even-epoch DEV187 required')
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and Path(spec['image_root']).is_absolute() and
        spec['CPU_affinity'] == [0, 1] and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_owner_limit_MiB'] == 8192 and
        type(spec['deadline_seconds']) is int and spec['deadline_seconds'] > 0 and os.environ.get('CUDA_VISIBLE_DEVICES') == '0',
        'ROOT fresh absolute output/CPU01/CUDA0/allocator4096/whole8192/deadline required')
    training = qualified_training(spec)
    for source in spec['sources']:
        bound(source)
    require(reference(__file__) in spec['sources'] and all(spec[key] in spec['sources'] for key in
        ('training_source', 'training_CPU_reader', 'initializer_source', 'micro_preflight_source', 'native_helpers', 'adapter_source', 'hooks_source')),
        'Current evaluator and executing source roles are not bound')
    output.mkdir(parents=True)
    write(output / 'START.json', dict(schema='native-geometry-paired-residual-off-controls-evaluation-start-v1', phase=spec['phase'],
        source=reference(__file__), spec=reference(spec_path), modes=list(MODES), expected_epoch=spec['expected_epoch'],
        evaluation_images=187, checkpointmap=spec['checkpointmap'], qualificationmap=spec['qualificationmap'],
        paired_residual_off_probe=True, fg_fine_residual_off_probe=True, endpoints=list(ENDPOINTS),
        no_AP=True, no_TEST=True, no_retry=True,
        all_training_terminal_gates_passed_before_scientific_access=True))
    import psutil
    import numpy as np
    import torch
    from pycocotools import mask as mask_utils
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'Actual CPU01/cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    register_all_modules(init_default_scope=False); init_default_scope('mmdet')
    pref = load_source(spec['micro_preflight_source'], 'paired_evaluation_qualified_native_preflight')
    initializer = load_source(spec['initializer_source'], 'paired_evaluation_qualified_initializer')
    old = load_source(spec['native_helpers'], 'paired_evaluation_native_helpers'); old.torch = torch
    am = load_source(spec['adapter_source'], 'paired_evaluation_sealed_context')
    hooks = load_source(spec['hooks_source'], 'paired_evaluation_sealed_hooks')
    require(am.INITIALIZATION_SEED == 2026100911 and am.TRAINABLE_TENSORS == 2 and am.TRAINABLE_SCALARS == 393216,
        'Actual sealed adapter architecture differs')
    classes = tuple(CocoDataset.METAINFO['classes'])
    require(len(classes) == 80 and classes[8] == 'boat', 'Official COCO class ordering differs')
    prepared = {mode: prepare_model(torch, pref, initializer, am, hooks, MODELS, revert_sync_batchnorm, Config,
        classes, spec, training[mode], mode) for mode in MODES}
    dataset = CocoDataset(ann_file=str(bound(spec['DEV'])), data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
    coco = COCO(str(bound(spec['DEV'])))
    require(len(dataset) == len(coco.imgs) == 187 and len(coco.anns) == 380, 'Complete actual DEV187/380 required')
    torch.cuda.init()
    capacity = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 4096 * 1024 ** 2 / capacity), 0)
    torch.cuda.reset_peak_memory_stats(0)
    results, error_record = {}, None
    try:
        for mode in MODES:
            item = prepared.pop(mode)
            labels = [label for label in ENDPOINTS if ENDPOINT_MODE[label] == mode]
            for label in labels:
                result = evaluate_endpoint(torch, np, mask_utils, pref, old, hooks, spec, training[mode], item,
                    dataset, coco, output, label, started)
                # The complete per-image function frame and GPU temporaries are dead.
                result['cleanup'] = pref.clear_gpu(torch)
                results[label] = result
            del item
        for source in spec['sources']:
            bound(source)
        for mode in MODES:
            bound(spec['checkpointmap'][mode]); bound(spec['qualificationmap'][mode])
        pref.check_deadline(started, spec['deadline_seconds'])
    except Exception as error:
        error_record = dict(type=type(error).__name__, message=str(error), no_retry=True, no_fallback=True,
            OOM_technical_stop=isinstance(error, torch.cuda.OutOfMemoryError))
        traceback.clear_frames(error.__traceback__)
        if 'item' in locals():
            item[0].to('cpu'); del item
        prepared.clear(); gc.collect()
        try:
            cleanup = pref.clear_gpu(torch)
        except Exception as cleanup_error:
            cleanup = dict(error=str(cleanup_error))
        error_record['cleanup'] = cleanup
    write(output / 'REPORT.json', dict(schema=REPORT_SCHEMA, status=('TECHNICAL_FIVE_ENDPOINT_RESIDUAL_OFF_EXPORT_COMPLETE_NO_AP'
        if error_record is None else 'TECHNICAL_STOP_NO_RESIDUAL_OFF_EVALUATION_QUALIFICATION'), phase=spec['phase'], source=reference(__file__),
        spec=reference(spec_path), modes=list(MODES), expected_epoch=spec['expected_epoch'], checkpointmap=spec['checkpointmap'],
        qualificationmap=spec['qualificationmap'], mode_reports=results, error=error_record, DEV=spec['DEV'],
        evaluation_images=187, raw_GT=380, predictions_per_image=100,
        paired_residual_off_probe=True, fg_fine_residual_off_probe=True, endpoints=list(ENDPOINTS),
        optional_off_is_same_CP_inference_ablation=True, trained_modes_count=3, inference_endpoints_count=5,
        training_path_or_regularization_mechanism_claimed=False,
        residual_off_skips_adapter_forward=True, residual_off_compute_matched=False,
        pruned610_deployment_qualified=False, zero_additional_inference_cost_claimed=False,
        all_training_terminal_gates_passed_before_scientific_access=True, loader_accesses_annotations=True,
        GT_removed_from_forward=True, matching_executed=False, loss_executed=False, no_TEST=True, AP_computed=False,
        DEV_historically_exposed=True, cross_model_query_ID_object_identity_claimed=False, cross_model_factor_exchange=False,
        added_NMS=False, added_score_cutoff=False, scientific_method_decision_computed=False,
        elapsed_seconds=time.monotonic() - started, torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0)))
    if error_record:
        raise RuntimeError('TECHNICAL_STOP: ' + error_record['type'] + ': ' + error_record['message'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)


if __name__ == '__main__':
    main()
