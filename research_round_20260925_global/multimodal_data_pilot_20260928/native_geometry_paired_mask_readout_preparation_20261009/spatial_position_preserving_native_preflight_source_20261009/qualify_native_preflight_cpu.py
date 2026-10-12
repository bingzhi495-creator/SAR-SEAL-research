"""Inert manual ROOT CPU consumer of spatial-position-preserving technical preflight.

Importing this source performs no scientific imports or payload access. ROOT
alone runs it after the independently observed owned terminal. Digest-based
inference observations do not constitute a CPU forward replay or method result.
This fresh reader independently qualifies three cold spatial modes and three
fresh four-image/two-update supervised native-loss cases. Direct CPU Linear
constructors reconstruct both weights. Observed small geometry is decoded and
hashed; counts, fallback and metadata support are independently checked. No
F/e0/M0 packet is saved, so PCA and hard-sign eligibility are not CPU replayed.
No scientific execution was performed during SOURCE preparation. Explicit
ROOT execution and actual owned terminal evidence are mandatory.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

SPEC_SCHEMA = 'ROOT-native-spatial-position-preserving-native-bridge-preflight-spec-v1'
REPORT_SCHEMA = 'native-spatial-position-preserving-native-bridge-preflight-report-v1'
CP_SCHEMA = 'native-spatial-position-preserving-native-bridge-technical-checkpoint-v1'
MODES = ('global', 'spatial_halves', 'spatial_reversed')
TRAINING_MODES = MODES
IMAGE_IDS = (1, 2, 3, 5)
PREFIX = 'panoptic_head.spatial_mask_readout.'
DRIVER_BYTES = 38723
DRIVER_SHA = '840bdf376ee0d9eb291e24cb605c110d310d32338ccba6d3f231c827c0e6e1ba'
REUSED_BYTES = 46683
REUSED_SHA = 'ae810fb940fed62588b1a4e0bd7ee199647ff656dbbef5d0bb1247864662d960'
GEOMETRY_SHA = '7ad19e356342afc15a2a41f1358634b23486c19021e63ccd94cfed03f6841d15'
HOOKS_SHA = '23f36f3fab9b4a006bd0db65427b68223dc99c4929da458504ed5a3aa9185806'
READOUT_SHA = '7da9561ed30675fd0d692ce2ad883e0c8cccfddfd9f7913761da0fe4fff0a09d'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and set(('path', 'bytes', 'sha256')) <= set(row),
            'Incomplete absolute reference')
    require(Path(row['path']).is_absolute() and reference(row['path']) ==
            {key: row[key] for key in ('path', 'bytes', 'sha256')},
            'Declared artifact differs: ' + str(row.get('path')))
    return Path(row['path'])


def bind_tree(value):
    if isinstance(value, dict):
        if set(('path', 'bytes', 'sha256')) <= set(value):
            bound(value)
        else:
            for item in value.values():
                bind_tree(item)
    elif isinstance(value, list):
        for item in value:
            bind_tree(item)


def load_source(row, name):
    definition = importlib.util.spec_from_file_location(name, str(bound(row)))
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


def terminal_guard(receipt, phase):
    keys = ('status', 'returncode', 'root_wait_reaped', 'held_handle_exit', 'numeric_birth_absence',
            'abort', 'error', 'wait_errors', 'GPU_observation_errors', 'original_identity',
            'admission_sha256')
    require(all(key in receipt for key in keys) and
            receipt['status'] == 'ACTUAL_OWNED_TERMINAL' and receipt['original_identity'] is not None and
            type(receipt.get('returncode')) is int and receipt['returncode'] == 0 and
            receipt.get('root_wait_reaped') is True and
            receipt.get('held_handle_exit') ==
            {'wait_state': 'WAIT_OBJECT_0', 'actual_exit_code': 0} and
            receipt.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
            receipt.get('abort') is None and receipt.get('error') is None and
            receipt.get('wait_errors') is False and receipt.get('GPU_observation_errors') == [],
            'STOP: actual owned exit0, held reap, birth absence and no errors required')
    if 'phase' in receipt:
        require(receipt['phase'] == phase, 'Owned receipt phase differs')


def admission_guard(spec_path, spec, receipt):
    """Bind this new entry and consumer; never use the old run.py admission."""
    spec_path = spec_path.resolve(strict=True)
    admission_name = spec['GPU_admission_file']
    require(isinstance(admission_name, str) and Path(admission_name).is_absolute(),
            'Absolute admission filename string required')
    admission_path = Path(admission_name).resolve(strict=True)
    admission_ref = reference(admission_path)
    require(receipt['admission_sha256'] == admission_ref['sha256'], 'Receipt/admission SHA differs')
    admission = read_json(admission_path)
    spec_ref, reader_ref = reference(spec_path), reference(__file__)
    entry = Path(__file__).resolve().with_name('native_preflight.py')
    entry_ref = reference(entry)
    owner = spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py'
    owner_ref = reference(owner)
    require(entry_ref['bytes'] == DRIVER_BYTES and entry_ref['sha256'] == DRIVER_SHA and
            owner_ref['sha256'] == OWNER_SHA and admission['driver_sha256'] == OWNER_SHA,
            'Exact producer/owned-driver source differs')
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            admission['automatic_retry'] is False and admission['phase'] == spec['phase'] and
            admission['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and
            admission['cuda_visible_devices'] == '0' and
            type(admission['qualified_leaf_deadline_seconds']) is int and
            admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
            admission['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192 and
            Path(admission['entry']).is_absolute() and Path(admission['entry']).resolve(strict=True) == entry and
            admission['entry_args'] == ['--spec', str(spec_path), '--owner-execute'], 'New phase/resource/entry contract differs')
    sources = [{key: row[key] for key in ('path', 'bytes', 'sha256')} for row in spec['sources']]
    hashes = admission['bound_source_sha256']
    reused_ref = spec['reused_CPU_reader']
    require(isinstance(reused_ref, dict) and Path(reused_ref['path']).is_absolute() and
            reused_ref['bytes'] == REUSED_BYTES and reused_ref['sha256'] == REUSED_SHA and
            spec['entry_source'] == entry_ref and spec['CPU_reader'] == reader_ref,
            'Explicit original46683 helper/new entry/new consumer roles differ')
    require(entry_ref in sources and reader_ref in sources and reused_ref in sources and
            hashes.get(str(spec_path)) == spec_ref['sha256'],
            'Actual producer/reader/reused reader/SPEC are not source-bound')
    for row in sources:
        require(hashes.get(row['path']) == row['sha256'], 'Admission source closure differs')
        bound(row)
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Admission scientific SPEC differs')
    peak = receipt['whole_GPU_observed_peak_MiB']
    require(type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            receipt['whole_GPU_used_limit_MiB'] == 8192 and receipt['whole_leaf_VRAM_certified'] is False and
            receipt['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
            'Actual GPU observation/scope differs')
    return admission_ref, spec_ref, entry_ref, owner_ref, reused_ref


def digest_contract(row, shape, dtype):
    require(isinstance(row, dict) and set(row) == {'shape', 'dtype', 'sha256'} and
            row['shape'] == shape and row['dtype'] == dtype and isinstance(row['sha256'], str) and
            len(row['sha256']) == 64 and set(row['sha256']) <= set('0123456789abcdef'),
            'Observed tensor digest contract differs')


def hook_contract(row, mode, readout_ref, geometry_ref, diagnostics):
    expected = dict(forward_calls=1, completed_forwards=1, pixel_decoder_calls=1,
        coefficient_calls=10, passive_coefficient_captures=1, unchanged_early_readouts=9,
        final_mask_replacements=1, adapter_calls=1, apply=True, mode=mode,
        readout_source=readout_ref, geometry_source=geometry_ref,
        metadata_fields_used=['img_shape', 'batch_input_shape'], class_list_object_preserved=True,
        first_nine_mask_objects_preserved=True, native_coefficient_outputs_untouched=True,
        earlier_gate_paths_untouched=True, all_queries_retained=True,
        half_eligible_counts=diagnostics['geometry_packets']['half_eligible_counts']['values'],
        global_fallback=diagnostics['geometry_packets']['global_fallback']['values'],
        adapter_geometry_phi_skipped=False, cache_cleared_after_forward=True,
        registered_state_not_pruned=True, hooks_removed=True,
        feature_shape=[1, 256, 200, 200], coefficient_shape=[1, 100, 256],
        final_native_mask_shape=[1, 100, 200, 200])
    require(row == expected, 'Actual whole-head hook/capture/only-final replacement differs')


def adapter_observation(torch, old, row, mode, live, actual_hw):
    require(row['mode'] == mode and row['eligibility_frame'] == 'canonical' and row['all_queries'] == 100 and
            row['input_F_shape'] == [1, 256, 200, 200] and row['input_e0_shape'] == [1, 100, 256] and
            row['input_F_requires_grad'] is live and row['input_e0_requires_grad'] is live and
            row['mask_logits_requires_grad'] is live and row['residual_requires_grad'] is live and
            row['coefficient_fields_shape'] == [1, 100, 2, 256] and row['all_queries_retained'] is True and
            row['GT_used'] is False and row['dense_feature_or_mask_exported'] is False,
            'Actual spatial F/e0/fields/live/all100/no-GT observation differs')
    shapes = dict(eligible=([1,100,16], 'bool'), inside=([1,100,16,2], 'float32'),
        outside=([1,100,16,2], 'float32'), centre_xy=([1,100,2], 'float32'),
        used_theta=([1,100], 'float32'), sigma_major=([1,100], 'float32'),
        sigma_minor=([1,100], 'float32'), rho_mass=([1,100], 'float32'),
        half_eligible_counts=([1,100,2], 'int64'), global_fallback=([1,100], 'bool'))
    require(set(row['geometry_packets']) == set(shapes), 'Actual ten small geometry packets required')
    tensors = {}
    for name, (shape, dtype) in shapes.items():
        packet = row['geometry_packets'][name]
        require(set(packet) == {'shape','dtype','sha256','values'}, 'Small packet keys differ: ' + name)
        digest_contract({key:packet[key] for key in ('shape','dtype','sha256')}, shape, dtype)
        value = torch.tensor(packet['values'], dtype=getattr(torch,dtype), device='cpu')
        require(list(value.shape) == shape and old.tensor_digest(torch,value,dtype) ==
                {key:packet[key] for key in ('shape','dtype','sha256')}, 'Small packet values/digest differ: ' + name)
        if dtype == 'float32':
            require(bool(torch.isfinite(value).all()), 'Nonfinite small geometry: ' + name)
        tensors[name] = value
    E, halves, mass = tensors['eligible'], tensors['half_eligible_counts'], tensors['rho_mass']
    expected_halves = torch.stack((E[...,:8].sum(-1),E[...,8:].sum(-1)),dim=-1)
    counts = E.sum(-1)
    require(torch.equal(halves,expected_halves) and
            torch.equal(tensors['global_fallback'],(expected_halves == 0).any(-1)) and
            bool((mass >= 0).all()) and bool((tensors['sigma_major'] > 0).all()) and
            bool((tensors['sigma_minor'] > 0).all()) and
            bool(((counts == 0) | (mass > 1e-12)).all()), 'Observed E/half/fallback/support relationship differs')
    expected_counts = dict(queries_with_support=int((mass > 1e-12).sum()),
        queries_with_eligible_pairs=int((counts > 0).sum()), eligible_pairs_total=int(counts.sum()),
        eligible_pairs_max=int(counts.max()), canonical_eligible_pairs_total=int(counts.sum()))
    require(all(type(row[key]) is int and row[key] == val for key,val in expected_counts.items()),
            'Observed small packets do not reproduce actual aggregate counts')
    require(isinstance(actual_hw,list) and len(actual_hw) == 2 and all(type(v) is int and 0 < v <= 800
            for v in actual_hw), 'Actual native input dimensions invalid')
    h,w = actual_hw
    axes = torch.arange(200,dtype=torch.float32,device='cpu')
    fy = ((h-4*axes).clamp(0,4)/4).reshape(1,1,200,1)
    fx = ((w-4*axes).clamp(0,4)/4).reshape(1,1,1,200)
    valid = fy*fx
    digest_contract(row['valid_fraction'],[1,1,200,200],'float32')
    require(old.tensor_digest(torch,valid,'float32') == row['valid_fraction'],
            'Independent metadata cell-valid fraction digest differs')
    # These are observations, not a reconstruction of PCA or M0-sign E.
    return row['geometry_packets']


def initialization_proof(torch, old, proof, initial, entry, closure, strict80, strict1, selections, prefix, with_cp):
    expected = dict(kind='mask2former', checkpoint=entry['checkpoint'], official_config=entry['config'],
        official_config_closure=closure, canonical_module_prefix_removed=prefix, official80=strict80,
        ship1=strict1, allowed_changed_keys=selections, initial_ship_tensors_exact=True,
        only_boat_classifier_rows_changed=True, random_missing_keys=False,
        public_recipe_equivalence_claimed=False, backbone_with_cp=with_cp,
        declared_preprocessing_overrides=dict(resize=[800, 800], keep_ratio=True, fixed_canvas=[800, 800],
                                            augmentation=False, mask_resize='nearest'))
    for key, value in expected.items():
        require(proof[key] == value, 'Independent initialization proof differs: ' + key)
    old.check_state(torch, initial, proof['initial_state'], 'independent native610 initialization')


def build_native(torch, old, initializer, registry, revert, Config, spec, accepted):
    entry = spec['initializers']['mask2former']
    require(entry['checkpoint'] == accepted['checkpoint'] and entry['config'] == accepted['official_config'] and
            accepted['strict_official80_and_ship1_compatible'] is True, 'Official initializer lineage differs')
    config = Config.fromfile(str(bound(entry['config'])))
    require(config.model.backbone.depth == 50 and config.model.backbone.style == 'pytorch', 'Native R50 required')
    closure = initializer.config_closure(bound(entry['config']))
    sources = {str(Path(row['path']).resolve()): row for row in spec['sources']}
    require(all(sources.get(row['path']) == row for row in closure), 'Official config inheritance unbound')
    checkpoint = torch.load(str(bound(entry['checkpoint'])), map_location='cpu')
    raw_keys = list(checkpoint['state_dict'])
    prefix = all(key.startswith('module.') for key in raw_keys)
    official = old.canonical_state(torch, checkpoint)
    official_model = initializer.build_cpu(torch, registry, revert, config.model)
    strict80 = initializer.strict_load(torch, official_model, official)
    initial, selected_rows = old.independent_ship_rows(torch, 'mask2former', official, official_model)
    ship_config, crosscheck, selections = initializer.ship_state(torch, 'mask2former', config.model, official_model, official)
    require(set(initial) == set(crosscheck) and all(torch.equal(initial[name], crosscheck[name]) for name in initial) and
            set(selected_rows) == set(selections) and all(selected_rows[name] == selections[name]['selected_rows']
                                                        for name in selected_rows), 'Independent boat8/null80 mapping differs')
    del checkpoint, official, official_model, crosscheck
    ship_config.backbone.frozen_stages = -1
    ship_config.backbone.norm_eval = True
    ship_config.backbone.norm_cfg.requires_grad = False
    ship_config.backbone.with_cp = spec['training_backbone_with_cp']
    ship_config.data_preprocessor.pad_mask = True
    ship_config.data_preprocessor.pad_size_divisor = 32
    ship_config.data_preprocessor.batch_augments = [dict(type='BatchFixedSizePad', size=(800, 800),
        img_pad_value=0, pad_mask=True, mask_pad_value=0, pad_seg=False)]
    model = initializer.build_cpu(torch, registry, revert, ship_config)
    strict1 = initializer.strict_load(torch, model, initial)
    require(len(initial) == 610 and len(list(model.parameters())) == 451, 'Native610/451 model differs')
    return model, initial, dict(entry=entry, closure=closure, strict80=strict80, strict1=strict1,
                               selections=selections, prefix=prefix)


def independent_spatial_state(torch):
    """Direct CPU 512->512->256 construction independent of source factories."""
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2026100911)
        hidden = torch.nn.Linear(512,512,bias=False,device='cpu',dtype=torch.float32)
        final = torch.nn.Linear(512,256,bias=False,device='cpu',dtype=torch.float32)
        with torch.no_grad():
            final.weight.zero_()
        state = {'hidden.weight':hidden.weight.detach().clone(),'final.weight':final.weight.detach().clone()}
    require(torch.equal(before,torch.get_rng_state()) and not torch.cuda.is_initialized() and
            sum(value.numel() for value in state.values()) == 393216 and
            bool((state['hidden.weight'] != 0).any()) and bool((state['final.weight'] == 0).all()),
            'Independent CPU spatial cold constructor/RNG/count differs')
    return state


def check_cold_inference(torch, old, readout, geometry, spec, result, initial, init_context, images, image_order):
    packet = read_json(bound(result['identity']))
    require(result['all_three_cold_modes_exact'] is True, 'Three cold modes incomplete')
    initialization_proof(torch, old, packet['initializer'], initial, with_cp=False, **init_context)
    old.check_state(torch, initial, packet['native_before'], 'inference native before')
    old.check_state(torch, initial, packet['native_after'], 'inference native after')
    identity = packet['identity']
    image = images[1]
    ann_ids = [int(ann['id']) for ann in images['_annotations'][1]]
    height, width = int(image['height']), int(image['width'])
    scale = 800. / max(height, width)
    actual = [int(height * scale + .5), int(width * scale + .5)]
    require(identity['image_id'] == 1 and identity['dataset_index'] == image_order.index(1) and
            identity['ann_ids'] == ann_ids and identity['original_shape'] == [height, width] and
            identity['actual_img_shape'] == actual and identity['batch_input_shape'] == [800, 800] and
            identity['native_input_gamma'] == 1 and identity['target_gray_resize'] == 'nearest' and
            identity['scale_factor'] == [actual[1] / width, actual[0] / height] and
            identity['parser_rows'] == [dict(annotation_id=key, native_parser_retained=True) for key in ann_ids],
            'Cold inference original identity/all raw GT loader order differs')
    bound(identity['image'])
    digest_contract(packet['input'], [1, 3, 800, 800], 'float32')
    require(packet['same_cached_backbone_features_for_all_modes'] is True and packet['native610_exact'] is True and
            packet['GT_removed_from_forward'] is True and packet['loader_accesses_annotations'] is True and
            packet['AP_computed'] is False and packet['prediction_payload_exported'] is False and
            packet['equality_is_TorchEqual_not_claimed_byte_identity'] is True and
            packet['baseline_scope'] == 'bare native610 head; no spatial bridge installed',
            'Cold inference scope differs')
    require('baseline_hook' not in packet, 'Bare native baseline cannot claim a spatial hook')
    fields = packet['baseline_fields']
    require(set(fields) == {'classes', 'masks'} and len(fields['classes']) == len(fields['masks']) == 10,
            'All ten baseline native fields required')
    for value in fields['classes']:
        digest_contract(value, [1, 100, 2], 'float32')
    for value in fields['masks']:
        digest_contract(value, [1, 100, 200, 200], 'float32')
    official = packet['baseline_official']
    for name, shape, dtype in (('query_ids', [100], 'int64'), ('masks', [100, height, width], 'bool'),
                               ('scores', [100], 'float32'), ('labels', [100], 'int64')):
        digest_contract(official[name], shape, dtype)
    for name, value in dict(predictions=100, original_shape=[height, width], official_fusion_predict=True,
        official_query_mapping_verified=True, added_NMS=False, added_score_cutoff=False).items():
        require(official[name] == value, 'Official native output scope differs: ' + name)

    rows = packet['mode_rows']
    require(len(rows) == 3 and tuple(row['mode'] for row in rows) == MODES, 'Three cold modes/order required')
    canonical,cold_states,shared_rng = None,{},None
    expected_state = independent_spatial_state(torch)
    for row in rows:
        mode = row['mode']
        adapter = readout.build_adapter(torch,geometry,mode=mode)
        state = dict(adapter.state_dict())
        require(set(state) == set(expected_state) and all(torch.equal(state[key],value)
                for key,value in expected_state.items()) and not list(adapter.buffers()),
                'Actual spatial factory differs from independent CPU cold weights')
        old.check_state(torch,expected_state,row['cold_adapter'], mode+' cold spatial adapter')
        cold_states[mode]=row['cold_adapter']
        require(row['cold_adapter'] == rows[0]['cold_adapter'], 'Spatial cold modes weights differ')
        for key in ('cold_constructor_repeated_exact','cold_final_weight_zero',
                    'canonical_coordinates_and_eligibility_exact','classes10_exact','masks10_exact',
                    'official100_exact','query_ID_order_exact','private_RNG_matches_baseline','native610_exact',
                    'all_parameter_grads_None'):
            require(row[key] is True, 'Source-observed cold identity failed: '+key)
        require(row['official'] == official and row['private_seed'] == 100001 and
                row['private_RNG_before'] == packet['baseline_private_RNG_before'] and
                row['private_RNG_after'] == packet['baseline_private_RNG_after'],
                'Actual official digests/private RNG differ from bare baseline')
        old.validate_rng(torch,row)
        if shared_rng is None:
            shared_rng=row['external_RNG_before']
        require(row['external_RNG_before'] == shared_rng, 'Cold external RNG stream differs')
        observed=adapter_observation(torch,old,row['diagnostics'],mode,False,actual)
        hook_contract(row['hook'],mode,spec['readout_source'],spec['geometry_source'],row['diagnostics'])
        if canonical is None:
            canonical=observed
        require(observed == canonical, 'Actual cold geometry/E/halves packets differ')
        del adapter,state
    return dict(identity=result['identity'],baseline_fields=fields,baseline_official=official,
        cold_adapter_digests=cold_states,cold_three_mode_source_observations_consistent=True,
        canonical_geometry_packets_exact=True,metadata_valid_fraction_independently_verified=True,
        all100_official_masks_scores_query_order_source_observed=True,
        PCA_and_M0_sign_eligibility_CPU_replayed=False,CPU_forward_replayed=False)


def check_matches(torch, old, matches, ann_ids):
    require(isinstance(matches, list) and len(matches) == 10, 'All ten native matching observations required')
    for match in matches:
        q, assigned = match['positive_query_indices'], match['assigned_GT_indices']
        require(type(match['positives']) is int and match['positives'] == len(ann_ids) == len(q) == len(assigned) and
                len(q) == len(set(q)) and all(type(value) is int and 0 <= value < 100 for value in q) and
                all(type(value) is int for value in assigned) and sorted(assigned) == list(range(len(ann_ids))),
                'Native one-to-one positive assignment omitted/duplicated raw GT')
        labels = torch.ones(100, dtype=torch.int64)
        labels[q] = 0
        require(old.tensor_digest(torch, labels, 'int64') == match['target_labels'],
                'Native actual Q100 classification target digest differs from observed positives')


def check_trace(torch, np, cv2, coco, old, spec, result, policy, images, image_order, mode='global'):
    require(len(result['traces']) == 4, 'Exactly four actual trace references required')
    rows, consumed = [], []
    activity = {name: 0 for name in policy['active_names']}
    update_counts = dict(activity)
    external_rng = None
    for micro, (ref, image_id) in enumerate(zip(result['traces'], IMAGE_IDS)):
        row = read_json(bound(ref))
        ann_ids = [int(ann['id']) for ann in coco.imgToAnns[image_id]]
        require(row['microstep'] == micro and row['image_id'] == image_id and
                row['identity']['dataset_index'] == image_order.index(image_id) and
                row['private_seed'] == 1000000 + micro and row['accumulation_divisor'] == 2 and
                row['optimizer_step'] is bool(micro % 2) and row['optimizer_updates'] == (micro + 1) // 2 and
                row['gradient_clipping_applied'] is bool(micro % 2) and
                row['gradients_are_accumulated_preclip'] is True and row['frozen_BN_parameter_grads_None'] is True,
                'Four-micro native accumulation/seed/update/frozen-BN contract differs')
        adapted_GT = dict(row, actual_input_shape=row['input']['shape'])
        old.validate_GT(torch, np, cv2, coco, images[image_id], ann_ids, adapted_GT)
        require(row['native_loss_only'] is True and row['Sobel_added'] is False and row['auxiliary_loss_added'] is False,
                'Original native30 only required')
        old.validate_losses(dict(row, loss_is_native_model_loss=row['native_loss_only']), 'mask2former')
        check_matches(torch, old, row['matches'], ann_ids)
        adapter_observation(torch,old,row['diagnostics'],mode,True,row['identity']['actual_img_shape'])
        hook_contract(row['spatial_hook'],mode,spec['readout_source'],spec['geometry_source'],row['diagnostics'])
        head_fields = row['head_fields']
        require(set(head_fields) == {'classes', 'masks', 'first_micro_fields'} and
                len(head_fields['classes']) == len(head_fields['masks']) == 10, 'Native ten live fields missing')
        for kind, shape in (('classes', [1, 100, 2]), ('masks', [1, 100, 200, 200])):
            require(all(value == dict(shape=shape, dtype='torch.float32', requires_grad=True)
                        for value in head_fields[kind]), 'Native ten field shape/live/dtype differs')
        if micro == 0:
            initial_fields = head_fields['first_micro_fields']
            require(set(initial_fields) == {'classes', 'masks'} and all(len(initial_fields[key]) == 10
                    for key in initial_fields), 'First supervised native ten digest packet missing')
            for kind, shape in (('classes', [1, 100, 2]), ('masks', [1, 100, 200, 200])):
                for digest in initial_fields[kind]:
                    digest_contract(digest, shape, 'float32')
        else:
            require(head_fields['first_micro_fields'] is None, 'Unexpected extra first-micro packet')
        norms = row['gradient_norms_aligned_active_names']
        require(isinstance(norms, list) and len(norms) == len(policy['active_names']), 'Active-gradient alignment differs')
        for name, value in zip(policy['active_names'], norms):
            if value is not None:
                old.finite(value, 'Accumulated preclip gradient ' + name, True)
                activity[name] += int(value > 0)
                update_counts[name] += int(micro % 2 == 1)
        if micro == 0:
            require(norms[policy['active_names'].index(PREFIX + 'hidden.weight')] == 0,
                    'Cold zero-final implies zero first hidden-weight gradient')
        if micro % 2:
            old.finite(row['preclip_global_norm'], 'Global native-plus-adapter preclip norm', True)
        else:
            require(row['preclip_global_norm'] is None, 'A non-update microstep reported a clip call')
        old.validate_rng(torch, row)
        if external_rng is None:
            external_rng = row['external_RNG_before']
        require(row['external_RNG_before'] == external_rng, 'Restored external RNG changed between micros')
        consumed.extend(ann_ids)
        rows.append(row)
    return rows, consumed, activity, update_counts


def check_training(torch, np, cv2, coco, old, readout, geometry, hooks, spec, result, model, native_initial,
                   init_context, images, image_order, spec_ref, driver_ref, mode='global'):
    require(mode in TRAINING_MODES, 'Actual three independent spatial technical cases required')
    adapter = hooks.register_spatial_mask_readout(torch,readout,geometry,model,mode,
        spec['readout_source'],spec['geometry_source'])
    direct_cold = independent_spatial_state(torch)
    require(set(adapter.state_dict()) == set(direct_cold) and all(torch.equal(adapter.state_dict()[name],value)
            for name,value in direct_cold.items()), 'Spatial training cold weights differ from direct CPU constructor')
    del direct_cold
    initial = dict(model.state_dict())
    policy = old.independent_policy(torch, model, 'mask2former')
    require(policy['state_entries'] == 612 and policy['parameter_tensors'] == 453 and
            policy['trainable_tensors'] == 347 and policy['frozen_tensors'] == 106 and
            len(policy['backbone_conv_names']) == 53 and len(policy['BN_buffer_names']) == 159,
            'Independent native345+adapter2 / frozen BN106+159 / full612 schema differs')
    groups, audit = old.independent_optimizer(torch, model, spec['initializers']['mask2former'], policy)
    require(audit['type'] == 'AdamW' and audit['recipe']['lr'] == 0.0001 and
            audit['clip_grad'] == dict(max_norm=0.01, norm_type=2), 'Native AdamW/.01 clip policy differs')
    checkpoint = torch.load(str(bound(result['checkpoint'])), map_location='cpu')
    require(set(checkpoint) == {'state_dict', 'adapter_state_dict', 'optimizer_state_dict', 'external_rng_state', 'meta'},
            'Full native612 technical checkpoint container differs')
    final, meta = checkpoint['state_dict'], checkpoint['meta']
    proof = read_json(bound(result['state_proof']))
    require(proof['schema'] == 'native-spatial-position-preserving-native-bridge-technical-state-proof-v1' and proof['policy'] == policy and
            old.canonical_sha(proof['optimizer']) == old.canonical_sha(audit), 'Independent typed policy/optimizer differs')
    initialization_proof(torch, old, proof['initializer'], native_initial,
        with_cp=spec['training_backbone_with_cp'], **init_context)
    old.check_state(torch, native_initial, proof['native_cold'], 'training native cold610')
    old.check_state(torch, initial, proof['before'], 'independent native612+fresh adapter before')
    old.check_state(torch, final, proof['after'], 'actual full612 checkpoint after')
    require(set(final) == set(initial), 'Final full state key boundary differs')
    for name, value in final.items():
        require(value.shape == initial[name].shape and value.dtype == initial[name].dtype,
                'Final state shape/dtype differs: ' + name)
    changed = {name: not torch.equal(initial[name], value) for name, value in final.items()}
    require(changed == proof['changed'] and all(not changed[name] for name in
            set(initial) - set(policy['active_names'])), 'Frozen BN affine/buffers or another nonactive state changed')
    model.load_state_dict(final, strict=True)
    require(all(torch.equal(value, final[name]) for name, value in model.state_dict().items()),
            'Strict full612 CPU load differs')
    rows, consumed, activity, counts = check_trace(torch, np, cv2, coco, old, spec, result, policy, images, image_order, mode=mode)
    added = [PREFIX + key for key in ('hidden.weight', 'final.weight')]
    require(proof['adapter_names'] == added and proof['positive_gradient_microcounts'] == activity and
            all(activity[name] > 0 and changed[name] for name in added) and
            proof['native_changed_active_tensors'] == sum(changed[name] for name in policy['active_names'] if name not in added) and
            proof['all_adapter_weights_positive_and_changed'] is True and proof['frozen_BN_all_exact'] is True and
            proof['CPU_checkpoint_readback_exact'] is True and proof['initial_hidden_gradient_zero_is_expected'] is True and
            proof['no_every_native_parameter_positive_claim'] is True,
            'Actual adapter gradient/changed-state/frozen-state evidence differs')
    require(set(checkpoint['adapter_state_dict']) == {'hidden.weight', 'final.weight'} and all(
            torch.equal(checkpoint['adapter_state_dict'][name], final[PREFIX + name])
            for name in checkpoint['adapter_state_dict']), 'Compact adapter alias differs from full checkpoint')
    expected_meta = dict(schema=CP_SCHEMA, mode=mode, phase=spec['phase'], spec=spec_ref, source=driver_ref,
        hooks_source=spec['hooks_source'], readout_source=spec['readout_source'],
        geometry_source=spec['geometry_source'],
        initial_checkpoint=spec['initializers']['mask2former']['checkpoint'], initializer_source=spec['initializer_source'],
        image_ids=list(IMAGE_IDS), consumed_ann_ids=consumed, microsteps=4, optimizer_updates=2, batch_size=1,
        accumulation=2, backbone_with_cp=spec['training_backbone_with_cp'], full_R50_conv_trainable=True,
        BN_stats_and_affine_frozen=True, policy=policy, native_loss_only=True, no_Sobel_or_auxiliary_loss=True,
        not_formal_training=True, eligible_for_formal_resume=False, no_AP=True, no_pending_accumulation_gradients=True)
    for key, value in expected_meta.items():
        require(meta[key] == value, 'Technical checkpoint metadata differs: ' + key)
    require(old.canonical_sha(meta['optimizer']) == old.canonical_sha(audit), 'CP optimizer policy differs')
    optimizer = old.validate_optimizer(torch, checkpoint, groups, audit, counts)
    rng = checkpoint['external_rng_state']
    require(isinstance(rng, list) and len(rng) == 2 and all(isinstance(value, torch.Tensor) and
        value.device.type == 'cpu' and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0
        for value in rng), 'Actual CPU/CUDA external RNG byte tensors invalid')
    require([old.tensor_digest(torch, value, 'uint8') for value in rng] == rows[-1]['external_RNG_after'],
            'Final checkpoint RNG differs from actual restored trace')
    for key, value in dict(mode=mode, readout_source=spec['readout_source'],
        geometry_source=spec['geometry_source'],
        no_pending_accumulation_gradients=True, microsteps=4, optimizer_updates=2, image_ids=list(IMAGE_IDS), consumed_ann_ids=consumed,
        training_forward_uses_native_GT_samples=True, GT_not_used_in_spatial_module=True,
        eligible_for_formal_resume=False, native_loss_only=True, AP_computed=False).items():
        require(result[key] == value, 'Spatial training report differs: ' + key)
    result_proof = dict(mode=mode, readout_source=spec['readout_source'],
        geometry_source=spec['geometry_source'],
        no_pending_accumulation_gradients_source_observed=True,
        checkpoint=result['checkpoint'], state_proof=result['state_proof'], traces=result['traces'],
        native_state_entries=610, full_state_entries=612, parameter_tensors=453, trainability=policy,
        actual_microsteps=4, actual_optimizer_updates=2, all_raw_ann_ids=consumed,
        positive_gradient_microcounts=activity, zero_positive_gradient_names=[name for name in activity if not activity[name]],
        changed_adapter_names=added, native_changed_active_tensors=proof['native_changed_active_tensors'],
        optimizer=optimizer, full_CPU_state_and_frozen_BN_verified=True,
        all_GT_nearest800_independently_reconstructed=True, ten_assignments_retain_all_raw_GT=True,
        native30_only=True, eligible_for_formal_resume=False)
    del model, initial, native_initial, adapter, final, checkpoint
    gc.collect()
    require(not torch.cuda.is_initialized(), 'CPU consumer initialized CUDA')
    return result_proof


def qualify(spec_file, acceptance_file):
    spec_path, target = Path(spec_file), Path(acceptance_file)
    require(spec_path.is_absolute() and target.is_absolute() and not target.exists() and target.parent.is_dir(),
            'Absolute SPEC and fresh acceptance with existing parent required')
    spec = read_json(spec_path)
    require(spec['schema'] == SPEC_SCHEMA and tuple(spec['modes']) == MODES and
            tuple(spec['image_ids']) == IMAGE_IDS and tuple(spec['training_modes']) == TRAINING_MODES and spec['inference_image_id'] == 1 and
            spec['phase'].startswith('native_spatial_position_preserving_native_bridge_preflight_') and
            spec['backbone_with_cp'] is False and type(spec['training_backbone_with_cp']) is bool and
            spec['head_seed_base'] == 100000 and spec['loss_seed_base'] == 1000000 and
            type(spec['deadline_seconds']) is int and spec['deadline_seconds'] > 0 and
            spec['CPU_affinity'] == [0, 1] and spec['allocator_limit_MiB'] == 4096 and
            spec['whole_GPU_owner_limit_MiB'] == 8192, 'Exact technical spatial SPEC differs')
    receipt_path = spec_path.resolve().parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    receipt = read_json(receipt_path)
    terminal_guard(receipt, spec['phase'])
    # No artifact hash, scientific payload or scientific import occurs before this actual terminal gate.
    admission_ref, spec_ref, driver_ref, owner_ref, reused_ref = admission_guard(spec_path, spec, receipt)
    receipt_ref = reference(receipt_path)
    require(spec['geometry_source']['bytes'] == 17500 and spec['geometry_source']['sha256'] == GEOMETRY_SHA and
            spec['readout_source']['bytes'] == 9802 and spec['readout_source']['sha256'] == READOUT_SHA and
            spec['hooks_source']['bytes'] == 21688 and spec['hooks_source']['sha256'] == HOOKS_SHA,
            'Exact sealed spatial/original geometry/whole-head hook sources differ')
    for key in ('initializer_source','micro_preflight_source','native_helpers','readout_source',
                'geometry_source','hooks_source','reused_CPU_reader'):
        require(spec[key] in spec['sources'], 'Runtime SOURCE role absent from owner-bound closure: '+key)
    bind_tree(spec)
    accepted = read_json(bound(spec['initializer_qualification']))
    require(accepted['schema'] == 'ROOT-native-coco-comparator-initializer-CPU-inspection-v1' and
            accepted['all_initializers_strictly_compatible'] is True and accepted['source'] == spec['initializer_source'] and
            set(accepted['initializers']) == {'mask2former', 'condinst', 'cascade_maskrcnn'},
            'Actual qualified three-initializer lineage differs')
    pref_accepted = read_json(bound(spec['micro_preflight_qualification']))
    require(pref_accepted['schema'] == 'ROOT-native-coco-comparator-micro-preflight-CPU-qualification-v1' and
            pref_accepted['status'] == 'ROOT_NATIVE_COCO_COMPARATOR_MICRO_PREFLIGHT_TECHNICALLY_QUALIFIED' and
            pref_accepted['source'] == spec['micro_preflight_source'], 'Original native-loss ROOT qualification differs')
    qualified_spec = read_json(bound(pref_accepted['spec']))
    require(qualified_spec['initializers']['mask2former'] == spec['initializers']['mask2former'] and
            qualified_spec['initializer_source'] == spec['initializer_source'] and
            qualified_spec['initializer_qualification'] == spec['initializer_qualification'],
            'Native checkpoint/config/optimizer/paramwise/clip policy changed since qualification')
    bind_tree(accepted)
    bind_tree(pref_accepted)
    output = Path(spec['output_directory'])
    require(output.is_absolute(), 'Absolute actual output directory required')
    report_path, start_path = output / 'REPORT.json', output / 'START.json'
    report = read_json(report_path)
    for key, value in dict(schema=REPORT_SCHEMA, status='TECHNICAL_SPATIAL_POSITION_PRESERVING_NATIVE_BRIDGE_PREFLIGHT_COMPLETE_NO_METHOD_RESULT',
        phase=spec['phase'], spec=spec_ref, source=driver_ref,
        initializer_qualification=spec['initializer_qualification'], micro_preflight_qualification=spec['micro_preflight_qualification'],
        error=None, no_DEV=True, no_AP=True, cold_inference_GT_removed=True, training_is_supervised_native_loss=True,
        no_Sobel_or_auxiliary_loss=True, formal_resume_eligible=False, scientific_method_decision_computed=False,
        published_recipe_reproduction_claimed=False).items():
        require(report[key] == value, 'Actual completed report differs: ' + key)
    cases = {'inference'} | {mode+'_training' for mode in TRAINING_MODES}
    require(set(report['results']) == cases and set(report['cleanup']) == cases,
            'All three independent fresh training cases and inference cleanups required')
    for row in report['cleanup'].values():
        require(row['allocated_after'] == 0 and type(row['allocated_before']) is int and row['allocated_before'] >= 0,
                'GPU cleanup was not actual exact0')
    require(type(report['torch_peak_allocated_bytes']) is int and 0 <= report['torch_peak_allocated_bytes'] <= 4096 * 1024 ** 2 and
            type(report['elapsed_seconds']) in (int, float) and math.isfinite(report['elapsed_seconds']) and
            0 <= report['elapsed_seconds'] < spec['deadline_seconds'], 'Actual allocator/wall resource scope differs')
    require(read_json(start_path) == dict(schema='native-spatial-position-preserving-native-bridge-technical-start-v1', phase=spec['phase'],
        source=driver_ref, spec=spec_ref, modes=list(MODES), image_ids=list(IMAGE_IDS),
        training_modes=list(TRAINING_MODES), native_runtime_qualification_pending_before_owner_run=True, inference_image_id=1,
        no_formal_resume=True, no_DEV=True, no_AP=True, no_retry=True), 'Actual technical START differs')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    import numpy as np
    import cv2
    import torch
    from pycocotools.coco import COCO
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'CPU01/no initialized CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    cv2.setNumThreads(2)
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    require(len(CocoDataset.METAINFO['classes']) == 80 and CocoDataset.METAINFO['classes'][8] == 'boat',
            'Official COCO boat8 order differs')
    old = load_source(reused_ref, 'independent_original_preflight_CPU_helpers')
    initializer = load_source(spec['initializer_source'], 'qualified_original_initializer_CPU_helpers')
    geometry = load_source(spec['geometry_source'],'sealed_original_geometry_CPU_constructor')
    readout = load_source(spec['readout_source'],'actual_spatial_readout_CPU_constructor')
    hooks = load_source(spec['hooks_source'],'registered_spatial_whole_head_CPU_interface')
    require(tuple(readout.MODES) == MODES and readout.INITIALIZATION_SEED == 2026100911 and
            geometry.INITIALIZATION_SEED == 2026100911 and tuple(hooks.MODES) == MODES and
            hooks.REGISTRATION_NAME == 'spatial_mask_readout', 'Actual sealed spatial modules/prefix differ')
    fit = read_json(bound(spec['FIT']))
    images = {int(row['id']): row for row in fit['images']}
    image_order = [int(row['id']) for row in fit['images']]
    require(len(images) == len(image_order) == 741 and len(fit['annotations']) == 1661 and
            len({int(ann['id']) for ann in fit['annotations']}) == 1661 and set(IMAGE_IDS) <= set(images),
            'Complete native FIT population/unique raw annotations differs')
    annotations = {image_id: [] for image_id in images}
    for ann in fit['annotations']:
        annotations[int(ann['image_id'])].append(ann)
    images['_annotations'] = annotations
    coco = COCO(str(bound(spec['FIT'])))
    rng_before = torch.get_rng_state().clone()
    model, native_initial, init_context = build_native(torch, old, initializer, MODELS,
        revert_sync_batchnorm, Config, spec, accepted['initializers']['mask2former'])
    inference = check_cold_inference(torch,old,readout,geometry,spec,report['results']['inference'],
        native_initial,init_context,images,image_order)
    trainings={}
    initial_digests={name:old.tensor_digest(torch,value,str(value.dtype).replace('torch.',''))
                     for name,value in native_initial.items()}
    for index,mode in enumerate(TRAINING_MODES):
        if index:
            model,native_initial,init_context=build_native(torch,old,initializer,MODELS,
                revert_sync_batchnorm,Config,spec,accepted['initializers']['mask2former'])
            old.check_state(torch,native_initial,initial_digests,'fresh independent common native610')
        trainings[mode]=check_training(torch,np,cv2,coco,old,readout,geometry,hooks,spec,
            report['results'][mode+'_training'],model,native_initial,init_context,images,image_order,
            spec_ref,driver_ref,mode=mode)
        del model,native_initial
        gc.collect()
    require(torch.equal(rng_before, torch.get_rng_state()) and not torch.cuda.is_initialized(),
            'CPU constructor/qualification changed external RNG or initialized CUDA')
    acceptance = dict(schema='ROOT-native-spatial-position-preserving-native-bridge-preflight-CPU-qualification-v1',
        status='ROOT_NATIVE_SPATIAL_POSITION_PRESERVING_NATIVE_BRIDGE_PREFLIGHT_TECHNICALLY_QUALIFIED',
        phase=spec['phase'], spec=spec_ref, source=driver_ref, reader_source=reference(__file__),
        report=reference(report_path), start=reference(start_path), parent_receipt=receipt_ref,
        GPU_admission=admission_ref, owner_driver=owner_ref, reused_independent_CPU_reader=reused_ref,
        initializer_qualification=spec['initializer_qualification'], micro_preflight_qualification=spec['micro_preflight_qualification'],
        inference_proof=inference, training_proofs=trainings,
        checkpointmap={mode:proof['checkpoint'] for mode,proof in trainings.items()},
        cold_modes=list(MODES), training_modes=list(TRAINING_MODES), geometry_source=spec['geometry_source'],
        readout_source=spec['readout_source'], hooks_source=spec['hooks_source'], native_runtime_qualified=True,
        execution_source_binding=dict(admission=admission_ref, spec=spec_ref, entry=driver_ref,
            consumer=reference(__file__), owner=owner_ref, entry_args=['--spec', str(spec_path.resolve()), '--owner-execute'],
            CPU_affinity=[0, 1], CUDA_visible_devices='0', deadline_seconds=spec['deadline_seconds'],
            whole_GPU_used_limit_MiB=8192, automatic_retry=False,
            receipt_admission_SHA_join_verified_before_scientific_access=True),
        terminal_gate_passed_before_scientific_access=True, CPU_affinity=[0, 1], torch_threads=2,
        CUDA_initialized=False, external_CPU_RNG_restored=True, all_raw_GT_preserved=True,
        frozen_BN_affine_and_buffers_independently_exact=True, full_R50_conv_active=True,
        no_Sobel_or_auxiliary_loss=True, formal_training_qualified=False, eligible_for_formal_resume=False,
        AP_computed=False, scientific_method_verdict=None, public_recipe_provenance_verified=False,
        evidence_limits=[
            'Only three cold spatial identity modes and three independent fresh four-micro/two-update cases are technically qualified; no AP, utility or convergence evidence.',
            'CPU independently reconstructs official boat8/null80 initialization, all native/new endpoint tensors, original GT rasters and optimizer membership/moments; it does not replay CUDA forward/backward.',
            'Cold native10/all100 equality and live F/e0/gradients are source-observed runtime evidence checked through consistent digests and bound source, not independently recomputed CPU forward or gradients.',
            'All three cold modes share actual observed small geometry packets; CPU independently checks their digests/counts/fallback and metadata valid fraction, not PCA or M0-sign eligibility. Training cases may naturally acquire different F/e0/geometry.',
            'Native training legally uses original GT; GT-free module/cold forward observations do not imply training is GT-free.',
            'The technical checkpoint is explicitly ineligible for formal resume; no full public pretraining provenance or method superiority is certified.'])
    with target.open('x', encoding='utf-8') as stream:
        json.dump(acceptance, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(dict(status=acceptance['status'], acceptance=reference(target)), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--acceptance', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'SOURCE_ONLY: explicit ROOT CPU execution after owned terminal is required')
    qualify(args.spec, args.acceptance)


if __name__ == '__main__':
    main()
