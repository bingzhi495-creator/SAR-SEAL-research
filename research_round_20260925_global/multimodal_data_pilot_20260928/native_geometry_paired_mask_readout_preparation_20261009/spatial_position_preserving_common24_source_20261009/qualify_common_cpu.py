"""Inert independent CPU consumer of one complete spatial common24 attempt.

All scientific imports, payload reads and artifact hashes require the whole
owned actual exit0/reap and the exact admission/SPEC/source metadata chain.
ROOT alone executes this reader. It audits all six live node snapshots after
24 epochs, without replaying CUDA gradients, geometry, forward or matching.
No scientific execution or payload access occurred during SOURCE preparation.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

KINDS = ('mask2former', 'condinst', 'cascade_maskrcnn')
SPEC_SCHEMA = 'ROOT-native-spatial-position-preserving-common24-spec-v1'
REPORT_SCHEMA = 'native-spatial-position-preserving-common24-report-v1'
CP_SCHEMA = 'native-spatial-position-preserving-common24-checkpoint-v1'
FIT_IMAGES, FIT_GT, TOTAL_EPOCHS = 741, 1661, 24
TOTAL_MICROSTEPS, TOTAL_UPDATES = 17784, 8892
TRAIN_BYTES = 43363
TRAIN_SHA = 'a50b7c95691af7884dcf0bf23ad331d0c1a4055c11caa3da255d0efafb25cc09'
C_ROOT = Path(__file__).resolve().parent.parent
PREF_READER = dict(path=str(C_ROOT.parent / 'native_coco_comparator_micro_preflight_preparation_20261008' / 'qualify_cpu.py'),
    bytes=46683, sha256='ae810fb940fed62588b1a4e0bd7ee199647ff656dbbef5d0bb1247864662d960')
PARENT_READER = dict(path=str(C_ROOT / 'qualify_common_cpu_v2.py'), bytes=60171,
    sha256='5f1eac5900e314556b5041b9c2446b9a872ab0f6875d87f544945102e0d1a559')
MODES = ('native0','global','spatial_halves','spatial_reversed')
NODES = (2,4,8,12,18,24)
ADAPTER_PREFIX = 'panoptic_head.spatial_mask_readout.'
QUAL_SCHEMA = 'ROOT-native-spatial-position-preserving-common24-CPU-qualification-v1'
QUAL_STATUS = 'ROOT_NATIVE_SPATIAL_POSITION_PRESERVING_COMMON24_CPU_QUALIFIED'
READOUT_SHA = '7da9561ed30675fd0d692ce2ad883e0c8cccfddfd9f7913761da0fe4fff0a09d'
GEOMETRY_SHA = '7ad19e356342afc15a2a41f1358634b23486c19021e63ccd94cfed03f6841d15'
HOOKS_SHA = '23f36f3fab9b4a006bd0db65427b68223dc99c4929da458504ed5a3aa9185806'


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
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and
            set(('path', 'bytes', 'sha256')) <= set(row), 'Incomplete reference')
    require(Path(row['path']).is_absolute(), 'Absolute artifact reference required')
    require(reference(row['path']) == {key: row[key] for key in
                                     ('path', 'bytes', 'sha256')},
            'Artifact reference differs: ' + row['path'])
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


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                     allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


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


def formal_admission(spec_path, spec, receipt):
    """Owned whole24 terminal has already passed; bind exact metadata chain."""
    spec_path=spec_path.resolve(strict=True)
    require(isinstance(spec['GPU_admission_file'],str) and Path(spec['GPU_admission_file']).is_absolute(),
            'Absolute current GPU admission string required')
    admission_path=Path(spec['GPU_admission_file']).resolve(strict=True)
    admission_ref=reference(admission_path)
    require(receipt['admission_sha256']==admission_ref['sha256'],'Receipt admission SHA join differs')
    admission=read_json(admission_path)
    spec_ref,reader_ref=reference(spec_path),reference(__file__)
    producer=Path(__file__).resolve().with_name('train_common.py')
    producer_ref=reference(producer)
    owner_ref=reference(spec_path.parent/'finite_point_cache_execution_20261005_v2'/'run_owned_gpu_limited.py')
    require(TRAIN_BYTES>0 and producer_ref['bytes']==TRAIN_BYTES and producer_ref['sha256']==TRAIN_SHA,
            'Final producer SOURCE pin missing or differs')
    require(owner_ref['sha256']=='ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924' and
            admission['driver_sha256']==owner_ref['sha256'],'Owned driver source differs')
    require(admission['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            admission['automatic_retry'] is False and admission['phase']==spec['phase'] and
            admission['allowed_cpus']==spec['CPU_affinity']==[0,1] and admission['cuda_visible_devices']=='0' and
            type(admission['qualified_leaf_deadline_seconds']) is int and
            admission['qualified_leaf_deadline_seconds']==spec['deadline_seconds'] and
            admission['whole_GPU_used_limit_MiB']==spec['whole_GPU_owner_limit_MiB']==8192 and
            Path(admission['entry']).is_absolute() and Path(admission['entry']).resolve(strict=True)==producer and
            admission['entry_args']==['--spec',str(spec_path),'--owner-execute'],
            'Actual whole24 entry/phase/no-retry/CPU/CUDA/resources differ')
    hashes=admission['bound_source_sha256']
    require(isinstance(hashes,dict) and hashes.get(str(spec_path))==spec_ref['sha256'],
            'Actual SPEC not owner bound')
    sources=[]
    for row in spec['sources']:
        bound(row)
        require(hashes.get(row['path'])==row['sha256'],'SOURCE closure not owner bound')
        sources.append({key:row[key] for key in ('path','bytes','sha256')})
    require(producer_ref in sources and reader_ref in sources and PREF_READER in sources and
            PARENT_READER in sources and spec['entry_source']==producer_ref and spec['CPU_reader']==reader_ref,
            'Actual producer/consumer/reused independent helpers not explicitly source bound')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec']==spec_ref,'Admission scientific SPEC differs')
    peak=receipt['whole_GPU_observed_peak_MiB']
    require(type(peak) in (int,float) and math.isfinite(peak) and 0<=peak<=8192 and
            receipt['whole_GPU_used_limit_MiB']==8192 and receipt['whole_leaf_VRAM_certified'] is False and
            receipt['GPU_scope']=='whole GPU observed, not per-process WDDM allocation',
            'Actual whole-GPU observation absent or outside scope')
    return spec_ref,admission_ref,producer_ref,reader_ref,owner_ref


def lr_multiplier(updates, warmup, start, gamma, milestones):
    """Independent hand formula; updates means already completed updates."""
    require(type(updates) is int and updates >= 0 and type(warmup) is int and warmup >= 0,
            'Integer nonnegative scheduler counters required')
    require(type(start) in (int, float) and math.isfinite(start) and 0 < start <= 1 and
            type(gamma) in (int, float) and math.isfinite(gamma) and 0 < gamma <= 1 and
            isinstance(milestones, list) and milestones == sorted(set(milestones)) and
            all(type(mark) is int and mark >= 0 for mark in milestones), 'Scheduler policy invalid')
    ramp = start + (1. - start) * updates / warmup if warmup and updates < warmup else 1.
    return ramp * gamma ** sum(mark <= updates for mark in milestones)



def load_source(row, name):
    loader = importlib.util.spec_from_file_location(name, str(bound(row)))
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def schedule_factor(schedule, updates):
    return lr_multiplier(updates, schedule['warmup_updates'], schedule['warmup_start_factor'],
                         schedule['gamma'], schedule['milestones'])


def independent_plan(np, seed):
    generator = np.random.RandomState(seed)
    orders, states = [], []
    for epoch in range(25):
        state = generator.get_state()
        states.append([state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])])
        if epoch < 24:
            orders.append(generator.permutation(741).tolist())
    require(all(sorted(row) == list(range(741)) for row in orders), 'Noncomplete independent epoch order')
    return dict(seed=seed, orders=orders, RNG_states_at_epoch_boundaries=states,
                order_sha256=canonical_sha(orders), total_epochs=24, private_numpy_rng=True)


def protocol(spec, order_sha, source):
    return dict(kind=spec['kind'],mode=spec['mode'],checkpoint=spec['initializers']['mask2former']['checkpoint'],
        config=spec['initializers']['mask2former']['config'],FIT=spec['FIT'],image_root=spec['image_root'],
        initializers_policy=spec['initializers']['mask2former'],native_helpers=spec['native_helpers'],
        initializer_source=spec['initializer_source'],micro_preflight_source=spec['micro_preflight_source'],
        training_source=source,common_schedule_source=spec['common_schedule_source'],
        readout_source=spec['readout_source'],geometry_source=spec['geometry_source'],hooks_source=spec['hooks_source'],
        native_preflight_source=spec['native_preflight_source'],native_preflight_qualification=spec['native_preflight_qualification'],
        adapter_factory_hidden_seed=2026100911 if spec['mode']!='native0' else None,
        backbone_with_cp=spec['backbone_with_cp'],schedule=spec['schedule'],order_seed=spec['order_seed'],
        order_sha256=order_sha,total_epochs=24,total_microsteps=17784,total_updates=8892,total_raw_GT_visits=39864,
        checkpoint_nodes=list(NODES),batch_size=1,accumulation=2,loss_seed_base=1000000,
        no_augmentation=True,full_R50_conv=True,typed_BN_frozen=True,native_loss_only=True,native_loss_atoms=30,Sobel=False,
        final_mask_logit_readout=spec['mode']!='native0',adapter_geometry_has_no_GT_input=True,
        realized_cross_mode_points_or_eligibility_equal_claimed=False,equal_whole_compute_claimed=False,
        scheduler_index='completed_updates_before_next_step',fresh_single_owned_24=True,resume_allowed=False)

def cpu_state(torch, old, state, proof, label):
    old.check_state(torch, state, proof, label)
    return {name: old.tensor_digest(torch, value, proof[name]['dtype']) for name, value in state.items()}


def optimizer_snapshot(torch, old, state, group_names):
    require(set(state) == {'state', 'param_groups'} and len(state['param_groups']) == len(group_names),
            'Optimizer state container/name groups differ')
    groups, named, ids = [], {}, []
    for group, names in zip(state['param_groups'], group_names):
        require(len(group['params']) == len(names) == 1, 'Exactly one named active parameter per group')
        groups.append(dict(names=names, settings={key: value for key, value in group.items() if key != 'params'}))
        identifier, name = group['params'][0], names[0]
        ids.append(identifier)
        require(name not in named, 'Duplicate named optimizer parameter')
        moments = {}
        for key, value in state['state'].get(identifier, {}).items():
            if isinstance(value, torch.Tensor):
                require(value.device.type == 'cpu' and bool(torch.isfinite(value).all()), 'Nonfinite/nonCPU moment')
                moments[key] = dict(tensor=old.tensor_digest(torch, value, str(value.numpy().dtype)),
                                    scalar=float(value) if value.ndim == 0 else None)
            else:
                require(type(value) in (int, float) and math.isfinite(value), 'Unsupported optimizer scalar')
                moments[key] = dict(value=value)
        named[name] = moments
    require(len(ids) == len(set(ids)) and set(state['state']) <= set(ids), 'Foreign/duplicate optimizer state ID')
    # Match the JSON representation exported by the actual START/final proof.
    # This changes tuple containers to lists, preserving numeric values and all checks.
    return json.loads(json.dumps(dict(parameter_groups=groups, named_state=named, state_entries=len(state['state']),
                optimizer_state_sha256=canonical_sha(dict(groups=groups, state=named))),
                sort_keys=True, allow_nan=False))


def check_rng_tensors(torch, old, values):
    require(isinstance(values, list) and len(values) == 2 and all(
        isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.uint8 and
        value.ndim == 1 and value.numel() > 0 for value in values), 'Complete external CPU/CUDA RNG CPU bytes required')
    return [old.tensor_digest(torch, value, 'uint8') for value in values]


def groups_at(base_groups, schedule, updates):
    multiplier = schedule_factor(schedule, updates)
    return [dict(group, lr=group['lr'] * multiplier) for group in base_groups]


def validate_moments(torch, saved, expected_groups, names, values, family, recipe, counts):
    require(set(saved) == {'state', 'param_groups'} and canonical_sha(saved['param_groups']) == canonical_sha(expected_groups),
            'Independent native optimizer defaults/named groups/scheduled LR differ')
    ids = [group['params'][0] for group in expected_groups]
    require(len(ids) == len(set(ids)) == len(names) and all(len(group['params']) == 1 for group in expected_groups),
            'Unique active optimizer parameter IDs required')
    expected_ids = {identifier for name, identifier in zip(names, ids)
                    if counts[name] and (family == 'AdamW' or recipe.get('momentum', 0) != 0)}
    require(set(saved['state']) == expected_ids, 'Independent per-parameter actual optimizer state membership differs')
    for name, identifier in zip(names, ids):
        require(type(counts[name]) is int and counts[name] >= 0, 'Invalid optimizer update count')
        if identifier not in expected_ids:
            continue
        moment = saved['state'][identifier]
        value = values[name]
        keys = ('exp_avg', 'exp_avg_sq') if family == 'AdamW' else ('momentum_buffer',)
        require(set(moment) == ({'step'} | set(keys) if family == 'AdamW' else set(keys)), 'Native moment fields differ')
        if family == 'AdamW':
            step = moment['step']
            require(isinstance(step, torch.Tensor) and step.device.type == 'cpu' and step.numel() == 1 and
                    bool(torch.isfinite(step).all()) and float(step) == counts[name], 'Actual AdamW step differs')
        for key in keys:
            array = moment[key]
            require(isinstance(array, torch.Tensor) and array.device.type == 'cpu' and array.shape == value.shape and
                    array.dtype == value.dtype and bool(torch.isfinite(array).all()), 'Moment shape/dtype/finite differs')
            if key == 'exp_avg_sq':
                require(bool((array >= 0).all()), 'Negative AdamW squared moment')


def build_initial_native(torch, old, initializer, registry, revert, Config, spec, qualified, initialization):
    kind, entry = spec['kind'], spec['initializers'][spec['kind']]
    accepted = qualified['initializers'][kind]
    require(entry['checkpoint'] == accepted['checkpoint'] and entry['config'] == accepted['official_config'] and
            accepted['strict_official80_and_ship1_compatible'] is True, 'Qualified official COCO initializer lineage differs')
    config = Config.fromfile(str(bound(entry['config'])))
    require(config.model.backbone.style == 'pytorch' and config.model.backbone.depth == 50, 'Official R50 contract differs')
    closure = initializer.config_closure(bound(entry['config']))
    supplied = {row['path']: {key: row[key] for key in ('path', 'bytes', 'sha256')} for row in spec['sources']}
    require(all(supplied.get(row['path']) == row for row in closure), 'Unbound native config inheritance')
    official_cp = torch.load(str(bound(entry['checkpoint'])), map_location='cpu')
    official = old.canonical_state(torch, official_cp)
    prefix_removed = all(key.startswith('module.') for key in official_cp['state_dict'])
    official_model = initializer.build_cpu(torch, registry, revert, config.model)
    official_proof = initializer.strict_load(torch, official_model, official)
    cold, selections = old.independent_ship_rows(torch, kind, official, official_model)
    ship, helper_cold, helper_selections = initializer.ship_state(torch, kind, config.model, official_model, official)
    require(set(cold) == set(helper_cold) and all(torch.equal(cold[name], helper_cold[name]) for name in cold) and
            set(selections) == set(helper_selections) and all(
            selections[name] == helper_selections[name]['selected_rows'] for name in selections),
            'Independent COCO boat8/null80/box32:36 mapping differs')
    del official_cp, official, official_model, helper_cold
    ship.backbone.frozen_stages = -1
    ship.backbone.norm_eval = True
    ship.backbone.norm_cfg.requires_grad = False
    ship.backbone.with_cp = spec['backbone_with_cp']
    ship.data_preprocessor.pad_mask = True
    ship.data_preprocessor.pad_size_divisor = 32
    ship.data_preprocessor.batch_augments = [dict(type='BatchFixedSizePad', size=(800, 800),
        img_pad_value=0, pad_mask=True, mask_pad_value=0, pad_seg=False)]
    model = initializer.build_cpu(torch, registry, revert, ship)
    one_class_proof = initializer.strict_load(torch, model, cold)
    expected = dict(kind=kind, checkpoint=entry['checkpoint'], official_config=entry['config'], official_config_closure=closure,
        canonical_module_prefix_removed=prefix_removed, official80=official_proof, ship1=one_class_proof,
        allowed_changed_keys=helper_selections, initial_ship_tensors_exact=True, only_boat_classifier_rows_changed=True,
        random_missing_keys=False, public_recipe_equivalence_claimed=False, backbone_with_cp=spec['backbone_with_cp'],
        declared_preprocessing_overrides=dict(resize=[800, 800], keep_ratio=True, fixed_canvas=[800, 800],
                                             augmentation=False, mask_resize='nearest'))
    require(set(initialization) == set(expected) | {'initial_state'} and all(initialization[key] == value for key, value in expected.items()),
            'Actual initialization proof differs from independently remapped official native state')
    cpu_state(torch, old, cold, initialization['initial_state'], 'Independently remapped cold official COCO state')
    return model, cold


def build_initial(torch, old, initializer, registry, revert, Config, spec, qualified, initialization):
    extra_keys = {'mode', 'cold_adapter', 'factory_hidden_seed', 'registered_native_boundary_exact'}
    require(extra_keys <= set(initialization), 'Registered cold initialization proof missing')
    native_proof = {key: value for key, value in initialization.items() if key not in extra_keys}
    model, native_cold = build_initial_native(torch, old, initializer, registry, revert, Config,
                                              spec, qualified, native_proof)
    require(len(native_cold) == 610 and len(list(model.parameters())) == 451,
            'Original strict COCO-to-ship native610/451 boundary differs')
    mode = spec['mode']
    adapter_cold = {}
    if mode != 'native0':
        readout=load_source(spec['readout_source'],'sealed_spatial_readout_CPU_factory')
        geometry=load_source(spec['geometry_source'],'sealed_original_geometry_CPU_factory')
        hooks=load_source(spec['hooks_source'],'sealed_whole_head_spatial_CPU_registration')
        require(readout.INITIALIZATION_SEED==geometry.INITIALIZATION_SEED==2026100911 and
                readout.TRAINABLE_TENSORS==2 and readout.TRAINABLE_SCALARS==393216 and
                tuple(readout.MODES)==tuple(hooks.MODES)==MODES[1:],'Exact spatial factory modes/capacity differs')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2026100911)
            hidden=torch.nn.Linear(512,512,bias=False,device='cpu',dtype=torch.float32)
            final=torch.nn.Linear(512,256,bias=False,device='cpu',dtype=torch.float32)
            with torch.no_grad(): final.weight.zero_()
        expected={'hidden.weight':hidden.weight.detach(),'final.weight':final.weight.detach()}
        adapter=hooks.register_spatial_mask_readout(torch,readout,geometry,model,mode,
            spec['readout_source'],spec['geometry_source'])
        require(set(adapter.state_dict())==set(expected) and not list(adapter.buffers()) and
                all(torch.equal(adapter.state_dict()[name],value) for name,value in expected.items()) and
                bool((hidden.weight!=0).any()) and bool((final.weight==0).all()),
                'Independent CPU hidden/zero-final construction differs')
        adapter_cold={name:old.tensor_digest(torch,value,'float32') for name,value in expected.items()}
    cold = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    added = int(mode != 'native0') * 2
    require(len(cold) == 610 + added and len(list(model.parameters())) == 451 + added and
            set(cold) - set(native_cold) == ({ADAPTER_PREFIX + name for name in adapter_cold}) and
            all(torch.equal(cold[name], value) for name, value in native_cold.items()),
            'Optional registration changed the original native boundary')
    require(initialization['mode'] == mode and initialization['cold_adapter'] == adapter_cold and
            initialization['factory_hidden_seed'] == (2026100911 if added else None) and
            initialization['registered_native_boundary_exact'] is True,
            'Actual registered factory proof differs from independent cold CPU tensors')
    return model, cold


def digest_contract(row, shape, dtype):
    require(isinstance(row, dict) and set(row) == {'shape', 'dtype', 'sha256'} and
            row['shape'] == shape and row['dtype'] == dtype and isinstance(row['sha256'], str) and
            len(row['sha256']) == 64 and set(row['sha256']) <= set('0123456789abcdef'),
            'Observed tensor digest contract differs')


def validate_native_observations(torch, old, spec, row, ann_ids):
    mode, G = spec['mode'], len(ann_ids)
    require(row['mode'] == mode and row['original_native_loss_atoms'] == 30 and
            row['support_statistics_are_posterior_only'] is True and row['support_used_for_loss_or_selection'] is False,
            'Mode/native30/posterior-only support scope differs')
    fields = row['native_head_fields']
    require(fields['readouts'] == 10 and fields['classes_shape'] == [1, 100, 2] and
            fields['masks_shape'] == [1, 100, 200, 200] and fields['all_fields_require_grad'] is True,
            'Actual original ten native readouts/query count differs')
    cold = fields['cold_micro_digests']
    if row['global_microstep'] == 0:
        require(set(cold) == {'classes', 'masks'} and len(cold['classes']) == len(cold['masks']) == 10,
                'First global micro native ten-field digests missing')
        for value in cold['classes']:
            digest_contract(value, [1, 100, 2], 'float32')
        for value in cold['masks']:
            digest_contract(value, [1, 100, 200, 200], 'float32')
    else:
        require(cold is None, 'Unexpected cold-digest packet after global micro0')
    matches = row['native_matches']
    require(isinstance(matches, list) and len(matches) == 10, 'All ten native matching observations required')
    for match in matches:
        q, assigned = match['positive_query_indices'], match['assigned_GT_indices']
        require(type(match['positives']) is int and match['positives'] == G == len(q) == len(assigned) and
                len(q) == len(set(q)) and all(type(value) is int and 0 <= value < 100 for value in q) and
                all(type(value) is int for value in assigned) and sorted(assigned) == list(range(G)),
                'Observed native one-to-one matches omitted/duplicated raw GT')
        labels = torch.ones(100, dtype=torch.int64)
        labels[q] = 0
        require(old.tensor_digest(torch, labels, 'int64') == match['target_labels'],
                'Native Q100 actual target-label digest differs from observed assignment')
    hook, diagnostics, support = row['spatial_hook'], row['adapter_diagnostics'], row['final_positive_GT_eligibility']
    if mode == 'native0':
        require(hook is None and diagnostics is None and support is None, 'Native0 acquired an adapter/hook/support route')
        return
    expected_hook=dict(forward_calls=1,completed_forwards=1,pixel_decoder_calls=1,coefficient_calls=10,
        passive_coefficient_captures=1,unchanged_early_readouts=9,final_mask_replacements=1,adapter_calls=1,
        apply=True,mode=mode,readout_source=spec['readout_source'],geometry_source=spec['geometry_source'],
        metadata_fields_used=['img_shape','batch_input_shape'],class_list_object_preserved=True,
        first_nine_mask_objects_preserved=True,native_coefficient_outputs_untouched=True,
        earlier_gate_paths_untouched=True,all_queries_retained=True,
        half_eligible_counts=[diagnostics['half_eligible_counts']],global_fallback=[diagnostics['global_fallback']],
        adapter_geometry_phi_skipped=False,cache_cleared_after_forward=True,registered_state_not_pruned=True,
        hooks_removed=True,feature_shape=[1,256,200,200],coefficient_shape=[1,100,256],
        final_native_mask_shape=[1,100,200,200])
    require(hook==expected_hook,'Actual final whole-head replacement/observer scope differs')
    require(diagnostics['mode'] == mode and diagnostics['eligibility_frame'] ==
            'canonical' and diagnostics['all_queries'] == 100 and
            diagnostics['input_F_live'] is True and diagnostics['input_e0_live'] is True and
            diagnostics['residual_live'] is True and diagnostics['geometry_stop_gradient'] is True and
            diagnostics['GT_used_for_geometry'] is False and diagnostics['no_query_dropped'] is True,
            'Actual live F/e0 and detached GT-free geometry observation differs')
    counts = diagnostics['eligible_counts_by_query']
    require(isinstance(counts, list) and len(counts) == 100 and all(type(value) is int and 0 <= value <= 16 for value in counts),
            'Complete all100 eligibility counts required')
    require(diagnostics['eligible_pairs_total'] == sum(counts) and
            diagnostics['queries_with_eligible_pairs'] == sum(value > 0 for value in counts) and
            type(diagnostics['queries_with_support']) is int and
            diagnostics['queries_with_eligible_pairs'] <= diagnostics['queries_with_support'] <= 100 and
            type(diagnostics['canonical_eligible_pairs_total']) is int and
            0 <= diagnostics['canonical_eligible_pairs_total'] <= 1600,
            'Observed query support aggregate arithmetic differs')
    halves,fallback=diagnostics['half_eligible_counts'],diagnostics['global_fallback']
    require(isinstance(halves,list) and len(halves)==100 and isinstance(fallback,list) and len(fallback)==100 and
            all(isinstance(value,list) and len(value)==2 and all(type(n) is int and 0<=n<=8 for n in value)
                for value in halves) and all(type(v) is bool for v in fallback) and
            all(sum(halves[q])==counts[q] and fallback[q]==any(n==0 for n in halves[q]) for q in range(100)) and
            diagnostics['canonical_eligible_pairs_total']==sum(counts),'Observed half counts/fallback arithmetic differs')
    final = matches[-1]
    expected = [dict(annotation_id=ann_ids[gi], assigned_GT_index=gi, query_index=q, eligible_pairs=counts[q])
                for q, gi in zip(final['positive_query_indices'], final['assigned_GT_indices'])]
    require(support == expected, 'Posterior support did not join actual final assigned raw GT order')


def validate_trace(torch, np, cv2, coco, old, spec, path, policy, images, image_order, plan, start_rng):
    kind=spec['kind']
    order=[index for epoch_order in plan['orders'] for index in epoch_order]
    active, seen = policy['active_names'], 0
    positive = {name: 0 for name in active}
    updates = {name: 0 for name in active}
    epoch_GT={epoch:0 for epoch in range(TOTAL_EPOCHS)}
    prefixes={}
    GT_cache, previous_rng, consumed = {}, start_rng, 0
    first_row, boundary = None, []
    with Path(path).open('r', encoding='utf-8') as stream:
        for local, line in enumerate(stream):
            require(local < TOTAL_MICROSTEPS and line.strip(), 'Extra/blank micro trace row')
            row = json.loads(line)
            micro = local
            epoch, within = divmod(micro, 741)
            index = order[local]
            image_id = image_order[index]
            step, u = (micro + 1) % 2 == 0, micro // 2
            require(row['kind'] == kind and row['global_microstep'] == micro and row['segment_microstep'] == local and
                    row['epoch'] == epoch and row['within_epoch'] == within and row['identity']['dataset_index'] == index and
                    row['identity']['image_id'] == image_id and row['accumulation_divisor'] == 2 and
                    row['optimizer_step'] is step and row['global_updates'] == (micro + 1) // 2 and
                    row['scheduler_u_before_step'] == u and row['lr_factor'] == schedule_factor(spec['schedule'], u) and
                    row['gradients_are_accumulated_preclip'] is True and row['private_seed'] == 1000000 + micro,
                    'Actual complete epoch/order/global odd-even accumulation/LR/seed contract differs')
            require(row['gradient_clipping_applied'] is (step and spec['initializers'][kind]['clip_grad'] is not None),
                    'Actual nullable native clipping application differs')
            ann_ids = [int(ann['id']) for ann in coco.imgToAnns.get(image_id, [])]
            validate_native_observations(torch, old, spec, row, ann_ids)
            if image_id not in GT_cache:
                adapted = dict(row, actual_input_shape=row['input']['shape'])
                old.validate_GT(torch, np, cv2, coco, images[image_id], ann_ids, adapted)
                GT_cache[image_id] = (row['identity'], row['GT'], row['input'])
            else:
                require(GT_cache[image_id] == (row['identity'], row['GT'], row['input']),
                        'Same nonaugmented image/complete raw GT/normalized input differs across the complete24 epochs')
            require(row['native_model_loss_only'] is True, 'Trace is not native model.loss')
            old.validate_losses(dict(row, loss_is_native_model_loss=True, Sobel_added=False), kind)
            old.validate_rng(torch, row)
            require(row['external_RNG_before'] == previous_rng, 'External RNG continuity/actual START restore differs')
            previous_rng = row['external_RNG_after']
            norms = row['gradient_norms_aligned_active_names']
            require(isinstance(norms, list) and len(norms) == len(active), 'Complete aligned active gradient vector required')
            for name, magnitude in zip(active, norms):
                if magnitude is not None:
                    old.finite(magnitude, 'Actual accumulated gradient '+name, True)
                    positive[name] += int(magnitude > 0)
                    if step:
                        updates[name] += 1
            require(row['frozen_BN_parameter_grads_None'] is True and
                    row['BN_module_training'] == {name: False for name in policy['BN_modules']},
                    'Actual typed BN mode/affine gradients differ')
            if step:
                old.finite(row['preclip_global_norm'], 'Actual native global gradient norm', True)
            else:
                require(row['preclip_global_norm'] is None, 'Unexpected step/clip at odd accumulated micro')
            if within in (0,740) and (epoch % 2 == 0 or within == 740):
                boundary.append(dict(global_microstep=micro, epoch=epoch, within_epoch=within,
                                     optimizer_step=step, global_updates=row['global_updates']))
            if first_row is None:
                first_row = row
            epoch_GT[epoch] += len(ann_ids)
            consumed += len(ann_ids)
            seen += 1
            if within==740 and epoch+1 in NODES:
                prefixes[epoch+1]=dict(rows=seen,raw_GT=consumed,positive=dict(positive),
                    per_parameter_updates=dict(updates),last_external_rng=previous_rng)
    require(seen==TOTAL_MICROSTEPS and len(GT_cache)==FIT_IMAGES and consumed==TOTAL_EPOCHS*FIT_GT and
            all(value==FIT_GT for value in epoch_GT.values()) and set(prefixes)==set(NODES),
            'Not complete24 epochs/all six node prefixes/all raw GT retained')
    return dict(rows=seen, raw_GT=consumed, epoch_GT=epoch_GT, positive=positive, per_parameter_updates=updates,
                last_external_rng=previous_rng, odd_epoch_boundary=boundary, first_row=first_row, prefixes=prefixes)



def qualify_endpoint(torch,np,cv2,coco,old,initializer,registry,revert,Config,
                     spec,report,source,spec_ref,plan,initialized,images,image_order):
    kind,mode,result=spec['kind'],spec['mode'],report['result']
    require(result['kind']==kind and result['mode']==mode and set(result['nodes'])=={str(e) for e in NODES},
            'Actual one-mode full24/six nodes missing')
    initialization=read_json(bound(result['initialization_proof']))
    model,cold=build_initial(torch,old,initializer,registry,revert,Config,spec,initialized,initialization)
    policy=old.independent_policy(torch,model,kind)
    added=int(mode!='native0')*2
    require(policy['trainable_tensors']==345+added and policy['parameter_tensors']==451+added and
            policy['state_entries']==610+added and policy['frozen_tensors']==106 and
            len(policy['BN_buffer_names'])==159 and len(policy['backbone_conv_names'])==53,
            'Independent native345/610 versus spatial347/612 typed policy differs')
    base_groups,audit=old.independent_optimizer(torch,model,spec['initializers'][kind],policy)
    require(audit['type']=='AdamW','Native M2F AdamW required')
    active=policy['active_names']; names=[[name] for name in active]
    base_lrs=[group['lr'] for group in base_groups]
    signature=protocol(spec,plan['order_sha256'],source)
    loaded=read_json(bound(result['start_state_proof']))
    expected_resume=dict(resumed=False,checkpoint=None,qualification=None,global_microsteps=0,
        global_updates=0,consumed_raw_GT=0,cold_ship_initialization=True)
    require(loaded['kind']==kind and loaded['mode']==mode and loaded['policy']==policy and
            loaded['base_lrs']==base_lrs and loaded['training_protocol']==signature and loaded['resume']==expected_resume,
            'Actual START is not the original fresh cold model/empty optimizer')
    cpu_state(torch,old,cold,loaded['actual_loaded_model_state'],'Independent original fresh START state')
    require(loaded['actual_optimizer']==optimizer_snapshot(torch,old,dict(state={},param_groups=base_groups),names),
            'Actual START optimizer settings/membership differ from independent cold optimizer')
    require(isinstance(loaded['external_RNG'],list) and len(loaded['external_RNG'])==2,
            'START actual CPU/CUDA RNG summaries missing')
    for rng_digest in loaded['external_RNG']:
        require(isinstance(rng_digest,dict) and set(rng_digest)=={'shape','dtype','sha256'} and
                rng_digest['dtype']=='uint8' and isinstance(rng_digest['shape'],list) and
                len(rng_digest['shape'])==1 and type(rng_digest['shape'][0]) is int and rng_digest['shape'][0]>0,
                'START RNG byte digest contract differs')
        digest_contract(rng_digest,rng_digest['shape'],'uint8')
    trace=validate_trace(torch,np,cv2,coco,old,spec,bound(result['trace']),policy,images,image_order,
                         plan,loaded['external_RNG'])
    epoch_rows=read_json(bound(result['epoch_rows']))
    expected_epochs=[dict(epoch=e,dataset_indices_sha256=canonical_sha(plan['orders'][e]),images=741,
        raw_GT=1661,global_microsteps=(e+1)*741,global_updates=(e+1)*741//2,
        pending_accumulation_microsteps=((e+1)*741)%2) for e in range(24)]
    require(epoch_rows==expected_epochs,'Actual continuous24 epoch/order/odd accumulation boundaries differ')
    fixed=set(cold)-set(active)
    native_cold={name:value for name,value in cold.items() if not name.startswith(ADAPTER_PREFIX)}
    extra=[ADAPTER_PREFIX+name for name in ('hidden.weight','final.weight')] if added else []
    nodes={}
    for epoch in NODES:
        node=result['nodes'][str(epoch)]; prefix=trace['prefixes'][epoch]
        require(prefix['rows']==epoch*FIT_IMAGES and prefix['raw_GT']==epoch*FIT_GT,
                'Actual node trace prefix population differs')
        cp=torch.load(str(bound(node['checkpoint'])),map_location='cpu')
        require(isinstance(cp,dict) and set(cp)=={'state_dict','optimizer_state_dict',
                'external_rng_state','private_order_state','meta'},'Complete full node CPU checkpoint required')
        final,meta=cp['state_dict'],cp['meta']
        U=epoch*FIT_IMAGES//2
        expected_meta=dict(schema=CP_SCHEMA,kind=kind,mode=mode,phase=spec['phase'],spec=spec_ref,source=source,
            training_protocol=signature,initial_epoch=0,completed_epoch=epoch,global_microsteps=epoch*FIT_IMAGES,
            global_updates=U,consumed_raw_GT=epoch*FIT_GT,optimizer_param_names=names,base_lrs=base_lrs,
            stored_lr_factor=schedule_factor(spec['schedule'],U-1),resume_checkpoint=None,resume_qualification=None,
            no_pending_accumulation_gradients=True,not_technical_preflight_checkpoint=True,
            fresh_single_owned_24=True,eligible_for_automatic_resume=False)
        require(set(meta)==set(expected_meta)|{'optimizer'} and all(meta[key]==value for key,value in expected_meta.items())
                and canonical_sha(meta['optimizer'])==canonical_sha(audit),'Actual node metadata/protocol/counters differ')
        proof=read_json(bound(node['state_proof']))
        expected_proof=dict(schema='native-spatial-position-preserving-common24-node-state-proof-v1',
            kind=kind,mode=mode,completed_epoch=epoch,policy=policy,original_checkpoint=spec['initializers'][kind]['checkpoint'],
            before_is_original_fresh_cold=True,initialization_proof=result['initialization_proof'],
            native_microsteps=epoch*FIT_IMAGES,optimizer_updates=U,raw_GT_visits=epoch*FIT_GT,
            no_pending_accumulation_gradients=True,node_does_not_restart_training=True,
            gradient_activity_is_accumulated=True,frozen_BN_all_exact=True,all_final_state_finite=True,
            CPU_checkpoint_readback_exact=True)
        require(all(proof[key]==value for key,value in expected_proof.items()) and
                canonical_sha(proof['optimizer'])==canonical_sha(audit),'Actual node state proof metadata differs')
        cpu_state(torch,old,native_cold,proof['original_native_cold'],'Independent original native610')
        cpu_state(torch,old,cold,proof['cold'],'Independent full cold native/optional spatial weights')
        cpu_state(torch,old,cold,proof['before'],'Node before means original fresh cold, not a resumed snapshot')
        cpu_state(torch,old,final,proof['after'],'Actual full node CPU state')
        require(set(final)==set(cold) and all(value.shape==cold[name].shape and value.dtype==cold[name].dtype
                for name,value in final.items()),'Full node keys/shapes/dtypes differ')
        require(all(torch.equal(final[name],cold[name]) for name in fixed),'A nonactive original tensor/BN state changed')
        changed={name:not torch.equal(final[name],cold[name]) for name in final}
        require(proof['changed']==changed and proof['positive_gradient_microcounts']==prefix['positive'] and
                proof['optimizer_step_nonNone_gradient_counts']==prefix['per_parameter_updates'] and
                any(changed[name] and prefix['positive'][name]>0 for name in active) and
                all(changed[name] and prefix['positive'][name]>0 for name in extra),
                'Node genuine-gradient activity/changed tensors differ; weight decay alone is insufficient')
        validate_moments(torch,cp['optimizer_state_dict'],groups_at(base_groups,spec['schedule'],U-1),active,final,
                         audit['type'],audit['recipe'],prefix['per_parameter_updates'])
        require(proof['actual_final_optimizer']==optimizer_snapshot(torch,old,cp['optimizer_state_dict'],names),
                'Node optimizer digest/settings differ from independently CPU-read actual moments')
        rng=check_rng_tensors(torch,old,cp['external_rng_state'])
        require(rng==prefix['last_external_rng'] and cp['private_order_state']==dict(seed=plan['seed'],
            order_sha256=plan['order_sha256'],next_epoch=epoch,numpy_state=plan['RNG_states_at_epoch_boundaries'][epoch]),
            'Node actual external RNG or private continuous24 order position differs')
        expected_result=dict(completed_epoch=epoch,global_microsteps=epoch*FIT_IMAGES,global_updates=U,
            consumed_raw_GT=epoch*FIT_GT,CPU_checkpoint_readback_exact=True,frozen_BN_all_exact=True,
            no_pending_accumulation_gradients=True,stored_lr_factor=schedule_factor(spec['schedule'],U-1),
            node_does_not_restart_training=True)
        require(all(node[key]==value for key,value in expected_result.items()),'Actual node result counters/LR differ')
        model.load_state_dict(final,strict=True)
        require(all(torch.equal(value,final[name]) for name,value in model.state_dict().items()),
                'Strict full node CPU state readback differs')
        nodes[str(epoch)]=dict(checkpoint=node['checkpoint'],state_proof=node['state_proof'],completed_epoch=epoch,
            global_microsteps=epoch*FIT_IMAGES,global_updates=U,consumed_raw_GT=epoch*FIT_GT,
            optimizer_update_counts_per_parameter=prefix['per_parameter_updates'],positive_gradient_microcounts=prefix['positive'],
            changed=changed,endpoint_external_RNG=rng,stored_lr_factor=schedule_factor(spec['schedule'],U-1),
            actual_unchanged_active_names=[name for name in active if not changed[name]],
            zero_positive_gradient_names=[name for name in active if not prefix['positive'][name]],
            full_CPU_state_optimizer_frozen_BN_verified=True,checkpoint_CPU_strict_loaded=True)
        del cp,final,proof
        gc.collect()
    require(result['checkpoint']==result['nodes']['24']['checkpoint'] and
            result['state_proof']==result['nodes']['24']['state_proof'],'Actual final state/checkpoint is not E24 node')
    for key,value in dict(completed_epoch=24,global_microsteps=17784,global_updates=8892,consumed_raw_GT=39864,
            CPU_checkpoint_readback_exact=True,frozen_BN_all_exact=True,single_owned_full24=True,
            no_pending_accumulation_gradients=True).items():
        require(result[key]==value,'Actual full24 result differs: '+key)
    require(not torch.cuda.is_initialized(),'Independent CPU check initialized CUDA')
    technical=dict(full_native_state_CPU_finite_verified=True,all_nonactive_states_original_exact=True,
        original_typed_BN_affine_and_buffers_exact=True,every_actual_R50_conv_active=True,
        independent_boat8_null80_mapping_verified=True,full_optimizer_CPU_moments_groups_verified=True,
        original_fresh_model_empty_optimizer_START_verified=True,external_RNG_node_CP_and_trace_verified=True,
        order24_independently_recomputed=True,complete_741_and_1661_per_epoch=True,all_six_nodes_verified=True,
        odd_epoch_boundaries=trace['odd_epoch_boundary'],actual_ten_stage_assignment_observations_complete=True,
        assignment_costs_or_CUDA_matching_independently_recomputed=False,geometry_or_M0_sign_reconstructed=False,
        cross_mode_query_semantics_or_realized_support_equality_claimed=False,CUDA_training_trajectory_replayed=False,
        gradient_elements_independently_recomputed=False,node_state_or_RNG_restarts_observed=False)
    refs={key:result[key] for key in ('initialization_proof','state_proof','start_state_proof','trace','epoch_rows')}
    out=dict(checkpoint=result['checkpoint'],training_protocol=signature,completed_epoch=24,global_microsteps=17784,
        global_updates=8892,consumed_raw_GT=39864,optimizer_update_counts_per_parameter=trace['per_parameter_updates'],
        positive_gradient_microcounts=trace['positive'],changed=nodes['24']['changed'],trainable=policy,optimizer=audit,
        endpoint_external_RNG=trace['last_external_rng'],nodes=nodes,technical_proof=technical,proof_refs=refs)
    del model,cold
    gc.collect()
    return out


def qualify(spec_file,acceptance_file):
    spec_path,target=Path(spec_file),Path(acceptance_file)
    require(spec_path.is_absolute() and target.is_absolute() and not target.exists() and target.parent.is_dir(),
            'Absolute SPEC/fresh acceptance with existing parent required')
    spec=read_json(spec_path)
    require(spec['schema']==SPEC_SCHEMA and spec['kind']=='mask2former' and spec['mode'] in MODES and
            isinstance(spec['admitted_modes'],list) and len(spec['admitted_modes'])==len(set(spec['admitted_modes'])) and
            tuple(spec['admitted_modes'])==MODES and spec['mode'] in spec['admitted_modes'] and
            spec['phase'].startswith('native_spatial_position_preserving_common24_') and spec['total_epochs']==24 and
            type(spec['initial_epoch']) is int and spec['initial_epoch']==0 and spec['epochs_this_run']==24 and spec['checkpoint_nodes']==list(NODES) and
            spec['resume_checkpoint'] is None and spec['resume_qualification'] is None and
            type(spec['order_seed']) is int and spec['order_seed']==2026100917 and type(spec['backbone_with_cp']) is bool,
            'Fresh one-mode whole24/six nodes/no-resume contract differs')
    receipt_path=spec_path.resolve().parent/'finite_point_cache_execution_20261005_v2'/'runs'/spec['phase']/'PARENT_RECEIPT.json'
    receipt=read_json(receipt_path)
    terminal_guard(receipt,spec['phase'])
    # Whole owned actual exit0/reap precedes any source/artifact hash, SCI import or payload.
    spec_ref,admission_ref,source,reader_ref,owner_ref=formal_admission(spec_path,spec,receipt)
    require(spec['allocator_limit_MiB']==4096 and spec['whole_GPU_owner_limit_MiB']==8192 and
            type(spec['deadline_seconds']) is int and 0<spec['deadline_seconds']<=10800 and
            Path(spec['image_root']).is_absolute(),'Locked whole24 allocator/deadline/image root differs')
    require(spec['readout_source']['bytes']==9802 and spec['readout_source']['sha256']==READOUT_SHA and
            spec['geometry_source']['bytes']==17500 and spec['geometry_source']['sha256']==GEOMETRY_SHA and
            spec['hooks_source']['bytes']==21688 and spec['hooks_source']['sha256']==HOOKS_SHA,
            'Exact readout/original geometry/whole-head bridge SOURCE roles differ')
    for key in ('readout_source','geometry_source','hooks_source','native_preflight_source',
                'native_preflight_CPU_reader','common_schedule_source','native_helpers','initializer_source','micro_preflight_source'):
        require(spec[key] in spec['sources'],'Explicit owner-bound SOURCE role missing: '+key)
    schedule=spec['schedule']
    require(set(schedule)=={'warmup_updates','warmup_start_factor','milestones','gamma'} and
            type(schedule['warmup_updates']) is int and 0<=schedule['warmup_updates']<8892 and
            (schedule['warmup_updates']>0 or schedule['warmup_start_factor']==1) and
            schedule['milestones']==[7904,8563],'Fixed update-index LR schedule differs')
    schedule_factor(schedule,0)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='2'
    os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
    import psutil
    psutil.Process().cpu_affinity([0,1])
    bind_tree(spec)
    initialized=read_json(bound(spec['initializer_qualification']))
    micro=read_json(bound(spec['micro_preflight_qualification']))
    require(initialized['schema']=='ROOT-native-coco-comparator-initializer-CPU-inspection-v1' and
            initialized['all_initializers_strictly_compatible'] is True and initialized['source']==spec['initializer_source'] and
            set(initialized['initializers'])==set(KINDS),'Qualified original official COCO initializer differs')
    require(micro['schema']=='ROOT-native-coco-comparator-micro-preflight-CPU-qualification-v1' and
            micro['status']=='ROOT_NATIVE_COCO_COMPARATOR_MICRO_PREFLIGHT_TECHNICALLY_QUALIFIED' and
            micro['source']==spec['micro_preflight_source'] and micro['reader_source']==PREF_READER,
            'Qualified native loss technical lineage differs')
    bind_tree(initialized); bind_tree(micro)
    technical=read_json(bound(spec['native_preflight_qualification']))
    require(technical['schema']=='ROOT-native-spatial-position-preserving-native-bridge-preflight-CPU-qualification-v1' and
            technical['status']=='ROOT_NATIVE_SPATIAL_POSITION_PRESERVING_NATIVE_BRIDGE_PREFLIGHT_TECHNICALLY_QUALIFIED' and
            technical['source']==spec['native_preflight_source'] and technical['reader_source']==spec['native_preflight_CPU_reader'] and
            technical['native_runtime_qualified'] is True and technical['eligible_for_formal_resume'] is False and
            technical['formal_training_qualified'] is False and technical['AP_computed'] is False,
            'Qualified spatial technical seam required, never its technical CP as formal initializer')
    bind_tree(technical)
    technical_spec=read_json(bound(technical['spec']))
    for key in ('readout_source','geometry_source','hooks_source','native_helpers','initializer_source'):
        require(technical_spec[key]==spec[key],'Qualified technical SOURCE seam differs: '+key)
    for key in ('checkpoint','config'):
        require(technical_spec['initializers']['mask2former'][key]==spec['initializers']['mask2former'][key],
                'Qualified technical native architecture/COCO initializer differs')
    require(technical_spec['initializers']['mask2former']==spec['initializers']['mask2former'],
            'Qualified technical native optimizer/paramwise/clip policy differs')
    require(spec['initializers']['mask2former']['optimizer']['type']=='AdamW' and
            spec['initializers']['mask2former']['optimizer']['lr']==0.0001 and
            spec['initializers']['mask2former']['clip_grad']==dict(max_norm=0.01,norm_type=2),
            'Explicit fixed native AdamW/1e-4/.01 clip contract differs')
    require(technical_spec['training_backbone_with_cp'] is spec['backbone_with_cp'],
            'Qualified native checkpointing architecture differs')
    output=Path(spec['output_directory']); require(output.is_absolute(),'Absolute formal output required')
    report_path,start_path=output/'REPORT.json',output/'START.json'
    report=read_json(report_path)
    expected=dict(schema=REPORT_SCHEMA,status='NATIVE_SPATIAL_COMMON24_OWNED_COMPLETE_NO_AP',phase=spec['phase'],
        source=source,spec=spec_ref,kind=spec['kind'],mode=spec['mode'],error=None,
        initializer_qualification=spec['initializer_qualification'],micro_preflight_qualification=spec['micro_preflight_qualification'],
        native_preflight_qualification=spec['native_preflight_qualification'],readout_source=spec['readout_source'],
        geometry_source=spec['geometry_source'],hooks_source=spec['hooks_source'],initial_epoch=0,epochs_this_run=24,
        total_epochs=24,checkpoint_nodes=list(NODES),full_R50_conv_trainable=True,typed_BN_frozen=True,native_loss_only=True,
        original_native_loss_atoms=30,adapter_geometry_has_no_GT_input=True,
        realized_cross_mode_points_or_eligibility_equal_claimed=False,unchanged_class_scores_after_training_claimed=False,
        no_inference=True,no_DEV=True,no_AP=True,GT_free_inference_runtime_verified=False,
        single_owned_fresh_full24=True,no_resume=True,equal_whole_compute_claimed=False,
        scientific_method_decision_computed=False,published_recipe_reproduction_claimed=False)
    require(all(report[key]==value for key,value in expected.items()),'Actual completed whole24 REPORT differs')
    require(report['cleanup']['allocated_after']==0,'Actual GPU cleanup is not exact0')
    require(type(report['torch_peak_allocated_bytes']) is int and 0<=report['torch_peak_allocated_bytes']<=4096*1024**2 and
            type(report['elapsed_seconds']) in (int,float) and math.isfinite(report['elapsed_seconds']) and
            0<=report['elapsed_seconds']<spec['deadline_seconds'],'Actual allocator/deadline scope differs')
    require(read_json(start_path)==dict(schema='native-spatial-position-preserving-common24-start-v1',phase=spec['phase'],
        source=source,spec=spec_ref,kind=spec['kind'],mode=spec['mode'],initial_epoch=0,epochs_this_run=24,total_epochs=24,
        checkpoint_nodes=list(NODES),no_resume=True,no_AP=True,no_DEV=True,no_retry=True),'Actual START contract differs')
    bind_tree(report)
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
    require(psutil.Process().cpu_affinity()==[0,1] and not torch.cuda.is_initialized(),'CPU01/no initialized CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2); cv2.setNumThreads(2)
    register_all_modules(init_default_scope=False); init_default_scope('mmdet')
    require(len(CocoDataset.METAINFO['classes'])==80 and CocoDataset.METAINFO['classes'][8]=='boat',
            'Official COCO boat8 class order differs')
    old=load_source(PREF_READER,'qualified_original_independent_CPU_helpers')
    initializer=load_source(spec['initializer_source'],'qualified_official_initializer_CPU_functions')
    fit=read_json(bound(spec['FIT']))
    image_order=[int(row['id']) for row in fit['images']]; images={int(row['id']):row for row in fit['images']}
    require(len(images)==len(image_order)==741 and len(fit['annotations'])==1661 and
            len({int(ann['id']) for ann in fit['annotations']})==1661,'Full original FIT741/1661 population differs')
    coco=COCO(str(bound(spec['FIT'])))
    plan=independent_plan(np,spec['order_seed'])
    require(read_json(bound(report['order']))==plan and report['order_sha256']==plan['order_sha256'],
            'All24 private permutations and boundary RNG states differ')
    before_rng=torch.get_rng_state().clone()
    proof=qualify_endpoint(torch,np,cv2,coco,old,initializer,MODELS,revert_sync_batchnorm,Config,
        spec,report,source,spec_ref,plan,initialized,images,image_order)
    require(torch.equal(before_rng,torch.get_rng_state()) and not torch.cuda.is_initialized(),
            'CPU reconstruction consumed external RNG or initialized CUDA')
    require(report['checkpoint']==proof['checkpoint'] and report['nodes']==report['result']['nodes'] and
            report['completed_epoch']==24 and report['global_microsteps']==17784 and report['global_updates']==8892,
            'Actual top-level node/final counters do not join')
    acceptance=dict(schema=QUAL_SCHEMA,status=QUAL_STATUS,kind=spec['kind'],mode=spec['mode'],phase=spec['phase'],
        source=source,reader_source=reader_ref,spec=spec_ref,report=reference(report_path),start=reference(start_path),
        parent_receipt=reference(receipt_path),GPU_admission=admission_ref,owner_driver=owner_ref,
        checkpoint=proof['checkpoint'],checkpointmap={spec['mode']:proof['checkpoint']},
        node_checkpointmap={epoch:node['checkpoint'] for epoch,node in proof['nodes'].items()},
        completed_epoch=24,global_microsteps=17784,global_updates=8892,consumed_raw_GT=39864,
        optimizer_update_counts_per_parameter=proof['optimizer_update_counts_per_parameter'],
        training_protocol=proof['training_protocol'],model_proof=proof,native_preflight_qualification=spec['native_preflight_qualification'],
        readout_source=spec['readout_source'],geometry_source=spec['geometry_source'],hooks_source=spec['hooks_source'],
        copied_reader_source=PARENT_READER,terminal_and_admission_bound_before_scientific_access=True,
        CPU_affinity=[0,1],torch_threads=2,CUDA_initialized=False,external_CPU_RNG_restored=True,
        technical_training_checkpoint_eligible=True,automatic_resume_eligible=False,all_six_nodes_CPU_qualified=True,
        inference_executed=False,AP_computed=False,convergence_claimed=False,SOTA_claimed=False,
        GT_free_inference_runtime_verified=False,scientific_method_verdict=None,completed_24_epoch_schedule=True,
        next_action='Use a separately qualified official evaluator; this whole24 training consumer makes no scientific ranking.',
        evidence_limits=[
            'Fresh original official COCO-to-boat initialization and declared800/no-augmentation transfer, not full public recipe or pretraining provenance equivalence.',
            'All six live-training node snapshots are CPU-qualified only after the single whole24 owned terminal; no resume, node reconstruction or early metric selection.',
            'Independent CPU tensors/moments/order/GT audits do not replay CUDA forward/backward, gradient elements, PCA or M0-sign eligibility.',
            'All raw GT and actual ten-stage assignment coverage are checked from trace observations; native sampled costs/Hungarian optimum are not independently recomputed.',
            'Geometry/half/support counts are observed posthoc; joint arms may realize different F/e0/E/coordinates/matches and no same-support assertion is made.',
            'Odd741 accumulation continuity uses counters, gradient observations and bound execution source; the actual optimization trajectory is not reproduced.',
            'No inference, AP, convergence, model ranking or innovation verdict is computed.'])
    with target.open('x',encoding='utf-8') as stream:
        json.dump(acceptance,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=acceptance['status'],acceptance=reference(target)),sort_keys=True),flush=True)



def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--acceptance',required=True)
    args=parser.parse_args()
    qualify(args.spec,args.acceptance)


if __name__ == '__main__':
    main()
