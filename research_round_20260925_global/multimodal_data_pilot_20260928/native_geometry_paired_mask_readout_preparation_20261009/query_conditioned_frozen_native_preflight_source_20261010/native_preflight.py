"""Inert ROOT-owned four-mode frozen E24 teacher technical preflight.

The RAW original teacher terminal receipt passes before the first source,
SPEC, qualification or checkpoint hash/import. The current owned admission
then binds the complete SOURCE closure. Four fresh CPU native0 architectures
strict-load the SAME independently qualified full610 E24 teacher before CUDA.
Every native parameter is frozen and every native module remains in eval.
Only two fresh query pair-generator weights receive AdamW updates, using all
30 native loss atoms, four FIT images and two accumulation-two updates.

This entry reuses sealed initialization, image, native loss and CPU proof
helpers. It never restores teacher optimizer, data order or RNG. It produces
technical artifacts for an independent CPU reader; no efficacy, AP, novelty,
convergence, equal-compute or formal-resume claim follows from this fixture.
SOURCE authoring performed no scientific execution or payload access.
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


SCHEMA = 'ROOT-query-conditioned-frozen-native-preflight-spec-v1'
REPORT_SCHEMA = 'query-conditioned-frozen-native-preflight-report-v1'
CP_SCHEMA = 'query-conditioned-frozen-native-technical-checkpoint-v1'
PROOF_SCHEMA = 'query-conditioned-frozen-native-technical-state-proof-v1'
MODES = ('raw', 'raw_rms', 'query_rms', 'permuted_query_rms')
IMAGE_IDS = (1, 2, 3, 5)
ADAPTER_PREFIX = 'panoptic_head.query_pair_readout.'
ACTIVE = [ADAPTER_PREFIX + name for name in ('hidden.weight', 'final.weight')]
GT_FIELDS = ('gt_instances', 'ignored_instances', 'gt_sem_seg', 'gt_panoptic_seg')
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
READOUT_SHA = '5199354074939e335e0d0b5c19940ad56d637b750843cb38cc9c7213e81b4b2a'
GEOMETRY_SHA = '7ad19e356342afc15a2a41f1358634b23486c19021e63ccd94cfed03f6841d15'
HOOKS_SHA = 'd043e86d5699193533d4a8279ff7961e10adef916028b8250ae38ebe58a42dcc'
EVALUATION_SHA = 'b4ca717bd2018ea8404808c03510f830081a6660c893eda04fce72d7adb43a48'
FIXTURE_SHA = 'e6458035b8924eb8e1b4e5f8a9dffd1bc61030515fd96b3e72ff4b81bf4c3dd6'
TRAIN_SHA = 'a50b7c95691af7884dcf0bf23ad331d0c1a4055c11caa3da255d0efafb25cc09'
TRAIN_READER_SHA = '51443701bf31273a1df771a597b05b7540b75366c2e09d4919bf49587062d97d'
ROLES = ('CPU_reader', 'readout_source', 'geometry_source', 'hooks_source',
         'initializer_source', 'micro_preflight_source', 'native_helpers',
         'native_preflight_helpers', 'reused_CPU_reader', 'evaluation_source',
         'training_source', 'training_CPU_reader', 'operator_CPU_fixture_source')


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
    require(isinstance(row, dict) and set(('path', 'bytes', 'sha256')) <= set(row)
            and Path(row['path']).is_absolute(), 'Absolute complete input binding required')
    require(reference(row['path']) == {key: row[key] for key in ('path', 'bytes', 'sha256')},
            'Pinned input differs: ' + row['path'])
    return Path(row['path'])


def write(path, value):
    """Publish JSON atomically, only within the fresh attempt directory."""
    path = Path(path)
    require(not path.exists(), 'Refuse to replace an existing artifact')
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    require(not path.exists(), 'Artifact appeared during publication')
    temporary.rename(path)


def load_source(row, name):
    path = bound(row)
    value = path.read_bytes()
    require(len(value) == row['bytes'] and hashlib.sha256(value).hexdigest() == row['sha256'],
            'SOURCE changed between binding and compilation')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(value, str(path), 'exec'), module.__dict__)
    return module


def terminal_guard(receipt, phase):
    require(receipt['status'] == 'ACTUAL_OWNED_TERMINAL' and
            receipt['original_identity'] is not None and
            type(receipt['returncode']) is int and receipt['returncode'] == 0 and
            receipt['root_wait_reaped'] is True and receipt['held_handle_exit'] ==
            dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
            receipt['numeric_birth_absence']['original_birth_absent'] is True and
            receipt['abort'] is None and receipt['error'] is None and
            receipt['wait_errors'] is False and receipt['GPU_observation_errors'] == [],
            'STOP before hashes/imports: actual teacher held0/reap/birth absence/no errors required')
    if 'phase' in receipt:
        require(receipt['phase'] == phase, 'Actual teacher receipt phase differs')
    peak = receipt['whole_GPU_observed_peak_MiB']
    require(type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            receipt['whole_GPU_used_limit_MiB'] == 8192 and
            receipt['whole_leaf_VRAM_certified'] is False and
            receipt['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
            'Actual teacher whole-GPU observed scope/peak differs')


def raw_teacher_gate(spec):
    """RAW receipt read ONLY. No reference(), SOURCE, Q or CP access here."""
    require(Path(spec['teacher_receipt']['path']).is_absolute(), 'Absolute raw teacher receipt required')
    receipt = read(spec['teacher_receipt']['path'])
    terminal_guard(receipt, spec['teacher_phase'])
    return receipt


def owner_source_gate(spec_path, spec):
    require(Path(spec['GPU_admission_file']).is_absolute(), 'Absolute current owner admission required')
    admission = read(Path(spec['GPU_admission_file']).resolve(strict=True))
    entry, spec_ref = reference(__file__), reference(spec_path)
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            admission['automatic_retry'] is False and admission['phase'] == spec['phase'] and
            admission['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and
            admission['cuda_visible_devices'] == '0' and
            admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
            admission['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192 and
            Path(admission['entry']).resolve(strict=True) == Path(entry['path']) and
            admission['entry_args'] == ['--spec', str(spec_path), '--owner-execute'] and
            admission['driver_sha256'] == OWNER_SHA, 'Current owner entry/resource metadata differs')
    hashes = admission['bound_source_sha256']
    require(spec['entry_source'] == entry and entry in spec['sources'] and
            spec['CPU_reader'] in spec['sources'] and hashes.get(str(spec_path)) == spec_ref['sha256'],
            'Current entry/reader/SPEC are not owner-bound')
    for source in spec['sources']:
        require(hashes.get(source['path']) == source['sha256'], 'Current complete SOURCE closure not owner-bound')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Current owner scientific SPEC differs')
    return spec_ref


def qualified_teacher(spec, receipt):
    """Original terminal0 already passed before the first teacher Q/CP hash."""
    bound(spec['teacher_receipt'])
    q = read(bound(spec['teacher_qualification']))
    require(q['schema'] == 'ROOT-native-spatial-position-preserving-common24-CPU-qualification-v1' and
            q['status'] == 'ROOT_NATIVE_SPATIAL_POSITION_PRESERVING_COMMON24_CPU_QUALIFIED' and
            q['kind'] == 'mask2former' and q['mode'] == 'native0' and q['phase'] == spec['teacher_phase'] and
            q['source'] == spec['training_source'] and q['reader_source'] == spec['training_CPU_reader'] and
            q['spec'] == spec['teacher_training_spec'] and q['parent_receipt'] == spec['teacher_receipt'] and
            q['all_six_nodes_CPU_qualified'] is True and q['completed_24_epoch_schedule'] is True and
            q['automatic_resume_eligible'] is False and q['inference_executed'] is False and
            q['AP_computed'] is False, 'Actual independently qualified native0 teacher differs')
    for key in ('reader_source', 'source', 'report', 'start', 'owner_driver'):
        bound(q[key])
    train = read(bound(spec['teacher_training_spec']))
    admission_path = bound(q['GPU_admission'])
    admission = read(admission_path)
    require(receipt['admission_sha256'] == q['GPU_admission']['sha256'] and
            Path(train['GPU_admission_file']).resolve(strict=True) == admission_path.resolve(strict=True) and
            admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            admission['automatic_retry'] is False and admission['phase'] == q['phase'] and
            admission['allowed_cpus'] == train['CPU_affinity'] == [0, 1] and
            admission['cuda_visible_devices'] == '0' and
            admission['qualified_leaf_deadline_seconds'] == train['deadline_seconds'] and
            admission['whole_GPU_used_limit_MiB'] == train['whole_GPU_owner_limit_MiB'] == 8192 and
            Path(admission['entry']).resolve(strict=True) == Path(spec['training_source']['path']).resolve(strict=True) and
            admission['entry_args'] == ['--spec', q['spec']['path'], '--owner-execute'] and
            admission['driver_sha256'] == q['owner_driver']['sha256'] == OWNER_SHA,
            'Original teacher receipt/admission/SPEC/source join differs')
    hashes = admission['bound_source_sha256']
    require(hashes.get(q['spec']['path']) == q['spec']['sha256'] and
            hashes.get(q['source']['path']) == q['source']['sha256'], 'Teacher admission hashes differ')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == q['spec'], 'Teacher scientific SPEC differs')
    for source in train['sources']:
        bound(source)
        require(source in spec['sources'] and hashes.get(source['path']) == source['sha256'],
                'Original teacher SOURCE closure not retained/bound')
    require(q['checkpoint'] == spec['teacher_checkpoint'] == q['node_checkpointmap']['24'] and
            q['checkpointmap'] == {'native0': q['checkpoint']} and
            set(q['node_checkpointmap']) == {'2', '4', '8', '12', '18', '24'} and
            q['completed_epoch'] == 24 and q['global_microsteps'] == 17784 and
            q['global_updates'] == 8892 and q['consumed_raw_GT'] == 39864 and
            train['schema'] == 'ROOT-native-spatial-position-preserving-common24-spec-v1' and
            train['kind'] == 'mask2former' and train['mode'] == 'native0' and train['phase'] == q['phase'] and
            train['initial_epoch'] == 0 and train['epochs_this_run'] == train['total_epochs'] == 24 and
            train['checkpoint_nodes'] == [2, 4, 8, 12, 18, 24] and
            train['resume_checkpoint'] is None and train['resume_qualification'] is None,
            'Only actual full24 native0 teacher endpoint is accepted')
    for key in ('initializer_source', 'micro_preflight_source', 'native_helpers', 'geometry_source', 'image_root'):
        require(train[key] == spec[key], 'Actual teacher native construction/input differs: ' + key)
    require(train['FIT'] == spec['FIT_annotation'], 'Same original FIT annotation binding required')
    protocol = q['training_protocol']
    require(protocol['mode'] == 'native0' and protocol['training_source'] == spec['training_source'] and
            protocol['native_loss_only'] is True and protocol['native_loss_atoms'] == 30 and
            protocol['Sobel'] is False and protocol['fresh_single_owned_24'] is True and
            protocol['resume_allowed'] is False and protocol['total_microsteps'] == 17784 and
            protocol['total_updates'] == 8892, 'Native0 teacher training protocol differs')
    proof = q['model_proof']
    require(proof['checkpoint'] == q['checkpoint'] == proof['nodes']['24']['checkpoint'] and
            proof['proof_refs']['state_proof'] == proof['nodes']['24']['state_proof'],
            'Teacher E24 proof/CP join differs')
    return dict(training_spec=train, qualification=q)


def fixture_gate(spec):
    fixture = read(bound(spec['operator_CPU_fixture']))
    require(spec['operator_CPU_fixture']['bytes'] == 3719 and
            spec['operator_CPU_fixture']['sha256'] == FIXTURE_SHA and
            fixture['status'] == 'QUERY_CONDITIONED_GLOBAL_PAIR_SYNTHETIC_CPU_PASS' and
            fixture['synthetic_execution_performed'] is True and
            fixture['native_integration_qualified'] is False and fixture['efficacy_claimed'] is False and
            fixture['sources']['fixture'] == spec['operator_CPU_fixture_source'] and
            fixture['sources']['operator'] == spec['readout_source'] and
            fixture['sources']['geometry'] == spec['geometry_source'] and
            fixture['execution'] == dict(device='cpu', cpu_affinity=[0, 1], threads=2, cuda_visible_devices=''),
            'Actual pinned CPU synthetic fixture PASS required')
    for key in ('all_four_cold_native_bit_identity', 'seeded_state_and_rng_restoration',
                'permutation_cycle_derangement_multiset_norm', 'normalization_scale_sign_zero_gradients',
                'auxiliary_e0_detached_native_e0_live', 'all_E0_connected_zero',
                'raw_original_spatial_global_bit_equal', 'real_original_rectangle_E'):
        require(fixture[key] is True, 'CPU synthetic assertion absent: ' + key)
    for source in fixture['sources'].values():
        require(source in spec['sources'], 'CPU synthetic SOURCE closure absent')
        bound(source)


def frozen_policy(torch, model):
    named = dict(model.named_parameters())
    native = [name for name in named if not name.startswith(ADAPTER_PREFIX)]
    active = [name for name, value in named.items() if value.requires_grad]
    BN_modules, BN_parameters, BN_buffers = [], [], []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            BN_modules.append(name)
            BN_parameters.extend(name + '.' + local for local, value in module.named_parameters(recurse=False))
            BN_buffers.extend(name + '.' + local for local, value in module.named_buffers(recurse=False))
    require(active == ACTIVE and len(native) == 451 and len(named) == 453 and
            len(model.state_dict()) == 612 and len(BN_modules) == 53 and
            len(BN_parameters) == 106 and len(BN_buffers) == 159 and
            all(not named[name].requires_grad for name in native) and
            all(not module.training for module in model.modules()) and
            sum(named[name].numel() for name in ACTIVE) == 393216,
            'Frozen native451/all-eval/BN106+159 plus exactly two fresh weights required')
    return dict(kind='mask2former', state_entries=612, parameter_tensors=453,
                trainable_tensors=2, frozen_tensors=451, active_names=ACTIVE,
                native_names=native, frozen_names=native, BN_module_names=BN_modules,
                BN_parameter_names=BN_parameters, BN_buffer_names=BN_buffers,
                all_native_modules_eval=True)


def equal_rng(torch, left, right):
    return len(left) == len(right) == 2 and all(torch.equal(a, b) for a, b in zip(left, right))


def hook_inventory(model):
    return {name: [list(module._forward_pre_hooks), list(module._forward_hooks)]
            for name, module in model.named_modules()}


def image_data(native, old, model, dataset, coco, image_id):
    # O returns copied inference samples. Add the fourth GT-field removal before
    # N's all-four assertion, without changing the original supervised samples.
    def load_image(*args):
        result = old.load_image(*args)
        for sample in result[1]:
            for key in GT_FIELDS:
                if key in sample:
                    delattr(sample, key)
        return result
    return native.image_data(types.SimpleNamespace(load_image=load_image), model, dataset, coco, image_id)


def training_fixture(torch, pref, old, native, hooks, teacher_reader, reused_reader,
                     spec, spec_ref, train, prepared, teacher, dataset, coco, output, started,
                     mode, shared_native):
    model, loading_proof, cold = prepared
    directory = output / (mode + '_training')
    directory.mkdir()
    before_state = dict(teacher)
    before_state.update({ADAPTER_PREFIX + name: value for name, value in cold.items()})
    before = pref.state_proof(before_state)
    model.to('cuda:0')
    policy = frozen_policy(torch, model)
    entry = train['initializers']['mask2former']
    recipe = entry['optimizer']
    require(recipe['type'] == 'AdamW' and recipe['lr'] == 0.0001 and recipe['weight_decay'] == 0.05 and
            tuple(recipe.get('betas', (0.9, 0.999))) == (0.9, 0.999) and recipe.get('eps', 1e-8) == 1e-8 and
            entry['clip_grad'] == dict(max_norm=0.01, norm_type=2), 'Exact fresh native AdamW/clip recipe required')
    optimizer, optimizer_audit = pref.make_optimizer(torch, model, entry, policy)
    require(len(optimizer.state) == 0 and len(optimizer.param_groups) == 2 and
            all(group['lr'] == 0.0001 and group['weight_decay'] == 0.05 and
                tuple(group['betas']) == (0.9, 0.999) and group['eps'] == 1e-8 for group in optimizer.param_groups),
            'Fresh optimizer must contain ONLY two unmodified query weight groups')
    names = dict(model.named_parameters())
    adapter = model.panoptic_head.query_pair_readout
    head = model.panoptic_head
    baseline_hooks = hook_inventory(model)
    activity = {name: 0 for name in ACTIVE}
    consumed, trace_refs, updates, warm_both = [], [], 0, False
    optimizer.zero_grad(set_to_none=True)
    try:
        for micro, image_id in enumerate(IMAGE_IDS):
            pref.check_deadline(started, spec['deadline_seconds'])
            data, samples, identity = image_data(native, old, model, dataset, coco, image_id)
            target = pref.check_targets(torch, data, identity)
            external = pref.rng_snapshot(torch)
            native_outputs, native_fields, adapted_fields, bridge_inputs = [], [], [], []

            def observe_native(module, inputs, output):
                require(len(output) == 2 and len(output[0]) == len(output[1]) == 10,
                        'Original native head must produce ten class/mask readouts')
                native_outputs.append(output)
                native_fields.append(dict(classes=[pref.digest_tensor(x) for x in output[0]],
                    native_masks=[pref.digest_tensor(x) for x in output[1]],
                    native_classes_require_grad=[x.requires_grad for x in output[0]],
                    native_masks_require_grad=[x.requires_grad for x in output[1]]))

            def observe_adapted(module, inputs, output):
                original = native_outputs[0]
                require(output[0] is original[0] and
                        all(output[1][i] is original[1][i] for i in range(9)) and
                        output[1][-1].requires_grad, 'Only final mask may acquire new weight autograd')
                adapted_fields.append(dict(adapted_final_mask=pref.digest_tensor(output[1][-1]),
                    classes10_objects_preserved=True, first9_mask_objects_preserved=True,
                    adapted_final_requires_grad=True))

            def observe_inputs(module, inputs, output):
                require(len(inputs) == 3 and not inputs[0].requires_grad and not inputs[1].requires_grad and
                        output['logit_residual'].requires_grad and output['mask_logits'].requires_grad,
                        'Frozen F/e0 and live new-weight residual required')
                bridge_inputs.append(dict(feature=pref.digest_tensor(inputs[0]),
                    coefficient=pref.digest_tensor(inputs[1]), valid_fraction=pref.digest_tensor(inputs[2]),
                    feature_requires_grad=False, coefficient_requires_grad=False, residual_requires_grad=True))

            original_handle = head.register_forward_hook(observe_native)
            input_handle = adapter.register_forward_hook(observe_inputs)
            adapted_handle = None
            seed = 1000000 + micro
            try:
                with torch.random.fork_rng(devices=[0]):
                    torch.random.default_generator.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    private_before = pref.rng_snapshot(torch)
                    require(torch.is_grad_enabled() and not data['inputs'].requires_grad,
                            'Ambient autograd enabled with ordinary nongrad input required')
                    with hooks.query_pair_readout_context(torch, head,
                             [sample.metainfo for sample in samples]) as record:
                        adapted_handle = head.register_forward_hook(observe_adapted)
                        features = model.extract_feat(data['inputs'])
                        classes, masks = head(features, samples)
                    gt = old.posterior_gt(head, data, identity)
                    posterior_mask = gt.masks.detach().to(device='cpu', dtype=torch.uint8)
                    raw_mask = data['data_samples'][0].gt_instances.masks.to_tensor(dtype=torch.uint8, device='cpu')
                    require(gt.masks.dtype == torch.int64 and torch.equal(posterior_mask, raw_mask),
                            'Posterior native int64 masks differ from original padded GT raster')
                    posterior = dict(mask=pref.digest_tensor(gt.masks), labels=pref.digest_tensor(gt.labels),
                        mask_uint8=pref.digest_tensor(posterior_mask), GT_count=len(gt),
                        complete_all_raw_annotations=True, preprocess_after_head_forward=True)
                    with old.observe_targets(head) as matches:
                        losses = head.loss_by_feat(classes, masks, [gt], [data['data_samples'][0].metainfo])
                    require(set(losses) == old.ATOMIC_KEYS and len(losses) == 30 and len(matches) == 10,
                            'Exact original native30/ten actual target records required')
                    summed, scalars = pref.loss_terms(torch, losses)
                    total, parsed = model.parse_losses(losses)
                    require(total.requires_grad and bool(torch.isfinite(total)) and torch.equal(total, summed),
                            'Native parse_losses total differs/disconnected/nonfinite')
                    require(len(native_fields) == len(adapted_fields) == len(bridge_inputs) == 1 and
                            not any(native_fields[0]['native_classes_require_grad']) and
                            not any(native_fields[0]['native_masks_require_grad']), 'Native graph must stay frozen')
                    for match in matches:
                        require(len(set(match['positive_query_indices'])) == len(identity['ann_ids']) and
                                sorted(match['assigned_GT_indices']) == list(range(len(identity['ann_ids']))) and
                                all(0 <= q < 100 for q in match['positive_query_indices']),
                                'Every original raw GT must survive each native assignment')
                    native_key = dict(classes=native_fields[0]['classes'], masks=native_fields[0]['native_masks'],
                                      feature=bridge_inputs[0]['feature'], coefficient=bridge_inputs[0]['coefficient'])
                    if mode == MODES[0]:
                        shared_native[micro] = native_key
                    require(native_key == shared_native[micro], 'Frozen native fields differ across fresh modes')
                    require(record['grad_mode_enabled_at_start'] is True and
                            record['native_graph_properties']['feature_requires_grad'] is False and
                            record['native_graph_properties']['coefficient_requires_grad'] is False and
                            record['native_graph_properties']['native_final_mask_requires_grad'] is False,
                            'Bridge observed a live native graph')
                    if micro < 2:
                        require(record['cold_zero_final_weight'] is True and
                                record['cold_native_value_equal'] is True, 'Cold native final value identity required')
                    (total / 2).backward()
                    private_after = pref.rng_snapshot(torch)
            finally:
                original_handle.remove()
                input_handle.remove()
                if adapted_handle is not None:
                    adapted_handle.remove()
            external_after = pref.rng_snapshot(torch)
            require(equal_rng(torch, external, external_after), 'Private seeded loss RNG not externally restored')
            require(hook_inventory(model) == baseline_hooks and hooks._contexts_inactive(head) and
                    record['hooks_removed'] is True and record['cache_cleared_after_forward'] is True,
                    'Scoped hook/cache cleanup differs')
            norms = []
            for name in ACTIVE:
                gradient = names[name].grad
                require(gradient is not None and bool(torch.isfinite(gradient).all()),
                        'Every new weight needs a finite actual accumulated gradient')
                norms.append(float(gradient.detach().float().norm().cpu()))
                activity[name] += int(norms[-1] > 0)
            cold_hidden_zero = bool(torch.count_nonzero(names[ACTIVE[0]].grad).item() == 0)
            if micro < 2:
                require(cold_hidden_zero and norms[1] > 0, 'Expected cold hidden-zero/final-positive gradients absent')
            else:
                warm_both = warm_both or all(value > 0 for value in norms)
            require(all(names[name].grad is None for name in policy['native_names']) and
                    all(not module.training for module in model.modules()), 'Frozen native grad/eval contract broken')
            stepped = (micro + 1) % 2 == 0
            preclip = None
            if stepped:
                global_norm = torch.nn.utils.clip_grad_norm_([names[name] for name in ACTIVE], max_norm=0.01, norm_type=2)
                require(bool(torch.isfinite(global_norm)), 'Nonfinite accumulated global clip norm')
                preclip = float(global_norm.detach().cpu())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            fields = dict(native_fields[0], **adapted_fields[0])
            consumed.extend(identity['ann_ids'])
            trace = dict(mode=mode, microstep=micro, image_id=image_id, identity=identity, GT=target,
                posterior_GT=posterior, input=pref.digest_tensor(data['inputs']), loss_fields=scalars,
                native_total=float(total.detach().cpu()), native_loss_only=True, Sobel_added=False,
                auxiliary_loss_added=False, matches=list(matches), head_fields=fields,
                bridge_inputs=bridge_inputs[0], native_hook=dict(record),
                readout_observation=record['readout_observation'], accumulation_divisor=2,
                optimizer_step=stepped, optimizer_updates=updates, preclip_global_norm=preclip,
                gradient_clipping_applied=stepped, gradient_norms_aligned_active_names=norms,
                gradients_are_accumulated_preclip=True,
                cold_hidden_gradient_exact_zero=(cold_hidden_zero if micro < 2 else None),
                native_parameter_grads_None=True, all_native_modules_eval=True, hook_cache_clear=True,
                private_seed=seed, private_RNG_before=[pref.digest_tensor(x) for x in private_before],
                private_RNG_after=[pref.digest_tensor(x) for x in private_after],
                external_RNG_before=[pref.digest_tensor(x) for x in external],
                external_RNG_after=[pref.digest_tensor(x) for x in external_after], external_RNG_restored=True,
                GT_removed_from_forward=True, native_all10_fields_equal_across_modes=True,
                common_matching_or_points_after_divergence_required=False)
            path = directory / ('MICRO_%02d.json' % micro)
            write(path, trace)
            trace_refs.append(reference(path))
            print(json.dumps(dict(status='TECHNICAL_PROGRESS', mode=mode, microsteps=micro + 1, updates=updates)), flush=True)
            del data, samples, features, classes, masks, gt, posterior_mask, raw_mask, losses, total, summed, parsed, matches
            del native_outputs, native_fields, adapted_fields, bridge_inputs, record, trace
        require(updates == 2 and warm_both and all(activity[name] > 0 for name in ACTIVE) and
                all(parameter.grad is None for parameter in model.parameters()),
                'Four micro/two updates/warm both-weight gradients/no pending grads required')
        final = pref.cpu_tree(torch, model.state_dict())
        require(all(torch.equal(value.detach().cpu(), final[name]) for name, value in model.state_dict().items()),
                'Actual full-state GPU-to-CPU transfer differs')
        require(final.keys() == before_state.keys() and all(bool(torch.isfinite(x).all()) for x in final.values()) and
                all(torch.equal(final[name], value) for name, value in teacher.items()),
                'Full612 finite schema and exact all610 teacher preservation required')
        changed = {name: not torch.equal(before_state[name], value) for name, value in final.items()}
        require(all(changed[name] for name in ACTIVE) and not any(changed[name] for name in teacher),
                'Only the two freshly learned weights may change')
        final_optimizer = pref.cpu_tree(torch, optimizer.state_dict())
        require(len(final_optimizer['state']) == 2 and all(float(row['step']) == 2 for row in final_optimizer['state'].values()),
                'Exactly two fresh AdamW moment states at step2 required')
        group_names = [[name] for name in ACTIVE]
        meta = dict(schema=CP_SCHEMA, mode=mode, kind='mask2former', phase=spec['phase'], source=spec['entry_source'],
            spec=spec_ref, readout_source=spec['readout_source'], geometry_source=spec['geometry_source'],
            hooks_source=spec['hooks_source'], teacher_checkpoint=spec['teacher_checkpoint'],
            teacher_qualification=spec['teacher_qualification'], teacher_training_spec=spec['teacher_training_spec'],
            teacher_receipt=spec['teacher_receipt'], teacher_mode='native0', teacher_epoch=24,
            image_ids=list(IMAGE_IDS), consumed_ann_ids=consumed, microsteps=4, optimizer_updates=2,
            batch_size=1, accumulation=2, optimizer=optimizer_audit, optimizer_param_names=group_names, policy=policy,
            frozen_native=True, native610_exact=True, native_loss_only=True, native_loss_atoms=30,
            no_Sobel_or_auxiliary_loss=True, not_formal_training=True, eligible_for_formal_resume=False,
            no_AP=True, no_pending_accumulation_gradients=True, teacher_optimizer_loaded=False,
            teacher_order_RNG_loaded=False, teacher_external_RNG_loaded=False)
        checkpoint = dict(state_dict=final, adapter_state_dict={name: final[ADAPTER_PREFIX + name] for name in cold},
            optimizer_state_dict=final_optimizer, external_rng_state=pref.cpu_tree(torch, pref.rng_snapshot(torch)), meta=meta)
        checkpoint_path = directory / (mode + '_technical_final.pth')
        with checkpoint_path.open('xb') as stream:
            torch.save(checkpoint, stream)
        require(pref.tree_equal(torch, checkpoint, torch.load(str(checkpoint_path), map_location='cpu')),
                'Fresh technical checkpoint CPU readback differs')
        proof = dict(schema=PROOF_SCHEMA, mode=mode, native_teacher610=pref.state_proof(teacher), before=before,
            after=pref.state_proof(final), changed=changed, policy=policy, optimizer=optimizer_audit,
            positive_gradient_microcounts=activity, adapter_names=ACTIVE,
            actual_final_optimizer=teacher_reader.optimizer_snapshot(torch, reused_reader, final_optimizer, group_names),
            fresh_optimizer_state_entries=0, teacher_optimizer_loaded=False, teacher_order_RNG_loaded=False,
            teacher_external_RNG_loaded=False, native610_exact=True, native_grad_None=True,
            all_native_modules_eval=True, hook_cache_clear=True, frozen_BN_all_exact=True,
            all_adapter_weights_positive_and_changed=True, warm_both_weights_positive=True,
            CPU_checkpoint_readback_exact=True, GPU_to_CPU_readback_state_exact=True, teacher_loading=loading_proof,
            cold_adapter=pref.state_proof(cold), initial_hidden_gradient_zero_is_expected=True,
            cold_constructor_repeated_exact=True, cold_constructor_RNG_restored=True,
            fresh_identical_adapter_initialization_across_modes=True)
        proof_path = directory / 'TRAIN_STATE_PROOF.json'
        write(proof_path, proof)
        return dict(mode=mode, checkpoint=reference(checkpoint_path), state_proof=reference(proof_path), traces=trace_refs,
            image_ids=list(IMAGE_IDS), consumed_ann_ids=consumed, microsteps=4, optimizer_updates=2,
            native610_exact=True, frozen_native=True, native_loss_only=True, AP_computed=False,
            eligible_for_formal_resume=False, no_pending_accumulation_gradients=True,
            GT_removed_from_forward=True, native_training_GT_complete=True)
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.to('cpu')


def run(spec_file):
    started = time.monotonic()
    require(Path(spec_file).is_absolute(), 'Absolute --spec required')
    spec_path = Path(spec_file).resolve(strict=True)
    spec = read(spec_path)  # Metadata allowed; deliberately NOT hashed yet.
    receipt = raw_teacher_gate(spec)  # FIRST payload gate before SOURCE/Q/CP/SPEC hashes.
    require(spec['schema'] == SCHEMA and spec['phase'].startswith('query_conditioned_frozen_native_preflight_') and
            spec['modes'] == list(MODES) and spec['image_ids'] == list(IMAGE_IDS) and
            spec['loss_seed_base'] == 1000000 and spec['expected_epoch'] == 24 and
            spec['hidden_seed'] == 2026100911 and spec['permutation_seed'] == 2026101013 and
            spec['CPU_affinity'] == [0, 1] and spec['allocator_limit_MiB'] == 4096 and
            spec['whole_GPU_owner_limit_MiB'] == 8192 and type(spec['deadline_seconds']) is int and
            spec['deadline_seconds'] > 0 and os.environ.get('CUDA_VISIBLE_DEVICES') == '0',
            'Exact four-mode frozen E24/4micro2update resource protocol required')
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and Path(spec['image_root']).is_absolute(),
            'Fresh absolute output/image root required')
    spec_ref = owner_source_gate(spec_path, spec)
    for source in spec['sources']:
        bound(source)
    require(all(spec[key] in spec['sources'] for key in ROLES), 'Required complete SOURCE helper roles absent')
    for role, size, sha in (('readout_source', 10481, READOUT_SHA), ('geometry_source', 17500, GEOMETRY_SHA),
                           ('hooks_source', 27567, HOOKS_SHA), ('evaluation_source', 40816, EVALUATION_SHA),
                           ('training_source', 43363, TRAIN_SHA), ('training_CPU_reader', 60137, TRAIN_READER_SHA)):
        require(spec[role]['bytes'] == size and spec[role]['sha256'] == sha, 'Exact sealed SOURCE role differs: ' + role)
    teacher_row = qualified_teacher(spec, receipt)
    fixture_gate(spec)
    output.mkdir(parents=True)
    write(output / 'START.json', dict(schema='query-conditioned-frozen-native-preflight-start-v1',
        phase=spec['phase'], source=spec['entry_source'], spec=spec_ref, modes=list(MODES), image_ids=list(IMAGE_IDS),
        teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
        teacher_receipt=spec['teacher_receipt'], operator_CPU_fixture=spec['operator_CPU_fixture'],
        teacher_terminal_gate_before_scientific_access=True, all_four_CPU_models_before_CUDA=True,
        teacher_optimizer_loaded=False, teacher_order_RNG_loaded=False, teacher_external_RNG_loaded=False,
        fresh_optimizer_state_entries=0, no_formal_resume=True, no_DEV=True, no_AP=True, no_retry=True))
    import psutil
    import torch
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(),
            'Actual CPU01 and uninitialized CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 2, 'Actual two-thread runtime required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    pref = load_source(spec['micro_preflight_source'], 'query_frozen_native_pref')
    initializer = load_source(spec['initializer_source'], 'query_frozen_native_initializer')
    old = load_source(spec['native_helpers'], 'query_frozen_native_images'); old.torch = torch
    native = load_source(spec['native_preflight_helpers'], 'query_frozen_native_preflight_helpers')
    evaluation = load_source(spec['evaluation_source'], 'query_frozen_native_E24_loader')
    teacher_reader = load_source(spec['training_CPU_reader'], 'query_frozen_teacher_CPU_helpers')
    reused_reader = load_source(spec['reused_CPU_reader'], 'query_frozen_original_CPU_helpers')
    readout = load_source(spec['readout_source'], 'query_frozen_readout')
    geometry = load_source(spec['geometry_source'], 'query_frozen_geometry')
    hooks = load_source(spec['hooks_source'], 'query_frozen_whole_head_bridge')
    require(tuple(readout.MODES) == tuple(hooks.MODES) == MODES, 'Actual four query modes required')
    classes = tuple(CocoDataset.METAINFO['classes'])
    require(len(classes) == 80 and classes[8] == 'boat', 'Original official COCO80 ordering required')
    helper_spec = dict(initializer_source=spec['initializer_source'], training_source=spec['training_source'],
        checkpointmap={'native0': spec['teacher_checkpoint']}, qualificationmap={'native0': spec['teacher_qualification']},
        expected_epoch=24)
    prepared, teacher, shared_cold = {}, None, None
    for mode in MODES:
        pref.check_deadline(started, spec['deadline_seconds'])
        model, state, meta, loading = evaluation.prepare_model(torch, pref, initializer, None, None, None,
            MODELS, revert_sync_batchnorm, Config, classes, helper_spec, teacher_row, 'native0')
        require(not torch.cuda.is_initialized() and all(not module.training for module in model.modules()),
                'CPU teacher loading must precede CUDA and remain entirely eval')
        if teacher is None:
            teacher = state
        require(pref.tree_equal(torch, teacher, state), 'Four CPU teachers must load the SAME exact610 state')
        rng_before = torch.get_rng_state().clone()
        adapter = hooks.register_query_pair_readout(torch, readout, geometry, model, mode,
            spec['readout_source'], spec['geometry_source'], hidden_seed=2026100911, permutation_seed=2026101013)
        cold = pref.cpu_tree(torch, adapter.state_dict())
        repeated = readout.build_adapter(torch, geometry, mode=mode, hidden_seed=2026100911, permutation_seed=2026101013)
        require(torch.equal(rng_before, torch.get_rng_state()) and
                pref.tree_equal(torch, cold, pref.cpu_tree(torch, repeated.state_dict())) and
                bool((cold['final.weight'] == 0).all()), 'Repeated fresh constructor/RNG/zero final differs')
        if shared_cold is None:
            shared_cold = cold
        require(pref.tree_equal(torch, shared_cold, cold), 'Four fresh adapters have unequal initial weights')
        model.requires_grad_(False).eval()
        adapter.requires_grad_(True)
        frozen_policy(torch, model)
        prepared[mode] = (model, loading, cold)
        del state, meta, repeated, adapter
    require(not torch.cuda.is_initialized() and len(prepared) == 4, 'All four CPU models before any CUDA required')
    dataset = CocoDataset(ann_file=str(bound(spec['FIT_annotation'])), data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
    coco = COCO(str(bound(spec['FIT_annotation'])))
    require(len(dataset) == len(coco.imgs) == 741 and len(coco.anns) == 1661 and
            sorted(map(int, coco.imgs))[:4] == list(IMAGE_IDS), 'Complete FIT741/1661 and fixed first four image IDs required')
    torch.cuda.init()
    capacity = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(4096 * 1024 ** 2 / capacity, 0)
    results, shared_native, cleanup = {}, {}, {}
    for mode in MODES:
        current = prepared.pop(mode)
        results[mode] = training_fixture(torch, pref, old, native, hooks, teacher_reader, reused_reader,
            spec, spec_ref, teacher_row['training_spec'], current, teacher, dataset, coco, output, started,
            mode, shared_native)
        del current
        cleanup[mode] = pref.clear_gpu(torch)
    report = dict(schema=REPORT_SCHEMA, phase=spec['phase'], source=spec['entry_source'], spec=spec_ref,
        teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
        teacher_training_spec=spec['teacher_training_spec'], teacher_receipt=spec['teacher_receipt'],
        operator_CPU_fixture=spec['operator_CPU_fixture'], modes=list(MODES), mode_rows=results,
        native_loss_only=True, native_loss_atoms=30, frozen_native=True, native610_exact=True,
        all_four_CPU_models_before_CUDA=True, only_one_model_on_CUDA_at_a_time=True,
        teacher_optimizer_loaded=False, teacher_order_RNG_loaded=False, teacher_external_RNG_loaded=False,
        image_ids=list(IMAGE_IDS), microsteps_per_mode=4, optimizer_updates_per_mode=2,
        status='TECHNICAL_QUERY_CONDITIONED_FROZEN_NATIVE_PREFLIGHT_COMPLETE_NO_METHOD_RESULT', error=None,
        AP_computed=False, no_AP=True, no_DEV=True, efficacy_claimed=False, formal_resume_eligible=False,
        scientific_method_decision_computed=False, cleanup=cleanup,
        elapsed_seconds=time.monotonic() - started,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
        torch_allocated_peak_MiB=torch.cuda.max_memory_allocated(0) / 1024 ** 2,
        allocator_limit_MiB=4096, GPU_scope='whole GPU must be verified from actual owner receipt')
    report_path = output / 'REPORT.json'
    write(report_path, report)
    write(output / 'QUALIFIED.json', dict(schema='query-conditioned-frozen-native-preflight-technical-aggregate-v1',
        status='ROOT_QUERY_CONDITIONED_FROZEN_NATIVE_PREFLIGHT_TECHNICAL_PENDING_CPU_READER',
        source=spec['entry_source'], spec=spec_ref, report=reference(report_path), mode_rows=results,
        teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
        teacher_receipt=spec['teacher_receipt'], independently_CPU_qualified=False,
        efficacy_claimed=False, AP_computed=False, formal_resume_eligible=False))
    print(json.dumps(dict(status='TECHNICAL_COMPLETE_CPU_QUALIFICATION_PENDING', modes=list(MODES))), flush=True)


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
