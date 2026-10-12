"""Independent CPU state reader for a fresh native phase from native0 E24.

ROOT alone executes this SOURCE after current and all ancestral raw terminals.
The four-microstep branch qualifies restoration/wiring, never efficacy or a
formal initializer. Formal additional24 reloads original E24 with empty AdamW,
and audits six complete native CPU nodes after the single terminal attempt.
No CUDA forward, sampled matching, loss/backward or optimizer trajectory is
replayed. Original independent mapping, typed policy and optimizer helpers
are imported only after the complete admitted SOURCE closure is verified.
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
import types

SCHEMA = 'ROOT-native-finetune24-fresh-optimizer-spec-v1'
REPORT_SCHEMA = 'native-finetune24-fresh-optimizer-report-v1'
CP_SCHEMA = 'native-finetune24-fresh-optimizer-checkpoint-v1'
TECH_SCHEMA = 'ROOT-native-finetune24-fresh-optimizer-restoration-CPU-qualification-v1'
TECH_STATUS = 'ROOT_NATIVE_FINETUNE24_FRESH_OPTIMIZER_RESTORATION_CPU_QUALIFIED'
QUAL_SCHEMA = 'ROOT-native-finetune24-fresh-optimizer-CPU-qualification-v1'
QUAL_STATUS = 'ROOT_NATIVE_FINETUNE24_FRESH_OPTIMIZER_CPU_QUALIFIED'
LEGACY_MODES = ('native0', 'global', 'spatial_halves', 'spatial_reversed')
NODES = (2, 4, 8, 12, 18, 24)
SCHEDULE = dict(gamma=.1, milestones=[7904, 8563], warmup_start_factor=1, warmup_updates=0)
ORDER_SEED, LOSS_SEED = 2026101021, 1000000
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
M_ROOT = Path(__file__).resolve().parent.parent.parent
MICRO_CPU = dict(path=str(M_ROOT / 'native_coco_comparator_micro_preflight_preparation_20261008' / 'qualify_cpu.py'),
    bytes=46683, sha256='ae810fb940fed62588b1a4e0bd7ee199647ff656dbbef5d0bb1247864662d960')


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


def bound(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'bytes', 'sha256'} and
        Path(ref['path']).is_absolute() and type(ref['bytes']) is int and ref['bytes'] > 0 and
        isinstance(ref['sha256'], str) and len(ref['sha256']) == 64 and
        set(ref['sha256']) <= set('0123456789abcdef'), 'Exact absolute reference required')
    require(reference(ref['path']) == ref, 'Pinned reference differs: ' + ref['path'])
    return Path(ref['path'])


def read_ref(ref):
    return read(bound(ref))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')).hexdigest()


def expect(row, expected, label):
    require(isinstance(row, dict), label + ' must be an object')
    for key, value in expected.items():
        require(key in row and row[key] == value and
            (type(value) not in (bool, int) or type(row[key]) is type(value)), label + ' differs: ' + key)


def load_source(ref, name):
    path = bound(ref)
    require(path.suffix == '.py', 'Bound Python SOURCE required')
    source = path.read_bytes()
    require(len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'],
        'SOURCE changed immediately before import')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def terminal(raw, phase):
    expect(raw, dict(status='ACTUAL_OWNED_TERMINAL', returncode=0, root_wait_reaped=True,
        held_handle_exit=dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0), abort=None, error=None,
        wait_errors=False, GPU_observation_errors=[], whole_GPU_used_limit_MiB=8192,
        whole_leaf_VRAM_certified=False, GPU_scope='whole GPU observed, not per-process WDDM allocation'),
        'STOP before hashes/imports: owned raw terminal')
    require(raw.get('original_identity') is not None and
        raw.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
        raw.get('phase', phase) == phase, 'STOP: original owned identity/reap/birth absence differs')
    peak = raw.get('whole_GPU_observed_peak_MiB')
    require(type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192,
        'STOP: actual whole GPU peak outside 8192MiB')


def phase_gate(spec):
    expect(spec, dict(schema=SCHEMA, kind='mask2former', mode='native0', expected_epoch=24,
        teacher_epoch=24, phase_initial_epoch=0, phase_epochs=24, phase_microsteps=17784,
        phase_updates=8892, phase_raw_GT_visits=39864, order_seed=ORDER_SEED,
        phase_loss_seed_base=LOSS_SEED, CPU_affinity=[0, 1], allocator_limit_MiB=4096,
        whole_GPU_owner_limit_MiB=8192, schedule=SCHEDULE), 'Fresh native phase SPEC')
    require(spec['execution_kind'] in ('technical_restore_four_microsteps', 'formal_additional24') and
        type(spec['deadline_seconds']) is int and 0 < spec['deadline_seconds'] <= 10800,
        'Fresh execution kind/deadline differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU reader requires CUDA_VISIBLE_DEVICES empty')


def source_gate(spec_path, spec):
    # Current raw, original four raws, and formal technical raw precede ALL hashes.
    receipt_path = spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    raw = {'current': read(receipt_path)}
    terminal(raw['current'], spec['phase'])
    require(set(spec['training_receiptmap']) == set(spec['training_phasemap']) == set(LEGACY_MODES),
        'All four original raw identities required')
    for mode in LEGACY_MODES:
        ref = spec['training_receiptmap'][mode]
        require(Path(ref['path']).is_absolute(), 'Absolute original raw path required')
        raw[mode] = read(ref['path'])
        terminal(raw[mode], spec['training_phasemap'][mode])
    if spec['execution_kind'] == 'formal_additional24':
        require(Path(spec['restoration_receipt']['path']).is_absolute(), 'Absolute technical raw path required')
        raw['restoration'] = read(spec['restoration_receipt']['path'])
        terminal(raw['restoration'], spec['restoration_phase'])
    require(Path(spec['GPU_admission_file']).is_absolute(), 'Absolute current admission required')
    admission = read(spec['GPU_admission_file'])
    spec_ref, reader_ref = reference(spec_path), reference(__file__)
    source = reference(Path(__file__).with_name('train.py'))
    admission_ref, receipt_ref = reference(spec['GPU_admission_file']), reference(receipt_path)
    owner = reference(spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py')
    require(raw['current']['admission_sha256'] == admission_ref['sha256'] and owner['sha256'] == OWNER_SHA,
        'Current raw/admission/owner SHA join differs')
    expect(admission, dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE', automatic_retry=False,
        phase=spec['phase'], allowed_cpus=[0, 1], cuda_visible_devices='0',
        qualified_leaf_deadline_seconds=spec['deadline_seconds'], whole_GPU_used_limit_MiB=8192,
        entry_args=['--spec', str(spec_path), '--owner-execute'], driver_sha256=OWNER_SHA), 'Current owned admission')
    require(Path(admission['entry']).resolve(strict=True) == Path(source['path']) and
        spec['entry_source'] == source and spec['CPU_reader'] == reader_ref and
        source in spec['sources'] and reader_ref in spec['sources'] and MICRO_CPU in spec['sources'] and
        len({r['path'] for r in spec['sources']}) == len(spec['sources']), 'Exact producer/reader/helper closure differs')
    hashes = admission['bound_source_sha256']
    require(hashes.get(str(spec_path)) == spec_ref['sha256'], 'SPEC is not owner-bound')
    for ref in spec['sources']:
        bound(ref)
        require(hashes.get(ref['path']) == ref['sha256'], 'Complete SOURCE closure is not owner-bound')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Current admitted scientific SPEC differs')
    proposal = read_ref(spec['SOURCE_proposal'])
    expect(proposal, dict(kind='ROOT_INERT_NATIVE_FINETUNE24_SOURCE_PROPOSAL_NOT_SCIENCE_ADMISSION', schema=SCHEMA,
        phase_epochs=24, phase_microsteps=17784, phase_updates=8892, phase_raw_GT_visits=39864,
        teacher_epoch=24, order_seed=ORDER_SEED, phase_loss_seed_base=LOSS_SEED,
        old_optimizer_restored=False, old_RNG_restored=False), 'Canonical inert SOURCE proposal')
    require(spec['SOURCE_proposal'] in spec['sources'], 'Inert proposal must be admitted')
    for role, ref in dict(proposal['immutable_SOURCE_refs'], **proposal['library_SOURCE_refs']).items():
        require(spec[role] == ref and ref in spec['sources'], 'Canonical SOURCE role differs: ' + role)
    require(proposal['producer_SOURCE'] == source, 'Inert proposal producer SOURCE pin differs')
    bound(owner)
    return dict(source=source, reader_source=reader_ref, spec=spec_ref, parent_receipt=receipt_ref,
        GPU_admission=admission_ref, owner_driver=owner, raw=raw, proposal=proposal)


def restoration_gate(spec, refs):
    if spec['execution_kind'] != 'formal_additional24':
        return None
    q = read_ref(spec['restoration_CPU_qualification'])
    expect(q, dict(schema=TECH_SCHEMA, status=TECH_STATUS, source=refs['source'],
        reader_source=spec['restoration_CPU_reader'], spec=spec['restoration_spec'],
        parent_receipt=spec['restoration_receipt'], phase=spec['restoration_phase'],
        teacher_checkpoint=spec['checkpointmap']['native0'], teacher_qualification=spec['qualificationmap']['native0'],
        initial_native610_CPU_exact=True, typed_BN106_affine159_buffers_exact=True, native345_active=True,
        fresh_optimizer_state_entries=0, old_optimizer_restored=False, old_RNG_restored=False,
        native_loss_atoms=30, technical_microsteps=4, technical_updates=2,
        formal_training_or_resume_qualified=False), 'Actual fresh technical restoration qualification')
    for role in ('source', 'reader_source', 'spec', 'report', 'start', 'parent_receipt', 'GPU_admission', 'owner_driver'):
        bound(q[role])
    for role in ('restoration_CPU_qualification', 'restoration_CPU_reader', 'restoration_spec', 'restoration_receipt'):
        require(spec[role] in spec['sources'], 'Technical lineage role absent from current closure: ' + role)
    technical = read_ref(spec['restoration_spec'])
    expect(technical, dict(schema=SCHEMA, execution_kind='technical_restore_four_microsteps',
        phase=spec['restoration_phase'], entry_source=refs['source'], CPU_reader=spec['restoration_CPU_reader']),
        'Qualified technical SPEC')
    for role in ('initializers', 'schedule', 'backbone_with_cp', 'order_seed', 'phase_loss_seed_base',
                 'checkpointmap', 'qualificationmap', 'FIT', 'image_root'):
        require(technical[role] == spec[role], 'Formal/technical recipe differs: ' + role)
    adm = read_ref(q['GPU_admission'])
    require(refs['raw']['restoration']['admission_sha256'] == q['GPU_admission']['sha256'] and
        q['owner_driver']['sha256'] == OWNER_SHA,
        'Technical raw/admission SHA join differs')
    expect(adm, dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE', automatic_retry=False,
        phase=technical['phase'], allowed_cpus=[0, 1], cuda_visible_devices='0', driver_sha256=OWNER_SHA,
        qualified_leaf_deadline_seconds=technical['deadline_seconds'], whole_GPU_used_limit_MiB=8192,
        entry_args=['--spec', q['spec']['path'], '--owner-execute']), 'Original technical admission')
    require(Path(adm['entry']).resolve(strict=True) == Path(refs['source']['path']) and
        adm['bound_source_sha256'].get(q['spec']['path']) == q['spec']['sha256'], 'Technical entry/SPEC binding differs')
    for ref in technical['sources']:
        bound(ref)
        require(ref in spec['sources'] and adm['bound_source_sha256'].get(ref['path']) == ref['sha256'],
            'Original technical closure must be preserved and bound')
    if 'scientific_spec' in adm:
        require(adm['scientific_spec'] == q['spec'], 'Original technical admitted SPEC differs')
    return q


def validate_recipe(spec, train):
    require(spec['initializers'] == train['initializers'] and spec['schedule'] == train['schedule'] == SCHEDULE and
        spec['backbone_with_cp'] is train['backbone_with_cp'], 'Exact original qualified recipe/checkpointing required')
    entry = spec['initializers']['mask2former']
    expect(entry['optimizer'], dict(type='AdamW', lr=.0001), 'Original native optimizer recipe')
    require(entry['clip_grad'] == dict(max_norm=.01, norm_type=2), 'Original global L2 clip.01 required')
    for key in ('FIT', 'image_root', 'native_helpers', 'initializer_source', 'micro_preflight_source'):
        require(spec[key] == train[key], 'Original construction/input differs: ' + key)


def digest_contract(row, shape, dtype):
    require(isinstance(row, dict) and set(row) == {'shape', 'dtype', 'sha256'} and row['shape'] == shape and
        row['dtype'] == dtype and isinstance(row['sha256'], str) and len(row['sha256']) == 64 and
        set(row['sha256']) <= set('0123456789abcdef'), 'Tensor digest shape/dtype/SHA contract differs')


def build_start(torch, old, common, initializer, registry, revert, Config, spec, teacher, start, refs):
    train, q = teacher['training_spec'], teacher['qualification']
    validate_recipe(spec, train)
    loading = start['teacher_loading']
    initialized = read_ref(train['initializer_qualification'])
    model, cold = common.build_initial_native(torch, old, initializer, registry, revert, Config,
        train, initialized, loading['original_initialization'])
    policy = old.independent_policy(torch, model, 'mask2former')
    require(policy['trainable_tensors'] == 345 and policy['parameter_tensors'] == 451 and policy['state_entries'] == 610 and
        policy['frozen_tensors'] == 106 and len(policy['BN_buffer_names']) == 159 and len(policy['BN_modules']) == 53 and
        len(policy['backbone_conv_names']) == 53 and not any('readout' in n for n in model.state_dict()),
        'Independent bare native345/451/610 and typedBN106+159 policy differs')
    cp = torch.load(str(bound(spec['checkpointmap']['native0'])), map_location='cpu')
    require(set(cp) == {'state_dict', 'optimizer_state_dict', 'external_rng_state', 'private_order_state', 'meta'},
        'Full qualified original teacher container required')
    meta = cp['meta']
    expect(meta, dict(schema=common.CP_SCHEMA, kind='mask2former', mode='native0', source=spec['training_source'],
        spec=q['spec'], phase=q['phase'], training_protocol=q['training_protocol'], initial_epoch=0, completed_epoch=24,
        global_microsteps=17784, global_updates=8892, consumed_raw_GT=39864, no_pending_accumulation_gradients=True,
        not_technical_preflight_checkpoint=True, fresh_single_owned_24=True, eligible_for_automatic_resume=False,
        resume_checkpoint=None, resume_qualification=None), 'Original qualified E24 teacher metadata')
    anchor = cp['state_dict']
    require(set(anchor) == set(cold) and len(anchor) == 610 and all(isinstance(v, torch.Tensor) and
        v.device.type == 'cpu' and v.shape == cold[n].shape and v.dtype == cold[n].dtype and
        bool(torch.isfinite(v).all()) for n, v in anchor.items()), 'Full teacher native610 keys/types/finite differ')
    qualified_proof = read_ref(q['model_proof']['proof_refs']['state_proof'])
    common.cpu_state(torch, old, anchor, qualified_proof['after'], 'Qualified original E24 CPU state')
    fixed = policy['frozen_names'] + policy['BN_buffer_names']
    require(set(anchor) - set(policy['active_names']) == set(fixed) and
        all(torch.equal(anchor[n], cold[n]) for n in fixed), 'TypedBN teacher state must be initializer-exact')
    common.cpu_state(torch, old, anchor, loading['teacher_state'], 'Loading teacher610')
    common.cpu_state(torch, old, anchor, start['actual_native_state'], 'Actual fresh START native610')
    expect(loading, dict(schema='native-finetune24-fresh-optimizer-loading-proof-v1',
        teacher_checkpoint=spec['checkpointmap']['native0'], teacher_qualification=spec['qualificationmap']['native0'],
        teacher_training_source=spec['training_source'], teacher_epoch=24, strict_full_native610_loaded=True,
        typed_BN_initializer_exact=True, native_policy=policy, old_checkpoint_optimizer_and_RNG_containers_deserialized=True,
        old_optimizer_restored=False, old_RNG_restored=False, old_automatic_resume_eligible=False,
        fresh_phase_not_formal_resume=True, registered_adapter_tensors=0), 'Actual teacher loading proof')
    model.requires_grad_(True).train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            module.requires_grad_(False)
    loaded = model.load_state_dict(anchor, strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys and
        all(torch.equal(v.detach().cpu(), anchor[n]) for n, v in model.state_dict().items()) and
        all(v.dtype == torch.float32 for v in model.parameters()) and model.backbone.with_cp is spec['backbone_with_cp'],
        'Strict independently constructed native E24 model/policy differs')
    groups, audit = old.independent_optimizer(torch, model, spec['initializers']['mask2former'], policy)
    names = [[n] for n in policy['active_names']]
    base_lrs = [g['lr'] for g in groups]
    require(len(groups) == 345 and audit['type'] == 'AdamW' and
        canonical(audit) == canonical(start['optimizer']), 'Independent exact native345 paramwise AdamW differs')
    expect(start, dict(schema='native-finetune24-fresh-optimizer-start-state-v1', source=refs['source'], spec=refs['spec'],
        policy=policy, base_lrs=base_lrs, fresh_optimizer_state_entries=0, old_optimizer_restored=False,
        old_RNG_restored=False, phase_microsteps=0, phase_updates=0, initial_native_epoch=24,
        phase_initial_epoch=0, fresh_phase_not_formal_resume=True), 'Fresh actual START state')
    require(start['actual_optimizer'] == common.optimizer_snapshot(torch, old,
        dict(state={}, param_groups=groups), names), 'Actual START optimizer must have empty moments/exact groups')
    require(isinstance(start['external_RNG'], list) and len(start['external_RNG']) == 2, 'START CPU/CUDA RNG digests required')
    for d in start['external_RNG']:
        digest_contract(d, d['shape'], 'uint8')
        require(len(d['shape']) == 1 and type(d['shape'][0]) is int and d['shape'][0] > 0, 'RNG byte vector invalid')
    del cp, qualified_proof, cold
    gc.collect()
    return model, anchor, policy, groups, audit, names, base_lrs, loading


def imported_library_gate(torch, model, spec):
    head = model.panoptic_head
    actual = dict(native_mask_head_SOURCE=type(head), native_parent_head_SOURCE=type(head).__mro__[1],
        panoptic_GT_SOURCE=head.preprocess_gt.__func__.__globals__['preprocess_panoptic_gt'],
        hungarian_SOURCE=type(head.assigner), mask_sampler_SOURCE=type(head.sampler),
        mask_sampling_result_SOURCE=head.sampler.sample.__func__.__globals__['MaskSamplingResult'],
        point_sample_SOURCE=head._get_targets_single.__func__.__globals__['point_sample'],
        adamw_SOURCE=torch.optim.AdamW, clip_grad_SOURCE=torch.nn.utils.clip_grad_norm_)
    for role, obj in actual.items():
        require(Path(inspect.getsourcefile(obj)).resolve() == bound(spec[role]).resolve(), 'Actual imported SOURCE differs: ' + role)
    require(torch.__version__ == '2.0.1+cu118' and
        (Path(torch.__file__).resolve().parent / 'version.py').resolve() == bound(spec['torch_version_source']).resolve(),
        'Actual bound Torch version differs')
    require(head.num_queries == 100 and head.num_classes == 1 and head.num_points == 12544 and
        [type(c).__name__ for c in head.assigner.match_costs] == ['ClassificationCost', 'CrossEntropyLossCost', 'DiceCost'] and
        [c.weight for c in head.assigner.match_costs] == [2., 5., 5.], 'Original native dynamic matching recipe differs')
    for c in head.assigner.match_costs:
        require(Path(inspect.getsourcefile(type(c))).resolve() == bound(spec['match_cost_SOURCE']).resolve(),
            'Actual native match cost SOURCE differs')


def independent_orders(torch, old, rows, roster, technical):
    count = 1 if technical else 24
    require(isinstance(rows, list) and len(rows) == count, 'Complete private order rows required')
    gen = torch.Generator(device='cpu').manual_seed(ORDER_SEED)
    orders, states = [], []
    for epoch in range(count):
        before = gen.get_state().clone()
        order = torch.randperm(741, generator=gen).tolist()
        after = gen.get_state().clone()
        require(sorted(order) == list(range(741)), 'Independent complete randperm741 required')
        require(rows[epoch] == dict(phase_epoch=epoch, order_seed=ORDER_SEED, dataset_indices=order,
            image_ids=[roster[i] for i in order], private_CPU_RNG_before=old.tensor_digest(torch, before, 'uint8'),
            private_CPU_RNG_after=old.tensor_digest(torch, after, 'uint8')), 'Private Torch order seed/continuity differs')
        orders.append(order)
        states.append(after)
    return orders, states


def native_observations(torch, old, row, ann_ids, micro):
    head = row['native_head_fields']
    expect(head, dict(readouts=10, classes_shape=[1, 100, 2], masks_shape=[1, 100, 200, 200],
        all_fields_require_grad=True), 'Actual native ten live readouts')
    cold = head['cold_micro_digests']
    if micro == 0:
        require(set(cold) == {'classes', 'masks'} and len(cold['classes']) == len(cold['masks']) == 10,
            'First qualified E24 forward ten-field digests required')
        for d in cold['classes']:
            digest_contract(d, [1, 100, 2], 'float32')
        for d in cold['masks']:
            digest_contract(d, [1, 100, 200, 200], 'float32')
    else:
        require(cold is None, 'Dense first-forward digest scope must remain micro0 only')
    matches = row['native_matches']
    G = len(ann_ids)
    require(isinstance(matches, list) and len(matches) == 10, 'Actual original ten assignments required')
    for match in matches:
        qs, gs = match['positive_query_indices'], match['assigned_GT_indices']
        require(type(match['positives']) is int and match['positives'] == G == len(qs) == len(gs) and
            len(set(qs)) == G and all(type(q) is int and 0 <= q < 100 for q in qs) and
            all(type(g) is int for g in gs) and sorted(gs) == list(range(G)), 'One-to-one raw GT/query assignment coverage differs')
        labels = torch.ones(100, dtype=torch.int64)
        labels[qs] = 0
        require(old.tensor_digest(torch, labels, 'int64') == match['target_labels'], 'Actual assignment class target digest differs')


def trace_check(torch, np, cv2, coco, old, common, spec, result, policy, shapes, images, roster, orders, start_rng, started):
    technical = spec['execution_kind'] == 'technical_restore_four_microsteps'
    limit = 4 if technical else 17784
    active = policy['active_names']
    positive, counts = ({n: 0 for n in active} for _ in range(2))
    cache, prefixes, epoch_GT = {}, {}, [0] * (1 if technical else 24)
    previous_rng, total_GT, seen, first, boundaries = start_rng, 0, 0, None, []
    pending_boundary, exact_boundary_joins = None, []
    with bound(result['trace']).open('r', encoding='utf-8') as stream:
        for micro, line in enumerate(stream):
            require(micro < limit and line.strip(), 'Extra/empty trace record')
            require(time.monotonic() - started < spec['deadline_seconds'], 'CPU qualification deadline exceeded')
            row = json.loads(line)
            epoch, within = divmod(micro, 741)
            index = orders[epoch][within]
            image_id, stepped, U = roster[index], (micro + 1) % 2 == 0, micro // 2
            expect(row, dict(mode='native0', execution_kind=spec['execution_kind'], phase_microstep=micro,
                phase_epoch=epoch, effective_native_epoch_index=24 + epoch, within_epoch=within,
                native_model_loss_only=True, original_native_loss_atoms=30,
                first_forward_digest_scope='phase_micro0_from_qualified_nativeE24', accumulation_divisor=2,
                optimizer_step=stepped, phase_updates=(micro + 1) // 2, scheduler_u_before_step=U,
                lr_factor=common.schedule_factor(spec['schedule'], U), gradient_clipping_applied=stepped,
                gradients_are_accumulated_preclip=True, frozen_BN_parameter_grads_None=True,
                BN_module_training={n: False for n in policy['BN_modules']}, private_seed=LOSS_SEED + micro,
                external_RNG_restored=True), 'Actual native phase trace counters/recipe')
            expect(row['identity'], dict(dataset_index=index, image_id=image_id), 'Actual private dataset identity')
            ann_ids = [int(a['id']) for a in coco.imgToAnns.get(image_id, [])]
            native_observations(torch, old, row, ann_ids, micro)
            if image_id not in cache:
                old.validate_GT(torch, np, cv2, coco, images[image_id], ann_ids,
                    dict(row, actual_input_shape=row['input']['shape']))
                cache[image_id] = (row['identity'], row['GT'], row['input'])
            else:
                require(cache[image_id] == (row['identity'], row['GT'], row['input']),
                    'Repeated nonaugmented complete raw GT/input identity differs')
            old.validate_losses(dict(row, loss_is_native_model_loss=True, Sobel_added=False), 'mask2former')
            old.validate_rng(torch, row)
            require(row['external_RNG_before'] == previous_rng, 'Actual START/trace external RNG continuity differs')
            previous_rng = row['external_RNG_after']
            norms = row['gradient_norms_aligned_active_names']
            require(isinstance(norms, list) and len(norms) == 345, 'All345 aligned accumulated gradient entries required')
            if technical:
                require(all(v is not None for v in norms), 'Technical wiring requires all345 finite present gradients')
            for n, v in zip(active, norms):
                if v is not None:
                    old.finite(v, 'Actual accumulated gradient ' + n, True)
                    positive[n] += int(v > 0)
                    counts[n] += int(stepped)
            if stepped:
                old.finite(row['preclip_global_norm'], 'Actual native global preclip norm', True)
            else:
                require(row['preclip_global_norm'] is None, 'No clipping/update on pending first micro')
            odd_end = within == 740 and (micro + 1) % 2 == 1
            odd_start = within == 0 and micro % 2 == 1
            gradient_map = row['odd_epoch_end_pending_gradients']
            if odd_end:
                require(isinstance(gradient_map, dict) and set(gradient_map) == set(active) and
                    pending_boundary is None, 'Complete pending345 gradients at odd741 epoch end required')
                for n, norm in zip(active, norms):
                    if norm is None:
                        require(gradient_map[n] is None, 'Missing gradient acquired a boundary tensor proof')
                    else:
                        digest_contract(gradient_map[n], shapes[n], 'float32')
                pending_boundary = dict(micro=micro, gradients=gradient_map)
            else:
                require(gradient_map is None, 'Unexpected odd-epoch pending-gradient packet')
            join = row['pending_boundary_gradient_join']
            if odd_start:
                require(pending_boundary is not None and pending_boundary['micro'] == micro - 1,
                    'Missing preceding odd741 pending-gradient proof')
                require(join == dict(previous_phase_microstep=micro - 1,
                    after_previous_odd_epoch=pending_boundary['gradients'], before_next_forward=pending_boundary['gradients'],
                    exact=True, SOURCE_observed_not_CPU_gradient_replay=True),
                    'Actual odd741 accumulated-gradient digests changed before next forward')
                exact_boundary_joins.append(dict(previous_phase_microstep=micro - 1, next_phase_microstep=micro,
                    active_gradient_fields=345, actual_digest_join_exact=True, SOURCE_observed_not_CPU_gradient_replay=True))
                pending_boundary = None
            else:
                require(join is None, 'Unexpected odd-epoch boundary join')
            if within in (0, 740):
                boundaries.append(dict(phase_microstep=micro, phase_epoch=epoch, within_epoch=within,
                    optimizer_step=stepped, phase_updates=(micro + 1) // 2, pending_microsteps=(micro + 1) % 2))
            epoch_GT[epoch] += len(ann_ids)
            total_GT += len(ann_ids)
            seen += 1
            if first is None:
                first = row
            if seen == limit or (within == 740 and epoch + 1 in NODES):
                prefixes[str(0 if technical else epoch + 1)] = dict(rows=seen, raw_GT=total_GT,
                    positive=dict(positive), per_parameter_updates=dict(counts), last_external_rng=previous_rng)
    require(seen == limit and pending_boundary is None and len(exact_boundary_joins) == (0 if technical else 12) and
        (technical or (len(cache) == 741 and total_GT == 39864 and epoch_GT == [1661] * 24)),
        'Complete actual trace/FIT741/all raw1661 per epoch differs')
    return dict(rows=seen, raw_GT=total_GT, epoch_GT=epoch_GT, unique_images=len(cache), prefixes=prefixes,
        positive=positive, per_parameter_updates=counts, last_external_rng=previous_rng,
        first_forward=first['native_head_fields'], odd_epoch_boundaries=boundaries, exact_boundary_joins=exact_boundary_joins)


def proof_state(torch, old, common, state, anchor, proof, policy, audit, loading, prefix):
    require(set(state) == set(anchor) and len(state) == 610 and all(isinstance(v, torch.Tensor) and
        v.device.type == 'cpu' and v.shape == anchor[n].shape and v.dtype == anchor[n].dtype and
        bool(torch.isfinite(v).all()) for n, v in state.items()), 'Complete native610 CPU endpoint types/finite differ')
    common.cpu_state(torch, old, anchor, proof['before'], 'Proof before means original E24 teacher')
    common.cpu_state(torch, old, state, proof['after'], 'Actual native endpoint CPU state')
    fixed = policy['frozen_names'] + policy['BN_buffer_names']
    require(all(torch.equal(state[n], anchor[n]) for n in fixed), 'A typedBN affine/buffer changed')
    changed = {n: not torch.equal(v, anchor[n]) for n, v in state.items()}
    expect(proof, dict(changed=changed, policy=policy, teacher_loading=loading, frozen_BN_all_exact=True,
        no_pending_accumulation_gradients=True, formal_resume_eligible=False,
        positive_accumulated_gradient_microcounts=prefix['positive'],
        optimizer_step_nonNone_gradient_counts=prefix['per_parameter_updates']), 'Actual native state/gradient proof')
    require(canonical(proof['optimizer']) == canonical(audit), 'Actual state-proof optimizer audit differs')
    return changed


def checkpoint_check(torch, old, common, model, anchor, policy, groups, audit, names, base_lrs, loading,
                     spec, refs, result, trace, order_states):
    technical = spec['execution_kind'] == 'technical_restore_four_microsteps'
    endpoint = read_ref(result['endpoint_proof'])
    endpoint_expected = dict(schema='native-finetune24-fresh-optimizer-endpoint-proof-v1',
        phase_microsteps=trace['rows'], phase_updates=trace['rows'] // 2, phase_raw_GT_visits=trace['raw_GT'],
        completed_phase_epochs=0 if technical else 24, effective_native_epoch=24 if technical else 48,
        native_loss_atoms=30, technical_weights_are_not_formal_initialization=True,
        old_optimizer_restored=False, old_RNG_restored=False)
    expect(endpoint, endpoint_expected, 'Actual native endpoint proof')
    nodes = {}
    targets = [0] if technical else list(NODES)
    require(set(result['nodes']) == (set() if technical else {str(e) for e in NODES}), 'Formal six nodes/technical no nodes differ')
    for epoch in targets:
        prefix = trace['prefixes'][str(epoch)]
        U = prefix['rows'] // 2
        if technical:
            cp_ref = result['technical_CPU_snapshot']
            cp = torch.load(str(bound(cp_ref)), map_location='cpu')
            require(set(cp) == {'schema', 'state_dict', 'optimizer_state_dict', 'external_rng_state', 'private_order_state', 'meta'} and
                cp['schema'] == 'native-finetune24-fresh-optimizer-technical-CPU-snapshot-v1', 'Complete technical CPU debug snapshot required')
            proof = endpoint
            expect(cp['meta'], dict(source=refs['source'], spec=refs['spec'], phase=spec['phase'],
                teacher_checkpoint=spec['checkpointmap']['native0'], phase_microsteps=4, phase_updates=2,
                phase_raw_GT_visits=prefix['raw_GT'], optimizer_param_names=names, base_lrs=base_lrs,
                schedule=spec['schedule'], policy=policy, technical_only=True, formal_resume_eligible=False,
                technical_weights_are_not_formal_initialization=True), 'Technical debug snapshot metadata')
            require(canonical(cp['meta']['optimizer']) == canonical(audit), 'Technical snapshot optimizer audit differs')
            order_meta = dict(seed=ORDER_SEED, generated_phase_epochs=1)
            state_index = 0
        else:
            node = result['nodes'][str(epoch)]
            cp_ref = node['checkpoint']
            cp = torch.load(str(bound(cp_ref)), map_location='cpu')
            require(set(cp) == {'state_dict', 'optimizer_state_dict', 'external_rng_state', 'private_order_state', 'meta'},
                'Complete formal native node container required')
            factor = common.schedule_factor(spec['schedule'], U - 1)
            expect(cp['meta'], dict(schema=CP_SCHEMA, kind='mask2former', mode='native0', phase=spec['phase'],
                source=refs['source'], spec=refs['spec'], execution_kind='formal_additional24',
                teacher_checkpoint=spec['checkpointmap']['native0'], teacher_epoch=24, phase_initial_epoch=0,
                completed_phase_epoch=epoch, effective_native_epoch=24 + epoch, phase_microsteps=epoch * 741,
                phase_updates=U, phase_raw_GT_visits=epoch * 1661, effective_native_microsteps=17784 + epoch * 741,
                effective_native_updates=8892 + U, effective_native_raw_GT_visits=39864 + epoch * 1661,
                phase_recipe=spec['initializers']['mask2former'], schedule=spec['schedule'], base_lrs=base_lrs,
                stored_lr_factor=factor, optimizer_param_names=names, fresh_optimizer_at_phase_start=True,
                old_optimizer_restored=False, old_RNG_restored=False, no_pending_accumulation_gradients=True,
                formal_resume_eligible=False), 'Complete formal native node metadata')
            proof = read_ref(node['state_proof'])
            expect(proof, dict(schema='native-finetune24-fresh-optimizer-node-state-proof-v1',
                source=refs['source'], spec=refs['spec'], completed_phase_epoch=epoch, effective_native_epoch=24 + epoch,
                phase_microsteps=epoch * 741, phase_updates=U, phase_raw_GT_visits=epoch * 1661,
                CPU_checkpoint_readback_exact=True), 'Actual formal node state proof')
            expect(node, dict(completed_phase_epoch=epoch, effective_native_epoch=24 + epoch,
                phase_microsteps=epoch * 741, phase_updates=U, phase_raw_GT_visits=epoch * 1661,
                stored_lr_factor=factor, frozen_BN_all_exact=True, CPU_checkpoint_readback_exact=True), 'Actual formal node result')
            order_meta = dict(seed=ORDER_SEED, next_phase_epoch=epoch)
            state_index = epoch - 1
        state = cp['state_dict']
        changed = proof_state(torch, old, common, state, anchor, proof, policy, audit, loading, prefix)
        common.validate_moments(torch, cp['optimizer_state_dict'], common.groups_at(groups, spec['schedule'], U - 1),
            policy['active_names'], state, 'AdamW', audit['recipe'], prefix['per_parameter_updates'])
        snapshot = common.optimizer_snapshot(torch, old, cp['optimizer_state_dict'], names)
        require(snapshot == proof['actual_optimizer'], 'Independent actual optimizer moments/groups/digests differ')
        rng = common.check_rng_tensors(torch, old, cp['external_rng_state'])
        require(rng == prefix['last_external_rng'], 'Actual node/snapshot external RNG differs from trace endpoint')
        private = cp['private_order_state']
        require(set(private) == set(order_meta) | {'torch_CPU_generator_state'}, 'Private order container fields differ')
        expect(private, order_meta, 'Actual node private order counters')
        generator_state = private['torch_CPU_generator_state']
        require(isinstance(generator_state, torch.Tensor) and generator_state.device.type == 'cpu' and
            generator_state.dtype == torch.uint8 and torch.equal(generator_state, order_states[state_index]),
            'Actual native node private CPU order position differs')
        loaded = model.load_state_dict(state, strict=True)
        require(not loaded.missing_keys and not loaded.unexpected_keys and all(torch.equal(v.detach().cpu(), state[n])
            for n, v in model.state_dict().items()), 'Strict full native610 CPU endpoint load differs')
        if technical or epoch == 24:
            proof_state(torch, old, common, state, anchor, endpoint, policy, audit, loading, prefix)
            require(endpoint['actual_optimizer'] == snapshot, 'Final endpoint optimizer proof differs')
        nodes[str(epoch)] = dict(checkpoint=cp_ref, state_proof=result['endpoint_proof'] if technical else node['state_proof'],
            phase_microsteps=prefix['rows'], phase_updates=U, phase_raw_GT_visits=prefix['raw_GT'],
            completed_phase_epoch=epoch, effective_native_epoch=24 + epoch, full_native610_CPU_verified=True,
            typed_BN106_affine159_buffers_exact=True, full_AdamW_moments_and_step_counts_verified=True,
            private_order_position_verified=True, external_RNG_verified=True, changed=changed,
            positive_gradient_microcounts=prefix['positive'], optimizer_update_counts_per_parameter=prefix['per_parameter_updates'],
            zero_positive_gradient_names=[n for n in policy['active_names'] if not prefix['positive'][n]],
            unchanged_active_names=[n for n in policy['active_names'] if not changed[n]])
        del cp, state, proof
        gc.collect()
    require(result['checkpoint'] is None if technical else result['checkpoint'] == result['nodes']['24']['checkpoint'],
        'Technical must not emit formal CP; formal final CP must be phase24 node')
    require((result['technical_CPU_snapshot'] is not None) if technical else result['technical_CPU_snapshot'] is None,
        'Technical debug snapshot scope differs')
    return nodes


def qualify(spec_file, acceptance_file):
    started = time.monotonic()
    spec_path, target = Path(spec_file), Path(acceptance_file)
    require(spec_path.is_absolute() and target.is_absolute() and not target.exists() and target.parent.is_dir(),
        'Absolute SPEC and fresh absolute acceptance in existing directory required')
    spec_path = spec_path.resolve(strict=True)
    spec = read(spec_path)
    phase_gate(spec)
    refs = source_gate(spec_path, spec)
    restoration = restoration_gate(spec, refs)
    # Legacy helper executes only metadata/SOURCE lineage checks before ML imports.
    legacy = load_source(spec['evaluation_source'], 'native_phase_original_qualified_teacher_lineage')
    teacher = legacy.qualified_training(spec)['native0']
    validate_recipe(spec, teacher['training_spec'])
    output = Path(spec['output_directory'])
    require(output.is_absolute() and output.is_dir(), 'Actual completed output directory required')
    report_ref, start_ref = reference(output / 'REPORT.json'), reference(output / 'START.json')
    report, start = read_ref(report_ref), read_ref(start_ref)
    common_metadata = dict(source=refs['source'], spec=refs['spec'], phase=spec['phase'],
        execution_kind=spec['execution_kind'], teacher_checkpoint=spec['checkpointmap']['native0'],
        teacher_qualification=spec['qualificationmap']['native0'], old_optimizer_restored=False,
        old_RNG_restored=False, fresh_phase_not_formal_resume=True, order_seed=ORDER_SEED,
        phase_loss_seed_base=LOSS_SEED, no_DEV=True, no_AP=True)
    expect(start, dict(common_metadata, schema='native-finetune24-fresh-optimizer-start-v1', teacher_epoch=24,
        phase_epochs=24, no_retry=True, technical_CPU_qualification=None if restoration is None else spec['restoration_CPU_qualification']),
        'Completed fresh native START')
    expect(report, dict(common_metadata, schema=REPORT_SCHEMA, status='NATIVE_FINETUNE24_FRESH_OPTIMIZER_OWNED_COMPLETE_NO_AP',
        GPU_admission=refs['GPU_admission'], error=None, SOURCE_proposal=spec['SOURCE_proposal'],
        phase_recipe=spec['initializers']['mask2former'], schedule=SCHEDULE, native345_trainable=True,
        native451_parameters_610_states=True, typed_BN106_affine159_buffers_frozen=True, native_loss_atoms=30,
        dynamic_ten_stage_native_matching=True, feature_cache_used=False, no_TEST=True, no_automatic_retry=True,
        formal_resume_eligible=False, additional_imagevisits_and_updates_comparator=True, compute_matched_claimed=False,
        objective_matched_claimed=False, efficacy_claimed=False, convergence_claimed=False, novelty_claimed=False, SOTA_claimed=False),
        'Completed native phase report')
    cleanup = report['cleanup']
    require(isinstance(cleanup, dict) and set(cleanup) == {'allocated_before', 'allocated_after'} and
        type(cleanup['allocated_before']) is int and cleanup['allocated_before'] >= 0 and
        type(cleanup['allocated_after']) is int and cleanup['allocated_after'] == 0,
        'Actual pref.clear_gpu allocator cleanup must be zero')
    require(type(report['torch_peak_allocated_bytes']) is int and 0 <= report['torch_peak_allocated_bytes'] <= 4096 * 1024 ** 2 and
        type(report['elapsed_seconds']) in (int, float) and math.isfinite(report['elapsed_seconds']) and
        0 <= report['elapsed_seconds'] <= spec['deadline_seconds'], 'Actual allocator/deadline metadata differs')
    result = report['result']
    technical = spec['execution_kind'] == 'technical_restore_four_microsteps'
    expect(result, dict(execution_kind=spec['execution_kind'], phase_microsteps=4 if technical else 17784,
        phase_updates=2 if technical else 8892, completed_phase_epochs=0 if technical else 24,
        effective_native_epoch=24 if technical else 48, frozen_BN_all_exact=True, fresh_optimizer_at_phase_start=True,
        old_optimizer_restored=False, old_RNG_restored=False, formal_resume_eligible=False), 'Completed native phase result')
    actual_start = read_ref(result['start_state_proof'])
    import psutil
    import torch
    import numpy as np
    import cv2
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'ROOT CPU01 and uninitialized CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    require(len(CocoDataset.METAINFO['classes']) == 80 and CocoDataset.METAINFO['classes'][8] == 'boat',
        'Original COCO boat8/null80 class identity required')
    old = load_source(MICRO_CPU, 'native_phase_independent_micro_CPU_helpers')
    common = load_source(spec['training_CPU_reader'], 'native_phase_independent_common24_CPU_helpers')
    require(common.PREF_READER == MICRO_CPU, 'Canonical reused independent helper differs')
    initializer = load_source(spec['initializer_source'], 'native_phase_original_initializer')
    prepared = build_start(torch, old, common, initializer, MODELS, revert_sync_batchnorm, Config,
        spec, teacher, actual_start, refs)
    imported_library_gate(torch, prepared[0], spec)
    dataset = CocoDataset(ann_file=str(bound(spec['FIT'])), data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
    coco = COCO(str(bound(spec['FIT'])))
    images = {int(k): v for k, v in coco.imgs.items()}
    roster = [int(dataset.get_data_info(i)['img_id']) for i in range(len(dataset))]
    require(len(dataset) == len(images) == 741 and len(coco.anns) == 1661 and roster == spec['FIT_image_ids'] and
        len(set(roster)) == 741 and set(roster) == set(images) and
        sum(len(coco.imgToAnns.get(i, [])) for i in roster) == 1661, 'Independent complete ordered FIT741/1661 population differs')
    orders, order_states = independent_orders(torch, old, read_ref(result['order']), roster, technical)
    model, anchor, policy, groups, audit, names, base_lrs, loading = prepared
    trace = trace_check(torch, np, cv2, coco, old, common, spec, result, policy,
        {r['name']: r['shape'] for r in audit['parameter_groups']}, images, roster,
        orders, actual_start['external_RNG'], started)
    require(result['phase_raw_GT_visits'] == trace['raw_GT'], 'Actual raw GT counter differs')
    epoch_rows = read_ref(result['epoch_rows'])
    expected_epochs = [] if technical else [dict(phase_epoch=e, images=741, raw_GT=1661,
        phase_microsteps=(e + 1) * 741, epoch_updates=(e + 1) * 741 // 2 - e * 741 // 2,
        phase_updates=(e + 1) * 741 // 2, pending_accumulation_microsteps=((e + 1) * 741) % 2) for e in range(24)]
    require(epoch_rows == expected_epochs, 'Actual24 complete odd741 accumulation epoch summaries differ')
    nodes = checkpoint_check(torch, old, common, model, anchor, policy, groups, audit, names, base_lrs,
        loading, spec, refs, result, trace, order_states)
    require(not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
        torch.get_num_threads() == torch.get_num_interop_threads() == 2 and psutil.Process().cpu_affinity() == [0, 1],
        'Independent CPU reader resource/CUDA scope changed')
    for ref in spec['sources']:
        bound(ref)
    require(time.monotonic() - started < spec['deadline_seconds'], 'CPU qualification deadline exceeded')
    acceptance = dict(schema=TECH_SCHEMA if technical else QUAL_SCHEMA, status=TECH_STATUS if technical else QUAL_STATUS,
        source=refs['source'], reader_source=refs['reader_source'], spec=refs['spec'], phase=spec['phase'],
        report=report_ref, start=start_ref, parent_receipt=refs['parent_receipt'], GPU_admission=refs['GPU_admission'],
        owner_driver=refs['owner_driver'], execution_kind=spec['execution_kind'], SOURCE_proposal=spec['SOURCE_proposal'],
        teacher_checkpoint=spec['checkpointmap']['native0'], teacher_qualification=spec['qualificationmap']['native0'],
        initial_native610_CPU_exact=True, typed_BN106_affine159_buffers_exact=True, native345_active=True,
        fresh_optimizer_state_entries=0, old_optimizer_restored=False, old_RNG_restored=False, native_loss_atoms=30,
        technical_microsteps=4 if technical else None, technical_updates=2 if technical else None,
        formal_additional24_completed=not technical,
        formal_training_completion_CPU_state_qualified=not technical, formal_resume_qualified=False,
        completed_phase_epochs=0 if technical else 24, effective_native_epoch=24 if technical else 48,
        phase_microsteps=trace['rows'], phase_updates=trace['rows'] // 2, phase_raw_GT_visits=trace['raw_GT'],
        checkpoint=result['checkpoint'], technical_CPU_snapshot=result['technical_CPU_snapshot'], nodes=nodes,
        technical_restoration_qualification=None if technical else spec['restoration_CPU_qualification'],
        formal_rebuilds_original_teacher_without_technical_states=not technical,
        formal_resume_eligible=False, inference_executed=False, AP_computed=False, efficacy_claimed=False,
        convergence_claimed=False, novelty_claimed=False, SOTA_claimed=False,
        terminal_and_complete_admission_closure_bound_before_scientific_access=True,
        independent_COCO_boat8_null80_mapping_verified=True, independently_verified_private_Torch_order=True,
        complete_raw_GT_annotation_and_padded_mask_proofs_verified=True,
        native_ten_stage_assignment_coverage_verified=True, native_loss30_exported_sum_verified=True,
        all_active_technical_gradient_entries_present_finite=technical, full_native_CPU_states_and_AdamW_moments_verified=True,
        actual_gradient_observations=dict(positive=trace['positive'], update_counts=trace['per_parameter_updates']),
        first_native_forward_digest=trace['first_forward'], odd_epoch_boundaries=trace['odd_epoch_boundaries'],
        actual_odd_epoch_pending_gradient_digest_joins=trace['exact_boundary_joins'],
        proof_refs={k: result[k] for k in ('start_state_proof', 'endpoint_proof', 'trace', 'order', 'epoch_rows')},
        CPU_execution=dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2, torch_interop_threads=2,
            cuda_visible_devices='', CUDA_initialized=False),
        evidence_limits=[
            'Technical4micro2updates checks native restoration and wiring only; its snapshot is never a formal initializer or a resume checkpoint.',
            'Formal additional24 is a new optimizer/RNG/scheduler phase from qualified original E24 weights, not continuation of the original optimizer trajectory.',
            'Full native states, typed BN, named paramwise optimizer groups/moments/steps, private order and raw GT descriptors are independently checked on CPU.',
            'Native CUDA forward, sampled matching cost derivation/optimality, loss values and gradient elements are SOURCE-observed, not independently numerically replayed.',
            'Thirty exported atomic loss means and their FP32 total, ten per-stage assignment coverage and observed finite gradient/update counters are checked for consistency.',
            'Odd741 accumulation is checked through global counters/epoch summaries and all12 actual pending-gradient digest joins before the next forward; these are observed CUDA proofs, not CPU-gradient replay.',
            'No positivity or parameter-change condition is treated as effectiveness; valid finite zero gradients and unchanged active parameters are reported.',
            'No DEV/TEST/inference/AP, convergence, causal mechanism, global failure, novelty, SOTA or matched-compute/objective claim is qualified.'])
    if technical:
        acceptance['formal_training_or_resume_qualified'] = False
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(acceptance, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=acceptance['status'], acceptance=str(target), execution_kind=spec['execution_kind']),
        sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--acceptance', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'SOURCE_ONLY: explicit ROOT CPU execution required')
    qualify(args.spec, args.acceptance)


if __name__ == '__main__':
    main()
