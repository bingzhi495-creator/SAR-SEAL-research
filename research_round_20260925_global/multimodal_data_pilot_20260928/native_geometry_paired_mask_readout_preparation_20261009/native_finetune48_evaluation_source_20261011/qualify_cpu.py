"""Independent CPU qualification of the fixed native phase48/nativeE96 endpoint.

ROOT alone executes this SOURCE. Current and all eight ancestral actual raw
terminals precede every hash, imported SOURCE and scientific payload. The
original independent COCO-to-ship construction and typed policy establish the
bare native610 boundary; evaluation state is checked against NEW E96 weights.
All100 FP32 final class/M200 fields support independent CPU rendering and
scalar score checks. No original GPU raster logits are saved: threshold flips
are counted within a prelocked practical near-zero envelope, not a certified
GPU interpolation bound or bitwise replay. Original COCO12 and raw-GT best100
diagnostics use the actual exported GPU RLE, retaining crowd/ignore/empty GT.
No training, CUDA initialization, matching, optimizer replay or node selection.
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

SCHEMA = 'ROOT-native-finetune48-finalE96-evaluation-spec-v1'
REPORT_SCHEMA = 'native-finetune48-finalE96-evaluation-report-v1'
REPORT_STATUS = 'NATIVE_FINETUNE48_FINAL_E96_EVALUATION_COMPLETE_NO_GPU_AP'
QUAL_SCHEMA = 'ROOT-native-finetune48-finalE96-evaluation-CPU-qualification-v1'
QUAL_STATUS = 'ROOT_NATIVE_FINETUNE48_FINAL_E96_EVALUATION_CPU_QUALIFIED'
PROPOSAL_KIND = 'ROOT_INERT_NATIVE_FINETUNE48_FINAL_E96_EVALUATION_SOURCE_PROPOSAL_NOT_SCIENCE_ADMISSION'
CP_SCHEMA = 'native-finetune48-E48warm-weights-fresh-optimizer-checkpoint-v1'
ENDPOINT = 'nativeE96'
LEGACY_MODES = ('native0', 'global', 'spatial_halves', 'spatial_reversed')
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
PRODUCER_BYTES = 39209
PRODUCER_SHA = 'fdecb19a63feafc8e47e1b1bfc89c68bbad31fe71babeace0c043f50a678120a'
CPU_NUMERIC = dict(CPU_pixel_envelope='1e-4+1e-4*abs(CPU_rendered_logit)',
    class_probability_atol=1e-6, class_probability_rtol=1e-6,
    renderer_quality_atol=1e-4, renderer_quality_rtol=1e-4,
    scope='practical near-zero/scalar checks, not a certified GPU original-logit error bound')


REQUIRED_SOURCE_ROLES = ('DEV', 'entry_source', 'CPU_reader', 'SOURCE_proposal',
    'native_phase_source', 'native_phase_CPU_reader', 'native_phase_SOURCE_proposal',
    'native_phase_spec', 'native_phase_qualification', 'native_phase_receipt',
    'native_phase_GPU_admission', 'native_phase_checkpoint',
    'restoration_receipt', 'restoration_CPU_qualification', 'restoration_CPU_reader', 'restoration_spec',
    'native48_receipt', 'native48_restoration_receipt')

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
        set(ref['sha256']) <= set('0123456789abcdef'), 'Exact absolute pinned reference required')
    require(reference(ref['path']) == ref, 'Pinned input differs: ' + ref['path'])
    return Path(ref['path'])


def read_ref(ref):
    return read(bound(ref))


def expect(row, values, label):
    require(isinstance(row, dict), label + ' must be an object')
    for key, value in values.items():
        require(key in row and row[key] == value and
            (type(value) not in (int, bool) or type(row[key]) is type(value)), label + ' differs: ' + key)


def finite(value, label, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value) and
        (not nonnegative or value >= 0), label + ' must be finite' + (' and nonnegative' if nonnegative else ''))


def digest_contract(row, shape, dtype):
    require(isinstance(row, dict) and set(row) == {'shape', 'dtype', 'sha256'} and row['shape'] == shape and
        row['dtype'] == dtype and isinstance(row['sha256'], str) and len(row['sha256']) == 64 and
        set(row['sha256']) <= set('0123456789abcdef'), 'Tensor digest shape/dtype/SHA differs')


def tensor_digest(torch, value):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu', 'CPU tensor required')
    array = value.detach().contiguous().numpy()
    return dict(shape=list(value.shape), dtype=str(array.dtype), sha256=hashlib.sha256(array.tobytes()).hexdigest())


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
        'STOP before hashes/imports: actual owned terminal')
    require(raw.get('original_identity') is not None and
        raw.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
        raw.get('phase', phase) == phase, 'STOP: actual original identity/reap/birth absence differs')
    finite(raw.get('whole_GPU_observed_peak_MiB'), 'Actual whole GPU peak', True)
    require(raw['whole_GPU_observed_peak_MiB'] <= 8192, 'STOP: whole GPU peak exceeds 8192MiB')


def source_gate(spec_path, spec):
    # SPEC metadata is necessary to locate raw identities; no input is hashed
    # and no SOURCE/payload is imported or read until all nine raws pass.
    receipt_path = spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    raw = {'current': read(receipt_path)}
    terminal(raw['current'], spec['phase'])
    require(set(spec['training_receiptmap']) == set(spec['training_phasemap']) == set(LEGACY_MODES),
        'All four original raw identities required')
    for mode in LEGACY_MODES:
        require(Path(spec['training_receiptmap'][mode]['path']).is_absolute(), 'Absolute original raw path required')
        raw[mode] = read(spec['training_receiptmap'][mode]['path'])
        terminal(raw[mode], spec['training_phasemap'][mode])
    for label, receipt_key, phase_key in (('native48_restoration', 'native48_restoration_receipt', 'native48_restoration_phase'),
            ('native48', 'native48_receipt', 'native48_phase'), ('restoration', 'restoration_receipt', 'restoration_phase'),
            ('native_phase', 'native_phase_receipt', 'native_phase_phase')):
        require(Path(spec[receipt_key]['path']).is_absolute(), 'Absolute new native raw path required')
        raw[label] = read(spec[receipt_key]['path'])
        terminal(raw[label], spec[phase_key])
    require(Path(spec['GPU_admission_file']).is_absolute(), 'Absolute current admission required')
    admission = read(spec['GPU_admission_file'])
    source, reader = reference(Path(__file__).with_name('evaluate.py')), reference(__file__)
    spec_ref, admission_ref, receipt_ref = reference(spec_path), reference(spec['GPU_admission_file']), reference(receipt_path)
    owner = reference(spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py')
    require(source['bytes'] == PRODUCER_BYTES and source['sha256'] == PRODUCER_SHA and owner['sha256'] == OWNER_SHA and
        raw['current']['admission_sha256'] == admission_ref['sha256'], 'Exact current producer/raw/admission/owner join differs')
    expect(admission, dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE', automatic_retry=False,
        phase=spec['phase'], allowed_cpus=[0, 1], cuda_visible_devices='0',
        qualified_leaf_deadline_seconds=spec['deadline_seconds'], whole_GPU_used_limit_MiB=8192,
        entry_args=['--spec', str(spec_path), '--owner-execute'], driver_sha256=OWNER_SHA), 'Current owned admission')
    require(Path(admission['entry']).resolve(strict=True) == Path(source['path']) and
        spec['entry_source'] == source and spec['CPU_reader'] == reader and
        source in spec['sources'] and reader in spec['sources'] and
        len({ref['path'] for ref in spec['sources']}) == len(spec['sources']), 'Actual producer/reader closure differs')
    hashes = admission['bound_source_sha256']
    require(hashes.get(str(spec_path)) == spec_ref['sha256'], 'Current SPEC is not owner-bound')
    for ref in spec['sources']:
        bound(ref)
        require(hashes.get(ref['path']) == ref['sha256'], 'Complete current closure is not owner-bound')
    for role in REQUIRED_SOURCE_ROLES:
        require(spec[role] in spec['sources'] and hashes.get(spec[role]['path']) == spec[role]['sha256'],
            'Required DEV/native/TECH SOURCE or payload role must be current GPU owner-bound: ' + role)
        bound(spec[role])
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Current admitted scientific SPEC differs')
    proposal = read_ref(spec['SOURCE_proposal'])
    expect(proposal, dict(kind=PROPOSAL_KIND, schema=SCHEMA, producer_SOURCE=source,
        endpoint=ENDPOINT, completed_phase_epoch=48, effective_native_epoch=96, images=187, raw_GT=380,
        predictions=18700, evaluation_report_schema=REPORT_SCHEMA, evaluation_report_status=REPORT_STATUS,
        evaluation_CPU_schema=QUAL_SCHEMA, evaluation_CPU_status=QUAL_STATUS, CPU_numeric=CPU_NUMERIC), 'Canonical SOURCE proposal')
    require(spec['SOURCE_proposal'] in spec['sources'] and spec['DEV'] == proposal['canonical_DEV'] and
        proposal['required_source_roles'] == list(REQUIRED_SOURCE_ROLES), 'Canonical DEV/proposal differs')
    for role, ref in dict(proposal['immutable_SOURCE_refs'], **proposal['library_SOURCE_refs']).items():
        require(spec[role] == ref and ref in spec['sources'], 'Canonical SOURCE role differs: ' + role)
    for role in ('DEV', 'native_phase_spec', 'native_phase_qualification', 'native_phase_receipt', 'native_phase_checkpoint',
                 'restoration_receipt'):
        require(spec[role] in spec['sources'], 'Current scientific lineage role must be owner-bound: ' + role)
        bound(spec[role])
    return dict(source=source, reader=reader, spec=spec_ref, admission=admission_ref,
        receipt=receipt_ref, owner=owner, raw=raw, proposal=proposal)


def native_lineage(spec, refs):
    # Reuse verified provenance-only checks; neither function reconstructs
    # native CUDA matching/gradients or loads weights. CPU tensor checks below
    # use the original independent reader, not the producer's prepare_model.
    producer = load_source(refs['source'], 'bound_nativeE96_provenance_ONLY')
    row = producer.qualified_native_phase(spec, refs['raw'])
    train, q = row['training_spec'], row['qualification']
    for role in ('native_phase_source', 'native_phase_CPU_reader', 'native_phase_SOURCE_proposal'):
        require(spec[role] in spec['sources'], 'New native SOURCE lineage absent: ' + role)
    native_reader = load_source(spec['native_phase_CPU_reader'], 'bound_native_phase_technical_provenance_ONLY')
    technical = native_reader.restoration_gate(train, dict(source=spec['native_phase_source'], raw=refs['raw']))
    require(technical is not None and q['technical_restoration_qualification'] == train['restoration_CPU_qualification'] and
        q['formal_rebuilds_qualified_E48_without_technical_states'] is True,
        'Actual technical gate/fresh original teacher rebuild differs')
    for ref in (train['restoration_CPU_qualification'], train['restoration_CPU_reader'], train['restoration_spec']):
        require(ref in spec['sources'], 'Preserved technical lineage must be owner-bound')
    legacy = load_source(spec['evaluation_source'], 'bound_original_native0_lineage_ONLY')
    original = legacy.qualified_training(spec)['native0']
    require(train['initializers'] == original['training_spec']['initializers'] and
        all(spec[k] == train[k] for k in ('image_root', 'native_helpers', 'initializer_source', 'micro_preflight_source',
            'training_source', 'training_CPU_reader', 'readout_source', 'geometry_source', 'hooks_source')),
        'Original/new native architecture/initializer/helper provenance differs')
    predecessor = native_reader.qualified_native48(train, refs['raw'])
    require(predecessor['qualification']['checkpoint'] == train['native48_checkpoint'],
        'Independent predecessor E48 native lineage differs')
    row['predecessor_qualification'] = predecessor['qualification']
    row['original_training_spec'], row['original_qualification'] = original['training_spec'], original['qualification']
    row['technical_qualification'] = technical
    return row


def report_check(spec, refs):
    output = Path(spec['output_directory'])
    require(output.is_absolute() and output.is_dir(), 'Absolute completed evaluation output required')
    report_ref, start_ref = reference(output / 'REPORT.json'), reference(output / 'START.json')
    report, start = read_ref(report_ref), read_ref(start_ref)
    common = dict(source=refs['source'], spec=refs['spec'], phase=spec['phase'], endpoint=ENDPOINT,
        native_phase_checkpoint=spec['native_phase_checkpoint'], native_phase_qualification=spec['native_phase_qualification'],
        native_CPU_tool_terminal=spec['native_CPU_tool_terminal'], no_training=True, no_GPU_AP=True, no_TEST=True,
        no_node_selection=True, no_score_threshold_selection=True)
    expect(start, dict(common, schema='native-finetune48-finalE96-evaluation-start-v1', completed_phase_epoch=48,
        effective_native_epoch=96, no_retry=True), 'Evaluation START')
    expect(report, dict(common, schema=REPORT_SCHEMA, status=REPORT_STATUS, GPU_admission=refs['admission'],
        SOURCE_proposal=spec['SOURCE_proposal'], error=None, no_automatic_retry=True,
        full_pixelwise_GPU_original_logit_replay_claimed=False, blind_DEV_claimed=False,
        historical_class_or_score_equality_assumed=False, matched_compute_claimed=False, matched_objective_claimed=False,
        efficacy_claimed=False, convergence_claimed=False, novelty_claimed=False, SOTA_claimed=False), 'Completed evaluation REPORT')
    expect(report['cleanup'], dict(allocated_after=0), 'Actual GPU cleanup')
    finite(report['cleanup']['allocated_before'], 'GPU allocation before cleanup', True)
    finite(report['elapsed_seconds'], 'GPU elapsed', True)
    require(report['elapsed_seconds'] <= spec['deadline_seconds'] and type(report['torch_peak_allocated_bytes']) is int and
        0 <= report['torch_peak_allocated_bytes'] <= 4096 * 1024 ** 2, 'GPU deadline/allocator observations differ')
    record = report['result']
    expect(record, dict(endpoint=ENDPOINT, checkpoint=spec['native_phase_checkpoint'], qualification=spec['native_phase_qualification'],
        completed_phase_epoch=48, effective_native_epoch=96, CP_qualified=True, images=187, raw_GT=380,
        predictions_count=18700, query_score_factors_count=18700, all100_fields_saved=True, final_field_packets=187,
        GPU_original_logits_saved=False, full_native610_state_exact=True, registered_adapter_tensors=0, params_gradNone=True,
        native_official_fusion=True, GT_removed_from_forward=True, loader_accesses_annotations=True,
        query_score_factors_verified=True, class_probability_mask_quality_product_TorchEqual_official_score=True,
        source_native_query_ID_mapping_verified=True, added_NMS=False, added_score_cutoff=False, query_filter=False,
        AP_computed_on_GPU=False, historical_class_or_score_equality_assumed=False), 'Fixed nativeE96 result')
    directory = output / ENDPOINT
    for role, filename in dict(image_rows='IMAGE_ROWS.json', predictions='ALL_PREDICTIONS.json',
                              query_score_factors='QUERY_SCORE_FACTORS.json', state_proof='STATE_PROOF.json').items():
        require(Path(bound(record[role])).resolve() == (directory / filename).resolve(strict=True), 'Endpoint artifact path differs: ' + role)
    return record, report_ref, start_ref


def full_state_check(torch, old, common, initializer, registry, revert, Config, spec, refs, lineage, record):
    proof = read_ref(record['state_proof'])
    expect(proof, dict(schema='native-finetune48-finalE96-evaluation-state-proof-v1', source=refs['source'], spec=refs['spec'],
        endpoint=ENDPOINT, full_native610_state_exact=True, params_gradNone=True, all_parameters_frozen=True,
        all_modules_eval=True, no_adapter_registered=True, optimizer_or_RNG_restored=False), 'Actual evaluation state proof')
    loading = proof['loading_proof']
    train, q, original_train = lineage['training_spec'], lineage['qualification'], lineage['original_training_spec']
    initialized = read_ref(original_train['initializer_qualification'])
    expect(initialized, dict(schema='ROOT-native-coco-comparator-initializer-CPU-inspection-v1',
        all_initializers_strictly_compatible=True, source=spec['initializer_source']), 'Original strict initializer qualification')
    construction = dict(original_train, backbone_with_cp=False)
    model, cold = common.build_initial_native(torch, old, initializer, registry, revert, Config,
        construction, initialized, loading['initialization'])
    policy = old.independent_policy(torch, model, 'mask2former')
    require(policy['trainable_tensors'] == 345 and policy['parameter_tensors'] == 451 and policy['state_entries'] == 610 and
        policy['frozen_tensors'] == 106 and len(policy['BN_buffer_names']) == 159 and len(policy['BN_modules']) == 53 and
        len(policy['backbone_conv_names']) == 53 and not any('readout' in n for n in model.state_dict()),
        'Independent bare native451/610 and typedBN106+159 boundary differs')
    common.cpu_state(torch, old, cold, loading['initial_native_state'], 'Independent strict initial ship native610')
    fixed = policy['frozen_names'] + policy['BN_buffer_names']
    # Complete original nativeE24 is independently joined, but active E96
    # weights are allowed to change. Only typed-BN state remains initializer-exact.
    teacher = torch.load(str(bound(spec['checkpointmap']['native0'])), map_location='cpu')
    require(set(teacher) == {'state_dict', 'optimizer_state_dict', 'external_rng_state', 'private_order_state', 'meta'},
        'Full qualified original E24 checkpoint required')
    original_q = lineage['original_qualification']
    expect(teacher['meta'], dict(schema=common.CP_SCHEMA, kind='mask2former', mode='native0',
        source=spec['training_source'], spec=original_q['spec'], phase=original_q['phase'],
        training_protocol=original_q['training_protocol'], completed_epoch=24, global_microsteps=17784,
        global_updates=8892, consumed_raw_GT=39864, no_pending_accumulation_gradients=True), 'Original E24 teacher metadata')
    teacher_proof = read_ref(original_q['model_proof']['proof_refs']['state_proof'])
    common.cpu_state(torch, old, teacher['state_dict'], teacher_proof['after'], 'Qualified original E24 full CPU610')
    require(set(teacher['state_dict']) == set(cold) and all(teacher['state_dict'][n].shape == cold[n].shape and
        teacher['state_dict'][n].dtype == cold[n].dtype for n in cold) and
        all(torch.equal(teacher['state_dict'][n], cold[n]) for n in fixed), 'Original E24 typedBN/shape mapping differs')
    del teacher, teacher_proof
    gc.collect()
    cp = torch.load(str(bound(spec['native_phase_checkpoint'])), map_location='cpu')
    require(set(cp) == {'state_dict', 'optimizer_state_dict', 'external_rng_state', 'private_order_state', 'meta'},
        'Complete NEW native phase48 checkpoint required')
    meta, state = cp['meta'], cp['state_dict']
    expect(meta, dict(schema=CP_SCHEMA, kind='mask2former', mode='native0', execution_kind='formal_additional48',
        source=q['source'], spec=q['spec'], phase=q['phase'], teacher_checkpoint=q['teacher_checkpoint'], teacher_epoch=48,
        phase_initial_epoch=0, completed_phase_epoch=48, effective_native_epoch=96, phase_microsteps=35568,
        phase_updates=17784, phase_raw_GT_visits=79728, effective_native_microsteps=71136,
        effective_native_updates=35568, effective_native_raw_GT_visits=159456, phase_recipe=train['initializers']['mask2former'],
        schedule=train['schedule'], fresh_optimizer_at_phase_start=True, old_optimizer_restored=False, old_RNG_restored=False,
        no_pending_accumulation_gradients=True, formal_resume_eligible=False), 'NEW native phase48/E96 metadata')
    require(isinstance(state, dict) and set(state) == set(cold) and len(state) == 610 and all(
        isinstance(v, torch.Tensor) and v.device.type == 'cpu' and v.shape == cold[n].shape and v.dtype == cold[n].dtype and
        bool(torch.isfinite(v).all()) for n, v in state.items()), 'NEW full610 native keys/types/finite differ')
    node = read_ref(q['nodes']['48']['state_proof'])
    expect(node, dict(schema='native-finetune48-E48warm-weights-fresh-optimizer-node-state-proof-v1', source=q['source'], spec=q['spec'],
        completed_phase_epoch=48, effective_native_epoch=96, phase_microsteps=35568, phase_updates=17784,
        phase_raw_GT_visits=79728, frozen_BN_all_exact=True, CPU_checkpoint_readback_exact=True,
        no_pending_accumulation_gradients=True, formal_resume_eligible=False), 'Qualified phase48 node proof')
    for label, state_proof in (('Qualified nativeE96 full610', node['after']), ('Evaluation loaded nativeE96', loading['trained_fullstate']),
                               ('Evaluation nativeE96 before', proof['before']), ('Evaluation nativeE96 after', proof['after'])):
        common.cpu_state(torch, old, state, state_proof, label)
    require(proof['before'] == proof['after'] == loading['trained_fullstate'] == node['after'] and
        all(torch.equal(state[n], cold[n]) for n in fixed), 'Evaluation610/E96 and original typedBN exact state differ')
    bn_params, bn_buffers = [], []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            bn_params.extend(name + '.' + local for local, _ in module.named_parameters(recurse=False))
            bn_buffers.extend(name + '.' + local for local, _ in module.named_buffers(recurse=False))
    expect(loading, dict(schema='native-finetune48-finalE96-loading-proof-v1', checkpoint=spec['native_phase_checkpoint'],
        qualification=spec['native_phase_qualification'], endpoint=ENDPOINT, strict_fullstate_loaded=True,
        native_state_tensors=610, registered_parameters=451, registered_adapter_tensors=0,
        BN_parameter_names=bn_params, BN_buffer_names=bn_buffers, original_frozen_BN_exact=True,
        completed_phase_epoch=48, effective_native_epoch=96, inference_backbone_with_cp=False,
        training_backbone_with_cp=train['backbone_with_cp'], optimizer_or_RNG_restored_for_inference=False,
        checkpoint_optimizer_and_RNG_containers_deserialized=True, no_random_missing_keys=True), 'Actual inference loading proof')
    loaded = model.load_state_dict(state, strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys and all(torch.equal(v.detach().cpu(), state[n])
        for n, v in model.state_dict().items()), 'Independent strict full E96 model load differs')
    model.requires_grad_(False).eval()
    require(all(not p.requires_grad and p.grad is None for p in model.parameters()) and all(not m.training for m in model.modules()) and
        model.backbone.with_cp is False, 'CPU inference model eval/frozen/noncheckpointed boundary differs')
    head, fusion = model.panoptic_head, model.panoptic_fusion_head
    require(head.num_queries == 100 and head.num_classes == 1 and fusion.num_classes == fusion.num_things_classes == 1 and
        fusion.num_stuff_classes == 0 and fusion.test_cfg.get('max_per_image', 100) == 100 and
        fusion.test_cfg.get('instance_on', False) is True and fusion.test_cfg.get('panoptic_on', True) is False and
        fusion.test_cfg.get('semantic_on', False) is False, 'Original all100 single-class fusion configuration differs')
    for role, obj in dict(native_mask_head_SOURCE=type(head), native_parent_head_SOURCE=type(head).__mro__[1],
            native_fusion_SOURCE=type(fusion), interpolation_SOURCE=torch.nn.functional.interpolate).items():
        require(Path(inspect.getsourcefile(obj)).resolve() == bound(spec[role]).resolve(), 'Actual inference SOURCE differs: ' + role)
    result = dict(checkpoint=spec['native_phase_checkpoint'], state_proof=record['state_proof'], node_state_proof=q['nodes']['48']['state_proof'],
        full_native610_CPU_exact=True, registered_parameters=451, registered_adapter_tensors=0, native345_architecture_verified=True,
        typed_BN106_affine159_buffers_original_exact=True, evaluation_before_after610_exact_to_E96=True,
        original_E24_active_weights_equality_required=False, strict_original_COCO_boat8_null80_mapping_verified=True,
        all_parameters_frozen=True, all_modules_eval=True, optimizer_or_RNG_restored=False, CPU_native_forward_replayed=False)
    del model, cp, state, cold, node, initialized, proof, loading
    gc.collect()
    return result


def image_population(spec, refs, record, coco):
    ids = list(coco.imgs)
    rows = read_ref(record['image_rows'])
    require(isinstance(rows, list) and len(ids) == len(rows) == 187 and len(coco.anns) == 380 and
        [r['image_id'] for r in rows] == ids and set(coco.cats) == {1}, 'Complete ordered original DEV187/380 population differs')
    previous_rng, packet_paths = None, set()
    for index, row in enumerate(rows):
        image = coco.imgs[ids[index]]
        h, w = image['height'], image['width']
        require(type(h) is int and type(w) is int and h > 0 and w > 0, 'Positive original image dimensions required')
        scale = 800. / max(h, w)
        actual = [int(h * scale + .5), int(w * scale + .5)]
        anns = coco.imgToAnns[ids[index]]
        expect(row, dict(endpoint=ENDPOINT, dataset_index=index, image_id=ids[index],
            ann_ids=[ann['id'] for ann in anns], annotation_count=len(anns), original_shape=[h, w], actual_img_shape=actual,
            batch_input_shape=[800, 800], scale_factor=[actual[1] / w, actual[0] / h], checkpoint=spec['native_phase_checkpoint'],
            qualification=spec['native_phase_qualification'], CP_qualified=True, GT_removed_from_forward=True,
            loader_accesses_annotations=True, matching_or_loss_executed=False, head_seed=100000 + ids[index],
            external_RNG_restored=True, predictions=100, official_instance_postprocess=True,
            source_native_query_ID_mapping_verified=True, official_masks_query_ID_bitwise_equal=True,
            official_scores_query_ID_bitwise_equal=True, query_score_factors_verified=True,
            class_probability_mask_quality_product_TorchEqual_official_score=True, added_NMS=False,
            added_score_cutoff=False, query_filter=False, all100_fields_saved=True, GPU_original_logits_saved=False,
            historical_class_or_score_equality_assumed=False), 'Actual DEV image row')
        require(Path(row['image']['path']).resolve() == (Path(spec['image_root']) / image['file_name']).resolve(),
            'Original image path differs')
        bound(row['image'])
        require(row['external_RNG_before'] == row['external_RNG_after'] and
            (previous_rng is None or previous_rng == row['external_RNG_before']), 'Actual external RNG restoration/continuity differs')
        rng = row['external_RNG_before']
        require(isinstance(rng, list) and len(rng) == 2, 'CPU/CUDA RNG digest pair required')
        for digest in rng:
            require(isinstance(digest.get('shape'), list) and len(digest['shape']) == 1 and type(digest['shape'][0]) is int and
                digest['shape'][0] > 0, 'Actual RNG length differs')
            digest_contract(digest, digest['shape'], 'uint8')
        previous_rng = row['external_RNG_after']
        queries = row['prediction_query_ids']
        require(isinstance(queries, list) and len(queries) == 100 and all(type(q) is int for q in queries) and
            set(queries) == set(range(100)), 'Every original query ordinal must appear exactly once')
        require(set(row['final_fields']) == {'classes', 'masks'}, 'Final field digest names differ')
        digest_contract(row['final_fields']['classes'], [1, 100, 2], 'float32')
        digest_contract(row['final_fields']['masks'], [1, 100, 200, 200], 'float32')
        packet = bound(row['final_field_packet'])
        require(packet.resolve() == (Path(spec['output_directory']) / ENDPOINT / 'FINAL_FIELDS' / ('IMAGE_%04d.pth' % index)).resolve(strict=True) and
            str(packet) not in packet_paths, 'Unique canonical final packet path required')
        packet_paths.add(str(packet))
    require(sum(len(r['ann_ids']) for r in rows) == 380 and len(packet_paths) == 187, 'RawGT/final packet coverage differs')
    return ids, rows


def render_original(torch, field, actual_shape, original_shape):
    # Independent expression of the bound official two-stage renderer, no
    # producer renderer/forward call. Batch query chunks limit CPU memory.
    require(field.device.type == 'cpu' and field.dtype == torch.float32 and field.ndim == 4 and
        field.shape[0] == 1 and tuple(field.shape[-2:]) == (200, 200), 'CPU FP32 M200 renderer input required')
    ih, iw = actual_shape
    padded = torch.nn.functional.interpolate(field, size=(800, 800), mode='bilinear', align_corners=False)
    cropped = padded[0, :, :ih, :iw].unsqueeze(1)
    original = torch.nn.functional.interpolate(cropped, size=tuple(original_shape), mode='bilinear', align_corners=False)[:, 0]
    require(bool(torch.isfinite(original).all()), 'Nonfinite independently rendered logits')
    return original


def field_replay(torch, np, masks, spec, refs, record, rows, grouped, started):
    factors = read_ref(record['query_score_factors'])
    factor_map = {(r['image_id'], r['query_id']): r for r in factors}
    require(len(factor_map) == len(factors) == 18700, 'Unique full score-factor population required')
    per_image, total_flips, changed_queries, pixels, max_class_error, max_quality_error = [], 0, 0, 0, 0., 0.
    packet_keys = {'schema', 'source', 'spec', 'checkpoint', 'endpoint', 'dataset_index', 'image_id', 'original_shape',
        'actual_img_shape', 'final_class_logits', 'final_mask_logits200', 'final_field_digests', 'prediction_query_ids',
        'GPU_original_logits_saved', 'all100_fields_saved'}
    with torch.no_grad():
        for row in rows:
            require(time.monotonic() - started <= spec['deadline_seconds'], 'CPU qualification deadline exceeded')
            packet = torch.load(str(bound(row['final_field_packet'])), map_location='cpu')
            require(isinstance(packet, dict) and set(packet) == packet_keys, 'Exact complete final100 packet schema required')
            expect(packet, dict(schema='native-finetune48-finalE96-final-fields-v1', source=refs['source'], spec=refs['spec'],
                checkpoint=spec['native_phase_checkpoint'], endpoint=ENDPOINT, dataset_index=row['dataset_index'], image_id=row['image_id'],
                original_shape=row['original_shape'], actual_img_shape=row['actual_img_shape'], final_field_digests=row['final_fields'],
                prediction_query_ids=row['prediction_query_ids'], GPU_original_logits_saved=False, all100_fields_saved=True), 'Final100 field packet')
            classes, fields = packet['final_class_logits'], packet['final_mask_logits200']
            for value, shape, name in ((classes, (1, 100, 2), 'classes'), (fields, (1, 100, 200, 200), 'masks')):
                require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.float32 and
                    tuple(value.shape) == shape and not value.requires_grad and bool(torch.isfinite(value).all()) and
                    tensor_digest(torch, value) == row['final_fields'][name], 'Finite exact exported final fields differ: ' + name)
            probabilities = classes[0].softmax(dim=-1)[:, 0]
            population = {p['query_id']: p for p in grouped[row['image_id']]}
            image_flips, image_changed, image_class_error, image_quality_error, max_flipped_abs_logit = 0, 0, 0., 0., 0.
            for start in range(0, 100, 10):
                logits = render_original(torch, fields[:, start:start + 10], row['actual_img_shape'], row['original_shape'])
                for local, query in enumerate(range(start, min(start + 10, 100))):
                    p, factor = population[query], factor_map[(row['image_id'], query)]
                    gpu_binary_np = masks.decode(dict(size=p['segmentation']['size'], counts=p['segmentation']['counts'].encode('ascii')))
                    require(gpu_binary_np.dtype == np.uint8 and bool(((gpu_binary_np == 0) | (gpu_binary_np == 1)).all()),
                        'Exported GPU RLE must decode to binary uint8')
                    gpu_binary = torch.from_numpy(np.ascontiguousarray(gpu_binary_np)).to(dtype=torch.bool)
                    cpu_logit = logits[local]
                    require(tuple(gpu_binary.shape) == tuple(cpu_logit.shape), 'GPU RLE/CPU original renderer shapes differ')
                    flip = (cpu_logit > 0) != gpu_binary
                    count = int(flip.sum())
                    pixels += gpu_binary.numel()
                    if count:
                        envelope = 1e-4 + 1e-4 * cpu_logit.abs()
                        require(bool((cpu_logit.abs()[flip] <= envelope[flip]).all()),
                            'STOP: GPU RLE/CPU threshold disagreement outside prelocked near-zero envelope: image %s query %s' % (row['image_id'], query))
                        max_flipped_abs_logit = max(max_flipped_abs_logit, float(cpu_logit.abs()[flip].max()))
                    support = gpu_binary.to(dtype=torch.float32)
                    quality = (cpu_logit.sigmoid() * support).sum() / (support.sum() + 1e-6)
                    expected_class = torch.tensor(factor['class_probability'], dtype=torch.float32)
                    expected_quality = torch.tensor(factor['mask_quality'], dtype=torch.float32)
                    require(bool(torch.isclose(probabilities[query], expected_class, rtol=1e-6, atol=1e-6)),
                        'Independent final class softmax differs from actual factor')
                    require(bool(torch.isclose(quality, expected_quality, rtol=1e-4, atol=1e-4)),
                        'CPU rendered quality on exported GPU support differs from actual factor')
                    image_class_error = max(image_class_error, float((probabilities[query] - expected_class).abs()))
                    image_quality_error = max(image_quality_error, float((quality - expected_quality).abs()))
                    image_flips += count
                    image_changed += int(count > 0)
                    del gpu_binary_np, gpu_binary, cpu_logit, flip, support, quality, expected_class, expected_quality
                del logits
            total_flips += image_flips
            changed_queries += image_changed
            max_class_error = max(max_class_error, image_class_error)
            max_quality_error = max(max_quality_error, image_quality_error)
            per_image.append(dict(image_id=row['image_id'], dataset_index=row['dataset_index'], packet=row['final_field_packet'],
                queries=100, binary_threshold_disagreements=image_flips, queries_with_disagreement=image_changed,
                maximum_abs_CPU_logit_at_disagreement=max_flipped_abs_logit,
                max_abs_class_probability_error=image_class_error, max_abs_mask_quality_error=image_quality_error))
            del packet, classes, fields, probabilities, population
            if (len(per_image) % 50 == 0) or len(per_image) == 187:
                print(json.dumps(dict(CPU_rendered_images=len(per_image), endpoint=ENDPOINT,
                    threshold_disagreements=total_flips, elapsed_seconds=time.monotonic() - started)), flush=True)
    return dict(images=187, queries=18700, final_packets=187, CPU_fields_digest_exact=True,
        all100_original_query_ordinals_independently_checked=True, CPU_FP32_class_softmax_checked=True,
        CPU_FP32_two_stage_bilinear_renderer_checked=True, CPU_quality_on_exported_GPU_RLE_support_independently_checked=True,
        numeric_protocol=CPU_NUMERIC, total_original_query_pixels=pixels, binary_threshold_disagreements=total_flips,
        queries_with_disagreement=changed_queries, all_disagreements_within_CPU_nearzero_envelope=True,
        max_abs_class_probability_error=max_class_error, max_abs_mask_quality_error=max_quality_error, per_image=per_image,
        GPU_original_logits_saved=False, GPU_original_logits_full_allclose_measured=False,
        certified_GPU_interpolation_error_bound=False, CPU_native_forward_replayed=False,
        original_GPU_TorchEqual_fusion_checks_SOURCE_observed=True, cross_device_bitwise_renderer_equality_claimed=False)


def geometry_rows(ids, grouped, coco, np, masks):
    # Preserve the original diagnostic and export-row tie convention, with an
    # explicit all-zero raster for a raw empty segmentation. No raw GT filter.
    rows, empty_predictions, raw_annotations = [], 0, []
    for image_id in ids:
        population = grouped[image_id]
        binary = np.stack([masks.decode(dict(size=p['segmentation']['size'], counts=p['segmentation']['counts'].encode('ascii'))).astype(bool)
                           for p in population], axis=0)
        areas = binary.sum(axis=(1, 2), dtype=np.int64)
        empty_predictions += int((areas == 0).sum())
        for ann in coco.imgToAnns[image_id]:
            raw_annotations.append(ann)
            require(ann['image_id'] == image_id and ann['category_id'] == 1, 'Original raw GT image/category differs')
            gt = np.zeros(binary.shape[1:], dtype=np.uint8) if ann.get('segmentation') == [] else coco.annToMask(ann)
            require(gt.shape == binary.shape[1:] and bool(((gt == 0) | (gt == 1)).all()), 'Original GT raster geometry/binary differs')
            gt = gt.astype(bool)
            area = int(gt.sum(dtype=np.int64))
            inter = (binary & gt[None]).sum(axis=(1, 2), dtype=np.int64)
            union = areas + area - inter
            iou = np.divide(inter, union, out=np.ones(100, dtype=np.float64), where=union > 0)
            best = int(np.argmax(iou))
            tp, fp, fn = int(inter[best]), int(areas[best] - inter[best]), int(area - inter[best])
            rows.append(dict(image_id=image_id, annotation_id=ann['id'], category_id=ann['category_id'],
                iscrowd=int(ann.get('iscrowd', 0)), ignore=int(ann.get('ignore', 0)), original_shape=list(gt.shape), GT_pixels=area,
                empty_GT_raster=area == 0, best_export_row=best, best_query_id=population[best]['query_id'],
                best_official_score=population[best]['score'], max_mask_IoU=float(iou[best]), TP=tp, FP=fp, FN=fn,
                FN_over_GT_area=fn / max(area, 1), FP_over_GT_area=fp / max(area, 1)))
        del binary, areas
    require(len(rows) == len(raw_annotations) == len(coco.anns) == 380 and
        len({r['annotation_id'] for r in rows}) == 380 and {r['annotation_id'] for r in rows} == set(coco.anns),
        'Every original raw GT annotation must be retained exactly once')
    annotation_sha = hashlib.sha256(json.dumps(raw_annotations, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    return rows, empty_predictions, annotation_sha


def run(spec_file, acceptance_file):
    started = time.monotonic()
    spec_path, target = Path(spec_file), Path(acceptance_file)
    require(spec_path.is_absolute() and target.is_absolute() and not target.exists() and target.parent.is_dir(),
        'Absolute SPEC/fresh absolute acceptance in existing directory required')
    spec_path = spec_path.resolve(strict=True)
    spec = read(spec_path)
    expect(spec, dict(schema=SCHEMA, kind='mask2former', endpoint=ENDPOINT, completed_phase_epoch=48,
        effective_native_epoch=96, CPU_affinity=[0, 1], allocator_limit_MiB=4096, whole_GPU_owner_limit_MiB=8192), 'Fixed finalE96 SPEC')
    require(type(spec['deadline_seconds']) is int and 0 < spec['deadline_seconds'] <= 10800 and
        os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU01/CUDA-empty/fixed deadline required')
    refs = source_gate(spec_path, spec)
    lineage = native_lineage(spec, refs)
    record, report_ref, start_ref = report_check(spec, refs)
    import psutil
    import torch
    import numpy as np
    from pycocotools import mask as masks
    from pycocotools.cocoeval import COCOeval
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'ROOT CPU01/coldCUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 2 and torch.__version__ == '2.0.1+cu118' and
        (Path(torch.__file__).resolve().parent / 'version.py').resolve() == bound(spec['torch_version_source']).resolve(), 'Bound Torch2.0.1/threads2+2 required')
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    require(len(CocoDataset.METAINFO['classes']) == 80 and CocoDataset.METAINFO['classes'][8] == 'boat', 'Official COCO boat8/null80 mapping required')
    old = load_source(spec['reused_CPU_reader'], 'bound_original_independent_CPU_state_reader')
    common = load_source(spec['training_CPU_reader'], 'bound_original_independent_COCO_model_constructor')
    initializer = load_source(spec['initializer_source'], 'bound_original_strict_initializer')
    official = load_source(spec['official_evaluation_CPU_reader'], 'bound_original_generic_RLE_score_COCO_helpers')
    state_check = full_state_check(torch, old, common, initializer, MODELS, revert_sync_batchnorm, Config,
        spec, refs, lineage, record)
    coco = COCO(str(bound(spec['DEV'])))
    dataset = CocoDataset(ann_file=str(bound(spec['DEV'])), data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
    require(len(dataset) == len(coco.imgs) == 187 and len(coco.anns) == 380 and
        [int(dataset.get_data_info(i)['img_id']) for i in range(187)] == list(coco.imgs), 'Canonical ordered DEV187/380 differs')
    ids, rows = image_population(spec, refs, record, coco)
    predictions, grouped = official.prediction_population(record, ids, coco, np, masks, rows)
    score_proof = official.score_factors(record, predictions, rows, np)
    # The old arithmetic helper did not possess logits. Keep that exact proof,
    # label its scope, and attach the new independent scalar renderer check.
    score_proof['scope'] = 'Original factor/product arithmetic helper only; independent M200 renderer quality check is in field_replay'
    replay = field_replay(torch, np, masks, spec, refs, record, rows, grouped, started)
    geometry, empty_predictions, raw_ann_sha = geometry_rows(ids, grouped, coco, np, masks)
    metrics = official.coco_metrics(coco, predictions, ids, COCOeval)
    summary = official.geometry_summary(geometry)
    for ref in spec['sources']:
        bound(ref)
    for ref in (report_ref, start_ref, refs['receipt'], record['image_rows'], record['predictions'], record['query_score_factors'], record['state_proof']):
        bound(ref)
    for row in rows:
        bound(row['final_field_packet'])
    require(time.monotonic() - started <= spec['deadline_seconds'] and not torch.cuda.is_initialized() and
        psutil.Process().cpu_affinity() == [0, 1] and torch.get_num_threads() == torch.get_num_interop_threads() == 2,
        'CPU deadline/affinity/threads/noCUDA final proof differs')
    acceptance = dict(schema=QUAL_SCHEMA, status=QUAL_STATUS, phase=spec['phase'], endpoint=ENDPOINT,
        source=refs['source'], reader_source=refs['reader'], spec=refs['spec'], SOURCE_proposal=spec['SOURCE_proposal'],
        report=report_ref, start=start_ref, parent_receipt=refs['receipt'], GPU_admission=refs['admission'], owner_driver=refs['owner'],
        native_phase_source=spec['native_phase_source'], native_phase_CPU_reader=spec['native_phase_CPU_reader'],
        native_phase_spec=spec['native_phase_spec'], native_phase_qualification=spec['native_phase_qualification'],
        native_phase_receipt=spec['native_phase_receipt'], native_CPU_tool_terminal=spec['native_CPU_tool_terminal'],
        checkpoint=spec['native_phase_checkpoint'], completed_phase_epoch=48, effective_native_epoch=96,
        technical_restoration_qualification=lineage['training_spec']['restoration_CPU_qualification'],
        original_teacher_checkpoint=spec['checkpointmap']['native0'], original_teacher_qualification=spec['qualificationmap']['native0'],
        actual_current_and_eight_prior_raw_terminals_passed_before_hashes_imports=True,
        whole_admitted_source_closure_bound_before_imports=True, sources=spec['sources'],
        new_formal_nativeE96_qualification_required=True, DEV_current_GPU_owner_bound=True,
        predecessor_nativeE48_qualification=lineage['training_spec']['native48_CPU_qualification'],
        phase48_technical_restoration_qualification=lineage['training_spec']['restoration_CPU_qualification'], fixed_phase48_only=True, intermediate_nodes_evaluated=False,
        state_check=state_check, images=187, raw_GT=380, predictions_count=18700,
        image_rows=record['image_rows'], predictions=record['predictions'], query_score_factors=record['query_score_factors'],
        state_proof=record['state_proof'], score_factor_proof=score_proof, field_replay=replay,
        COCO=metrics, GT_best100=dict(summary=summary, rows=geometry, empty_predictions=empty_predictions,
            raw_annotation_order_sha256=raw_ann_sha, tie_rule='first actual export row; actual native query ID recorded',
            all_original_annotation_IDs_retained=True, crowd_ignore_empty_GT_retained=True),
        AP_computed=True, AP_computed_on_GPU=False, independent_CPU_metric_reconstruction=True,
        CPU_execution=dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2, torch_interop_threads=2,
            cuda_visible_devices='', CUDA_initialized=False), elapsed_seconds=time.monotonic() - started,
        training_executed=False, native_forward_replayed=False, optimizer_or_RNG_restored=False,
        blind_DEV_claimed=False, no_TEST=True, no_node_selection=True, no_score_threshold_selection=True,
        historical_class_or_score_equality_assumed=False, efficacy_claimed=False, convergence_claimed=False,
        novelty_claimed=False, SOTA_claimed=False, matched_compute_claimed=False, matched_objective_claimed=False,
        evidence_limits=[
            'ROOT actual native CPU tool-terminal0 is prelocked metadata; this consumer does not independently OS-reap that CPU process.',
            'Evaluation native610 before/after equals the NEW trained E96 checkpoint; only typedBN106+159 is original-initializer-exact.',
            'Final class/M200 packet digests and CPU renderer/softmax/scalar quality are independently checked; native CUDA forward and training gradients are not replayed.',
            'All threshold flips are counted and accepted only within the prelocked practical CPU near-zero envelope. Original GPU raster logits were not saved: no measured full pixelwise allclose, certified GPU error bound or cross-device bitwise renderer equality.',
            'Independent mask-quality uses CPU rendered sigmoid on the EXPORTED GPU RLE support. The older score-factor helper omission flag concerns its arithmetic-only scope; new quality evidence is field_replay.',
            'Official GPU fusion masks/scores TorchEqual and GT-free-forward/RNG proofs remain SOURCE-observed; CPU all100 RLE and COCO12 checks use the actual exported predictions.',
            'GT-best100 is score-free reusable per-GT pairing with first-export-row ties, includes crowd/ignore/empty rasters, and is not AP, one-to-one recall or a deployable GT oracle.',
            'DEV187/380 is historically exposed, not blind. Only the prelocked final phase48/nativeE96 endpoint is evaluated; no efficacy, convergence, novelty, SOTA or matched-compute/objective claim follows.'
        ])
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(acceptance, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=QUAL_STATUS, acceptance=str(target), endpoint=ENDPOINT,
        images=187, predictions=18700, AP=metrics['stats'][0], threshold_disagreements=replay['binary_threshold_disagreements'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--acceptance', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'SOURCE_ONLY: explicit ROOT-owned CPU execution required')
    run(args.spec, args.acceptance)


if __name__ == '__main__':
    main()
