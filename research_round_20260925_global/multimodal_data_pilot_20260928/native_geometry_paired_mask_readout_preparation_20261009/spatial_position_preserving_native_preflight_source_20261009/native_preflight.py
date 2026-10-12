"""Inert ROOT-owned spatial final-mask technical preflight, no method result.

SOURCE only: three cold modes share cached backbone/native baseline; three
fresh COCO models train FIT1,2,3,5 independently (4micro/2updates). All CPU
models are built before CUDA. Native30 only/noSobel/noaux/AP/formal resume.
Explicit readout9802/geometry17500/bridge21688/reused CPU46683 source roles.
Only whole-head final mask is replaced; output observer registers AFTER that
replacement hook. Small geometry packets are observations, not CPU M0/sign
replay. The new owner admission and --owner-execute are required. No SCI
import or numerical execution occurred during SOURCE creation.
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


SCHEMA = 'ROOT-native-spatial-position-preserving-native-bridge-preflight-spec-v1'
REPORT_SCHEMA = 'native-spatial-position-preserving-native-bridge-preflight-report-v1'
MODES = ('global', 'spatial_halves', 'spatial_reversed')
TRAINING_MODES = MODES
READOUT_BYTES = 9802
READOUT_SHA = '7da9561ed30675fd0d692ce2ad883e0c8cccfddfd9f7913761da0fe4fff0a09d'
GEOMETRY_BYTES = 17500
GEOMETRY_SHA = '7ad19e356342afc15a2a41f1358634b23486c19021e63ccd94cfed03f6841d15'
BRIDGE_BYTES = 21688
BRIDGE_SHA = '23f36f3fab9b4a006bd0db65427b68223dc99c4929da458504ed5a3aa9185806'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
IMAGE_IDS = (1, 2, 3, 5)
ADAPTER_PREFIX = 'panoptic_head.spatial_mask_readout.'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def reference(path):
    path = Path(path).resolve(strict=True)
    value = path.read_bytes()
    return dict(path=str(path), bytes=len(value), sha256=hashlib.sha256(value).hexdigest())


def bound(row):
    require(isinstance(row, dict) and Path(row['path']).is_absolute(), 'Full absolute binding required')
    require(reference(row['path']) == row, 'Declared binding mismatch: ' + row['path'])
    return Path(row['path'])


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def load_source(row, name):
    spec = importlib.util.spec_from_file_location(name, str(bound(row)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def owner_source_gate(spec_path, spec):
    """No scientific hash/import before explicit current owned-admission metadata."""
    spec_path = Path(spec_path).resolve(strict=True)
    require(spec['phase'].startswith('native_spatial_position_preserving_native_bridge_preflight_') and
            Path(spec['GPU_admission_file']).is_absolute(), 'Fresh owned phase/admission required')
    admission_path = Path(spec['GPU_admission_file']).resolve(strict=True)
    admission = json.loads(admission_path.read_text(encoding='utf-8-sig'))
    entry, spec_ref = reference(__file__), reference(spec_path)
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            admission['automatic_retry'] is False and admission['phase'] == spec['phase'] and
            admission['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and
            admission['cuda_visible_devices'] == '0' and
            admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
            admission['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192 and
            Path(admission['entry']).resolve(strict=True) == Path(entry['path']) and
            admission['entry_args'] == ['--spec', str(spec_path), '--owner-execute'] and
            admission['driver_sha256'] == OWNER_SHA,
            'STOP before SCI: fresh explicit owner admission/entry/resources differ')
    require(spec['entry_source'] == entry and entry in spec['sources'] and spec['CPU_reader'] in spec['sources'] and
            admission['bound_source_sha256'].get(str(spec_path)) == spec_ref['sha256'],
            'Current entry/consumer/SPEC are not owner-bound')
    for source in spec['sources']:
        require(admission['bound_source_sha256'].get(source['path']) == source['sha256'],
                'Current preflight source closure absent from owner admission')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Owner scientific SPEC differs')


def equal_rng(torch, left, right):
    return len(left) == len(right) == 2 and all(torch.equal(a, b) for a, b in zip(left, right))


def image_data(old, model, dataset, coco, image_id):
    index = next(i for i in range(len(dataset)) if int(dataset.get_data_info(i)['img_id']) == image_id)
    data, samples, gray, raw, identity = old.load_image(model, dataset, coco, dataset.get_data_info(index))
    del gray, raw
    identity['dataset_index'] = index
    require(all(not any(key in sample for key in
                ('gt_instances', 'ignored_instances', 'gt_sem_seg', 'gt_panoptic_seg'))
                for sample in samples), 'GT was not entirely removed from inference samples')
    return data, samples, identity


def observe_adapter(torch, adapter, pref, records, geometry_packets=None):
    """Small actual geometry only; no dense F/e0/logit CPU export."""
    def observed(module, inputs, output):
        d=output['diagnostics']
        names=('eligible','inside','outside','centre_xy','used_theta','sigma_major','sigma_minor',
               'rho_mass','half_eligible_counts','global_fallback')
        packets,actual={},{}
        for name in names:
            value=d[name].detach().cpu().clone()
            actual[name]=value
            packets[name]=dict(pref.digest_tensor(value),values=value.tolist())
        if geometry_packets is not None:
            geometry_packets.append(actual)
        counts=d['eligible_counts']
        records.append(dict(mode=d['mode'],eligibility_frame=d['eligibility_frame'],
            all_queries=int(counts.numel()),queries_with_support=int((d['rho_mass']>1e-12).sum().cpu()),
            queries_with_eligible_pairs=int((counts>0).sum().cpu()),eligible_pairs_total=int(counts.sum().cpu()),
            eligible_pairs_max=int(counts.max().cpu()),
            canonical_eligible_pairs_total=int(d['canonical_eligible_counts'].sum().cpu()),
            geometry_packets=packets,valid_fraction=pref.digest_tensor(inputs[2]),
            input_F_shape=list(inputs[0].shape),input_e0_shape=list(inputs[1].shape),
            input_F_requires_grad=inputs[0].requires_grad,input_e0_requires_grad=inputs[1].requires_grad,
            mask_logits_requires_grad=output['mask_logits'].requires_grad,
            residual_requires_grad=output['logit_residual'].requires_grad,
            coefficient_fields_shape=list(output['coefficient_fields'].shape),
            all_queries_retained=True,GT_used=False,dense_feature_or_mask_exported=False))
        return None
    return adapter.register_forward_hook(observed)


def official_output(torch, model, classes, masks, samples, identity, pref):
    """Use native fusion.predict crop/resize and instance_postprocess, once."""
    fusion = model.panoptic_fusion_head
    require(fusion.num_classes == fusion.num_things_classes == 1 and
            fusion.test_cfg.get('instance_on') is True and
            fusion.test_cfg.get('panoptic_on') is False and
            not fusion.test_cfg.get('semantic_on', False) and fusion.test_cfg['max_per_image'] == 100,
            'Official single-class all100 instance-only fusion required')
    padded = torch.nn.functional.interpolate(masks[-1], size=(800, 800),
                                             mode='bilinear', align_corners=False)
    instances = fusion.predict(classes[-1], padded, samples, rescale=True)[0]['ins_results']
    cls = classes[-1][0]
    ship = torch.nn.functional.softmax(cls, dim=-1)[:, :-1]
    scores, ids = ship.flatten(0, 1).topk(100, sorted=False)
    qids = ids // fusion.num_classes
    require(torch.equal(qids.sort().values, torch.arange(100, device=qids.device)) and
            torch.equal(scores, ship[qids, 0]) and len(instances) == 100,
            'Official all100 query mapping differs')
    ih, iw = identity['actual_img_shape']
    rendered = torch.nn.functional.interpolate(padded[:, :, :ih, :iw],
        size=tuple(identity['original_shape']), mode='bilinear', align_corners=False)[0]
    ordered = rendered[qids]
    binary = (ordered > 0).float()
    quality = (ordered.sigmoid() * binary).flatten(1).sum(1) / (binary.flatten(1).sum(1) + 1e-6)
    require(torch.equal(instances.masks, binary.bool()) and
            torch.equal(instances.scores, ship[qids, 0] * quality) and
            bool((instances.labels == 0).all()) and bool(torch.isfinite(instances.scores).all()),
            'Official binary masks/scores do not match native query IDs')
    packet = dict(query_ids=qids.detach().cpu().clone(), masks=instances.masks.detach().cpu().clone(),
                  scores=instances.scores.detach().cpu().clone(), labels=instances.labels.detach().cpu().clone())
    summary = {name: pref.digest_tensor(value) for name, value in packet.items()}
    summary.update(predictions=100, original_shape=identity['original_shape'],
                   official_fusion_predict=True, official_query_mapping_verified=True,
                   added_NMS=False, added_score_cutoff=False)
    return packet, summary


def inference_fixture(torch,pref,old,hooks,readout,geometry,spec,prepared,dataset,coco,output,started):
    model,native_initial,initialization=prepared
    model.to('cuda:0').requires_grad_(False).eval()
    head=model.panoptic_head
    require(len(model.state_dict())==610 and len(list(model.parameters()))==451,'Cold native610/451 required')
    before=pref.state_proof(native_initial)
    rows,shared_cold,canonical_geometry=[],None,None
    try:
        data,samples,identity=image_data(old,model,dataset,coco,1)
        metadata=[sample.metainfo for sample in samples]
        external=pref.rng_snapshot(torch)
        seed=100001
        with torch.random.fork_rng(devices=[0]):
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            with torch.no_grad():
                features=model.extract_feat(data['inputs'])
                torch.random.default_generator.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                baseline_private_before=pref.rng_snapshot(torch)
                # Bare native610 baseline; new applyFalse correctly requires registered612.
                baseline_classes,baseline_masks=head(features,samples)
                require(len(baseline_classes)==len(baseline_masks)==10,'Native baseline readouts differ')
                baseline_instances,baseline_summary=official_output(
                    torch,model,baseline_classes,baseline_masks,samples,identity,pref)
                baseline_private_after=pref.rng_snapshot(torch)
        require(equal_rng(torch,external,pref.rng_snapshot(torch)),'Baseline external RNG not restored')
        baseline_fields=dict(classes=[pref.digest_tensor(x) for x in baseline_classes],
                             masks=[pref.digest_tensor(x) for x in baseline_masks])
        for mode in MODES:
            pref.check_deadline(started,spec['deadline_seconds'])
            before_constructor_rng=pref.rng_snapshot(torch)
            adapter=hooks.register_spatial_mask_readout(torch,readout,geometry,model,mode,
                spec['readout_source'],spec['geometry_source'])
            repeated=readout.build_adapter(torch,geometry,mode=mode)
            cold=pref.cpu_tree(torch,adapter.state_dict())
            require(pref.tree_equal(torch,cold,pref.cpu_tree(torch,repeated.state_dict())) and
                equal_rng(torch,before_constructor_rng,pref.rng_snapshot(torch)) and
                bool((cold['final.weight']==0).all()),'Cold constructor/zero-last/RNG differs')
            del repeated
            if shared_cold is None:
                shared_cold=cold
            require(pref.tree_equal(torch,shared_cold,cold),'Three modes have different cold weights')
            adapter.to('cuda:0').requires_grad_(False).eval()
            diagnostics,geometry_packets=[],[]
            observer=observe_adapter(torch,adapter,pref,diagnostics,geometry_packets)
            try:
                with torch.random.fork_rng(devices=[0]):
                    torch.random.default_generator.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    private_before=pref.rng_snapshot(torch)
                    with torch.no_grad(),hooks.spatial_readout_context(torch,head,metadata) as record:
                        classes,masks=head(features,samples)
                        require(len(classes)==len(masks)==10 and
                            all(torch.equal(a,b) for a,b in zip(classes,baseline_classes)) and
                            all(torch.equal(a,b) for a,b in zip(masks,baseline_masks)),'Cold native fields changed')
                        packet,fusion_summary=official_output(torch,model,classes,masks,samples,identity,pref)
                    private_after=pref.rng_snapshot(torch)
                require(len(geometry_packets)==len(diagnostics)==1,'Cold observation count differs')
                if canonical_geometry is None:
                    canonical_geometry=geometry_packets[0]
                require(pref.tree_equal(torch,canonical_geometry,geometry_packets[0]),'Cold coordinates/E/halves differ')
                require(equal_rng(torch,external,pref.rng_snapshot(torch)) and
                    equal_rng(torch,private_before,baseline_private_before) and
                    equal_rng(torch,private_after,baseline_private_after) and
                    pref.tree_equal(torch,baseline_instances,packet) and
                    all(parameter.grad is None for parameter in model.parameters()),'Cold official output/RNG/gradNone differs')
                require(all(torch.equal(v.detach().cpu(),native_initial[n]) for n,v in model.state_dict().items()
                    if not n.startswith(ADAPTER_PREFIX)),'Native state changed during inference')
                rows.append(dict(mode=mode,cold_adapter=pref.state_proof(cold),cold_constructor_repeated_exact=True,
                    cold_final_weight_zero=True,canonical_coordinates_and_eligibility_exact=True,
                    classes10_exact=True,masks10_exact=True,official100_exact=True,query_ID_order_exact=True,
                    hook=dict(record),diagnostics=diagnostics[0],official=fusion_summary,private_seed=seed,
                    external_RNG_before=[pref.digest_tensor(x) for x in external],
                    private_RNG_before=[pref.digest_tensor(x) for x in private_before],
                    private_RNG_after=[pref.digest_tensor(x) for x in private_after],private_RNG_matches_baseline=True,
                    external_RNG_after=[pref.digest_tensor(x) for x in pref.rng_snapshot(torch)],
                    external_RNG_restored=True,native610_exact=True,all_parameter_grads_None=True))
                del classes,masks,packet,record
            finally:
                observer.remove()
                delattr(head,hooks.REGISTRATION_NAME)
                adapter.to('cpu')
            del adapter,observer,cold,diagnostics,geometry_packets
        require(len(model.state_dict())==610 and all(torch.equal(v.detach().cpu(),native_initial[n])
            for n,v in model.state_dict().items()),'Final native state differs')
        result=dict(identity=identity,initializer=initialization,
            baseline_scope='bare native610 head; no spatial bridge installed',
            input=pref.digest_tensor(data['inputs']),same_cached_backbone_features_for_all_modes=True,
            baseline_private_RNG_before=[pref.digest_tensor(x) for x in baseline_private_before],
            baseline_private_RNG_after=[pref.digest_tensor(x) for x in baseline_private_after],
            baseline_fields=baseline_fields,baseline_official=baseline_summary,mode_rows=rows,
            native_before=before,native_after=pref.state_proof(model.state_dict()),native610_exact=True,
            GT_removed_from_forward=True,loader_accesses_annotations=True,AP_computed=False,
            prediction_payload_exported=False,equality_is_TorchEqual_not_claimed_byte_identity=True)
        write(output/'INFERENCE_IDENTITY.json',result)
        return dict(identity=reference(output/'INFERENCE_IDENTITY.json'),all_three_cold_modes_exact=True)
    finally:
        if hasattr(head,hooks.REGISTRATION_NAME):
            delattr(head,hooks.REGISTRATION_NAME)
        model.to('cpu')


def training_fixture(torch, pref, old, hooks, readout, geometry, spec, spec_ref, prepared, dataset, coco, output, started,
                     mode='global'):
    model, native_initial, initialization = prepared
    require(mode in TRAINING_MODES,'Only actual spatial independent technical modes allowed')
    output=output/(mode+'_training')
    output.mkdir()
    adapter=hooks.register_spatial_mask_readout(torch,readout,geometry,model,mode,
        spec['readout_source'],spec['geometry_source'])
    adapter_ref=spec['readout_source']
    initial = pref.cpu_tree(torch, model.state_dict())
    model.to('cuda:0')
    policy = pref.trainability(torch, model)
    policy['kind'] = 'mask2former'
    require(policy['trainable_tensors'] == 347 and policy['frozen_tensors'] == 106 and
            policy['state_entries'] == 612 and policy['parameter_tensors'] == 453 and
            len(policy['backbone_conv_names']) == 53 and len(policy['BN_buffer_names']) == 159,
            'Expected native345 plus adapter2, frozen BN106+159, full612 state')
    optimizer, optimizer_audit = pref.make_optimizer(torch, model, spec['initializers']['mask2former'], policy)
    named = dict(model.named_parameters())
    activity = {name: 0 for name in policy['active_names']}
    traces, consumed, updates = [], [], 0
    head = model.panoptic_head
    optimizer.zero_grad(set_to_none=True)
    try:
        for micro, image_id in enumerate(IMAGE_IDS):
            pref.check_deadline(started, spec['deadline_seconds'])
            data, stripped, identity = image_data(old, model, dataset, coco, image_id)
            del stripped
            target = pref.check_targets(torch, data, identity)
            external = pref.rng_snapshot(torch)
            diagnostics, field_observation = [], []
            adapter_observer = observe_adapter(torch, adapter, pref, diagnostics)
            def observe_fields(module, inputs, output):
                require(len(output) == 2 and len(output[0]) == len(output[1]) == 10,
                        'Training native head did not produce ten fields')
                field_observation.append(dict(
                    classes=[dict(shape=list(x.shape), dtype=str(x.dtype), requires_grad=x.requires_grad)
                             for x in output[0]],
                    masks=[dict(shape=list(x.shape), dtype=str(x.dtype), requires_grad=x.requires_grad)
                           for x in output[1]],
                    first_micro_fields=(dict(classes=[pref.digest_tensor(x) for x in output[0]],
                                              masks=[pref.digest_tensor(x) for x in output[1]]) if micro == 0 else None)))
                return None
            fields_observer = None
            seed = 1000000 + micro
            try:
                with torch.random.fork_rng(devices=[0]):
                    torch.random.default_generator.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    private_before = pref.rng_snapshot(torch)
                    with hooks.spatial_readout_context(torch, head,
                             [sample.metainfo for sample in data['data_samples']]) as record, \
                             old.observe_targets(head) as matches:
                        # Observe AFTER the bridge whole-head replacement hook.
                        fields_observer = head.register_forward_hook(observe_fields)
                        losses = model.loss(data['inputs'], data['data_samples'])
                    total, scalars = pref.loss_terms(torch, losses)
                    require(len(scalars) == 30 and len(matches) == 10 and len(diagnostics) == len(field_observation) == 1,
                            'Native30/ten match/one final adapter call contract differs')
                    for match in matches:
                        require(len(set(match['positive_query_indices'])) == len(identity['ann_ids']) and
                                sorted(match['assigned_GT_indices']) == list(range(len(identity['ann_ids']))) and
                                all(0 <= q < 100 for q in match['positive_query_indices']),
                                'All raw GT must remain in each native assignment')
                    require(diagnostics[0]['input_F_requires_grad'] and diagnostics[0]['input_e0_requires_grad'] and
                            diagnostics[0]['residual_requires_grad'], 'Spatial F/e0/adapter path is not live')
                    (total / 2).backward()
                    private_after = pref.rng_snapshot(torch)
            finally:
                adapter_observer.remove()
                if fields_observer is not None:
                    fields_observer.remove()
            external_after = pref.rng_snapshot(torch)
            require(equal_rng(torch, external, external_after), 'Native supervised loss RNG not restored')
            gradients = []
            present = []
            for name in policy['active_names']:
                gradient = named[name].grad
                if gradient is not None:
                    gradients.append(gradient.detach().float().norm())
                    present.append(name)
            values = torch.stack(gradients).cpu().tolist()
            require(all(math.isfinite(value) for value in values), 'Nonfinite actual accumulated gradient')
            norm_map = dict(zip(present, values))
            for name, value in norm_map.items():
                activity[name] += int(value > 0)
            require(all(named[name].grad is None for name in policy['frozen_names']) and
                    all(not module.training for name, module in model.named_modules() if name in policy['BN_modules']),
                    'Frozen BN received gradients or training statistics')
            preclip, stepped = None, False
            if (micro + 1) % 2 == 0:
                settings = spec['initializers']['mask2former']['clip_grad']
                require(settings == dict(max_norm=0.01, norm_type=2), 'Technical native M2F clip must be explicit .01/L2')
                global_norm = torch.nn.utils.clip_grad_norm_([named[n] for n in policy['active_names']], **settings)
                require(bool(torch.isfinite(global_norm)), 'Nonfinite global clip norm')
                preclip = float(global_norm.detach().cpu())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                stepped = True
                del global_norm
            consumed.extend(identity['ann_ids'])
            traces.append(dict(microstep=micro, image_id=image_id, identity=identity, GT=target,
                input=pref.digest_tensor(data['inputs']), loss_fields=scalars, native_total=float(total.detach().cpu()),
                native_loss_only=True, Sobel_added=False, auxiliary_loss_added=False, matches=list(matches),
                head_fields=field_observation[0], spatial_hook=dict(record), diagnostics=diagnostics[0],
                accumulation_divisor=2, optimizer_step=stepped, optimizer_updates=updates,
                preclip_global_norm=preclip, gradient_clipping_applied=stepped,
                gradient_norms_aligned_active_names=[norm_map.get(n) for n in policy['active_names']],
                gradients_are_accumulated_preclip=True, frozen_BN_parameter_grads_None=True,
                private_seed=seed, private_RNG_before=[pref.digest_tensor(x) for x in private_before],
                private_RNG_after=[pref.digest_tensor(x) for x in private_after],
                external_RNG_before=[pref.digest_tensor(x) for x in external],
                external_RNG_after=[pref.digest_tensor(x) for x in external_after], external_RNG_restored=True))
            write(output / ('MICRO_%02d.json' % micro), traces[-1])
            print(json.dumps(dict(status='TECHNICAL_PROGRESS', microsteps=micro + 1, updates=updates)), flush=True)
            del data, losses, total, gradient, gradients, record, matches, diagnostics, field_observation
            del private_before, private_after, external, external_after
        require(updates == 2 and all(parameter.grad is None for parameter in model.parameters()),
                'Exactly four microsteps/two updates and no pending accumulation gradients required')
        final = pref.cpu_tree(torch, model.state_dict())
        require(final.keys() == initial.keys() and all(bool(torch.isfinite(x).all()) for x in final.values()),
                'Full state changed schema or became nonfinite')
        frozen = policy['frozen_names'] + policy['BN_buffer_names']
        require(all(torch.equal(initial[n], final[n]) for n in frozen), 'Frozen BN affine/buffers changed')
        changed = {n: not torch.equal(initial[n], value) for n, value in final.items()}
        adapter_names = [ADAPTER_PREFIX + suffix for suffix in ('hidden.weight', 'final.weight')]
        require(all(activity[n] > 0 and changed[n] for n in adapter_names),
                'Four-micro fixture did not demonstrate learning in both adapter weights')
        meta = dict(schema='native-spatial-position-preserving-native-bridge-technical-checkpoint-v1', mode=mode, phase=spec['phase'],
            spec=spec_ref, source=reference(__file__), hooks_source=spec['hooks_source'], readout_source=adapter_ref,
            initial_checkpoint=spec['initializers']['mask2former']['checkpoint'], initializer_source=spec['initializer_source'],
            image_ids=list(IMAGE_IDS), consumed_ann_ids=consumed, microsteps=4, optimizer_updates=2,
            batch_size=1, accumulation=2, backbone_with_cp=spec['training_backbone_with_cp'], full_R50_conv_trainable=True,
            BN_stats_and_affine_frozen=True, optimizer=optimizer_audit,
            optimizer_param_names=[[row['name']] for row in optimizer_audit['parameter_groups']], policy=policy,
            native_loss_only=True, no_Sobel_or_auxiliary_loss=True, not_formal_training=True,
            eligible_for_formal_resume=False, no_AP=True)
        meta['no_pending_accumulation_gradients'] = True
        meta['geometry_source'] = spec['geometry_source']
        checkpoint = dict(state_dict=final, adapter_state_dict={n[len(ADAPTER_PREFIX):]: final[n] for n in adapter_names},
            optimizer_state_dict=pref.cpu_tree(torch, optimizer.state_dict()),
            external_rng_state=pref.cpu_tree(torch, pref.rng_snapshot(torch)), meta=meta)
        path = output / (mode + '_technical_final.pth')
        with path.open('xb') as stream:
            torch.save(checkpoint, stream)
        require(pref.tree_equal(torch, checkpoint, torch.load(path, map_location='cpu')),
                'Full technical model/optimizer/RNG CPU readback differs')
        proof = dict(schema='native-spatial-position-preserving-native-bridge-technical-state-proof-v1',
            native_cold=pref.state_proof(native_initial), before=pref.state_proof(initial), after=pref.state_proof(final),
            changed=changed, policy=policy, optimizer=optimizer_audit, positive_gradient_microcounts=activity,
            adapter_names=adapter_names, all_adapter_weights_positive_and_changed=True,
            native_changed_active_tensors=sum(changed[n] for n in policy['active_names'] if n not in adapter_names),
            frozen_BN_all_exact=True, CPU_checkpoint_readback_exact=True, initializer=initialization,
            initial_hidden_gradient_zero_is_expected=True, no_every_native_parameter_positive_claim=True)
        write(output / 'TRAIN_STATE_PROOF.json', proof)
        return dict(mode=mode, readout_source=adapter_ref,
            geometry_source=spec['geometry_source'],
            no_pending_accumulation_gradients=True,
            checkpoint=reference(path), state_proof=reference(output / 'TRAIN_STATE_PROOF.json'),
            traces=[reference(output / ('MICRO_%02d.json' % i)) for i in range(4)],
            microsteps=4, optimizer_updates=2, image_ids=list(IMAGE_IDS), consumed_ann_ids=consumed,
            training_forward_uses_native_GT_samples=True, GT_not_used_in_spatial_module=True,
            eligible_for_formal_resume=False, native_loss_only=True, AP_computed=False)
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.to('cpu')


def run(specfile):
    started = time.monotonic()
    require(Path(specfile).is_absolute(), 'Absolute --spec required')
    spec_ref = reference(specfile)
    spec = json.loads(Path(specfile).read_text(encoding='utf-8-sig'))
    require(spec['schema'] == SCHEMA and tuple(spec['modes']) == MODES and spec['inference_image_id'] == 1 and
            tuple(spec['image_ids']) == IMAGE_IDS and tuple(spec['training_modes']) == TRAINING_MODES and
            spec['backbone_with_cp'] is False and
            type(spec['training_backbone_with_cp']) is bool and
            spec['loss_seed_base'] == 1000000 and spec['head_seed_base'] == 100000 and
            spec['deadline_seconds'] > 0 and spec['CPU_affinity'] == [0, 1] and
            spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_owner_limit_MiB'] == 8192,
            'Exact three-cold/three fresh four-image technical cases required')
    owner_source_gate(specfile, spec)
    require(spec['readout_source']['bytes']==READOUT_BYTES and spec['readout_source']['sha256']==READOUT_SHA and
        spec['geometry_source']['bytes']==GEOMETRY_BYTES and spec['geometry_source']['sha256']==GEOMETRY_SHA and
        spec['hooks_source']['bytes']==BRIDGE_BYTES and spec['hooks_source']['sha256']==BRIDGE_SHA and
        all(spec[key] in spec['sources'] for key in ('readout_source','geometry_source','hooks_source','reused_CPU_reader')),
        'Exact spatial readout/original geometry/new bridge SOURCE roles required')
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES') == '0',
            'ROOT CUDA0/fresh absolute output required')
    for row in spec['sources']:
        bound(row)
    accepted = json.loads(bound(spec['initializer_qualification']).read_text(encoding='utf-8-sig'))
    require(accepted['schema'] == 'ROOT-native-coco-comparator-initializer-CPU-inspection-v1' and
            accepted['all_initializers_strictly_compatible'] is True and accepted['source'] == spec['initializer_source'],
            'ROOT exact COCO80-to-boat initializer qualification required')
    pref_accepted = json.loads(bound(spec['micro_preflight_qualification']).read_text(encoding='utf-8-sig'))
    require(pref_accepted['schema'] == 'ROOT-native-coco-comparator-micro-preflight-CPU-qualification-v1' and
            pref_accepted['status'] == 'ROOT_NATIVE_COCO_COMPARATOR_MICRO_PREFLIGHT_TECHNICALLY_QUALIFIED' and
            pref_accepted['source'] == spec['micro_preflight_source'], 'ROOT original native-loss qualification required')
    qualified_native_spec = json.loads(bound(pref_accepted['spec']).read_text(encoding='utf-8-sig'))
    require(qualified_native_spec['initializers']['mask2former'] == spec['initializers']['mask2former'] and
            qualified_native_spec['initializer_source'] == spec['initializer_source'] and
            qualified_native_spec['initializer_qualification'] == spec['initializer_qualification'],
            'Checkpoint/config/optimizer/paramwise/clip must equal the qualified native preflight policy')
    require(spec['initializers']['mask2former']['optimizer']['type'] == 'AdamW' and
            spec['initializers']['mask2former']['optimizer']['lr'] == 0.0001,
            'Explicit native M2F AdamW technical LR required')
    output.mkdir(parents=True)
    write(output / 'START.json', dict(schema='native-spatial-position-preserving-native-bridge-technical-start-v1', phase=spec['phase'],
        source=reference(__file__), spec=spec_ref, modes=list(MODES), image_ids=list(IMAGE_IDS),
        training_modes=list(TRAINING_MODES), native_runtime_qualification_pending_before_owner_run=True,
        inference_image_id=1, no_formal_resume=True, no_DEV=True, no_AP=True, no_retry=True))
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
            'ROOT CPU01 and initially uninitialized CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    initializer = load_source(spec['initializer_source'], 'paired_preflight_initializer')
    pref = load_source(spec['micro_preflight_source'], 'paired_preflight_native_source')
    old = load_source(spec['native_helpers'], 'paired_preflight_image_helper')
    readout=load_source(spec['readout_source'],'actual_spatial_readout')
    geometry=load_source(spec['geometry_source'],'sealed_original_geometry')
    hooks=load_source(spec['hooks_source'],'actual_spatial_native_bridge')
    old.torch=torch
    require(tuple(readout.MODES)==MODES and geometry.INITIALIZATION_SEED==2026100911,'Actual spatial/geometry modules required')
    classes = tuple(CocoDataset.METAINFO['classes'])
    require(len(classes) == 80 and classes[8] == 'boat', 'Official COCO80 class order differs')
    # Inference and ALL three fresh CPU training models BEFORE CUDA initialization.
    inference_model=pref.prepare_cpu(torch,initializer,MODELS,revert_sync_batchnorm,Config,
        classes,spec,'mask2former',accepted['initializers']['mask2former'])
    training_spec=dict(spec,backbone_with_cp=spec['training_backbone_with_cp'])
    training_models={mode:pref.prepare_cpu(torch,initializer,MODELS,revert_sync_batchnorm,Config,
        classes,training_spec,'mask2former',accepted['initializers']['mask2former']) for mode in TRAINING_MODES}
    require(all(pref.tree_equal(torch,inference_model[1],value[1]) for value in training_models.values()),
        'Fresh COCO native initial states differ')
    dataset = CocoDataset(ann_file=str(bound(spec['FIT'])), data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)), filter_cfg=None, pipeline=[], test_mode=True, lazy_init=False)
    coco = COCO(str(bound(spec['FIT'])))
    require(len(dataset) == len(coco.imgs) == 741 and len(coco.anns) == 1661 and
            set(IMAGE_IDS) <= set(coco.imgs), 'Actual complete FIT/native fixed image IDs differ')
    torch.cuda.init()
    capacity = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 4096 * 1024 ** 2 / capacity), 0)
    torch.cuda.reset_peak_memory_stats(0)
    results, cleanups, error_record = {}, {}, None
    try:
        results['inference']=inference_fixture(torch,pref,old,hooks,readout,geometry,spec,inference_model,
            dataset,coco,output,started)
        del inference_model
        cleanups['inference']=pref.clear_gpu(torch)
        for mode in TRAINING_MODES:
            prepared=training_models.pop(mode)
            results[mode+'_training']=training_fixture(torch,pref,old,hooks,readout,geometry,spec,spec_ref,
                prepared,dataset,coco,output,started,mode=mode)
            del prepared
            cleanups[mode+'_training']=pref.clear_gpu(torch)
        for row in spec['sources']:
            bound(row)
        pref.check_deadline(started, spec['deadline_seconds'])
    except Exception as error:
        error_record = dict(type=type(error).__name__, message=str(error),
            OOM_technical_stop=isinstance(error, torch.cuda.OutOfMemoryError), no_fallback=True, no_retry=True)
        traceback.clear_frames(error.__traceback__)
        if 'inference_model' in locals():
            inference_model[0].to('cpu')
            del inference_model
        if 'prepared' in locals():
            prepared[0].to('cpu')
            del prepared
        for pending in training_models.values():
            pending[0].to('cpu')
        training_models.clear()
        gc.collect()
        try:
            cleanups['error_path'] = pref.clear_gpu(torch)
        except Exception as cleanup_error:
            error_record['cleanup_error'] = str(cleanup_error)
    report = dict(schema=REPORT_SCHEMA,
        status='TECHNICAL_SPATIAL_POSITION_PRESERVING_NATIVE_BRIDGE_PREFLIGHT_COMPLETE_NO_METHOD_RESULT' if error_record is None else
               'TECHNICAL_SPATIAL_POSITION_PRESERVING_BRIDGE_STOP_NO_FORMAL_RESUME_NO_METHOD_RESULT',
        phase=spec['phase'], spec=spec_ref, source=reference(__file__), results=results, cleanup=cleanups,
        initializer_qualification=spec['initializer_qualification'], micro_preflight_qualification=spec['micro_preflight_qualification'],
        error=error_record, torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
        elapsed_seconds=time.monotonic() - started, no_DEV=True, no_AP=True,
        cold_inference_GT_removed=True if 'inference' in results else None,
        training_is_supervised_native_loss=True, no_Sobel_or_auxiliary_loss=True,
        formal_resume_eligible=False, scientific_method_decision_computed=False,
        published_recipe_reproduction_claimed=False)
    write(output / 'REPORT.json', report)
    if error_record is not None:
        raise RuntimeError('TECHNICAL_STOP: ' + error_record['type'] + ': ' + error_record['message'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'SOURCE_ONLY: explicit ROOT owner execution and fresh admission are required')
    run(args.spec)


if __name__ == '__main__':
    main()
