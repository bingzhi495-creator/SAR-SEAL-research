"""Injected one-image native integration check; SOURCE is not qualification.

ROOT must first gate provenance/resources and supply the strictly loaded E24
teacher, qualified SOURCE modules and one prelocked FIT sample with 1..100 raw
GT rows. raw_gt_masks/processed_gt must come from qualified shared.original_targets:
that caller check establishes raw annotation order and native processed rasters.
This helper reads no files, constructs no matching substitute, and writes nothing.
One detached teacher forward serves cold and warm backward checks. The fixed
warm perturbation is a graph-connection test, not an optimizer/training result.
No dense new-head inference, AP, convergence, calibration, or efficacy is proved.
"""

import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def gradients(torch, readout):
    result = {}
    for name, parameter in readout.named_parameters():
        value = parameter.grad
        require(value is not None and bool(torch.isfinite(value).all()), 'Missing/nonfinite readout gradient: ' + name)
        result[name] = dict(norm=float(value.detach().norm()), nonzero=int(torch.count_nonzero(value)))
    require(len(result) == 8, 'Exactly eight readout parameter gradients required')
    return result


def run_preflight(torch, shared, pref, core, old, readout_module, adapter_module,
                  renderer, model, data, samples, identity, raw_gt_masks,
                  processed_gt, readout_seed, point_seed):
    """Qualified objects are caller-injected; return a compact tensor-free dict.

    renderer is bound as renderer(masks, identity), using the qualified composite
    200->800->crop->original helper. GT supplies loss/matching only, never samples
    passed to the neural head. All original GT rows are retained; G=0 is rejected
    here because this check requires real cold/warm activity and native cost calls.
    """
    require(torch.is_grad_enabled(), 'Run outside the teacher no_grad scope')
    require(type(point_seed) is int and 0 <= point_seed <= 2**64 - 1, 'Explicit private CPU point seed required')
    require(all(not module.training for module in model.modules()) and
            all(not p.requires_grad and p.grad is None for p in model.parameters()), 'Frozen eval native teacher required')
    require(all(not getattr(module, 'with_cp', False) for module in model.modules()), 'Qualified noncheckpoint native inference required')
    require(data['inputs'].dtype == torch.float32 and tuple(data['inputs'].shape) == (1, 3, 800, 800) and
            data['inputs'].device == torch.device('cuda:0'), 'Actual batch1 FP32 native GPU0 input required')
    require(len(samples) == 1 and all(not any(key in sample for key in core.GT_FIELDS) for sample in samples),
            'All four GT fields must be absent from teacher forward samples')
    require(type(identity['image_id']) is int and identity['image_id'] > 0, 'Actual image ID required')
    count, original = len(identity['ann_ids']), tuple(identity['original_shape'])
    require(0 < count <= 100 and len(set(identity['ann_ids'])) == count, 'One image with 1..100 unique raw GT rows required')
    require(isinstance(raw_gt_masks, torch.Tensor) and raw_gt_masks.dtype in (torch.bool, torch.float32) and
            raw_gt_masks.device == data['inputs'].device and tuple(raw_gt_masks.shape) == (count,) + original and
            bool(((raw_gt_masks == 0) | (raw_gt_masks == 1)).all()), 'All aligned original binary GT rasters required')
    require(isinstance(processed_gt.labels, torch.Tensor) and isinstance(processed_gt.masks, torch.Tensor) and
            processed_gt.labels.dtype == processed_gt.masks.dtype == torch.int64 and
            tuple(processed_gt.labels.shape) == (count,) and tuple(processed_gt.masks.shape) == (count, 800, 800) and
            processed_gt.labels.device == processed_gt.masks.device == data['inputs'].device and
            bool((processed_gt.labels == 0).all()) and
            bool(((processed_gt.masks == 0) | (processed_gt.masks == 1)).all()), 'Native processed InstanceData labels/masks required')
    require(getattr(old, 'torch', None) is torch, 'Qualified native target observer needs injected Torch')
    anchor = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    require(len(anchor) == 610 and len(list(model.parameters())) == 451, 'Bare native610/451 teacher boundary required')
    bn = [(name, module) for name, module in model.named_modules()
          if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    bn_names = [name + '.' + key for name, module in bn
                for key, value in list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False))]
    require(len(bn) == 53 and sum(len(list(m.named_parameters(recurse=False))) for n, m in bn) == 106 and
            sum(len(list(m.named_buffers(recurse=False))) for n, m in bn) == 159, 'Typed native BN106+159 required')
    outside, hooks_before = pref.rng_snapshot(torch), core.hook_inventory(model)
    readout = readout_module.QueryPixelPosteriorReadout(seed=readout_seed)
    require(len(list(readout.parameters())) == 8 and sum(p.numel() for p in readout.parameters()) == 9441 and
            not list(readout.buffers()), 'Fresh standalone 9441/eight-parameter readout required')
    readout.to(device=data['inputs'].device).train()
    captured, forward, protected = shared.teacher_forward(torch, pref, core, model, data, samples, identity['image_id'])
    local_identity = dict(identity, metainfo=dict(img_shape=list(identity['actual_img_shape']),
        batch_input_shape=[800, 800], ori_shape=list(original)))
    head = model.panoptic_head
    require(head.num_points == 12544 and
            [type(cost).__name__ for cost in head.assigner.match_costs] == ['ClassificationCost', 'CrossEntropyLossCost', 'DiceCost'] and
            [float(cost.weight) for cost in head.assigner.match_costs] == [2., 5., 5.], 'Actual native final Hungarian recipe required')
    target_method, target_override = head._get_targets_single, '_get_targets_single' in head.__dict__
    costs_before = head.assigner.match_costs
    assigner_module = sys.modules[type(head.assigner).__module__]
    solver_before = assigner_module.linear_sum_assignment
    try:
        matching = shared.fixed_matching(torch, pref, core, old, head, captured, processed_gt, local_identity)
    finally:
        require(head._get_targets_single == target_method and ('_get_targets_single' in head.__dict__) == target_override and
                head.assigner.match_costs is costs_before and assigner_module.linear_sum_assignment is solver_before,
                'Native target/cost/SciPy observers were not restored')
    query_ids = torch.tensor(matching['query_by_GT'], dtype=torch.int64, device=data['inputs'].device)
    generator = torch.Generator(device='cpu').manual_seed(point_seed)
    indices = torch.randint(original[0] * original[1], (count, 12544), generator=generator, dtype=torch.int64).to(data['inputs'].device)
    targets = raw_gt_masks.detach().to(dtype=torch.bool)
    stages = {}
    for stage in ('cold', 'warm'):
        readout.zero_grad(set_to_none=True)
        if stage == 'warm':
            with torch.no_grad():
                readout.final.weight.copy_(torch.linspace(1e-3, 2e-3, 32, dtype=torch.float32,
                    device=data['inputs'].device).reshape(1, 32))
                readout.final.bias.fill_(1e-3)
        fields = adapter_module.project_native_fields(readout, captured['F'], captured['e0'], captured['M200'],
            identity['actual_img_shape'], identity['original_shape'], renderer)
        loss = adapter_module.matched_point_loss(readout, fields, query_ids, targets, indices)
        require(loss.shape == () and loss.requires_grad and bool(torch.isfinite(loss)), 'Finite live matched point BCE required')
        cold_native_BCE_exact = None
        if stage == 'cold':
            require(bool((readout.final.weight == 0).all()) and bool((readout.final.bias == 0).all()), 'Cold final layer must be exactly zero')
            with torch.no_grad():
                native_points = fields['native_logits'][0, query_ids].reshape(count, -1).gather(1, indices)
                gt_points = targets.reshape(count, -1).gather(1, indices).to(dtype=torch.float32)
                native_loss = torch.nn.functional.binary_cross_entropy_with_logits(native_points, gt_points, reduction='none').mean(1).mean()
                cold_native_BCE_exact = bool(torch.equal(loss.detach(), native_loss))
            require(cold_native_BCE_exact, 'Cold matched point BCE changed from native logits')
        loss.backward()
        audit = gradients(torch, readout)
        expected = all(v['nonzero'] > 0 if name.startswith('final.') else v['nonzero'] == 0 for name, v in audit.items()) if stage == 'cold' else all(v['nonzero'] > 0 for v in audit.values())
        require(expected, 'Constructed real-sample cold/warm gradient activity check failed; no efficacy inference')
        stages[stage] = dict(loss=float(loss.detach()), gradients=audit, cold_native_point_BCE_exact=cold_native_BCE_exact)
    after = model.state_dict()
    require(list(after) == list(anchor) and all(torch.equal(value.detach().cpu(), anchor[name]) for name, value in after.items()),
            'A native610 teacher state changed')
    require(all(not module.training for module in model.modules()) and
            all(not p.requires_grad and p.grad is None for p in model.parameters()), 'Native eval/frozen/gradNone invariant changed')
    require(all(value._version == version for value, version in zip(protected['tensors'], protected['versions'])) and
            core.hook_inventory(model) == hooks_before and core.equal_rng(torch, outside, pref.rng_snapshot(torch)),
            'Native output tensor version/hook/external RNG invariant changed')
    return dict(schema='query-pixel-posterior-native-injected-preflight-v1',
        status='QUERY_PIXEL_POSTERIOR_NATIVE_INJECTED_PREFLIGHT_BACKWARD_PASS', image_id=identity['image_id'],
        raw_ann_ids=list(identity['ann_ids']), native_forward_calls=1, backward_calls=2, matching=matching,
        readout_seed=readout_seed, point_seed=point_seed, points_per_GT=12544, stages=stages,
        native610_exact=True, native451_frozen_gradNone=True, BN106_affine159_buffers_exact=all(
            torch.equal(after[name].detach().cpu(), anchor[name]) for name in bn_names), all_native_modules_eval=True,
        native_all10_tensor_versions_unchanged=True, external_RNG_restored=True, hooks_removed=True,
        forward=forward, raw_processed_GT_source_row_validation='ROOT caller qualified shared.original_targets',
        no_optimizer_step=True, warm_perturbation_is_structural_only=True, all100_new_head_inference_qualified=False,
        AP_computed=False, calibration_qualified=False, efficacy_claimed=False,
        GPU_backward_bitwise_determinism_claimed=False)
