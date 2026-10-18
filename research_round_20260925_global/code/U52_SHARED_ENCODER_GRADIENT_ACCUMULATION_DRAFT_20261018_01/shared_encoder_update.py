"""UNEXECUTED U52 engineering candidate; no module-level Torch import or execution.

This lands the earlier U34 SHARED_FEATURE_VJP_REVIEW idea, not a new algorithm.
Only the future original owner may run numerical/state/timing comparisons.
"""
import math
import time

ORIGINAL_U39_SHA256 = '378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d'
ORIGINAL_ADAPTER_SHA256 = '9ad221170f69c028474f76700c991cb76dfecf19b11f5198767843b8e378d8b3'
ORIGINAL_U36_HELPERS_SHA256 = '1e726092a19d80a06f58d6a56f85f4c15bff2b744f0468a7e0a35fa2c85999e5'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def query_order(G, step):
    """Pure original metadata order, shared by the real candidate query loop."""
    need(type(G) is int and 1 <= G <= 100 and type(step) is int and step >= 1,
         'Original whole-image G and positive carried Adam step')
    return list(range(G))


def _elapsed(start):
    value = time.perf_counter()-start
    need(math.isfinite(value) and value >= 0, 'Finite elapsed time')
    return value


def _boundary(torch, adapter, optimizer, params, effective, inactive):
    """Reject unsupported sharing/hooks; this does not authenticate source provenance."""
    need(adapter.training and torch.is_grad_enabled() and not torch.is_autocast_enabled(),
         'Original training mode, first-order FP32, no AMP')
    actual = dict(adapter.named_parameters())
    need(set(actual) == set(params) == set(effective)|set(inactive)
         and not set(effective).intersection(inactive)
         and all(actual[name] is params[name] for name in actual), 'Actual original parameter objects and partition')
    encoder, readout = {}, {}
    for root in ('backbone','feature_projection'):
        encoder.update((root+'.'+name,value) for name,value in getattr(adapter,root).named_parameters())
    for root in ('query_projection','mask_projection','head'):
        readout.update((root+'.'+name,value) for name,value in getattr(adapter,root).named_parameters())
    need(set(encoder)|set(readout) == set(params)
         and not set(encoder).intersection(readout)
         and not {id(p) for p in encoder.values()}.intersection(id(p) for p in readout.values())
         and all(params[name] is value for name,value in list(encoder.items())+list(readout.items())),
         'Current encode/backbone+feature_projection and predict-only parameters must be disjoint and exhaustive')
    storages = set()
    for value in params.values():
        need(value.dtype == torch.float32 and value.device.type == 'cuda' and value.numel() > 0
             and value.grad is None and not getattr(value,'_backward_hooks',None),
             'Fresh original CUDA FP32 gradients; no parameter gradient transforms')
        storage = (str(value.device),value.untyped_storage().data_ptr())
        need(storage not in storages, 'Distinct Parameter objects cannot share writable storage in this minimal candidate')
        storages.add(storage)
    for module in adapter.modules():
        need(not getattr(module,'_backward_hooks',None) and not getattr(module,'_backward_pre_hooks',None),
             'No module backward hooks whose timing or transforms depend on repeated encoder traversals')
    optimized = [value for group in optimizer.param_groups for value in group['params']]
    need(len(optimized) == len({id(p) for p in optimized}) == len(effective)
         and {id(p) for p in optimized} == {id(params[name]) for name in effective},
         'Exactly original active Parameter objects, one optimizer update only after complete VJP')
    return set(encoder)


def update_shared_encoder(torch, c, reader, index, adapter, optimizer, params, effective, inactive,
            helpers, step, stage, arm_record, report):
    """UNEXECUTED candidate: accumulate a detached shared-feature leaf, then one encoder VJP.

    Same arguments as original U39._update. Future owner must source-close this
    module, original adapter/helpers and a new explicit caller; never patch U40.
    """
    item = x = encoded = leaf = encoder_gradient = e = m = native = gt = delta = rendered = logits = probability = bce = dice = loss = None
    started = time.perf_counter()
    try:
        c.check()
        item = reader.load(index)  # Unmodified source/GT/matcher/RGB validations.
        row = reader.records[index]
        G = item['G']
        order = query_order(G, step)
        need(item['identity'] == row['identity'] and item['query_ids'] == row['query_ids']
             and G == row['G'] and type(G) is int and 1 <= G <= 100,
             'Entire original matched GT/query order; never rematch/filter')
        stage.update(dataset_index=index, image_id=item['identity']['image_id'], G=G,
            query_ids=list(item['query_ids']), ann_ids=list(item['identity']['ann_ids']),
            original_shape=list(item['identity']['original_shape']),
            actual_img_shape=list(item['identity']['actual_img_shape']),
            original_image=item['identity']['image'], data_read_elapsed_seconds=_elapsed(started))
        need(tuple(item['e'].shape) == (G, 256) and tuple(item['M'].shape) == (G, 200, 200)
             and tuple(item['L'].shape) == tuple(item['targets'].shape)
             == (G,)+tuple(item['identity']['original_shape']), 'Original matched reader tensor ABI')
        # Reader-validated GT/F stay CPU; only this query's target/native-L move
        # to CUDA. Model inputs contain RGB, native e and native coarse M only.
        torch.cuda.reset_peak_memory_stats()
        x = item['rgb400'].to(device='cuda:0', dtype=torch.float32)
        need(tuple(x.shape) == (1, 3, 400, 400), 'Single RGB400 image')
        optimizer.zero_grad(set_to_none=True)
        encoder_names = _boundary(torch, adapter, optimizer, params, effective, inactive)
        need(all(not item[name].requires_grad for name in ('rgb400','e','M','L','targets')), 'Frozen original inputs; no hidden upstream teacher/target path')
        stage['optimizer_state_entries_before'] = len(optimizer.state)
        need(len(optimizer.state) == (0 if step == 1 else len(effective)),
             'AdamW moments remain resident across successive image updates')
        torch.cuda.synchronize()
        train_started = time.perf_counter()
        saved = stage['encoder_saved_on_cpu'] = {}
        with helpers._observed_save_on_cpu(torch, saved):
            encoded = adapter.encode(x)
        need(tuple(encoded.shape) == (1, 32, 400, 400) and encoded.dtype == torch.float32
             and encoded.requires_grad and encoded.grad_fn is not None
             and bool(torch.isfinite(encoded).all()), 'One finite intact shared encoding per image update')
        need(not getattr(encoded, '_backward_hooks', None), 'No encoded-output gradient transformation hooks')
        leaf = encoded.detach().requires_grad_(True)
        versions = (encoded._version, leaf._version)
        need(leaf.is_leaf and leaf.grad is None and leaf.data_ptr() == encoded.data_ptr(), 'Unmodified FP32 detached leaf with shared value storage')
        stage.update(engineering_candidate='U52_shared_encoder_gradient_accumulation', encoder_backward_calls=0, query_backward_calls=0, detached_leaf_used=True)
        chunks = []
        weighted_loss = weighted_bce = weighted_dice = 0.
        zero_identity = True if step == 1 else None
        for ordinal in order:
            e = item['e'][ordinal:ordinal+1].to(device='cuda:0', dtype=torch.float32)
            m = item['M'][ordinal:ordinal+1].to(device='cuda:0', dtype=torch.float32)
            native = item['L'][ordinal:ordinal+1].to(device='cuda:0', dtype=torch.float32)
            gt = item['targets'][ordinal:ordinal+1].to(device='cuda:0', dtype=torch.float32)
            delta = adapter.predict(leaf, e, m)
            rendered = item['geometry'].render_delta(delta)[:, 0]
            logits = native+rendered
            if step == 1:
                identical = bool((delta == 0).all() and (rendered == 0).all() and torch.equal(logits, native))
                zero_identity = zero_identity and identical
                need(identical, 'Pristine zero readout preserves matched native logits exactly')
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none')
            bce = bce.flatten(1).mean(1)
            probability = logits.sigmoid()
            dice = 1-(2*(probability*gt).flatten(1).sum(1)+1)/(probability.flatten(1).sum(1)+gt.flatten(1).sum(1)+1)
            loss = (bce+dice).mean()/G
            need(bool(torch.isfinite(loss)), 'Finite original-grid BCE+softDice')
            weighted_loss += float(loss.detach().item())
            weighted_bce += float(bce.detach().item())/G
            weighted_dice += float(dice.detach().item())/G
            chunks.append(dict(raw_GT_ordinal=ordinal, query_id=item['query_ids'][ordinal],
                               weight=1./G, retain_graph=False))
            loss.backward(retain_graph=False, create_graph=False)
            stage['query_backward_calls'] += 1
            e = m = native = gt = delta = rendered = logits = probability = bce = dice = loss = None
            c.check()
        need((encoded._version, leaf._version) == versions, 'Neither shared tensor view was modified in place')
        encoder_gradient = leaf.grad
        need(encoder_gradient is not None and encoder_gradient.dtype == torch.float32
             and encoder_gradient.shape == encoded.shape and not encoder_gradient.requires_grad
             and bool(torch.isfinite(encoder_gradient).all()), 'Complete finite shared-feature gradient including legitimate all-zero cases')
        need(all(params[name].grad is None for name in encoder_names), 'Detached query paths cannot silently bypass leaf into encoder parameters')
        encoded.backward(encoder_gradient, retain_graph=False, create_graph=False)
        stage['encoder_backward_calls'] = 1
        stage['shared_gradient_shape'] = list(encoder_gradient.shape)
        encoded = leaf = encoder_gradient = x = None
        helpers._require_actual_cpu_saves(saved)
        stage.update(loss=weighted_loss, BCE=weighted_bce, softDice=weighted_dice,
            chunks=chunks, image_encodes=1, all_matched_GT=G, initial_matched_zero_identity=zero_identity)
        helpers._gradient_report(torch, params, effective, inactive, stage)
        stage['optimizer_step_entered'] = True
        report['new_training'] = True
        report['optimizer_steps_entered'] += 1
        optimizer.step()
        # Count a returned real optimizer step immediately, even if a later
        # metadata/resource check fails; partial output must preserve this fact.
        arm_record['updates_completed'] += 1
        report['successful_optimizer_updates'] += 1
        stage['optimizer_step_returned'] = True
        stage['optimizer_state'] = helpers._optimizer_report(torch, optimizer, params, effective, step)
        need(all(params[name].grad is None for name in inactive), 'Inactive gradients remain None')
        torch.cuda.synchronize()
        stage['training_with_checks_elapsed_seconds'] = _elapsed(train_started)
    finally:
        # Adam/BN persist; gradients, each image, query and graph do not.
        optimizer.zero_grad(set_to_none=True)
        item = x = encoded = leaf = encoder_gradient = e = m = native = gt = delta = rendered = logits = probability = bce = dice = loss = None
        stage['image_scope_elapsed_seconds'] = _elapsed(started)
