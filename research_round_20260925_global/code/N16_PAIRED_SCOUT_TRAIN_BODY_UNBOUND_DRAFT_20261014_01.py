"""UNBOUND paired scout training numerical body, no IO/admission/RAW/source gate.

ROOT supplies a qualified thin-packet loader and owns device/resources, checkpoint
IO, endpoint CPU qualification, actual termination and official all100 evaluation.
Only actual random CPU SYNTHETIC fixtures have been run on this ordinary draft.
No actual science/model weights/data/protocol references were read here.
"""
import copy

ARMS = ('UTILITY', 'ERROR')
INIT_SEED = 202610140161
ORDER_SEED = 202610140160001
EPOCHS = 24
ENDPOINTS = (8, 24)
ADAMW = dict(lr=3e-4, weight_decay=1e-4, betas=(.9, .999), eps=1e-8)
LABEL_KEYS = dict(UTILITY='utility_labels', ERROR='error_labels')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def weighted_cross_entropy(torch, logits, labels, weights):
    n = len(labels)
    require(n > 0 and logits.dtype == weights.dtype == torch.float32 and labels.dtype == torch.int64
            and tuple(logits.shape) == (n, 3) and tuple(weights.shape) == (n,)
            and logits.device == labels.device == weights.device,
            'Finite same-device FP32 logits/weights and INT64 labels required')
    require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(weights).all())
            and bool((weights > 0).all()) and bool(((labels >= 0) & (labels <= 2)).all()), 'Invalid weighted CE inputs')
    # Deliberately mean over slots. Never normalize by the batch sum of weights.
    return (weights * torch.nn.functional.cross_entropy(logits, labels, reduction='none')).mean()


def _cpu_tree(torch, value):
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), 'Nonfinite snapshot tensor')
        return value.detach().cpu().clone()
    if type(value) is dict:
        return {k: _cpu_tree(torch, v) for k, v in value.items()}
    if type(value) in (tuple, list):
        return type(value)(_cpu_tree(torch, v) for v in value)
    return copy.deepcopy(value)


def make_paired_state(torch, scout_module, device='cpu'):
    heads = {arm: scout_module.make_scout(torch, INIT_SEED).to(device) for arm in ARMS}
    for head in heads.values():
        require(len(head.state_dict()) == len(tuple(head.parameters())) == 4
                and sum(p.numel() for p in head.parameters()) == 16547 and not tuple(head.buffers())
                and all(p.dtype == torch.float32 for p in head.parameters()), 'Original four-tensor16547 scout required')
    require(all(torch.equal(v, heads['ERROR'].state_dict()[k]) for k, v in heads['UTILITY'].state_dict().items()),
            'Paired normal initialization must be exactly identical')
    optimizers = {arm: torch.optim.AdamW(heads[arm].parameters(), **ADAMW) for arm in ARMS}
    return dict(schema='N16-paired-scout-training-body-state-v1', heads=heads, optimizers=optimizers,
                initialization_seed=INIT_SEED, order_seed=ORDER_SEED, completed_epoch=0,
                updates={arm: 0 for arm in ARMS}, visited_images=0, empty_images=0, sampled_points=0,
                history=[], epoch_orders=[], last_order_rng=None,
                scientific_source_qualified=False, actual_owned_terminal_self_certified=False)


def _inputs(torch, packet, device):
    """Thin packet contract; ROOT loader must not load full natural_L for training."""
    samples = packet['samples']
    sq, mapping = samples['query_ids'], packet['sample_to_unique']
    n = len(sq)
    for t in (sq, mapping):
        require(t.device.type == 'cpu' and t.dtype == torch.int64 and tuple(t.shape) == (n,), 'CPU common slot indices required')
    features, embeddings = packet['sampled_F_unique'], packet['native_e0']
    require(features.device.type == embeddings.device.type == 'cpu' and features.dtype == embeddings.dtype == torch.float32
            and features.ndim == 2 and features.shape[1] == 256 and tuple(embeddings.shape) == (1, 100, 256)
            and bool(torch.isfinite(features).all()) and bool(torch.isfinite(embeddings).all()), 'Actual GPU captured input field ABI differs')
    require(bool(((sq >= 0) & (sq < 100)).all()) and (n == 0 or bool(((mapping >= 0) & (mapping < len(features))).all())),
            'Sample/unique indices outside captured inputs')
    for key in ('native_L', 'importance_weights'):
        t = samples[key]
        require(t.device.type == 'cpu' and t.dtype == torch.float32 and tuple(t.shape) == (n,) and bool(torch.isfinite(t).all()), 'FP32 common sample input required')
    for key in LABEL_KEYS.values():
        t = samples[key]
        require(t.device.type == 'cpu' and t.dtype == torch.int64 and tuple(t.shape) == (n,)
                and bool(((t >= 0) & (t <= 2)).all()), 'Fixed threeclass labels required')
    require(samples['stratum_ids'].device.type == 'cpu' and tuple(samples['stratum_ids'].shape) == (n,)
            and samples['stratum_ids'].dtype == torch.uint8
            and bool((samples['stratum_ids'] <= 1).all()), 'Matched0/background1 strata required')
    require(samples['delta'].device.type == 'cpu' and samples['delta'].dtype == torch.bool
            and tuple(samples['delta'].shape) == (n,)
            and torch.equal(samples['delta'], samples['utility_labels'] != 2), 'Actual flip flag, not signed PR residual, required')
    if n == 0:
        return None
    require(bool((samples['importance_weights'] > 0).all()), 'Positive locked proposal importance weights required')
    return dict(F=features[mapping].to(device), e=embeddings[0, sq].to(device),
                L=samples['native_L'].to(device), weights=samples['importance_weights'].to(device),
                labels={arm: samples[LABEL_KEYS[arm]].to(device) for arm in ARMS}, samples=samples, n=n)


def train_one_epoch(torch, state, loader_callback, epoch, image_count=741, observer=None):
    require(type(epoch) is int and 1 <= epoch <= EPOCHS and state['completed_epoch'] == epoch - 1
            and type(image_count) is int and image_count > 0, 'Contiguous fixed1..24 epoch required')
    rng = torch.Generator(device='cpu')
    rng.manual_seed(ORDER_SEED + epoch)
    before = rng.get_state().clone()
    order = torch.randperm(image_count, generator=rng).tolist()
    after = rng.get_state().clone()
    device = next(state['heads']['UTILITY'].parameters()).device
    require(all(next(state['heads'][arm].parameters()).device == device for arm in ARMS), 'Both scout devices must match')
    for head in state['heads'].values():
        head.train()
    report = dict(epoch=epoch, order_seed=ORDER_SEED + epoch, order=order,
                  visited_images=image_count, updated_images=0, empty_images=0, points=0,
                  online_sum_image_weighted_CE={arm: 0. for arm in ARMS},
                  online_loss_is_not_fixed_endpoint=True)
    for index in order:
        if observer is not None:
            observer(dict(event='before_image', epoch=epoch, dataset_index=index))
        packet = loader_callback(index)  # exactly once for both arms
        inputs = _inputs(torch, packet, device)
        if inputs is None:
            report['empty_images'] += 1
            state['empty_images'] += 1
            del packet
            continue
        metadata = dict(epoch=epoch, dataset_index=index, points=inputs['n'], arms={})
        for arm in ARMS:
            head, optimizer = state['heads'][arm], state['optimizers'][arm]
            optimizer.zero_grad(set_to_none=True)
            logits = head(inputs['F'], inputs['e'], inputs['L'])
            loss = weighted_cross_entropy(torch, logits, inputs['labels'][arm], inputs['weights'])
            require(bool(torch.isfinite(loss)), 'Paired training CE must be finite')
            loss.backward()
            require(all(p.grad is not None and p.grad.dtype == torch.float32 and bool(torch.isfinite(p.grad).all())
                        for p in head.parameters()), 'All four trainable gradients must be finite FP32')
            optimizer.step()
            require(all(bool(torch.isfinite(p).all()) for p in head.parameters()), 'AdamW produced nonfinite scout parameter')
            state['updates'][arm] += 1
            metadata['arms'][arm] = dict(weighted_CE=float(loss.detach().cpu()), updates=state['updates'][arm])
            report['online_sum_image_weighted_CE'][arm] += metadata['arms'][arm]['weighted_CE']
            optimizer.zero_grad(set_to_none=True)
            del logits, loss, head, optimizer
        report['updated_images'] += 1
        report['points'] += inputs['n']
        state['sampled_points'] += inputs['n']
        if observer is not None:
            observer(dict(event='paired_image_update', **metadata))
        del packet, inputs, metadata
    state['visited_images'] += image_count
    state['completed_epoch'] = epoch
    state['epoch_orders'].append(list(order))
    state['last_order_rng'] = dict(seed=ORDER_SEED + epoch, before=before, after=after)
    for arm in ARMS:
        report['online_mean_image_weighted_CE_' + arm] = (report['online_sum_image_weighted_CE'][arm] / report['updated_images']
                                                       if report['updated_images'] else None)
    state['history'].append(copy.deepcopy(report))
    return report


def _statistics_empty():
    return {name: dict(actual_help=0., actual_harm=0., actual_noop=0.,
                       positive_help=0., positive_harm=0., positive_noop=0., sample_slots=0)
            for name in ('matched', 'background')}


def _statistics_add(torch, statistics, logits, samples, arm):
    # FIT sample diagnostic only. Deployment canonical NumPy FP64/topK is separate.
    probability = logits.detach().cpu().double().softmax(1)
    score = probability[:, 0] - probability[:, 1] if arm == 'UTILITY' else 2 * (probability[:, 0] + probability[:, 1]) - 1
    positive = score > 0
    for name, group_id in (('matched', 0), ('background', 1)):
        group = samples['stratum_ids'] == group_id
        statistics[name]['sample_slots'] += int(group.sum())
        for class_id, action in enumerate(('help', 'harm', 'noop')):
            actual = group & (samples['utility_labels'] == class_id)
            weights = samples['importance_weights'].double()
            statistics[name]['actual_' + action] += float(weights[actual].sum())
            statistics[name]['positive_' + action] += float(weights[actual & positive].sum())


def evaluate_fixed(torch, state, loader_callback, image_count=741, observer=None):
    """Recompute FULL cached FIT at current fixed weights, not online training loss."""
    device = next(state['heads']['UTILITY'].parameters()).device
    previous_modes = {arm: state['heads'][arm].training for arm in ARMS}
    for head in state['heads'].values():
        head.eval()
    result = dict(epoch=state['completed_epoch'], nonempty_images=0, empty_images=0, sampled_points=0,
                  arms={arm: dict(sum_image_weighted_CE=0., per_stratum_flip_statistics=_statistics_empty()) for arm in ARMS},
                  scope='fixed cached FIT samples; no AP/all100 deployment qualification',
                  primary_loss='mean of per-nonempty-image mean(weight*CE); no sumweights renormalization',
                  statistics_scope='importance-weighted sample action types/scout-positive, no overall accuracy, no topK claim')
    try:
        with torch.no_grad():
            for index in range(image_count):
                if observer is not None:
                    observer(dict(event='before_fixed_endpoint_image', epoch=state['completed_epoch'], dataset_index=index))
                packet = loader_callback(index)
                inputs = _inputs(torch, packet, device)
                if inputs is None:
                    result['empty_images'] += 1
                    del packet
                    continue
                result['nonempty_images'] += 1
                result['sampled_points'] += inputs['n']
                for arm in ARMS:
                    logits = state['heads'][arm](inputs['F'], inputs['e'], inputs['L'])
                    loss = weighted_cross_entropy(torch, logits, inputs['labels'][arm], inputs['weights'])
                    result['arms'][arm]['sum_image_weighted_CE'] += float(loss.cpu())
                    _statistics_add(torch, result['arms'][arm]['per_stratum_flip_statistics'], logits, inputs['samples'], arm)
                    del logits, loss
                del packet, inputs
        for arm in ARMS:
            item = result['arms'][arm]
            item['mean_image_weighted_CE'] = item['sum_image_weighted_CE'] / result['nonempty_images'] if result['nonempty_images'] else None
            for group in item['per_stratum_flip_statistics'].values():
                group['help_recall_on_actual_flips'] = group['positive_help'] / group['actual_help'] if group['actual_help'] else None
                group['harm_selection_rate_on_actual_flips'] = group['positive_harm'] / group['actual_harm'] if group['actual_harm'] else None
                selected_flips = group['positive_help'] + group['positive_harm']
                group['help_precision_among_positive_actual_flips'] = group['positive_help'] / selected_flips if selected_flips else None
        return result
    finally:
        for arm in ARMS:
            state['heads'][arm].train(previous_modes[arm])


def cpu_snapshot(torch, state):
    require(state['updates']['UTILITY'] == state['updates']['ERROR'], 'Paired update count equality required')
    optimizers = {arm: _cpu_tree(torch, state['optimizers'][arm].state_dict()) for arm in ARMS}
    for arm in ARMS:
        optimizer = optimizers[arm]
        require(len(optimizer['param_groups']) == 1 and len(optimizer['param_groups'][0]['params']) == 4, 'Exactly one full four-parameter AdamW group required')
        require(len(optimizer['state']) == (4 if state['updates'][arm] else 0), 'All four AdamW states must exist after an actual update')
        for value in optimizer['state'].values():
            step = value['step']
            require(step.numel() == 1 and float(step) == state['updates'][arm]
                    and bool((value['exp_avg_sq'] >= 0).all()), 'Exact AdamW step/count and nonnegative second moment required')
    return dict(schema='N16-paired-scout-training-CPU-snapshot-v1',
                models={arm: _cpu_tree(torch, state['heads'][arm].state_dict()) for arm in ARMS}, optimizers=optimizers,
                completed_epoch=state['completed_epoch'], updates=dict(state['updates']),
                visited_images=state['visited_images'], empty_images=state['empty_images'], sampled_points=state['sampled_points'],
                initialization_seed=INIT_SEED, order_seed=ORDER_SEED, AdamW=dict(ADAMW), fixed_epochs=EPOCHS,
                history=copy.deepcopy(state['history']), epoch_orders=copy.deepcopy(state['epoch_orders']),
                last_order_rng=_cpu_tree(torch, state['last_order_rng']),
                allowed_fixed_checkpoints=list(ENDPOINTS), qualification_pending=True,
                scientific_source_qualified=False, actual_owned_terminal_self_certified=False)


def train_until(torch, state, loader_callback, last_epoch=24, image_count=741, observer=None,
                checkpoint_callback=None):
    """Fixed8/24 endpoints only; optional callback owns concrete checkpoint IO.

    A first8 leaf can return state for continuation through24 WITHOUT resetting
    AdamW. The callback receives only JSON metadata and a complete CPU snapshot.
    Body never changes the endpoint or uses FIT loss to choose a best epoch.
    """
    require(last_epoch in ENDPOINTS and state['completed_epoch'] < last_epoch, 'Only fixed8/24 continuation target allowed')
    endpoints = {}
    for epoch in range(state['completed_epoch'] + 1, last_epoch + 1):
        train_one_epoch(torch, state, loader_callback, epoch, image_count, observer)
        if epoch in ENDPOINTS:
            endpoint = evaluate_fixed(torch, state, loader_callback, image_count, observer)
            snapshot = cpu_snapshot(torch, state)
            endpoints[epoch] = endpoint
            if checkpoint_callback is not None:
                checkpoint_callback(epoch, endpoint, snapshot)
            del snapshot
    return dict(completed_epoch=state['completed_epoch'], updates=dict(state['updates']),
                fixed_endpoints=endpoints, qualification_pending=True, AP_evaluated=False,
                actual_owned_terminal_self_certified=False)
