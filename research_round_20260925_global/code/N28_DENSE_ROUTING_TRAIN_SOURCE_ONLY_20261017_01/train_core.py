"""UNBOUND N28 paired routing training computation; no imports or file I/O.

The caller owns admission, checkpoint/RNG restoration, and fixed AdamW setup.
All masks may change. The two heads see the same images and contiguous original
GT microbatches. Weighted chunk losses equal the image mean of the losses
computed under this fixed microbatch recipe; BN means this is not a claim of
equivalence to a single full-G forward pass.
"""


ARMS = ('SELECTIVE', 'WHOLE_NODE')


def validate_order(order, population=741):
    """Return the given image order after exact one-visit validation."""
    if type(population) is not int or population < 1:
        raise ValueError('population must be a positive Python int')
    if not isinstance(order, (list, tuple)):
        raise ValueError('order must be a Python list or tuple')
    result = tuple(order)
    if (len(result) != population or any(type(index) is not int for index in result)
            or set(result) != set(range(population))):
        raise ValueError('order must contain each image ordinal exactly once')
    return result


def chunk_weight(chunk_GT, image_GT):
    """Weight a chunk mean so every original GT has equal image-level weight."""
    if (type(chunk_GT) is not int or type(image_GT) is not int
            or image_GT < 1 or not 1 <= chunk_GT <= image_GT):
        raise ValueError('chunk_GT and image_GT must be positive Python ints, chunk<=image')
    return chunk_GT / image_GT


def chunk_plan(image_GT, microbatch=4):
    """Contiguous original GT ranges and their image-mean accumulation weights."""
    if (type(image_GT) is not int or image_GT < 1
            or type(microbatch) is not int or microbatch < 1):
        raise ValueError('positive Python integer image_GT and microbatch required')
    result = []
    for start in range(0, image_GT, microbatch):
        stop = min(start + microbatch, image_GT)
        result.append((start, stop, chunk_weight(stop-start, image_GT)))
    return tuple(result)


def _spatial_shape(value, name, maximum=None):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(dimension) is not int or dimension < 1 for dimension in value)
            or (maximum is not None and any(dimension > maximum for dimension in value))):
        raise ValueError(name + ' must be two positive Python ints within canvas')
    return tuple(value)


def render_delta(torch, delta_M, identity):
    """The GPU246-qualified 200->800->actual crop->original FP32 renderer."""
    actual = _spatial_shape(identity['actual_img_shape'], 'actual_img_shape', 800)
    original = _spatial_shape(identity['original_shape'], 'original_shape')
    if delta_M.ndim != 4 or tuple(delta_M.shape[1:]) != (1, 200, 200):
        raise ValueError('delta_M must have shape [Q,1,200,200]')
    padded = torch.nn.functional.interpolate(
        delta_M.transpose(0, 1), size=(800, 800), mode='bilinear', align_corners=False)
    cropped = padded[0, :, :actual[0], :actual[1]]
    return torch.nn.functional.interpolate(
        cropped[:, None], size=original, mode='bilinear', align_corners=False)[:, 0]


def dense_instance_losses(torch, logits, targets):
    """Per-instance class-balanced BCE plus soft Dice, both on the full mask."""
    if logits.ndim != 3 or logits.shape != targets.shape:
        raise ValueError('original-shape logits and targets must both be [Q,H,W]')
    foreground = targets == 1
    background = targets == 0
    foreground_count = foreground.flatten(1).sum(dim=1)
    background_count = background.flatten(1).sum(dim=1)
    if not bool(((foreground_count > 0) & (background_count > 0)).all()):
        raise ValueError('every GT must have both foreground and background pixels')
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction='none')
    foreground_bce = (bce * foreground).flatten(1).sum(dim=1) / foreground_count
    background_bce = (bce * background).flatten(1).sum(dim=1) / background_count
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).flatten(1).sum(dim=1)
    total = probabilities.flatten(1).sum(dim=1) + targets.flatten(1).sum(dim=1)
    soft_dice = 1 - (2 * intersection + 1) / (total + 1)
    return 0.5 * (foreground_bce + background_bce) + soft_dice


def _validate_image(torch, item):
    """Validate the reader ABI without permuting, dropping, or resampling GTs."""
    if not isinstance(item, dict) or not isinstance(item.get('identity'), dict):
        raise ValueError('reader must return an image dict with its original identity')
    G = item['G']
    if type(G) is not int or G < 1:
        raise ValueError('reader.G must be a positive Python int')
    query_ids = item['query_ids']
    if (not isinstance(query_ids, (list, tuple)) or len(query_ids) != G
            or any(type(q) is not int or not 0 <= q < 100 for q in query_ids)
            or len(set(query_ids)) != G):
        raise ValueError('one distinct natural query id per original GT is required')
    identity = item['identity']
    _spatial_shape(identity['actual_img_shape'], 'actual_img_shape', 800)
    height, width = _spatial_shape(identity['original_shape'], 'original_shape')
    shapes = {'F': (1, 256, 200, 200), 'e': (G, 256),
              'M': (G, 200, 200), 'L': (G, height, width),
              'targets': (G, height, width)}
    for name, shape in shapes.items():
        value = item[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != torch.float32 or value.device.type != 'cpu'
                or value.requires_grad or not bool(torch.isfinite(value).all())):
            raise ValueError('reader.' + name + ' must be finite frozen CPU FP32 with exact shape')
    targets = item['targets']
    if not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError('targets must be binary, with no ignored or soft pixels')
    counts = targets.flatten(1).sum(dim=1)
    if not bool(((counts > 0) & (counts < height*width)).all()):
        raise ValueError('each original GT must contain foreground and background')
    return G


def _head_parameters(torch, head, optimizer, arm, device):
    if getattr(head, 'mode', None) != arm:
        raise ValueError('head mode does not match its arm key')
    parameters = [(name, parameter) for name, parameter in head.named_parameters()
                  if parameter.requires_grad]
    if (not parameters or any(parameter.dtype != torch.float32 or parameter.device != device
                              for _, parameter in parameters)):
        raise ValueError('all active head parameters must be FP32 on cuda:0')
    optimizer_parameters = [parameter for group in optimizer.param_groups
                            for parameter in group['params']]
    expected = {id(parameter) for _, parameter in parameters}
    actual = [id(parameter) for parameter in optimizer_parameters]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('optimizer must own exactly its head active parameters once')
    return parameters


def train_epoch(torch, heads, optimizers, reader, order, c, microbatch=4):
    """One fixed 741-image epoch, one optimizer update/image/arm, 1661 GTs.

    Inputs and teacher logits remain detached. One image F is transferred once;
    e/M/L/targets are transferred by shared GT chunk for the two arms. No model,
    optimizer, RNG, or cache file is read or written here. The caller must save
    and restore optimizer and all private RNG states between admitted segments.
    """
    order = validate_order(order, 741)  # Must precede any model or reader access.
    chunk_plan(1, microbatch)  # Reject an invalid size before state changes.
    if (not isinstance(heads, dict) or set(heads) != set(ARMS)
            or not isinstance(optimizers, dict) or set(optimizers) != set(ARMS)):
        raise ValueError('heads and optimizers require exactly the two fixed arm keys')
    device = torch.device('cuda:0')
    parameters = {}
    totals = {}
    for arm in ARMS:
        parameters[arm] = _head_parameters(torch, heads[arm], optimizers[arm], arm, device)
        heads[arm].train()
        totals[arm] = dict(updates=0, GT_visits=0, microbatches=0,
                           image_loss_sum=0.0, instance_loss_sum=0.0,
                           finite_gradient_checks=0,
                           active_parameter_tensors=len(parameters[arm]))
    GT_visits = 0
    chunk_visits = 0
    for index in order:
        c.check()
        item = reader.load(index)
        G = _validate_image(torch, item)
        plan = chunk_plan(G, microbatch)
        # F is shared by both arms and transferred only once for this image.
        F = item['F'].detach().to(device=device, dtype=torch.float32, non_blocking=False)
        for arm in ARMS:
            optimizers[arm].zero_grad(set_to_none=True)
        for start, stop, weight in plan:
            c.check()
            e = item['e'][start:stop].detach().to(device=device, dtype=torch.float32, non_blocking=False)
            M = item['M'][start:stop].detach().to(device=device, dtype=torch.float32, non_blocking=False)
            L = item['L'][start:stop].detach().to(device=device, dtype=torch.float32, non_blocking=False)
            targets = item['targets'][start:stop].detach().to(device=device, dtype=torch.float32, non_blocking=False)
            for arm in ARMS:
                c.check()
                delta_M = heads[arm](F, e, M)
                logits = L + render_delta(torch, delta_M, item['identity'])
                losses = dense_instance_losses(torch, logits, targets)
                loss = losses.mean() * weight
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(arm + ' nonfinite full-mask training loss')
                loss.backward()
                value = float(loss.detach().item())
                totals[arm]['image_loss_sum'] += value
                totals[arm]['instance_loss_sum'] += value * G
                totals[arm]['GT_visits'] += stop-start
                totals[arm]['microbatches'] += 1
                del delta_M, logits, losses, loss
            chunk_visits += 1
            del e, M, L, targets
        for arm in ARMS:
            c.check()
            finite = []
            for name, parameter in parameters[arm]:
                if parameter.grad is None:
                    raise RuntimeError(arm + ' missing active gradient: ' + name)
                finite.append(torch.isfinite(parameter.grad).all())
            if not bool(torch.stack(finite).all()):
                raise FloatingPointError(arm + ' nonfinite active parameter gradient')
            totals[arm]['finite_gradient_checks'] += len(finite)
            optimizers[arm].step()
            totals[arm]['updates'] += 1
            optimizers[arm].zero_grad(set_to_none=True)
            del finite
        GT_visits += G
        del F, item
    c.check()
    if GT_visits != 1661:
        raise RuntimeError('fixed FIT epoch must retain all 1661 original GTs')
    for arm in ARMS:
        if totals[arm]['updates'] != 741 or totals[arm]['GT_visits'] != 1661:
            raise RuntimeError('paired arm image/GT visit budget mismatch')
        totals[arm]['mean_image_loss'] = totals[arm]['image_loss_sum'] / 741
        totals[arm]['mean_instance_loss'] = totals[arm]['instance_loss_sum'] / 1661
    return dict(images=741, GT_visits=GT_visits, image_order=list(order),
                microbatch=microbatch, shared_microbatches=chunk_visits, arms=totals,
                full_mask_writable=True, teacher_updated=False,
                optimizer_updates_per_image_per_arm=1,
                loss_weighting='chunk_GT/image_GT; fixed microbatch BN semantics')
