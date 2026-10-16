"""UNBOUND N23 compute core. No torch import, I/O, or autonomous execution.

The Root-owned runtime supplies torch, already validated CPU packets, heads,
optimizers, loss_terms and resource control. Real GPU equivalence is required
before admission; metadata fixtures do not qualify scientific computation.
"""
import math

ARMS = ("A_COMMON", "A_ACTIVE", "B_C")


def validate_order(order, image_count):
    if type(image_count) is not int or image_count < 0:
        raise ValueError("image_count must be a nonnegative integer")
    result = tuple(order)
    if (len(result) != image_count or any(type(x) is not int for x in result)
            or set(result) != set(range(image_count))):
        raise ValueError("order must be a complete image-index permutation")
    return result


def iter_epoch_gt(images, order):
    """Visit images by the supplied permutation, retaining each raw GT order."""
    checked = validate_order(order, len(images))
    for image_index in checked:
        group = images[image_index]
        if not isinstance(group, (list, tuple)):
            raise ValueError("each images[index] must be an ordered GT list")
        for record in group:
            yield image_index, record


def fast_residual(torch, head, F, L):
    """Original PR-FL91 arithmetic; caller qualifies inputs and equivalence."""
    fine = torch.nn.functional.layer_norm(
        F.detach(), (256,), weight=None, bias=None, eps=1e-5)
    coarse = L.detach().clamp(-8, 8).unsqueeze(1) / 8
    hidden = torch.relu(head.hidden1(torch.cat((fine, coarse), dim=1)))
    hidden = torch.relu(head.hidden2(torch.cat((hidden, coarse), dim=1)))
    hidden = torch.relu(head.hidden3(torch.cat((hidden, coarse), dim=1)))
    return head.final(torch.cat((hidden, coarse), dim=1)).squeeze(-1)


def train_one_epoch(torch, heads, optimizers, images, order, loss_terms,
                    control, extension_weight=.25):
    """One fixed shared-image epoch; one AdamW step per GT and arm.

    All sampled points and repeats are retained. Inputs remain CPU tensors and
    are never mutated. Old and new point rows share one concatenated forward;
    Root must use that same concatenation for canonical equivalence checks.
    """
    checked_order = validate_order(order, len(images))
    if (type(extension_weight) not in (int, float)
            or not math.isfinite(extension_weight) or extension_weight < 0):
        raise ValueError("extension_weight must be finite and nonnegative")
    if set(heads) != set(ARMS) or set(optimizers) != set(ARMS):
        raise ValueError("exactly A_COMMON, A_ACTIVE and B_C are required")
    parameters = {}
    for arm in ARMS:
        heads[arm].train()
        parameters[arm] = tuple(heads[arm].parameters())
        if len(parameters[arm]) != 8:
            raise ValueError("each PR-FL91 head must have exactly eight parameters")

    sums = {arm: None for arm in ARMS}
    updates = {arm: 0 for arm in ARMS}
    visits = []
    zero_old = zero_new_omega = empty_new = 0
    device = "cuda:0"
    for image_index, record in iter_epoch_gt(images, checked_order):
        control.check()
        old = record["old"]
        new = record["new"]
        old_n = int(old["n"])
        if old_n != 1024:
            raise ValueError("original anchor must retain exactly 1024 points")
        old_flag_cpu = old["point_in_region"]
        zero_old += int(not bool(old_flag_cpu.any()))
        if new is None:
            new_n = 0
            new_F_cpu = old["F"][:0]
            new_L_cpu = old["L"][:0]
            new_target_cpu = old["targets"][:0]
            new_omega_cpu = old_flag_cpu[:0]
            new_weight_cpu = old["L"][:0]
        else:
            new_n = int(new["n"])
            if new_n < 0:
                raise ValueError("new point count must be nonnegative")
            new_F_cpu, new_L_cpu = new["F"], new["L"]
            new_target_cpu = new["targets"]
            new_omega_cpu = new["Omega_flag"]
            new_weight_cpu = new["importance_weights"]
        empty_new += int(new_n == 0)
        zero_new_omega += int(not bool(new_omega_cpu.any()))
        F = torch.cat((old["F"], new_F_cpu), dim=0).to(device=device)
        L = torch.cat((old["L"], new_L_cpu), dim=0).to(device=device)
        targets = torch.cat((old["targets"], new_target_cpu), dim=0).to(
            device=device, dtype=L.dtype)
        old_flag = old_flag_cpu.to(device=device, dtype=torch.bool)
        new_omega = new_omega_cpu.to(device=device, dtype=torch.bool)
        new_weights = new_weight_cpu.to(device=device)
        if (F.shape[0] != old_n + new_n or L.shape != (old_n + new_n,)
                or targets.shape != L.shape or old_flag.shape != (old_n,)
                or new_omega.shape != (new_n,) or new_weights.shape != (new_n,)):
            raise ValueError("shared batch tensor lengths are inconsistent")
        old_L, new_L = L[:old_n], L[old_n:]
        old_target, new_target = targets[:old_n], targets[old_n:]
        visits.append(dict(image_index=image_index,
                           GT_annotation_id=record["GT_annotation_id"],
                           query_id=record["query_id"]))
        for arm in ARMS:
            optimizer = optimizers[arm]
            optimizer.zero_grad(set_to_none=True)
            residual = fast_residual(torch, heads[arm], F, L)
            old_u, new_u = residual[:old_n], residual[old_n:]
            old_logits = torch.where(old_flag, old_L + old_u, old_L)
            anchor = torch.nn.functional.binary_cross_entropy_with_logits(
                old_logits, old_target, reduction="none").mean()
            if arm == "B_C":
                new_logits = new_L + new_u
            else:
                new_logits = torch.where(new_omega, new_L + new_u, new_L)
            new_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                new_logits, new_target, reduction="none")
            terms = loss_terms.combine_loss(
                anchor, new_losses, new_omega, new_weights, arm,
                extension_weight=extension_weight)
            total = terms["total"]
            total.backward()
            grads = tuple(parameter.grad for parameter in parameters[arm])
            if any(gradient is None for gradient in grads):
                raise RuntimeError("all eight gradients must exist, including K0")
            flattened = torch.cat(tuple(gradient.reshape(-1) for gradient in grads))
            if not bool(torch.isfinite(flattened).all()):
                raise RuntimeError("nonfinite PR-FL91 gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            denom = torch.as_tensor(terms["denom"], device=device,
                                    dtype=total.dtype)
            observed = torch.stack((total.detach(), anchor.detach(),
                                    terms["extension"].detach(), denom.detach()))
            sums[arm] = observed if sums[arm] is None else sums[arm] + observed
            updates[arm] += 1
    result = {}
    for arm in ARMS:
        values = (sums[arm] / updates[arm]).cpu().tolist() if updates[arm] else [None]*4
        result[arm] = dict(total_loss_mean=values[0], anchor_loss_mean=values[1],
                           extension_loss_mean=values[2], extension_denom_mean=values[3],
                           updates=updates[arm])
    return dict(arms=result, GT_visits=len(visits), gt_visit_order=visits,
                zero_old_Omega_GT=zero_old, zero_new_Omega_GT=zero_new_omega,
                empty_new_GT=empty_new, image_order=list(checked_order),
                extension_weight=extension_weight)
