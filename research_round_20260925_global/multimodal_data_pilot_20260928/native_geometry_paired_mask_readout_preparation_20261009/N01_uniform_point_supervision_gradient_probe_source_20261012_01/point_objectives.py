"""Hard original-raster point BCE decomposition; Torch is caller-injected."""
def point_loss_components(torch, point_logits, hard_targets):
    if (not isinstance(point_logits, torch.Tensor) or point_logits.ndim != 2
            or point_logits.dtype not in (torch.float32, torch.float64)
            or min(point_logits.shape) <= 0 or not bool(torch.isfinite(point_logits).all())):
        raise ValueError('Finite FP32/FP64 nonempty [GT,points] logits required')
    if (not isinstance(hard_targets, torch.Tensor) or hard_targets.dtype != torch.bool
            or hard_targets.shape != point_logits.shape or hard_targets.device != point_logits.device):
        raise ValueError('Aligned hard bool point labels required')
    y = hard_targets.to(point_logits.dtype)
    positive_terms = y * torch.nn.functional.softplus(-point_logits)
    negative_terms = (1-y) * torch.nn.functional.softplus(point_logits)
    foreground = positive_terms.mean(1).mean()
    background = negative_terms.mean(1).mean()
    positive_mass = y.sum(1).clamp_min(1)
    negative_mass = (1-y).sum(1).clamp_min(1)
    balanced = (.5*positive_terms.sum(1)/positive_mass
                +.5*negative_terms.sum(1)/negative_mass).mean()
    return dict(foreground=foreground, background=background,
                uniform=foreground+background, balanced=balanced)

