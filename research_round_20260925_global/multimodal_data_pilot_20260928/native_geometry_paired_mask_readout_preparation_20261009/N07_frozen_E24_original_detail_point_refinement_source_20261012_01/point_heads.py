"""SOURCE-only original-detail point heads; numerical qualification is pending.

C and R share the same 40,257-parameter architecture and seed initialization.
Only their direct 7x7 raw-image patch input differs: C replaces it with zero.
Native F/e already contain image and query information, so this is not an
information-matched control or a claim of a new refinement primitive.
"""


def make_point_head(torch, seed):
    """Build a CPU FP32 head without changing external CPU/CUDA RNG state.

    Torch is supplied by the caller. This factory performs no file access,
    native forward, matching, sampling, GT supervision, or optimizer action.
    Module transfer to a later device is a caller responsibility.
    """
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be a nonnegative int below 2**63")

    class OriginalDetailPointHead(torch.nn.Module):
        """Frozen inputs -> 562 -> 64 ReLU -> 64 ReLU -> raw scalar u.

        Input columns are LN(F256), LN(e256), clamp(L,-8,8)/8, patch49.
        The two LayerNorm operations have no affine parameters. Returned
        residual is the ungated u; band is abs(native L)<=2. Returned logits
        equal L+u in that band and native L outside it. This head does not
        estimate a quality score. Native inputs are detached internally.
        """

        def __init__(self):
            super().__init__()
            self.hidden1 = torch.nn.Linear(562, 64, bias=True,
                                           device="cpu", dtype=torch.float32)
            self.hidden2 = torch.nn.Linear(64, 64, bias=True,
                                           device="cpu", dtype=torch.float32)
            self.final = torch.nn.Linear(64, 1, bias=True,
                                         device="cpu", dtype=torch.float32)

        def forward(self, F, e, L, patch, arm="C"):
            """Return logits/residual [S] and bool band [S]; S=0 is allowed.

            All four tensors must be finite FP32 strided tensors on the
            module device. Shapes are [S,256], [S,256], [S], [S,49]. Both
            arms validate the supplied patch in [0,1] before C zeroes it.
            Inputs are never modified. A zero final layer gives numerical
            native-logit identity while allowing cold final-layer gradients.
            Activity of every parameter is not guaranteed (C has zero patch
            columns); exact signed-zero bit patterns are not asserted.
            """
            if type(arm) is not str or arm not in ("C", "R"):
                raise ValueError("arm must be C or R")
            values = (("F", F), ("e", e), ("L", L), ("patch", patch))
            for name, value in values:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(name + " must be a tensor")
                if value.layout != torch.strided or value.dtype != torch.float32:
                    raise TypeError(name + " must be a strided FP32 tensor")
            if F.ndim != 2 or F.shape[1] != 256:
                raise ValueError("F must have shape [S,256]")
            count = F.shape[0]
            if tuple(e.shape) != (count, 256) or tuple(L.shape) != (count,):
                raise ValueError("e/L must have shape [S,256]/[S]")
            if tuple(patch.shape) != (count, 49):
                raise ValueError("patch must have shape [S,49]")
            parameters = tuple(self.parameters())
            for parameter in parameters:
                if parameter.dtype != torch.float32 or parameter.device != F.device:
                    raise TypeError("head parameters must share input device and FP32")
                if not bool(torch.isfinite(parameter).all()):
                    raise ValueError("head parameters must be finite")
            for name, value in values:
                if value.device != F.device:
                    raise ValueError("all inputs must share the head device")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(name + " must be finite")
            if not bool(((patch >= 0) & (patch <= 1)).all()):
                raise ValueError("patch intensities must be in [0,1]")

            native = L.detach()
            normalized_F = torch.nn.functional.layer_norm(
                F.detach(), (256,), weight=None, bias=None, eps=1e-5)
            normalized_e = torch.nn.functional.layer_norm(
                e.detach(), (256,), weight=None, bias=None, eps=1e-5)
            raw_patch = patch.detach()
            if arm == "C":
                raw_patch = torch.zeros_like(raw_patch)
            inputs = torch.cat((normalized_F, normalized_e,
                                native.clamp(-8, 8).unsqueeze(1) / 8,
                                raw_patch), dim=1)
            hidden = torch.nn.functional.relu(self.hidden1(inputs))
            hidden = torch.nn.functional.relu(self.hidden2(hidden))
            residual = self.final(hidden).squeeze(-1)
            band = native.abs() <= 2
            logits = torch.where(band, native + residual, native)
            if not bool(torch.isfinite(residual).all() and
                        torch.isfinite(logits).all()):
                raise ValueError("head outputs must be finite")
            return dict(logits=logits, residual=residual, band=band)

    private = torch.Generator(device="cpu")
    private.manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.set_state(private.get_state())
        head = OriginalDetailPointHead()
        with torch.no_grad():
            head.final.weight.zero_()
            head.final.bias.zero_()
    expected = ("hidden1.weight", "hidden1.bias", "hidden2.weight",
                "hidden2.bias", "final.weight", "final.bias")
    if tuple(name for name, _ in head.named_parameters()) != expected:
        raise RuntimeError("unexpected point-head parameter names")
    if len(tuple(head.parameters())) != 6 or sum(
            parameter.numel() for parameter in head.parameters()) != 40257:
        raise RuntimeError("point-head parameter count must be 6/40257")
    if tuple(head.buffers()):
        raise RuntimeError("point head must have no buffers")
    return head
