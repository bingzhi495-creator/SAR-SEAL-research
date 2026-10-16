"""Unbound SOURCE: PointRend-style shared F/L residual control.

PR-FL-91 uses three 91-wide hidden layers and coarse-logit reinjection.
It is a frozen-carrier structural adaptation, not the author pipeline or a
new primitive. No scientific import, file access, sampling or execution occurs
at module import. The caller supplies Torch and owns numerical admission.
"""


def make_point_head(torch, seed):
    """Return a CPU FP32 PR head without changing outside CPU/CUDA RNG.

    e and patch remain in the forward ABI for the original qualified cache.
    They are validated but never used to compute the PR residual.
    """
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be a nonnegative int below 2**63")

    class CoarseMaskConditionedPointHead(torch.nn.Module):
        """F256/L -> shared point residual; 8 tensors / 40,497 scalars.

        LN(F) has no affine parameters. The normalized native coarse logit
        c=clamp(L,-8,8)/8 is concatenated before all four linear layers.
        Returned u is ungated; returned logits are L+u in |L|<=2, else L.
        No score prediction, pixel selection, GT matching or query filtering
        is implemented here. Native inputs are detached, never mutated.
        """

        def __init__(self):
            super().__init__()
            self.hidden1 = torch.nn.Linear(257, 91, bias=True,
                                           device="cpu", dtype=torch.float32)
            self.hidden2 = torch.nn.Linear(92, 91, bias=True,
                                           device="cpu", dtype=torch.float32)
            self.hidden3 = torch.nn.Linear(92, 91, bias=True,
                                           device="cpu", dtype=torch.float32)
            self.final = torch.nn.Linear(92, 1, bias=True,
                                         device="cpu", dtype=torch.float32)

        def forward(self, F, e, L, patch, arm="PR"):
            """Return logits/residual [S], band [S]; S=0 is allowed.

            All operands must be finite FP32 strided tensors on the module
            device with shapes [S,256], [S,256], [S], [S,49]. The original
            patch range [0,1] remains an ABI requirement, although PR does
            not use e or patch. A zero final layer gives numerical native
            identity, without asserting bitwise signed-zero equality.
            """
            if type(arm) is not str or arm != "PR":
                raise ValueError("arm must be PR")
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
            for parameter in self.parameters():
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
            coarse = native.clamp(-8, 8).unsqueeze(1) / 8
            hidden = torch.nn.functional.relu(self.hidden1(
                torch.cat((normalized_F, coarse), dim=1)))
            hidden = torch.nn.functional.relu(self.hidden2(
                torch.cat((hidden, coarse), dim=1)))
            hidden = torch.nn.functional.relu(self.hidden3(
                torch.cat((hidden, coarse), dim=1)))
            residual = self.final(torch.cat((hidden, coarse), dim=1)).squeeze(-1)
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
        head = CoarseMaskConditionedPointHead()
        with torch.no_grad():
            head.final.weight.zero_()
            head.final.bias.zero_()
    expected = ("hidden1.weight", "hidden1.bias", "hidden2.weight", "hidden2.bias",
                "hidden3.weight", "hidden3.bias", "final.weight", "final.bias")
    if tuple(name for name, _ in head.named_parameters()) != expected:
        raise RuntimeError("unexpected PR point-head parameter names")
    if len(tuple(head.parameters())) != 8 or sum(
            parameter.numel() for parameter in head.parameters()) != 40497:
        raise RuntimeError("PR point-head parameter count must be 8/40497")
    if tuple(head.buffers()):
        raise RuntimeError("PR point head must have no buffers")
    return head
