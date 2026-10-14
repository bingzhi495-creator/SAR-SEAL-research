"""UNBOUND / UNEXECUTED DRAFT: one pointwise context-memory PR add-on.

No imports, file access, checkpoint loading, data access, training, or device
initialization occur at module import. The caller supplies Torch and the
existing qualified PR head. Numerical admission and execution belong to ROOT.

This adapts the context-retrieval idea of UCMNet; it is not UCMNet, a calibrated
uncertainty estimator, or a verified improvement on SAR instance segmentation.
"""


def make_context_memory_point_head(torch, base_head, seed, context_dim=8,
                                   memory_tokens=8, temperature=0.25):
    """Wrap an existing PR head without reinitializing or loading its weights.

    The existing ABI is forward(F, e, L, patch, arm='PR') returning a dict with
    logits, residual and band. The caller continues to apply its existing
    actual-region gating to the returned ungated residual. This wrapper never
    selects points, changes LOG support, scores queries, or modifies labels.

    Defaults add 2,201 parameters to the existing 40,497-parameter base head.
    Construct on CPU, then move the complete wrapper before creating AdamW.
    Use a fresh base instance with the same parent weights for each branch;
    do not share the same base object between two concurrent training arms.
    """
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be a nonnegative int below 2**63")
    if type(context_dim) is not int or context_dim < 1:
        raise ValueError("context_dim must be a positive integer")
    if type(memory_tokens) is not int or memory_tokens < 1:
        raise ValueError("memory_tokens must be a positive integer")
    if type(temperature) not in (int, float) or not 0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive")
    if not isinstance(base_head, torch.nn.Module):
        raise TypeError("base_head must be the caller's existing PR module")
    expected_shapes = ((91, 257), (91,), (91, 92), (91,),
                       (91, 92), (91,), (1, 92), (1,))
    if tuple(tuple(p.shape) for p in base_head.parameters()) != expected_shapes:
        raise ValueError("base_head must retain the original eight PR tensors")

    class ContextMemoryPointHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base_head
            self.context_dim = context_dim
            self.memory_tokens = memory_tokens
            self.temperature = float(temperature)
            self.query = torch.nn.Linear(257, context_dim, bias=True,
                                         device="cpu", dtype=torch.float32)
            self.memory_keys = torch.nn.Parameter(torch.empty(
                memory_tokens, context_dim, device="cpu", dtype=torch.float32))
            self.memory_values = torch.nn.Parameter(torch.empty(
                memory_tokens, context_dim, device="cpu", dtype=torch.float32))
            self.output = torch.nn.Linear(context_dim, 1, bias=True,
                                          device="cpu", dtype=torch.float32)
            torch.nn.init.normal_(self.memory_keys, std=context_dim ** -0.5)
            torch.nn.init.normal_(self.memory_values, std=0.1)
            with torch.no_grad():
                self.output.weight.zero_()
                self.output.bias.zero_()

        def forward(self, F, e, L, patch, arm="PR"):
            # The base owns validation of its ABI, including unused e/patch.
            result = self.base(F, e, L, patch, arm=arm)
            for parameter in (self.query.weight, self.query.bias,
                              self.memory_keys, self.memory_values,
                              self.output.weight, self.output.bias):
                if parameter.dtype != torch.float32 or parameter.device != F.device:
                    raise TypeError("add-on parameters must share input device/FP32")
                if not bool(torch.isfinite(parameter).all()):
                    raise ValueError("add-on parameters must be finite")

            normalized_F = torch.nn.functional.layer_norm(
                F.detach(), (256,), weight=None, bias=None, eps=1e-5)
            coarse = L.detach().clamp(-8, 8).unsqueeze(1) / 8
            query = self.query(torch.cat((normalized_F, coarse), dim=1))
            query = torch.nn.functional.normalize(query, p=2, dim=1, eps=1e-6)
            keys = torch.nn.functional.normalize(
                self.memory_keys, p=2, dim=1, eps=1e-6)
            routing = torch.softmax(
                (query @ keys.transpose(0, 1)) / self.temperature, dim=1)
            context = routing @ self.memory_values
            correction = self.output(context).squeeze(-1)
            residual = result["residual"] + correction
            native = L.detach()
            logits = torch.where(result["band"], native + residual, native)
            if not bool(torch.isfinite(residual).all() and
                        torch.isfinite(logits).all()):
                raise ValueError("context-memory outputs must be finite")
            updated = dict(result)
            updated.update(logits=logits, residual=residual)
            return updated

    # Only the CPU default RNG is temporarily replaced. Avoid torch.manual_seed,
    # which can alter CUDA RNG as well. CPU states restore even on exceptions;
    # no CUDA random operations occur during this explicitly CPU construction.
    private = torch.Generator(device="cpu")
    private.manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.set_state(private.get_state())
        head = ContextMemoryPointHead()
    expected_addon = 258 * context_dim + 2 * memory_tokens * context_dim + context_dim + 1
    if sum(p.numel() for p in head.parameters()) != 40497 + expected_addon:
        raise RuntimeError("unexpected context-memory parameter count")
    if tuple(head.buffers()):
        raise RuntimeError("context-memory point head must have no buffers")
    return head


def make_context_memory_head(torch, base_head, seed=202610140801,
                             tokens=8, dim=8, temperature=0.25):
    """Stable ROOT training-carrier API; no execution occurs at import.

    Added parameter names are query.weight/bias, memory_keys, memory_values,
    output.weight/bias. Existing parameters retain names under base.*.
    """
    return make_context_memory_point_head(
        torch, base_head, seed, context_dim=dim, memory_tokens=tokens,
        temperature=temperature)


def context_memory_parameter_counts(hidden_width=91, context_dim=8,
                                    memory_tokens=8):
    """Pure integer accounting only; does not construct or execute a model.

    The PR native coarse scalar is concatenated before ALL FOUR linear layers.
    Linear multiply-add estimates exclude LN, normalization, softmax and bias.
    """
    for value in (hidden_width, context_dim, memory_tokens):
        if type(value) is not int or value < 1:
            raise ValueError("dimensions must be positive integers")
    base = 2 * hidden_width ** 2 + 263 * hidden_width + 2
    addon = 259 * context_dim + 2 * memory_tokens * context_dim + 1
    base_macs = 2 * hidden_width ** 2 + 260 * hidden_width + 1
    addon_macs = 258 * context_dim + 2 * memory_tokens * context_dim
    return dict(base_parameters=base, addon_parameters=addon,
                total_parameters=base + addon,
                base_linear_macs_per_point=base_macs,
                addon_linear_macs_per_point=addon_macs)
