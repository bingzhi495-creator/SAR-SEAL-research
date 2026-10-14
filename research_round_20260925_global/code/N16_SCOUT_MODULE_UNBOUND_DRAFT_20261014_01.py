"""UNBOUND_DRAFT: ordinary scout module, not scientific training or qualification.

Canonical implementation only: non-affine LayerNorm of F256/e256, eps=1e-5,
raw L/8, Linear513->32, ReLU, Linear32->3. Four parameter tensors, 16547
scalars, no buffers/dropout. Label semantics belong to the caller.
No actual SAR data, weights, files, Source/Spec/Q/RAW or CUDA execution here.
"""
EPS = 1e-5
PARAMETERS = 16547
STATE_SHAPES = {"hidden.weight": (32,513), "hidden.bias": (32,),
                "output.weight": (3,32), "output.bias": (3,)}


def _validate_inputs(torch, F, e, L, device):
    for name,value in (("F",F),("e",e),("L",L)):
        if not isinstance(value,torch.Tensor) or value.layout != torch.strided or value.dtype != torch.float32:
            raise ValueError(name + " must be a dense FP32 Tensor")
        if value.device != device:
            raise ValueError(name + " must share the scout parameter device")
    if F.ndim != 2 or F.shape[1] != 256:
        raise ValueError("F must have shape [N,256]")
    n = F.shape[0]
    if e.shape != (n,256) or L.shape != (n,):
        raise ValueError("e must be [N,256] and L must be [N]")
    for name,value in (("F",F),("e",e),("L",L)):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(name + " contains nonfinite values")
    return n


def _validate_state(torch, state, device):
    if not isinstance(state,dict) or set(state) != set(STATE_SHAPES):
        raise ValueError("Scout state must contain exactly four named tensors")
    for name,shape in STATE_SHAPES.items():
        value = state[name]
        if not isinstance(value,torch.Tensor) or value.layout != torch.strided or value.dtype != torch.float32 or value.device != device or value.shape != shape:
            raise ValueError("Scout parameter shape/dtype/device differs: " + name)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite scout parameter: " + name)


def _validate_output(torch, logits, n, device):
    if logits.shape != (n,3) or logits.dtype != torch.float32 or logits.device != device or not bool(torch.isfinite(logits).all()):
        raise ValueError("Scout must emit finite FP32 [N,3] logits on its input device")
    return logits


def independent_logits(torch, state, F, e, L):
    """CPU qualification formula, independent of nn.LayerNorm/nn.Linear calls.

    Mean/centered-square population variance implement non-affine LN. Explicit
    matmul+bias, ReLU, matmul+bias use the exact four supplied FP32 tensors.
    Same arithmetic graph in real numbers; FP32 bit equivalence or universal
    error bounds are not claimed. Caller must lock actual numeric criteria.
    """
    device = torch.device("cpu")
    n = _validate_inputs(torch,F,e,L,device)
    _validate_state(torch,state,device)

    def normalize(value):
        centered = value - value.mean(dim=1,keepdim=True)
        variance = (centered*centered).mean(dim=1,keepdim=True)
        return centered * torch.rsqrt(variance + EPS)

    descriptor = torch.cat((normalize(F),normalize(e),(L/8).unsqueeze(1)),dim=1)
    hidden = descriptor.matmul(state["hidden.weight"].t()) + state["hidden.bias"]
    hidden = torch.clamp_min(hidden,0)
    logits = hidden.matmul(state["output.weight"].t()) + state["output.bias"]
    return _validate_output(torch,logits,n,device)


def make_scout(torch, seed):
    """Build a CPU FP32 module from a private seed; preserve external CPU RNG.

    Seed is a nonnegative Python integer smaller than 2**63. Only the CPU
    generator is seeded: no CUDA seeding call, query or initialization.
    """
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative integer <2**63, excluding bool")

    class Scout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_F = torch.nn.LayerNorm(256, eps=EPS, elementwise_affine=False,
                                             device="cpu", dtype=torch.float32)
            self.norm_e = torch.nn.LayerNorm(256, eps=EPS, elementwise_affine=False,
                                             device="cpu", dtype=torch.float32)
            self.hidden = torch.nn.Linear(513,32,device="cpu",dtype=torch.float32)
            self.activation = torch.nn.ReLU()
            self.output = torch.nn.Linear(32,3,device="cpu",dtype=torch.float32)

        def forward(self, F, e, L):
            device = self.hidden.weight.device
            n = _validate_inputs(torch,F,e,L,device)
            _validate_state(torch,self.state_dict(),device)
            descriptor = torch.cat((self.norm_F(F), self.norm_e(e),
                                    (L/8).unsqueeze(1)), dim=1)
            logits = self.output(self.activation(self.hidden(descriptor)))
            return _validate_output(torch,logits,n,device)

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return Scout()
