# UNBOUND_DRAFT: source-only prototype; no SAR training or GPU qualification.
"""Zero-initialized depthwise 3x3 feature adapter plus the fixed-width N16 MLP.

Default architecture has five parameter tensors / 18851 scalar parameters.
No I/O, data, CUDA calls, device changes, seeding, dropout or implicit neighbors.
Construction uses local CPU FP32 tensors and restores the external CPU RNG.
"""


def create_head(torch, feature_channels=256, hidden=32, num_classes=3):
    """Build a CPU FP32 head; caller supplies all nine spatial feature samples.

    head(center_F[S,256], query_e[S,256], native_L[S],
         neighbors[S,9,256]) -> finite FP32 logits[S,3].
    Architecture arguments are explicit fixed-contract checks, not tunable widths.
    """
    if (type(feature_channels) is not int or feature_channels != 256 or
            type(hidden) is not int or hidden != 32 or
            type(num_classes) is not int or num_classes != 3):
        raise ValueError('The fixed head requires feature_channels=256, hidden=32, num_classes=3')

    class FeatureAdapterHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.spatial_weight = torch.nn.Parameter(
                torch.zeros((9, 256), dtype=torch.float32, device='cpu'))
            self.norm_F = torch.nn.LayerNorm(256, eps=1e-5, elementwise_affine=False,
                                             device='cpu', dtype=torch.float32)
            self.norm_e = torch.nn.LayerNorm(256, eps=1e-5, elementwise_affine=False,
                                             device='cpu', dtype=torch.float32)
            self.hidden = torch.nn.Linear(513, 32, device='cpu', dtype=torch.float32)
            self.activation = torch.nn.ReLU()
            self.output = torch.nn.Linear(32, 3, device='cpu', dtype=torch.float32)

        def forward(self, center_F, query_e, native_L, neighbors):
            device = self.spatial_weight.device
            for name, value in (('center_F', center_F), ('query_e', query_e),
                                ('native_L', native_L), ('neighbors', neighbors)):
                if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or
                        value.dtype != torch.float32 or value.device != device):
                    raise ValueError(name + ' must be a dense FP32 tensor on the parameter device')
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(name + ' must be finite')
            if center_F.ndim != 2 or center_F.shape[1] != 256:
                raise ValueError('center_F must have shape [S,256]')
            count = center_F.shape[0]
            if (query_e.shape != (count, 256) or native_L.shape != (count,) or
                    neighbors.shape != (count, 9, 256)):
                raise ValueError('Expected query_e[S,256], native_L[S], neighbors[S,9,256]')
            # Use exactly the supplied neighbors; center-copy is a caller decision.
            adapted_F = center_F + (neighbors * self.spatial_weight.unsqueeze(0)).sum(dim=1)
            descriptor = torch.cat((self.norm_F(adapted_F), self.norm_e(query_e),
                                    (native_L / 8).unsqueeze(1)), dim=1)
            logits = self.output(self.activation(self.hidden(descriptor)))
            if (logits.shape != (count, 3) or logits.dtype != torch.float32 or
                    logits.device != device or not bool(torch.isfinite(logits).all())):
                raise ValueError('Head must return finite FP32 logits[S,3] on the input device')
            return logits

    # No seed or global device setter; constructor draws are externally invisible.
    with torch.random.fork_rng(devices=[]):
        return FeatureAdapterHead()
