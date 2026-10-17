"""UNBOUND small query-conditioned adaptation of the user's model_new1 routing.

This is not a reproduction of the full legacy semantic model. Both arms share
the same input adapter, widths, ordinary convolutions/BN, independent up edges,
scale-shared pure CBAM, and zero-initialized dense residual readout. Only the
destination of already computed gated nodes differs. No scientific imports,
data access, model construction or CUDA work occurs at module import.
"""


MODES = ('SELECTIVE', 'WHOLE_NODE')


def _mode(mode):
    if type(mode) is not str or mode not in MODES:
        raise ValueError('mode must be SELECTIVE or WHOLE_NODE')
    return mode


def route_node(D, gate, mode):
    """One gate call; return (cross-scale source, other outgoing-edge source)."""
    _mode(mode)
    T = gate(D)
    return T, D if mode == 'SELECTIVE' else T


def make_head(torch, mode):
    """Build a CPU FP32 residual head without importing Torch in this module.

    Inputs are frozen F[1,256,H,W], e[Q,256], coarse_M[Q,H,W], all finite
    FP32 on the head's device. H/W must be >=16. The output is an unconstrained
    delta_M[Q,1,H,W], never an Omega-masked residual or a binary semantic map.
    Standard training BN requires >1 value/channel at the deepest scale;
    Q=1,H=W=16 numerical shape checks should use eval(), or Q>=2.
    """
    _mode(mode)  # Reject an invalid mode before any Torch access/construction.
    nn = torch.nn
    functional = torch.nn.functional
    factory_kwargs = dict(device='cpu', dtype=torch.float32)

    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels, pre):
            super().__init__()
            if pre:
                # Exact legacy ContinusParalleConv pre-BN operation order.
                self.layers = nn.Sequential(
                    nn.BatchNorm2d(in_channels, **factory_kwargs), nn.ReLU(),
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, **factory_kwargs),
                    nn.BatchNorm2d(out_channels, **factory_kwargs), nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, **factory_kwargs))
            else:
                self.layers = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, **factory_kwargs),
                    nn.BatchNorm2d(out_channels, **factory_kwargs), nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, **factory_kwargs),
                    nn.BatchNorm2d(out_channels, **factory_kwargs), nn.ReLU())

        def forward(self, value):
            return self.layers(value)

    class PureCBAM(nn.Module):
        def __init__(self, channels):
            super().__init__()
            # Fixed C//16 and spatial k3 from model_new1; no residual/SimAM.
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc1 = nn.Conv2d(channels, channels//16, 1, bias=False, **factory_kwargs)
            self.relu = nn.ReLU()
            self.fc2 = nn.Conv2d(channels//16, channels, 1, bias=False, **factory_kwargs)
            self.spatial = nn.Conv2d(2, 1, 3, padding=1, bias=False, **factory_kwargs)

        def forward(self, value):
            avg = self.fc2(self.relu(self.fc1(self.avg_pool(value))))
            maximum = self.fc2(self.relu(self.fc1(self.max_pool(value))))
            channel_gated = value * torch.sigmoid(avg + maximum)
            descriptor = torch.cat((torch.mean(channel_gated, dim=1, keepdim=True),
                                    torch.max(channel_gated, dim=1, keepdim=True)[0]), dim=1)
            return channel_gated * torch.sigmoid(self.spatial(descriptor))

    class DenseQueryRoutingHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = mode
            self.widths = (16, 32, 64, 128, 256)
            self.feature_projection = nn.Conv2d(256, 16, 1, **factory_kwargs)
            self.embedding_projection = nn.Linear(256, 16, **factory_kwargs)
            self.encoders = nn.ModuleList([
                DoubleConv(33 if level == 0 else self.widths[level-1], channels, pre=False)
                for level, channels in enumerate(self.widths)])
            self.gates = nn.ModuleList([PureCBAM(channels) for channels in self.widths])
            self.pool = nn.MaxPool2d(2, stride=2)
            self.upsamples = nn.ModuleDict()
            self.decoders = nn.ModuleDict()
            for column in range(1, 5):
                for level in range(5-column):
                    key = '%d_%d' % (level, column)
                    self.upsamples[key] = nn.ConvTranspose2d(
                        self.widths[level+1], self.widths[level],
                        kernel_size=4, stride=2, padding=1, **factory_kwargs)
                    self.decoders[key] = DoubleConv(
                        (column+1)*self.widths[level], self.widths[level], pre=True)
            # Public probe endpoint: Root may change a disposable copy only.
            self.readout = nn.Conv2d(16, 1, 1, **factory_kwargs)
            nn.init.zeros_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)

        def _inputs(self, F, e, coarse_M):
            values = (F, e, coarse_M)
            if (not all(isinstance(value, torch.Tensor) for value in values)
                    or F.ndim != 4 or F.shape[0] != 1 or F.shape[1] != 256
                    or e.ndim != 2 or e.shape[1] != 256 or e.shape[0] < 1
                    or coarse_M.ndim != 3 or coarse_M.shape[0] != e.shape[0]
                    or F.shape[-2:] != coarse_M.shape[-2:]
                    or min(F.shape[-2:]) < 16
                    or any(value.dtype != torch.float32 for value in values)
                    or any(value.device != self.readout.weight.device for value in values)
                    or not all(bool(torch.isfinite(value).all()) for value in values)):
                raise ValueError('Finite FP32 frozen F[1,256,H,W], e[Q,256], M[Q,H,W], H/W>=16 required')
            height, width = coarse_M.shape[-2:]
            queries = e.shape[0]
            # Per-pixel channel LN has no affine parameters, followed by 1x1.
            fine = functional.layer_norm(F.detach().permute(0, 2, 3, 1), (256,),
                                         weight=None, bias=None, eps=1e-5)
            fine = self.feature_projection(fine.permute(0, 3, 1, 2).contiguous())
            fine = fine.expand(queries, 16, height, width)
            embedding = functional.layer_norm(e.detach(), (256,),
                                              weight=None, bias=None, eps=1e-5)
            embedding = self.embedding_projection(embedding)[:, :, None, None]
            embedding = embedding.expand(queries, 16, height, width)
            coarse = coarse_M.detach().clamp(-8, 8).unsqueeze(1) / 8
            combined = torch.cat((fine, embedding, coarse), dim=1)
            pad_right, pad_bottom = (-width) % 16, (-height) % 16
            if pad_right or pad_bottom:
                combined = functional.pad(combined, (0, pad_right, 0, pad_bottom), mode='replicate')
            return combined, height, width

        def forward(self, F, e, coarse_M):
            combined, height, width = self._inputs(F, e, coarse_M)
            # Stage0 is a common already-gated node; never route/gate it twice.
            D0 = self.gates[0](self.encoders[0](combined))
            memory = {(0, 0):D0}
            cross = {}
            for level in range(1, 5):
                D = self.encoders[level](self.pool(memory[(level-1, 0)]))
                cross[(level, 0)], memory[(level, 0)] = route_node(
                    D, self.gates[level], self.mode)
            for column in range(1, 5):
                for level in range(5-column):
                    key = '%d_%d' % (level, column)
                    transported = self.upsamples[key](cross[(level+1, column-1)])
                    # Legacy concat order: transported tensor first, then all
                    # same-scale historical nodes in their original order.
                    incoming = [transported] + [memory[(level, old)] for old in range(column)]
                    D = self.decoders[key](torch.cat(incoming, dim=1))
                    if level == 0:
                        memory[(0, column)] = D  # Highest-resolution decoder has no gate.
                    else:
                        cross[(level, column)], memory[(level, column)] = route_node(
                            D, self.gates[level], self.mode)
            delta_M = self.readout(memory[(0, 4)])
            return delta_M[:, :, :height, :width].contiguous()

    return DenseQueryRoutingHead()
