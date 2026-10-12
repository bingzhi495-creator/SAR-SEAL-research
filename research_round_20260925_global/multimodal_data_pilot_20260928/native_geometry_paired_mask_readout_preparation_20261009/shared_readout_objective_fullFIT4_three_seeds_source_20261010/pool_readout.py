"""Inert SOURCE for two pooling controls on the frozen native mask basis.

Torch is injected by the caller after its SOURCE and execution gates. Both
controls use the existing RAW two-weight initialization and a pooled 512-vector
before the shared SiLU MLP. Freezing the native model remains a caller contract.
"""
CHANNELS = 256
PAIR_COUNT = 16
TRAINABLE_SCALARS = 393216
KINDS = ('pair_pooled', 'mask_pooled')
DEFAULT_HIDDEN_SEED = 2026100911


def build_adapter(torch, kind='pair_pooled', hidden_seed=DEFAULT_HIDDEN_SEED):
    if kind not in KINDS or type(hidden_seed) is not int or hidden_seed < 0:
        raise ValueError('Known pooling kind and nonnegative integer hidden seed required')

    class PooledPairGenerator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kind, self.hidden_seed = kind, hidden_seed
            outside = torch.get_rng_state().clone()
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(hidden_seed)
                self.hidden = torch.nn.Linear(512, 512, bias=False, device='cpu', dtype=torch.float32)
                self.final = torch.nn.Linear(512, 256, bias=False, device='cpu', dtype=torch.float32)
                with torch.no_grad():
                    self.final.weight.zero_()
            if not torch.equal(outside, torch.get_rng_state()):
                raise RuntimeError('Pooled control construction changed ambient CPU RNG')

        def forward(self, F, e0, valid_fraction, native_logits=None, context=None):
            if not torch.is_tensor(F) or F.ndim != 4 or F.shape[1] != CHANNELS or \
                    min(F.shape[0], F.shape[2], F.shape[3]) <= 0 or F.dtype != torch.float32 or \
                    not bool(torch.isfinite(F).all()):
                raise ValueError('Finite FP32 F[B,256,H,W] required')
            B, _, H, W = F.shape
            if not torch.is_tensor(e0) or e0.ndim != 3 or e0.shape[0] != B or \
                    e0.shape[2] != CHANNELS or e0.shape[1] <= 0 or e0.dtype != F.dtype or \
                    e0.device != F.device or not bool(torch.isfinite(e0).all()):
                raise ValueError('Finite same-device FP32 e0[B,Q,256] required')
            Q = e0.shape[1]
            if not torch.is_tensor(valid_fraction) or tuple(valid_fraction.shape) != (B, 1, H, W) or \
                    valid_fraction.dtype != F.dtype or valid_fraction.device != F.device or \
                    not bool(torch.isfinite(valid_fraction).all()) or \
                    not bool(((valid_fraction >= 0) & (valid_fraction <= 1)).all()):
                raise ValueError('Same-device FP32 metadata valid fraction[B,1,H,W] in[0,1] required')
            supplied = native_logits is not None
            if native_logits is None:
                native_logits = torch.einsum('bqc,bchw->bqhw', e0, F)
            if not torch.is_tensor(native_logits) or tuple(native_logits.shape) != (B, Q, H, W) or \
                    native_logits.dtype != F.dtype or native_logits.device != F.device or \
                    not bool(torch.isfinite(native_logits).all()):
                raise ValueError('Finite same-device native additive anchor[B,Q,H,W] required')
            diagnostics = dict(kind=self.kind, hidden_seed=self.hidden_seed, activation='silu',
                pool_location='before_shared_MLP', input_RMS_normalized=False,
                global_pool=True, dense_readout='same_unnormalized_F', auxiliary_inputs_detached=True,
                native_anchor='supplied_native_logits' if supplied else 'native_e0_dot_F',
                parameters=TRAINABLE_SCALARS, teacher_freeze_is_caller_contract=True,
                frozen_native_or_native_integration_qualified=False)
            if self.kind == 'pair_pooled':
                if not isinstance(context, dict) or set(context) != {'h', 'eligible'}:
                    raise ValueError('Pair pooling requires actual RAW context {h,eligible}')
                h, eligible = context['h'], context['eligible']
                if not torch.is_tensor(h) or tuple(h.shape) != (B, Q, PAIR_COUNT, 512) or \
                        h.dtype != F.dtype or h.device != F.device or not bool(torch.isfinite(h).all()) or \
                        not torch.is_tensor(eligible) or tuple(eligible.shape) != (B, Q, PAIR_COUNT) or \
                        eligible.dtype != torch.bool or eligible.device != F.device:
                    raise ValueError('Finite actual h[B,Q,16,512] and Boolean eligible[B,Q,16] required')
                eligible = eligible.detach()
                counts = eligible.sum(-1)
                weights = eligible.to(F.dtype) / counts.clamp_min(1)[..., None]
                pooled = (h.detach() * weights[..., None]).sum(2)
                diagnostics.update(pooling='eligible_pair_mean', eligible_counts=counts.detach(),
                    eligible=eligible, empty_E='zero_input_to_biasfree_MLP',
                    fg_support=None, bg_support=None, denominator_clamp=None)
            else:
                if context is not None:
                    raise ValueError('Global mask pooling derives its input from native logits, without external context')
                probability = native_logits.detach().sigmoid()
                foreground = probability * valid_fraction.detach()
                background = (1 - probability) * valid_fraction.detach()
                fg_support, bg_support = foreground.sum((-2, -1)), background.sum((-2, -1))
                fg_mean = torch.einsum('bqhw,bchw->bqc', foreground, F.detach()) / fg_support.clamp_min(1e-6)[..., None]
                bg_mean = torch.einsum('bqhw,bchw->bqc', background, F.detach()) / bg_support.clamp_min(1e-6)[..., None]
                pooled = torch.cat((fg_mean, bg_mean), dim=-1)
                diagnostics.update(pooling='native_soft_FG_BG_valid_fraction_weighted_mean',
                    eligible_counts=None, eligible=None, empty_E=None,
                    fg_support=fg_support.detach(), bg_support=bg_support.detach(), denominator_clamp=1e-6)
            if not bool(torch.isfinite(pooled).all()):
                raise ValueError('Pooling produced a nonfinite 512-vector')
            coefficient = self.final(torch.nn.functional.silu(self.hidden(pooled)))
            residual = torch.einsum('bqc,bchw->bqhw', coefficient, F)
            mask_logits = native_logits + residual
            if not bool(torch.isfinite(coefficient).all()) or not bool(torch.isfinite(mask_logits).all()):
                raise ValueError('Pooled control produced nonfinite coefficients or mask logits')
            diagnostics['pooled_input'] = pooled.detach()
            return dict(mask_logits=mask_logits, logit_residual=residual,
                coefficient_fields=torch.stack((coefficient, coefficient), dim=2), diagnostics=diagnostics)

    adapter = PooledPairGenerator()
    expected = {'hidden.weight': (512, 512), 'final.weight': (256, 512)}
    parameters = dict(adapter.named_parameters())
    if list(parameters) != list(expected) or len(adapter.state_dict()) != 2 or list(adapter.named_buffers()) or \
            any(tuple(parameters[name].shape) != shape for name, shape in expected.items()) or \
            sum(value.numel() for value in parameters.values()) != TRAINABLE_SCALARS:
        raise RuntimeError('Pooling controls must register only the original two RAW weights')
    return adapter
