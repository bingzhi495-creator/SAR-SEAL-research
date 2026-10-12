"""Candidate pixel-posterior readout; SOURCE authoring is not qualification.

Inputs to point_outputs/dense_outputs must already share the original raster.
Project 256-channel features at their native resolution before using the external
renderer on the 16 projected channels. This module does not resize, use GT,
change query/class identities, freeze the teacher, or promise calibrated IoU.

A zero final layer preserves logits/masks, but the budget score is a new score.
Dense evaluation factorizes the first linear layer and changes floating-point
operation order relative to point evaluation; numerical agreement is toleranced.
"""

import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F


class QueryPixelPosteriorReadout(nn.Module):
    """9441 trainable scalars in eight parameter tensors; no buffers.

    Constructed explicitly on CPU in float32 without touching CUDA RNG.
    Call .double() or .to(...) externally before providing matching inputs.
    point_outputs is trainable; dense_outputs is inference-only.
    """

    def __init__(self, seed: int, kappa: float = 1.0, eps: float = 1e-6):
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("seed must be a non-bool integer in [0, 2**64 - 1]")
        kappa = self._positive_scalar("kappa", kappa)
        eps = self._positive_scalar("eps", eps)
        with torch.random.fork_rng(devices=[]):
            private_cpu_generator = torch.Generator(device="cpu")
            private_cpu_generator.manual_seed(seed)
            torch.random.set_rng_state(private_cpu_generator.get_state())
            super().__init__()
            self.feature_projection = nn.Conv2d(
                256, 16, kernel_size=1, bias=True, device="cpu",
                dtype=torch.float32,
            )
            self.query_projection = nn.Linear(
                256, 16, bias=True, device="cpu", dtype=torch.float32,
            )
            self.hidden = nn.Linear(
                36, 32, bias=True, device="cpu", dtype=torch.float32,
            )
            self.final = nn.Linear(
                32, 1, bias=True, device="cpu", dtype=torch.float32,
            )
            nn.init.zeros_(self.final.weight)
            nn.init.zeros_(self.final.bias)
        self.seed = seed
        self.kappa = kappa
        self.eps = eps

    @staticmethod
    def _positive_scalar(name, value):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(name + " must be a finite positive real scalar")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be finite and strictly positive")
        return value

    def _reference(self):
        reference = self.hidden.weight
        if reference.dtype not in (torch.float32, torch.float64):
            raise TypeError("module parameters must be float32 or float64")
        for parameter in self.parameters():
            if parameter.device != reference.device:
                raise ValueError("all module parameters must share one device")
            if parameter.dtype != reference.dtype:
                raise TypeError("all module parameters must share one dtype")
        limits = torch.finfo(reference.dtype)
        smallest_positive = limits.tiny * limits.eps
        for name in ("kappa", "eps"):
            value = self._positive_scalar(name, getattr(self, name))
            if not smallest_positive <= value <= limits.max:
                raise ValueError(name + " is not positive finite in the module dtype")
        return reference

    @staticmethod
    def _finite(name, value):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(name + " must contain only finite values")

    def _tensor(self, name, value, ndim, reference):
        if not isinstance(value, torch.Tensor):
            raise TypeError(name + " must be a torch.Tensor")
        if value.layout != torch.strided:
            raise TypeError(name + " must be a dense strided tensor")
        if value.ndim != ndim:
            raise ValueError(name + " has the wrong number of dimensions")
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(name + " must be float32 or float64")
        if value.dtype != reference.dtype:
            raise TypeError(name + " dtype must match the module dtype")
        if value.device != reference.device:
            raise ValueError(name + " device must match the module device")
        self._finite(name, value)

    def project_features(self, features):
        """features[B,256,Hf,Wf] -> pixel[B,16,Hf,Wf]; no resizing."""
        reference = self._reference()
        self._tensor("features", features, 4, reference)
        if features.shape[1] != 256 or any(size <= 0 for size in features.shape):
            raise ValueError("features must have positive shape [B,256,Hf,Wf]")
        pixel = self.feature_projection(features)
        self._finite("projected features", pixel)
        return pixel

    def project_queries(self, queries):
        """queries[B,Q,256] -> query[B,Q,16]; query order is preserved."""
        reference = self._reference()
        self._tensor("queries", queries, 3, reference)
        if queries.shape[2] != 256 or any(size <= 0 for size in queries.shape):
            raise ValueError("queries must have positive shape [B,Q,256]")
        query = self.query_projection(queries)
        self._finite("projected queries", query)
        return query

    def _outputs(self, native_logits, u, include_posterior_logits=False):
        self._finite("MLP output", u)
        posterior_logits = native_logits + u
        self._finite("posterior_logits", posterior_logits)
        posterior = torch.sigmoid(posterior_logits)
        bounded_delta = self.kappa * torch.tanh(u)
        corrected_logits = native_logits + bounded_delta
        self._finite("posterior", posterior)
        self._finite("bounded_delta", bounded_delta)
        self._finite("corrected_logits", corrected_logits)
        outputs = {
            "posterior": posterior,
            "corrected_logits": corrected_logits,
            "bounded_delta": bounded_delta,
        }
        if include_posterior_logits:
            outputs["posterior_logits"] = posterior_logits
        return outputs

    def point_outputs(self, pixel, query, native_logits, contrast):
        """Inputs: pixel/query[S,16], native_logits/contrast[S].

        contrast must be computed on the full corresponding query raster before
        selecting points. posterior_logits supports stable BCEWithLogits loss.
        S=0 returns connected empty outputs.
        """
        reference = self._reference()
        self._tensor("pixel", pixel, 2, reference)
        self._tensor("query", query, 2, reference)
        self._tensor("native_logits", native_logits, 1, reference)
        self._tensor("contrast", contrast, 1, reference)
        count = pixel.shape[0]
        if pixel.shape != (count, 16) or query.shape != (count, 16):
            raise ValueError("pixel and query must have identical shape [S,16]")
        if native_logits.shape != (count,) or contrast.shape != (count,):
            raise ValueError("native_logits and contrast must have shape [S]")
        probability = torch.sigmoid(native_logits)
        binary = (native_logits > 0).to(dtype=reference.dtype)
        inputs = torch.cat(
            (pixel, query, native_logits[:, None], probability[:, None],
             binary[:, None], contrast[:, None]),
            dim=1,
        )
        u = self.final(F.silu(self.hidden(inputs))).squeeze(-1)
        return self._outputs(native_logits, u, include_posterior_logits=True)

    @torch.no_grad()
    def dense_outputs(self, pixel, query, native_logits, query_chunk_size=4):
        """Inference only: three [B,Q,H,W] fields and budget[B,Q].

        Uses original-raster pixel[B,16,H,W], query[B,Q,16], and
        native_logits[B,Q,H,W]. No full [B,Q,H,W,36] tensor is created.
        The first layer uses a shared per-pixel term, a per-query term, and
        four scalar terms, in that order. Query blocks bound hidden workspaces;
        the three requested output fields are necessarily dense over all Q.
        """
        reference = self._reference()
        self._tensor("pixel", pixel, 4, reference)
        self._tensor("query", query, 3, reference)
        self._tensor("native_logits", native_logits, 4, reference)
        if type(query_chunk_size) is not int or query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be a positive non-bool integer")
        batch, channels, height, width = pixel.shape
        if channels != 16 or any(size <= 0 for size in pixel.shape):
            raise ValueError("pixel must have positive shape [B,16,H,W]")
        if query.shape[0] != batch or query.shape[2] != 16 or query.shape[1] <= 0:
            raise ValueError("query must have positive matching shape [B,Q,16]")
        queries = query.shape[1]
        if native_logits.shape != (batch, queries, height, width):
            raise ValueError("native_logits must have matching shape [B,Q,H,W]")

        weight = self.hidden.weight
        pixel_hidden = F.linear(
            pixel.permute(0, 2, 3, 1), weight[:, :16], self.hidden.bias,
        ).permute(0, 3, 1, 2)
        query_hidden = F.linear(query, weight[:, 16:32])
        self._finite("pixel hidden term", pixel_hidden)
        self._finite("query hidden term", query_hidden)
        fields = {
            name: torch.empty_like(native_logits)
            for name in ("posterior", "corrected_logits", "bounded_delta")
        }
        budget = native_logits.new_empty((batch, queries))
        for start in range(0, queries, query_chunk_size):
            end = min(start + query_chunk_size, queries)
            native = native_logits[:, start:end]
            probability = torch.sigmoid(native)
            binary = (native > 0).to(dtype=reference.dtype)
            local_mean = F.avg_pool2d(
                probability.reshape(batch * (end - start), 1, height, width),
                kernel_size=3, stride=1, padding=1, count_include_pad=False,
            ).reshape(batch, end - start, height, width)
            contrast = probability - local_mean
            hidden = (
                pixel_hidden[:, None]
                + query_hidden[:, start:end, :, None, None]
            )
            for index, scalar in enumerate((native, probability, binary, contrast)):
                hidden.addcmul_(
                    scalar[:, :, None],
                    weight[:, 32 + index].reshape(1, 1, 32, 1, 1),
                )
            u = F.linear(
                F.silu(hidden).permute(0, 1, 3, 4, 2),
                self.final.weight, self.final.bias,
            ).squeeze(-1)
            block = self._outputs(native, u)
            support = (block["corrected_logits"] > 0).to(dtype=reference.dtype)
            intersection = (support * block["posterior"]).sum(dim=(-2, -1))
            union = (
                support.sum(dim=(-2, -1))
                + ((1 - support) * block["posterior"]).sum(dim=(-2, -1))
            )
            block_budget = intersection / (union + self.eps)
            self._finite("budget", block_budget)
            budget[:, start:end] = block_budget
            for name in fields:
                fields[name][:, start:end] = block[name]
        fields["budget"] = budget
        return fields
