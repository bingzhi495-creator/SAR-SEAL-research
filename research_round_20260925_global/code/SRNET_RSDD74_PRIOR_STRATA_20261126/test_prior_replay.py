import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prior_replay import posterior_arms


class KnownResidual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Linear(28, 1, bias=False)
        with torch.no_grad():
            self.network.weight.zero_()
            self.network.weight[0, 0] = 2.0
            self.network.weight[0, 14] = 1.0

    def forward(self, own, partner, probability):
        return torch.logit(probability.clamp(1e-4, 1 - 1e-4)) + self.network(
            torch.cat((own, partner), -1)).squeeze(-1)


def test_separates_additive_prior_without_changing_input_probability():
    features = np.zeros((2, 2, 14), dtype=np.float32)
    features[..., 0] = [[0.0, 1.0], [0.25, 0.75]]
    center = np.full(14, 0.1, dtype=np.float64)
    scale = np.full(14, 2.0, dtype=np.float64)
    p, residual, original = posterior_arms(KnownResidual(), features, center, scale)
    raw = torch.from_numpy(features[..., 0])
    logits = torch.logit(raw.clamp(1e-4, 1 - 1e-4))
    expected_residual = 3.0 * ((raw - 0.1) / 2.0)
    assert p.dtype == residual.dtype == original.dtype == np.float32
    assert np.array_equal(p, logits.sigmoid().numpy())
    assert np.allclose(residual, expected_residual.sigmoid().numpy(), rtol=0, atol=1e-7)
    assert np.allclose(original, (logits + expected_residual).sigmoid().numpy(), rtol=0, atol=1e-7)


if __name__ == "__main__":
    test_separates_additive_prior_without_changing_input_probability()
    print("1 synthetic frozen-readout test passed")

