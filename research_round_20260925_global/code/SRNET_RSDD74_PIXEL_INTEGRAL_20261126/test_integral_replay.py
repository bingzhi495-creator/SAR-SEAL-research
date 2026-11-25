from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import torch

spec = spec_from_file_location("integral_replay", Path(__file__).with_name("integral_replay.py"))
replay = module_from_spec(spec)
spec.loader.exec_module(replay)


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Linear(28, 1, bias=False)
        with torch.no_grad():
            self.network.weight.zero_()
            self.network.weight[0, 0] = 2

    def forward(self, own, context, p):
        return self.network(torch.cat((own, context), dim=-1)).squeeze(-1) + torch.logit(p.clamp(1e-4, 1-1e-4))


def test_only_direct_prior_changes_with_alternate_samples():
    head = Probe()
    x = np.zeros((2, 2, 14), dtype=np.float32)
    x[:, :, 0] = [[.12, .8], [.44, .59]]
    a = np.array([[.33, .73], [.22, .85]], dtype=np.float32)
    s = np.array([[.21, .81], [.38, .69]], dtype=np.float32)
    scale = np.ones((14,), dtype=np.float32)
    center = np.zeros((14,), dtype=np.float32)
    got = replay.posteriors((head, center, scale), x, a, s)
    with torch.inference_mode():
        visible = torch.from_numpy(x[:, :, 0])
        expected = [(torch.logit(torch.from_numpy(p)) + 2 * visible).sigmoid().numpy()
                    for p in (x[:, :, 0].copy(), a.copy(), s.copy())]
    assert all(np.array_equal(pred, want) for pred, want in zip(got, expected, strict=True))
    assert all(np.array_equal(x[:, :, 0], [[np.float32(.12), np.float32(.8)], [np.float32(.44), np.float32(.59)]])
               for _ in got)


if __name__ == "__main__":
    test_only_direct_prior_changes_with_alternate_samples()
    print("FP32 original posterior and static residual with two alternative priors: PASS")

