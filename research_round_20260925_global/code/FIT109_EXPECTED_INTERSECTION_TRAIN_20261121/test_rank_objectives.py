import itertools
import math

import pytest
import torch

from rank_objectives import rank_objectives


def _softplus(value):
    return max(0.0, value) + math.log1p(math.exp(-abs(value)))


def _winning_subset(values, k):
    # itertools lists subsets in native order; max retains the first winner.
    return max(itertools.combinations(range(len(values)), k),
               key=lambda subset: sum(values[i] for i in subset))


def _reference_row(z, q, k):
    size = len(z)
    if not size:
        return (0.0,) * 4
    bce = sum(_softplus(logit) - logit * label
              for zs, qs in zip(z, q) for logit, label in zip(zs, qs)) / (2 * size)
    pairwise = spo = regret = 0.0
    for side, sign in enumerate([1, -1]):
        logit = [p[side] for p in z]
        truth = [sign * p[side] for p in q]
        predicted = [sign / (1 + math.exp(-value)) for value in logit]
        margins = [_softplus(-sign * (logit[a] - logit[b]))
                   for a in range(size) for b in range(size) if truth[a] > truth[b]]
        pairwise += sum(margins) / max(len(margins), 1) / 2
        augmented = [2 * p - g for p, g in zip(predicted, truth)]
        chosen = _winning_subset(predicted, k)
        optimum = _winning_subset(truth, k)
        worst = _winning_subset(augmented, k)
        spo += (sum(augmented[i] for i in worst) - sum(augmented[i] for i in optimum))
        regret += sum(truth[i] for i in optimum) - sum(truth[i] for i in chosen)
    return bce, pairwise, spo / max(1, 2 * k), regret / max(1, 2 * k)


def test_exhaustive_small_rankings_and_budgets():
    # Every 3-point binary occupancy list, all 3-point score permutations,
    # and all budgets. The second column uses an independent ordering.
    rows = [(scores, truth, k)
            for scores in itertools.permutations([-1.0, 0.0, 1.0])
            for truth in itertools.product([0.125, 0.875], repeat=3)
            for k in range(4)]
    logits = [[[scores[i], scores[(i + 1) % 3]] for i in range(3)]
              for scores, truth, k in rows]
    labels = [[[truth[i], truth[(i + 2) % 3]] for i in range(3)]
              for scores, truth, k in rows]
    budgets = [k for _, _, k in rows]
    expected = [_reference_row(z, q, k) for z, q, k in zip(logits, labels, budgets)]
    assert all(r[2] + 1e-12 >= r[3] >= -1e-12 for r in expected)
    bce, pairwise, spo, _ = (sum(v) / len(rows) for v in zip(*expected))
    result = rank_objectives(torch.tensor(logits, dtype=torch.float64),
                             torch.tensor(labels, dtype=torch.float64),
                             [3] * len(rows), budgets)
    for name, value in dict(bce=bce, pairwise=bce + .25 * pairwise, spo=bce + .25 * spo).items():
        assert result[name].item() == pytest.approx(value, abs=1e-12)


def test_mixed_lengths_equal_row_weights_padding_and_empty():
    z = torch.tensor([[[0.6, -0.4], [-1., 2.], [1.1, 0.5]],
                      [[0.0, 0.6], [math.nan, math.inf], [math.inf, math.nan]],
                      [[math.nan, math.nan]] * 3], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[[0.1, 0.2], [0.7, 0.9], [0.8, 0.3]],
                      [[0.8, 0.8], [math.nan, 99], [-1, math.inf]],
                      [[math.nan, -99]] * 3], dtype=torch.float64)
    lengths, k = [3, 1, 0], [2, 1, 0]
    result = rank_objectives(z, q, lengths, k)
    reference = [_reference_row(z[i, :n].tolist(), q[i, :n].tolist(), k[i])
                 for i, n in enumerate(lengths)]
    bce, pairwise, spo, _ = [sum(v) / 3 for v in zip(*reference)]
    assert [v.item() for v in result.values()] == pytest.approx(
        [bce, bce + .25 * pairwise, bce + .25 * spo], abs=1e-12)
    sum(result.values()).backward()
    assert torch.isfinite(z.grad).all()
    assert not z.grad[1, 1:].count_nonzero() and not z.grad[2].count_nonzero()


def test_native_tie_priority_and_signed_spo_gradient():
    # Truth has three-way ties on both sides. The augmented utility chooses
    # the second point for both sides. At k=1, stable truth selection uses 0.
    z = torch.tensor([[[-1.0, 1.0], [1.0, -1.0], [0.0, 0.0]]],
                     dtype=torch.float64, requires_grad=True)
    q = torch.full_like(z, 0.5)
    result = rank_objectives(z, q, [3], [1])
    assert result['pairwise'].item() == result['bce'].item()
    grad, = torch.autograd.grad((result['spo'] - result['bce']) / .25, z)
    derivative = z.sigmoid() * (1 - z.sigmoid())
    expected = derivative * z.new_tensor([[[-1, 1], [1, -1], [0, 0]]])
    assert torch.allclose(grad, expected, atol=1e-12, rtol=0)

    # On an augmented tie choose the first native point as well. Binary
    # probabilities allow the augmented scores to tie exactly at zero.
    probs = torch.tensor([[[0.25, 0.75], [0.5, 0.5]]], dtype=torch.float64)
    z2 = probs.logit().requires_grad_()
    q2 = 2 * z2.sigmoid().detach() - 0.5
    r2 = rank_objectives(z2, q2, [2], [1])
    g2, = torch.autograd.grad(r2['spo'] - r2['bce'], z2)
    assert g2[0, 0, 0] > 0 and g2[0, 1, 0] < 0
    assert g2[0, 0, 1] < 0 and g2[0, 1, 1] > 0


def test_gradients_match_finite_differences_away_from_cutoff_ties():
    z = torch.tensor([[[0.8, -1.2], [-0.2, 0.3], [1.8, 0.5]]],
                     dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[[0.9, 0.88], [0.7, 0.21], [0.14, 0.68]]], dtype=torch.float64)

    def losses(value):
        return tuple(rank_objectives(value, q, [3], [1]).values())

    assert torch.autograd.gradcheck(losses, (z,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_all_padding_and_invalid_inputs():
    z = torch.empty((2, 0, 2), requires_grad=True)
    result = rank_objectives(z, z.detach(), [0, 0], [0, 0])
    assert all(v.item() == 0 for v in result.values())
    sum(result.values()).backward()
    assert z.grad.shape == z.shape
    z = torch.zeros(1, 9, 2)
    for lengths, k in [([2], [3]), ([9], [9]), ([10], [1]), ([-1], [0]),
                       ([3], [-1]), ([True], [0]), ([3], [0.5]), ([2, 3], [0])]:
        with pytest.raises(ValueError):
            rank_objectives(z, z, lengths, k)
    with pytest.raises(ValueError):
        rank_objectives(z, z + 1.01, [2], [1])
    with pytest.raises(ValueError):
        rank_objectives(z + math.inf, z, [1], [1])
