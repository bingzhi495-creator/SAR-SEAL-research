"""Exact expected IoU for two Bernoulli reference columns with disjoint endpoint dependence."""

from operator import index

import numpy as np


def _pair_tables(probabilities, pairs, kappa):
    """Mix each independent Bernoulli table with one extremal table, preserving both marginals."""
    p, q = probabilities[pairs[:, 0]], probabilities[pairs[:, 1]]
    independent = np.stack(((1 - p) * (1 - q), (1 - p) * q, p * (1 - q), p * q))
    positive = np.stack((1 - np.maximum(p, q), np.maximum(q - p, 0), np.maximum(p - q, 0), np.minimum(p, q)))
    negative = np.stack((np.maximum((1 - p) - q, 0), np.minimum(q, 1 - p),
                         np.minimum(p, 1 - q), np.maximum(p - (1 - q), 0)))
    extreme = np.where(kappa >= 0, positive, negative)
    strength = np.abs(kappa)
    return (1 - strength) * independent + strength * extreme, strength * np.abs(extreme[3] - independent[3])


def _multiply(state, factor):
    """Multiply two per-reference polynomials without mixing the reference columns."""
    expanded = np.zeros((len(state) + len(factor) - 1, 2), dtype=np.float64)
    for lag, weight in enumerate(factor):
        expanded[lag:lag + len(state)] += state * weight
    return expanded


class _Evaluator:
    def __init__(self, probabilities, pairs, kappa, native_fixed_area):
        self.probabilities = probabilities
        self.pairs = pairs
        self.native_fixed_area = native_fixed_area
        paired_sites = np.zeros(len(probabilities), dtype=bool)
        paired_sites[pairs.ravel()] = True
        self.singletons = np.flatnonzero(~paired_sites)
        self.tables, self.abs_covariance = _pair_tables(probabilities, pairs, kappa)

    def _selected_and_area(self, selected, area, *, batch):
        chosen = np.asarray(selected)
        if (chosen.dtype != np.bool_ or chosen.ndim not in ((1, 2) if batch else (1,))
                or chosen.shape[-1] != len(self.probabilities)):
            raise ValueError("Expected a bool vector on the exact ordered support, or a batch for bound")
        dimensions = np.asarray(area)
        if (not np.issubdtype(dimensions.dtype, np.integer) or (dimensions.ndim and dimensions.shape != chosen.shape[:-1])
                or np.any(dimensions <= 0) or np.any(dimensions != self.native_fixed_area + chosen.sum(axis=-1))):
            raise ValueError("Edited area must equal native fixed area plus the count of selected endpoints")
        return chosen, dimensions

    def utility(self, selected, area):
        """Return two exact expectations for one edited mask."""
        chosen, edited_area = self._selected_and_area(selected, area, batch=False)
        p = self.probabilities
        singletons = self.singletons
        left = chosen[self.pairs[:, 0]]
        right = chosen[self.pairs[:, 1]]
        total_only_inside = (p[singletons[chosen[singletons]]].sum(axis=0)
                             + p[self.pairs[left & right]].sum(axis=(0, 1)))
        mass = np.ones((1, 2), dtype=np.float64)
        overlap = (self.native_fixed_area + total_only_inside)[None, :]

        for ordinal in np.flatnonzero(~(left & right)):
            first, second = self.pairs[ordinal]
            p00, p01, p10, p11 = self.tables[:, ordinal]
            if not (left[ordinal] or right[ordinal]):
                factor = np.stack((p00, p01 + p10, p11))
                overlap = _multiply(overlap, factor)
            else:
                if left[ordinal]:
                    remaining, selected_only = p[second], p10
                else:
                    remaining, selected_only = p[first], p01
                factor = np.stack((1 - remaining, remaining))
                contribution = np.stack((selected_only, p11))
                overlap = _multiply(overlap, factor) + _multiply(mass, contribution)
            mass = _multiply(mass, factor)

        for remaining in p[singletons[~chosen[singletons]]]:
            factor = np.stack((1 - remaining, remaining))
            overlap = _multiply(overlap, factor)

        denom = float(edited_area) + np.arange(len(overlap), dtype=np.float64)
        return (overlap / denom[:, None]).sum(axis=0)

    def bound(self, selected, area):
        """Bound the absolute change from independent expectation for one mask or a batch."""
        chosen, edited_area = self._selected_and_area(selected, area, batch=True)
        left = chosen[..., self.pairs[:, 0]]
        right = chosen[..., self.pairs[:, 1]]
        size = edited_area.astype(np.float64)[..., None]
        mixed = (left ^ right) @ self.abs_covariance
        both_unselected = (~left & ~right) @ self.abs_covariance
        return mixed / (size * (size + 1)) + 2 * both_unselected / ((size + 1) * (size + 2))


def prepare(probabilities, pairs, kappa, native_fixed_area):
    """Freeze pair tables for p[2*n,2], disjoint integer pairs and two signed strengths."""
    probabilities = np.array(probabilities, dtype=np.float64, copy=True)
    pairs = np.asarray(pairs)
    kappa = np.asarray(kappa, dtype=np.float64)
    outside_area = index(native_fixed_area)
    if (probabilities.ndim != 2 or probabilities.shape[1] != 2 or len(probabilities) % 2
            or not np.all(np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1))):
        raise ValueError("Expected two finite probability columns on flattened signed endpoint pairs")
    if pairs.shape == (0,):
        pairs = np.empty((0, 2), dtype=np.intp)
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer)
            or np.any((pairs < 0) | (pairs >= len(probabilities))) or np.unique(pairs).size != pairs.size):
        raise ValueError("Expected distinct in-bounds endpoints in disjoint integer pairs")
    if kappa.shape != (2,) or not np.all(np.isfinite(kappa) & (np.abs(kappa) <= 1)) or outside_area < 0:
        raise ValueError("Expected one signed dependence strength per reference and a nonnegative fixed area")
    return _Evaluator(probabilities, pairs.astype(np.intp, copy=True), kappa, outside_area)
