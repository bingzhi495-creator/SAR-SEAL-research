"""Label-free rival evidence and equal-count ranking for one frozen scene."""

from hashlib import sha256

import numpy as np
from pycocotools import mask as codec


MIN_RIVAL_SCORE = 0.50
DUPLICATE_FRACTION = 0.50
SHUFFLE_SEED = b"FIT109_NATIVE_RIVALS_Q_QUARTILES_V1"


class NativeRivals:
    """Compute the strongest credible other mask score at sparse native cells."""

    def __init__(self, rows):
        self.masks = [row["segmentation"] for row in rows]
        self.scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
        self.areas = np.asarray(codec.area(self.masks), dtype=np.int64)
        self.shape = tuple(self.masks[0]["size"])
        if (not all(tuple(mask["size"]) == self.shape for mask in self.masks)
                or not np.isfinite(self.scores).all()
                or not np.all((0 <= self.scores) & (self.scores <= 1))):
            raise ValueError("Invalid frozen scene mask dimensions or scores")
        self.ranks = np.flatnonzero((self.scores >= MIN_RIVAL_SCORE) & (self.areas > 0))
        self.occupancy = codec.decode([self.masks[i] for i in self.ranks]) if len(self.ranks) else None

    def penalties(self, rank, coordinates):
        """Use zero-based native rank and flat row-major background cells."""
        cells = np.asarray(coordinates)
        height, width = self.shape
        if (not 0 <= rank < len(self.masks) or cells.ndim != 1
                or not np.issubdtype(cells.dtype, np.integer)
                or not np.all((0 <= cells) & (cells < height * width))
                or len(np.unique(cells)) != len(cells)):
            raise ValueError("Invalid sparse native addition cells")
        if not len(cells):
            return np.zeros(0, dtype=np.float64)
        y, x = np.divmod(cells, width)
        own = codec.decode(self.masks[rank])
        if np.any(own[y, x]):
            raise ValueError("Addition overlaps its original native mask")
        evidence = np.zeros(len(cells), dtype=np.float64)
        if self.occupancy is None:
            return evidence
        coverage = self.occupancy[y, x, :].astype(bool)
        for column in np.flatnonzero(coverage.any(axis=0) & (self.ranks != rank)):
            other = self.ranks[column]
            overlap = int(codec.area(codec.merge([self.masks[rank], self.masks[other]], intersect=True)))
            if overlap >= DUPLICATE_FRACTION * min(self.areas[rank], self.areas[other]):
                continue
            covered = coverage[:, column]
            evidence[covered] = np.maximum(evidence[covered], self.scores[other])
        return evidence


def placebo_penalties(probabilities, penalties, image_id, rank):
    """Exchange native penalties only within quartiles of addition probability."""
    ordinal = np.arange(len(penalties))
    order = np.lexsort((ordinal, np.asarray(probabilities)))
    key = SHUFFLE_SEED + b"|" + str(image_id).encode("ascii") + b"|" + str(rank).encode("ascii")
    seed = int.from_bytes(sha256(key).digest()[:16], "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    shuffled = np.asarray(penalties).copy()
    for block in np.array_split(order, 4):
        shuffled[block] = penalties[rng.permutation(block)]
    return shuffled


def rank_additions(probabilities, penalties, cap, count, choose):
    """Apply rival evidence to the addition column and preserve the exact count."""
    values = np.asarray(probabilities, dtype=np.float64).copy()
    penalties = np.asarray(penalties, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 2 or penalties.shape != (len(values),)
            or not np.isfinite(values).all() or not np.all((0 <= penalties) & (penalties <= 1))):
        raise ValueError("Invalid readout probabilities or rival penalties")
    values[:, 0] *= 1 - penalties
    return choose(values, cap, fixed_count=count)
