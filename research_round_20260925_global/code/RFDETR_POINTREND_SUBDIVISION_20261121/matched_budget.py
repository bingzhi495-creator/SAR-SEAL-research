"""Compare two readouts of identical refined probabilities at a locked edit count."""

from math import isfinite
from typing import NamedTuple, Tuple


class Edits(NamedTuple):
    additions: Tuple[int, ...]
    removals: Tuple[int, ...]
    nonpositive_flips: int

    @property
    def area_delta(self):
        return len(self.additions) - len(self.removals)


def project_matched_budget(additions, removals, swaps):
    """Inputs are disjoint (original row-major pixel index, foreground probability).

    The input sides come from the unchanged native binary mask of the same
    candidate. Both outputs flip exactly 2*swaps cells of that mask. Scores,
    candidate indices and the lookup of original sides remain with the caller.
    """
    if type(swaps) is not int or not 0 <= swaps <= min(8, len(additions), len(removals)):
        raise ValueError("A frozen swap count available on both sides is required")

    def scored(endpoints, is_addition):
        for index, probability in endpoints:
            if type(index) is not int or index < 0 or not isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("A native pixel index and finite foreground probability are required")
            gain = probability - .5 if is_addition else .5 - probability
            yield gain, index, is_addition

    add = sorted(scored(additions, True), key=lambda row: (-row[0], row[1]))
    remove = sorted(scored(removals, False), key=lambda row: (-row[0], row[1]))
    pool = add + remove
    if len({row[1] for row in pool}) != len(pool):
        raise ValueError("Each native endpoint must appear once")

    def assemble(rows):
        return Edits(tuple(index for _, index, is_addition in rows if is_addition),
                     tuple(index for _, index, is_addition in rows if not is_addition),
                     sum(gain <= 0 for gain, _, _ in rows))

    balanced = assemble(add[:swaps] + remove[:swaps])
    joint = assemble(sorted(pool, key=lambda row: (-row[0], row[1]))[:2 * swaps])
    return balanced, joint
