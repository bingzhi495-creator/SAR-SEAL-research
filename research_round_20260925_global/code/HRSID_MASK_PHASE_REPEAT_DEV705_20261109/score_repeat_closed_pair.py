"""Score closed HRSID repeat DEV705 exports with the original paired geometry."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType


REPEAT_PROTOCOL = "a49361b92977a605267025b1e7798420936cbe53"
REPEAT_SEEDS = (5202, 5203)
DRIVER_SHA = "a046cab04371b2e0932ad817c8ad41381ccb7134a30842c37d9797e54b2f6e14"
DRIVER_SOURCE = (
    Path(__file__).resolve().parent.parent / "HRSID_MASK_PHASE_PAIRED_GEOMETRY_20261109" / "score_closed_pair.py"
)


def load_driver():
    path = DRIVER_SOURCE
    if path.resolve(strict=True) != path:
        raise ValueError("Canonical immutable geometry driver path required")
    with path.open("rb") as stream:
        source = stream.read(64 * 1024 + 1)
    if not 0 < len(source) <= 64 * 1024 or sha256(source).hexdigest() != DRIVER_SHA:
        raise ValueError("Exact immutable closed geometry driver required")
    driver = ModuleType("_repeat_scoped_p204_geometry")
    driver.__file__ = str(path)
    exec(compile(source, str(path), "exec"), driver.__dict__)
    return driver


def score_pair(pair_path, pair_sha256, training_seed):
    """Require the same exact seed in the authorized training and export pair receipts."""
    if type(training_seed) is not int or training_seed not in REPEAT_SEEDS:
        raise ValueError("Exact prelocked repeat training seed required")
    driver = load_driver()
    original_exact = driver.exact
    seeded_statuses = (driver.PAIR_STATUS, driver.TRAINING_PAIR_STATUS)

    def require_repeat_seed(actual, expected, reason, *, extra=()):
        if expected.get("status") in seeded_statuses:
            expected = dict(expected, seed=training_seed)
            reason += " with its authorized repeat seed"
        return original_exact(actual, expected, reason, extra=extra)

    # Each worker has its own byte-pinned driver namespace. Only two receipt
    # schemas and the training protocol are scoped to the authorized repeat.
    driver.exact = require_repeat_seed
    driver.TRAINING_PROTOCOL = REPEAT_PROTOCOL
    return dict(driver.score_pair(pair_path, pair_sha256), seed=training_seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--training-seed", type=int, choices=REPEAT_SEEDS, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--pair-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if (not output.is_absolute() or output.parent.resolve(strict=True) != output.parent
            or output.exists() or output.is_symlink()):
        raise ValueError("Fresh canonical output file required")
    result = score_pair(args.pair, args.pair_sha256, args.training_seed)
    data = (json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(data)


if __name__ == "__main__":
    main()
