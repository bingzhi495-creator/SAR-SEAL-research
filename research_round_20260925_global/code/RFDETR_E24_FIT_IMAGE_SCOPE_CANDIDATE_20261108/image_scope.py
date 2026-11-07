"""Deterministic FIT head split with exact encoded-image groups kept intact."""
from hashlib import sha256
import re

SALT = "SSDD-FIT741-RFDETR-E24-BOUNDARY-20261108-v1"


def split_by_content(rows):
    groups = {}
    image_ids = set()
    for row in rows:
        identity, digest = row["image_id"], row["image_sha256"]
        if (type(identity) is not int or identity < 0 or identity in image_ids or
                type(digest) is not str or re.fullmatch("[0-9a-f]{64}", digest) is None):
            raise ValueError("Expected unique integer images and exact content digests")
        image_ids.add(identity)
        groups.setdefault(digest, []).append(identity)
    if len(groups) < 3:
        raise ValueError("At least three distinct encoded image groups are required")
    order = sorted(groups, key=lambda key: (sha256(f"{SALT}\0{key}".encode("ascii")).digest(), key))
    count = len(order)
    fit_end = min(count - 2, max(1, 7 * count // 10))
    calibration_end = fit_end + max(1, 15 * count // 100)
    names = ("head_fit", "calibration", "head_holdout")
    slices = (order[:fit_end], order[fit_end:calibration_end], order[calibration_end:])
    return {identity: name for name, keys in zip(names, slices)
            for key in keys for identity in groups[key]}
