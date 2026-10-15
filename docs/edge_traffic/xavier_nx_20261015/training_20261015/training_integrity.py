"""Reject exact image content shared by different data splits."""
import hashlib


def audit_split_leakage(paths_by_split):
    seen = {}
    hashes = {}
    for split, paths in paths_by_split.items():
        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen and seen[digest][0] != split:
                raise ValueError(f"cross-split duplicate: {seen[digest]} and {(split, str(path))}")
            seen[digest] = (split, str(path))
            hashes[str(path)] = digest
    return hashes
