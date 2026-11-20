"""Keep the accepted native audit byte-identical while saving probabilities."""
from hashlib import sha256
from io import BytesIO
import json
import os

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_line(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def flush(stream):
    stream.flush()
    os.fsync(stream.fileno())


def plane_path(ordinal):
    return f"PROBABILITY_PRIVATE/{ordinal:04d}.npy"


class Capture:
    def __init__(self, directory):
        self.directory = directory
        self.rows = self.files = self.nbytes = 0
        self.digest = sha256()

    def __enter__(self):
        (self.directory / "PROBABILITY_PRIVATE").mkdir()
        self.stream = (self.directory / "PROBABILITIES_PRIVATE.jsonl").open("xb")
        return self

    def __exit__(self, *details):
        try:
            flush(self.stream)
        finally:
            self.stream.close()

    def write(self, ordinal, image_id, audit, planes):
        candidates = audit["candidates"]
        n = min(100, len(candidates))
        qids = [r["query_id"] for r in candidates]
        require(ordinal == self.files + 1 and image_id == audit["image_id"] and
                len(set(qids)) == len(qids) and all(type(q) is int and 0 <= q < 300 for q in qids) and
                all(r["native_rank"] == i + 1 and r["probability_row"] == (i if i < n else None)
                    for i, r in enumerate(candidates)), "Exact native rank and query lineage")
        require(planes.dtype == np.dtype("<f4") and planes.shape == (n, 160, 160),
                "Exact native FP32 probability shape")
        file = plane_path(ordinal)
        out = BytesIO()
        np.save(out, planes, allow_pickle=False)
        data = out.getvalue()
        with (self.directory / file).open("xb") as stream:
            require(stream.write(data) == len(data), "Complete native probability archive")
            flush(stream)
        row = dict(image_id=image_id, query_ids=qids, probability=dict(
            file=file, bytes=len(data), sha256=sha256(data).hexdigest(), rows=n))
        raw = canonical_line(row)
        require(self.stream.write(raw) == len(raw), "Complete native probability index")
        self.digest.update(raw)
        self.files += 1
        self.rows += n
        self.nbytes += len(data)
        return dict(audit, candidates=[{k: v for k, v in r.items()
                     if k not in ("query_id", "probability_row")} for r in candidates])

    def receipt(self):
        return dict(probability_files=self.files, probability_rows=self.rows,
                    probability_bytes=self.nbytes, probability_index_sha256=self.digest.hexdigest())


def verify_probability_file(directory, ordinal, ref, expected_rows):
    require(set(ref) == {"file", "bytes", "sha256", "rows"} and ref["file"] == plane_path(ordinal)
            and type(ref["bytes"]) is int and 0 < ref["bytes"] <= 11 * 1024**2
            and ref["rows"] == expected_rows, "Exact bounded probability reference")
    path = directory / ref["file"]
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == ref["bytes"],
            "Exact probability file identity")
    raw = path.read_bytes()
    require(len(raw) == ref["bytes"] and sha256(raw).hexdigest() == ref["sha256"],
            "Exact probability bytes")
    stream = BytesIO(raw)
    array = np.load(stream, allow_pickle=False)
    require(not stream.read(1) and array.dtype == np.dtype("<f4")
            and array.shape == (expected_rows, 160, 160) and np.isfinite(array).all()
            and (array.min(initial=0) >= 0) and (array.max(initial=1) <= 1),
            "Exact bounded finite decoder probabilities")
    return array

