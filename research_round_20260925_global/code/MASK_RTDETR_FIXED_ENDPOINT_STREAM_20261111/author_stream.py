"""Write and consume one immutable author box/mask pair per COCO image."""

from hashlib import sha256
import json
import os

from MASK_RTDETR_AUTHOR_MASK_ROWS_20261111.author_mask_rows import prepare_mask_rows


def write_author_stream(path, image_shapes, blocks, *, per_image=100):
    """Write an ordered native stream and return its SHA-256 after every image closes."""
    if not image_shapes:
        raise ValueError("An ordered image roster is required")
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_name(path.name + ".partial")
    digest = sha256()
    with partial.open("xb") as stream:
        for (image_id, shape), block in zip(image_shapes.items(), blocks, strict=True):
            prepare_mask_rows(block, {image_id: shape}, expected_per_image=per_image)
            raw = (json.dumps(block, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            stream.write(raw)
            digest.update(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    partial.rename(path)
    return digest.hexdigest()


def read_mask_stream(path, image_shapes, expected_sha256, *, per_image=100):
    """Validate the closed, ordered stream and keep only native segmentation rows."""
    if not image_shapes:
        raise ValueError("An ordered image roster is required")
    digest = sha256()
    predictions = []
    with path.open("rb") as stream:
        for (image_id, shape), raw in zip(image_shapes.items(), stream, strict=True):
            if not raw.endswith(b"\n"):
                raise ValueError("Truncated native prediction block")
            digest.update(raw)
            predictions.extend(prepare_mask_rows(json.loads(raw), {image_id: shape}, expected_per_image=per_image))
    if digest.hexdigest() != expected_sha256:
        raise ValueError("Prediction stream does not match the closed export digest")
    return predictions
