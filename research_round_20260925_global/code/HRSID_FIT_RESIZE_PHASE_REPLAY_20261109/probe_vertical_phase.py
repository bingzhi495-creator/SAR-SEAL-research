"""CPU synthetic check of sampling direction and common strip survival."""

import json

import numpy as np
import torch
from torch.nn.functional import interpolate


def sampled_rows(source, output_height, mode, *, reflected=False):
    if reflected:
        source = source.flip(-2)
    output = interpolate(source, size=(output_height, 1), mode=mode)
    if reflected:
        output = output.flip(-2)
    return output.numpy().astype(np.int64).ravel()


def run():
    height, output_height = 800, 576
    source = torch.arange(height, dtype=torch.float32).reshape(1, 1, height, 1)
    observed = [
        sampled_rows(source, output_height, "nearest"),
        sampled_rows(source, output_height, "nearest-exact"),
        sampled_rows(source, output_height, "nearest", reflected=True),
    ]
    output_rows = np.arange(output_height)
    ideal = [
        output_rows * height // output_height,
        (2 * output_rows + 1) * height // (2 * output_height),
        ((output_rows + 1) * height - 1) // output_height,
    ]
    backprojected = (output_rows + 0.5) * height / output_height
    result = dict(
        torch=torch.__version__,
        scale=[height, output_height],
        index_disagreements=[int(np.count_nonzero(a != b)) for a, b in zip(observed, ideal)],
        rows=[],
    )
    for thickness in (1, 2, 4, 8, 16):
        starts = np.arange(height - thickness + 1)
        masks = [
            (indices[None, :] >= starts[:, None]) & (indices[None, :] < starts[:, None] + thickness)
            for indices in observed
        ]
        counts = [mask.sum(axis=1) for mask in masks]
        shared = np.logical_and.reduce([count > 0 for count in counts])
        original_centroids = starts[shared] + thickness / 2
        errors = [
            (mask[shared] * backprojected).sum(axis=1) / count[shared] - original_centroids
            for mask, count in zip(masks, counts)
        ]
        result["rows"].append(dict(
            height=thickness,
            starts=len(starts),
            nonempty=[int(np.count_nonzero(count)) for count in counts],
            shared=int(shared.sum()),
            mean_signed_error=[round(float(np.mean(error)), 6) for error in errors],
            mean_absolute_error=[round(float(np.mean(np.abs(error))), 6) for error in errors],
        ))
    return result


if __name__ == "__main__":
    print(json.dumps(run()))
