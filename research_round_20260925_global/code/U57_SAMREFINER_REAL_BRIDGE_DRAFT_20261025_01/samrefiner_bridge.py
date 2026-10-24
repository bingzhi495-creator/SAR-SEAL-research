"""Unbound bridge to a caller-supplied, source-verified author function."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class IndexedMasks:
    query_ids: Tuple[int, ...]
    masks: Tuple[object, ...]


def _binary_array(masks, shape):
    if not isinstance(masks, (np.ndarray, list, tuple)):
        raise ValueError('Only NumPy or nested Python masks are accepted')
    array = np.asarray(masks)
    if array.shape != shape or array.dtype.kind not in ('b', 'i', 'u'):
        raise ValueError('Mask shape or integral/bool dtype violates the declared frame')
    if not np.logical_or(array == 0, array == 1).all():
        raise ValueError('Masks must already contain exactly zero or one')
    return array


def make_samrefiner_backend(image_path, sam, author_refiner):
    def backend(*, query_ids, masks, image_size, iters):
        ids = tuple(query_ids)
        size = tuple(image_size)
        if type(iters) is not int or iters != 5:
            raise ValueError('This bridge fixes the author default at five iterations')
        if len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
            raise ValueError('Declared original image size must be positive integer H,W')
        if not 1 <= len(ids) <= 100 or len(set(ids)) != len(ids):
            raise ValueError('Require one to one hundred unique submitted query IDs')
        shape = (len(ids),) + size
        checked = _binary_array(masks, shape)
        batch = np.array(checked, dtype=np.uint8, order='C', copy=True)
        output = author_refiner(
            image_path, batch, sam,
            resize_transform=None, use_point=True, use_box=True, use_mask=True,
            add_neg=True, iters=5, margin=0.0, gamma=4.0, strength=30,
            use_samhq=False, ddp=False, is_train=False)
        if not isinstance(output, tuple) or len(output) != 3:
            raise ValueError('Author function must return its three-element tuple')
        if type(output[0]) is not np.ndarray or output[0].dtype != np.uint8:
            raise ValueError('Author refined masks must be an actual uint8 NumPy array')
        checked_output = _binary_array(output[0], shape)
        refined = tuple(tuple(tuple(row) for row in mask)
                        for mask in checked_output.tolist())
        return IndexedMasks(ids, refined)

    return backend
