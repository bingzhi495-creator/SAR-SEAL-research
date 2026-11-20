"""Fixed query-conditioned channel controls for an existing sealed point roster.

No models, labels, fitted statistics, files or learned parameters. Inputs F/Q may
be learned frozen representations. This primitive makes no empirical claim.
"""
from collections import defaultdict
from hashlib import sha256
import json

import numpy as np

PROJECTION_SEED = 5241
PROJECTION_SHA256 = '96db66d62d88541d31254f408a18330c0b03f423364b64d2f50cdf19ce466448'
DIVISOR = 16
PERMUTATION_RECIPE = 'candidate-sign-layer-pixel-sha256-single-cycle-v1'
CONTROL_NAMES = ('zero', 'ordinary_centered', 'query_conditioned_centered', 'query_conditioned_permuted')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def rademacher_matrix():
    """Fresh immutable int8 matrix, using only a private PCG64 generator."""
    bits = np.random.Generator(np.random.PCG64(PROJECTION_SEED)).integers(
        0, 2, size=(256, 16), dtype=np.int8)
    matrix = bits * 2 - 1
    need(matrix.dtype == np.int8 and sha256(matrix.tobytes(order='C')).hexdigest() == PROJECTION_SHA256,
         'Exact fixed Rademacher matrix identity')
    matrix.setflags(write=False)
    return matrix


def _float_input(value, shape, name):
    array = np.asarray(value)
    need(array.shape == shape and array.dtype.kind == 'f' and array.dtype.itemsize in (4, 8)
         and np.isfinite(array).all(), 'Finite float32/float64 ' + name + ' with exact shape')
    return array


def _project(values, projection):
    with np.errstate(over='ignore', invalid='ignore'):
        centered = values - values.mean(axis=-1, dtype=np.float64, keepdims=True)
        projected = centered @ projection
        output = projected.astype(np.float32)
    need(np.isfinite(centered).all() and np.isfinite(projected).all() and np.isfinite(output).all(),
         'Finite float64 channel centering/projection and float32 output, without clipping')
    return output


def _permutation(coordinates, layers, candidate_id):
    flat_coords, flat_layers = coordinates.reshape(-1), layers.reshape(-1)
    groups = defaultdict(list)
    for i, layer in enumerate(flat_layers):
        groups[(i % 2, int(layer))].append(i)
    source = np.arange(len(flat_coords), dtype=np.int64)
    for (sign, layer), members in groups.items():
        def key(i):
            pixel = int(flat_coords[i])
            identity = [PERMUTATION_RECIPE, PROJECTION_SEED, *candidate_id, sign, layer, pixel]
            raw = json.dumps(identity, separators=(',', ':'), allow_nan=False).encode('ascii')
            return sha256(raw).digest(), pixel
        ordered = sorted(members, key=key)
        if len(ordered) > 1:
            for receiver, donor in zip(ordered, ordered[1:] + ordered[:1]):
                source[receiver] = donor
    return source.reshape(coordinates.shape)


def point_feature_controls(old14, spatial, query, coordinates, boundary_layers, *, candidate_id):
    """Return four [K,2,30] controls for one candidate, with 0 <= K <= 32.

    Axis 1 is fixed [add, drop]. F is [K,2,256], Q is [256]. Both channel
    centerings are per point over its 256 channels. Only the new conditional
    16-vector is permuted, within candidate/sign/layer, never old14 or coordinates.
    candidate_id is the immutable (image_id, native_rank, native_query_id) tuple.
    """
    coordinates = np.asarray(coordinates)
    layers = np.asarray(boundary_layers)
    need(coordinates.ndim == 2 and coordinates.shape[1] == 2 and len(coordinates) <= 32
         and coordinates.dtype.kind in 'iu' and np.all(coordinates >= 0)
         and np.unique(coordinates).size == coordinates.size,
         'Unique nonnegative sealed [K,2] native indices, at most 32 complete pairs')
    need(layers.shape == coordinates.shape and layers.dtype.kind in 'iu' and np.all(layers >= 0),
         'One explicit nonnegative integer boundary layer per endpoint')
    need(type(candidate_id) is tuple and len(candidate_id) == 3
         and all(type(v) is int and v >= 0 for v in candidate_id) and candidate_id[1] > 0,
         'Immutable original integer image/rank/query identity')
    k = len(coordinates)
    old = _float_input(old14, (k, 2, 14), 'old14')
    need(old.dtype == np.float32, 'Original old14 must already be float32')
    F = _float_input(spatial, (k, 2, 256), 'spatial F').astype(np.float64, copy=False)
    Q = _float_input(query, (256,), 'query Q').astype(np.float64, copy=False)
    matrix = rademacher_matrix()
    projection = matrix.astype(np.float64) / DIVISOR
    ordinary = _project(F, projection)
    with np.errstate(over='ignore', invalid='ignore'):
        product = F * Q
    need(np.isfinite(product).all(), 'Finite float64 Q*F before channel centering')
    conditional = _project(product, projection)
    source = _permutation(coordinates, layers, candidate_id)
    permuted = conditional.reshape(-1, 16)[source.reshape(-1)].reshape(k, 2, 16)
    residuals = dict(zero=np.zeros((k, 2, 16), dtype=np.float32), ordinary_centered=ordinary,
                     query_conditioned_centered=conditional, query_conditioned_permuted=permuted)
    features = {name:np.concatenate((old, residuals[name]), axis=-1) for name in CONTROL_NAMES}
    return dict(features=features, residuals=residuals, coordinates=coordinates.copy(),
                boundary_layers=layers.copy(), permutation_source_flat_indices=source,
                metadata=dict(schema='RFDETR-fixed-query-channel-residual-controls-v1',
                              channels=256, projected_channels=16, old_channels=14, head_channels=30,
                              projection_seed=PROJECTION_SEED, projection_generator='PCG64',
                              projection_matrix_dtype='int8', projection_matrix_sha256=PROJECTION_SHA256,
                              projection_divisor=DIVISOR, arithmetic_dtype='float64', output_dtype='float32',
                              centering='per-point 256-channel mean', learned_projection=False,
                              fitted_dataset_statistics=False, global_rng_consumed=False,
                              permutation_recipe=PERMUTATION_RECIPE,
                              permutation_scope='same candidate, add/drop sign and boundary layer',
                              singleton_points_kept=int(np.count_nonzero(source.reshape(-1) == np.arange(k*2))),
                              point_order_unchanged=True, old14_unchanged=True))
