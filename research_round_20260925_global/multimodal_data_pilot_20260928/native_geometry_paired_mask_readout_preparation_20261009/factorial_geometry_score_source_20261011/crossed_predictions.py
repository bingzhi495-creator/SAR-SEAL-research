"""Pair qualified exports for a descriptive geometry/score factorial diagnostic.

This stdlib helper does not decode RLE, evaluate AP, read files, or certify
checkpoint/qualification lineage. The owned caller must certify those inputs.
"""
import copy
import math


def _rle_schema(row, label):
    if set(row) != {'image_id', 'query_id', 'category_id', 'score', 'segmentation'}:
        raise ValueError(label + ': exact prediction fields required')
    if type(row['category_id']) is not int or row['category_id'] != 1:
        raise ValueError(label + ': single category1 required')
    score = row['score']
    if type(score) not in (int, float) or not 0 <= score <= 1 or not math.isfinite(score):
        raise ValueError(label + ': finite probability score in [0,1] required')
    rle = row['segmentation']
    if not isinstance(rle, dict) or set(rle) != {'size', 'counts'}:
        raise ValueError(label + ': compressed RLE fields required')
    size, counts = rle['size'], rle['counts']
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError(label + ': positive integer original dimensions required')
    if not isinstance(counts, str) or not counts:
        raise ValueError(label + ': nonempty ASCII compressed counts required')
    try:
        counts.encode('ascii')
    except UnicodeEncodeError as error:
        raise ValueError(label + ': non-ASCII compressed RLE') from error


def _population(rows, expected, query_count, label):
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError(label + ': incomplete or extra query population')
    mapped = {}
    for row in rows:
        if not isinstance(row, dict) or not {'image_id', 'query_id'} <= set(row):
            raise ValueError(label + ': prediction identity missing')
        _rle_schema(row, label)
        image, query = row['image_id'], row['query_id']
        if type(image) is not int or type(query) is not int or not 0 <= query < query_count:
            raise ValueError(label + ': exact integer image/query identity required')
        key = (image, query)
        if key not in expected or key in mapped:
            raise ValueError(label + ': unexpected or duplicate query identity')
        mapped[key] = row
    if set(mapped) != expected:
        raise ValueError(label + ': missing query identity')
    return mapped


def assemble_crossed_predictions(native_official, posterior_official,
                                 posterior_budget, *, image_ids, query_count=100):
    """Return four cells in unchanged native order, including empty masks.

    S1 belongs to corrected geometry G1. Its reuse on G0 is a counterfactual
    diagnostic, so scores are neither recomputed nor reset for empty G0 masks.
    """
    if type(query_count) is not int or query_count <= 0:
        raise ValueError('Positive integer query_count required')
    if not isinstance(image_ids, (list, tuple)) or not image_ids or any(type(i) is not int for i in image_ids):
        raise ValueError('Explicit nonempty integer image population required')
    if len(set(image_ids)) != len(image_ids):
        raise ValueError('Duplicate requested image identity')
    expected = {(image, query) for image in image_ids for query in range(query_count)}
    native_map = _population(native_official, expected, query_count, 'native_official')
    posterior = _population(posterior_official, expected, query_count, 'posterior_official')
    budget = _population(posterior_budget, expected, query_count, 'posterior_budget')
    image_shapes = {}
    for key, native in native_map.items():
        size = native['segmentation']['size']
        original = image_shapes.setdefault(key[0], size)
        if size != original or posterior[key]['segmentation']['size'] != original or budget[key]['segmentation']['size'] != original:
            raise ValueError('Original dimensions differ within or across an image')
        if posterior[key]['segmentation'] != budget[key]['segmentation']:
            raise ValueError('Posterior official and budget geometry must be identical')
    cells = {name: [] for name in ('F00', 'F10', 'F01', 'F11')}
    for native in native_official:
        key = (native['image_id'], native['query_id'])
        for name, geometry, score in (
            ('F00', native, native['score']),
            ('F10', posterior[key], native['score']),
            ('F01', native, budget[key]['score']),
            ('F11', posterior[key], budget[key]['score']),
        ):
            prediction = copy.deepcopy(geometry)
            prediction['score'] = score
            cells[name].append(prediction)
    return cells


def summarize_factorial_AP(main_AP):
    """Describe main-AP fractions at a single checkpoint; never infer causality."""
    if not isinstance(main_AP, dict) or set(main_AP) != {'F00', 'F10', 'F01', 'F11'}:
        raise ValueError('Exactly four main AP fractions required')
    if any(type(value) not in (int, float) for value in main_AP.values()):
        raise ValueError('Numeric main AP fractions required')
    if any(value == -1 for value in main_AP.values()):
        raise ValueError('undefined AP, decomposition unavailable')
    if any(not 0 <= value <= 1 or not math.isfinite(value) for value in main_AP.values()):
        raise ValueError('Finite main AP fractions in [0,1] required')
    f00, f10, f01, f11 = (main_AP[name] for name in ('F00', 'F10', 'F01', 'F11'))
    total = f11 - f00
    effects = dict(geometry_at_S0=f10-f00, score_at_G1=f11-f10,
                   score_at_G0=f01-f00, geometry_at_S1=f11-f01)
    return dict(
        AP=dict(main_AP), units='AP fraction, not percentage points',
        effects=effects, total_delta=total, interaction=f11-f10-f01+f00,
        decomposition_residuals=dict(
            geometry_then_score=effects['geometry_at_S0']+effects['score_at_G1']-total,
            score_then_geometry=effects['score_at_G0']+effects['geometry_at_S1']-total),
        unique_causal_attribution=False, contribution_percentages=None,
        interpretation='Reference-dependent descriptive effects for nonlinear AP; not unique causal contributions.',
        counterfactual_scores='S1 was computed for G1. Reusing S1 on G0 does not calibrate quality for G0; crossed cells are diagnostics, not deployment methods.',
    )
