"""Cross qualified geometry and scores by actual image/query identity.

Pure stdlib row manipulation: no filtering, NMS, RLE interpretation, metrics,
scientific I/O, additive attribution, or deployment claim.
"""

from copy import deepcopy
import math


def _index(rows, label, geometry):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError(label + ' must be a nonempty list or tuple of rows')
    index, images = {}, {}
    required = ('image_id', 'query_id', 'category_id', 'score')
    for row in rows:
        if not isinstance(row, dict) or any(key not in row for key in required):
            raise ValueError(label + ' row lacks image/query/category/score fields')
        if geometry and 'segmentation' not in row:
            raise ValueError('Geometry row lacks segmentation')
        image_id, query_id, category_id = row['image_id'], row['query_id'], row['category_id']
        if type(image_id) is not int or type(category_id) is not int:
            raise ValueError(label + ' image/category identities must be integers, not booleans')
        if type(query_id) is not int or not 0 <= query_id < 100:
            raise ValueError(label + ' query identity must be an integer from 0 through 99')
        key = (image_id, query_id)
        if key in index:
            raise ValueError(label + ' has a repeated image/query identity')
        if not geometry:
            score = row['score']
            if type(score) not in (int, float) or not 0 <= score <= 1 or not math.isfinite(score):
                raise ValueError('Replacement score must be a finite numeric value in [0, 1]')
        index[key] = row
        images.setdefault(image_id, set()).add(query_id)
    all_queries = set(range(100))
    if any(queries != all_queries for queries in images.values()):
        raise ValueError(label + ' must retain each image with exactly all 100 query identities')
    return index, set(images)


def cross_score_rows(geometry_rows, score_rows):
    """Return a deep-copied list in geometry order with only scores replaced.

    Both nonempty populations must contain exactly one row for every actual
    query 0..99 per image, have identical image sets, and agree on category at
    each (image_id, query_id). Replacement scores must be finite in [0, 1].
    Geometry rows require an existing score and segmentation; segmentation and
    every other value are copied unchanged, including empty RLE and metadata.
    Score-row geometry/metadata are not transferred. Neither input is mutated.
    """
    geometry, geometry_images = _index(geometry_rows, 'Geometry', True)
    scores, score_images = _index(score_rows, 'Scores', False)
    if geometry_images != score_images or geometry.keys() != scores.keys():
        raise ValueError('Geometry and score rows must have a strict one-to-one identity join')
    for key, row in geometry.items():
        if row['category_id'] != scores[key]['category_id']:
            raise ValueError('Geometry and score category differ at the same image/query identity')
    crossed = deepcopy(list(geometry_rows))
    for row in crossed:
        row['score'] = scores[(row['image_id'], row['query_id'])]['score']
    return crossed
