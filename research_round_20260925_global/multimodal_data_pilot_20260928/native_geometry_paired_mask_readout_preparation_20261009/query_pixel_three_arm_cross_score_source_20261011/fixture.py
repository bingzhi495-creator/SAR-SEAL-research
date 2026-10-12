"""Small stdlib-only fixture for score crossing; authoring does not run it.

Synthetic rows test identity and copying only. No RLE decoding, COCO evaluation,
scientific artifact access, additive attribution, or deployment claim.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import types


def fixture():
    source = Path(__file__).with_name('cross_score_rows.py')
    module = types.ModuleType('cross_score_rows_fixture_target')
    module.__file__ = str(source)
    exec(compile(source.read_bytes(), str(source), 'exec'), module.__dict__)
    cross_score_rows = module.cross_score_rows

    geometry = []
    for image_id, order in ((17, list(reversed(range(100)))),
                            (29, list(range(17, 100)) + list(range(17)))):
        for query_id in order:
            geometry.append(dict(image_id=image_id, query_id=query_id,
                category_id=1, score=(query_id + 1) / 101.,
                segmentation=dict(size=[3, 5], counts='?' if query_id == 0 else '1>'),
                opaque=dict(values=[image_id, query_id], nested={'keep': True})))
    scores = [dict(image_id=row['image_id'], query_id=row['query_id'],
        category_id=1, score=0. if row['query_id'] == 99 else (100 - row['query_id']) / 100.,
        segmentation={'unused': True}, unused_metadata=['not transferred'])
        for row in reversed(geometry)]
    scores = scores[37:] + scores[:37]
    original_geometry, original_scores = deepcopy(geometry), deepcopy(scores)
    expected = {(row['image_id'], row['query_id']): row['score'] for row in scores}
    crossed = cross_score_rows(geometry, scores)
    assert len(crossed) == 200
    for before, after in zip(geometry, crossed):
        assert set(before) == set(after)
        assert after['score'] == expected[(before['image_id'], before['query_id'])]
        assert all(after[key] == before[key] for key in before if key != 'score')
        assert after is not before and after['segmentation'] is not before['segmentation']
    assert geometry == original_geometry and scores == original_scores
    assert [(r['image_id'], r['query_id']) for r in crossed] == [
        (r['image_id'], r['query_id']) for r in geometry]
    assert all(after['segmentation'] == before['segmentation'] for before, after in zip(geometry, crossed)
        if before['query_id'] == 0)
    crossed[0]['segmentation']['size'][0] = 777
    crossed[0]['opaque']['nested']['keep'] = False
    assert geometry == original_geometry and scores == original_scores

    rejected = []

    def reject(label, changed_geometry, changed_scores):
        before_geometry, before_scores = deepcopy(changed_geometry), deepcopy(changed_scores)
        try:
            cross_score_rows(changed_geometry, changed_scores)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError('Expected rejection: ' + label)
        assert changed_geometry == before_geometry
        # NaN needs its own comparison when checking rejection did not mutate input.
        assert len(changed_scores) == len(before_scores)
        for before, after in zip(before_scores, changed_scores):
            assert set(before) == set(after)
            for key in before:
                if key == 'score' and isinstance(before[key], float) and before[key] != before[key]:
                    assert isinstance(after[key], float) and after[key] != after[key]
                else:
                    assert after[key] == before[key]

    for which in ('geometry', 'scores'):
        g, s = deepcopy(geometry), deepcopy(scores)
        selected = g if which == 'geometry' else s
        selected.append(deepcopy(selected[0]))
        reject('repeated identity in ' + which, g, s)
        g, s = deepcopy(geometry), deepcopy(scores)
        (g if which == 'geometry' else s).pop()
        reject('missing query in ' + which, g, s)
        for query_id in (-1, 100, True, 0.5):
            g, s = deepcopy(geometry), deepcopy(scores)
            (g if which == 'geometry' else s)[0]['query_id'] = query_id
            reject('invalid query in ' + which + ': ' + repr(query_id), g, s)

    g, s = deepcopy(geometry), deepcopy(scores)
    s[0]['category_id'] = 2
    reject('category mismatch', g, s)
    g, s = deepcopy(geometry), deepcopy(scores)
    for row in s:
        if row['image_id'] == 29:
            row['image_id'] = 31
    reject('different complete image sets', g, s)
    for value in (float('nan'), float('inf'), -float('inf'), -0.01, 1.01, True, '0.5'):
        g, s = deepcopy(geometry), deepcopy(scores)
        s[0]['score'] = value
        reject('invalid score: ' + repr(value), g, s)
    for field in ('image_id', 'query_id', 'category_id', 'score'):
        g, s = deepcopy(geometry), deepcopy(scores)
        del s[0][field]
        reject('score row missing ' + field, g, s)
    for field in ('score', 'segmentation'):
        g, s = deepcopy(geometry), deepcopy(scores)
        del g[0][field]
        reject('geometry row missing ' + field, g, s)
    g, s = deepcopy(geometry), deepcopy(scores)
    s[0]['image_id'] = True
    reject('boolean image identity', g, s)
    reject('empty populations', [], [])
    assert geometry == original_geometry and scores == original_scores
    return dict(status='CROSS_SCORE_ROWS_SYNTHETIC_STDLIB_PASS',
        images=2, queries_per_image=100, crossed_rows=200,
        unordered_score_identity_join=True, only_score_changed=True,
        geometry_order_and_empty_RLE_retained=True, independent_deepcopy=True,
        original_inputs_unchanged=True, rejected_cases=rejected,
        scientific_artifacts_read=False, RLE_decoded=False, COCO_computed=False,
        additive_attribution_claimed=False, deployment_claimed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner-execute', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(fixture(), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
