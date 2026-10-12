"""Pure boundary/classification/count fixture SOURCE; no numerical libraries.

Future ROOT may run this separately. Saving/AST compilation is not a test run.
"""
import importlib.util
from pathlib import Path


def check(module):
    for threshold in (.5, .75, .95):
        assert module.fp_kind(threshold, threshold) == 'duplicate_or_assignment_conflict'
        assert module.fp_kind(threshold-1e-6, threshold) == 'geometry_candidate'
        assert module.fp_kind(.1, threshold) == 'geometry_candidate'
        assert module.fp_kind(.1-1e-6, threshold) == 'background_candidate'
        assert module.fp_kind(0., threshold) == 'background_candidate'
    assert [module.area_group(v) for v in (0, 1024, 1025, 9216, 9217)] == ['small', 'small', 'medium', 'medium', 'large']
    assert [module.density_group(v) for v in (0, 1, 2, 4, 5)] == ['0', '1', '2-4', '2-4', '>=5']
    rows = [dict(ignored=False, matched_GT=11, FP_type=None),
            dict(ignored=False, matched_GT=0, FP_type='duplicate_or_assignment_conflict'),
            dict(ignored=False, matched_GT=0, FP_type='geometry_candidate'),
            dict(ignored=False, matched_GT=0, FP_type='background_candidate'),
            dict(ignored=True, matched_GT=99, FP_type=None),
            dict(ignored=False, matched_GT=12, FP_type=None)]
    assert module.tally(rows[:1], {11, 12, 13}, 1) == dict(TP=1, FN=2, FP=0, ignored_DT=0, ignored_GT=1,
            duplicate_or_assignment_conflict=0, geometry_candidate=0, background_candidate=0)
    assert module.tally(rows, {11, 12, 13}, 1) == dict(TP=2, FN=1, FP=3, ignored_DT=1, ignored_GT=1,
            duplicate_or_assignment_conflict=1, geometry_candidate=1, background_candidate=1)
    assert module.tally([], set(), 0) == {key: 0 for key in module.COUNT_KEYS}
    for value in (-.1, 1.1, float('nan'), float('inf')):
        try:
            module.fp_kind(value, .5)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Invalid IoU accepted')
    try:
        module.tally([rows[0], rows[0]], {11}, 0)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Duplicate regular match accepted')
    return dict(status='PURE_N06_BOUNDARY_AND_COUNT_FIXTURE_PASS', scientific_libraries_imported=False)


if __name__ == '__main__':
    path = Path(__file__).with_name('run.py')
    spec = importlib.util.spec_from_file_location('_N06_source_component_test', str(path))
    module = importlib.util.module_from_spec(spec)
    # Execute exact SOURCE, avoiding an unrelated cached bytecode module.
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    import json
    print(json.dumps(check(module)))
