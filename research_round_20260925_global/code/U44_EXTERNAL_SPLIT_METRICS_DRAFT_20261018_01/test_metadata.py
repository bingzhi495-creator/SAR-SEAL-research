"""Standard-library metadata only; opaque tokens are not valid mask fixtures."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import split_metrics


CONTRACT_PATH = (Path(__file__).resolve().parents[3] / 'github_release_20261011' /
    'research_round_20260925_global/code/N28_FIXED24_DEV_EVAL_DRAFT_20261017_01/'
    'population_contract.py')
_spec = importlib.util.spec_from_file_location('u44_original_population_contract', CONTRACT_PATH)
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


def population_fixture():
    manifest = [dict(dataset_index=i, image_id=7000+3*i) for i in range(1962)]
    native = [dict(image_id=row['image_id'], query_id=q, category_id=1,
                   score=0.0 if q == 0 else 0.5,
                   segmentation={'opaque_fixture': 'native'},
                   bbox=[10, 20, 3, 4], area=12)
              for row in manifest for q in reversed(range(100))]
    full = [dict(row, segmentation={'opaque_fixture': 'FULL'}) for row in native]
    methods = ['native', 'PLAIN', 'FULL', 'R']
    ranges = [[0, 1000], [1000, 1962]]
    shards = [dict(dataset_range=span,
                   predictions={name: (full if name == 'FULL' else native)[a*100:b*100]
                                for name in methods}) for span in ranges for a, b in [span]]
    return manifest, native, methods, ranges, shards


class ExternalMetadataTests(unittest.TestCase):
    def test_complete_population_keeps_original_order_and_all_metadata(self):
        manifest, native, methods, ranges, shards = population_fixture()
        merged = split_metrics.merge_population(manifest, methods, ranges, shards, contract)
        self.assertEqual(set(merged), set(methods))
        self.assertTrue(all(len(rows) == 196200 for rows in merged.values()))
        self.assertEqual([(r['image_id'], r['query_id']) for r in merged['native']],
                         [(r['image_id'], r['query_id']) for r in native])
        self.assertEqual(merged['FULL'][0]['segmentation'], {'opaque_fixture': 'FULL'})
        self.assertEqual(merged['FULL'][-1]['bbox'], [10, 20, 3, 4])
        self.assertEqual(merged['FULL'][-1]['area'], 12)
        self.assertEqual(merged['FULL'][-1]['score'], 0.0)
        merged['FULL'][0]['bbox'][0] = 999
        self.assertEqual(shards[0]['predictions']['FULL'][0]['bbox'][0], 10)

    def test_ranges_required_methods_and_original_query_metadata_cannot_drift(self):
        manifest, native, methods, ranges, shards = population_fixture()
        with self.assertRaises(ValueError):
            split_metrics.merge_population(manifest, methods, ranges+[[1962, 1962]], shards, contract)
        with self.assertRaises(ValueError):
            split_metrics.merge_population(manifest, methods, [[0, 1000], [999, 1962]], shards, contract)
        with self.assertRaises(ValueError):
            split_metrics.merge_population(manifest, methods, [[False, 1000], [1000, 1962]], shards, contract)
        missing = [dict(shards[0], predictions={k: v for k, v in shards[0]['predictions'].items() if k != 'R'}), shards[1]]
        with self.assertRaises(ValueError):
            split_metrics.merge_population(manifest, methods, ranges, missing, contract)
        drift = dict(shards[0]['predictions'])
        drift['FULL'] = list(drift['FULL'])
        drift['FULL'][0], drift['FULL'][1] = drift['FULL'][1], drift['FULL'][0]
        with self.assertRaises(ValueError):
            split_metrics.merge_population(manifest, methods, ranges, [dict(shards[0], predictions=drift), shards[1]], contract)
        drift['FULL'] = list(shards[0]['predictions']['FULL'])
        drift['FULL'][0] = dict(drift['FULL'][0], area=13)
        with self.assertRaises(ValueError):
            split_metrics.merge_population(manifest, methods, ranges, [dict(shards[0], predictions=drift), shards[1]], contract)

    def test_evaluation_rejects_unlocked_method_and_unclosed_official_before_any_COCO(self):
        manifest, native, methods, ranges, shards = population_fixture()
        calls = []

        def forbidden_COCO(*args, **kwargs):
            calls.append('COCO')
            raise AssertionError('No scientific constructor may run in this metadata fixture')

        class ClosureStop(RuntimeError):
            pass

        def blocked_closed_ref(reference):
            calls.append(reference)
            raise ClosureStop('Deliberate source-closure stop, without opening a file')

        sources = {key: dict(path='never_open_'+key, bytes=1, sha256='0'*64)
                   for key in ('official', 'coco', 'cocoeval')}
        options = dict(COCO=forbidden_COCO, COCOeval=forbidden_COCO,
                       official=SimpleNamespace(), module_sources=sources,
                       test_ref={'path': 'never_open_TEST'}, closed_ref=blocked_closed_ref,
                       population_contract=contract)
        with self.assertRaises(ValueError):
            split_metrics.evaluate_one_method('not_prelocked', methods, manifest, native, **options)
        with self.assertRaises(ValueError):
            split_metrics.evaluate_one_method('native', methods, manifest, native, **options)
        self.assertEqual(calls, [])
        sources['official'] = dict(path='never_open_original_official', bytes=49923,
            sha256='2357929fcd38ffc3020c5b8d204ee371da8da3972ff2e2e03398c8225c55e916')
        with self.assertRaises(ClosureStop):
            split_metrics.evaluate_one_method('native', methods, manifest, native, **options)
        self.assertEqual(calls, [sources['official']])


if __name__ == '__main__':
    unittest.main()
