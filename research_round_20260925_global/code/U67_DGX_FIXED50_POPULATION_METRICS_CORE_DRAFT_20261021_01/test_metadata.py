"""Synthetic JSON/stub checks only; no actual COCO, masks, or learned outputs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parent
PUBLIC = Path('E:/SAR/seal/github_release_20261011/research_round_20260925_global/code')
ORDINARY = Path('E:/SAR/seal/research_round_20260925_global/code')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def fixture():
    events = []
    population = load(PUBLIC/'N28_FIXED24_DEV_EVAL_DRAFT_20261017_01/population_contract.py', 'original_population')
    old = load(ORDINARY/'U40_FIXED50_DEV_METRICS_DRAFT_20261017_01/population_metrics.py', 'original_metrics')
    ids = list(range(187))
    # Deliberately reverse original query export order; natural query identities
    # are not a license to sort/renumber these JSON candidates.
    native = [dict(image_id=i, query_id=q, category_id=1, score=-0.0 if q == 0 else q/100,
                   bbox=[], area=0, segmentation=dict(size=[1, 1], counts='1'))
              for i in ids for q in reversed(range(100))]
    strong = deepcopy(native)
    for value in strong:
        value['segmentation']['counts'] = 'synthetic_R'
    endpoint = dict(schema='U58-U57-DGX-fixed-endpoint-metadata-v1', mode='FORMAL',
        formal_endpoint=True, cursor=dict(global_updates=37050, completed_epochs=50, within_epoch=0),
        run_id='synthetic-only', metadata_only=True, numerical_runtime_qualified=False,
        checkpoint_refs={'PLAIN': 'synthetic-P', 'FULL': 'synthetic-F'}, pair_commit_ref='synthetic-pair')
    shards = []
    for index, span in enumerate(((0, 47), (47, 94), (94, 141), (141, 187))):
        first, stop = span
        original = deepcopy(native[first*100:stop*100])
        arms = {}
        for arm in ('PLAIN', 'FULL'):
            predictions = deepcopy(original)
            for value in predictions:
                value['segmentation']['counts'] = 'synthetic_'+arm
            support = [dict(dataset_index=first+n//100, image_id=v['image_id'], query_id=v['query_id'],
                            Omega=dict(size=[1, 1], counts='original_native_L_support'))
                       for n, v in enumerate(original)] if arm == 'PLAIN' else []
            arms[arm] = dict(status='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE',
                predictions=predictions, identity_predictions=deepcopy(original), Omega_support=support,
                cost=dict(encode_calls=stop-first, readout_calls=25*(stop-first), queries=(stop-first)*100, synthetic_timing=3.0),
                proofs=[dict(dataset_index=i, image_id=i, queries=100, zero_residual_native_L_exact=True,
                    native_RLE_and_prediction_dictionaries_exact=True, original_full_S0_class_order_preserved=True)
                    for i in range(first, stop)])
        shards.append(dict(schema='U66-DGX-fixed50-DEV-shard-pending-v1',
            status='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE', endpoint=deepcopy(endpoint),
            shard_index=index, dataset_range=list(span), images=stop-first, queries=(stop-first)*100,
            fixed_epoch=50, updates_per_arm=37050, query_batch=4, source_refs={'synthetic': 'same'},
            arms=arms, predictions={a: arms[a]['predictions'] for a in arms},
            identity_predictions=arms['PLAIN']['identity_predictions'], Omega_support=arms['PLAIN']['Omega_support'],
            cost={a: arms[a]['cost'] for a in arms}, qualified=False, AP=False, DEV=True,
            DEV_technical_input=True, TEST=False, new_training=False, DEV_metrics_computed=False,
            whole_population_predicted=False, GT_used_for_prediction=False, GT_rasterization_executed=False,
            original_full_S0_class_order_preserved=True))
    reference = {method: dict(stats=[0.25 if method == 'native_official' else 0.3]*11+[-1.0],
        iouType='segm', maxDets=[1, 10, 100], original_GT_preserved=True,
        score_filter=None, official_COCO_crowd_ignore_semantics=True) for method in ('native_official', 'R')}
    computed_reference = deepcopy(reference)
    native_evaluation = object()
    official = SimpleNamespace(coco_metrics=lambda *a: None)
    cocoeval, masks = SimpleNamespace(COCOeval=lambda *a: None), SimpleNamespace(decode=lambda *a: None)

    def official_metrics(np, c, helper, evaluator, coco, values, received_ids):
        assert helper is official and evaluator is cocoeval.COCOeval and received_ids == ids
        counts = values[0]['segmentation']['counts']
        events.append(('official', counts, len(values)))
        method = 'native_official' if counts == '1' else 'R' if counts == 'synthetic_R' else counts
        result = deepcopy(computed_reference[method] if method in computed_reference else computed_reference['native_official'])
        return result, {'synthetic_only': True}, native_evaluation if method == 'native_official' else object()

    def diagnose(np, mask_api, coco, evaluation, predictions, supports):
        assert mask_api is masks and evaluation is native_evaluation
        assert set(predictions) == {'native_official', 'R', 'SELECTIVE', 'WHOLE_NODE'}
        assert predictions['SELECTIVE'][0]['segmentation']['counts'] == 'synthetic_FULL'
        assert predictions['WHOLE_NODE'][0]['segmentation']['counts'] == 'synthetic_PLAIN'
        assert len(supports) == 18700 and supports[0]['Omega']['counts'] == 'original_native_L_support'
        events.append(('geometry', len(supports)))
        return dict(summary=dict(schema='N28-original-native-fixed-pair-geometry-summary-v1',
            scope=dict(total_DEV_GT=380, selected_native_nonignored_GT=0,
                conditioned_on_original_native_matched_GT=True, private_pair_records_not_for_publication=True),
            methods={k: {'synthetic_only': True} for k in predictions},
            new_method_GT_pair_selection=False, AP_computed=False), private_pair_records=[])

    modules = dict(population_contract=population, eval_body=SimpleNamespace(official_metrics=official_metrics),
        official=official, cocoeval=cocoeval, mask_api=masks,
        geometry_diagnostic=SimpleNamespace(diagnose=diagnose), population_metrics=old)
    c = SimpleNamespace(check=lambda: events.append(('check',)))
    coco = SimpleNamespace(imgs={i: {} for i in ids}, anns={i: {} for i in range(380)}, cats={1: {}})
    return dict(np=object(), c=c, shards=shards, native=native, strong=strong,
                reference_COCO=reference, coco=coco, ids=ids, modules=modules), events


class WiringTests(unittest.TestCase):
    def test_four_shards_original_helpers_and_private_geometry(self):
        core = load(HERE/'population_metrics_core.py', 'u67')
        args, events = fixture()
        before = json.dumps(args['shards'], sort_keys=True)
        result = core.evaluate_u66_population(**args)
        summary = result['summary']
        self.assertEqual([e[1:] for e in events if e[0] == 'official'],
                         [('1', 18700), ('synthetic_R', 18700), ('synthetic_PLAIN', 18700), ('synthetic_FULL', 18700)])
        self.assertEqual(len([e for e in events if e[0] == 'geometry']), 1)
        self.assertEqual(summary['queries'], 18700)
        self.assertFalse(summary['qualified'])
        self.assertFalse(summary['ROOT_actual_terminal_self_certified'])
        self.assertTrue(summary['AP'])
        self.assertFalse(summary['standalone_FPS_claimed'])
        self.assertNotIn('private_pair_records', summary)
        self.assertEqual(result['private_pair_records'], [])
        self.assertEqual(set(summary['geometry']['methods']), {'native_official', 'R', 'PLAIN', 'FULL'})
        self.assertEqual(summary['Delta_COCO_stats']['FULL_minus_PLAIN'][-1], None)
        self.assertEqual([r['dataset_range'] for r in summary['GPU_costs_by_shard']],
                         [[0, 47], [47, 94], [94, 141], [141, 187]])
        self.assertEqual(before, json.dumps(args['shards'], sort_keys=True))

    def test_exact_endpoint_ranges_and_real_prediction_status_required(self):
        core = load(HERE/'population_metrics_core.py', 'u67')
        args, events = fixture()
        args['shards'][2]['endpoint']['checkpoint_refs']['FULL'] = 'different-pair'
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            core.evaluate_u66_population(**args)
        self.assertFalse(any(e[0] == 'official' for e in events))
        args['shards'][2]['endpoint'] = deepcopy(args['shards'][0]['endpoint'])
        for shard in args['shards']:
            shard['endpoint']['cursor']['completed_epochs'] = 24
        with self.assertRaisesRegex(ValueError, 'E50'):
            core.evaluate_u66_population(**args)
        for shard in args['shards']:
            shard['endpoint']['cursor']['completed_epochs'] = 50
        args['shards'][0]['status'] = 'READER_VALIDATION_COMPLETE'
        with self.assertRaisesRegex(ValueError, 'prediction'):
            core.evaluate_u66_population(**args)
        args['shards'][0]['status'] = 'PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE'
        args['shards'][3]['dataset_range'] = [141, 186]
        with self.assertRaisesRegex(ValueError, 'range'):
            core.evaluate_u66_population(**args)

    def test_entire_native_and_strong_COCO_dictionaries_must_reproduce(self):
        core = load(HERE/'population_metrics_core.py', 'u67')
        args, events = fixture()
        # AP (stats[0]) is unchanged; the last undefined stat still matters.
        args['reference_COCO']['R']['stats'][11] = -0.9
        with self.assertRaisesRegex(ValueError, 'complete.*COCO12'):
            core.evaluate_u66_population(**args)
        self.assertFalse(any(e[0] == 'geometry' for e in events))
        args['reference_COCO']['R']['stats'][11] = -1.0
        args['reference_COCO']['native_official']['maxDets'] = [1, 10, 99]
        with self.assertRaisesRegex(ValueError, 'complete.*COCO12'):
            core.evaluate_u66_population(**args)

    def test_exact_original_API_and_original_failures_propagate(self):
        core = load(HERE/'population_metrics_core.py', 'u67')
        args, events = fixture()
        original = args['modules']['eval_body']
        args['modules']['eval_body'] = SimpleNamespace(evaluate_population=lambda: 'wrong-old-entry')
        with self.assertRaisesRegex(ValueError, 'official_metrics'):
            core.evaluate_u66_population(**args)
        self.assertFalse(any(e[0] == 'official' for e in events))
        marker = RuntimeError('original scientific helper failure')

        def fail(*args):
            raise marker

        args['modules']['eval_body'] = SimpleNamespace(official_metrics=fail)
        with self.assertRaises(RuntimeError) as caught:
            core.evaluate_u66_population(**args)
        self.assertIs(caught.exception, marker)
        self.assertFalse(any(e[0] == 'geometry' for e in events))
        args['modules']['eval_body'] = original

    def test_original_support_identity_aliases_and_scores_are_not_reinterpreted(self):
        core = load(HERE/'population_metrics_core.py', 'u67')
        args, events = fixture()
        shard = args['shards'][0]
        shard['Omega_support'][0]['query_id'] = 0
        with self.assertRaisesRegex(ValueError, 'Omega'):
            core.evaluate_u66_population(**args)
        self.assertFalse(any(e[0] == 'official' for e in events))
        shard['Omega_support'][0]['query_id'] = 99
        shard['Omega_support'] = deepcopy(shard['Omega_support'])
        shard['Omega_support'][0]['Omega']['counts'] = 'derived_from_changed_method_mask'
        with self.assertRaisesRegex(ValueError, 'PLAIN.*support'):
            core.evaluate_u66_population(**args)
        shard['Omega_support'] = shard['arms']['PLAIN']['Omega_support']
        shard['arms']['FULL']['identity_predictions'][0]['segmentation']['counts'] = 'changed_zero_mask'
        with self.assertRaisesRegex(ValueError, 'identity'):
            core.evaluate_u66_population(**args)
        shard['arms']['FULL']['identity_predictions'] = deepcopy(shard['identity_predictions'])
        shard['predictions']['FULL'][0]['score'] = 0.123456
        with self.assertRaisesRegex(ValueError, 'non-segmentation fields'):
            core.evaluate_u66_population(**args)
        self.assertFalse(any(e[0] == 'geometry' for e in events))

    def test_full_population_is_not_truncated_or_reordered(self):
        core = load(HERE/'population_metrics_core.py', 'u67')
        args, events = fixture()
        args['strong'].append(deepcopy(args['strong'][-1]))
        with self.assertRaisesRegex(ValueError, '18700'):
            core.evaluate_u66_population(**args)
        args['strong'].pop()
        args['ids'].reverse()
        with self.assertRaisesRegex(ValueError, '187.*380'):
            core.evaluate_u66_population(**args)
        args['ids'].reverse()
        args['coco'].anns.pop(0)
        with self.assertRaisesRegex(ValueError, '187.*380'):
            core.evaluate_u66_population(**args)
        self.assertFalse(any(e[0] == 'official' for e in events))

    def test_every_image_calibration_and_full_query_cost_must_have_completed(self):
        core = load(HERE/'population_metrics_core.py', 'u67')
        args, events = fixture()
        arm = args['shards'][1]['arms']['FULL']
        arm['proofs'][0]['zero_residual_native_L_exact'] = False
        with self.assertRaisesRegex(ValueError, 'calibration'):
            core.evaluate_u66_population(**args)
        arm['proofs'][0]['zero_residual_native_L_exact'] = True
        arm['cost']['queries'] -= 1
        with self.assertRaisesRegex(ValueError, 'full100'):
            core.evaluate_u66_population(**args)
        self.assertFalse(any(e[0] == 'official' for e in events))


if __name__ == '__main__':
    unittest.main()
