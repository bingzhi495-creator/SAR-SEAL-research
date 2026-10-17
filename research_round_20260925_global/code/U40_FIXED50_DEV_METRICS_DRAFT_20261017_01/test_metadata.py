"""Only fabricated JSON metadata; no scientific imports, masks or payload I/O."""
import copy
import unittest

import population_metrics as core


def ref(label):
    return dict(path='E:/SYNTHETIC_ONLY/'+label+'.json', bytes=len(label), sha256='a'*64)


def endpoint():
    spec = dict(seed=202610173101, fixed_epoch=50, updates_per_arm=37050,
        training_qualified=ref('training'), pair_commit_ref=ref('pair'),
        checkpoint_refs={a: ref(a) for a in ('PLAIN', 'FULL')}, order_ref=ref('order'))
    training = dict(schema='ROOT-U40-continuous-training-segment-qualified-v1',
        qualified=True, paired_prefix_advanced=True, result=dict(seed=spec['seed'],
        span=dict(start=36864, stop=37050), paired_checkpoint_prefix=dict(
        global_updates=37050, completed_epochs=50, within_epoch=0),
        checkpoint_refs=copy.deepcopy(spec['checkpoint_refs']),
        pair_commit_ref=spec['pair_commit_ref'], order_ref=spec['order_ref']))
    return spec, training


def shard_metadata(ordinal):
    spec, _ = endpoint()
    roles = ('shared_initialization', 'full_legacy', 'full_dsconv', 'full_head',
        'full_feature', 'plain_legacy', 'plain_feature', 'instance_adapter',
        'u36_helpers', 'u39_training_ops', 'image_geometry', 'training_state',
        'dev_reader', 'dev_rgb_reader', 'population_contract', 'encoder', 'mask_api')
    spec.update(dev_context=ref('DEV_context'), prediction_core_source=ref('predict_core'),
        training_sources={'source_%02d' % i: ref('training_%02d' % i) for i in range(17)},
        prediction_sources={key: ref(key) for key in roles})
    start, stop = [[0, 47], [47, 94], [94, 141], [141, 187]][ordinal]
    report = dict(schema='U40-fixed50-full-method-DEV-shard-pending-v1',
        seed=spec['seed'], fixed_epoch=50, updates_per_arm=37050,
        dataset_range=[start, stop], shard_index=ordinal, images=stop-start,
        queries=(stop-start)*100, training_qualified=spec['training_qualified'],
        pair_commit_ref=spec['pair_commit_ref'], checkpoint_refs=spec['checkpoint_refs'],
        order_ref=spec['order_ref'], dev_context=spec['dev_context'], query_batch=4,
        predictions={arm: ref(arm+'_%d' % ordinal) for arm in ('PLAIN', 'FULL')},
        identity_predictions=ref('identity_%d' % ordinal), Omega_support=ref('Omega_%d' % ordinal),
        training_sources=spec['training_sources'], prediction_sources=spec['prediction_sources'],
        prediction_core_source=spec['prediction_core_source'],
        new_training=False, AP=False, DEV=True, TEST=False)
    report.update({key: True for key in ('native_all_RLE_and_identity_prediction_dictionaries_exact',
        'zero_residual_native_L_exact', 'original_full_S0_class_order_preserved',
        'historical_target_fields_may_be_in_bundle', 'model_BN_RNG_unchanged')})
    report.update({key: False for key in ('original_teacher_forward', 'GT_used_for_prediction',
        'DEV_COCO_annotation_opened', 'new_training')})
    shared = ('predictions', 'identity_predictions', 'Omega_support', 'dataset_range',
        'shard_index', 'images', 'queries', 'seed', 'fixed_epoch', 'updates_per_arm',
        'training_qualified', 'pair_commit_ref', 'checkpoint_refs', 'order_ref',
        'dev_context', 'training_sources', 'prediction_sources',
        'new_training', 'AP', 'DEV', 'TEST')
    result = {key: copy.deepcopy(report[key]) for key in shared}
    result['report_ref'] = ref('report_%d' % ordinal)
    accepted = dict(schema='ROOT-U40-fixed50-full-method-DEV-shard-qualified-v1',
                    qualified=True, result=result)
    return spec, accepted, report


class MetadataContract(unittest.TestCase):
    def test_only_actual_root_accepted_fixed50_pair_is_a_usable_reference(self):
        spec, training = endpoint()
        before = copy.deepcopy((spec, training))
        core.validate_fixed_endpoint(spec, training)
        self.assertEqual((spec, training), before)
        for replacement in ({'qualified': False}, {'paired_prefix_advanced': False}):
            bad = copy.deepcopy(training)
            bad.update(replacement)
            with self.assertRaises(RuntimeError):
                core.validate_fixed_endpoint(spec, bad)
        bad = copy.deepcopy(training)
        bad['result']['paired_checkpoint_prefix']['global_updates'] = 36864
        with self.assertRaises(RuntimeError):
            core.validate_fixed_endpoint(spec, bad)
        bad = copy.deepcopy(training)
        bad['result']['checkpoint_refs']['FULL'] = ref('wrong_pair')
        with self.assertRaises(RuntimeError):
            core.validate_fixed_endpoint(spec, bad)

    def test_all_four_shards_require_exact_range_pair_population_and_root_acceptance(self):
        for ordinal in range(4):
            spec, accepted, report = shard_metadata(ordinal)
            core.validate_shard_metadata(spec, accepted, report, ordinal)
            with self.subTest(ordinal=ordinal, corruption='pending'):
                bad = copy.deepcopy(accepted)
                bad['qualified'] = False
                with self.assertRaises(RuntimeError):
                    core.validate_shard_metadata(spec, bad, report, ordinal)
            bad = copy.deepcopy(accepted)
            bad['result']['training_sources']['source_00'] = ref('different_training_source')
            with self.assertRaises(RuntimeError):
                core.validate_shard_metadata(spec, bad, report, ordinal)
            for key, value in [('dataset_range', [0, 47]), ('queries', 1),
                               ('checkpoint_refs', {'PLAIN': ref('other'), 'FULL': ref('FULL')})]:
                if report[key] == value:
                    continue
                with self.subTest(ordinal=ordinal, corruption=key):
                    bad = copy.deepcopy(report)
                    bad[key] = value
                    with self.assertRaises(RuntimeError):
                        core.validate_shard_metadata(spec, accepted, bad, ordinal)

    def test_legacy_geometry_labels_are_disclosed_then_replaced_without_changing_values(self):
        names = ('native_official', 'R', 'SELECTIVE', 'WHOLE_NODE')
        original = dict(summary=dict(schema='N28-original-native-fixed-pair-geometry-summary-v1',
            scope=dict(total_DEV_GT=380, selected_native_nonignored_GT=1),
            methods={name: {'unchanged_integer_count': i} for i, name in enumerate(names)}),
            private_pair_records=[dict(GT_annotation_id=17,
                methods={name: {'unchanged_integer_count': i} for i, name in enumerate(names)})])
        before = copy.deepcopy(original)
        result = core.relabel_geometry(original)
        self.assertEqual(original, before)
        self.assertEqual(set(result['summary']['methods']), {'native_official', 'R', 'FULL', 'PLAIN'})
        self.assertEqual(result['summary']['methods']['FULL'], original['summary']['methods']['SELECTIVE'])
        self.assertEqual(result['private_pair_records'][0]['methods']['PLAIN'],
                         original['private_pair_records'][0]['methods']['WHOLE_NODE'])
        self.assertEqual(result['summary']['method_label_adapter'],
                         {'SELECTIVE': 'FULL', 'WHOLE_NODE': 'PLAIN'})
        self.assertFalse(result['summary']['old_method_models_evaluated'])
        self.assertEqual(result['summary']['scope']['total_DEV_GT'], 380)
        self.assertEqual(result['summary']['scope']['selected_native_nonignored_GT'], 1)


if __name__ == '__main__':
    unittest.main()
