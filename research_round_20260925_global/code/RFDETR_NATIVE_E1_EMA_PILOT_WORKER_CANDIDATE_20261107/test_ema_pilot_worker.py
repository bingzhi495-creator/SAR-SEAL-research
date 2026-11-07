import unittest
from collections import OrderedDict

from ema_pilot_worker import ema_state_for_model


def checkpoint():
    average = OrderedDict([
        ('n_averaged', 186),
        ('module.model.class_embed.weight', 'ema-classifier'),
        ('module.model.stem.weight', 'ema-stem'),
        ('module.criterion.buffer', 'ema-criterion'),
    ])
    average._metadata = OrderedDict([
        ('', {'version': 1}),
        ('module', {'version': 1}),
        ('module.model', {'version': 2}),
        ('module.model.stem', {'version': 3}),
    ])
    return {
        'global_step': 186,
        'state_dict': OrderedDict([
            ('model.class_embed.weight', 'raw-classifier'),
            ('model.stem.weight', 'raw-stem'),
            ('criterion.buffer', 'raw-criterion'),
            ('criterion._extra_state', 'omitted-by-native-ema'),
        ]),
        'callbacks': {'RFDETREMACallback': {
            'latest_update_step': 186, 'average_model_state_dict': average,
        }},
    }


class EMATests(unittest.TestCase):
    def test_selects_the_native_ema_model_and_preserves_scoped_metadata(self):
        mapped, raw = ema_state_for_model(checkpoint(), {'class_embed.weight', 'stem.weight'})
        self.assertEqual(list(mapped.items()), [
            ('class_embed.weight', 'ema-classifier'), ('stem.weight', 'ema-stem')])
        self.assertEqual(mapped._metadata, {'': {'version': 2}, 'stem': {'version': 3}})
        self.assertEqual(raw, 'raw-classifier')


    def test_missing_ema_key_does_not_fall_back_to_raw(self):
        value = checkpoint()
        del value['callbacks']['RFDETREMACallback']['average_model_state_dict']['module.model.stem.weight']
        with self.assertRaisesRegex(ValueError, 'EMA wrapper keys'):
            ema_state_for_model(value, {'class_embed.weight', 'stem.weight'})

    def test_eager_model_schema_must_match_the_full_mapped_namespace(self):
        with self.assertRaisesRegex(ValueError, 'eager model key set'):
            ema_state_for_model(checkpoint(), {'class_embed.weight'})


if __name__ == '__main__':
    unittest.main()
