import unittest
from types import SimpleNamespace

from e1_export import export_e1_image


class Tensor:
    def __init__(self, values, shape, dtype, device='cpu', finite=True):
        self.values = values
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.finite = finite

    def is_floating_point(self):
        return self.dtype == 'float32'

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values

    def __getitem__(self, index):
        return Tensor(self.values[index], self.shape[1:], self.dtype, self.device)


def tensor(data, *, device, dtype):
    return Tensor(data, (len(data), len(data[0])), dtype, device)


def finite(value):
    return SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: value.finite))


torch = SimpleNamespace(bool='bool', int64='int64', is_tensor=lambda value: isinstance(value, Tensor),
                        tensor=tensor, isfinite=finite)


def raw_outputs(queries=100):
    return dict(pred_logits=Tensor(None, (1, queries, 2), 'float32'),
                pred_boxes=Tensor(None, (1, queries, 4), 'float32'),
                pred_masks=Tensor(None, (1, queries, 3, 3), 'float32'))


def native_outputs(rows=100):
    return dict(scores=Tensor([0.9, 0.42] + [0.1] * (rows - 2), (rows,), 'float32'),
                labels=Tensor([1, 0] + [1] * (rows - 2), (rows,), 'int64'),
                boxes=Tensor([[0, 0, 0, 0]] * rows, (rows, 4), 'float32'),
                masks=Tensor([[[[False, True], [False, False]]]] * rows, (rows, 1, 2, 2), 'bool'))


class ExportTests(unittest.TestCase):
    def test_all_native_rows_survive_and_coco_preserves_low_mapped_score(self):
        raw = raw_outputs()
        native = native_outputs()
        calls = []
        exported = []

        def create_postprocess(**options):
            self.assertEqual(options, dict(num_select=100, upsample_masks_to_image_size=True))

            def postprocess(outputs, sizes, *, score_threshold):
                self.assertEqual(outputs, raw)
                self.assertEqual(sizes.tolist(), [[2, 2]])
                self.assertEqual(sizes.device, raw['pred_logits'].device)
                self.assertIsNone(score_threshold)
                calls.append('native')
                return [native]

            return postprocess

        def canonical_export(predictions, *, image_id, original_hw, label2cat, encode_mask):
            self.assertEqual(label2cat, {0: 1})
            self.assertEqual(original_hw, (2, 2))
            label, = predictions['labels']
            score, = predictions['scores']
            mask, = predictions['masks']
            self.assertEqual(mask, [[[False, True], [False, False]]])
            exported.append((label, score))
            return ([dict(image_id=image_id, category_id=label2cat[label], score=score,
                          segmentation=encode_mask(mask[0]))] if label in label2cat else [])

        out = export_e1_image(raw, image_id=7, original_hw=(2, 2), torch_api=torch,
                              postprocess_type=create_postprocess, canonical_export=canonical_export,
                              encode_mask=lambda mask: dict(size=[2, 2], counts='fixture'))
        self.assertEqual(calls, ['native'])
        self.assertIs(out['native'], native)
        self.assertEqual(len(exported), 100)
        self.assertEqual(exported[:2], [(1, 0.9), (0, 0.42)])
        self.assertEqual(out['native_count'], 100)
        self.assertEqual(out['unmapped_slot1_count'], 99)
        self.assertEqual(out['coco'], [dict(image_id=7, category_id=1, score=0.42,
                                            segmentation=dict(size=[2, 2], counts='fixture'))])


    def test_wrong_query_count_is_rejected_before_selection(self):
        def never_called(**kwargs):
            self.fail('Wrong raw query count must not be padded or ranked')

        with self.assertRaisesRegex(ValueError, '100 actual queries'):
            export_e1_image(raw_outputs(49), image_id=7, original_hw=(2, 2), torch_api=torch,
                            postprocess_type=never_called, canonical_export=never_called,
                            encode_mask=never_called)

    def test_shortened_native_selection_is_rejected_before_coco_export(self):
        def create_postprocess(**options):
            return lambda *args, **kwargs: [native_outputs(rows=99)]

        def never_called(**kwargs):
            self.fail('Shortened native rows must not reach the canonical exporter')

        with self.assertRaisesRegex(ValueError, 'top100 tensor row count'):
            export_e1_image(raw_outputs(), image_id=7, original_hw=(2, 2), torch_api=torch,
                            postprocess_type=create_postprocess, canonical_export=never_called,
                            encode_mask=never_called)


if __name__ == '__main__':
    unittest.main()
