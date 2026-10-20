"""Tiny stdlib metadata tests only; opaque tokens are not masks or packages."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import package_io as package


PINS = {
    'dev_reader': 'daabec8388e23d9ff646ae78eba69ba2168099a72971578a8d34cef04f1268c0',
    'dev_rgb_reader': 'b91375e5366da3f74272ce811f453be1379b46a4bb96103f9d8039a4998a1204',
    'image_geometry': '65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
}


def reference(name, sha='a'*64):
    return dict(path='/synthetic-never-open/'+name, bytes=0, sha256=sha)


class NeverSerialize:
    def __deepcopy__(self, memo):
        raise AssertionError('Tensor/F/geometry must not enter metadata projection')


class BooleanToken:
    def __init__(self, value=True): self.value = value
    def all(self): return self.value
    def __and__(self, other): return BooleanToken(self.value and other.value)


class TensorToken:
    """Opaque dependency-injection token; contains no tensor or image values."""
    def __init__(self, shape, label):
        self.shape, self.label = tuple(shape), label
        self.dtype, self.device, self.requires_grad = 'float32', SimpleNamespace(type='cpu'), False
    def detach(self): return self
    def contiguous(self): return self
    def clone(self): return deepcopy(self)
    def numpy(self): return self
    def tobytes(self): return ('opaque-token-'+self.label).encode('ascii')
    def __ge__(self, other): return BooleanToken()
    def __le__(self, other): return BooleanToken()


class TorchToken:
    """Only orchestration checks, never a genuine Torch numerical test."""
    Tensor, float32 = TensorToken, 'float32'
    @staticmethod
    def isfinite(value): return BooleanToken()
    @staticmethod
    def equal(left, right): return left.shape == right.shape and left.tobytes() == right.tobytes()
    saved, loads = {}, []
    @classmethod
    def save(cls, value, handle):
        marker = ('opaque-synthetic-package-'+str(len(cls.saved))).encode('ascii')
        cls.saved[marker] = deepcopy(value)
        handle.write(marker)
    @classmethod
    def load(cls, path, *, map_location, weights_only):
        cls.loads.append(dict(map_location=map_location, weights_only=weights_only))
        if map_location != 'cpu' or weights_only is not True:
            raise AssertionError('No unsafe/reinterpreted load path allowed')
        return deepcopy(cls.saved[Path(path).read_bytes()])


class ControlToken:
    def __init__(self): self.closed_refs, self.checks = [], 0
    def closed(self, reference):
        self.closed_refs.append(deepcopy(reference))
        return reference['path']
    def check(self): self.checks += 1


class GeometryToken:
    def __init__(self, arguments): self.constructor_arguments = arguments
    def render_delta(self, *args): raise AssertionError('No numerical geometry test allowed')
    def prepare_rgb(self, *args): raise AssertionError('No numerical image test allowed')


def runtime_fixture():
    item, native, provenance = fixture()
    shapes = dict(rgb400=(1, 3, 400, 400), e=(100, 256), M=(100, 200, 200), L=(100, 40, 60))
    item.update({key: TensorToken(shape, key) for key, shape in shapes.items()})
    geometry = SimpleNamespace(__file__=provenance['source_refs']['image_geometry']['path'],
        make_geometry=lambda *args: GeometryToken(args[1:]))
    return item, native, provenance, ControlToken(), geometry


def fixture():
    image = dict(path='D:/synthetic-never-open/original.jpg', bytes=0, sha256='b'*64)
    qids = list(range(7, 100))+list(range(7))
    identity = dict(image_id=501, image=image, original_shape=[40, 60],
        actual_img_shape=[533, 800], batch_input_shape=[800, 800],
        historical_target_information=NeverSerialize())
    rgb = dict(image=image, original_shape=[40, 60], actual_img_shape=[533, 800],
        canvas_size=800, branch_size=400, decoded_mode='RGB', exif_orientation=None,
        auto_orientation=False, teacher_photometric_reconstruction=False,
        input_convention='RGB uint8 -> FP32 /255; unchanged reviewed geometry',
        baseline='original all100 N16 native_L; not GT-matched selection',
        image_source_control='same explicitly admitted control')
    item = dict(F=NeverSerialize(), e=NeverSerialize(), M=NeverSerialize(),
        L=NeverSerialize(), rgb400=NeverSerialize(), geometry=NeverSerialize(),
        identity=identity, query_ids=qids, rgb_receipt=rgb)
    native = [dict(image_id=501, query_id=qid, category_id=1,
        score=(-0.0 if offset == 0 else 0),
        segmentation=dict(size=[40, 60], counts='opaque-not-a-rasterized-RLE'))
        for offset, qid in enumerate(qids)]
    native[0].update(bbox=[1, 2, 3, 4], area=12.5)
    provenance = dict(source_refs={role: reference(role+'.py', pin) for role, pin in PINS.items()},
        input_refs={role: reference(role+'.json') for role in
                    ('dev_context', 'native_predictions', 'original_capture_manifest', 'bundle', 'dense_packet')})
    return item, native, provenance


class MetadataTests(unittest.TestCase):
    def test_projection_preserves_all_original_candidates_without_copying_unrelated_fields(self):
        item, native, provenance = fixture()
        before = deepcopy(native)
        result = package.project_metadata(11, item, native, provenance=provenance)
        self.assertEqual(result['dataset_index'], 11)
        self.assertEqual(result['query_ids'], item['query_ids'])
        self.assertEqual(result['natural_query_ids'], list(range(100)))
        self.assertEqual(result['native_predictions'], before)
        self.assertEqual(native, before)
        self.assertIn('"score":-0.0', package.canonical_json(result))
        self.assertEqual(set(result['identity']),
            {'image_id', 'image', 'original_shape', 'actual_img_shape', 'batch_input_shape'})
        self.assertEqual(result['geometry'], dict(original_shape=[40, 60],
            actual_img_shape=[533, 800], canvas_size=800, branch_size=400))
        result['native_predictions'][0]['bbox'][0] = 99
        self.assertEqual(native[0]['bbox'], [1, 2, 3, 4])

    def test_nested_targets_and_query_drift_fail_before_unrelated_objects_are_copied(self):
        item, native, provenance = fixture()
        native[4]['targets'] = NeverSerialize()
        with self.assertRaises(ValueError):
            package.project_metadata(11, item, native, provenance=provenance)
        del native[4]['targets']
        metadata = package.project_metadata(11, item, native, provenance=provenance)
        metadata['identity']['targets'] = [1]
        with self.assertRaises(ValueError):
            package.validate_metadata(metadata)
        for bad in (item['query_ids'][:-1], [7]*100):
            changed = dict(item, query_ids=bad)
            with self.assertRaises(ValueError):
                package.project_metadata(11, changed, native, provenance=provenance)
        changed = dict(item, query_ids=list(range(100)))
        with self.assertRaises(ValueError):
            package.project_metadata(11, changed, native, provenance=provenance)
        polluted = deepcopy(provenance)
        polluted['input_refs']['dev_context']['targets'] = NeverSerialize()
        with self.assertRaises(ValueError):
            package.project_metadata(11, item, native, provenance=polluted)

    def test_metadata_coverage_requires_original187_unique_images_without_claiming_tensor_validation(self):
        summary = package.validate_population_indices(list(range(187)), list(range(1000, 1187)))
        self.assertEqual(summary, dict(images=187, queries=18700, tensor_population_revalidated=False))
        for indices, ids in ((list(range(186)), list(range(186))),
                             (list(reversed(range(187))), list(range(187))),
                             (list(range(187)), [5]*187)):
            with self.assertRaises(ValueError):
                package.validate_population_indices(indices, ids)

    def test_export_has_only_four_owned_natural_axis_fields_and_minimal_proofs(self):
        item, native, provenance, control, geometry = runtime_fixture()
        result = package.export_item(TorchToken, control, 11, item, native,
            image_geometry=geometry, source_refs=provenance['source_refs'],
            input_refs=provenance['input_refs'])
        self.assertEqual(set(result), {'schema', 'metadata', 'tensors', 'proof'})
        self.assertEqual(set(result['tensors']), {'rgb400', 'e', 'M', 'L'})
        self.assertEqual(result['metadata']['query_ids'], item['query_ids'])
        self.assertEqual(result['metadata']['native_predictions'], native)
        self.assertEqual(result['proof']['metadata_sha256'], package.metadata_sha256(result['metadata']))
        self.assertEqual(result['proof']['fields']['e']['shape'], [100, 256])
        for key in package.FIELDS:
            self.assertIsNot(result['tensors'][key], item[key])
        self.assertEqual(control.closed_refs, list(provenance['source_refs'].values()))
        self.assertGreater(control.checks, 0)

    def test_save_and_reader_use_cpu_weights_only_and_rebuild_original_geometry(self):
        item, native, provenance, control, geometry = runtime_fixture()
        value = package.export_item(TorchToken, control, 11, item, native,
            image_geometry=geometry, source_refs=provenance['source_refs'], input_refs=provenance['input_refs'])
        with tempfile.TemporaryDirectory(prefix='u59_metadata_only_') as directory:
            target = Path(directory)/'ONE_OPAQUE_TEST_TOKEN.pt'
            entry = package.save_export(TorchToken, control, value, target,
                image_geometry=geometry, source_refs=provenance['source_refs'])
            self.assertTrue(target.is_file())
            self.assertEqual(set(entry), {'dataset_index', 'image_id', 'query_ids', 'package_ref'})
            entries = [dict(dataset_index=index, image_id=1000+index,
                query_ids=list(range(100)), package_ref=reference('unused-'+str(index)+'.pt')) for index in range(187)]
            entries[11] = entry
            reader = package.prepare(TorchToken, control, entries,
                image_geometry=geometry, source_refs=provenance['source_refs'])
            loaded, row, originals = reader.load(11)
            self.assertEqual(set(loaded), {'rgb400', 'e', 'M', 'L', 'identity', 'query_ids', 'geometry'})
            self.assertEqual(loaded['geometry'].constructor_arguments, ([40, 60], [533, 800], 800, 400))
            self.assertEqual(row, dict(dataset_index=11, image_id=501, qids=item['query_ids']))
            self.assertEqual(originals, native)
            self.assertEqual(loaded['e'].label, 'e')
            self.assertTrue(all(call == dict(map_location='cpu', weights_only=True) for call in TorchToken.loads))
            with self.assertRaises(ValueError):
                package.save_export(TorchToken, control, value, target,
                    image_geometry=geometry, source_refs=provenance['source_refs'])
            wrong_geometry = SimpleNamespace(__file__=geometry.__file__, make_geometry=lambda *args: None)
            bad_reader = package.prepare(TorchToken, control, entries,
                image_geometry=wrong_geometry, source_refs=provenance['source_refs'])
            with self.assertRaises(ValueError):
                bad_reader.load(11)

    def test_unknown_rle_shape_types_are_rejected_without_reinterpreting_native_fields(self):
        item, native, provenance = fixture()
        native[0]['segmentation']['size'] = [40.0, 60.0]
        with self.assertRaises(ValueError):
            package.project_metadata(11, item, native, provenance=provenance)

    def test_corrupted_proof_and_polluted_population_fail_without_writing_or_loading_packages(self):
        item, native, provenance, control, geometry = runtime_fixture()
        value = package.export_item(TorchToken, control, 11, item, native,
            image_geometry=geometry, source_refs=provenance['source_refs'], input_refs=provenance['input_refs'])
        value['proof']['fields']['e']['sha256'] = '0'*64
        with tempfile.TemporaryDirectory(prefix='u59_metadata_only_') as directory:
            target = Path(directory)/'MUST_NOT_BE_CREATED.pt'
            with self.assertRaises(ValueError):
                package.save_export(TorchToken, control, value, target,
                    image_geometry=geometry, source_refs=provenance['source_refs'])
            self.assertFalse(target.exists())
        entries = [dict(dataset_index=index, image_id=1000+index,
            query_ids=list(range(100)), package_ref=reference('unused-'+str(index)+'.pt')) for index in range(187)]
        entries[50]['package_ref']['targets'] = NeverSerialize()
        closed_before = len(control.closed_refs)
        with self.assertRaises(ValueError):
            package.prepare(TorchToken, control, entries,
                image_geometry=geometry, source_refs=provenance['source_refs'])
        self.assertEqual(len(control.closed_refs)-closed_before, 3)  # source refs only, no package

    def test_unsupported_weights_only_load_has_no_unsafe_retry_and_preserves_failed_output(self):
        class UnsupportedTorch(TorchToken):
            attempts = 0
            @classmethod
            def load(cls, *args, **kwargs):
                cls.attempts += 1
                raise TypeError('synthetic unsupported weights_only ABI')
        item, native, provenance, control, geometry = runtime_fixture()
        value = package.export_item(TorchToken, control, 11, item, native,
            image_geometry=geometry, source_refs=provenance['source_refs'], input_refs=provenance['input_refs'])
        with tempfile.TemporaryDirectory(prefix='u59_metadata_only_') as directory:
            target = Path(directory)/'FAILED_OPAQUE_TEST_TOKEN.pt'
            with self.assertRaisesRegex(ValueError, 'weights_only=True'):
                package.save_export(UnsupportedTorch, control, value, target,
                    image_geometry=geometry, source_refs=provenance['source_refs'])
            self.assertTrue(target.is_file())
            self.assertEqual(UnsupportedTorch.attempts, 1)


if __name__ == '__main__':
    unittest.main()
