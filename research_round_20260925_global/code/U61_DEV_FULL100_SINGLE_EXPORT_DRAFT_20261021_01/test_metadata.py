"""Tiny stdlib contract tests; no main(), scientific import, or real input."""
from copy import deepcopy
import unittest

import entry


PINS = dict(export_source='8fabc800e4d36e2da3988e42acf88058c80790f9c8b3d664a1a4fa9232233473',
    dev_reader_source='daabec8388e23d9ff646ae78eba69ba2168099a72971578a8d34cef04f1268c0',
    dev_rgb_reader_source='b91375e5366da3f74272ce811f453be1379b46a4bb96103f9d8039a4998a1204',
    geometry_source='65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7')


def reference(label, sha='a'*64, size=0):
    return dict(path='D:/synthetic-never-open/'+label, bytes=size, sha256=sha)


def context_fixture():
    rows = [dict(dataset_index=index, image_id=1000+index,
        qids=list(range(13, 100))+list(range(13)), original_shape=[40, 60],
        bundle=reference('bundle-'+str(index)+'.pt'),
        original_capture_manifest=reference('manifest-'+str(index)+'.json'),
        dense_packet=reference('packet-'+str(index)+'.pt')) for index in range(187)]
    context = dict(schema='ROOT-N28-fixed24-dense-native-DEV-context-v1',
        image_ids=list(range(1000, 1187)), rows=rows,
        legacy_context=reference('legacy.json'), legacy_ROOT_result=reference('accepted-cache.json'),
        native_predictions=reference('native.json'), R_predictions=reference('R.json'),
        encoder_source=reference('encoder.py'), teacher_checkpoint=reference('NEVER_LOAD_TEACHER.pt'))
    images = [reference('image-'+str(index)+'.jpg') for index in range(187)]
    return context, images


def spec_fixture():
    context, images = context_fixture()
    roles = dict(entry_source=reference('entry.py'))
    roles.update({role: reference(role+'.py', pin, 20627 if role == 'export_source' else 0)
                  for role, pin in PINS.items()})
    libraries = {name: reference(name+'.py') for name in
        ('psutil', 'torch', 'numpy', 'PIL', 'PIL.Image', 'PIL.ImageFile',
         'pycocotools', 'pycocotools.coco', 'pycocotools.mask', 'pycocotools._mask')}
    inputs = [context[key] for key in
        ('legacy_context', 'native_predictions', 'R_predictions', 'encoder_source', 'legacy_ROOT_result')]
    inputs += [context['rows'][0][key] for key in ('original_capture_manifest', 'bundle', 'dense_packet')]+[images[0]]
    outer = reference('dev-context.json')
    return dict(schema='U61-DEV-full100-single-CPU-export-spec-v1', operation='CPU_U61_DEV_FULL100_SINGLE_EXPORT',
        phase='SYNTHETIC_NEVER_EXECUTE', expected_versions=dict(torch='2.0.1+cu118', numpy='1.23.5'),
        limits=dict(allowed_cpus=[0, 1], threads=2, maximum_RSS_bytes=16*1024**3,
                    minimum_available_RAM_bytes=16*1024**3, deadline_seconds=1800, CUDA_visible_devices=''),
        dataset_indices=[0], source_roles=roles, library_sources=libraries, context_ref=outer,
        original_image_refs=images, input_sources=inputs,
        sources=list(roles.values())+list(libraries.values())+[outer]+inputs,
        output_directory='D:/synthetic-never-created/U61_OUTPUT')


class MetadataTests(unittest.TestCase):
    def test_single_original_zero_with_full_mapping_and_explicit_sources(self):
        spec = spec_fixture()
        entry.validate_spec(spec)
        for indices in ([1], [0, 1], [], [False]):
            changed = deepcopy(spec)
            changed['dataset_indices'] = indices
            with self.assertRaises(ValueError): entry.validate_spec(changed)
        changed = deepcopy(spec)
        changed['limits']['deadline_seconds'] = 180
        with self.assertRaises(ValueError): entry.validate_spec(changed)

    def test_unselected_images_remain_metadata_without_becoming_opened_sources(self):
        spec = spec_fixture()
        context, images = context_fixture()
        expected = entry.expected_input_references(context, images)
        self.assertEqual(expected, spec['input_sources'])
        self.assertNotIn(context['teacher_checkpoint'], expected)
        self.assertNotIn(images[1], expected)
        self.assertNotIn(context['rows'][1]['bundle'], expected)
        entry.validate_context_inputs(spec, context)
        changed = deepcopy(spec)
        changed['input_sources'].append(context['rows'][1]['bundle'])
        with self.assertRaises(ValueError): entry.validate_context_inputs(changed, context)

    def test_all18700_native_candidates_keep_original_order_and_unknown_fields(self):
        context, images = context_fixture()
        native = [dict(image_id=row['image_id'], query_id=qid, category_id=1, score=0.,
                    segmentation=dict(size=[40, 60], counts='opaque-not-RLE'),
                    unreviewed_field='retained-for-U59-to-reject-if-selected')
                  for row in context['rows'] for qid in row['qids']]
        before = deepcopy(native)
        result = entry.validate_native_population(context['rows'], native)
        self.assertEqual(result, dict(images=187, queries=18700, tensors_revalidated=False))
        self.assertEqual(native, before)
        native[0], native[1] = native[1], native[0]
        with self.assertRaises(ValueError): entry.validate_native_population(context['rows'], native)

    def test_summary_contains_only_four_verified_field_proofs_and_scalar_identity_digests(self):
        package = dict(proof=dict(schema='U59-full100-DEV-field-proof-v1', metadata_sha256='b'*64,
            fields={name: dict(shape=[1], dtype='float32', sha256='c'*64, forbidden='NOT_COPIED')
                    for name in ('rgb400', 'e', 'M', 'L')}),
            metadata=dict(forbidden='NOT_COPIED'))
        receipt = dict(package_ref=reference('one.pt', 'd'*64, 123))
        checks = {name: dict(source_bitexact=True, independent_storage=True, compact_storage=True)
                  for name in ('rgb400', 'e', 'M', 'L')}
        result = entry.summarize_export(package, receipt, checks, 'e'*64)
        self.assertEqual(result['images'], 1)
        self.assertEqual(result['queries'], 100)
        self.assertEqual(result['native_predictions_metadata_sha256'], 'e'*64)
        self.assertEqual(set(result['fields']['e']),
            {'shape', 'dtype', 'sha256', 'source_bitexact', 'independent_storage', 'compact_storage'})
        self.assertNotIn('NOT_COPIED', str(result))
        checks['e']['source_bitexact'] = False
        with self.assertRaises(ValueError): entry.summarize_export(package, receipt, checks, 'e'*64)

    def test_source_membership_and_context_union_reject_unselected_or_conflicting_inputs(self):
        spec = spec_fixture()
        context, images = context_fixture()
        changed = deepcopy(spec)
        changed['input_sources'].append(context['rows'][1]['bundle'])
        changed['sources'].append(context['rows'][1]['bundle'])
        entry.validate_spec(changed)  # The context is needed to identify this excess.
        with self.assertRaises(ValueError): entry.validate_context_inputs(changed, context)
        for mutate in (
            lambda value: value['sources'].append(images[2]),
            lambda value: value['sources'].append(deepcopy(value['sources'][0])),
            lambda value: value['original_image_refs'].pop(),
            lambda value: value['original_image_refs'].__setitem__(1, value['original_image_refs'][0]),
            lambda value: value['source_roles']['dev_reader_source'].__setitem__('sha256', '0'*64),
            lambda value: value.__setitem__('accounting_parent', reference('FORBIDDEN.json')),
        ):
            changed = deepcopy(spec)
            mutate(changed)
            with self.assertRaises(ValueError): entry.validate_spec(changed)

    def test_full_population_rejects_duplicate_images_or_renumbered_ordinals(self):
        context, images = context_fixture()
        context['image_ids'][1] = context['image_ids'][0]
        with self.assertRaises(ValueError): entry.expected_input_references(context, images)
        context, images = context_fixture()
        context['rows'][0]['dataset_index'] = False
        with self.assertRaises(ValueError): entry.expected_input_references(context, images)


if __name__ == '__main__':
    unittest.main()
