"""Finite stdlib tests only; no main, scientific import, process or real input."""
from copy import deepcopy
import unittest

import entry


PINS = dict(export_source='8fabc800e4d36e2da3988e42acf88058c80790f9c8b3d664a1a4fa9232233473',
    dev_reader_source='daabec8388e23d9ff646ae78eba69ba2168099a72971578a8d34cef04f1268c0',
    dev_rgb_reader_source='b91375e5366da3f74272ce811f453be1379b46a4bb96103f9d8039a4998a1204',
    geometry_source='65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7')


def reference(label, sha='a'*64, size=0):
    return dict(path='D:/synthetic-never-open/'+label, bytes=size, sha256=sha)


def fixture():
    rows = [dict(dataset_index=index, image_id=1000+index, qids=list(range(13,100))+list(range(13)),
        bundle=reference('bundle-'+str(index)+'.pt'), original_capture_manifest=reference('manifest-'+str(index)+'.json'),
        dense_packet=reference('packet-'+str(index)+'.pt')) for index in range(187)]
    context = dict(schema='ROOT-N28-fixed24-dense-native-DEV-context-v1', image_ids=list(range(1000,1187)), rows=rows,
        **{key: reference(key+'.json') for key in
           ('legacy_context','native_predictions','R_predictions','encoder_source','legacy_ROOT_result')})
    context.update(teacher_checkpoint=reference('NEVER_OPEN_TEACHER.pt'), annotation=reference('NEVER_OPEN_GT.json'))
    images = [reference('image-'+str(index)+'.jpg') for index in range(187)]
    roles = dict(entry_source=reference('entry.py'))
    roles.update({role: reference(role+'.py', pin, 20627 if role == 'export_source' else 0) for role,pin in PINS.items()})
    libraries = {name: reference(name+'.py') for name in
        ('psutil','torch','numpy','PIL','PIL.Image','PIL.ImageFile','pycocotools','pycocotools.coco','pycocotools.mask','pycocotools._mask')}
    inputs = [context[key] for key in ('legacy_context','native_predictions','R_predictions','encoder_source','legacy_ROOT_result')]
    for index in range(1,187):
        inputs.extend(rows[index][key] for key in ('original_capture_manifest','bundle','dense_packet'))
        inputs.append(images[index])
    outer = reference('context.json')
    spec = dict(schema='U62-DEV-remaining186-full100-CPU-export-spec-v1', operation='CPU_U62_DEV_REMAINING186_FULL100_EXPORT',
        phase='SYNTHETIC_NEVER_EXECUTE', expected_versions=dict(torch='2.0.1+cu118',numpy='1.23.5'),
        limits=dict(allowed_cpus=[0,1],threads=2,maximum_RSS_bytes=16*1024**3,
            minimum_available_RAM_bytes=16*1024**3,deadline_seconds=1800,CUDA_visible_devices=''),
        minimum_output_free_bytes=20*1024**3, dataset_indices=list(range(1,187)),source_roles=roles,library_sources=libraries,
        context_ref=outer,original_image_refs=images,input_sources=inputs,
        sources=list(roles.values())+list(libraries.values())+[outer]+inputs,
        output_directory='D:/synthetic-never-created/U62_OUTPUT')
    return context,images,spec


class MetadataTests(unittest.TestCase):
    def test_fixed_original_remaining_range_and_fixed_D_reserve(self):
        context,images,spec = fixture()
        entry.validate_spec(spec)
        for indices in ([0]+list(range(1,187)), list(range(1,186)), list(reversed(range(1,187))), [True]+list(range(2,187))):
            changed=deepcopy(spec);changed['dataset_indices']=indices
            with self.assertRaises(ValueError): entry.validate_spec(changed)
        for reserve in (0, 19*1024**3, 21*1024**3, True):
            changed=deepcopy(spec);changed['minimum_output_free_bytes']=reserve
            with self.assertRaises(ValueError): entry.validate_spec(changed)

    def test_exact_selected_union_excludes_ordinal0_and_recursive_weight_or_annotation(self):
        context,images,spec = fixture()
        inputs=entry.expected_input_references(context,images)
        self.assertEqual(inputs,spec['input_sources'])
        entry.validate_context_inputs(spec,context)
        for forbidden in (context['teacher_checkpoint'],context['annotation'],images[0],context['rows'][0]['bundle'],context['rows'][0]['dense_packet']):
            self.assertNotIn(forbidden,inputs)
            changed=deepcopy(spec);changed['input_sources'].append(forbidden);changed['sources'].append(forbidden)
            entry.validate_spec(changed)
            with self.assertRaises(ValueError): entry.validate_context_inputs(changed,context)

    def test_each_package_uses_its_own_original_refs_and_never_compact_zero(self):
        context,images,spec = fixture()
        for index in (1,2,186):
            value=entry.item_input_references(spec,context,index)
            self.assertEqual(set(value),{'dev_context','native_predictions','original_capture_manifest','bundle','dense_packet'})
            self.assertEqual(value['bundle'],context['rows'][index]['bundle'])
            self.assertEqual(value['dense_packet'],context['rows'][index]['dense_packet'])
        for index in (0,187,True):
            with self.assertRaises(ValueError): entry.item_input_references(spec,context,index)

    def test_low_D_disk_sample_records_failure_and_cannot_be_erased_by_later_space(self):
        observed=dict(peak_RSS_bytes=0,minimum_available_RAM_bytes=None,samples=0,violations=[],sampling_errors=[],
            minimum_output_disk_free_bytes=None,output_disk_samples=0,output_disk_sample_path='D:/synthetic-never-open')
        entry.observe_resources(observed,rss_bytes=1,available_ram_bytes=32*1024**3,output_free_bytes=20*1024**3)
        self.assertEqual(observed['violations'],[])
        entry.validate_observations(observed)
        entry.observe_resources(observed,rss_bytes=2,available_ram_bytes=32*1024**3,output_free_bytes=20*1024**3-1)
        self.assertIn('D output free-space boundary',observed['violations'])
        with self.assertRaises(ValueError): entry.validate_observations(observed)
        entry.observe_resources(observed,rss_bytes=1,available_ram_bytes=32*1024**3,output_free_bytes=40*1024**3)
        self.assertEqual(observed['minimum_output_disk_free_bytes'],20*1024**3-1)
        self.assertEqual(observed['output_disk_samples'],3)
        self.assertIn('D output free-space boundary',observed['violations'])
        with self.assertRaises(ValueError): entry.validate_observations(observed)


if __name__=='__main__': unittest.main()
