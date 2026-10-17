"""Stdlib-only metadata/failed-flow checks; no scientific libraries or inputs."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('u43_entry_test', HERE/'run_check.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def reference(name):
    return dict(path=str((HERE/name).resolve()), bytes=1, sha256='a'*64)


def scope():
    return dict(schema=entry.SPEC_SCHEMA, operation=entry.OPERATION,
        owner_index=10, normal_RAW_count=10, total_failed_RAW_count=2,
        total_terminal_identity_pin_count=12, legacy_pin_count=13,
        deadline_seconds=1800, automatic_retry=False, cuda_visible_devices='',
        allowed_cpus=[0,1], thread_limit=2, maximum_process_RSS_bytes=16*1024**3,
        minimum_available_RAM_bytes=16*1024**3, whole_GPU_limit_MiB=8192,
        allocator_limit_MiB=4096, expected_images=5604, expected_annotations=16969,
        splits=['train','test'], no_new_training=True, no_AP=True,
        no_model=True, no_prediction=True, no_DEV=True,
        TEST_annotations_read=True, TEST_used_for_model_selection=False,
        library_sources={name:reference('lib_'+str(i)) for i,name in enumerate(entry.LIBRARY_ROLES)})


class MetadataTests(unittest.TestCase):
    def test_exact_population_and_resource_scope(self):
        value=scope()
        entry.validate_scope(value)
        # Legacy aliases are deliberately independent of normal+failed.
        for key,bad in [('expected_images',1962),('expected_annotations',16951),
                        ('deadline_seconds',1801),('cuda_visible_devices','0'),
                        ('no_new_training',False),('TEST_annotations_read',False)]:
            changed=copy.deepcopy(value); changed[key]=bad
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                entry.validate_scope(changed)

    def test_complete_direct_inputs_no_unrelated_payload(self):
        value=scope()
        for role in entry.CODE_ROLES+('context','prelock','plan','parent_ROOT_qualified'):
            value[role]=reference(role)
        root=HERE/'synthetic_images'
        context=dict(schema=entry.CONTEXT_SCHEMA,image_root=str(root),
            annotations={s:reference(s+'.json') for s in ('train','test')},
            image_refs_by_filename={str(i)+'.jpg':reference('synthetic_images/'+str(i)+'.jpg') for i in range(5604)})
        value['sources']=[value[k] for k in entry.CODE_ROLES+('context','prelock','plan','parent_ROOT_qualified')]
        value['sources']+=list(value['library_sources'].values())+list(context['annotations'].values())
        value['sources']+=list(context['image_refs_by_filename'].values())
        entry.validate_sources(value,context)
        for mode in ('missing_image','missing_split','extra_weight'):
            changed=copy.deepcopy(value)
            if mode=='missing_image': changed['sources'].pop()
            elif mode=='missing_split': changed['sources'].remove(context['annotations']['test'])
            else: changed['sources'].append(reference('unrelated.pth'))
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                entry.validate_sources(changed,context)

    def test_resource_exception_preserves_private_partial_and_never_Q(self):
        ms=importlib.util.spec_from_file_location('u43_body_test',HERE/'check_body.py')
        body=importlib.util.module_from_spec(ms); ms.loader.exec_module(body)
        class ResourceStop(RuntimeError): pass
        calls=[]
        def check(): calls.append('check')
        def work(record_image):
            record_image(dict(split='train',global_image_ordinal=0,
                              private_filename='fixture_only.jpg',status='FAILED'))
            raise ResourceStop('synthetic resource interruption; no decoder was invoked')
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'owned'
            report=dict(population={},qualified=False)
            with self.assertRaises(ResourceStop):
                body.persist_run(output,report,work,check)
            partial=json.loads((output/'REPORT.json').read_text())
            rows=(output/'IMAGE_CHECKS.jsonl').read_text().splitlines()
            self.assertEqual(partial['status'],'FAILED')
            self.assertEqual(partial['image_records_written'],1)
            self.assertEqual(len(rows),1)
            self.assertEqual(json.loads(rows[0])['private_filename'],'fixture_only.jpg')
            self.assertFalse(partial['qualified'])
            self.assertFalse((output/'Q.json').exists())
            self.assertFalse((output/'SAFE_SUMMARY.json').exists())


if __name__ == '__main__':
    unittest.main()
