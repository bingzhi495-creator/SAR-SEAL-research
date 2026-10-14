"""One metadata/SYN fixture only: no Torch/CUDA/SAR/actual refs or qualification."""
import importlib.util
from pathlib import Path
import unittest
ENTRY=Path(__file__).with_name('N17_FIXEDDEV_FULL187_CPU_COCO_FINAL_ROOT_UNBOUND_DRAFT_20261015_01.py')

class Contract(unittest.TestCase):
    def test_fixed_four_scopes_and_original_ordered_population(self):
        info=importlib.util.spec_from_file_location('_N17_final_fake',ENTRY);m=importlib.util.module_from_spec(info);info.loader.exec_module(m)
        fake=lambda name:dict(path=str(ENTRY.parent/name),bytes=0,sha256='0'*64)
        descriptors=[]
        for sid,scope in enumerate(m.RANGES):
            descriptors.append(dict(cpu_shard_id=sid,cpu_dataset_ordinal_range=list(scope),source=fake('SAME_FAKE_CPU.py'),spec=fake('CPU%d_SPEC.json'%sid),
                report=fake('CPU%d_REPORT.json'%sid),RAW=fake('CPU%d_RAW.json'%sid),admission=fake('CPU%d_AD.json'%sid),
                phase_binding=fake('CPU%d_PHASE.json'%sid),bootstrap=fake('CPU%d_BOOT.py'%sid),cpu_shard_owner_index=126+sid))
        spec=dict(schema=m.CPU_SCHEMA,entry_source=fake(ENTRY.name),expected_normal_RAW_count=130,normal_RAW_count=130,
            total_failed_RAW_count=19,total_terminal_identity_pin_count=149,deadline_seconds=1800,automatic_retry=False,
            initialization_seed=202610150171,maximum_process_RSS_bytes=16*1024**3,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192)
        self.assertTrue(m.validate_final_header(spec));self.assertEqual(m.validated_CPU_descriptors(descriptors,130),descriptors)
        for key,wrong in (('schema',m.SCHEMA),('total_failed_RAW_count',18),('total_terminal_identity_pin_count',148)):
            with self.subTest(key=key),self.assertRaises(RuntimeError):m.validate_final_header(dict(spec,**{key:wrong}))
        wrong=[dict(item) for item in descriptors];wrong[1]['cpu_dataset_ordinal_range']=[47,95]
        with self.assertRaises(RuntimeError):m.validated_CPU_descriptors(wrong,130)
        ids=list(range(1,188));native=[dict(image_id=image,query_id=q,category_id=1,score=.3,segmentation=dict(size=[1,1],counts='1'))
            for image in ids for q in range(99,-1,-1)]
        pieces=[native[a*100:z*100] for a,z in m.RANGES]
        self.assertEqual(m.merge_policy_predictions(pieces,ids,native,native),native)
        bad=[list(part) for part in pieces];bad[0]=list(reversed(bad[0]))
        with self.assertRaises(RuntimeError):m.merge_policy_predictions(bad,ids,native,native)
        bad=[list(part) for part in pieces];bad[0][0]=dict(bad[0][0],score=.9)
        with self.assertRaises(RuntimeError):m.merge_policy_predictions(bad,ids,native,native)
        self.assertEqual(m.AP_percent(-1),-1);self.assertEqual(m.AP_percent(.68),68)
        with self.assertRaises(RuntimeError):m.AP_percent(True)

if __name__=='__main__':unittest.main()
