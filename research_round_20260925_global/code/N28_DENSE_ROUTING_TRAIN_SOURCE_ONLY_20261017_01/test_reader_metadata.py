"""Synthetic metadata only: no Torch, NumPy, annotations, tensors or data I/O."""
import copy
import importlib.util
from pathlib import Path
import unittest


def fixture():
    def ref(name):
        return dict(path='E:/SYNTHETIC_N28/'+name,bytes=1,sha256='0'*64)
    teacher=ref('teacher');report_ref=ref('report.json')
    fit=dict(images=741,GT=1661,dataset_indices=list(range(741)),teacher_checkpoint=teacher,
             identities=[],image_ids=[],existing_packet_refs=[],existing_native_L_refs=[])
    report=dict(schema='N13-sameK-region-FIT-cache-report-v1',images=741,raw_GT=1661,
                teacher_checkpoint=teacher,rows=[],validation_carrier_refs=[])
    annotation=1
    for index in range(741):
        G=3 if index<179 else 2
        identity=dict(image_id=10000+index,ann_ids=list(range(annotation,annotation+G)),
                      original_shape=[8,11],actual_img_shape=[580,800],batch_input_shape=[800,800])
        annotation+=G
        carrier=ref('carrier_%04d.pth'%index)
        fit['identities'].append(identity);fit['image_ids'].append(identity['image_id'])
        fit['existing_packet_refs'].append(ref('packet_%04d.pth'%index))
        fit['existing_native_L_refs'].append(ref('L_%04d.pth'%index))
        report['validation_carrier_refs'].append(carrier)
        report['rows'].append(dict(dataset_index=index,image_id=identity['image_id'],identity=identity,
            raw_GT=G,query_ids=list(range(2,2+G)),validation_carrier=carrier))
    qualified=dict(status='ROOT_N13_SAMEK_REGION_FIT_CACHE_CPU_QUALIFIED_V2',GPU_report=report_ref,
        teacher_checkpoint=teacher,images=741,raw_GT=1661,
        validation_carrier_refs=report['validation_carrier_refs'])
    return fit,qualified,report,report_ref


class PopulationMetadataTest(unittest.TestCase):
    def test_complete_order_and_corrupt_metadata(self):
        p=Path(__file__).with_name('reader.py')
        spec=importlib.util.spec_from_file_location('N28_METADATA_ONLY_READER',str(p))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        fit,qualified,report,report_ref=fixture()
        rows=module._population_records(fit,qualified,report,report_ref)
        self.assertEqual(len(rows),741)
        self.assertEqual(sum(row['G'] for row in rows),1661)
        self.assertEqual(rows[0],dict(dataset_index=0,image_id=10000,identity=fit['identities'][0],
                                     G=3,query_ids=[2,3,4]))
        bad=copy.deepcopy(report);bad['rows'][0]['query_ids']=[2,2,4]
        with self.assertRaises((ValueError,RuntimeError,AssertionError)):
            module._population_records(fit,qualified,bad,report_ref)
        bad=copy.deepcopy(report);bad['raw_GT']=1660
        with self.assertRaises((ValueError,RuntimeError,AssertionError)):
            module._population_records(fit,qualified,bad,report_ref)


if __name__=='__main__':
    unittest.main()
