"""Finite stdlib/stub delta tests; no genuine tensor, ML, input or process."""
import json
from types import SimpleNamespace
import unittest

import validation_body as body


class TensorToken:
    """Weak-referenceable opaque token, not a tensor or raster."""
    pass


class TorchToken:
    Tensor=TensorToken
    cuda=SimpleNamespace(is_initialized=lambda:False)


class ControlToken:
    def __init__(self):self.checks=0
    def check(self):self.checks+=1


def fixture():
    entries=[dict(dataset_index=index,image_id=1000+index,query_ids=list(range(7,100))+list(range(7)),
        package_ref=dict(path='/synthetic-never-open/'+str(index)+'.pt',bytes=0,sha256='a'*64)) for index in range(187)]
    return dict(entries=entries,synthetic_manifest_metadata_only=True)


class ReaderToken:
    def __init__(self,entries,loads,fail=None):
        self.entries,self.loads,self.fail=entries,loads,fail
        self.records=[dict(dataset_index=row['dataset_index'],image_id=row['image_id'],qids=list(row['query_ids'])) for row in entries]
    def load(self,index):
        self.loads.append(index)
        if index==self.fail:raise ValueError('synthetic load failure')
        row=dict(self.records[index])
        item={name:TensorToken() for name in ('rgb400','e','M','L')}
        item.update(identity=dict(image_id=row['image_id']),query_ids=list(row['qids']),geometry=object())
        originals=[dict(image_id=row['image_id'],query_id=qid) for qid in row['qids']]
        return item,row,originals


class ModuleToken:
    """Intentionally has no visits/failed_loads/last_audit legacy properties."""
    def __init__(self,fail=None):self.loads=[];self.fail=fail;self.prepared=None
    def prepare(self,torch,c,entries,*,image_geometry,source_refs):
        self.prepared=(entries,image_geometry,source_refs)
        return ReaderToken(entries,self.loads,self.fail)


class BodyTests(unittest.TestCase):
    def test_new_prelock_has_full187_queries_without_matched_targets(self):
        pre=body.prelock_contract();body.validate_prelock(pre)
        self.assertEqual(pre['images'],187);self.assertEqual(pre['queries'],18700)
        self.assertEqual(pre['dataset_indices'],list(range(187)))
        self.assertTrue(pre['no_DEV'] and pre['DEV_technical_input'])
        self.assertNotIn('GT',pre)
        pre['dataset_indices']=list(range(186))
        with self.assertRaises(ValueError):body.validate_prelock(pre)

    def test_full187_tuple_reader_and_release_recording_without_old_reader_attributes(self):
        manifest=fixture();module=ModuleToken();report={};audits=[];control=ControlToken()
        refs=dict(dev_reader='source-token',dev_rgb_reader='source-token',image_geometry='source-token',reader='extra-not-forwarded')
        body.run(TorchToken,control,manifest,reader_module=module,image_geometry='geometry-module-token',
            source_refs=refs,prelock=body.prelock_contract(),report=report,record_image=audits.append)
        body.completed_population(report)
        self.assertIs(module.prepared[0],manifest['entries'])
        self.assertEqual(set(module.prepared[2]),{'dev_reader','dev_rgb_reader','image_geometry'})
        self.assertEqual(module.loads,list(range(187)));self.assertEqual(len(audits),187)
        self.assertEqual(report['queries'],18700);self.assertEqual(report['remaining_returned_tensor_objects'],0)
        self.assertFalse(report['DEV']);self.assertTrue(report['DEV_technical_input'])
        self.assertNotIn('matched_instances',report)
        for field in ('original_annotation_records_parsed','raw_provenance_inputs_revalidated'):
            report[field]=True
            with self.assertRaises(ValueError):body.completed_population(report)
            report[field]=False
        report['original_metadata_population']['queries']=18600
        with self.assertRaises(ValueError):body.completed_population(report)
        report['original_metadata_population']['queries']=18700
        for audit in audits:
            self.assertEqual(audit['queries'],100);self.assertTrue(audit['returned_tensor_objects_released'])
            self.assertEqual(set(audit),{'dataset_index','image_id','queries','status','source_package','original_record_sha256',
                'native_predictions_metadata_sha256','query_ids_metadata_sha256','CUDA_initialized','returned_tensor_objects_released'})
            json.dumps(audit,allow_nan=False)

    def test_loaded_metadata_refuses_target_fields_and_changed_query_order(self):
        manifest=fixture();reader=ReaderToken(manifest['entries'],[])
        item,row,originals=reader.load(0)
        item['targets']=TensorToken()
        with self.assertRaises(ValueError):body.validate_loaded_metadata(0,item,row,originals,manifest['entries'][0])
        del item['targets'];item['query_ids']=list(reversed(item['query_ids']))
        with self.assertRaises(ValueError):body.validate_loaded_metadata(0,item,row,originals,manifest['entries'][0])

    def test_failed_load_keeps_partial_real_attempt_counts_and_no_population_success(self):
        report={};module=ModuleToken(fail=3);audits=[]
        with self.assertRaises(ValueError):
            body.run(TorchToken,ControlToken(),fixture(),reader_module=module,image_geometry='geometry-module-token',
                source_refs=dict(dev_reader=0,dev_rgb_reader=0,image_geometry=0),prelock=body.prelock_contract(),
                report=report,record_image=audits.append)
        self.assertEqual(report['status'],'FAILED');self.assertEqual(report['attempted_images'],4)
        self.assertEqual(report['validated_images'],3);self.assertEqual(report['queries'],300)
        self.assertEqual(report['reader_visits'],[0,1,2,3]);self.assertEqual(report['reader_failed_loads'],1)
        self.assertEqual(len(audits),3);self.assertEqual(report['remaining_returned_tensor_objects'],0)
        with self.assertRaises(ValueError):body.completed_population(report)


if __name__=='__main__':unittest.main()
