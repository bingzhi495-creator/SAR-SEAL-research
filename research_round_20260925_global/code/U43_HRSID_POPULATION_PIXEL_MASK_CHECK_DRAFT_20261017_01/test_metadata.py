"""Fabricated split metadata/counters only; no real/fake pixel or mask success."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).with_name('population.py')
spec=importlib.util.spec_from_file_location('U43_POPULATION_METADATA_TARGET',str(path))
population=importlib.util.module_from_spec(spec)
spec.loader.exec_module(population)


def reference(name):
    return dict(path='E:/SYNTHETIC_ONLY/'+name,bytes=1,sha256=hashlib.sha256(name.encode()).hexdigest())


def fixture():
    context=dict(schema='U43-HRSID-official-split-pixel-mask-check-context-v1',
        image_root='E:/SYNTHETIC_ONLY/images',annotations={key:reference(key+'.json') for key in ('train','test')},
        image_refs_by_filename={})
    documents={}
    for split,n_images,n_annotations in (('train',3642,11047),('test',1962,5922)):
        images=[dict(id=i+1,file_name='%s_%04d.jpg'%(split,i),width=800,height=800) for i in range(n_images)]
        annotations=[dict(id=i+1,image_id=i%n_images+1,category_id=1,iscrowd=int(i==0),
            area=1.5,bbox=[0,0,2,2],segmentation=[[0,0,2,0,0,2]]) for i in range(n_annotations)]
        annotations[0]['segmentation'].append([3,3,5,3,3,5])
        documents[split]=dict(images=images,annotations=annotations,categories=[dict(id=1,name='ship')])
        for image in images:
            value=reference('images/'+image['file_name']);context['image_refs_by_filename'][image['file_name']]=value
    return context,documents


class PopulationMetadata(unittest.TestCase):
    def test_split_id_namespaces_keep_original_order_and_annotation_groups(self):
        context,documents=fixture();before=deepcopy((context,documents))
        plan=population.validate_population_metadata(context,documents)
        self.assertEqual(set(plan['splits']),{'train','test'})
        for split,n_images,n_annotations in (('train',3642,11047),('test',1962,5922)):
            current=plan['splits'][split]
            self.assertIs(current['images'],documents[split]['images'])
            self.assertEqual(len(current['images']),n_images)
            self.assertEqual(sum(len(rows) for rows in current['annotations_by_image'].values()),n_annotations)
            first=current['annotations_by_image'][1][0]
            self.assertIs(first,documents[split]['annotations'][0])
            self.assertEqual((first['id'],first['iscrowd'],len(first['segmentation'])),(1,1,2))
            self.assertEqual([a['id'] for a in current['annotations_by_image'][1]],
                [a['id'] for a in documents[split]['annotations'] if a['image_id']==1])
        self.assertEqual((context,documents),before)

    def test_wrong_split_counts_or_duplicate_local_ids_and_json_keys_are_rejected(self):
        context,documents=fixture()
        removed=documents['train']['annotations'].pop()
        with self.assertRaises(ValueError):population.validate_population_metadata(context,documents)
        documents['train']['annotations'].append(removed)
        original=documents['test']['annotations'][1]['id']
        documents['test']['annotations'][1]['id']=documents['test']['annotations'][0]['id']
        with self.assertRaises(ValueError):population.validate_population_metadata(context,documents)
        documents['test']['annotations'][1]['id']=original
        with self.assertRaises(ValueError):population.decode_metadata_json(b'{"images":[],"images":[]}')

    def test_failure_or_unvisited_input_counters_never_grant_complete_status(self):
        # Scalar bookkeeping only, not an image/mask checker replacement.
        summaries={}
        for split,n_images,n_annotations in (('train',3642,11047),('test',1962,5922)):
            summaries[split]=dict(declared_images=n_images,declared_annotations=n_annotations,
                processed_images=n_images,recorded_images=n_images,original_annotations_seen=n_annotations,
                image_checks_passed=n_images,image_checks_failed=0,pixel_decode_passed_images=n_images,
                pixel_decode_failed_images=0,empty_annotation_images=0,raster_attempted_annotations=n_annotations,
                raster_passed_annotations=n_annotations,raster_failed_annotations=0,
                annotation_metadata_failed_count=0,not_attempted_annotations=0)
        failure=deepcopy(summaries)
        failure['test']['image_checks_passed']-=1;failure['test']['image_checks_failed']+=1
        result=population.completion_status(failure)
        self.assertFalse(result['input_check_complete'])
        self.assertNotEqual(result['status'],'INPUT_CHECK_COMPLETE_PENDING_ROOT')
        partial=deepcopy(summaries);partial['train']['processed_images']-=1;partial['train']['recorded_images']-=1
        self.assertFalse(population.completion_status(partial)['input_check_complete'])
        missing=deepcopy(summaries);missing['test']['raster_passed_annotations']-=1
        missing['test']['raster_attempted_annotations']-=1;missing['test']['not_attempted_annotations']+=1
        self.assertFalse(population.completion_status(missing)['input_check_complete'])


if __name__=='__main__':unittest.main()
