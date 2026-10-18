"""Tiny pure metadata only: no actual manifests, ML, tensors or GT masks."""
import copy
import unittest
import portable_reader as core


def row_fixture():
    def ref(path,sha='a'*64):return dict(path=path,bytes=100,sha256=sha)
    identity=dict(image_id=7,ann_ids=[92,21],image=ref('E:/synthetic/image.jpg'),
        original_shape=[2,4],actual_img_shape=[400,800],batch_input_shape=[800,800])
    record=dict(dataset_index=0,image_id=7,G=2,query_ids=[73,5],identity=identity)
    fields={}
    for name,shape in dict(rgb400=[1,3,400,400],e=[2,256],M=[2,200,200],L=[2,2,4],targets=[2,2,4]).items():
        size=4
        for dimension in shape:size*=dimension
        fields[name]=dict(shape=shape,dtype='torch.float32',logical_bytes=size,
            source_sha256='a'*64,sha256='a'*64,bitexact=True,independent_storage=True,compact_storage=True)
    row=dict(dataset_index=0,record=record,package_ref=ref('/synthetic/portable0.pt'),
        original_package_ref=ref('D:/synthetic/MATCHED_FIT_0.pt'),export_index=0,
        summary=dict(images=1,matched_instances=2,fields=fields,package_bytes=100,
            package_sha256='a'*64,weights_only_CPU_roundtrip_verified=True))
    export=dict(root_qualified_ref=ref('E:/synthetic/accepted.json'),
        root_qualified_schema='ROOT-U51-matched-CPU-export-qualified-v1',
        package_schema='U51-matched-CPU-portable-item-v1',dataset_indices=list(range(32)))
    return row,export


class ManifestMetadata(unittest.TestCase):
    def test_full_original_index_coverage_never_claims_tensor_validation(self):
        indices=list(range(741))
        self.assertEqual(core.validate_coverage(indices,1661),
            dict(images=741,GT=1661,tensor_population_revalidated=False))
        for invalid in (indices[:-1],indices+[741],list(reversed(indices)),[0]+indices[0:740]):
            with self.subTest(length=len(invalid)):
                with self.assertRaises(ValueError):core.validate_coverage(invalid,1661)
        with self.assertRaises(ValueError):core.validate_coverage(indices,1660)

    def test_original_query_annotation_order_and_transfer_proof_remain_independent_metadata(self):
        row,export=row_fixture();before=copy.deepcopy(row)
        result=core.validate_row(row,export,0)
        self.assertEqual(result,row['record'])
        self.assertEqual(result['query_ids'],[73,5])
        self.assertEqual(result['identity']['ann_ids'],[92,21])
        result['query_ids'].reverse()
        self.assertEqual(row,before)
        changed=copy.deepcopy(row);changed['summary']['matched_instances']=1
        with self.assertRaises(ValueError):core.validate_row(changed,export,0)
        changed=copy.deepcopy(row);changed['package_ref']['sha256']='b'*64
        with self.assertRaises(ValueError):core.validate_row(changed,export,0)

    def test_historical_batches_require_exact_u51_u53_schema_and_indices(self):
        _,first=row_fixture()
        second=copy.deepcopy(first);second['dataset_indices']=list(range(32,64))
        third=copy.deepcopy(first)
        third.update(root_qualified_schema='ROOT-U53-matched-CPU-export-qualified-v1',
            package_schema='U53-matched-CPU-portable-item-v1',dataset_indices=list(range(64,741)))
        core.validate_export(first,0);core.validate_export(second,1);core.validate_export(third,2)
        wrong=copy.deepcopy(first);wrong['package_schema']='U49-matched-CPU-portable-item-v1'
        with self.assertRaises(ValueError):core.validate_export(wrong,0)
        wrong=copy.deepcopy(third);wrong['dataset_indices'].reverse()
        with self.assertRaises(ValueError):core.validate_export(wrong,2)
        with self.assertRaises(ValueError):core.validate_export(first,1)


if __name__=='__main__':unittest.main()
