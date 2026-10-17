"""Invented reference dictionaries only; no tensor, GT or scientific execution."""
import copy
import unittest
import dev_reader


def reference(name):
    return dict(path='E:/invented/'+name, bytes=0, sha256='a'*64)


def fixture():
    rows=[]
    for index in range(187):
        rows.append(dict(dataset_index=index,image_id=index+1000,qids=list(range(99,-1,-1)),
                         original_shape=[100,120],bundle=reference('b%d'%index),
                         shared_locator='i%03d/SHARED_ACTUAL_INPUTS.pth'%index,
                         original_capture_manifest=reference('manifest'),capture_binding={'fake':True}))
    legacy=dict(schema='ROOT-N23-qualified-cache-DEV-context-v1',rows=rows,
                image_ids=[row['image_id'] for row in rows])
    for key in ('native_predictions','R_predictions','encoder_source','legacy_ROOT_result'):
        legacy[key]=reference(key)
    context=dict(schema='ROOT-N28-fixed24-dense-native-DEV-context-v1',legacy_context=reference('legacy'),
                 rows=[dict(row,dense_packet=reference('dense%d'%i)) for i,row in enumerate(rows)],
                 image_ids=list(legacy['image_ids']))
    for key in ('native_predictions','R_predictions','encoder_source','legacy_ROOT_result'):
        context[key]=copy.deepcopy(legacy[key])
    return context,legacy


class ReaderMetadataTests(unittest.TestCase):
    def test_complete_original_dense_refs_keep_export_query_order(self):
        context,legacy=fixture()
        records,refs=dev_reader.prepare_records(context,legacy)
        self.assertEqual(records,context['rows'])
        self.assertEqual(len(records),187)
        self.assertEqual(records[0]['qids'],list(range(99,-1,-1)))
        self.assertEqual(len(refs),380)
        records[0]['qids'][0]=-1
        self.assertEqual(context['rows'][0]['qids'][0],99)
        for mutation in ('missing_image','query_duplicate','wrong_dense_ref','changed_source'):
            bad=copy.deepcopy(context)
            if mutation=='missing_image':bad['rows'].pop()
            elif mutation=='query_duplicate':bad['rows'][1]['qids'][0]=bad['rows'][1]['qids'][1]
            elif mutation=='wrong_dense_ref':bad['rows'][1]['dense_packet']['sha256']='z'*64
            else:bad['native_predictions']=reference('different')
            with self.assertRaises((ValueError,RuntimeError)):
                dev_reader.prepare_records(bad,legacy)


if __name__=='__main__':unittest.main()
