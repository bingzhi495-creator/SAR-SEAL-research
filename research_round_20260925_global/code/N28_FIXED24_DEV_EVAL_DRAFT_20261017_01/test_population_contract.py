"""Synthetic JSON metadata only; not mask decoding, NN inference or COCO tests."""
import copy
import unittest

import population_contract as contract


def image_predictions(image_id=501):
    qids=[(ordinal*37)%100 for ordinal in range(100)]
    values=[dict(image_id=image_id,query_id=q,category_id=1,score=(q+1)/101,
        segmentation=dict(size=[2,3],counts='SYNTHETIC_OLD_%d'%q),
        trace=dict(ordinal=ordinal)) for ordinal,q in enumerate(qids)]
    return values,qids


def full_population():
    original=[]
    for index in range(187):
        rows,_=image_predictions(1000+13*index)
        original.extend(rows)
    ranges=[[0,47],[47,94],[94,141],[141,187]]
    shards=[]
    for first,stop in ranges:
        values=copy.deepcopy(original[first*100:stop*100])
        for row in values:
            row['segmentation']['counts']='SYNTHETIC_CHANGED'
        shards.append(dict(dataset_range=[first,stop],predictions=values))
    return original,shards


class PopulationContractTest(unittest.TestCase):
    def test_replace_only_geometry_in_non_natural_order_without_aliases(self):
        original,qids=image_predictions()
        new=[dict(size=[2,3],counts='SYNTHETIC_NEW_%d'%q) for q in qids]
        old_before,new_before=copy.deepcopy(original),copy.deepcopy(new)
        actual=contract.replace_segmentations(original,new,501,qids)
        self.assertEqual([row['query_id'] for row in actual],qids)
        self.assertEqual([row['segmentation'] for row in actual],new)
        self.assertEqual([{k:v for k,v in row.items() if k!='segmentation'} for row in actual],
                         [{k:v for k,v in row.items() if k!='segmentation'} for row in original])
        actual[0]['trace']['ordinal']=-1
        actual[0]['segmentation']['size'][0]=999
        self.assertEqual(original,old_before)
        self.assertEqual(new,new_before)

    def test_replace_rejects_invalid_population_order_and_scores(self):
        original,qids=image_predictions()
        new=[dict(size=[2,3],counts='SYNTHETIC_NEW') for _ in qids]
        cases=[]
        cases.append((original[:-1],new,501,qids))
        cases.append((original,new[:-1],501,qids))
        cases.append((original,new,999,qids))
        cases.append((original,new,501,list(reversed(qids))))
        duplicate=list(qids);duplicate[1]=duplicate[0]
        cases.append((original,new,501,duplicate))
        for field,value in (('image_id',999),('category_id',2),('score',float('nan')),
                            ('score',float('inf')),('score',True),('query_id',True)):
            bad=copy.deepcopy(original);bad[0][field]=value
            cases.append((bad,new,501,qids))
        for arguments in cases:
            with self.subTest(arguments=arguments[0][0]):
                with self.assertRaises(ValueError):
                    contract.replace_segmentations(*arguments)

    def test_merge_four_complete_shards_preserves_canonical_order_without_aliases(self):
        original,shards=full_population()
        expected=[row for shard in shards for row in shard['predictions']]
        actual=contract.merge_predictions(shards,original)
        self.assertEqual(len(actual),18700)
        self.assertEqual(actual,expected)
        actual[0]['trace']['ordinal']=-1
        actual[0]['segmentation']['size'][0]=999
        self.assertEqual(shards[0]['predictions'][0]['trace']['ordinal'],0)
        self.assertEqual(shards[0]['predictions'][0]['segmentation']['size'],[2,3])
        self.assertEqual(original[0]['segmentation']['counts'],'SYNTHETIC_OLD_0')

    def test_replace_requires_explicit_original_query_order(self):
        original,_=image_predictions()
        new=[dict(size=[2,3],counts='SYNTHETIC_NEW') for _ in range(100)]
        with self.assertRaises(ValueError):
            contract.replace_segmentations(original,new,501,None)

    def test_merge_rejects_missing_duplicate_score_identity_or_order_changes(self):
        original,shards=full_population()
        def missing(values):
            values[0]['predictions'].pop()
        def duplicate(values):
            values[0]['predictions'][1]=copy.deepcopy(values[0]['predictions'][0])
        def changed_score(values):
            values[0]['predictions'][0]['score']+=.125
        def changed_extra_field(values):
            values[0]['predictions'][0]['trace']['ordinal']=-1
        def changed_order(values):
            rows=values[0]['predictions'];rows[0],rows[1]=rows[1],rows[0]
        def changed_range(values):
            values[0]['dataset_range']=[0,46]
        for change in (missing,duplicate,changed_score,changed_extra_field,changed_order,changed_range):
            bad=copy.deepcopy(shards);change(bad)
            with self.subTest(change=change.__name__):
                with self.assertRaises(ValueError):
                    contract.merge_predictions(bad,original)
        for bad in (shards[:-1],list(reversed(shards))):
            with self.assertRaises(ValueError):
                contract.merge_predictions(bad,original)
        with self.assertRaises(ValueError):
            contract.merge_predictions(shards,original[:-1])
        with self.assertRaises(ValueError):
            contract.merge_predictions(shards,original,ranges=[[0,46],[46,94],[94,141],[141,187]])


if __name__=='__main__':
    unittest.main()
