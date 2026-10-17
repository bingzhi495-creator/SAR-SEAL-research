"""Three pure metadata contracts; no scientific imports or numerical fixtures."""
import copy
import hashlib
import unittest
import prediction_core as P


def ref(name):
    return dict(path='E:/synthetic/'+name+'.json',bytes=10,sha256=hashlib.sha256(name.encode()).hexdigest())


def endpoint():
    sources={k:ref(k) for k in P.TRAINING_ROLES}
    for k,v in P.TRAINING_PINS.items(): sources[k]['sha256']=v
    refs={k:ref(k) for k in ('training_qualified','pair_commit','preparation','order')}
    cp={a:ref(a) for a in P.ARMS}
    identity=dict(sources=dict(core_source=sources['core_source'],
        component_sources={k:sources[k] for k in P.TRAINING_COMPONENTS}),
        recipe=dict(seed=P.SEED,images=741,epochs=50,updates=37050),
        context={'training':True},runtime={'python':'3.8.17'},order_sha256=P.ORDER_SHA)
    result=dict(checkpoint_refs=cp,pair_commit_ref=refs['pair_commit'],order_ref=refs['order'],
                paired_checkpoint_prefix=copy.deepcopy(P.FINAL_CURSOR),paired_prefix_advanced=False)
    qualified=dict(schema=P.QUALIFIED_SCHEMA,qualified=True,paired_prefix_advanced=True,seed=P.SEED,
        span={'start':36864,'stop':37050},result=result,source=sources['entry_source'],
        body_source=sources['body_source'],runtime_source=sources['runtime_source'],core_source=sources['core_source'],
        module_sources=identity['sources']['component_sources'],context=ref('training_context'))
    pair=dict(schema='U40-paired-checkpoint-commit-v1',status='PAIRED_CHECKPOINTS_WRITTEN_PENDING_ROOT_ACCEPTANCE',
        cursor=copy.deepcopy(P.FINAL_CURSOR),span={'start':36864,'stop':37050},identity=identity,
        order_ref=refs['order'],checkpoints={a:dict(reference=cp[a]) for a in P.ARMS},
        traces={a:ref(a+'trace') for a in P.ARMS},actual_trace_sha256='a'*64)
    prep=dict(schema='ROOT-U40-continuous-training-preparation-v1',seed=P.SEED,images=741,GT=1661,
        epochs=50,total_updates_per_arm=37050,executable_sources=sources,
        module_sources=identity['sources']['component_sources'],recipe=identity['recipe'],
        context=qualified['context'],order_sha256=P.ORDER_SHA)
    return dict(refs=refs,training_qualified=qualified,pair_commit=pair,preparation=prep,
                order=dict(seed=P.SEED,order_sha256=P.ORDER_SHA))


class MetadataTests(unittest.TestCase):
    def test_only_actual_full_endpoint_and_exact_sources(self):
        m=endpoint()
        self.assertEqual(P.validate_final_endpoint(m)['checkpoint_refs'],m['training_qualified']['result']['checkpoint_refs'])
        for mutation in ('midpoint','worker_only','wrong_seed','wrong_source','one_arm'):
            bad=copy.deepcopy(m)
            if mutation=='midpoint': bad['training_qualified']['result']['paired_checkpoint_prefix']['global_updates']=256
            if mutation=='worker_only': bad['training_qualified']['paired_prefix_advanced']=False
            if mutation=='wrong_seed': bad['training_qualified']['seed']=3101
            if mutation=='wrong_source': bad['preparation']['executable_sources']['core_source']=ref('wrongcore')
            if mutation=='one_arm': del bad['training_qualified']['result']['checkpoint_refs']['FULL']
            with self.subTest(mutation=mutation),self.assertRaises((ValueError,KeyError)):
                P.validate_final_endpoint(bad)

    def test_export_queries_select_natural_axis_not_export_ordinal(self):
        qids=list(reversed(range(100)))
        originals=[dict(image_id=7,query_id=q,category_id=1,score=.25,segmentation={}) for q in qids]
        chunks=P.export_query_chunks(qids,originals,7)
        self.assertEqual(chunks[0],[99,98,97,96])
        self.assertEqual([q for chunk in chunks for q in chunk],qids)
        natural=['natural-'+str(q) for q in range(100)]
        self.assertEqual([natural[q] for q in chunks[0]],['natural-99','natural-98','natural-97','natural-96'])
        with self.assertRaises(ValueError): P.export_query_chunks(list(range(100)),originals,7)

    def test_complete_population_and_four_fixed_shards(self):
        records=[dict(dataset_index=i,image_id=1000+i,qids=list(range(100))) for i in range(187)]
        native=[dict(image_id=r['image_id'],query_id=q,category_id=1,score=.25,segmentation={})
                for r in records for q in r['qids']]
        self.assertEqual(P.validate_population(records,native,3,[141,187]),(141,187))
        for bad_records,bad_native,bad_span in ((records[:-1],native,[0,47]),
                (records,native[:-1],[0,47]),(records,native,[0,46])):
            with self.assertRaises(ValueError): P.validate_population(bad_records,bad_native,0,bad_span)


if __name__=='__main__': unittest.main()
