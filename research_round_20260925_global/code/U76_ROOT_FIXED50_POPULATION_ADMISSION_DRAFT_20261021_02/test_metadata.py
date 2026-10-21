"""U76-only stdlib synthetic byte/metadata checks; never actual proof/data I/O."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent


def load():
    spec=importlib.util.spec_from_file_location('_U76_TEST',HERE/'admission.py')
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def ref(path,raw):
    return dict(path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode('utf-8')


def population_fixture():
    """Explicitly synthetic proof/source bytes; no genuine Root observations."""
    helper=b'''def validate_accepted_proof(proof,prediction):
    if proof.get('schema')!='Root-U75-DGX-fixed50-shard-accepted-v1' or proof.get('status')!='ROOT_ACCEPTED_SINGLE_SHARD_PREDICTION' or proof.get('root_accepted') is not True:
        raise ValueError('Synthetic accepted proof validator rejects owner/pending input')
    if proof['endpoint']!=prediction['endpoint'] or proof['source_refs']!=prediction['source_refs']:
        raise ValueError('Synthetic accepted proof content join')
    return proof
'''
    helper_ref=ref('/synthetic/U75_acceptance.py',helper)
    roles=('training_state','training_entry','u36_helpers','u39_training_ops','shared_initialization',
        'full_legacy','full_dsconv','full_head','full_feature','plain_legacy','plain_feature','instance_adapter',
        'restore_state','portable_reader','image_geometry','prediction_core','encoder','mask_api',
        'population_contract','dev_reader','dev_rgb_reader')
    sources={k:ref('/synthetic/source/'+k+'.py',b'SYNTHETIC_SOURCE') for k in roles}
    fit=ref('/synthetic/source/fit_portable_reader.py',b'SYNTHETIC_FIT_SOURCE')
    libraries={k:ref('/synthetic/lib/'+k,b'SYNTHETIC_LIBRARY') for k in ('torch','numpy','psutil','pycocotools','mask_extension')}
    endpoint=dict(schema='U58-U57-DGX-fixed-endpoint-metadata-v1',mode='FORMAL',formal_endpoint=True,
        cursor=dict(global_updates=37050,completed_epochs=50,within_epoch=0),run_id='SYNTHETIC_ONLY',
        checkpoint_refs={k:ref('/synthetic/checkpoints/'+k+'.pth',b'NEVER_READ') for k in ('PLAIN','FULL')},
        pair_commit_ref=ref('/synthetic/PAIR_COMMIT.json',b'NEVER_READ'))
    proofs,predictions,proof_refs,prediction_refs,raws=[],[],[],[],{helper_ref['path']:helper}
    ranges=((0,47),(47,94),(94,141),(141,187))
    for index,(first,stop) in enumerate(ranges):
        identity=[dict(image_id=i+1000,query_id=q,category_id=1,score=0.,segmentation={'synthetic_only':True})
            for i in range(first,stop) for q in reversed(range(100))]
        support=[dict(dataset_index=first+n//100,image_id=p['image_id'],query_id=p['query_id'],Omega={'synthetic_only':True})
            for n,p in enumerate(identity)]
        arms={a:dict(status='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE',predictions=deepcopy(identity),
            identity_predictions=deepcopy(identity),Omega_support=deepcopy(support) if a=='PLAIN' else [],
            cost=dict(encode_calls=stop-first,readout_calls=25*(stop-first),queries=100*(stop-first)),
            proofs=[dict(dataset_index=i,image_id=i+1000,queries=100,zero_residual_native_L_exact=True,
                native_RLE_and_prediction_dictionaries_exact=True,original_full_S0_class_order_preserved=True) for i in range(first,stop)])
            for a in ('PLAIN','FULL')}
        prediction=dict(schema='U66-DGX-fixed50-DEV-shard-pending-v1',status='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE',
            endpoint=deepcopy(endpoint),source_refs=deepcopy(sources),shard_index=index,dataset_range=[first,stop],
            images=stop-first,queries=100*(stop-first),fixed_epoch=50,updates_per_arm=37050,query_batch=4,
            arms=arms,predictions={a:arms[a]['predictions'] for a in arms},cost={a:arms[a]['cost'] for a in arms},
            identity_predictions=deepcopy(identity),Omega_support=deepcopy(support),qualified=False,AP=False,DEV=True,
            DEV_technical_input=True,TEST=False,new_training=False,DEV_metrics_computed=False,whole_population_predicted=False,
            GT_used_for_prediction=False,GT_rasterization_executed=False,original_full_S0_class_order_preserved=True)
        prediction_raw=encoded(prediction);prediction_ref=ref('/synthetic/SHARD_'+str(index)+'.json',prediction_raw)
        proof=dict(schema='Root-U75-DGX-fixed50-shard-accepted-v1',status='ROOT_ACCEPTED_SINGLE_SHARD_PREDICTION',
            root_accepted=True,qualified=False,AP=False,DEV=True,DEV_technical_input=True,TEST=False,new_training=False,
            formal_training_budget_credit=0,endpoint=deepcopy(endpoint),source_refs=deepcopy(sources),
            outer_source_refs=dict(deepcopy(sources),fit_portable_reader=deepcopy(fit)),library_sources=deepcopy(libraries),
            evidence_refs={'manifest':ref('/synthetic/ORIGINAL_DEV187_MANIFEST.json',b'SYNTHETIC_MANIFEST_ONLY')},
            prediction_ref=prediction_ref,shard_index=index,dataset_range=[first,stop],images=stop-first,queries=100*(stop-first),
            fixed_epoch=50,updates_per_arm=37050)
        proof_raw=encoded(proof);proof_ref=ref('/synthetic/ROOT_ACCEPTED_'+str(index)+'.json',proof_raw)
        proofs.append(proof);predictions.append(prediction);proof_refs.append(proof_ref);prediction_refs.append(prediction_ref)
        raws[prediction_ref['path']]=prediction_raw;raws[proof_ref['path']]=proof_raw
    cpu_request=dict(worker_source=ref('/synthetic/U70.py',b'NEVER_READ'),core_source=ref('/synthetic/U67.py',b'NEVER_READ'),
        module_sources={k:sources[k] if k in ('population_contract','mask_api') else ref('/synthetic/'+k+'.py',b'NEVER_READ')
            for k in ('population_contract','eval_body','official','cocoeval','mask_api','geometry_diagnostic','population_metrics')},
        library_sources={k:libraries[k] if k in libraries else ref('/synthetic/lib/'+k,b'NEVER_READ')
            for k in ('numpy','psutil','pycocotools','coco_loader','mask_extension')},
        input_refs={k:ref('/synthetic/input/'+k+'.json',b'NEVER_READ')
            for k in ('context','legacy_context','legacy_result','native_predictions','R_predictions','DEV')},
        shard_refs=prediction_refs,output_directory='/synthetic/FUTURE_CPU_WORKER')
    refs=sorted([helper_ref]+proof_refs+prediction_refs,key=lambda r:r['path'])
    return dict(input_refs=refs,raw_inputs=raws,proof_refs=proof_refs,prediction_refs=prediction_refs,
        endpoint=endpoint,cpu_request=cpu_request,acceptance_source_ref=helper_ref),proofs,predictions


def refresh(args,proofs,predictions):
    raw={args['acceptance_source_ref']['path']:args['raw_inputs'][args['acceptance_source_ref']['path']]}
    prediction_refs=[]
    for i,value in enumerate(predictions):
        blob=encoded(value);value_ref=ref(args['prediction_refs'][i]['path'],blob)
        prediction_refs.append(value_ref);raw[value_ref['path']]=blob
    proof_refs=[]
    for i,value in enumerate(proofs):
        value['prediction_ref']=deepcopy(prediction_refs[i])
        blob=encoded(value);value_ref=ref(args['proof_refs'][i]['path'],blob)
        proof_refs.append(value_ref);raw[value_ref['path']]=blob
    args.update(raw_inputs=raw,proof_refs=proof_refs,prediction_refs=prediction_refs)
    args['cpu_request']['shard_refs']=deepcopy(prediction_refs)
    args['input_refs']=sorted([args['acceptance_source_ref']]+proof_refs+prediction_refs,key=lambda r:r['path'])


def admit_synthetic(module,args):
    with patch.object(module,'ACCEPTANCE_PIN',{k:args['acceptance_source_ref'][k] for k in ('bytes','sha256')}):
        return module.admit_population(**args)


def genuine_validator_fixture():
    """Use frozen U75's original validator and its synthetic-only public fixture.

    The fixture patches component pin globals only in temporary definitions;
    it never changes the frozen U75 validator or production U76 source pin.
    """
    directory=HERE.parent/'U75_ROOT_FIXED50_SHARD_ACCEPTANCE_DRAFT_20261021_01'
    spec=importlib.util.spec_from_file_location('acceptance',directory/'acceptance.py')
    acceptance=importlib.util.module_from_spec(spec);spec.loader.exec_module(acceptance)
    fixture_spec=importlib.util.spec_from_file_location('_U75_SYNTHETIC_FIXTURE',directory/'test_metadata.py')
    fixture=importlib.util.module_from_spec(fixture_spec)
    with patch.dict(sys.modules,{'acceptance':acceptance}):
        fixture_spec.loader.exec_module(fixture)
    raw=(directory/'acceptance.py').read_bytes()
    source_ref=ref(str(directory/'acceptance.py'),raw)
    raws={source_ref['path']:raw};proofs=[];prediction_refs=[];proof_refs=[]
    for index in range(4):
        proof,data=fixture.accept_synthetic(index)
        prediction_ref=deepcopy(proof['prediction_ref'])
        # This is the original byte array produced by U75's synthetic fixture.
        raws[prediction_ref['path']]=data['objects'][prediction_ref['path']]
        proof_raw=encoded(proof);proof_ref=ref('/synthetic/ROOT_U75_'+str(index)+'.json',proof_raw)
        raws[proof_ref['path']]=proof_raw
        proofs.append(proof);proof_refs.append(proof_ref);prediction_refs.append(prediction_ref)
    cpu=population_fixture()[0]['cpu_request']
    for role in ('population_contract','mask_api'):
        cpu['module_sources'][role]=deepcopy(proofs[0]['source_refs'][role])
    for role in ('pycocotools','mask_extension'):
        cpu['library_sources'][role]=deepcopy(proofs[0]['library_sources'][role])
    cpu['shard_refs']=deepcopy(prediction_refs)
    return dict(input_refs=sorted([source_ref]+proof_refs+prediction_refs,key=lambda r:r['path']),
        raw_inputs=raws,proof_refs=proof_refs,prediction_refs=prediction_refs,
        endpoint=deepcopy(proofs[0]['endpoint']),cpu_request=cpu,acceptance_source_ref=source_ref)


class Tests(unittest.TestCase):
    def test_original_raw_bytes_bind_the_parsed_artifact(self):
        module=load()
        raw=b'{ "synthetic_only": true, "ordinal": 0 }\n'
        reference=ref('/synthetic/proof0.json',raw)
        self.assertEqual(module.decode_artifact(reference,raw),dict(synthetic_only=True,ordinal=0))
        # Equal parsed JSON has different actual source bytes and must not pass.
        normalized=b'{"ordinal":0,"synthetic_only":true}\n'
        with self.assertRaisesRegex(ValueError,'bytes'):
            module.decode_artifact(reference,normalized)

    def test_explicit_input_ref_list_and_actual_raw_keys_are_exact(self):
        module=load()
        raw={('/synthetic/'+str(i)+'.json'):b'{}\n' for i in range(9)}
        refs=[ref(path,raw[path]) for path in sorted(raw)]
        self.assertEqual(module.close_inputs(refs,raw,refs),raw)
        bad=deepcopy(raw);bad['/synthetic/extra.json']=b'{}\n'
        with self.assertRaisesRegex(ValueError,'input'):
            module.close_inputs(refs,bad,refs)
        changed=deepcopy(refs);changed[0]['path']='/synthetic/substituted.json'
        with self.assertRaisesRegex(ValueError,'input'):
            module.close_inputs(changed,raw,refs)

    def test_four_proof_population_is_only_future_CPU_material(self):
        module=load();args,proofs,predictions=population_fixture()
        with patch.object(module,'ACCEPTANCE_PIN',{k:args['acceptance_source_ref'][k] for k in ('bytes','sha256')}):
            result=module.admit_population(**args)
        self.assertEqual(result['external_shard_admission_refs'],args['proof_refs'])
        self.assertEqual(result['shard_refs'],args['prediction_refs'])
        self.assertEqual((result['images'],result['queries']),(187,18700))
        self.assertEqual(result['image_ids'],list(range(1000,1187)))
        self.assertFalse(result['qualified']);self.assertFalse(result['AP'])
        self.assertFalse(result['CPU_execution_performed'])

    def test_fixed_order_and_original_file_links_reject_splicing(self):
        module=load()
        for case in ('order','copied_proof','new_prediction_path','missing','owner_self_report'):
            args,proofs,predictions=population_fixture()
            if case=='order': args['proof_refs'].reverse()
            elif case=='copied_proof':
                proofs[1]=deepcopy(proofs[0]);refresh(args,proofs,predictions)
            elif case=='new_prediction_path':
                old=args['prediction_refs'][0]['path'];new='/synthetic/RELOCATED_SHARD_0.json'
                args['raw_inputs'][new]=args['raw_inputs'].pop(old)
                args['prediction_refs'][0]=dict(args['prediction_refs'][0],path=new)
                args['input_refs']=sorted([args['acceptance_source_ref']]+args['proof_refs']+args['prediction_refs'],key=lambda r:r['path'])
            elif case=='missing': args['proof_refs'].pop()
            else:
                proofs[0].update(schema='U74-Linux-owned-fixed50-single-shard-prediction-receipt-v1',
                    status='OWNED_CHILD_COMPLETE_PENDING_ROOT_REVIEW')
                refresh(args,proofs,predictions)
            with self.subTest(case=case),self.assertRaises(ValueError):
                admit_synthetic(module,args)

    def test_whole_endpoint_sources_and_CPU_mask_identity_cannot_drift(self):
        module=load()
        for case in ('endpoint_arg','run','checkpoint','source','outer_fit','population_path','mask_path','extension'):
            args,proofs,predictions=population_fixture()
            if case=='endpoint_arg': args['endpoint']['run_id']='OTHER_RUN'
            elif case in ('run','checkpoint'):
                for value in (proofs[2],predictions[2]):
                    if case=='run': value['endpoint']['run_id']='OTHER_RUN'
                    else: value['endpoint']['checkpoint_refs']['PLAIN']['path']='/synthetic/OTHER.pth'
                refresh(args,proofs,predictions)
            elif case=='source':
                for value in (proofs[2],predictions[2]): value['source_refs']['encoder']['path']='/synthetic/OTHER_ENCODER.py'
                proofs[2]['outer_source_refs']['encoder']['path']='/synthetic/OTHER_ENCODER.py'
                refresh(args,proofs,predictions)
            elif case=='outer_fit':
                proofs[2]['outer_source_refs']['fit_portable_reader']['path']='/synthetic/OTHER_FIT.py'
                refresh(args,proofs,predictions)
            elif case in ('population_path','mask_path'):
                role='population_contract' if case=='population_path' else 'mask_api'
                args['cpu_request']['module_sources'][role]['path']='/synthetic/COPIED_'+role+'.py'
            else: args['cpu_request']['library_sources']['mask_extension']['path']='/synthetic/OTHER_mask.so'
            with self.subTest(case=case),self.assertRaises(ValueError):
                admit_synthetic(module,args)

    def test_duplicate_image_across_individually_accepted_ranges_is_rejected(self):
        module=load()
        args,proofs,predictions=population_fixture();value=predictions[1]
        old=value['identity_predictions'][0]['image_id'];new=predictions[0]['identity_predictions'][0]['image_id']
        for rows in [value['identity_predictions'],value['Omega_support']]+[
                rows for arm in value['arms'].values() for rows in (arm['predictions'],arm['identity_predictions'],arm['Omega_support'])]:
            for row in rows:
                if row['image_id']==old: row['image_id']=new
        for arm in value['arms'].values(): arm['proofs'][0]['image_id']=new
        refresh(args,proofs,predictions)
        with self.assertRaises(ValueError): admit_synthetic(module,args)

    def test_genuine_fixed_U75_validator_is_used_without_source_pin_override(self):
        module=load();args=genuine_validator_fixture()
        result=module.admit_population(**args)
        self.assertEqual(result['external_shard_admission_refs'],args['proof_refs'])
        self.assertEqual(result['shard_refs'],args['cpu_request']['shard_refs'])
        self.assertEqual(result['image_ids'],list(range(1000,1187)))
        self.assertEqual((result['images'],result['queries']),(187,18700))
        self.assertFalse(result['qualified']);self.assertFalse(result['AP'])
        # Keep the original eleven embedded raw JSON artifacts untouched. A
        # fabricated top-level terminal exit conflicts with their genuine gate.
        first=args['proof_refs'][0];proof=json.loads(args['raw_inputs'][first['path']])
        proof['original_terminal']['owner_process_exit_code']=1
        blob=encoded(proof);replacement=ref(first['path'],blob)
        args['proof_refs'][0]=replacement;args['raw_inputs'][first['path']]=blob
        args['input_refs']=sorted([args['acceptance_source_ref']]+args['proof_refs']+args['prediction_refs'],key=lambda r:r['path'])
        with self.assertRaises(ValueError): module.admit_population(**args)

    def test_individually_accepted_shards_must_share_the_same_actual_manifest_ref(self):
        module=load();args,proofs,predictions=population_fixture()
        proofs[2]['evidence_refs']['manifest']['path']='/synthetic/COPIED_DEV187_MANIFEST.json'
        # Same content identity and same selected image IDs are insufficient:
        # each single-shard gate can admit its own distinct actual manifest.
        refresh(args,proofs,predictions)
        with self.assertRaisesRegex(ValueError,'manifest'):
            admit_synthetic(module,args)


if __name__=='__main__':
    unittest.main()
