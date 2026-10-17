"""Fabricated stdlib metadata only; no Torch, tensors, model or COCO execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

path = Path(__file__).with_name('run_eval.py')
loader = importlib.util.spec_from_file_location('U40_E50_ENTRY_METADATA_TARGET', str(path))
entry = importlib.util.module_from_spec(loader)
loader.loader.exec_module(entry)


def ref(label):
    return dict(path='E:/SYNTHETIC_ONLY/'+label, bytes=1, sha256='a'*64)


def contract(gpu=True):
    spec = dict(schema=('ROOT-U40-fixed50-full-method-DEV-prediction-spec-v1' if gpu
                       else 'ROOT-U40-fixed50-complete-DEV-metrics-spec-v1'),
        operation=('GPU_U40_FIXED50_FULL_METHOD_DEV_SHARD' if gpu else 'CPU_U40_FIXED50_FULL_DEV_COCO'),
        seed=202610173101, fixed_epoch=50, updates_per_arm=37050,
        owner_index=7, normal_RAW_count=7, total_failed_RAW_count=2,
        total_terminal_identity_pin_count=9, legacy_pin_count=8,
        allowed_cpus=[0,1], thread_limit=2, cuda_visible_devices='0' if gpu else '',
        whole_GPU_limit_MiB=8192, allocator_limit_MiB=4096,
        maximum_process_RSS_bytes=16*1024**3, minimum_available_RAM_bytes=16*1024**3,
        deadline_seconds=1800, automatic_retry=False, no_new_training=True, no_TEST=True,
        no_AP=gpu, no_DEV=False, output_directory='E:/SYNTHETIC_ONLY/OUTPUT')
    if gpu:
        spec.update(shard_index=3, dataset_range=[141,187])
    return spec


def declared_inputs(gpu=True):
    spec = contract(gpu)
    for key in ('entry_source','body_source','runtime_source','prediction_core_source','metrics_source',
                'prelock','parent_ROOT_qualified','training_qualified','pair_commit_ref','order_ref',
                'dev_context','coco_context'):
        spec[key] = ref(key+'.py' if key.endswith('source') else key+'.json')
    training_roles = ('shared_initialization','rgb_matched_reader','full_legacy','full_dsconv',
        'full_head','full_feature','plain_legacy','plain_feature','instance_adapter',
        'matched_reader','image_geometry','u36_helpers','u39_training_ops',
        'entry_source','body_source','runtime_source','core_source')
    prediction_roles = ('shared_initialization','full_legacy','full_dsconv','full_head','full_feature',
        'plain_legacy','plain_feature','instance_adapter','u36_helpers','u39_training_ops',
        'image_geometry','training_state','dev_reader','dev_rgb_reader','population_contract','mask_api','encoder')
    spec['training_sources'] = {key:ref('trained_'+key+'.py') for key in training_roles}
    spec['prediction_sources'] = {key:spec['training_sources'].get(key,ref('prediction_'+key+'.py'))
                                  for key in prediction_roles}
    spec['prediction_sources']['training_state'] = spec['training_sources']['core_source']
    spec['checkpoint_refs'] = {arm:ref(arm+'.pth') for arm in ('PLAIN','FULL')}
    spec['original_image_refs'] = [ref('image%03d.jpg'%i) for i in range(187)]
    spec['endpoint_refs'] = dict(training_qualified=spec['training_qualified'], pair_commit=spec['pair_commit_ref'],
                                 preparation=ref('preparation.json'), order=spec['order_ref'])
    spec['metrics_module_sources'] = {key:ref('metrics_'+key+'.py')
        for key in ('legacy_eval','population','geometry','official','coco','cocoeval','masks')}
    spec['accepted_GPU_shards'] = [ref('Root_shard%d.json'%i) for i in range(4)]
    allrefs = [value for key,value in spec.items() if type(value) is dict and set(value)=={'path','bytes','sha256'}]
    for key in ('training_sources','prediction_sources','checkpoint_refs','endpoint_refs','metrics_module_sources'):
        allrefs.extend(spec[key].values())
    allrefs.extend(spec['accepted_GPU_shards'])
    if gpu:
        first,stop=spec['dataset_range'];allrefs.extend(spec['original_image_refs'][first:stop])
    spec['sources'] = list({value['path']:value for value in allrefs}.values())
    pre = dict(schema='U40-fixed50-complete-DEV-metrics-prelock-v1',seed=202610173101,
        fixed_epoch=50,updates_per_arm=37050,images=187,GT=380,queries=18700,
        GPU_ranges=[[0,47],[47,94],[94,141],[141,187]],query_batch=4,binary_rule='>0',
        fixed_original_S0=True,GT_used_for_prediction=False,score_update=False,
        query_filter=None,NMS=False,no_new_training=True,no_TEST=True,
        original_image_references_metadata_sha256=hashlib.sha256(json.dumps(spec['original_image_refs'],
            sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest())
    return spec,pre


class MetadataContract(unittest.TestCase):
    def test_fixed50_operation_resources_and_cpu_cuda_scope_reject_wrong_contract(self):
        for gpu in (True, False):
            spec = contract(gpu)
            before = copy.deepcopy(spec)
            self.assertEqual(entry.validate_operation(spec), gpu)
            self.assertEqual(spec, before)
            for key, value in [('updates_per_arm', 36864), ('fixed_epoch', 49),
                               ('cuda_visible_devices', '' if gpu else '0'), ('allocator_limit_MiB', 8192)]:
                bad = copy.deepcopy(spec); bad[key] = value
                with self.subTest(gpu=gpu,key=key), self.assertRaises(RuntimeError):
                    entry.validate_operation(bad)

    def test_declared_source_closure_and_fixed_original_rgb_refs_are_required(self):
        for gpu in (True,False):
            spec,pre=declared_inputs(gpu)
            entry.validate_declared_inputs(spec)
            entry.validate_prelock(spec,pre)
            bad=copy.deepcopy(spec)
            missing=spec['prediction_core_source'] if gpu else spec['metrics_module_sources']['official']
            bad['sources']=[value for value in bad['sources'] if value!=missing]
            with self.assertRaises(RuntimeError): entry.validate_declared_inputs(bad)
            badpre=copy.deepcopy(pre);badpre['original_image_references_metadata_sha256']='f'*64
            with self.assertRaises(RuntimeError): entry.validate_prelock(spec,badpre)

    def test_core_results_remain_pending_and_owned_without_endpoint_substitution(self):
        for gpu in (True,False):
            spec,_=declared_inputs(gpu)
            out=Path(spec['output_directory'])
            owned=lambda name:dict(ref(name),path=str(out/name))
            result=dict(report_ref=owned('DEV_REPORT.json' if gpu else 'COCO_REPORT.json'),
                seed=202610173101,fixed_epoch=50,updates_per_arm=37050,new_training=False,
                AP=not gpu,DEV=True,TEST=False,qualified=False)
            if gpu:
                result.update(predictions={arm:owned(arm+'.json') for arm in ('PLAIN','FULL')},
                    identity_predictions=owned('identity.json'),Omega_support=owned('Omega.json'),
                    images=46,queries=4600,dataset_range=[141,187],shard_index=3,
                    **{key:spec[key] for key in ('training_qualified','pair_commit_ref','checkpoint_refs',
                        'order_ref','dev_context','training_sources','prediction_sources')})
            else:
                result.update(images=187,GT=380,queries=18700,utility_gate_decided=False)
            before=copy.deepcopy(result)
            entry.validate_result(spec,result)
            self.assertEqual(result,before)
            for key,value in [('qualified',True),('updates_per_arm',36864),('report_ref',ref('outside.json'))]:
                bad=copy.deepcopy(result);bad[key]=value
                with self.subTest(gpu=gpu,key=key),self.assertRaises(RuntimeError):
                    entry.validate_result(spec,bad)


if __name__ == '__main__':
    unittest.main()
