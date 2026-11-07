"""SYNTHETIC contract dictionaries only; never import tensor libraries."""
from copy import deepcopy
import json
from types import SimpleNamespace as N
import unittest
import worker as w

def synthetic_spec():
    def ref(name,pin=None):
        return dict(path='/synthetic/source/'+name,**(pin or dict(bytes=1,sha256='a'*64)))
    return dict(schema=w.SCHEMA,operation=w.OPERATION,run_id='DiffSAR-R101-CPU-load-v01-SYNTHETIC',
        source_refs=dict(worker=ref('worker.py'),config_adapter=ref('adapter.py',w.ADAPTER_PIN),coverage=ref('coverage.py',w.COVERAGE_PIN),actual_load=ref('actual.py',w.ACTUAL_LOAD_PIN)),
        library_sources={k:ref(k+'.py',w.D2_EXTENSION_PIN if k=='d2_extension' else None) for k in w.LIBRARY_NAMES},
        config_ref=ref('configs/SSDD.yaml',w.CONFIG_PIN),checkpoint_ref=ref('V1.pth',w.CHECKPOINT_PIN),root_lease_ref=ref('LEASE.json'),output_directory='/synthetic/output/new',prelock=deepcopy(w.PRELOCK))

def synthetic_lease(spec):
    return dict(schema=w.LEASE_SCHEMA,status='ROOT_GRANTED',**{k:deepcopy(spec[k]) for k in ('operation','run_id','prelock','source_refs','library_sources','config_ref','checkpoint_ref','output_directory')},boot_id='a'*8+'-'+'b'*4+'-'+'c'*4+'-'+'d'*4+'-'+'e'*12,single_attempt=True,main_scientific_processes_reaped=True,exclusive_project_scientific_leaf=True,whole_device_exclusive=False,main_accepted_prefix=32000,worker_ref=deepcopy(spec['source_refs']['worker']),prior_scientific_groups=[dict(pid=123,SYNTHETIC=True)],main_acceptance_ref=dict(path='/synthetic/main/ROOT_ACCEPTANCE.json',bytes=1,sha256='b'*64))

class ContractTests(unittest.TestCase):
    def test_explicit_load_only_spec_without_any_image_or_training_input(self):
        spec=synthetic_spec()
        self.assertIsNone(w.validate_spec(spec))
        self.assertNotIn('input_root',spec)
        self.assertFalse(spec['prelock']['forward'])

    def test_lease_requires_closed_main_and_identical_scientific_bindings(self):
        spec=synthetic_spec();lease=synthetic_lease(spec)
        self.assertIsNone(w.validate_lease(spec,lease))
        for key,value in [('main_scientific_processes_reaped',False),('single_attempt',1),('whole_device_exclusive',True),('main_accepted_prefix',31999)]:
            wrong=deepcopy(lease);wrong[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):w.validate_lease(spec,wrong)
        wrong=deepcopy(lease);wrong['checkpoint_ref']['sha256']='b'*64
        with self.assertRaises(ValueError):w.validate_lease(spec,wrong)

    def test_raw_duplicate_keys_and_unsupported_spec_routes_cannot_be_hidden(self):
        spec=synthetic_spec();self.assertEqual(w.parse_json(json.dumps(spec)),spec)
        for text in ('{"x":1,"x":2}','{"x":{"a":1,"a":2}}','{"x":NaN}'):
            with self.subTest(text=text),self.assertRaises(ValueError):w.parse_json(text)
        for mutate in ('image','prelock','alias','badpin'):
            s=synthetic_spec()
            if mutate=='image':s['input_root']='/synthetic/GT'
            if mutate=='prelock':s['prelock']['new_training']=0
            if mutate=='alias':s['library_sources']['numpy']=deepcopy(s['library_sources']['torch'])
            if mutate=='badpin':s['source_refs']['coverage']['sha256']='f'*64
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):w.validate_spec(s)

    def test_configuration_changes_only_the_declared_CPU_seed_worker_fields(self):
        model=N(META_ARCHITECTURE='DiffSARShipInst',BACKBONE=N(NAME='build_resnet_scje_fpn_backbone'),RESNETS=N(DEPTH=101,DEFORM_MODULATED=True,DEFORM_ON_PER_STAGE=[True]*4,DEFORM_NUM_GROUPS=1,NUM_GROUPS=1,STRIDE_IN_1X1=False,RES5_DILATION=1,NORM='FrozenBN'),DiffSARShipInst=N(NUM_PROPOSALS=500,NUM_HEADS=6,SAMPLE_STEP=1,NUM_CLASSES=1,DEEP_SUPERVISION=True,USE_FED_LOSS=False),PIXEL_MEAN=[123.675,116.28,103.53],PIXEL_STD=[58.395,57.12,57.375],WEIGHTS=None,DEVICE='cuda')
        cfg=N(MODEL=model,MODEL_EMA=N(ENABLED=False),INPUT=N(FORMAT='RGB'),SOLVER=N(MAX_ITER=18560,STEPS=[14848,17632]),DATALOADER=N(NUM_WORKERS=4),SEED=-1)
        cfg.merge_from_file=lambda p:None;cfg.freeze=lambda:setattr(cfg,'frozen',True)
        modules={'d2_config':N(get_cfg=lambda:cfg),'config':N(add_DiffSARShipInst_config=lambda c:None),'model_ema':N(add_model_ema_configs=lambda c:None)}
        adapter=N(register_inactive_bifpn_schema=lambda c:None,verify_inactive_bifpn_schema=lambda c:None)
        actual=w.configuration(modules,adapter,synthetic_spec())
        self.assertIs(actual,cfg);self.assertEqual(cfg.MODEL.DEVICE,'cpu');self.assertEqual(cfg.SEED,40244023);self.assertEqual(cfg.DATALOADER.NUM_WORKERS,0)
        self.assertEqual((cfg.SOLVER.MAX_ITER,cfg.SOLVER.STEPS),(18560,[14848,17632]));self.assertTrue(cfg.frozen)

    def test_native_modulated_DCN_scalar_spatial_attributes_are_checked(self):
        self.assertEqual(w.spatial_pair(2),(2,2))
        self.assertEqual(w.spatial_pair((1,1)),(1,1))
        for bad in (True,'2',[1,1],(1,2,3),(True,1)):
            with self.subTest(bad=bad),self.assertRaises(ValueError):w.spatial_pair(bad)

    def test_actual_tensor_comparison_helper_cannot_be_replaced_by_a_success_stub(self):
        spec=synthetic_spec();spec['source_refs']['actual_load']['sha256']='e'*64
        with self.assertRaisesRegex(ValueError,'byte pin'):w.validate_spec(spec)

if __name__=='__main__':unittest.main()
