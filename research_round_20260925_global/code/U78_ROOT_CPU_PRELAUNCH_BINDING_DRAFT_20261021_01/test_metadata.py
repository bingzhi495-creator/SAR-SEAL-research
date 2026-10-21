"""U78-only stdlib synthetic metadata; never real research or process execution."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent


def load_file(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def load():
    return load_file(HERE/'binding.py','_U78_TEST')


def ref(path,raw):
    return dict(path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


@lru_cache(maxsize=1)
def fixture_base():
    """Frozen real U76/U75 metadata functions; all observations remain invented."""
    root=HERE.parent
    u76=root/'U76_ROOT_FIXED50_POPULATION_ADMISSION_DRAFT_20261021_02'
    helper=load_file(u76/'test_metadata.py','_U76_U78_FIXTURE')
    args=helper.genuine_validator_fixture()
    raw=dict(args['raw_inputs']);codes={}
    for role,relative in (
        ('admission_source_ref','U76_ROOT_FIXED50_POPULATION_ADMISSION_DRAFT_20261021_02/admission.py'),
        ('entry_source','U71_DGX_FIXED50_CPU_METRICS_SUPERVISION_DRAFT_20261021_01/entry.py'),
        ('owner_source','U71_DGX_FIXED50_CPU_METRICS_SUPERVISION_DRAFT_20261021_01/linux_owner.py'),
        ('worker_source','U70_DGX_FIXED50_POPULATION_METRICS_WORKER_DRAFT_20261021_01/worker.py')):
        blob=(root/relative).read_bytes();value=ref('/synthetic/root_code/'+role+'.py',blob)
        codes[role]=value;raw[value['path']]=blob
    cpu=deepcopy(args['cpu_request'])
    worker=load_file(root/'U70_DGX_FIXED50_POPULATION_METRICS_WORKER_DRAFT_20261021_01/worker.py','_U70_U78_FIXTURE')
    cpu['worker_source']=codes['worker_source']
    cpu['core_source']=dict(path='/synthetic/CPU_CORE.py',**worker.CORE_PIN)
    for role,pin in worker.SOURCE_PINS.items():
        if role not in ('population_contract','mask_api'):
            cpu['module_sources'][role].update(pin)
    request=dict(schema='Root-U78-CPU-prelaunch-binding-request-v1',phase='SYNTHETIC_CPU_ONLY',
        admission_source_ref=codes['admission_source_ref'],acceptance_source_ref=args['acceptance_source_ref'],
        entry_source=codes['entry_source'],owner_source=codes['owner_source'],
        proof_refs=args['proof_refs'],prediction_refs=args['prediction_refs'],expected_endpoint=args['endpoint'],
        cpu_request=cpu,output_directory='/synthetic/FUTURE_CPU_ENTRY',
        owner_output_directory='/synthetic/FUTURE_CPU_OWNER',persistence_directory='D:/synthetic/FUTURE_U78_BINDING')
    return request,raw


def fixture():
    request,raw=fixture_base()
    return deepcopy(request),dict(raw)


def synthetic_definition_pins(module,request):
    """Patch only ephemeral U70 metadata globals for invented U75 components.

    Frozen source bytes/pins, U76/U75 validators and U71 functions stay intact.
    This accommodates harmless synthetic pop/mask refs, not real source proof.
    """
    original=module.definitions
    def definitions(reference,raw):
        result=original(reference,raw)
        if reference==request['entry_source']:
            original_load=result.load_closed
            def load_closed(close,value,alias):
                worker=original_load(close,value,alias)
                for role in ('population_contract','mask_api'):
                    worker.SOURCE_PINS[role]={k:request['cpu_request']['module_sources'][role][k]
                                              for k in ('bytes','sha256')}
                return worker
            result.load_closed=load_closed
        return result
    return patch.object(module,'definitions',side_effect=definitions)


class Tests(unittest.TestCase):
    def test_true_admission_and_U71_metadata_produce_the_same_bound_request(self):
        module=load();request,raw=fixture();visits=[]
        def read_raw(value):
            visits.append(value['path']);return raw[value['path']]
        with synthetic_definition_pins(module,request):
            result=module.prepare_binding(request,read_raw=read_raw)
        spec=result['spec'];admission=result['admission']
        self.assertEqual(len(spec),11)
        self.assertEqual(spec['request'],request['cpu_request'])
        self.assertEqual(spec['external_shard_admission_refs'],admission['external_shard_admission_refs'])
        self.assertEqual(spec['external_shard_admission_refs'],request['proof_refs'])
        self.assertEqual(admission['shard_refs'],request['prediction_refs'])
        self.assertEqual((admission['images'],admission['queries']),(187,18700))
        self.assertEqual(len(visits),len(set(visits)))
        self.assertEqual(set(visits),set(raw))
        self.assertFalse(result['record']['qualified']);self.assertFalse(result['record']['AP'])
        self.assertFalse(result['record']['CPU_launched'])
        self.assertFalse(result['record']['persistence_performed'])
        self.assertEqual(result['record']['cpu_request_metadata_sha256'],admission['cpu_request_metadata_sha256'])

    def test_exclusive_persistence_binds_exact_original_spec_and_admission_bytes(self):
        module=load();request,raw=fixture()
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            destination=Path(temporary)/'NEW_BINDING';request['persistence_directory']=str(destination)
            with synthetic_definition_pins(module,request),patch.object(module.os,'fsync',wraps=os.fsync) as sync:
                result=module.write_binding(request,read_raw=lambda value:raw[value['path']])
            for role,filename in (('spec','SPEC.json'),('admission','ADMISSION.json'),('record','PREPARATION.json')):
                data=(destination/filename).read_bytes()
                self.assertEqual(json.loads(data),result[role])
                self.assertEqual(result['persisted_refs'][role],ref(str(destination/filename),data))
            self.assertGreaterEqual(sync.call_count,3)
            self.assertEqual(result['record']['spec_ref'],result['persisted_refs']['spec'])
            self.assertEqual(result['record']['admission_ref'],result['persisted_refs']['admission'])
            self.assertTrue(result['record']['persistence_performed']);self.assertFalse(result['record']['CPU_launched'])
            original=(destination/'SPEC.json').read_bytes()
            with self.assertRaises(FileExistsError):
                module.write_binding(request,read_raw=lambda value:self.fail('existing attempt must fail before reads'))
            self.assertEqual((destination/'SPEC.json').read_bytes(),original)

    def test_persistence_and_execution_outputs_cannot_enclose_inputs_or_each_other(self):
        module=load()
        for case in ('persistence_inputs','persistence_execution','execution_control_source'):
            request,raw=fixture()
            if case=='persistence_inputs':request['persistence_directory']='/synthetic'
            elif case=='persistence_execution':request['persistence_directory']=request['output_directory']
            else:request['output_directory']=request['admission_source_ref']['path']
            with self.subTest(case=case),synthetic_definition_pins(module,request),self.assertRaisesRegex(ValueError,'output|persistence'):
                module.prepare_binding(request,read_raw=lambda value:raw[value['path']])

    def test_CLI_request_SHA_precedes_all_ref_reads_and_output_creation(self):
        module=load()
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            source=Path(temporary)/'SYNTHETIC_REQUEST.json'
            request,raw=fixture();request['persistence_directory']=str(Path(temporary)/'BINDING')
            source.write_text(json.dumps(request),encoding='utf-8')
            with patch.object(module,'write_binding',side_effect=AssertionError('No binding before original request SHA')):
                with self.assertRaisesRegex(ValueError,'SHA'):
                    module.main(['--request',str(source),'--request-sha','0'*64])
            self.assertFalse((Path(temporary)/'BINDING').exists())

    def test_fixed_sources_and_actual_admission_failure_never_persist_a_spec(self):
        module=load()
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            for case in ('old_U76','owner_pin','raw_prediction','CPU_mask_path'):
                request,raw=fixture();destination=Path(temporary)/case
                request['persistence_directory']=str(destination)
                if case=='old_U76':request['admission_source_ref']['bytes']=8829
                elif case=='owner_pin':request['owner_source']['sha256']='0'*64
                elif case=='raw_prediction':raw[request['prediction_refs'][0]['path']]+=b' '
                else:request['cpu_request']['module_sources']['mask_api']['path']='/synthetic/COPIED_MASK.py'
                with self.subTest(case=case),synthetic_definition_pins(module,request),self.assertRaises(ValueError):
                    module.write_binding(request,read_raw=lambda value:raw[value['path']])
                self.assertFalse(destination.exists())

    def test_persistence_failure_retains_partial_evidence_and_original_exception(self):
        module=load();request,raw=fixture();failure=OSError('SYNTHETIC_WRITE_FAILURE')
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            destination=Path(temporary)/'FAILED_BINDING';request['persistence_directory']=str(destination)
            original=module.write_json_exclusive
            def write(path,value):
                if path.name=='ADMISSION.json':raise failure
                return original(path,value)
            with synthetic_definition_pins(module,request),patch.object(module,'write_json_exclusive',side_effect=write):
                with self.assertRaises(OSError) as caught:
                    module.write_binding(request,read_raw=lambda value:raw[value['path']])
            self.assertIs(caught.exception,failure)
            self.assertTrue((destination/'SPEC.json').is_file())
            self.assertFalse((destination/'PREPARATION.json').exists())
            audit=json.loads((destination/'FAILED.json').read_text())
            self.assertFalse(audit['CPU_launched']);self.assertFalse(audit['qualified'])
            self.assertIn('SYNTHETIC_WRITE_FAILURE',audit['error']['traceback'])


if __name__=='__main__':unittest.main()
