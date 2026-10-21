"""U81-only synthetic saved artifacts and launch-boundary stubs; no SSH."""
from copy import deepcopy
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent


def load_file(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def ref(path,raw):
    return dict(path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True)+'\n').encode()


def replace_saved(request,raw,role,value):
    """Invent a newly hashed artifact; never modify an actual research file."""
    prior=request['persisted_refs'][role];blob=encoded(value)
    request['persisted_refs'][role]=ref(prior['path'],blob);raw[prior['path']]=blob
    if role!='record':
        record_ref=request['persisted_refs']['record'];record=json.loads(raw[record_ref['path']])
        record[role+'_ref']=request['persisted_refs'][role]
        replace_saved(request,raw,'record',record)
    if role=='spec':
        request['launch_request']['spec_copy_ref']=deepcopy(request['persisted_refs']['spec'])
        request['launch_request']['spec_ref'].update({key:request['persisted_refs']['spec'][key] for key in ('bytes','sha256')})
    request['binding_terminal']['persisted_refs']=deepcopy(request['persisted_refs'])


class RecordingTransport:
    """Initial isolated call-boundary fixture, not U80/SSH process evidence."""
    def __init__(self):self.calls=[]
    def launch_saved_spec(self,request):
        self.calls.append(deepcopy(request))
        return dict(status='SYNTHETIC_SSH_WAIT_RETURNED_PENDING_ROOT',SSH_returncode=0,qualified=False,AP=False)


def synthetic_metadata_globals(module,binding_request):
    original=module.definitions
    def definitions(reference,raw):
        value=original(reference,raw)
        if reference==binding_request['entry_source']:
            original_load=value.load_closed
            def load_closed(close,source,alias):
                worker=original_load(close,source,alias)
                for role in ('population_contract','mask_api'):
                    worker.SOURCE_PINS[role]={key:binding_request['cpu_request']['module_sources'][role][key]
                                             for key in ('bytes','sha256')}
                return worker
            value.load_closed=load_closed
        return value
    return patch.object(module,'definitions',side_effect=definitions)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(dir=HERE);cls.addClassCleanup(cls.temporary.cleanup)
        directory=Path(cls.temporary.name)
        u78=HERE.parent/'U78_ROOT_CPU_PRELAUNCH_BINDING_DRAFT_20261021_01'
        helper=load_file(u78/'test_metadata.py','_U78_U81_FIXTURE')
        binding=helper.load();original,raw=helper.fixture()
        original['persistence_directory']=str(directory/'ORIGINAL_U78_OUTPUT')
        with helper.synthetic_definition_pins(binding,original):
            saved=binding.write_binding(original,read_raw=lambda value:raw[value['path']])
        code=(u78/'binding.py').read_bytes();source_ref=ref('/synthetic/U78_binding.py',code);raw[source_ref['path']]=code
        original_raw=encoded(original);original_ref=ref(str(directory/'ORIGINAL_U78_REQUEST.json'),original_raw)
        raw[original_ref['path']]=original_raw
        for value in saved['persisted_refs'].values():raw[value['path']]=Path(value['path']).read_bytes()
        # Only this isolated fixture uses a tiny launch boundary. A genuine
        # fixed-U80 integration is added after the shared source freezes.
        launcher=b'def launch_cpu_saved_spec(request,*,transport=None):\n    return transport.launch_saved_spec(request)\n'
        launcher_ref=ref('/synthetic/U80_fixture.py',launcher);raw[launcher_ref['path']]=launcher
        remote_spec=dict(saved['persisted_refs']['spec'],path='/synthetic/DGX_SAVED_SPEC.json')
        launch_request=dict(schema='Root-U80-original-saved-spec-launch-request-v1',mode='U71_CPU',
            spec_ref=remote_spec,spec_copy_ref=saved['persisted_refs']['spec'],
            transport_config_ref=ref('/synthetic/PRIVATE_CONFIG.json',b'NEVER_READ_BY_ADAPTER'),
            authority_ref=ref('/synthetic/ACTUAL_ROOT_AUTHORITY.json',b'NEVER_READ_BY_ADAPTER'),
            expected_endpoint=original['expected_endpoint'],
            expected_final_acceptance_ref=original['expected_endpoint']['training_qualified'],
            acceptance_source_ref=original['acceptance_source_ref'],persistence_directory=str(directory/'FUTURE_U80_ATTEMPT'))
        terminal=dict(root_observed_tool_exit_code=0,root_observed_tool_chunk_id='SYNTHETIC_U78_TOOL_CHUNK',
            binding_source_ref=deepcopy(source_ref),binding_request_ref=deepcopy(original_ref),persisted_refs=deepcopy(saved['persisted_refs']))
        cls.base=dict(schema='Root-U81-U71-saved-binding-launch-request-v1',binding_source_ref=source_ref,
            binding_request_ref=original_ref,persisted_refs=saved['persisted_refs'],binding_terminal=terminal,
            launcher_source_ref=launcher_ref,launch_request=launch_request)
        cls.raw=raw;cls.binding_request=original

    def fixture(self):return deepcopy(self.base),dict(self.raw)

    def load(self):return load_file(HERE/'launch.py','_U81_TEST')

    def test_saved_spec_bytes_are_consumed_and_CPU_launcher_is_really_called(self):
        module=self.load();request,raw=self.fixture();transport=RecordingTransport()
        with synthetic_metadata_globals(module,self.binding_request),patch.object(module,'LAUNCHER_PIN',
                {key:request['launcher_source_ref'][key] for key in ('bytes','sha256')}):
            result=module.launch_saved_binding(request,read_raw=lambda value:raw[value['path']],transport=transport)
        self.assertEqual(transport.calls,[request['launch_request']])
        self.assertEqual(result['SSH_returncode'],0)
        self.assertFalse(result['qualified']);self.assertFalse(result['AP'])

    def test_prepared_json_and_false_or_mismatched_terminal_cannot_launch(self):
        module=self.load()
        for case in ('boolean_zero','nonzero','missing_chunk','other_saved_record','owner_self_report'):
            request,raw=self.fixture();transport=RecordingTransport()
            terminal=request['binding_terminal']
            if case=='boolean_zero':terminal['root_observed_tool_exit_code']=False
            elif case=='nonzero':terminal['root_observed_tool_exit_code']=1
            elif case=='missing_chunk':terminal['root_observed_tool_chunk_id']=''
            elif case=='other_saved_record':terminal['persisted_refs']['record']['path']='/synthetic/OTHER_PREPARATION.json'
            else:request['binding_terminal']={'controller_returning':True,'qualified':True}
            with self.subTest(case=case),synthetic_metadata_globals(module,self.binding_request),patch.object(module,'LAUNCHER_PIN',
                    {key:request['launcher_source_ref'][key] for key in ('bytes','sha256')}),self.assertRaises(ValueError):
                module.launch_saved_binding(request,read_raw=lambda value:raw[value['path']],transport=transport)
            self.assertEqual(transport.calls,[])

    def test_final_endpoint_ordinal_proofs_and_preparation_flags_are_not_replaceable(self):
        module=self.load()
        for case in ('other_endpoint','midpoint','other_final_acceptance','proof_order','not_admitted'):
            request,raw=self.fixture();transport=RecordingTransport()
            if case=='other_endpoint':request['launch_request']['expected_endpoint']['run_id']='OTHER'
            elif case=='midpoint':request['launch_request']['expected_endpoint']['cursor']['global_updates']=7168
            elif case=='other_final_acceptance':request['launch_request']['expected_final_acceptance_ref']=dict(
                request['launch_request']['expected_final_acceptance_ref'],path='/synthetic/OTHER_FINAL.json')
            elif case=='proof_order':
                value=json.loads(raw[request['persisted_refs']['admission']['path']])
                value['external_shard_admission_refs'].reverse();replace_saved(request,raw,'admission',value)
            else:
                value=json.loads(raw[request['persisted_refs']['record']['path']]);value['U76_admission_called']=False
                replace_saved(request,raw,'record',value)
            with self.subTest(case=case):
                with synthetic_metadata_globals(module,self.binding_request),patch.object(module,'LAUNCHER_PIN',
                        {key:request['launcher_source_ref'][key] for key in ('bytes','sha256')}),self.assertRaises(ValueError):
                    module.launch_saved_binding(request,read_raw=lambda value:raw[value['path']],transport=transport)
                self.assertFalse(transport.calls)

    def test_CPU_CLI_checks_original_request_SHA_and_returns_actual_SSH_code(self):
        module=self.load();request,raw=self.fixture()
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            path=Path(temporary)/'SYNTHETIC_U81_REQUEST.json';blob=encoded(request);path.write_bytes(blob)
            with patch.object(module,'launch_saved_binding',side_effect=AssertionError('No launch before request SHA')):
                with self.assertRaisesRegex(ValueError,'SHA'):
                    module.main(['--request-file',str(path),'--request-sha256','0'*64])
            pending=dict(status='SYNTHETIC_WAIT_ONLY',SSH_returncode=255,owner_process_exit_code=None,qualified=False,AP=False)
            with patch.object(module,'launch_saved_binding',return_value=pending) as launch,contextlib.redirect_stdout(io.StringIO()) as stdout:
                code=module.main(['--request-file',str(path),'--request-sha256',hashlib.sha256(blob).hexdigest()])
            self.assertEqual(code,255);self.assertEqual(json.loads(stdout.getvalue()),pending)
            self.assertEqual(launch.call_args[0],(request,))

    def test_genuine_fixed_shared_CPU_launch_uses_original_saved_files_and_wait(self):
        module=self.load();request,raw=self.fixture()
        source=HERE.parent/'U80_ROOT_U74_ORIGINAL_LAUNCH_DRAFT_20261021_02'
        shared=load_file(source/'launch.py','_U80_U81_GENUINE')
        with patch.dict(sys.modules,{'launch':shared}):
            helper=load_file(source/'test_metadata.py','_U80_U81_PUBLIC_FIXTURE')
        blob=(source/'launch.py').read_bytes();source_ref=ref(str(source/'launch.py'),blob)
        request['launcher_source_ref']=source_ref;raw[source_ref['path']]=blob
        for scenario in ('prefix7168','missing_shard','valid'):
            with self.subTest(scenario=scenario),tempfile.TemporaryDirectory(dir=HERE) as temporary:
                directory=Path(temporary)
                launch_request,unused=helper.launch_fixture(directory,mode='U71_CPU')
                launch_request.update(spec_ref=deepcopy(request['launch_request']['spec_ref']),
                    spec_copy_ref=deepcopy(request['persisted_refs']['spec']),
                    expected_endpoint=deepcopy(request['launch_request']['expected_endpoint']),
                    expected_final_acceptance_ref=deepcopy(request['launch_request']['expected_final_acceptance_ref']))
                authority=Path(launch_request['authority_ref']['path']);state=json.loads(authority.read_bytes())
                spec=json.loads(raw[request['persisted_refs']['spec']['path']])
                state[shared.TRAINING_KEY]['latest_accepted']['FORMAL']=launch_request['expected_final_acceptance_ref']
                state[shared.PREDICTION_KEY]['accepted_shards']=[dict(shard_index=index,
                    proof_ref=spec['external_shard_admission_refs'][index],prediction_ref=spec['request']['shard_refs'][index]) for index in range(4)]
                if scenario=='prefix7168':state[shared.TRAINING_KEY]['FORMAL_prefix']=7168
                elif scenario=='missing_shard':
                    state[shared.PREDICTION_KEY]['accepted_shards'].pop();state[shared.PREDICTION_KEY]['accepted_count']=3
                launch_request['authority_ref']=helper.save(authority,state)
                request['launch_request']=launch_request
                transport=helper.SyntheticTransport(spec,authority)
                with synthetic_metadata_globals(module,self.binding_request):
                    if scenario!='valid':
                        with self.assertRaisesRegex(ValueError,'FORMAL37050' if scenario=='prefix7168' else 'Four actual'):
                            module.launch_saved_binding(request,read_raw=lambda value:raw[value['path']],transport=transport)
                        self.assertFalse(transport.events)
                    else:
                        result=module.launch_saved_binding(request,read_raw=lambda value:raw[value['path']],transport=transport)
                        self.assertEqual(transport.events,['preflight','popen','wait'])
                        self.assertEqual(result['SSH_returncode'],0);self.assertIsNone(result['owner_process_exit_code'])
                        self.assertIsNone(result['AP'])
                        self.assertEqual(Path(result['stdout_ref']['path']).read_bytes(),b'\x00SYNTHETIC_RAW_STDOUT\xff')
                        self.assertFalse(json.loads(authority.read_bytes())['current_all_leaf_sessions_reaped'])


if __name__=='__main__':unittest.main()
