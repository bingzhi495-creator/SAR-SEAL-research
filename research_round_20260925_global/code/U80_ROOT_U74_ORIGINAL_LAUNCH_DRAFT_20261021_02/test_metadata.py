"""Synthetic files and process-boundary stubs only; no SSH or real processes."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
import io
import platform
from contextlib import redirect_stdout,redirect_stderr

import launch


def ref(path,token='a'):
    return dict(path=path,bytes=1,sha256=token*64)


def authority_fixture():
    final=ref('/synthetic/FINAL_ACCEPTED.json')
    state=dict(current_all_leaf_sessions_reaped=True,GPU_owner_or_worker_active=False,gpu_training_active=False,
        unchanged_training_cost={'hours':17},**{launch.TRAINING_KEY:dict(FORMAL_prefix=37050,active_phase=None,
            latest_accepted={'FORMAL':final},failures=3)})
    endpoint=dict(schema='U58-U57-DGX-fixed-endpoint-metadata-v1',mode='FORMAL',formal_endpoint=True,
        cursor=dict(global_updates=37050,completed_epochs=50,within_epoch=0),training_qualified=final)
    spec=dict(phase='SYNTHETIC_PHASE',request={'endpoint_refs':{'qualified':final}})
    return state,spec,endpoint,final


def save(path,value):
    raw=value if type(value) is bytes else (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
    path.write_bytes(raw)
    import hashlib
    return dict(path=path.resolve().as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def preflight_fixture(root):
    entry=save(root/'entry.py',b"def validate_envelope(s):\n return {r['path']:r for r in s['sources']}\ndef close_source_union(s,closed):\n closed(s['request']['worker_source'])\n return validate_envelope(s)\n")
    owner=save(root/'owner.py',b'# SYNTHETIC ONLY NOT AN OWNER\n')
    worker=save(root/'worker.py',b'# SYNTHETIC ONLY NOT A WORKER\n')
    helper=save(root/'acceptance.py',b"def endpoint_join(e,d,r):\n assert e['training_qualified']==r['endpoint_refs']['qualified']\n assert d['qualified']['mode']=='FORMAL'\n")
    refs={key:save(root/('ENDPOINT_'+key+'.json'),{'mode':'FORMAL','synthetic':key}) for key in ('qualified','spec','pair_commit','order')}
    cps={arm:save(root/(arm+'.pth'),b'SYNTHETIC_NOT_TORCH') for arm in ('PLAIN','FULL')}
    spec=dict(schema=launch.PINS['U74_GPU']['schema'],operation=launch.PINS['U74_GPU']['operation'],phase='SYNTHETIC_PHASE',
        entry_source=entry,owner_source=owner,request=dict(worker_source=worker,endpoint_refs=refs,checkpoint_refs=cps,
            output_directory=(root/'WORKER').as_posix()),output_directory=(root/'ENTRY').as_posix(),owner_output_directory=(root/'OWNER').as_posix())
    spec['sources']=sorted([entry,owner,worker]+list(refs.values())+list(cps.values()),key=lambda r:r['path'])
    spec_ref=save(root/'SPEC.json',spec)
    endpoint=dict(training_qualified=refs['qualified'])
    pins=deepcopy(launch.PINS);pins['U74_GPU']['entry']=(entry['bytes'],entry['sha256']);pins['U74_GPU']['owner']=(owner['bytes'],owner['sha256'])
    payload=dict(mode='U74_GPU',spec_ref=spec_ref,expected_endpoint=endpoint,acceptance_source_ref=helper)
    return payload,spec,pins,(helper['bytes'],helper['sha256'])


class SyntheticTransport:
    """No subprocess: records order and writes invented original binary streams."""
    def __init__(self,spec,authority_path,code=0,failure=None):
        self.spec=spec;self.authority_path=authority_path;self.code=code;self.failure=failure;self.events=[]
    def preflight(self,payload,directory):
        self.events.append('preflight')
        result=dict(schema='Root-U80-remote-preflight-v1',status='PREFLIGHT_COMPLETE_NO_OWNER_LAUNCHED',
            mode=payload['mode'],spec_ref=payload['spec_ref'],phase=self.spec['phase'],
            expected_endpoint=payload['expected_endpoint'],source_refs=self.spec['sources'],
            output_directories=[self.spec['output_directory'],self.spec['owner_output_directory'],self.spec['request']['output_directory']],
            no_owner_launched=True,resource_observations={'SYNTHETIC':True})
        return dict(SSH_returncode=0,result=result,command=['SYNTHETIC_PREFLIGHT_NOT_EXECUTED'],
            stdout_ref=save(directory/'PREFLIGHT.stdout',json.dumps(result).encode()),stderr_ref=save(directory/'PREFLIGHT.stderr',b''))
    def popen(self,remote_command,stdout,stderr):
        state=json.loads(self.authority_path.read_bytes())
        assert state['current_all_leaf_sessions_reaped'] is False
        assert state[launch.PROFILE_KEY]['active_attempt']['spec_ref']['path']==remote_command[-3]
        assert (Path(state[launch.PROFILE_KEY]['active_attempt']['persistence_directory'])/'ATTEMPT.json').is_file()
        self.events.append('popen')
        if self.failure=='popen':raise OSError('SYNTHETIC_POPEN_FAILURE')
        stdout.write(b'\x00SYNTHETIC_RAW_STDOUT\xff');stderr.write(b'')
        transport=self
        class Child:
            pid=4321
            def wait(self):
                transport.events.append('wait')
                if transport.failure=='wait':raise OSError('SYNTHETIC_WAIT_UNKNOWN')
                return transport.code
        return Child()


def launch_fixture(root,mode='U71_CPU',code=0,failure=None):
    """Public synthetic builder for U81 integration; never an actual launch input."""
    root=Path(root).resolve();state,_,endpoint,final=authority_fixture();pins=launch.PINS[mode]
    spec=dict(schema=pins['schema'],operation=pins['operation'],phase='SYNTHETIC_'+mode,
        entry_source=dict(path='/synthetic/entry.py',bytes=pins['entry'][0],sha256=pins['entry'][1]),
        owner_source=dict(path='/synthetic/owner.py',bytes=pins['owner'][0],sha256=pins['owner'][1]),
        request=dict(output_directory='/synthetic/WORKER'),output_directory='/synthetic/ENTRY',owner_output_directory='/synthetic/OWNER')
    if mode=='U74_GPU':spec['request']['endpoint_refs']={'qualified':final}
    else:
        spec['external_shard_admission_refs']=[ref('/synthetic/proof'+str(i)) for i in range(4)]
        spec['request']['shard_refs']=[ref('/synthetic/prediction'+str(i)) for i in range(4)]
        state[launch.PREDICTION_KEY]=dict(accepted_count=4,AP=False,accepted_shards=[dict(shard_index=i,
            proof_ref=spec['external_shard_admission_refs'][i],prediction_ref=spec['request']['shard_refs'][i]) for i in range(4)])
    spec['sources']=sorted([spec['entry_source'],spec['owner_source']],key=lambda r:r['path'])
    copied=save(root/'SAVED_SPEC.json',spec);remote=dict(copied,path='/synthetic/SPEC.json')
    authority_path=root/'SYNTHETIC_AUTHORITY.json';authority_ref=save(authority_path,state)
    config=dict(schema='Root-U80-private-SSH-transport-v1',ssh_executable=sys.executable,ssh_arguments=[],
        destination='synthetic.invalid',remote_python='/synthetic/bin/python',authority_path=authority_path.as_posix())
    request=dict(schema=launch.SCHEMA,mode=mode,spec_ref=remote,spec_copy_ref=copied,
        transport_config_ref=save(root/'PRIVATE_CONFIG.json',config),authority_ref=authority_ref,expected_endpoint=endpoint,
        expected_final_acceptance_ref=final,acceptance_source_ref=dict(path='/synthetic/U75.py',bytes=launch.U75_PIN[0],sha256=launch.U75_PIN[1]),
        persistence_directory=(root/'ATTEMPT').as_posix())
    return request,SyntheticTransport(spec,authority_path,code,failure)


class Launch(unittest.TestCase):
    def test_byte_preflight_sets_actual_cpu01_and_nice10_or_lower_priority(self):
        for initial_nice in (0,12):
            with self.subTest(initial_nice=initial_nice):
                affinity={0,1,2,3};nice=[initial_nice];actions=[]
                def set_affinity(pid,cpus):actions.append('set_affinity');affinity.clear();affinity.update(cpus)
                def set_priority(which,pid,value):actions.append('set_priority');nice[0]=value
                with patch.object(platform,'system',return_value='Linux'),patch.object(platform,'machine',return_value='aarch64'),\
                    patch.object(platform,'python_version',return_value='3.12.14'),\
                    patch.object(launch.os,'sched_getaffinity',side_effect=lambda pid:set(affinity),create=True),\
                    patch.object(launch.os,'sched_setaffinity',side_effect=set_affinity,create=True),\
                    patch.object(launch.os,'getpriority',side_effect=lambda which,pid:nice[0],create=True),\
                    patch.object(launch.os,'setpriority',side_effect=set_priority,create=True),\
                    patch.object(launch.os,'PRIO_PROCESS',0,create=True),\
                    patch.object(Path,'read_text',return_value='MemAvailable: 32000000 kB\n'):
                    result=launch.linux_resources()
                self.assertEqual(affinity,{0,1});self.assertGreaterEqual(nice[0],10)
                self.assertEqual(result['CPU_affinity'],[0,1]);self.assertEqual(result['nice'],max(10,initial_nice))
                self.assertIn('set_affinity',actions)
                if initial_nice==12:self.assertNotIn('set_priority',actions)

    def test_unknown_original_wait_reaps_only_owned_local_handle_and_preserves_primary(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory).resolve();request,transport=launch_fixture(root,'U74_GPU')
            original_popen=transport.popen;actions=[]
            class Child:
                pid=4321
                killed=False
                done=False
                def poll(self):actions.append('poll');return -9 if self.done else None
                def wait(self,timeout=None):
                    actions.append(('wait',timeout))
                    if timeout is None:raise OSError('SYNTHETIC_ORIGINAL_WAIT_PRIMARY')
                    if not self.killed:raise TimeoutError('SYNTHETIC_TERM_TIMEOUT')
                    self.done=True;return -9
                def terminate(self):actions.append('terminate')
                def kill(self):actions.append('kill');self.killed=True
            child=Child()
            def popen(command,stdout,stderr):original_popen(command,stdout,stderr);return child
            transport.popen=popen
            with self.assertRaisesRegex(OSError,'SYNTHETIC_ORIGINAL_WAIT_PRIMARY'):
                launch.launch_saved_spec(request,transport=transport)
            report=json.loads((Path(request['persistence_directory'])/'REPORT.json').read_bytes())
            self.assertIn('terminate',actions);self.assertIn('kill',actions)
            self.assertTrue(child.done)
            self.assertTrue(report['local_SSH_cleanup']['wait_reaped'])
            self.assertEqual(report['local_SSH_cleanup']['returncode'],-9)
            self.assertTrue(any(row['stage']=='terminate_wait' and row['type']=='TimeoutError'
                for row in report['cleanup_errors']))
            self.assertIsNone(report['SSH_returncode']);self.assertIsNone(report['owner_process_exit_code'])
            self.assertFalse(report['local_SSH_wait_returned'])
            self.assertIn('SYNTHETIC_ORIGINAL_WAIT_PRIMARY',report['error']['traceback'])
            self.assertFalse(json.loads(transport.authority_path.read_bytes())['current_all_leaf_sessions_reaped'])

    def test_original_native_windows_saved_reference_is_preserved_without_relocation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory).resolve();request,transport=launch_fixture(root)
            original=deepcopy(request['spec_copy_ref'])
            original['path']=str(Path(original['path']))
            request['spec_copy_ref']=original
            result=launch.launch_cpu_saved_spec(request,transport=transport)
            self.assertEqual(result['request']['spec_copy_ref'],original)
            self.assertEqual(result['request']['spec_ref'],request['spec_ref'])
            self.assertEqual(result['SSH_returncode'],0)
            self.assertEqual(transport.events,['preflight','popen','wait'])

    def test_nonzero_popen_failure_and_unknown_wait_keep_active_without_owner_exit_claim(self):
        for scenario in ('ssh255','popen','wait'):
            with self.subTest(scenario=scenario),tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
                root=Path(directory).resolve();request,transport=launch_fixture(root,'U74_GPU',
                    code=255 if scenario=='ssh255' else 0,failure=None if scenario=='ssh255' else scenario)
                if scenario=='ssh255':
                    result=launch.launch_saved_spec(request,transport=transport)
                    self.assertEqual(result['SSH_returncode'],255);self.assertTrue(result['local_SSH_wait_returned'])
                else:
                    with self.assertRaisesRegex(OSError,'SYNTHETIC_'):launch.launch_saved_spec(request,transport=transport)
                    result=json.loads((Path(request['persistence_directory'])/'REPORT.json').read_bytes())
                    self.assertIn('traceback',result['error']);self.assertIsNone(result['SSH_returncode'])
                    self.assertFalse(result['local_SSH_wait_returned'])
                self.assertIsNone(result['owner_process_exit_code'])
                state=json.loads(Path(request['authority_ref']['path']).read_bytes())
                self.assertFalse(state['current_all_leaf_sessions_reaped'])
                self.assertIsNotNone(state[launch.PROFILE_KEY]['active_attempt'])
                self.assertNotIn('root_observed_tool_exit_code',result)

    def test_cli_is_gpu_only_and_returns_original_wait_code_not_planned_zero(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory).resolve();request,_=launch_fixture(root,'U74_GPU')
            request_ref=save(root/'LAUNCH_REQUEST.json',request)
            argv=['--request-file',request_ref['path'],'--request-sha256',request_ref['sha256']]
            with patch.object(launch,'launch_saved_spec',return_value={'SSH_returncode':255,'synthetic':True}) as run:
                output=io.StringIO()
                with redirect_stdout(output):code=launch.main(argv)
                self.assertEqual(code,255);self.assertEqual(json.loads(output.getvalue())['SSH_returncode'],255)
                run.assert_called_once_with(request)
                run.reset_mock();request['mode']='U71_CPU';cpu_ref=save(root/'CPU_REQUEST.json',request)
                errors=io.StringIO()
                with redirect_stderr(errors):code=launch.main(['--request-file',cpu_ref['path'],'--request-sha256',cpu_ref['sha256']])
                self.assertNotEqual(code,0);run.assert_not_called()
                with redirect_stderr(errors):code=launch.main(argv[:-1]+['0'*64])
                self.assertNotEqual(code,0);run.assert_not_called()

    def test_cpu_shared_api_requires_original_four_committed_refs_and_keeps_gpu_flags_false(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory).resolve();request,transport=launch_fixture(root)
            authority_path=Path(request['authority_ref']['path']);prior=json.loads(authority_path.read_bytes())
            bad=deepcopy(prior);bad[launch.PREDICTION_KEY]['accepted_shards'][2]['proof_ref']=ref('/OTHER/PROOF.json')
            bad_ref=save(authority_path,bad);bad_request=deepcopy(request);bad_request['authority_ref']=bad_ref
            with self.assertRaises(ValueError):launch.launch_cpu_saved_spec(bad_request,transport=transport)
            self.assertEqual(transport.events,[])
            save(authority_path,prior)
            result=launch.launch_cpu_saved_spec(request,transport=transport)
            self.assertEqual(result['SSH_returncode'],0);self.assertIsNone(result['AP'])
            state=json.loads(authority_path.read_bytes())
            self.assertFalse(state['current_all_leaf_sessions_reaped'])
            self.assertFalse(state['GPU_owner_or_worker_active']);self.assertFalse(state['gpu_training_active'])
            self.assertEqual(state[launch.PREDICTION_KEY],prior[launch.PREDICTION_KEY])
            self.assertEqual(state[launch.TRAINING_KEY],prior[launch.TRAINING_KEY])

    def test_original_launch_reserves_first_waits_once_preserves_raw_streams_and_never_self_accepts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory).resolve();request,transport=launch_fixture(root,'U74_GPU')
            prior=json.loads(Path(request['authority_ref']['path']).read_bytes())
            result=launch.launch_saved_spec(request,transport=transport)
            self.assertEqual(transport.events,['preflight','popen','wait'])
            self.assertEqual(result['SSH_returncode'],0)
            self.assertIsNone(result['owner_process_exit_code'])
            self.assertNotIn('root_observed_tool_exit_code',result)
            self.assertNotIn('root_observed_tool_chunk_id',result)
            self.assertEqual(Path(result['stdout_ref']['path']).read_bytes(),b'\x00SYNTHETIC_RAW_STDOUT\xff')
            self.assertEqual(result['stderr_ref']['bytes'],0)
            state=json.loads(Path(request['authority_ref']['path']).read_bytes())
            self.assertFalse(state['current_all_leaf_sessions_reaped']);self.assertTrue(state['GPU_owner_or_worker_active'])
            self.assertFalse(state['gpu_training_active'])
            self.assertEqual(state[launch.TRAINING_KEY],prior[launch.TRAINING_KEY])
            self.assertEqual(state['unchanged_training_cost'],prior['unchanged_training_cost'])
            with self.assertRaises((ValueError,FileExistsError)):launch.launch_saved_spec(request,transport=transport)
            self.assertEqual(transport.events,['preflight','popen','wait'])

    def test_remote_preflight_closes_fixed_definitions_exact_union_endpoint_cp_and_fresh_outputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory).resolve();payload,spec,pins,helper_pin=preflight_fixture(root)
            with patch.object(launch,'PINS',pins),patch.object(launch,'U75_PIN',helper_pin),patch.object(launch,'linux_resources',return_value={'synthetic':True}):
                result=launch.remote_preflight(payload)
                self.assertEqual(result['source_refs'],spec['sources'])
                self.assertTrue(result['no_owner_launched'])
                self.assertEqual(result['spec_ref'],payload['spec_ref'])
                Path(spec['request']['checkpoint_refs']['FULL']['path']).write_bytes(b'CHANGED')
                with self.assertRaises(ValueError):launch.remote_preflight(payload)
                save(Path(spec['request']['checkpoint_refs']['FULL']['path']),b'SYNTHETIC_NOT_TORCH')
                Path(spec['output_directory']).mkdir()
                with self.assertRaises(ValueError):launch.remote_preflight(payload)

    def test_authority_requires_genuine_idle_current_final_and_no_unresolved_attempt(self):
        state,spec,endpoint,final=authority_fixture()
        self.assertIsNone(launch.check_authority(state,spec,'U74_GPU',endpoint,final))
        for change in ('prefix','boolean_prefix','active','busy','other_final','unresolved'):
            bad=deepcopy(state)
            if change=='prefix':bad[launch.TRAINING_KEY]['FORMAL_prefix']=7424
            elif change=='boolean_prefix':bad[launch.TRAINING_KEY]['FORMAL_prefix']=True
            elif change=='active':bad[launch.TRAINING_KEY]['active_phase']='LIVE'
            elif change=='busy':bad['current_all_leaf_sessions_reaped']=False
            elif change=='other_final':bad[launch.TRAINING_KEY]['latest_accepted']['FORMAL']=ref('/other/FINAL.json')
            else:bad[launch.PROFILE_KEY]=dict(active_attempt={'SYNTHETIC':True},attempts=[])
            with self.subTest(change=change),self.assertRaises(ValueError):
                launch.check_authority(bad,spec,'U74_GPU',endpoint,final)


if __name__=='__main__':unittest.main()
