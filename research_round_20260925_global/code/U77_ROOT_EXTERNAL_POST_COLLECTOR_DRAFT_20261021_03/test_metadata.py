"""Only new synthetic files/procfs fixtures; never production main or real /proc."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout,redirect_stderr

import collector


BOOT='11111111-2222-3333-4444-555555555555'


def owner_fixture():
    return dict(boot_id=BOOT,owner_pid=200,
        owner_identity=dict(pid=200,state='S',parent_pid=100,process_group=100,session=100,start_ticks=1000),
        child_identity=dict(pid=201,state='R',parent_pid=200,process_group=201,session=201,start_ticks=1001))


def stat_text(identity):
    fields=[identity['state'],str(identity['parent_pid']),str(identity['process_group']),str(identity['session'])]+['0']*15+[str(identity['start_ticks'])]
    return str(identity['pid'])+' (SYNTHETIC (name)) '+' '.join(fields)+'\n'


def proc_fixture(root):
    proc=root/'synthetic_proc';boot=proc/'sys/kernel/random/boot_id';boot.parent.mkdir(parents=True);boot.write_text(BOOT+'\n')
    other=dict(pid=300,state='S',parent_pid=1,process_group=300,session=300,start_ticks=2000)
    (proc/'300').mkdir();(proc/'300/stat').write_text(stat_text(other))
    return proc


def file_ref(path):
    data=path.read_bytes();return dict(path=path.resolve().as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def save_json(path,value):
    path.write_bytes((json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode());return file_ref(path)


def collection_fixture(root):
    source=Path(collector.__file__).resolve()
    acceptance=source.parent.parent/'U75_ROOT_FIXED50_SHARD_ACCEPTANCE_DRAFT_20261021_01/acceptance.py'
    owner_out,entry_out,worker_out=(root/name for name in ('ORIGINAL_OWNER','ORIGINAL_ENTRY','ORIGINAL_WORKER'))
    for directory in (owner_out,entry_out,worker_out):directory.mkdir()
    # Full prediction is deliberately opaque here: U77 must hash, not evaluate it.
    (worker_out/'U66_SHARD.json').write_bytes(b'SYNTHETIC_OPAQUE_PREDICTION_NOT_A_REAL_RESULT\n')
    prediction=file_ref(worker_out/'U66_SHARD.json')
    body=dict(schema='U73-single-shard-worker-report-v1',status='COMPLETE_PENDING_ROOT_ACCEPTANCE',prediction_ref=prediction,
        original_restore_source_closure_verified=True)
    body_ref=save_json(worker_out/'REPORT.json',body)
    code={role:dict(path='/synthetic/'+role+'.py',bytes=values[0],sha256=values[1]) for role,values in {
        'entry':(25044,'26a0e5066371829c84610d421919957fc587bcbcaf14e25f6a0401488e61e066'),
        'owner':(26696,'5286468fcc4ab98a29be181251e1aee950d5d8becb59cb46c6c6d68c3ec6648e'),
        'worker':(21286,'af3860ae0df9de7efa338f725ad05f9d7f4c1c76ee8fba84bc7c6be00a8b907f'),
        'core':(18285,'71e10c6539f34105c9726c73e9a0953973a76334da07059cf7a5fc2b619606be')}.items()}
    spec=dict(schema='ROOT-U74-DGX-fixed50-single-shard-supervision-spec-v1',operation='GPU_U74_DGX_FIXED50_SINGLE_SHARD_PREDICTION',
        phase='SYNTHETIC_PHASE',entry_source=code['entry'],owner_source=code['owner'],
        request=dict(worker_source=code['worker'],prediction_core_source=code['core'],output_directory=worker_out.as_posix()),
        output_directory=entry_out.as_posix(),owner_output_directory=owner_out.as_posix())
    spec_ref=save_json(root/'ORIGINAL_SPEC.json',spec)
    entry=dict(schema='U74-DGX-fixed50-single-shard-entry-report-v1',status='COMPLETE_PENDING_ROOT_ACCEPTANCE',
        spec_ref=spec_ref,phase=spec['phase'],worker_report_ref=body_ref,prediction_ref=prediction,worker_report=body)
    entry_ref=save_json(entry_out/'REPORT.json',entry)
    logs={}
    for name in ('stdout.log','stderr.log'):(owner_out/name).write_bytes(b'');logs[name]=file_ref(owner_out/name)
    owner=owner_fixture();owner.update(schema='U74-Linux-owned-fixed50-single-shard-prediction-receipt-v1',
        status='OWNED_CHILD_COMPLETE_PENDING_ROOT_REVIEW',spec_ref=spec_ref,phase=spec['phase'],operation=spec['operation'],
        sys_executable='/synthetic/python',returncode=0,reaped=True,child_launched=True,original_birth_absent=True,
        final_group_scan_resolved=True,remaining_owned_group=[],all_owned_processes_absent=True,
        entry_report_ref=entry_ref,worker_report_ref=body_ref,prediction_ref=prediction,logs=logs)
    save_json(owner_out/'OWNER_RECEIPT.json',owner);save_json(owner_out/'RUN_STATE.json',owner)
    streams={}
    for name in ('original.stdout','original.stderr'):(root/name).write_bytes(b'');streams[name]=file_ref(root/name)
    terminal=dict(root_observed_tool_exit_code=0,root_observed_tool_chunk_id='SYNTHETIC_ORIGINAL_NOT_ACTUAL',
        SSH_returncode=0,owner_process_exit_code=0,original_exec_wait_observed=True,
        command=['exec','/synthetic/python','-u',code['owner']['path'],'--spec',spec_ref['path'],'--spec-sha',spec_ref['sha256']],
        stdout_ref=streams['original.stdout'],stderr_ref=streams['original.stderr'],attempt_id='SYNTHETIC_ATTEMPT',phase=spec['phase'],spec_ref=spec_ref)
    request=dict(schema='U77-external-collect-request-v1',operation='COLLECT',collector_source_ref=file_ref(source),
        acceptance_source_ref=file_ref(acceptance),spec_ref=spec_ref,original_terminal=terminal,output_directory=(root/'COLLECT').as_posix())
    request_ref=save_json(root/'COLLECT_REQUEST.json',request)
    return request,request_ref,proc_fixture(root)


class Collector(unittest.TestCase):
    def test_generic_zero_fields_parse_and_census_but_owned_identity_remains_strict(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory).resolve();request,request_ref,proc=collection_fixture(root)
            generic=[dict(pid=1,state='S',parent_pid=0,process_group=1,session=1,start_ticks=0),
                dict(pid=2,state='I',parent_pid=0,process_group=0,session=0,start_ticks=0)]
            for identity in generic:
                self.assertEqual(collector.parse_proc_stat(stat_text(identity)),identity)
                (proc/str(identity['pid'])).mkdir()
                (proc/str(identity['pid'])/'stat').write_text(stat_text(identity))
            with patch.object(collector,'production_proc_root',return_value=proc):
                result=collector.collect(request,request_ref)
            report=json.loads(Path(result['collection_ref']['path']).read_bytes())
            self.assertEqual(report['status'],'COLLECTED_AWAITING_ROOT_TOOL_TERMINAL')
            self.assertTrue(report['observations']['owned_group_scan_resolved'])
            self.assertEqual(report['observations']['remaining_owned_group'],[])
            self.assertFalse(report['root_accepted'])
            for role in ('owner_identity','child_identity'):
                for key in ('pid','process_group','session','start_ticks'):
                    for invalid in (0,False):
                        bad=owner_fixture();bad[role][key]=invalid
                        with self.subTest(role=role,key=key,invalid=invalid),self.assertRaises(ValueError):
                            collector.observe_processes(proc,bad)
            for key in ('pid','parent_pid','process_group','session','start_ticks'):
                bad=deepcopy(generic[1]);bad[key]=-1
                with self.subTest(generic_negative=key),self.assertRaises(ValueError):
                    collector.parse_proc_stat(stat_text(bad))
            bad=deepcopy(generic[1]);bad['pid']=0
            with self.assertRaises(ValueError):collector.parse_proc_stat(stat_text(bad))

    def test_enumerated_process_disappears_before_stat_is_unknown_not_group_absence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory).resolve();request,request_ref,proc=collection_fixture(root)
            member=dict(pid=300,state='S',parent_pid=1,process_group=201,session=201,start_ticks=1010)
            (proc/'300/stat').write_text(stat_text(member))
            original_read=Path.read_text;disappeared=[]
            def read(path,*args,**kwargs):
                if path==proc/'300/stat':
                    # A forks B and disappears after directory enumeration/check.
                    child=dict(pid=301,state='S',parent_pid=1,process_group=201,session=201,start_ticks=1011)
                    (proc/'301').mkdir();(proc/'301/stat').write_text(stat_text(child))
                    path.unlink();path.parent.rmdir();disappeared.append(300)
                return original_read(path,*args,**kwargs)
            with patch.object(collector,'production_proc_root',return_value=proc),patch.object(Path,'read_text',read):
                with self.assertRaisesRegex(ValueError,'UNKNOWN'):
                    collector.collect(request,request_ref)
            self.assertEqual(disappeared,[300])
            report=json.loads((Path(request['output_directory'])/'COLLECTION.json').read_bytes())
            self.assertEqual(report['status'],'UNKNOWN_OR_FAILED')
            self.assertIsNone(report['observations'])
            self.assertFalse(report['root_accepted'])

    def test_fixture_birth_and_group_scan_reports_absence_without_future_terminal(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            proc=proc_fixture(Path(directory));owner=owner_fixture()
            result=collector.observe_processes(proc,owner)
            self.assertTrue(result['owner_original_birth_absent'])
            self.assertTrue(result['child_original_birth_absent'])
            self.assertTrue(result['owned_group_scan_resolved'])
            self.assertEqual(result['remaining_owned_group'],[])
            self.assertIsNone(result['owner_current_identity']);self.assertIsNone(result['child_current_identity'])
            self.assertNotIn('root_observed_tool_exit_code',result)
            self.assertNotIn('root_observed_tool_chunk_id',result)
            (proc/'200').mkdir();(proc/'200/stat').write_text(stat_text(owner['owner_identity']))
            with self.assertRaises(ValueError):collector.observe_processes(proc,owner)

    def test_collection_closes_five_outputs_and_empty_logs_but_never_self_certifies(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory).resolve();request,request_ref,proc=collection_fixture(root)
            with patch.object(collector,'production_proc_root',return_value=proc):result=collector.collect(request,request_ref)
            report=json.loads(Path(result['collection_ref']['path']).read_bytes())
            self.assertEqual(report['status'],'COLLECTED_AWAITING_ROOT_TOOL_TERMINAL')
            self.assertEqual(set(report['evidence_refs']),{'owner_receipt','owner_run_state','entry_report','worker_report','prediction'})
            self.assertTrue(report['observations']['owned_group_scan_resolved'])
            self.assertFalse(report['root_accepted']);self.assertFalse(report['U75_acceptance_called'])
            self.assertNotIn('root_observed_tool_exit_code',report)
            self.assertNotIn('root_observed_tool_chunk_id',report)
            self.assertEqual(report['log_refs']['owner_stdout']['bytes'],0)
            with patch.object(collector,'production_proc_root',return_value=proc),self.assertRaises(FileExistsError):collector.collect(request,request_ref)

    def test_later_envelope_consumes_external_collector_terminal_and_exact17_post(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory).resolve();request,request_ref,proc=collection_fixture(root)
            with patch.object(collector,'production_proc_root',return_value=proc):result=collector.collect(request,request_ref)
            report=json.loads(Path(result['collection_ref']['path']).read_bytes())
            terminal=dict(root_observed_tool_exit_code=0,root_observed_tool_chunk_id='SYNTHETIC_LATER_COLLECTOR_TOOL_NOT_ACTUAL',
                collection_ref=result['collection_ref'],request_ref=request_ref,collector_source_ref=request['collector_source_ref'],
                spec_ref=request['spec_ref'],phase=report['phase'],attempt_id=report['attempt_id'])
            envelope=dict(schema='U77-external-envelope-request-v1',operation='ENVELOPE',
                collector_source_ref=request['collector_source_ref'],acceptance_source_ref=request['acceptance_source_ref'],
                collection_ref=result['collection_ref'],collector_terminal=terminal,output_directory=(root/'ENVELOPE').as_posix())
            envelope_ref=save_json(root/'ENVELOPE_REQUEST.json',envelope)
            output=collector.envelope(envelope,envelope_ref)
            post=json.loads(Path(output['post_ref']['path']).read_bytes())
            self.assertEqual(len(post),17)
            self.assertEqual(post['root_observed_tool_chunk_id'],terminal['root_observed_tool_chunk_id'])
            self.assertEqual(post['original_terminal'],request['original_terminal'])
            self.assertFalse(json.loads(Path(output['envelope_ref']['path']).read_bytes())['U75_acceptance_called'])
            for mutation in ('boolean_zero','other_attempt','reused_original_chunk','missing_tool'):
                bad=deepcopy(envelope);bad['output_directory']=(root/('BAD_'+mutation)).as_posix()
                if mutation=='boolean_zero':bad['collector_terminal']['root_observed_tool_exit_code']=False
                elif mutation=='other_attempt':bad['collector_terminal']['attempt_id']='OTHER_ATTEMPT'
                elif mutation=='reused_original_chunk':bad['collector_terminal']['root_observed_tool_chunk_id']=request['original_terminal']['root_observed_tool_chunk_id']
                else:del bad['collector_terminal']['root_observed_tool_chunk_id']
                bad_ref=save_json(root/(mutation+'.json'),bad)
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):collector.envelope(bad,bad_ref)
                self.assertFalse(Path(bad['output_directory']).exists())

    def test_cli_dispatches_only_pinned_request_and_has_no_proc_override(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory).resolve();request,request_ref,_=collection_fixture(root)
            argv=['--request-file',request_ref['path'],'--request-sha256',request_ref['sha256']]
            descriptor=dict(status='SYNTHETIC_PENDING',collection_ref={'synthetic':True})
            with patch.object(collector.platform,'system',return_value='Linux'),patch.object(collector,'collect',return_value=descriptor) as run:
                output=io.StringIO()
                with redirect_stdout(output):code=collector.main(argv)
                self.assertEqual(code,0);self.assertEqual(json.loads(output.getvalue()),descriptor)
                run.assert_called_once_with(request,request_ref)
                run.reset_mock();errors=io.StringIO()
                with redirect_stderr(errors):code=collector.main(argv[:-1]+['f'*64])
                self.assertEqual(code,1);run.assert_not_called()
                with redirect_stderr(errors),self.assertRaises(SystemExit):collector.main(argv+['--proc-root',str(root)])
                run.assert_not_called()

    def test_unknown_permission_reboot_zombie_reuse_and_changed_bytes_preserve_failure(self):
        original_read=Path.read_text
        for scenario in ('permission','reboot','zombie','reused_pid','incomplete_stat','surviving_group','changed_prediction'):
            with self.subTest(scenario=scenario),tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
                root=Path(directory).resolve();request,request_ref,proc=collection_fixture(root)
                owner=owner_fixture()
                if scenario=='reboot':(proc/'sys/kernel/random/boot_id').write_text('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n')
                elif scenario in ('zombie','reused_pid'):
                    identity=deepcopy(owner['child_identity'] if scenario=='zombie' else owner['owner_identity'])
                    if scenario=='zombie':identity['state']='Z'
                    else:identity['start_ticks']+=100
                    (proc/str(identity['pid'])).mkdir();(proc/str(identity['pid'])/'stat').write_text(stat_text(identity))
                elif scenario=='incomplete_stat':(proc/'300/stat').unlink()
                elif scenario=='surviving_group':
                    member=dict(pid=302,state='S',parent_pid=1,process_group=201,session=201,start_ticks=1010)
                    (proc/'302').mkdir();(proc/'302/stat').write_text(stat_text(member))
                elif scenario=='changed_prediction':(root/'ORIGINAL_WORKER/U66_SHARD.json').write_bytes(b'CHANGED_SYNTHETIC_BYTES')
                def read(path,*args,**kwargs):
                    if scenario=='permission' and path==proc/'300/stat':raise PermissionError('SYNTHETIC_PERMISSION_DENIED')
                    return original_read(path,*args,**kwargs)
                with patch.object(collector,'production_proc_root',return_value=proc),patch.object(Path,'read_text',read):
                    with self.assertRaises((ValueError,PermissionError)):collector.collect(request,request_ref)
                report=json.loads((Path(request['output_directory'])/'COLLECTION.json').read_bytes())
                self.assertEqual(report['status'],'UNKNOWN_OR_FAILED')
                self.assertIsNone(report['observations'])
                self.assertIn('traceback',report['error']);self.assertFalse(report['root_accepted'])
                self.assertNotIn('root_observed_tool_exit_code',report)
                if scenario=='permission':self.assertIn('SYNTHETIC_PERMISSION_DENIED',report['error']['traceback'])


if __name__=='__main__':unittest.main()
