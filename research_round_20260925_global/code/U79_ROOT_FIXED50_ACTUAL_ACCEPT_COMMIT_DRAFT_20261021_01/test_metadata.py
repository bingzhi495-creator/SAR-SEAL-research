"""Finite synthetic tests of actual U75 caller and exclusive authority update.

No real chain, SSH, /proc, data, checkpoints, numerical model or old test suite.
Fixture source pin substitutions are confined to an ephemeral U75 module.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
U75=HERE.parent/'U75_ROOT_FIXED50_SHARD_ACCEPTANCE_DRAFT_20261021_01'
sys.path.insert(0,str(U75))

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def reference(path,raw):
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

class Caller(unittest.TestCase):
    def test_actual_u75_call_is_persisted_once_and_reused_directory_refused(self):
        ctl=load(HERE/'control.py','u79')
        fixture=load(U75/'test_metadata.py','u79_fixture')
        data=fixture.synthetic_inputs()
        acceptance=fixture.acceptance
        original=acceptance.definitions
        def definitions(ref,raw):
            module=original(ref,raw)
            if hasattr(module,'ENCODER_SHA256'):module.ENCODER_SHA256=data['encoder']['sha256']
            if hasattr(module,'SOURCE_PINS'):
                module.SOURCE_PINS=data['synthetic_source_pins'];module.Path=fixture.MetadataPath
            return module
        source_raw=(U75/'acceptance.py').read_bytes()
        source_ref=reference(U75/'acceptance.py',source_raw)
        def raw(ref):
            if ref==source_ref:return source_raw
            return data['objects'][ref['path']]
        with tempfile.TemporaryDirectory(dir=str(HERE)) as td:
            request=dict(schema=ctl.ACCEPT_SCHEMA,acceptance_source_ref=source_ref,
                spec_ref=data['spec_ref'],postcheck_ref=data['post_ref'],
                original_terminal=data['terminal'],persistence_directory=str(Path(td)/'accepted'))
            with patch.object(ctl,'load_source',return_value=acceptance),patch.object(acceptance,'definitions',side_effect=definitions):
                result=ctl.accept_and_persist(request,read_raw=raw,closed=data['closed'])
                with self.assertRaises(FileExistsError):
                    ctl.accept_and_persist(request,read_raw=raw,closed=data['closed'])
            actual=json.loads(Path(result['proof_ref']['path']).read_bytes())
            self.assertEqual(actual['prediction_ref'],result['prediction_ref'])
            self.assertEqual(actual['original_terminal'],data['terminal'])
            self.assertIs(result['CAS_performed'],False)
            self.assertIs(result['AP'],False)
            self.assertEqual(result['formal_training_budget_credit'],0)

    def test_fixed_source_pin_checked_before_call_or_output(self):
        ctl=load(HERE/'control.py','u79_badpin')
        with tempfile.TemporaryDirectory(dir=str(HERE)) as td:
            out=Path(td)/'must_not_exist'
            request=dict(schema=ctl.ACCEPT_SCHEMA,acceptance_source_ref=reference('/fake/u75.py',b'x'),
                spec_ref={},postcheck_ref={},original_terminal={},persistence_directory=str(out))
            with patch.object(ctl,'load_source',side_effect=AssertionError('must not call')):
                with self.assertRaises(ValueError):ctl.accept_and_persist(request,read_raw=lambda ref:b'x',closed=None)
            self.assertFalse(out.exists())

    def test_commit_cas_preserves_state_refuses_duplicate_stale_and_future_terminal(self):
        ctl=load(HERE/'control.py','u79_commit')
        # Isolated commit protocol fixture, not an accepted scientific proof.
        proof=dict(shard_index=0,prediction_ref=reference('/remote/prediction.json',b'{}'),
            original_terminal=dict(root_observed_tool_chunk_id='ORIGINAL'),
            evidence_refs={'postcheck':reference('/remote/POST.json',b'{}')},
            endpoint={'fixed_epoch':50,'updates_per_arm':37050,'training_qualified':{'path':'/final'}},phase='fixture',attempt_id='fixture0')
        post=dict(root_observed_tool_chunk_id='COLLECTOR')
        raw_proof=(json.dumps(proof)+'\n').encode();remote_proof=reference('/remote/PROOF.json',raw_proof)
        report=dict(schema=ctl.PERSIST_SCHEMA,status='PERSISTED_AWAITING_ROOT_ACCEPT_TOOL_TERMINAL',
            proof_ref=remote_proof,prediction_ref=proof['prediction_ref'],phase='fixture',attempt_id='fixture0',
            AP=False,CAS_performed=False,formal_training_budget_credit=0)
        raw_report=(json.dumps(report)+'\n').encode();report_ref=reference('/remote/PERSISTENCE.json',raw_report)
        facts=dict(root_observed_tool_exit_code=0,root_observed_tool_chunk_id='ACCEPT',
            accept_report_ref=report_ref,proof_ref=remote_proof)
        with tempfile.TemporaryDirectory(dir=str(HERE)) as td:
            authority=Path(td)/'ACTIVE.json'
            initial=dict(current_all_leaf_sessions_reaped=True,GPU_owner_or_worker_active=False,gpu_training_active=False,
                unrelated={'preserve':[1,2,3]},U57_independent_DGX_original_update_training_20261018_01={
                    'FORMAL_prefix':37050,'active_phase':None,'latest_accepted':{'FORMAL':{'path':'/final'}}})
            original_raw=(json.dumps(initial)+'\n').encode();authority.write_bytes(original_raw)
            with patch.object(ctl,'validate_commit_inputs',return_value=(proof,report,post)):
                with self.assertRaises(ValueError):
                    ctl.commit_shard(authority,hashlib.sha256(original_raw).hexdigest(),facts,
                                     {},{}, {},read_raw=lambda ref:b'')
                bad=copy.deepcopy(facts);bad['root_observed_tool_exit_code']=False
                with self.assertRaises(ValueError):
                    ctl.commit_shard(authority,hashlib.sha256(original_raw).hexdigest(),bad,
                                     report_ref,remote_proof,{},read_raw=lambda ref:b'')
                self.assertEqual(authority.read_bytes(),original_raw)
                nonfinal=copy.deepcopy(initial);nonfinal[ctl.TRAINING_KEY]['FORMAL_prefix']=7168
                nonfinal_raw=(json.dumps(nonfinal)+'\n').encode();authority.write_bytes(nonfinal_raw)
                with self.assertRaisesRegex(ValueError,'fixed E50'):
                    ctl.commit_shard(authority,hashlib.sha256(nonfinal_raw).hexdigest(),facts,
                        report_ref,remote_proof,{},read_raw=lambda ref:b'')
                self.assertEqual(authority.read_bytes(),nonfinal_raw)
                authority.write_bytes(original_raw)
                result=ctl.commit_shard(authority,hashlib.sha256(original_raw).hexdigest(),facts,
                    report_ref,remote_proof,{},read_raw=lambda ref:b'')
                saved=json.loads(authority.read_bytes())
                self.assertEqual(saved['unrelated'],initial['unrelated'])
                self.assertEqual(saved[ctl.ACCOUNT_KEY]['accepted_shards'][0]['proof_ref'],remote_proof)
                self.assertEqual(saved['U57_independent_DGX_original_update_training_20261018_01'],
                                 initial['U57_independent_DGX_original_update_training_20261018_01'])
                with self.assertRaises(ValueError):
                    ctl.commit_shard(authority,hashlib.sha256(original_raw).hexdigest(),facts,
                        report_ref,remote_proof,{},read_raw=lambda ref:b'')
                newhash=hashlib.sha256(authority.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    ctl.commit_shard(authority,newhash,facts,report_ref,remote_proof,{},read_raw=lambda ref:b'')
                self.assertEqual(result['accepted_count'],1)

    def test_cleanup_failure_cannot_replace_original_write_failure(self):
        ctl=load(HERE/'control.py','u79_failure')
        proof=dict(shard_index=0,original_terminal={'root_observed_tool_chunk_id':'ORIGINAL'},
            endpoint={'training_qualified':{'path':'/final'}},phase='fixture',attempt_id='fixture0')
        report_ref={'path':'/report'};proof_ref={'path':'/proof'}
        facts=dict(root_observed_tool_exit_code=0,root_observed_tool_chunk_id='ACCEPT',
            accept_report_ref=report_ref,proof_ref=proof_ref)
        with tempfile.TemporaryDirectory(dir=str(HERE)) as td:
            authority=Path(td)/'ACTIVE.json'
            raw=encoded_state=dict(current_all_leaf_sessions_reaped=True,GPU_owner_or_worker_active=False,
                gpu_training_active=False,U57_independent_DGX_original_update_training_20261018_01={
                    'FORMAL_prefix':37050,'active_phase':None,'latest_accepted':{'FORMAL':{'path':'/final'}}})
            raw=(json.dumps(encoded_state)+'\n').encode();authority.write_bytes(raw)
            with patch.object(ctl,'validate_commit_inputs',return_value=(proof,{},dict(root_observed_tool_chunk_id='COLLECTOR'))),\
                 patch.object(ctl,'write_new',side_effect=OSError('PRIMARY_WRITE_FAILURE')),\
                 patch.object(ctl.Path,'unlink',side_effect=OSError('SECONDARY_CLEANUP_FAILURE')):
                with self.assertRaisesRegex(OSError,'PRIMARY_WRITE_FAILURE'):
                    ctl.commit_shard(authority,hashlib.sha256(raw).hexdigest(),facts,
                        report_ref,proof_ref,{},read_raw=lambda ref:b'')
            self.assertEqual(authority.read_bytes(),raw)

if __name__=='__main__':unittest.main()
