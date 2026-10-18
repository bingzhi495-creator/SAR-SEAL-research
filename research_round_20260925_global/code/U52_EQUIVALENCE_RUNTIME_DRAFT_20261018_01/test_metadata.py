"""Tiny stdlib spec/terminal guards only; no processes, ML or actual file refs."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest


def module(name):
    spec=importlib.util.spec_from_file_location('_U52_TEST_'+name,Path(__file__).with_name(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


entry,owner=module('entry'),module('linux_owner')


def spec_fixture():
    ref=lambda name,sha='a'*64,size=1:dict(path='/sources/'+name,bytes=size,sha256=sha)
    value=dict(schema='ROOT-U52-DGX-equivalence-spec-v1',operation='GPU_U52_DGX_SHARED_ENCODER_EQUIVALENCE',
        phase='metadata_fixture',entry_source=ref('entry.py'),owner_source=ref('linux_owner.py'),
        body_source=ref('equivalence_body.py',entry.BODY_SHA256,entry.BODY_BYTES),
        source_refs={k:ref(k+'.py',sha) for k,sha in entry.SOURCE_PINS.items()},
        library_sources={k:ref(k+'.py') for k in ('psutil','torch','numpy')},
        context=dict(portable_refs={'54':ref('a.pt'),'710':ref('b.pt')},scratch_directory='/out/worker/AUDIT'),
        prelock=dict(limits=deepcopy(owner.LIMITS)),output_directory='/out/worker',owner_output_directory='/out/owner')
    value['sources']=sorted([value[k] for k in ('entry_source','body_source','owner_source')]
        +list(value['source_refs'].values())+list(value['library_sources'].values())
        +list(value['context']['portable_refs'].values()),key=lambda row:row['path'])
    return value


def completed(candidate_pass):
    result=dict(status='COMPARISON_COMPLETED_PENDING_ROOT',new_training=True,pilot_only=True,
        main_training=False,formal_training_budget_credit=0,U40_frontier_modified=False,qualified=False,
        AP=False,DEV=False,TEST=False,innovation_established=False,speedup_established=False,
        successful_optimizer_updates=14,optimizer_steps_entered=14,
        caller_CPU_RNG_restored=True,caller_NumPy_RNG_restored=True,caller_Python_RNG_restored=True,
        caller_postinitialization_CUDA_RNG_restored=True,module_aliases_restored=True,cleanup_errors=[],
        final_CUDA_cleanup=dict(allocated_bytes=0,reserved_bytes=0),
        verdict='WITHIN_PRELOCKED_TOLERANCES' if candidate_pass else 'CANDIDATE_OUTSIDE_PRELOCKED_TOLERANCES',
        reader_visits=[54,710,54,710,54,710,54]*2,arms={})
    for arm in ('PLAIN','FULL'):
        stages=[]
        for trajectory,indices,steps in (('WARMUP',[54],[1]),('BASE_A',[710,54],[2,3]),
                                         ('BASE_B',[710,54],[2,3]),('CANDIDATE',[710,54],[2,3])):
            for index,step in zip(indices,steps):
                stages.append(dict(trajectory=trajectory,dataset_index=index,step=step,G=1 if index==54 else 27,
                    optimizer_step_returned=True,calls=dict(backbone=1,feature_exit=1,legacy_forward=0,
                        **{'final_super_0_'+str(i):0 for i in range(1,5)})))
        result['arms'][arm]=dict(status='COMPARISON_COMPLETED',updates_completed=7,baseline_control_exact=True,
            candidate_within=candidate_pass,warm_head_nonzero=True,stages=stages,
            final_CUDA_cleanup=dict(allocated_bytes=0,reserved_bytes=0),
            trajectories={k:dict(steps=[2,3],passed=candidate_pass if k=='CANDIDATE' else True,
                after_release=dict(allocated_bytes=0,reserved_bytes=0)) for k in ('BASE_A','BASE_B','CANDIDATE')})
    return result


class MetadataTests(unittest.TestCase):
    def test_exact_body_context_and_22_sources(self):
        value=spec_fixture()
        self.assertEqual(len(entry.source_table(value)),22)
        owner.validate_spec(value)
        for change in ('body','hidden','scratch'):
            bad=deepcopy(value)
            if change=='body':bad['body_source']['sha256']='b'*64
            if change=='hidden':bad['sources'].append(dict(path='/extra.pt',bytes=1,sha256='b'*64))
            if change=='scratch':bad['context']['scratch_directory']='/out/owner/AUDIT'
            with self.assertRaises(ValueError):entry.source_table(bad)

    def test_negative_numerical_result_is_completed_diagnostic(self):
        entry.completed_equivalence(completed(True))
        entry.completed_equivalence(completed(False))
        for change in ('baseline','count','RNG','verdict'):
            bad=completed(False)
            if change=='baseline':bad['arms']['FULL']['baseline_control_exact']=False
            if change=='count':bad['successful_optimizer_updates']=13
            if change=='RNG':bad['caller_CPU_RNG_restored']=False
            if change=='verdict':bad['verdict']='WITHIN_PRELOCKED_TOLERANCES'
            with self.assertRaises(ValueError):entry.completed_equivalence(bad)

    def test_actual_terminal_and_new_body_status(self):
        self.assertTrue(owner.terminal_success(0,True,True,[],[],'COMPARISON_COMPLETED_PENDING_ROOT'))
        for code,reaped,absent,group,violations,status in (
            (1,True,True,[],[],'COMPARISON_COMPLETED_PENDING_ROOT'),
            (0,False,True,[],[],'COMPARISON_COMPLETED_PENDING_ROOT'),
            (0,True,False,[],[],'COMPARISON_COMPLETED_PENDING_ROOT'),
            (0,True,True,[],['RSS'],'COMPARISON_COMPLETED_PENDING_ROOT'),
            (0,True,True,[],[],'BASELINE_CONTROL_STOPPED')):
            self.assertFalse(owner.terminal_success(code,reaped,absent,group,violations,status))


if __name__=='__main__':unittest.main()
