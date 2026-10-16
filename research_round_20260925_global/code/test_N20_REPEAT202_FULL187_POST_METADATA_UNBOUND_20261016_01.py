"""Pure-stdlib synthetic metadata only; never import an entry or open its refs."""
import argparse
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
DEFAULT = HERE/'N20_REPEAT202_FULL187_POST_ACTUAL_ROOT_CPU_UNBOUND_DRAFT_20261016_01.py'
POST_SCHEMA = 'ROOT-N20-repeat202-fixedDEV-full187-post-actual-CPU-spec-v1'


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def ref_shape(value):
    require(type(value) is dict and set(value) == {'path','bytes','sha256'}, 'synthetic strictref')


def load_header(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'N20_FINAL_CPU_SCHEMA','N20_POST_RECORD_KEYS','N20_REPEAT202_POST_SPEC_SCHEMA'}
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'n20_validate_post_header']
    ns = {'Path':Path,'require':require,'ref_shape':ref_shape,'__file__':str(Path(__file__).resolve())}
    exec(compile(ast.Module(body=nodes,type_ignores=[]), '<synthetic-header-only>', 'exec'), ns)
    return ns['n20_validate_post_header']


def preview_call(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
           and n.func.attr=='n20_final_main_preview']
    require(len(calls)==1,'One actual post interpretation delegation')
    controls={'SYNTHETIC_ONLY':True};gate={'first201_root_permission':'SYNTHETIC_QUALIFIED_FIRST201'}
    received=[]
    def expected(interpreter, observed, context):
        received.append((interpreter,observed,context));return 'SYNTHETIC_PREVIEW'
    ns=dict(final=SimpleNamespace(n20_final_main_preview=expected),interpreter=object(),Q={'control_results':controls},gate=gate)
    result=eval(compile(ast.Expression(body=calls[0]),'<actual-preview-call-only>','eval'),ns)
    require(result=='SYNTHETIC_PREVIEW' and received[0][1] is controls and received[0][2] is gate,
        'Two-seed parent receives both actual202 controls and qualified-first201 context')


def provenance(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='n20_post_validate_pending_provenance']
    require(len(nodes)==1,'One pending provenance validator')
    ns={'require':require}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<synthetic-provenance-only>','exec'),ns)
    return ns['n20_post_validate_pending_provenance']


def pending_pair():
    Q={key:False for key in ('qualified','ROOT_actual_CPU0_verified','current_owned_CPU_terminal_self_certified',
        'numerical_forward_rerun','head_loaded','CUDA_initialized','local_AP_averaged',
        'CPU_scout_forward_used_for_ranking','AP_results_interpreted_as_qualified',
        'task_efficacy_established','innovation_established','TGRS_readiness_established','failed26_partials_reused')}
    Q.update(original_training_GPU_actual0_verified=True,training_saved_artifact_recovery=False,independent_endpoint_CPU_qualified=True)
    return Q,copy.deepcopy(Q)


def fixture():
    def r(name):
        return {'path':str(HERE/(name+'.SYNTHETIC_ONLY')), 'bytes':0, 'sha256':'0'*64}
    record = {k:r(k) for k in ('source','spec','Q','RAW','admission','phase_binding','bootstrap')}
    record['owner_index'] = 267
    return dict(schema=POST_SCHEMA, post_owner_index=268, final_owner_index=267,
        normal_RAW_count=268, expected_normal_RAW_count=268,
        total_failed_RAW_count=26, total_terminal_identity_pin_count=294,
        deadline_seconds=1800,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,
        maximum_process_RSS_bytes=16*1024**3,initialization_seed=202610150202,
        automatic_retry=False,failed26_partials_reused=False,final_record=record,
        completed_final_CPU_source=record['source'], entry_source=dict(r('own'),path=str(Path(__file__).resolve())),
        CPU_admission_file=str(HERE/'OWN_ADMISSION.SYNTHETIC_ONLY'),
        RAW_inputs=[{'phase':'ancestor_%d'%i} for i in range(268)],phase='SYNTHETIC_NEW_POST')


class Metadata(unittest.TestCase):
    def test_dynamic_post_owner_with_independent_schema(self):
        self.assertTrue(HEADER(fixture()))

    def test_invalid_identity_seed_counts_or_live_post_are_rejected(self):
        mutations=(('schema','ROOT-N20-repeat202-fixedDEV-full187-CPU-COCO-final-spec-v1'),
                   ('initialization_seed',202610150201),('post_owner_index',267),
                   ('normal_RAW_count',269),('expected_normal_RAW_count',269),
                   ('total_failed_RAW_count',25),('total_terminal_identity_pin_count',293),
                   ('automatic_retry',True),('failed26_partials_reused',True))
        for key,value in mutations:
            with self.subTest(key=key):
                s=fixture();s[key]=value
                with self.assertRaises((AssertionError,KeyError,ValueError)):
                    HEADER(s)
        s=fixture();s['final_record']['owner_index']=266
        with self.assertRaises(AssertionError):HEADER(s)
        s=fixture();s['RAW_inputs'][0]['phase']=s['phase']
        with self.assertRaises(AssertionError):HEADER(s)

    def test_actual_preview_call_preserves_first201_context_and_actual202_controls(self):
        preview_call(SOURCE)

    def test_pending_normal_ancestry_rejects_recovery_alias_and_self_certification(self):
        validate=provenance(SOURCE);self.assertTrue(validate(*pending_pair()))
        Q,details=pending_pair();Q['training_artifacts_recovery_qualified']=True
        with self.assertRaises(AssertionError):validate(Q,details)
        Q,details=pending_pair();Q['qualified']=True
        with self.assertRaises(AssertionError):validate(Q,details)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=DEFAULT)
    args, rest=parser.parse_known_args();SOURCE=args.source;HEADER=load_header(args.source)
    unittest.main(argv=[__file__]+rest)
