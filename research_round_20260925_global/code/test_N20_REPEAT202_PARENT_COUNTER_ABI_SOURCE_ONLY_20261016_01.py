"""Metadata-only fixture for the newly observed parent-spec counter ABI."""
import json,runpy,sys,tempfile,unittest
from pathlib import Path

FRONT=Path(__file__).resolve().parent/'N20_REPEAT202_TRAIN_GPU_FRONT_UNBOUND_DRAFT_20261016_01.py'

class ParentCounterABI(unittest.TestCase):
    def exercise(self,schema,index,expected_marker):
        fn=runpy.run_path(str(FRONT))['n20_repeat202_completed']
        with tempfile.TemporaryDirectory(prefix='N20_parent_counter_fixture_') as folder:
            base=Path(folder);out=base/'out';out.mkdir()
            def ref(name):
                p=base/name;p.touch();return dict(path=str(p),bytes=0,sha256='0'*64)
            source=ref('source.py');sp=ref('spec.json');ad=ref('ad.json')
            output_key='metadata' if index==179 else ('report' if index==180 else 'Q')
            cuda='0' if index==180 else '';ad_key='GPU_admission' if cuda else 'CPU_admission'
            output=dict(path=str(out/'output.json'),bytes=0,sha256='0'*64)
            record=dict(source=source,spec=sp,RAW=ref('RAW.json'),admission=ad,
                phase_binding=ref('phase.json'),bootstrap=ref('bootstrap.py'),owner_index=index)
            record[output_key]=output
            row=dict(receipt=record['RAW'],phase='fixture',admission=ad,
                phase_binding=record['phase_binding'],bootstrap=record['bootstrap'])
            prior=[None]*index
            parent=dict(schema=schema,normal_RAW_count=index,entry_source=source,phase='fixture',
                RAW_inputs=prior,sources=[],terminal_identity_pins={},output_directory=str(out))
            parent[ad_key+'_file']=ad['path']
            if expected_marker is not None:parent['expected_normal_RAW_count']=expected_marker
            Path(sp['path']).write_bytes(json.dumps(parent).encode())
            Path(output['path']).write_bytes(json.dumps(dict(source=source,spec=sp,phase='fixture',**{ad_key:ad})).encode())
            class Lineage:
                def owner_ref_join(self,*args):return index
            gate=dict(closure={},context={},owners={},lineage=Lineage())
            spec=dict(RAW_inputs=prior+[row],terminal_identity_pins={})
            def require(ok,message):
                if not ok:raise RuntimeError(message)
            fn.__globals__.update(require=require,closed=lambda c,r:Path(r['path']),
                read=lambda p:json.loads(Path(p).read_text()),Path=Path)
            value=fn(record,output_key,index,cuda,gate,spec)
            assert 'torch' not in sys.modules and 'numpy' not in sys.modules
            return value

    def test_exact_source_only_originals_do_not_invent_expected_field(self):
        self.exercise('ROOT-N20-source-only-repeat202-initialization-spec-v1',179,None)
        self.exercise('ROOT-N20-repeat202-synthetic-GPU-cleanup-probe-spec-v1',180,None)

    def test_legacy_expected_counter_is_still_required_and_exact(self):
        schema='ROOT-N20-fixedDEV-full187-CPU-COCO-final-spec-v1'
        self.exercise(schema,177,177)
        for expected in (None,176):
            with self.assertRaises((KeyError,RuntimeError)):self.exercise(schema,177,expected)

    def test_exception_does_not_extend_to_other_owner_or_add_field(self):
        schema='ROOT-N20-source-only-repeat202-initialization-spec-v1'
        with self.assertRaises((KeyError,RuntimeError)):self.exercise(schema,178,None)
        with self.assertRaises((KeyError,RuntimeError)):self.exercise(schema,179,179)

if __name__=='__main__':unittest.main()
