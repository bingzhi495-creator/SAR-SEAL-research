"""Synthetic endpoint/source metadata only; no checkpoint or NN execution."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import unittest


PUBLIC=Path('E:/SAR/seal/github_release_20261011/research_round_20260925_global/code/U40_FIXED50_DEV_PREDICTION_DRAFT_20261017_01')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def ref(name):return dict(path='E:/synthetic-only/'+name+'.py',bytes=10,sha256=hashlib.sha256(name.encode()).hexdigest())


def fixture(core):
    sources={k:ref(k) for k in core.TRAINING_ROLES}
    for k,v in core.TRAINING_PINS.items():sources[k]['sha256']=v
    sources['u36_helpers']['sha256']=core.SOURCE_PINS['u36_helpers']
    refs={k:ref(k) for k in ('training_qualified','pair_commit','preparation','order')}
    cp={arm:ref(arm) for arm in core.ARMS}
    component={k:sources[k] for k in core.TRAINING_COMPONENTS}
    identity=dict(sources=dict(core_source=sources['core_source'],component_sources=component),
        recipe={'synthetic_only':True},order_sha256=core.ORDER_SHA)
    result=dict(checkpoint_refs=cp,pair_commit_ref=refs['pair_commit'],order_ref=refs['order'],paired_checkpoint_prefix=deepcopy(core.FINAL_CURSOR))
    q=dict(schema=core.QUALIFIED_SCHEMA,qualified=True,paired_prefix_advanced=True,seed=core.SEED,
        span=dict(start=36864,stop=37050),result=result,source=sources['entry_source'],body_source=sources['body_source'],
        runtime_source=sources['runtime_source'],core_source=sources['core_source'],module_sources=component,context=ref('FIT_context'))
    pair=dict(schema='U40-paired-checkpoint-commit-v1',status='PAIRED_CHECKPOINTS_WRITTEN_PENDING_ROOT_ACCEPTANCE',
        cursor=deepcopy(core.FINAL_CURSOR),span=q['span'],identity=identity,order_ref=refs['order'],
        checkpoints={arm:dict(reference=cp[arm]) for arm in core.ARMS},traces={arm:ref(arm+'trace') for arm in core.ARMS})
    prep=dict(schema='ROOT-U40-continuous-training-preparation-v1',seed=core.SEED,images=741,GT=1661,epochs=50,total_updates_per_arm=37050,
        executable_sources=sources,module_sources=component,recipe=identity['recipe'],context=q['context'],order_sha256=core.ORDER_SHA)
    metadata=dict(refs=refs,training_qualified=q,pair_commit=pair,preparation=prep,order=dict(seed=core.SEED,order_sha256=core.ORDER_SHA))
    restore=dict(prediction_core=ref('prediction_core'),training_state=sources['core_source'],u39_training_ops=sources['u39_training_ops'],u36_helpers=sources['u36_helpers'])
    restore['prediction_core']['sha256']='819a2350487394265980cb77f8ff7e89547dbcac8e0d3e3c175a67a2794c9680'
    return metadata,restore


class Metadata(unittest.TestCase):
    def test_only_original_complete_fixed_endpoint_is_accepted(self):
        core=load(PUBLIC/'prediction_core.py','U44_PUBLIC_CORE_METADATA_ONLY')
        api=load(Path(__file__).with_name('restore_state.py'),'U44_RESTORE_METADATA_ONLY')
        metadata,sources=fixture(core)
        endpoint=api.validate_restore_request(core,metadata,'FULL',sources)
        self.assertEqual(endpoint['checkpoint_refs'],metadata['training_qualified']['result']['checkpoint_refs'])
        bad=deepcopy(metadata);bad['training_qualified']['result']['paired_checkpoint_prefix']['global_updates']=256
        with self.assertRaises(ValueError):api.validate_restore_request(core,bad,'FULL',sources)

    def test_restore_helpers_cannot_be_substituted_or_crossed_with_another_training_source(self):
        core=load(PUBLIC/'prediction_core.py','U44_PUBLIC_CORE_SOURCE_METADATA_ONLY')
        api=load(Path(__file__).with_name('restore_state.py'),'U44_RESTORE_SOURCE_METADATA_ONLY')
        metadata,sources=fixture(core)
        for role in ('prediction_core','training_state','u39_training_ops','u36_helpers'):
            changed=deepcopy(sources);changed[role]=ref('different_'+role)
            with self.subTest(role=role),self.assertRaises(ValueError):api.validate_restore_request(core,metadata,'PLAIN',changed)
        with self.assertRaises(ValueError):api.validate_restore_request(core,metadata,'SELECTIVE',sources)
        bad=deepcopy(metadata);bad['training_qualified']['paired_prefix_advanced']=False
        with self.assertRaises(ValueError):api.validate_restore_request(core,bad,'FULL',sources)


if __name__=='__main__':unittest.main()
