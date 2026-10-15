"""One synthetic metadata-only contract check; no scientific imports/RAW/files.

Only exact ordinary TMP source text is read and AST-extracted pure metadata
helpers execute on invented descriptors/predictions. No source_gate/run executes.
"""
from pathlib import Path
import ast
import copy
import json

root = Path('E:/SAR/seal/tmp').resolve(strict=True)
front_path = root/'N20_FIXEDDEV_FULL187_COCO_CPU_FRONT_UNBOUND_DRAFT_20261015_02.py'
tail_path = root/'N20_FULL187_CPU_COCO_FINAL_RUN_TAIL_UNBOUND_DRAFT_20261015_02.py'
post_path = root/'N20_FIXEDDEV_FULL187_POST_ACTUAL_ROOT_CPU_UNBOUND_DRAFT_20261015_02.py'
for path in (front_path,tail_path,post_path):
    assert path.resolve(strict=True).parent == root and not path.is_symlink()


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def ref_shape(value):
    assert type(value) is dict and set(value)=={'path','bytes','sha256'}


def extract(path, wanted):
    source = path.read_text(encoding='utf-8-sig'); tree = ast.parse(source)
    selected = []
    for node in tree.body:
        names = {node.name} if isinstance(node,ast.FunctionDef) else (
            {t.id for t in node.targets if isinstance(t,ast.Name)} if isinstance(node,ast.Assign) else set())
        if names & wanted:
            selected.append(node)
    namespace = {'Path':Path,'require':require,'ref_shape':ref_shape,'__file__':str(path)}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    assert wanted <= namespace.keys()
    return namespace


def rejected(callback):
    try:
        callback()
    except AssertionError:
        return
    raise AssertionError('Invalid old/partial metadata was accepted')


f = extract(front_path,{'N20_FINAL_CPU_SCHEMA','N20_FINAL_CPU_RANGES','N20_FINAL_GPU_RANGES',
    'N20_FINAL_PARENT_IDS','N20_FINAL_CPU_KEYS','N20_FINAL_GPU_KEYS','n20_validate_final_header'})
t = extract(tail_path,{'N20_FINAL_CPU_RANGES','n20_merge_policy_predictions'})
expected = tuple((s,min(s+6,last)) for first,last in ((0,47),(47,94),(94,141),(141,187))
    for s in range(first,last,6))
assert f['N20_FINAL_CPU_RANGES']==t['N20_FINAL_CPU_RANGES']==expected
assert f['N20_FINAL_PARENT_IDS']==tuple(p for p in range(4) for _ in range(8))
assert [i for first,last in expected for i in range(first,last)]==list(range(187))
assert [b-a for a,b in expected][7::8]==[5,5,5,4]


def invented_ref(label):
    return dict(path='E:/SAR/seal/tmp/SYNTHETIC_ONLY_'+label,bytes=1,sha256='0'*64)


def descriptor(index, gpu=False):
    values={key:invented_ref(('%s_%d_'%('GPU' if gpu else 'CPU',index))+key)
        for key in ('source','spec','report','RAW','admission','phase_binding','bootstrap')}
    if gpu:
        values.update(shard_id=index,dataset_ordinal_range=list(f['N20_FINAL_GPU_RANGES'][index]),shard_owner_index=141+index)
    else:
        values.update(cpu_shard_id=index,cpu_dataset_ordinal_range=list(expected[index]),cpu_shard_owner_index=145+index)
    return values


spec = dict(schema=f['N20_FINAL_CPU_SCHEMA'],normal_RAW_count=177,expected_normal_RAW_count=177,
    total_failed_RAW_count=26,total_terminal_identity_pin_count=203,initialization_seed=202610150201,
    deadline_seconds=1800,automatic_retry=False,maximum_process_RSS_bytes=16*1024**3,
    allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,failed26_partials_reused=False,
    entry_source={'path':str(front_path)},numeric_CPU_shards=[descriptor(i) for i in range(32)],
    completed_GPU_shards=[descriptor(i,True) for i in range(4)])
assert len(f['n20_validate_final_header'](spec))==32
bad=copy.deepcopy(spec);bad['numeric_CPU_shards']=bad['numeric_CPU_shards'][:8]
rejected(lambda:f['n20_validate_final_header'](bad))
bad=copy.deepcopy(spec);bad['failed26_partials_reused']=True
rejected(lambda:f['n20_validate_final_header'](bad))
bad=copy.deepcopy(spec);bad['normal_RAW_count']=bad['expected_normal_RAW_count']=153
rejected(lambda:f['n20_validate_final_header'](bad))

ids=list(range(187)); native=[dict(image_id=i,category_id=1,score=0.5,query_id=q,
    segmentation={'size':[1,1],'counts':'SYNTHETIC'}) for i in ids for q in range(100)]
pieces=[native[first*100:last*100] for first,last in expected]
assert t['n20_merge_policy_predictions'](pieces,ids,native,native)==native
rejected(lambda:t['n20_merge_policy_predictions'](pieces[:8],ids,native,native))
badpieces=copy.deepcopy(pieces);badpieces[0][0]['score']=0.6
rejected(lambda:t['n20_merge_policy_predictions'](badpieces,ids,native,native))

# Preserve pending lifecycle and frozen first201 decision; no model/COCO execution.
post=post_path.read_text(encoding='utf-8-sig')
assert "record['owner_index'] == 177" in post and "len(spec['RAW_inputs']) == 178" in post
assert 'normal_after=178,preserved_failures=26,pins_after=204' in post
assert 'qualified=False,ROOT_post_actual_CPU0_verified=False,current_post_owned_terminal_self_certified=False' in post
assert 'COCO_rerun=False' in post and 'n20_failed26_raw_only(spec)' in post
print(json.dumps(dict(status='SYNTHETIC_METADATA_ONLY_GREEN',CPU_scopes=32,GPU_parents=4,
    images=187,queries=18700,old8_rejected=True,old_final153_rejected=True,
    failed26_partial_reuse_rejected=True,nonseg_mutation_rejected=True,
    scientific_import_or_execution=False,actual_qualification=False)))
