"""SYNTHETIC_ONLY stdlib scope/origin-binding test; no science refs/files."""
import copy
import importlib.util
from pathlib import Path

p=Path(__file__).with_name('N16_FIXED_E24_ALL100_SCOUT_SHARDS_FULLDEV_CPU_PROOF_ROOT_UNBOUND_DRAFT_20261014_01.py')
s=importlib.util.spec_from_file_location('ordinary_shard_cpu_proof',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def ref(name):return dict(path='E:/SYNTHETIC_ONLY/'+name,bytes=0,sha256='0'*64)


def evidence():
    result=[]
    for i,(first,last) in enumerate(m.RANGES):
        record=dict(shard_id=i,dataset_ordinal_range=[first,last],shard_owner_index=109+i,
            source=ref('SOURCE.py'),spec=ref('s%d.json'%i),report=ref('r%d.json'%i),RAW=ref('raw%d.json'%i),admission=ref('ad%d.json'%i))
        rows=[dict(dataset_index=n,image_id=1000+n,bundle=ref('b%d.pth'%n),TEST_ONLY=True) for n in range(first,last)]
        result.append(dict(descriptor=record,parent={},report={'capture_manifest':ref('manifest%d.json'%i)},
            manifest={'rows':rows},binding={'SYNTHETIC_ONLY':True,'shard_id':i,'OriginalTag':'GPU%d'%i}))
    return result


def rejects(f):
    try:f()
    except (RuntimeError,ValueError,KeyError):return
    raise AssertionError('Invalid fixed shard scope/binding accepted')


def main():
    e=evidence();rows=m.origin_manifest_rows(e)
    assert len(rows)==187 and [r['dataset_index'] for r in rows]==list(range(187))
    assert [sum(r['origin_shard_id']==i for r in rows) for i in range(4)]==[47,47,47,46]
    for row in rows:
        assert row['original_GPU_binding']==e[row['origin_shard_id']]['binding']
        assert row['original_GPU_binding'].get('CPU_admission') is None
    rejects(lambda:m.origin_manifest_rows(e[:-1]))
    bad=copy.deepcopy(e);bad[1]['manifest']['rows'][0]['dataset_index']=46
    rejects(lambda:m.origin_manifest_rows(bad))
    bad=copy.deepcopy(e);bad[3]['descriptor']['shard_owner_index']=113
    rejects(lambda:m.origin_manifest_rows(bad))
    bad=copy.deepcopy(e);bad[0]['descriptor']['dataset_ordinal_range'][0]=False
    rejects(lambda:m.origin_manifest_rows(bad))
    bad=copy.deepcopy(e);bad[0]['binding']['shard_id']=3
    rejects(lambda:m.origin_manifest_rows(bad))
    print('SYNTHETIC_ONLY full187 fixed47/47/47/46 + originalGPUbindings +5 scope rejects PASS; no actual qualification')


if __name__=='__main__':main()
