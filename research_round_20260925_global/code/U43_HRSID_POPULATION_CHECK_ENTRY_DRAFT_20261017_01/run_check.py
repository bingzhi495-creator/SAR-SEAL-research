"""UNBOUND U43 CPU input-check leaf. No model, Torch, retry, binder or launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

SPEC_SCHEMA='ROOT-U43-HRSID-population-pixel-mask-check-spec-v1'
OPERATION='CPU_HRSID_FULL_PIXEL_MASK_INPUT_CHECK'
CONTEXT_SCHEMA='U43-HRSID-official-split-pixel-mask-check-context-v1'
PRELOCK_SCHEMA='U43-HRSID-population-pixel-mask-check-prelock-v1'
RUNTIME_SHA256='9b17a7bdd9c44fe159583f4c82d38f1b42970662507986654900767e8ba5f383'
CHECKER_SHA256='cb247465226a77afc1d8c9bba00ad5a46ca1a10ffebf4931ba4a17a1c09ea69f'
CODE_ROLES=('entry_source','body_source','population_source','checker_source','runtime_source')
LIBRARY_ROLES=('PIL','PIL.Image','PIL.ImageFile','PIL.JpegImagePlugin','PIL._imaging',
               'numpy','numpy.core._multiarray_umath','pycocotools','pycocotools.mask',
               'pycocotools._mask','psutil','psutil._psutil_windows')


def need(ok,message):
    if not ok:
        raise RuntimeError(message)


def no_torch():
    need(not any(name=='torch' or name.startswith('torch.') for name in sys.modules),
         'Torch must not be imported by this CPU image/mask-only leaf')


def strict_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'}
         and type(value['path']) is str and Path(value['path']).is_absolute()
         and type(value['bytes']) is int and value['bytes']>=0
         and type(value['sha256']) is str and len(value['sha256'])==64
         and set(value['sha256'])<=set('0123456789abcdef'),'Exact strict reference')


def path_key(path):
    return str(path).replace('\\','/').casefold()


def ref(path):
    path=Path(path); digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def load(path,name):
    path=Path(path); module=types.ModuleType(name); module.__file__=str(path)
    sys.modules[name]=module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def validate_scope(spec):
    counts=('owner_index','normal_RAW_count','total_failed_RAW_count',
            'total_terminal_identity_pin_count','legacy_pin_count')
    need(spec['schema']==SPEC_SCHEMA and spec['operation']==OPERATION
         and all(type(spec[k]) is int and spec[k]>=0 for k in counts)
         and spec['owner_index']==spec['normal_RAW_count']
         and spec['total_terminal_identity_pin_count']==spec['normal_RAW_count']+spec['total_failed_RAW_count']
         and type(spec['deadline_seconds']) is int and spec['deadline_seconds']==1800
         and spec['automatic_retry'] is False and spec['cuda_visible_devices']==''
         and spec['allowed_cpus']==[0,1] and spec['thread_limit']==2
         and spec['maximum_process_RSS_bytes']==spec['minimum_available_RAM_bytes']==16*1024**3
         and spec['whole_GPU_limit_MiB']==8192 and spec['allocator_limit_MiB']==4096
         and spec['expected_images']==5604 and spec['expected_annotations']==16969
         and spec['splits']==['train','test']
         and all(spec[k] is True for k in ('no_new_training','no_AP','no_model','no_prediction','no_DEV'))
         and spec['TEST_annotations_read'] is True and spec['TEST_used_for_model_selection'] is False,
         'Exact complete HRSID train/test input-only scope and CPU resource policy')
    need(type(spec['library_sources']) is dict and set(spec['library_sources'])==set(LIBRARY_ROLES),
         'Explicit selected runtime module/source footprint; no whole-environment claim')
    for value in spec['library_sources'].values():
        strict_ref(value)


def validate_sources(spec,context):
    """Metadata only: exact direct closure, never open an image/annotation here."""
    need(type(context) is dict and set(context)=={'schema','image_root','annotations','image_refs_by_filename'}
         and context['schema']==CONTEXT_SCHEMA and type(context['image_root']) is str
         and Path(context['image_root']).is_absolute()
         and type(context['annotations']) is dict and set(context['annotations'])=={'train','test'}
         and type(context['image_refs_by_filename']) is dict and len(context['image_refs_by_filename'])==5604,
         'Explicit two original splits and all 5604 admitted JPEG refs')
    required=[spec[k] for k in CODE_ROLES+('context','prelock','plan','parent_ROOT_qualified')]
    required+=list(spec['library_sources'].values())+list(context['annotations'].values())
    image_keys=[]
    for name,value in context['image_refs_by_filename'].items():
        strict_ref(value)
        need(type(name) is str and name==Path(name).name and '/' not in name and '\\' not in name
             and Path(name).suffix.lower() in ('.jpg','.jpeg')
             and path_key(Path(value['path']))==path_key(Path(context['image_root'])/name),
             'Canonical original image_root/file_name exact reference')
        image_keys.append(path_key(value['path'])); required.append(value)
    need(len(set(image_keys))==5604,'No duplicate original JPEG paths')
    expected={}
    for value in required:
        strict_ref(value); key=path_key(value['path'])
        need(key not in expected or expected[key]==value,'Conflicting direct refs')
        expected[key]=value
    need(type(spec['sources']) is list,'Explicit source list')
    supplied={}
    for value in spec['sources']:
        strict_ref(value); key=path_key(value['path'])
        need(key not in supplied,'Duplicate source path')
        supplied[key]=value
    need(supplied==expected,'Complete exact direct closure; no unrelated cache, model or weight inputs')
    need(len({path_key(v['path']) for v in context['annotations'].values()})==2,
         'Distinct original train/test JSON refs, no merged substitute')
    return supplied


def validate_prelock(pre,spec):
    need(pre['schema']==PRELOCK_SCHEMA and pre['context']==spec['context'] and pre['plan']==spec['plan']
         and pre['splits']==['train','test']
         and pre['split_counts']=={'train':{'images':3642,'annotations':11047},
                                  'test':{'images':1962,'annotations':5922}}
         and pre['expected_images']==5604 and pre['expected_annotations']==16969
         and pre['original_canvas']==[800,800] and pre['automatic_retry'] is False
         and pre['automatic_sharding'] is False and pre['population_filter'] is None
         and all(pre[k] is True for k in ('no_new_training','no_AP','no_model','no_prediction','no_DEV'))
         and pre['TEST_annotations_read'] is True and pre['TEST_used_for_model_selection'] is False,
         'Complete original train/test population; no repair, filtering, model or AP')


def run(path):
    no_torch(); started=time.monotonic(); path=Path(path)
    need(path.is_absolute(),'Absolute Root-owned spec')
    spec=json.loads(path.read_bytes()); validate_scope(spec)
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only original environment')
    need(ref(__file__)==spec['entry_source']
         and spec['runtime_source']['sha256']==RUNTIME_SHA256
         and spec['checker_source']['sha256']==CHECKER_SHA256,
         'Actual entry and original approved runtime/U42 source pins')
    spec_ref=ref(path); admission_path=Path(spec['admission_file'])
    need(admission_path.is_absolute(),'Absolute original admission')
    admission=json.loads(admission_path.read_bytes())
    keys={'status','phase','driver_sha256','entry','entry_args','allowed_cpus','thread_limit',
          'cuda_visible_devices','whole_GPU_used_limit_MiB','qualified_leaf_deadline_seconds',
          'automatic_retry','scientific_spec','bound_source_sha256'}
    need(set(admission)==keys and len(admission)==13
         and admission['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE'
         and admission['scientific_spec']==spec_ref and admission['phase']==spec['phase']
         and admission['entry']==spec['entry_source']['path']
         and admission['entry_args']==['--spec',str(path),'--owner-execute']
         and admission['allowed_cpus']==[0,1] and admission['thread_limit']==2
         and admission['cuda_visible_devices']=='' and admission['automatic_retry'] is False
         and admission['whole_GPU_used_limit_MiB']==8192
         and admission['qualified_leaf_deadline_seconds']==1800
         and admission['bound_source_sha256']=={v['path']:v['sha256'] for v in spec['sources']+[spec_ref]},
         'Original thirteen-field admission/source bindings')
    strict_ref(spec['context'])
    need(spec['context'] in spec['sources'] and ref(spec['context']['path'])==spec['context'],
         'Context itself is admitted and byte-exact before metadata parsing')
    context=json.loads(Path(spec['context']['path']).read_bytes())
    validate_sources(spec,context)
    need(ref(spec['runtime_source']['path'])==spec['runtime_source'],'Original runtime bytes')
    runtime=load(spec['runtime_source']['path'],'U43_CPU_RESOURCE_CONTROL')
    no_torch()
    c=runtime.Control(spec)
    c.started=started  # Include entry metadata/source work in the original 1800s check.
    c.execution_spec_ref,c.execution_admission_ref=spec_ref,ref(admission_path)
    c.check()
    need(len(c.allowed)==len(spec['sources']),'Unique resolved direct input paths')
    for value in spec['sources']: c.closed(value)
    parent=json.loads(Path(c.closed(spec['parent_ROOT_qualified'])).read_bytes())
    need(parent['qualified'] is True and all(parent[k]==spec[k] for k in
         ('normal_RAW_count','total_failed_RAW_count','total_terminal_identity_pin_count','legacy_pin_count')),
         'Current Root-accepted account parent; independent of image population')
    pre=json.loads(Path(c.closed(spec['prelock'])).read_bytes()); validate_prelock(pre,spec)
    output=Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists()
         and not output.is_symlink() and c.psutil.disk_usage(str(output.parent)).free>=1024**3,
         'Fresh exclusive output and 1GiB metadata/log headroom')
    no_torch()
    body=load(c.closed(spec['body_source']),'U43_CPU_INPUT_CHECK_BODY')
    result=body.run(spec,c,pre,context)
    no_torch(); c.check()
    need(result['images']==5604 and result['annotations']==16969 and result['input_check_complete'] is True
         and result['TEST_annotations_read'] is True and result['TEST_used_for_model_selection'] is False
         and all(result[k] is False for k in ('qualified','Torch_imported','CUDA_initialized','new_training','AP',
                                             'semantic_alignment_proven')),
         'Actual complete input check only; never self-qualified or model/semantic evidence')
    for key,name in (('report_ref','REPORT.json'),('log_ref','probe.log'),
                     ('image_records_ref','IMAGE_CHECKS.jsonl'),('safe_summary_ref','SAFE_SUMMARY.json')):
        value=result[key]; strict_ref(value)
        need(Path(value['path']).resolve()==(output/name).resolve() and not Path(value['path']).is_symlink()
             and c.ref(value['path'])==value,'Actual complete owned artifact '+name)
    report=json.loads(Path(result['report_ref']['path']).read_bytes())
    need(report['schema']=='U43-HRSID-population-CPU-wrapper-report-pending-v1'
         and report['status']=='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE'
         and report['population']['status']=='INPUT_CHECK_COMPLETE_PENDING_ROOT'
         and report['image_records_written']==5604 and report['Torch_imported'] is False
         and report['source_aliases_restored_exact'] is True,
         'Full worker report matches original body completion, without numerical replay')
    for value in [spec[k] for k in CODE_ROLES]+list(spec['library_sources'].values()):
        need(c.ref(value['path'])==value,'Selected executed source unchanged')
    q=dict(schema='U43-owned-pending-HRSID-population-input-check-v1',
        status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',operation=OPERATION,
        phase=spec['phase'],source=spec['entry_source'],body_source=spec['body_source'],
        population_source=spec['population_source'],checker_source=spec['checker_source'],
        runtime_source=spec['runtime_source'],library_sources=spec['library_sources'],
        spec=spec_ref,admission=c.execution_admission_ref,context=spec['context'],prelock=spec['prelock'],
        plan=spec['plan'],parent_ROOT_qualified=spec['parent_ROOT_qualified'],result=result,
        owner_index=spec['owner_index'],normal_RAW_count=spec['normal_RAW_count'],
        total_failed_RAW_count=spec['total_failed_RAW_count'],
        total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],legacy_pin_count=spec['legacy_pin_count'],
        new_training=False,AP=False,DEV=False,TEST=True,TEST_annotations_read=True,
        TEST_used_for_model_selection=False,Torch_imported=False,CUDA_initialized=False,
        model_forward_executed=False,optimizer_created=False,semantic_alignment_proven=False,
        qualified=False,original_actual0_self_certified=False,innovation_established=False,goal_complete=False,
        elapsed_seconds=time.monotonic()-c.started,peak_RSS_bytes=c.peak_rss,
        minimum_available_RAM_bytes=c.minimum_available,cleanup=dict(Torch_imported=False,CUDA_initialized=False))
    qref=c.write_json(output/'Q.json',q)
    print(json.dumps(dict(status=q['status'],Q=qref,qualified=False,input_check_only=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args()
    need(args.owner_execute,'Only the original Root-owned CPU leaf may execute')
    run(args.spec)
