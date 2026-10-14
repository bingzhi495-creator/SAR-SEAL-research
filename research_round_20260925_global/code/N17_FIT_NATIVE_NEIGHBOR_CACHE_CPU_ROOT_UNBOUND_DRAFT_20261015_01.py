"""UNBOUND N17 FIT-only neighbor cache CPU entry; actual execution remains ROOT-owned."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import types

LEGACY_SCHEMA = 'ROOT-N16-fixed-E24-all100-scout-CPU-numeric-shards-aggregate-spec-v1'
N17_SCHEMA = 'ROOT-N17-native-eight-neighbor-FIT-cache-spec-v1'
THREAD_ENV = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def validate_spec(spec):
    require(spec['schema']==LEGACY_SCHEMA and spec['N17_schema']==N17_SCHEMA,
        'Explicit N17 schema with unchanged parent-gate ABI required')
    for key, expected in [('normal_RAW_count',119),('expected_normal_RAW_count',119),
            ('total_failed_RAW_count',17),('total_terminal_identity_pin_count',136),
            ('deadline_seconds',1800),('allocator_limit_MiB',4096),('whole_GPU_limit_MiB',8192)]:
        require(type(spec[key]) is int and spec[key]==expected,'N17 own119/resource count differs: '+key)
    require(spec['automatic_retry'] is False and type(spec['RAW_inputs']) is list and
        len(spec['RAW_inputs'])==119 and type(spec['terminal_identity_pins']) is dict and
        len(spec['terminal_identity_pins'])==127,'Complete normal119 plus original legacy failed8 pins')
    scope=spec['N17_cache_scope']
    require(type(scope) is dict and scope['split']=='FIT' and
        scope['images']==741 and scope['natural_queries']==74100 and
        scope['sampled_points']==3035136 and scope['payload_bytes']==97124352 and
        scope['no_DEV_inputs'] is True and scope['no_teacher_forward'] is True and
        scope['no_new_training'] is True,'N17 cache must cover original full FIT only')

def read(path):
    def reject(value):raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)

def reference(path):
    path=Path(path).resolve(strict=True);h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())

def strict_ref(ref):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and
        type(ref['path']) is str and Path(ref['path']).is_absolute() and
        type(ref['bytes']) is int and ref['bytes']>=0 and type(ref['sha256']) is str and
        len(ref['sha256'])==64 and set(ref['sha256'])<=set('0123456789abcdef'),
        'Malformed strict three-field reference')

def load(ref,name):
    strict_ref(ref);require(reference(ref['path'])==ref,'Pinned module changed')
    m=types.ModuleType(name);m.__file__=ref['path']
    exec(compile(Path(ref['path']).read_bytes(),ref['path'],'exec'),m.__dict__);return m

def write_json(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def source_gate(path,spec):
    validate_spec(spec)
    require(Path(__file__).resolve(strict=True)==Path(spec['entry_source']['path']).resolve(strict=True),
        'N17 own entry must match actual executed source')
    # The independently frozen Root prefix checks all119 terminal owners and
    # retained17 failures before this entry hashes/imports a scientific source.
    prefix=spec['pure_RAW_prefix'];strict_ref(prefix)
    code=Path(prefix['path']).read_text(encoding='utf-8-sig')
    exec(compile(code,prefix['path'],'exec'),{})
    require(reference(prefix['path'])==prefix,'Original pure RAW prefix changed')
    initial_closure={ref['path']:ref for ref in spec['sources']}
    require(len(initial_closure)==len(spec['sources']) and
        initial_closure.get(spec['parent_metadata_gate_source']['path'])==spec['parent_metadata_gate_source'] and
        initial_closure.get(prefix['path'])==prefix,'Parent metadata module and RAW prefix must be exact Source members')
    legacy=load(spec['parent_metadata_gate_source'],'_N17_closed_parent_metadata_gate')
    gate=legacy.inherited_source_gate(path,spec,cpu=True)
    closure=gate['closure']
    for key in ('parent_metadata_gate_source','pure_RAW_prefix','neighbor_experiment_lock',
            'neighbor_kernel_source','neighbor_cache_body_source','N17_initialization_metadata',
            'ROOTcache_result','cache_GPU_report','cache_CPU_qualification',
            'N16_completed_ROOT_result','N16_completed_CPU_qualification'):
        legacy.closed(closure,spec[key])
    lock=read(legacy.closed(closure,spec['neighbor_experiment_lock']))
    require(lock['schema']=='N17-local-native-logit-error-paired-experiment-locked-v1' and
        lock['locked_before_actual_neighbor_cache_and_training'] is True and
        lock['parent_protocol']==spec['locked_protocol'] and
        lock['qualified_parent_cache']==spec['ROOTcache_result'] and
        lock['qualified_previous_result']==spec['N16_completed_ROOT_result'] and
        lock['previous_N16_decision']=='STOP' and
        lock['population']==dict(FIT_images=741,FIT_instances=1661,FIT_natural_queries=74100,
            FIT_sampled_points=3035136,DEV_images=187,DEV_GT=380,DEV_natural_queries=18700,official_TEST=False) and
        lock['architecture']['parameters']==16803 and lock['architecture']['first_seed']==202610150171 and
        lock['architecture']['repeat_seeds']==[202610150172,202610150173] and
        lock['architecture']['initialization_metadata']==spec['N17_initialization_metadata'] and
        lock['architecture']['neighbor_module_SHA256']==spec['neighbor_kernel_source']['sha256'] and
        lock['inputs']['offsets']==[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]] and
        lock['inputs']['border']=='axis-wise replicate' and lock['cache']['payload_bytes']==97124352,
        'N17 independently locked method, inputs and seeds differ')
    require(lock==spec['expected_neighbor_experiment_lock'],'Complete new experiment lock must match')
    init=read(legacy.closed(closure,spec['N17_initialization_metadata']))
    require(init['synthetic_parameters_only'] is True and init['training_executed'] is False and
        init['old_factory_SHA256']==spec['scouter_source']['sha256'] and
        init['lift_module_SHA256']==spec['neighbor_kernel_source']['sha256'] and
        {str(r['seed']):r['combined_digest'] for r in init['rows']}==
            lock['architecture']['initialization_digest_by_seed'] and
        all(r['paired_state_exact'] is True and r['paired_storage_independent'] is True and
            r['new_columns_zero'] is True for r in init['rows']), 'Prebound initialization identity differs')
    root=read(legacy.closed(closure,spec['ROOTcache_result']))
    report=read(legacy.closed(closure,spec['cache_GPU_report']))
    q=read(legacy.closed(closure,spec['cache_CPU_qualification']))
    require(root['status']=='ROOT_N16_FULLFIT_ALL100_ACTION_CACHE_FULLY_QUALIFIED' and
        root['fully_qualified'] is True and root['GPU_actual_exit_code']==root['CPU_actual_exit_code']==0 and
        root['GPU_report']==spec['cache_GPU_report'] and root['CPU_Q']==spec['cache_CPU_qualification'] and
        root['GPU_RAW']==spec['cache_GPU_RAW'] and root['CPU_RAW']==spec['cache_CPU_RAW'] and
        root['locked_protocol']==spec['locked_protocol'] and
        [root[k] for k in ('images','raw_GT','natural_queries','sampled_points')]==[741,1661,74100,3035136] and
        q['numerical_semantic_checks_complete'] is True and q['cache_GPU_report']==spec['cache_GPU_report'] and
        [q[k] for k in ('images','raw_GT','natural_queries','sampled_points')]==[741,1661,74100,3035136],
        'Only completed qualified original FIT cache may supply new neighbors')
    require(report['source']==spec['cache_gpu_source'] and report['spec']==spec['cache_gpu_spec'] and
        report['GPU_admission']==spec['cache_GPU_admission_refs'] and report['phase']==spec['cachephase'] and
        len(report['rows'])==len(report['packet_refs'])==len(report['native_L_refs'])==741 and
        [report[k] for k in ('images','raw_GT','natural_queries','sampled_points')]==[741,1661,74100,3035136] and
        report['original_native610_unchanged'] is True and report['original_PR8_unchanged'] is True,
        'Exact full741 original GPU FIT cache identity differs')
    gi=gate['lineage'].owner_ref_join(gate['context'],gate['owners'],spec['cache_GPU_RAW'],
        spec['cache_GPU_admission_refs'],spec['cache_gpu_source'],spec['cache_gpu_spec'],'0')
    ci=gate['lineage'].owner_ref_join(gate['context'],gate['owners'],spec['cache_CPU_RAW'],
        spec['cache_CPU_admission'],spec['cache_CPU_source'],spec['cache_CPU_spec'],'')
    require(gi==105 and ci==106,'Original FIT GPU105/CPU106 owners required')
    completed=read(legacy.closed(closure,spec['N16_completed_ROOT_result']))
    require(completed['qualified'] is True and completed['final_CPU_qualification']==spec['N16_completed_CPU_qualification'] and
        completed['normal_RAW_count_after_final']==119 and completed['failed_RAW_count']==17,
        'Completed N16 evidence, including preserved failure17, required')
    for ref in report['packet_refs']+report['native_L_refs']:legacy.closed(closure,ref)
    gate.update(legacy=legacy,new_lock=lock,FIT_cache_report=report)
    return gate

def run(spec_path):
    started=time.monotonic();path=Path(spec_path).resolve(strict=True)
    require(Path(spec_path).is_absolute(),'Absolute N17 spec required')
    spec=read(path);gate=source_gate(path,spec);legacy=gate['legacy'];closure=gate['closure']
    output=Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and output.parent.is_dir(),'Fresh independent N17 output required')
    import psutil
    process=psutil.Process()
    require(process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
        all(os.environ.get(k)=='2' for k in THREAD_ENV),'CPUempty/affinity01/two threads required')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(),'Cold qualified CPU Torch required')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    import numpy as np,cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    require(np.__version__=='1.23.5' and cv2.__version__=='4.8.0','Qualified numeric runtime required')
    cv2.setNumThreads(2)
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,closure,gate['admission'],ndimage)
    outside=torch.get_rng_state().clone()
    def check():
        require(time.monotonic()-started<1800 and process.memory_info().rss<16*1024**3 and
            not torch.cuda.is_initialized() and process.cpu_affinity()==[0,1] and
            os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in THREAD_ENV) and
            torch.get_num_threads()==torch.get_num_interop_threads()==2 and torch.equal(outside,torch.get_rng_state()),
            'Bounded CPU119 resources/RNG/threads/noCUDA required')
    legacy.closed(closure,spec['neighbor_kernel_source'])
    kernel=legacy.load(spec['neighbor_kernel_source'],'_N17_bound_neighbor_kernel')
    legacy.closed(closure,spec['neighbor_cache_body_source'])
    body=legacy.load(spec['neighbor_cache_body_source'],'_N17_bound_FIT_cache_body')
    check();output.mkdir()
    def write_npy(index,array):
        require(type(index) is int and 0<=index<741 and type(array) is np.ndarray and
            array.shape==(4096,8) and array.dtype==np.float32,'Fixed FIT-only neighbor output required')
        target=output/('NEIGHBORS_%04d.npy'%index)
        with target.open('xb') as f:np.save(f,array,allow_pickle=False)
        return reference(target)
    def read_generated(ref):
        strict_ref(ref);require(Path(ref['path']).parent.resolve()==output.resolve() and reference(ref['path'])==ref,
            'Read only this fresh generated neighbor output')
        return np.load(ref['path'],allow_pickle=False)
    def write_manifest(value):
        target=output/'MANIFEST.json';write_json(target,value);return reference(target)
    report=gate['FIT_cache_report']
    callbacks=dict(closed=lambda ref:legacy.closed(closure,ref),
        read_torch=lambda p:torch.load(str(p),map_location='cpu'),write_npy=write_npy,
        read_generated_npy=read_generated,write_manifest=write_manifest,check=check)
    binding=dict(source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],phase=spec['phase'],
        neighbor_experiment_lock=spec['neighbor_experiment_lock'],neighbor_kernel_source=spec['neighbor_kernel_source'],
        neighbor_cache_body_source=spec['neighbor_cache_body_source'],qualified_parent_cache=spec['ROOTcache_result'])
    manifest_ref,summary=body.extract_fit(torch,np,kernel,report,spec['cache_GPU_report'],
        dict(locked_protocol=spec['locked_protocol']),callbacks,output_binding=binding)
    require([summary[k] for k in ('images','queries','points','payload_bytes')]==[741,74100,3035136,97124352] and
        summary['qualified'] is False,'Complete FIT scope, never self-certified output')
    for ref in spec['sources']:check();legacy.bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and
        reference(spec['CPU_admission_file'])==gate['admission_ref'] and torch.equal(outside,torch.get_rng_state()) and
        gate['common'].imported_files(torch,np,cv2,masks,COCO,closure,gate['admission'],ndimage)==libraries,
        'Final Source/spec/admission/libraries/RNG unchanged')
    gate['lineage'].final_endpoint_check(gate['parent']);check()
    q=dict(schema='N17-fullFIT-native-eight-neighbor-cache-CPU-proof-v1',
        status='N17_FULLFIT_NEIGHBOR_CACHE_COMPLETE_PENDING_ROOT_ACTUAL0',**binding,
        manifest=manifest_ref,summary=summary,normal_RAW_count=119,total_failed_RAW_count=17,
        total_terminal_identity_pin_count=136,actual_imported_files=libraries,
        qualified=False,ROOT_actual_CPU0_verified=False,no_new_training=True,
        AP_evaluated=False,innovation_established=False,elapsed_seconds=time.monotonic()-started)
    write_json(output/'Q.json',q)
    print(json.dumps(dict(status=q['status'],Q=reference(output/'Q.json'),qualified=False)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true');args=parser.parse_args()
    require(args.owner_execute,'ROOT owner-execute required');run(args.spec)
