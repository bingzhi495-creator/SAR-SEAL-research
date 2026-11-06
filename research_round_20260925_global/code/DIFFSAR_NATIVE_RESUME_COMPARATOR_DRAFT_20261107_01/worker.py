"""Unexecuted CPU resume comparator draft; module import uses stdlib only."""
import argparse,datetime,gc,hashlib,importlib,json,math,os,platform,sys,time,traceback,types
from pathlib import Path,PurePosixPath

SCHEMA='Root-DiffSAR-native-resume-compare-spec-v1'
LEASE_SCHEMA='Root-DiffSAR-native-resume-compare-lease-v1'
OPERATION='CPU_DIFFSAR_NATIVE_RESUME_C8_S8_COMPARISON_V01'
LIBRARY_MODULES={'torch':'torch','numpy':'numpy'}
LIBRARY_NAMES=tuple(LIBRARY_MODULES)
PRELOCK=dict(continuous=[0,8],split=[4,8],total_actual_training_updates=12,
    normalized_maxdiff_limit=1e-4,normalization='absmax(C-S)/(1+absmax(C))',
    exact_restored_start=True,exact_input_trace=True,exact_end_RNG_loader_scheduler=True,
    floating_final_components=['model','gradients','optimizer'],integer_bool_static_exact=True,
    theoretical_roundoff_bound=False,bitexact_continuation_claim=False,
    CPU_affinity=[0,1],threads=2,peak_group_RSS_cap_bytes=16*1024**3,
    free_RAM_floor_bytes=16*1024**3,start_RAM_bytes=48*1024**3,
    external_deadline_seconds=1800,single_attempt=True,automatic_retry=False,
    GPU_hidden=True,checkpoint_write=False,AP=False,formal_training_credit=0)

def need(c,m):
    if not c:raise ValueError(m)

def valid_ref(r):
    need(type(r) is dict and set(r)=={'path','bytes','sha256'},'Exact byte reference')
    p=r['path'];need(type(p) is str and PurePosixPath(p).is_absolute() and str(PurePosixPath(p))==p and '..' not in PurePosixPath(p).parts,'Canonical reference')
    need(type(r['bytes']) is int and r['bytes']>0 and type(r['sha256']) is str and len(r['sha256'])==64 and all(c in '0123456789abcdef' for c in r['sha256']),'Byte identity')

def reference(path):
    p=Path(path);need(str(p)==str(p.resolve(strict=True)) and p.is_file(),'Canonical regular input')
    a=p.stat();h=hashlib.sha256();n=0
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b);n+=len(b)
    z=p.stat();need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns),'Stable input bytes')
    return dict(path=str(p),bytes=n,sha256=h.hexdigest())

def verify(r):valid_ref(r);need(reference(r['path'])==r,'Pinned input mismatch')

def spec_refs(s):
    return list(s['source_refs'].values())+list(s['library_sources'].values())+list(s['checkpoint_refs'].values())+list(s['training_report_refs'].values())+[s['root_lease_ref']]

def validate_spec(s):
    need(type(s) is dict and set(s)=={'schema','operation','run_id','source_refs','library_sources','checkpoint_refs','training_report_refs','root_lease_ref','output_directory','prelock'},'Exact CPU comparison spec')
    need(s['schema']==SCHEMA and s['operation']==OPERATION and s['prelock']==PRELOCK,'Fixed CPU comparison operation')
    need(type(s['run_id']) is str and s['run_id'].startswith('DiffSAR-SSDD-resume-compare-v01-'),'Comparison run identity')
    need(set(s['source_refs'])=={'worker','training_worker'} and set(s['library_sources'])==set(LIBRARY_NAMES),'Own selected source identities')
    need(set(s['checkpoint_refs'])=={'C4','C8','S8'} and set(s['training_report_refs'])=={'C','S'},'Fixed real checkpoint/report roles')
    refs=spec_refs(s)
    for r in refs:valid_ref(r)
    need(len({r['path'] for r in refs})==len(refs),'Distinct selected inputs')
    out=s['output_directory'];need(type(out) is str and PurePosixPath(out).is_absolute() and str(PurePosixPath(out))==out and out!='/' and '..' not in PurePosixPath(out).parts,'Canonical fresh output')
    need(all(not r['path'].startswith(out+'/') and r['path']!=out for r in refs),'Outputs outside immutable inputs')

def validate_lease(s,l):
    need(l.get('schema')==LEASE_SCHEMA and l.get('status')=='ROOT_GRANTED','Explicit CPU Root lease')
    for k in ('operation','run_id','output_directory','prelock','source_refs','checkpoint_refs','training_report_refs'):need(l.get(k)==s[k],'Lease '+k)
    need(l.get('worker_ref')==s['source_refs']['worker'],'Lease worker identity')
    need(l.get('single_attempt') is l.get('main_scientific_processes_reaped') is l.get('exclusive_project_scientific_leaf') is True and l.get('whole_device_exclusive') is False,'Single scientific leaf')

def checkpoint_entry(r,n):
    found=[x for x in r['checkpoints'] if x['updates']==n];need(len(found)==1,'Unique actual checkpoint')
    return found[0]
def validate_pair(c,s,refs):
    if s.get('resume_start_exact') is not True:raise ValueError('Exact restored start required')
    for r,added in ((c,8),(s,4)):
        if r.get('all_requested_updates_completed') is not True or type(r.get('worker_exit_code')) is not int or r['worker_exit_code']!=0:
            raise ValueError('Both real training workers must complete')
        if type(r.get('optimizer_steps_completed')) is not int or r['optimizer_steps_completed']!=8 or r.get('new_optimizer_updates')!=added:
            raise ValueError('Actual completed update counts')
    if [x['global_update'] for x in c['steps']]!=list(range(1,9)) or [x['global_update'] for x in s['steps']]!=list(range(5,9)):
        raise ValueError('Only continuous0to8 and same4to8')
    for r,n,k in ((c,4,'C4'),(c,8,'C8'),(s,8,'S8')):
        if checkpoint_entry(r,n)['ref']!=refs[k]:raise ValueError('Exact checkpoint reference join')
    if s.get('resume_start_component_digests')!=checkpoint_entry(c,4)['component_digests']:
        raise ValueError('Same complete restored-start digests')
    keys=('global_update','dataset_index','image_id','mapped_image_shape','mapped_instances',
          'augmentation_and_target_sha256','used_learning_rates','post_scheduler_learning_rates')
    a=[x for x in c['steps'] if x['global_update']>=5];b=s['steps']
    if len(a)!=4 or len(b)!=4:raise ValueError('Four actual continuation steps')
    for x,y in zip(a,b):
        if any(x[k]!=y[k] for k in keys):raise ValueError('Actual input/order/LR differs')
    return True

def scalar_difference(a,b):
    need(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b),'Finite recorded scalar')
    d=abs(a-b);need(math.isfinite(d),'Finite scalar difference')
    return dict(max_abs_difference=d,normalized_maxdiff=d/(1+abs(a)),passed=d/(1+abs(a))<=PRELOCK['normalized_maxdiff_limit'])

def compare_tree(t,a,b,path,result,check):
    need(type(a) is type(b),'Typed structure mismatch '+path)
    if t.is_tensor(a):
        need(a.device.type==b.device.type=='cpu' and a.dtype==b.dtype and a.shape==b.shape,'Tensor type/shape '+path)
        need(not a.is_complex(),'No complex tensor in native checkpoint')
        floating=a.is_floating_point();exact=not floating or path.startswith('optimizer/') and path.endswith('/step')
        aa=a.reshape(-1);bb=b.reshape(-1);different=0;maximum=0.0;refmax=0.0
        for lo in range(0,a.numel(),1048576):
            x=aa[lo:lo+1048576];y=bb[lo:lo+1048576]
            different+=int(t.count_nonzero(x!=y).item())
            if floating:
                need(bool(t.isfinite(x).all().item()) and bool(t.isfinite(y).all().item()),'Finite tensors '+path)
                xd=x.to(dtype=t.float64);yd=y.to(dtype=t.float64)
                maximum=max(maximum,float((xd-yd).abs().max().item()));refmax=max(refmax,float(xd.abs().max().item()))
            check()
        need(math.isfinite(maximum) and math.isfinite(refmax),'Finite measured tensor differences')
        norm=maximum/(1+refmax);passed=(different==0 if exact else norm<=PRELOCK['normalized_maxdiff_limit'])
        row=dict(path=path,dtype=str(a.dtype),shape=list(a.shape),elements=a.numel(),different_elements=different,
            exact_required=exact,max_abs_difference=maximum if floating else None,normalized_maxdiff=norm if floating else None,passed=passed)
        result['tensors'].append(row);result['different_elements']+=different;result['passed']=result['passed'] and passed
        result['max_abs_difference']=max(result['max_abs_difference'],maximum);result['max_normalized_difference']=max(result['max_normalized_difference'],norm)
    elif isinstance(a,dict):
        need(list(a)==list(b),'Dictionary keys/order '+path)
        for k in a:compare_tree(t,a[k],b[k],path+'/'+str(k),result,check)
        need(hasattr(a,'_metadata')==hasattr(b,'_metadata'),'State metadata presence')
        if hasattr(a,'_metadata'):compare_tree(t,a._metadata,b._metadata,path+'/_metadata',result,check)
    elif isinstance(a,(list,tuple)):
        need(len(a)==len(b),'Sequence size '+path)
        for i,(x,y) in enumerate(zip(a,b)):compare_tree(t,x,y,path+'/'+str(i),result,check)
    else:
        need(a==b and (not isinstance(a,float) or math.isfinite(a)),'Exact static state '+path)

def perform_comparison(t,helper,s,c_report,s_report,check):
    validate_pair(c_report,s_report,s['checkpoint_refs']);out={}
    expected_schema='DiffSAR-SSDD-native-full-train-segment-report-v1'
    for r in (c_report,s_report):
        need(r.get('schema')==expected_schema and r.get('status')=='NATIVE_FULL_SEGMENT_COMPLETE_PENDING_ROOT' and r.get('stage')=='NATIVE_FULL_SEGMENT_COMPLETE','Completed actual training reports')
        need(r.get('samples_consumed_after_segment')==8 and r.get('selected_sources_and_all_FIT_inputs_unchanged_after') is True,'Actual input/counter completion')
    # Read C4 separately and release it before holding the two final checkpoints.
    c4=t.load(s['checkpoint_refs']['C4']['path'],map_location='cpu',weights_only=False)
    helper.validate_checkpoint_header(c4,4,'RESUME_VALIDATION',helper.PRELOCK)
    digests=helper.component_digests(t,c4);need(digests==checkpoint_entry(c_report,4)['component_digests']==s_report['resume_start_component_digests'],'Actual C4/restored-start full state exact')
    worker_pin={k:s['source_refs']['training_worker'][k] for k in ('bytes','sha256')}
    need(c4['scientific_pins']['sources']['worker']==worker_pin,'Same pinned training-worker digest algorithm')
    pins=c4['scientific_pins'];out['restored_start_component_digests']=digests;out['restored_start_verified_against_actual_C4']=True
    del c4;gc.collect();check()
    a=t.load(s['checkpoint_refs']['C8']['path'],map_location='cpu',weights_only=False);check()
    b=t.load(s['checkpoint_refs']['S8']['path'],map_location='cpu',weights_only=False);check()
    for c in (a,b):
        helper.validate_checkpoint_header(c,8,'RESUME_VALIDATION',helper.PRELOCK)
        need(c['scientific_pins']==pins,'Same scientific code/input pins at endpoint')
    ad=helper.component_digests(t,a);bd=helper.component_digests(t,b)
    need(ad==checkpoint_entry(c_report,8)['component_digests'] and bd==checkpoint_entry(s_report,8)['component_digests'],'Final checkpoint bytes vs reported state digests')
    out['exact_endpoint_components']={k:dict(C=ad[k],S=bd[k],exact=ad[k]==bd[k]) for k in ('rng','loader','scheduler')}
    need(all(v['exact'] for v in out['exact_endpoint_components'].values()),'Exact endpoint RNG/loader/scheduler')
    out['components']={}
    for k in ('model','gradients','optimizer'):
        r=dict(tensors=[],different_elements=0,max_abs_difference=0.0,max_normalized_difference=0.0,passed=True)
        compare_tree(t,a[k],b[k],k,r,check);r['tensor_count']=len(r['tensors']);out['components'][k]=r
    out['losses']=[]
    for x,y in zip(c_report['steps'][4:],s_report['steps']):
        need(set(x['losses'])==set(y['losses']),'Same original loss terms')
        losses={k:scalar_difference(x['losses'][k],y['losses'][k]) for k in x['losses']}
        losses['total_loss']=scalar_difference(x['total_loss'],y['total_loss'])
        out['losses'].append(dict(global_update=x['global_update'],values=losses,passed=all(v['passed'] for v in losses.values())))
    out.update(exact_trace_verified=True,actual_training_updates=12,endpoint_updates=8,
        comparison_passed=all(v['passed'] for v in out['components'].values()) and all(v['passed'] for v in out['losses']),
        normalized_maxdiff_limit=PRELOCK['normalized_maxdiff_limit'],bitexact_continuation_claim=False,theoretical_roundoff_bound=False)
    del a,b;gc.collect();return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--spec-sha',required=True);args=ap.parse_args()
    sr=reference(str(Path(args.spec).resolve(strict=True)));need(sr['sha256']==args.spec_sha,'Actual Root spec SHA')
    s=json.loads(Path(sr['path']).read_bytes());validate_spec(s);out=Path(s['output_directory']);need(not out.exists(),'Fresh output');out.mkdir(mode=0o700)
    start=time.monotonic();r=dict(schema='DiffSAR-native-resume-CPU-comparison-report-v1',operation=OPERATION,run_id=s['run_id'],spec_ref=sr,
        source_refs=s['source_refs'],library_sources=s['library_sources'],checkpoint_refs=s['checkpoint_refs'],training_report_refs=s['training_report_refs'],prelock=PRELOCK,
        pid=os.getpid(),pgid=os.getpgid(0),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),status='STARTED',qualified=False,
        Root_external_admission_required=True,comparison_passed=False,CUDA_initialized=False,AP=False,formal_training_credit=0,checkpoint_written=False,new_training=False,optimizer_steps=0)
    def save():
        r['elapsed_seconds']=time.monotonic()-start;p=out/'REPORT.next'
        with p.open('w',encoding='utf-8') as f:json.dump(r,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(str(p),str(out/'REPORT.json'))
    def check():
        import resource
        need(os.sched_getaffinity(0)=={0,1},'CPU01 affinity');rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        avail=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        r['peak_worker_RSS_bytes']=rss;r['minimum_available_RAM_bytes']=min(avail,r.get('minimum_available_RAM_bytes',avail))
        need(rss<=PRELOCK['peak_group_RSS_cap_bytes'] and avail>=PRELOCK['free_RAM_floor_bytes'] and time.monotonic()-start<1800,'CPU resource contract')
    code=1;t=None;save()
    try:
        check();need(r['minimum_available_RAM_bytes']>=PRELOCK['start_RAM_bytes'],'48GiB startup RAM')
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('PYTHONHASHSEED')=='40244023','GPU hidden/hash seed')
        need(all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'Thread environment')
        for ref in spec_refs(s):verify(ref)
        need(reference(str(Path(__file__).resolve()))==s['source_refs']['worker'],'Actual selected comparator')
        lease=json.loads(Path(s['root_lease_ref']['path']).read_bytes());validate_lease(s,lease);need(lease.get('boot_id')==r['boot_id'],'Same boot lease')
        modules={k:importlib.import_module(v) for k,v in LIBRARY_MODULES.items()};t=modules['torch']
        need(t.__version__=='2.10.0+cu130' and modules['numpy'].__version__=='1.26.4' and platform.machine()=='aarch64','Pinned CPU runtime')
        for k,m in modules.items():need(reference(m.__file__)==s['library_sources'][k],'Actual runtime import '+k)
        r['actual_library_refs']={k:reference(m.__file__) for k,m in modules.items()}
        t.set_num_threads(2);t.set_num_interop_threads(2);need(not t.cuda.is_initialized() and t.cuda.device_count()==0,'No CUDA runtime context')
        helper=types.ModuleType('_Root_DiffSAR_Full_State_Helper');helper.__file__=s['source_refs']['training_worker']['path'];sys.modules[helper.__name__]=helper
        exec(compile(Path(helper.__file__).read_bytes(),helper.__file__,'exec'),helper.__dict__)
        reports={k:json.loads(Path(v['path']).read_bytes()) for k,v in s['training_report_refs'].items()}
        result=perform_comparison(t,helper,s,reports['C'],reports['S'],check);r['comparison']=result;r['comparison_passed']=result['comparison_passed'];save()
        need(r['comparison_passed'],'Declared finite resume-comparison checks')
        for ref in spec_refs(s):verify(ref)
        need(not t.cuda.is_initialized(),'CUDA stayed uninitialized');check()
        r.update(status='CPU_RESUME_COMPARISON_COMPLETE_PENDING_ROOT',selected_sources_and_inputs_unchanged_after=True,CUDA_initialized=False);code=0
    except BaseException as e:r.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),automatic_retry=False)
    finally:
        if t is not None:r['CUDA_initialized']=t.cuda.is_initialized()
        r.update(worker_exit_code=code,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
        print(json.dumps({k:r[k] for k in ('status','worker_exit_code','comparison_passed','CUDA_initialized','AP')}),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
