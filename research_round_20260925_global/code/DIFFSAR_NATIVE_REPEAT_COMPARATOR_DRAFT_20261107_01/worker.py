"""Unexecuted S/R repeat characterization draft; stdlib only at module import."""
import argparse,datetime,gc,hashlib,importlib,json,os,platform,sys,time,traceback,types
from pathlib import Path,PurePosixPath
SCHEMA='Root-DiffSAR-native-repeat-compare-spec-v1'
LEASE_SCHEMA='Root-DiffSAR-native-repeat-compare-lease-v1'
OPERATION='CPU_DIFFSAR_NATIVE_S8_R8_VARIANCE_CHARACTERIZATION_V01'
LIBRARY_MODULES={'torch':'torch','numpy':'numpy'}
LIBRARY_NAMES=tuple(LIBRARY_MODULES)
PRELOCK=dict(S=[4,8],R=[4,8],new_training_updates=4,independent_seed=False,
    reference='S',normalized_difference='maxabs(S-R)/(1+maxabs(S))',prior_threshold=1e-4,
    numerical_excess_is_valid_observation=True,strict_state_trace_RNG_and_finiteness=True,
    prior_failed_comparison_preserved=True,full_continuity_qualified=False,
    theoretical_roundoff_bound=False,bitexact_claim=False,
    CPU_affinity=[0,1],threads=2,peak_group_RSS_cap_bytes=16*1024**3,
    free_RAM_floor_bytes=16*1024**3,start_RAM_bytes=48*1024**3,external_deadline_seconds=1800,
    single_attempt=True,automatic_retry=False,GPU_hidden=True,checkpoint_write=False,AP=False,formal_training_credit=0)
REFERENCE_COMPARATOR_PIN=dict(bytes=16721,sha256='68a1c3e1c9ab097da0440aa85461eddf458d8141bf4f010e42097350545bf18a')

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
    z=p.stat();need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns),'Stable selected bytes')
    return dict(path=str(p),bytes=n,sha256=h.hexdigest())

def verify(r):valid_ref(r);need(reference(r['path'])==r,'Pinned input mismatch')

def spec_refs(s):
    return list(s['source_refs'].values())+list(s['library_sources'].values())+list(s['checkpoint_refs'].values())+list(s['training_report_refs'].values())+[s['root_lease_ref']]

def validate_spec(s):
    need(type(s) is dict and set(s)=={'schema','operation','run_id','source_refs','library_sources','checkpoint_refs','training_report_refs','root_lease_ref','output_directory','prelock'},'Exact characterization spec')
    need(s['schema']==SCHEMA and s['operation']==OPERATION and s['prelock']==PRELOCK and s['run_id'].startswith('DiffSAR-SSDD-repeat-compare-v01-'),'Fixed characterization')
    need(set(s['source_refs'])=={'worker','training_worker','reference_comparator'} and set(s['library_sources'])==set(LIBRARY_NAMES),'Selected source roles')
    need(set(s['checkpoint_refs'])=={'C4','S8','R8'} and set(s['training_report_refs'])=={'S','R'},'Actual identical-restored-start roles')
    refs=spec_refs(s)
    for r in refs:valid_ref(r)
    need(len({r['path'] for r in refs})==len(refs),'Distinct selected inputs')
    need(all(s['source_refs']['reference_comparator'][k]==v for k,v in REFERENCE_COMPARATOR_PIN.items()),'Unchanged prior numerical algorithm')
    out=s['output_directory'];need(type(out) is str and PurePosixPath(out).is_absolute() and str(PurePosixPath(out))==out and '..' not in PurePosixPath(out).parts and out!='/','Canonical fresh output')
    need(all(r['path']!=out and not r['path'].startswith(out+'/') for r in refs),'Read-only inputs outside output')

def validate_lease(s,l):
    need(l.get('schema')==LEASE_SCHEMA and l.get('status')=='ROOT_GRANTED','Explicit characterization lease')
    for k in ('operation','run_id','output_directory','prelock','source_refs','checkpoint_refs','training_report_refs'):need(l.get(k)==s[k],'Lease '+k)
    need(l.get('worker_ref')==s['source_refs']['worker'],'Selected own worker')
    need(l.get('single_attempt') is l.get('main_scientific_processes_reaped') is l.get('exclusive_project_scientific_leaf') is True and l.get('whole_device_exclusive') is False,'Single reaped scientific leaf')
    need(l.get('prior_failed_comparison_preserved') is True and l.get('full_continuity_qualified') is False,'No retroactive qualification')

def load_module(ref,name):
    m=types.ModuleType(name);m.__file__=ref['path'];sys.modules[name]=m
    exec(compile(Path(m.__file__).read_bytes(),m.__file__,'exec'),m.__dict__);return m

def validate_repeat_metadata(s,r,refs):
    if any(x.get('resume_checkpoint_ref')!=refs['C4'] for x in (s,r)):raise ValueError('Both runs must load same C4')
    for report,key in ((s,'S8'),(r,'R8')):
        if report.get('resume_start_exact') is not True or report.get('all_requested_updates_completed') is not True or type(report.get('worker_exit_code')) is not int or report['worker_exit_code']!=0:
            raise ValueError('Actual finite restored training completion')
        if report.get('optimizer_steps_completed')!=8 or report.get('new_optimizer_updates')!=4 or report.get('samples_consumed_after_segment')!=8:
            raise ValueError('Exactly four new real optimizer updates')
        cp=[x for x in report['checkpoints'] if x['updates']==8]
        if len(cp)!=1 or cp[0]['ref']!=refs[key]:raise ValueError('Pinned own endpoint join')
        if [x['global_update'] for x in report['steps']]!=list(range(5,9)):raise ValueError('Global updates5to8 only')
    if s.get('resume_start_component_digests')!=r.get('resume_start_component_digests'):raise ValueError('Both restored complete starting states exact')
    keys=('global_update','dataset_index','image_id','mapped_image_shape','mapped_instances','augmentation_and_target_sha256','used_learning_rates','post_scheduler_learning_rates')
    for a,b in zip(s['steps'],r['steps']):
        if any(a[k]!=b[k] for k in keys):raise ValueError('Actual input/augmentation/order/LR differs')
    return True

def classify_numeric_results(components,losses):
    if any(not row['passed'] for c in components.values() for row in c['tensors'] if row['exact_required']):raise ValueError('Integer/semantic-counter state differs')
    return all(x['passed'] for x in components.values()) and all(x['passed'] for x in losses)

def characterize(t,base,helper,s,reports,check):
    a_report=reports['S'];b_report=reports['R'];validate_repeat_metadata(a_report,b_report,s['checkpoint_refs'])
    for r in reports.values():
        need(r.get('schema')=='DiffSAR-SSDD-native-full-train-segment-report-v1' and r.get('status')=='NATIVE_FULL_SEGMENT_COMPLETE_PENDING_ROOT' and r.get('stage')=='NATIVE_FULL_SEGMENT_COMPLETE','Actual completed training reports')
        need(r.get('selected_sources_and_all_FIT_inputs_unchanged_after') is True,'Actual unchanged training inputs')
    need(base.PRELOCK['normalized_maxdiff_limit']==PRELOCK['prior_threshold'],'Identical prior threshold')
    c4=t.load(s['checkpoint_refs']['C4']['path'],map_location='cpu',weights_only=False)
    helper.validate_checkpoint_header(c4,4,'RESUME_VALIDATION',helper.PRELOCK);digests=helper.component_digests(t,c4)
    need(digests==a_report['resume_start_component_digests']==b_report['resume_start_component_digests'],'Both restored starts against actual C4')
    need(c4['scientific_pins']['sources']['worker']=={k:s['source_refs']['training_worker'][k] for k in ('bytes','sha256')},'Same frozen training math')
    pins=c4['scientific_pins'];out=dict(actual_C4_restored_start_exact=True,restored_start_component_digests=digests)
    del c4;gc.collect();check()
    a=t.load(s['checkpoint_refs']['S8']['path'],map_location='cpu',weights_only=False);check()
    b=t.load(s['checkpoint_refs']['R8']['path'],map_location='cpu',weights_only=False);check()
    for c in (a,b):
        helper.validate_checkpoint_header(c,8,'RESUME_VALIDATION',helper.PRELOCK);need(c['scientific_pins']==pins,'Same endpoint scientific pins')
    ad=helper.component_digests(t,a);bd=helper.component_digests(t,b)
    need(ad==base.checkpoint_entry(a_report,8)['component_digests'] and bd==base.checkpoint_entry(b_report,8)['component_digests'],'Actual endpoints joined to their own reports')
    out['exact_endpoint_components']={k:dict(S=ad[k],R=bd[k],exact=ad[k]==bd[k]) for k in ('rng','loader','scheduler')}
    need(all(v['exact'] for v in out['exact_endpoint_components'].values()),'Exact endpoint RNG/loader/scheduler')
    components={}
    for k in ('model','gradients','optimizer'):
        row=dict(tensors=[],different_elements=0,max_abs_difference=0.0,max_normalized_difference=0.0,passed=True)
        base.compare_tree(t,a[k],b[k],k,row,check);row['tensor_count']=len(row['tensors']);components[k]=row
    losses=[]
    for x,y in zip(a_report['steps'],b_report['steps']):
        need(set(x['losses'])==set(y['losses']),'Identical original loss terms')
        values={k:base.scalar_difference(x['losses'][k],y['losses'][k]) for k in x['losses']}
        values['total_loss']=base.scalar_difference(x['total_loss'],y['total_loss'])
        losses.append(dict(global_update=x['global_update'],values=values,passed=all(v['passed'] for v in values.values())))
    within=classify_numeric_results(components,losses)
    out.update(components=components,losses=losses,exact_trace_verified=True,strict_integer_state_match=True,mandatory_state_trace_RNG_finite_checks_passed=True,
        within_prior_threshold=within,reference='S',prior_threshold=PRELOCK['prior_threshold'],numerical_excess_is_execution_failure=False,
        prior_failed_comparison_upgraded=False,full_continuity_qualified=False,independent_seed=False)
    del a,b;gc.collect();return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--spec-sha',required=True);args=ap.parse_args()
    sr=reference(str(Path(args.spec).resolve(strict=True)));need(sr['sha256']==args.spec_sha,'Actual spec SHA')
    s=json.loads(Path(sr['path']).read_bytes());validate_spec(s);out=Path(s['output_directory']);need(not out.exists(),'Fresh output');out.mkdir(mode=0o700)
    start=time.monotonic();r=dict(schema='DiffSAR-native-repeat-CPU-characterization-report-v1',operation=OPERATION,run_id=s['run_id'],spec_ref=sr,
        source_refs=s['source_refs'],library_sources=s['library_sources'],checkpoint_refs=s['checkpoint_refs'],training_report_refs=s['training_report_refs'],prelock=PRELOCK,
        pid=os.getpid(),pgid=os.getpgid(0),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),status='STARTED',qualified=False,Root_external_admission_required=True,
        characterization_complete=False,within_prior_threshold=None,prior_failed_comparison_upgraded=False,full_continuity_qualified=False,CUDA_initialized=False,
        new_training=False,optimizer_steps=0,formal_training_credit=0,AP=False,checkpoint_written=False)
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
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('PYTHONHASHSEED')=='40244023','GPU hidden/fixed hash seed')
        need(all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'Thread environment')
        for ref in spec_refs(s):verify(ref)
        need(reference(str(Path(__file__).resolve()))==s['source_refs']['worker'],'Selected own worker')
        lease=json.loads(Path(s['root_lease_ref']['path']).read_bytes());validate_lease(s,lease);need(lease.get('boot_id')==r['boot_id'],'Same boot lease')
        mods={k:importlib.import_module(v) for k,v in LIBRARY_MODULES.items()};t=mods['torch']
        need(t.__version__=='2.10.0+cu130' and mods['numpy'].__version__=='1.26.4' and platform.machine()=='aarch64','Pinned CPU runtime')
        for k,m in mods.items():need(reference(m.__file__)==s['library_sources'][k],'Selected actual library '+k)
        r['actual_library_refs']={k:reference(m.__file__) for k,m in mods.items()};t.set_num_threads(2);t.set_num_interop_threads(2)
        need(not t.cuda.is_initialized() and t.cuda.device_count()==0,'No CUDA context')
        base=load_module(s['source_refs']['reference_comparator'],'_Root_Frozen_Prior_Comparator');helper=load_module(s['source_refs']['training_worker'],'_Root_Frozen_Training_Helper')
        reports={k:json.loads(Path(ref['path']).read_bytes()) for k,ref in s['training_report_refs'].items()}
        result=characterize(t,base,helper,s,reports,check);r.update(characterization=result,within_prior_threshold=result['within_prior_threshold']);save()
        for ref in spec_refs(s):verify(ref)
        need(not t.cuda.is_initialized(),'CUDA stayed uninitialized');check()
        r.update(status='CPU_REPEAT_CHARACTERIZATION_COMPLETE_PENDING_ROOT',characterization_complete=True,selected_sources_and_inputs_unchanged_after=True);code=0
    except BaseException as e:r.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),automatic_retry=False)
    finally:
        if t is not None:r['CUDA_initialized']=t.cuda.is_initialized()
        r.update(worker_exit_code=code,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
        print(json.dumps({k:r[k] for k in ('status','worker_exit_code','characterization_complete','within_prior_threshold','CUDA_initialized','AP')}),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
