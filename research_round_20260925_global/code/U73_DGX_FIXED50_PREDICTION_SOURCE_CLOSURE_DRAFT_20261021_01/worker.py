"""UNBOUND U73 source-closure repair for original U66 single-shard prediction."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import random
import sys
import time
import traceback
import types

LOAD_ORDER=(('full_dsconv','dsconv'),('full_head','head'),('full_legacy','model_full'),
    ('full_feature','feature_backbone'),('plain_legacy','plain_model1'),
    ('plain_feature','plain_feature'),('instance_adapter','full_instance_adapter'),
    ('image_geometry','image_geometry'),('shared_initialization','shared_initialization'),
    ('u36_helpers','_U68_U36'),('u39_training_ops','_U68_U39'),
    ('training_state','_U68_TRAINING'),('training_entry','_U68_TRAINING_ENTRY'),
    ('portable_reader','_U68_READER'),('restore_state','_U68_RESTORE'),
    ('prediction_core','_U68_U40_MATH'),('encoder','_U68_ENCODER'),
    ('population_contract','_U68_POPULATION'))
PREDICTION_SOURCE_ROLES=tuple(role for role,_ in LOAD_ORDER)+('mask_api','dev_reader','dev_rgb_reader')
SOURCE_ROLES=PREDICTION_SOURCE_ROLES+('fit_portable_reader',)
FIT_READER_BYTES=19618
RESTORE_ROLES=('training_state','training_entry','u36_helpers','u39_training_ops','shared_initialization',
    'full_legacy','full_dsconv','full_head','full_feature','plain_legacy','plain_feature','instance_adapter')
RANGES=((0,47),(47,94),(94,141),(141,187))
CORE_SHA256='71e10c6539f34105c9726c73e9a0953973a76334da07059cf7a5fc2b619606be'
SOURCE_PINS=dict(training_state='d80ab26ad7092431f4001535705e7afcc4fda7cbf66bacb4dc2cb66edd3d6724',
    training_entry='cd62a640e63210bcb63e58e09bb1e429f1a68ea66dcfec240831a5f1db31aca9',
    shared_initialization='4ba968db7688a03d5975c8546607791a5e09e82ef6c5c7d096cd6a368d1b0292',
    full_legacy='c210b318be29aca7d26516e9f47c8849e2c6a76642d4049fe7b133726ee809b9',
    full_dsconv='b4b612edf9f87cf29dc48bb675d501658fdec5dbee1ea363a7f02d86d197e5f0',
    full_head='6caba966ac7601b611fd1b36fdbd1d758875c5d029758fefe586738c8009f16e',
    full_feature='f7953780580449dd246cbdf1803652183c1dd178f2fd60b37adb238a6d2b6a5a',
    plain_legacy='902e13e5c60b1579459854c6bf4393ad8f268092a73a1d851f0fa640c810f48d',
    plain_feature='d93ff973a57ec3abfde98bb7b9ab4fcf5a3443445a512c2acd16460fce5bfbad',
    instance_adapter='9ad221170f69c028474f76700c991cb76dfecf19b11f5198767843b8e378d8b3',
    u36_helpers='1e726092a19d80a06f58d6a56f85f4c15bff2b744f0468a7e0a35fa2c85999e5',
    u39_training_ops='378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d',
    restore_state='0943096b6d9787bdb370212c8c311427c37f5ef1eb4c32ab397c1ea35faf3d14',
    portable_reader='8fabc800e4d36e2da3988e42acf88058c80790f9c8b3d664a1a4fa9232233473',
    image_geometry='65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
    prediction_core='819a2350487394265980cb77f8ff7e89547dbcac8e0d3e3c175a67a2794c9680',
    population_contract='733a208c5df49113e9dff3338b2b88ad2d0d4b26b2bbdd3676ce10417e05c9a5',
    mask_api='1281ce0b5436ffe924c96521b2ae1d589f6f0e2a7b7281b6bb284f7b2245545c',
    dev_reader='daabec8388e23d9ff646ae78eba69ba2168099a72971578a8d34cef04f1268c0',
    dev_rgb_reader='b91375e5366da3f74272ce811f453be1379b46a4bb96103f9d8039a4998a1204',
    fit_portable_reader='b65ed9ecb7a9a704899f3c75983b2cb9676e26b42d87f65336cfbcb16bd94557')
LIBRARIES=('torch','numpy','psutil','pycocotools','mask_extension')


def need(ok,message):
    if not ok:raise ValueError(message)


def strict_ref(ref,*,allow_empty=False):
    need(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and type(ref['path']) is str
        and PurePosixPath(ref['path']).anchor=='/' and str(PurePosixPath(ref['path']))==ref['path']
        and '..' not in PurePosixPath(ref['path']).parts and '\\' not in ref['path']
        and type(ref['bytes']) is int and ref['bytes']>=(0 if allow_empty else 1) and type(ref['sha256']) is str
        and len(ref['sha256'])==64 and set(ref['sha256'])<=set('0123456789abcdef'),'Exact canonical POSIX file ref')
    return ref


def validate_request(request):
    need(type(request) is dict and set(request)=={'worker_source','prediction_core_source','source_refs',
        'library_sources','endpoint_refs','checkpoint_refs','manifest_ref','shard_index','dataset_range',
        'output_directory'},'Exact U68 worker arguments, no owner/Root admission fields')
    need(type(request['shard_index']) is int and 0<=request['shard_index']<4
        and type(request['dataset_range']) is list and all(type(i) is int for i in request['dataset_range'])
        and tuple(request['dataset_range'])==RANGES[request['shard_index']],'One fixed original shard')
    need(set(request['source_refs'])==set(SOURCE_ROLES) and set(SOURCE_PINS)==set(SOURCE_ROLES)-{'encoder'},
        'Exact22 source refs including byte-only original FIT reader; encoder is separately Root-pinned')
    for role,ref in request['source_refs'].items():
        strict_ref(ref)
        if role!='encoder':need(ref['sha256']==SOURCE_PINS[role],'Original reviewed code pin: '+role)
    need(request['source_refs']['fit_portable_reader']['bytes']==FIT_READER_BYTES,'Original19618-byte FIT reader')
    for field,roles in (('library_sources',set(LIBRARIES)),('endpoint_refs',{'qualified','spec','pair_commit','order'}),
                        ('checkpoint_refs',{'PLAIN','FULL'})):
        need(type(request[field]) is dict and set(request[field])==roles,'Explicit complete roles: '+field)
        for role,ref in request[field].items():strict_ref(ref,allow_empty=field=='library_sources' and role=='pycocotools')
    for key in ('worker_source','prediction_core_source','manifest_ref'):strict_ref(request[key])
    need(request['prediction_core_source']['sha256']==CORE_SHA256,'Exact frozen U66 core')
    out=Path(request['output_directory'])
    need(type(request['output_directory']) is str and out.is_absolute() and out.resolve()==out
        and not out.is_symlink() and out.parent.is_dir(),'Canonical explicit output with existing parent')
    return request


def direct_references(request,entries):
    """Pure metadata union for a future owner; only selected packages are opened."""
    validate_request(request)
    need(type(entries) is list and len(entries)==187,'Full187 manifest metadata, not a selected-only manifest')
    refs=list(request['source_refs'].values())+list(request['library_sources'].values())
    refs+=list(request['endpoint_refs'].values())+list(request['checkpoint_refs'].values())
    refs += [request[key] for key in ('worker_source','prediction_core_source','manifest_ref')]
    image_ids=set()
    for i,row in enumerate(entries):
        need(type(row) is dict and set(row)=={'dataset_index','image_id','query_ids','package_ref'}
            and type(row['dataset_index']) is int and row['dataset_index']==i
            and type(row['image_id']) is int and row['image_id'] not in image_ids
            and type(row['query_ids']) is list and len(row['query_ids'])==100
            and all(type(q) is int for q in row['query_ids']) and set(row['query_ids'])==set(range(100)),
            'Original187 image order and original full100 permutation')
        image_ids.add(row['image_id']);strict_ref(row['package_ref'])
        if request['dataset_range'][0]<=i<request['dataset_range'][1]:refs.append(row['package_ref'])
    union={}
    for ref in refs:
        strict_ref(ref,allow_empty=ref==request['library_sources']['pycocotools'])
        need(ref['path'] not in union or union[ref['path']]==ref,'Conflicting direct source refs')
        union[ref['path']]=ref
        out=PurePosixPath(request['output_directory']);path=PurePosixPath(ref['path'])
        need(out!=path and out not in path.parents,'Output cannot contain direct inputs')
    return [deepcopy(union[path]) for path in sorted(union)]


def validate_endpoint_source_closure(endpoint,source_refs):
    """Pure extra closure check, after the unchanged original U58 endpoint/12-role gates."""
    need(type(source_refs) is dict and set(source_refs)==set(SOURCE_ROLES),'Exact22 source refs')
    original=endpoint['restore_source_refs']
    need(type(original) is dict and set(original)==set(RESTORE_ROLES)|{'image_geometry','portable_reader'},
        'Original endpoint14 source refs, not a synthetic partial closure')
    for role,original_role in (('image_geometry','image_geometry'),('fit_portable_reader','portable_reader')):
        strict_ref(source_refs[role]);strict_ref(original[original_role])
        need(source_refs[role]==original[original_role],
            'Exact original accepted path/bytes/SHA for '+role)
    return True


def import_runtime(c,request):
    """Future genuine imports only; not called by the finite metadata tests."""
    need(platform.system()=='Linux' and platform.machine()=='aarch64' and platform.python_version()=='3.12.14',
        'Future pinned Linux/aarch64 Python3.12.14 worker')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and all(os.environ.get(k)=='2' for k in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'Original CUDA0/CPU01/two-thread environment')
    need(not any(name.split('.')[0] in ('torch','numpy','pycocotools') for name in sys.modules),
        'Fresh child; no scientific imports before source admission')
    refs=request['library_sources']
    for ref in refs.values():_bytes(c,ref)
    _bytes(c,request['source_refs']['mask_api'])
    loaded={key:importlib.import_module(name) for key,name in
        (('psutil','psutil'),('torch','torch'),('numpy','numpy'),('pycocotools','pycocotools'),
         ('mask_extension','pycocotools._mask'),('mask_api','pycocotools.mask'))}
    for role,module in loaded.items():
        ref=request['source_refs']['mask_api'] if role=='mask_api' else refs[role]
        need(type(module) is types.ModuleType and str(Path(module.__file__).resolve())==ref['path'],
            'Actual admitted scientific library/module: '+role)
        _bytes(c,ref)
    torch,np,psutil=(loaded[k] for k in ('torch','numpy','psutil'))
    need(str(torch.__version__)=='2.10.0+cu130' and str(np.__version__)=='1.26.4'
        and str(psutil.__version__)=='6.1.1' and c.psutil is psutil,'Exact genuine Torch/NumPy/psutil and original control')
    need(not torch.cuda.is_initialized() and torch.get_default_dtype()==torch.float32
        and not torch.is_autocast_enabled() and psutil.Process().cpu_affinity()==[0,1],
        'Cold CUDA, original FP32/noAMP and CPU01')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    need(torch.get_num_threads()==torch.get_num_interop_threads()==2,'Actual two Torch threads')
    c.check()
    return torch,np,loaded['mask_api']


def _bytes(c,ref):
    c.check();path=c.closed(ref)
    need(str(path)==ref['path'],'Exact admitted path')
    raw=path.read_bytes()
    need(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'Actual admitted bytes')
    return raw


@contextmanager
def source_modules(c,source_refs,core_ref,mask_api):
    """Execute explicitly closed definitions under the original dependency aliases."""
    need(type(source_refs) is dict and set(source_refs)==set(SOURCE_ROLES),'Exact22 source roles, original19 modules')
    originals={};loaded={};core=None
    def load(ref,alias):
        raw=_bytes(c,ref)
        originals.setdefault(alias,(alias in sys.modules,sys.modules.get(alias)))
        module=types.ModuleType(alias);module.__file__=ref['path'];sys.modules[alias]=module
        exec(compile(raw,ref['path'],'exec'),module.__dict__)
        need(module.__file__==ref['path'],'Unchanged source-backed module identity')
        return module
    try:
        for role in ('dev_reader','dev_rgb_reader','fit_portable_reader'):_bytes(c,source_refs[role])
        _bytes(c,source_refs['mask_api'])
        need(type(mask_api) is types.ModuleType and mask_api.__file__==source_refs['mask_api']['path'],
            'Genuine already source-closed pycocotools.mask')
        for role,alias in LOAD_ORDER:loaded[role]=load(source_refs[role],alias)
        loaded['mask_api']=mask_api
        need(loaded['full_legacy'].DSConv is loaded['full_dsconv'].DSConv
            and loaded['full_legacy'].CoordConv is loaded['full_head'].CoordConv,'Original dependency objects')
        core=load(core_ref,'_U68_U66_CORE')
        yield core,loaded
    finally:
        loaded.clear();core=None
        for alias,(existed,old) in reversed(list(originals.items())):
            if existed:sys.modules[alias]=old
            else:sys.modules.pop(alias,None)


def _write_json(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode('utf-8')
    with Path(path).open('xb') as stream:
        need(stream.write(raw)==len(raw),'Complete JSON write');stream.flush();os.fsync(stream.fileno())
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def persist_prediction(output_directory,prediction):
    """Keep the original U66 JSON body consumable by U67; never grant acceptance."""
    need(prediction['schema']=='U66-DGX-fixed50-DEV-shard-pending-v1'
        and prediction['status']=='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE'
        and all(prediction[key] is False for key in ('qualified','AP','TEST','new_training'))
        and type(prediction['formal_training_budget_credit']) is int
        and prediction['formal_training_budget_credit']==0,'Only completed pending prediction output')
    directory=Path(output_directory)
    if not directory.exists():directory.mkdir()
    return _write_json(directory/'U66_SHARD.json',prediction)


def run_worker(c,request,*,report):
    """Owned future body, not a launcher: caller already owns deadline/resources.

    Its external control must close the exact refs and enforce the original
    Linux process-group/resource contract. This function creates no owner/Q.
    """
    need(type(report) is dict and not report,'Fresh worker report')
    started=time.monotonic();created=False;out=None;primary=None;torch=np=mask_api=None
    outside=None;prediction={};references=[]
    report.update(schema='U73-single-shard-worker-report-v1',status='STARTED',qualified=False,
        AP=False,DEV=True,DEV_technical_input=True,TEST=False,new_training=False,
        formal_training_budget_credit=0,ROOT_actual_terminal_self_certified=False,
        automatic_retry=False,full_binary_closure_claimed=False,cleanup_errors=[],
        original_restore_source_closure_verified=False)
    try:
        validate_request(request);c.check()
        need(request['worker_source']['path']==str(__file__),'This exact admitted future worker')
        for ref in list(request['source_refs'].values())+list(request['library_sources'].values())+[
            request['worker_source'],request['prediction_core_source']]:_bytes(c,ref)
        metadata=dict(refs=deepcopy(request['endpoint_refs']))
        for role,ref in request['endpoint_refs'].items():metadata[role]=json.loads(_bytes(c,ref))
        manifest=json.loads(_bytes(c,request['manifest_ref']));entries=manifest['entries']
        references=direct_references(request,entries)
        out=Path(request['output_directory']);out.mkdir();created=True
        report.update(shard_index=request['shard_index'],dataset_range=deepcopy(request['dataset_range']),
            direct_input_refs=references,worker_source=deepcopy(request['worker_source']),
            prediction_core_source=deepcopy(request['prediction_core_source']),
            source_refs=deepcopy(request['source_refs']))
        torch,np,mask_api=import_runtime(c,request)
        outside=(torch.get_rng_state().clone(),np.random.get_state(),random.getstate(),
            torch.get_default_dtype(),torch.is_grad_enabled(),
            (torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,
             torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32))
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        with source_modules(c,request['source_refs'],request['prediction_core_source'],mask_api) as (core,modules):
            restore=modules['restore_state']
            endpoint=restore.validate_final_endpoint(modules['training_state'],modules['training_entry'],metadata)
            restore.validate_restore_sources(endpoint,{r:request['source_refs'][r] for r in RESTORE_ROLES})
            validate_endpoint_source_closure(endpoint,request['source_refs'])
            report['original_restore_source_closure_verified']=True
            need(endpoint['checkpoint_refs']==request['checkpoint_refs'],'Original actual two-arm endpoint checkpoint refs')
            modules['portable_reader'].validate_entries(entries)
            c.check()
            returned=core.predict_shard(torch,np,c,endpoint_metadata=metadata,entries=entries,modules=modules,
                source_refs={r:request['source_refs'][r] for r in PREDICTION_SOURCE_ROLES},shard_index=request['shard_index'],
                dataset_range=request['dataset_range'],report=prediction)
            need(returned is prediction,'Original core returns the caller partial report')
        report['source_aliases_restored']=True
        c.check()
        for ref in list(request['source_refs'].values())+list(request['library_sources'].values())+[
            request['worker_source'],request['prediction_core_source'],request['manifest_ref']]+list(request['endpoint_refs'].values()):
            _bytes(c,ref)
        report['source_and_metadata_bytes_rechecked']=True
        need(prediction['shard_index']==request['shard_index'] and prediction['dataset_range']==request['dataset_range'],
            'Original requested shard identity')
        report['prediction_ref']=persist_prediction(out,prediction)
        report['status']='PREDICTION_SAVED_PENDING_CLEANUP'
    except BaseException as error:
        primary=error;report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc()))
        traceback.clear_frames(error.__traceback__);error.__traceback__=None
    finally:
        if outside is not None:
            cpu,numpy_rng,python_rng,dtype,grad,backend=outside
            def restore_backend():
                (torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,
                    torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)=backend
            for label,restore_value in (
                ('CPU_RNG',lambda:torch.set_rng_state(cpu)),('NumPy_RNG',lambda:np.random.set_state(numpy_rng)),
                ('Python_RNG',lambda:random.setstate(python_rng)),('dtype',lambda:torch.set_default_dtype(dtype)),
                ('grad',lambda:torch.set_grad_enabled(grad)),('backend',restore_backend)):
                try:restore_value()
                except BaseException as error:
                    report['cleanup_errors'].append(dict(operation=label,type=type(error).__name__,message=str(error)))
                    report['status']='FAILED'
                    if primary is None:primary=error
            try:
                current=np.random.get_state()
                need(torch.equal(torch.get_rng_state(),cpu) and current[0]==numpy_rng[0]
                    and bool(np.array_equal(current[1],numpy_rng[1])) and current[2:]==numpy_rng[2:]
                    and random.getstate()==python_rng and torch.get_default_dtype()==dtype
                    and torch.is_grad_enabled()==grad and torch.get_num_threads()==torch.get_num_interop_threads()==2,
                    'Outer CPU/NumPy/Python RNG, dtype/grad/thread state restored')
                if torch.cuda.is_initialized():
                    need(torch.cuda.memory_allocated()==torch.cuda.memory_reserved()==0,'U66 final GPU release remains zero')
                report['outer_runtime_states_restored']=not report['cleanup_errors']
            except BaseException as error:
                report['cleanup_errors'].append(dict(type=type(error).__name__,message=str(error)))
                report['status']='FAILED'
                if primary is None:primary=error
        report['elapsed_seconds']=time.monotonic()-started
        try:
            c.check();need(report['elapsed_seconds']<=1800,'Original complete worker deadline')
        except BaseException as error:
            report['status']='FAILED';report['cleanup_errors'].append(dict(type=type(error).__name__,message=str(error)))
            if primary is None:primary=error
        if report['status']=='PREDICTION_SAVED_PENDING_CLEANUP':report['status']='COMPLETE_PENDING_ROOT_ACCEPTANCE'
        if report['status']=='FAILED':report['partial_prediction_report']=prediction
        if created:
            try:_write_json(out/'REPORT.json',report)
            except BaseException as error:
                report['status']='FAILED';report['report_write_error']=dict(type=type(error).__name__,message=str(error))
                if primary is None:primary=error
    if primary is not None:raise primary
    need(report['status']=='COMPLETE_PENDING_ROOT_ACCEPTANCE','No completed worker without actual core output/cleanup')
    return report
