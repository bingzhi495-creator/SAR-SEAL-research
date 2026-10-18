"""UNBOUND independent DGX two-image pilot body; no module-level ML import.

The actual owner supplies imported source-closed modules, two authenticated U49
packages and a continuous/hard resource controller. This is not a worker entry,
loader for arbitrary Python objects, remote launcher, or U40 continuation.
"""
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback


SOURCE_PINS={
    'shared_initialization':'4ba968db7688a03d5975c8546607791a5e09e82ef6c5c7d096cd6a368d1b0292',
    'full_legacy':'c210b318be29aca7d26516e9f47c8849e2c6a76642d4049fe7b133726ee809b9',
    'full_dsconv':'b4b612edf9f87cf29dc48bb675d501658fdec5dbee1ea363a7f02d86d197e5f0',
    'full_head':'6caba966ac7601b611fd1b36fdbd1d758875c5d029758fefe586738c8009f16e',
    'full_feature':'f7953780580449dd246cbdf1803652183c1dd178f2fd60b37adb238a6d2b6a5a',
    'plain_legacy':'902e13e5c60b1579459854c6bf4393ad8f268092a73a1d851f0fa640c810f48d',
    'plain_feature':'d93ff973a57ec3abfde98bb7b9ab4fcf5a3443445a512c2acd16460fce5bfbad',
    'instance_adapter':'9ad221170f69c028474f76700c991cb76dfecf19b11f5198767843b8e378d8b3',
    'image_geometry':'65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
    'u36_helpers':'1e726092a19d80a06f58d6a56f85f4c15bff2b744f0468a7e0a35fa2c85999e5',
    'u39_training_ops':'378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d',
}
ORIGINAL_READER_PINS={
    'rgb_matched_reader':'678ce3781872e21e961b24a7b0a4d0ecc69f0f8a08b8feb7eb0b2e8624d57025',
    'matched_reader':'82565caa9c42b6c2ed501e13af9554c6ef47a99b6b525ccfce0bcef620534870',
    'image_geometry':SOURCE_PINS['image_geometry'],
}
FIELDS=('rgb400','e','M','L','targets')
EXPECTED_G={54:1,710:27}


def need(ok,message):
    if not ok:raise ValueError(message)


def pilot_contract():
    return dict(schema='U50-DGX-paired-disposable-pilot-prelock-v1',
        environment=dict(python='3.12.14',torch='2.10.0+cu130',numpy='1.26.4',system='Linux',machine='aarch64'),
        image_order=[54,710,54],arms=['PLAIN','FULL'],initialization_seed=202610173101,
        image_batch_size=1,branch_size=400,canvas_size=800,dtype='float32',query_chunk=1,
        all_matched_GT=True,updates_per_arm=3,total_disposable_updates=6,
        optimizer=dict(name='AdamW',lr=3e-4,weight_decay=1e-4,betas=[.9,.999],eps=1e-8,foreach=False,fused=False),
        encoder_saves='save_on_cpu_pin_memory_true',loss='query_mean_original_grid_BCE_plus_softDice_smooth1',
        AMP=False,compile=False,VJP=False,checkpoint_recompute=False,automatic_retry=False,
        resume_U40_checkpoint=False,resume_U40_optimizer_or_RNG=False,formal_training_budget_credit=0,
        main_training=False,AP=False,DEV=False,TEST=False,
        limits=dict(allowed_cpus=[0,1],threads=2,allocator_bytes=4*1024**3,
            maximum_RSS_bytes=16*1024**3,minimum_available_RAM_bytes=16*1024**3,deadline_seconds=1800))


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def validate_prelock(pre):
    need(type(pre) is dict and _canonical(pre)==_canonical(pilot_contract()),
         'Exact independent DGX environment/order/FP32/full-G/optimizer/resource recipe; no U40 continuation')
    return list(pre['image_order'])


def _strict_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'}
         and type(value['path']) is str and Path(value['path']).is_absolute()
         and type(value['bytes']) is int and value['bytes']>=0
         and type(value['sha256']) is str and len(value['sha256'])==64
         and set(value['sha256'])<=set('0123456789abcdef'),'Actual admitted local source/package strictref')
    return value


def _closed(c,value):
    _strict_ref(value);path=Path(c.closed(value))
    need(path.is_absolute() and path.resolve()==Path(value['path']).resolve(),'Exact admitted local path')
    return path


def _sources(c,modules,refs,pilot_source_ref):
    need(set(modules)==set(refs)==set(SOURCE_PINS),'Exact eleven numeric/source component roles')
    for role,digest in SOURCE_PINS.items():
        value=refs[role];path=_closed(c,value);raw=path.read_bytes()
        need(value['sha256']==digest and len(raw)==value['bytes'] and hashlib.sha256(raw).hexdigest()==digest
             and Path(modules[role].__file__).resolve()==path.resolve(),'Original immutable component bytes/object: '+role)
    need(_closed(c,pilot_source_ref).resolve()==Path(__file__).resolve(),'This actual new independent pilot body is admitted')


def _value_digest(value):
    # Exact U49 raw contiguous tensor-byte digest; shape/dtype are checked apart.
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes(order='C')).hexdigest()


def _load_item(torch,c,index,reference,geometry_module):
    c.check();path=_closed(c,reference)
    package=torch.load(str(path),map_location='cpu',weights_only=True)
    expected={'schema','metadata','tensors','equivalence','selected_indices','images','qualified',
              'new_training','AP','DEV','TEST','whole_population_exported'}
    need(type(package) is dict and set(package)==expected
         and package['schema']=='U49-matched-CPU-portable-item-v1'
         and package['selected_indices']==[54,710] and package['images']==1
         and all(package[k] is False for k in ('qualified','new_training','AP','DEV','TEST','whole_population_exported')),
         'Original pending matched-only U49 item, never a full-population or self-qualified package')
    meta=package['metadata'];tensors=package['tensors'];proofs=package['equivalence']
    need(type(meta) is dict and set(meta)=={'dataset_index','image_id','G','query_ids','ann_ids',
         'identity','rgb_receipt','geometry','provenance'} and meta['dataset_index']==index
         and type(meta['G']) is int and meta['G']==EXPECTED_G[index], 'Complete original selected-image G and identity')
    identity=meta['identity'];G=meta['G'];queries=meta['query_ids'];annotations=meta['ann_ids']
    need(type(identity) is dict and identity['image_id']==meta['image_id'] and identity['ann_ids']==annotations
         and type(queries) is list and len(queries)==len(set(queries))==G
         and all(type(q) is int and 0<=q<100 for q in queries)
         and type(annotations) is list and len(annotations)==len(set(annotations))==G
         and all(type(a) is int for a in annotations),'Retained GT/query one-to-one original order, never natural-axis reindex')
    geo=meta['geometry'];rgb=meta['rgb_receipt'];provenance=meta['provenance']
    need(type(geo) is dict and set(geo)=={'original_shape','actual_img_shape','canvas_size','branch_size'}
         and list(geo['original_shape'])==list(identity['original_shape'])
         and list(geo['actual_img_shape'])==list(identity['actual_img_shape'])
         and geo['canvas_size']==800 and geo['branch_size']==400 and identity['batch_input_shape']==[800,800],
         'Original integer content/canvas/original geometry is retained')
    need(rgb['image']==identity['image'] and list(rgb['original_shape'])==list(geo['original_shape'])
         and list(rgb['actual_img_shape'])==list(geo['actual_img_shape']) and rgb['canvas_size']==800 and rgb['branch_size']==400
         and rgb['decoded_mode']=='RGB' and rgb['auto_orientation'] is False
         and (rgb['exif_orientation'] is None or type(rgb['exif_orientation']) is int and rgb['exif_orientation']==1)
         and rgb['teacher_photometric_reconstruction'] is False
         and rgb['input_convention']=='RGB uint8 -> FP32 /255; common reviewed geometry'
         and rgb['baseline']=='N16 native_L selected by unchanged original GT-matched query ids'
         and rgb['image_source_control'] in ('base','separate_explicit_admitted_control'),
         'Original RGB/native-L receipt; not teacher F, strong-R L, or another photometric convention')
    need(type(provenance) is dict and set(provenance)=={'context','source_refs','input_refs'}
         and set(provenance['context'])=={'FIT_context','N13_qualified','N13_report'}
         and set(provenance['source_refs'])==set(ORIGINAL_READER_PINS)
         and all(provenance['source_refs'][key]['sha256']==value for key,value in ORIGINAL_READER_PINS.items()),
         'Retain original export source lineage metadata without reopening Windows inputs on DGX')
    # Original Windows refs remain provenance only. Root authenticates export and
    # transfer; this process opens only its two admitted portable data files.
    H,W=geo['original_shape']
    shapes=dict(rgb400=(1,3,400,400),e=(G,256),M=(G,200,200),L=(G,H,W),targets=(G,H,W))
    need(type(tensors) is dict and type(proofs) is dict and set(tensors)==set(proofs)==set(FIELDS),
         'Exactly five matched CPU fields; no hidden F/full100 carrier')
    pointers=set();checks={}
    for name in FIELDS:
        c.check();value=tensors[name];record=proofs[name]
        need(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.float32
             and value.layout==torch.strided and tuple(value.shape)==shapes[name]
             and value.storage_offset()==0 and value.is_contiguous() and not value.requires_grad and value.grad_fn is None
             and bool(torch.isfinite(value).all()),'Finite compact detached CPU FP32 field: '+name)
        logical=value.numel()*value.element_size();pointer=value.untyped_storage().data_ptr()
        need(value.untyped_storage().nbytes()==logical and pointer not in pointers,'Independent compact storage: '+name)
        pointers.add(pointer)
        digest=_value_digest(value)
        need(type(record) is dict and set(record)=={'shape','dtype','logical_bytes','source_sha256','sha256',
             'bitexact','independent_storage','compact_storage'} and record['shape']==list(shapes[name])
             and record['dtype']=='torch.float32' and record['logical_bytes']==logical
             and record['source_sha256']==record['sha256']==digest
             and all(record[k] is True for k in ('bitexact','independent_storage','compact_storage')),
             'Actual transferred value matches the exact U49 export-byte record: '+name)
        checks[name]=dict(shape=list(value.shape),dtype=str(value.dtype),logical_bytes=logical,sha256=digest)
    value=None
    need(bool(((tensors['targets']==0)|(tensors['targets']==1)).all()),'Original binary instance targets, not GT proxies')
    geometry=geometry_module.make_geometry(torch,ori_shape=tuple(geo['original_shape']),
        content_shape=tuple(geo['actual_img_shape']),canvas_size=800,branch_size=400)
    item=dict(tensors,G=G,query_ids=deepcopy(queries),identity=deepcopy(identity),geometry=geometry)
    row=dict(dataset_index=index,image_id=meta['image_id'],G=G,query_ids=deepcopy(queries),identity=deepcopy(identity))
    audit=dict(dataset_index=index,G=G,source_package=deepcopy(reference),tensor_checks=checks,
        original_metadata=deepcopy(meta),original_source_authentication='Root-owned U49 export acceptance and transfer required; not rederived here')
    package=None;c.check()
    return item,row,audit


class _PortableReader:
    """Only U39._update's consumed ABI; not the original F-validating reader."""
    def __init__(self,items,records):
        self.items=items;self.records=records;self.visits=[]

    def load(self,index):
        need(type(index) is int and index in EXPECTED_G,'Only the two prelocked original ordinals')
        self.visits.append(index)
        return self.items[index]


def _error(error):
    result=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    traceback.clear_frames(error.__traceback__);error.__traceback__=None
    return result


def run_pilot(torch,np,c,*,portable_refs,modules,source_refs,prelock,pilot_source_ref,report):
    """Return a JSON-compatible partial/completed report; never save weights.

    Caller supplies c.closed(strictref), c.check(), c.psutil and the eleven
    admitted modules. It must reject non-COMPLETED status, preserve real partial
    updates and provide actual resource/terminal ownership. No automatic retry.
    """
    need(type(report) is dict and not report,'Caller retains a fresh partial report')
    started=time.perf_counter();adapters=reader=items=records=None;outside_cpu=outside_cuda=None
    report.update(schema='U50-DGX-paired-disposable-pilot-report-pending-v1',status='STARTED',
        prelock=deepcopy(prelock),source_refs=deepcopy(source_refs),portable_refs=deepcopy(portable_refs),arms={},
        successful_optimizer_updates=0,optimizer_steps_entered=0,new_training=False,pilot_only=True,
        main_training=False,formal_training_budget_credit=0,U40_frontier_modified=False,
        U40_optimizer_or_RNG_resumed=False,model_weights_loaded=False,weights_written=False,
        AP=False,DEV=False,TEST=False,qualified=False,whole_population_exported=False,
        innovation_established=False,speedup_established=False,cross_environment_bitexact_claimed=False,
        same_seed_cross_environment_initialization_equal_claimed=False,automatic_retry=False,
        whole_GPU_observed_peak_MiB=None,whole_GPU_WDDM8GiB_contract_certified=False,
        whole_GPU_observation_scope='Independent GB10 UMA protocol remains with actual owner; unavailable is None, never zero')
    try:
        order=validate_prelock(prelock)
        actual=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=str(np.__version__),
                    system=platform.system(),machine=platform.machine())
        need(actual==prelock['environment'] and not torch.cuda.is_initialized() and torch.get_default_dtype()==torch.float32,
             'New prelocked DGX runtime, cold CUDA and FP32 default; never the old U40 runtime')
        need(torch.get_num_threads()==torch.get_num_interop_threads()==2 and c.psutil.Process().cpu_affinity()==[0,1]
             and torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
             and torch.backends.cudnn.allow_tf32 is False and torch.backends.cuda.matmul.allow_tf32 is False
             and not torch.is_autocast_enabled(),'Explicit CPU01/two threads/noAMP/noTF32/deterministic policy')
        need(callable(getattr(torch.autograd.graph,'save_on_cpu',None))
             and callable(getattr(torch._C,'_cuda_clearCublasWorkspaces',None)),
             'Original offload and CUDA-release interfaces must exist; no silent runtime fallback')
        report['actual_environment']=actual;c.check();_sources(c,modules,source_refs,pilot_source_ref)
        need(type(portable_refs) is dict and set(portable_refs)==set(EXPECTED_G),'Exactly two admitted portable files')
        outside_cpu=torch.get_rng_state().clone()
        items={};records={};report['portable_inputs']={}
        for index in (54,710):
            item,row,audit=_load_item(torch,c,index,portable_refs[index],modules['image_geometry'])
            items[index]=item;records[index]=row;report['portable_inputs'][str(index)]=audit
            item=row=audit=None
        reader=_PortableReader(items,records);items=records=None
        ops=modules['u39_training_ops'];helpers=modules['u36_helpers'];initializer=modules['shared_initialization']
        adapters,initialization=initializer.build_pair(torch,
            {key:modules[key] for key in initializer.SOURCE_SHA256},
            {key:source_refs[key] for key in initializer.SOURCE_SHA256})
        need(initialization['schema']=='U34-common-operation-initialization-v1'
             and initialization['seed']==initialization['adapter_seed']==prelock['initialization_seed']
             and len(initialization['common_state'])==104 and initialization['common_parameter_tensors']==68
             and initialization['common_BN_buffers']==36,'Actual fresh common104/adapter paired initialization in this runtime')
        report['initialization']=initialization;initialization=None;gc.collect();c.check()
        need(not torch.cuda.is_initialized(),'Portable loading and U34 initialization remain CPU only')
        torch.cuda.init();torch.cuda.set_device(0)
        properties=torch.cuda.get_device_properties(0);total=int(properties.total_memory);limit=prelock['limits']['allocator_bytes']
        need(total>=limit,'Device supports unchanged original 4GiB allocator cap')
        torch.cuda.set_per_process_memory_fraction(limit/float(total),device=0)
        need(torch.cuda.memory_allocated()==torch.cuda.memory_reserved()==0,'Cap precedes any model/input GPU allocation')
        report['CUDA_allocator_limit']=dict(limit_bytes=limit,device_total_bytes=total,fraction=limit/float(total),
            device_name=properties.name,set_before_model_or_input_transfer=True)
        outside_cuda=torch.cuda.get_rng_state(0).cpu().clone()
        for arm in ('PLAIN','FULL'):
            record=report['arms'][arm]={}
            # Fresh independent pilot streams, not states recovered from U40.
            # fork_rng stores CPU RNG snapshots and does not keep CUDA models.
            with torch.random.fork_rng(devices=[0]):
                torch.cuda.manual_seed(prelock['initialization_seed'])
                ops._run_arm(torch,c,reader,adapters,order,helpers,arm,record,report)
            need(record['status']=='PASS','Original U39 arm failed; preserve partial record and do not retry')
            need(record['updates_completed']==3 and [row['dataset_index'] for row in record['updates']]==order
                 and [row['G'] for row in record['updates']]==[1,27,1]
                 and sum(row['all_matched_GT'] for row in record['updates'])==29,
                 'Three original updates, all matched GT and 29 GT visits on this arm')
            c.check()
        need(not adapters and report['successful_optimizer_updates']==report['optimizer_steps_entered']==6
             and reader.visits==order+order,'Exactly six actual disposable updates on the same input sequence')
        report.update(status='COMPLETED_PENDING_ROOT_REVIEW',distinct_images=2,distinct_matched_GT=28,
            GT_visits_per_arm=29,total_GT_visits=58,image_order=order)
        _sources(c,modules,source_refs,pilot_source_ref)
    except BaseException as error:
        report.update(status='FAILED',error=_error(error))
    finally:
        adapters=reader=items=records=None;gc.collect()
        report['elapsed_seconds']=time.perf_counter()-started
        if torch.cuda.is_initialized():
            try:
                peaks=[int(torch.cuda.max_memory_allocated())]
                for record in report['arms'].values():
                    if 'before_first_image' in record:peaks.append(record['before_first_image']['CUDA_peak_allocated_bytes'])
                    for stage in record.get('updates',[]):
                        if 'after_image_release' in stage:peaks.append(stage['after_image_release']['CUDA_peak_allocated_bytes'])
                report['recorded_Torch_peak_allocated_bytes']=max(peaks)
                report['Torch_peak_scope']='Max of actual per-image reset intervals and arm-before-first samples; not whole-UMA physical peak'
                torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();torch.cuda.synchronize()
                report['final_CUDA_cleanup']=dict(allocated_bytes=int(torch.cuda.memory_allocated()),reserved_bytes=int(torch.cuda.memory_reserved()))
                need(report['final_CUDA_cleanup']==dict(allocated_bytes=0,reserved_bytes=0),'Final allocated/reserved CUDA zero')
                if outside_cuda is not None:
                    report['postinitialization_CUDA_RNG_preserved']=bool(torch.equal(outside_cuda,torch.cuda.get_rng_state(0).cpu()))
                    need(report['postinitialization_CUDA_RNG_preserved'],'Independent arm streams restore post-init caller CUDA RNG')
            except BaseException as error:
                report.update(status='FAILED',cleanup_error=_error(error))
        if outside_cpu is not None:
            report['caller_CPU_RNG_preserved']=bool(torch.equal(outside_cpu,torch.get_rng_state()))
            if not report['caller_CPU_RNG_preserved']:report['status']='FAILED'
        outside_cpu=outside_cuda=None
        if report['elapsed_seconds']>1800:report.update(status='FAILED',deadline_exceeded=True)
    json.dumps(report,allow_nan=False)
    return report
