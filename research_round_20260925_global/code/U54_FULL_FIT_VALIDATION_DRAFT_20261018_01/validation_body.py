"""UNBOUND full741 portable CPU validation; no ML imports, I/O or execution on import."""
import gc
import hashlib
import json
import time
import weakref


SCHEMA='U54-full-FIT-CPU-validation-body-report-v1'
LIMITS=dict(allowed_cpus=[0,1],threads=2,maximum_RSS_bytes=16*1024**3,
            minimum_available_RAM_bytes=16*1024**3,deadline_seconds=1800)
FLAGS={'CUDA_uninitialized_before','CUDA_uninitialized_after','metadata_population_verified',
       'all_field_proofs_verified','all_returned_tensor_objects_released'}
FALSE_FIELDS=('qualified','new_training','AP','DEV','TEST','main_training','model_constructed',
              'model_forward_executed','optimizer_created','GT_matching_recomputed')
FIELDS=('rgb400','e','M','L','targets')


def need(ok,message):
    if not ok:raise ValueError(message)


def prelock_contract():
    return dict(schema='U54-full-FIT-CPU-validation-prelock-v1',
        environment=dict(python='3.12.14',torch='2.10.0+cu130',numpy='1.26.4',system='Linux',machine='aarch64'),
        dataset_indices=list(range(741)),images=741,GT=1661,CUDA_visible_devices='',limits=dict(LIMITS),
        no_new_training=True,no_forward=True,no_matching=True,no_AP=True,no_DEV=True,no_TEST=True)


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _metadata_sha(value):
    return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


def validate_prelock(prelock):
    need(type(prelock) is dict and _canonical(prelock)==_canonical(prelock_contract()),
         'Exact CPU-only741/1661 independent DGX validation prelock; no training or CUDA')


def completed_population(report):
    """Complete worker metadata guard, never self-grants actual Root acceptance."""
    need(type(report) is dict and report.get('schema')==SCHEMA and report.get('status')=='COMPLETE_PENDING_ROOT'
         and all(type(report.get(k)) is int and report[k]==741
                 for k in ('attempted_images','validated_images','released_images','recorded_images'))
         and type(report.get('matched_instances')) is int and report['matched_instances']==1661
         and type(report.get('reader_failed_loads')) is int and report['reader_failed_loads']==0
         and type(report.get('reader_visits')) is list and all(type(i) is int for i in report['reader_visits'])
         and report['reader_visits']==list(range(741))
         and set(report.get('flags',{}))==FLAGS and all(v is True for v in report['flags'].values())
         and all(report.get(k) is False for k in FALSE_FIELDS)
         and type(report.get('formal_training_budget_credit')) is int and report['formal_training_budget_credit']==0,
         'All741 actual loads/audits/releases and1661 matched targets, no partial or numerical-training qualification')


def run(torch,c,manifest,*,reader_module,image_geometry,source_refs,prelock,report,record_image):
    """Actual future owner invokes once; retain only scalar audit via callback.

    The caller holds the same report object across failures and owns source-
    closed module loading, private JSONL, resource monitoring and actual exit.
    """
    need(type(report) is dict and not report and callable(record_image),'Fresh caller report and audit callback')
    validate_prelock(prelock)
    started=time.monotonic();reader=item=None;live=[]
    flags={name:False for name in FLAGS}
    report.update(schema=SCHEMA,status='STARTED',flags=flags,
        attempted_images=0,validated_images=0,released_images=0,recorded_images=0,
        matched_instances=0,reader_visits=[],reader_failed_loads=0,
        original_metadata_population=dict(images=741,GT=1661,tensor_population_revalidated=False),
        manifest_sha256=_metadata_sha(manifest),record_metadata_sha256=None,
        private_scalar_audit_only=True,original_annotation_records_parsed=False,
        FIT_GT_tensor_read_attempted=False,FIT_GT_tensors_read=False,
        formal_training_budget_credit=0,**{name:False for name in FALSE_FIELDS})
    try:
        c.check();need(not torch.cuda.is_initialized(),'CUDA cold before CPU-only input verification')
        flags['CUDA_uninitialized_before']=True
        reader=reader_module.prepare(torch,c,manifest,image_geometry=image_geometry,source_refs=source_refs)
        need(len(reader.records)==741 and [row['dataset_index'] for row in reader.records]==list(range(741))
             and sum(row['G'] for row in reader.records)==1661,'Exact full original metadata population')
        records_sha=_metadata_sha(reader.records);report['record_metadata_sha256']=records_sha
        flags['metadata_population_verified']=True
        for index in range(741):
            c.check();report['attempted_images']+=1
            report['FIT_GT_tensor_read_attempted']=True
            item=reader.load(index)
            need(type(item) is dict and set(item)==set(FIELDS)|{'G','query_ids','identity','geometry'},
                 'Only U39-consumed matched fields, no extra F or cached whole population')
            live=[weakref.ref(item[name]) for name in FIELDS]
            record=reader.records[index]
            need(item['G']==record['G'] and item['query_ids']==record['query_ids']
                 and item['identity']==record['identity'] and not torch.cuda.is_initialized(),
                 'Original complete matched GT/query identities and CPU-only numerical field verification')
            audit=reader.last_audit
            need(audit['status']=='LOADED' and audit['dataset_index']==index and audit['G']==item['G']
                 and set(audit['tensor_checks'])==set(FIELDS),'Actual reader returned all five validated field proofs')
            scalar=dict(dataset_index=index,status='VALIDATED',G=item['G'],
                original_record_sha256=_metadata_sha(record),source_package=audit['source_package'],
                original_package=audit['original_package'],export_qualified=audit['export_qualified'],
                tensor_checks=audit['tensor_checks'],CUDA_initialized=False,returned_tensor_objects_released=False)
            _canonical(scalar)  # Reject tensor/model objects or nonfinite audit values before callback.
            report['validated_images']+=1;report['matched_instances']+=item['G']
            # A successful load returned GT; False does not prove no failed I/O.
            report['FIT_GT_tensors_read']=True
            item=None;gc.collect()
            need(all(reference() is None for reference in live),'All current image tensor objects released before next load')
            report['released_images']+=1
            scalar['returned_tensor_objects_released']=True
            record_image(scalar)
            report['recorded_images']+=1
            live=[];audit=scalar=record=None
            c.check()
        need(_metadata_sha(reader.records)==records_sha,'Original population records unchanged during all loads')
        report['reader_visits']=list(reader.visits);report['reader_failed_loads']=reader.failed_loads
        flags['all_field_proofs_verified']=report['validated_images']==741
        flags['all_returned_tensor_objects_released']=report['released_images']==741
        flags['CUDA_uninitialized_after']=not torch.cuda.is_initialized()
        c.check();report['status']='COMPLETE_PENDING_ROOT'
        completed_population(report)
        return report
    except BaseException as error:
        report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        if reader is not None:
            report['reader_visits']=list(reader.visits);report['reader_failed_loads']=reader.failed_loads
        item=reader=None;gc.collect()
        report['remaining_returned_tensor_objects']=sum(reference() is not None for reference in live)
        flags['CUDA_uninitialized_after']=not torch.cuda.is_initialized()
        report['cost']=dict(elapsed_seconds=time.monotonic()-started)
