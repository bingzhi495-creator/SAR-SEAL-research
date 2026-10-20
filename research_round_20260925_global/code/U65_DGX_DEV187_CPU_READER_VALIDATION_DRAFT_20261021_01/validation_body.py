"""UNBOUND full187/full100 portable DEV CPU input validation; no ML imports."""
import gc
import hashlib
import json
import time
import weakref


SCHEMA='U65-DEV187-CPU-reader-validation-body-report-v1'
LIMITS=dict(allowed_cpus=[0,1],threads=2,maximum_RSS_bytes=16*1024**3,
            minimum_available_RAM_bytes=16*1024**3,deadline_seconds=1800)
FLAGS={'CUDA_uninitialized_before','CUDA_uninitialized_after','metadata_population_verified',
       'all_field_proofs_verified','all_returned_tensor_objects_released'}
FALSE_FIELDS=('qualified','new_training','AP','DEV','TEST','main_training','model_constructed',
              'model_forward_executed','optimizer_created','GT_matching_recomputed','GT_rasterization_executed')
FIELDS=('rgb400','e','M','L')
READER_SOURCE_ROLES=('dev_reader','dev_rgb_reader','image_geometry')


def need(ok,message):
    if not ok:raise ValueError(message)


def prelock_contract():
    return dict(schema='U65-DEV187-CPU-reader-validation-prelock-v1',
        environment=dict(python='3.12.14',torch='2.10.0+cu130',numpy='1.26.4',system='Linux',machine='aarch64'),
        dataset_indices=list(range(187)),images=187,queries=18700,CUDA_visible_devices='',limits=dict(LIMITS),
        no_new_training=True,no_forward=True,no_matching=True,no_AP=True,no_DEV=True,no_TEST=True,
        no_GT_rasterization=True,DEV_technical_input=True)


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _metadata_sha(value):
    return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


def validate_prelock(prelock):
    need(type(prelock) is dict and _canonical(prelock)==_canonical(prelock_contract()),
         'Exact CPU-only187/18700 independent DGX input prelock; no model, training, evaluation or CUDA')


def completed_population(report):
    """Complete worker metadata guard, never self-grants original Root acceptance."""
    need(type(report) is dict and report.get('schema')==SCHEMA and report.get('status')=='COMPLETE_PENDING_ROOT'
         and all(type(report.get(k)) is int and report[k]==187
                 for k in ('attempted_images','validated_images','released_images','recorded_images'))
         and type(report.get('queries')) is int and report['queries']==18700
         and type(report.get('reader_failed_loads')) is int and report['reader_failed_loads']==0
         and type(report.get('reader_visits')) is list and all(type(i) is int for i in report['reader_visits'])
         and report['reader_visits']==list(range(187))
         and type(report.get('remaining_returned_tensor_objects')) is int and report['remaining_returned_tensor_objects']==0
         and report.get('original_metadata_population')==dict(images=187,queries=18700,tensor_population_revalidated=False)
         and report.get('original_annotation_records_parsed') is False
         and report.get('raw_provenance_inputs_revalidated') is False
         and set(report.get('flags',{}))==FLAGS and all(v is True for v in report['flags'].values())
         and all(report.get(k) is False for k in FALSE_FIELDS) and report.get('DEV_technical_input') is True
         and type(report.get('formal_training_budget_credit')) is int and report['formal_training_budget_credit']==0,
         'All187 actual input loads/audits/releases and18700 queries; no partial, evaluation or training qualification')


def validate_loaded_metadata(index,item,row,originals,entry):
    """Join the genuine U59 three-value return without copying/decoding masks.

    Real tensor proofs, identity schema, native dictionaries and geometry are
    validated inside unchanged U59 load. This checks only the consumed ABI and
    canonical ordinal joins; it does not invent a replacement field proof.
    """
    need(type(index) is int and 0<=index<187 and type(item) is dict
         and set(item)==set(FIELDS)|{'identity','query_ids','geometry'},
         'Only U59 full100 CPU fields and minimal metadata; no F/targets/GT fields')
    need(type(row) is dict and set(row)=={'dataset_index','image_id','qids'}
         and type(entry) is dict and set(entry)=={'dataset_index','image_id','query_ids','package_ref'}
         and type(row['dataset_index']) is int and row['dataset_index']==entry['dataset_index']==index
         and type(row['image_id']) is int and row['image_id']==entry['image_id']
         and type(item['identity']) is dict and item['identity'].get('image_id')==row['image_id'],
         'Original full187 manifest/reader image and ordinal identities')
    qids=item['query_ids']
    need(type(qids) is list and len(qids)==100 and all(type(qid) is int for qid in qids)
         and set(qids)==set(range(100)) and qids==row['qids']==entry['query_ids'],
         'Natural tensors retain distinct original full100 export permutation')
    need(type(originals) is list and len(originals)==100
         and all(type(value) is dict and type(value.get('image_id')) is int and value['image_id']==row['image_id']
                 and type(value.get('query_id')) is int and value['query_id']==qid for qid,value in zip(qids,originals)),
         'All original100 native dictionaries stay in their original order')


def run(torch,c,manifest,*,reader_module,image_geometry,source_refs,prelock,report,record_image):
    """Root-owned future CPU leaf; entry preserves caller runtime/RNG state.

    Source-closed real U59 is injected. Windows provenance inside its packages
    remains historical metadata: this path does not re-open or re-authenticate
    original JPEG/bundle/packet/annotation/teacher inputs on DGX.
    """
    need(type(report) is dict and not report and callable(record_image),'Fresh caller report and audit callback')
    validate_prelock(prelock)
    need(type(manifest) is dict and type(manifest.get('entries')) is list and len(manifest['entries'])==187,
         'U64 full187 real portable entries required')
    need(type(source_refs) is dict and all(role in source_refs for role in READER_SOURCE_ROLES),
         'Three original reader/geometry refs within the caller source closure')
    started=time.monotonic();reader=item=row=originals=loaded=None;live=[]
    flags={name:False for name in FLAGS}
    report.update(schema=SCHEMA,status='STARTED',flags=flags,
        attempted_images=0,validated_images=0,released_images=0,recorded_images=0,queries=0,
        reader_visits=[],reader_failed_loads=0,remaining_returned_tensor_objects=0,
        original_metadata_population=dict(images=187,queries=18700,tensor_population_revalidated=False),
        manifest_sha256=_metadata_sha(manifest),record_metadata_sha256=None,
        private_scalar_audit_only=True,original_annotation_records_parsed=False,
        raw_provenance_inputs_revalidated=False,DEV_technical_input=True,
        formal_training_budget_credit=0,**{name:False for name in FALSE_FIELDS})
    try:
        c.check();need(not torch.cuda.is_initialized(),'CUDA cold before CPU-only input verification')
        flags['CUDA_uninitialized_before']=True
        reader=reader_module.prepare(torch,c,manifest['entries'],image_geometry=image_geometry,
            source_refs={role:source_refs[role] for role in READER_SOURCE_ROLES})
        need(len(reader.records)==187 and [record['dataset_index'] for record in reader.records]==list(range(187))
             and len({record['image_id'] for record in reader.records})==187,'Exact full187 original metadata population')
        records_sha=_metadata_sha(reader.records);report['record_metadata_sha256']=records_sha
        flags['metadata_population_verified']=True
        for index in range(187):
            c.check();report['attempted_images']+=1;report['reader_visits'].append(index)
            try:
                loaded=reader.load(index)
            except BaseException:
                report['reader_failed_loads']+=1
                raise
            need(type(loaded) is tuple and len(loaded)==3,'Original U59 load returns item,row,originals tuple')
            item,row,originals=loaded;loaded=None
            validate_loaded_metadata(index,item,row,originals,manifest['entries'][index])
            need(all(isinstance(item[name],torch.Tensor) for name in FIELDS),'All four genuine U59-returned tensors')
            live=[weakref.ref(item[name]) for name in FIELDS]
            report['remaining_returned_tensor_objects']=len(live)
            record=reader.records[index]
            need(_canonical(row)==_canonical(record) and not torch.cuda.is_initialized(),
                 'Original U59 canonical row and CPU-only field verification')
            scalar=dict(dataset_index=index,image_id=row['image_id'],queries=100,status='VALIDATED',
                source_package=dict(manifest['entries'][index]['package_ref']),
                original_record_sha256=_metadata_sha(record),
                native_predictions_metadata_sha256=_metadata_sha(originals),
                query_ids_metadata_sha256=_metadata_sha(item['query_ids']),
                CUDA_initialized=False,returned_tensor_objects_released=False)
            _canonical(scalar)  # Never pass tensors, geometry or full native masks to the callback.
            report['validated_images']+=1;report['queries']+=100
            item=row=originals=None;gc.collect()
            need(all(reference() is None for reference in live),'All four current-image tensor objects released before next load')
            report['released_images']+=1;report['remaining_returned_tensor_objects']=0
            scalar['returned_tensor_objects_released']=True
            record_image(scalar)
            report['recorded_images']+=1
            live=[];scalar=record=None
            c.check()
        need(_metadata_sha(reader.records)==records_sha,'Original population records unchanged during all loads')
        flags['all_field_proofs_verified']=report['validated_images']==187
        flags['all_returned_tensor_objects_released']=report['released_images']==187
        flags['CUDA_uninitialized_after']=not torch.cuda.is_initialized()
        c.check();report['status']='COMPLETE_PENDING_ROOT'
        completed_population(report)
        return report
    except BaseException as error:
        report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        item=row=originals=loaded=reader=None;gc.collect()
        report['remaining_returned_tensor_objects']=sum(reference() is not None for reference in live)
        flags['CUDA_uninitialized_after']=not torch.cuda.is_initialized()
        report['cost']=dict(elapsed_seconds=time.monotonic()-started)
