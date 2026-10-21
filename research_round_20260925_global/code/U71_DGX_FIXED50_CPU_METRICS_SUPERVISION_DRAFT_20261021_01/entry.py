"""UNBOUND U71 CPU entry. Stdlib-only import; actual owner and Root gate external."""
from copy import deepcopy
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import traceback
import types

SCHEMA = 'ROOT-U71-DGX-fixed50-population-metrics-supervision-spec-v1'
OPERATION = 'CPU_U71_DGX_FIXED50_POPULATION_METRICS'
WORKER_PIN = dict(bytes=22009, sha256='efc944c1c8a8f839813669565c5b19e0439b945397a77fff88c5e4d83ca691c5')
LIMITS = dict(allowed_cpus=[0,1], threads=2, maximum_RSS_bytes=16*1024**3,
              minimum_available_RAM_bytes=16*1024**3, deadline_seconds=1800)
REQUEST_KEYS = {'worker_source','core_source','module_sources','library_sources','input_refs','shard_refs','output_directory'}
FALSE_FIELDS = ('qualified','TEST','new_training','ROOT_actual_terminal_self_certified','automatic_retry','full_binary_closure_claimed')
FLAGS = {'source_union_verified','source_bytes_before_worker','no_scientific_imports_before_worker',
         'worker_report_bytes_verified','summary_bytes_verified','private_pairs_bytes_verified',
         'source_bytes_after_worker','module_aliases_restored','child_resources_within_limits'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def signature(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def strict_ref(value, *, allow_empty=False):
    need(type(value) is dict and set(value) == {'path','bytes','sha256'} and type(value['path']) is str
         and '\\' not in value['path'] and PurePosixPath(value['path']).anchor == '/'
         and str(PurePosixPath(value['path'])) == value['path'] and '..' not in PurePosixPath(value['path']).parts
         and type(value['bytes']) is int and value['bytes'] >= (0 if allow_empty else 1)
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'), 'Strict canonical explicit reference')
    return value


def base_references(spec):
    request = spec['request']
    values = [spec['entry_source'],spec['owner_source'],request['worker_source'],request['core_source']]
    values += spec['external_shard_admission_refs']+request['shard_refs']
    for value in values:
        strict_ref(value)
    for key in ('module_sources','library_sources','input_refs'):
        need(type(request[key]) is dict, 'Explicit U70 reference roles')
        for role, value in request[key].items():
            strict_ref(value, allow_empty=key == 'library_sources' and role == 'pycocotools')
            values.append(value)
    return values


def validate_envelope(spec):
    need(type(spec) is dict and set(spec) == {'schema','operation','phase','entry_source','owner_source','request',
         'limits','sources','output_directory','owner_output_directory','external_shard_admission_refs'}
         and spec['schema'] == SCHEMA and spec['operation'] == OPERATION, 'Exact U71 CPU envelope')
    need(type(spec['phase']) is str and 1 <= len(spec['phase']) <= 160 and '/' not in spec['phase']
         and '\\' not in spec['phase'], 'Explicit finite phase identity')
    need(signature(spec['limits']) == signature(LIMITS), 'Original five CPU limits; no GPU allocator')
    request = spec['request']
    need(type(request) is dict and set(request) == REQUEST_KEYS, 'Original U70 seven-key request')
    need(type(spec['external_shard_admission_refs']) is list and len(spec['external_shard_admission_refs']) == 4
         and len({signature(v) for v in spec['external_shard_admission_refs']}) == 4,
         'Exactly four original external Root evidence refs in range order; no self qualification')
    need(type(request['shard_refs']) is list and len(request['shard_refs']) == 4, 'Four original U70 shard refs')
    need(all(request['worker_source'][k] == v for k,v in WORKER_PIN.items()), 'Exact frozen U70 worker bytes')
    outputs = []
    for value in (spec['output_directory'],spec['owner_output_directory'],request['output_directory']):
        need(type(value) is str and '\\' not in value and PurePosixPath(value).anchor == '/'
             and str(PurePosixPath(value)) == value and '..' not in PurePosixPath(value).parts and value != '/',
             'Canonical output directory')
        outputs.append(PurePosixPath(value))
    need(all(a != b and a not in b.parents and b not in a.parents
             for i,a in enumerate(outputs) for b in outputs[i+1:]), 'Three fresh separate non-nested outputs')
    admitted = {}
    need(type(spec['sources']) is list and spec['sources'], 'Finite actual source allowlist')
    for value in spec['sources']:
        strict_ref(value, allow_empty=True)
        need(value['path'] not in admitted, 'Unique source-table paths')
        admitted[value['path']] = deepcopy(value)
        path = PurePosixPath(value['path'])
        need(all(out != path and out not in path.parents for out in outputs), 'Outputs must not contain inputs')
    need(spec['sources'] == [admitted[k] for k in sorted(admitted)], 'Canonical sorted source table')
    for value in base_references(spec):
        need(admitted.get(value['path']) == value, 'Every explicit actual input must be proposed')
    return admitted


def closed_bytes(close, reference):
    path = close(reference)
    need(str(path) == reference['path'], 'Exact admitted actual path')
    raw = path.read_bytes()
    need(len(raw) == reference['bytes'] and hashlib.sha256(raw).hexdigest() == reference['sha256'], 'Exact admitted bytes')
    return raw


def load_closed(close, reference, alias):
    value = types.ModuleType(alias)
    value.__file__ = reference['path']
    exec(compile(closed_bytes(close, reference), reference['path'], 'exec'), value.__dict__)
    need(value.__file__ == reference['path'], 'Source-backed module identity')
    return value


def close_source_union(spec, closed):
    admitted = validate_envelope(spec)
    worker = load_closed(closed, spec['request']['worker_source'], '_U71_U70_METADATA')
    worker.validate_request(spec['request'])
    refs = worker.direct_references(spec['request'])+[spec['entry_source'],spec['owner_source']]+spec['external_shard_admission_refs']
    expected = {}
    for value in refs:
        need(value['path'] not in expected or expected[value['path']] == value, 'No conflicting union refs')
        expected[value['path']] = value
    need(expected == admitted and spec['sources'] == [expected[k] for k in sorted(expected)],
         'Exact U70 finite union plus entry/owner/four external evidence refs')
    return admitted


def summary_metadata(summary, pairs):
    need(type(summary) is dict and summary['status'] == 'METRICS_COMPLETE_PENDING_ROOT_ACCEPTANCE'
         and summary['qualified'] is summary['ROOT_actual_terminal_self_certified'] is False
         and summary['AP'] is summary['DEV'] is True and summary['TEST'] is summary['new_training'] is False
         and type(summary['formal_training_budget_credit']) is int and summary['formal_training_budget_credit'] == 0,
         'Original complete pending U67 summary scope')
    for key, value in (('images',187),('GT',380),('queries',18700)):
        need(type(summary[key]) is int and summary[key] == value, 'Complete population summary counters')
    selected = summary['geometry']['scope']['selected_native_nonignored_GT']
    need(type(selected) is int and 0 <= selected <= 380 and type(pairs) is list and len(pairs) == selected,
         'Original conditional matched private pair count')
    result={key:deepcopy(summary[key]) for key in ('status','qualified','ROOT_actual_terminal_self_certified','AP','DEV',
        'TEST','new_training','formal_training_budget_credit','images','GT','queries')}
    result['private_pair_count']=selected
    return result


def completed_entry(report, spec, spec_ref):
    validate_envelope(spec);strict_ref(spec_ref)
    need(report['schema'] == 'U71-CPU-population-metrics-entry-report-v1'
         and report['status'] == 'COMPLETE_PENDING_ROOT_ACCEPTANCE' and report['operation'] == OPERATION
         and report['phase'] == spec['phase'] and report['spec_ref'] == spec_ref, 'Exact pending CPU entry identity')
    for key in ('entry_source','owner_source','request','limits','sources','output_directory','owner_output_directory','external_shard_admission_refs'):
        need(signature(report[key]) == signature(spec[key]), 'Exact actual CPU input identity: '+key)
    need(set(report['flags']) == FLAGS and all(v is True for v in report['flags'].values())
         and report['cleanup_errors'] == [] and all(report[k] is False for k in FALSE_FIELDS)
         and report['AP'] is report['DEV'] is report['DEV_technical_input'] is True
         and report['AP_computation_status'] == 'COMPLETE_PENDING_ROOT_ACCEPTANCE'
         and type(report['formal_training_budget_credit']) is int and report['formal_training_budget_credit'] == 0,
         'Actual complete CPU metrics checks; no Root qualification')
    _resources(report['resource_observations'])
    worker=report['worker_report']
    need(worker['schema'] == 'U70-population-metrics-worker-report-v1' and worker['status'] == 'COMPLETE_PENDING_ROOT_ACCEPTANCE'
         and worker['AP'] is worker['DEV'] is worker['DEV_technical_input'] is True
         and worker['AP_computation_status'] == 'COMPLETE_PENDING_ROOT_ACCEPTANCE'
         and all(worker[k] is False for k in FALSE_FIELDS)
         and worker['CPU_only'] is True and worker['Torch_imported'] is False and worker['CUDA_visible_devices'] == ''
         and worker['cleanup_errors'] == [] and worker['source_aliases_restored'] is True
         and worker['outer_runtime_states_restored'] is True and worker['all_direct_input_bytes_rechecked_after'] is True
         and worker['annotation_JSON_parsed'] is worker['evaluation_started'] is True
         and type(worker['formal_training_budget_credit']) is int and worker['formal_training_budget_credit'] == 0
         and signature(worker['request']) == signature(spec['request']), 'Actual complete U70 CPU report')
    out=Path(spec['request']['output_directory'])
    for role,name in (('worker_report_ref','REPORT.json'),('summary_ref','SUMMARY.json'),('private_pair_records_ref','FIXED_NATIVE_PAIRS_PRIVATE.json')):
        value=report[role]
        need(type(value) is dict and value['path'] == str(out/name), 'Actual original output path: '+role)
        if role != 'worker_report_ref':
            need(value == worker[role], 'Actual U70 output reference: '+role)
    need(report['metrics_metadata']['AP'] is True and report['metrics_metadata']['private_pair_count'] == report['private_pair_record_count'],
         'Actual complete summary/private metadata projection')
    return worker


def execute_owned(spec, spec_ref, c, worker, *, report):
    need(type(report) is dict and not report, 'Fresh caller report')
    started=time.monotonic();primary=None;out=None;created=False
    report.update(schema='U71-CPU-population-metrics-entry-report-v1',status='STARTED',operation=OPERATION,
        worker_report={},flags={k:False for k in FLAGS},cleanup_errors=[],AP=None,AP_computation_status='WORKER_SCOPE_UNOBSERVED',
        DEV=True,DEV_technical_input=True,formal_training_budget_credit=0,**{k:False for k in FALSE_FIELDS})
    try:
        validate_envelope(spec);strict_ref(spec_ref)
        report.update(phase=spec['phase'],spec_ref=deepcopy(spec_ref))
        for key in ('entry_source','owner_source','request','limits','sources','output_directory','owner_output_directory','external_shard_admission_refs'):
            report[key]=deepcopy(spec[key])
        out=Path(spec['output_directory'])
        need(out.is_absolute() and out.resolve() == out and not out.exists() and not out.is_symlink() and out.parent.is_dir(),
             'Fresh canonical entry output')
        out.mkdir();created=True
        close_source_union(spec,c.closed);report['flags']['source_union_verified']=True
        for reference in spec['sources']:
            c.closed(reference)
        report['flags']['source_bytes_before_worker']=True
        report['external_evidence_refs_closed']=deepcopy(spec['external_shard_admission_refs'])
        need(no_scientific_imports(), 'No scientific imports before U70 admission')
        report['flags']['no_scientific_imports_before_worker']=True
        need(type(worker) is types.ModuleType and worker.__file__ == spec['request']['worker_source']['path'], 'Real closed U70 module')
        c.check();returned=worker.run_worker(c,spec['request'],report=report['worker_report'])
        need(returned is report['worker_report'], 'Original U70 caller-owned report')
        worker_out=Path(spec['request']['output_directory'])
        value,persisted=read_output(worker_out/'REPORT.json')
        need(signature(persisted) == signature(returned), 'Persisted actual U70 report equals return')
        report['worker_report_ref']=value;report['flags']['worker_report_bytes_verified']=True
        summary_ref,summary=read_output(worker_out/'SUMMARY.json')
        private_ref,pairs=read_output(worker_out/'FIXED_NATIVE_PAIRS_PRIVATE.json')
        need(summary_ref == returned['summary_ref'] and private_ref == returned['private_pair_records_ref'], 'Actual persisted summary/private bytes')
        report.update(summary_ref=summary_ref,private_pair_records_ref=private_ref,
            metrics_metadata=summary_metadata(summary,pairs),private_pair_record_count=len(pairs))
        report['flags']['summary_bytes_verified']=report['flags']['private_pairs_bytes_verified']=True
        del summary,pairs
        for reference in spec['sources']:
            c.closed(reference)
        report['flags']['source_bytes_after_worker']=True
        need(returned['source_aliases_restored'] is True, 'Original U70 aliases restored')
        report['flags']['module_aliases_restored']=True
        report['status']='COMPLETE_PENDING_FINAL_CHECK'
    except BaseException as error:
        primary=error
        report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        traceback.clear_frames(error.__traceback__);error.__traceback__=None
    finally:
        partial=report['worker_report']
        report['AP']=partial.get('AP')
        report['AP_computation_status']=partial.get('AP_computation_status','WORKER_SCOPE_UNOBSERVED')
        report['evaluation_started']=partial.get('evaluation_started')
        report['annotation_JSON_parsed']=partial.get('annotation_JSON_parsed')
        if created and 'worker_report_ref' not in report:
            try:
                path=Path(spec['request']['output_directory'])/'REPORT.json'
                if path.is_file():
                    value,persisted=read_output(path);report['worker_report_ref']=value
                    need(signature(persisted) == signature(partial), 'Actual failed U70 report equals partial evidence')
            except BaseException as error:
                report['cleanup_errors'].append(dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
                report['status']='FAILED'
                if primary is None:primary=error
        try:
            c.check();report['resource_observations']=c.summary();_resources(report['resource_observations'])
            report['flags']['child_resources_within_limits']=True
        except BaseException as error:
            report['status']='FAILED';report['cleanup_errors'].append(dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
            try:report['resource_observations']=c.summary()
            except BaseException as summary_error:report['cleanup_errors'].append(dict(type=type(summary_error).__name__,message=str(summary_error)))
            if primary is None:primary=error
        report['elapsed_seconds']=time.monotonic()-started
        if report['elapsed_seconds']>1800:
            report['status']='FAILED'
            if primary is None:primary=TimeoutError('CPU entry1800s deadline exceeded')
        if report['status']=='COMPLETE_PENDING_FINAL_CHECK':
            report['status']='COMPLETE_PENDING_ROOT_ACCEPTANCE'
            try:completed_entry(report,spec,spec_ref)
            except BaseException as error:
                report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
                if primary is None:primary=error
        if created:
            try:write_json(out/'REPORT.json',report)
            except BaseException as error:
                report.update(status='FAILED',report_write_error=dict(type=type(error).__name__,message=str(error)))
                if primary is None:primary=error
    if primary is not None:raise primary
    need(report['status']=='COMPLETE_PENDING_ROOT_ACCEPTANCE','Only actual complete CPU entry may return')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True);parser.add_argument('--spec-sha',required=True)
    args=parser.parse_args(argv)
    spec=spec_ref=None;validated=False;entered=False
    try:
        path=Path(args.spec)
        need(path.is_absolute() and path.is_file() and not path.is_symlink() and path.stat().st_size<=2*1024**2,
             'Explicit bounded actual spec')
        raw=path.read_bytes()
        need(hashlib.sha256(raw).hexdigest()==args.spec_sha,'Actual Root-pinned spec SHA')
        spec=json.loads(raw);spec_ref=dict(path=str(path),bytes=len(raw),sha256=args.spec_sha)
        admitted=validate_envelope(spec);validated=True
        need(platform.system()=='Linux' and platform.machine()=='aarch64' and platform.python_version()=='3.12.14',
             'Original actual Linux/aarch64 Python3.12.14 CPU entry')
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in
             ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')) and no_scientific_imports(),
             'Fresh noTorch/scientific imports and GPU-hidden CPU two-thread environment')
        need(spec['entry_source']['path']==str(Path(__file__).resolve(strict=True)),'This actual admitted CPU entry')

        def closed(reference):
            need(admitted.get(reference['path'])==reference,'Explicitly admitted actual file')
            return verify_file(reference)

        closed(spec['entry_source'])
        owner=load_closed(closed,spec['owner_source'],'_U71_CPU_OWNER')
        need(owner.SCHEMA==SCHEMA and owner.OPERATION==OPERATION and signature(owner.LIMITS)==signature(LIMITS),
             'Actual matching CPU owner/control ABI')
        close_source_union(spec,closed)  # Only U70 definitions before exact union.
        expected=spec['request']['library_sources']['psutil']
        closed(expected)
        psutil=importlib.import_module('psutil')
        need(type(psutil) is types.ModuleType and str(Path(psutil.__file__).resolve(strict=True))==expected['path']
             and str(psutil.__version__)=='6.1.1','Actual admitted psutil6.1.1')
        closed(expected)
        control=owner.make_control(spec,psutil)
        control.check()
        worker=load_closed(control.closed,spec['request']['worker_source'],'_U71_U70_WORKER')
        entered=True
        report=execute_owned(spec,spec_ref,control,worker,report={})
        return 0 if report['status']=='COMPLETE_PENDING_ROOT_ACCEPTANCE' else 1
    except BaseException as error:
        visible=traceback.format_exc()
        if validated and not entered:
            try:
                out=Path(spec['output_directory'])
                if out.is_absolute() and out.resolve()==out and not out.exists() and not out.is_symlink() and out.parent.is_dir():
                    out.mkdir()
                    write_json(out/'REPORT.json',dict(schema='U71-CPU-population-metrics-entry-report-v1',status='FAILED',
                        spec_ref=spec_ref,phase=spec['phase'],operation=OPERATION,qualified=False,AP=None,
                        AP_computation_status='WORKER_NOT_ENTERED',DEV=True,TEST=False,new_training=False,
                        formal_training_budget_credit=0,ROOT_actual_terminal_self_certified=False,
                        error=dict(type=type(error).__name__,message=str(error),traceback=visible)))
            except BaseException:
                pass  # Original exception remains primary; actual owner retains stderr.
        raise





def no_scientific_imports():
    return not any(name.split('.')[0] in ('torch','numpy','pycocotools','PIL') for name in sys.modules)


def write_json(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode('utf-8')
    with path.open('xb') as stream:
        need(stream.write(raw)==len(raw),'Complete entry report write');stream.flush();os.fsync(stream.fileno())
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def read_output(path):
    need(path.is_absolute() and path.resolve(strict=True)==path and not path.is_symlink(),'Canonical actual output')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    key=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    need(key(before)==key(after) and len(raw)==after.st_size,'Stable complete output bytes')
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),json.loads(raw)


def _resources(observed):
    need(type(observed) is dict and observed['schema']=='U71-Linux-child-control-observations-v1'
        and type(observed['samples']) is int and observed['samples']>0 and observed['violations']==[]
        and type(observed['peak_RSS_bytes']) is int and 0<=observed['peak_RSS_bytes']<=LIMITS['maximum_RSS_bytes']
        and type(observed['minimum_available_RAM_bytes']) is int
        and observed['minimum_available_RAM_bytes']>=LIMITS['minimum_available_RAM_bytes'],
        'Actual child resource samples within unchanged limits')
    elapsed=observed['elapsed_seconds']
    need(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<=1800,'Bounded actual child duration')


def verify_file(value):
    """Stdlib bootstrap for explicitly selected small source/metadata refs only."""
    strict_ref(value)
    path=Path(value['path'])
    need(path.is_absolute() and path.resolve(strict=True)==path and not path.is_symlink(),'Canonical bootstrap file')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    key=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    need(key(before)==key(after) and len(raw)==value['bytes'] and hashlib.sha256(raw).hexdigest()==value['sha256'],
        'Stable pinned bootstrap bytes')
    return path


if __name__=='__main__':
    sys.exit(main())
