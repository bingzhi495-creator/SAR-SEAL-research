"""Owned N21 FIT metadata geometry only; pending until external actual0 acceptance."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

def need(ok,message):
    if not ok:raise AssertionError(message)
def read(path):
    return json.loads(Path(path).read_bytes())
def ref(path,b=None):
    path=Path(path);b=path.read_bytes() if b is None else b
    return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):
    need(type(r) is dict and set(r)=={'path','bytes','sha256'} and Path(r['path']).is_absolute(),'Exact reference shape')
    b=Path(r['path']).read_bytes()
    need(len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],'N21 immutable input: '+r['path'])
    return b
def write(path,value):
    b=(json.dumps(value,ensure_ascii=True,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf8')
    with Path(path).open('xb') as f:f.write(b)
    need(Path(path).read_bytes()==b,'N21 output write readback')
    return ref(path,b)

def run(path):
    import psutil
    started=time.monotonic();process=psutil.Process();peak=0;minimum=None
    def check():
        nonlocal peak,minimum
        rss=process.memory_info().rss;available=psutil.virtual_memory().available
        peak=max(peak,rss);minimum=available if minimum is None else min(minimum,available)
        need(process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')==''
             and all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
             and rss<16*1024**3 and available>=16*1024**3 and time.monotonic()-started<1800
             and 'torch' not in sys.modules and 'numpy' not in sys.modules,'N21 declared CPU/RAM/deadline/no-tensor scope')
    check();path=Path(path)
    need(path.is_absolute(),'Absolute N21 spec')
    spec=read(path)
    need(spec['schema']=='N21-FIT-native-support-geometry-owned-spec-v1' and spec['owner_index']==221
         and spec['normal_RAW_count']==221 and spec['total_failed_RAW_count']==27
         and spec['total_terminal_identity_pin_count']==248 and spec['legacy_pin_count']==229
         and spec['automatic_retry'] is False and spec['deadline_seconds']==1800
         and spec['maximum_process_RSS_bytes']==16*1024**3 and spec['minimum_available_RAM_bytes']==16*1024**3
         and spec['whole_GPU_limit_MiB']==8192 and spec['allowed_cpus']==[0,1]
         and spec['thread_limit']==2 and spec['cuda_visible_devices']==''
         and spec['seed']==202610170301 and spec['images']==32 and spec['centers_per_image']==64,'Locked N21 current owner and protocol')
    need(ref(__file__)==spec['entry_source'],'N21 actual source entry')
    for r in spec['sources']:checked(r);check()
    ad=read(spec['CPU_admission_file'])
    need(len(ad)==13 and ad['phase']==spec['phase'] and ad['scientific_spec']==ref(path)
         and ad['entry']==spec['entry_source']['path'] and ad['entry_args']==['--spec',str(path),'--owner-execute']
         and ad['allowed_cpus']==[0,1] and ad['cuda_visible_devices']=='' and ad['thread_limit']==2
         and ad['qualified_leaf_deadline_seconds']==1800 and ad['automatic_retry'] is False
         and ad['whole_GPU_used_limit_MiB']==8192
         and ad['bound_source_sha256']=={r['path']:r['sha256'] for r in spec['sources']+[ref(path)]},'Own immutable CPU admission')
    prelock=json.loads(checked(spec['prelock']))
    need(prelock['schema']=='N21-native-support-FIT-geometry-prelock-v1'
         and prelock['seed']==spec['seed'] and prelock['images']==32 and prelock['centers_per_image']==64
         and prelock['dataset_report']==spec['dataset_report'] and prelock['mapping_sources']==spec['mapping_sources']
         and prelock['parent_ROOT_qualified']==spec['parent_ROOT_qualified']
         and prelock['shared_cell_threshold']==dict(numerator=4,denominator=5)
         and prelock['minimum_main_fraction']==dict(numerator=1,denominator=2)
         and prelock['native_shape']==[200,200] and prelock['canvas_shape']==[800,800]
         and prelock['align_corners'] is False and prelock['antialias'] is False
         and prelock['crop_origin']==[0,0] and prelock['result_read'] is False,'Original pre-result lock unchanged')
    parent=json.loads(checked(spec['parent_ROOT_qualified']))
    need(parent['qualified'] is True and parent['normal_RAW_count']==221 and parent['total_failed_RAW_count']==27
         and parent['total_terminal_identity_pin_count']==248 and parent['legacy_pin_count']==229
         and parent['primary_interpretation']['decision']=='STOP_CURRENT_RECIPE'
         and parent['pure221_actual_exit_code']==0,'Completed parent and N20 stop, not rescue')
    output=Path(spec['output_directory'])
    need(output.is_absolute() and not output.exists() and not output.is_symlink()
         and output.parent.is_dir() and psutil.disk_usage(str(output.parent)).free>=1024**3,'Fresh N21 output with free1GiB')
    report=json.loads(checked(spec['dataset_report']))
    need(type(report['rows']) is list and len(report['rows'])==741,'All original FIT rows, no subsampling by content')
    rows=[]
    for row in report['rows']:
        identity=row['identity']
        need(type(row['image_id']) in (int,str) and row['image_id']==identity['image_id']
             and type(row['dataset_index']) is int and identity['batch_input_shape']==[800,800],
             'Actual FIT row ID and observed padded canvas')
        rows.append(dict(image_key=str(row['image_id']),dataset_index=row['dataset_index'],
            native_shape=[200,200],canvas_shape=[800,800],crop_shape=identity['actual_img_shape'],image_shape=identity['original_shape']))
    need(len({r['image_key'] for r in rows})==741 and {r['dataset_index'] for r in rows}==set(range(741)),
         'No FIT identity collisions or missing indices')
    del report;gc.collect();check()
    module_path=Path(spec['fit_geometry_source']['path'])
    need(module_path.parent==Path(__file__).parent,'Bound geometry modules reside in own source directory')
    sys.path.insert(0,str(module_path.parent))
    module_spec=importlib.util.spec_from_file_location('N21_owned_fit_geometry',module_path)
    geometry=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(geometry)
    selected=geometry.select_rows(rows,seed=spec['seed'],count=32)
    need(len(selected)==32 and all(len(geometry.unique_centers(r,seed=spec['seed'],cap=64))==min(64,r['image_shape'][0]*r['image_shape'][1]) for r in selected),
         'Exactly32 fixed FIT rows and fixed center cap64')
    analysis=geometry.analyze_rows(selected,seed=spec['seed']);check()
    need(set(analysis)=={'cases','summary','decision'} and len(analysis['cases'])==32,'Complete registered result ABI')
    summary=analysis['summary'];all_count=summary['all']['denominator'];main=summary['no_any_clamp']['denominator']
    same=summary['no_any_clamp']['same_cell_count']
    need(all_count==sum(min(64,r['image_shape'][0]*r['image_shape'][1]) for r in selected)
         and summary['clamped']['denominator']+main==all_count,'All registered groups retained')
    expected='INCONCLUSIVE_CLAMP_COVERAGE' if 2*main<all_count else (
        'GEOMETRY_PREMISE_SUPPORTED' if 5*same>=4*main else 'SHARED_FOUR_PREMISE_REJECTED')
    need(analysis['decision']==expected,'Exact prelocked50/80 branch, no posthoc threshold')
    output.mkdir();details_ref=write(output/'DETAILS.json',analysis);check()
    value=dict(schema='N21-FIT-native-support-geometry-pending-result-v1',
        status='N21_FIT_IDEAL_GEOMETRY_COMPLETE_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
        source=spec['entry_source'],spec=ref(path),CPU_admission=ref(spec['CPU_admission_file']),phase=spec['phase'],
        prelock=spec['prelock'],dataset_report=spec['dataset_report'],mapping_sources=spec['mapping_sources'],
        parent_ROOT_qualified=spec['parent_ROOT_qualified'],details=details_ref,images=32,groups=all_count,
        seed=spec['seed'],centers_per_image=64,selected_image_keys=[r['image_key'] for r in selected],
        summary=summary,decision=expected,normal_RAW_count=221,total_failed_RAW_count=27,total_terminal_identity_pin_count=248,
        qualified=False,current_owned_terminal_self_certified=False,ROOT_actual0_verified=False,
        tensor_loaded=False,GT_or_target_tensors_loaded=False,GT_or_target_values_used_for_selection_or_decision=False,
        dataset_report_metadata_loaded=True,image_pixels_loaded=False,DEV_or_TEST_loaded=False,
        actual_FP32_feature_rank_measured=False,LN_feature_rank_measured=False,task_efficacy_established=False,
        innovation_established=False,TGRS_readiness_established=False,goal_complete=False,
        scope=prelock['scope'],point_distribution=prelock['point_distribution'],
        peak_observed_process_RSS_bytes=peak,minimum_observed_available_RAM_bytes=minimum,
        elapsed_seconds=time.monotonic()-started)
    for r in spec['sources']:checked(r);check()
    qref=write(output/'Q.json',value);check()
    print(json.dumps(dict(status=value['status'],Q=qref,qualified=False)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();need(args.owner_execute,'One Root-owned scientific execution only');run(args.spec)
