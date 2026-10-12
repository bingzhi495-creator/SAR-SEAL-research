"""Inert future ROOT authoring of own qualified joint first24 weights plus fresh48.

Original four raws, native-restoration/tiny priors, tiny and own first24 actual
raws precede every hash, qualification read and provenance SOURCE execution.
Only metadata/SOURCE are interpreted; no model/CP load, GT decode, scientific
library, launch, AUTH, retry or selection. ROOT must separately confirm actual
first24 CPU tool terminal0: a qualification JSON is not operating-system evidence.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import types

PHASE='query_pixel_joint_native_E24warm_additional48_20261011_02'
PREFIX='ROOT_QUERY_PIXEL_JOINT_NATIVE_E24WARM_ADDITIONAL48'
FIRST24_PHASE='query_pixel_joint_native_fullFIT24_20261011_02'
FIRST24_PREFIX='query-pixel-joint-native-fullFIT24'
MODES=('native0','global','spatial_halves','spatial_reversed')
OWNER_SHA='ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
ENTRY_SHA='65d72eecff1e7614cfdbc3d67fc2b2272e2661185ebe06e205c9ee616efa8783'
READER_SHA='c946b4265139ee6e7d18e9287bec0c96447d1a6062074672e45cf749b4400afe'
FIRST24_ENTRY_SHA='6c9103a2d9507332d6396897f564474ca0cc9ba371c45b76a6a775eceed47e86'
FIRST24_READER_SHA='154a320a6ea6b00f6f0aa883f3274911e8f9e44797fb4d3001467c02282226ac'


def require(ok,message):
    if not ok: raise RuntimeError(message)


def read(path):
    require(Path(path).is_absolute(),'Absolute JSON locator required')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path=Path(path).resolve(strict=True); digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def bound(ref):
    require(isinstance(ref,dict) and set(ref)=={'path','bytes','sha256'} and Path(ref['path']).is_absolute() and
        type(ref['bytes'])is int and ref['bytes']>0 and isinstance(ref['sha256'],str) and len(ref['sha256'])==64 and
        set(ref['sha256'])<=set('0123456789abcdef') and reference(ref['path'])==ref,'Pinned input differs')
    return Path(ref['path'])


def load(ref,name):
    path=bound(ref); source=path.read_bytes()
    require(path.suffix=='.py' and len(source)==ref['bytes'] and hashlib.sha256(source).hexdigest()==ref['sha256'],'Verified SOURCE changed')
    module=types.ModuleType(name); module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module


def terminal(raw,phase):
    require(raw.get('status')=='ACTUAL_OWNED_TERMINAL' and raw.get('original_identity')is not None and
        type(raw.get('returncode'))is int and raw['returncode']==0 and raw.get('root_wait_reaped')is True and
        raw.get('held_handle_exit')==dict(wait_state='WAIT_OBJECT_0',actual_exit_code=0) and
        raw.get('numeric_birth_absence',{}).get('original_birth_absent')is True and
        all(k in raw and raw[k]is None for k in ('abort','error')) and raw.get('wait_errors')is False and
        raw.get('GPU_observation_errors')==[],'STOP before hashes/imports: actual owned raw0 required')
    peak=raw.get('whole_GPU_observed_peak_MiB')
    require(raw.get('phase',phase)==phase and type(peak)in(int,float) and math.isfinite(peak) and 0<=peak<=8192 and
        raw.get('whole_GPU_used_limit_MiB')==8192 and raw.get('whole_leaf_VRAM_certified')is False and
        raw.get('GPU_scope')=='whole GPU observed, not per-process WDDM allocation','Raw scope differs')


def union_refs(rows):
    refs={}
    for ref in rows:
        require(isinstance(ref,dict) and set(ref)=={'path','bytes','sha256'} and
            (ref['path'] not in refs or refs[ref['path']]==ref),'Conflicting closure reference')
        refs[ref['path']]=ref
    return [refs[path] for path in sorted(refs)]


def write_fresh(path,value):
    require(path.is_absolute() and path.parent.is_dir() and not path.exists(),'Fresh absolute output required')
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')


def build(args):
    c=Path(__file__).resolve(strict=True).parent.parent; h=Path(args.artifact_root)
    require(h.is_absolute() and h.resolve(strict=True)==c.parent/'goal_continuation_20261005','Canonical absolute artifact root required')
    h=h.resolve(strict=True)
    paths={k:h/(PREFIX+'_'+k+'_20261011_02.json') for k in ('SPEC','ADMISSION','CPU_QUALIFICATION')}
    output=h/'QUERY_PIXEL_JOINT_NATIVE_E24WARM_ADDITIONAL48_02'
    runtime=h/'finite_point_cache_execution_20261005_v2'/'runs'/PHASE
    require(all(not p.exists() for p in list(paths.values())+[output,runtime]),'Fresh outputs/runtime required; no overwrite/retry')
    first_path=h/'ROOT_QUERY_PIXEL_JOINT_NATIVE_FULLFIT24_SPEC_20261011_02.json'
    q_path=h/'ROOT_QUERY_PIXEL_JOINT_NATIVE_FULLFIT24_CPU_QUALIFICATION_20261011_02.json'
    template_path=h/'ROOT_NATIVE_FINETUNE24_FORMAL_SPEC_20261010_01.json'
    first,template=read(first_path),read(template_path)  # Raw locator metadata only.
    tiny=read(first['joint_preflight_spec']['path'])
    require(first['phase']==FIRST24_PHASE and first['schema']=='ROOT-'+FIRST24_PREFIX+'-spec-v1' and
        set(tiny['training_receiptmap'])==set(tiny['training_phasemap'])==set(MODES) and
        first['training_receiptmap']==tiny['training_receiptmap']==template['training_receiptmap'] and
        first['training_phasemap']==tiny['training_phasemap']==template['training_phasemap'],'Original four raw maps differ')
    raw_rows=[(tiny['training_receiptmap'][k],tiny['training_phasemap'][k]) for k in MODES]
    require(all(set(row)=={'receipt','phase'} for row in tiny['prior_receipts']),'Explicit tiny prior identities required')
    raw_rows += [(row['receipt'],row['phase']) for row in tiny['prior_receipts']]
    require((template['restoration_receipt'],template['restoration_phase']) in raw_rows,'Actual native restoration raw must be retained')
    for ref,phase in raw_rows: terminal(read(ref['path']),phase)
    terminal(read(first['joint_preflight_receipt']['path']),first['joint_preflight_phase'])
    raw_path=h/'finite_point_cache_execution_20261005_v2'/'runs'/FIRST24_PHASE/'PARENT_RECEIPT.json'
    raw=read(raw_path); terminal(raw,FIRST24_PHASE)
    # Every original/restoration/tiny/ownfirst24 raw precedes the first SHA or SOURCE execution.

    first_ref,q_ref,raw_ref,template_ref=[reference(p) for p in (first_path,q_path,raw_path,template_path)]
    require(first['entry_source']['sha256']==FIRST24_ENTRY_SHA and first['CPU_reader']['sha256']==FIRST24_READER_SHA and
        Path(first['CPU_qualification_output']).resolve(strict=True)==q_path.resolve(strict=True),'Sealed first24 SOURCE/qualification locator differs')
    prior=load(first['entry_source'],'_ROOT_own_first24_provenance_ONLY')
    actual_entry,actual_spec,first_owner,tiny_entry=prior.source_gate(first_path,first)
    require(actual_entry==first['entry_source'] and actual_spec==first_ref and raw['admission_sha256']==first_owner['sha256'],'Actual own first24 owner/SPEC/raw differs')
    q=read(bound(q_ref)); report_ref=q['report']; report=read(bound(report_ref))
    require(q['schema']=='ROOT-'+FIRST24_PREFIX+'-CPU-qualification-v1' and q['status']=='ROOT_QUERY_PIXEL_JOINT_NATIVE_FULLFIT24_CPU_QUALIFIED' and
        q['source']==actual_entry and q['reader_source']==first['CPU_reader'] and q['spec']==first_ref and q['parent_receipt']==raw_ref and
        q['GPU_admission']==first_owner and q['phase']==FIRST24_PHASE and q['completed_phase_epochs']==24 and q['effective_native_epoch']==48 and
        q['counters_per_arm']==prior.COUNTERS==dict(epochs=24,microsteps=17784,optimizer_updates=8892,raw_GT_visits_per_arm=39864) and
        q['phase_microsteps_per_arm']==17784 and q['native_updates_per_arm']==q['B_readout_updates']==8892 and q['phase_raw_GT_visits_per_arm']==39864 and
        q['teacher_checkpoint']==first['checkpointmap']['native0'] and q['teacher_qualification']==first['qualificationmap']['native0'] and
        q['joint_preflight_CPU_qualification']==first['joint_preflight_CPU_qualification'] and q['CPU_numeric_protocol']==prior.NUMERIC and
        q['CPU_execution']==dict(device='cpu',affinity=[0,1],CUDA_VISIBLE_DEVICES='',torch_num_threads=2,torch_num_interop_threads=2,CUDA_initialized=False) and
        q['no_AP']is q['no_DEV']is q['no_TEST']is True and q['technical_weights_loaded']is q['formal_resume_eligible']is False and
        report_ref==reference(Path(first['output_directory'])/'REPORT.json'),'Actual own first24 CPU qualification differs')
    require(report['schema']==FIRST24_PREFIX+'-report-v1' and report['status']=='QUERY_PIXEL_JOINT_NATIVE_FULLFIT24_GPU_COMPLETE_PENDING_CPU' and
        report['source']==actual_entry and report['spec']==first_ref and report['GPU_admission']==first_owner and report['phase']==FIRST24_PHASE and
        report['error']is None and report['cleanup']['allocated_after']==0 and report['counters_per_arm']==prior.COUNTERS and
        report['actual_total_backward_calls']==35568 and report['result']==q['result'] and set(q['result'])==set(q['arms'])=={'A','B'},'Actual completed first24 report differs')
    primary=[]; warm={}
    for arm in ('A','B'):
        row,summary=q['result'][arm],q['arms'][arm]; warm[arm]=row['checkpoint']
        require(row['checkpoint']==summary['checkpoint'] and row['completed_phase_epochs']==24 and row['effective_native_epoch']==48 and
            row['phase_microsteps']==17784 and row['phase_updates']==8892 and row['phase_raw_GT_visits']==39864 and
            summary['strict_native610_CPU_population']is summary['initial_native610_CPU_exact']is summary['typed_BN106_affine159_buffers_CPU_exact']is True and
            summary['native345_moments_match_actual_trace_counts']is True and summary['readout_moment_entries']==(8 if arm=='B' else 0) and
            len(row['audits'])==4,'Own qualified arm primary endpoint differs')
        primary += [row[k] for k in ('checkpoint','endpoint_proof','trace','order','epochs','start')] + row['audits']
    require(all(r in first['sources'] for r in template['sources']),'Complete original template closure must be retained')
    native=load(first['native_loader_source'],'_ROOT_original_E24_constructor_provenance_ONLY')
    legacy=load(first['evaluation_source'],'_ROOT_original_teacher_lineage_ONLY'); teachers=legacy.qualified_training(first)
    native.validate_phase_recipe(first,teachers['native0']['training_spec'])
    require(first['schedule']==native.SCHEDULE==dict(gamma=.1,milestones=[7904,8563],warmup_start_factor=1,warmup_updates=0),
        'Original E24 CPU constructor view differs; it must not use the new48 loop schedule')

    d=c/'query_pixel_joint_native_E24warm_additional48_source_20261011_deadline28h_02'
    sources=dict(entry_source=reference(d/'train.py'),CPU_reader=reference(d/'qualify_cpu.py'),manifest_builder_SOURCE=reference(__file__),
        first24_training_spec=first_ref,first24_CPU_qualification=q_ref,first24_report=report_ref,first24_receipt=raw_ref,
        first24_entry_source=actual_entry,first24_CPU_reader_source=first['CPU_reader'])
    require(sources['entry_source']['sha256']==ENTRY_SHA and sources['CPU_reader']['sha256']==READER_SHA,'Sealed warm48 producer/reader differs')
    writer=load(sources['entry_source'],'_ROOT_joint_warm48_provenance_ONLY'); spec=copy.deepcopy(first)
    spec.update(sources,schema=writer.SCHEMA,phase=PHASE,first24_phase=FIRST24_PHASE,warm_checkpoints=warm,arms=list(writer.ARMS),
        **writer.COUNTERS,accumulation=2,auxiliary_weight=1.,schedule=writer.SCHEDULE,B_readout_optimizer=writer.HEAD_RECIPE,B_readout_LR_schedule=False,
        CPU_affinity=[0,1],allocator_limit_MiB=4096,whole_GPU_owner_limit_MiB=8192,deadline_seconds=100800,CPU_numeric_protocol=writer.NUMERIC,
        kappa=1.,eps=1e-6,readout_seed=2026100911,order_seed=2026101021,loss_seed_base=1000000,point_seed_base=3800000,
        backbone_with_cp=True,no_feature_cache=True,no_DEV_selection=True,GPU_admission_file=str(paths['ADMISSION']),output_directory=str(output),
        CPU_qualification_output=str(paths['CPU_QUALIFICATION']),serial_arms=True,weights_warm_start=True,technical_weights_loaded=False,
        old_optimizer_restored=False,old_RNG_restored=False,manifest_builder_scope='STDLIB_AUTHORING_ONLY_NO_SCIENCE_OR_LAUNCH',
        first24_CPU_tool_gate_scope='Actual CPU tool terminal0 requires ROOT confirmation; qualification file alone is not OS evidence')
    require(writer.COUNTERS==dict(epochs=48,microsteps=35568,optimizer_updates=17784,raw_GT_visits_per_arm=79728) and
        writer.SCHEDULE==dict(gamma=.1,milestones=[15808,17126],warmup_start_factor=1,warmup_updates=0) and writer.NUMERIC==prior.NUMERIC and
        writer.HEAD_RECIPE==dict(type='AdamW',lr=1e-4,weight_decay=.05,betas=[.9,.999],eps=1e-8),'Authoritative warm48 recipe changed')
    extras=[template_ref,first_owner]+list(sources.values())+primary+[q[k] for k in ('GPU_admission','start','GPU_qualification')]
    for row in teachers.values():
        qualification=row['qualification']; extras += row['training_spec']['sources']+[qualification[k] for k in
            ('source','reader_source','spec','report','start','GPU_admission','owner_driver')]+list(qualification['model_proof']['proof_refs'].values())
    spec['sources']=union_refs(first['sources']+template['sources']+extras)
    require(all(spec[k] in spec['sources'] for k in writer.MODULES+writer.LIBRARIES+writer.PREFLIGHT_ROLES+writer.FIRST24_ROLES+
        ('entry_source','CPU_reader','training_CPU_reader','reused_CPU_reader','readout_source','geometry_source','hooks_source','FIT')) and
        all(r in spec['sources'] for r in warm.values()),'Complete own first24/warm/original role closure required')
    for ref in spec['sources']: bound(ref)
    driver=h/'finite_point_cache_execution_20261005_v2'/'run_owned_gpu_limited.py'
    require(reference(driver)['sha256']==OWNER_SHA,'Original owner driver differs')
    admission=dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE',phase=PHASE,driver_sha256=OWNER_SHA,
        entry=sources['entry_source']['path'],entry_args=['--spec',str(paths['SPEC']),'--owner-execute'],allowed_cpus=[0,1],cuda_visible_devices='0',
        qualified_leaf_deadline_seconds=100800,whole_GPU_used_limit_MiB=8192,automatic_retry=False,manifest_builder_SOURCE=sources['manifest_builder_SOURCE'],
        bound_source_sha256={ref['path']:ref['sha256'] for ref in spec['sources']})
    write_fresh(paths['SPEC'],spec); spec_ref=reference(paths['SPEC'])
    admission['scientific_spec']=spec_ref; admission['bound_source_sha256'][spec_ref['path']]=spec_ref['sha256']
    write_fresh(paths['ADMISSION'],admission)
    actual_source,actual_spec,actual_admission,_=writer.source_gate(paths['SPEC'],spec)
    require(actual_source==sources['entry_source'] and actual_spec==spec_ref and actual_admission==reference(paths['ADMISSION']),
        'Final SOURCE-only compatibility gate differs')
    print(json.dumps(dict(status='ROOT_JOINT_NATIVE_E24WARM_ADDITIONAL48_MANIFESTS_BUILT_NOT_EXECUTED',spec=spec_ref,admission=actual_admission,
        CPU_qualification_output=str(paths['CPU_QUALIFICATION']),output_directory=str(output),owner_runtime=str(runtime),source_count=len(spec['sources']),
        counters_per_arm=writer.COUNTERS,initial_native_epoch=48,effective_native_epoch=96,effective_B_readout_epoch=72,
        science_executed=False,training_launched=False),sort_keys=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--artifact-root',required=True); parser.add_argument('--owner-author',action='store_true')
    args=parser.parse_args(); require(args.owner_author,'Inert SOURCE: explicit future ROOT authoring required'); build(args)


if __name__=='__main__':
    main()
