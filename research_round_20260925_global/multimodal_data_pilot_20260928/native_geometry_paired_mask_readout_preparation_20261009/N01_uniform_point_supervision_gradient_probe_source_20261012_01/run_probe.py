"""Fixed E24/FIT local gradient diagnosis. No optimizer, update, DEV, TEST or AP."""
import argparse
from functools import partial
import gc, hashlib, json, math, os, statistics, time, types
from pathlib import Path

SCHEMA = 'ROOT-N01-uniform-point-supervision-gradient-probe-spec-v1'
STATUS = 'N01_FIXED_FIT_LOCAL_GRADIENT_PROBE_GPU_COMPLETE_PENDING_CPU'
PARAM_PREFIXES = ('panoptic_head.mask_embed.', 'panoptic_head.pixel_decoder.mask_feature.')
NAMES = ('native', 'foreground', 'background', 'balanced')
def require(ok, message):
    if not ok: raise RuntimeError(message)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def reference(path):
    path=Path(path).resolve(strict=True); digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())
def bound(ref):
    require(reference(ref['path'])==ref,'Actual pinned input differs')
    return Path(ref['path'])
def gate(path, phase):
    raw=read(path)
    require(Path(path).is_absolute() and Path(path).name=='PARENT_RECEIPT.json'
        and Path(path).parent.name==phase and raw['status']=='ACTUAL_OWNED_TERMINAL'
        and type(raw['returncode']) is int and raw['returncode']==0
        and isinstance(raw['original_identity'],dict) and bool(raw['original_identity'])
        and raw['held_handle_exit']==dict(wait_state='WAIT_OBJECT_0',actual_exit_code=0)
        and type(raw['held_handle_exit']['actual_exit_code']) is int
        and raw['root_wait_reaped'] is True and raw['numeric_birth_absence']['original_birth_absent'] is True
        and raw.get('error') is None and raw.get('abort') is None and raw['wait_errors'] is False
        and raw['GPU_observation_errors']==[], 'All ancestral owned terminal0 required before hashes/imports')
    peak=raw['whole_GPU_observed_peak_MiB']
    require(type(peak) in (int,float) and math.isfinite(peak) and 0<=peak<=8192
        and raw['whole_GPU_used_limit_MiB']==8192 and raw['whole_leaf_VRAM_certified'] is False
        and raw['GPU_scope']=='whole GPU observed, not per-process WDDM allocation','RAW resource scope differs')
def load(ref,name):
    module=types.ModuleType(name);module.__file__=str(bound(ref))
    exec(compile(Path(module.__file__).read_bytes(),module.__file__,'exec'),module.__dict__)
    return module
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def metric(torch,a,b=None):
    aa={k:v.double() for k,v in a.items()}
    n=math.sqrt(math.fsum(float((v*v).sum()) for v in aa.values()))
    if b is None:return n
    bb={k:v.double() for k,v in b.items()}
    other=math.sqrt(math.fsum(float((v*v).sum()) for v in bb.values()))
    dot=math.fsum(float((aa[k]*bb[k]).sum()) for k in aa)
    return dict(norm=n,other_norm=other,dot=dot,cosine=None if min(n,other)<=1e-12 else max(-1.,min(1.,dot/(n*other))))
def sum_gradients(a,b):return {k:a[k]+b[k] for k in a}
def scalar_pair(torch,a,b):return metric(torch,dict(field=a),dict(field=b))
def select(torch,model):
    named=dict(model.named_parameters())
    selected={n:p for n,p in named.items() if any(n.startswith(prefix) for prefix in PARAM_PREFIXES)}
    require(len(selected)==8 and all(p.requires_grad for p in selected.values())
        and all(any(n.startswith(prefix) for n in selected) for prefix in PARAM_PREFIXES),'Expected six mask_embed and two mask_feature trainable parameters')
    return selected

def probe_batch(torch,np,cv2,modules,preflight,objective,model,policy,dataset,coco,spec,index,position,selected):
    native,legacy,pref,initializer,common,old,diag,posterior,capture,joint,adapter,nativeobjective=modules
    torch.cuda.synchronize();started=time.monotonic()
    data,stripped,gray,raw_gray,identity=old.load_image(model,dataset,coco,dataset.get_data_info(index))
    del stripped,gray,raw_gray
    identity['dataset_index']=index
    expected=spec['diagnostic_image_rows'][position]
    require(identity['image_id']==expected['image_id'] and identity['image']==expected['image'],'Selected actual image differs')
    target=pref.check_targets(torch,data,identity)
    raw_GT,mapping,validator=preflight.raw_GT_validator(torch,np,cv2,coco,data,identity)
    point_seed=spec['point_seed_base']+position
    point_rng=torch.Generator(device='cpu').manual_seed(point_seed)
    indices=torch.randint(raw_GT.shape[1]*raw_GT.shape[2],(len(raw_GT),12544),generator=point_rng,dtype=torch.int64).to('cuda:0')
    observed_targets=[]
    native_target_packet={}
    def observe_mask_targets(module,args):
        require(len(args)>=2 and len(observed_targets)<10,'One native point-label observation per stage')
        labels=args[1].detach()
        require(labels.dtype==torch.float32 and labels.numel()==len(raw_GT)*12544
            and bool(torch.isfinite(labels).all()) and bool(((labels>=0)&(labels<=1)).all()),'Actual native point labels differ')
        y=labels.reshape(len(raw_GT),12544)
        observed_targets.append(dict(stage=len(observed_targets),shape=list(y.shape),
            interpolation_soft_count=int(((y>0)&(y<1)).sum()),total_points=y.numel(),
            positive_mass_by_row=y.sum(1).cpu().tolist(),
            hard_positive_count=int((y==1).sum()),hard_negative_count=int((y==0).sum())))
        if position<5:native_target_packet['stage_'+str(len(observed_targets)-1)]=y.cpu().clone()
    hooks_before={n:(tuple(v._forward_pre_hooks),tuple(v._forward_hooks)) for n,v in model.named_modules()}
    handle=model.panoptic_head.loss_mask.register_forward_pre_hook(observe_mask_targets)
    external=pref.rng_snapshot(torch)
    try:
        with torch.random.fork_rng(devices=[0]):
            seed=spec['loss_seed_base']+position
            torch.random.default_generator.manual_seed(seed);torch.cuda.manual_seed(seed)
            model.zero_grad(set_to_none=True)
            r=joint.joint_native_losses(torch,pref,common,old,capture,adapter,nativeobjective,model,data,None,
                partial(diag.render_original,torch),identity,identity['ann_ids'],raw_GT,mapping,validator,indices,'A',position)
            require(len(observed_targets)==10,'All ten actual native point-label stages required')
            live=r['live_native_fields'];L=r['auxiliary_fields']['native_logits'];queries=r['query_ids']
            z=L[0,queries].flatten(1).gather(1,indices)
            y=raw_GT.flatten(1).gather(1,indices)
            require(y.dtype==torch.bool,'Additional original-raster supervision is hard binary')
            parts=objective.point_loss_components(torch,z,y)
            torch.testing.assert_close(parts['uniform'],r['auxiliary'],rtol=1e-5,atol=1e-6)
            fields={k:live[k] for k in ('F256','e0','M200')}
            for tensor in fields.values():tensor.retain_grad()
            snapshots={}
            full_gradient_counts={}
            def backward_snapshot(loss,key,retain):
                model.zero_grad(set_to_none=True)
                for tensor in fields.values():tensor.grad=None
                loss.backward(retain_graph=retain)
                parameter_grad={}
                for name,param in selected.items():
                    require(param.grad is not None and bool(torch.isfinite(param.grad).all()),'Selected native parameter gradient absent/nonfinite')
                    parameter_grad[name]=param.grad.detach().cpu().clone()
                field_grad={}
                for name,tensor in fields.items():
                    require(tensor.grad is not None and bool(torch.isfinite(tensor.grad).all()),'Live field gradient absent/nonfinite')
                    field_grad[name]=tensor.grad.detach().cpu().clone()
                active=dict(model.named_parameters())
                counts=dict(non_none=0,nonzero=0)
                for name in policy['active_names']:
                    g=active[name].grad
                    if g is not None:
                        require(bool(torch.isfinite(g).all()),'Native gradient nonfinite')
                        counts['non_none']+=1;counts['nonzero']+=int(bool((g!=0).any()))
                require(all(active[name].grad is None for name in policy['frozen_names']),'Frozen BN affine received gradients')
                full_gradient_counts[key]=counts
                return dict(parameters=parameter_grad,fields=field_grad)
            snapshots['native']=backward_snapshot(r['native_total'],'native',True)
            direct_uniform=backward_snapshot(r['auxiliary'],'uniform_direct',True) if position==0 else None
            snapshots['foreground']=backward_snapshot(parts['foreground'],'foreground',True)
            snapshots['background']=backward_snapshot(parts['background'],'background',True)
            snapshots['balanced']=backward_snapshot(parts['balanced'],'balanced',False)
            uniform=dict(parameters=sum_gradients(snapshots['foreground']['parameters'],snapshots['background']['parameters']),
                         fields=sum_gradients(snapshots['foreground']['fields'],snapshots['background']['fields']))
            linear_check=None
            if direct_uniform is not None:
                maximum=0.
                for group in ('parameters','fields'):
                    for name,value in uniform[group].items():
                        torch.testing.assert_close(value,direct_uniform[group][name],rtol=1e-3,atol=1e-7)
                        maximum=max(maximum,float((value-direct_uniform[group][name]).abs().max()))
                linear_check=dict(status='DIRECT_U_VS_FOREGROUND_PLUS_BACKGROUND_PASS',max_abs_error=maximum,rtol=1e-3,atol=1e-7)
            p=torch.sigmoid(z.detach())
            Gplus=(y.to(p.dtype)*(1-p)).mean(1)
            Gminus=((~y).to(p.dtype)*p).mean(1)
            parameter_relation=metric(torch,uniform['parameters'],snapshots['native']['parameters'])
            balanced_relation=metric(torch,snapshots['balanced']['parameters'],snapshots['native']['parameters'])
            field_relations={name:dict(uniform_vs_native=scalar_pair(torch,uniform['fields'][name],snapshots['native']['fields'][name]),
                balanced_vs_native=scalar_pair(torch,snapshots['balanced']['fields'][name],snapshots['native']['fields'][name]),
                foreground_norm=float(snapshots['foreground']['fields'][name].double().norm()),
                background_norm=float(snapshots['background']['fields'][name].double().norm())) for name in fields}
            packet=dict(schema='N01-point-gradient-packet-v1',position=position,dataset_index=index,identity=identity,
                point_indices=indices.cpu().clone(),point_logits=z.detach().cpu().clone(),point_targets=y.cpu().clone(),
                parameter_gradients={name:snapshots[name]['parameters'] for name in NAMES},
                losses=dict(native=float(r['native_total'].detach()),uniform=float(r['auxiliary'].detach()),
                    foreground=float(parts['foreground'].detach()),background=float(parts['background'].detach()),balanced=float(parts['balanced'].detach())),
                final_query_by_raw_GT=queries.cpu().tolist(),native_metadata=r['metadata'],native_point_labels_observed=observed_targets,
                backward_count=5 if position==0 else 4,optimizer_created=False,weights_updated=False)
            if position<5:
                packet.update(M200=live['M200'].detach().cpu().clone(),raw_GT=raw_GT.cpu().clone(),
                    local_M_gradients={name:snapshots[name]['fields']['M200'][0,queries.cpu()].clone() for name in ('foreground','background','balanced')},
                    native_point_targets=native_target_packet)
            row=dict(position=position,dataset_index=index,image_id=identity['image_id'],annotation_ids=identity['ann_ids'],
                identity=identity,raw_GT_count=len(raw_GT),original_GT_area=raw_GT.flatten(1).sum(1).cpu().tolist(),
                sampled_foreground_count=y.sum(1).cpu().tolist(),sampled_zero_foreground_instances=int((y.sum(1)==0).sum()),
                point_count_per_GT=12544,hard_additional_labels=True,point_seed=point_seed,loss_seed=seed,
                point_indices_digest=pref.digest_tensor(indices),native_point_labels=observed_targets,
                output_gradient_foreground_mass=Gplus.cpu().tolist(),output_gradient_background_mass=Gminus.cpu().tolist(),
                losses=packet['losses'],parameters_uniform_vs_native=parameter_relation,parameters_balanced_vs_native=balanced_relation,
                foreground_parameter_norm=metric(torch,snapshots['foreground']['parameters']),
                background_parameter_norm=metric(torch,snapshots['background']['parameters']),
                balanced_parameter_norm=balanced_relation['norm'],field_relations=field_relations,
                direct_uniform_gradient_linearity=linear_check,actual_native_gradient_counts=full_gradient_counts,
                gradient_scope='eight mask_embed and mask_feature parameter tensors; F/e/M each reported separately',
                native_full_backward_CPU_replayed=False,optimizer_created=False,weights_updated=False)
            model.zero_grad(set_to_none=True)
            for tensor in fields.values():tensor.grad=None
    finally:
        handle.remove()
    require(pref.tree_equal(torch,external,pref.rng_snapshot(torch)),'External RNG not restored')
    require({n:(tuple(v._forward_pre_hooks),tuple(v._forward_hooks)) for n,v in model.named_modules()}==hooks_before,'Passive hooks not restored')
    require(not any(dict(model.named_modules())[n].training for n in policy['BN_modules']),'Frozen BN mode changed')
    torch.cuda.synchronize();row['elapsed_seconds']=time.monotonic()-started
    return row,packet

def summarize(rows):
    valid=[r for r in rows if r['parameters_uniform_vs_native']['cosine'] is not None and r['parameters_balanced_vs_native']['cosine'] is not None]
    differences=[r['parameters_balanced_vs_native']['cosine']-r['parameters_uniform_vs_native']['cosine'] for r in valid]
    negative=sum(r['parameters_uniform_vs_native']['cosine']<0 for r in valid)
    improved=sum(d>0 for d in differences)
    median_U=None if not valid else statistics.median(r['parameters_uniform_vs_native']['cosine'] for r in valid)
    median_difference=None if not differences else statistics.median(differences)
    ratios=[r['parameters_uniform_vs_native']['norm']/r['balanced_parameter_norm'] for r in valid if r['balanced_parameter_norm']>1e-12]
    candidate=bool(len(valid)>=24 and negative>=20 and improved>=20 and median_U<0 and median_difference>=.10 and len(ratios)>=24)
    return dict(valid_images=len(valid),uniform_negative_cosine_images=negative,balanced_improves_cosine_images=improved,
        median_uniform_native_cosine=median_U,median_balanced_minus_uniform_cosine=median_difference,
        FIT_initial_selected_parameter_norm_calibration_c=None if not ratios else statistics.median(ratios),
        N02_worth_finite_local_followup_pending_CPU=candidate,mechanism_established=False,harmful_training_drift_proven=False,
        criterion='valid>=24/32; U negative cosine>=20; balance improves>=20; median U cosine<0; median balance-minus-U>=0.10',
        gradient_scope='eight selected parameter tensors only; not whole native345',
        not_statistical_significance_or_novelty=True)

def run(spec_path):
    started=time.monotonic();path=Path(spec_path).resolve(strict=True);spec=read(path)
    require(spec['schema']==SCHEMA and len(spec['RAW_inputs'])==14 and len({r['receipt']['path'] for r in spec['RAW_inputs']})==14,'Fixed N01 source schema/14 RAWs required')
    for row in spec['RAW_inputs']:gate(row['receipt']['path'],row['phase'])
    # Every current/ancestor terminal precedes all hashes and scientific imports.
    for row in spec['RAW_inputs']:
        require(all(row[k] in spec['sources'] for k in ('receipt','admission','phase_binding','bootstrap')),
            'Actual ancestral receipt/owner bindings absent from sources')
        raw=read(bound(row['receipt']))
        ancestral_admission=read(bound(row['admission']))
        ancestral_binding=read(bound(row['phase_binding']))
        bound(row['bootstrap'])
        require(ancestral_admission['phase']==row['phase']
            and raw['admission_sha256']==row['admission']['sha256']==ancestral_binding['admission_sha256']
            and ancestral_binding['bootstrap_sha256']==row['bootstrap']['sha256']
            and ancestral_binding['argv'][-1]==row['bootstrap']['path'], 'Actual ancestral owner provenance join differs')
    for ref in spec['sources']:bound(ref)
    source=reference(__file__);spec_ref=reference(path);admit_ref=reference(spec['GPU_admission_file']);admit=read(admit_ref['path'])
    require(source==spec['entry_source'] and admit['scientific_spec']==spec_ref and admit['phase']==spec['phase']
        and admit['entry']==source['path'] and admit['entry_args']==['--spec',str(path),'--owner-execute']
        and admit['allowed_cpus']==[0,1] and admit['cuda_visible_devices']=='0' and admit['automatic_retry'] is False
        and admit['qualified_leaf_deadline_seconds']==spec['deadline_seconds'] and admit['whole_GPU_used_limit_MiB']==8192,'Own actual admission differs')
    require(admit['bound_source_sha256'].get(str(path))==spec_ref['sha256']
        and all(admit['bound_source_sha256'].get(r['path'])==r['sha256'] for r in spec['sources']),'Source closure not owner-bound')
    require(spec['parameter_prefixes']==list(PARAM_PREFIXES) and spec['diagnostic_batches']==32
        and len(spec['diagnostic_dataset_indices'])==32 and len(set(spec['diagnostic_dataset_indices']))==32
        and spec['audit_positions']==[0,1,2,3,4] and spec['optimizer_created'] is False and spec['weights_updated'] is False
        and os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Prelocked diagnostic scope differs')
    output=Path(spec['output_directory']);require(output.is_absolute() and not output.exists(),'Fresh output required');output.mkdir()
    legacy=load(spec['evaluation_source'],'_N01_original_teacher_provenance');teachers=legacy.qualified_training(spec)
    import psutil,torch,numpy as np,cv2
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity()==[0,1] and not torch.cuda.is_initialized(),'CPU01/cold CUDA required')
    require(all(os.environ.get(k)=='2' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']),'Thread environment differs')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    objective=load(spec['point_objectives_source'],'_N01_point_objectives');fixture=load(spec['test_components_source'],'_N01_public_objective_fixture')
    GREEN=fixture.verify_cpu_components(torch,objective)
    require(not torch.cuda.is_initialized(),'CPU fixture unexpectedly initialized CUDA')
    register_all_modules(init_default_scope=False);init_default_scope('mmdet')
    template=load(spec['first24_entry_source'],'_N01_existing_module_roles')
    modules=[load(spec[k],'_N01_'+k) for k in template.MODULES]
    native,legacy,pref,initializer,common,old,diag,posterior,capture,joint,adapter,nativeobjective=modules
    old.torch=torch
    preflight=load(spec['joint_preflight_entry_source'],'_N01_qualified_GT_validator')
    construction=dict(teachers['native0']['training_spec'],training_source=spec['training_source'],checkpointmap=spec['checkpointmap'],
        qualificationmap=spec['qualificationmap'],order_seed=native.ORDER_SEED,phase_loss_seed_base=native.LOSS_SEED_BASE)
    model,anchor,policy,loading=native.prepare_training_model(torch,pref,initializer,common,MODELS,revert_sync_batchnorm,Config,
        tuple(CocoDataset.METAINFO['classes']),construction,teachers['native0'])
    native.imported_library_gate(torch,model,spec)
    dataset=CocoDataset(ann_file=str(bound(spec['FIT'])),data_prefix=dict(img=spec['image_root']),metainfo=dict(classes=('ship',)),pipeline=[],filter_cfg=None,test_mode=True)
    coco=COCO(str(bound(spec['FIT'])));require(len(dataset)==741 and len(coco.anns)==1661,'Original FIT population differs')
    torch.cuda.init();torch.cuda.set_per_process_memory_fraction(min(1.,spec['allocator_limit_MiB']*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0);model.to('cuda:0');selected=select(torch,model)
    require(dict(pref.trainability(torch,model),kind='mask2former')==policy,'Native CUDA policy differs')
    write(output/'START.json',dict(schema='N01-start-v1',source=source,spec=spec_ref,GPU_admission=admit_ref,phase=spec['phase'],
        teacher_loading=loading,CPU_public_fixture=GREEN,parameter_names=list(selected),parameter_count=sum(p.numel() for p in selected.values()),
        native345_active=True,typed_BN106_and159_frozen=True,optimizer_created=False,weights_updated=False,no_DEV=True,no_TEST=True,no_AP=True))
    rows=[];packet_refs=[]
    with (output/'TRACE.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        for position,index in enumerate(spec['diagnostic_dataset_indices']):
            require(time.monotonic()-started<spec['deadline_seconds'],'N01 fixed deadline')
            row,packet=probe_batch(torch,np,cv2,modules,preflight,objective,model,policy,dataset,coco,spec,index,position,selected)
            packet_path=output/('PACKET_'+str(position).zfill(2)+'.pth')
            with packet_path.open('xb') as target:torch.save(packet,target)
            row['packet']=reference(packet_path);packet_refs.append(row['packet']);rows.append(row)
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            del packet
    final=pref.cpu_tree(torch,model.state_dict())
    require(final.keys()==anchor.keys() and all(torch.equal(final[n],anchor[n]) for n in anchor),'Diagnostic must not change any native610 weight/buffer')
    peak=torch.cuda.max_memory_allocated(0);model.cpu();del model,selected,anchor,final;gc.collect()
    cleanup=pref.clear_gpu(torch);require(cleanup['allocated_after']==0,'N01 CUDA cleanup differs')
    for ref in spec['sources']:bound(ref)
    require(reference(path)==spec_ref and reference(__file__)==source and reference(spec['GPU_admission_file'])==admit_ref,'Final source/spec/admission changed')
    report=dict(schema='N01-fixed-FIT-local-gradient-report-v1',status=STATUS,source=source,spec=spec_ref,GPU_admission=admit_ref,phase=spec['phase'],
        CPU_public_fixture=GREEN,images=32,rows=rows,packets=packet_refs,summary=summarize(rows),
        native610_unchanged=True,optimizer_created=False,weights_updated=False,one_forward_per_image=True,actual_backward_calls=129,
        no_DEV=True,no_TEST=True,no_AP=True,core_innovation_established=False,TGRS_success=False,native30_CPU_replayed=False,
        harmful_training_drift_proven=False,elapsed_seconds=time.monotonic()-started,torch_peak_allocated_bytes=peak,cleanup=cleanup)
    write(output/'REPORT.json',report)
    print(json.dumps(dict(status=STATUS,report=reference(output/'REPORT.json'),summary=report['summary'])),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();require(args.owner_execute,'ROOT owner-only execution required');run(args.spec)
