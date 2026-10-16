"""N28 owned initialization and first-real-instance interface validation.
No numerical import or I/O occurs at module import. No efficacy training.
"""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time

SEEDS=[202610172801,202610172802,202610172803]

def need(ok,msg):
    if not ok:
        raise RuntimeError(msg)

def read(c,v):
    return json.loads(c.closed(v).read_bytes())

def load(c,v,name):
    spec=importlib.util.spec_from_file_location(name,str(c.closed(v)))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def digest(value):
    array=value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape),dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())

def render(torch,value,identity):
    """Native200 -> padded800 -> actual crop -> original, on caller device."""
    h,w=identity['actual_img_shape'];H,W=identity['original_shape']
    padded=torch.nn.functional.interpolate(value,size=(800,800),mode='bilinear',align_corners=False)
    crop=padded[0,:,:h,:w]
    return torch.nn.functional.interpolate(crop[:,None],size=(H,W),mode='bilinear',align_corners=False)[:,0]

def same_state(torch,left,right):
    return list(left)==list(right) and all(torch.equal(left[k],right[k]) for k in left)

def dense_loss(torch,logits,target):
    """Per-instance equally weighted FG/BG BCE plus soft Dice, then mean."""
    need(logits.shape==target.shape and logits.ndim==3,'Q,H,W loss tensors')
    pixel=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')
    dims=(1,2);fg=target.sum(dims);bg=(1-target).sum(dims)
    need(bool((fg>0).all()) and bool((bg>0).all()),'Real ship and background pixels')
    balanced=.5*((pixel*target).sum(dims)/fg+(pixel*(1-target)).sum(dims)/bg)
    prob=torch.sigmoid(logits)
    dice=1-(2*(prob*target).sum(dims)+1)/(prob.sum(dims)+fg+1)
    return (balanced+dice).mean()

def run(torch,np,spec,c,pre):
    from pycocotools.coco import COCO
    started=time.monotonic()
    need(pre['schema']=='N28-dense-routing-initialize-prelock-v1'
         and pre['seeds']==SEEDS and pre['real_FIT_dataset_indices']==[0],
         'Fixed initialization/one-real-image validation scope')
    context=read(c,spec['context'])
    need(context['schema']=='ROOT-N28-dense-initialize-context-v1','Exact context')
    fit=read(c,context['FIT_context'])
    old=read(c,context['N13_qualified']);report=read(c,context['N13_report'])
    need(old['status']=='ROOT_N13_SAMEK_REGION_FIT_CACHE_CPU_QUALIFIED_V2'
         and old['GPU_report']==context['N13_report']
         and old['teacher_checkpoint']==report['teacher_checkpoint']==fit['teacher_checkpoint']
         and old['images']==report['images']==fit['images']==741
         and old['raw_GT']==report['raw_GT']==fit['GT']==1661
         and old['validation_carrier_refs']==report['validation_carrier_refs'],
         'Previously qualified complete F population, same frozen teacher')
    row=report['rows'][0];identity=fit['identities'][0]
    need(row['dataset_index']==0 and row['identity']==identity
         and row['validation_carrier']==context['first_carrier']==report['validation_carrier_refs'][0]
         and context['first_N16_packet']==fit['existing_packet_refs'][0]
         and context['first_native_L_carrier']==fit['existing_native_L_refs'][0],
         'Original first image and exact carrier references')
    carrier=torch.load(str(c.closed(context['first_carrier'])),map_location='cpu')
    packet=torch.load(str(c.closed(context['first_N16_packet'])),map_location='cpu')
    native=torch.load(str(c.closed(context['first_native_L_carrier'])),map_location='cpu')
    need(carrier['schema']=='N13-sameK-region-FIT-validation-carrier-v1'
         and packet['schema']=='N16-all100-FIT-action-sampled-cache-body-v1'
         and native['schema']=='N16-actual-GPU-natural100-L-carrier-v1'
         and all(v['dataset_index']==0 and v['identity']==identity for v in (carrier,packet,native))
         and carrier['teacher_checkpoint']==fit['teacher_checkpoint']
         and packet['original_cache_packet']==row['original_cache_packet'],
         'Actual tensor containers and original matching lineage')
    need({k:digest(carrier[k]) for k in ('query_ids','native_F256','matched_M200')}
         ==row['carrier_digests'],'Exact original full F/coarse tensor digests')
    F=carrier['native_F256'];M=carrier['matched_M200'];e=packet['native_e0']
    H,W=identity['original_shape'];G=len(identity['ann_ids'])
    need(F.shape==(1,256,200,200) and M.shape==(1,G,200,200) and e.shape==(1,100,256)
         and native['native_L'].shape==(100,H,W)
         and all(v.device.type=='cpu' and v.dtype==torch.float32 and not v.requires_grad
                 and bool(torch.isfinite(v).all()) for v in (F,M,e,native['native_L'])),
         'Finite original detached FP32 full coarse F/M, natural e/L')
    queries=carrier['query_ids'].tolist()
    need(queries==row['query_ids'] and len(queries)==G
         and [int(packet['assignment']['raw_GT_index_by_query'][q]) for q in queries]==list(range(G))
         and packet['raw_GT_ids']==identity['ann_ids'],'Real query-GT order')
    coco=COCO(str(c.closed(fit['FIT'])))
    raw=np.stack([coco.annToMask(coco.anns[a]).astype(np.bool_) for a in identity['ann_ids']])
    need(digest(torch.from_numpy(raw))==packet['raw_GT_digest'],'Real annotated first-image GT digest')
    factory=load(c,spec['factory_source'],'N28_actual_factory')
    outside_cpu_rng=torch.get_rng_state().clone()
    out=Path(spec['output_directory']);out.mkdir()
    checkpoints=[];initials={};parameter_count=None
    for seed in SEEDS:
        c.check()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            A=factory.make_head(torch,'SELECTIVE')
            torch.random.default_generator.manual_seed(seed)
            B=factory.make_head(torch,'WHOLE_NODE')
        need(same_state(torch,A.state_dict(),B.state_dict()),'Paired initial parameters and BN buffers exact')
        count=sum(p.numel() for p in A.parameters())
        need(count==sum(p.numel() for p in B.parameters()) and (parameter_count is None or parameter_count==count),
             'Same fixed capacity across arms and seeds')
        parameter_count=count
        state={k:v.detach().cpu().clone() for k,v in A.state_dict().items()}
        path=out/('COMMON_INITIAL_SEED%d.pth'%seed)
        torch.save(dict(schema='N28-virgin-common-dense-routing-initial-v1',seed=seed,
                        state_dict=state,modes=['SELECTIVE','WHOLE_NODE'],factory_source=spec['factory_source'],
                        trained_updates=0),str(path))
        checkpoints.append(c.ref(path))
        if seed==SEEDS[0]:
            initials=state
        del A,B,state
    need(torch.equal(torch.get_rng_state(),outside_cpu_rng),'CPU global initialization RNG restored')
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((4096*1024**2)/total,0)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
    device='cuda:0';c.check()
    fgpu=F.to(device);egpu=e[0,queries[:1]].to(device);mgpu=M[:,0].to(device)
    base=native['native_L'][queries[:1]].to(device);target=torch.from_numpy(raw[:1].astype(np.float32)).to(device)
    # Old CPU replay allowed practical tolerances. New GPU replay is reported;
    # the saved original native_L, never replayed M, remains the actual base.
    replay=render(torch,mgpu[:,None],identity)
    replay_error=float((replay-base).abs().max())
    replay_bits=bool(torch.equal(replay.view(torch.int32),base.view(torch.int32)))
    need(torch.allclose(replay,base,rtol=1e-4,atol=1e-4)
         and torch.equal(replay>0,base>0),'GPU coarse-to-current-native practical values and exact signs')
    del replay
    indices=packet['unique_flat_indices'].to(device)
    feature_error=0.;feature_exact=True
    with torch.no_grad():
        for first in range(0,256,16):
            c.check()
            projected=render(torch,fgpu[:,first:first+16],identity)
            actual=projected.flatten(1)[:,indices].t().contiguous()
            expected=packet['sampled_F_unique'][:,first:first+16].to(device)
            need(torch.allclose(actual,expected,rtol=1e-4,atol=1e-4),
                 'Existing full F to N16 centers; prelocked practical tolerance')
            feature_error=max(feature_error,float((actual-expected).abs().max()))
            feature_exact=feature_exact and bool(torch.equal(actual.view(torch.int32),expected.view(torch.int32)))
            del projected,actual,expected
    heads={}
    for mode in ('SELECTIVE','WHOLE_NODE'):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(SEEDS[0])
            head=factory.make_head(torch,mode)
        head.load_state_dict(initials,strict=True);heads[mode]=head.to(device).eval()
    tests={};nonzero_outputs={};outside=base.abs()>2
    need(bool(outside.any()),'Real first instance has outside-Omega pixels')
    for mode,head in heads.items():
        c.check()
        with torch.no_grad():
            delta=head(fgpu,egpu,mgpu)
            need(delta.shape==(1,1,200,200) and bool(torch.isfinite(delta).all())
                 and int(torch.count_nonzero(delta))==0,'Zero-initialized dense output')
            combined=base+render(torch,delta.transpose(0,1),identity)
            need(torch.equal(combined,base) and torch.equal(combined>0,base>0),'Zero head preserves exact native logits values and masks')
            head.readout.bias.fill_(1.)
            writable=base+render(torch,head(fgpu,egpu,mgpu).transpose(0,1),identity)
            need(bool((writable[outside]!=base[outside]).all()),'Full-mask write permission including original outside Omega')
            head.load_state_dict(initials,strict=True)
            head.readout.weight.fill_(.001)
            nonzero_outputs[mode]=head(fgpu,egpu,mgpu).detach().cpu().clone()
            del delta,combined,writable
        head.train()
        head.zero_grad(set_to_none=True)
        delta=head(fgpu,egpu,mgpu)
        combined=base+render(torch,delta.transpose(0,1),identity)
        loss=dense_loss(torch,combined,target)
        need(bool(torch.isfinite(loss)),'Finite real one-instance validation loss')
        loss.backward()
        params=list(head.parameters())
        need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params),
             'Every live model parameter has a finite gradient')
        head.zero_grad(set_to_none=True)
        tests[mode]=dict(zero_delta_exact=True,native_logits_values_and_masks_preserved=True,
                        full_mask_outside_write_verified=True,all_parameter_gradients_finite=True,
                        disposable_backward_probes=1,optimizer_steps=0,
                        probe_is_efficacy_training=False)
        del loss,delta,combined,params
        head.eval()
    route_difference=float((nonzero_outputs['SELECTIVE']-nonzero_outputs['WHOLE_NODE']).abs().max())
    need(route_difference>0,'The unique route difference reaches nonzero readout')
    need(torch.equal(torch.get_rng_state(),outside_cpu_rng),'All constructor CPU RNG forks restored')
    for ref in checkpoints:
        need(c.ref(ref['path'])==ref,'Virgin initialization files unchanged by disposable probes')
    elapsed=time.monotonic()-started
    report_value=dict(schema='N28-dense-routing-initialization-report-pending-v1',
        status='INITIALIZATION_AND_FIRST_REAL_INTERFACE_VALIDATION_COMPLETE_PENDING_ROOT_ACTUAL0',
        source=spec['body_source'],factory_source=spec['factory_source'],spec=c.execution_spec_ref,
        GPU_admission=c.execution_admission_ref,phase=spec['phase'],context=spec['context'],prelock=spec['prelock'],
        checkpoint_refs=checkpoints,seeds=SEEDS,modes=['SELECTIVE','WHOLE_NODE'],images=1,samples=1,
        dataset_indices=[0],query_ids=queries[:1],parameter_count_per_arm=parameter_count,
        paired_state_dict_exact_all_seeds=True,virgin_trained_updates=0,CPU_global_RNG_restored=True,
        first_full_F_digest_exact=True,first_coarse_M_digest_exact=True,first_query_GT_identity_exact=True,
        coarse_to_native_max_abs_error=replay_error,coarse_to_native_bitexact=replay_bits,
        coarse_to_native_signs_exact=True,F_to_N16_centers_max_abs_error=feature_error,
        F_to_N16_centers_bitexact=feature_exact,F_to_N16_centers_allclose_rtol=1e-4,
        F_to_N16_centers_allclose_atol=1e-4,base='Original saved FP32 natural native_L, not replayed M or binary R',
        feature_scope='Single frozen pixel-decoder mask feature at 200x200; learned hierarchy, not captured FPN levels',
        tests=tests,nonzero_readout_route_max_abs_difference=route_difference,
        original_models_or_weights_modified=False,disposable_backward_probes_total=2,
        disposable_readout_perturbations=True,optimizer_created=False,optimizer_steps=0,
        full_training_executed=False,training_eligible=False,new_training=False,AP=False,DEV=False,TEST=False,
        qualified=False,ROOT_actual_GPU0_verified=False,elapsed_seconds=elapsed)
    report_ref=c.write_json(out/'INITIALIZATION_REPORT.json',report_value)
    del head,heads,fgpu,egpu,mgpu,base,target,indices,outside,carrier,packet,native,F,M,e,initials,nonzero_outputs
    gc.collect();torch.cuda.synchronize(0);c.check()
    return dict(report_ref=report_ref,checkpoint_refs=checkpoints,images=1,qualified=False,
                training_eligible=False,new_training=False,AP=False,DEV=False,TEST=False)

