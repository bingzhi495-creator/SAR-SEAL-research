"""Unbound N28 fixed24 geometry-only DEV export and full-population COCO.

Draft code, not executed on actual DEV. No scientific libraries are imported
until the admitted caller supplies them. GPU prediction never opens DEV GT.
"""
import copy
import gc
import hashlib
import json
from pathlib import Path
import time
import types

MODES=['SELECTIVE','WHOLE_NODE']
SEEDS=[202610172801,202610172802,202610172803]
R_AP=68.09938494745545
N26_AP=69.53736693986352


def need(ok,message):
    if not ok:raise RuntimeError(message)


def read(c,value):
    return json.loads(Path(c.closed(value)).read_bytes())


def load(c,value,name):
    path=Path(c.closed(value));raw=path.read_bytes()
    need(len(raw)==value['bytes'] and hashlib.sha256(raw).hexdigest()==value['sha256'],
         'Exact closed source')
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def validate_endpoint(spec,training,checkpoint):
    """Pure metadata gate; finite tensor/state equality is checked by GPU body."""
    seed=spec['seed'];result=training['result']
    need(type(seed) is int and seed in SEEDS
         and training['schema']=='ROOT-N28-dense-routing-paired-train-qualified-v1'
         and training['qualified'] is True and training['full_24_epochs_complete'] is True
         and training['seed']==result['seed']==checkpoint['seed']==seed
         and training['end_epoch']==result['end_epoch']==checkpoint['completed_epoch']==24
         and result['checkpoint_ref']==spec['checkpoint']
         and result['cumulative_updates_per_arm']==checkpoint['cumulative_updates_per_arm']==17784
         and result['new_training'] is True
         and all(result[key] is False for key in ('AP','DEV','TEST')),
         'Only the requested seed and actual accepted fixed24 endpoint')
    need(checkpoint['schema']=='N28-dense-routing-paired-train-state-v1'
         and checkpoint['modes']==MODES and set(checkpoint['models'])==set(MODES)
         and checkpoint['factory_source']==training['factory_source']==spec['factory_source']
         and checkpoint['train_core_source']==training['train_core_source']==spec['train_core_source']
         and checkpoint['reader_source']==training['reader_source']==spec['training_reader_source']
         and len(checkpoint['history'])==24 and checkpoint['history']==training['history']
         and [row['epoch'] for row in checkpoint['history']]==list(range(1,25)),
         'Exact original construction/source and full original training history')
    return seed


def state_digest(state):
    records=[]
    for name,value in state.items():
        array=value.detach().cpu().contiguous().numpy()
        records.append(dict(name=name,shape=list(array.shape),dtype=str(array.dtype),
                            sha256=hashlib.sha256(array.tobytes()).hexdigest()))
    return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def gpu(torch,np,spec,c,pre,context):
    """Draft real inference path; requires a separately owned actual execution."""
    from pycocotools import mask as masks
    started=time.monotonic()
    need(not torch.cuda.is_initialized(),'Cold CUDA before exact CPU checkpoint restoration')
    training=read(c,spec['training_qualified'])
    checkpoint=torch.load(str(c.closed(spec['checkpoint'])),map_location='cpu')
    seed=validate_endpoint(spec,training,checkpoint)
    factory=load(c,spec['factory_source'],'N28_DEV_ORIGINAL_FACTORY')
    renderer=load(c,spec['train_core_source'],'N28_DEV_ORIGINAL_RENDERER')
    reader_module=load(c,spec['reader_source'],'N28_DEV_DENSE_READER')
    contract=load(c,spec['population_source'],'N28_DEV_POPULATION_CONTRACT')
    encoder=load(c,context['encoder_source'],'N28_DEV_ORIGINAL_RLE_ENCODER')
    reader=reader_module.prepare(torch,c,context)
    native=read(c,context['native_predictions'])
    need(len(native)==18700,'Complete original18700 prediction dictionaries')
    outside_cpu=torch.get_rng_state().clone()
    heads={};initial_digests={}
    with torch.random.fork_rng(devices=[]):
        for mode in MODES:
            head=factory.make_head(torch,mode)
            head.load_state_dict(checkpoint['models'][mode],strict=True)
            need(sum(p.numel() for p in head.parameters())==2978555
                 and all(t.device.type=='cpu' and (not t.is_floating_point() or
                     (t.dtype==torch.float32 and bool(torch.isfinite(t).all())))
                         for t in head.state_dict().values()),'Exact finite original dense head/BN state')
            head.eval().requires_grad_(False)
            initial_digests[mode]=state_digest(head.state_dict());heads[mode]=head
    del head,checkpoint
    need(torch.equal(torch.get_rng_state(),outside_cpu) and not torch.cuda.is_initialized(),
         'CPU-only construction/restoration with original external RNG preserved')
    torch.cuda.set_per_process_memory_fraction(4096*1024**2/torch.cuda.get_device_properties(0).total_memory,0)
    torch.cuda.reset_peak_memory_stats(0)
    for head in heads.values():head.to('cuda:0')
    del head
    outside_cuda=torch.cuda.get_rng_state(0).cpu().clone()
    first,stop=spec['dataset_range']
    need([first,stop]==pre['GPU_ranges'][spec['shard_index']],'Fixed preregistered image shard')
    out=Path(spec['output_directory']);out.mkdir()
    predictions={mode:[] for mode in MODES};identity_predictions=[];support_records=[];proofs=[]
    cost={mode:dict(forward_calls=0,queries=0,forward_seconds=0.,render_seconds=0.,RLE_seconds=0.,
                    changed_inside_Omega=0,changed_outside_Omega=0) for mode in MODES}
    io_seconds=0.;identity_seconds=0.
    with torch.no_grad():
        for index in range(first,stop):
            c.check();tick=time.monotonic();item=reader.load(index)
            io_seconds+=time.monotonic()-tick
            row=reader.records[index];image_id=row['image_id'];qids=item['query_ids']
            originals=native[index*100:(index+1)*100]
            need([(v['image_id'],v['query_id']) for v in originals]==[(image_id,q) for q in qids],
                 'Canonical original image/export-query sequence')
            full_F=item['F'].to('cuda:0')
            rles={mode:[] for mode in MODES};identity_rles=[]
            for begin in range(0,100,4):
                c.check();queries=qids[begin:begin+4]
                selected=torch.tensor(queries,dtype=torch.int64,device='cpu')
                e=item['e'].index_select(0,selected).to('cuda:0')
                M=item['M'].index_select(0,selected).to('cuda:0')
                L=item['L'].index_select(0,selected).to('cuda:0')
                native_binary=L>0;Omega=L.abs()<=2
                torch.cuda.synchronize(0);tick=time.monotonic()
                zero_delta=torch.zeros((len(queries),1,200,200),dtype=torch.float32,device='cuda:0')
                rendered_zero=renderer.render_delta(torch,zero_delta,item['identity'])
                need(torch.equal(rendered_zero,torch.zeros_like(L)) and torch.equal(L+rendered_zero,L),
                     'All query/grid zero-residual renderer must exactly preserve original GPU native_L')
                zero_binary=(L+rendered_zero)>0
                for offset,q in enumerate(queries):
                    zero_rle=encoder.segmentation(np,masks,zero_binary[offset].cpu().numpy().astype(np.uint8))
                    need(zero_rle==originals[begin+offset]['segmentation'],
                         'All zero-residual original native RLEs exact, including empty queries')
                    identity_rles.append(zero_rle)
                    support_records.append(dict(dataset_index=index,image_id=image_id,query_id=q,
                        Omega=encoder.segmentation(np,masks,Omega[offset].cpu().numpy().astype(np.uint8))))
                torch.cuda.synchronize(0);identity_seconds+=time.monotonic()-tick
                del zero_delta,rendered_zero,zero_binary
                for mode in MODES:
                    torch.cuda.synchronize(0);tick=time.monotonic()
                    delta=heads[mode](full_F,e,M)
                    torch.cuda.synchronize(0);cost[mode]['forward_seconds']+=time.monotonic()-tick
                    need(tuple(delta.shape)==(len(queries),1,200,200) and delta.dtype==torch.float32
                         and bool(torch.isfinite(delta).all()),'Finite FP32 full-field residual for every query')
                    tick=time.monotonic()
                    corrected=L+renderer.render_delta(torch,delta,item['identity'])
                    need(corrected.shape==L.shape and corrected.dtype==torch.float32
                         and bool(torch.isfinite(corrected).all()),'Original-grid all-mask-writable L+delta')
                    binary=corrected>0;changed=binary!=native_binary
                    cost[mode]['changed_inside_Omega']+=int((changed&Omega).sum().item())
                    cost[mode]['changed_outside_Omega']+=int((changed&~Omega).sum().item())
                    torch.cuda.synchronize(0);cost[mode]['render_seconds']+=time.monotonic()-tick
                    tick=time.monotonic()
                    for offset in range(len(queries)):
                        rles[mode].append(encoder.segmentation(np,masks,binary[offset].cpu().numpy().astype(np.uint8)))
                    cost[mode]['RLE_seconds']+=time.monotonic()-tick
                    cost[mode]['forward_calls']+=1;cost[mode]['queries']+=len(queries)
                    del delta,corrected,binary,changed
                del selected,e,M,L,native_binary,Omega
            calibrated=contract.replace_segmentations(originals,identity_rles,image_id,qids)
            need(calibrated==originals,'Every original native prediction dictionary exactly calibrated')
            identity_predictions.extend(calibrated)
            for mode in MODES:
                predictions[mode].extend(contract.replace_segmentations(originals,rles[mode],image_id,qids))
            proofs.append(dict(dataset_index=index,image_id=image_id,queries=100,
                               native_RLE_and_prediction_dictionaries_exact=True,
                               zero_residual_native_L_exact=True,original_full_S0_class_order_preserved=True))
            del item,full_F,rles,identity_rles,calibrated,originals
            gc.collect();c.check()
            print(json.dumps(dict(stage='N28_FIXED24_DENSE_DEV',dataset_index=index,
                                  complete_images=len(proofs),scope_images=stop-first)),flush=True)
    need(all(cost[mode]['queries']==(stop-first)*100 and cost[mode]['forward_calls']==(stop-first)*25
             and initial_digests[mode]==state_digest(heads[mode].state_dict())
             and all(p.grad is None and not p.requires_grad for p in heads[mode].parameters())
             for mode in MODES),'All query coverage and completely frozen trained model/BN state')
    need(torch.equal(torch.get_rng_state(),outside_cpu)
         and torch.equal(torch.cuda.get_rng_state(0).cpu(),outside_cuda),'No inference randomness consumed')
    peak=int(torch.cuda.max_memory_allocated(0))
    for head in heads.values():head.cpu()
    del head,heads
    gc.collect();torch.cuda.synchronize(0);torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache();torch.cuda.synchronize(0)
    cleanup=dict(allocated_after=int(torch.cuda.memory_allocated(0)),reserved_after=int(torch.cuda.memory_reserved(0)))
    need(cleanup['allocated_after']==cleanup['reserved_after']==0 and peak<=4096*1024**2,
         'Release all CUDA aliases before ending owned prediction leaf')
    refs={mode:c.write_json(out/(mode+'_PREDICTIONS.json'),values) for mode,values in predictions.items()}
    identity_ref=c.write_json(out/'IDENTITY_PREDICTIONS.json',identity_predictions)
    support_ref=c.write_json(out/'OMEGA_SUPPORT.json',support_records)
    report=dict(schema='N28-fixed24-dense-DEV-export-pending-v1',seed=seed,dataset_range=[first,stop],
        shard_index=spec['shard_index'],images=stop-first,queries=(stop-first)*100,
        predictions=refs,identity_predictions=identity_ref,Omega_support=support_ref,proofs=proofs,
        training_qualified=spec['training_qualified'],checkpoint=spec['checkpoint'],
        factory_source=spec['factory_source'],train_core_source=spec['train_core_source'],
        source=spec['body_source'],spec=c.execution_spec_ref,GPU_admission=c.execution_admission_ref,
        phase=spec['phase'],prelock=spec['prelock'],dev_context=spec['dev_context'],
        fixed_epoch=24,updates_per_arm=17784,query_batch=4,original_full_S0_class_order_preserved=True,
        native_all_RLE_and_identity_prediction_dictionaries_exact=True,zero_residual_native_L_exact=True,
        original_teacher_forward=False,GT_opened_for_prediction=False,new_training=False,DEV=True,AP=False,TEST=False,
        frozen_head_state_digests=initial_digests,model_BN_RNG_unchanged=True,
        cost=cost,cache_IO_checks_seconds=io_seconds,identity_renderer_and_export_seconds=identity_seconds,
        elapsed_seconds=time.monotonic()-started,cleanup=cleanup,torch_peak_allocated_bytes=peak,
        end_to_end_backbone_cost_included=False,standalone_FPS_claimed=False,
        qualified=False,innovation_established=False,goal_complete=False)
    report_ref=c.write_json(out/'DEV_REPORT.json',report)
    del reader,native,predictions,identity_predictions,support_records
    gc.collect();c.check()
    return dict(report_ref=report_ref,predictions=refs,identity_predictions=identity_ref,
                Omega_support=support_ref,images=stop-first,queries=(stop-first)*100,
                dataset_range=[first,stop],seed=seed,fixed_epoch=24,AP=False)


def official_metrics(np,c,official,COCOeval,coco,values,ids):
    """Original COCO12 plus raw per-IoU precision; one complete population."""
    observed=[]
    def observe(*args,**kwargs):
        evaluation=COCOeval(*args,**kwargs);observed.append(evaluation)
        return evaluation
    result=official.coco_metrics(coco,copy.deepcopy(values),ids,observe)
    need(len(observed)==1 and len(result['stats'])==12 and result['iouType']=='segm'
         and result['maxDets']==[1,10,100] and result['original_GT_preserved'] is True
         and result['score_filter'] is None and result['official_COCO_crowd_ignore_semantics'] is True,
         'Full original official COCO12 semantics')
    evaluation=observed[0];thresholds=np.linspace(.5,.95,10)
    need(list(evaluation.params.maxDets)==[1,10,100] and list(evaluation.params.catIds)==[1]
         and len(evaluation.params.imgIds)==187 and set(evaluation.params.imgIds)==set(ids)
         and np.array_equal(evaluation.params.iouThrs,thresholds),'Complete187/380 single-category COCO')
    per_IoU=[]
    for ti,threshold in enumerate(thresholds):
        precision=evaluation.eval['precision'][ti,:,0,0,2]
        valid=precision[precision>-1]
        need(len(valid)>0,'Valid original precision entries')
        per_IoU.append(dict(IoU=float(threshold),AP_percent=100*float(valid.mean())))
    need(abs(sum(v['AP_percent'] for v in per_IoU)/1000-result['stats'][0])<=1e-12,
         'Raw precision reconstructs official AP')
    c.check()
    return result,per_IoU,evaluation


def cpu(torch,np,spec,c,pre,context):
    """Draft complete-population CPU evaluation, never shard-AP averaging."""
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    need(not torch.cuda.is_initialized(),'CPU evaluation starts and stays CUDA cold')
    started=time.monotonic();seed=spec['seed']
    training=read(c,spec['training_qualified'])
    need(training['schema']=='ROOT-N28-dense-routing-paired-train-qualified-v1'
         and training['qualified'] is True and training['full_24_epochs_complete'] is True
         and training['end_epoch']==24 and training['seed']==seed
         and training['result']['checkpoint_ref']==spec['checkpoint']
         and training['result']['cumulative_updates_per_arm']==17784,
         'Only actual accepted fixed24, no DEV-selected endpoint')
    contract=load(c,spec['population_source'],'N28_COCO_POPULATION_CONTRACT')
    geometry=load(c,spec['geometry_source'],'N28_FIXED_NATIVE_PAIR_GEOMETRY')
    coco_context=read(c,spec['coco_context'])
    need(coco_context['schema']=='ROOT-N28-fixed24-original-COCO-context-v1'
         and coco_context['image_ids']==context['image_ids']
         and coco_context['legacy_ROOT_result']==context['legacy_ROOT_result'],
         'Original prediction and COCO scope association')
    official=load(c,coco_context['COCO_source'],'N28_ORIGINAL_OFFICIAL_COCO')
    coco=COCO(str(c.closed(coco_context['DEV'])))
    ids=list(coco.imgs)
    need(ids==context['image_ids'] and len(ids)==187 and len(coco.anns)==380 and set(coco.cats)=={1},
         'Exact original187 images/380GT/category1, no annotation conversion')
    native=read(c,context['native_predictions']);strong=read(c,context['R_predictions'])
    need(len(native)==len(strong)==18700
         and [(p['image_id'],p['query_id']) for p in native]
             ==[(p['image_id'],p['query_id']) for p in strong]
         and all({k:v for k,v in p.items() if k!='segmentation'}
                 =={k:v for k,v in q.items() if k!='segmentation'} for p,q in zip(native,strong)),
         'Original native/R complete population and original fixed S0 metadata')
    legacy=read(c,context['legacy_ROOT_result'])
    need(legacy['qualified'] is True,'Original native/R official result Root accepted')
    pieces={mode:[] for mode in MODES+['identity']};support_records=[];gpu_costs=[]
    need(len(spec['accepted_GPU_shards'])==4,'All four original owned GPU shards required')
    for ordinal,reference in enumerate(spec['accepted_GPU_shards']):
        accepted=read(c,reference)
        need(accepted['schema']=='ROOT-N28-fixed24-dense-DEV-export-qualified-v1'
             and accepted['qualified'] is True and accepted['operation']=='GPU_N28_DENSE_FIXED24_DEV',
             'Original actual0 and Root accepted dense GPU shard')
        report=read(c,accepted['result']['report_ref'])
        need(report['schema']=='N28-fixed24-dense-DEV-export-pending-v1'
             and report['seed']==seed and report['fixed_epoch']==24
             and report['training_qualified']==spec['training_qualified']
             and report['checkpoint']==spec['checkpoint'] and report['shard_index']==ordinal
             and report['dataset_range']==pre['GPU_ranges'][ordinal]
             and report['dev_context']==spec['dev_context'] and report['prelock']==spec['prelock']
             and report['native_all_RLE_and_identity_prediction_dictionaries_exact'] is True
             and report['zero_residual_native_L_exact'] is True
             and report['original_full_S0_class_order_preserved'] is True
             and report['model_BN_RNG_unchanged'] is True
             and report['original_teacher_forward'] is report['GT_opened_for_prediction'] is False,
             'Same seed/fixed endpoint/context and original complete native calibration')
        for mode in MODES:
            pieces[mode].append(dict(dataset_range=report['dataset_range'],predictions=read(c,report['predictions'][mode])))
        pieces['identity'].append(dict(dataset_range=report['dataset_range'],
                                      predictions=read(c,report['identity_predictions'])))
        current_support=read(c,report['Omega_support'])
        first,stop=report['dataset_range']
        need(len(current_support)==(stop-first)*100
             and [(v['image_id'],v['query_id']) for v in current_support]
                 ==[(v['image_id'],v['query_id']) for v in native[first*100:stop*100]]
             and all(v['dataset_index']==first+i//100 for i,v in enumerate(current_support)),
             'Complete original query-aligned Omega support from GPU native_L')
        support_records.extend(current_support);gpu_costs.append(report['cost'])
    predictions={mode:contract.merge_predictions(pieces[mode],native,pre['GPU_ranges']) for mode in MODES}
    identity=contract.merge_predictions(pieces['identity'],native,pre['GPU_ranges'])
    need(identity==native and len(support_records)==18700,'All18700 exact zero-residual identity predictions')
    predictions.update(native_official=identity,R=strong)
    out=Path(spec['output_directory']);out.mkdir()
    results={};native_evaluation=None
    for mode in ['native_official','R']+MODES:
        c.check();tick=time.monotonic()
        metrics,per_IoU,evaluation=official_metrics(np,c,official,COCOeval,coco,predictions[mode],ids)
        if mode in ('native_official','R'):
            legacy_key='native_official' if mode=='native_official' else 'strong_fixed_E24_Omega'
            need(metrics==legacy['control_results'][legacy_key]['COCO'],
                 'Every original native/R COCO12 field reproduced exactly')
        if mode=='native_official':native_evaluation=evaluation
        results[mode]=dict(COCO=metrics,per_IoU=per_IoU,official_COCO_seconds=time.monotonic()-tick,
                           predictions=c.write_json(out/(mode+'_PREDICTIONS.json'),predictions[mode]))
        del evaluation
    geometry_full=geometry.diagnose(np,masks,coco,native_evaluation,predictions,support_records)
    geometry_result=geometry_full['summary']
    need(geometry_result['scope']['total_DEV_GT']==380,'Geometry coverage denominator is complete DEV GT')
    pairs=geometry_full['private_pair_records']
    pair_ref=c.write_json(out/'FIXED_NATIVE_MATCHED_GEOMETRY_PRIVATE.json',pairs)
    del native_evaluation
    scores={mode:100*value['COCO']['stats'][0] for mode,value in results.items()}
    need(scores['R']==R_AP,'Original strong R may not move')
    paired_delta=scores['SELECTIVE']-scores['WHOLE_NODE']
    threshold=max(scores['WHOLE_NODE'],N26_AP);delta=scores['SELECTIVE']-threshold
    suggested='STOP_CURRENT_RECIPE' if delta<=0 else ('AT_MOST_SECOND_PRELOCKED_SEED' if delta<1 else 'PROCEED_PRELOCKED_REPEATS')
    need(not torch.cuda.is_initialized(),'Official evaluation and geometry are CPU only')
    report=dict(schema='N28-fixed24-complete187-COCO-pending-v1',seed=seed,fixed_epoch=24,
        images=187,GT=380,queries=18700,source=spec['body_source'],spec=c.execution_spec_ref,
        CPU_admission=c.execution_admission_ref,phase=spec['phase'],prelock=spec['prelock'],
        training_qualified=spec['training_qualified'],checkpoint=spec['checkpoint'],
        dev_context=spec['dev_context'],coco_context=spec['coco_context'],
        accepted_GPU_shards=spec['accepted_GPU_shards'],results=results,AP_points=scores,
        paired_Delta_AP_points=paired_delta,conservative_reference_AP=N26_AP,
        utility_threshold_AP=threshold,Delta_AP_points=delta,first_seed_gate_suggestion=suggested,
        seed_gate_decided=False,geometry=geometry_result,private_pair_records=pair_ref,
        GPU_costs_by_shard=gpu_costs,original_native_R_COCO12_exact=True,
        fixed_full_S0_class_query_order_preserved=True,epoch_selected_on_DEV=False,
        no_shard_AP_average=True,geometry_pair_selection_uses_new_method=False,
        new_training=False,AP=True,DEV=True,TEST=False,formal_TEST=False,cross_dataset=False,
        elapsed_seconds=time.monotonic()-started,qualified=False,innovation_established=False,goal_complete=False)
    report_ref=c.write_json(out/'COCO_REPORT.json',report)
    del predictions,pieces,native,strong,identity,support_records
    gc.collect();c.check()
    return dict(report_ref=report_ref,seed=seed,fixed_epoch=24,images=187,GT=380,queries=18700,
                AP=True,AP_points=scores,paired_Delta_AP_points=paired_delta,
                Delta_AP_points=delta,first_seed_gate_suggestion=suggested)


def run(torch,np,spec,c,pre):
    context=read(c,spec['dev_context'])
    need(context['schema']=='ROOT-N28-fixed24-dense-native-DEV-context-v1',
         'Fixed original complete DEV prediction context')
    need(pre['schema']=='N28-fixed24-dense-routing-DEV-prelock-v1'
         and pre['modes']==MODES and pre['seeds']==SEEDS and spec['seed'] in SEEDS
         and pre['images']==187 and pre['GT']==380 and pre['queries_per_image']==100
         and pre['GPU_ranges']==[[0,47],[47,94],[94,141],[141,187]]
         and pre['fixed_epoch']==24 and pre['updates_per_arm']==17784
         and pre['query_batch']==4 and pre['binary_rule']=='>0'
         and pre['fixed_original_S0'] is True and pre['GT_used_for_prediction'] is False
         and pre['score_update'] is False and pre['query_filter'] is None
         and pre['NMS'] is False and pre['historical_R_AP']==R_AP and pre['N26_mean_AP']==N26_AP,
         'Exact preregistered masks-only whole-population evaluation')
    if spec['operation']=='GPU_N28_DENSE_FIXED24_DEV':return gpu(torch,np,spec,c,pre,context)
    need(spec['operation']=='CPU_N28_DENSE_FIXED24_COCO','Only GPU prediction or CPU full COCO operation')
    return cpu(torch,np,spec,c,pre,context)
