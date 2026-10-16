"""GT-free four-head cached GPU export; separate complete official CPU COCO."""
import gc
import importlib.util
import json
import time
from pathlib import Path


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def load(c, ref, name):
    sp = importlib.util.spec_from_file_location(name, str(c.closed(ref)))
    module = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(module)
    return module


def read(c, ref):
    return json.loads(c.closed(ref).read_bytes())


def gpu(torch, np, spec, c, pre, ctx):
    from pycocotools import mask as masks
    core = load(c, spec['core_source'], 'N23_FIXEDDEV_OMEGA_COORDINATE_CORE')
    factory = load(c, ctx['factory_source'], 'N23_FIXEDDEV_ORIGINAL_POINT_FACTORY')
    n10 = load(c, ctx['encoder_source'], 'N23_FIXEDDEV_ORIGINAL_RLE_EXPORT')
    trained = read(c, ctx['training_ROOT_result'])
    legacy = read(c, ctx['legacy_ROOT_result'])
    need(trained['qualified'] and trained['last_extra_epoch'] == 24
         and trained['result']['cumulative_extra_updates_each'] == 39864
         and legacy['qualified'], 'Accepted fixed training endpoint and original DEV cache')
    native, strong = read(c, ctx['native_predictions']), read(c, ctx['R_predictions'])
    need(len(native) == len(strong) == 18700, 'Complete original unfiltered predictions')
    native_by, strong_by = {}, {}
    for target, values in ((native_by, native), (strong_by, strong)):
        for v in values:
            key = (v['image_id'], v['query_id'])
            need(key not in target and v['category_id'] == 1, 'Unique original image/query/category')
            target[key] = v
    need(set(native_by) == set(strong_by), 'Original native/R population identities exact')
    heads, weight_proofs = {}, {}
    outside_CPU_RNG = torch.get_rng_state().clone()
    for arm in pre['heads']:
        cpref = ctx['checkpoints'][arm]
        cp = torch.load(str(c.closed(cpref)), map_location='cpu')
        if arm != 'R':
            need(cpref == trained['result']['checkpoint_refs'][arm]
                 and cp['meta']['arm'] == arm and cp['meta']['completed_extra_epochs'] == 24
                 and cp['meta']['updates_per_arm'] == 39864, 'Exact trained arm and endpoint metadata')
        else:
            need(cpref == ctx['strong_initial_checkpoint'], 'Original strong R weight identity')
        head = factory.make_point_head(torch, 0)
        need(set(cp['state_dict']) == set(head.state_dict()), 'Exact eight weight names')
        head.load_state_dict(cp['state_dict'], strict=True)
        need(all(p.dtype == torch.float32 and p.device.type == 'cpu' and bool(torch.isfinite(p).all())
                 for p in head.parameters()) and len(list(head.parameters())) == 8
             and sum(p.numel() for p in head.parameters()) == 40497 and not list(head.buffers()),
             'Same finite CPU FP32 8-tensor/40497-parameter architecture')
        head.eval().requires_grad_(False)
        heads[arm] = head
        weight_proofs[arm] = dict(checkpoint=cpref, parameter_tensors=8, parameter_scalars=40497)
        del cp, head
    need(torch.equal(outside_CPU_RNG, torch.get_rng_state()) and not torch.cuda.is_initialized(),
         'Private construction preserves CPU RNG; all CPU loads precede CUDA')
    torch.cuda.set_per_process_memory_fraction(4096 * 1024**2 / torch.cuda.get_device_properties(0).total_memory, 0)
    heads = {a: h.to('cuda:0') for a, h in heads.items()}
    start, stop = spec['dataset_range']
    need([start, stop] == pre['GPU_ranges'][spec['shard_index']], 'Fixed physical shard only')
    rows = ctx['rows'][start:stop]
    output = Path(spec['output_directory'])
    output.mkdir()
    predictions = {a: [] for a in pre['heads']}
    proofs, sumK, zeroK = [], 0, 0
    cost = {a: dict(forward_calls=0, forward_points=0, forward_seconds=0., RLE_seconds=0.) for a in pre['heads']}
    io_seconds = 0.
    head = None  # Also valid when an entire fixed shard has K0.
    manifest_cache = {}
    for row in rows:
        c.check()
        idx, image_id = row['dataset_index'], row['image_id']
        need(idx == start + len(proofs) and image_id == ctx['image_ids'][idx], 'Canonical image order')
        mr = row['original_capture_manifest']
        if mr['path'] not in manifest_cache:
            manifest_cache[mr['path']] = read(c, mr)
        manifest = manifest_cache[mr['path']]
        original_row = manifest['rows'][idx-manifest['dataset_ordinal_range'][0]]
        need(original_row['bundle'] == row['bundle'] and original_row['dataset_index'] == idx
             and original_row['image_id'] == image_id and original_row['qids'] == row['qids']
             and original_row['shared_actual_image_inputs'] == row['shared_locator']
             and manifest['capture_binding'] == row['capture_binding'], 'Original qualified manifest join')
        tick = time.monotonic()
        packet = torch.load(str(c.closed(row['bundle'])), map_location='cpu')
        need(packet['schema'] == 'N16-actual-GPU-image-capture-bundle-v1'
             and packet['binding'] == row['capture_binding']
             and packet['dataset_index'] == idx and packet['image_id'] == image_id,
             'Original physical bundle identity/binding')
        shared = packet['members'][row['shared_locator']]
        need(shared['schema'] == 'N16-actual-GPU-image-shared-inputs-v1'
             and shared['actual_GPU_F_captured'] is True and shared['CPU_F_rerendered'] is False
             and shared['dataset_index'] == idx and shared['image_id'] == image_id
             and shared['natural_query_ids'] == list(range(100))
             and shared['native_export_query_ids'] == row['qids']
             and sorted(row['qids']) == list(range(100)), 'Actual GPU shared natural100 identities')
        F, e, L, union_t = shared['F_unique'], shared['e0'], shared['native_L'], shared['union_flat_indices']
        need(F.device.type == e.device.type == L.device.type == union_t.device.type == 'cpu'
             and F.dtype == e.dtype == L.dtype == torch.float32
             and tuple(L.shape) == (100,) + tuple(row['original_shape'])
             and tuple(e.shape) == (100, 256) and F.shape == (len(union_t), 256)
             and union_t.dtype == torch.int64 and union_t.ndim == 1
             and all(not t.requires_grad and bool(torch.isfinite(t).all()) for t in (F,e,L)),
             'Original finite CPU F/e/L and exact original-grid shapes')
        # Drop unrelated saved scouts/chunks before predicting. No teacher, image or GT is loaded.
        del packet
        union = union_t.numpy()
        io_seconds += time.monotonic()-tick
        image_K, image_zero = 0, 0
        with torch.no_grad():
            for q in row['qids']:
                c.check()
                native_L = L[q].numpy()
                plan = core.query_plan(np, native_L, union, expected_K=shared['K_counts'][q])
                indices, positions, K = plan['Omega_flat_indices'], plan['union_positions'], plan['K']
                baseline = native_by[(image_id,q)]
                native_rle = n10.segmentation(np, masks, (native_L > 0).astype(np.uint8))
                need(native_rle == baseline['segmentation'], 'Reconstruct exact original native mask/RLE')
                logits = {a: np.empty(K, dtype=np.float32) for a in pre['heads']}
                for first in range(0, K, 2048):
                    c.check()
                    end = min(first+2048, K)
                    f = F.index_select(0, torch.from_numpy(positions[first:end].copy())).contiguous().to('cuda:0')
                    emb = e[q].to('cuda:0').unsqueeze(0).expand(end-first,256)
                    n = L[q].reshape(-1).index_select(0,torch.from_numpy(indices[first:end].copy())).to('cuda:0')
                    patch = n.new_zeros((end-first,49))
                    for arm, head in heads.items():
                        torch.cuda.synchronize(0)
                        tick = time.monotonic()
                        result = head(f,emb,n,patch,arm='PR')
                        combined = n + result['residual']
                        need(combined.dtype == torch.float32 and combined.shape == (end-first,)
                             and bool(torch.isfinite(combined).all()), 'Actual original-device FP32 L+u')
                        logits[arm][first:end] = combined.detach().cpu().numpy()
                        torch.cuda.synchronize(0)
                        cost[arm]['forward_seconds'] += time.monotonic()-tick
                        cost[arm]['forward_calls'] += 1
                        cost[arm]['forward_points'] += end-first
                        del combined, result
                    del f,emb,n,patch
                for arm in pre['heads']:
                    tick = time.monotonic()
                    binary = core.apply_query_logits(np,native_L,indices,logits[arm])
                    item = dict(baseline)
                    item['segmentation'] = n10.segmentation(np,masks,binary.astype(np.uint8))
                    need({k:v for k,v in item.items() if k!='segmentation'} ==
                         {k:v for k,v in baseline.items() if k!='segmentation'}, 'Score/class/query/order never updated')
                    if arm == 'R':
                        need(item == strong_by[(image_id,q)], 'Recomputed R must match historical prediction dictionary exactly')
                    predictions[arm].append(item)
                    cost[arm]['RLE_seconds'] += time.monotonic()-tick
                    del binary,item
                image_K += K
                image_zero += int(K==0)
                del native_L,plan,indices,positions,logits,baseline,native_rle
        need(all(len(predictions[a]) == (len(proofs)+1)*100 for a in pre['heads']), 'Every natural query exported')
        proofs.append(dict(dataset_index=idx,image_id=image_id,queries=100,sumK=image_K,K0=image_zero,
                           native_all100_RLE_exact=True,R_all100_prediction_dictionaries_exact=True,
                           every_head_same_Omega_K_original_score=True))
        sumK += image_K
        zeroK += image_zero
        del shared,F,e,L,union_t,union,original_row,manifest
        gc.collect()
        c.check()
        print(json.dumps(dict(stage='N23_FIXED_E24_DEV_EXPORT',dataset_index=idx,completed=len(proofs),
                              scope_images=len(rows),R_exact=True)),flush=True)
    refs = {a:c.write_json(output/(a+'_PREDICTIONS.json'),v) for a,v in predictions.items()}
    need(all(cost[a]['forward_points'] == sumK for a in pre['heads']), 'All four heads evaluated same exact K')
    report = dict(schema='N23-fourhead-cached-DEV-export-report-v1',dataset_range=[start,stop],
                  shard_index=spec['shard_index'],images=len(rows),queries=len(rows)*100,
                  image_ids=[r['image_id'] for r in rows],heads=pre['heads'],weight_proofs=weight_proofs,
                  predictions=refs,proofs=proofs,sumK=sumK,K0=zeroK,cost=cost,
                  cache_hash_load_validation_seconds=io_seconds,teacher_forward_calls=0,GT_used_for_prediction=False,
                  score_update=False,query_filter=None,point_chunk=2048,AP=False,qualified=False,
                  R_all_prediction_dictionaries_exact=True,native_all_RLE_exact=True,
                  full_backbone_inference_cost_included=False,standalone_end_to_end_FPS_claimed=False)
    rr = c.write_json(output/'DEV_REPORT.json',report)
    del heads,predictions,native,strong,native_by,strong_by,manifest_cache,head,outside_CPU_RNG
    gc.collect()
    return dict(report_ref=rr,predictions=refs,images=len(rows),queries=len(rows)*100,
                dataset_range=[start,stop],sumK=sumK,K0=zeroK,R_exact=True,AP=False)


def cpu(torch,np,spec,c,pre,ctx):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    reader = load(c,ctx['reader_source'],'N23_FIXEDDEV_ORIGINAL_POPULATION_READER')
    official = load(c,ctx['COCO_source'],'N23_FIXEDDEV_ORIGINAL_OFFICIAL_COCO')
    coco = COCO(str(c.closed(ctx['DEV'])))
    ids = list(coco.imgs)
    need(ids == ctx['image_ids'] and len(ids)==187 and len(coco.anns)==380 and set(coco.cats)=={1},
         'Complete canonical187/380GT/category1, no GT conversion')
    native,strong = read(c,ctx['native_predictions']),read(c,ctx['R_predictions'])
    reader.population(coco,native,ids)
    reader.population(coco,strong,ids)
    legacy = read(c,ctx['legacy_ROOT_result'])
    results = {}
    output = Path(spec['output_directory'])
    output.mkdir()
    for arm in pre['heads']+['native_official']:
        c.check()
        if arm == 'native_official':
            values = native
        else:
            values=[]
            ranges=[]
            for sr in spec['accepted_GPU_shards']:
                qualified=read(c,sr)
                need(qualified['qualified'] and qualified['operation']=='GPU_DEV', 'Each actual GPU shard Root accepted')
                report=read(c,qualified['result']['report_ref'])
                need(report['R_all_prediction_dictionaries_exact'] and report['point_chunk']==2048,
                     'Original R calibrated in each shard')
                ranges.append(report['dataset_range'])
                values.extend(read(c,report['predictions'][arm]))
            need(ranges==pre['GPU_ranges'], 'All four ordered disjoint fixed GPU scopes')
        need([(v['image_id'],v['query_id']) for v in values] == [(v['image_id'],v['query_id']) for v in native],
             'Entire18700 original image/query order, no local AP averages')
        need(all({k:v for k,v in a.items() if k!='segmentation'} ==
                 {k:v for k,v in b.items() if k!='segmentation'} for a,b in zip(values,native)),
             'Full original class/score/query dictionaries preserved')
        if arm=='R':
            need(values==strong, 'Entire original R dictionaries exact before COCO')
        reader.population(coco,values,ids)
        predictions_ref=c.write_json(output/(arm+'_PREDICTIONS.json'),values)
        metrics=official.coco_metrics(coco,values,ids,COCOeval)
        need(len(metrics['stats'])==12 and metrics['iouType']=='segm' and metrics['maxDets']==[1,10,100]
             and metrics['original_GT_preserved'] is True and metrics['score_filter'] is None
             and metrics['official_COCO_crowd_ignore_semantics'] is True, 'Original complete COCO12 semantics')
        if arm in ('R','native_official'):
            key='strong_fixed_E24_Omega' if arm=='R' else arm
            need(metrics==legacy['control_results'][key]['COCO'], 'Native/R all12 raw metrics must exactly reproduce')
        results[arm]=dict(predictions=predictions_ref,COCO=metrics,stats=metrics['stats'])
        c.check()
    scores={a:results[a]['stats'][0]*100 for a in pre['heads']}
    need(scores['R']==pre['historical_R_AP_points'], 'Original strong reference cannot silently move')
    threshold=max(scores['A_COMMON'],scores['A_ACTIVE'],scores['R'],pre['historical_R_AP_points'])
    delta=scores['B_C']-threshold
    decision='STOP_CURRENT_RECIPE' if delta<=0 else ('AT_MOST_SECOND_PRELOCKED_SEED' if delta<1 else 'PROCEED_PRELOCKED_REPEATS')
    report=dict(schema='N23-full187-official-COCO-result-pending-v1',images=187,GT=380,queries=18700,
                results=results,AP_points=scores,Delta_AP_points=delta,decision=decision,
                strong_reference_threshold_points=threshold,accepted_GPU_shards=spec['accepted_GPU_shards'],
                AP=True,qualified=False,innovation_established=False,goal_complete=False,
                frozen_R_all_predictions_and_COCO12_exact=True,formal_TEST=False,cross_dataset=False)
    rr=c.write_json(output/'COCO_REPORT.json',report)
    need(not torch.cuda.is_initialized(), 'Independent full COCO remains CPU only')
    return dict(report_ref=rr,images=187,GT=380,queries=18700,AP=True,AP_points=scores,
                Delta_AP_points=delta,decision=decision,R_exact=True)


def run(torch,np,spec,c,pre):
    ctx=read(c,spec['dev_context'])
    need(ctx['schema']=='ROOT-N23-qualified-cache-DEV-context-v1'
         and len(ctx['rows'])==len(ctx['image_ids'])==187
         and [r['dataset_index'] for r in ctx['rows']]==list(range(187))
         and [r['image_id'] for r in ctx['rows']]==ctx['image_ids'], 'Original complete DEV context')
    if spec['operation']=='GPU_DEV':
        return gpu(torch,np,spec,c,pre,ctx)
    return cpu(torch,np,spec,c,pre,ctx)
