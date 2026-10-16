# encoding: utf-8
"""UNBOUND_NOT_EXECUTED fixed-R, all-query learned-quality scoring evaluation.

No GT is read until the quality head and every CUDA tensor are released.
Only S0*qhat is evaluated; no mask, class, candidate or query is changed.
"""
import copy
import gc
import hashlib
import json
from pathlib import Path
import time
import types

SEEDS = [202610172601, 202610172602, 202610172603]
R_AP = 68.09938494745545
MERGED_KEYS = {'FIT_features','FIT_targets_FP64','FIT_identity_order',
               'DEV_features','DEV_prediction_order','R_prediction_reference'}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def validate_seed(pre, spec):
    need(type(pre['seeds']) is list and pre['seeds'] == SEEDS
         and all(type(s) is int for s in pre['seeds'])
         and type(spec['seed']) is int and spec['seed'] in SEEDS and spec['seed'] in pre['seeds'],
         'Only the three complete prelocked N26 seeds; execution requires separate Root authorization')
    return spec['seed']


def read(c, ref):
    return json.loads(Path(c.closed(ref)).read_text(encoding='utf-8-sig'))


def load(c, ref, name):
    path = Path(c.closed(ref)); data = path.read_bytes()
    need(len(data) == ref['bytes'] and hashlib.sha256(data).hexdigest() == ref['sha256'],
         'Closed source changed: ' + name)
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(data,str(path),'exec'),module.__dict__)
    return module


def state_digest(state):
    records = []
    for name,value in state.items():
        a = value.detach().cpu().contiguous().numpy()
        records.append(dict(name=name,shape=list(a.shape),dtype=str(a.dtype),
                            sha256=hashlib.sha256(a.tobytes()).hexdigest()))
    return dict(sha256=hashlib.sha256(json.dumps(records,sort_keys=True,
                       separators=(',',':')).encode()).hexdigest(),tensors=records)


def rescore(np, quality, original, qualities, expected_order):
    need(len(original) == 18700 and [(v['image_id'],v['query_id']) for v in original] == expected_order
         and all(v['category_id'] == 1 for v in original), 'Complete original R identity, class and order required')
    scores = np.asarray([v['score'] for v in original],dtype=np.float64)
    product = quality.score_product(np,scores,qualities)
    changed = copy.deepcopy(original)
    for index,value in enumerate(changed):
        value['score'] = float(product[index])
    need(all(set(a) == set(b) and {k:v for k,v in a.items() if k != 'score'}
             == {k:v for k,v in b.items() if k != 'score'} for a,b in zip(original,changed)),
         'Only score may change; original RLE/class/query/all100 order remain exact')
    return changed, scores, product


def official_metrics(np,c,official,COCOeval,coco,values,ids):
    observed = []
    def factory(*args,**kwargs):
        ev = COCOeval(*args,**kwargs); observed.append(ev); return ev
    # Original evaluator/loadRes may add bookkeeping: never give it immutable input dictionaries.
    result = official.coco_metrics(coco,copy.deepcopy(values),ids,factory)
    need(len(observed) == 1 and len(result['stats']) == 12 and result['iouType'] == 'segm'
         and result['maxDets'] == [1,10,100] and result['original_GT_preserved'] is True
         and result['score_filter'] is None and result['official_COCO_crowd_ignore_semantics'] is True,
         'Complete original official COCO12 semantics required')
    ev = observed.pop(); thresholds = np.linspace(.5,.95,10)
    need(list(ev.params.maxDets) == [1,10,100] and list(ev.params.catIds) == [1]
         and len(ev.params.imgIds) == 187 and set(ev.params.imgIds) == set(ids)
         and np.array_equal(ev.params.iouThrs,thresholds), 'Full187 original COCO scope and IoU thresholds')
    per_T = []
    for ti,t in enumerate(thresholds):
        precision = ev.eval['precision'][ti,:,0,0,2]
        valid = precision[precision > -1]
        need(len(valid) > 0,'Original valid AP precision entries required at each threshold')
        per_T.append(dict(IoU=float(t),AP_percent=100*float(valid.mean())))
    need(abs(sum(v['AP_percent'] for v in per_T)/1000-result['stats'][0]) <= 1e-12,
         'Original precision tensor reconstructs official mean AP')
    del ev,observed; gc.collect(); c.check()
    return result,per_T


def run(torch,np,spec,c,pre):
    started = time.monotonic(); c.check(); seed = validate_seed(pre,spec)
    need(spec['operation'] == 'GPU_QUERY_MASK_QUALITY_EVAL'
         and spec['allocator_limit_MiB'] == 4096 and pre == read(c,spec['prelock']),
         'Pinned prelocked-seed/product-only evaluation and prelock required')
    need(torch.__version__ == '2.0.1+cu118' and torch.get_num_threads() == 2
         and not torch.cuda.is_initialized(), 'Pinned Torch/two threads and cold CUDA required')
    need(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
         and not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic,
         'Same deterministic FP32 policy as quality training required')
    accepted = read(c,spec['inputs_qualified'])
    training = read(c,spec['training_qualified'])
    need(accepted['qualified'] is True and accepted['quality_training_eligible'] is True
         and accepted['result']['merged_ref'] == spec['merged_inputs'], 'Root-qualified complete quality inputs required')
    need(training['qualified'] is True and training['result']['checkpoint_ref'] == spec['checkpoint'],
         'Root-qualified actual training checkpoint must be joined before loading')
    merged = torch.load(str(c.closed(spec['merged_inputs'])),map_location='cpu',weights_only=True)
    need(type(merged) is dict and set(merged) == MERGED_KEYS, 'Exact merged FIT/DEV input container')
    X = merged['DEV_features']
    need(isinstance(X,torch.Tensor) and X.shape == (18700,1040) and X.dtype == torch.float32
         and X.device.type == 'cpu' and not X.requires_grad and bool(torch.isfinite(X).all()),
         'Full detached natural100 DEV quality features')
    context = read(c,spec['prediction_context'])
    need(context['schema'] == 'ROOT-N25-qualified-fixedmask-score-context-v1'
         and merged['R_prediction_reference'] == context['predictions']['R'], 'Exact original qualified R reference')
    order = [tuple(v) for v in merged['DEV_prediction_order']]
    ids = context['image_ids']
    need(order == [tuple(v) for v in context['prediction_order']]
         and len(ids) == len(set(ids)) == 187 and len(order) == len(set(order)) == 18700
         and all(type(q) is int and image_id in ids and 0 <= q < 100 for image_id,q in order)
         and all({q for image_id,q in order if image_id == current} == set(range(100)) for current in ids),
         'Every original image and natural100 query in fixed context order')
    original = read(c,context['predictions']['R'])
    quality = load(c,spec['quality_source'],'N26_EVAL_QUALITY_FACTORY')
    calibrated,original_scores,_ = rescore(np,quality,original,np.ones(18700,dtype=np.float64),order)
    need(calibrated == original, 'qhat=1 must reproduce every original prediction dictionary exactly')
    checkpoint = torch.load(str(c.closed(spec['checkpoint'])),map_location='cpu',weights_only=True)
    need(checkpoint['schema'] == 'N26-query-mask-quality-fixedE24-checkpoint-v1'
         and checkpoint['seed'] == seed and checkpoint['epoch'] == 24 and checkpoint['updates'] == 624
         and checkpoint['parameters'] == 299521 and checkpoint['tensors'] == 6
         and all(checkpoint[k] == spec[k] for k in ('merged_inputs','inputs_qualified','quality_source')),
         'Same factory/input bindings and only the fixedE24 requested-seed checkpoint')
    need(state_digest(checkpoint['state_dict']) == checkpoint['final_digest'], 'Qualified checkpoint full state digest')
    external_cpu = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = quality.build_head(torch)
    need(not torch.cuda.is_initialized() and torch.equal(torch.get_rng_state(),external_cpu),
         'CPU head construction must restore external RNG and keep CUDA cold')
    need(state_digest(head.state_dict()) == checkpoint['initialization_digest'], 'Same complete CPU standard initialization')
    head.load_state_dict(checkpoint['state_dict'],strict=True); head.eval()
    for parameter in head.parameters(): parameter.requires_grad_(False)
    need(sum(p.numel() for p in head.parameters()) == 299521 and len(head.state_dict()) == 6,
         'Only the standard six-tensor quality head is instantiated')
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(4096*1024**2/torch.cuda.get_device_properties(0).total_memory,0)
    torch.cuda.reset_peak_memory_stats(0); external_cuda = torch.cuda.get_rng_state(0).cpu().clone()
    head.to('cuda:0'); qhat = torch.empty(18700,dtype=torch.float32,device='cpu')
    batch_counts = []; prediction_started = time.monotonic()
    with torch.no_grad():
        for first in range(0,18700,64):
            c.check(); xb = X[first:first+64].to('cuda:0'); values = head(xb)
            need(values.shape == (len(xb),) and values.dtype == torch.float32
                 and bool(torch.isfinite(values).all()) and bool(((values >= 0) & (values <= 1)).all()),
                 'One finite FP32 [0,1] prediction for every natural query')
            qhat[first:first+len(xb)] = values.detach().cpu(); batch_counts.append(len(xb))
            del xb,values
    torch.cuda.synchronize(); prediction_seconds = time.monotonic()-prediction_started
    need(batch_counts == [64]*292+[12], 'Exactly293 fixed64 batches and true12-query tail')
    need(state_digest(head.state_dict()) == checkpoint['final_digest']
         and all(p.grad is None and not p.requires_grad for p in head.parameters())
         and torch.equal(torch.get_rng_state(),external_cpu)
         and torch.equal(torch.cuda.get_rng_state(0).cpu(),external_cuda), 'Frozen quality state/gradNone/external RNG unchanged')
    peak = int(torch.cuda.max_memory_allocated(0)); head.cpu(); del head
    gc.collect(); torch.cuda.synchronize(); torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    cleanup = dict(allocated_after=int(torch.cuda.memory_allocated(0)),reserved_after=int(torch.cuda.memory_reserved(0)))
    need(cleanup['allocated_after'] == cleanup['reserved_after'] == 0 and peak <= 4096*1024**2,
         'Release every CUDA alias before official CPU COCO')
    changed,_,product = rescore(np,quality,original,qhat.numpy(),order)
    # GT is first accessed below, after all model prediction and CUDA cleanup.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    official = load(c,context['COCO_source'],'N26_ORIGINAL_OFFICIAL_COCO')
    coco = COCO(str(c.closed(context['DEV'])))
    need(list(coco.imgs) == ids and len(coco.anns) == 380 and set(coco.cats) == {1},
         'Original whole187/380 same-category GT used only by official evaluation')
    c.check(); coco_started = time.monotonic()
    baseline,baseline_T = official_metrics(np,c,official,COCOeval,coco,calibrated,ids)
    need(baseline == context['expected_COCO']['R'] and 100*baseline['stats'][0] == R_AP,
         'qhat1 must exactly reproduce every original R COCO12 field and AP')
    evaluated,evaluated_T = official_metrics(np,c,official,COCOeval,coco,changed,ids)
    coco_seconds = time.monotonic()-coco_started
    need(calibrated == original and all({k:v for k,v in a.items() if k != 'score'}
         == {k:v for k,v in b.items() if k != 'score'} for a,b in zip(original,changed)),
         'COCO observation must leave input masks/classes/query identities unchanged')
    output = Path(spec['output_directory']); output.mkdir()
    binding = dict(source=c.ref(Path(__file__)),spec=c.execution_spec_ref,
        GPU_admission=c.execution_admission_ref,phase=spec['phase'],prelock=spec['prelock'],
        merged_inputs=spec['merged_inputs'],inputs_qualified=spec['inputs_qualified'],
        checkpoint=spec['checkpoint'],training_qualified=spec['training_qualified'],
        quality_source=spec['quality_source'],prediction_context=spec['prediction_context'],
        original_R_predictions=context['predictions']['R'],COCO_source=context['COCO_source'],
        DEV=context['DEV'],seed=seed)
    predictions_ref = c.write_json(output/'QUALITY_PREDICTIONS.json',changed)
    audit_ref = c.write_json(output/'SCORE_AUDIT.json',dict(
        schema='N26-fixed-R-allquery-quality-score-audit-v1',**binding,
        score_mode='original_S0_times_learned_quality_FP64',records=[dict(
            ordinal=i,image_id=image_id,query_id=q,original_score_FP64=float(original_scores[i]),
            qhat_FP32_as_float=float(qhat[i]),product_score_FP64=float(product[i]))
            for i,(image_id,q) in enumerate(order)],private_not_for_publication=True))
    AP = 100*evaluated['stats'][0]
    per_T = [dict(IoU=a['IoU'],baseline_AP_percent=a['AP_percent'],
        quality_AP_percent=b['AP_percent'],Delta_AP_percent=b['AP_percent']-a['AP_percent'])
        for a,b in zip(baseline_T,evaluated_T)]
    report_ref = c.write_json(output/'EVAL_REPORT.json',dict(
        schema='N26-fixed-R-allquery-quality-score-evaluation-report-v1',
        status='N26_QUALITY_PRODUCT_FULLDEV_EVALUATION_COMPLETE_PENDING_ROOT_ACTUAL0',**binding,
        images=187,GT=380,queries=18700,predictions_ref=predictions_ref,score_audit_ref=audit_ref,
        original_COCO=baseline,quality_COCO=evaluated,per_IoU=per_T,
        original_AP_percent=R_AP,quality_AP_percent=AP,Delta_AP_percent=AP-R_AP,
        score_mode='S0*qhat',qhat_only_evaluated=False,epoch_selected_on_DEV=False,
        fixed_epoch=24,parameters=299521,tensors=6,batch_size=64,batches=293,last_batch=12,
        all100_original_RLE_class_query_order_exact=True,qhat1_original_predictions_exact=True,
        qhat1_original_COCO12_exact=True,top100_candidate_mask_geometry_unchanged=True,
        GT_used_in_quality_features_or_predictions=False,GT_only_in_official_evaluation=True,
        FIT_container_deserialized_but_unused=True,training_executed=False,
        head_batching_and_transfers_wall_seconds=prediction_seconds,official_two_COCO_wall_seconds=coco_seconds,
        elapsed_seconds=time.monotonic()-started,cleanup=cleanup,torch_peak_allocated_bytes=peak,
        qualified=False,ROOT_actual_GPU0_verified=False,current_owned_terminal_self_certified=False,
        seed_gate_decided=False))
    return dict(report_ref=report_ref,predictions_ref=predictions_ref,score_audit_ref=audit_ref)
