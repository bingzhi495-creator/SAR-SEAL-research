"""ROOT-owned frozen-native two-mode adapter learning; author source-only.

This script executes only under a manually admitted ROOT owner.  Original 610
states remain frozen/eval, four new eight-tensor adapters are independently
trained. All GT is post-forward supervision or evaluation; no AP/DEV/TEST.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import types

HERE = Path(__file__).resolve().parent
SPEC_SHA = '93a6abf80bc49070c903750ed5bc432f0babbd25f3027bb7ab20c70b18c7fbf4'
SPEC_BYTES = 21882
PHASE = 'native_ordinal_two_mode_mask_adapter_learning_20261008_01'
ARMS = ('union_only', 'random_balanced', 'spatial_balanced', 'ordinal')
EDGE_WEIGHT = .00032159743246726324
K = 12544
AUX_POINTS = 512
ATOMIC_KEYS = {'loss_cls', 'loss_mask', 'loss_dice'}
ATOMIC_KEYS.update('d%d.%s' % (i,k) for i in range(9) for k in ('loss_cls','loss_mask','loss_dice'))
OUTPUT_CAP = 64 * 1024**2

def require(ok,message):
    if not ok:raise ValueError(message)

def ref(path):
    p=Path(path).resolve(strict=True);digest=hashlib.sha256()
    with p.open('rb') as handle:
        for block in iter(lambda:handle.read(1024**2),b''):digest.update(block)
    return dict(path=str(p),sha256=digest.hexdigest(),bytes=p.stat().st_size)

def bound(row):
    require(Path(row['path']).is_absolute(),'Absolute input required')
    actual=ref(row['path'])
    require(actual['sha256']==row['sha256'] and actual['bytes']==row['bytes'],'Bound input changed')
    return Path(actual['path'])

def read(row):
    path=bound(row);value=json.loads(path.read_text(encoding='utf-8-sig'));bound(row);return value

def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(value,handle,sort_keys=True,indent=2,allow_nan=False);handle.write('\n')

def render_native(logits,resized_hw,original_hw):
    """Exact native logit interpolation, padding crop, then original-size resize."""
    import torch.nn.functional as F
    require(tuple(logits.shape)==(1,1,200,200),'One native200 logit field required')
    h,w=resized_hw;oh,ow=original_hw
    require(0<h<=800 and 0<w<=800 and oh>0 and ow>0,'Actual render dimensions required')
    padded=F.interpolate(logits,size=(800,800),mode='bilinear',align_corners=False)
    cropped=padded[:,:,:h,:w]
    return F.interpolate(cropped,size=(oh,ow),mode='bilinear',align_corners=False)[0,0]

def annotation_mapping(dataset,coco,image_id,info):
    """Use the actual native parser per annotation, never infer ID from match order."""
    raw_image=deepcopy(coco.load_imgs([image_id])[0]);raw_image['img_id']=image_id
    raw_ids=coco.get_ann_ids(img_ids=[image_id]);raw_annotations=coco.load_anns(raw_ids)
    rows=[];instances=[];parsed_ids=[]
    for ann in raw_annotations:
        parsed=dataset.parse_data_info(dict(raw_img_info=raw_image,raw_ann_info=[ann]))['instances']
        require(len(parsed) in (0,1),'Single-annotation parser changed multiplicity')
        rows.append(dict(annotation_id=int(ann['id']),native_parser_retained=bool(parsed)))
        if parsed:instances.extend(parsed);parsed_ids.append(int(ann['id']))
    require(instances==info['instances'],'Per-ann actual parser order differs from native dataset')
    require(len(set(raw_ids))==len(raw_ids) and len(set(parsed_ids))==len(parsed_ids),'Annotation IDs must be unique')
    return raw_annotations,parsed_ids,rows

def same_reference(left,right):
    return (Path(left['path'])==Path(right['path']) and left['sha256']==right['sha256']
            and left['bytes']==right['bytes'])

def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def rng_snapshot():
    import torch
    return (torch.get_rng_state().clone(),torch.cuda.get_rng_state(0).clone())

def rng_equal(left,right):
    import torch
    return all(torch.equal(a,b) for a,b in zip(left,right))

def digest_tensor(value):
    array = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())

@contextmanager
def observe_targets(head):
    """Observe the already installed target method, never recompute assignment."""
    original = head._get_targets_single
    had_instance = '_get_targets_single' in head.__dict__
    previous = head.__dict__.get('_get_targets_single')
    records = []
    def observed(current, *args, **kwargs):
        result = original(*args, **kwargs)
        require(len(result) == 7, 'Native seven-tuple targets required')
        sampling = result[6]
        pos = result[4].detach().cpu().tolist()
        assigned = sampling.pos_assigned_gt_inds.detach().cpu().tolist()
        require(len(pos) == len(assigned), 'Matched query/GT order differs')
        records.append(dict(positive_query_indices=pos, assigned_GT_indices=assigned,
                            positives=len(pos), target_labels=digest_tensor(result[0])))
        return result
    head._get_targets_single = types.MethodType(observed, head)
    try:
        yield records
    finally:
        if had_instance:
            head._get_targets_single = previous
        else:
            delattr(head, '_get_targets_single')

def capture_head_class(original):
    class PilotHead(original):
        def _get_targets_single(self, *args, **kwargs):
            result = super()._get_targets_single(*args, **kwargs)
            if getattr(self, "pilot_capture", None) is not None:
                self.pilot_capture["call_count"] += 1
                self.pilot_capture["final"] = result
            return result
    return PilotHead

def RNG_digest():
    return [tensor_digest(value) for value in rng_snapshot()]

def gradient_summary(model,names):
    import torch
    parameters=dict(model.named_parameters());rows={}
    for name in names:
        gradient=parameters[name].grad
        if gradient is None:rows[name]=dict(present=False,norm=0.,positive=False)
        else:
            require(bool(torch.isfinite(gradient).all()),'Nonfinite gradient: '+name)
            norm=float(gradient.detach().norm());rows[name]=dict(present=True,norm=norm,positive=norm>0)
    return rows

def clear_GPU():
    import torch
    before=torch.cuda.memory_allocated(0);collected=gc.collect();torch.cuda.synchronize(0)
    after_gc=torch.cuda.memory_allocated(0);torch._C._cuda_clearCublasWorkspaces()
    after_workspace=torch.cuda.memory_allocated(0);torch.cuda.empty_cache();after=torch.cuda.memory_allocated(0)
    require(after==0,'Model/graph/workspace cleanup retained '+str(after)+' CUDA bytes')
    return dict(before_GC=before,GC_objects_collected=collected,after_GC=after_gc,
                after_workspace_clear=after_workspace,after_empty_cache=after,allocated_exact0=True)

@contextmanager
def observe_native_costs(head):
    """Observe each real weighted cost and the exact CPU sum sent to SciPy once.

    The owner uses a single serial Python model call. All wrappers delegate without
    changing arguments/results and are removed even on a native matching failure.
    """
    import torch
    from mmdet.models.task_modules.assigners import hungarian_assigner
    original_costs=head.assigner.match_costs
    original_solver=hungarian_assigner.linear_sum_assignment
    rows=[];totals=[]
    class Cost:
        def __init__(self,delegate,index):self.delegate,self.index=delegate,index
        def __call__(self,*args,**kwargs):
            value=self.delegate(*args,**kwargs)
            rows.append(dict(cost_index=self.index,tensor=value.detach().cpu().clone()))
            return value
    def observed_solver(cost,*args,**kwargs):
        require(isinstance(cost,torch.Tensor) and cost.device.type=='cpu',
            'Installed native assigner must send its actual CPU tensor to SciPy')
        totals.append(cost.detach().clone())
        return original_solver(cost,*args,**kwargs)
    head.assigner.match_costs=[Cost(value,index) for index,value in enumerate(original_costs)]
    hungarian_assigner.linear_sum_assignment=observed_solver
    try:yield dict(components=rows,totals=totals)
    finally:
        head.assigner.match_costs=original_costs
        hungarian_assigner.linear_sum_assignment=original_solver


def load_source(row, name):
    path = bound(row)
    definition = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


@contextmanager
def capture_frozen_readout(head):
    holder = dict(pixel_decoder_calls=0, mask_embed_calls=0)
    def pixel_hook(module, args, output):
        holder['pixel_decoder_calls'] += 1
        holder['F'] = output[0].detach()
        return None
    def embedding_hook(module, args):
        holder['mask_embed_calls'] += 1
        holder['h'] = args[0].detach()
        return None
    handles = [head.pixel_decoder.register_forward_hook(pixel_hook),
               head.mask_embed.register_forward_pre_hook(embedding_hook)]
    try:
        yield holder
    finally:
        for handle in handles:
            handle.remove()
        holder['hooks_removed'] = True


def frozen_forward(model, data, samples, image_id):
    outside = rng_snapshot()
    with capture_frozen_readout(model.panoptic_head) as holder:
        with torch.random.fork_rng(devices=[0]), torch.no_grad():
            torch.manual_seed(100000 + image_id)
            torch.cuda.manual_seed_all(100000 + image_id)
            private_before = RNG_digest()
            features = model.extract_feat(data['inputs'])
            classes, masks = model.panoptic_head(features, samples)
            private_after = RNG_digest()
    require(holder['pixel_decoder_calls'] == 1 and holder['mask_embed_calls'] == 10 and
            holder['hooks_removed'] and len(classes) == len(masks) == 10,
            'One passive native forward with ten embeddings required')
    require(tuple(holder['F'].shape) == (1,256,200,200) and
            tuple(holder['h'].shape) == (1,100,256), 'Actual F/final postnorm h shape differs')
    require(all(tuple(x.shape) == (1,100,2) and x.dtype == torch.float32 and not x.requires_grad
                and bool(torch.isfinite(x).all()) for x in classes), 'Original class fields differ')
    require(all(tuple(x.shape) == (1,100,200,200) and x.dtype == torch.float32 and not x.requires_grad
                and bool(torch.isfinite(x).all()) for x in masks), 'Original mask fields differ')
    require(not holder['F'].requires_grad and not holder['h'].requires_grad and
            rng_equal(outside, rng_snapshot()), 'Frozen fields/RNG were not detached/restored')
    return dict(F=holder['F'], h=holder['h'], classes=classes, masks=masks), dict(
        image_id=image_id, head_calls=1, backbone_extract_calls=1, pixel_decoder_calls=1,
        mask_embedding_calls=10, hooks_removed=True, no_GT_forward=True,
        head_seed=100000+image_id, private_RNG_before=private_before, private_RNG_after=private_after,
        external_RNG_before=[tensor_digest(v) for v in outside], external_RNG_after=RNG_digest(),
        external_RNG_restored=True)


def load_image(model, dataset, coco, info):
    import cv2
    import numpy as np
    from mmengine.dataset import Compose, pseudo_collate
    from mmdet.datasets.transforms import PackDetInputs
    info = deepcopy(info)
    image_id = int(info['img_id'])
    image_ref = ref(info['img_path'])
    raw_annotations, parsed, parser_rows = annotation_mapping(dataset, coco, image_id, info)
    pipeline = Compose([dict(type='LoadImageFromFile',to_float32=True),
                        dict(type='LoadAnnotations',with_bbox=True,with_mask=True),
                        dict(type='Resize',scale=(800,800),keep_ratio=True)])
    transformed = pipeline(info)
    require(transformed is not None, 'Fixed image discarded')
    flags = transformed['gt_ignore_flags'].tolist()
    require(len(flags) == len(parsed), 'Loaded annotation order differs')
    valid_ids = [int(a) for a, ignored in zip(parsed, flags) if not ignored]
    raw_ids = [int(a['id']) for a in raw_annotations]
    require(valid_ids == raw_ids, 'All raw annotations including empty rasters must be retained')
    raw_gray = cv2.imread(str(image_ref['path']), cv2.IMREAD_GRAYSCALE)
    require(raw_gray is not None and raw_gray.dtype == np.uint8, 'Original uint8 gray decode failed')
    packed = PackDetInputs(meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))(transformed)
    with torch.no_grad():
        data = model.data_preprocessor(pseudo_collate([packed]), training=True)
    sample = data['data_samples'][0]
    hw = tuple(map(int, sample.img_shape[:2]))
    ori = tuple(map(int, sample.ori_shape[:2]))
    require(tuple(data['inputs'].shape) == (1,3,800,800) and
            tuple(sample.batch_input_shape[:2]) == (800,800) and raw_gray.shape == ori,
            'Actual input/gray canvas differs')
    gray800 = np.zeros((800,800), dtype=np.uint8)
    gray800[:hw[0],:hw[1]] = cv2.resize(raw_gray, (hw[1],hw[0]), interpolation=cv2.INTER_NEAREST)
    samples = deepcopy(data['data_samples'])
    for current in samples:
        for key in ('gt_instances','ignored_instances','gt_sem_seg'):
            if key in current:
                delattr(current, key)
    require(all(not any(key in current for key in ('gt_instances','ignored_instances','gt_sem_seg'))
                for current in samples), 'GT was forwarded to native head')
    identity = dict(image_id=image_id,image=image_ref,ann_ids=raw_ids,parser_rows=parser_rows,
                    original_shape=list(ori),actual_img_shape=list(hw),batch_input_shape=[800,800],
                    scale_factor=[float(v) for v in sample.scale_factor],
                    native_input_gamma=1., gray_decode='cv2.IMREAD_GRAYSCALE',
                    target_gray_resize='nearest', gray800_sha256=hashlib.sha256(gray800.tobytes()).hexdigest())
    return data, samples, gray800, raw_gray, identity


def posterior_gt(head, data, identity):
    with torch.no_grad():
        gt = head.preprocess_gt([data['data_samples'][0].gt_instances],[None])[0]
    h,w = identity['actual_img_shape']
    require(len(gt) == len(identity['ann_ids']) and tuple(gt.masks.shape) == (len(gt),800,800)
            and bool(((gt.masks == 0)|(gt.masks == 1)).all()) and bool((gt.labels == 0).all())
            and not bool(gt.masks[:,h:,:].any()) and not bool(gt.masks[:,:,w:].any()),
            'Native GT after forward differs from raw complete population')
    identity['native_GT'] = digest_tensor(gt.masks)
    return gt


def prepare_auxiliary(core, gt, gray800, ann_ids, micro):
    import numpy as np
    binary = gt.masks.detach().cpu().numpy().astype(np.uint8)
    splits = {arm: core.split_foreground_targets(binary,gray800,ann_ids,mode=arm)
              for arm in ARMS[1:]}
    coords = np.zeros((len(ann_ids),AUX_POINTS,2),dtype=np.float32)
    labels = {arm: np.zeros((len(ann_ids),2,AUX_POINTS),dtype=np.float32) for arm in ARMS[1:]}
    rows = []
    for gi,ann_id in enumerate(ann_ids):
        pixels = np.flatnonzero(binary[gi].reshape(-1))
        origin = 202610080000 + int(ann_id) + 65536 * micro
        seed = origin % (2**32)
        if pixels.size:
            chosen = pixels[np.random.RandomState(seed).randint(0,len(pixels),size=AUX_POINTS)]
            coords[gi,:,0] = (chosen % 800 + .5)/800
            coords[gi,:,1] = (chosen // 800 + .5)/800
            for arm in ARMS[1:]:
                labels[arm][gi,0] = splits[arm]['bright_masks'][gi].reshape(-1)[chosen]
                labels[arm][gi,1] = splits[arm]['dark_masks'][gi].reshape(-1)[chosen]
                require(np.array_equal(labels[arm][gi].sum(axis=0),np.ones(AUX_POINTS)),
                        'Direct sampled positive-cell labels fail exact foreground partition')
        rows.append(dict(GT_index=gi,annotation_id=int(ann_id),GT_positive_pixels=int(pixels.size),
                         empty_GT=(not bool(pixels.size)),seed_origin=origin,seed_uint32=seed,
                         coordinates_sha256=hashlib.sha256(coords[gi].tobytes()).hexdigest()))
    coordinate_tensor = torch.from_numpy(coords).to(gt.masks.device)
    label_tensors = {arm:torch.from_numpy(value).to(gt.masks.device) for arm,value in labels.items()}
    diagnostic = dict(points_per_GT=AUX_POINTS,raw_GT_order=list(ann_ids),sampling_rows=rows,
        coordinates_sha256=hashlib.sha256(coords.tobytes()).hexdigest(),
        target_sha256={arm:hashlib.sha256(value.tobytes()).hexdigest() for arm,value in labels.items()},
        split_rows={arm:splits[arm]['diagnostics'] for arm in ARMS[1:]},
        private_NumPy_global_rng_untouched=True,uniform_positive_cell_centers_with_replacement=True,
        native_PointRend_coords_not_replaced_or_reused=True)
    return coordinate_tensor,label_tensors,diagnostic


def conditional_loss(core, result, positive, assigned, coords, labels, arm):
    from mmcv.ops import point_sample
    sub = result['subresiduals'][0].index_select(0,positive)
    selected_coords = coords.index_select(0,assigned)
    count = int(positive.numel())
    if count:
        expanded_coords = selected_coords[:,None].expand(count,2,AUX_POINTS,2).reshape(count*2,AUX_POINTS,2)
        sampled = point_sample(sub.reshape(count*2,1,200,200),expanded_coords,
                               align_corners=False).reshape(count,2,AUX_POINTS)
    else:
        sampled = sub.reshape(0,2,40000)[:,:,:AUX_POINTS]
    if arm == 'union_only':
        target = labels['ordinal'].index_select(0,assigned)
    else:
        target = labels[arm].index_select(0,assigned)
    value, diagnostic = core.foreground_conditional_ce(sampled,target[:,0],target[:,1])
    weight = 0. if arm == 'union_only' else 1.
    serial = {key:float(v.detach()) if isinstance(v,torch.Tensor) else v for key,v in diagnostic.items()}
    serial.update(weight=weight,GT_reorder=assigned.detach().cpu().tolist(),
                  query_reorder=positive.detach().cpu().tolist(),point_coords_shared_in_raw_GT_order=True)
    return weight*value,value,serial


def adapter_objective(model, core, edge, adapter, cached, data, gt, identity, coords, labels, arm, micro):
    head = model.panoptic_head
    result = adapter(cached['h'],cached['F'],cached['masks'][-1])
    masks = list(cached['masks'][:-1]) + [result['union_logits']]
    require(all(a is b for a,b in zip(masks[:-1],cached['masks'][:-1])), 'First nine fields changed')
    outside = rng_snapshot()
    head.pilot_capture = dict(call_count=0,final=None)
    try:
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(1000000+micro);torch.cuda.manual_seed_all(1000000+micro)
            private_before = RNG_digest()
            with observe_targets(head) as observations:
                losses = head.loss_by_feat(cached['classes'],masks,[gt],[data['data_samples'][0].metainfo])
            private_after = RNG_digest()
        captured = head.pilot_capture['final']
        calls = head.pilot_capture['call_count']
    finally:
        head.pilot_capture = None
    require(rng_equal(outside,rng_snapshot()) and calls == len(observations) == 10 and
            set(losses) == ATOMIC_KEYS and isinstance(captured,tuple) and len(captured) == 7,
            'Native30/defaultfive/ten actual matching capture differs')
    positive,assigned = captured[4],captured[6].pos_assigned_gt_inds
    require(torch.equal(torch.nonzero(captured[3]>0,as_tuple=False).flatten(),positive) and
            len(positive) == len(assigned) == len(captured[2]) and
            torch.equal(captured[2],gt.masks[assigned]), 'Final native positive/GT correspondence differs')
    sobel = edge.edge_agreement_loss(masks[-1],positive,gt.masks[assigned])
    conditional,unweighted,cond_record = conditional_loss(core,result,positive,assigned,coords,labels,arm)
    combined = dict(losses)
    combined['loss_sobel_edge_agreement'] = EDGE_WEIGHT * sobel
    combined['loss_foreground_conditional'] = conditional
    total,_ = model.parse_losses(combined)
    require(len(combined) == 32 and total.requires_grad and all(bool(torch.isfinite(v)) for v in combined.values()),
            'Finite native31 plus conditional32 graph required')
    initial = None
    if micro == 0:
        same = torch.equal(masks[-1],cached['masks'][-1])
        close = torch.allclose(masks[-1],cached['masks'][-1],rtol=1.3e-6,atol=1e-5)
        require(same or close,'Initial union changed native200 beyond locked FP32 tolerance')
        initial = dict(union_native200_TorchEqual=same,union_native200_numerically_close=bool(close),
                       union_native200_max_abs=float((masks[-1]-cached['masks'][-1]).abs().max()),
                       classes_all10_same_objects=True,first9_masks_same_objects=True,
                       native31_values={k:float(v.detach()) for k,v in combined.items() if k != 'loss_foreground_conditional'},
                       native_matching=deepcopy(observations))
    return total,dict(arm=arm,zero_based_micro=micro,microstep=micro+1,image_id=identity['image_id'],
        raw_GT_count=len(gt),GT_annotation_ids=identity['ann_ids'],native_loss_atoms=30,Sobel_loss_atom=1,
        total_loss_atoms=32,native30_values={k:float(v.detach()) for k,v in losses.items()},
        Sobel_unweighted=float(sobel.detach()),Sobel_weight=EDGE_WEIGHT,
        Sobel_weighted=float((EDGE_WEIGHT*sobel).detach()),conditional_unweighted=float(unweighted.detach()),
        conditional_weight=(0. if arm=='union_only' else 1.),conditional_weighted=float(conditional.detach()),
        total_loss=float(total.detach()),conditional=cond_record,native_matching=observations,
        native_matching_calls=calls,final_positive_query_indices=positive.detach().cpu().tolist(),
        final_assigned_GT_indices=assigned.detach().cpu().tolist(),GT_only_after_native_head=True,
        first9_masks_same_objects=True,all10_classes_same_objects=True,
        native_loss_private_seed=1000000+micro,private_RNG_before=private_before,private_RNG_after=private_after,
        external_RNG_before=[tensor_digest(v) for v in outside],external_RNG_after=RNG_digest(),external_RNG_restored=True,
        native_matching_and_PointRend_implementation_unchanged=True,native_uncertain_points=K),initial


def make_optimizer(adapter):
    names = dict(adapter.named_parameters())
    require(len(names) == 8 and sum(p.numel() for p in names.values()) == 66176, 'New8 adapter shapes/scalars differ')
    weights = [p for name,p in names.items() if name.endswith('.weight')]
    biases = [p for name,p in names.items() if name.endswith('.bias')]
    require(len(weights) == len(biases) == 4, 'Eight Linear weight/bias tensors required')
    optimizer = torch.optim.AdamW([dict(params=weights,weight_decay=.05),dict(params=biases,weight_decay=0.)],lr=.0001)
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    require(len(ids) == len(set(ids)) == 8 and set(ids) == {id(p) for p in names.values()} and not optimizer.state,
            'Only new8 may enter the cold optimizer')
    rows = []
    by_id = {id(p):name for name,p in names.items()}
    for index,group in enumerate(optimizer.param_groups):
        rows.append(dict(index=index,names=[by_id[id(p)] for p in group['params']],
                         lr=group['lr'],weight_decay=group['weight_decay']))
    optimizer.zero_grad(set_to_none=True)
    return optimizer,rows


def encode_selected(logit, resized_hw, original_hw):
    import numpy as np
    from pycocotools import mask as mask_utils
    binary = (render_native(logit,resized_hw,original_hw)>0).detach().cpu().numpy().astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(binary))
    if isinstance(rle['counts'],bytes):rle['counts']=rle['counts'].decode('ascii')
    return dict(size=[int(v) for v in rle['size']],counts=rle['counts'])


def evaluate_image(model, core, cached, gt, data, identity, adapters):
    head = model.panoptic_head
    outside = rng_snapshot()
    head.pilot_capture = dict(call_count=0,final=None)
    try:
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            seed = 100000 + identity['image_id']
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            before = RNG_digest()
            with observe_native_costs(head) as costs:
                targets = head.get_targets([cached['classes'][-1][0]],[cached['masks'][-1][0]],
                    [gt],[data['data_samples'][0].metainfo])
            after = RNG_digest()
        captured,calls = head.pilot_capture['final'],head.pilot_capture['call_count']
    finally:
        head.pilot_capture = None
    require(len(targets) == 5 and calls == 1 and rng_equal(outside,rng_snapshot()),
            'One actual frozen base target call required')
    positive,assigned = captured[4],captured[6].pos_assigned_gt_inds
    require(len(positive) == len(assigned) == len(gt) and torch.equal(captured[2],gt.masks[assigned]),
            'Every heldout raw GT requires actual complete base query anchor')
    mapping = [None]*len(gt)
    for q,j in zip(positive.detach().cpu().tolist(),assigned.detach().cpu().tolist()):mapping[j]=int(q)
    require(all(q is not None for q in mapping) and len(set(mapping)) == len(mapping), 'Base query map incomplete')
    if len(gt):
        require([row['cost_index'] for row in costs['components']] == [0,1,2] and len(costs['totals']) == 1,
                'One actual frozen base Hungarian and three native costs required')
        components = [row['tensor'] for row in costs['components']]
        total = costs['totals'][0]
    else:
        require(not costs['components'] and not costs['totals'], 'Native zero-GT early return differs')
        # Empty matrices encode the vacuous population; no cost or SciPy call occurred.
        components = [torch.empty((100,0),dtype=torch.float32) for _ in range(3)]
        total = torch.empty((100,0),dtype=torch.float32)
    require(all(v.dtype == torch.float32 and tuple(v.shape) == (100,len(gt)) and bool(torch.isfinite(v).all())
                for v in components+[total]), 'Actual installed weightedFP32 costs differ')
    keys = ('classification_weight2','BCE_weight5','Dice_weight5','total')
    matrices = dict(zip(keys,components+[total]))
    cost_row = dict(image_id=identity['image_id'],ann_ids=identity['ann_ids'],query_by_GT=mapping,
        components={k:v.tolist() for k,v in matrices.items()},component_digests={k:digest_tensor(v) for k,v in matrices.items()},
        components_already_weighted=True,weights=[2.,5.,5.],FP32_native_arithmetic=True,
        total_captured_actual_CPU_SciPy_argument=bool(len(gt)),total_recomputed=False,
        zero_GT_native_early_return=(len(gt)==0),empty_matrices_are_vacuous_not_observed_costs=(len(gt)==0),
        points=K,naive_Dice_eps=1.,native_cost_calls=len(costs['components']),native_Hungarian_calls=len(costs['totals']),
        native_get_targets_calls=1,native_get_targets_default_return_sampling_results=False,
        positive_query_indices=positive.detach().cpu().tolist(),assigned_GT_indices=assigned.detach().cpu().tolist(),
        matching_seed=seed,private_RNG_before=before,private_RNG_after=after,
        external_RNG_before=[tensor_digest(v) for v in outside],external_RNG_after=RNG_digest(),external_RNG_restored=True,
        final_cls_logits=cached['classes'][-1][0].detach().cpu().tolist())
    predictions = {}
    hw,ori = identity['actual_img_shape'],identity['original_shape']
    with torch.no_grad():
        for arm in ('native',)+ARMS:
            logits = cached['masks'][-1] if arm=='native' else adapters[arm](cached['h'],cached['F'],cached['masks'][-1])['union_logits']
            rows = []
            for gi,(ann_id,q) in enumerate(zip(identity['ann_ids'],mapping)):
                rows.append(dict(annotation_id=ann_id,GT_index=gi,query_index=q,
                    RLE=encode_selected(logits[:,q:q+1],hw,ori)))
            predictions[arm] = rows
    return dict(image_id=identity['image_id'],ann_ids=identity['ann_ids'],query_by_GT=mapping,
        original_shape=ori,actual_img_shape=hw,predictions=predictions,
        reference_queries_are_actual_native_base_matching=True,all_raw_GT_retained=True,
        no_best100_or_score_filter=True,native_render_before_threshold=True,
        new_adapter_eval_updates=0,no_GT_in_adapter_or_native_forward=True),cost_row


def run(spec_path, out):
    started = time.perf_counter()
    spec_ref = ref(spec_path)
    require(Path(spec_path).is_absolute() and spec_ref['sha256']==SPEC_SHA and spec_ref['bytes']==SPEC_BYTES,'Exact locked SPEC01 required')
    spec = read(spec_ref)
    require(spec['schema']=='ROOT-ordinal-two-mode-mask-adapter-learning-spec-v1' and spec['phase']==PHASE and
            Path(spec['source_directory'])==HERE and spec['runtime_entry']=='train.py' and tuple(spec['arms'])==ARMS,
            'Locked source/phase/arms differ')
    out = Path(out)
    require(out.is_absolute() and not out.exists() and out==Path(spec['output_directory']),'Exact fresh output directory required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','ROOT owner must expose CUDA0 only')
    out.mkdir(parents=True)
    binding_ref = ref(HERE/'SOURCE_BINDINGS.json')
    binding = read(binding_ref)
    require(binding['schema']=='native-ordinal-two-mode-mask-adapter-learning-source-v1' and
            same_reference(binding['scientific_spec'],spec_ref),'Source/spec binding differs')
    for row in binding['files']:bound(row)
    require(any(same_reference(row,ref(__file__)) for row in binding['files']),'Entry is not source bound')
    for key in ('adapter_core','CPU_numeric_qualification','checkpoint','original_Sobel_source','old_training_source_bindings'):
        require(same_reference(binding[key],spec[key]),'Declared lineage differs: '+key);bound(spec[key])
    require(binding['inputs']==spec['inputs'],'Input refs differ')
    for row in spec['inputs'].values():bound(row)
    import numpy as np
    import torch as torch_runtime
    import psutil
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmengine.runner import CheckpointLoader
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    from mmdet.models.dense_heads.mask2former_head import Mask2FormerHead
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    globals()['torch'] = torch_runtime
    require(psutil.Process().cpu_affinity()==[0,1],'ROOT CPU0/1 required')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.init();capacity=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,4*1024**3/capacity),0)
    torch.cuda.reset_peak_memory_stats(0)
    register_all_modules(init_default_scope=False);init_default_scope('mmdet')
    core = load_source(spec['adapter_core'],'root_ordinal_adapter_core')
    edge = load_source(spec['original_Sobel_source'],'root_ordinal_original_edge')
    require(tuple(core.ARMS)==ARMS and core.ADDED_PARAMETER_TENSORS==8 and core.ADDED_PARAMETER_SCALARS==66176,
            'Actual core contract differs')
    cfg = Config.fromfile(str(bound(spec['inputs']['config'])));cfg.load_from=None
    CaptureHead = capture_head_class(Mask2FormerHead)
    cfg.model.panoptic_head.type = CaptureHead
    checkpoint = CheckpointLoader.load_checkpoint(str(bound(spec['checkpoint'])),map_location='cpu')
    state,meta = checkpoint['state_dict'],checkpoint['meta']
    require(len(state)==610 and meta['schema']=='native-final-mask-sobel-edge-agreement-seed-repeat-checkpoint-v1'
            and meta['arm']=='sobel_edge_agreement' and meta['seed']==meta['training_image_order_seed']==1
            and meta['microsteps']==2964 and meta['optimizer_updates']==1482 and meta['trainable_tensors']==181
            and meta['frozen_state_tensors']==429 and meta['edge_weight']==EDGE_WEIGHT
            and meta['source_bindings_sha256']==spec['old_training_source_bindings']['sha256'], 'Strong fixed610 initializer differs')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = revert_sync_batchnorm(MODELS.build(cfg.model));model.load_state_dict(state,strict=True)
    require(len(model.state_dict())==610 and len(tuple(model.parameters()))==451 and
            all(torch.equal(value,state[name]) for name,value in model.state_dict().items()), 'Strict full610 CPU readback failed')
    native_keys = set(state)
    native_parameters = list(model.parameters())
    before = {name:digest_tensor(value) for name,value in model.state_dict().items()}
    model.requires_grad_(False);model.eval()
    outside_constructor = rng_snapshot()
    adapters = torch.nn.ModuleDict({arm:core.TwoModeMaskAdapter() for arm in ARMS})
    require(rng_equal(outside_constructor,rng_snapshot()),'Adapter constructors changed CPU/CUDA RNG')
    cold_state = {arm:{name:p.detach().cpu().clone() for name,p in adapters[arm].state_dict().items()} for arm in ARMS}
    require(all(set(v)==set(cold_state[ARMS[0]]) and all(torch.equal(t,cold_state[ARMS[0]][name]) for name,t in v.items())
                for v in cold_state.values()),'Four cold adapter states differ')
    require(not torch.equal(adapters[ARMS[0]].bright_embed[0].weight,adapters[ARMS[0]].dark_embed[0].weight),
            'Distinct hidden seeds must avoid artificial identical-parameter symmetry')
    model.panoptic_head.ordinal_two_mode_adapters = adapters
    require(len(model.state_dict())==642 and len(tuple(model.parameters()))==483 and
            sum(p.requires_grad for p in model.parameters())==32 and all(not p.requires_grad for p in native_parameters),
            'Shared workspace642/483/new32 differs')
    model.to('cuda:0');head=model.panoptic_head
    require(not model.training and not head.training and not head.pixel_decoder.training and
            head.num_queries==100 and head.num_points==K and head.oversample_ratio==3 and head.importance_sample_ratio==.75
            and [c.weight for c in head.assigner.match_costs]==[2.,5.,5.] and head.assigner.match_costs[2].eps==1.,
            'Frozen native modes/PointRend/costs differ')
    optimizers,group_rows = {},{}
    for arm in ARMS:optimizers[arm],group_rows[arm]=make_optimizer(adapters[arm])
    require(all(not p.requires_grad and p.grad is None for p in native_parameters),'Original parameters trainable or have gradients')
    dataset = CocoDataset(ann_file=str(bound(spec['inputs']['fit_annotation'])),
        data_prefix=dict(img=spec['image_root']),metainfo=dict(classes=('ship',)),pipeline=[],test_mode=True,
        filter_cfg=None,serialize_data=True)
    coco = COCO(str(bound(spec['inputs']['fit_annotation'])))
    require(len(dataset)==len(coco.imgs)==741 and len(coco.anns)==1661,'Complete FIT741/1661 differs')
    infos = [dataset.get_data_info(i) for i in range(len(dataset))]
    permutation = np.random.RandomState(20261008).permutation(list(range(741))).tolist()
    train_indices,heldout_indices = permutation[:480],permutation[480:]
    require(len(set(train_indices))==480 and len(set(heldout_indices))==261 and not(set(train_indices)&set(heldout_indices)),
            'Adapter train/heldout indices overlap')
    image_identities,eval_rows,cost_rows = [],[],[]
    consumed_GT = 0
    activity = {arm:{name:0 for name in adapters[arm].state_dict()} for arm in ARMS}
    initial_qualification = {}
    updates = {arm:0 for arm in ARMS}
    write(out/'COLD_ADAPTER_STATE_DIGESTS.json',{arm:{name:digest_tensor(value) for name,value in st.items()} for arm,st in cold_state.items()})
    write(out/'START.json',dict(schema='native-ordinal-two-mode-mask-adapter-learning-start-v1',phase=PHASE,
        spec=spec_ref,source=ref(__file__),source_bindings=binding_ref,checkpoint=spec['checkpoint'],
        adapter_core=spec['adapter_core'],CPU_numeric_qualification=spec['CPU_numeric_qualification'],
        arms=list(ARMS),permutation_dataset_indices=permutation,training_dataset_indices=train_indices,
        heldout_dataset_indices=heldout_indices,training_image_ids=[int(infos[i]['img_id']) for i in train_indices],
        heldout_image_ids=[int(infos[i]['img_id']) for i in heldout_indices],microsteps_per_arm=480,
        optimizer_updates_per_arm=240,original610_frozen=True,workspace_state642=True,
        gamma=1.,native_loss_atoms=30,total_loss_atoms=32,baseline_already_trained_on_all_FIT=True,
        heldout_only_for_new_adapters_not_blind_TEST=True))
    with (out/'TRAIN_TRACE.jsonl').open('x',encoding='utf-8',newline='\n') as trace:
        for micro,index in enumerate(train_indices):
            require(time.perf_counter()-started < 1800,'Locked owner deadline exceeded')
            data,samples,gray800,raw_gray,identity = load_image(model,dataset,coco,infos[index])
            cached,forward_record = frozen_forward(model,data,samples,identity['image_id'])
            gt = posterior_gt(head,data,identity)
            coords,labels,aux_sampling = prepare_auxiliary(core,gt,gray800,identity['ann_ids'],micro)
            identity.update(dataset_index=index,partition='adapter_training',forward=forward_record)
            image_identities.append(identity);consumed_GT += len(gt)
            for arm in ARMS:
                total,row,initial = adapter_objective(model,core,edge,adapters[arm],cached,data,gt,identity,coords,labels,arm,micro)
                (total/2).backward()
                gradients = gradient_summary(adapters[arm],list(adapters[arm].state_dict()))
                for name,record in gradients.items():
                    if record['positive']:activity[arm][name]+=1
                row.update(shared_frozen_forward=forward_record,auxiliary_sampling=aux_sampling,
                           new8_preclip_accumulated_gradient_rows=gradients,accumulation_microsteps=2,
                           optimizer_step=False,old610_gradient_None=all(p.grad is None for p in native_parameters))
                require(row['old610_gradient_None'],'Frozen native610 received gradients')
                if initial is not None:
                    initial['first_micro_new8_gradients'] = gradients
                    initial_qualification[arm] = initial
                    if arm != ARMS[0]:
                        baseline=initial_qualification[ARMS[0]]
                        require(initial['native_matching']==baseline['native_matching'] and
                            all(math.isclose(v,baseline['native31_values'][k],rel_tol=1.3e-6,abs_tol=1e-5)
                                for k,v in initial['native31_values'].items()), 'Initial native31/matching differ between arms')
                if micro%2==1:
                    norm=torch.nn.utils.clip_grad_norm_(adapters[arm].parameters(),.01,norm_type=2)
                    require(bool(torch.isfinite(norm)),'Nonfinite new8 global preclip norm')
                    row['new8_global_preclip_norm']=float(norm)
                    row['new8_postclip_gradient_rows']=gradient_summary(adapters[arm],list(adapters[arm].state_dict()))
                    optimizers[arm].step();optimizers[arm].zero_grad(set_to_none=True)
                    updates[arm]+=1;row['optimizer_step']=True
                row['optimizer_updates']=updates[arm]
                trace.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                del total,row,gradients,initial
                if micro%2==1:del norm
            del data,samples,cached,gt,coords,labels,gray800,raw_gray
            if micro%40==39:print(json.dumps(dict(microsteps_per_arm=micro+1,updates_per_arm=updates,elapsed=time.perf_counter()-started)),flush=True)
    require(all(v==240 for v in updates.values()),'Actual updates differ from240')
    arm_reports = {}
    for arm in ARMS:
        directory=out/arm;directory.mkdir()
        final_state={name:value.detach().cpu().clone() for name,value in adapters[arm].state_dict().items()}
        require(len(final_state)==8 and all(bool(torch.isfinite(v).all()) for v in final_state.values()),'Final8 state nonfinite')
        changed={name:not torch.equal(v,cold_state[arm][name]) for name,v in final_state.items()}
        require(all(changed.values()) and all(activity[arm][name]>0 for name in changed),'All8 genuine learned activity/change required')
        final_digest={name:digest_tensor(v) for name,v in final_state.items()}
        write(directory/'ADAPTER_FINAL_STATE_DIGESTS.json',final_digest)
        cp_meta=dict(schema='native-ordinal-two-mode-mask-adapter-checkpoint-v1',arm=arm,phase=PHASE,
            dataset='pseg_ssdd',adapter_state_tensors=8,adapter_parameter_scalars=66176,
            original_native_state_tensors=610,logical_deployment_state_tensors=618,logical_parameter_tensors=459,
            native_trainable_tensors=0,new_trainable_tensors=8,original610_frozen=True,
            original_checkpoint=spec['checkpoint'],adapter_core_source=spec['adapter_core'],
            source=ref(__file__),source_bindings=binding_ref,source_bindings_sha256=binding_ref['sha256'],
            microsteps=480,optimizer_updates=240,training_unique_images=480,adapter_heldout_images=261,
            consumed_GT_annotations=consumed_GT,training_order_seed=20261008,training_passes=1,
            gamma=1.,Sobel_weight=EDGE_WEIGHT,conditional_weight=(0. if arm=='union_only' else 1.),
            native_loss_atoms=30,total_loss_atoms=32,learning_rate=.0001,accumulation=2,
            clip_grad_norm=.01,weight_decay_weights=.05,weight_decay_bias=0.,
            inference_GT_free=True,only_adapter_state_payload=True,not_blind_TEST=True)
        cp_path=directory/'adapter_final.pth'
        with cp_path.open('xb') as stream:torch.save(dict(state_dict=final_state,meta=cp_meta),stream)
        readback=torch.load(cp_path,map_location='cpu')
        require(readback['meta']==cp_meta and set(readback['state_dict'])==set(final_state) and
                all(torch.equal(value,readback['state_dict'][name]) for name,value in final_state.items()),'Complete8 CPU checkpoint readback failed')
        arm_reports[arm]=dict(checkpoint=ref(cp_path),checkpoint_meta=cp_meta,
            final_state_digests=ref(directory/'ADAPTER_FINAL_STATE_DIGESTS.json'),all8_CPUreadback_exact=True,
            optimizer_groups=group_rows[arm],microsteps=480,optimizer_updates=240,
            consumed_GT_annotations=consumed_GT,actual_changed=changed,genuine_positive_gradient_microsteps=activity[arm],
            cold_state_same_as_all_controls=True,first_micro=initial_qualification[arm])
        del final_state,readback
    for index in heldout_indices:
        require(time.perf_counter()-started<1800,'Locked owner deadline exceeded')
        data,samples,gray800,raw_gray,identity=load_image(model,dataset,coco,infos[index])
        cached,forward_record=frozen_forward(model,data,samples,identity['image_id'])
        gt=posterior_gt(head,data,identity)
        evaluation,cost=evaluate_image(model,core,cached,gt,data,identity,adapters)
        identity.update(dataset_index=index,partition='adapter_heldout',forward=forward_record)
        image_identities.append(identity);eval_rows.append(evaluation);cost_rows.append(cost)
        del data,samples,cached,gt,gray800,raw_gray,evaluation,cost
    require(len(image_identities)==741 and len(eval_rows)==len(cost_rows)==261 and
            sum(len(row['ann_ids']) for row in image_identities)==1661,'Complete FIT rawGT population differs')
    after={name:digest_tensor(model.state_dict()[name]) for name in native_keys}
    exact=all(torch.equal(model.state_dict()[name].detach().cpu(),state[name]) for name in native_keys)
    require(before==after and exact and all(not p.requires_grad and p.grad is None for p in native_parameters),
            'Original610 state/gradients changed during adapter learning')
    state_proof=dict(schema='native-ordinal-two-mode-mask-adapter-state-proof-v1',checkpoint=spec['checkpoint'],
        before=before,after=after,state_entries610=610,all610_CPU_TorchEqual=exact,
        original_parameter_tensors451=451,all_original_parameters_frozen=True,all_original_parameter_grad_None=True,
        strict_initial_CPU_readback=True,workspace_state_entries=642,workspace_parameter_tensors=483,
        workspace_trainable_tensors=32,logical_deployment_state_entries=618,logical_trainable_tensors=8,
        constructor_private_CPU_CUDA_RNG_restored=True,optimizer_original_parameter_membership=0)
    peak=torch.cuda.max_memory_allocated(0)
    # Clear all CUDA owners including optimizer moments and retained native parameters.
    del model,head,adapters,optimizers,native_parameters,cold_state,state,checkpoint,cfg
    cleanup=clear_GPU()
    write(out/'IMAGE_IDENTITIES.json',image_identities)
    write(out/'EVALUATION_ROWS.json',eval_rows)
    write(out/'NATIVE_COST_ROWS.json',cost_rows)
    write(out/'STATE_PROOF.json',state_proof)
    outputs={key:ref(out/name) for key,name in dict(trace='TRAIN_TRACE.jsonl',image_identities='IMAGE_IDENTITIES.json',
        evaluation_rows='EVALUATION_ROWS.json',native_cost_rows='NATIVE_COST_ROWS.json',state_proof='STATE_PROOF.json',
        cold_adapter_state_digests='COLD_ADAPTER_STATE_DIGESTS.json',start='START.json').items()}
    for row in image_identities:bound(row['image'])
    for row in binding['files']:bound(row)
    bound(spec_ref);bound(spec['checkpoint'])
    for row in spec['inputs'].values():bound(row)
    report=dict(schema='native-ordinal-two-mode-mask-adapter-learning-report-v1',
        status='COMPLETE_ORDINAL_TWO_MODE_ADAPTER_LEARNING_PENDING_ROOT_TERMINAL',phase=PHASE,spec=spec_ref,
        source=ref(__file__),source_bindings=binding_ref,source_files=binding['files'],inputs=spec['inputs'],
        checkpoint=spec['checkpoint'],checkpoint_meta=meta,adapter_core=spec['adapter_core'],
        CPU_numeric_qualification=spec['CPU_numeric_qualification'],arms=list(ARMS),arm_reports=arm_reports,
        outputs=outputs,order_seed=20261008,permutation_dataset_indices=permutation,
        training_dataset_indices=train_indices,heldout_dataset_indices=heldout_indices,
        training_image_ids=[int(infos[i]['img_id']) for i in train_indices],heldout_image_ids=[int(infos[i]['img_id']) for i in heldout_indices],
        training_microsteps_per_arm=480,optimizer_updates_per_arm=240,total_microsteps_all4=1920,total_optimizer_updates_all4=960,
        training_raw_GT=consumed_GT,heldout_raw_GT=sum(len(row['ann_ids']) for row in eval_rows),all_FIT_raw_GT=1661,
        native_head_forwards=741,native_backbone_extracts=741,native_embeddings_per_forward=10,
        native_training_target_calls_all4=19200,heldout_base_target_calls=261,
        original610_frozen_CPU_equal=True,original451_allgradNone=True,shared_workspace_state642=True,
        constructor_private_CPU_RNG_restored=True,all_input_images_SHA_before_after=True,
        native30_original_PointRend_and_matching_executed=True,total_loss_atoms=32,native_loss_atoms=30,
        Sobel_weight=EDGE_WEIGHT,conditional_weights={arm:0. if arm=='union_only' else 1. for arm in ARMS},
        new_aux_points_per_GT=512,new_aux_GT_positive_cell_centers=True,
        same_aux_coordinates_all_arms_before_assignment_reorder=True,all_native_loss_private_forks_restored=True,
        first9_native_masks_and_all_classes_unchanged=True,all_passive_hooks_removed=True,
        gamma=1.,no_GT_forward=True,no_AP_DEV_TEST=True,no_best100_or_score_filter=True,
        baseline_already_trained_on_all_FIT=True,heldout_only_for_adapters_not_independent_TEST=True,
        no_threeclass_rendered_native_loss_equivalence_claim=True,producer_science_decision_computed=False,
        cleanup=cleanup,torch_peak_allocated_bytes=peak,wall_seconds=time.perf_counter()-started,
        scientific_export_payload_limit_bytes=OUTPUT_CAP)
    write(out/'REPORT.json',report)
    require(sum(path.stat().st_size for path in out.rglob('*') if path.is_file())<=OUTPUT_CAP,'Locked64MiB export cap exceeded')
    print(json.dumps(dict(status=report['status'],report=ref(out/'REPORT.json')),sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();run(args.spec,args.out)


if __name__=='__main__':
    main()
