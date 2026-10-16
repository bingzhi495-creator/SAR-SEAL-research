"""UNBOUND CPU construction for N26 capture; no CUDA initialization or training.

Root owns the current admission/terminal chain. This module reuses only the
original numerical constructors and exact endpoint-state loader, never the old
N16 source_gate or its historical RAW-count admission. Imported private helper
modules route their single-reference bound() through the new Control.closed().
"""
import importlib.util
import inspect
import json
from pathlib import Path


MODULE_ROLES = ('native_loader_source','evaluation_source','micro_preflight_source',
                'initializer_source','training_source','native_helpers','renderer_source',
                'joint_losses_source','joint_preflight_entry_source')


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def read(c, ref):
    return json.loads(c.closed(ref).read_bytes())


def load(c, ref, name):
    path = c.closed(ref)
    module_spec = importlib.util.spec_from_file_location(name,str(path))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    # All affected old helpers use the statically confirmed bound(row/ref) ABI.
    # Only this private module object changes; original SOURCE bytes stay frozen.
    if hasattr(module,'bound'):
        need(callable(module.bound) and len(inspect.signature(module.bound).parameters) == 1,
             'Original single-reference bound helper ABI')
        module.bound = c.closed
    return module


def declared(c, refs, label):
    """Check membership without opening or hashing unrelated historical payloads."""
    for ref in refs:
        need(type(ref) is dict and set(ref) == {'path','bytes','sha256'}
             and Path(ref['path']).is_absolute()
             and c.allowed.get(str(Path(ref['path']).resolve()).lower()) == ref,
             label + ': original input absent or changed in current direct closure')


def strong_head(torch, c, carrier, evaluation, lineage):
    """Minimal exact endpoint view for the original evaluation.load_heads ABI."""
    endpoint = carrier['endpoint']
    need(endpoint['type'] == 'QUALIFIED_SUCCESSFUL_GPU_ENDPOINT'
         and carrier['seed'] == 202610120701 and carrier['head_endpoint_epoch'] == 24,
         'Only the original qualified strong seed701 E24 endpoint')
    training = read(c,endpoint['training_spec'])
    report = read(c,endpoint['training_report'])
    q = read(c,endpoint['CPU_qualification'])
    need(training['schema'] in lineage.PARENT_ABIS, 'Explicit original successful endpoint schema')
    abi = lineage.PARENT_ABIS[training['schema']]
    need(report['schema'] == abi['report'] and q['schema'] == abi['q']
         and q['status'] == abi['status'] and q['training_spec'] == endpoint['training_spec']
         and q['training_report'] == endpoint['training_report']
         and q['source'] == endpoint['training_entry_source']
         and q['reader_source'] == endpoint['CPU_entry_source']
         and q['spec'] == endpoint['CPU_spec'] and q['CPU_admission'] == endpoint['CPU_admission']
         and q['training_GPU_RAW'] == endpoint['training_GPU_RAW']
         and q['training_GPU_admission'] == endpoint['training_GPU_admission']
         and training['entry_source'] == report['source'] == endpoint['training_entry_source']
         and report['spec'] == endpoint['training_spec']
         and report['GPU_admission'] == endpoint['training_GPU_admission']
         and report['phase'] == training['phase']
         and training['seed'] == report['seed'] == q['seed'] == carrier['seed']
         and training['last_epoch'] == report['last_epoch'] == q['last_epoch'] == 24
         and training['recipe'] == report['recipe'] == q['recipe'] == carrier['training_recipe']
         and q['checkpoints'] == report['checkpoints'],
         'Original successful E24 training/report/independent CPU endpoint metadata')
    declared(c,training['sources'],'Original PR training closure')
    need(set(report['checkpoints']) == {'LOG_PR','GEO_PR'}
         and all(report['checkpoints'][a]['checkpoint'] == endpoint['checkpoints'][a]
                 and report['checkpoints'][a]['completed_head_epochs'] == 24
                 and report['checkpoints'][a]['updates'] == 39864 for a in ('LOG_PR','GEO_PR')),
         'Both original qualified complete-epoch checkpoint identities')
    declared(c,[carrier['point_heads_source']] + [report['checkpoints'][a][k]
        for a in ('LOG_PR','GEO_PR') for k in ('checkpoint','proof')], 'Original factory and endpoint artifacts')
    parent = dict(spec=training,checkpoints=report['checkpoints'],Q=q,CP_schema=abi['cp'])
    # The unchanged original loader checks exact 8 names/shapes/40497 scalars,
    # checkpoint metadata, factory/cache identities, and each state proof. Its
    # bound() and factory load() now close every read in the current own scope.
    heads,states,proofs = evaluation.load_heads(torch,carrier,dict(parent=parent))
    head = heads['LOG_PR']
    anchor = {name:value.detach().clone() for name,value in states['LOG_PR'].items()}
    proof = proofs['LOG_PR']
    need(len(head.state_dict()) == len(list(head.parameters())) == len(anchor) == 8
         and sum(p.numel() for p in head.parameters()) == 40497 and not list(head.buffers())
         and not head.training and all(p.device.type == 'cpu' and p.dtype == torch.float32
             and not p.requires_grad and p.grad is None for p in head.parameters())
         and all(torch.equal(value.detach(),anchor[name]) for name,value in head.state_dict().items()),
         'Original strong LOG_PR frozen CPU state and independent snapshot')
    del heads,states,proofs
    return head,anchor,proof


def setup(torch, np, cv2, ndimage, ctx, c, spec):
    c.check()
    need(ctx['schema'] == 'ROOT-N26-quality-input-context-v1'
         and torch.__version__ == '2.0.1+cu118' and np.__version__ == '1.23.5'
         and cv2.__version__ == '4.8.0' and not torch.cuda.is_initialized(),
         'Original numerical versions and CPU-only setup')
    old_spec = read(c,ctx['N16_GPU_spec'])
    n16_report = read(c,ctx['N16_GPU_report'])
    context = read(c,ctx['native_context_spec'])
    carrier = read(c,ctx['carrier_GPU_spec'])
    declared(c,old_spec['sources'],'Original N16 source closure')
    declared(c,context['sources'],'Original native context source closure')
    declared(c,carrier['sources'],'Original strong carrier source closure')
    for role in ('native_context_spec','carrier_GPU_spec','carrier_evaluation_source',
                 'original_fit_cache_entry_source','original_fit_cache_report','region_common_source','FIT'):
        need(ctx[role] == old_spec[role], 'Unchanged original capture setup role: ' + role)
    need(context['FIT'] == ctx['FIT'] and context['image_root'] == ctx['image_root'] == old_spec['image_root']
         and n16_report['source'] == old_spec['entry_source'] and n16_report['spec'] == ctx['N16_GPU_spec']
         and n16_report['teacher_checkpoint'] == context['checkpointmap']['native0']
         and n16_report['fixed_PR_checkpoint'] == carrier['endpoint']['checkpoints']['LOG_PR']
         and n16_report['images'] == 741 and n16_report['raw_GT'] == 1661
         and n16_report['natural_queries'] == 74100
         and carrier['entry_source'] == ctx['carrier_evaluation_source'],
         'Same native E24, strong LOG E24, canonical FIT and original source/report identities')
    from pycocotools import mask as masks
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    cache7 = load(c,ctx['original_fit_cache_entry_source'],'N26_ORIGINAL_FIT_CACHE_HELPERS')
    region_common = load(c,ctx['region_common_source'],'N26_ORIGINAL_REGION_HELPERS')
    need(cache7.MODULES == MODULE_ROLES, 'Original nine-module numerical setup ABI')
    loaded = [load(c,context[role],'N26_NUMERIC_'+role) for role in MODULE_ROLES]
    native,legacy,pref,initializer,common,old,diag,joint,preflight = loaded
    old.torch = torch
    # qualified_training first observes its original RAW receipts before scientific
    # lineage access. Close these direct references here; no current-count gate is reused.
    for ref in context['training_receiptmap'].values():
        c.closed(ref)
    teachers = legacy.qualified_training(context)
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    construction = dict(teachers['native0']['training_spec'],training_source=context['training_source'],
        checkpointmap=context['checkpointmap'],qualificationmap=context['qualificationmap'],
        order_seed=native.ORDER_SEED,phase_loss_seed_base=native.LOSS_SEED_BASE)
    model,anchor,policy,loading = native.prepare_training_model(torch,pref,initializer,common,MODELS,
        revert_sync_batchnorm,Config,tuple(CocoDataset.METAINFO['classes']),construction,teachers['native0'])
    native.imported_library_gate(torch,model,context)
    need(Path(inspect.getsourcefile(torch.nn.functional.interpolate)).resolve() ==
         c.closed(context['interpolation_SOURCE']).resolve(), 'Original actual renderer interpolation source')
    model.requires_grad_(False)
    model.eval()
    model.backbone.with_cp = False
    need(len(anchor) == len(model.state_dict()) == 610 and len(tuple(model.parameters())) == 451
         and all(p.device.type == 'cpu' and p.dtype == torch.float32
                 and not p.requires_grad and p.grad is None for p in model.parameters())
         and all(v.device.type == 'cpu' and torch.equal(v.detach(),anchor[n])
                 for n,v in model.state_dict().items()), 'Exact frozen native610/451 CPU teacher')
    evaluation = load(c,ctx['carrier_evaluation_source'],'N26_ORIGINAL_STRONG_HEAD_LOADER')
    lineage = load(c,carrier['endpoint_lineage_source'],'N26_ORIGINAL_ENDPOINT_ABI_CONSTANTS')
    head,pr_anchor,pr_proof = strong_head(torch,c,carrier,evaluation,lineage)
    dataset = CocoDataset(ann_file=str(c.closed(ctx['FIT'])),data_prefix=dict(img=ctx['image_root']),
        metainfo=dict(classes=('ship',)),pipeline=[],filter_cfg=None,test_mode=True)
    coco = COCO(str(c.closed(ctx['FIT'])))
    ids = [dataset.get_data_info(i)['img_id'] for i in range(len(dataset))]
    need(len(dataset) == len(coco.imgs) == 741 and len(coco.anns) == 1661 and set(coco.cats) == {1}
         and ids == list(coco.imgs) == ctx['image_ids'], 'Full canonical FIT741/1661 and original image order')
    need(sum(v['height']*v['width'] for v in coco.imgs.values()) == 119780015
         and max(v['height']*v['width'] for v in coco.imgs.values()) == 306257
         and all(str(Path(dataset.get_data_info(i)['img_path']).resolve()).lower() in c.allowed
                 for i in range(741)), 'Original full FIT geometry and bound image paths')
    old_report = read(c,ctx['original_fit_cache_report'])
    need(old_report['images'] == 741 and old_report['raw_GT'] == 1661
         and len(old_report['packet_refs']) == len(old_report['rows']) == 741,
         'Original matched FIT cache metadata; no packet tensor read by setup')
    closure = {ref['path']:ref for ref in spec['sources']}
    need(len(closure) == len(spec['sources']), 'Unique exact-path current source closure')
    # The new entry independently verifies its 13-field admission and exact map.
    # imported_files uses only this map; no old N16 admission/owner count is reused.
    admission_view = dict(bound_source_sha256={ref['path']:ref['sha256'] for ref in spec['sources']})
    libraries = region_common.imported_files(torch,np,cv2,masks,COCO,closure,admission_view,ndimage)
    need(not torch.cuda.is_initialized(), 'Setup completed entirely on CPU before CUDA initialization')
    c.check()
    loading = dict(loading,setup_scope='CPU-only numerical setup; current Root owns admission and actual terminal',
        strong_PR_checkpoint=carrier['endpoint']['checkpoints']['LOG_PR'],strong_PR_state_proof=pr_proof,
        imported_libraries=libraries,old_gate_reused=False,original_helpers_bound_to_current_Control=True)
    modules = dict(pref=pref,old=old,diag=diag,joint=joint,preflight=preflight,cache7=cache7,common=common,
        native=native,legacy=legacy,region_common=region_common)
    return dict(modules=modules,model=model,head=head,dataset=dataset,coco=coco,context=context,
                anchor=anchor,pr_anchor=pr_anchor,loading=loading)
