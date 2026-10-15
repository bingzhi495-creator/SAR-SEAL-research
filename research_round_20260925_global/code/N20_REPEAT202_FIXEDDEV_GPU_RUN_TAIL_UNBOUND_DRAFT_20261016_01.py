"""UNBOUND / NOT_EXECUTED N20 repeat202 fused DEV GPU tail; no admission front or CLI.

Concatenate after ROOT's typed RAW front, repeat202 normally-qualified DEV front and
prior DEV metadata hook. Existing helpers: require/read/closed/bound/load/
reference/write. No Source/ref is invented here; old frozen files stay unchanged.

Activation requires future external ROOT-N20-repeat202-paired-training-endpoints-
qualified-result-v1 after original GPU181 actual0 and independent CPU182 actual0.
This UNEXECUTED draft neither reads nor qualifies the currently active CPU leaf.

Additional spec source refs: capture_body_source (N20 DEV capture02),
coordinates_source, legacy_eval_body_source (N16 numerical body), labels_source;
new_eval_body_source/scout_module_source already belong to the DEV front.
capture_budget uses the six original N17 disk/RSS fields. Additional gate refs
prior_shared_validator_source/prior_common_decoder_source come from the prior
hook, not guessed spec aliases. Required gate runtime metadata:
DEV_packet_module_refs={n10,reader,renderer}, DEV_packet_gate (original prior
report), DEV_annotation_ref, DEV_packet_rows/DEV_packet_refs[187],
prior_common_rows[187] items={row,binding}; legacy_rows/populations/training/
trained_checkpoints/context/evaluation/common/lineage/parent and own ref fields.

Per-image: read old N16 bundle once, read old N17 common bundle once, capture
extra/map into an in-memory member, score, then save/readback/hash one physical
bundle. No teacher forward or new full-C PR; selected PR is freshly executed.
The new reader must understand F9/FP64-score schemas; old extra8 decoder is used
only to authenticate/read the PRIOR common PR, never the new N20 bundle.
"""

import gc
import json
import math
import os
from pathlib import Path
import time

N20_BUNDLE_SCHEMA = 'N20-actual-GPU-feature-neighbors-selector-image-bundle-v1'
N20_POINT_SCHEMA = 'N20-actual-GPU-feature-neighbors-selector-point-chunk-v1'
N20_PROFILE_SCHEMA = 'N20-actual-GPU-feature-neighbors-selector-query-profile-v1'
N20_SCORE_SCHEMA = 'N20-actual-GPU-logits-canonical-FP64-route-scores-v1'
N20_MANIFEST_SCHEMA = 'N20-fixedDEV-feature-neighbors-GPU-shard-capture-manifest-v1'
N20_REPORT_SCHEMA = 'N20-fixedDEV-feature-error-GPU-shard-report-v1'
N20_STATUS = 'N20_FIXEDDEV_FEATURE_ERROR_SHARD_COMPLETE_PENDING_ROOT_ACTUAL0_AND_CPU'
N20_ARMS = ('LOCAL_ERROR', 'CENTER_ERROR')
N20_EPOCHS = (24, 8)
N20_MODES = ('atmost', 'exact')
N20_SHAPES = {'spatial_weight': (9,256), 'hidden.weight': (32,513),
              'hidden.bias': (32,), 'output.weight': (3,32), 'output.bias': (3,)}
N20_RANGES = ((0,47), (47,94), (94,141), (141,187))
N20_THREAD_ENV = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def n20_policy_names():
    return ('native_official','strong_fixed_E24_Omega') + tuple(
        'E%d_%s_%s' % (epoch,arm,mode) for epoch in N20_EPOCHS for arm in N20_ARMS for mode in N20_MODES)


def n20_tensor_bytes(torch, value):
    if isinstance(value, torch.Tensor):
        require(value.device.type=='cpu' and value.layout==torch.strided and not value.requires_grad
            and value.dtype in (torch.float32,torch.float64,torch.int64,torch.uint8,torch.bool)
            and bool(torch.isfinite(value).all()), 'Finite detached CPU bundle tensor')
        return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(n20_tensor_bytes(torch,v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(n20_tensor_bytes(torch,v) for v in value)
    return 0


def n20_same_bits(torch, left, right):
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.numel() == 0:
        return True
    def logical_bytes(value):
        # Normal contiguous F9 uses aliasing views; no additional tensor copy.
        flat = value.contiguous().reshape(-1)
        if flat.stride(0) != 1:
            # A singleton expand can be logically contiguous with stride zero.
            # Allocate a standard-stride view only for this pathological case.
            safe = torch.empty(flat.numel(), dtype=flat.dtype, device=flat.device)
            safe.copy_(flat)
            flat = safe
        return flat.view(torch.uint8)
    return torch.equal(logical_bytes(left), logical_bytes(right))

def n20_tree_equal(torch, left, right):
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and n20_same_bits(torch,left,right)
    if isinstance(left,dict):return isinstance(right,dict) and list(left)==list(right) and all(n20_tree_equal(torch,left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)):return type(left) is type(right) and len(left)==len(right) and all(n20_tree_equal(torch,a,b) for a,b in zip(left,right))
    return type(left) is type(right) and left==right


def n20_member_descriptor(torch, value):
    return dict(schema=value['schema'],query_id=value.get('query_id'),stage=value.get('stage','profile'),
        scout_epoch=value.get('scout_epoch',value.get('epoch')),control=value.get('control',value.get('arm')),
        evalmode=value.get('evalmode',value.get('mode')),chunk_first=value.get('chunk_first'),
        raw_tensor_payload_bytes=n20_tensor_bytes(torch,value))


def n20_point_member_id(index,value):
    q,first=value['query_id'],value['chunk_first'];stage=value['stage'];epoch=value['scout_epoch'];arm=value['control'];mode=value['evalmode']
    require(type(q) is int and 0<=q<100 and type(first) is int and first>=0 and first%2048==0,'Actual point role/query/chunk')
    if stage=='full_C_PR':
        require(epoch is None and arm=='COMMON_C_REFERENCE' and mode=='shared','Original common-C role');role='commonC'
    else:
        require(epoch in N20_EPOCHS and arm in N20_ARMS and ((stage=='scout_C_logits' and mode=='shared-atmost-exact')
            or (stage=='selected_PR' and mode in N20_MODES)),'Fixed N20 scout/selected role')
        role='e%d_%s_%s_%s'%(epoch,arm,mode,stage)
    return 'i%03d/q%03d/%s_c%06d.pth'%(index,q,role,first)


def n20_capture_event_cpu(torch,np,event,shared,combined_CPU,mapping,input_member,common_origin):
    require(event['actual_device_type']=='cuda' and event['synthetic_only'] is False,'Actual CUDA callback only')
    value={k:(v.detach().cpu().clone().contiguous() if isinstance(v,torch.Tensor) else v) for k,v in event.items()}
    flat,q=value['flat_indices'],value['query_id'];union=shared['union_flat_indices'].numpy()
    require(flat.dtype==torch.int64 and flat.ndim==1 and 0<len(flat)<=2048,'Canonical nonempty callback block')
    positions=np.searchsorted(union,flat.numpy())
    require(bool(np.all(positions<len(union))) and np.array_equal(union[positions],flat.numpy()),'Actual callback union coverage')
    positions_tensor=torch.from_numpy(positions.astype(np.int64))
    expected_F=shared['F_unique'][positions_tensor]
    require(n20_same_bits(torch,value['sampled_F'],expected_F) and n20_same_bits(torch,value['e0'],shared['e0'][q])
        and n20_same_bits(torch,value['native_L'],shared['native_L'][q].reshape(-1)[flat]),'Actual F/e/L must equal old input bytes')
    if value['stage']=='scout_C_logits':
        expected=(combined_CPU[mapping[positions_tensor]] if value['control']=='LOCAL_ERROR'
                  else expected_F[:,None,:].repeat(1,9,1))
        require(value['neighbor_center_column']==4 and torch.equal(value['neighbor_union_positions'],positions_tensor)
            and n20_same_bits(torch,value['neighbors_F'],expected)
            and n20_same_bits(torch,value['actual_GPU_logits'],value['scout_logits']), 'Actual nine-F inputs/logit alias byteexact before deduplication')
        for key in ('neighbors_F','neighbor_union_positions','neighbor_center_column','scout_logits'):del value[key]
        value.update(neighbor_input_member=input_member,actual_observed_neighbors_F_bitexact_to_input_carrier=True,
                     actual_GPU_logits_alias_byte_exact=True)
    elif value['stage']=='full_C_PR':
        require(value['reused_qualified_common_PR'] is True and value['fresh_PR_forward_executed'] is False
            and value['result_origin']=='qualified_prior_GPU_common_C_capture','Common is original qualified reference, never new execution')
        origin=common_origin[(q,value['chunk_first'])]
        value.update(original_common_bundle=origin['bundle'],original_common_member=origin['member'],
                     original_common_GPU_binding=origin['binding'])
    else:require(value['stage']=='selected_PR','Fresh PR only on the new selected support')
    for key in ('sampled_F','e0','native_L'):del value[key]
    value.update(schema=N20_POINT_SCHEMA,actual_observed_F_e_L_bitexact_to_input_carrier=True)
    return value


class N20ImageWriter:
    def __init__(self,torch,output,spec,binding,check):
        self.torch,self.output,self.spec,self.binding,self.check=torch,Path(output),spec,binding,check
        self.refs=[];self.raw=0;self.disk=0
    def begin(self,item,common):
        self.item=item;self.members={};self.registry={};self.image_raw=0;row=item['row']
        self.input_carrier=dict(bundle=row['bundle'],shared_member=row['shared_actual_image_inputs'],original_GPU_binding=item['binding'],
            common_PR_bundle=common['row']['bundle'],common_PR_original_GPU_binding=common['binding'])
    def put(self,name,value):
        self.check();require(type(name) is str and len(name.encode('utf8'))<=256 and name not in self.members,'Unique bounded internal member ID')
        size=n20_tensor_bytes(self.torch,value);b=self.spec['capture_budget']
        require(self.image_raw+size<=b['maximum_perimage_raw_capture_bytes'] and self.raw+self.image_raw+size<=b['maximum_total_raw_capture_payload_bytes'],'Actual raw tensor cap')
        self.members[name]=value;self.registry[name]=n20_member_descriptor(self.torch,value);self.image_raw+=size
        return name
    def finish(self):
        index=self.item['row']['dataset_index'];target=self.output/('IMAGE_%03d_ACTUAL_GPU_BUNDLE.pth'%index);self.check()
        payload=dict(schema=N20_BUNDLE_SCHEMA,binding=self.binding,input_carrier=self.input_carrier,dataset_index=index,
            image_id=self.item['row']['image_id'],members=self.members,member_registry=self.registry,
            member_count=len(self.members),raw_tensor_payload_bytes=self.image_raw)
        with target.open('xb') as stream:self.torch.save(payload,stream)
        replay=self.torch.load(str(target),map_location='cpu');require(n20_tree_equal(self.torch,payload,replay),'Exact physical bundle readback')
        del replay
        ref=reference(target);require(self.disk+ref['bytes']<=self.spec['capture_budget']['maximum_serialized_capture_bytes'],'Actual serialized capture cap')
        self.refs.append(ref);self.raw+=self.image_raw;self.disk+=ref['bytes'];members=list(self.members)
        self.members={};self.registry={};return ref,members


def n20_capture_budget_gate(spec,gate):
    b=spec['capture_budget'];keys=('maximum_process_RSS_bytes','maximum_perimage_raw_capture_bytes','maximum_total_raw_capture_payload_bytes',
        'maximum_serialized_capture_bytes','minimum_free_output_bytes','auxiliary_output_budget_bytes')
    require(type(b) is dict and all(type(b[k]) is int and b[k]>0 for k in keys)
        and b['maximum_process_RSS_bytes']<=16*1024**3 and b['maximum_perimage_raw_capture_bytes']<=1024**3
        and b['maximum_serialized_capture_bytes']+b['auxiliary_output_budget_bytes']<=72*1024**3
        and b['minimum_free_output_bytes']>=b['maximum_serialized_capture_bytes']+b['auxiliary_output_budget_bytes']+16*1024**3,'Concrete unchanged disk/RAM admission')
    first,last=N20_RANGES[spec['shard_id']];rows=[];total=0
    for item in gate['legacy_rows'][first:last]:
        row=item['row'];N=math.prod(row['identity']['original_shape']);cost=row['shared_capture_cost'];U,C,K=cost['U'],cost['sumC'],cost['sumK']
        require(all(type(v) is int for v in (N,U,C,K)) and 0<=U<=N and C>=K>=0,'Qualified native geometry/C/K/union metadata')
        E_upper=min(8*U,N-U)
        per=1032*E_upper+72*U+160*C+192*K+800*((N+7)//8)+16*1024**2
        require(per<=b['maximum_perimage_raw_capture_bytes'],'F9 difference/FP64 score/profile raw upper bound')
        rows.append(dict(dataset_index=row['dataset_index'],N=N,U=U,C=C,K=K,extra_count_upper=E_upper,raw_and_metadata_upper_bytes=per));total+=per
    require(total<=b['maximum_total_raw_capture_payload_bytes'] and total<=b['maximum_serialized_capture_bytes'],'Full fixed-shard conservative bound before any scout/PR')
    return dict(rows=rows,total_upper_bytes=total,extra_count_bound_is_conservative=True,
                formula='1032Eupper+72U+160C+192K+800ceil(N/8)+16MiB/image')


def n20_model_load_cpu(torch,spec,gate):
    require(not torch.cuda.is_initialized(),'Cold CUDA through all original PR/checkpoint loads')
    outside=torch.get_rng_state().clone();heads,states,proofs=gate['evaluation'].load_heads(torch,gate['context'],gate)
    strong=heads['LOG_PR'];pr=states['LOG_PR'];require(len(pr)==8 and sum(v.numel() for v in pr.values())==40497 and not tuple(strong.buffers()),'Frozen original LOG E24 PR40497 only')
    strong.eval().requires_grad_(False);frozen={'PR':{k:v.detach().cpu().clone() for k,v in pr.items()}};del heads,states,pr
    scout=load(spec['scout_module_source'],'_N20_DEV_feature_adapter');models={};prefix={}
    original_binding=gate['training']['original_binding']
    for epoch in (8,24):
        cp=torch.load(str(closed(gate['closure'],gate['trained_checkpoints'][str(epoch)])),map_location='cpu')
        require(cp['schema']=='N20-paired-local-center-feature-error-training-CPU-snapshot-v1' and cp['completed_epoch']==epoch
            and cp['parameters_per_arm']==18851 and cp['initialization_seed']==202610150202 and set(cp['models'])==set(N20_ARMS)
            and cp['metadata']==dict(original_binding,endpoint_epoch=epoch),'Actual normally-qualified repeat202 nested checkpoint binding')
        prefix[epoch]=(cp['history'],cp['epoch_orders'])
        for arm in N20_ARMS:
            state=cp['models'][arm];require(set(state)==set(N20_SHAPES) and all(v.device.type=='cpu' and v.dtype==torch.float32
                and tuple(v.shape)==N20_SHAPES[k] and bool(torch.isfinite(v).all()) for k,v in state.items()),'Five finite trained N20 tensors, no zero/divergence restriction')
            head=scout.create_head(torch);head.load_state_dict(state,strict=True);head.eval().requires_grad_(False)
            models[(epoch,arm)]=head;frozen[(epoch,arm)]={k:v.detach().clone() for k,v in state.items()}
        del cp,state,head
    require(prefix[24][0][:8]==prefix[8][0] and prefix[24][1][:8]==prefix[8][1] and torch.equal(outside,torch.get_rng_state())
        and not torch.cuda.is_initialized(),'Both fixed trajectory endpoints and private CPU construction')
    return strong,models,frozen,proofs


def run(spec_path):
    started=time.monotonic();require(Path(spec_path).is_absolute(),'Absolute N20 DEV shard specification');path=Path(spec_path).resolve(strict=True)
    spec=read(path);gate=source_gate(path,spec);c=gate['closure'];budget=n20_capture_budget_gate(spec,gate)
    output=Path(spec['output_directory']);require(output.is_absolute() and not output.exists() and not output.is_symlink() and output.parent.is_dir(),'Fresh shard directory with existing parent')
    import psutil
    process=psutil.Process();require(process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
        and all(os.environ.get(k)=='2' for k in N20_THREAD_ENV) and psutil.virtual_memory().available>=16*1024**3
        and psutil.disk_usage(str(output.parent)).free>=spec['capture_budget']['minimum_free_output_bytes'],'Actual admitted CPU01/GPU0/thread/disk/RAM')
    import torch
    import numpy as np,cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    require(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and cv2.__version__=='4.8.0' and not torch.cuda.is_initialized(),'Closed cold runtime')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    sources=('capture_body_source','coordinates_source','legacy_eval_body_source','labels_source','new_eval_body_source')
    for key in sources:closed(c,spec[key])
    for key in ('prior_shared_validator_source','prior_common_decoder_source'):closed(c,gate[key])
    refs=gate['DEV_packet_module_refs'];require(type(refs) is dict and set(refs)=={'n10','reader','renderer'},'Explicit original native packet/renderer refs')
    for ref in refs.values():closed(c,ref)
    closed(c,gate['DEV_annotation_ref'])
    require(len(gate['DEV_packet_rows'])==len(gate['DEV_packet_refs'])==len(gate['legacy_rows'])==len(gate['prior_common_rows'])==187,'Full187 original/prior carrier records')
    capture_body=load(spec['capture_body_source'],'_N20_DEV_extra_capture');coordinates=load(spec['coordinates_source'],'_N20_DEV_coordinates')
    body=load(spec['new_eval_body_source'],'_N20_DEV_numerical');legacy=load(spec['legacy_eval_body_source'],'_N16_original_numerical')
    helper=load(spec['labels_source'],'_N16_FP64_selection');validator=load(gate['prior_shared_validator_source'],'_N17_actual_shared_validator')
    decoder=load(gate['prior_common_decoder_source'],'_N17_prior_common_decoder')
    packet_modules={k:load(ref,'_N20_original_'+k) for k,ref in refs.items()};coco=COCO(str(closed(c,gate['DEV_annotation_ref'])))
    require(tuple(body.ARMS)==N20_ARMS and tuple(body.EPOCHS)==N20_EPOCHS and tuple(body.MODES)==N20_MODES
        and body.STATE_SHAPES==N20_SHAPES and body.POINT_CHUNK==2048 and body.ABSOLUTE_TOLERANCE==1e-4,'Exact fixed N20 evalbody ABI')
    strong,models,frozen,model_proofs=n20_model_load_cpu(torch,spec,gate)
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,c,gate['admission'],ndimage)
    require(not torch.cuda.is_initialized(),'All closed modules and recovered models precede CUDA')
    torch.cuda.set_device(0);torch.cuda.init();require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Only actual CUDA0')
    torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0);torch.cuda.reset_peak_memory_stats(0)
    outside=(torch.get_rng_state().clone(),torch.cuda.get_rng_state(0).clone());strong.to('cuda:0')
    for head in models.values():head.to('cuda:0')
    del head
    memory_observations=[]
    def check():
        require(time.monotonic()-started<1800 and process.memory_info().rss<spec['capture_budget']['maximum_process_RSS_bytes']
            and psutil.virtual_memory().available>=16*1024**3 and process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
            and all(os.environ.get(k)=='2' for k in N20_THREAD_ENV) and torch.get_num_threads()==torch.get_num_interop_threads()==2
            and cv2.getNumThreads()==2 and torch.cuda.current_device()==0 and torch.cuda.memory_reserved(0)<=4096*1024**2,'Bounded single leaf/resource/deadline; no retry')
    binding=dict(source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],phase=spec['phase'],
        N20_repeat202_ROOT_training_qualification=spec['N20_repeat202_ROOT_training_qualification'],
        initialization_seed=202610150202,N20_locked_protocol=spec['N20_locked_protocol'],
        legacy_qualified_result=spec['legacy_qualified_result'],new_eval_body_source=spec['new_eval_body_source'],capture_body_source=spec['capture_body_source'],
        coordinates_source=spec['coordinates_source'],checkpoints=gate['trained_checkpoints'],shard_id=spec['shard_id'],dataset_ordinal_range=spec['dataset_ordinal_range'])
    output.mkdir();(output/'CAPTURES').mkdir();writer=N20ImageWriter(torch,output/'CAPTURES',spec,binding,check)
    predictions={name:[] for name in n20_policy_names()};rows=[];costs=[];auxiliary=0
    def auxiliary_write(target,value):
        nonlocal auxiliary
        write(target,value);auxiliary+=target.stat().st_size;require(auxiliary<=spec['capture_budget']['auxiliary_output_budget_bytes'],'Actual auxiliary output cap')
    def encode(mask):
        encoded=masks.encode(np.asfortranarray(mask.astype(np.uint8)));return dict(size=[int(v) for v in encoded['size']],counts=encoded['counts'].decode('ascii'))
    auxiliary_write(output/'START.json',dict(schema='N20-fixedDEV-feature-GPU-shard-start-v1',**binding,model_proofs=model_proofs,
        capture_budget=budget,policies=n20_policy_names(),original_training_GPU_actual0_verified=True,
        training_saved_artifact_recovery=False,independent_endpoint_CPU_qualified=True,
        main_epoch=24,curve_epoch=8,native_model_loaded=False,AP_evaluated=False))
    first,last=N20_RANGES[spec['shard_id']]
    # Reuse the exact original CV2-domain facade from the same known validator source.
    engine=validator.old_candidate_engine(np,cv2)
    for item in gate['legacy_rows'][first:last]:
        check();row=item['row'];index=row['dataset_index'];tick=time.monotonic();common_item=gate['prior_common_rows'][index]
        require(common_item['row']['dataset_index']==index and common_item['row']['image_id']==row['image_id']
            and common_item['row']['identity']==row['identity'] and common_item['row']['qids']==row['qids'],'N16/N17 same whole-image/common carrier')
        bundle=torch.load(str(closed(c,row['bundle'])),map_location='cpu');shared=validator.validate_actual_shared(torch,np,cv2,bundle,item)
        native_ref=gate['DEV_packet_refs'][index];require(native_ref==row['native_packet'],'Original packet ref must equal the qualified shared carrier')
        packet=torch.load(str(closed(c,native_ref)),map_location='cpu')
        extra_row,extra_payload=capture_body.capture_image(torch,np,packet_modules,packet,row,index,native_ref,shared,
            gate['DEV_packet_gate'],coordinates,coco=coco,packet_row=gate['DEV_packet_rows'][index],check=check)
        del packet
        writer.begin(item,common_item);input_member=writer.put('i%03d/FEATURE_NEIGHBORS_INPUTS.pth'%index,extra_payload)
        baseline=gate['populations']['native_official'][row['image_id']];strong_predictions=gate['populations']['strong_fixed_E24_Omega'][row['image_id']]
        old_common=torch.load(str(closed(c,common_item['row']['bundle'])),map_location='cpu')
        prior=decoder.decode_image(torch,np,shared,old_common,common_item['row'],common_item['binding'],common_item['row']['input_carrier'],baseline)
        require(common_item['row']['input_carrier']['bundle']==row['bundle'] and common_item['row']['input_carrier']['shared_member']==row['shared_actual_image_inputs'],
            'Old common PR and new scouts use identical original shared carrier')
        L=shared['native_L'].to('cuda:0');e=shared['e0'].to('cuda:0');F=shared['F_unique'].to('cuda:0');extra_GPU=extra_payload['extra_F'].to('cuda:0')
        require(all(n20_same_bits(torch,a.cpu(),b) for a,b in ((L,shared['native_L']),(e,shared['e0']),(F,shared['F_unique']),(extra_GPU,extra_payload['extra_F']))),'Actual old/new features survive one GPU transfer byteexact')
        neighbor_inputs=dict(extra_F=extra_GPU,extra_flat_indices=extra_payload['extra_flat_indices'],neighbor_to_combined=extra_payload['neighbor_to_combined'])
        combined_CPU=torch.cat((shared['F_unique'],extra_payload['extra_F']),0);common_origin={};refs_by_role={};query_members=[]
        current={name:{} for name in n20_policy_names()[2:]};by_q={v['query_id']:v for v in baseline}
        def common_provider(q,active,K):
            query=prior.query(q);require(query['K']==K and np.array_equal(query['C_flat_indices'].numpy(),shared['C_indices'][q].numpy()),'Prior common query/C/K unchanged')
            profile_id=common_item['row']['query_profiles'][q]['profiles']['E24_LOCAL_ERROR_atmost']
            names=old_common['members'][profile_id]['common_C_PR_chunks']
            for i,name in enumerate(names):common_origin[(q,i*2048)]=dict(bundle=common_item['row']['bundle'],member=name,binding=common_item['binding'])
            return query['common_PR_chunks']
        def capture(event):
            value=n20_capture_event_cpu(torch,np,event,shared,combined_CPU,extra_payload['neighbor_to_combined'],input_member,common_origin)
            name=writer.put(n20_point_member_id(index,value),value)
            refs_by_role.setdefault((value['query_id'],value['scout_epoch'],value['control'],value['evalmode'],value['stage']),[]).append(name)
        for result in body.evaluate_all100(torch,np,engine,helper,legacy,L,e,shared['union_flat_indices'],F,neighbor_inputs,models,strong,
                point_callback=capture,check_callback=check,common_PR_provider=common_provider,synthetic_only=False):
            q=result['query_id'];profiles={};scores_members={}
            for name,profile in result['profiles'].items():
                epoch,arm,mode=profile['epoch'],profile['arm'],profile['mode'];scout_chunks=refs_by_role.get((q,epoch,arm,'shared-atmost-exact','scout_C_logits'),[])
                model_key=(epoch,arm)
                if model_key not in scores_members:
                    score_value=dict(schema=N20_SCORE_SCHEMA,query_id=q,epoch=epoch,arm=arm,K=result['K'],C_count=len(result['C_flat_indices']),
                        flat_indices=torch.from_numpy((result['C_flat_indices'] if result['K'] else np.empty(0,dtype=np.int64)).copy()),
                        C_scout_chunks=scout_chunks,canonical_CPU_FP64_C_scores=torch.from_numpy(profile['canonical_CPU_FP64_C_scores'].copy()),actual_GPU_logits_used=bool(result['K']))
                    scores_members[model_key]=writer.put('i%03d/q%03d/e%d_%s_SCORES.pth'%(index,q,epoch,arm),score_value)
                encoded=encode(profile['mask']);prediction=dict(by_q[q],segmentation=encoded);current[name][q]=prediction
                value=dict(schema=N20_PROFILE_SCHEMA,query_id=q,epoch=epoch,arm=arm,mode=mode,K=result['K'],C_scout_chunks=scout_chunks,
                    common_C_PR_chunks=refs_by_role.get((q,None,'COMMON_C_REFERENCE','shared','full_C_PR'),[]),
                    selected_PR_chunks=refs_by_role.get((q,epoch,arm,mode,'selected_PR'),[]),
                    route_scores_member=scores_members[model_key],neighbor_input_member=input_member,
                    actual_selected_flat_indices=torch.from_numpy(profile['selected_flat_indices'].copy()),
                    actual_mask_packbits=torch.from_numpy(np.packbits(profile['mask'].ravel(),bitorder='little')),actual_mask_shape=list(profile['mask'].shape),actual_mask_bitorder='little',
                    actual_RLE=encoded,original_prediction_nonsegmentation_fields={k:v for k,v in by_q[q].items() if k!='segmentation'},PR_compatibility=profile['PR_compatibility'])
                profiles[name]=writer.put('i%03d/q%03d/e%d_%s_%s_PROFILE.pth'%(index,q,epoch,arm,mode),value)
            query_members.append(dict(query_id=q,profiles=profiles,K=result['K'],C_count=len(result['C_flat_indices'])))
            del result,profile,profiles,scores_members
        require(len(query_members)==100 and all(set(v)==set(range(100)) for v in current.values()),'Full natural100/8profiles actual outputs')
        predictions['native_official'].extend(baseline);predictions['strong_fixed_E24_Omega'].extend(strong_predictions)
        for name,values in current.items():
            ordered=[values[q] for q in row['qids']]
            require(all({k:v for k,v in x.items() if k!='segmentation'}=={k:v for k,v in y.items() if k!='segmentation'} for x,y in zip(ordered,baseline)),'Original class/S0/query/export ordering unchanged')
            predictions[name].extend(ordered)
        ref,members=writer.finish();rows.append(dict(dataset_index=index,image_id=row['image_id'],identity=row['identity'],qids=row['qids'],
            input_carrier=writer.input_carrier,bundle=ref,member_locators=members,query_profiles=query_members,natural_queries=100,neighbor_input_member=input_member))
        costs.append(dict(dataset_index=index,inclusive_seconds=time.monotonic()-tick,feature_capture=extra_row,standalone_FPS_measured=False))
        memory_observations.append(dict(dataset_index=index,allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0)))
        del bundle,shared,old_common,prior,extra_payload,neighbor_inputs,combined_CPU,L,e,F,extra_GPU,current,refs_by_role,common_origin,query_members
        gc.collect();check()
    require(all(len(v)==(last-first)*100 for v in predictions.values()),'Complete fixed shard; no local AP')
    for key,head in [('PR',strong)]+list(models.items()):
        require(all(n20_same_bits(torch,v.detach().cpu(),frozen[key][k]) for k,v in head.state_dict().items())
            and all(not p.requires_grad and p.grad is None for p in head.parameters()),'Frozen PR/five-tensor scouts unchanged')
    require(torch.equal(outside[0],torch.get_rng_state()) and torch.equal(outside[1],torch.cuda.get_rng_state(0)),'Evaluation external RNG unchanged')
    peak=torch.cuda.max_memory_allocated(0);before=dict(allocated=torch.cuda.memory_allocated(0),reserved=torch.cuda.memory_reserved(0));strong.cpu()
    for head in models.values():head.cpu()
    models.clear();del head,strong,models;gc.collect();torch.cuda.synchronize(0);torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();torch.cuda.synchronize(0)
    cleanup=dict(allocated_after=torch.cuda.memory_allocated(0),reserved_after=torch.cuda.memory_reserved(0))
    require(cleanup==dict(allocated_after=0,reserved_after=0) and peak<=4096*1024**2,'Actual complete GPU cleanup including cuBLAS workspaces')
    outputs={}
    for name,values in predictions.items():
        target=output/(name+'_PREDICTIONS.json');auxiliary_write(target,values);outputs[name]=dict(predictions=reference(target))
    auxiliary_write(output/'CAPTURE_MANIFEST.json',dict(schema=N20_MANIFEST_SCHEMA,**binding,images=last-first,queries=(last-first)*100,
        global_images=187,global_queries=18700,rows=rows,physical_capture_refs=writer.refs,internal_member_locators_are_strings=True,
        raw_tensor_payload_bytes=writer.raw,serialized_bytes=writer.disk))
    for ref in spec['sources']:check();bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and reference(spec['GPU_admission_file'])==gate['admission_ref']
        and gate['common'].imported_files(torch,np,cv2,masks,COCO,c,gate['admission'],ndimage)==libraries,'Final Source/spec/admission/imported libraries unchanged')
    gate['lineage'].final_endpoint_check(gate['parent']);check()
    auxiliary_write(output/'REPORT.json',dict(schema=N20_REPORT_SCHEMA,status=N20_STATUS,**binding,controls=n20_policy_names(),control_results=outputs,
        scope_dataset_indices=list(range(first,last)),scope_image_ids=[r['image_id'] for r in rows],images=last-first,queries=(last-first)*100,
        capture_manifest=reference(output/'CAPTURE_MANIFEST.json'),capture_refs=writer.refs,START=reference(output/'START.json'),
        normal_RAW_count=spec['normal_RAW_count'],total_failed_RAW_count=spec['total_failed_RAW_count'],total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],
        capture_budget=budget,serialized_bytes=writer.disk,auxiliary_bytes_before_REPORT=auxiliary,image_costs=costs,memory_observations=memory_observations,
        GPU_peak_allocated_bytes=peak,pre_cleanup=before,cleanup=cleanup,models_frozen=True,qualified=False,ROOT_actual_GPU0_verified=False,
        original_training_GPU_actual0_verified=True,training_saved_artifact_recovery=False,
        independent_endpoint_CPU_qualified=True,CPU_numerical_qualification_pending=True,
        teacher_forward_calls=0,new_fullC_PR_forward_calls=0,fresh_selected_PR=True,reused_common_C_reference=True,
        native_model_loaded=False,AP_evaluated=False,local_AP_computed=False,full187_qualification_claimed=False,standalone_FPS_measured=False,
        shared_diagnostic_costs_nonadditive=True,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(status=N20_STATUS,report=reference(output/'REPORT.json'),qualified=False)),flush=True)
