"""UNBOUND_DRAFT: pure already-loaded N20 true-F9 bundle -> selected CPU-core adapter.
No IO/Source/RAW/terminal/model load/COCO qualification. ROOT authenticates carriers.
Old shared tensors stay external; member strings are IDs, never physical file refs.
"""
from pathlib import Path
import hashlib
ARMS=('LOCAL_ERROR','CENTER_ERROR');EPOCHS=(24,8);MODES=('atmost','exact')
POLICIES=tuple('E%d_%s_%s'%(epoch,arm,mode) for epoch in EPOCHS for arm in ARMS for mode in MODES)
BUNDLE_KEYS={'schema','binding','input_carrier','dataset_index','image_id','members','member_registry','member_count','raw_tensor_payload_bytes'}
POINT_BASE={'schema','query_id','control','scout_epoch','evalmode','stage','point_chunk','chunk_first','flat_indices','actual_device_type','synthetic_only',
    'actual_input_scope','actual_Source_REF_qualified_here','actual_observed_F_e_L_bitexact_to_input_carrier'}
PROFILE_KEYS={'route_scores_member','neighbor_input_member','schema','query_id','epoch','arm','mode','K','C_scout_chunks','common_C_PR_chunks','selected_PR_chunks','actual_selected_flat_indices',
    'actual_mask_packbits','actual_mask_shape','actual_mask_bitorder','actual_RLE','original_prediction_nonsegmentation_fields','PR_compatibility'}

def need(ok,message):
    if not ok:raise ValueError(message)
def _ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'} and type(value['path']) is str and Path(value['path']).is_absolute() and
        type(value['bytes']) is int and value['bytes']>=0 and type(value['sha256']) is str and len(value['sha256'])==64 and
        set(value['sha256'])<=set('0123456789abcdef'),'Strict physical three-field ref required; members are strings')
def _tensor(torch,value,dtype,shape):
    need(isinstance(value,torch.Tensor) and value.layout==torch.strided and value.device.type=='cpu' and value.dtype==dtype and
        tuple(value.shape)==tuple(shape) and not value.requires_grad and bool(torch.isfinite(value).all()),'Finite exact detached CPU array shape/dtype')
def _bytes(torch,value):
    if isinstance(value,torch.Tensor):
        need(value.dtype in (torch.float32,torch.int64,torch.uint8,torch.bool,torch.float64),'Capture dtype allowed');_tensor(torch,value,value.dtype,value.shape)
        return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(_bytes(torch,v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(_bytes(torch,v) for v in value)
    return 0

def _profile_id(index,q,epoch,arm,mode):return 'i%03d/q%03d/e%d_%s_%s_PROFILE.pth'%(index,q,epoch,arm,mode)
def _point_id(index,value):
    q=value['query_id'];first=value['chunk_first'];stage=value['stage'];epoch=value['scout_epoch'];arm=value['control'];mode=value['evalmode']
    need(type(q) is int and 0<=q<100 and type(first) is int and first>=0 and first%2048==0,'Canonical query/chunk metadata')
    if stage=='full_C_PR':need(arm=='COMMON_C_REFERENCE' and epoch is None and mode=='shared','Common-C role exact');role='commonC'
    else:
        need(type(epoch) is int and epoch in EPOCHS and arm in ARMS and ((stage=='scout_C_logits' and mode=='shared-atmost-exact') or
            (stage=='selected_PR' and mode in MODES)),'Only new fixed scout/selected PR roles');role='e%d_%s_%s_%s'%(epoch,arm,mode,stage)
    return 'i%03d/q%03d/%s_c%06d.pth'%(index,q,role,first)
def _registry(torch,value):
    return dict(schema=value['schema'],query_id=value.get('query_id'),stage=value.get('stage','profile'),scout_epoch=value.get('scout_epoch',value.get('epoch')),
        control=value.get('control',value.get('arm')),evalmode=value.get('evalmode',value.get('mode')),chunk_first=value.get('chunk_first'),raw_tensor_payload_bytes=_bytes(torch,value))


INPUT_KEYS={'schema','parent_bundle','parent_shared_member','native_packet','identity','dataset_index','image_id','selector',
    'natural_query_ids','native_export_query_ids','original_unique_count','extra_count','center_column','neighbor_order','combined_order',
    'original_union_flat_sha256','extra_F','extra_flat_indices','neighbor_to_combined'}
SCORE_KEYS={'schema','query_id','epoch','arm','K','C_count','flat_indices','C_scout_chunks','canonical_CPU_FP64_C_scores','actual_GPU_logits_used'}
COMMON_KEYS={'reused_qualified_common_PR','fresh_PR_forward_executed','result_origin','original_common_bundle','original_common_member','original_common_GPU_binding'}
def _score_id(index,q,epoch,arm):return 'i%03d/q%03d/e%d_%s_SCORES.pth'%(index,q,epoch,arm)

def _feature_input(torch,np,value,shared,row,carrier,union,H,W):
    need(type(value) is dict and set(value)==INPUT_KEYS and value['schema']=='N20-DEV-true-feature-neighbor-extra-capture-body-v1' and
        value['dataset_index']==row['dataset_index'] and value['image_id']==row['image_id'] and value['identity']==shared['identity'] and
        value['parent_bundle']==carrier['bundle'] and value['parent_shared_member']==carrier['shared_member'] and
        value['native_packet']==shared['native_packet'] and value['selector']==shared['selector'] and
        value['natural_query_ids']==list(range(100)) and value['native_export_query_ids']==row['qids'] and value['center_column']==4 and
        value['neighbor_order']==['NW','N','NE','W','C','E','SW','S','SE'] and
        value['combined_order']=='old shared union order then sorted extra union-minus-centers','Exact true-F9 input provenance/ABI, never extra8')
    _ref(value['native_packet']);U=len(union);E=value['extra_count']
    need(type(E) is int and type(value['original_unique_count']) is int and value['original_unique_count']==U and
        0<=E<=min(8*U,H*W-U),'Exact extra-union count bound')
    extra=value['extra_flat_indices'];mapping=value['neighbor_to_combined']
    _tensor(torch,extra,torch.int64,(E,));_tensor(torch,value['extra_F'],torch.float32,(E,256));_tensor(torch,mapping,torch.int64,(U,9))
    need(bool(((extra>=0)&(extra<H*W)).all()) and bool((extra[1:]>extra[:-1]).all()) and
        not bool(np.intersect1d(union.numpy(),extra.numpy()).size) and bool(((mapping>=0)&(mapping<U+E)).all()) and
        torch.equal(mapping[:,4],torch.arange(U,dtype=torch.int64)),'Unique disjoint extra points and exact original-center map')
    y,x=union.numpy()//W,union.numpy()%W
    coordinates=np.stack([np.clip(y+dy,0,H-1)*W+np.clip(x+dx,0,W-1) for dy in (-1,0,1) for dx in (-1,0,1)],axis=1)
    combined=np.concatenate((union.numpy(),extra.numpy()))
    need(np.array_equal(combined[mapping.numpy()],coordinates) and np.array_equal(np.setdiff1d(np.unique(coordinates),union.numpy()),extra.numpy()) and
        value['original_union_flat_sha256']==hashlib.sha256(union.numpy().tobytes(order='C')).hexdigest(),'Independent original-grid replicate equation/extra difference/old union digest')
    return dict(union_flat_indices=union,F_unique=shared['F_unique'],extra_flat_indices=extra,
                extra_F=value['extra_F'],neighbor_to_combined=mapping)

class ImageDecoder:
    def __init__(self,torch,np,shared,bundle,row,binding,carrier,native):
        self.torch,self.np,self.shared,self.bundle,self.row=torch,np,shared,bundle,row;self.used=set()
        _ref(row['bundle']);_ref(carrier['bundle'])
        need(type(carrier) is dict and set(carrier)=={'bundle','shared_member','original_GPU_binding','common_PR_bundle','common_PR_original_GPU_binding'} and type(carrier['shared_member']) is str and
            carrier['shared_member']=='i%03d/SHARED_ACTUAL_INPUTS.pth'%row['dataset_index'] and type(carrier['original_GPU_binding']) is dict and carrier['original_GPU_binding'] and
            carrier['bundle']!=row['bundle'] and type(carrier['common_PR_original_GPU_binding']) is dict and carrier['common_PR_original_GPU_binding'] and
            carrier['common_PR_bundle']!=row['bundle'] and carrier['common_PR_bundle']!=carrier['bundle'],'Original physical carriers exact; no shared copy or self-ref')
        _ref(carrier['common_PR_bundle'])
        need(type(bundle) is dict and set(bundle)==BUNDLE_KEYS and bundle['schema']=='N20-actual-GPU-feature-neighbors-selector-image-bundle-v1' and
            type(binding) is dict and binding and bundle['binding']==binding and bundle['input_carrier']==row['input_carrier']==carrier and
            type(row['dataset_index']) is int and 0<=row['dataset_index']<187 and type(row['image_id']) is int and
            bundle['dataset_index']==row['dataset_index'] and bundle['image_id']==row['image_id'],'New physical nine-key bundle/immutable bindings/identity exact')
        self.index=row['dataset_index'];self.H,self.W=shared['native_L'].shape[1:]
        need(shared['schema']=='N16-actual-GPU-image-shared-inputs-v1' and shared['dataset_index']==self.index and shared['image_id']==row['image_id'] and
            shared['identity']==row['identity'] and list(shared['identity']['original_shape'])==[self.H,self.W] and shared['natural_query_ids']==list(range(100)) and
            shared['native_export_query_ids']==row['qids'] and type(row['qids']) is list and sorted(row['qids'])==list(range(100)) and
            shared['actual_GPU_F_captured'] is True and shared['CPU_F_rerendered'] is False and row['natural_queries']==100,'Original qualified same-image natural100 carrier/order')
        _tensor(torch,shared['native_L'],torch.float32,(100,self.H,self.W));_tensor(torch,shared['e0'],torch.float32,(100,256))
        self.union=shared['union_flat_indices'];_tensor(torch,self.union,torch.int64,(len(self.union),));_tensor(torch,shared['F_unique'],torch.float32,(len(self.union),256))
        need(shared['U']==len(self.union) and self.H>0 and self.W>0 and (len(self.union)==0 or bool(((self.union>=0)&(self.union<self.H*self.W)).all()) and
            bool((self.union[1:]>self.union[:-1]).all())) and len(shared['C_indices'])==len(shared['K_counts'])==len(shared['C_counts'])==100,'Exact original union/C/K shape mapping')
        need(type(native) is list and len(native)==100 and [p['query_id'] for p in native]==row['qids'] and all(p['image_id']==row['image_id'] for p in native),
            'Caller supplies bound original native predictions in original export order')
        self.native={p['query_id']:{k:v for k,v in p.items() if k!='segmentation'} for p in native}
        members=bundle['members'];registry=bundle['member_registry']
        need(type(members) is dict and type(registry) is dict and type(row['member_locators']) is list and list(members)==list(registry)==row['member_locators'] and
            type(bundle['member_count']) is int and bundle['member_count']==len(members),'Strict ordered member registry/manifest complete')
        input_id='i%03d/FEATURE_NEIGHBORS_INPUTS.pth'%self.index
        need(row['neighbor_input_member']==input_id and input_id in members,'One actual registered F9 input member')
        self.feature_context=_feature_input(torch,np,members[input_id],shared,row,carrier,self.union,self.H,self.W)
        self.used.add(input_id);total=0
        for name,value in members.items():
            need(type(name) is str and len(name.encode('utf8'))<=256 and type(value) is dict,'Only actual registered member IDs')
            schema=value.get('schema')
            if name==input_id:need(schema=='N20-DEV-true-feature-neighbor-extra-capture-body-v1','Actual input member')
            elif schema=='N20-actual-GPU-feature-neighbors-selector-point-chunk-v1':
                scout=value['stage']=='scout_C_logits';common=value['stage']=='full_C_PR'
                extra={'actual_GPU_logits','neighbor_input_member','actual_observed_neighbors_F_bitexact_to_input_carrier','actual_GPU_logits_alias_byte_exact'} if scout else {'GPU_raw_residual','GPU_combined'} | (COMMON_KEYS if common else set())
                need(set(value)==POINT_BASE|extra and value['point_chunk']==2048 and value['actual_device_type']=='cuda' and value['synthetic_only'] is False and
                    value['actual_input_scope']=='actual_GPU_old_union_plus_new_extra_feature_neighbors' and value['actual_Source_REF_qualified_here'] is False and
                    value['actual_observed_F_e_L_bitexact_to_input_carrier'] is True and name==_point_id(self.index,value),'Exact actual point roles/input-byte proof')
                n=len(value['flat_indices']);_tensor(torch,value['flat_indices'],torch.int64,(n,));need(0<n<=2048,'No empty fabricated point member')
                if scout:
                    _tensor(torch,value['actual_GPU_logits'],torch.float32,(n,3))
                    need(value['neighbor_input_member']==input_id and value['actual_observed_neighbors_F_bitexact_to_input_carrier'] is True and
                        value['actual_GPU_logits_alias_byte_exact'] is True,'Actual observed true F9/center inputs and actual logits alias proof')
                else:
                    for key in ('GPU_raw_residual','GPU_combined'):_tensor(torch,value[key],torch.float32,(n,))
                    if common:need(value['reused_qualified_common_PR'] is True and value['fresh_PR_forward_executed'] is False and
                        value['result_origin']=='qualified_prior_GPU_common_C_capture' and value['original_common_bundle']==carrier['common_PR_bundle'] and
                        value['original_common_member']==_point_id(self.index,value) and value['original_common_GPU_binding']==carrier['common_PR_original_GPU_binding'],
                        'Original N17 common-C actual output lineage; never label it fresh PR')
            elif schema=='N20-actual-GPU-logits-canonical-FP64-route-scores-v1':
                need(set(value)==SCORE_KEYS and type(value['query_id']) is int and value['query_id'] in range(100) and value['epoch'] in EPOCHS and value['arm'] in ARMS and
                    name==_score_id(self.index,value['query_id'],value['epoch'],value['arm']) and type(value['K']) is int and value['actual_GPU_logits_used'] is (value['K']>0),'Fixed actual-GPU-logit FP64 score member; K0 has no scout call')
                n=len(value['flat_indices']);_tensor(torch,value['flat_indices'],torch.int64,(n,));_tensor(torch,value['canonical_CPU_FP64_C_scores'],torch.float64,(n,))
            elif schema=='N20-actual-GPU-feature-neighbors-selector-query-profile-v1':
                need(set(value)==PROFILE_KEYS and type(value['query_id']) is int and 0<=value['query_id']<100 and value['epoch'] in EPOCHS and value['arm'] in ARMS and value['mode'] in MODES and
                    value['neighbor_input_member']==input_id and name==_profile_id(self.index,value['query_id'],value['epoch'],value['arm'],value['mode']),'Exact profile roles/F9 input')
                self._mask(value);need(value['original_prediction_nonsegmentation_fields']==self.native[value['query_id']],'Original class/S0/query order unchanged')
            else:raise ValueError('Unknown schema/extra8/shared-copy/orphan input forbidden')
            need(registry[name]==_registry(torch,value),'Exact registry role/query/epoch/chunk/actual raw bytes');total+=_bytes(torch,value)
        need(type(bundle['raw_tensor_payload_bytes']) is int and bundle['raw_tensor_payload_bytes']==total,'Exact complete physical payload bytes')
        roster=row['query_profiles'];need(type(roster) is list and len(roster)==100 and [v['query_id'] for v in roster]==list(range(100)),'Full natural0..99 roster')
        expected_order=[input_id];self.profile_ids={};self.score_ids={}
        for q,query in enumerate(roster):
            C=shared['C_indices'][q];_tensor(torch,C,torch.int64,(shared['C_counts'][q],));K=shared['K_counts'][q]
            need(type(K) is int and 0<=K<=len(C) and bool((C[1:]>C[:-1]).all()) and query['K']==K and query['C_count']==len(C) and
                type(query['profiles']) is dict and tuple(query['profiles'])==POLICIES,'Exact original per-query C/K/policy order')
            active=C.numpy() if K else np.empty(0,dtype=np.int64);ids=query['profiles'];common_ids=None
            for epoch in EPOCHS:
                for arm in ARMS:
                    pair=[]
                    for mode in MODES:
                        name=ids['E%d_%s_%s'%(epoch,arm,mode)];need(name in members and name==_profile_id(self.index,q,epoch,arm,mode),'Exact profile ID')
                        value=members[name];need(value['K']==K,'Original physical budget');self.profile_ids[(q,epoch,arm,mode)]=name;pair.append(value)
                    if common_ids is None:common_ids=pair[0]['common_C_PR_chunks'];expected_order.extend(common_ids)
                    score_id=_score_id(self.index,q,epoch,arm);self.score_ids[(q,epoch,arm)]=score_id
                    need(pair[0]['common_C_PR_chunks']==pair[1]['common_C_PR_chunks']==common_ids and
                        pair[0]['C_scout_chunks']==pair[1]['C_scout_chunks'] and pair[0]['route_scores_member']==pair[1]['route_scores_member']==score_id and score_id in members,
                        'One common/scout/score member shared between the same model modes')
                    score=members[score_id];need(score['query_id']==q and score['epoch']==epoch and score['arm']==arm and score['K']==K and score['C_count']==len(C) and
                        score['C_scout_chunks']==pair[0]['C_scout_chunks'] and np.array_equal(score['flat_indices'].numpy(),active),'Saved score exact full active C and actual scout source')
                    self._support(common_ids,q,None,'COMMON_C_REFERENCE','shared','full_C_PR',active)
                    self._support(pair[0]['C_scout_chunks'],q,epoch,arm,'shared-atmost-exact','scout_C_logits',active);expected_order.extend(pair[0]['C_scout_chunks'])
                    for mode,value in zip(MODES,pair):
                        selected=value['actual_selected_flat_indices'];_tensor(torch,selected,torch.int64,(len(selected),))
                        need(len(selected)<=K and (mode!='exact' or len(selected)==K) and bool((selected[1:]>selected[:-1]).all()),'Ascending real selected K support')
                        pos=np.searchsorted(C.numpy(),selected.numpy());need(len(selected)==0 or (bool((pos<len(C)).all()) and np.array_equal(C.numpy()[pos],selected.numpy())),'Actual selected inside C')
                        self._support(value['selected_PR_chunks'],q,epoch,arm,mode,'selected_PR',selected.numpy());expected_order.extend(value['selected_PR_chunks'])
            for epoch in EPOCHS:
                for arm in ARMS:
                    expected_order.append(self.score_ids[(q,epoch,arm)])
                    expected_order.extend(self.profile_ids[(q,epoch,arm,mode)] for mode in MODES)
        need(expected_order==list(members) and len(set(expected_order))==len(expected_order),'Canonical complete input/point/score/profile insertion order, no missing/orphan/duplicate member')
    def _mask(self,value):
        torch,np=self.torch,self.np;N=self.H*self.W;packed=value['actual_mask_packbits'];_tensor(torch,packed,torch.uint8,((N+7)//8,))
        need(value['actual_mask_shape']==[self.H,self.W] and value['actual_mask_bitorder']=='little' and type(value['actual_RLE']) is dict,'Exact real mask shape/bitorder/RLE')
        bits=np.unpackbits(packed.numpy(),bitorder='little');need(not bool(bits[N:].any()),'Packed mask padding must be zero')
        return torch.from_numpy(bits[:N].astype(np.bool_).reshape(self.H,self.W))
    def _support(self,names,q,epoch,arm,mode,stage,indices):
        need(type(names) is list and len(names)==(len(indices)+2047)//2048,'Complete canonical chunk list')
        for i,name in enumerate(names):
            need(type(name) is str and name in self.bundle['members'],'Internal member ID string only')
            value=self.bundle['members'][name];need(value['schema']=='N20-actual-GPU-feature-neighbors-selector-point-chunk-v1' and value['query_id']==q and value['scout_epoch']==epoch and
                value['control']==arm and value['evalmode']==mode and value['stage']==stage and value['chunk_first']==i*2048 and
                self.np.array_equal(value['flat_indices'].numpy(),indices[i*2048:(i+1)*2048]),'Chunk exact query/role/epoch/support/order')
    def _point(self,name):
        torch,np=self.torch,self.np;value=self.bundle['members'][name];self.used.add(name);flat=value['flat_indices'];q=value['query_id']
        positions=np.searchsorted(self.union.numpy(),flat.numpy());need(bool((positions<len(self.union)).all()) and np.array_equal(self.union.numpy()[positions],flat.numpy()),'Exact old shared F coverage')
        F=self.shared['F_unique'][torch.from_numpy(positions.astype(np.int64))];e=self.shared['e0'][q];L=self.shared['native_L'][q].reshape(-1)[flat]
        if value['stage']=='scout_C_logits':
            if value['control']=='CENTER_ERROR':neighbors=F[:,None,:].repeat(1,9,1)
            else:
                mapped=self.feature_context['neighbor_to_combined'][torch.from_numpy(positions.astype(np.int64))].reshape(-1);old=mapped<len(self.union)
                neighbors=torch.empty((len(mapped),256),dtype=torch.float32)
                neighbors[old]=self.shared['F_unique'][mapped[old]];neighbors[~old]=self.feature_context['extra_F'][mapped[~old]-len(self.union)]
                neighbors=neighbors.reshape(len(flat),9,256)
            return flat,F,e,L,neighbors,value['actual_GPU_logits']
        return dict(flat_indices=flat,sampled_F=F,e0=e,native_L=L,GPU_raw_residual=value['GPU_raw_residual'],GPU_combined=value['GPU_combined'])
    def query(self,q):
        need(type(q) is int and 0<=q<100,'Natural query only');common=self.bundle['members'][self.profile_ids[(q,24,ARMS[0],'atmost')]]['common_C_PR_chunks']
        def scout(epoch,arm):
            need(epoch in EPOCHS and arm in ARMS,'Qualified fixed model only');profile=self.bundle['members'][self.profile_ids[(q,epoch,arm,'atmost')]]
            for name in profile['C_scout_chunks']:yield self._point(name)
        def profile(epoch,arm,mode):
            need(epoch in EPOCHS and arm in ARMS and mode in MODES,'Fixed policy only');name=self.profile_ids[(q,epoch,arm,mode)];self.used.add(name);value=self.bundle['members'][name]
            return value['actual_selected_flat_indices'],[self._point(member) for member in value['selected_PR_chunks']],self._mask(value),value['actual_RLE']
        def scores(epoch,arm):
            need(epoch in EPOCHS and arm in ARMS,'Fixed score model only');name=self.score_ids[(q,epoch,arm)];self.used.add(name)
            return self.bundle['members'][name]['canonical_CPU_FP64_C_scores']
        return dict(feature_context=self.feature_context,read_scores=scores,query_id=q,native_L_grid=self.shared['native_L'],e0=self.shared['e0'][q],C_flat_indices=self.shared['C_indices'][q],K=self.shared['K_counts'][q],
            common_PR_chunks=[self._point(name) for name in common],iter_scout_chunks=scout,read_profile=profile)
    def predictions(self,policy):
        need(policy in POLICIES,'Only fixed new policy export');values=[]
        for q in self.row['qids']:
            name=self.row['query_profiles'][q]['profiles'][policy];self.used.add(name);value=self.bundle['members'][name]
            values.append(dict(value['original_prediction_nonsegmentation_fields'],segmentation=value['actual_RLE']))
        return values
    def require_all_consumed(self):
        need(self.used==set(self.bundle['members'])==set(self.bundle['member_registry']),'Every registered actual member must be decoded, not merely inspected')
        return dict(all_registered_members_consumed=True,members=len(self.used),queries=100,ROOT_scientific_qualification_conferred=False)

def decode_image(torch,np,old_shared,new_bundle,row,expected_binding,expected_input_carrier,native_predictions):
    return ImageDecoder(torch,np,old_shared,new_bundle,row,expected_binding,expected_input_carrier,native_predictions)
