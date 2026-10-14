"""UNBOUND_DRAFT: CPU FIT-only thin native-L neighbor cache extraction body.
ROOT supplies qualified input bindings and all closed/read/write/deadline callbacks.
No direct file loading, teacher/PR, new GT files/matching, DEV, training or owner gate.
Original packets contain GT-derived training labels and metadata.
"""
import hashlib
from pathlib import Path
FIT_IMAGES=741
NATURAL_QUERIES=100
SAMPLES_PER_IMAGE=4096
EXPECTED_PAYLOAD_BYTES=97124352
OFFSETS=((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
CALLBACKS=('closed','read_torch','write_npy','read_generated_npy','write_manifest','check')

def _need(ok,message):
    if not ok:raise ValueError(message)
def _ref(value):
    _need(type(value) is dict and set(value)=={'path','bytes','sha256'} and type(value['path']) is str and
        Path(value['path']).is_absolute() and type(value['bytes']) is int and value['bytes']>=0 and
        type(value['sha256']) is str and len(value['sha256'])==64 and set(value['sha256'])<=set('0123456789abcdef'),
        'Strict absolute3field reference required')
def _callbacks(callbacks):
    _need(type(callbacks) is dict and all(callable(callbacks.get(name)) for name in CALLBACKS),'ROOT must provide all six named IO/check callbacks')
def _read(ref,callbacks):
    _ref(ref);callbacks['check']();path=callbacks['closed'](ref)
    _need(isinstance(path,Path) and path.is_absolute(),'closed(ref) must return absolute Path for read_torch, never module load')
    return callbacks['read_torch'](path)
def _tensor(torch,value,shape,dtype):
    _need(isinstance(value,torch.Tensor) and value.layout==torch.strided and value.device.type=='cpu' and
        value.dtype==dtype and tuple(value.shape)==shape and not value.requires_grad and bool(torch.isfinite(value).all()),
        'Exact finite CPU tensor shape/dtype required')
def _bits_equal(torch,a,b):
    return torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))
def _numpy_bits_equal(np,a,b):
    return (isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype==np.float32 and a.shape==b.shape and
        np.array_equal(np.ascontiguousarray(a).view(np.uint32),np.ascontiguousarray(b).view(np.uint32)))

def extract_image(torch,np,kernel,index,row,packet_ref,native_L_ref,cache_report,expected_bindings,callbacks,expected_samples=4096):
    _callbacks(callbacks);_need(type(index) is int and index>=0 and type(expected_samples) is int and expected_samples>0,'Actual image/sample count required')
    _need(tuple(kernel.OFFSETS)==OFFSETS,'Frozen original-native eight-neighbor order required')
    _need(type(row) is dict and row['dataset_index']==index and type(row['image_id']) is int,'Original FIT row/index/image identity required')
    _need(row['packet']==packet_ref and row['natural_native_L_carrier']==native_L_ref,
        'Complete original row packet/native carrier reference association required')
    _ref(expected_bindings['locked_protocol'])
    packet=_read(packet_ref,callbacks);carrier=_read(native_L_ref,callbacks);callbacks['check']()
    _need(type(packet) is dict and type(carrier) is dict and carrier['schema']=='N16-actual-GPU-natural100-L-carrier-v1' and
        packet['natural_native_L_carrier']==native_L_ref,'Only exact closed original packet/native carrier pair allowed')
    for key in ('source','spec','GPU_admission','phase'):
        _need(packet[key]==carrier[key]==cache_report[key],'Original cache payload binding differs: '+key)
    _need(packet['dataset_index']==carrier['dataset_index']==index and packet['identity']==carrier['identity']==row['identity'],
        'Original row/packet/nativeL index and full identity must be exact')
    identity=packet['identity'];shape=identity['original_shape']
    _need(identity['dataset_index']==index and identity['image_id']==row['image_id'] and
        type(shape) in (list,tuple) and len(shape)==2 and all(type(v) is int and v>0 for v in shape),
        'True original native shape/image/query namespace required')
    _need(packet['protocol']==expected_bindings['locked_protocol'],'Original packet.protocol must equal caller-closed old locked protocol')
    for key in ('fixed_PR_checkpoint','teacher_checkpoint'):
        _need(packet[key]==cache_report[key],'Original cache checkpoint metadata cannot change: '+key)
    L=carrier['native_L'];H,W=shape;_tensor(torch,L,(100,H,W),torch.float32)
    sample=packet['samples'];q=sample['query_ids'];flat=sample['flat_indices'];original_L=sample['native_L']
    _tensor(torch,q,(expected_samples,),torch.int64);_tensor(torch,flat,(expected_samples,),torch.int64)
    _tensor(torch,original_L,(expected_samples,),torch.float32)
    _need(not bool(((q<0)|(q>=100)).any()) and not bool(((flat<0)|(flat>=H*W)).any()),'Original query/point outside native grid')
    before_q=q.clone();before_flat=flat.clone();before_L=original_L.clone()
    centers=L.flatten(1)[q,flat]
    _need(_bits_equal(torch,centers,original_L),'Actual same-query native center must be byte exact to old samples.native_L')
    callbacks['check']();features=kernel.neighbor_features(torch,L,q,flat);_tensor(torch,features,(expected_samples,8),torch.float32)
    expected=kernel.neighbor_features_numpy(np,L.numpy(),q.numpy(),flat.numpy())
    _need(_numpy_bits_equal(np,features.numpy(),expected),'Independent NumPy original-coordinate features must be FP32 byte exact for every sample')
    control=kernel.center_control_features(torch,L,q,flat)
    expected_control=kernel.center_control_features_numpy(np,L.numpy(),q.numpy(),flat.numpy())
    old_center=(original_L.clamp(-8,8)/8).unsqueeze(1).repeat(1,8)
    _need(_bits_equal(torch,control,old_center) and _numpy_bits_equal(np,control.numpy(),expected_control),
        'Eight real center copies must be exact to original sampled nativeL')
    callbacks['check']();array=np.ascontiguousarray(features.numpy());output_ref=callbacks['write_npy'](index,array);_ref(output_ref)
    reread=callbacks['read_generated_npy'](output_ref)
    _need(_numpy_bits_equal(np,reread,expected),'Actual NPY roundtrip dtype/shape/bits/sample order differs')
    _need(torch.equal(q,before_q) and torch.equal(flat,before_flat) and _bits_equal(torch,original_L,before_L),
        'Original point/query order and sampled values must not mutate')
    payload=expected_samples*8*4
    _need(array.nbytes==payload and output_ref['bytes']>payload,'Real NPY must contain the exact FP32 payload plus header')
    point_digest=hashlib.sha256(before_q.numpy().tobytes()+before_flat.numpy().tobytes()).hexdigest();callbacks['check']()
    return dict(dataset_index=index,image_id=row['image_id'],original_shape=list(shape),queries=100,points=expected_samples,
        packet_ref=packet_ref,native_L_ref=native_L_ref,neighbors8_ref=output_ref,point_order_sha256=point_digest,
        payload_bytes=payload,serialized_bytes=output_ref['bytes'],header_bytes=output_ref['bytes']-payload,
        same_query_native_center_byte_exact=True,all_samples_NumPy_neighbors_byte_exact=True,center_control_byte_exact=True,
        original_labels_weights_points_preserved_by_packet_ref=True,center_control_stored=False)

def extract_fit(torch,np,kernel,cache_report,cache_report_ref,expected_bindings,callbacks,output_binding=None):
    _callbacks(callbacks);_ref(cache_report_ref);callbacks['check']();callbacks['closed'](cache_report_ref)
    rows=cache_report['rows'];packet_refs=cache_report['packet_refs'];native_L_refs=cache_report['native_L_refs']
    _need(type(rows) is list and type(packet_refs) is list and type(native_L_refs) is list and
        len(rows)==len(packet_refs)==len(native_L_refs)==741 and [r['dataset_index'] for r in rows]==list(range(741)) and
        len({r['image_id'] for r in rows})==741,'Exactly complete original ordered FIT741 required, no DEV or partial scope')
    for refs in (packet_refs,native_L_refs):
        for ref in refs:_ref(ref)
        _need(len({r['path'] for r in refs})==741,'Each original image has one distinct real packet/nativeL reference')
    extracted=[]
    for index,row in enumerate(rows):
        extracted.append(extract_image(torch,np,kernel,index,row,packet_refs[index],native_L_refs[index],cache_report,expected_bindings,callbacks,4096))
    payload=sum(row['payload_bytes'] for row in extracted);serialized=sum(row['serialized_bytes'] for row in extracted)
    _need(payload==EXPECTED_PAYLOAD_BYTES and sum(r['points'] for r in extracted)==3035136,'Full741×4096×8 FP32 payload bound exact')
    summary=dict(images=741,queries=74100,points=3035136,payload_bytes=payload,serialized_array_bytes=serialized,
        array_header_bytes=serialized-payload,feature_files=741,qualified=False,teacher_loaded=False,PR_loaded=False,
        GT_files_opened=False,GT_used_to_choose_neighbors=False,
        DEV_generated=False,training_performed=False,old_packet_or_Source_mutated=False)
    manifest=dict(schema='N17-FIT-native-L-neighbors8-cache-manifest-v1',status='BODY_COMPLETE_PENDING_ROOT_ACTUAL0_AND_QUALIFICATION',
        output_binding={} if output_binding is None else output_binding,original_cache_report=cache_report_ref,
        original_source=cache_report['source'],original_spec=cache_report['spec'],original_GPU_admission=cache_report['GPU_admission'],
        original_phase=cache_report['phase'],original_locked_protocol=expected_bindings['locked_protocol'],
        fixed_PR_checkpoint=cache_report['fixed_PR_checkpoint'],teacher_checkpoint=cache_report['teacher_checkpoint'],
        offsets=[list(v) for v in OFFSETS],normalization='gather8 then clamp[-8,8]/8',border='same-query original native replicate',
        center_control='derive eight copies from original samples.native_L; not stored twice',rows=extracted,summary=summary)
    callbacks['check']();manifest_ref=callbacks['write_manifest'](manifest);_ref(manifest_ref);callbacks['check']()
    return manifest_ref,dict(summary,manifest_bytes=manifest_ref['bytes'])
