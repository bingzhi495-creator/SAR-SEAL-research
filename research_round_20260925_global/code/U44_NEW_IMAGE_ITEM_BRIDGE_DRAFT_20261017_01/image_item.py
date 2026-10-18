"""UNBOUND U41-to-U40 single-image bridge; numerical path has not been run.

Stdlib only on import. This file does not construct/restore a model, select a
population, read GT, change native scores, free the caller's teacher, or qualify
an execution. Torch, NumPy, PIL.Image and the original geometry are injected.
"""
import hashlib
import math
from pathlib import Path


METADATA_KEYS=frozenset(('identity','export_query_ids','native_predictions','natural_query_ids'))
IDENTITY_KEYS=frozenset(('image_id','image','original_shape','actual_img_shape','batch_input_shape',
    'pad_shape','scale_factor','native_input_gamma'))
CAPTURE_KEYS=METADATA_KEYS|frozenset(('F','e','M','C','L','S0_by_query','score_factors_by_query',
    'class_packet','category_ids_by_query','receipt'))
GEOMETRY_SHA256='65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7'
JPEG_LOGIC_PROVENANCE=dict(
    source='U37_FULL_RGB_MATCHED_READER_DRAFT_20261017_01/rgb_matched_reader.py:103-138',
    sha256='678ce3781872e21e961b24a7b0a4d0ecc69f0f8a08b8feb7eb0b2e8624d57025')


def need(ok,message):
    if not ok:raise ValueError(message)


def strict_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'} and type(value['path']) is str
         and Path(value['path']).is_absolute() and type(value['bytes']) is int and value['bytes']>=0
         and type(value['sha256']) is str and len(value['sha256'])==64
         and set(value['sha256'])<=set('0123456789abcdef'),'Explicit strict absolute ref required')
    return value


def prepare_metadata(capture_metadata,dataset_index,image_reference):
    """Public metadata contract only: preserve original objects and export order."""
    need(type(capture_metadata) is dict and set(capture_metadata)==METADATA_KEYS,'Only image/query/export metadata, no GT')
    need(type(dataset_index) is int and dataset_index>=0,'Caller-owned image ordinal, no fixed population')
    reference=strict_ref(image_reference);identity=capture_metadata['identity']
    need(type(identity) is dict and set(identity)==IDENTITY_KEYS and type(identity['image_id']) is int
         and identity['image_id']>=0 and strict_ref(identity['image'])==reference,'Exact original capture identity.image')
    for key in ('original_shape','actual_img_shape'):
        need(type(identity[key]) in (list,tuple) and len(identity[key])==2
             and all(type(v) is int and v>0 for v in identity[key]),'Original positive H/W: '+key)
    need(identity['batch_input_shape']==[800,800] and identity['pad_shape']==[800,800]
         and max(identity['actual_img_shape'])==800 and min(identity['actual_img_shape'])<=800,'Original native800 geometry')
    qids=capture_metadata['export_query_ids'];natural=capture_metadata['natural_query_ids']
    need(type(qids) is list and len(qids)==100 and all(type(q) is int for q in qids)
         and set(qids)==set(range(100)) and type(natural) is list and natural==list(range(100)),'Separate natural100 axis and export permutation')
    originals=capture_metadata['native_predictions'];need(type(originals) is list and len(originals)==100,'All100 original export dictionaries')
    for qid,value in zip(qids,originals):
        need(type(value) is dict and set(value)=={'image_id','query_id','category_id','score','segmentation'}
             and type(value['image_id']) is int and value['image_id']==identity['image_id']
             and type(value['query_id']) is int and value['query_id']==qid
             and type(value['category_id']) is int and value['category_id']==1
             and type(value['score']) in (int,float) and math.isfinite(value['score']) and 0<=value['score']<=1,
             'Exact source-export schema/query/category/S0; no GT or additional association fields')
        rle=value['segmentation'];need(type(rle) is dict and set(rle)=={'size','counts'}
            and rle['size']==list(identity['original_shape']) and type(rle['counts']) is str,'Original JSON RLE metadata, no decode here')
    return dict(row=dict(dataset_index=dataset_index,image_id=identity['image_id']),identity=identity,query_ids=qids,originals=originals)


def _source(module,reference,closed_ref,expected_sha=None):
    """Canonical refs are explicit; compare actual file and bytes separately."""
    strict_ref(reference);declared=Path(reference['path']);canonical=declared.resolve(strict=True)
    need(str(declared)==str(canonical),'Caller must provide the canonical source ref, not a junction alias')
    need(Path(closed_ref(reference)).resolve(strict=True)==canonical,'Source closure returned a different file')
    filename=getattr(module,'__file__',None)
    need(type(filename) is str and Path(filename).resolve(strict=True)==canonical,'Actually injected module source differs')
    raw=canonical.read_bytes()
    need(len(raw)==reference['bytes'] and hashlib.sha256(raw).hexdigest()==reference['sha256'],'Exact admitted source bytes')
    if expected_sha is not None:need(reference['sha256']==expected_sha,'Fixed original geometry source changed')


def _cpu_copy(torch,value,shape,name):
    need(isinstance(value,torch.Tensor) and tuple(value.shape)==tuple(shape) and value.dtype==torch.float32
         and not value.requires_grad and value.device.type in ('cpu','cuda')
         and (value.device.type!='cuda' or value.device.index==0) and bool(torch.isfinite(value).all()),
         'Original finite detached FP32 natural-axis '+name)
    # copy=True avoids retaining a view into the complete returned capture even
    # when the caller already moved that capture to CPU. Never reorders queries.
    result=value.detach().to(device='cpu',copy=True).contiguous()
    need(result.device.type=='cpu' and result.dtype==torch.float32 and not result.requires_grad,'Owned CPU copy '+name)
    return result


def make_image_item(torch,np,Image,capture,dataset_index,image_reference,*,image_geometry,
                    source_refs,closed_ref,check):
    """Future genuine numerical bridge -> {'item','row','originals'}.

    Required inputs are the unchanged U41 returned capture and explicit original
    image ref. Only e/M batch0 and natural-axis L are copied to CPU; no F/C or
    complete capture reference is returned. The caller still owns/releases all
    original capture/teacher aliases before placing either trained arm on GPU.
    No libraries are imported here, and no GT argument is accepted.
    """
    need(type(capture) is dict and set(capture)==CAPTURE_KEYS,'Exact U41 image-only capture; GT/extra fields forbidden')
    metadata=prepare_metadata({key:capture[key] for key in METADATA_KEYS},dataset_index,image_reference)
    receipt=capture['receipt']
    need(type(receipt) is dict and receipt['schema']=='U41-unbound-single-image-native-capture-v1'
         and receipt['e_is_mask_embed_output'] is True and receipt['original_fusion_masks_scores_labels_exact'] is True
         and receipt['natural_query_ids']==list(range(100)) and receipt['images']==1 and receipt['queries']==100
         and all(receipt[k] is False for k in ('annotations_opened','GT_forwarded','loss_or_matching_called',
             'optimizer_created','new_training','qualified','population_qualification','numerical_runtime_qualified')),
         'Actual capture receipt is required; it is not numerical or population qualification')
    need(capture['category_ids_by_query']==[1]*100 and type(capture['S0_by_query']) is list
         and len(capture['S0_by_query'])==100,'Original natural-axis category and S0 metadata')
    need(all(value['score']==capture['S0_by_query'][value['query_id']] for value in metadata['originals']),
         'Original native S0 must match the exact export query mapping')
    need(type(source_refs) is dict and set(source_refs)=={'image_geometry','PIL.Image'},'Two explicit canonical source refs')
    check();_source(image_geometry,source_refs['image_geometry'],closed_ref,GEOMETRY_SHA256)
    _source(Image,source_refs['PIL.Image'],closed_ref)
    identity=metadata['identity'];height,width=identity['original_shape']
    e_batch=_cpu_copy(torch,capture['e'],(1,100,256),'e mask_embed output')
    m_batch=_cpu_copy(torch,capture['M'],(1,100,200,200),'native M200')
    e=e_batch[0];M=m_batch[0];del e_batch,m_batch
    L=_cpu_copy(torch,capture['L'],(100,height,width),'original-grid L')
    check()
    geometry=image_geometry.make_geometry(torch,identity['original_shape'],identity['actual_img_shape'],800,400)
    path=Path(closed_ref(image_reference)).resolve(strict=True)
    need(path==Path(image_reference['path']).resolve(strict=True),'Exact original image closure')
    # Same JPEG/EXIF/RGB conversion as U37:103-138 and DEV bridge:167-195.
    # This is the new branch's RGB convention, never teacher normalization.
    with Image.open(path) as image:
        need(image.format=='JPEG' and image.size==(width,height),'Original JPEG and stored pixel dimensions')
        orientation=image.getexif().get(274)
        need(orientation is None or (type(orientation) is int and orientation==1),'Absent/identity EXIF only; never rotate')
        original_mode=image.mode;rgb=image.convert('RGB')
        try:pixels=np.array(rgb,dtype=np.uint8,copy=True)
        finally:rgb.close()
    need(pixels.dtype==np.uint8 and pixels.shape==(height,width,3),'Decoded original RGB uint8 grid')
    image_uint8=torch.from_numpy(pixels).permute(2,0,1).unsqueeze(0).contiguous()
    need(image_uint8.device.type=='cpu' and image_uint8.dtype==torch.uint8,'CPU RGB uint8 before original geometry')
    rgb400=geometry.prepare_rgb(image_uint8)
    need(tuple(rgb400.shape)==(1,3,400,400) and rgb400.device.type=='cpu' and rgb400.dtype==torch.float32
         and not rgb400.requires_grad and bool(torch.isfinite(rgb400).all())
         and bool(((rgb400>=0)&(rgb400<=1)).all()),'Original finite detached CPU RGB400/255 path')
    del pixels,image_uint8
    rgb_receipt=dict(schema='U44-new-image-item-bridge-pending-v1',image=image_reference,
        source_refs=source_refs,original_shape=list(identity['original_shape']),actual_img_shape=list(identity['actual_img_shape']),
        canvas_size=800,branch_size=400,original_mode=original_mode,decoded_mode='RGB',exif_orientation=orientation,
        auto_orientation=False,input_convention='RGB uint8 -> FP32 /255; original image_geometry',
        teacher_photometric_reconstruction=False,native_tensor_query_axis='natural query ids 0..99; not reordered',
        prediction_query_order='unchanged actual U41 export permutation',capture_reference_retained=False,
        F_or_C_retained=False,teacher_release_verified=False,GT_forwarded=False,qualified=False,
        numerical_runtime_qualified=False,population_qualification=False)
    item=dict(e=e,M=M,L=L,identity=metadata['identity'],query_ids=metadata['query_ids'],
        rgb400=rgb400,geometry=geometry,rgb_receipt=rgb_receipt)
    check();return dict(item=item,row=metadata['row'],originals=metadata['originals'])
