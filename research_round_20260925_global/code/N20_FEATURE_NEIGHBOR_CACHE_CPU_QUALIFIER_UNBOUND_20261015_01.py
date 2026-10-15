# UNBOUND_DRAFT: independent CPU numeric/index body; ROOT alone qualifies sources/terminals.
"""No IO, coordinate-module import, renderer, teacher, GT, PR, model or RNG use.
ROOT authenticates physical refs/source/spec/admission and actual GPU provenance
before calling. This body verifies the known capture schema and supplied arrays;
it cannot independently establish actual extra-F values without a renderer.
"""
import hashlib


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _tensor(torch, value, dtype, shape, name):
    _need(isinstance(value,torch.Tensor) and value.layout==torch.strided and
          value.device.type=='cpu' and value.dtype==dtype and tuple(value.shape)==tuple(shape)
          and value.is_contiguous() and not value.requires_grad, name+': detached contiguous CPU dtype/shape')
    _need(bool(torch.isfinite(value).all()),name+': finite values')


def _same_bytes(torch, left, right):
    return left.dtype==right.dtype and left.shape==right.shape and torch.equal(
        left.contiguous().view(torch.uint8),right.contiguous().view(torch.uint8))


def _digest(value):
    array=value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape),dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def qualify_image(torch,np,old_packet,new_packet,old_row,new_row):
    """Check independently supplied old/new arrays; return small proof, never Q.

    Generic sample count permits hand fixtures. ROOT separately enforces actual
    FIT741/4096, closed refs, full Source/RAW ownership and GPU capture proofs.
    """
    _need(all(type(v) is dict for v in (old_packet,new_packet,old_row,new_row)),'Explicit decoded packet/row dictionaries')
    _need(new_packet['schema']=='N20-true-F-neighbor-capture-body-v1','Known numeric capture schema')
    identity=old_packet['identity'];index=old_packet['dataset_index']
    _need(type(index) is int and index>=0 and all(v['dataset_index']==index for v in (new_packet,old_row,new_row))
          and type(identity) is dict and type(identity['image_id']) is int and
          all(v['identity']==identity for v in (new_packet,old_row,new_row)) and
          old_row['image_id']==new_row['image_id']==identity['image_id'],'Exact old/new row, image and full identity association')
    shape=identity['original_shape']
    _need(type(shape) in (list,tuple) and len(shape)==2 and all(type(x) is int and x>0 for x in shape),'Positive original spatial shape')
    H,W=shape;pixels=H*W
    samples=old_packet['samples'];unique=old_packet['unique_flat_indices'];old_F=old_packet['sampled_F_unique']
    inverse=old_packet['sample_to_unique'];flats=samples['flat_indices'];queries=samples['query_ids']
    _need(isinstance(unique,torch.Tensor) and unique.ndim==1 and isinstance(flats,torch.Tensor) and flats.ndim==1,'Original one-dimensional coordinates')
    U,N=len(unique),len(flats)
    for value,dtype,expected,name in ((unique,torch.int64,(U,),'unique'),(inverse,torch.int64,(N,),'inverse'),
            (flats,torch.int64,(N,),'sample flats'),(queries,torch.int64,(N,),'sample queries'),(old_F,torch.float32,(U,256),'old F')):
        _tensor(torch,value,dtype,expected,name)
    _need(bool(((flats>=0)&(flats<pixels)).all()) and bool(((queries>=0)&(queries<100)).all()),'Original sample coordinates/query IDs within physical domain')
    expected_unique,expected_inverse=np.unique(flats.numpy(),return_inverse=True)
    _need(np.array_equal(unique.numpy(),expected_unique) and np.array_equal(inverse.numpy(),expected_inverse),
          'Original sorted unique/inverse must exactly recover duplicate slot order')
    _need(old_packet['natural_query_ids']==list(range(100)) and old_row['natural_queries']==new_row['natural_queries']==100
          and type(old_row['points']) is int and old_row['points']==new_row['points']==new_packet['sampled_points']==N
          and old_row['unique_sampled_F_points']==new_row['unique_centers']==new_row['original_unique_count']==new_packet['original_unique_count']==U,
          'Original natural100/sample/unique counts')
    _need(type(old_packet['matching_loss_seed']) is int and
          old_packet['matching_loss_seed']==old_row['matching_loss_seed']==new_row['matching_loss_seed']==new_packet['matching_loss_seed']
          and type(samples['seed']) is int and samples['seed']==old_row['point_seed']==new_row['point_seed']==new_packet['point_seed']
          and samples['replacement'] is True and samples['duplicates_preserved'] is True,'Original matching and point seeds/replacement preserved')
    teacher=old_packet['teacher_checkpoint']
    _need(type(teacher) is dict and teacher and new_packet['teacher_checkpoint']==teacher,'Original teacher reference metadata exact; no checkpoint load')
    digests={name:_digest(value) for name,value in (('unique_flat_indices',unique),('sample_to_unique',inverse),
             ('sample_flat_indices',flats),('sample_query_ids',queries))}
    _need(new_packet['input_index_digests']==digests,'Independent original index shape/dtype/raw-byte digests')
    _need(type(new_packet['center_column']) is int and new_packet['center_column']==4 and
          new_packet['neighbor_order']==['NW','N','NE','W','C','E','SW','S','SE'] and
          new_packet['combined_order']=='old unique order then sorted extra union-minus-centers','Fixed nine-neighbor/combined ordering')
    extra=new_packet['extra_flat_indices'];extra_F=new_packet['extra_F'];mapping=new_packet['neighbor_to_combined']
    _need(isinstance(extra,torch.Tensor) and extra.ndim==1,'Extra coordinate vector')
    E=len(extra)
    _tensor(torch,extra,torch.int64,(E,),'extra flats');_tensor(torch,extra_F,torch.float32,(E,256),'extra F')
    _tensor(torch,mapping,torch.int64,(U,9),'neighbor mapping')
    _need(type(new_packet['extra_count']) is int and new_packet['extra_count']==new_row['extra_count']==E and
          bool(((mapping>=0)&(mapping<U+E)).all()),'Extra count and strict combined bounds')
    # Independent scalar original-grid coordinates, never GPU build_coordinates.
    expected_coordinates=np.empty((U,9),dtype=np.int64)
    for row,flat in enumerate(unique.numpy().tolist()):
        cy,cx=divmod(int(flat),W);column=0
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                yy=min(max(cy+dy,0),H-1);xx=min(max(cx+dx,0),W-1)
                expected_coordinates[row,column]=yy*W+xx;column+=1
    expected_extra=np.setdiff1d(np.unique(expected_coordinates),unique.numpy())
    _need(np.array_equal(extra.numpy(),expected_extra),'Exactly sorted distinct neighbor union minus old centers')
    combined_indices=np.concatenate((unique.numpy(),extra.numpy()))
    lookup={int(flat):i for i,flat in enumerate(combined_indices.tolist())}
    expected_map=np.array([[lookup[int(flat)] for flat in row] for row in expected_coordinates],dtype=np.int64).reshape(U,9)
    _need(np.array_equal(mapping.numpy()[:,4],np.arange(U,dtype=np.int64)),'Center column4 must identify each original row')
    _need(np.array_equal(mapping.numpy(),expected_map) and np.array_equal(combined_indices[mapping.numpy()],expected_coordinates),
          'Exact original-grid replicate mapping, no cross-row wrap')
    _need(np.array_equal(unique.numpy()[inverse.numpy()],flats.numpy()) and
          np.array_equal(combined_indices[mapping.numpy()][inverse.numpy()],expected_coordinates[expected_inverse]),
          'All nine sample neighbors follow the original repeated/ordered slots exactly')
    combined_F=torch.cat((old_F,extra_F),dim=0)
    _need(_same_bytes(torch,combined_F[mapping[:,4]],old_F) and
          _same_bytes(torch,combined_F[mapping[inverse,4]],old_F[inverse]),'Original unique and repeated sample center feature bytes preserved')
    payload_bytes=E*256*4+E*8+U*9*8
    _need(new_row['extra_feature_bytes']==E*256*4 and new_row['raw_tensor_payload_bytes']==payload_bytes,'Exact numerical tensor payload accounting')
    return dict(dataset_index=index,image_id=identity['image_id'],original_unique_count=U,extra_count=E,sampled_points=N,
                neighbor_coordinates_exact=True,extra_union_exact=True,center_features_byte_exact=True,
                original_sample_order_exact=True,input_index_digests_exact=True,identity_teacher_seeds_exact=True,
                raw_tensor_payload_bytes=payload_bytes,qualified=False,ROOT_external_source_and_terminal_checks_required=True,
                extra_feature_values_independently_rerendered=False,models_loaded=False,
                GT_files_opened=False,GT_used_for_coordinates=False)
