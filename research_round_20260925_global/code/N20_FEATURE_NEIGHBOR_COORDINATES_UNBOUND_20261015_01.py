"""Pure original-pixel 3x3 coordinate construction; no I/O or scientific run."""

OFFSETS=((-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1))


def _need(ok,message):
    if not ok:
        raise ValueError(message)


def build_coordinates(np,unique_flat_indices,original_shape):
    """Return extra sorted flats and Ux9 indices into [old centers, extras].

    Old centers must already be strictly ascending INT64; their order is never
    changed. Offsets are row-major dy/dx=-1,0,1, clamped independently in the
    original image frame. The center occupies slot4.
    """
    _need(isinstance(unique_flat_indices,np.ndarray) and
          unique_flat_indices.dtype==np.dtype(np.int64) and unique_flat_indices.ndim==1,
          'unique_flat_indices must be a one-dimensional INT64 NumPy array')
    _need(type(original_shape) in (tuple,list) and len(original_shape)==2,
          'original_shape must contain exactly H and W')
    _need(all(isinstance(v,(int,np.integer)) and not isinstance(v,(bool,np.bool_))
              and int(v)>0 for v in original_shape),
          'H and W must be positive nonboolean integers')
    H,W=map(int,original_shape)
    _need(H*W<=np.iinfo(np.int64).max,'Original flat coordinate frame must fit INT64')
    centers=unique_flat_indices
    U=len(centers)
    _need(bool(((centers>=0)&(centers<H*W)).all()),'Old center flat index outside original image')
    _need(bool((centers[1:]>centers[:-1]).all()),'Old center flat indices must be strictly ascending and distinct')
    if U==0:
        return dict(extra_flat_indices=np.empty(0,dtype=np.int64),
                    neighbor_to_combined=np.empty((0,9),dtype=np.int64))

    dy=np.array([offset[0] for offset in OFFSETS],dtype=np.int64)
    dx=np.array([offset[1] for offset in OFFSETS],dtype=np.int64)
    neighbor_y=np.clip((centers//W)[:,None]+dy[None,:],0,H-1)
    neighbor_x=np.clip((centers%W)[:,None]+dx[None,:],0,W-1)
    neighbors=neighbor_y*W+neighbor_x
    union=np.unique(neighbors)
    extra=np.setdiff1d(union,centers,assume_unique=True).astype(np.int64,copy=False)

    flat_neighbors=neighbors.reshape(-1)
    center_positions=np.searchsorted(centers,flat_neighbors)
    in_centers=(center_positions<U)
    safe_positions=np.minimum(center_positions,U-1)
    in_centers&=(centers[safe_positions]==flat_neighbors)
    mapping=(U+np.searchsorted(extra,flat_neighbors)).astype(np.int64,copy=False)
    mapping[in_centers]=center_positions[in_centers]
    mapping=mapping.reshape(U,9)
    _need(np.array_equal(mapping[:,4],np.arange(U,dtype=np.int64)),
          'Center slot must preserve the exact old center order')
    _need(len(extra)<=8*U and bool(((mapping>=0)&(mapping<U+len(extra))).all()),
          'Neighbor mapping must reference only old centers and the extra union')
    return dict(extra_flat_indices=extra,neighbor_to_combined=mapping)
