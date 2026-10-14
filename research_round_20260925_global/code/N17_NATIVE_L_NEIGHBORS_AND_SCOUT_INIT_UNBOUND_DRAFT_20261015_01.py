"""UNBOUND_DRAFT: N17 native-coordinate features and fair ERROR scout initialization only.
No file IO, pipeline, owner/qualification gate, actual weights or scientific execution.
"""
OFFSETS=((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
EPS=1e-5
PARAMETERS=16803
BASE_STATE_SHAPES={'hidden.weight':(32,513),'hidden.bias':(32,),'output.weight':(3,32),'output.bias':(3,)}
STATE_SHAPES=dict(BASE_STATE_SHAPES,**{'hidden.weight':(32,521)})

def _coordinates(torch,L,q,flat):
    if not isinstance(L,torch.Tensor) or L.layout!=torch.strided or L.dtype!=torch.float32 or L.ndim!=3 or min(L.shape)<1:
        raise ValueError('native_L must be dense finite FP32[Q,H,W] in its original native space')
    for index in (q,flat):
        if not isinstance(index,torch.Tensor) or index.layout!=torch.strided or index.dtype!=torch.int64 or index.ndim!=1 or index.device!=L.device:
            raise ValueError('query_ids/flat_indices must be same-device dense INT64[N], no bool/float indices')
    if q.shape!=flat.shape or not bool(torch.isfinite(L).all()) or bool(((q<0)|(q>=L.shape[0])).any()) or bool(((flat<0)|(flat>=L.shape[1]*L.shape[2])).any()):
        raise ValueError('Invalid finite native grid, matching point count or original index range')
    return flat//L.shape[2],flat%L.shape[2]

def neighbor_features(torch,native_L,query_ids,flat_indices):
    """Gather same-query eight native neighbors, replicate borders; clip only gathered[N,8]."""
    y,x=_coordinates(torch,native_L,query_ids,flat_indices);H,W=native_L.shape[1:]
    values=torch.stack([native_L[query_ids,(y+dy).clamp(0,H-1),(x+dx).clamp(0,W-1)] for dy,dx in OFFSETS],dim=1)
    return values.clamp(-8,8)/8

def center_control_features(torch,native_L,query_ids,flat_indices):
    """Capacity/input-shape control: eight real copies of clipped native center."""
    y,x=_coordinates(torch,native_L,query_ids,flat_indices)
    return (native_L[query_ids,y,x].clamp(-8,8)/8).unsqueeze(1).repeat(1,8)

def _numpy_coordinates(np,L,q,flat):
    if not isinstance(L,np.ndarray) or L.dtype!=np.float32 or L.ndim!=3 or min(L.shape)<1 or not bool(np.isfinite(L).all()):
        raise ValueError('native_L must be finite FP32[Q,H,W] in original native space')
    if any(not isinstance(v,np.ndarray) or v.dtype!=np.int64 or v.ndim!=1 for v in (q,flat)) or q.shape!=flat.shape:
        raise ValueError('query_ids/flat_indices must be INT64[N], no bool/float indices')
    if bool(np.any((q<0)|(q>=L.shape[0]))) or bool(np.any((flat<0)|(flat>=L.shape[1]*L.shape[2]))):
        raise ValueError('Original query/flat index outside native grid')
    return flat//L.shape[2],flat%L.shape[2]

def neighbor_features_numpy(np,native_L,query_ids,flat_indices):
    """Independent explicit scalar-coordinate equation; no resize, Torch padding or pseudo-grid."""
    y,x=_numpy_coordinates(np,native_L,query_ids,flat_indices);H,W=native_L.shape[1:]
    out=np.empty((len(query_ids),8),dtype=np.float32)
    for i,q in enumerate(query_ids):
        for j,(dy,dx) in enumerate(OFFSETS):
            row=min(max(int(y[i])+dy,0),H-1);col=min(max(int(x[i])+dx,0),W-1)
            out[i,j]=min(max(native_L[int(q),row,col],np.float32(-8)),np.float32(8))/np.float32(8)
    return out

def center_control_features_numpy(np,native_L,query_ids,flat_indices):
    y,x=_numpy_coordinates(np,native_L,query_ids,flat_indices)
    out=np.empty((len(query_ids),8),dtype=np.float32)
    for i,q in enumerate(query_ids):
        out[i,:]=min(max(native_L[int(q),int(y[i]),int(x[i])],np.float32(-8)),np.float32(8))/np.float32(8)
    return out

def _state(torch,state,shapes):
    if not isinstance(state,dict) or set(state)!=set(shapes):raise ValueError('Exactly four scout tensors required')
    for name,shape in shapes.items():
        value=state[name]
        if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.dtype!=torch.float32 or tuple(value.shape)!=shape or not bool(torch.isfinite(value).all()):
            raise ValueError('Finite FP32 scout parameter shape required: '+name)

def lift_initial_state(torch,base_state):
    """Preserve old four initial tensors; append only eight zero hidden columns, no RNG."""
    _state(torch,base_state,BASE_STATE_SHAPES)
    if any(v.device.type!='cpu' for v in base_state.values()):raise ValueError('Initial state lifting is CPU-only')
    result={name:value.detach().clone() for name,value in base_state.items()}
    result['hidden.weight']=torch.cat((result['hidden.weight'],torch.zeros((32,8),dtype=torch.float32,device='cpu')),dim=1)
    return result

def make_scout(torch,seed,base_factory):
    """Call ROOT-bound old513 factory first; lift its four tensors, never sample a521 layer.

    ROOT supplies the closed N16 make_scout callable, not a path or trained checkpoint.
    External CPU RNG is restored; all construction/initial lifting is CPU FP32.
    """
    if type(seed) is not int or not 0<=seed<2**63 or not callable(base_factory):raise ValueError('Valid private seed and ROOT-bound old factory required')
    class LinearFromState(torch.nn.Module):
        def __init__(self,weight,bias):
            super().__init__();self.weight=torch.nn.Parameter(weight.detach().clone());self.bias=torch.nn.Parameter(bias.detach().clone())
        def forward(self,value):return torch.nn.functional.linear(value,self.weight,self.bias)
    class Scout(torch.nn.Module):
        def __init__(self,state):
            super().__init__()
            self.norm_F=torch.nn.LayerNorm(256,eps=EPS,elementwise_affine=False,device='cpu',dtype=torch.float32)
            self.norm_e=torch.nn.LayerNorm(256,eps=EPS,elementwise_affine=False,device='cpu',dtype=torch.float32)
            self.hidden=LinearFromState(state['hidden.weight'],state['hidden.bias'])
            self.output=LinearFromState(state['output.weight'],state['output.bias'])
        def forward(self,F,e,L,extra8):
            if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():raise ValueError('FP32 only; AMP forbidden')
            device=self.hidden.weight.device;n=len(F) if isinstance(F,torch.Tensor) and F.ndim else -1
            for value,shape in ((F,(n,256)),(e,(n,256)),(L,(n,)),(extra8,(n,8))):
                if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.dtype!=torch.float32 or value.device!=device or tuple(value.shape)!=shape or not bool(torch.isfinite(value).all()):
                    raise ValueError('Same-device finite FP32 F[N,256]/e[N,256]/rawL[N]/extra8[N,8] required')
            _state(torch,self.state_dict(),STATE_SHAPES)
            descriptor=torch.cat((self.norm_F(F),self.norm_e(e),(L/8).unsqueeze(1),extra8),dim=1)
            logits=self.output(torch.relu(self.hidden(descriptor)))
            if logits.dtype!=torch.float32 or tuple(logits.shape)!=(n,3) or not bool(torch.isfinite(logits).all()):raise ValueError('Finite FP32[N,3] output required')
            return logits
    with torch.random.fork_rng(devices=[]):
        old=base_factory(torch,seed)
        if not isinstance(old,torch.nn.Module) or list(old.named_buffers()):raise ValueError('Old factory must produce original four-tensor buffer-free scout')
        initial=lift_initial_state(torch,old.state_dict());del old
        result=Scout(initial)
        if sum(v.numel() for v in result.parameters())!=PARAMETERS:raise ValueError('16803 parameter contract')
        return result
