"""UNBOUND query/mask quality baseline; injected Torch/NumPy, no top-level imports.

Only the quality MLP is trainable. The predicted-mask feature takes no GT input;
matched-mask supervision is computed separately on the complete original grid.
"""


def build_head(torch):
    """Fixed 1040->256->128->1 sigmoid MLP; the cached teacher input is detached.

    The first 256 values must be the original teacher embedding, followed by the
    784 predicted-mask values. No teacher module or parameter is attached here.
    """
    class QueryMaskQualityHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(1040,256,device='cpu',dtype=torch.float32),torch.nn.ReLU(),
                torch.nn.Linear(256,128,device='cpu',dtype=torch.float32),torch.nn.ReLU(),
                torch.nn.Linear(128,1,device='cpu',dtype=torch.float32),torch.nn.Sigmoid())

        def forward(self, features):
            if (not isinstance(features,torch.Tensor) or features.ndim != 2
                    or features.shape[1] != 1040 or features.dtype != torch.float32
                    or features.device != next(self.parameters()).device
                    or not bool(torch.isfinite(features).all())):
                raise ValueError('Finite FP32 batch-by-1040 features on the quality-head device required')
            quality = self.layers(features.detach()).squeeze(-1)
            if not bool(torch.isfinite(quality).all()) or not bool(((quality >= 0) & (quality <= 1)).all()):
                raise ValueError('Quality-head prediction must remain finite and in [0,1]')
            return quality

    return QueryMaskQualityHead()


def _binary_array(np, value, label):
    if (not isinstance(value,np.ndarray) or value.ndim != 2 or min(value.shape) <= 0
            or value.dtype.kind not in 'buif' or not bool(np.isfinite(value).all())
            or not bool(((value == 0) | (value == 1)).all())):
        raise ValueError(label + ': finite binary original-grid 2D array required')
    return value.astype(np.bool_,copy=False)


def matched_mask_iou(np, pred, GT):
    """FP64 full-grid binary IoU; empty union is zero, with no crop or padding."""
    pred = _binary_array(np,pred,'Prediction')
    GT = _binary_array(np,GT,'Matched true GT')
    if pred.shape != GT.shape:
        raise ValueError('Prediction and matched GT must retain the same complete original shape')
    intersection = int(np.count_nonzero(pred & GT))
    union = int(np.count_nonzero(pred | GT))
    return np.float64(intersection)/np.float64(union) if union else np.float64(0)


def make_mask_features(torch, pred_binary2D):
    """Predicted-mask-only unpadded bbox crop resized to a CPU FP32 784-vector."""
    pred = pred_binary2D
    if (not isinstance(pred,torch.Tensor) or pred.ndim != 2 or min(pred.shape) <= 0
            or pred.device.type != 'cpu' or pred.dtype != torch.bool or pred.requires_grad):
        raise ValueError('Predicted mask must be a positive-shape CPU bool 2D non-gradient tensor')
    coordinates = torch.nonzero(pred,as_tuple=False)
    if not coordinates.numel():
        return torch.zeros(784,dtype=torch.float32,device='cpu')
    row0 = int(coordinates[:,0].min().item())
    row1 = int(coordinates[:,0].max().item())+1
    col0 = int(coordinates[:,1].min().item())
    col1 = int(coordinates[:,1].max().item())+1
    crop = pred[row0:row1,col0:col1].to(dtype=torch.float32).reshape(1,1,row1-row0,col1-col0)
    resized = torch.nn.functional.interpolate(crop,size=(28,28),mode='bilinear',align_corners=False)
    return resized.reshape(784)


def score_product(np, S0, qhat):
    """Position-preserving FP64 multiplication; original scores are never clipped."""
    if (not isinstance(S0,np.ndarray) or not isinstance(qhat,np.ndarray)
            or S0.ndim != 1 or qhat.shape != S0.shape
            or S0.dtype.kind not in 'buif' or qhat.dtype.kind not in 'buif'):
        raise ValueError('Original score and quality arrays must be equal-shaped real 1D vectors')
    scores = S0.astype(np.float64,copy=False)
    quality = qhat.astype(np.float64,copy=False)
    if (not bool(np.isfinite(scores).all()) or not bool(np.isfinite(quality).all())
            or not bool(((quality >= 0) & (quality <= 1)).all())):
        raise ValueError('Finite original scores and predicted quality in [0,1] required')
    product = scores*quality
    if not bool(np.isfinite(product).all()):
        raise ValueError('FP64 score product must remain finite')
    return product
