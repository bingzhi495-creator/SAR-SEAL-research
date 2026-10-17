"""UNBOUND shared-image adapter, not an executed full-UNet++ reproduction.

Only the caller supplies Torch and the chosen backbone. No module-level Torch
import, weight loading, backbone surgery, training-mode change, teacher freezing,
GT, class, score, query filtering or logit renderer is introduced here.
Root owns future actual tensor tests and scientific execution.
"""


def make_adapter(torch,backbone,feature_module_name,feature_channels,query_channels=256,hidden=32):
    """Create a shared-image encoder and one small query-conditioned residual head.

    encode(image) returns caller-owned projected features [1,hidden,H,W] with
    their autograd graph intact; no encoded features are retained on the module.
    predict(encoded,e_q,M200) returns [Q,1,H,W] residuals. M200 is the original
    coarse mask logit tensor [Q,200,200], not a GT mask. The fixed outer renderer
    and original baseline-logit addition remain the caller's responsibility.
    """
    def need(ok,message):
        if not ok:raise ValueError(message)

    need(isinstance(backbone,torch.nn.Module),'A real caller-owned Torch backbone is required')
    need(type(feature_module_name) is str and bool(feature_module_name)
         and feature_module_name in dict(backbone.named_modules()),'Named feature module must exist')
    need(all(type(value) is int and value>0 for value in (feature_channels,query_channels,hidden)),
         'Positive integer feature/query/hidden channel counts')

    class SharedImageInstanceAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=backbone
            # Store only the name: a second module alias would create duplicate
            # backbone paths in the adapter state_dict.
            self.feature_module_name=feature_module_name
            self.feature_channels=feature_channels
            self.query_channels=query_channels
            self.hidden=hidden
            self.feature_projection=torch.nn.Conv2d(feature_channels,hidden,1)
            self.query_projection=torch.nn.Linear(query_channels,hidden)
            self.mask_projection=torch.nn.Conv2d(1,hidden,1)
            self.head=torch.nn.Sequential(torch.nn.ReLU(inplace=False),torch.nn.Conv2d(hidden,1,1))
            torch.nn.init.zeros_(self.head[-1].weight)
            torch.nn.init.zeros_(self.head[-1].bias)

        def encode(self,image):
            """Run the original backbone once, removing only our temporary hook."""
            need(isinstance(image,torch.Tensor) and image.ndim==4 and image.shape[0]==1
                 and image.is_floating_point(),'Encode exactly one floating-point image')
            target=dict(self.backbone.named_modules()).get(self.feature_module_name)
            need(target is not None,'Configured feature module must remain present')
            captured=[]

            def capture(module,arguments,output):
                need(isinstance(output,torch.Tensor) and output.ndim==4
                     and output.shape[0]==1 and output.shape[1]==self.feature_channels
                     and min(output.shape[-2:])>0 and output.is_floating_point(),
                     'Named module must emit one shared NCHW floating-point feature tensor')
                # Keep the exact hook-time value if a later original layer uses
                # an in-place operation; clone preserves the autograd connection.
                captured.append(output.clone())

            handle=target.register_forward_hook(capture)
            try:
                self.backbone(image)
            finally:
                handle.remove()
            need(len(captured)==1,'Configured feature module must execute exactly once per image')
            feature=captured.pop()
            need(feature.device==self.feature_projection.weight.device
                 and feature.dtype==self.feature_projection.weight.dtype,
                 'Caller must place the backbone and adapter projections on one device/dtype')
            return self.feature_projection(feature)

        def _query_count(self,encoded,e_q,M200):
            need(isinstance(encoded,torch.Tensor) and encoded.ndim==4
                 and encoded.shape[0]==1 and encoded.shape[1]==self.hidden
                 and min(encoded.shape[-2:])>0 and encoded.is_floating_point(),
                 'Use caller-owned projected shared features [1,hidden,H,W]')
            need(isinstance(e_q,torch.Tensor) and e_q.ndim==2
                 and e_q.shape[1]==self.query_channels,'Original query embeddings [Q,query_channels]')
            need(isinstance(M200,torch.Tensor) and M200.ndim==3
                 and tuple(M200.shape[1:])==(200,200) and M200.shape[0]==e_q.shape[0],
                 'Original query-aligned coarse logits [Q,200,200]')
            need(e_q.device==M200.device==encoded.device
                 and e_q.dtype==M200.dtype==encoded.dtype,
                 'Shared features, embeddings and coarse logits must share device and floating dtype')
            return e_q.shape[0]

        def predict(self,encoded,e_q,M200):
            """Only small shared readout operations; never call the backbone."""
            count=self._query_count(encoded,e_q,M200)
            height,width=encoded.shape[-2:]
            if count==0:return encoded.new_empty((0,1,height,width))
            queries=self.query_projection(e_q).unsqueeze(-1).unsqueeze(-1)
            coarse=torch.nn.functional.interpolate(M200.unsqueeze(1),size=(height,width),
                mode='bilinear',align_corners=False)
            combined=encoded.expand(count,-1,-1,-1)+queries+self.mask_projection(coarse)
            return self.head(combined)

        def forward(self,image,e_q,M200,query_batch_size=4):
            """Encode once; query grouping changes only the small residual readout."""
            need(type(query_batch_size) is int and query_batch_size>0,'Positive query microbatch size')
            encoded=self.encode(image)
            count=self._query_count(encoded,e_q,M200)
            if count==0:return self.predict(encoded,e_q,M200)
            residuals=[]
            for first in range(0,count,query_batch_size):
                residuals.append(self.predict(encoded,e_q[first:first+query_batch_size],
                                              M200[first:first+query_batch_size]))
            return torch.cat(residuals,dim=0)

    return SharedImageInstanceAdapter()
