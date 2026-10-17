"""UNBOUND feature-only view of the user's explicit full convolutional graph.

The caller supplies Torch and a real already-constructed model_full.UnetPlusPlus.
No Torch import, weight loading, file access, mode/parameter change, transformer
replacement, DSConv repair or BN repair occurs here. B=1 is mandatory; this is
not a claim that the original DSConv batch-index problem has been fixed.
"""


def make_feature_backbone(torch,legacy_backbone):
    """Preserve original module objects and return U=original CONV0_4 output.

    Register as .legacy; the existing shared adapter's feature_module_name is
    'legacy.CONV0_4', feature_channels=64. The wrapper has no new parameters.
    This factory has not yet been executed on the actual original backbone.
    """
    def need(ok,message):
        if not ok:raise ValueError(message)

    def validate(legacy):
        need(isinstance(legacy,torch.nn.Module),'A real caller-owned legacy Torch module is required')
        need(getattr(legacy,'use_x2_transformer',None) is False
             and hasattr(legacy,'encoder_x2') and legacy.encoder_x2 is None
             and not hasattr(legacy,'encoder_x4'),
             'Explicit no-BatchFormer variant: use_x2_transformer=False, encoder_x2=None, no x4 allocation')
        need(getattr(legacy,'deep_supervision',None) is False
             and type(getattr(legacy,'num_classes',None)) is int and legacy.num_classes==1,
             'Original one-class, no-deep-supervision full configuration required')
        size=getattr(legacy,'input_size',None)
        # The original deepest SimAM divides by H*W-1; input_size16 would
        # produce a 1x1 deepest map and an undefined variance denominator.
        need(type(size) is int and size>=32 and size%16==0,'Explicit input_size>=32 divisible by16')
        return size

    validate(legacy_backbone)

    class FullFeatureBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.legacy=legacy_backbone

        def inactive_parameter_names(self):
            """Known original unused parameter paths, in wrapper name space.

            This only reports names. It does not change requires_grad or select
            an optimizer. The original six bn1 modules still execute their
            original discarded path and may update running statistics in train
            mode. Parameter inactivity does not mean module calls are removed.
            """
            prefixes=(
                'CONV3_1','CONV2_1','CONV2_2','CONV1_1','CONV1_2','CONV1_3',
                'cbam_stage_up_1_1.bn1','cbam_stage_up_1_2.bn1','cbam_stage_up_1_3.bn1',
                'cbam_stage_up_2_1.bn1','cbam_stage_up_2_2.bn1','cbam_stage_up_3_1.bn1',
                'final_super_0_1','final_super_0_2','final_super_0_3','final_super_0_4',
                'ds_stage_2x.dsc_conv_y','ds_stage_2y.dsc_conv_x')
            full_prefixes=tuple('legacy.'+prefix+'.' for prefix in prefixes)
            return tuple(name for name,_ in self.named_parameters() if name.startswith(full_prefixes))

        def forward(self,x):
            legacy=self.legacy;size=validate(legacy)
            need(isinstance(x,torch.Tensor) and tuple(x.shape)==(1,3,size,size)
                 and x.is_floating_point(),'One floating-point RGB-format image at the explicit fixed input_size')
            # model_full.py392-419: original five-scale encoder and actual H/4
            # semantic/ordinary-input x/y DSConv branches. The transformer is
            # explicitly absent; preserve even the original reshape pair.
            x_0_0=legacy.stage_0(x)
            x_0_0=legacy.cbam_stage_0(x_0_0)
            x_1_0=legacy.cbam_stage_1(legacy.pool(x_0_0))
            x_2_0=legacy.cbam_stage_2(legacy.pool(x_1_0))
            x_2_0_1=legacy.stage_2(legacy.pool(x_1_0))
            old_shape=x_2_0.shape
            x_2_0=x_2_0.view(old_shape[0],old_shape[1],old_shape[2]*old_shape[3])
            x_2_0=x_2_0.view(old_shape)
            x_2_0_x=legacy.ds_stage_2x(x_2_0_1)
            x_2_0_y=legacy.ds_stage_2y(x_2_0_1)
            x_2_0=legacy.con2(torch.cat([x_2_0,x_2_0_x,x_2_0_y],dim=1))
            x_3_0=legacy.cbam_stage_3(legacy.pool(x_2_0))
            x_4_0=legacy.cbam_stage_4(legacy.pool(x_3_0))
            # model_full.py435-463: original ordering, all six attention-based
            # lower decoder nodes and all four plain highest-resolution nodes.
            x_0_1=torch.cat([legacy.upsample_0_1(x_1_0),x_0_0],1)
            x_0_1=legacy.CONV0_1(x_0_1)
            x_1_1=torch.cat([legacy.upsample_1_1(x_2_0),x_1_0],1)
            x_1_1=legacy.cbam_stage_up_1_1(x_1_1)
            x_2_1=torch.cat([legacy.upsample_2_1(x_3_0),x_2_0],1)
            x_2_1=legacy.cbam_stage_up_2_1(x_2_1)
            x_3_1=torch.cat([legacy.upsample_3_1(x_4_0),x_3_0],1)
            x_3_1=legacy.cbam_stage_up_3_1(x_3_1)
            x_2_2=torch.cat([legacy.upsample_2_2(x_3_1),x_2_0,x_2_1],1)
            x_2_2=legacy.cbam_stage_up_2_2(x_2_2)
            x_1_2=torch.cat([legacy.upsample_1_2(x_2_1),x_1_0,x_1_1],1)
            x_1_2=legacy.cbam_stage_up_1_2(x_1_2)
            x_1_3=torch.cat([legacy.upsample_1_3(x_2_2),x_1_0,x_1_1,x_1_2],1)
            x_1_3=legacy.cbam_stage_up_1_3(x_1_3)
            x_0_2=torch.cat([legacy.upsample_0_2(x_1_1),x_0_0,x_0_1],1)
            x_0_2=legacy.CONV0_2(x_0_2)
            x_0_3=torch.cat([legacy.upsample_0_3(x_1_2),x_0_0,x_0_1,x_0_2],1)
            x_0_3=legacy.CONV0_3(x_0_3)
            x_0_4=torch.cat([legacy.upsample_0_4(x_1_3),x_0_0,x_0_1,x_0_2,x_0_3],1)
            return legacy.CONV0_4(x_0_4)

    return FullFeatureBackbone()
