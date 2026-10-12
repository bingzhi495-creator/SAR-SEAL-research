# A0R + GCNet ContextBlock (global context) plugin in ResNet c3-c5. Global-context
# anti-clutter is mechanistically different from the exhausted local CBAM; targets
# inshore land-sea false positives. Config-only, no download, negligible VRAM.
_base_ = './a0r_refine_pseg.py'

model = dict(backbone=dict(
    plugins=[dict(cfg=dict(type='ContextBlock', ratio=1.0 / 16),
                  stages=(False, True, True, True),
                  position='after_conv3')]))
