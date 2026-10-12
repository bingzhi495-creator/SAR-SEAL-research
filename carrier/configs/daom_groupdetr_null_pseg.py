# DAOM Group-DETR NULL control on PSeg-SSDD: aux group + one-to-many loss but
# NO evidence seeds (density off). This isolates "does the one-to-many auxiliary
# group help on its own" from "does the evidence-density seeding help". The aux
# reference points are the standard learnable (top-k encoder) init only -- no
# seed injection. Compare against daom_seed_pseg.py: the Delta is exactly the
# contribution of the evidence-density seeds beyond plain o2m.
#
# seed_source='off' makes _seed_boxes_norm return None for every image, so the
# aux group falls back to the pure learnable init (no seeds), while keeping the
# identical aux-group + k_o2m machinery.
_base_ = './daom_seed_pseg.py'

model = dict(
    seed_source='off',   # density off / no evidence seeds -> pure o2m control
    density_alpha=0.0,   # belt-and-suspenders: zero seed budget too
)
