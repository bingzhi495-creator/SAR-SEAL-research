# APEX arm: b1 (SAR init) + a1 (stride-4) + a2 (GW matching) + CAST-hard (physics-consistent hard-example
# paste). The full mutually-reinforcing chain: CAST makes hard examples -> SAR-pretrained features can encode
# them -> stride-4 reaches small ships -> GW matching assigns crowded scenes stably. Target: inshore .25+.
_base_ = './cast_hard_m2f_hrsid.py'   # = a1a2 + CAST-hard training data

load_from = 'pretrained/m2f_r50_coco_sarbackbone.pth'
