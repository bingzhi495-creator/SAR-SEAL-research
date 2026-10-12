# b1+a1a2 combined arm: SAR-domain backbone init (b1) + stride-4 resolution (a1) + GW matching (a2).
# The synergy-chain "b1 feeds a1/a2" hypothesis arm: SAR features give C2 content for stride-4 to lock onto
# and cleaner moments for GW matching. Controls: a1a2 (ImageNet/COCO init, .186 inshore) isolates the init
# effect on TOP of a1a2; b1-alone isolates a1a2's add on top of SAR init.
_base_ = './a1a2_m2f_hrsid.py'

load_from = 'pretrained/m2f_r50_coco_sarbackbone.pth'
