# b1-SARInit on HRSID (synergy P1-e): identical to a0_m2f_r50_hrsid EXCEPT the R50 backbone weights inside
# load_from come from SAR-domain pretraining (SARDet-100K/MSFA, arXiv:2403.06534, raw-SAR variant) instead of
# ImageNet/COCO. To avoid the confound "backbone init changed AND neck/head lost COCO init", load_from points
# to an OFFLINE-COMPOSED checkpoint: COCO M2F full weights with ONLY backbone.* tensors replaced by the SARDet
# R50 (composed by tools compose_b1_ckpt.py after download+leakage audit). Zero architecture change, zero new
# params -> GATE-0 immune to the capacity attack; control arm = a0_hrsid_seed0 (.125) at the same protocol.
# PREREQUISITES (attack queue checks): pretrained/sardet100k_r50.pth downloaded + runs/_B1_AUDIT_PASS (file-level
# leakage audit: SARDet-100K train list ∩ HRSID official_test = ∅ and ∩ PSeg test = ∅).
_base_ = './a0_m2f_r50_hrsid.py'

load_from = 'pretrained/m2f_r50_coco_sarbackbone.pth'   # composed: COCO M2F + SARDet-100K R50 backbone
