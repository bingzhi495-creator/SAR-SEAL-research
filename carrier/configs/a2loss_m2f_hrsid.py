# a2-loss CONTROL ARM on HRSID (the GATE-0 the paper's novelty claim needs, review w13gekell):
# identical Gaussian-Wasserstein distance as a2, but as an AUXILIARY LOSS on matched pairs -- matching stays
# stock. Same protocol as a0_m2f_r50_hrsid, so the triangle is: a0 baseline (.125 inshore) vs a2 matching-
# placement (.173) vs THIS loss-placement arm. a2 > a2-loss => "matching-cost placement matters" evidenced
# (clean cut vs O2Former-style NWD-in-loss); a2-loss ~= a2 => rescope the novelty framing honestly.
_base_ = './a0_m2f_r50_hrsid.py'

custom_imports = dict(imports=['carrier.wass_match'], allow_failed_imports=False)

model = dict(
    panoptic_head=dict(
        type='WassLossM2FHead',
        lambda_wass=2.0,     # matched to a2's matching-cost weight=2.0 for a fair placement comparison
        wass_C=0.5),         # same C as a2
)
