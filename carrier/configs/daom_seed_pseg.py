# DAOM-Seed on PSeg-SSDD: training-only one-to-many seeded auxiliary queries.
# Non-displacing fix for the SeedDINO recall@0.5 -3.5 regression: the evidence
# seeds are injected into a SEPARATE aux query group (appended after dn+matching)
# supervised one-to-many, and DROPPED at inference -> the test-time graph is
# byte-identical to the DINO baseline (strongest capacity control), zero new
# params. Reuses the exact SeedDINO-pred seed pipeline (LoadSeeds, flip+resize
# aligned NQS seeds).
_base_ = './seeddino_pred_pseg.py'

custom_imports = dict(
    imports=['carrier.daom_seed_dino', 'carrier.seed_dino', 'carrier.load_seeds'],
    allow_failed_imports=False)

model = dict(
    type='DAOMSeedDINO',
    seed_source='pred',     # real NQS evidence seeds from data_sample.seeds
    max_seeds=200,          # raise cap so density budget (k_max) is the limiter
    set_content=False,
    # --- DAOM aux-group kwargs ---
    aux_num_queries=300,
    k_o2m=6,
    lambda_aux=1.0,
    density_alpha=1.0,
    k_min=8,
    k_max=200,
)
