# SeedDINO oracle-seeding gate (verdict-3 required change 2):
# GT boxes injected as seeds into reference_points. If this does NOT beat the
# DINO baseline on inshore box AP, the inshore bottleneck is feature-level
# (not query-level) -> stop the seeding line, go to Plan B (SARATR-X backbone).
_base_ = './dino_r50_pseg.py'

custom_imports = dict(imports=['carrier.seed_dino'], allow_failed_imports=False)

model = dict(type='SeedDINO', seed_source='oracle', max_seeds=50, set_content=False)
