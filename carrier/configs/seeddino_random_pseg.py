# SeedDINO with RANDOM seed boxes (P1-5 trivial-prior control). Same as
# seeddino_pred but seed_source='random': injects the same NUMBER of seeds as the
# real NQS run, at random locations/scales. If this matches seeddino_pred's
# AR +1.85 -> the gain is "any high-recall proposal injection", and the MNE
# statistic adds nothing specific (claim must downgrade). If pred > random ->
# the evidence statistic specifically helps (claim defensible).
_base_ = './seeddino_pred_pseg.py'

model = dict(seed_source='random')
