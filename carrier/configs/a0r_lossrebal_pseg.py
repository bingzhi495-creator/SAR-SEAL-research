# A0R + matcher/loss rebalance toward mask/dice. On 1-class SAR the cls cost is
# near-informationless; AP@IoU is overlap-driven, so up-weight dice in BOTH the
# Hungarian matcher and the loss (kept proportional). Pure config, zero compute cost.
_base_ = './a0r_refine_pseg.py'

model = dict(
    panoptic_head=dict(
        loss_cls=dict(loss_weight=2.0),
        loss_mask=dict(loss_weight=5.0),
        loss_dice=dict(loss_weight=8.0)),
    train_cfg=dict(
        assigner=dict(match_costs=[
            dict(type='ClassificationCost', weight=2.0),
            dict(type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dict(type='DiceCost', weight=8.0, pred_act=True, eps=1.0)])))
