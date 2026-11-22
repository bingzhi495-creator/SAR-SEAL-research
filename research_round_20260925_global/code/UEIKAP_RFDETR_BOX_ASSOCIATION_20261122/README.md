# Box-associated UEIKAP hull diagnosis

[Frozen protocol](../../UEIKAP_RFDETR_BOX_ASSOCIATION_PRELOCK_20261122.md) · [Observed diagnosis](../../UEIKAP_RFDETR_BOX_ASSOCIATION_ACTUAL_20261122.md)

Run `box_association.py --help` in the original UEIKAP geometry runtime. Pass the closed [RF-DETR evaluation](../UEIKAP_RFDETR_CARRIER_20261122), its accepted label-free edits, and the prior original UEIKAP metadata. The worker independently reproduces the former mask-associated aggregate before emitting the box-associated results. Use the existing [observer](../HRSID_DEV705_CLASSICAL_ATTRACTION_20261122/observe.py) to record its actual exit and bind the output. Run `test_box_association.py` for synthetic threshold, tie-order and weak-mask assignment checks.
