"""Strict schema adapter for one unused node in the pinned author YAML.

No imports, source rewriting, network construction, or validation relaxation.
Call the author's DiffSAR and EMA initializers first; merge the unchanged YAML
after registration, then run verification before constructing the model.
"""

UPSTREAM_COMMIT = "bdc3362372213bc8c3a1f6d5b7b74dc224d1b871"
NODE = "BIFPN_1p_5p"


def _expected_values():
    return {"FUSE_TYPE": "sum", "IN_FEATURES": ["res2", "res3", "res4", "res5"],
            "NORM": " ", "OUT_CHANNELS": 256, "REPEAT": 1}


def register_inactive_bifpn_schema(cfg):
    """Register exactly five typed leaves; never overwrite an existing node."""
    if NODE in cfg.MODEL:
        raise ValueError("MODEL.BIFPN_1p_5p already exists; refusing to overwrite")
    node = type(cfg)()
    for key, value in _expected_values().items():
        setattr(node, key, value)
    setattr(cfg.MODEL, NODE, node)
    return {"adapter": "inactive-bifpn-schema-v1", "upstream_commit": UPSTREAM_COMMIT,
            "status": "REGISTERED_INACTIVE_METADATA", "node": "MODEL." + NODE,
            "implements_bifpn": False}


def verify_inactive_bifpn_schema(cfg):
    """After YAML merge, refuse routes where the unimplemented node could matter."""
    if cfg.MODEL.META_ARCHITECTURE != "DiffSARShipInst":
        raise ValueError("Only the pinned DiffSARShipInst meta architecture is supported")
    if cfg.MODEL.BACKBONE.NAME != "build_resnet_scje_fpn_backbone":
        raise ValueError("Only the pinned SCJE backbone route is supported")
    if NODE not in cfg.MODEL:
        raise ValueError("Missing explicit inactive BIFPN schema")
    node = getattr(cfg.MODEL, NODE)
    expected = _expected_values()
    if set(node.keys()) != set(expected):
        raise ValueError("Inactive BIFPN node must contain exactly the five verified keys")
    for key, value in expected.items():
        actual = getattr(node, key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError("Inactive BIFPN value/type differs from the pinned YAML: " + key)
    return {"adapter": "inactive-bifpn-schema-v1", "status": "VERIFIED_INACTIVE_METADATA",
            "implements_bifpn": False}
