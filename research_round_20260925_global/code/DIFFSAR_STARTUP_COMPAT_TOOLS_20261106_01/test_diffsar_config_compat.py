"""Text/configuration-only adapter checks; no Detectron2 or ML imports."""

import copy
import unittest

from diffsar_config_compat import register_inactive_bifpn_schema, verify_inactive_bifpn_schema


class Node(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def set_new_allowed(self, *args, **kwargs):
        raise AssertionError("The adapter must never relax configuration validation")


def active_config():
    return Node(MODEL=Node(
        BACKBONE=Node(NAME="build_resnet_scje_fpn_backbone"),
        META_ARCHITECTURE="DiffSARShipInst",
        SCJE_FPN=Node(OUT_CHANNELS=256, NORM=""),
        DiffSARShipInst=Node(NUM_CLASSES=1, NUM_PROPOSALS=500, NUM_HEADS=6),
    ), MODEL_EMA=Node(ENABLED=False), INPUT=Node(FORMAT="RGB"))


class ConfigAdapter(unittest.TestCase):
    def test_registration_preserves_every_existing_active_value(self):
        cfg = active_config()
        before = copy.deepcopy(cfg)
        result = register_inactive_bifpn_schema(cfg)
        self.assertEqual(result["status"], "REGISTERED_INACTIVE_METADATA")
        self.assertEqual(cfg.MODEL.BIFPN_1p_5p, {
            "FUSE_TYPE": "sum", "IN_FEATURES": ["res2", "res3", "res4", "res5"],
            "NORM": " ", "OUT_CHANNELS": 256, "REPEAT": 1,
        })
        after = copy.deepcopy(cfg)
        del after.MODEL["BIFPN_1p_5p"]
        self.assertEqual(after, before)

    def test_existing_node_is_never_overwritten(self):
        cfg = active_config()
        cfg.MODEL.BIFPN_1p_5p = Node(USER_VALUE=123)
        before = copy.deepcopy(cfg)
        with self.assertRaises(ValueError):
            register_inactive_bifpn_schema(cfg)
        self.assertEqual(cfg, before)

    def test_postmerge_guard_accepts_only_verified_unused_metadata_and_route(self):
        cfg = active_config()
        register_inactive_bifpn_schema(cfg)
        before = copy.deepcopy(cfg)
        self.assertEqual(verify_inactive_bifpn_schema(cfg)["status"], "VERIFIED_INACTIVE_METADATA")
        self.assertEqual(cfg, before)
        for path, key, value in (
            (("MODEL", "BACKBONE"), "NAME", "build_resnet_bifpn_p1_p5_backbone"),
            (("MODEL",), "META_ARCHITECTURE", "GeneralizedRCNN"),
            (("MODEL", "BIFPN_1p_5p"), "REPEAT", True),
            (("MODEL", "BIFPN_1p_5p"), "NORM", ""),
            (("MODEL", "BIFPN_1p_5p"), "UNREGISTERED", 1),
        ):
            with self.subTest(path=path, key=key):
                changed = copy.deepcopy(cfg)
                branch = changed
                for part in path:
                    branch = branch[part]
                branch[key] = value
                snapshot = copy.deepcopy(changed)
                with self.assertRaises(ValueError):
                    verify_inactive_bifpn_schema(changed)
                self.assertEqual(changed, snapshot)


if __name__ == "__main__":
    unittest.main()
