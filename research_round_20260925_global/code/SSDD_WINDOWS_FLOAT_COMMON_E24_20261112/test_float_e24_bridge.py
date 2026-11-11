import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import float_e24_bridge as bridge


class LongLedgerClockTest(unittest.TestCase):
    def test_epoch_boundaries_and_endpoint(self):
        self.assertEqual(bridge.visit_position(0), (1,1))
        self.assertEqual(bridge.visit_position(743), (1,744))
        self.assertEqual(bridge.visit_position(744), (2,1))
        self.assertEqual(bridge.visit_position(17855), (24,744))
        for invalid in (-1, 17856, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "visit cursor"):
                    bridge.visit_position(invalid)

    def test_actual_order_support_and_selected_masks(self):
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(bridge, FIT_VISITS=48, EPOCH_VISITS=2):
            paths = {name: Path(tmp)/name for name in ("common", "native", "center", "epoch")}
            rows = {name: [] for name in paths}
            a,b = "00"*32,"11"*32
            for epoch in range(1,25):
                rows["epoch"].append({"epoch":epoch,"visits":[[100,1],[101,0]]})
                for batch in (1,2):
                    ident = dict(epoch=epoch,batch=batch,ordinal=(epoch-1)*2+batch,image_id=99+batch)
                    count = int(batch == 1)
                    rows["common"].append(dict(ident, common_nonempty=count,native_common_masks_sha256=a,
                                               center_common_masks_sha256=b))
                    for mode,selected in (("native",a),("center",b)):
                        rows[mode].append(dict(ident,labels=count,postfilter_labels=count,boxes_sha256=a,
                                               postfilter_boxes_sha256=a,masks_sha256=selected,
                                               postfilter_masks_sha256=selected,cuda_rng_sha256=a))
            for name, path in paths.items():
                path.write_text("".join(json.dumps(row)+"\n" for row in rows[name]),encoding="utf-8")
            a_scope,a_mask = bridge.verified_actual(paths["common"],paths["native"],paths["epoch"],"native")
            b_scope,b_mask = bridge.verified_actual(paths["common"],paths["center"],paths["epoch"],"center")
            self.assertEqual(a_scope,b_scope)
            self.assertNotEqual(a_mask,b_mask)
            rows["center"][17]["postfilter_labels"] = 2
            paths["center"].write_text("".join(json.dumps(row)+"\n" for row in rows["center"]),encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "common support"):
                bridge.verified_actual(paths["common"],paths["center"],paths["epoch"],"center")


if __name__ == "__main__":
    unittest.main()
