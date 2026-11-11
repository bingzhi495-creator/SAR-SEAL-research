from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from pycocotools import mask

import float_dev187 as app


def make_fixture():
    ids = list(range(1, 188))
    gt = dict(images=[dict(id=i, height=16, width=16) for i in ids], categories=[dict(id=1, name="ship")],
              annotations=[])
    float_gt = deepcopy(gt)
    streams = dict(native=[], center=[])
    for image_id in ids:
        positions = [(1, 1), (8, 1)]
        if image_id <= 6:
            positions.append((1, 8))
        for x, y in positions:
            aid = len(gt["annotations"]) + 1
            row = dict(id=aid, image_id=image_id, category_id=1, iscrowd=0, area=8, bbox=[x,y,4,2],
                       segmentation=[[x,y,x+4,y,x+4,y+2,x,y+2]])
            gt["annotations"].append(row)
            fx = x + .65
            float_gt["annotations"].append(dict(row, bbox=[fx,y,4,2],
                                                segmentation=[[fx,y,fx+4,y,fx+4,y+2,fx,y+2]]))
            for mode, shift in (("native", 0), ("center", 1)):
                pixels = np.zeros((16,16), dtype=np.uint8)
                pixels[y:y+2,x+shift:x+shift+4] = 1
                rle = mask.encode(np.asfortranarray(pixels))
                streams[mode].append(dict(image_id=image_id, category_id=1, score=.7,
                                          segmentation=dict(size=rle["size"], counts=rle["counts"].decode("ascii"))))
    return gt, float_gt, streams, ids


class FloatPrimary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage, cls.modules = app.load_stock(cls.code_root)
        cls.scorer = cls.modules["ssdd_phase_dev_score"]
        cls.apis = app.score_apis(cls.scorer)

    def test_original_coco_backend_uses_float_primary_and_fixed_predictions(self):
        integer, floating, predictions, ids = make_fixture()
        before = deepcopy(predictions)
        out = app.score_variants(self.scorer, self.apis, integer, floating, predictions, ids)
        self.assertEqual(predictions, before)
        self.assertLess(out["sensitivity"]["center_minus_native_points"]["AP"], 0.)
        self.assertGreater(out["primary"]["center_minus_native_points"]["AP"], 0.)
        self.assertEqual(out["primary"]["center"]["AP"], out["sensitivity"]["native"]["AP"])
        for cell in ("primary", "sensitivity"):
            for mode in app.ARMS:
                scores = out[cell][mode]
                self.assertEqual(len(scores["all12"]), 12)
                self.assertEqual(scores["APm"], scores["all12"][4])
                self.assertEqual(scores["APl"], scores["all12"][5])
                self.assertIn("AP90", scores)
        floating["annotations"][0]["iscrowd"] = 1
        with self.assertRaisesRegex(ValueError, "Only three geometry fields"):
            app.score_variants(self.scorer, self.apis, integer, floating, predictions, ids)

    def test_gate_uses_both_paired_a_and_higher_fixed_reference(self):
        reference = {"AP": app.HISTORICAL_AP}
        self.assertEqual(app.gate({"AP": .69}, {"AP": .70}, reference)["status"], "NO_PROMOTION_NONPOSITIVE")
        b = app.HISTORICAL_AP + .005
        self.assertEqual(app.gate({"AP": .69}, {"AP": b}, reference)["status"], "AT_MOST_ONE_PRELOCKED_REPLICATION")
        self.assertEqual(app.gate({"AP": .72}, {"AP": .73}, reference)["status"], "QUALIFIES_NEXT_FIXED_SEED")

    def test_reference_uses_closed_seed_5201_float_report(self):
        row = app.historical_native(self.stage, app.file_ref(self.historical, app.REFERENCE_SHA))
        self.assertEqual(row["AP"], .7047324561474607)
        self.assertEqual(len(row["all12"]), 12)

    def test_float_training_receipts_are_validated_without_integer_schema_coercion(self):
        api = self.modules["ssdd_phase_dev_admission"]
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            arm_refs, arms = {}, {}

            def write(name, document):
                path = root / (name + ".json")
                raw = json.dumps(document).encode()
                path.write_bytes(raw)
                return app.file_ref(path, sha256(raw).hexdigest())

            for i, mode in enumerate(app.ARMS):
                digest = lambda key: sha256(f"{i}/{key}".encode()).hexdigest()
                phase = api.phase_marker(mode, app.TRAINING_PROTOCOL, "7" * 64, app.FIT_SOURCE_MANIFEST_SHA)
                arms[mode] = dict(status=app.ARM_STATUS, phase=phase, qualification_pair_sha256=app.FLOAT_SHORT_SHA,
                                  operation_commit="5" * 40, original_os_wait=True, original_os_exit=0,
                                  resource_stop=False, worker_closed=True, completed_epochs=24, visits=17856,
                                  optimizer_updates=4464, ema_updates=4464, checkpoint_bytes=520,
                                  common_ledger_sha256="1" * 64, identical_fields_sha256="2" * 64,
                                  common_targets_total=32000, common_empty_visits=4, changed_mask_visits=12000,
                                  filtered_empty_rows_total=0, **app.FIT_ONLY,
                                  **{field:digest(field) for field in app.ARM_DIGESTS})
                arms[mode]["visit_ledger_sha256"] = "3" * 64
                arm_refs[mode] = write(mode, arms[mode])
            pair = dict(status=app.PAIR_STATUS, training_protocol=app.TRAINING_PROTOCOL,
                        source_manifest_sha256=app.FIT_SOURCE_MANIFEST_SHA, qualification_pair_sha256=app.FLOAT_SHORT_SHA,
                        seed=5201, fit_images=741, completed_epochs_per_arm=24, visits_per_arm=17856,
                        optimizer_updates_per_arm=4464, ema_updates_per_arm=4464,
                        arms={mode:{key:arm_refs[mode][key] for key in ("bytes","sha256")} for mode in app.ARMS},
                        common_ledger_sha256="1" * 64, identical_fields_sha256="2" * 64,
                        common_targets_total=32000, common_empty_visits=4, changed_mask_visits=12000,
                        selected_mask_stream_sha256={mode:arms[mode]["selected_mask_stream_sha256"] for mode in app.ARMS},
                        identical_nonmask_inputs_and_rng=True, identical_common_support=True, **app.FIT_ONLY)
            pair_ref = write("pair", pair)
            args = dict(native_arm_ref=arm_refs["native"], center_arm_ref=arm_refs["center"],
                        training_protocol=app.TRAINING_PROTOCOL, evaluation_protocol="6" * 40)
            actual_pair, actual_arms = app.read_training_pair(api, pair_ref, **args)
            self.assertEqual(actual_pair, pair)
            self.assertEqual(actual_arms, arms)
            with self.assertRaises(ValueError):
                api.read_pair(pair_ref, **args)
            pair["identical_common_support"] = False
            pair_ref = write("pair", pair)
            with self.assertRaisesRegex(ValueError, "floating/common E24 pair"):
                app.read_training_pair(api, pair_ref, **args)

    def test_float_export_requires_both_original_child_and_outer_success(self):
        api = self.modules["ssdd_phase_dev_admission"]
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)

            def write(name, document):
                path = root / name
                raw = json.dumps(document).encode()
                path.write_bytes(raw)
                return app.file_ref(path, sha256(raw).hexdigest())

            pair, evaluation, request, report = "1"*64, "2"*40, "3"*64, "4"*64
            stage_name = "EXPORT_NATIVE"
            stdout = write("OWNER.stdout", dict(status=app.REPORT_EVENT, stage=stage_name, request_sha256=request,
                                                report_sha256=report))
            child = write("OWNER_TERMINAL_PRIVATE.json", dict(original_os_wait=True, original_os_exit=0,
                          job_assigned=True, job_active_before_drain=0, job_active_after_wait=0,
                          stop_reason=None, owner={"pid": 13}, stdout_ref=stdout))
            join = dict(stage=stage_name, request_sha256=request, export_report_sha256=report,
                        original_child_terminal_sha256=child["sha256"])
            result = dict(exit_code=0, output=json.dumps(dict(status=app.EXPORT_OWNER_CLOSED, **join)))
            outer = dict(status=app.EXPORT_WAIT_STATUS, original_session_id=123, exact_tool_wait=result, **join)
            accepted = dict(status=app.EXPORT_ACCEPTED_STATUS, mode="native", evaluation_protocol=evaluation,
                            training_pair_sha256=pair, export_report_sha256=report, request_sha256=request,
                            original_child_terminal_ref=child, original_outer_wait_ref=write("wait.json", outer))
            record = dict(closure=dict(report_sha256=report, terminal_sha256=child["sha256"]),
                          acceptance_ref=write("accept.json", accepted))
            args = dict(pair_sha=pair, evaluation_protocol=evaluation)
            self.assertEqual(app.verify_export_acceptance(self.stage, api, record, "native", **args), 123)
            outer["exact_tool_wait"]["exit_code"] = 5
            accepted["original_outer_wait_ref"] = write("wait.json", outer)
            record["acceptance_ref"] = write("accept.json", accepted)
            with self.assertRaisesRegex(ValueError, "outer float export wait"):
                app.verify_export_acceptance(self.stage, api, record, "native", **args)


if __name__ == "__main__":
    FloatPrimary.code_root = Path(sys.argv[1]).resolve(strict=True)
    FloatPrimary.historical = Path(sys.argv[2]).resolve(strict=True)
    unittest.main(argv=[sys.argv[0]])
