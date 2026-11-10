from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from export_frozen_hrsid_ssdd import (
    TRAINING_PROTOCOLS, admit_source, load_module, manifest_items, read_jpeg, validate_image_zip,
)


def manifest(rows):
    return dict(
        schema="Root-RFDETR-DEV187-image-only-manifest-v1", population="internal_DEV187",
        model_execution=False, AP=False, TEST=False,
        expected_ids=[item["image_id"] for item in rows], items=rows,
    )


def item(image_id, path, payload):
    return dict(image_id=image_id, original_hw=[9, 11],
                image_ref=dict(path=path, bytes=len(payload), sha256=sha256(payload).hexdigest()))


class FrozenHrsidSsddExportTest(unittest.TestCase):
    def test_exact_logical_jpegs_allow_repeated_bytes_without_opening_other_members(self):
        pixels = b"synthetic JPEG bytes"
        rows = [
            item(7, r"D:\archive\valid\first.jpg", pixels),
            item(3, "/root/archived/valid/other.jpg", pixels),
        ]
        selected = manifest_items(manifest(rows), expected_images=2)
        self.assertEqual([item["image_id"] for item in selected], [7, 3])
        self.assertEqual([item["member"] for item in selected], ["valid/first.jpg", "valid/other.jpg"])
        archive_bytes = BytesIO()
        with ZipFile(archive_bytes, "w") as archive:
            archive.writestr("valid/other.jpg", pixels)
            archive.writestr("valid/first.jpg", pixels)
        with ZipFile(archive_bytes) as archive:
            validate_image_zip(archive, selected)
            self.assertEqual([read_jpeg(archive, entry) for entry in selected], [pixels, pixels])



    def test_image_zip_rejects_extra_gt_members_and_wrong_jpeg_bytes(self):
        data = b"correct"
        selected = manifest_items(manifest([item(1, "/x/valid/one.jpg", data)]), expected_images=1)
        for wrong in ("extra_gt", "wrong_bytes"):
            with self.subTest(wrong=wrong):
                archive_bytes = BytesIO()
                with ZipFile(archive_bytes, "w") as archive:
                    archive.writestr("valid/one.jpg", data if wrong == "extra_gt" else b"changed")
                    if wrong == "extra_gt":
                        archive.writestr("valid/_annotations.coco.json", b"{}")
                with ZipFile(archive_bytes) as archive:
                    if wrong == "extra_gt":
                        with self.assertRaisesRegex(ValueError, "Exact image-only ZIP member set"):
                            validate_image_zip(archive, selected)
                    else:
                        validate_image_zip(archive, selected)
                        with self.assertRaisesRegex(ValueError, "Exact original SSDD JPEG bytes"):
                            read_jpeg(archive, selected[0])

    def test_original_first_and_repeat_source_receipts_bind_transported_checkpoint_sha(self):
        root = Path(__file__).resolve().parent.parent
        source_paths = dict(
            phase=root / "HRSID_MASK_PHASE_E24_DEV705_20261109/hrsid_mask_phase_dev705.py",
            repeat=root / "HRSID_MASK_PHASE_REPEAT_DEV705_20261109/hrsid_mask_phase_repeat_dev705.py",
            filtered=root / "HRSID_COCO_E24_FILTERED_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705_filtered.py",
            base=root / "HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py",
            fit_audit=root / "HRSID_OFFICIAL_PNG_FIT_AUDIT_20261109/hrsid_png_fit_audit.py",
            recipe=root.parent / "configs/RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml",
        )

        def ref(path):
            data = path.read_bytes()
            return dict(path=str(path.resolve()), bytes=len(data), sha256=sha256(data).hexdigest())

        pins = {role: ref(path) for role, path in source_paths.items()}
        phase = load_module("_fixture_source_phase", source_paths["phase"], source_paths["phase"].read_bytes())
        repeat = load_module("_fixture_source_repeat", source_paths["repeat"], source_paths["repeat"].read_bytes())
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp).resolve()
            for seed in (5201, 5203):
                with self.subTest(seed=seed):
                    arm_refs = {}
                    markers = {}
                    protocol = TRAINING_PROTOCOLS[seed]
                    pair = dict(status=phase.PAIR_STATUS, training_protocol=protocol,
                                ordered_fit_visits_match=True, completed_epochs=24, visits_per_arm=70_560)
                    if seed != 5201:
                        pair["seed"] = seed
                    for mode in ("native", "center"):
                        marker = phase.expected_phase(mode)
                        if seed != 5201:
                            marker.update(protocol=protocol, seed=seed, phase_trainer_sha256=repeat.REPEAT_TRAINER_SHA)
                        markers[mode] = marker
                        check = sha256(f"{seed}/{mode}/checkpoint".encode()).hexdigest()
                        arm = dict(status=phase.ARM_STATUS, original_os_wait=True, original_os_exit=0, resource_stop=False,
                                   completed_epochs=24, optimizer_updates=17_640, ema_updates=17_640,
                                   target_filter_sha256=phase.FILTER_SHA, checkpoint_sha256=check, phase=marker,
                                   filtered_empty_rows_total=1)
                        for field in ("visit_ledger", "result", "stdout", "terminal", "phase_provenance"):
                            arm[f"{field}_sha256"] = sha256(f"{seed}/{mode}/{field}".encode()).hexdigest()
                        if seed != 5201:
                            arm[repeat.WITNESS_FIELD] = sha256(f"{seed}/{mode}/witness".encode()).hexdigest()
                        path = tmp / f"{seed}_{mode}.json"
                        path.write_text(json.dumps(arm), encoding="utf8")
                        arm_refs[f"{mode}_receipt"] = ref(path)
                        pair[f"{mode}_arm_receipt_sha256"] = arm_refs[f"{mode}_receipt"]["sha256"]
                        pair[f"{mode}_checkpoint_sha256"] = check
                    pair_path = tmp / f"{seed}_pair.json"
                    pair_path.write_text(json.dumps(pair), encoding="utf8")
                    lock = dict(seed=seed, mode="center", sources=pins, pair_receipt=ref(pair_path),
                                checkpoint=dict(sha256=pair["center_checkpoint_sha256"]), **arm_refs)
                    base = admit_source(lock)
                    checkpoint = {
                        "pytorch-lightning_version": "2.6.6", "epoch": 23, "global_step": 17_640,
                        "args": dict(seed=seed), phase.PHASE_MARKER: markers["center"],
                        base.ENDPOINT_KEY: dict(
                            protocol=protocol, epoch=24, actual_updates=17_640, native_index_sha256=base.FIT_INDEX_SHA,
                            coco_sha256=base.COCO_SHA, original_recipe_sha256=base.RECIPE_SHA,
                            initial_expanded_sha256=base.INITIAL_SHA, FIT_only=True,
                            exact_restart_trajectory_verified=False, rng_state=None,
                            filtered_empty_rows_total=1, target_filter_sha256=phase.FILTER_SHA,
                        ),
                    }
                    base.check_endpoint(checkpoint, protocol)
                    self.assertEqual(base.RECIPE, source_paths["recipe"])
                    self.assertEqual(base.FIT_AUDIT, source_paths["fit_audit"])
                    with self.assertRaisesRegex(ValueError, "independently accepted source arm"):
                        admit_source(dict(lock, checkpoint=dict(sha256=pair["native_checkpoint_sha256"])))


if __name__ == "__main__":
    unittest.main()
