from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ssdd_phase_dev_admission as admission


TRAIN_PROTOCOL = "a" * 40
EVAL_PROTOCOL = "b" * 40


def expected_phase(mode):
    return dict(training_protocol=TRAIN_PROTOCOL, stage="e24", mask_resize_mode=mode, seed=5201,
                initialization_seed=5201, num_queries=100, num_select=100, query_groups=13, class_head_slots=2,
                coco_asset_sha256="a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b",
                initial_online_sha256="40524fc63797d2fcab7b50b6ee1f041f9429df73e505cb9e647a5a8cdf6f2e2f",
                recipe_sha256="ba9508d9acab5ff68b10db8a1aff5d9001868d6dda00e39f299d6f5cc1b8011b",
                fit_ids_sha256="c" * 64, source_manifest_sha256="d" * 64)


class DevAdmissionTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def write(self, name, value):
        raw = (json.dumps(value, sort_keys=True) + "\n").encode()
        path = self.root / name
        path.write_bytes(raw)
        return dict(path=str(path), bytes=len(raw), sha256=sha256(raw).hexdigest())

    def synthetic_checkpoint(self, legacy, phase):
        checkpoint = dict(
            epoch=23, global_step=4464, **{"pytorch-lightning_version": "2.6.6"},
            args={"seed": 5201}, model_config=dict(num_queries=100, num_select=100, group_detr=13, resolution=576,
                                                  num_classes=1, segmentation_head=True),
        )
        checkpoint["sar_seal_ssdd_mask_resize_phase_v1"] = phase
        checkpoint[legacy.BOUNDARY_KEY] = legacy.endpoint_marker(24, phase["recipe_sha256"])
        checkpoint["loops"] = {"fit_loop": {
            "epoch_progress": {key: dict(ready=24, started=24, processed=24, completed=23)
                               for key in ("current", "total")},
            "epoch_loop.batch_progress": dict(is_last_batch=True, **{
                key: {name: count for name in ("ready", "started", "processed", "completed")}
                for key, count in (("current", 744), ("total", 17856))
            }),
            "epoch_loop.automatic_optimization.optim_progress": {"optimizer": {"step": {
                key: dict(ready=count, completed=count) for key, count in (("current", 186), ("total", 4464))
            }}},
        }}
        lr = .01 * legacy.schedule_factor(4464)
        checkpoint["optimizer_states"] = [dict(state={1: dict(step=4464, exp_avg=None, exp_avg_sq=None)},
                                               param_groups=[dict(lr=lr)])]
        checkpoint["lr_schedulers"] = [dict(last_epoch=4464, base_lrs=[.01], _last_lr=[lr])]
        checkpoint["callbacks"] = {"RFDETREMACallback": dict(latest_update_step=4464,
            average_model_state_dict={"n_averaged": 4464, "module.model.class_embed.weight": None})}
        return checkpoint

    def test_phase_hook_preserves_native_boundary_and_seed_admission(self):
        deps = admission.load_sources()
        phase = expected_phase("center")
        checkpoint = self.synthetic_checkpoint(deps["long"], phase)
        hook = admission.bind_archive_validator({"phase": phase}, long_worker=deps["long"])
        kwargs = dict(boundary_api=deps["boundary"], recipe_sha256=phase["recipe_sha256"])
        hook(checkpoint, 24, **kwargs)
        for key, value, reason in (("args", {"seed": 5202}, "saved RF-DETR seed"),
                                   ("global_step", 4463, "cursor"),
                                   ("sar_seal_ssdd_mask_resize_phase_v1", expected_phase("native"), "phase marker")):
            with self.subTest(key=key), self.assertRaisesRegex((ValueError, RuntimeError), reason):
                altered = dict(checkpoint, **{key: value})
                hook(altered, 24, **kwargs)

    def pair_receipts(self, *, mutate_center=lambda arm: None):
        pair = dict(status="SSDD_MASK_PHASE_E24_PAIR_INDEPENDENTLY_ACCEPTED_V1", training_protocol=TRAIN_PROTOCOL,
                    seed=5201, qualification_pair_sha256="e" * 64, source_manifest_sha256="d" * 64,
                    fit_ids_sha256="c" * 64, initial_online_sha256=expected_phase("native")["initial_online_sha256"],
                    recipe_sha256=expected_phase("native")["recipe_sha256"],
                    coco_asset_sha256=expected_phase("native")["coco_asset_sha256"],
                    completed_epochs=24, visits_per_arm=17856, optimizer_updates_per_arm=4464,
                    ema_updates_per_arm=4464, ordered_fit_visits_match=True, ordered_fit_visits_sha256="f" * 64,
                    FIT=True, DEV=False, TEST=False, AP=False)
        refs = {}
        for mode in ("native", "center"):
            checkpoint = mode.encode("ascii")
            arm = dict(status="SSDD_MASK_PHASE_E24_ARM_INDEPENDENTLY_ACCEPTED_V1", phase=expected_phase(mode),
                       qualification_pair_sha256=pair["qualification_pair_sha256"], original_os_wait=True,
                       original_os_exit=0, resource_stop=False, worker_closed=True, completed_epochs=24,
                       visits=17856, optimizer_updates=4464, ema_updates=4464, checkpoint_sha256=sha256(checkpoint).hexdigest(),
                       checkpoint_bytes=len(checkpoint), filtered_empty_rows_total=1, FIT=True, DEV=False, TEST=False, AP=False)
            for key in ("result", "epoch_audit", "visit_ledger", "stdout", "terminal"):
                arm[key + "_sha256"] = sha256((mode + key).encode()).hexdigest()
            if mode == "center":
                mutate_center(arm)
            ref = self.write(mode + "_ARM_SAFE.json", arm)
            refs[f"{mode}_arm_ref"] = ref
            pair[f"{mode}_arm_receipt_sha256"] = ref["sha256"]
            pair[f"{mode}_checkpoint_sha256"] = arm["checkpoint_sha256"]
        return self.write("PAIR_SAFE.json", pair), refs

    def test_pair_requires_both_distinct_closed_same_seed_arms(self):
        pair_ref, refs = self.pair_receipts()
        protocols = dict(training_protocol=TRAIN_PROTOCOL, evaluation_protocol=EVAL_PROTOCOL)
        pair, arms = admission.read_pair(pair_ref, **refs, **protocols)
        self.assertEqual(pair["native_checkpoint_sha256"], arms["native"]["checkpoint_sha256"])
        for mutation, reason in (
            (lambda arm: arm.update(worker_closed=False), "closed SSDD Q100 E24 arm"),
            (lambda arm: arm["phase"].update(seed=5202), "phase marker identity"),
            (lambda arm: arm.update(qualification_pair_sha256="0" * 64), "closed SSDD Q100 E24 arm"),
        ):
            with self.subTest(reason=reason):
                pair_ref, refs = self.pair_receipts(mutate_center=mutation)
                with self.assertRaisesRegex(ValueError, reason):
                    admission.read_pair(pair_ref, **refs, **protocols)

    def image_manifest(self):
        valid = self.root / "valid"
        valid.mkdir()
        jpeg = b"synthetic duplicate encoding"
        items = []
        for image_id, name, hw in ((53, "image_53.jpg", [39, 25]), (21, "image_21.jpg", [39, 25])):
            (valid / name).write_bytes(jpeg)
            items.append(dict(image_id=image_id, original_hw=hw, image_ref=dict(
                path="/original/valid/" + name, bytes=len(jpeg), sha256=sha256(jpeg).hexdigest(),
            )))
        manifest = dict(schema="Root-RFDETR-DEV187-image-only-manifest-v1", population="internal_DEV187",
                        expected_ids=[53, 21], items=items, model_execution=False, AP=False, TEST=False)
        return self.write("manifest.json", manifest), manifest

    def test_windows_canonical_archive_reader_and_rebased_order_preserve_bytes(self):
        deps = admission.load_sources()
        archive = self.root / "checkpoint_23.ckpt"
        archive.write_bytes(b"synthetic native")
        ref = dict(path=archive.as_posix(), bytes=16, sha256=sha256(archive.read_bytes()).hexdigest())
        fake_torch = SimpleNamespace(load=lambda stream, **kwargs: (stream.read(), kwargs))
        self.assertEqual(deps["worker"].read_archive(ref, fake_torch, deps["pilot"]),
                         (b"synthetic native", {"map_location": "cpu", "weights_only": True}))
        manifest_ref, original = self.image_manifest()
        with patch.multiple(admission, DEV_IMAGES=2, IMAGE_MANIFEST_SHA=manifest_ref["sha256"]):
            rows, ids = admission.load_image_manifest(manifest_ref, self.root)
        rebased = deepcopy(original["items"])
        for row in rebased:
            filename = row["image_ref"]["path"].split("/")[-1]
            row["image_ref"]["path"] = str(self.root / "valid" / filename)
        self.assertEqual((rows, ids), (rebased, [53, 21]))
        self.assertEqual(rows[0]["image_ref"]["sha256"], rows[1]["image_ref"]["sha256"])

    def test_export_routes_original_hook_and_stops_wrong_archive_before_dev_manifest(self):
        pair_ref, refs = self.pair_receipts()
        manifest_ref, _ = self.image_manifest()
        archive = self.root / "checkpoint_23.ckpt"
        archive.write_bytes(b"native")
        archive_ref = dict(path=str(archive), bytes=6, sha256=sha256(b"native").hexdigest())
        kwargs = dict(**refs, training_protocol=TRAIN_PROTOCOL, evaluation_protocol=EVAL_PROTOCOL, mode="native",
                      archive_ref=archive_ref, dev_manifest_ref=manifest_ref, dev_root=self.root, model_config=object(),
                      train_config=object(), output_directory=self.root / "output")
        with patch.object(admission, "load_image_manifest", side_effect=AssertionError("premature DEV access")):
            with self.assertRaisesRegex(ValueError, "accepted SSDD checkpoint_23.ckpt"):
                admission.export_arm(pair_ref, **dict(kwargs, mode="center"))

        deps = admission.load_sources()
        original_result = dict(status="NATIVE_E24_EMA_DEV187_EXPORT_COMPLETE_PENDING_ROOT", fixed_epoch=24,
                               optimizer_updates=4464, checkpoint_sha256=archive_ref["sha256"],
                               recipe_sha256=expected_phase("native")["recipe_sha256"], image_count=2, native_count_total=200,
                               full_DEV=True, qualified=False, terminal_self_certified=False, GT_read=False, TEST=False,
                               AP=False, training=False)

        def original_worker(received, items, ids, model, train, *, validate_archive_progress, boundary_api,
                            recipe_sha256, **_):
            self.assertEqual(received, archive_ref)
            self.assertEqual(ids, [53, 21])
            self.assertEqual([row["image_id"] for row in items], ids)
            self.assertIs(model, kwargs["model_config"])
            self.assertIs(train, kwargs["train_config"])
            checkpoint = self.synthetic_checkpoint(deps["long"], expected_phase("native"))
            validate_archive_progress(checkpoint, 24, boundary_api=boundary_api, recipe_sha256=recipe_sha256)
            return original_result

        with patch.multiple(admission, DEV_IMAGES=2, IMAGE_MANIFEST_SHA=manifest_ref["sha256"]), \
                patch.object(admission, "load_sources", return_value=deps), \
                patch.object(deps["worker"], "run_e24_dev_export", side_effect=original_worker):
            report = admission.export_arm(pair_ref, **kwargs)
        self.assertIs(report["result"], original_result)
        self.assertEqual(report["training_seed"], 5201)
        self.assertEqual(report["training_pair_sha256"], pair_ref["sha256"])
        self.assertEqual(report["training_arm_sha256"], refs["native_arm_ref"]["sha256"])


if __name__ == "__main__":
    unittest.main()
