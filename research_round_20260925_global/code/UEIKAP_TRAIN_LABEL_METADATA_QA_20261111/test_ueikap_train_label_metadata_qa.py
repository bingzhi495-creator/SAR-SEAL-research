"""Synthetic Range ZIP tests for the independent train metadata audit."""

from hashlib import sha256
from io import BytesIO
import json
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

from ueikap_train_label_metadata_qa import BoundedZip, examine, expected_paths, verify_positive_control


class RangeMemberTest(unittest.TestCase):
    def test_reads_exact_approved_member_and_never_fetches_neighbor_image(self):
        member = "UEIKAP_SAR/labels/train/selected.txt"
        stream = BytesIO()
        with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
            archive.writestr("UEIKAP_SAR/images_tifs/test/unrelated.tif", bytes(range(256)) * 400)
            archive.writestr(member, "0 0.1 0.1 0.2 0.1 0.2 0.2\n")
            selected_start = archive.getinfo(member).header_offset
        payload = stream.getvalue()
        central = payload.index(b"PK\x01\x02")
        eocd = payload.index(b"PK\x05\x06")
        requests = []

        def fetch(start, end):
            requests.append((start, end))
            return payload[start:end]

        with BoundedZip(fetch, len(payload), sha256(payload[central:eocd]).hexdigest(), (member,)) as archive:
            self.assertEqual(archive.read_member(member), b"0 0.1 0.1 0.2 0.1 0.2 0.2\n")
            before = list(requests)
            with self.assertRaisesRegex(ValueError, "outside the fixed TRAIN metadata scope"):
                archive.read_member("UEIKAP_SAR/images_tifs/test/unrelated.tif")
            self.assertEqual(requests, before)
        self.assertTrue(all(start >= selected_start for start, _ in requests))

    def test_scores_only_same_train_metadata_and_leaves_hull_completeness_uninferred(self):
        stem = "synthetic_FalseColorVVVH_000000000"
        vv_stem = "synthetic_VV_000000000"
        header = dict(name=f"UEIKAP_SAR/images_tifs/train/{stem}.tif", dimensions=[512, 512],
                      timestamp_proxy="S1A_20240301T170225", bounds_lonlat=[0., 0., 1., 1.])
        stream = BytesIO()
        image = dict(file_name="synthetic_VV_annotations_000000000.tif", id=1, width=512, height=512)
        category = dict(id=0, name="wake")
        annotation = dict(id=1, category_id=0, image_id=1,
                          segmentation=[[256, 256, 384, 256, 384, 384, 256, 384]])
        paths = expected_paths(stem)
        self.assertEqual(paths, {
            "txt": f"UEIKAP_SAR/labels/train/{stem}.txt",
            "coco": "UEIKAP_SAR/labels_jsons/synthetic_VV_annotations_000000000_coco.json",
            "ais": f"UEIKAP_SAR/AIS_interpolated_SAR_UEIKAP/{vv_stem}.tif.csv",
        })
        blobs = {
            "txt": "0 .125 .125 .25 .125 .25 .25 .125 .25\n1 .5 .5 .75 .5 .75 .75 .5 .75\n",
            "coco": json.dumps(dict(images=[image], categories=[category], annotations=[annotation])),
            "ais": "MMSI,LAT,LON,BaseDateTime\n0,0.4,0.6,2024-03-01T17:02:25Z\n",
        }
        with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
            archive.writestr(f"UEIKAP_SAR/labels/test/other_{stem}.txt", "must not read")
            archive.writestr(f"UEIKAP_SAR/images_tifs/train/{stem}.tif", "must not read")
            archive.writestr(f"UEIKAP_SAR/labels_jsons/test/synthetic_VV_annotations_000000000_coco.json", "must not read")
            archive.writestr(f"UEIKAP_SAR/labels_jsons/train/{stem}.json", "must not read")
            for role, path in paths.items():
                archive.writestr(path, blobs[role])
            control = {role: (len(blobs[role]), archive.getinfo(path).CRC,
                              sha256(blobs[role].encode("ascii")).hexdigest())
                       for role, path in paths.items()}
        payload = stream.getvalue()
        central = payload.index(b"PK\x01\x02")
        eocd = payload.index(b"PK\x05\x06")
        digest = sha256(payload[central:eocd]).hexdigest()
        with BoundedZip(lambda start, end: payload[start:end], len(payload), digest, paths.values()) as archive:
            self.assertEqual(verify_positive_control(archive, header, control),
                             {role: record[2] for role, record in control.items()})
            size, crc, _ = control["ais"]
            with self.assertRaisesRegex(ValueError, "positive-control member SHA-256 differs"):
                verify_positive_control(archive, header, dict(control, ais=(size, crc, "0" * 64)))
            row, = examine(archive, [header])
        self.assertEqual(row["status"], "TRAIN_METADATA_COMPARED")
        self.assertEqual(row["labels"]["wake_exact_nonempty_raster_matches"], 1)
        self.assertEqual(row["labels"]["class0_wake_positive_iou_pairs"], 0)
        self.assertFalse(row["ais"]["point_in_raster_class0"])
        self.assertTrue(row["ais"]["point_in_raster_class1"])
        self.assertNotIn(stem, json.dumps(row))

    def test_another_split_with_same_name_disqualifies_flat_metadata_before_read(self):
        stem = "synthetic_FalseColorVVVH_000000000"
        paths = expected_paths(stem)
        stream = BytesIO()
        with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
            for path in paths.values():
                archive.writestr(path, "must not read")
            archive.writestr(f"UEIKAP_SAR/labels/val/{stem}.txt", "must not read")
        payload = stream.getvalue()
        central = payload.index(b"PK\x01\x02")
        eocd = payload.index(b"PK\x05\x06")
        requests = []

        def fetch(start, end):
            requests.append((start, end))
            return payload[start:end]

        with BoundedZip(fetch, len(payload), sha256(payload[central:eocd]).hexdigest(), paths.values()) as archive:
            header = dict(name=f"UEIKAP_SAR/images_tifs/train/{stem}.tif")
            row, = examine(archive, [header])
            self.assertEqual(row["unresolved"], {"other_split_matches": True})
        self.assertTrue(all(start >= central for start, _ in requests))


if __name__ == "__main__":
    unittest.main()
