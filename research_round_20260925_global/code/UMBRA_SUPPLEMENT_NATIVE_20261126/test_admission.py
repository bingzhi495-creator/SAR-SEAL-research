"""Synthetic checks for the supplemental admission and native write workflow."""
from argparse import Namespace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from uuid import UUID
import unittest

from admission import read_admission


def write(path, value):
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


def mock_case(root, ids=range(101, 119)):
    args = Namespace(manifest=root/"cohort.json", metadata_result=root/"metadata.json",
                     metadata_terminal=root/"terminal.json", published_lock=root/"LOCK.md",
                     published_commit="a"*40)
    items = [dict(window_id=f"UMB-{j:03d}", collect_id=str(UUID(int=j)), task_id=str(UUID(int=1000+j)),
                  author_url=f"https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/"
                             f"sar-data/tasks/ship_detection_testdata/synthetic/{j}_GEC.tif",
                  etag='"synthetic"', object_bytes=2**25) for j in ids]
    args.manifest_sha256 = write(args.manifest, dict(schema="umbra_gec_supplement_v1", items=items))
    args.metadata_result_sha256 = write(args.metadata_result, dict(status="UMBRA_SUPPLEMENT_METADATA_LOCKED",
        cohort_private_sha256=args.manifest_sha256, selected_scene_count=len(items), image_reads=0,
        label_reads=0, model_forwards=0, worker_sha256="b"*64, protocol_sha256="c"*64))
    args.metadata_terminal_sha256 = write(args.metadata_terminal, dict(phase="verify", actual_exit=0,
        output_footer_verified=True, result_sha256=args.metadata_result_sha256, worker_sha256="b"*64))
    args.published_lock.write_text("# Supplemental frozen roster\n" + args.manifest_sha256 + "\n", encoding="utf-8")
    args.published_lock_sha256 = sha256(args.published_lock.read_bytes()).hexdigest()
    return args


class NativeAdmission(unittest.TestCase):
    def test_published_closed_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            args = mock_case(Path(directory))
            admission = read_admission(args)
            self.assertEqual(len(admission.items), 18)
            self.assertEqual([row["window_id"] for row in admission.items],
                             [f"UMB-{j:03d}" for j in range(101, 119)])
            self.assertEqual(admission.binding["published_commit"], "a"*40)

    def test_specific_metadata_closure_rejects_unverified_footer(self):
        with tempfile.TemporaryDirectory() as directory:
            args = mock_case(Path(directory))
            terminal = json.loads(args.metadata_terminal.read_bytes())
            terminal["output_footer_verified"] = False
            args.metadata_terminal_sha256 = write(args.metadata_terminal, terminal)
            with self.assertRaisesRegex(ValueError, "Observed natural metadata"):
                read_admission(args)

    def test_stops_unpublished_manifest_and_changed_order(self):
        with tempfile.TemporaryDirectory() as directory:
            args = mock_case(Path(directory), ids=reversed(range(101, 119)))
            with self.assertRaisesRegex(ValueError, "Exact eligible supplemental cohort"):
                read_admission(args)
        with tempfile.TemporaryDirectory() as directory:
            args = mock_case(Path(directory))
            args.published_lock_sha256 = write(args.published_lock, dict(unother_manifest="d"*64))
            with self.assertRaisesRegex(ValueError, "Published exact pre-pixel cohort lock"):
                read_admission(args)


if __name__ == "__main__":
    unittest.main()
