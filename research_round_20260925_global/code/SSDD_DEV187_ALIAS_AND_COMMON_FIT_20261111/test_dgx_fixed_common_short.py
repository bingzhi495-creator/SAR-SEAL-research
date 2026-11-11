import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import dgx_fixed_common_short as subject


class DgxCommonShortTest(unittest.TestCase):
    def test_external_wait_requires_original_birth_and_ordered_marker(self):
        owner = dict(pid=300, created=2.5, boot_id="fixture")
        request = "a"*64
        summary = "b"*64
        events = [subject.arm_start(mode, request) for mode in subject.MODES]
        events.append(subject.pending_event(request, summary))
        capture = dict(status=subject.WAIT_STATUS, owner=owner, request_sha256=request,
                       original_os_wait=True, original_os_exit=0, process_group_closed=True,
                       combined_stdout="\n".join(map(json.dumps, events))+"\n")
        subject.verify_external_wait(capture, owner, request, summary, lambda _: False)
        with self.assertRaisesRegex(ValueError, "still live"):
            subject.verify_external_wait(capture, owner, request, summary, lambda _: True)
        reversed_order = "\n".join(map(json.dumps, events[::-1]))+"\n"
        with self.assertRaisesRegex(ValueError, "ordered"):
            subject.verify_external_wait(dict(capture, combined_stdout=reversed_order), owner, request, summary, lambda _: False)
        with self.assertRaisesRegex(ValueError, "original owner wait"):
            subject.verify_external_wait(dict(capture, original_os_exit=False), owner, request, summary, lambda _: False)

    def test_stock_scope_and_common_bytes_must_match_separately(self):
        base = dict(differing_prefilter_count_visits=0, differing_prefilter_labels_visits=0,
                    differing_prefilter_box_visits=0, differing_postfilter_count_visits=0,
                    differing_postfilter_box_visits=0, differing_postfilter_mask_visits=13)
        stats = dict(changed_mask_visits=13)
        subject.verify_common_pair((b"a", "1", stats), (b"a", "2", stats), base)
        with self.assertRaisesRegex(ValueError, "Ordered"):
            subject.verify_common_pair((b"a", "1", stats), (b"b", "2", stats), base)
        with self.assertRaisesRegex(ValueError, "nonmask"):
            subject.verify_common_pair((b"a", "1", stats), (b"a", "2", stats),
                                       dict(base, differing_prefilter_box_visits=1))

    def test_exact_source_preflight_does_not_open_stock_binding(self):
        local = Path(__file__).resolve().parents[1]
        release = local / "github_release_20261011/research_round_20260925_global"

        def ref(path):
            raw = path.read_bytes()
            return dict(path=path.as_posix(), bytes=len(raw), sha256=subject.digest(raw))

        with TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            source = Path(subject.__file__).resolve()
            cfg = dict(status=subject.REQUEST_STATUS, training_protocol="1"*40, implementation_commit="2"*40,
                python=sys.executable, code_root=(release / "code").as_posix(),
                work_root=(parent / "unused_study").as_posix(), lease_directory=parent.as_posix(),
                worker=ref(source), stock_binding_sha256="3"*64,
                stock_owner=ref(local / "P242_SSDD_MASK_PHASE_DGX_SHORT_TRANSPORT_PRIVATE_20261110_01/OWNER.py"),
                linux_guard=ref(release / "code/HRSID_BC_DGX_SHORT_PLATFORM_20261111/linux_bc_short_owner.py"),
                common_worker=ref(local / "P333_SSDD_WINDOWS_COMMON_SHORT_DRAFT_20261111_01/p333_common_short_worker.py"),
                p330_report=ref(release / "results/SSDD_FIT_FIXED_COMMON_NATIVE_DATASET_ACTUAL_SAFE_20261111.json"))
            request = parent / "fixture.json"
            raw = subject.encode(cfg)
            request.write_bytes(raw)
            actual, output, _, code = subject.get_request(request, subject.digest(raw))
            actual_deps = subject.dependencies(actual, code)
            self.assertEqual(actual_deps.stock.HOST_FLOOR, 40 * subject.GIB)
            self.assertFalse(output.exists())
            self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
