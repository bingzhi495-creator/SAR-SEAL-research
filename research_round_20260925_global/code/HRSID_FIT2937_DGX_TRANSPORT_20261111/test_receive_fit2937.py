"""Synthetic USTAR transport fixtures for the closed HRSID FIT receiver."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from shutil import which
import subprocess
import sys
from tarfile import DIRTYPE, SYMTYPE, TarInfo, USTAR_FORMAT, open as tar_open
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import receive_fit2937 as stage


def digest(raw):
    return sha256(raw).hexdigest()


def make_fixture():
    images = {"train/P1_demo.png": b"one PNG bytes", "train/P2_demo.png": b"two PNG bytes"}
    index = b'{"images":[],"annotations":[]}'
    png_size = sum(map(len, images.values()))
    manifest = dict(native_index_bytes=len(index), native_index_sha256=digest(index),
                    selected_encoded_png_bytes=png_size, protocol="a" * 40,
                    archive_bytes=stage.ORIGINAL_ARCHIVE_BYTES, archive_central_sha256=stage.ORIGINAL_CENTRAL_SHA,
                    archive_sha256_receipt=stage.ORIGINAL_ARCHIVE_SHA, fit_source_index_sha256=stage.FIT_SOURCE_INDEX_SHA,
                    images=[dict(image_id=i, native_basename=Path(name).name, member="images/" + Path(name).name,
                                 encoded_sha256=digest(raw), encoded_bytes=len(raw), compressed_bytes=len(raw),
                                 decoded_rgb_sha256="b" * 64, zip_crc32="a" * 8, zip_header_offset=i)
                            for i, (name, raw) in enumerate(images.items(), 1)])
    manifest_raw = json.dumps(manifest).encode("utf-8")
    args = dict(MANIFEST_BYTES=len(manifest_raw), MANIFEST_SHA=digest(manifest_raw), INDEX_BYTES=len(index),
                INDEX_SHA=digest(index), FIT_IMAGES=len(images), PNG_BYTES=png_size)
    return manifest_raw, {stage.INDEX_NAME: index, **images}, args


def tar_bytes(manifest, files, *, extra=()):
    buffer = BytesIO()
    with tar_open(fileobj=buffer, mode="w", format=USTAR_FORMAT) as writer:
        def entry(name, raw):
            header = TarInfo(name)
            header.size = len(raw)
            writer.addfile(header, BytesIO(raw))

        entry(stage.MANIFEST_NAME, manifest)
        folder = TarInfo("train/")
        folder.type = DIRTYPE
        writer.addfile(folder)
        for name, raw in files.items():
            entry(name, raw)
        for name, value in extra:
            if value is None:
                alias = TarInfo(name)
                alias.type = SYMTYPE
                alias.linkname = "../outside"
                writer.addfile(alias)
            else:
                entry(name, value)
    return buffer.getvalue()


class ReceiverTest(unittest.TestCase):
    def test_ustar_members_verified_and_manifest_is_a_sibling(self):
        manifest, files, pins = make_fixture()
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "ready"
            with patch.multiple(stage, **pins):
                result = stage.receive(BytesIO(tar_bytes(manifest, files)), output)
            self.assertEqual(set(output.iterdir()), {output / "train"})
            self.assertEqual({p.relative_to(output).as_posix(): p.read_bytes()
                              for p in (output / "train").iterdir()}, files)
            self.assertEqual(stage.manifest_sidecar(output).read_bytes(), manifest)
            self.assertEqual(result["images"], 2)
            self.assertEqual(result["manifest_sha256"], digest(manifest))
            self.assertNotIn(str(root), json.dumps(result))


    def test_bad_rosters_and_trailing_stream_never_publish(self):
        manifest, files, pins = make_fixture()
        missing = {name: raw for name, raw in files.items() if name != "train/P2_demo.png"}
        streams = (
            ("missing", tar_bytes(manifest, missing), "Incomplete FIT roster"),
            ("link", tar_bytes(manifest, missing, extra=(("train/P2_demo.png", None),)), "Only original FIT regular"),
            ("traversal", tar_bytes(manifest, files, extra=(("train/../outside.png", b"x"),)), "Unexpected, duplicate"),
            ("duplicate", tar_bytes(manifest, files, extra=(("train/P1_demo.png", b"x"),)), "Unexpected, duplicate"),
            ("data", tar_bytes(manifest, files) + b"unexpected", "Unexpected data after"),
            ("wrong_digest", tar_bytes(manifest, dict(files, **{"train/P1_demo.png": b"x" * 13})),
             "FIT member SHA-256 mismatch"),
        )
        for name, stream, error in streams:
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                output = root / "ready"
                with patch.multiple(stage, **pins), self.assertRaisesRegex(ValueError, error):
                    stage.receive(BytesIO(stream), output)
                self.assertEqual(list(root.iterdir()), [])

    @unittest.skipUnless(which("tar.exe"), "Windows USTAR producer unavailable")
    def test_stock_windows_tar_producer_exact_flags(self):
        manifest, files, pins = make_fixture()
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            (source / "train").mkdir(parents=True)
            (source / stage.MANIFEST_NAME).write_bytes(manifest)
            for name, raw in files.items():
                (source / name).write_bytes(raw)
            process = subprocess.run(["tar.exe", "--format=ustar", "-cf", "-", "-C", str(source),
                                      stage.MANIFEST_NAME, "train"], check=True, capture_output=True,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            output = root / "ready"
            with patch.multiple(stage, **pins):
                result = stage.receive(BytesIO(process.stdout), output)
            self.assertEqual(result["images"], 2)
            self.assertEqual(stage.manifest_sidecar(output).read_bytes(), manifest)


if __name__ == "__main__":
    unittest.main()
