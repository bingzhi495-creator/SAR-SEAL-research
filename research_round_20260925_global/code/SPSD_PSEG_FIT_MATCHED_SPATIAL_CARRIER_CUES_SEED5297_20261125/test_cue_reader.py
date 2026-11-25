from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
import unittest

import numpy as np


MODULE = Path(__file__).with_name("cue_reader.py")


class CueJoinTests(unittest.TestCase):
    def test_same_coordinates_and_fingerprint_required(self):
        f = run_path(str(MODULE))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ranks = np.array([1,8],dtype=np.int16)
            offsets = np.array([0,2,3],dtype=np.int32)
            coords = np.array([[4,8],[3,6],[0,7]],dtype=np.int32)
            np.savez(root/"point.npz",native_ranks=ranks,point_offsets=offsets,signed_coordinates=coords,
                     features=np.zeros((3,2,14),dtype=np.float32),static_offsets=np.array([0,1,1],dtype=np.int32))
            original = (root/"point.npz").read_bytes()
            source_sha = f["digest"](original)
            point = dict(filename="point.npz",sha256=source_sha,bytes=len(original),native_archive_ordinal=4,
                         supported_candidates=2,paired_sites=3)

            def add(raw_coords,sha):
                np.savez(root/"cue.npz",native_archive_ordinal=np.int32(4),point_sha256=sha,
                         native_ranks=ranks,point_offsets=offsets,signed_coordinates=raw_coords,
                         cue=np.ones((3,2,6), dtype=np.float32))
                saved = (root/"cue.npz").read_bytes()
                return dict(filename="cue.npz",sha256=f["digest"](saved),bytes=len(saved))

            admitted = f["image_arrays"](root,root,point,add(coords,source_sha))
            self.assertEqual(admitted[0], {1:0,8:1})
            self.assertEqual(admitted[-1].shape,(3,2,6))
            with self.assertRaisesRegex(ValueError,"coordinate"):
                f["image_arrays"](root,root,point,add(coords+1,source_sha))
            with self.assertRaisesRegex(ValueError,"binding"):
                f["image_arrays"](root,root,point,add(coords,"0"*64))


if __name__ == "__main__":
    unittest.main()
