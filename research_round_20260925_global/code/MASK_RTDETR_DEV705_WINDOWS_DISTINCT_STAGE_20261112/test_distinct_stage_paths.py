from pathlib import PureWindowsPath
import unittest

import p392_windows_dev705_owner as subject


class DistinctStagePathsTest(unittest.TestCase):
    def test_staged_dev_images_never_alias_a_job_stage_on_windows(self):
        root = PureWindowsPath("example")
        path = root / subject.IMAGE_DATA_DIR
        for stage in subject.STAGES:
            self.assertNotEqual(path, root / stage)
            self.assertNotIn(path, (root / stage / "CONTROL").parents)


if __name__ == "__main__":
    unittest.main()
