"""One tiny standard-library scheduling test; no ML or scientific inputs."""
import unittest

import shared_encoder_update as candidate


class MetadataTest(unittest.TestCase):
    def test_original_query_order_and_invalid_metadata(self):
        self.assertEqual(candidate.query_order(3,2),[0,1,2])
        for G,step in ((0,1),(101,1),(True,1),(1,0),(1,True)):
            with self.assertRaises(ValueError):
                candidate.query_order(G,step)


if __name__ == '__main__':
    unittest.main()
