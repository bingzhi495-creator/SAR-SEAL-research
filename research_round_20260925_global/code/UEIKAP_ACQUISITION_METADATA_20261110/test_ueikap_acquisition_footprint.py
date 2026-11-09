"""Synthetic tests for separate historical and family footprint criteria."""
from math import radians, sin
import unittest

from ueikap_acquisition_footprint import classify_chip


STAMP = "20240827T171424"
PROXY = f"S1A_{STAMP}"
BOUNDS = [0, 0, 10, 10]


def name(orbit, take, product, start=STAMP, mission="S1A", suffix=""):
    return f"{mission}_IW_GRDH_1SDV_{start}_20240827T172000_{orbit}_{take}_{product}{suffix}"


def rectangle(west, south, east, north):
    return {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north],
                                                  [west, north], [west, south]]]}


class AcquisitionFootprintTests(unittest.TestCase):
    def test_primary_requires_single_exact_product_and_secondary_unions_only_same_family(self):
        original = name("053214", "0672CD", "1111")
        left = name("053215", "0672CD", "2222")
        right = name("053215", "0672CD", "3333", start="20240827T171624")
        other = name("053216", "0672CD", "4444")
        other_take = name("053216", "0672CE", "5555")
        near = name("053217", "0672CD", "6666", start="20240827T171224")
        too_late = name("053218", "0672CD", "7777", start="20240827T171625")
        wrong_mission = name("053219", "0672CD", "8888", mission="S1B")
        result = classify_chip(BOUNDS, PROXY, [
            (original + ".SAFE", rectangle(0, 0, 9.5, 10)),
            (left + "_COG.SAFE", rectangle(0, 0, 5.2, 10)),
            (right + "-GRD_HD", rectangle(5, 0, 10, 10)),
            (left, rectangle(0, 0, 5.2, 10)),
            (other, rectangle(0, 0, 6, 10)),
            (other_take, rectangle(4, 0, 10, 10)),
            (near, rectangle(0, 0, 10, 10)),
            (too_late, rectangle(0, 0, 10, 10)),
            (wrong_mission, rectangle(0, 0, 10, 10)),
        ], set())
        self.assertEqual(result["primary_products"], [original])
        self.assertEqual(result["primary_families"], ["S1A_053214_0672CD"])
        self.assertEqual(result["secondary_families"], ["S1A_053214_0672CD", "S1A_053215_0672CD", "S1A_053217_0672CD"])
        families = result["family_overlaps"]
        family = families["S1A_053215_0672CD"]
        self.assertEqual(family["products"], [left, right])
        self.assertAlmostEqual(family["planar_fraction"], 1.)
        self.assertAlmostEqual(family["equal_area_fraction"], 1.)
        self.assertNotIn("S1A_053218_0672CD", families)
        self.assertNotIn("S1B_053219_0672CD", families)



    def test_reserved_contact_distinguishes_positive_area_from_line_point_and_disjoint(self):
        a, b, c, d = (f"05321{i}" for i in range(4))
        family_a, family_b, family_c, family_d = (f"S1A_{orbit}_0672CD" for orbit in (a, b, c, d))
        result = classify_chip(BOUNDS, PROXY, [
            (name(a, "0672CD", "AAAA"), rectangle(10, 0, 12, 10)),
            (name(a, "0672CD", "BBBB"), rectangle(9.75, 0, 12, 10)),
            (name(b, "0672CD", "CCCC"), rectangle(10, 2, 12, 3)),
            (name(c, "0672CD", "DDDD"), rectangle(10, 10, 12, 12)),
            (name(d, "0672CD", "EEEE"), rectangle(11, 0, 12, 1)),
        ], {family_a, family_b, family_c, family_d})
        self.assertEqual(result["reserved_positive_area_families"], [family_a])
        self.assertEqual(result["reserved_boundary_only_families"], [family_b, family_c])
        self.assertEqual(result["primary_products"], [])
        self.assertEqual(result["secondary_families"], [])
        self.assertAlmostEqual(result["family_overlaps"][family_a]["planar_fraction"], .025)
        self.assertFalse(result["family_overlaps"][family_a]["boundary_only"])
        self.assertTrue(result["family_overlaps"][family_b]["boundary_only"])
        self.assertFalse(result["family_overlaps"][family_d]["boundary_only"])

    def test_spherical_secondary_and_planar_primary_differ_and_multipolygon_holes_count(self):
        first = name("053214", "0672CD", "1111")
        second = name("053215", "0672CD", "2222")
        lower = rectangle(0, 0, 1, 56.4)
        outer = rectangle(0, 0, .4, 60)["coordinates"][0]
        hole = rectangle(.1, 10, .3, 20)["coordinates"][0]
        upper = rectangle(.6, 0, 1, 60)["coordinates"][0]
        result = classify_chip([0, 0, 1, 60], PROXY, [
            (first, lower),
            (second, {"type": "MultiPolygon", "coordinates": [[outer, hole], [upper]]}),
        ], set())
        self.assertEqual(result["primary_products"], [])
        self.assertEqual(result["secondary_families"], ["S1A_053214_0672CD"])
        families = result["family_overlaps"]
        self.assertAlmostEqual(families["S1A_053214_0672CD"]["planar_fraction"], .94)
        self.assertAlmostEqual(families["S1A_053214_0672CD"]["equal_area_fraction"],
                               sin(radians(56.4)) / sin(radians(60)))
        self.assertAlmostEqual(families["S1A_053215_0672CD"]["equal_area_fraction"],
                               .8 - .2 * (sin(radians(20)) - sin(radians(10))) / sin(radians(60)))

    def test_rejects_invalid_and_antimeridian_footprints_and_inconsistent_product_duplicates(self):
        product = name("053214", "0672CD", "1111")
        bow_tie = {"type": "Polygon", "coordinates": [[[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]]]}
        open_ring = {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2]]]}
        split = {"type": "MultiPolygon", "coordinates": [rectangle(-180, 0, -178, 2)["coordinates"],
                                                         rectangle(178, 0, 180, 2)["coordinates"]]}
        for geojson, reason in (
            (bow_tie, "Invalid or empty footprint geometry"),
            (open_ring, "unclosed"),
            (rectangle(179, 0, -179, 1), "Antimeridian"),
            (split, "Antimeridian"),
            (rectangle(0, 88, 2, 91), "Out-of-range"),
            (rectangle(0, 0, float("nan"), 2), "Out-of-range"),
            (rectangle(0, 0, 10 ** 500, 2), "Out-of-range"),
        ):
            with self.subTest(reason=reason, geometry=geojson):
                with self.assertRaisesRegex(ValueError, reason):
                    classify_chip(BOUNDS, PROXY, [(product, geojson)], set())
        with self.assertRaisesRegex(ValueError, "Conflicting footprints"):
            classify_chip(BOUNDS, PROXY, [(product, rectangle(0, 0, 10, 10)),
                                         (product + "_COG.SAFE", rectangle(0, 0, 9, 9))], set())
        with self.assertRaisesRegex(ValueError, "antimeridian-spanning"):
            classify_chip([170, 0, -170, 1], PROXY, [], set())
        with self.assertRaisesRegex(ValueError, "full Sentinel-1"):
            classify_chip(BOUNDS, PROXY, [(product.replace("_GRDH_", "_GRDM_"), rectangle(0, 0, 1, 1))], set())


if __name__ == "__main__":
    unittest.main()
