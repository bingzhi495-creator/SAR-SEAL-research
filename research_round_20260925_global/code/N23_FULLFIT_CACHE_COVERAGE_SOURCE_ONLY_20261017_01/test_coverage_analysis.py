# encoding: utf-8
"""UNIT_FIXTURE: only stdlib and synthetic lists, no cache/Torch/NumPy/models."""
import copy
import importlib.util
from pathlib import Path
import unittest

P=Path(__file__).with_name("coverage_analysis.py")
S=importlib.util.spec_from_file_location("n23_coverage_unit",P)
coverage=importlib.util.module_from_spec(S)
S.loader.exec_module(coverage)


def fixture(index):
    image_id=1000+index
    def batch(g,q,n,omega,weight,first=0):
        return dict(dataset_index=index,image_id=image_id,query_id=q,GT_index=g,
            GT_annotation_id=(100 if index==0 else 200)+g,n=n,
            sample_positions=list(range(first,first+n)),
            flat_indices=list(range(n//2))*2,
            Omega_flag=[True]*omega+[False]*(n-omega),C_flag=[True]*n,
            importance_weights=[weight]*n)
    if index==0:
        batches=[batch(0,10,1024,0,.5),batch(1,20,1024,256,1.5,1024)]
        unsampled=[dict(query_id=22,GT_index=2,GT_annotation_id=102,K=0)]
        GT=3
    else:
        batches=[batch(0,30,2048,2048,2.0)]
        unsampled=[dict(query_id=31,GT_index=1,GT_annotation_id=201,K=3)]
        GT=2
    return dict(schema="N23-shared-original-C-matched-query-batches-v1",batches=batches,
        audit=dict(original_draws=4096,retained_matched_draws=2048,dropped_background_draws=2048,
            ignored_draws=0,real_GT_instances=GT,nonempty_matched_batches=len(batches),
            unsampled_real_GT=unsampled,matched_query_ids=[b["query_id"] for b in batches],
            Omega_editable_draws=sum(sum(b["Omega_flag"]) for b in batches),C_editable_draws=2048,
            K0_queries_present_in_original_sample=False))


class CoverageTests(unittest.TestCase):
    def test_exact_coverage_K0_sparse_Omega_duplicates_and_weights(self):
        first=coverage.analyze_image(0,1000,fixture(0))
        second=coverage.analyze_image(1,1001,fixture(1))
        result=coverage.aggregate_images([first,second],expected_images=2,expected_real_GT=5)
        s=result["summary"]
        self.assertEqual(s["real_GT_instances"],5)
        self.assertEqual((s["unsampled_GT_K0"],s["unsampled_GT_K_positive"]),(1,1))
        self.assertEqual(s["nonempty_matched_batches"],3)
        self.assertEqual(s["Omega_zero_point_batches"],1)
        self.assertEqual((s["matched_draws"],s["background_draws"]),(4096,4096))
        self.assertEqual(s["Omega_points"],2304)
        self.assertEqual(s["Omega_fraction_of_matched_draws"],.5625)
        self.assertEqual(s["matched_duplicate_draws"],2048)
        self.assertEqual(s["matched_unique_query_pixel_pairs"],2048)
        self.assertEqual(s["matched_weight_stats"]["sum"],6144.0)
        self.assertEqual(s["Omega_weight_sum"],4480.0)
        self.assertFalse(s["background_duplicate_statistics_available"])
        self.assertFalse(s["importance_weights_claimed_unbiased_after_GT_reduction"])
        self.assertEqual(s["A_zero_data_gradient_step_rule"],"UNDETERMINED")
        self.assertEqual(len(result["images"][0]["GT_records"]),3)
        GT=result["images"][0]["GT_records"]
        self.assertEqual(GT[0]["Omega_unique_query_pixel_pairs"],0)
        self.assertIsNone(GT[0]["Omega_coordinate_ESS"])
        self.assertEqual(GT[0]["matched_coordinate_ESS"],512.0)
        self.assertEqual(GT[1]["Omega_unique_query_pixel_pairs"],256)
        self.assertEqual(GT[1]["Omega_coordinate_ESS"],256.0)
        self.assertEqual(s["Omega_unique_query_pixel_pairs"],1280)
        self.assertAlmostEqual(s["matched_coordinate_ESS"],6144.0**2/21504.0)
        self.assertAlmostEqual(s["Omega_coordinate_ESS"],4480.0**2/16960.0)


    def test_all_real_GT_unsampled_has_no_NaN_or_invented_steps(self):
        result=fixture(0)
        result["batches"]=[]
        result["audit"].update(retained_matched_draws=0,dropped_background_draws=4096,
            nonempty_matched_batches=0,matched_query_ids=[],Omega_editable_draws=0,C_editable_draws=0,
            unsampled_real_GT=[dict(query_id=q,GT_index=g,GT_annotation_id=100+g,K=0)
                for g,q in enumerate((10,20,22))])
        image=coverage.analyze_image(0,1000,result)
        final=coverage.aggregate_images([image],expected_images=1,expected_real_GT=3)
        summary=final["summary"]
        self.assertEqual(summary["unsampled_GT_K0"],3)
        self.assertEqual(summary["nonempty_matched_batches"],0)
        self.assertIsNone(summary["Omega_fraction_of_matched_draws"])
        self.assertIsNone(summary["matched_coordinate_ESS"])
        self.assertIsNone(summary["Omega_coordinate_ESS"])
        self.assertEqual(summary["Omega_unique_query_pixel_pairs"],0)
        self.assertEqual(summary["matched_draws_per_real_GT_histogram"],{"0":3})
        import json
        json.dumps(final,allow_nan=False)

    def test_inconsistent_counts_identity_slots_and_weights_are_rejected(self):
        mutations=(
            lambda d:d["audit"].update(original_draws=4095),
            lambda d:d["audit"].update(retained_matched_draws=2049),
            lambda d:d["audit"].update(real_GT_instances=4),
            lambda d:d["audit"].update(Omega_editable_draws=257),
            lambda d:d["audit"]["unsampled_real_GT"][0].update(GT_annotation_id=100),
            lambda d:d["batches"][1].update(sample_positions=list(range(1024))),
            lambda d:d["batches"][0]["C_flag"].__setitem__(0,False),
            lambda d:d["batches"][0]["importance_weights"].__setitem__(0,float("nan")))
        for mutation in mutations:
            data=fixture(0);mutation(data)
            with self.assertRaises(RuntimeError):coverage.analyze_image(0,1000,data)
        first=coverage.analyze_image(0,1000,fixture(0))
        with self.assertRaises(RuntimeError):
            coverage.aggregate_images([first,first],expected_images=2,expected_real_GT=6)
        data=fixture(1);data["batches"][0]["GT_annotation_id"]=100
        second=coverage.analyze_image(1,1001,data)
        with self.assertRaises(RuntimeError):
            coverage.aggregate_images([first,second],expected_images=2,expected_real_GT=5)

if __name__=="__main__":unittest.main()
