# encoding: utf-8
"""UNIT_FIXTURE: synthetic arrays only; no Torch, cache/weights/images or training."""
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np

P = Path(__file__).with_name("dataset_adapter.py")
S = importlib.util.spec_from_file_location("n23_adapter_unit", P)
adapter = importlib.util.module_from_spec(S)
S.loader.exec_module(adapter)


def fixture():
    ref = lambda label: dict(path="E:/UNIT/"+label, bytes=1, sha256="a"*64)
    binding = {k:ref(k) for k in ("source","spec","GPU_admission","teacher_checkpoint","fixed_PR_checkpoint")}
    binding["phase"]="UNIT_N16"
    binding["cache_body_source"]=dict(path="E:/UNIT/cache_body.py",bytes=23893,
        sha256="77de4c5445ac455672095b8c795e16046a4ed6822bae50f739f40cb4625ea5bd")
    binding["C_engine_source"]=dict(path="E:/UNIT/C_engine.py",bytes=7684,
        sha256="8f0bd504ce7899225d868381d9b656402f8882bb19f6bd880060085fe269f9cc")
    ids=np.array([0,2,0,0,1,1],np.int64)
    flat=np.array([0,3,0,4,2,2],np.int64)
    unique,inverse=np.unique(flat,return_inverse=True)
    raw=np.full(100,-2,np.int64);raw[0]=0;raw[2]=1;raw[1]=-1
    kinds=["ignored"]*100;kinds[0]=kinds[2]="matched";kinds[1]="background"
    identity=dict(image_id=42,ann_ids=[10,20],original_shape=[2,3])
    packet=dict(schema="N16-all100-FIT-action-sampled-cache-body-v1",dataset_index=3,identity=identity,
        natural_query_ids=list(range(100)),raw_GT_ids=[10,20],
        sampled_F_unique=np.arange(len(unique)*256,dtype=np.float32).reshape(-1,256),
        native_e0=np.arange(100*256,dtype=np.float32).reshape(1,100,256),
        unique_flat_indices=unique,sample_to_unique=inverse.astype(np.int64),
        assignment=dict(raw_GT_index_by_query=raw,raw_GT_annotation_id_by_query=[10,None,20]+[None]*97,
            supervision_kind_by_query=kinds,native_positive_query_indices=[0,2],
            native_negative_query_indices=[1],native_ignored_query_indices=list(range(3,100))),
        C_counts_all100=[6]*100,K_counts_all100=[1]*100,
        active_matched_queries=[0,2],active_background_queries=[1],active_queries=[0,1,2],
        samples=dict(query_ids=ids,flat_indices=flat,native_L=np.array([-.5,3,-.5,-5,-3,-3],np.float32),
            targets=np.array([1,0,1,1,0,0],np.bool_),importance_weights=np.array([.8,.8,.8,.8,1.2,1.2],np.float32),
            stratum_ids=np.array([0,0,0,0,1,1],np.uint8),seed=17,replacement=True,duplicates_preserved=True,
            draw_order="matched then background; vector query draw then sequential point draws"))
    packet.update({k:v for k,v in binding.items() if k not in ("cache_body_source","C_engine_source")})
    row=dict(dataset_index=3,image_id=42,identity=copy.deepcopy(identity),points=6,
             C_counts_all100=[6]*100,K_counts_all100=[1]*100)
    return packet,row,binding


class SharedPointAdapterTests(unittest.TestCase):
    def test_only_real_matches_keep_original_points_duplicates_and_both_domain_flags(self):
        packet,row,binding=fixture()
        result=adapter.build_shared_batches(np,packet,row,binding,synthetic_only=True)
        self.assertEqual([b["query_id"] for b in result["batches"]],[0,2])
        first=result["batches"][0]
        np.testing.assert_array_equal(first["sample_positions"],[0,2,3])
        np.testing.assert_array_equal(first["flat_indices"],[0,0,4])
        np.testing.assert_array_equal(first["F"],packet["sampled_F_unique"][[0,0,3]])
        np.testing.assert_array_equal(first["Omega_flag"],[True,True,False])
        np.testing.assert_array_equal(first["C_flag"],[True,True,True])
        np.testing.assert_array_equal(first["targets"],[True,True,True])
        self.assertEqual(result["audit"]["dropped_background_draws"],2)
        self.assertEqual(result["audit"]["retained_matched_draws"],4)
        self.assertEqual(first["GT_annotation_id"],10)

    def test_unsampled_real_GT_is_audited_not_filled_or_resampled(self):
        packet,row,binding=fixture()
        packet["samples"]["query_ids"][1]=0
        result=adapter.build_shared_batches(np,packet,row,binding,synthetic_only=True)
        self.assertEqual([b["query_id"] for b in result["batches"]],[0])
        self.assertEqual(result["audit"]["unsampled_real_GT"],
            [dict(query_id=2,GT_index=1,GT_annotation_id=20,K=1)])
        self.assertEqual(result["audit"]["retained_matched_draws"],4)

    def test_corrupt_mapping_proxy_ignore_and_unknown_writer_are_rejected(self):
        for mutation in (
            lambda p,r,b:p["sample_to_unique"].__setitem__(0,1),
            lambda p,r,b:p["samples"]["stratum_ids"].__setitem__(4,0),
            lambda p,r,b:p["samples"]["query_ids"].__setitem__(4,3),
            lambda p,r,b:b["cache_body_source"].update(sha256="f"*64),
            lambda p,r,b:b["C_engine_source"].update(sha256="f"*64),
            lambda p,r,b:p["K_counts_all100"].__setitem__(0,0)):
            packet,row,binding=fixture();mutation(packet,row,binding)
            with self.assertRaises(RuntimeError):
                adapter.build_shared_batches(np,packet,row,binding,synthetic_only=True)

    def test_outputs_do_not_mutate_cache_and_real_mode_rejects_array_fixture(self):
        packet,row,binding=fixture();before=copy.deepcopy(packet)
        result=adapter.build_shared_batches(np,packet,row,binding,synthetic_only=True)
        result["batches"][0]["F"][0,0]=-999
        result["batches"][0]["targets"][0]=False
        np.testing.assert_array_equal(packet["sampled_F_unique"],before["sampled_F_unique"])
        np.testing.assert_array_equal(packet["samples"]["targets"],before["samples"]["targets"])
        with self.assertRaises(RuntimeError):
            adapter.build_shared_batches(np,packet,row,binding)

if __name__=="__main__":unittest.main()
