import copy
import json
import os
import importlib.util
from pathlib import Path
from hashlib import sha256
import unittest
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
from pycocotools import mask as codec
from score_dev import mapped_identity_sha256, reconstruct_predictions


def native_scorer():
    path=Path(os.environ['RF_NATIVE_SCORER'])
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!='1ce0eaa2106fe52452129fad534d0fc11bb8ad5fc4611f70858dbdf6f87a5ec3':
        raise ValueError('Exact original ordinary native scorer')
    spec=importlib.util.spec_from_file_location('_synthetic_native_score',str(path))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only relocate the original pinned official adapter in this private module.
    module.OFFICIAL=Path(os.environ['RF_OFFICIAL_ADAPTER'])
    if sha256(module.OFFICIAL.read_bytes()).hexdigest()!=module.OFFICIAL_SHA:
        raise ValueError('Exact original ordinary official adapter')
    return module


def rle(mask):
    item=codec.encode(np.asfortranarray(mask,dtype=np.uint8))
    return dict(size=item['size'],counts=item['counts'].decode('ascii'))


def fixture():
    stock=np.zeros((8,8),dtype=bool)
    stock[:4]=True
    masks=[rle(stock),rle(np.zeros_like(stock))]
    pool=[dict(rank=rank,query_id=query,score=score,area=int(codec.area(mask)),rle=mask,
               box_xyxy=[0.,0.,8.,8.])
          for rank,query,score,mask in zip((7,98),(99,3),(.73,.61),masks)]
    native=dict(image_id=1,raw_query_count=164,native_count=100,unmapped_slot1_count=98,
                native_ranks=[7,98],coco=[dict(image_id=1,category_id=1,score=p['score'],segmentation=p['rle']) for p in pool])
    def change(indices,adds,drops,kind):
        return dict(flipped=indices,edits=len(indices),adds=adds,drops=drops,area_delta=adds-drops,
                    actions=len(indices) if kind=='flip' else len(indices)//2,action_kind=kind,
                    nonpositive_actions=0,nonpositive_flips=0,predicted_gain_sum=.1)
    arms={a:change([32,0],1,1,'swap') for a in ('SU','LF=S','LU=S','NP=S')}
    arms['LW=S']=change([32,33],2,0,'flip')
    sparse=dict(ordinal=1,image_id=1,group='prefix',original_hw=[8,8],
                mapped_identity_sha256=mapped_identity_sha256(pool),
                points=[dict(rank=7,query_id=99,coordinates=[[32,0],[33,1]])],
                seeds={str(seed):{'7':copy.deepcopy(arms)} for seed in (5217,5218,5219)})
    return [dict(native_block=native,pool=pool,source_prefix='prefix')],[sparse]


class ScoreTests(unittest.TestCase):
    def test_real_official_scoring_and_fixed_match_geometry_include_unsupported(self):
        from score_dev import score_matched_population, ARMS, SEEDS
        import evaluate_geometry
        source=Path(os.environ['FIT_STEP_SOURCE'])
        self.assertEqual(sha256(source.read_bytes()).hexdigest(),
                         '94e60f5b5557787687d1f228c33c249a1d7889172a225373b97bd789ec8b72b1')
        spec=importlib.util.spec_from_file_location('_score_real_fit_step',str(source))
        fit=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fit)
        helpers=fit.load_geometry(Path(os.environ['CODE_ROOT']),Path(os.environ['SHAPELY_SITE']))
        captures,sparse=fixture()
        second=copy.deepcopy(captures[0])
        second['native_block']['image_id']=2
        for row in second['native_block']['coco']:
            row['image_id']=2
        captures.append(second)
        unsupported=copy.deepcopy(sparse[0])
        unsupported.update(ordinal=2,image_id=2,points=[dict(rank=7,query_id=99,coordinates=[])],
                           seeds={str(seed):{} for seed in SEEDS})
        sparse.append(unsupported)
        gt=dict(info={},images=[dict(id=i,width=8,height=8) for i in (1,2)],
                categories=[dict(id=1,name='ship')],annotations=[
                    dict(id=i,image_id=i,category_id=1,iscrowd=0,area=32,bbox=[0,0,8,4],
                         segmentation=[[0,0,8,0,8,4,0,4]]) for i in (1,2)])
        original=copy.deepcopy(gt)
        base=native_scorer()
        historical=b''.join(base.encoded(row['native_block']) for row in captures)
        flat=[row for capture in captures for row in capture['native_block']['coco']]
        with redirect_stdout(StringIO()):
            expected=base.score_variants(gt,flat,set())
            result=score_matched_population(np,codec,base,gt=gt,empty_ids=set(),capture_rows=captures,
                sparse_rows=sparse,historical_native_bytes=historical,
                historical_native_sha256=sha256(historical).hexdigest(),historical_scores=expected,
                fit_step=fit,geometry_helpers=helpers,geometry_module=evaluate_geometry)
        self.assertEqual(result['geometry']['summary']['counts']['valid'],2)
        self.assertEqual(result['geometry']['summary']['counts']['valid_unsupported_noop'],1)
        self.assertEqual(len(result['geometry']['private_witness']),6)
        for seed in SEEDS:
            self.assertEqual(set(result['coco'][seed]),set(ARMS))
            self.assertEqual(result['coco'][seed]['N']['primary']['all12'],expected['primary']['all12'])
        witness=result['geometry']['private_witness'][0]['arms']
        self.assertAlmostEqual(witness['SU']['iou'],31/33)
        self.assertAlmostEqual(witness['LW=S']['iou'],32/34)
        self.assertIsNone(result['mean_COCO']['primary']['N']['all12'][5])
        self.assertFalse(result['gate']['advance'])
        self.assertEqual(gt,original)
        bad=copy.deepcopy(expected)
        bad['primary']['AP90']+=.001
        with redirect_stdout(StringIO()), self.assertRaises(ValueError):
            score_matched_population(np,codec,base,gt=gt,empty_ids=set(),capture_rows=captures,
                sparse_rows=sparse,historical_native_bytes=historical,
                historical_native_sha256=sha256(historical).hexdigest(),historical_scores=bad,
                fit_step=fit,geometry_helpers=helpers,geometry_module=evaluate_geometry)

    def test_sorted_query_objects_rebuild_exact_legacy_native_bytes(self):
        from score_dev import verify_native_baseline
        captures,_=fixture()
        base=native_scorer()
        historical=base.encoded(captures[0]['native_block'])
        digest=sha256(historical).hexdigest()
        sorted_capture=json.loads(json.dumps(captures,sort_keys=True))
        self.assertNotEqual(base.encoded(sorted_capture[0]['native_block']),historical)
        result=verify_native_baseline(base,sorted_capture,historical,digest)
        self.assertEqual(result['predictions_sha256'],digest)
        self.assertEqual(result['mapped_candidates'],2)
        self.assertTrue(result['exact_historical_bytes'])
        with self.assertRaises(ValueError):
            verify_native_baseline(base,sorted_capture,historical,'0'*64)
        sorted_capture[0]['native_block']['coco'][0]['score']=.74
        with self.assertRaises(ValueError):
            verify_native_baseline(base,sorted_capture,historical,digest)

    def test_actual_official_score_variants_empty_only_sensitivity(self):
        from score_dev import score_official
        captures,_=fixture()
        rows=captures[0]['native_block']['coco']
        gt=dict(info={},images=[dict(id=1,width=8,height=8)],categories=[dict(id=1,name='ship')],
                annotations=[dict(id=i+1,image_id=1,category_id=1,segmentation=row['segmentation'],
                                  area=int(codec.area(row['segmentation'])),bbox=[0,0,8,4] if i==0 else [0,0,0,0],
                                  iscrowd=0) for i,row in enumerate(rows)])
        original=copy.deepcopy(gt)
        actual=score_official(native_scorer(),gt,rows,{2})
        self.assertEqual(actual['primary']['annotations'],2)
        self.assertEqual(actual['sensitivity']['annotations'],1)
        self.assertEqual(len(actual['primary']['all12']),12)
        self.assertEqual(actual['primary']['AP50'],actual['primary']['all12'][1])
        self.assertAlmostEqual(actual['sensitivity']['AP'],1.)
        self.assertLess(actual['primary']['AP'],actual['sensitivity']['AP'])
        self.assertEqual(actual['primary']['all12'][5],-1.)
        self.assertEqual(gt,original)

    def test_sparse_reconstruction_preserves_all_raw_order_and_identity(self):
        captures,sparse=fixture()
        original=copy.deepcopy(captures)
        n=reconstruct_predictions(np,codec,captures,sparse,5217,'N')
        self.assertEqual(n,captures[0]['native_block']['coco'])
        lw=reconstruct_predictions(np,codec,captures,sparse,5217,'LW=S')
        self.assertEqual(len(lw),2)
        self.assertEqual([v['score'] for v in lw],[.73,.61])
        self.assertEqual(lw[1],n[1])
        self.assertEqual(int(codec.area(lw[0]['segmentation'])),34)
        self.assertEqual(int(codec.area(codec.merge([n[0]['segmentation'],lw[0]['segmentation']],intersect=True))),32)
        self.assertEqual(captures,original)
        bad=copy.deepcopy(sparse)
        bad[0]['seeds']['5217']['7']['LW=S']['edits']=1
        with self.assertRaises(ValueError):
            reconstruct_predictions(np,codec,captures,bad,5217,'LW=S')
        bad=copy.deepcopy(sparse)
        bad[0]['points'][0]['query_id']=98
        with self.assertRaises(ValueError):
            reconstruct_predictions(np,codec,captures,bad,5217,'N')


if __name__=='__main__':
    unittest.main()
