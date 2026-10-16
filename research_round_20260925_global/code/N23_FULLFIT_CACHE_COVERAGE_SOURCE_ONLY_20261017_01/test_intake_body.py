"""Original population/parent metadata checks, with no real inputs or tensors."""
import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).with_name('intake_body.py')


def load_body():
    ms = importlib.util.spec_from_file_location('n23_intake_body_unit', path)
    body = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(body)
    return body


def fixture():
    rows, count = [], 0
    for i in range(741):
        n = 3 if i < 179 else 2
        annotations = list(range(count, count + n)); count += n
        rows.append(dict(dataset_index=i, image_id=1000+i, raw_GT=n, points=4096,
            natural_queries=100, identity=dict(image_id=1000+i,ann_ids=annotations),
            packet=dict(path='E:/UNIT/packet_%d.pth'%i,bytes=1,sha256='a'*64)))
    report = dict(images=741,raw_GT=1661,natural_queries=74100,sampled_points=3035136,
                  rows=rows,packet_refs=[r['packet'] for r in rows])
    accepted = dict(fully_qualified=True,scout_training_eligible=True,
                   images=741,raw_GT=1661,natural_queries=74100,sampled_points=3035136)
    pre = dict(dataset_indices=list(range(741)), fit_image_ids=[r['image_id'] for r in rows],
               fit_ann_ids_by_image=[r['identity']['ann_ids'] for r in rows])
    return report, accepted, pre


class IntakePopulationTests(unittest.TestCase):
    def test_original_complete_population(self):
        body=load_body(); r,q,p=fixture(); saved=copy.deepcopy((r,q,p))
        body.check_population(r,q,p)
        self.assertEqual((r,q,p),saved)

    def test_pending_cache_does_not_qualify_itself(self):
        body=load_body()
        for field in ['fully_qualified','scout_training_eligible']:
            r,q,p=fixture();q[field]=False
            with self.assertRaises(AssertionError):body.check_population(r,q,p)

    def test_true_rows_and_packet_order_must_match_population_claims(self):
        body=load_body()
        for mutate in (
            lambda r:r['rows'][0].update(raw_GT=1),
            lambda r:r['rows'][0].update(points=1024),
            lambda r:r['packet_refs'].reverse(),
            lambda r:r['rows'][0]['identity']['ann_ids'].pop(),
            lambda r:r['rows'].reverse(),
        ):
            r,q,p=fixture();mutate(r)
            with self.assertRaises(AssertionError):body.check_population(r,q,p)


if __name__ == '__main__':unittest.main()
