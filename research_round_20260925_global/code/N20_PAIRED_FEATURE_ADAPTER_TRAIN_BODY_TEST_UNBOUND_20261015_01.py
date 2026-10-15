# SOURCE_ONLY_SYNTHETIC_TEST: Root actual exit0; no SAR inputs.
"""ROOT runs RED before N20 body implementation. Only ordinary sibling modules load."""
import importlib.util
from pathlib import Path
import unittest


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticPairedFeatureUpdate(unittest.TestCase):
    def test_one_image_shares_initial_function_but_trains_different_neighbor_gradients(self):
        root = Path(__file__).resolve().parent
        # Expected RED: the new body is not implemented yet.
        body = _load('n20_pair_body_under_test',
            root / 'N20_PAIRED_LOCAL_CENTER_ERROR_TRAIN_BODY_UNBOUND_DRAFT_20261015_01.py')
        head_module = _load('n20_feature_head_ordinary',
            root / 'N20_FEATURE_ADAPTER_UNBOUND_20261015_01.py')
        base = _load('n20_old_513_factory_ordinary',
            root / 'N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py')
        old = _load('n20_old_training_helpers_ordinary',
            root / 'N16_PAIRED_SCOUT_TRAIN_BODY_UNBOUND_DRAFT_20261014_02.py')
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        seed = 202610150201
        self.assertEqual(tuple(body.ARMS), ('LOCAL_ERROR', 'CENTER_ERROR'))
        self.assertEqual(tuple(body.REPEAT_SEEDS), (202610150201, 202610150202, 202610150203))
        before = torch.get_rng_state().clone()
        state = body.make_paired_state(torch, head_module, base, old, device='cpu', seed=seed)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        left, right = [state['heads'][arm] for arm in body.ARMS]
        reference = base.make_scout(torch, seed)
        for head in (left, right):
            self.assertEqual(len(tuple(head.parameters())), 5)
            self.assertEqual(sum(p.numel() for p in head.parameters()), 18851)
            self.assertTrue(torch.equal(head.spatial_weight, torch.zeros((9,256))))
            for key, expected in reference.state_dict().items():
                self.assertTrue(torch.equal(head.state_dict()[key], expected))
        for key, value in left.state_dict().items():
            self.assertTrue(torch.equal(value, right.state_dict()[key]))
            self.assertNotEqual(value.data_ptr(), right.state_dict()[key].data_ptr())

        # A single image, three sample slots, duplicate/nonmonotone unique mapping.
        first = torch.linspace(1.0, 2.0, 256, dtype=torch.float32)
        second = torch.linspace(2.0, 3.0, 256, dtype=torch.float32)
        unique_F = torch.stack((first, second))
        extra_F = -unique_F.clone()
        slot_to_unique = torch.tensor([1,0,1], dtype=torch.int64)
        neighbor_map = torch.tensor([[2,2,2,2,0,2,2,2,2],
                                     [3,3,3,3,1,3,3,3,3]], dtype=torch.int64)
        embedding = torch.linspace(1.0, -1.0, 256, dtype=torch.float32)
        weights = torch.tensor([.5,2.0,1.5], dtype=torch.float32)
        samples = dict(query_ids=torch.tensor([1,0,1], dtype=torch.int64),
                       native_L=torch.full((3,), .25, dtype=torch.float32),
                       importance_weights=weights,
                       utility_labels=torch.full((3,),2,dtype=torch.int64),
                       error_labels=torch.zeros(3,dtype=torch.int64),
                       stratum_ids=torch.tensor([0,1,0],dtype=torch.uint8),
                       delta=torch.zeros(3,dtype=torch.bool))
        packet = dict(samples=samples, sample_to_unique=slot_to_unique,
                      sampled_F_unique=unique_F, native_e0=embedding.reshape(1,1,256).repeat(1,100,1))
        loaded = dict(packet=packet, neighbor_packet=dict(extra_F=extra_F, neighbor_to_combined=neighbor_map))
        # Expected tensors are assembled explicitly, independently of body gather.
        expected_F = torch.stack((second, first, second))
        expected_local = torch.stack([torch.stack([center if offset==4 else -center
                                    for offset in range(9)]) for center in (second,first,second)])
        expected_center = expected_F[:,None,:].repeat(1,9,1)
        e = embedding[None,:].repeat(3,1)
        with torch.no_grad():
            expected_logits = reference(expected_F,e,samples['native_L'])
            for head, neighbors in ((left,expected_local),(right,expected_center)):
                actual = head(expected_F,e,samples['native_L'],neighbors)
                self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                            expected_logits.contiguous().view(torch.uint8)))
        # Independent FP64 manual CE; old target is mean(weight*CE), not /sum(w).
        scores64 = expected_logits.to(torch.float64)
        per_slot = torch.logsumexp(scores64,dim=1)-scores64[:,0]
        expected_loss = float((weights.to(torch.float64)*per_slot).mean())
        captured = {arm: [] for arm in body.ARMS}
        handles = []
        for arm in body.ARMS:
            def save_input(module, args, name=arm):
                captured[name].append(tuple(v.detach().clone() for v in args))
            handles.append(state['heads'][arm].register_forward_pre_hook(save_input))
        loads, events = [], []
        def loader(index):
            loads.append(index)
            return loaded
        try:
            report = body.train_one_epoch(torch,state,loader,epoch=1,image_count=1,observer=events.append)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(loads,[0])
        self.assertEqual(report['updated_images'],1)
        self.assertEqual(report['points'],3)
        self.assertEqual(state['updates'],{arm:1 for arm in body.ARMS})
        self.assertEqual(state['sampled_points'],3)
        self.assertEqual(state['completed_epoch'],1)
        for arm, neighbors in (('LOCAL_ERROR',expected_local),('CENTER_ERROR',expected_center)):
            self.assertEqual(len(captured[arm]),1)
            for actual, expected in zip(captured[arm][0],(expected_F,e,samples['native_L'],neighbors)):
                self.assertTrue(torch.equal(actual,expected))
            self.assertAlmostEqual(report['online_sum_image_weighted_CE'][arm],expected_loss,delta=2e-6)
            optimizer = state['optimizers'][arm]
            self.assertEqual(len(optimizer.state),5)
            for value in optimizer.state.values():
                self.assertEqual(float(value['step']),1.0)
                self.assertTrue(torch.isfinite(value['exp_avg']).all().item())
                self.assertTrue(torch.isfinite(value['exp_avg_sq']).all().item())
        lm = state['optimizers']['LOCAL_ERROR'].state[left.spatial_weight]['exp_avg'][0]
        cm = state['optimizers']['CENTER_ERROR'].state[right.spatial_weight]['exp_avg'][0]
        self.assertGreater(torch.count_nonzero(lm).item(),0)
        self.assertGreater(torch.count_nonzero(cm).item(),0)
        self.assertTrue((lm*cm<0).any().item(), 'True noncenter features must change a trainable gradient direction')
        self.assertFalse(torch.equal(left.spatial_weight,right.spatial_weight))
        self.assertEqual(sum(e['event']=='paired_image_update' for e in events),1)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
