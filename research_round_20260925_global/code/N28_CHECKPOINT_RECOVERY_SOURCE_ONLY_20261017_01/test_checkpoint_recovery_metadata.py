"""Synthetic completion-log/history join only; no checkpoint or tensor access."""
import copy
import json
import unittest

import checkpoint_probe_body as body


class CompletionMetadataTest(unittest.TestCase):
    def test_fixed_CPU_recovery_prelock_allows_plan_fields_but_rejects_CHANGED_scope(self):
        pre=dict(schema='N28-dense-routing-checkpoint-recovery-prelock-v1',seed=202610172801,
            completed_epoch=5,cumulative_updates_per_arm=3705,images_per_epoch=741,GT_per_epoch=1661,
            modes=['SELECTIVE','WHOLE_NODE'],lr=3e-4,weight_decay=1e-4,betas=[.9,.999],eps=1e-8,
            optimizer='AdamW',no_new_training=True,no_AP=True,no_DEV=True,no_TEST=True,
            CUDA_RNG_restore_deferred=True,plan='SYNTHETIC metadata only')
        body.validate_prelock(pre)
        for key,value in [('completed_epoch',8),('no_new_training',False),('CUDA_RNG_restore_deferred',False)]:
            bad=copy.deepcopy(pre);bad[key]=value
            with self.assertRaises(ValueError):body.validate_prelock(bad)

    def test_original_five_contiguous_completions_join_history_without_selection(self):
        seed=202610172801
        history=[];lines=['Synthetic unrelated initialization log']
        for epoch in range(1,6):
            means={mode:epoch/10+(0 if mode=='SELECTIVE' else .01) for mode in ('SELECTIVE','WHOLE_NODE')}
            history.append(dict(epoch=epoch,cumulative_updates_per_arm=epoch*741,
                arms={mode:dict(mean_image_loss=value) for mode,value in means.items()}))
            lines.append(json.dumps(dict(N28_seed=seed,completed_epoch=epoch,
                cumulative_updates_per_arm=epoch*741,mean_image_loss=means)))
        text='\n'.join(lines)
        actual=body.validate_completion_log(text,history,seed)
        self.assertEqual([row['completed_epoch'] for row in actual],[1,2,3,4,5])
        bad=copy.deepcopy(history);bad[2]['arms']['SELECTIVE']['mean_image_loss']+=.001
        with self.assertRaises(ValueError):body.validate_completion_log(text,bad,seed)
        with self.assertRaises(ValueError):body.validate_completion_log(text+'\n'+lines[-1],history,seed)


if __name__=='__main__':unittest.main()
