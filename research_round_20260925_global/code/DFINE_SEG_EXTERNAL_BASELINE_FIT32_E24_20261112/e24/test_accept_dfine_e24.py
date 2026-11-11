from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path
import tempfile
import unittest


HERE=Path(__file__).resolve().parent
API=run_path(str(HERE/"accept_dfine_e24.py"))


def encode(data):
    return (json.dumps(data,sort_keys=True,separators=(",",":"))+"\n").encode()


class ClosedLedgerTest(unittest.TestCase):
    def test_actual_batched_clock_and_sequence_are_both_required(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root=Path(temp).resolve()
            self.assertTrue(root.is_relative_to(HERE))
            order=list(range(2937))+[0,1,2]
            batches=[order[i:i+4] for i in range(0,len(order),4)]
            replay_order=root/"FIT_ORDER_PRIVATE.jsonl"
            replay_names=root/"FIT_ID_BASENAME_PRIVATE.json"
            replay_order.write_bytes(b"".join(encode(dict(epoch=i,batches=batches)) for i in range(1,25)))
            replay_names.write_bytes(encode({str(i):f"fixture{i}.png" for i in range(2937)}))
            verify=API["verify_ledgers"]
            verify.__globals__.update(REPLAY=root,ORDER_SHA=sha256(replay_order.read_bytes()).hexdigest(),
                                      MAP_SHA=sha256(replay_names.read_bytes()).hexdigest())
            source=root/"RESULT_SAFE.json"
            source.write_bytes(b"synthetic evidence only\n")
            batch_file=root/"ACTUAL_BATCH_LEDGER_PRIVATE.jsonl"
            epoch_file=root/"ACTUAL_EPOCH_LEDGER_PRIVATE.jsonl"
            events=[dict(status="SYNTHETIC_START",initial_state_sha256=API["INIT_SHA"])]
            rows,epoch_rows=[],[]
            for epoch in range(1,25):
                for batch, ids in enumerate(batches,1):
                    update=(epoch-1)*735+batch
                    row=dict(epoch=epoch,batch=batch,update=update,image_ids=ids,optimizer_returns=update,
                             scheduler_returns=update,ema_returns=update,score_parameter_clock=update,
                             retained_targets=[1]*4,nonempty_targets=[1]*4,seconds=.1,grad_before_clip=1.,
                             loss=1.,mask_bce=.2,mask_dice=.2)
                    if update<=8:
                        row.update(mask_optimizer_clock=update,mask_loss_dtype="torch.float32",total_loss_dtype="torch.float32",
                                   mask_before_sha256=str(update-1),mask_after_sha256=str(update))
                        events.append(dict(status="DFINE_SEG_S_HRSID_E24_FIRST8_ACTUAL_RETURN",**row))
                    rows.append(row)
                epoch_row=dict(epoch=epoch,optimizer_updates=update,ema_updates=update,scheduler_updates=update,
                               fit_visits=update*4,retained_targets=2940,nonempty_targets=2940,wall_seconds=73.5)
                events.append(dict(status="DFINE_SEG_S_HRSID_E24_ACTUAL_EPOCH_COMPLETE",**epoch_row,**API["SCOPE"]))
                epoch_rows.append(epoch_row)
            events.append(dict(status="SYNTHETIC_RETURN",result_sha256=sha256(source.read_bytes()).hexdigest()))
            epoch_file.write_bytes(b"".join(map(encode,epoch_rows)))
            stdout=b"".join(map(encode,events))

            def report_with_rows():
                batch_file.write_bytes(b"".join(map(encode,rows)))
                return dict(batch_ledger=API["stable_ref"](batch_file),epoch_ledger=API["stable_ref"](epoch_file),
                            retained_targets=70560,nonempty_targets=70560)

            self.assertEqual(verify(root,report_with_rows(),stdout),epoch_rows)
            rows[900]["optimizer_returns"]=900
            with self.assertRaisesRegex(ValueError,"no-skip update clocks"):
                verify(root,report_with_rows(),stdout)


if __name__=="__main__":
    unittest.main()
