"""Replay the same source masks with two additional quota-only choices."""
from argparse import ArgumentParser
from collections import Counter
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import HERE, LOCK, QUOTA, WRITER, closed, load, need, pin, settings

RENDERER = "a0c11c5c37c3a8569e1a54047fbc09598604710861bcd19f4096a0e3abe35959"
NEW = ("calibrated_quota","reversed_quota")
KEYS = ("group_protocol","archive_loader","variable_writer","gate","archive","original_fit","points",
        "core","selector","attraction","base_renderer")
STATUS = "SPSD_PSEG_QUOTA_CALIBRATION_RENDER_COMPLETE"


def renderer_source(source):
    anchor = '    raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\\n").encode()'
    need(source.count(anchor)==1,"Expected base renderer closeout")
    return source.replace(anchor,'    result.update(extra_facts())\n'+anchor)


def run(cfg, root, output):
    fit,fit_sha = closed(root/"fit/RESULT_SAFE.json",root/"observed_fit/TERMINAL.json",HERE/"fit_quota.py")
    need(fit["status"]=="SPSD_PSEG_QUOTA_FIT_COMPLETE" and fit["protocol_sha256"]==LOCK and fit["fit_only"] is True
         and not any(fit[x] for x in ("source_gate_labels_opened","source_gate_points_opened","historical_target_opened","historical_test_opened")),
         "FIT-only calibration has closed")
    models = json.loads(pin(root/"fit/CALIBRATION_PRIVATE.json",fit["calibrator_sha256"]))
    need(set(models)==set(NEW) and all(len(m["scale"])==len(m["coefficients"])==4 for m in models.values()),
         "Exact fitted four scalar gains for each arm")
    q = load(HERE/"quota.py",QUOTA)
    old = load(cfg.base_renderer,RENDERER,renderer_source)
    original_loader = old.load
    old.LOCK = LOCK
    old.STATUS = STATUS
    old.__file__ = __file__
    diagnostics = Counter()

    def guarded_load(path,checksum):
        writer = original_loader(path,checksum)
        if checksum != WRITER:
            return writer
        raw_select = writer.select
        arms = (*writer.ARMS,*NEW)

        def choose(p,static,k,area):
            rows = q.panel(writer,p,static,k,area)
            result = raw_select(p,static,k,area)
            require = {
                "fixed_iou": rows[0],
                "variable_iou": max(rows,key=lambda x:x["score"]),
                "near_iou": max((row for row in rows if abs(row["adds"]-k)<=1),key=lambda x:x["score"]),
            }
            need(all(result[key]["action"]==row["action"] for key,row in require.items()),
                 "Per-split replay preserves every original robust decision")
            for name in NEW:
                raw,x = q.features(rows,k,area,name=="reversed_quota")
                result[name] = q.choose(rows,raw,x,models[name])
            diagnostics.update(legal_splits=len(rows),fixed_only=int(len(rows)==1),
                               differ_between_calibrations=int(result[NEW[0]]["action"]!=result[NEW[1]]["action"]))
            return result

        return SimpleNamespace(ARMS=arms,select=choose,write_mask=writer.write_mask)

    old.load = guarded_load
    old.extra_facts = lambda: dict(quota_core_sha256=QUOTA,calibration_fit_result_sha256=fit_sha,
                                   calibrator_sha256=fit["calibrator_sha256"],quota_diagnostics=dict(diagnostics))
    cfg.protocol = HERE/"PRELOCK.md"
    cfg.output = output
    old.render(cfg)


def main():
    p=ArgumentParser(description=__doc__)
    for arg in ("inputs","root","output"):
        p.add_argument("--"+arg,type=Path,required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
         and not a.output.exists(),"Fresh isolated CPU-only label-free render")
    run(settings(a.inputs,KEYS),a.root,a.output)

if __name__=="__main__":
    main()
