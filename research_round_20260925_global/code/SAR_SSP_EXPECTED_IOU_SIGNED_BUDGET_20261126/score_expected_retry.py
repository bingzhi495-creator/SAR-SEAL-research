"""Replay the exact old SAR-SSP writer using two expectation-based sibling arms."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


HERE = Path(__file__).resolve().parent
LOCK = "ca91d6bb1004ae12993dabb1fcd502584fd589c5ad65f50649ce433dc16bda47"
EXPECTED = "d303ecfa5d88af0e6cfb102c905fbc71d5cab9b1b5dd3b18fd173600e331efff"
BASE_WORKER = "83ff61406235b142b86c2bcdbebbc92b3b17a7d3eb48032cd6c6619ec7e6f8a2"
BASE_RENDER = "9cf1753d4818df64717217831d29421f329eae21014b4656e52be83ecefd1701"
BASE_SCORE = "a857f06b37ca65e38205e24c5fd024e837f533dbf1ba6c448fed5d2e4727e837"
BASE_RENDER_TERMINAL = "1ea6b73fc7f977ad7f3cce632c97083fb2d3fa6b8c153ce4480dd212ec67c582"
BASE_SCORE_TERMINAL = "faa2c69b75cb23bfd5077a338a6682a9fd6233e6c66720fc713c1f64108f57c3"
RENDER_AUTHORITY = "b9fb0078d2ca028b8f2dc79cb29ad6c344964bd3a745b8e54c45f07ce66629a7"
NEW_RENDER = "7ad72dfc542ad5b3f22e9609aa0c2cac723f4f48966b875e655f0f609ac59db6"
NEW_RENDER_TERMINAL = "180a56fea7b67e2cc31a22acbf213322f5458b4bee04b998a0b40756cd098b11"
OLD_ARMS = ("fixed_iou","variable_iou","near_iou","bce_fixed_robust")
NEW_ARMS = ("expectation_exact","expectation_fixed")
RENDER_STATUS = "SAR_SSP_EXPECTED_IOU_SIGNED_REPLAY_COMPLETE"
SCORE_STATUS = "SAR_SSP_EXPECTED_IOU_SIGNED_SCORE_COMPLETE"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path,expected):
    raw = path.read_bytes()
    need(digest(raw)==expected,"Exact bound artifact "+path.name)
    return raw


def patch(source,before,after):
    need(source.count(before)==1,"Expected unique trusted source seam: "+before.splitlines()[0])
    return source.replace(before,after,1)


def adapted(source):
    source=patch(source,'    worker=digest(Path(__file__).read_bytes())',
                 '    worker=RENDER_AUTHORITY')
    source=patch(source,'            raw=line(dict(image_id=item["image_id"],rgb_sha256=item["rgb_sha256"],edits=results))',
        '            document=dict(image_id=item["image_id"],rgb_sha256=item["rgb_sha256"],edits=results)\n'
        '            verify_old(ordinal-1,document)\n'
        '            raw=line(document)')
    source=patch(source,'        now,old_row=rendered[j],archived[j]',
        '        now,old_row=rendered[j],archived[j]\n'
        '        verify_old(j,now)')
    source=patch(source,'    scoring.PRIMARY=("variable_iou","fixed_iou")',
        '    scoring.PRIMARY=("expectation_exact","variable_iou")')
    source=patch(source,'    scoring.SECONDARY=("variable_iou","bce_fixed_robust")',
        '    scoring.SECONDARY=("expectation_exact","expectation_fixed")')
    source=patch(source,'("fixed_iou","static"),("legacy_free_matched","legacy_fixed_distance"))',
        '("fixed_iou","static"),("legacy_free_matched","legacy_fixed_distance"),\n'
        '                         ("expectation_fixed","fixed_iou"),("variable_iou","fixed_iou"),\n'
        '                         ("variable_iou","bce_fixed_robust"),("expectation_exact","fixed_iou"))')
    key='"variable_iou__vs__fixed_iou"'
    source=patch(source,'    main=geometry["comparisons"]['+key+']["iou_points"]',
                        '    main=geometry["comparisons"]["expectation_exact__vs__variable_iou"]["iou_points"]')
    return patch(source,'    primary=measured["coco_mask_deltas_points"]['+key+']["AP"]',
                        '    primary=measured["coco_mask_deltas_points"]["expectation_exact__vs__variable_iou"]["AP"]')


def closed_archive(base):
    receipts = {}
    for name, expected, terminal, phase in (
         ("render",BASE_RENDER,BASE_RENDER_TERMINAL,"capture"),
         ("score",BASE_SCORE,BASE_SCORE_TERMINAL,"verify")):
        d=json.loads(pin(base/name/"RESULT_SAFE.json",expected))
        t=json.loads(pin(base/("observed_"+name)/"TERMINAL.json",terminal))
        need(t["phase"]==phase and t["actual_exit"]==0 and t["output_footer_verified"] is True
             and t["result_sha256"]==expected and t["worker_sha256"]==d["worker_sha256"]==BASE_WORKER,
             "Exact actual original target closure")
        receipts[name]=d
    need(receipts["render"]["gate_labels_opened"] is False and receipts["score"]["gate_labels_opened"] is True
         and receipts["score"]["render_result_sha256"]==BASE_RENDER and receipts["score"]["original_control_parity"],
         "Frozen previously scored SAR-SSP author validation comparison")
    return receipts


def verify_old(saved, new):
    need(new["image_id"]==saved["image_id"] and new["rgb_sha256"]==saved["rgb_sha256"]
         and new["edits"].keys()==saved["edits"].keys(), "Exact image, decoded pixel hash and original edited roster")
    for rank,prev in saved["edits"].items():
        current=new["edits"][rank]
        need(current["k"]==prev["k"] and current["coordinates"]==prev["coordinates"]
             and set(current["arms"])==set((*OLD_ARMS,*NEW_ARMS))
             and all(current["arms"][a]==v for a,v in prev["arms"].items()),
             "Every historical action, exact RLE, signed quota and point order reproduced")


def equivalent(a,b):
    if a is None or b is None:
        return a is b
    return abs(a-b)<1e-11


def parity(previous,new):
    for key in ("coco_mask_metric_points","fixed_native_match_all_truth"):
        a,b=previous[key],new[key]
        if key=="fixed_native_match_all_truth":
            a,b=a["arms"],b["arms"]
        need(set(a).issubset(b) and all(a[arm].keys()<=b[arm].keys() and all(
             equivalent(v,b[arm][k]) for k,v in a[arm].items()) for arm in a),
             "Exact archived target arm and metric parity")


def run(a):
    term=json.loads(pin(a.work_root/"observed_render/TERMINAL.json",NEW_RENDER_TERMINAL))
    ready=json.loads(pin(a.work_root/"render/RESULT_SAFE.json",NEW_RENDER))
    pin(HERE/"transfer_expected.py",RENDER_AUTHORITY)
    need(a.phase=="score" and term["result_sha256"]==NEW_RENDER
         and term["actual_exit"]==0 and term["output_footer_verified"] is True
         and term["worker_sha256"]==ready["worker_sha256"]==RENDER_AUTHORITY
         and ready["status"]==RENDER_STATUS and ready["gate_labels_opened"] is False,
         "Exact prior label-free closed render with original worker")
    past=closed_archive(a.base_run)
    archive_rows=[json.loads(line) for line in pin(a.base_run/"render/RENDER_PRIVATE.jsonl",
                   past["render"]["raw_sha256"]).splitlines()]
    need(len(archive_rows)==232 and sum(len(r["edits"]) for r in archive_rows)==10393,
         "Exact complete original target render")
    base=a.base_run/"transfer.py"
    bundle=ModuleType("_admitted_exact_sarssp_parent")
    bundle.__file__=str(Path(__file__).resolve())
    exec(compile(adapted(pin(base,BASE_WORKER).decode("utf-8")),str(base),"exec"),bundle.__dict__)
    need(bundle.NEW_ARMS==OLD_ARMS and bundle.HERE==HERE,"Exact known original SAR-SSP arm layout")
    bundle.NEW_ARMS=(*OLD_ARMS[:-1],*NEW_ARMS,OLD_ARMS[-1])
    bundle.LOCK=LOCK
    bundle.RENDER_AUTHORITY=RENDER_AUTHORITY
    bundle.RENDER_STATUS=RENDER_STATUS
    bundle.SCORE_STATUS=SCORE_STATUS
    bundle.verify_old=lambda i,new:verify_old(archive_rows[i],new)
    metrics=Counter()
    old_finish=bundle.finish
    old_setup=bundle.setup
    holder=ModuleType("_verified_expected_iou")
    holder.__file__=str(a.method)
    sys.path.insert(0,str(bundle.OLD/"numpy_1_26_4_only"))
    sys.path.insert(0,str(bundle.OLD/"capture/CODEC_PRIVATE"))
    exec(compile(pin(a.method,EXPECTED),str(a.method),"exec"),holder.__dict__)

    def setup():
        *args,writer=old_setup()
        if a.phase!="render":
            return (*args,writer)

        def select(p,static,k,area):
            plugin=writer.select(p,static,k,area)
            exact,audit=holder.select(writer,p,static,k,area)
            need(set(exact)==set(NEW_ARMS) and len(exact[NEW_ARMS[1]]["action"][0])==k,
                 "Two expectation arms at fixed and unrestricted allocations")
            need(audit["menu_actions"]==plugin["menu_actions"] and
                 audit["evaluated"]+audit["pruned"]==audit["menu_actions"],
                 "Identical finite menu and exact evaluation ledger")
            for key,value in audit.items():
                if key=="max_gap_bound":
                    metrics[key]=max(metrics[key],value)
                else:
                    metrics[key]+=value
            for exact_arm, old_arm in zip(NEW_ARMS,("variable_iou","fixed_iou"),strict=True):
                metrics["changed_"+exact_arm]+=int(exact[exact_arm]["action"]!=plugin[old_arm]["action"])
            metrics["mask_pairs"]+=1
            return dict(plugin,**exact)

        return (*args,SimpleNamespace(select=select,write_mask=writer.write_mask))

    def finish(output,status,started,**outcome):
        if status==SCORE_STATUS:
            parity(past["score"]["scoring"],outcome["scoring"])
        else:
            need(metrics["mask_pairs"]==10393,"Every old positive quota has two expectation choices")
            outcome["expected_iou_audit"]=dict(metrics)
        old_finish(output,status,started,original_render_sha256=BASE_RENDER,original_score_sha256=BASE_SCORE,
                   expected_core_sha256=EXPECTED,all_before_labels_historical_rles_and_geometry_exact=True,**outcome)

    bundle.setup=setup
    bundle.finish=finish
    (bundle.render if a.phase=="render" else bundle.score)(a)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("render","score"))
    for name in ("base-run","method","work-root","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and sys.flags.isolated and sys.dont_write_bytecode
         and a.phase=="score" and a.output.resolve()==a.work_root.resolve()/"score_retry" and not a.output.exists(),
         "Fresh isolated offline CPU stage")
    pin(HERE/"PRELOCK.md",LOCK)

    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("Offline retrospective replay")
    sys.addaudithook(offline)
    run(a)


if __name__=="__main__":
    main()
