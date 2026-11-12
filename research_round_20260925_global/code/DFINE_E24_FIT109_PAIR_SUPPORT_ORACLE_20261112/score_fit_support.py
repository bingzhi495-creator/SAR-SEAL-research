"""Exact FIT109 fixed-point oracle with no detector training or AP."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from types import ModuleType

import numpy as np

PROTOCOL = "33424b59f1dade1853068ce445394edd0855a9a6e210bef7d2a53edf2629fa12"
PAIR = "6a2d1d14c8359f5aed18c54879bed5bf1e10bde575e59e47d3ce4583507f4a11"
SUPPORT = "dcf81861124032292b3dd0b3753ebc83b048032a21a32579355d2068de201927"
MENU = "4b84e51855683acae44c09c2d0859cf249da1daa31edabfb76ea45f09da64555"
OLD = "53ba7adae5d3697cd77e1b147a331889c78ee29d17dea4c51101f9fe421b5aff"
PAIR_PRELABEL = "7c763fd75a8870a84426b2e8a30f8b1e4be6eefedd68dd2e2c88b8c617992519"
PAIR_WAIT = "dc254ce8ff521a4508d0e68aff8e82371f7047c41121a249dc21c8e3b45bc07a"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, expected):
    data = path.read_bytes()
    need(sha256(data).hexdigest()==expected,"Exact "+path.name)
    return data


def load(path, expected, name):
    data = pin(path,expected)
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(data,str(path),"exec"),module.__dict__)
    return module


def metrics(rows):
    groups = defaultdict(list)
    for group,values in rows:
        groups[group].append(values)
    modes = tuple(rows[0][1])
    outcomes = {}
    for mode in modes:
        r = [v[mode] for _,v in rows]
        outcomes[mode] = dict(pair_equal_iou_points=100*mean(r),
                              group_equal_iou_points=100*mean(mean(x[mode] for x in g) for g in groups.values()),
                              strict75=sum(x>=.75 for x in r),strict90=sum(x>=.90 for x in r),
                              better_than_raw=sum(v[mode]>v["raw"]+1e-12 for _,v in rows))
    transitions = dict(rematch_extra90=sum(v["rematched"]>=.90 and v["paired"]<.90 for _,v in rows),
                       unrestricted_extra90=sum(v["unrestricted"]>=.90 and v["rematched"]<.90 for _,v in rows))
    return dict(rows=len(rows),groups=len(groups),outcomes=outcomes,transitions=transitions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","pair","pair-math","old","base","capture","accepted","prelabel","wait",
                "fit-gt","code","shapely-site","output"):
        p.add_argument("--"+key,type=Path,required=True)
    a = p.parse_args()
    start = perf_counter()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","CPU-only scope")
    pin(a.protocol,PROTOCOL)
    pair = load(a.pair,PAIR,"_closed_pair")
    base = load(a.base,pair.BASE,"_closed_base")
    math = load(a.pair_math,pair.MATH,"_closed_math")
    support = load(Path(__file__).with_name("fit_support.py"),SUPPORT,"_fit_support")
    signed = load(a.code/"DFINE_DEV705_INSIDE_EDIT_MENUS_20261112/inside_menus.py",MENU,"_signed_menu")
    old = json.loads(pin(a.old,OLD))
    need(old["FIT"] and not old["DEV"] and not old["TEST"] and not old["advance"],"Exact prior result role")
    _,blocks = base.read_core(a)
    a.prelabel_sha = PAIR_PRELABEL
    a.wait_sha = PAIR_WAIT
    original,planned = pair.sealed_points(a,base,math,blocks)
    need(original["counts"]["pairs"]==194640 and original["counts"]["supported_candidates"]==6167,
         "Exact label-free point population")
    a.output.mkdir()
    pre = base.seal(a.output/"PRELABEL_SAFE.json",base.encoded(dict(status="DFINE_FIT109_PAIR_SUPPORT_PRELABEL",
                    FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,protocol_sha256=PROTOCOL,
                    support_sha256=SUPPORT,menu_sha256=MENU,original_prelabel_sha256=PAIR_PRELABEL)))
    gt,anns = pair.read_labels(a,base,blocks)
    slots,scope = pair.fixed_slots(a,base,blocks,planned,gt,anns)
    rows = []
    for s in slots:
        if s["raw"] is not None:
            T,I = s["raw"]
            q = s["truth"] if s["truth"] is not None else np.empty((0,2))
            need((q>=0).all() and (q<=1+1e-12).all(),"Truth fractions")
            q = np.clip(q,0,1)
            r = support.bounds(s["area"],T,I,q,signed.menu)
            need(r["raw"]<=r["paired"]+1e-12 and r["paired"]<=r["rematched"]+1e-12 and
                 r["rematched"]<=r["unrestricted"]+1e-12,"Nested oracle support")
            rows.append((s["group"],r))
    m = metrics(rows)
    need(m["rows"]==412 and m["groups"]==93,"Exact fixed matched population")
    for current,previous in (("raw","raw"),("paired","oracle")):
        now = m["outcomes"][current]
        prior = old["fixed_geometry"]["metrics"][previous]
        need(abs(now["group_equal_iou_points"]-prior["group_equal_iou_points"])<1e-9
             and now["strict90"]==prior["strict_90"] and now["strict75"]==prior["strict_75"],
             "Exact prior raw and original-pair oracle")
    result = dict(status="DFINE_FIT109_PAIR_SUPPORT_ACTUAL",FIT=True,DEV=False,TEST=False,GT_opened=True,
                  AP=False,detector_forwards=0,optimizer_updates=0,protocol_sha256=PROTOCOL,
                  support_sha256=SUPPORT,menu_sha256=MENU,worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  prior_actual_sha256=OLD,prelabel_sha256=pre,match_scope=scope,
                  summary=m,worker_seconds=perf_counter()-start)
    digest = base.seal(a.output/"RESULT_SAFE.json",base.encoded(result))
    print(json.dumps(dict(result_sha256=digest,summary=m),sort_keys=True),flush=True)


if __name__=="__main__":
    def offline(event,_):
        if event.startswith(("socket.","subprocess.")):
            raise RuntimeError("Offline CPU only")
    sys.addaudithook(offline)
    main()
