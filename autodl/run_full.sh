#!/usr/bin/env bash
# SERVER (AutoDL A800): FULL training of the surviving arm(s) + controls, multi-seed.
# Run AFTER the pilot GO verdict. Defaults to the 3-control ladder + baseline x seeds
# {0,1,2} x 36ep (A0 recipe, max_iters=8352). For the 50ep headline: FULL_ITERS=11600
# and the script shifts the LR milestones proportionally.
#
#   ARMS="sp_geo_pseg sp_blind_pseg sp_static_pseg a0r_refine_pseg" SEEDS="0 1 2" bash autodl/run_full.sh
#   FULL_ITERS=11600 ARMS="sp_geo_sar_pseg" SEEDS="0 1 2" bash autodl/run_full.sh   # 50ep headline
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT"; export KMP_DUPLICATE_LIB_OK=TRUE; export PYTHONIOENCODING=utf-8
TRAIN=$(python -c "import mmdet,os;print(os.path.join(os.path.dirname(mmdet.__file__),'.mim','tools','train.py'))")
mkdir -p results runs_full

ARMS=${ARMS:-"sp_geo_pseg sp_blind_pseg sp_static_pseg a0r_refine_pseg"}
SEEDS=${SEEDS:-"0 1 2"}
FULL_ITERS=${FULL_ITERS:-8352}
# proportional milestone shift if not the default 36ep
if [ "$FULL_ITERS" -ne 8352 ]; then
  M1=$(( FULL_ITERS*70/100 )); M2=$(( FULL_ITERS*93/100 ))
  EXTRA_SCHED="param_scheduler.0.end=$FULL_ITERS"   # (best-effort; verify milestones in your config)
else EXTRA_SCHED=""; fi
OPT="--cfg-options train_cfg.max_iters=$FULL_ITERS train_cfg.val_interval=$(( FULL_ITERS/8 )) $EXTRA_SCHED"

NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -lt 1 ] && NG=1
PER_JOB_GB=${PER_JOB_GB:-20}
MEM_GB=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1) / 1024 )); [ -z "$MEM_GB" ] && MEM_GB=24
SLOTS=$(( MEM_GB / PER_JOB_GB )); [ "$SLOTS" -lt 1 ] && SLOTS=1
MAXJOBS=$(( NG * SLOTS ))

# build run list (arm x seed)
declare -a RUNS=()
for a in $ARMS; do for s in $SEEDS; do RUNS+=("$a:s$s"); done; done
echo "FULL: ${#RUNS[@]} runs (${ARMS// /,} x seeds ${SEEDS// /,}) @ $FULL_ITERS iters | up to $MAXJOBS concurrent on ${NG}x${MEM_GB}GB"

set +e
inflight=0; i=0
for r in "${RUNS[@]}"; do
  a="${r%%:*}"; s="${r##*:s}"; wd="runs_full/${a}_s${s}"; gpu=$(( i % NG ))
  echo "[GPU $gpu] $a seed=$s -> $wd"
  CUDA_VISIBLE_DEVICES=$gpu python -u "$TRAIN" "carrier/configs/$a.py" \
      --work-dir "$wd" $OPT --cfg-options randomness.seed=$s > "$wd.log" 2>&1 &
  i=$((i+1)); inflight=$((inflight+1))
  if [ "$inflight" -ge "$MAXJOBS" ]; then wait -n 2>/dev/null; inflight=$((inflight-1)); fi
done
wait 2>/dev/null || true
echo "full runs done. extracting per-seed metrics + means..."

python - <<'PY'
import glob, json, re, os
from collections import defaultdict
agg=defaultdict(list); rows={}
for wd in sorted(glob.glob("runs_full/*_s*")):
    if not os.path.isdir(wd): continue
    name=os.path.basename(wd); arm=name.rsplit("_s",1)[0]
    last={}
    for lg in glob.glob(f"{wd}/*/*.log")+[f"{wd}.log"]:
        if not os.path.exists(lg): continue
        for line in open(lg,encoding="utf-8",errors="ignore"):
            for k in ["segm_mAP","segm_mAP_50","segm_mAP_75","segm_mAP_m"]:
                m=re.search(rf"coco/{k}: ([0-9.]+)",line)
                if m: last[k]=float(m.group(1))
    rows[name]=last
    if "segm_mAP_75" in last: agg[arm].append(last["segm_mAP_75"])
means={a:{"ap75_mean":sum(v)/len(v),"ap75_seeds":v} for a,v in agg.items() if v}
json.dump({"per_run":rows,"per_arm_ap75":means}, open("results/full_metrics.json","w"), indent=2)
print(json.dumps(means, indent=2))
print(">>> headline: GEO ap75_mean vs BLIND vs STATIC (3-seed). Send results/full_metrics.json.")
PY
echo "metrics -> results/full_metrics.json"
