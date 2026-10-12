#!/usr/bin/env bash
# SERVER (AutoDL): re-run ONLY the 3 arms that crashed on the rasterizer NaN
# (sp_static / sp_geo / sp_geo_sar), now fixed. R0(baseline) + R2(blind) already
# trained in run_pilot.sh -> leave them. Then re-extract all 5 metrics.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT"; export KMP_DUPLICATE_LIB_OK=TRUE; export PYTHONIOENCODING=utf-8
TRAIN=$(python -c "import mmdet,os;print(os.path.join(os.path.dirname(mmdet.__file__),'.mim','tools','train.py'))")
mkdir -p results
PILOT_ITERS=1856
OPT="--cfg-options train_cfg.max_iters=$PILOT_ITERS train_cfg.val_interval=928 default_hooks.checkpoint.interval=$PILOT_ITERS"

declare -a RUNS=("sp_static_pseg:R1_static" "sp_geo_pseg:R3_geo" "sp_geo_sar_pseg:R4_geo_sar")
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -lt 1 ] && NG=1
PER_JOB_GB=${PER_JOB_GB:-20}
MEM_GB=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1) / 1024 )); [ -z "$MEM_GB" ] && MEM_GB=24
SLOTS=$(( MEM_GB / PER_JOB_GB )); [ "$SLOTS" -lt 1 ] && SLOTS=1
MAXJOBS=$(( NG * SLOTS ))
echo "re-running 3 fixed arms, up to $MAXJOBS concurrent on ${NG}x${MEM_GB}GB"

set +e
inflight=0; i=0
for r in "${RUNS[@]}"; do
  cfg="${r%%:*}"; wd="runs/${r##*:}"; gpu=$(( i % NG )); rm -rf "$wd"
  echo "[GPU $gpu] $cfg -> $wd"
  CUDA_VISIBLE_DEVICES=$gpu python -u "$TRAIN" "carrier/configs/$cfg.py" --work-dir "$wd" $OPT > "$wd.log" 2>&1 &
  i=$((i+1)); inflight=$((inflight+1))
  if [ "$inflight" -ge "$MAXJOBS" ]; then wait -n 2>/dev/null; inflight=$((inflight-1)); fi
done
wait 2>/dev/null || true
echo "fixed arms done. extracting ALL 5 metrics..."

python - <<'PY'
import glob, json, re, os
res={}
for wd in ["R0_rank1_base","R1_static","R2_blind","R3_geo","R4_geo_sar"]:
    last={}
    for lg in glob.glob(f"runs/{wd}/*/*.log")+[f"runs/{wd}.log"]:
        if not os.path.exists(lg): continue
        for line in open(lg,encoding="utf-8",errors="ignore"):
            for k in ["segm_mAP","segm_mAP_50","segm_mAP_75","segm_mAP_s","segm_mAP_m","segm_mAP_l"]:
                m=re.search(rf"coco/{k}: ([0-9.]+)",line)
                if m: last[k]=float(m.group(1))
    res[wd]=last or "NO_VAL_FOUND"
json.dump(res, open("results/pilot_metrics.json","w"), indent=2)
print(json.dumps(res, indent=2))
print("\n>>> KEY: R3_geo segm_mAP_75 vs R2_blind vs R1_static (delta-of-deltas). Send results/pilot_metrics.json.")
PY
echo "metrics -> results/pilot_metrics.json"
