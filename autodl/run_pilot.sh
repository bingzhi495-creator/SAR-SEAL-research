#!/usr/bin/env bash
# SERVER (AutoDL): run the GeoKernel M3 pilot (8 epochs each), auto-distributed
# across all visible GPUs, then extract val metrics to results/pilot_metrics.json.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT"; export KMP_DUPLICATE_LIB_OK=TRUE; export PYTHONIOENCODING=utf-8
TRAIN=$(python -c "import mmdet,os;print(os.path.join(os.path.dirname(mmdet.__file__),'.mim','tools','train.py'))")
mkdir -p results runs

# 8 epochs. PSeg train ~928 imgs; A0 recipe bs varies. We cap iters for an 8ep pilot.
# (A0 full = 8352 iters @36ep -> 8ep ~= 1856 iters. Override max_iters + val_interval.)
PILOT_ITERS=1856; VAL_INT=928
OPT="--cfg-options train_cfg.max_iters=$PILOT_ITERS train_cfg.val_interval=$VAL_INT default_hooks.checkpoint.interval=$PILOT_ITERS"

# essential 5 arms (delta-of-deltas make-or-break = R3 geo vs R2 blind vs R1 static)
declare -a RUNS=(
  "a0r_refine_pseg:R0_rank1_base"
  "sp_static_pseg:R1_static"
  "sp_blind_pseg:R2_blind"
  "sp_geo_pseg:R3_geo"
  "sp_geo_sar_pseg:R4_geo_sar"
)
# concurrency = sum over GPUs of floor(VRAM/PER_JOB_GB). One big card (A800-80G)
# -> several arms at once on the SAME GPU; multi-GPU -> one+ per card.
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -lt 1 ] && NG=1
PER_JOB_GB=${PER_JOB_GB:-20}                                  # Mask2Former-R50 bs4 ~16-20GB
MEM_GB=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1) / 1024 ))
[ -z "$MEM_GB" ] && MEM_GB=24
SLOTS=$(( MEM_GB / PER_JOB_GB )); [ "$SLOTS" -lt 1 ] && SLOTS=1
MAXJOBS=$(( NG * SLOTS ))
echo "GPUs=$NG  VRAM=${MEM_GB}GB/gpu  ${SLOTS} jobs/gpu  -> up to $MAXJOBS concurrent | ${#RUNS[@]} runs x $PILOT_ITERS iters"

set +e   # one crash must NOT abort the rest; metric extractor flags failures
inflight=0; i=0
for r in "${RUNS[@]}"; do
  cfg="${r%%:*}"; wd="runs/${r##*:}"; gpu=$(( i % NG ))
  echo "[GPU $gpu | slot] $cfg -> $wd"
  CUDA_VISIBLE_DEVICES=$gpu python -u "$TRAIN" "carrier/configs/$cfg.py" \
      --work-dir "$wd" $OPT > "$wd.log" 2>&1 &
  i=$((i+1)); inflight=$((inflight+1))
  if [ "$inflight" -ge "$MAXJOBS" ]; then wait -n 2>/dev/null; inflight=$((inflight-1)); fi
done
wait 2>/dev/null || true
echo "all pilot runs finished. extracting metrics..."

python - <<'PY'
import os, glob, json, re
res={}
for wd in ["R0_rank1_base","R1_static","R2_blind","R3_geo","R4_geo_sar"]:
    logs=sorted(glob.glob(f"runs/{wd}/*/*.log") + glob.glob(f"runs/{wd}.log"))
    last={}
    for lg in logs:
        for line in open(lg, encoding="utf-8", errors="ignore"):
            for k in ["segm_mAP","segm_mAP_50","segm_mAP_75","segm_mAP_s","segm_mAP_m","segm_mAP_l"]:
                m=re.search(rf"coco/{k}: ([0-9.]+)", line)
                if m: last[k]=float(m.group(1))
    res[wd]=last or "NO_VAL_FOUND"
json.dump(res, open("results/pilot_metrics.json","w"), indent=2)
print(json.dumps(res, indent=2))
print("\n>>> KEY: R3_geo segm_mAP_75 vs R2_blind vs R1_static (the delta-of-deltas).")
PY
echo "metrics -> results/pilot_metrics.json"
