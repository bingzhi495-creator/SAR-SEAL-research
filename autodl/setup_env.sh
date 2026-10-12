#!/usr/bin/env bash
# SERVER (AutoDL): install the exact env + remap hard-coded E: paths to this root.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # /root/autodl-tmp/seal
echo "SEAL root: $ROOT"

# 1. exact deps (match local: torch 2.0.1+cu118 from the base image; do NOT reinstall torch)
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
pip install -U openmim
mim install mmengine==0.10.4 "mmcv==2.1.0" mmdet==3.3.0
pip install scipy opencv-python-headless "pycocotools" "numpy<2"

# 2. remap absolute Windows paths in every config -> this root
cd "$ROOT"
mapfile -t FILES < <(grep -rl -e 'E:/SAR/seal' -e 'E:/PSeg_SSDD' carrier/configs/ 2>/dev/null || true)
for f in "${FILES[@]}"; do
  sed -i "s|E:/PSeg_SSDD|$ROOT/PSeg_SSDD|g; s|E:/SAR/seal|$ROOT|g" "$f"
done
echo "remapped ${#FILES[@]} config files."

# 3. sanity: can we build the geo config + see the data?
export PYTHONPATH="$ROOT"; export KMP_DUPLICATE_LIB_OK=TRUE
python - <<'PY'
import os, warnings; warnings.filterwarnings('ignore')
from mmengine.config import Config
import carrier.geokernel_m2f_head, carrier.seal_m2f_head  # noqa
c=Config.fromfile('carrier/configs/sp_geo_pseg.py')
ann=c.train_dataloader.dataset.ann_file
img=c.train_dataloader.dataset.data_prefix['img']
print('ann exists :', os.path.exists(ann), ann)
print('img dir ok :', os.path.isdir(img), img)
print('SETUP OK' if os.path.exists(ann) and os.path.isdir(img) else 'SETUP INCOMPLETE: fix paths above')
PY
