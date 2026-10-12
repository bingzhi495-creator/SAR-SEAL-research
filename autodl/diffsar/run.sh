#!/bin/bash
# ============================================================================
# Phase A: reproduce DiffSARShipInst on PSeg-SSDD (AutoDL Linux, single GPU).
# Goal: get the REAL reproducible-SOTA numbers (apples-to-apples) + a working
# pipeline to then ADD our SAR innovations to (Phase B).
#
# RUN FROM the DiffSARShipInst repo root, after:
#   git clone https://github.com/XiaowoXuSAR/DiffSARShipInst && cd DiffSARShipInst
# Set the two paths below to YOUR AutoDL data, then: bash run.sh
# ============================================================================
set -u

# ---- EDIT THESE TO YOUR AUTODL PATHS ----
PSEG=/root/autodl-tmp/pseg_ssdd                 # has annotations/{train,test,test_inshore,test_offshore}.json
IMG_TRAIN=/root/autodl-tmp/pseg_ssdd/PSeg_SSDD/JPEGImages_train   # train images dir
IMG_TEST=/root/autodl-tmp/pseg_ssdd/PSeg_SSDD/JPEGImages_test     # test images dir
# -----------------------------------------

ANN=$PSEG/annotations
REPO=$PWD

echo "========== [1/4] build detectron2 (bundled) + deps =========="
pip install -e . 2>&1 | tail -3
pip install timm einops 2>&1 | tail -1

echo "========== [2/4] lay out PSeg-SSDD as datasets/coco/ (detectron2 coco_2017_train/val) =========="
mkdir -p datasets/coco/annotations
ln -sf "$ANN/train.json"          datasets/coco/annotations/instances_train2017.json
ln -sf "$ANN/test.json"           datasets/coco/annotations/instances_val2017.json
ln -sf "$ANN/test.json"           datasets/coco/annotations/test.json
ln -sf "$ANN/test_inshore.json"   datasets/coco/annotations/test_inshore.json
ln -sf "$ANN/test_offshore.json"  datasets/coco/annotations/test_offshore.json
ln -sfn "$IMG_TRAIN"  datasets/coco/train2017
ln -sfn "$IMG_TEST"   datasets/coco/val2017
echo "linked. train imgs: $(ls datasets/coco/train2017 2>/dev/null | wc -l), val imgs: $(ls datasets/coco/val2017 2>/dev/null | wc -l)"
echo "NOTE: if config MODEL.WEIGHTS needs an R101 ImageNet pkl, detectron2 will fetch it; if offline, pre-download R-101.pkl."

echo "========== [3/4] TRAIN =========="
python train_net.py --num-gpus 1 --config-file configs/DiffSARShipInst-SSDD-train.yaml 2>&1 | tail -30
CK=$(ls -t output*/model_final.pth output*/*.pth 2>/dev/null | head -1)
echo "trained ckpt: $CK"

echo "========== [4/4] EVAL 3-split (swap val json) =========="
for split in test test_inshore test_offshore; do
  ln -sf "$ANN/${split}.json" datasets/coco/annotations/instances_val2017.json
  echo "----- EVAL $split -----"
  python train_net.py --num-gpus 1 --config-file configs/DiffSARShipInst-SSDD-test.yaml \
    --eval-only MODEL.WEIGHTS "$CK" 2>&1 | grep -iE "segm|AP|copypaste" | tail -8
done
echo "[diffsar] DONE. 把三段 segm AP 发回 (我们 R50 基线: 整体.613 近岸.533 远岸.648; DiffSAR论文远岸70.6/90.8)."
