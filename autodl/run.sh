#!/bin/bash
# AutoDL: 依次跑 Swin-L 基线 + 组合版(Swin-L+GeoField+边界loss)，各做 scene-split 评估。
# 冲 LFG-Net 64.2/81.2。在代码根目录运行。OOM 则调小 batch_size+按比例调 lr。
set -u
export PYTHONPATH="$PWD" KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8
TRAIN=$(python -c "import mmdet,os;print(os.path.join(os.path.dirname(mmdet.__file__),'.mim/tools/train.py'))")
TEST=$(python -c "import mmdet,os;print(os.path.join(os.path.dirname(mmdet.__file__),'.mim/tools/test.py'))")

run_one () {  # cfg  workdir_tag
  local CFG="carrier/configs/$1.py"; local WD="runs/$2"
  local ANN; ANN=$(python -c "from mmengine.config import Config; print(Config.fromfile('$CFG').val_evaluator['ann_file'].rsplit('/',1)[0])")
  echo "========== TRAIN $1 $(date) =========="
  python "$TRAIN" "$CFG" --work-dir "$WD"
  local CK; CK=$(ls -t "$WD"/best_coco_segm_mAP_*.pth 2>/dev/null | head -1); [ -z "$CK" ] && CK="$WD/iter_2900.pth"
  echo "========== EVAL $1  ($CK) =========="
  for s in test test_inshore test_offshore; do
    echo "----- $1 / $s -----"
    python "$TEST" "$CFG" "$CK" --cfg-options \
      test_dataloader.dataset.ann_file=$ANN/$s.json test_evaluator.ann_file=$ANN/$s.json \
      | grep -E "segm_mAP:|segm_mAP_50:|segm_mAP_75:|segm_mAP_s:|segm_mAP_m:|segm_mAP_l:"
  done
}

run_one autodl_swinl_pseg        autodl_swinl        # 1) Swin-L 基线(载体)
run_one autodl_swinl_combo_pseg  autodl_swinl_combo  # 2) Swin-L + GeoField + 边界loss(组合版)
echo "[autodl] ALL DONE. 把两段 AP + 两个 best_*.pth 发回。对手 LFG-Net 64.2/81.2。"
