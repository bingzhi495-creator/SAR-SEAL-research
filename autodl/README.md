# AutoDL Swin-L 跑包（80GB GPU，冲过 LFG-Net 64.2/81.2）

目标：用 80GB 大 batch 快速跑 **Mask2Former-Swin-L（COCO-panoptic 初始化）** on PSeg-SSDD，
拿到整体/近岸/近海 mask AP，看是否超过其他所有方法（最强对手 LFG-Net 64.2 / AP75 81.2）。
权重走 URL 自动下载（不用上传）。

## 1. 上传到 AutoDL（建议 /root/autodl-tmp/）
| 本地 | AutoDL 目标 |
|---|---|
| `E:\SAR\seal\carrier\` + `autodl\` （整个代码，含自定义模块）| `/root/seal/`（即代码根目录）|
| `E:\SAR\seal\data\pseg_ssdd\annotations\*.json`（4 个）| `/root/autodl-tmp/pseg_ssdd/annotations/` |
| `E:\PSeg_SSDD\voc_style\JPEGImages_train\`（928 图）| `/root/autodl-tmp/PSeg_SSDD/JPEGImages_train/` |
| `E:\PSeg_SSDD\voc_style\JPEGImages_test\`（232 图）| `/root/autodl-tmp/PSeg_SSDD/JPEGImages_test/` |

（权重 1.2GB 不用传——配置里 load_from 是 URL，AutoDL 联网自动下。）

## 2. 装环境
```bash
cd /root/seal && bash autodl/setup.sh
```

## 3. 改数据路径（若与上面不同）
编辑 `carrier/configs/autodl_swinl_pseg.py` 顶部 3 行 data_root/img_train/img_test。

## 4. 跑（依次：Swin-L 基线 + 组合版，各做三切分评估）
```bash
cd /root/seal && bash autodl/run.sh
```
跑两个：① `autodl_swinl_pseg`（Swin-L 载体单独）② `autodl_swinl_combo_pseg`（**组合版** = Swin-L + GeoField + 边界loss）。
对比这俩 = 载体 vs 载体+机制的消融。**只想跑组合版省钱**：把 run.sh 末尾第一行 `run_one autodl_swinl_pseg ...` 注释掉即可。

## 5. ⭐ 用满 80GB（省钱关键）
配置默认 `batch_size=16, lr=1e-4`。跑起来后 `nvidia-smi` 看显存：
- **显存 <70GB** → 调大 batch 更快：`--cfg-options train_dataloader.batch_size=24 optim_wrapper.optimizer.lr=1.5e-4`（lr 按 batch/16×1e-4 线性缩放）
- **OOM** → 调小：batch=12 → lr=7.5e-5；batch=8 → lr=5e-5
- 也可直接在 run.sh 的 train 行加 `--cfg-options ...`

50 epoch（batch16 ≈ 2900 iter）约 1-2h（80GB），成本可控。

## 6. 跑完发回
- 终端打印的 3 段 AP（test/inshore/offshore 的 segm_mAP / AP50 / AP75）
- `runs/autodl_swinl/best_coco_segm_mAP_*.pth`（最佳权重，回本地继续做消融/出图）

## 对照（PSeg-SSDD 整体 mask AP，要超过的）
Mask R-CNN 57.8/73.2 · HTC 57.6/72.3 · MAI-SE-Net 63.0/77.6 · **LFG-Net 64.2/81.2** · 本地 SEAL-R50 61.3/75.6
→ Swin-L 目标 **>64.2 / >81.2**（大概率能到）。
