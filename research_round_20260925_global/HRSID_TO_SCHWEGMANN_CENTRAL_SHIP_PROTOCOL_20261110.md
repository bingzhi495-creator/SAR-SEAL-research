# HRSID 到 Schwegmann 的中心船散射掩膜辅助验证

Schwegmann 的正样本 101×101 图块只标注中心船的散射像元。21×21 小窗以作者掩膜质心重新裁切，因此主推理只用原 101×101 图块。原论文与博士论文未明确逐像素绘制或阈值流程。原论文报告船宽相对 AIS 系统性偏高。评价范围为已知中心附近的散射掩膜，整幅船舶 AP、全图虚警和物理船体边界超出该标注口径。 [作者论文](https://sarmaps.gitlab.io/data/dataset_paper.pdf) · [字段](https://gitlab.com/sarmaps/sar-ship-dataset/-/raw/9decec2a18b9d7d6fd2d057a435808ca67d2bad1/ship_positives.md) · [博士论文](https://repository.up.ac.za/server/api/core/bitstreams/903dd928-b728-44c2-a150-74dc430837d1/content)

## 固定数据与实际资格

已按固定作者提交分段取得原始 289,068,441 字节。完整 SHA-256 为 `78ff85f49eb5102716ae6d39bdbafaf27fc9c63a630be0d86a2c434ab09f053b`，Git blob 为 `0df2987ad896f2b0f4b8db644cb5ca3cff1ce986`，均与官方 API 的元数据一致。完整读取 1,596 条正样本及 25 个产品名。作者短 ID 绑定产品末尾唯一后缀。初次资格入口误用数据段号，未生成样本。修正后核对 25 个短别名和产品名一一对应。9 对 Sentinel-1 产品共享日期、绝对轨道和数据段，按同源切片归并。固定 16 个保守采集组，其中 13 组 Sentinel-1A、3 组 RADARSAT-2。按两篇原论文的采集来源和 Sentinel-1B 的发射日期，可排除与 HRSID 的同次采集重用。两数据仍共享 Sentinel-1 任务系列。 [固定数据 API](https://gitlab.com/api/v4/projects/sarmaps%2Fsar-ship-dataset/repository/files/json%2Fship_positives.json/raw?ref=9decec2a18b9d7d6fd2d057a435808ca67d2bad1) · [作者许可](https://gitlab.com/sarmaps/sar-ship-dataset/-/raw/9decec2a18b9d7d6fd2d057a435808ca67d2bad1/README.md) · [HRSID 原论文](https://doi.org/10.1109/ACCESS.2020.3005861) · [ESA 发射记录](https://www.esa.int/Newsroom/Press_Releases/Sentinel-1B_launched_to_complete_radar_pair)

像素全部有限，18 条记录含负值，其中 VV 8 条、VH 10 条。全非负资格的第一次运行在包生成前退出，原失败保留。修正后按图像数值资格排除含负值记录。作者未明确存储单位或负值含义。非负记录使用统一的实验输入约定，将数值取 \(10\log_{10}(\max(x,10^{-4}))\)，截到 [−40, +10]，再线性映射到 8 位并取最近偶数。灰度复制为三通道。按 VV 或 HH、真实输入去重及固定哈希次序，每组取至多两图。782 条同极化记录符合数值资格，最终全部 16 组各选两图，共 32 图。选中输入的低高端截断比例为约 0.0003% 和 0.0126%。推理包与评分掩膜分离，图像和二值参考均经无损读回，模型前向为零。 [实际资格汇总](results/SCHWEGMANN_CENTRAL_SHIP_COHORT_ACTUAL_SAFE_20261110.json)

## 固定推理与评分

复用已闭合且独立验收的 HRSID 5201、5202、5203 原生和逐步中心 E24 EMA。六臂使用同一 Windows 单线程 CPU、FP32、原生 576 验证变换及 164 查询。以跨类别槽的原生 top100 为起点，再保留船类槽 0。将原生框和掩膜恢复到 101×101，取预测框中心在固定 [40,61)×[40,61) 的最高原生排名候选。输出保留完整 101 掩膜和原始分数。候选不存在或选中空掩膜时主 IoU 记零。推理端仅使用 32 张图及封存模型，所有导出闭合后评分端才读取作者 `patchgt`。

主读数为采集等权的漏选计零 IoU。另报 Dice、0.50/0.75/0.90 恢复率、各臂选中率、空掩膜和共同非空条件形状。三组训练种子使用同一图像和采集组，固定 4,096 次采集重采样，种子 254101。实用幅度为 1 个 IoU 点，正式方向判定至少需要 20 个物理采集组。当前 16 组用于描述性比较。输入 101→576 为固定源模型变换，最终指标回到 101 原网格。 [推理源码](code/SCHWEGMANN_FROZEN_TRANSFER_20261110/export_schwegmann.py) · [评分源码](code/SCHWEGMANN_FROZEN_TRANSFER_20261110/score_schwegmann.py)

## 六臂实测与独立复算

六个原导出进程和原评分进程均闭合并退出 0。每臂 32 次真实前向，合计 192 次。单臂推理主体用时 71.42 至 74.06 秒。独立复算用 pycocotools 解码六份闭合候选和隔离的原始二值掩膜，核对选中候选、IoU、Dice、恢复率、共同非空条件与 4,096 次采集组重采样。结果与原评分一致。 [原评分安全汇总](results/SCHWEGMANN_CENTRAL_SHIP_ACTUAL_SCORE_SAFE_20261110.json) · [独立复算凭据](results/SCHWEGMANN_CENTRAL_SHIP_INDEPENDENT_READBACK_SAFE_20261110.json)

| 源训练种子 | 原生平均 IoU | 中心平均 IoU | 差值，百分点 | 非空候选，原生→中心 |
| --- | ---: | ---: | ---: | ---: |
| 5201 | 38.861% | 40.630% | +1.769 | 26→29 |
| 5202 | 44.630% | 45.675% | +1.045 | 31→32 |
| 5203 | 36.042% | 44.946% | +8.905 | 23→28 |
| 均值 | 39.844% | 43.751% | +3.906 |  |

均值差的采集组重采样区间为 −0.482 至 +8.644 点。Dice 平均差为 +6.112 点，区间 +1.022 至 +11.847 点，为次要读数。两臂共同选中非空候选时，三组 IoU 分别变化 −4.223、−1.661、+7.464 点。IoU 0.75 达标数为 4→1、6→1、4→4，0.90 为 0→0、1→0、2→1。主 IoU 与非空候选数在三组都增加，高阈值恢复未改善。组数 16 低于预锁的 20，保留原判定 `INSUFFICIENT_PHYSICAL_ACQUISITIONS`。现有结果限于中心已知的船舶散射掩膜描述性比较，完整场景 AP、虚警和强外推仍未验证。
