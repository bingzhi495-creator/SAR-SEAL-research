# D-FINE-seg-S 固定 HRSID DEV705 实测

D-FINE-seg-S 使用锁定的 RF-DETR 原生种子 5202 样本访问顺序，完成 24 轮、70,560 次 FIT 访问和 17,640 次优化器、EMA 与调度器更新。固定末轮作者格式 EMA 在训练进程关闭后独立验收。此次采用作者 v0.7.0 原生推理，640 输入、300 个内部查询、置信阈值 0、框 NMS 0.70 和掩膜阈值 0.50。评估使用统一开发图像与 COCO 评价口径，架构、优化配方、原生掩膜处理和候选生成各自沿用其协议。 [训练协议](code/DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/PROTOCOL.md) · [评价协议](code/DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705/PROTOCOL.md)

| DEV705 评价口径 | mask AP | AP50 | AP75 | AP90 | APs | APm | APl | AR100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 1,782 条标注 | 48.160 | 87.934 | 51.507 | 0.145 | 47.196 | 58.440 | 20.817 | 54.018 |
| 去除四条空栅格标注 | 48.278 | 88.071 | 51.662 | 0.145 | 47.306 | 58.440 | 20.817 | 54.139 |

独立验证了 705 张图的预测、候选审计和原始自然终止。全部后 NMS 候选为 184,687 条，其中 57,431 条掩膜为空。每图 COCO top100 合计 70,500 条，其中 18,953 条为空。评分保留全部后 NMS 候选和原分数。评分端先写入标签前回执，之后以固定原标注计算指标。敏感性仅删除预先锁定的四条空栅格标注，预测完全相同。 [图像导出](results/DFINE_E24_DEV705_EXPORT_ACTUAL_SAFE_20261112.json) · [独立导出读回](results/DFINE_E24_DEV705_EXPORT_INDEPENDENT_SAFE_20261112.json) · [标签前回执](results/DFINE_E24_DEV705_SCORE_PRELABEL_SAFE_20261112.json) · [实际评分](results/DFINE_E24_DEV705_SCORE_ACTUAL_SAFE_20261112.json)

图像导出工作体用时 103.471 秒，其中 705 次前向与后处理合计 15.416 秒，CUDA 峰值已分配 866,940,416 字节。CPU 评分用时 10.995 秒。本次结果限于固定 HRSID DEV705 和该已验收单种子检查点。未访问官方 TEST。

另用原生 pycocotools 独立复算两个标注口径的 26 项 COCO 指标，全部与原评分一致至 10⁻¹²。复算进程正常退出，未运行模型或读取正式 TEST。 [独立指标复算](results/DFINE_E24_DEV705_SCORE_STOCK_INDEPENDENT_SAFE_20261112.json)
