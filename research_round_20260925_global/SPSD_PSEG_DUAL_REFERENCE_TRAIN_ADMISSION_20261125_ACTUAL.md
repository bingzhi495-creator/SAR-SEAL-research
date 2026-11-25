# SPSD 与 PSeg-SSDD 双参考训练数据验收

同一批 **928 张训练 JPEG** 和 **232 张测试 JPEG** 在两套作者档案中逐字节相同。沿用 Official-SSDD 发布的浮点船体多边形与 SLGSea 发布的原生 RGB 实例 ID。测试 JPG 在本次只用来筛除重复训练图。按 JPEG 字节与含原始高宽的解码 RGB 双签名，训练侧剔除 10 个测试影像对应的 **12 张训练副本**。12 张的编码及解码均相同，无仅解码相同的新增命中。训练侧另隔离一张含不完整实例标注的图。测试集维持原有 232 张。 [Official-SSDD 作者发布](https://github.com/TianwenZhang0825/Official-SSDD) · [SLGSea 作者发布](https://github.com/deng1218/SLGSea#datasets)

压缩包检查发现一张 SPSD 训练图的 JSON 声明了五个正面积船体，其中两个实例 ID 在对应 PNG 上没有像素。其余所有声明的实例 ID 和全部非空约定像元均已核验。第一次严格验收按预锁退出并保留失败记录。之后按影像隔离这条训练记录，保留全量审核分母及异常记录。两份参考仍各用原发布几何。 [预锁](code/SPSD_PSEG_DUAL_REFERENCE_TRAIN_ADMISSION_20261125/PRELOCK.md) · [输入澄清](code/SPSD_PSEG_DUAL_REFERENCE_TRAIN_ADMISSION_20261125/ERRATUM.md)

| 训练范围 | JPEG | PSeg 多边形 | SPSD 非空船体 | 指派配对 | IoU ≥ 0.50 | 中位配对 IoU | 中位面积比 SPSD／PSeg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原作者训练总体 | 928 | 2,041 | 2,039 | 2,034 | 1,904 | 0.7451 | 0.8269 |
| 准入双参考训练 | 915 | 1,990 | 1,990 | 1,985 | 1,858 | 0.7459 | 0.8305 |
| 跨测试精确重复 | 12 | 46 | 46 | 46 | 43 | 0.6909 | 0.7197 |
| 实例 ID 不完整 | 1 | 5 | 3 | 3 | 3 | 0.7285 | 0.7511 |

配对通过同一原生影像内最大总栅格 IoU 的一对一指派得到。准入训练样本中没有完全相同的单个参考掩膜，亦无完全相同的影像船体并集。该结果描述两种发布参考之间的形状差别。候选预测与图像配准不参与统计。

915 张准入图形成 **901 个精确 RGB 组**，包含 888 个单图组、12 个双图组和一个三图组。13 个多图组内，分别以 PSeg 或 SPSD 的栅格实例集合比较，均存在组内标注差异。后续训练和验证应保持组完整，并按组等权处理重复记录。原 COCO 影像记录、SPSD 标注与训练文本、归档的候选来源说明和 JPEG EXIF 字段未提供可核验的母景或采集产品标识。本次只确认精确图像重复，场景级重叠与独立性仍未验证。

可公开查到的同一 HRSID E24 D-FINE-seg-S 来源在现有 PSeg DEV187 与 TEST232 上有已核验冻结推理。既有 PSeg FIT109 的目标载体捕获是 RF-DETR。未找到可直接复用、能覆盖 915 张准入训练影像的同 D-FINE E24 捕获记录。后续先在上述名单作不读标注的冻结候选捕获，独立核验后才可拟合。 [已核验 D-FINE DEV187](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/main/research_round_20260925_global/DFINE_HRSID_TO_PSEG_SSDD_DEV187_FROZEN_TRANSFER_20261120.md) · [已核验 TEST232](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/main/research_round_20260925_global/code/PSEG_TEST232_DFINE_FROZEN_SCORE_20261121/README.md)

修正验收已实际退出 0，终端返回的结果 SHA-256 与文件一致。独立读回逐项核验组划分、准入条件、双参考计数与结果散列。当前阶段没有模型前向、优化更新或测试标注读取。公开汇总、程序与预锁可用于复查。专用可训练名单含文件名、影像与标注签名及实例关联，保存在本地。

## 源域拟合与留出

按已受理 RGB 组哈希的固定顺序划分，721 组 732 图用于拟合，180 组 183 图用于源域留出。跨两部分没有文件名、编码或解码重叠。独立读回核对划分完整与私有名单散列。该顺序使用影像身份与预锁盐值，未消费标注统计或模型分数。下一阶段在同一检测候选、同容量和固定 1,000 次更新下比较写回方向。[划分结果](results/SPSD_PSEG_DUAL_REFERENCE_GROUP_SPLIT_20261125_SAFE.json) · [拟合协议与划分源码](code/SPSD_PSEG_DUAL_REFERENCE_GROUP_SPLIT_20261125)

[完整验收聚合](results/SPSD_PSEG_DUAL_REFERENCE_TRAIN_ADMISSION_20261125_SAFE.json) · [独立读回](results/SPSD_PSEG_DUAL_REFERENCE_TRAIN_ADMISSION_20261125_READBACK_SAFE.json) · [验收源码](code/SPSD_PSEG_DUAL_REFERENCE_TRAIN_ADMISSION_20261125)
