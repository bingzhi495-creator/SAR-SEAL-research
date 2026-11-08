# RF-DETR E24 等点数边界采样诊断协议

本诊断的研究假设为，小实例的边界贡献在均匀 BCE 的有限点估计中波动较大，给边界固定配额可以降低边界梯度估计方差，同时保住全图估计。最强替代解释是单纯改了风险权重，或原生点已覆盖大量重要边界。本次用固定预测做覆盖和方差诊断，先于训练。

## 冻结样本及原生参照

- 仅用整数标注与冻结 E24 EMA（4,464 次更新，种子 5201），从已闭合的 516 张 `head_fit` 图按 JPEG SHA-256 去重，组内取最小数据索引。以 `SHA256("sampling-coverage-v1:" + image_sha256)` 字典序选前 96 组。保留这批图的全部已验收 COCO IoU 0.50 匹配候选。固定匹配来自原图评价，用于诊断重复、同候选的采样差异。
- 同一 Windows CUDA RF-DETR 1.12.0 原生 eval 载体，使用 576×576 原生方形变换并联动真值掩膜。无随机几何增强。只使用最终层单组查询和 144×144 的原始 logits；以重建原图掩膜核对已冻结候选。变换后真值面积介于 1 至 1023 像素为小实例。面积为零的匹配单列。
- 直接调用原生 `get_uncertain_point_coords_with_randomness`。每实例 K=1,296，从 3K 候选按插值后的 −|logit| 选 972 点，另取 324 均匀点。固定种子 `5201000 + dataset_index`，8 次重采。报告 GT 前景、两像素 Chebyshev 内外边界带和边界内错分零命中率、平均点数和错误像素数。错误由冻结 logits 在 576 网格中心的 0 阈值决定。
- 原生训练包含匹配器共享均匀点、每层独立的损失采样、13 组训练查询和最终、3 个辅助、编码器共 5 个输出层。本诊断的固定匹配、eval 单组和固定变换限定其解释范围。训练覆盖需另用真实变换和原训练匹配验证。 [原生 criterion](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/criterion.py#L970) · [原生匹配器](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/matcher.py#L853)

## 等目标、等点数的方差检验

两个受控方案以 576 网格所有像素中心的均匀 BCE 为同一目标。预测用 `align_corners=False` 的零填充双线性点读，标签由同一 GT 网格给出。边界层 B 为 5×5 零填充膨胀与腐蚀的差，另一层为其补集。均匀组用 K 次有放回均匀抽样。分层组给 B 分配约其面积份额且至少 K 的 5%，其余点给补集，每非空层至少一点，空层退化为均匀。层均值以实际面积份额 $w_h=N_h/N$ 加权，两估计对固定均匀目标及其梯度都无偏。

每个位置梯度为 $g_i=b_i[\sigma(b_i^\top z)-y_i]$，其中 $b_i$ 是向粗网格反传的四邻域插值权重。枚举所有像素中心，用稀疏矩阵计算精确均值和二阶矩。均匀组全局方差迹是 $\operatorname{tr}(\operatorname{Cov}g)/K$，分层组为 $\sum_h w_h^2 \operatorname{tr}(\operatorname{Cov}_h g)/K_h$。边界梯度贡献的均匀参照以 $1_{i\in B}g_i$ 为随机变量。按实例累加两组方差再求比值，分别报告全体、小实例、其它实例。原生不确定采样只提供覆盖参照，其原 BCE 风险和此均匀目标有区别。

门槛预先锁定为小实例的边界贡献方差下降至少 20%，小实例与全体的全局方差各不增加超过 5%。通过只允许进入真实训练式匹配、变换与成本验证，且须核对原生零命中是否集中于本项目目标失败群。若真实训练式验证仍支持，训练比较固定初始化和匹配、两组共享原生 Dice 的规则及 K、BCE 额外 K。新增采样和插值开销需实测，另列原生共享 K 的强对照和等算力对照。Dice 是带平滑的非线性比值，点上分子分母无偏不足以保证比值无偏。

PointRend/Mask2Former 已用不确定点和均匀点，OccFormer 已用不放回的类别引导抽样及类别权重。当前诊断的差异是对 SAR 的真实匹配预测做可核验的同均匀 BCE 风险与方差分解。是否能带来真实训练效用和创新仍需证据。 [PointRend 固定源码](https://github.com/facebookresearch/detectron2/blob/247691a1c292654867deb35cabb27f18d75a1363/projects/PointRend/point_rend/point_features.py#L63) · [Mask2Former 固定源码](https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/mask2former/modeling/criterion.py#L140) · [OccFormer 固定源码](https://github.com/zhangyp15/OccFormer/blob/44aee79434d6aedca0d97fe62965032505f583e7/projects/mmdet3d_plugin/occformer/mask2former/base/mmdet_utils.py#L92)
