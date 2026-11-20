# 固定查询条件通道特征

本原型只实现纯 NumPy 特征变换和合成正确性检查。没有读取真实载荷、运行网络、捕获特征或训练。实际有效性尚未检验。

`point_feature_controls(old14, spatial, query, coordinates, boundary_layers, *, candidate_id)` 接收单候选的既定点集。`old14` 为 float32 `[K,2,14]`，`spatial` 为 float32/64 `[K,2,256]`，`query` 为 float32/64 `[256]`。`K` 为 0 至 32，第二轴固定为添加、删除。`coordinates` 是唯一非负原图扁平像素索引 `[K,2]`，`boundary_layers` 是同形非负整数层号。`candidate_id` 为原始 `(image_id, native_rank, native_query_id)` 整数元组，rank 从 1 开始。调用者负责提供已封存的点序、符号及边界层，本函数不重算点集。

令 `C(x)=x−mean_channel(x)`。局部 PCG64(5241) 生成 int8 的 256×16 Rademacher 矩阵 R。先生成取值 0/1 的 int8 数组，再作 `2*bits−1`。R 的 C-order 原始字节 SHA256 固定为 `96db66d62d88541d31254f408a18330c0b03f423364b64d2f50cdf19ce466448`。中心化、元素乘积和投影使用 float64，结果转为有限 float32，不裁剪溢出或消费全局随机状态。

四项控制各产生 `[K,2,16]`，并与原14维拼接成同形 `[K,2,30]`，供小、大两个头使用相同输入契约。

- `zero` 为全零。
- `ordinary_centered` 为 `C(F)R/16`。
- `query_conditioned_centered` 为 `C(Q⊙F)R/16`。
- `query_conditioned_permuted` 仅置换上一项的16维向量。候选、添加或删除符号、边界层共同确定组；SHA256 对不可变候选身份及像素键排序后建立单环，单点组保持不变。这不是均匀随机置换。原14维、坐标和接收点次序不动。

返回 `features`、`residuals`、原坐标与层号副本、`permutation_source_flat_indices` 及算法元数据。输入不会被修改。拒绝重复点、不完整点对、错误形状、非有限值、非整数身份及超出 float32 有限范围的投影。

相同标量 `Q·F` 不决定中心化通道向量，因此合成例可在标量相等时产生不同投影。这只是表示能力的代数见证。F/Q 来自已学习的冻结网络时仍含学习所得信息；本算子不引入新的学习参数、拟合统计或标签。Q 为全一时条件项等于普通项，Q 为零时条件项为零。F 和 Q 都是通道常量时中心化结果为零；仅 F 为通道常量而 Q 非均匀时条件项可以非零，不能将其解释为纯空间变化信息。

四项小合成测试实际通过，用时 0.046 秒，未导入 Torch。覆盖固定矩阵身份、相同标量不同残差、float64 中心化、输入及全局随机状态不变、组内置换与单点组、点对重排后的像素身份一致、常量与空点集，以及非法输入拒绝。测试不提供 SAR 质量、AP、真实特征可用性或训练收益证据。
