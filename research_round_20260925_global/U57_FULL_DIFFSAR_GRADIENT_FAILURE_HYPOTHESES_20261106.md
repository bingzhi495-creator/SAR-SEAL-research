# 完整 DiffSAR 单批梯度覆盖失败：待检验解释

2026-11-06；作者固定提交 `bdc3362372213bc8c3a1f6d5b7b74dc224d1b871`。本稿由两位 agent 独立只读源码核查整合；未新增实验、科学导入、读取图像/GT 或修改算法。

实际完整单批已取得 24 项有限损失，并完成一次 backward 与同步；随后九个 scope 的最终非零断言失败，原记录保持 `qualified=false`。失败 scope 尚未知：原 worker 在断言后才保存 summary，不能先认定 IAMR 出错。五项 mask 损失为 5.0，另一项约 4.967989，均不能单独证明梯度通畅或中断。

九个范围是 `backbone.bottom_up.`、`backbone.`、`head.iamr.` 和 `head.head_series.0.` 至 `.5.`。统计按每个 prefix 独立累加，不存在首匹配漏计；但宽 `backbone.` 包含 bottom_up，不能当作独立 FPN 梯度证据。

- **初始化量级是具体竞争解释。** 作者 `head.py:193–202` 对全部 head 参数按最后一维等于类别数或类别数加一施加 focal prior。单类/.01 prior 配置下，IAMR 最后卷积 `(8,128,1,1)` 的整个权重也被填为约 −4.59512，而非只初始化分类 bias。它可能放大负 mask features，再造成动态 ReLU 闭合或 sigmoid 饱和；这是原源码可证行为，是否造成此次失败仍待观察，不能预先修改。
- **IAMR 与实例头的监督路径应区分。** IAMR 只经 mask loss；classification/box 损失有限不证明 IAMR 有非零梯度。动态 mask 路径有两层 ReLU，可能令 feature 梯度为零而控制器其他参数仍有梯度。六个实例头各有自身辅助监督；`head.py:281` 只 detach 传向下一阶段的框坐标，不能单独解释本阶段全部监督失效。
- **损失值不能替代导数。** 作者 Dice 为 `5×(1−2Σpt/(Σp²+Σt²+ε))`。显示 5.0 可能只是微小重叠后的舍入；损失低于 5 也可能对应 sigmoid 已全饱和为 1、其导数为零。共同目标下采样全空可导致恒定 Dice，但同一 GT/同一抽样处理下有一头低于 5，不能用“所有阶段目标全空”单独解释。作者无匹配分支返回 0，此次六项正损失也不支持整次均走该零分支。
- **固定、未用和命名是另组解释。** 作者目标 FREEZE_AT=0，未发现冻结全部 IAMR/六头的代码；确有未调用的附加 attention、SE、部分 mask_refine 参数，其 grad=None 本身不异常。但现有断言只要求每 scope 至少一项非零，少量未用参数不足以解释整组失败；实际名称、requires_grad 和 None/零必须分别核验。

最少诊断先用独立 v02 中新增的保存顺序：在原断言前落盘已有逐 scope 统计及 missing、present-but-zero、failed 列表，保留原门槛。**v02 目前未绑定、未执行，尚无这些新观测，不能自动复跑。** 如后续经 Root 监督取得结果，先定位具体组；只有指向 mask 路径时，再考虑 logits 范围、sigmoid 精确 0/1 比例、动态 ReLU 活动率及处理后目标面积等少量聚合信息。不得据本次单批诊断宣称方法总体失效、训练成功或达到发表标准。

精确来源：[初始化规则](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/head.py#L193)、[阶段输出与 detach](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/head.py#L272)、[Dice 定义](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/loss.py#L82)、[动态 mask 与损失](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/loss.py#L335)。
