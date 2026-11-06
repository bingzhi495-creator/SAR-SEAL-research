# DiffSAR 初始化：梯度证据、条件秩结构与预训练缺口

2026-11-07后续实际结果：[原生32步FIT健康训练](U57_DIFFSAR_NATIVE_HEALTH32_ACTUAL_20261107.md)已经完成，第6步首次出现非零IAMR梯度，32步17非零/15全零。首个样例的零信号不持续覆盖这段轨迹，不再以它阻止训练或据此修改初始化；不同输入和参数更新的作用未分离。原01/02/03失败及下文的静态结构推导保留，仍无完整AP/收敛/论文初始化来源认证。下文“尚未执行健康阶段”为形成时的历史状态。

2026-11-07。作者固定提交 `bdc3362372213bc8c3a1f6d5b7b74dc224d1b871`。两位 agent 独立只读源码/数学核查，另查作者官方仓库和论文原始入口；未 SSH、导入科学库、运行模型、读取图像/GT 或下载权重。以下对照仅为设计，未锁定或执行。

## 三次单批失败不能升级

01 在最终 scope 断言前未保存 summary，实际失败 scope 仍未知；02 同 case/seed 定位 IAMR 的14个参数均有有限但全零梯度；03 在共享 IAMR 输出 M 处取得有限全零总梯度，形状 `[1,8,208,256]`、max_abs=0。三次均维持失败，02/03不是独立种子、训练过程或新方法证据。

03 支持“本次总 mask 信号在 IAMR 输出边界已为零”，不能独自区分 sigmoid 数值饱和、动态两层 ReLU、Dice、跨实例/跨头梯度抵消等解释。head4 的 mask loss 约4.968也不排除全1饱和预测。不得将这项定位改写成 sigmoid 已被证实是根因。

## 条件结构：不是实际数值秩测量

源码 `_reset_parameters:193–202` 按参数**最后一维**是否等于单类数1或加一2设 focal prior，因而 IAMR 最后 `(8,128,1,1)` 权重的全部元素被填成同一 c；prior=.01 时 c≈−4.59512。最终 bias 有8个元素，不受该条件覆盖。设最后卷积之前的 ReLU 特征为 A∈R^(128×P)，sᵀ=1₁₂₈ᵀA，则在该 W 尚未被加载或优化更新改写的理想实数计算下：

`M_j(p)=c·Σ_i A_i(p)+b_j`，即 `M=c·1₈sᵀ+b1_Pᵀ`。

因此 rank(M)≤2；各通道减去空间均值后秩≤1；所有像素的8维特征位于同一仿射直线。又因 A 非负、c<0，有 M_j(p)≤b_j。此结构压缩了初始化时可独立变化的通道方向，是表征退化候选，但仍保留空间变化，不等于常数掩膜，也不推出输出梯度必为零。

FP32 的卷积累加、bias相加和中心化含舍入，不能据代数推导宣称实际存储输出严格秩2。此结论只属于共享 mask-feature 边界及特定初始化，不能扩展为整网秩、任意输入下的训练失败或训练后永久性质。[原 IAMR 结构](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/head.py#L143)、[初始化规则](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/head.py#L184)、[动态 mask 与损失](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/loss.py#L335)。

## 预训练：公开默认值与论文实际配方分开

| 来源 | 本轮可核事实 | 不能推出 |
|---|---|---|
| 固定作者 README | 给出训练命令、测试时指定本地模型路径；未要求COCO/ImageNet预训练，也未给官方checkpoint下载链接。 | 论文实际实验必然从零训练。 |
| 两份 train YAML | `MODEL.WEIGHTS` 均为空。`build_model` 明确只构造网络，不加载配置权重。 | 作者没有经命令行、resume或其他实际环境指定初值。 |
| 作者训练入口 | Trainer构造网络/checkpointer后，main再调用resume_or_load，故初始化和后续权重加载是不同阶段。 | 任意预训练checkpoint都能覆盖IAMR。 |
| 2026-11-07官方公开仓库检索 | releases和issues均返回空列表；固定归档无pth/pt/pkl/ckpt/safetensors；Predict.py只指向作者本地推理checkpoint。 | 私下或其他渠道不存在作者权重；本地推理模型说明了训练初值。 |
| 原论文全文 | DOI/出版社题名已核实，全文请求仍403，未取得能核验初始化的implementation段。 | 论文要求或不要求COCO/ImageNet。该项保持**未知**。 |

不使用搜索结果的相邻论文、二手自动评议或常见CV习惯填补论文初始化缺口。公开代码没有自动预训练指令，与论文实际实验初值尚未闭合，是两个不同结论。后续需合法论文全文或可核的作者说明/权重来源；当前不能称正式论文配方已复现。

若只加载 backbone，A的分布和后续饱和风险可能改变，但上述 W 的代数限制仍在；若完整checkpoint实际覆盖了 `head.iamr.4.weight`，原构造 W 的条件可能不再成立。必须查加载键/形状、missing与unexpected项，以及加载后该 W 的事实，不能只记录“使用COCO”。即使覆盖新 W，也不能未经数值核查反推实际样本必然升秩或梯度恢复。

初始化来源：[作者 README](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/README.md)、[SSDD train](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-SSDD-train.yaml#L321)、[HRSID train](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-HRSID-train.yaml#L321)、[build_model](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/meta_arch/build.py#L16)、[真实加载入口](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/train_net.py#L269)、[加载接线](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/checkpoint/detection_checkpoint.py#L116)、[作者 releases](https://github.com/XiaowoXuSAR/DiffSARShipInst/releases)、[论文原始入口](https://doi.org/10.1016/j.isprsjprs.2025.02.030)。

## 下一阶段的最少公平设计，尚未执行

优先核实论文初始化及其键覆盖；若仍不明确，显式命名为“公开代码构造初值下的适配”，而非作者正式配方。对已经计划开展的作者基线训练，另锁一个有限健康阶段，例如前32个 optimizer updates，在0/1/8/32记录原损失、IAMR及controller的grad None/有限/非零、G_M及更新情况；不为选最佳样本而换图。保持真实FIT顺序、原mapper/R101/SCJE/500 proposals/6 heads、原优化器及学习率。仅用于判断首图现象会否随其他样本和真实更新消失，不作AP或总体可训练性结论；AdamW权重衰减引起的参数移动也不能冒充数据梯度恢复。

若确有必要检验初始化机制，再独立锁定一次等预算、配对数据/增强/seed的A/B：A保留原初始化；B仅对该最后 W 保留原循环已生成的Xavier结果、不做后续常数覆盖，其他参数与RNG消耗对齐。只改变一个张量的初值，先检验梯度健康，不重命名为新SAR模块。若实际采用的checkpoint已覆盖W，需先撤回这个构造初值对正式基线的解释，不能继续拿无效因素做对照。

单种子的短健康对照只能支持该预算与初值下的机制诊断。即使B改善，也只是初始化适配候选；作者强对照仍需合法可核初值、充分训练、重复种子、同数据/评价及成本证据。它不能单独成为SAR核心创新，更不能据此宣称胜过作者论文或达到TGRS。
