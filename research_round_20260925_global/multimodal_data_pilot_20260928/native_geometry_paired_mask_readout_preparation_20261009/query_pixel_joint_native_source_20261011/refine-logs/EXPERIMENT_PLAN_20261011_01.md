# 整网长训实验计划

**问题**：冻结主干的单种子预测头收益，尚不能证明新结构优于充分训练的网络或强质量评分头。
**方法假设**：在相同原网络初始化、数据、像素监督和追加预算下，原图网格上的query/像素条件残差可以提供评分以外的实例掩膜收益。此假设尚未证实；接通梯度本身不是创新。
**日期**：2026-10-11。本文档是SOURCE准备，不是运行授权、实际SPEC或科学结果。

## 主张与反证

| 候选主张 | 最低需要的证据 | 当前缺口 |
| --- | --- | --- |
| C1：query/像素条件修正具有额外几何效用 | 同原生E24初始化、同监督和追加72轮配方的原生点BCE对照；固定评分几何评价；重复配对差异及失败例 | 单冻结teacher的局部结果不能代替此证据 |
| C2：方法提供有效的精度与代价折中 | 强特征条件质量头、强SAR方法统一协议重训、HRSID；同步测时、参数、显存和退化比例 | 当前9441 scalar对照不匹配像素信息；没有当前方法HRSID或强SAR长训结果 |

必须排除：增益仅来自额外监督、额外特征/参数、重新评分或选择有利随机种子。当前不声明SOTA、概率校准、收敛或TGRS-ready。

## 必须完成的五个实验块

### B1：固定端点与预设重复

PSeg-SSDD FIT741/DEV187；DEV已被历史研究暴露，不能称盲测。现有seed0的P/B/scalar各自24→自身权重fresh AdamW48已取得CPU端点资格。保留全部arm和推理对照，追加seed1/seed2的相同两阶段配方；不得根据seed0只选择uncapped P或改变采样/学习率。三seed改变readout初始化、顺序和点采样；teacher E24仍相同，不是独立预训练backbone种子。逐seed列结果和配对差异，不能用三seed宣称可靠显著性或总体收敛。

在seed1 first24 CPU资格后的自然SCI空档，优先评价已完成native E96的固定最终DEV端点。它可检验相同总图像访问预算下的实用差异，不能称参数、目标、计算或轨迹完全匹配。无中间节点选择，无TEST使用。

### B2：整网同监督、同预算的决定性对照

A：原生30项损失 + λ×原生L的原图点BCE。
B：原生30项损失 + λ×query/像素残差L+u的原图点BCE。

λ事前固定为1，不以DEV选择；它是待检验配方，不是已执行设置。两者从同一qualified native E24初始化，追加24→各自权重fresh AdamW48，共72轮；记录优化器重置，不能叫连续72轮。保留原生optimizer参数组与backbone倍率，新增8个readout张量使用1e-4/.05；λ和阶段学习率不得因端点结果改变。原生学习率/无衰减规则应从实际已资格的make_optimizer SOURCE接入，入口资格时明确完整组清单，不能凭本计划猜测。

共同规则：microbatch1/global accumulation2；741为奇数，不在epoch边界清空pending梯度；每arm首阶段17784 micro/8892 updates，第二阶段35568 micro/17784 updates。每个micro独立完成30项原生loss及附加loss后一次backward，动态使用本arm当前最终stage Hungarian，按raw GT annotation ID对齐。不可复用冻结teacher的首次matching缓存。两arm参数变化后assignment可以不同；不能强制跨arm保持旧assignment。

原生345个非BN参数训练，106个BN affine和159个buffers继续冻结；readout B另加8张量。原生训练checkpoint路径保留。共同附加点为每GT12544个原图整数索引，保留重复；采样发生在caller，两个loss模块不生成点或GT。主分析采用L+u一致的推理；同时保存事前固定的native/corrected geometry与official/budget评分对照。只将uncapped作为seed0之后预设的新整网方案，不伪称最初从未看过DEV。

成功：相对于A，B的配对mask AP和固定评分几何收益跨重复方向稳定，并公开退化实例和漏分代价。失败：若B不优于A，额外点监督可解释收益，不支持修正结构不可替代。即使B胜出，也仍需强特征/容量对照和直接先例的技术区别；不能自动计算一个创新点。

### B3：强评分与强SAR对照

现有feature-pooled9477质量头是优先信息更强的评分控制；与9441相比多36参数且计算不同，应明示。先CPU小型数值检查和native checkpoint ON完整backward/update资源接入；再同readout24+fresh48长训。旧scalar不足以支持pixel posterior优于强质量预测器。

外部方法优先DiffSAR官方实现。显式绑定公开数据FIT/DEV、可核查初始化、实际更新次数和统一COCO evaluator，保留官方proposal数量/NMS与架构。官方init/split未闭合时称“统一协议重训”，不得称忠实复现或直接比较论文数字。没有可运行Linux/extension环境时先做SOURCE兼容接入，不因未知历史轨迹而无限延期。暂不同时移植O2Former或多套foundation模型。

### B4：HRSID适用性

为正式HRSID划分记录图像/annotation ID和场景/重叠风险，在公开训练部分固定inner FIT/DEV；公开test只用于冻结后的报告，不能用来选模块。计划要按实际数据自动计算长度、GT和updates，不沿用741/187硬编码；不得截断>100 GT或跳过emptyGT。候选与上述强对照在同协议训练，报告逐seed结果和计算。旧HRSID试验不代替当前新方法证据。

### B5：机制及失败分析

在现有权重上固定native query ID比较native/Pbounded/Puncapped，而不是每端各挑best query；统计cap不可翻转区间、GT边界/非边界的FP/FN及每实例改善/退化。现有best100 oracle会切换query，不能据此因果归因。评分移植不加性，不能把AP差除成贡献比例。记录小目标与面积层级，但稀少大目标不得外推。

最新[UnfoldCRF预印本](https://arxiv.org/abs/2609.33996)采用匹配输入、容量、阶段与监督的控制，并检验改变mask generator后的收益和退化比例；这是参考的证据标准，不是已经复现的SAR结果。已知SDC/SegQC公式和共享像素概率不列为新原语。

## 执行顺序

| 顺序 | 操作 | 实际状态/门槛 |
| --- | --- | --- |
| 1 | seed1三arm first24 GPU→CPU资格 | GPU已实际启动57914/PID64132；原始handle仍LIVE |
| 2 | native E96固定DEV GPU→CPU评价 | SOURCE ready；实际SPEC/admission尚未创建 |
| 3 | seed1 own24→fresh48 GPU→CPU→预设cross | SOURCE ready；只能使用seed1自己的qualified权重 |
| 4 | seed2自己的first24→fresh48及cross | SOURCE ready，未实际启动 |
| 5 | joint synthetic CPU与原生整网一次更新接入 | 新SOURCE静态PASS；尚无数值/native资格 |
| 6 | A/B整网长预算及预设重复 | 未实现完整入口，未启动 |
| 7 | 强质量控制、DiffSAR及HRSID | 仍是必要证据，不能用前五项替代 |

每个SCI任务沿用当前单worker、隐藏进程和已设资源限制；必须观察原始工具的实际终止再安排依赖任务，不根据超时重启。当前worker LIVE时不运行新fixture、读取科学载荷、改变库或AUTH。原生joint原图梯度图的资源资格不由旧冻结readout资格继承。

## 计算预算与删减

单次warm48已观测GPU端约77分钟，包含导出和DEV评价，不是纯训练时间。整网A/B及HRSID的GPU小时应按实际原生完整更新与稳定吞吐估计；目前不编造工期。CPU资格耗时计入总排期，不能只报GPU小时。

必须优先B1/B2/B3/B4。暂不增加SAM3、多模态伪数据、更多backbone、短预算超参扫参、论文交付包或新的无对照模块。非frontier方案无需添加LLM/VLM/diffusion/RL组件。

## 当前完成审计

- [x] 公平问题与反证明确。
- [x] 两个主张和五个实验块固定。
- [x] 新live adapter及native监督控制保存且SOURCE静态检查通过。
- [ ] joint数值与native实际反向传播资格。
- [ ] 全部三seed长训和整网决定性对照。
- [ ] 当前方法强对照/HRSID证据。
- [ ] 相对于直接先例的核心创新获证。
