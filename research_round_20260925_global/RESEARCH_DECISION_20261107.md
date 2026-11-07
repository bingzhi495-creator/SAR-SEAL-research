# 数学、物理、雷达处理与现代实例模型：研究决策

日期：2026-11-07。状态：原始证据审阅与有限真实CPU接口确认；没有新SAR精度结果，不是投稿成熟度认证。

## 已完成什么

本轮将工作拆为 SAR 实例分割直接先例、数学/雷达处理、基础模型与多模态三条独立检索，并追加一次针对采样想法的批判审查。已形成 12 个数学/雷达方法族、8 类基础模型、3 个船舶跨模态数据项目和近年直接实例分割工作的重点资料。这个范围不等于穷尽世界全部数学、物理、雷达或 SOTA。具体函数、固定源码版本、输入条件和缺失证据都在附表中；只看到仓库首页或占位实现的项目，不算取得论文实现。

Root 另实际取得并静态检查 RF-DETR 官方 1.12.0 固定提交 `639eac4555ceb89ef399ccda3fde716bbcfe34cb` 的完整源码归档，585 个文件、11,752,574 字节；归档 SHA256 为 `a0835160ea208bbf530217a8e0384b9527e336cde4a7398d5e7639b2112c3e91`。实际取得 Nano 实例分割权重 134,545,398 字节，MD5 与官方固定源码一致，SHA256 为 `a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b`。已下载 RF-DETR v2（24 页）和 OccFormer（11 页）原始 PDF，并读取相关方法与实验。第三方全文、完整代码归档和权重只保留为研究输入，本仓库发布原创综述、来源索引与本项目方案。

**基础依赖已实际离线安装，限定空schema补齐后的CPU严格加载及一次合成前向已真实通过。** [实际限定结果](RFDETR_SCHEMA_COMPAT_CPU_FORWARD_ACTUAL_RESULT_20261107_02.md)保留原始工具/SSH/worker0、原state等值和输出有限性范围；[v01原始strict失败](RFDETR_RAW_STRICT_PROBE_ACTUAL_RESULT_20261107_01.md)和新的元数据定位失败不升级为成功。完整train/CLI依赖、576真实SAR输入、梯度/优化器/BF16及评估仍待。

## 这次查新如何改变选择

| 初始可考虑的想法 | 直接证据 | 当前决定 |
| --- | --- | --- |
| CFAR 软注意力 + 现代检测器 | 2026-11-05 的 CaSA-DETR 已在 RT-DETRv2 上结合局部 CFAR 启发门控和散射上下文 | 不把普通 CFAR gate 当新贡献；图像经验对比仍可作对照，但不能宣称恒虚警保证 |
| 小船前景/边界点采样 | PointRend、Mask2Former、OccFormer 已覆盖不确定性、均匀混合与稀疏前景引导；边界点监督也有更早先例 | 降为有限预算诊断，不预定为论文核心创新 |
| SAM + 小波/局部分支/自动框提示 | SARSAM、SAMSAR、2026 检测器→SAM2 已有直接先例 | 仅换模型名或叠分支不够；需要实际边界/实例竞争错误及具体技术差异 |
| 直接拿多模态遥感模型作强泛化证据 | DOFA-CLIP 预训练字幕来源包含 SSDD/HRSID；AnySat/CROMA 等的原生通道和时相不同 | 重叠未排除前不作干净基线；不得伪造极化、对应光学或传感器元数据 |
| 单图物理去斑似然/原始成像一致性 | Gamma/Wishart、MERLIN、SAR2SAR、可微成像均有明确数据前提 | 未标定的 8-bit 显示图不支持这些物理声明；条件路线保留，现阶段不强塞主模型 |
| 最新方法仓库存在就能复现 | SDS-SAR 当前固定源码核心只是明确标注的 medianBlur 占位；若干 SAR 新作没有核得完整作者实现 | 记录缺口，区分透明重建与作者复现；不以占位代码报告论文方法执行成功 |

来源详见 [数学/雷达审阅](RADAR_MATH_PHYSICS_SIGNAL_PRIMARY_REVIEW_20261107_01.md)、[基础模型审阅](MULTIMODAL_FOUNDATION_DATA_PRIMARY_REVIEW_20261107_01.md)、[SAR 直接先例](LATEST_SAR_INSTANCE_SOTA_PRIMARY_REVIEW_20261107_01.md) 与 [采样独立审查](RFDETR_SAMPLING_IDEA_INDEPENDENT_REVIEW_20261107_01.md)。所有外部分数是作者报告，不与本项目不同协议的 AP 排名混用。

后续补齐了 [6项经典雷达链路](CLASSICAL_RADAR_SIGNAL_SOURCE_SUPPLEMENT_20261107_01.md) 与 [6项变分/扩散/低秩/图竞争工作](VARIATIONAL_DIFFUSION_INVERSE_SOURCE_SUPPLEMENT_20261107_01.md)。这些是重叠家族的深化补录，不能与前述12族相加当不重复总数。经典部分追到真实 Stolt、BP CUDA kernel、MVDR/MUSIC 与复数求解器；同时发现 TorchSAR 有占位函数。扩散部分找到2025作者真实实现及低于单次U-Net、生成错误结构的论文反例；2026 GLCNet虽有源，默认张量流和loss存在待验证静态风险，不能列已跑通。补录保留 STAP、若干低秩/稀疏作者实现和原始论文全文的未完成项。

后续另完成[6项形状/拓扑/OT/等变性原文与真实作者函数](SHAPE_TOPOLOGY_OT_EQUIVARIANCE_PRIMARY_SUPPLEMENT_20261107_01.md)。TopoLoRA-SAM已经组合基础模型、LoRA、空间适配器和clDice，任务为含SAR海陆的二值语义分割；不能冒充船舶实例mask AP，但足以约束普通组合的新颖性叙事。所有6项保留固定commit、直接函数、实现陷阱、简单对照和否定条件。

## 当前选定的可执行起点

先用 **RF-DETR-Seg Nano 官方架构与 COCO 初始化** 建立一个可复现的新强基线。它原生含 DINOv2 表征和实例掩膜头，代码与权重可直接核验；不需要先等待 SAM3/DINOv3 的 gated 权重，也不要求新配对标注。选 Nano 是先控制实际运行成本，不表示已证明它是 SAR 最强规模。首轮隔离兼容与成本测量后才能判断是否需要 Small/Medium；若换规模，所有关键对照随之重做并单独成表。

本项目拟用固定 576 输入的候选配方，而非将官方 COCO Nano 312 配置原封不动称为复现。主比较两臂必须同尺度、初始化、数据、有效 batch、更新数、增强、精度和评估规则。官方论文已有分辨率/patch/掩膜特征分辨率的系统研究，不能把加大图像或提高 mask 分辨率单独当作新结构。也不把论文上的 COCO 40.3 AP 外推成 SAR 水平。

**SAM3/3.1 与 DINOv3 为第二层候选。** 前者适合检验固定 ship 文本的端到端实例能力；后者可作特征/教师。现未取得对应 gated 权重。SAM3 默认演示阈值可能先截断召回，DINOv3 SAT493M 是光学预训练；二者必须按各自真实输入和推理规则处理。旧 SAMRefiner 的负结果不否定这些不同模型。

## 先诊断，再决定哪一种改动

只检验两个问题，初期只晋级一条，不同时叠加采样、频域、蒸馏和残差。

1. **监督分配假设（未确认）**：已匹配的细小/细长实例是否在多个层、组和训练步累计后仍缺少有效前景/边界监督，并导致可测的梯度估计误差？记录 loss 与 matcher 的采样，不能只展示一次零命中。若原生已足覆盖或 dense BCE 梯度一致，停止把采样当瓶颈。独立审查认为目前混合采样 + HH 的已知组合不足以支撑 TGRS 核心，只允许限额诊断。
2. **表示分辨率假设（未确认）**：在框正确且实例被检出的条件下，原生低分辨率掩膜是否系统性损失细船边界/邻船间隙？若是，才研究预测条件下的选择性局部残差。先比较更高原生分辨率、同容量普通残差和全实例细化；若简单方案解释全部收益，承认新机制未成立。若主要是漏检/错误匹配，则转向查询/特征问题。

当前真实已有证据中，整数/浮点 GT 栅格化变化能影响部分方法排序，质量评分也能改变 AP；这要求新增模型把几何与排序、固定主 GT 与敏感性分析分开，不能继续仅凭一个 AP 归因边界改善。该历史现象也不证明新 RF-DETR 有同一问题，更不能把另一 GT 当物理真值。

## 数学设计必须守住的边界

RF-DETR 1.12.0 原生 mask loss 的点数是 `max(Hmask, Hmask*Wmask//16)`，不是固定 12,544。原生从 3K 候选选择 75% 不确定点并补 25% 均匀点，BCE/Dice 共享 K 点；matcher 另用共享均匀点。实际 GT 标签采样 nearest，预测 logit 为 bilinear。576 方形、stride4 的静态推算是 144² 掩膜、K=1296；这只是代码条件推算，实际增强、padding、匹配组/层与成本必须记录。

若独立有放回抽离散像素 i~q，`mean(loss_i/(N*q_i))` 可在确切密度、正支持、停止密度梯度等条件下估计明确的均匀 BCE 风险；它是 Hansen–Hurwitz 风格，不是把无放回初始 q 当 HT 的纳入概率。该目标与原生 uncertainty 风险不同，不能说保持原生目标。对 Dice 的比值也没有自动无偏保证。

若新 BCE 用 K 点、Dice 另保留 K 原生点，实际最多两套点，不能写相同总点预算。最低对照为原生、均匀 BCE、未校正混合 BCE、校正混合 BCE，并增加实际成本控制。若校正混合不优于均匀 BCE，不支持降方差机制；若未校正最好，应按风险重加权解释。更高命中率本身不是因果证据。

## 证据出口与执行顺序

详细 [基线与诊断计划](RFDETR_BASELINE_AND_DIAGNOSTIC_PLAN_20261107.md) 只含五个核心工作块。当前已锁定 PLAIN/FULL 长训保持配方继续，先完成其真实固定端点与既定评价。新模型的源码、字节和独立环境准备可并行；新科学执行须先取得对应资源与来源资格，不抢占正在运行任务，不在活动源码中改配方。

新主线先跑通官方基线，保留完整 100 轮候选预算和 24/50/100 检查点；方法与对照使用同样预算，不以短训负差判方法总体无效。候选只有通过实际误差诊断和公平效用检查，才扩大到三种子及 SSDD/HRSID。原来用过的开发集仍称开发集，不能改名为未接触测试。

本轮形成的是**更扎实的来源库、已取得的可执行输入和有否证条件的新基线计划**。尚无新 RF-DETR 指标，没有新 SOTA 结论，没有已成立的核心创新或 TGRS 成功判断。正式跨模型审查后端当前不可用，独立代理审阅使用继承模型；查新结论状态保留为 `[pending Codex review]`，不把同模型独立审阅说成跨模型认证。

## 原始入口

- [RF-DETR 原论文 v2](https://arxiv.org/abs/2511.09554v2)、[官方固定代码](https://github.com/roboflow/rf-detr/tree/639eac4555ceb89ef399ccda3fde716bbcfe34cb)、[1.12.0 发布](https://github.com/roboflow/rf-detr/releases/tag/1.12.0)、[官方模型许可](https://roboflow.com/model-licenses/rf-detr)。
- [实际 mask loss](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/criterion.py)、[mask head/sampler](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/heads/segmentation.py)、[官方权重校验项](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/assets/model_weights.py)。
- [OccFormer 原论文](https://arxiv.org/abs/2304.05316)、[固定采样实现](https://github.com/zhangyp15/OccFormer/blob/44aee79434d6aedca0d97fe62965032505f583e7/projects/mmdet3d_plugin/occformer/mask2former/base/mmdet_utils.py)。
- [CaSA-DETR 出版方](https://www.mdpi.com/2072-4292/18/21/3799)、[DOFA-CLIP 原文与数据附录](https://arxiv.org/html/2503.06312v2)。


P0新核查：[官方单类迁移默认截取COCO行0/1而非boat9](RFDETR_SINGLE_CLASS_TRANSFER_INITIALIZATION_REVIEW_20261107_01.md)。当前YAML保留为官方默认候选，语义对齐行9/1仅列同预算对照，不预判其更好。后续模块与强参考必须共享选定初始化。另已准备[严格单类逆映射和原生top100后过滤的canonical导出](code/RFDETR_CANONICAL_EXPORT_DRAFT_20261107_01/README.md)，7项标准库合成测试通过，实际PostProcess/codec/原GT仍待。完整依赖的候选及解析缺口见[train/CLI准备报告](RFDETR_FULL_TRAIN_CLI_DEPENDENCY_READINESS_20261107_01.md)。
