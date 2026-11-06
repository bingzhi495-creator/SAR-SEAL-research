# 完整外部基线：初始化与预算的下一步选择

2026-11-07。只读普通已发布说明/安全摘要，不读取运行SOURCE、权重或资格链，不SSH。Root新通知：32次真实健康更新已完成，IAMR在17步有非零梯度，首见第6步；因此不建议凭首图现象修改初始化。该通知不是本页独立复验，也没有AP。

**主E50的初值与成本需分层记账。** PLAIN/FULL的UNet主干和实例adapter按U34随机新建，seed202610173101；仅同操作104项state及共同adapter配对，不加载旧UNet权重，不能称整网相同初值。其输入却依赖冻结的项目Mask2Former/R50 E24教师：项目24轮17784次图像训练记录已核，与当前DEV187身份交集0；教师上游官方COCO initializer的完整文件身份已闭合。ImageNet-1K/COCO训练范围仍属发布方声明，全部上游数据祖谱、实际预训练GPU小时及teacher缓存生成总成本未独立闭合。故“UNet从头训练”不等于完整系统没有预训练。成本应分列官方预训练、项目教师24轮、特征/候选缓存、每臂50轮37050更新、额外失败/重复计算与部署时教师＋refiner推理；不能把24＋50写成同网连续74轮，也不能只报refiner耗时。

| 完整DiffSAR选择 | 可以说明什么 | 主要限制 |
|---|---|---|
| 保留公开默认18560更新与原LR计划 | 公开代码默认配方在当前FIT的实际表现 | 在741图、batch1下约25.05次图像覆盖，不是50轮；初值无checkpoint、论文实际预训练未知 |
| 新预锁50轮37050更新、按比例调整原LR里程碑 | 同FIT图像曝光预算下的DiffSAR同协议适配 | 不是作者原配方逐项复现；仍未匹配教师上游信息、输入尺度和计算成本 |

**建议以第二种作为下一完整外部基线，不再以短健康阶段替代。** 保持原模型数学、native optimizer/裁剪、base LR3.5e-5、wd1e-4、原mapper和batch1；从新正式状态0开始，预锁37050更新、衰减点29640/35198（原0.80/0.95进度），warmup20且factor仍1.0。明确称“公开构造初值下、固定FIT/50轮的同协议适配”。健康32步的顺序采样轨迹不拼入新正式shuffle轨迹。原18560方案可作另外预锁的作者默认参考，不能在看到AP后临时改变终点或把已低LR衰减的18560 checkpoint追加到37050冒充从头预锁的50轮计划。scheduler调整、种子、sampler/worker、标注、100候选评价及资源规则应在运行前锁定，当前主E50不改。

这一训练可先开展，同时继续核实论文实现段及合法作者初值/加载键。当前随机初始化结果可以透明报告为系统适配参考，不能称作者论文强基线已复现；也不能将其与“COCO教师＋项目E24＋refiner E50”的AP差唯一归因于新结构。后续正式强比较至少要有可核且可比的预训练条件（不要求不同架构载入同一权重文件），或明确分层报告不同初值；若需要严格从零端到端对照，应另立新实验，不能重写当前锁定主试验。统一划分/GT、评价人口和maxDets，端到端方法保留各自候选生成，固定候选精修另列；报告固定端点AP、AP75/AR、训练曝光和完整系统成本后再判断收敛预算，不能因短训低分宣称方法无效。

依据：[当前结构与共享初值范围](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/main/research_round_20260925_global/U57_CURRENT_PROGRESS_AND_METHOD_EXPLANATION_20261024.md)、[教师及预训练身份边界](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/main/research_round_20260925_global/U57_TEACHER_IDENTITY_AND_DATA_EXPOSURE_SCOPE_20261025.md)、[原E50设计](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/main/research_round_20260925_global/U57_DGX_ORIGINAL_UPDATE_FULL_FIT_TRAINING_PLAN_20261018.md)、[U38实际公共初始化](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/main/research_round_20260925_global/U38_FULL_FIT_RGB_AND_COMMON_INITIALIZATION_ACTUAL_RESULT_20261017.md)。这些普通说明支持上述范围，不等于本次重放原初值或认证全部训练谱系。
