# N11 后的区域选择方向：一个待检验假设

日期：2026-10-13。状态：普通设计备忘录；N11 下列汇总由 ROOT 完成 GPU/CPU 实际终止与独立资格闭合后提供。本文未读取 SCI、GT、缓存、权重或绑定 SOURCE，未开展新数值、源码或训练；下一方案尚未登记执行。仅作针对性 primary 文献核查，不声称检索穷尽或证明首次。

## 已知证据与结论边界

优先检验编辑区域，暂不把带内 IoU/Dice 损失作为主要突破。canonical 原生 .50 匹配中，374 个固定 query–GT 对的现有编辑带 B={x:|L(x)|≤2} 只覆盖既有 FP/FN 的 0.22530751073905203。即使允许带内每个像素自由取 GT，仍有明确条件上限；换损失不能突破同一个固定支持的自由 oracle。

| 固定原生匹配人口 | 已核验汇总（ROOT 提供） |
| --- | --- |
| .50：374 对，另保留 6 个原生 FN | FP_in=8790，FN_in=6578；FP_out=25716，FN_out=27125 |
| 同一 374 对的 mean IoU | native 0.8414231314153919；free-B oracle 0.8680848509281471 |
| 同一 374 对的条件 oracle 门槛 | ≥.90：147 对；≥.95：18 对 |
| .75：348 对 / .90：107 对 | 编辑带内既有错误比例分别 0.25685897698 / 0.29961653174 |

以上是固定已匹配对的像素及条件上限，既不是全 380 个 GT 的上限，也不是 AP；6 个原生 FN 没有被 oracle 变成新检测。逐对 GT 自由赋值也不保证共享网络或多实例输出能同时达到这些上限。不同匹配阈值可能改变 query，不能用 147−107 宣称多恢复 40 艘船。CPU 用保存的真实 GPU B 计算这些指标；其 near-zero/near-two 实用重放检查没有以 CPU 近似带替换 GPU B。

**尚未测量带外错误到原生预测几何边界的距离。** 因此，不能从现有计数推断这些错误主要是边界偏移、高自信幻觉、岸线干扰、船体结构缺失或 SAR 物理效应。也不能先验保证扩大几何带就会有效：新增支持可能多覆盖正确背景，或者遗漏离预测轮廓很远的 FN。

## 最近直接先例压缩核查

| 直接先例 | 实际选择/继承机制；对本项目定位的约束 |
| --- | --- |
| [RefineMask，CVPR 2021](https://arxiv.org/html/2104.08569v1)，[作者 loss](https://github.com/zhanggang001/RefineMask/blob/main/mmdet/models/losses/cross_entropy_loss.py)，[作者推理](https://github.com/zhanggang001/RefineMask/blob/main/mmdet/models/roi_heads/refine_roi_head.py) | BAR 已有区域 mean BCE、预测/GT 边界联合训练和边界外继承。论文/作者代码训练 width=2、推理 width=1，位于各级固定 RoI 网格；作者 block operator 是方形邻域响应阈值近似，并非精确欧氏距离。原图 width=2 对照必须称“RefineMask 式几何带适配”，不能称官方完整复现。 |
| [PointRend，CVPR 2020](https://arxiv.org/abs/1912.08193)，[作者实现](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend)；[Mask Transfiner，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.pdf)，[作者代码](https://github.com/SysCV/transfiner) | 不确定性预算点选择、学习不一致区域和稀疏细化均有直接先例。固定点数、学习区域、区域外保持旧结果不能单独作为新原语。 |
| [SparseRefine，ECCV 2024](https://arxiv.org/html/2407.19014v1)，[作者代码](https://github.com/mit-han-lab/sparserefine) | 冻结粗模型，熵阈值选稀疏像素，高分辨率稀疏特征和 gated ensembler；还比较 learned selector 的召回与延迟。其选择为阈值密度，并非本文实例 query 的硬 K 配额，但“冻结载体+不确定区域+稀疏残差/融合”已有先例。 |
| [SAM-REF，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_SAM-REF_Introducing_Image-Prompt_Synergy_during_Interaction_for_Detail_Enhancement_in_CVPR_2025_paper.html)，[最终 arXiv 正文](https://arxiv.org/html/2408.11535v3) | 有 GT 错误监督的 soft error gate、交互相关局部窗口和动态贴回。不是严格固定 K 的自动实例细化，但 query/prompt 条件、误差预测、局部修改和冻结阶段的组合不能笼统称首次。此次未确认作者可用代码，不以第三方复现作证。 |
| [SAMRefiner，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/90800f46f84381b7891e1378ee850013-Abstract-Conference.html)，[正文](https://arxiv.org/html/2502.06756v2)，[作者区域/提示代码](https://github.com/linyq2117/SAMRefiner/blob/main/utils.py) | 用粗 mask 池化的实例 embedding 与周围特征相似性辅助弹性框，另有距离引导点；基础流程为提示驱动完整 mask 修正。实例表示辅助区域选择已有先例，不是保持带外恒等的硬 K 稀疏残差；弹性扩框属于可选配置，不能把默认关闭分支当作每次实际推理。 |
| [UnfoldCRF，2026-09-27 预印本](https://arxiv.org/abs/2609.33996)，[v1 正文](https://arxiv.org/html/2609.33996v1) | 图像与初始 mask 条件的 latent-region CRF、共享能量展开；默认五阶段、stride-4 全格，冻结 DINOv2 为一个变体。当前为二值 mask 任务，非 query-conditioned 固定 K repair；作者称代码将发布。本核查未独立复现实验，其非边界发现不得迁移为本项目机制。 |

几何膨胀/腐蚀、logit 带与几何带的并集、error map、top-K、query 条件各自已有丰富先例。当前没有足够依据把它们的组合命名为新的核心创新。相对于上述直接方法，公共 SAR 实例 query、原图固定编辑预算及固定 S0 下的整合效用可以成为候选贡献，但必须先建立具体差异与可复现实验，不能由关键词没有命中证明新颖。

## ONE 下一可否证假设：同预算下区域选择比固定 logit 带更有价值

**H_region（未执行）：在相同冻结载体、同一个已知点头、相同每实例编辑预算与训练配方下，依据原生二值 mask 的几何边界选择编辑支持，比当前 logit 不确定性支持覆盖更多可修正错误，并在完整训练后带来可重复的高 IoU 收益。** 这是选择研究方向的假设，测试已知强几何对照，不是新方法或发表级成功声明。

记 P={x:L(x)>0}。先明确 B_geo,2(P) 的算子与坐标：优先用 RefineMask 作者 block-boundary operator 的 width=2 在原图 P 上作强适配对照，另注明与原论文 RoI 各级 width 的尺度差异；不得把这一定义写成原图欧氏半径 2 的官方复现。若另用精确 distance-to-contour≤2，须另命名，不能悄悄互换。推理只用 P/L，无 DEV/TEST GT。

公平机制检验需区分两个报告：原始 B_geo,2 的覆盖/代价曲线，用于判断 radius=2 强基线是否实用；**主对照在同一前瞻锁定 K_i 下比较几何排序与 logit 排序**，防止几何带只是多编辑了像素。K_i 由原生输出及锁定预算规则确定，不由 GT 或 DEV 最佳结果决定。几何排序优先 width=2 的边界区，截断/不足补齐、确定性 ties、空 mask、K_i=0 等规则必须锁定；补齐超出 width=2 时准确称预算匹配的几何排序变体。保留不确定性 top-K 的 PointRend 式适配基线，不能只对一个较弱固定阈值证明有效。

- 所有臂使用同一教师检查点、划分、实例/图像顺序和冻结 F/e/L 来源；点头结构、活跃参数、冷初始化、optimizer、损失及输出 S0/class/order 完全一致。第一轮不同时换区域、骨架和损失，否则无法解释收益。
- FIT 标注访问、每次更新的监督点数、区域内/全图采样比例、更新次数和每实例编辑 K 匹配。区域改变后须重新生成匹配的 FIT 采样缓存；旧 logit 带 cache 不能冒充几何带训练。RefineMask 原 BAR 的 GT/预测边界联合训练是强补充适配对照；若采用此采样规则，须单独报告这一差异，不能把单臂额外 GT 导向采样的收益归因于推理区域。
- 双方先登记同一完整长训预算与至少三个配对种子，固定最终轮、禁止 DEV 选 best。轮数本身不证明充分训练：终点未稳定时双方统一延长，并保留原固定预算结论。不得用旧 C/PR 的 8 轮结果代替新区域的公平训练。缓存构建、几何选择、feature sampling、训练和推理实耗均计入；同 K 不自动等于同计算，必要时再给相同延迟预算曲线。
- 先在同一固定 query–GT 人口上报告 coverage、oracle、新增正确像素被破坏与实际 IoU；正式结论用全部预测的固定评价协议和 COCO 全指标，保留 FN，不以 oracle 或已匹配样本的局部收益替代 AP。最终还须公共独立数据集证据；不以此阶段直接宣称胜过完整 RefineMask/PointRend 或外部 SOTA。

**反证规则：** 若几何支持在同 K 下没有更好的错误覆盖/条件 oracle，区域方向的这一具体假设被否定；若仅 oracle 提升而完整训练没有可重复实际收益，则只证明存在理论可编辑空间，不能宣称 head 已能利用它。若收益仅随 K/耗时增加或原 BAR 的 GT 导向采样出现，机制尚未分离。若强几何基线已解释收益，就承认已知方法有效，不给并集改名。只有在这一检查后发现可明确定位且强几何/不确定性选择都遗漏的错误，才值得另登记 query-conditioned 预算分配设计及其去 query/容量/算量对照。

本备忘录的行动建议止于上述待检验区域选择假设；未新增科学任务、代码或运行权限。
