# SSDD 原生候选的双相位几何与阈值上限

32 图匹配诊断的主要缺口集中在 IoU 0.90 的候选质量。原重放的两个代价相位选到同样的配对，两套配对沿用相位 A 的 IoU。这里分别重测相位 A 与 B 的二值几何，再检验固定原生查询的形状排序是否容许只调阈值就达到严格重合。 [实际匹配归因](RFDETR_E24_FIT_MATCHER_COST_ATTRIBUTION_PROTOCOL_20261109.md)

复用 SSDD FIT 预锁 32 个原图组的已闭合轨迹。30 个合格轨迹全部纳入，保留两个增强空视图。沿用原始增强、13 组各 100 个原生最终头查询、相同空间特征与查询系数、同一 BF16 点乘和严格 logit >0 二值化。逐轨迹复现原生 1,296 点 logits 和标签。两相位各独立覆盖 576×576 网格，双线性预测和最近邻真值。先独立重现相位 A 的全组、小目标 IoU 0.75/0.90 缺候选计数。报告相位 A、B、同一查询在两相位均合格三个口径。首训练组与其余训练组分列，首组仍处于训练增强条件。无新前向、权重更新、DEV、TEST 或 AP。 [原生掩膜投影](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/heads/segmentation.py#L310-L341)

阈值上限只取首组原生匹配行。若该匹配在两相位均达到 0.50 且首组没有在两相位均达到 0.90 的同一候选，即进入诊断。按相位 A 的真值枚举固定行所有严格阈值，取最高 IoU，并列时优先原零阈值，否则取最接近零的枚举值。把选定阈值应用到相位 B，记录两相位均达到 0.90 的比例。按原图组聚合 2,048 次 bootstrap。至少 15 个有资格图像组、25 个有资格实例、图像均值恢复率至少 15% 且区间下界大于零，才支持进一步检验无真值校准。该门槛只筛选后续机制，不把同图相位转移表述为独立泛化。

若仅调阈值的潜力有限，下一步可在相同匹配行、冻结空间基底的条件下检验查询系数上限，并以原双线性读出的自由节点作正控制，区分系数估计、空间基底和网格限制。真值参与的上限只用于排查。系数与基底组合已有直接先例，B2Inst 也已显式学习边界基底。 [Mask DINO](https://arxiv.org/abs/2206.02777) · [B2Inst](https://openaccess.thecvf.com/content/WACV2021/papers/Kim_The_Devil_Is_in_the_Boundary_Exploiting_Boundary_Representation_for_Basis-Based_Instance_Segmentation_WACV_2021_paper.pdf)

阈值枚举用 63 个非空合成标签与穷举核对，覆盖平分与 IoU 边界。真实入口复用 P125 的轨迹闭合验证器。DGX 用同一隔离运行环境，40 GiB 可用内存下限、10 GiB RSS、6 GiB CUDA 分配上限、600 秒工作体上限。 [诊断源码](code/RFDETR_FIT_MASK_GEOMETRY_PHASE_20261109/mask_geometry.py)
