# SSDD 训练匹配的采样归因

原有实测中，分层损失点采样的短程控制高于同损失均匀组，但没有超过原生热启动或成熟 E24。上一轮原生网格反事实的错分抵消降幅也低于预定门槛。下一项先核对匹配阶段是否因有限取点而丢失高 IoU 分配。 [同网格诊断](RFDETR_E24_FIT_NATIVE_GRID_CANCELLATION_PROTOCOL_20261109.md)

固定训练模式的原生最终头、E24 状态、实际增强样本、13 组各 100 查询、类别与框成本及原生掩膜成本权重。固定 1,296 个匹配点。比较原生整图均匀坐标和 36×36 等面积分层的格内随机坐标，所有组共用各臂同一批坐标。两臂共用基础随机数和全部候选，保持后续损失采样独立。原生实际均匀抽样与同坐标重算作为精度与实现对照。 [RF-DETR 匹配实现](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/matcher.py#L871-L908)

用两套固定的原图 576 网格相位作密集成本参考。预测双线性采样，真值最近邻。累积足够统计量时保留 1,296 点版 Dice 的平滑尺度，复用同一匈牙利求解器。将掩膜二值化和 IoU 0.75/0.90 阈值提前固定，区分没有可用高重合候选、一对一候选竞争、密集成本未选到合格掩膜，以及稳定密集参照选中但有限点采样失去合格分配。以原图组为单位聚合和配对，另报告小目标。 [原生 Dice 成本](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/utilities/box_ops.py#L303-L337)

原生训练匹配器的均匀采样，与匹配后的不确定性损失采样是两个阶段。Mask2Former 已共享均匀点估计掩膜代价，Stable Matching 已研究层间配对稳定和位置代价。本轮隔离固定成本下的有限采样误差，只作为可训练干预的准入证据。 [Mask2Former 匹配器](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py#L102-L137) · [Stable Matching](https://arxiv.org/abs/2304.04742)

八项合成测试通过，覆盖相同点预算、保持平滑的密集积分、原生代价分解、分组求解和一对一可达性。真实样本的精确入口和执行门槛将在实际启动前单独锁定。 [数值实现](code/RFDETR_E24_FIT_MATCHER_COST_ATTRIBUTION_20261109/matcher_cost_math.py)

## 原生匹配采集入口

使用已闭合的 SSDD E24 在线权重，沿用训练模式原生增强，固定 576、BF16、13 组各 100 查询。只运行最终头前向和原生匹配，实测权重与缓冲区前后身份一致。短预检在既有 P68 编码图像组的固定顺序中选首个合格增强视图，仅用于验证真实原生代价和分配重放。后续机制试验候选预定编码哈希排序的 32 个 FIT 图像组及同一增强种子规则，逐组保留空或不合格记录。实际机制准入条件与密集重算另行锁定。 [采集源码](code/RFDETR_E24_FIT_MATCHER_COST_ATTRIBUTION_20261109/capture_matcher_cost.py)

两项合成测试通过。远端短预检限定单独工作目录、40 GiB 可用统一内存下限、10 GiB 进程 RSS、6 GiB GPU 分配及 300 秒工作体上限。保留原进程退出和日志校验。图像、标注和原始匹配张量均只存私有设备。

