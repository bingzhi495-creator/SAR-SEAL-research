# 续训数值偏差：独立质疑与最少下一检验

公开副本措辞限定：本轮记录证明首个恢复步骤的聚合损失相同，不能据此推断全部中间forward张量逐位一致。原始独立审阅文件保留未改；本页仍是当时的静态分析，新R实际结果见[汇总](U57_DIFFSAR_STATE_AND_NUMERICAL_VARIATION_ACTUAL_20261107.md)。

2026-11-07。仅只读绑定前普通worker副本和固定作者源码ZIP，未SSH、导入科学库、读取checkpoint或重跑实验。实际数值来自Root已核记录：C连续0→8并在4保存，S载入同C4→8；加载起点六组件摘要、后续输入/增强/目标及末RNG/loader/scheduler一致。第5步全部损失相同，第6步仅小差，第7/8步明显放大；原CPU比较退出1、预锁1e-4未通过，结论保持失败。

| 解释 | 当前支持与反证 |
|---|---|
| 漏存必要数学状态 | 不能完全排除；但模型/梯度/优化器/调度器/RNG/MapDataset均保存且加载后复核，首个恢复步骤的损失完全相同，反对“大量显式状态没有恢复”的直接解释。摘要不证明所有后台算法/布局/非注册属性都相同。 |
| 原生CUDA微扰后放大 | 有具体源候选：活跃modulated DCN的grad输入累加使用atomicAdd；worker关闭TF32/benchmark但未要求确定性算法。第5步损失同、第6步小差、以后放大的模式相容，尚未证明首次差异来自哪次反向/更新，更不能唯一归因atomic。 |
| 比较器误判 | 未见步骤错配或差值公式错误：按实际global_update/输入哈希配对，逐tensor转float64计算归一差，损失另从原日志比较。第8步单项loss归一差达0.869，不能仅以BN buffer数值大解释成打印假象。 |

BN的running_mean/var本来属于model state；已有C4摘要核对，不能再无证据说“漏存BN”。最大绝对差2.143×10¹¹发生在attention.bn.running_var，不代表已证明可训练权重爆炸；BN统计在后续eval有实际意义，也不能一概忽略。核心训练forward、head及SCJE/FPN此次静态扫描未找到显式跨步自增计数器或未注册随机生成器写入，但未穷尽全部传递依赖。

**1e-4是本次前瞻预锁的sanity准则，不是正确checkpoint恢复必然满足的数学必要条件。** 它混合检验了状态恢复和原生执行重复性；若原生重复本身超界，就会像首图非零梯度门槛一样阻碍有效研究。当前仍不得事后调大阈值、改称原检验成功或直接绕过已锁正式启动条件。应分开报告“状态连续性已获支持”和“数值轨迹重复性未获支持”。

最小新假设检验：另行预锁一次R，复用同一不可改C4、同worker/配置/输入，从4→8，仅新增4次实际更新；先核同样的精确起点/输入/RNG，再以原公式和阈值比较S–R（主比较），可预声明C–R为辅助。若S–R也明显偏离，说明两个新进程恢复后的原生重复本身不稳定，削弱“只有连续→重启导致差异”的解释，但仍不唯一确定原因；若S–R接近而C与两者均偏离，则更支持重启相关的进程历史/后台算法差异，仍非根因证明。一次R不能估计训练方差，更不能替代完整种子或AP。无论结果，不自动扩展重复、修初始化或放宽原门槛；是否建立新的原生非确定性训练政策，应据新证据另行前瞻声明，保留旧失败。

源依据：普通full worker的capture/load/native_loader与backend设置（worker.py:276–307、314–370、470）；普通comparator的配对、逐tensor和loss公式（worker.py:62–156）。作者[modulated DCN累加](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/layers/csrc/deformable/deform_conv_cuda_kernel.cu#L944)、[训练forward](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/detector.py#L537)、[MACSA/BN](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/head.py#L587)。
