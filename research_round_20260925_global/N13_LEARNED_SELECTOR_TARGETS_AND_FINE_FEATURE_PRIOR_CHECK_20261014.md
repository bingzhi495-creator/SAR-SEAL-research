# 已学习的精修选择器：监督目标与完整模型边界

2026-10-14。普通文献补核；只核两个已有直接先例的具体缺口。未运行作者模型、取得新实验结果或改动当前N13配方。主任务与独立代理阅读作者原文及在线实现；正式CVF PDF本轮访问返回403，不宣称取得正式会议全文。

## Mask Transfiner：学习选择并非新的起点

作者[v1原文§3.1–3.3](https://arxiv.org/html/2111.13673v1#S3.SS2)已有受监督的incoherence区域检测器，并将检测节点交给四叉树Transformer精修。其目标来自GT掩膜缩小再放大时丢失的信息；v1以最近邻缩放、XOR及邻域OR定义，不能写成当前预测P与GT的实际错误，更不是执行某个固定修补动作后的净收益。

作者代码锁定[5b61fb53d8df5484f44c8b7d8415f398fd283ddc](https://github.com/SysCV/transfiner/blob/5b61fb53d8df5484f44c8b7d8415f398fd283ddc/detectron2/modeling/roi_heads/mask_head.py)：get_incoherent_mask使用双线性缩放残差，区域预测使用Dice与BCE。该[README](https://github.com/SysCV/transfiner/blob/5b61fb53d8df5484f44c8b7d8415f398fd283ddc/README.md)明确说明代码为重新实现；因此必须区分v1公式与该代码版本，不能混为论文原始执行轨迹。主任务已实际读取固定README、目标函数与损失文本，未执行代码。

## RefineMask：细粒度特征也必须公平保留

作者[v1§3.1–3.4](https://arxiv.org/html/2104.08569v1)从P2训练四卷积语义前景支路，通过SFM逐级融合实例/语义特征和掩膜，输出由14增至112；有语义BCE和阶段监督。BAR区域从上一阶段预测边界计算，训练还使用GT边界，不是独立学习的错误选择头。

作者[固定633ed2be1b36b3f3c798be484b6a5117004faab8的两版head](https://github.com/zhanggang001/RefineMask/blob/633ed2be1b36b3f3c798be484b6a5117004faab8/mmdet/models/roi_heads/mask_heads/refine_mask_head.py)同时有原版及Simple版本：Simple删去显式语义掩膜分类与损失，仍保留可训练特征支路及融合。主任务已实际打开作者v1和固定代码，两版必须注明。冻结低分辨率F的插值不能当作复现P2细粒度支路；仅适配BAR选点也不能称完整RefineMask。

## 对本项目下一步的约束

当前N13是固定(D,abs(L),flat)的同K几何支持与对应采样联合处理，没有训练选择器。先完成预锁三种子E48比较；若需机制检验，优先同教师、同头、同预算的(D,flat)普通几何控制，不以仅赢logit支持证明新颖性。[现有区域对照](N13_POINT_REGION_DIRECT_PRECEDENT_UPDATE_20261013.md)。

未来FIT行动效用候选只能讨论固定PR条件下的修复/破坏目标，以及能否在保持质量时降低实际计算。它必须保留同容量error head的δ_actual[2(p_FP+p_FN)−1]强替代，并报告全域PR候选、排序和稀疏执行成本；标签本身不提供不能由错误与候选变化表达的新信息。[既有目标及代数限制](N13_ACTION_UTILITY_AND_ALLOCATION_DIRECT_PRIOR_ADDENDUM_20261013.md)。

此补核支持收缩“首次学习哪里修”的措辞，以及区分机制适配与完整作者方法；不支持当前方法有效、胜过上述完整方法、首次组合或TGRS投稿就绪。当前内部DEV已有选型历史，正式测试、跨数据集和核心创新仍须实际证据。本文只保存独立概述与primary链接，未收录第三方全文。
