# DiffSAR恢复分歧：源码假设缩小

2026-11-07，静态审阅，非新实验。运行事实据Root已结束的C/S比较通知；本页仅读绑定前普通worker副本与固定作者ZIP，未读原checkpoint/资格载荷、未SSH、未改源码或执行科学计算。

**时间线不能直接写成漏状态。** C4与S加载后模型、梯度、优化器、scheduler、RNG和loader六项指纹严格一致；后续样本、增强/targets及结束RNG/loader/scheduler一致。第5步24项loss逐位相同，第6步最大归一差约8e-6，第7/8步增至约0.0396/0.8691；C8/S8最大model归一差约0.0393位于BN running_var。仅标量loss相同尚不证明全部forward张量逐位相同，也未记录step5反向/更新后权重差，不能唯一定位首次差异发生在反向。

| 候选解释 | 本轮源码与证据约束 |
|---|---|
| 普通checkpoint遗漏或重建优化器 | worker的capture/load保存并实值复核model、每个param.grad、原优化器/scheduler、全RNG和MapDataset状态；六指纹及结束调度相同，明显遗漏、LR重置、错误样本或增强解释已显著削弱。不是全部进程状态或所有随机中间张量的证明。 |
| 未注册的可变数学状态 | 对作者head/detector/loss/SCJE-FPN非构造方法，未发现直接self属性赋值形式的跨步缓存；attention前向使用局部张量。GaussianFourier权重是Parameter，扩散常量是注册buffer。未发现具体遗漏项；该有限静态检查不覆盖所有第三方/C++内部状态。 |
| 优化器或运行布局/缓存差异 | 作者包装仅在原AdamW.step前做全模型clip，并未增加自有跨步缓存。原步数、矩和groups已恢复；Torch内部实现本次未独立核验，分配地址、工作区或热身内核路径也不在checkpoint中，因此不能把“未保存缓存”自动等同“漏数学状态”。 |
| CUDA累加差异及非线性放大 | 作者启用的modulated DCN反向源码确有atomicAdd，浮点累加次序存在候选自由度；worker关闭TF32/benchmark并不构成全算子确定性证明。它与“首个标量相同、次步微差、之后放大”相容，仍不是本次差异的唯一已证根因。 |

**BN差异需要正确解释。** head.mask_refine、IAMR和共享MACSA.bn均含BatchNorm；MACSA在p2/p3两次调用。其running状态已属于model.state_dict，不能再称未存。训练态BN使用当前batch统计；巨大的running_var差异是后续状态/推理风险，也可能只是上游特征分歧的记录，不能单凭它解释训练loss跳变。另有明确离散放大候选：loss.py634–638采用IoU topk和求和转int的dynamic_k、再按cost topk，**不是Hungarian匹配**；微差可能跨整数或排序边界，但本轮未观测匹配变化，不能写成已证机制。

当前保留原数值恢复比较失败、不授完整连续训练资格；也不据此断言模型不可训练或必定缺状态。Root另行限定的同C4控制用于区分重复运行波动与恢复相关效应，仍不能单独把运行布局效应与未注册数学状态唯一分解；本页不扩展探针或放宽原阈值。

来源：普通worker的capture_state/load_training_state/native_loader；固定作者[head.py](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/head.py#L123)（BN及MACSA L587–613）、[loss.py](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/loss.py#L634)、[原优化器](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/train_net.py#L124)、[DCN原子累加](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/layers/csrc/deformable/deform_conv_cuda_kernel.cu#L944)。
