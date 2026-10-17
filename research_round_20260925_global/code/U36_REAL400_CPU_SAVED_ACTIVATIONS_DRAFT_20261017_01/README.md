# U36：FP32编码激活CPU暂存资源试点（未执行）

保留原plain64与完整自主full卷积结构、真实RGB400、chunk1、9GT各1/9、每图一次编码/更新及相同AdamW。只在训练encode的反向保存机制中使用save_on_cpu(pin_memory=True)；不采用checkpoint重算、压缩、混合精度或VJP。

原两CPU合约先运行且CUDA保持cold；设置原4GiB配额后，先执行tiny Conv/BN/非线性/head的两种保存方式、各2次AdamW更新，逐张量torch.equal比较loss、每参数梯度、模型/BN和Adam步/矩；失败不放宽容差。tiny **4次真实更新**与之后主图两臂各2次、共 **4次真实更新**分别计数，总计划8次一次性更新，均不进入正式主训练。

同一标准save_on_cpu实例的pack/unpack委托包装透明计数，返回原payload/value，不用外层hook覆盖CPU暂存，也不额外保留张量。要求实际CUDA→pinned CPU保存和恢复到CUDA；累计事件字节包含重复，不能称独立激活、峰值、显存节省比例或加速。

stage0全部100查询按4、stage1非零Q2分组/置换不变；真实训练两步每臂都确认有效梯度、优化器状态、原语义头零调用与臂间CUDA清0。CPU0/1两线程、allocator4GiB/整卡观测8GiB、RSS与可用RAM各16GiB、1800秒，无自动重试。普通tiny一致性不能保证完整400等价或资源通过；活跃skip、参数、梯度、Adam与工作区可能仍占GPU。

源码只完成静态/AST及指定接线审阅，尚无数值结果；不授741图、正式50轮、收敛、AP或创新资格。数据/权重/spec/资格链不发布。原full组件见相邻[U33来源副本](../U33_REAL_IMAGE_400_PROBE_DRAFT_20261017_01/README.md)，原失败见[U35实际结果](../../U35_REAL400_PARTIAL_RESULT_AND_FP32_CPU_ACTIVATION_DESIGN_20261017.md)。新[RGB训练读取接口](../U37_FULL_RGB_MATCHED_READER_DRAFT_20261017_01/DESIGN.md)仍是单独未执行草稿，不自动进入本探针。
