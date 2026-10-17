# U36：FP32编码激活CPU暂存资源试点（已完成单图两步实测）

保留原plain64与完整自主full卷积结构、真实RGB400、chunk1、9GT各1/9、每图一次编码/更新及相同AdamW。只在训练encode的反向保存机制中使用save_on_cpu(pin_memory=True)；不采用checkpoint重算、压缩、混合精度或VJP。

原两CPU合约实际通过，随后tiny Conv/BN/非线性/head的ordinary与cpu_saved各2次AdamW更新实际完成。10组逐张量torch.equal比较loss、每参数梯度、模型/BN和Adam步/矩全部通过，零失败、无容差回退。tiny **4次更新**与真实400单图PLAIN/FULL各2次、共 **4次更新**分开记录，总计 **8次一次性更新**，均不进入正式主训练。tiny等价仅限所测微型合约。

同一标准save_on_cpu实例的pack/unpack委托包装透明计数，返回原payload/value，不用外层hook覆盖CPU暂存，也不额外保留张量。要求实际CUDA→pinned CPU保存和恢复到CUDA；累计事件字节包含重复，不能称独立激活、峰值、显存节省比例或加速。

stage0全部100查询零残差原logits逐位复现，stage1非零Q2分组/置换通过；真实训练两步每臂的有限梯度、AdamW状态、旧语义头零调用和臂间CUDA allocated/reserved归零均通过。FULL第二步245个有效参数张量全部非零。记录的GPU阶段分配峰PLAIN/FULL为1.829/1.974GiB，50ms采样RSS峰6.611/12.996GiB。FULL每步保存/恢复事件张量字节和为7.114/57.059GiB，不能解释为去重激活量或实测带宽。

固定CPU0/1两线程、allocator4GiB、整卡观测上限8GiB、RSS上限/可用RAM下限各16GiB和1800秒未放宽。Root补充的FULL每步5.844–7.218秒仅指这张9实例影像，最低空闲RAM18.221GiB；不能外推epoch耗时或效率。原科学任务一次正常退出，0.0秒短CPU测试引起的接收门修复只涉及独立V2元数据助手，没有重跑或新增科学失败。

见[完整实测报告](../../U36_REAL400_TRAINING_RESOURCE_ACTUAL_RESULT_20261017.md)及[脱敏摘要](../../results/U36_REAL400_CPU_SAVED_ACTIVATIONS_ACTUAL_20261017.json)。[U33](../../U33_REAL400_OOM_RESULT_AND_CHUNK1_REVISION_20261017.md)/[U35](../../U35_REAL400_PARTIAL_RESULT_AND_FP32_CPU_ACTIVATION_DESIGN_20261017.md)失败继续保留。不同主干的公共层同源初始化、全741图、正式50轮、收敛及AP均未完成，不授完整方法训练、效率、创新或投稿资格。Root核验为终态/源码摘要/标量关系；张量、梯度与等价检查仍是原worker实测。

原源码保留执行前状态字样作为历史版本，不回写冻结源；实际状态以上述结果为准。数据/权重/spec/资格链不发布。原full组件见相邻[U33来源副本](../U33_REAL_IMAGE_400_PROBE_DRAFT_20261017_01/README.md)。新[RGB训练读取接口](../U37_FULL_RGB_MATCHED_READER_DRAFT_20261017_01/DESIGN.md)仍是独立未执行草稿，不自动进入本探针。
