# R101显式骨干覆盖候选：有限实现审阅与实际加载边界

2026-11-07。独立代理只读新coverage_check.py与test_coverage.py，Root另读完整源码及测试；均未读取权重/真实模型或运行SOURCE，没有SSH、科学库执行或重跑已有14项测试。结论为PASS，严格限定纯metadata转移计划的静态范围；采用继承模型的独立代理，不冒称跨模型审计。有限审阅没有授予来源、张量加载、模型前向或训练资格。

固定3/4/23/3与各stage通道规范独立于输入字典，必须完整映射520条、42605504个state元素（含FrozenBN缓冲，不是全部可训练参数）；目标只接586键的完整backbone.bottom_up子集，含66项不迁移offset。每处FrozenBN四项缺一拒收；源和目标同时给出相同错误shape，也不能绕过独立规范。dtype/numel/bool、非法块号、shortcut、未知字段、源offset及额外FPN/头键有明确拒收逻辑，JSON在每层拒重复键，映射另外防重复目标。

这项检查不能把调用方metadata变成真实模型事实。可选SHA只是格式检查，加载前source与target-initial摘要允许不同。实际加载、来源认证、模型身份、未加载数值不变和DCN函数等价等输出均保持False。目标参数必须是586键骨干子集，不能直接拿fullmodel.state_dict()调用；未来执行器还须证明该子集确从完整模型提取，并检查子集外所有参数/缓冲未变。

14项标准库SYNTHETIC测试由实现代理按5个RED→GREEN周期完成，最终实际退出0/0.228秒。它们只检验明确小字典的解析、架构与危险边界，不是真实ResNet101、D2转换器、权重或科学suite测试。原TDD日志保持私有，公开源码和有限摘要见[候选README](code/R101_EXPLICIT_BACKBONE_COVERAGE_DRAFT_20261107_01/README.md)。Root此次没有重复已有绿色suite。

后续真实加载至少还要完成：从固定checkpoint和实际完整模型现场提取张量；保存独立克隆的加载前状态而非共享存储的浅拷贝；显式逐key/dtype/shape/numel与有限值检查；逐张量验证已载入值等于源、全部未载入参数/缓冲不变；核实际RGB量纲/单次归一化、stride、FrozenBN/DCN属性；固定构造种子/RNG并在加载后建立新空优化器；按原资源资格执行有限真实前向后，从独立update0建立预训练轨迹。当前随机初值512不能接入新初始化。

FrozenBN的加载器可能自动补齐缺失running_mean/var，所以strict=True未报错也不充分；完整统计必须加载前查全。DCN zero offset经sigmoid得到mask0.5，复制普通卷积权重不证明函数等价，不为制造等价而乘2或改mask。论文实际初值、完整系统上游信息/成本匹配仍未知；不称作者论文复现。

原始依据：[作者FrozenBN加载](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/layers/batch_norm.py#L68)、[作者R101构造](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/backbone/resnet.py#L646)、[官方转换器](https://github.com/facebookresearch/detectron2/blob/247691a1c292654867deb35cabb27f18d75a1363/tools/convert-torchvision-to-d2.py#L39)、[公开初始化入口与风险](DIFFSAR_INITIALIZATION_GAP_NEW_EVIDENCE_20261107_02.md)。本轮没有新方法AP或模型效用结论。
