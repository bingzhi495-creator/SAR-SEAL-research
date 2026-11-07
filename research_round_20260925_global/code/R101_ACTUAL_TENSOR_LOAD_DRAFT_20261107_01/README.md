# UNBOUND R101真实张量加载核验 helper候选

2026-11-07。源码会在明确调用后对调用方提供的实际Torch张量和模型执行加载及比较；本次仅运行11项标准库fake-tensor测试，**没有导入真实Torch/NumPy、读取权重或模型、执行前向/训练**。源码尚未取得真实PyTorch、作者模型或资源执行资格。

## 接口

`load_backbone_verified(torch, model, source_state, coverage) -> dict`

- `torch` 为未来受控执行器导入并核验的真实Torch模块；本helper不自行导入ML库，也不认证所传模块。
- `model` 为Root在独立受控操作中刚构造的CPU完整模型。模型身份、配置、构造种子、上游来源及资源资格由执行器核验。
- `source_state` 为执行器从固定checkpoint实际提取的原始Torchvision state映射。本helper不打开checkpoint或认证权重文件来源。
- `coverage` 为前轮固定的显式R101覆盖模块。实际执行器必须核验它的来源；本helper现场从张量提取dtype/shape/numel，调用其`check_coverage`，并要求520项/42,605,504元素、586键骨干子集。没有可缩小正式规范的测试开关。

调用成功只返回普通可序列化dict，不返回或保存模型/张量/权重。`actual_tensor_load_checked=True`只在下面全部比较通过后赋予；这不是Root科学接收标志，`qualified=False`、`source_provenance_verified=False`、`model_identity_verified=False`、`forward_executed=False`、`AP=False`保持。

## 实际调用时执行的检查

1. 检查源和整个model state均为普通、非子类的`torch.Tensor`，在CPU、strided、非量化且所有值finite。按明确名字只提取`backbone.bottom_up.*`元数据给完整R101覆盖器，拒绝不完整规范。所有named parameters都必须在完整state中。
2. 独立clone完整模型state和完整源state；另外clone全部named buffers，包括不进state_dict的nonpersistent buffers。快照不与原存储或加载payload共享；不会用浅拷贝冒充“未变”。
3. 再独立clone加载payload，只将520项显式源映射替换进去，调用`model.load_state_dict(payload, strict=True, assign=False)`。不做后缀匹配、dtype转换、`.to/.float`、修改DCN、建优化器、forward或保存checkpoint。
4. 重新提取完整state，确认键集、dtype、shape、numel、finite及逐张量`torch.equal`。已映射项等于独立源快照；所有未映射state（含66 offset、头/FPN、float64 schedule等）等于独立初始快照。
5. 源state本身也须键集/dtype/shape/numel/值不变。核全部named buffers（含nonpersistent），mapped buffer等于源，其余等于原快照。核named parameter名字、对象身份、requires_grad及CPU Torch RNG不变。

输出含逐stage覆盖计数、未加载state的key/numel及dtype汇总，和已执行的比较标志。不将state元素数称为全可训练参数。比较采用**显式dtype/shape/numel加torch.equal数值相等，不是逐字节bitwise证明**。Python/NumPy/CUDA RNG、normalizer lambda中未注册张量、实际RGB/stride/FrozenBN/DCN模块属性及真实前向仍由外层执行器负责。

加载失败可能已改变这个新构造的独立模型；函数抛异常，不声称回滚，也不重试。外层必须清理并保留真实失败。它绝不能指向已运行512轨迹的模型对象，或把该轨迹接入新初始化。

## 有限测试及证据范围

从本目录运行`python -B -m unittest -v test_actual_load`。测试依赖相邻`R101_EXPLICIT_BACKBONE_COVERAGE_DRAFT_20261107_01`中的已公开标准库checker和独立synthetic架构fixture，所有张量是显式fake对象，仅以一格常量存储模拟逻辑shape，绝未分配真实R101参数。

六个RED→GREEN周期先覆盖基本转移、相同数值下float64被静默cast、加载前finite、RNG/参数身份和trainability、源state被改及nonpersistent buffer被改。后续新增回归检验独立payload快照、offset改动、非普通/非CPU/非strided/量化输入、dtype/两端同错shape、strict返回缺项、没有科学库导入。最终11个测试方法通过；原始TDD日志保持私有。

这些测试证明的是Python核验算法在明确fake语义下的行为，**不证明真实torch.load、PyTorch load_state_dict、作者R101/DCN/FrozenBN、权重内容或前向已通过**。fake Torch可以驱动成功返回，说明本helper不认证调用方框架；只有Root固定实际模块/来源/执行身份并完成真实操作后，才能解释其运行观察。没有科学训练credit或AP结论。

可发布：actual_load.py、test_actual_load.py、本README、TEST_SUMMARY.json。原日志和私有发布清单不发布。
