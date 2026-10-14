# N16 OrderedDict 快照修复：普通 UNBOUND 合同

状态：仅普通源码与 CPU 合成验证，实际 scout 训练未启动。旧训练body01与GPUwrapper02均保留不改。本次不是新增科学失败、AP结果或协议变更。

新body：[N16_PAIRED_SCOUT_TRAIN_BODY_UNBOUND_DRAFT_20261014_02.py](N16_PAIRED_SCOUT_TRAIN_BODY_UNBOUND_DRAFT_20261014_02.py)。新GPU入口：[N16_FAIR_PAIRED_SCOUT_TRAIN_GPU_ROOT_UNBOUND_DRAFT_20261014_03.py](N16_FAIR_PAIRED_SCOUT_TRAIN_GPU_ROOT_UNBOUND_DRAFT_20261014_03.py)。修复仅 `_cpu_tree` 与 `tree_equal` 对 `dict` 子类（包括 `OrderedDict`）递归识别。原4张量16547、正常共同初始化、AdamW、样本/权重/图序、固定8/24、12个历史失败门、Source门和原科学协议不变。

PyTorch `state_dict()` 是 OrderedDict。旧 `_cpu_tree` 的 `type(value) is dict` 未进入模型字典，落入 deepcopy：GPU模型不会迁CPU；旧精确比较也会落入 OrderedDict 的多元素Tensor布尔比较而报歧义。新 `_cpu_tree` 对每个Tensor执行 detach/cpu/clone，映射输出普通dict；新tree_equal逐成员验证CPU/device/dtype/shape/torch.equal，接受dict/OrderedDict。

真实回归文件：[N16_ORDERED_DICT_CPU_SNAPSHOT_REGRESSION_SYNTHETIC_TEST_20261014_01.py](N16_ORDERED_DICT_CPU_SNAPSHOT_REGRESSION_SYNTHETIC_TEST_20261014_01.py)。CPU RED→GREEN覆盖旧分支的requires_grad泄漏见证、递归CPU调用计数、detach/独立存储、旧多元素张量布尔歧义、新精确比较与篡改拒绝，以及实际scout CPU快照。CPU global RNG保持，未初始化CUDA。**未执行实际GPU迁移验证**；ROOT须在其真实运行环境按源准入另验，不能将CPU测试称GPU科学资格。

新独立端点CPU资格允许 dict/OrderedDict 模型字典，但每个Tensor必须真正CPU FP32、无梯度、有限且形状精确；不会凭字典类型或 torch.load(map_location='cpu') 声称GPU保存端的CPU迁移已证明。实际Source、原owner终止、checkpoint双读回和CPU资格仍由ROOT闭合。
