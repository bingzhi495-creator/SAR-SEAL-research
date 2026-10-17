# N28：固定24轮DEV评价草稿

2026-10-17。状态：未绑定、未执行的研发源码。当前没有N28 DEV/AP结果，不能以源码或合成检查代替真实评价。最终训练端点、真实输入上下文、执行绑定及Root接收入口尚需核验与准备。

- dev_reader.py：原完整187图的N07密集F/e/M与N16实际GPU native_L读取，不用抽样F_unique代替dense特征。
- population_contract.py：保留原100query/S0/class/order；四段完整合并，拒绝缺失、重复、改分或改序。
- eval_body.py：第24轮来源门、两臂GPU预测、全query零残差native校准、完整CPU COCO及固定原native匹配的条件几何诊断。
- geometry_diagnostic.py：固定原native在IoU=.50/area-all/maxDets100的非忽略GT/query，不按新方法选择对应关系；统计Omega内外错误并报告覆盖范围。
- run_eval.py：拟用的单次原任务、Source、资源及待接收输出入口；不自行认证实际终止或结果资格。
- test_*_metadata.py / test_population_contract.py：仅标准库捏造JSON元数据的行为检查，不运行Torch/NumPy、实际数据、像素或COCO。

原factory与200→800→裁剪→原图renderer分别绑定既有[N28训练源码](../N28_DENSE_ROUTING_TRAIN_SOURCE_ONLY_20261017_01/README.md)的相同来源，资源控制沿用原受控入口；该草稿不替换正在运行的冻结源码。本目录不含原始数据、全量标注、权重、预测或运维绑定，尚不是一键复现镜像。

[评价设计与证据范围](../../N28_FIXED24_DEV_EVALUATION_DESIGN_20261017.md)。
