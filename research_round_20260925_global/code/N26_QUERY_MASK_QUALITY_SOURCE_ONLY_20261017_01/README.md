# N26 query-mask质量回归基线：未绑定模块

状态：模块已实现，7项纯合成测试已实际通过；数据捕获器、独立数据验证、训练与真实评价尚未实现或执行。这里只发布研发模块，不是已训练模型或一键实验包。

固定MLP为1040→256 ReLU→128 ReLU→1 Sigmoid，299521个参数/6张量；输入原FP32 query embedding256维与预测二值mask自身bbox无padding裁剪后28×28双线性784维。empty预测输入全0。模块不接收GT作特征；真实监督须在完整原图网格以原matcher配对GT计算IoU，不是max-any-GT。teacher输入detach，只有质量头可训练，Torch/NumPy由受控入口注入，模块无顶层导入。

score_product只作FP64位置不变乘积；qhat=1保持原S0，包括signed zero。目标/输入验证拒绝非二值、非有限、维度或原shape错配，不裁剪/伪造质量值。

同团队执行7项合成测试通过，Root已静态复核代码，未重跑相同测试。覆盖已知像素IoU、空union、非方形bbox/手算插值/平移、GT输入隔离、teacher梯度隔离、FP64评分与非法输入；测试只在CPU0/1、线程2、空CUDA用手工/随机合成张量运行，无真实数据/权重/CUDA初始化。合成反向不是实际训练或端到端正确性的证明。

`python -B -m unittest -v test_quality_head`运行纯合成检查。空union=0是模块通用行为；正式捕获阶段须独立拒绝无效真实GT，不能用空GT放宽匹配监督。

这是MaskIoU风格的query-mask输入适配，不含原作者ROI图像特征，不是完整Mask Scoring R-CNN复现或核心创新。正/负结论均须限定所用信息与固定配方。

[预锁计划](../../../refine-logs/N26_QUERY_MASK_QUALITY_BASELINE_20261017_01/EXPERIMENT_PLAN.md)、[当前阶段](../../../refine-logs/N26_QUERY_MASK_QUALITY_BASELINE_20261017_01/EXPERIMENT_TRACKER.md)、[N25真实oracle结果](../../N25_FIXEDMASK_SCORE_ORACLE_RESULT_20261017.md)、[直接先例](../../N25_MASK_QUALITY_SCORING_DIRECT_PRIOR_CHECK_20261017.md)。
