# N20：真实F邻域的同预算信息检验

2026-10-15 UTC 2026-10-15T04:16:40.469Z 事前锁定。以[前期设计](../../research_round_20260925_global/N20_TRUE_FEATURE_NEIGHBOR_SCOUT_DESIGN_AND_IMPLEMENTATION_STATUS_20261015.md)为依据，新取数不是重开N17/N19评分配方。此普通文档不包含运维配置/数据/权重/执行资格载荷。

- 新输入：原始两级renderer、16通道FP32，原图整数±1九邻、逐轴replicate。不是原生F格点±1。重新捕获的中心须逐byte等于旧中心，所有旧query/flat/inverse/标签/importance保持；没有DEV训练。
- 压缩：combined=[旧unique中心，升序新增位置]，只存extra_F、extra_flat_indices与U×9索引。741图/3035136样本/1786820中心，FP32新增上界14637629440B；序列化上界18GiB另加2GiB余量，含索引与读回成本。
- 模型：两臂均18851参数，9×256零空间适配后旧513→32 ReLU→3；LOCAL真邻域/CENTER同中心复制。共同MLP来自旧factory seed202610150201，重复202/203；初态digest须在实际训练前新锁，不复用N17。
- 公平性：同FIT741/内部DEV187、旧ERROR监督、图序/点序/importance、AdamW(lr3e-4,wd1e-4)、固定24轮17784步/臂；E8及exact仅次要，不救主门。teacher/PR/C/K/query/class/S0固定。
- 主门：E24 atmost的Delta=AP_LOCAL-max(AP_CENTER,68.09938494745545)。≤0停止；0至1最多补202，只有两组各胜CENTER且meanDelta≥1再补203；≥1补202/203。仍按总目标解释重复范围。
- 顺序：真实冻结teacher邻域捕获 → 独立CPU坐标及actualGPU载体资格 → 共同初态绑定 → 成对24轮 → 固定全量评价/资格 → 主门决定。每leaf单所有者、CPU01/两线程/1800秒/无自动重试及原资源上限；失败保留。

已有源码检查：坐标手写2×3边界映射actual0（0.013s）；单图成对更新actual0（2.155s），核相同初态、真实邻域输入、相同加权ERROR损失与步数及五参数优化器；此前原型两行为actual0。新增捕获入口完成一次定向静态审阅，未将静态/合成检查当真实资格。

此轮仅测试路由观测，未改变PR可修上界。普通DW残差不是已成立创新，中心复制不匹配输入秩/协方差；正收益后才做预锁破位机制、重复和跨集。没有正式TEST、跨集、新AP或TGRS就绪结论。
