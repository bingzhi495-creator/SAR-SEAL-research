# N20 两种子预锁主结果解释器：未绑定合同

状态：仅纯标准库决策组件及合成测试；**未读取或解释第二种子真实 AP**，不执行资格认证、模型、数据或科学载荷。调用方须先独立闭合两次完整 DEV 的原任务、来源和 Root 资格，也须确认首 201 的既有规则确实允许开展 202。本函数不替代这些门。

入口：`interpret_two_seed_main(results)`。`results` 仅可为恰含两个不同预锁种子 `202610150201`、`202610150202` 的 list/tuple，顺序不限。每行恰有 `seed, epoch, mode, AP_unit, LOCAL_AP, CENTER_AP, strong_AP` 七键；要求 `epoch=24`、`mode="atmost"`、`AP_unit="percent"`、固定强参考 `strong_AP=68.09938494745545`。LOCAL/CENTER 须为有限的百分制数值 `[0,100]`；`-1` 未定义值、bool、NaN、Inf、额外 E8/exact 字段均拒绝。接口只接主结果，不接受其他端点替换。

每种子报告两个 AP **百分点**差值：`LOCAL_i−CENTER_i` 和 `Δ_i=LOCAL_i−max(CENTER_i,68.09938494745545)`。决策只依赖“两组 `LOCAL_i>CENTER_i`”且 `mean(Δ_201,Δ_202)≥1`：均成立才输出 `ALLOW_SEED203`、仅允许 `202610150203`；否则 `STOP_CURRENT_RECIPE`。不以延长预算、融合、E8/exact 或 `mean(LOCAL)−max(mean(CENTER),R)` 补救。后一聚合式另列为不参与决策的诊断值；它可能高于预锁的 `mean(Δ_i)`。

输出复制并按种子排序输入行，含单位、两配对差值、两 `Δ_i`、均值、决策和允许种子；`qualification_performed`、`source_authentication_performed`、`task_efficacy_established`、`innovation_established`、`TGRS_readiness_established`、增加预算/融合和次结果替代主结果标志均为 `False`。`ALLOW_SEED203` 只是后续试验准入条件，不等于效用、创新或论文录用判断。

验证：`test_N20_TWO_SEED_PRELOCKED_RESULT_INTERPRETER_UNBOUND_20261016_01.py` 使用纯合成数值逐步 RED→GREEN，覆盖均值恰达 1、某配对失败、聚合替代式误放行、两配对赢但均值不足、种子/模式/单位/强参考/额外字段/非法数值拒绝和不认证输出。未使用 Torch/NumPy、项目科学代码或任何真实第二种子结果。
