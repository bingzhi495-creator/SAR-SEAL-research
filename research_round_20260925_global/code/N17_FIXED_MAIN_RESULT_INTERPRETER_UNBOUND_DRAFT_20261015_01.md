# N17 固定主结果解释器：UNBOUND

`interpret_fixed_main(results)` 只解释 ROOT 已完整资格核验后提供的1–3条有序171/172/173摘要，不读路径、不执行模型/COCO、不自行宣布 qualified。每条严格七字段：`{seed,epoch:24,mode:'atmost',AP_unit:'percent',LOCAL_AP,CENTER_AP,strong_AP}`；整数种子202610150171/172/173必须为有序前缀，strong_AP必须原精度68.09938494745545。

主 AP 范围0..100，-1表示undefined并返回RULE_NOT_EVALUABLE，不造0或STOP。显式percent与fraction单位不可混用；真实0.5百分点合法，不能仅由数值小猜成比例。拒绝bool、NaN/Inf、越界、错seed/epoch/mode/unit、缺R或额外次要策略字段。函数保存未舍入输入，门不加容差。

依据[公开预锁计划](../github_release_20261011/research_round_20260925_global/N17_LOCAL_NATIVE_LOGIT_ERROR_SELECTOR_EXPERIMENT_PLAN_20261015.md)“首组预锁判定与证据边界”：首组Δ=LOCAL−max(CENTER,R)，≤0 STOP，0<Δ<1最多172，≥1允许172/173。计划明确两组门用 `mean(AL)-max(mean(AC),R)`，且“两组各自AL>AC”才允许173；三组最终使用同一公式并要求每组paired win。这里用math.fsum计算均值，不采用mean(per-group Delta)。首STOP/undefined后提交额外组、首under1且两组门未过却提交173均拒绝；首≥1的173权限已经预锁，不另设两组止损门。

返回原主输入、LOCAL/CENTER均值百分数、原R、Delta百分点、逐组paired win、decision、allowed_next_seeds和accepted_under_locked_rule。只有完成三个合格可评估主结果且meanDelta≥1、三组逐一胜CENTER时accepted=True。E8/exact/其他策略从不替代主结果。结论仅限共同冻结teacher/PR、数据点与短预算下的私有scout初始化重复，不是全网重复、创新或TGRS证据。

开发验证：一项纯stdlib手工摘要测试RED→GREEN，覆盖0/1边界、首under1的172→173条件、Δ-of-means与逐组max平均差异、最终均值和逐组胜条件、undefined与上述拒绝项。最终1 test passed/exit0（0.004s）；未读取真实AP或科学资格链、未运行科学实验或Git。
