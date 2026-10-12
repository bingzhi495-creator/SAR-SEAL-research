# 长训实验跟踪

这是SOURCE准备记录，不是实际运行SPEC或最终科学权威状态；current worker以原始handle和receipt为准。更新日期2026-10-11。

| Run | 目的 | 系统/预算 | 划分 | 优先级 | 状态 | 限定 |
| --- | --- | --- | --- | --- | --- | --- |
| R01 | 冻结条件端点 | seed0 P/B/scalar own24+fresh48 | PSeg FIT/DEV | MUST | CPU-qualified | 累计72，teacher固定；创新未确认 |
| R02 | 预设重复 | seed1 P/B/scalar first24 | 同上 | MUST | GPU-running | 实际original57914/PID64132 |
| R03 | 整网长训基线 | native E96最终端点评价 | 同上 | MUST | SOURCE-ready | 训练已完成；最终AP未计算 |
| R04 | 完成seed1长预算 | seed1 own24+fresh48+cross | 同上 | MUST | SOURCE-ready | 未admit/launch；不可复用seed0 CP |
| R05 | 第二重复 | seed2 own24+fresh48+cross | 同上 | MUST | SOURCE-ready | 未admit/launch |
| R06 | 真正joint梯度 | synthetic + native一次完整更新 | synthetic/FIT | MUST | SOURCE-static-PASS | 尚未执行数值检查或native接入 |
| R07 | 排除额外监督 | native30 + nativeL点BCE，追加72 | 同上 | MUST | Planned | fresh24→own权重fresh48 |
| R08 | 检验结构效用 | native30 + L+u点BCE，追加72 | 同上 | MUST | Planned | 同初始化/λ/预算/评估，非简单解冻创新 |
| R09 | 强信息评分 | feature-pooled9477，24+fresh48 | 同上 | MUST | SOURCE-ready | 未synthetic/native资格/训练 |
| R10 | 强SAR方法 | DiffSAR统一协议长训 | 同上 | MUST | Planned | 环境/明确init/实际数据接入待完成 |
| R11 | 适用性 | 候选+强对照HRSID | HRSID | MUST | Planned | 当前方法未跑，先绑定划分 |
| R12 | 几何反证 | 固定native query修正/退化 | 既有DEV | MUST | Planned | 无额外训；不是各端best100归因 |

不把Planned/SOURCE-ready当实际结果，不将旧分支的跨数据集或强方法结果移用于当前方法。

2026-10-11 revision02: seed1 first24 GPU original57914 actualexit0; strict8raw0 before endpoint hashes; CPU original17087 LIVE. Do not promote provisional metrics or schedule dependent SCI before actualCPU terminal. A/B controls extra supervision only; new F/e0 inputs and parameters remain a confound.
