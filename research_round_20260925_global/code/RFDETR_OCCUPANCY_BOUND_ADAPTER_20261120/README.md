# 固定匹配占据上界适配器

本草稿完成小型合成 CPU 验证，尚未用于实际 DEV。它接收调用者已经准入的记录，不打开文件、加载实际标签、运行模型或计算 AP。

```python
evaluate_admitted(records, edits_by_seed, *, eligibility,
                  geometry_module, oracle_module)

from_prepared(np, gt, prepared, *, eligibility, score_module, fit_step,
              geometry_helpers, geometry_module, oracle_module)
```

第一入口接原 `attach_truth` 记录和三种子八策略编辑表，先由原 `evaluate_geometry` 验证完整候选身份、固定匹配及逐候选预算。第二入口直接调用公开 `score_dev.fixed_match_geometry`，利用其评价函数参数接入本模块。原 `fit_step.attach_truth` 仍负责 stock `.50` 非忽略匹配与连续单元占据。没有复制匹配实现或修改原模块。

每个有效匹配按每个种子实际 `SU.edits=2k` 计算 `O_U/O_F`，保留同预算 `LF=S/LU=S/LW=S/NP=S`、原八策略观测值及零预算候选。无支持候选保持 no-op，neutral/unmatched 保留在原人口计数中。主诊断为 `O_U−SU`、`O_F−O_U`、`O_F−LW=S`。同时给出平衡 oracle 对 LF/LU/NP 的差值与原 SU 对各观察策略的差值。SF 的预算可以不同，只保留其原观察对照，不将其当作同预算 oracle 比较。

汇总沿原三种子先平均，再按固定源前缀计算组内均值和组间等权均值。4096 次配对源组重抽样使用 `PCG64(5217)`，所有差值共享同一抽样矩阵。结果包含 IoU 百分点均值与 95% 分位区间，不新增显著性门。原始数值、容差和精确平局沿已发布 oracle，差值不截零。微小负差也原样保留并单列计数，方便核对两条浮点求和路径。

`summary` 与 `observed_geometry_summary` 只含聚合统计。`private_witness` 保留原匹配身份、源前缀、每种子实际预算、观测几何及 oracle 索引，应留在私有结果中。

`eligibility` 精确包含三个布尔字段 `original_score_closed`、`lw_ap_blocker`、`lw_geometry_blocker`。原评分关闭且后两项至少一项成立才计算。这些字段是调用者提供的已核事实，本模块不验证终态或输入证明，输出明确保留 `input_admission_verified_here=False`。实际接线应沿原评分的来源、独立读回、历史 stock、标签准入及输出独占规则，不能用该布尔接口替代。当前没有新增 CLI 或运行入口。

解释边界保持明确。公开 `feature_adapter.readouts_sparse` 的 `free(qS)` 分别排序添加端和删除端，SU 已采用独立端点选择并保持等量添加/删除。`O_U` 在相同独立支持和实际 `k` 上优化，因此 `O_U−SU` 是给定预算的位置选择余量，不额外解除配对绑定。对 `LF=S` 的 `fixed(qL,ks)` 比较才同时包含解除原配对绑定的影响。`O_F−O_U` 是进一步解除面积约束的几何可达差值。这些上界使用真值，不能作为可部署策略、AP 改善或单一因果证明。实现依据为公开 `code/RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/feature_adapter.py` 第 107–124 行。

依赖来自普通公开源码。未来调用者仍需闭合这些准确版本及原匹配库依赖。

| 公开相对路径 | SHA256 |
| --- | --- |
| `code/RFDETR_MATCHED_POINT_FORMAL_20261120/score_dev.py` | `271554ea01f5e5e24e48ff2c53c3229755615aa514624d7d89e5c9ec559fb79c` |
| `code/RFDETR_MATCHED_POINT_FORMAL_20261120/evaluate_geometry.py` | `fdee52489ba2df0e9db147f26d41432a3276638360f2858fb5b8c5b739953a44` |
| `code/RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/fit_step.py` | `94e60f5b5557787687d1f228c33c249a1d7889172a225373b97bd789ec8b72b1` |
| `code/RFDETR_FIXED_BUDGET_GEOMETRY_BOUND_20261120/oracle_geometry.py` | `16e5929035d2e95f23b240d755bf9c9a80b3c53fd5eb6570467ad975b454edcf` |

3 项测试通过，用时 1.615 秒。缺模块及缺组合入口分别实测 RED 后 GREEN。测试覆盖三种子不同实际预算、源组等权与确定重抽样、缺候选或错预算拒绝、触发条件、空人口，以及两个手工 8×8 图形经公开真实 matcher/占据接口的 `.5` 分数单元和无支持记录。导入原 `fit_step` 会连带导入 Torch，但未构造或运行模型，CUDA 保持未初始化。没有读取实际图像、标签或科学产物，合成图形不代表 SAR 质量。
