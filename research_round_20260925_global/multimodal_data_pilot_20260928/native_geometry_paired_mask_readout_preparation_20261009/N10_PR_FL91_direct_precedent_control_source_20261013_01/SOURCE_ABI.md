# N10 C/PR-FL-91 SOURCE 与后续 ABI

日期：2026-10-13。状态：未绑定、未运行的 SOURCE。只从独立发布副本读取原 N07 普通源码进行适配，没有读取原 SCI、GT、权重或缓存，没有计算科学文件 SHA，没有导入或执行科学数值。ROOT 统一资格、来源绑定、运行与最终结论。本说明不是 SPEC、回执或资格证明。

## 文件及接口

- `point_heads.py` 只定义新 PR factory `make_point_head(torch,seed)`；不复制或改写旧 C factory。
- PR `forward(F,e,L,patch,arm='PR')` 保留原缓存的四输入 ABI，返回 `logits/residual/band`。F/e/L/patch 分别为 `[S,256] / [S,256] / [S] / [S,49]`，有限 FP32 同设备。e 和 patch 仅验证，不进入 PR 函数。patch 仍须在 [0,1]。
- 三层 91 隐层均重新拼接 `clamp(L,-8,8)/8`；F 使用无 affine LayerNorm；零初始化最终层；在原生 `|L|≤2` 内加残差，带外恒等。空 S 支持。该头不读取 GT、不采样、不筛 query、不计算或改变 S0。
- `train.py` 为新 N10 C/PR 两种子固定 8 轮训练器。C 从旧 `point_heads_source` 的原 factory 原路径构造；PR 从新 `PR_point_heads_source` 构造。每 seed 共用一个私有 image-order generator，产生一次顺序供两臂共享。没有两臂权重相同的声明。

## ROOT 需要绑定的训练 SPEC 字段

训练 schema：`ROOT-N10-C-PR-FL91-point-training-spec-v1`。训练 status：`N10_C_PR_FL91_POINT_TRAINING_COMPLETE_PENDING_ROOT_QUALIFICATION`。checkpoint schema：`N10-C-PR-FL91-point-head-checkpoint-v1`。

沿用 N07 的入口和缓存字段：`entry_source`、`cache_entry_source`、`point_heads_source`、`cache_CPU_reader_source`、`cache_spec`、`cache_report`、`cache_GPU_receipt/admission/phase`、`cache_CPU_qualification/receipt/admission/phase`、`sources`、`RAW_inputs`、`prior_failed_CPU_receipt`、`phase`、`GPU_admission_file`、`deadline_seconds`、`output_directory`、`allocator_limit_MiB`、`whole_GPU_limit_MiB`、`recipe`。

新增字段：

| 字段 | 含义 |
| --- | --- |
| `PR_point_heads_source` | 新 PR 点头 SOURCE；旧 `point_heads_source` 仍是原 C/cache factory，不能互换 |
| `additional_failed_RAW_inputs` | 两个额外历史失败的 exact `receipt/phase/returncode` 三键行，returncode=1；与头 CPU SPEC/Q 原样一致，连同 prior CPU 失败保留为 3 个 actual1 |
| `head_CPU_qualification` | 独立合成头 CPU 资格 Q 引用，不等于任务效用证明 |
| `head_CPU_entry_source/spec/admission/receipt/phase` | 上述资格作业的完整来源、实际终止及阶段标识 |

`RAW_inputs` 固定 29 个正常祖先：既有正常 28 个后追加头合成 CPU actual0。所有 29 normal 与 3 failed 的原始 terminal gate 位于第一次 SHA、SOURCE load 和科学 import 前；失败不转成正常祖先。旧 cache 祖先 `[:17]`、旧 cache CPU 祖先 `[:18]`、GPU/CPU 原来源索引 17/18 保持不变；新头 CPU 位于索引 28。

全部 cached `sources`、741 packet、START/TRACE、原 E24 依赖、头 CPU SPEC 的完整闭包和 29 个正常 RAW 的 owner/PB/bootstrap 都须含在新的训练 `sources`。额外失败行的 receipt 必须闭包绑定并通过原 terminal actual1/held/reaped/birthabsence 检查；不假设额外失败行有 admission/PB/bootstrap 字段，也不改写其形状。新训练 admission 与 SPEC 由 ROOT 创建，源码准备者不生成或改写科学 metadata。

训练 `recipe` 要与 `train.RECIPE` 的逐字段值相同：两种子 `202610120701/202610120702`，`ARMS=('C','PR')`，固定 8 轮；每 GT 一次 1,024 点 mean BCE，每轮 1,661 更新；AdamW lr=3e−4、weight_decay=1e−4、恒定 lr、无增广与累积；不重新采样、不读 DEV/TEST、不算 AP、不延长、不挑最佳轮。新头本轮没有独立调参或长期有效性声明。

头 CPU SPEC/Q 的 schema 分别为 `ROOT-N10-C-PR-FL91-synthetic-head-CPU-spec-v1` / `ROOT-N10-C-PR-FL91-synthetic-head-CPU-qualification-v1`，Q status 为 `ROOT_N10_C_PR_FL91_SYNTHETIC_HEAD_CPU_QUALIFIED`；来源、两个 factory、参数数、合成 identity/band/backward/CPU-only 检查以及 RAW/failure 前缀须与 ROOT 新资格作业一致。Q 自身不能代替其原始 held-terminal0 owner 核验。

## 新输出与后续读取器

四个端点键为 `C_seed...` 与 `PR_seed...`。必须使用 `key.split('_seed',1)[0]` 解析臂，`key[0]` 会将 PR 错派为 P。

| 契约 | C | PR |
| --- | --- | --- |
| 参数名 | hidden1/hidden2/final 的 weight、bias | hidden1/hidden2/hidden3/final 的 weight、bias |
| 参数张量/标量 | 6 / 40,257 | 8 / 40,497 |
| state shape | (64,562)、(64)、(64,64)、(64)、(1,64)、(1) | (91,257)、(91)、(91,92)、(91)、(91,92)、(91)、(1,92)、(1) |
| 实际头工厂 | `point_heads_source` | `PR_point_heads_source` |
| 固定轮/更新 | 8 / 13,288 | 8 / 13,288 |

`train.NAMES/SHAPES/PARAMETERS/FACTORY_SPEC_KEYS` 为按 arm 索引的字典。`check_head(torch,head,arm)` 和 `optimizer_proof(torch,opt,head,updates,arm)` 已采用按臂契约，不能沿用旧六张量全头断言。

START 的 `initial_head_state_proofs` 以完整 `arm_seed` 键分别保存冷态。checkpoint meta 保留 `point_heads_source` 为旧 cache/C 依赖，同时记录 `PR_point_heads_source`、每臂实际 `head_factory_source` 和 `head_CPU_qualification`。按臂保存完整 optimizer moment、私有 order state、全部 epoch orders、endpoint BCE 和梯度活动；`training_resume_eligible=False` 沿用原定义，不把端点声明为合法续训入口。

TRACE 为 2 seeds × 8 epochs 共 16 行，每行包含两臂的完整 FIT endpoint BCE、同样的图像顺序与更新人口。没有旧 FIT-BCE 触发 8→24 执行路径；`FIXED8_DECISION.json` 只记录已锁的固定轮数和不延长事实。REPORT 为 `N10-C-PR-FL91-point-training-report-v1`，明确 paired order/seed/cold-native identity，同时 `paired_initialization=False`，不伪装 N07 报告 ABI。

旧 `qualify_train_cpu.py/evaluate.py/qualify_evaluation_cpu.py` 不能直接运行：其 schemas、固定 cache ancestry 后的新作业索引、C/R 字符解析、初始化同权重假设、参数张量数与扩训资格都不同。后续必须写新 N10 SOURCE。独立训练 CPU reader 对 PR 需独立重建三层91/F LN/coarse再注入及 band残差，不调用训练 forward 充当独立复核；C 保留旧计算。

新评价器仍须精确复现全100原生候选、类别、顺序和原生 S0，再仅替换 segmentation。四个头与原生共五种控制；不得在新增 PR 或空掩膜情况下重排评分、筛候选或放宽一致性。ROOT 应在新 SOURCE 实际终止后进行独立 CPU 复核。

## 验证状态

当前只做标准库 AST 3.8 与静态编译检查，以及独立普通 SOURCE 审阅；不把它们称为头数值资格、训练成功、速度测量或科研结果。数值资格与训练均待 ROOT 完成。此目录尚未由源码准备者哈希或绑定，后续冻结后不得原地修改。
