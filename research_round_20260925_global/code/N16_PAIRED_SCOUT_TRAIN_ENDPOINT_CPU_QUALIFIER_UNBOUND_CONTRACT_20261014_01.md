# N16 独立训练端点 CPU 资格：普通 UNBOUND 合同

状态：新研发草稿，未读取实际SCI/GT/cache/状态/Source/SPEC/Q/RAW，未运行真实科学训练或CPU资格任务，未执行Git。只执行随机CPU合成fixtures。ROOT须在真实缓存/GPU训练分别原始actual0并完整SOURCE闭合后才能绑定。

文件：

- `N16_PAIRED_SCOUT_TRAIN_ENDPOINT_CPU_QUALIFIER_BODY_UNBOUND_DRAFT_20261014_01.py`：无IO/Source/RAW门的独立数学主体。
- `N16_FAIR_PAIRED_SCOUT_TRAIN_ENDPOINT_CPU_ROOT_UNBOUND_DRAFT_20261014_01.py`：完整普通CPU入口，继承修复后的GPUtrainwrapper03纯资格门，自有新CPU_SCHEMA与cpu=True。
- `N16_PAIRED_SCOUT_TRAIN_ENDPOINT_CPU_QUALIFIER_SYNTHETIC_TEST_20261014_01.py`：真实CPU随机fixtures及拒绝反例。
- `N16_PAIRED_SCOUT_ENDPOINT_CPU_NUMERIC_PROTOCOL_UNBOUND_DRAFT_20261014_01.json`：无真实refs的协议建议，ROOT在GPU训练前落实不可变实际副本。
- `N16_ORDERED_DICT_CPU_SNAPSHOT_FIX_UNBOUND_CONTRACT_20261014_01.md`：已发现并修复的训练快照OrderedDict问题，旧body01/wrapper02保留，训练须采用body02/wrapper03或ROOT审阅的实际修复副本。

## 必需 SPEC 与前置门

CPU SPEC schema=`ROOT-N16-fair-paired-scout-training-endpoint-CPU-spec-v1`。沿用GPU03的全部旧carrier、机会GO、cache、fail9/10/11/12、完整SOURCE/RAW/admission和locked scientific protocol字段，另需：

| 字段 | 合同 |
|---|---|
| `entry_source` / `CPU_admission_file` / `phase` / `output_directory` | ROOT自有CPU源码、准入、phase、全新输出 |
| `training_GPU_source/spec/report/RAW` | 已实际完成的训练GPU真实refs |
| `training_GPU_admission_refs` | 单actual ref字典，不是虚拟ref/列表 |
| `qualifier_body_source` | 独立CPU数学body真实ref |
| `scouter_source` / `scout_formula_source` | 原四参数factory及显式独立CPU公式；本稿要求同一个scout module ref |
| `CPU_endpoint_numeric_protocol` | GPU训练前预锁的真实不可变JSONref，GPU training SPEC同名field及sources必须已包含它 |

当前计划：GPU training SPEC的107个正常祖先中cacheGPU/CPU索引为105/106；CPU资格SPEC108个正常祖先中最新训练GPU索引107。CPU自身own gate先完整闭合108祖先，再单列join训练GPU；缓存资格helper必须以父GPU training SPEC的107计数核105/106，不能对CPU当前108套父计数。每个失败记录保留不抹除，来源门函数保持新GPU03 semantics，仅CPU_SCHEMA切换自有literal。

ROOT实际 `ROOTcache_result` 必须证明缓存GPU/CPU各original held actual0和完整SOURCE闭合。CPU check自报status/True字段不能替代该证据。祖先GPU训练报告、initial/E8/E24检查点、START、完整TRACE与缓存packet/L carrier refs均须闭合。只load薄packet，不load完整naturalL/教师/固定PR或GT；缓存GT/target科学资格通过原qualifiedcache链继承。

## 数字协议与严格适用范围

建议JSON本身状态UNBOUND，不包含实际refs。ROOT在GPU训练前设actual status=`LOCKED_BEFORE_GPU_TRAINING`、锁定abs_logits=abs_meanCE=1e-4、rtol=0、epochs8/24、两个preexecution标志，并将JSON真实ref加入GPU SPEC field及sources。CPU SPEC后来绑定同一ref及已完成GPU report，避免protocol↔SPEC/report引用循环。

四个同GPU检查点FP32权重和同GPU捕获F/e/L输入不变：CPU标准factory前向与独立显式LN/matmul公式逐点logits纯绝对1e-4；两CPU路径的固定全FIT mean-image weightedCE分别对GPU报告aggregate纯绝对1e-4。使用原算法：每图完整FP32 batch logits，FP32 CE reductionnone、FP32 mean(weight*CE)，逐dataset顺序将每图scalar转Python float累加，再除非空图数；不sumweights归一。

这是具体固定状态/输入下的资格门，没有普适误差界。GPU训练报告没有逐点logits；聚合损失接近不能证明逐点GPU/CPU一致、符号、排名、selectedindices或AP。部署要另用实际GPU logits的独立CPU选点证明。本body不重放全部AdamW训练trajectory，只证明actual存储state、moments形状/dtype/finite/非负、精确step/超参、真实图序/计数与实际TRACE/history一致。moments读取精确不等于独立推导每步梯度。

## Body API 与检查

`validate_initial(torch, scout_module, initial_snapshot)`：factory私有正常seed202610140161，双方四FP32张量16547无buffers，原始counter0/空AdamW state、原hyperparameters与初始化逐Tensor相同。

`validate_state_evidence(torch, scout_module, snapshot, cache_rows, epoch)`：只E8/E24；dict/OrderedDict模型映射接受，但每个Tensor必须实际CPU FP32、无梯度、有限、精确shape。AdamW参数ID顺序来自真实factory named_parameters和reference optimizer，四state全覆盖，moments/step逐参数匹配；step finite scalar exact等于epoch×nonempty图，拒绝int截断。空图真实跳过、points/exposure及每轮seed202610140160001+epoch/private randperm/history/最后RNG before/after精确。无更新时原空moments合法。

`validate_trace(trace_records, E24_snapshot, cache_rows)`：完整before-image/paired-update/固定E8、E24重算图序，更新计数、points、每臂finite在线CE与各history累计逐值一致；在线mean不冒充固定端点。

`validate_endpoint(torch, scout_module, formula_module, initial_snapshot, endpoint_snapshot, GPU_endpoint, loader_callback, cache_rows, epoch, observer=None)`：薄packet同训练body字段、同份CPU输入恢复、独立两logit公式、固定全缓存CE核对；所有空图保持空且不造loss。完整actualGPUaggregate和same-cache计数必须一致。

## 报告与执行状态

Q schema=`N16-paired-scout-training-endpoint-CPU-qualification-v1`；status=`N16_PAIRED_SCOUT_E8_E24_CPU_CHECKS_COMPLETE_PENDING_ROOT_ACTUAL0`。fields包括自source/spec/CPU_admission/phase、training_GPU各refs、qualifiedcache/ROOTcache_result、numeric/scientificprotocol、factory/formula/body sources、details真ref、actualinitial及E8/E24refs和checks。

`qualified=False`、`ROOT_actual_CPU0_verified=False`、`current_owned_CPU_terminal_self_certified=False`。完成自身checks后仍须ROOT original CPU held actual0/reap/birth absence和完整Source闭合。`GPU_pointwise_logits_qualified=False`、`selected_indices_qualified=False`、`optimizer_trajectory_replayed=False`、`AP_evaluated=False`，不把loss下降当AP/效用/创新或TGRS资格。

## 已运行的合成验证

真正CPU随机fixtures由同训练数学主体训练24轮、两个非空图及一个空图形成E8/E24快照。新tests已RED→GREEN PASS：共同初始化、初始optimizer字段、两CPU公式、原weightedCE算法、固定端点/计数/privateRNG/完整TRACE；拒绝step48.25（期望48，不允许int截断）、负secondmoment、错误order、伪造trace更新、错误aggregateCE与不同共同初始化。保持外部CPU RNG、未初始化CUDA。AST静态通过。Root真实Torch环境OrderedDict回归由ROOT另验；本文不伪称已实际GPU训练或科学CPU资格。
