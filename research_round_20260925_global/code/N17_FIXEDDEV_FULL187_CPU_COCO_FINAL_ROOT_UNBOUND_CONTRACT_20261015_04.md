# N17 final04：七份 CPU 数值证据、原五份 GPU 与统一 full187 COCO（UNBOUND）

STOP EDIT。本版本仅作执行分段适配；未读取实际科学载荷、未执行模型/COCO、未操作 Git。旧 final03/post02、前三 CPU03 报告和五份原 GPU 证据保持冻结。Root 完成真实 Source 绑定、执行、原 actual0 与最终发布。

own schema 为 `ROOT-N17-fixedDEV-full187-CPU-COCO-final-spec-v3`。typed21 前门与 final03 逐函数 AST 相同，仅替换 own CPU_SCHEMA；失败21、pins=normal+21，所有正常和保留失败 RAW 先于首 Source SHA。normal/expected 使用 Root 真实完整前缀，首次合法 final owner134；更多真实已完成 owner 允许。资源 CPUempty/CPU01/threads2/RSS16GiB/1800秒、allocator4096/wholeGPU8192、noRetry 均不变。

`numeric_CPU_shards` 恰七个 exact10 descriptor：`cpu_shard_id,cpu_dataset_ordinal_range,source,spec,report,RAW,admission,phase_binding,bootstrap,cpu_shard_owner_index`。CPU ranges 固定 `[0,47],[47,70],[70,94],[94,117],[117,141],[141,164],[164,187]`，父 GPU ids 为 `[0,1,2,3,3,4,4]`。前三 owner 必须127/128/129，后四显式递增且早于 final；不得从 normal 倒推或复用失败 partial。

spec `legacy_numeric_CPU_shard_gate_source` 指冻结 CPU03/numeric-v3，`numeric_CPU_shard_gate_source` 指冻结 CPU04/numeric-v4（85239B，SHA `9f1141a4763766cd4e440c916b291fc07588675427b01f812d66eb0e1166c592`）。各 descriptor.source 按所属组核 bytes/SHA，再按其真实路径独立闭合加载，以保留 own `__file__` header。共同公式、PR、decoder、训练/初始化、GT-independent动作/排名、八种新 profiles与两种参考全部保持。原 GPU owners122..126、Source03/04各自19/20失败门、GPU五段、N16 legacy四段不改。

新增 `CPU_execution_partition_supplement` strict3ref 与 `expected_CPU_execution_partition_supplement` 全 JSON。锁 schema 为 `N17-fixedDEV-remaining-CPU-seven-partition-execution-supplement-v1`；expected 须与实际锁完整逐型相等，含 Root 原三份 completed_CPU_descriptors、owners127/128/129、七范围/父映射、reuse ids0..2/new ids3..6、全187/380/18700与10策略、方法不变/不复用失败/同时一叶/1800/noRetry。总评价墙钟时间可增加，不提高每叶资源上限。

公开 `validate_CPU_execution_partition_supplement(spec,gate)` 返回 parsed JSON，final 存入 gate 后传给 child；它只核 metadata/既有真实 owner，不重跑前三数值。前三 parent module 使用原 CPU03 `parent_GPU_gate`：完整父恰等于 CPU scope，旧 spec/report 不补 CPU_parent_GPU_id 或新 CPU supplement。后四使用 CPU04 `parent_GPU_gate` 保留完整父，再调用 `scope_parent_GPU`，读取 `scope_rows/scope_expected_predictions/scope_capture_refs`。新报告必须含映射正确的 CPU_parent_GPU_id及相同新锁ref。

各 CPU `capture_manifest` 仍指完整原 GPU manifest；其 capture_refs/image_proofs/predictions、images/queries/成员计数仅限自己的完整图像范围。final 每独立scope核一次所有成员，合计187图/187 imageproof/187真实bundle refs；重复父3/4允许，DETAILS只登记五个唯一完整父 manifest。原 bundle字节、原GPU binding、export qids/class/S0不改。所有报告、proof、预测、锁与祖先 refs 必须在 Sources 闭合。

七份每策略预测严格等于对应原五份 GPU 预测切片的完整拼接。统一全18700、380 GT、原官方 COCO12，十策略各调用一次；不能平均局部 AP。native/R 完整 predictions、stats12及 inline COCO语义 exact。原 AP比例→percent、固定171/E24-atmost 主解释、数值 proof摘要函数 AST不变，不重做模型/数值 forward。

output_directory 仍写十策略 PREDICTIONS.json、DETAILS.json和 pending Q.json，不写 REPORT.json。新 Q/details 绑定两 CPU Source模板及 CPU执行锁；checks区分五GPU/七CPU。资格和当前owner终止自报仍False。`root_qualified_result` 在原 final actual0后的真实新 own_gate上再核相同七 descriptors/锁/ref与原预览，返回独立 Root结果；final_record exact8不变，不改 pending Q。

一项 stdlib combined metadata fixture：旧五范围 RED，新七范围 GREEN；覆盖18700、原五GPU切片、原非升序qids、Source错版/原三owner/范围/RAW拒绝、post-only接口与冻结ref接线。它只证明普通接口，不提供实际科学资格。禁止自动启动、将合成通过当实际 qualified 或借分段改变主检验。
