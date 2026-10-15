# N20 full187 final：32 CPU 执行分片合同

UNBOUND_NOT_EXECUTED。新 front02、final run-tail02、assembler02/entry02；旧01均保留。仅改变执行分区、当前正常 owner 与失败保留接口，不改变科学输入、数值证明、COCO 语义、训练或 seed201 继续规则。

- 四个完整 GPU 父保持原 owners141..144、范围0:47/47:94/94:141/141:187，以及原 failed25、报告/Source/绑定。不得把 failed26 施加到这些历史 GPU spec。
- 每父从起点每6图连续分割，尾段分别5/5/5/4图，共32段。CPU ids0..31，owners145..176；parents为0×8、1×8、2×8、3×8。完整范围由新源码常量 `N20_FINAL_CPU_RANGES` 固定，严格覆盖187图一次。
- 新 final own177/failed26/pins203，CPUempty/CPU01/threads2/RSS16GiB/deadline1800/noRetry。独立 own schema仍为 `ROOT-N20-fixedDEV-full187-CPU-COCO-final-spec-v1`；历史 schema 常量不改。
- `numeric_CPU_shards` 为32个有序 exact10-key descriptor，字段不变。`numeric_CPU_shard_gate_source` 是新完成的完整 CPU ENTRY03，不是 front 片段；逐 descriptor 闭合并按原路径加载模块。每项 evidence 仍 EXACT `{descriptor,parent,report,child}`，所有 scope_* 只在 child。四 GPU parent/full manifest保持原件；CPU输出仅自身 scope。
- `CPU_execution_partition_supplement` strictref 与 expected 完整 JSON 由 Root 预锁，schema必须为 `N20-fixed-thirty-two-CPU-numeric-subscopes-after-resource-deadline-v1`；不得将旧8段计划用于新32段。
- 新必需字段 `N20_failed_CPU_numeric_record` 为原失败26 exact10记录；`N20_failed26_helper_source` strictref；`failed26_partials_reused=False`。失败CPU原 spec145/25/170、缺失REPORT与原RAW/resourceabort保留，不计正常 owner、不消费其部分产物。
- typed22..26 的 raw_only 依序在 inherited(cpu=True)/首次SHA前，closed在后。typed26独立record.pin，不放入 `terminal_identity_pins`；全局总pins=normal+26，旧8失败pin字典仍normal+8。

CPU report binding新增前述 record/helper 两字段，top-level `failed26_partials_reused=False`，原 `failed_partials_reused=False`仍必须成立。finalQ/DETAILS与post结果保留相同两字段和False限制。其余 recovered训练、cache、init、协议、公式、FP64排序、RLE、类别/S0/原query顺序均沿原门，不允许原失败训练GPU被记为actual0。

final front须按所有32个原CPU RAW实际0/source/spec/ad/phase/bootstrap/prefix完整join后，提供 `numeric_CPU_evidence[32]`、187 rows/imageproofs和注册成员总数。32子段预测按固定图序合并为18700条后，才运行每个策略一次完整187/380GT官方COCO，禁止局部AP平均。10策略、原native/R完整预测与全部12stats、inline COCO结构、seed201解释器字节与阈值不变。

final Q保持qualified/ROOT_actual_CPU0/current_owned_CPU_terminal_self_certified=False。Root必须在final177原actual0及纯normal178前缀actual0后，另行运行post178。普通装配/AST/metadata合成检查不构成科学执行或资格。
