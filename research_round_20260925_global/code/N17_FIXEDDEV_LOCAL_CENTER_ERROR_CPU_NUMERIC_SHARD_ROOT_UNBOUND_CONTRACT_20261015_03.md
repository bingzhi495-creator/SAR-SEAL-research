# N17 CPU numeric03：UNBOUND 失败21最小补丁

独立普通03由CPU02 75773B/SHA2a5f846fded2c9db1c71a7cbcf4cc6f5bb016466e82e563bc841b594be202805复制，旧02不改。未读实际科学链/数据/权重，未运行科学诊断或Git。ROOT确认原失败是binder遗漏adapter SOURCE条目，144.547秒实际exit1/no resource abort，无REPORT；此补丁只保留该失败和当前计数。

当前 `CPU_SCHEMA='ROOT-N17-fixedDEV-local-center-error-CPU-numeric-shard-spec-v3'`；normal仍由ROOT显式提供，首次修正版127、failed21、pinsnormal+21=148；127下限保留。原GPU父Source03 sid0依旧failed19/旧binding，Source04 sid1..4依旧failed20/supplement新binding，原父spec/report计数不更新。当前REPORTschema/status和全部数值proof ABI不改，仅REPORT自身失败/总pin计数升21。

新增 `additional_preserved_N17_independent_CPU_SOURCE_closure_failure` 严格十字段：`RAW_ref,phase,pin,returncode,admission,phase_binding,bootstrap,entry_source,spec,original_report_absent`。`twentyfirst_independent_CPU_SOURCE_closure_failure_raw_only(spec)` 在1..20门之后、pre_gate第一SHA之前核原actual1/held1/reaped/birthabsent/errorNone/abortNone、phase/pin和全部normal及failed1..20身份分离；六refs严格形状。normal/legacy8与原1..20前门均保留。

SHA后 `closed_independent_CPU_SOURCE_failure21(spec,closure,lineage,raw)` 闭合六原metadata refs并 `owner_join` 核原CPU CUDAempty/source/spec/ad/phase/bootstrap；原 `PRIOR_CPU_SCHEMA` 必须独立字面v2，原normal127/expected127/owner127、failed20/pins147、shard0 range0:47、全部127前缀、实际已有旧SOURCE条目/历史pins/独立9..20extras保持、REPORT缺失。该函数返回失败owner，**不把不完整旧SOURCE认作成功资格**；不要求old.sources包含原来遗漏的adapter，不改写旧spec。当前source_gate的CPU_NUMERIC_REFS逐项closed保持AST原样，必须包括真实adapter完整ref；ROOT负责新SOURCE/预检查修复，任何遗漏仍拒绝。

前门复制给final703时，保留新增 `PRIOR_CPU_SCHEMA` 和 `FAILURE21_PRIOR_KEYS`，仅替换自己的CPU_SCHEMA assignment，不能把原失败v2字面改成当前schema。前门边界在首 `CPU_REPORT_SCHEMA=` 之前。相对02原函数AST变化仅：ninth_failed_raw_only当前总计、inherited_source_gate调用21 raw/closed、validate_cpu_header当前schema/count、run仅输出当前count；新增上述两21 helper。旧1..20谓词/parent_GPU_gate/source_gate/补充校验完整AST同02；run所有数值循环/局部公式函数AST同02，PR/decoder/ranking不改。

一项stdlib fakeRAW/fakeRefs fixture RED旧02拒新v3/21→GREEN：合法21、缺记录/假intbool/REPORTflag/假pin/身份复用/controlledabort拒绝；旧v2 127/20/147且adapter不在old.sources的失败保留例通过，GPU0冒CPUempty、错spec/旧count拒绝；SHA/import spy在RAW-only不被调用，前门次序与数值AST保持检查通过。最终1 test passed/exit0（0.111s），双AST检查；不运行真实数值/COCO/科学资格。本普通副本STOP EDIT，实际绑定执行仅ROOT。
