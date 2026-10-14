# N17 fixedDEV GPU03：UNBOUND 最小失败谱系补丁

普通03副本由冻结普通02复制，01/02不改；未读实际科学文件、未执行科学诊断或 Git。失败19由 ROOT 证实为新 SPEC SOURCE 缺少旧导出预测 refs 的绑定错误，无 START/REPORT；缺失 refs 由 ROOT 补入新 SOURCE，本补丁不放宽 closure 检查。

新增 SPEC key：`additional_preserved_N17_secondDEV_SOURCE_closure_failure`，严格十字段 `RAW_ref,phase,pin,returncode,admission,phase_binding,bootstrap,entry_source,spec,original_report_absent`。六物理 refs 为原 RAW/admission/phase_binding/bootstrap/entry_source/spec；member 或 Path 不能代替严格 refs。

`nineteenth_secondDEV_SOURCE_closure_failure_raw_only(spec)` 在失败18之后、`pre_gate` 首 SHA 之前执行：原 actual1/held1/reaped/birth absent/errorNone/abortNone、phase/pin/来源与全部 normal 和 failed1..18 分离，minimum_ram_bytes≥16GiB、wholeGPU limit8192MiB、finite elapsed0..1800秒。RAW-only 不哈希或导入 SOURCE。

SHA 后维持六 refs 完全闭合、原 GPU CUDA0 owner/source/spec/phase/admission/bootstrap；旧 GPU02 SPEC 保持 normal122/failed18/pins140、全部122前缀与历史 pins、完整旧 SOURCE 和独立失败9..18 extras、资源1800/no retry/4096 allocator/8192 wholeGPU，原 REPORT.json 仍不存在。新03自身 normal122+shard_id、failed19、total pins normal+19；原失败18及更早所有资格门、旧训练/缓存计数不改。schema、N17_SCHEMA=SCHEMA marker、数值算法、COCO helper、新九字段 bundle/tuple ABI完全不变。

一次扩展原02虚构 fixture 的 stdlib 合成测试 RED→GREEN：旧18计数拒新19 header；新19 raw-only 合法例通过，bool/错 pin/重复 owner/错误 held或reap或birth/error或abort/低RAM/超deadline/GPUlimit拒绝；禁止 SHA/import spy通过；函数 AST 只允许 ninth counts、inherited gate、new header、report counts 变化，新 writer 类 AST 完全相同。最终 1 test passed，exit0，0.101s。首次 GREEN 尝试的测试类名拼写已更正；没有科学数据或真实权重。

ROOT 后续负责补全实际 SOURCE/Spec/admission、终态读回与资格。本普通代码不自证 actual0、独立 CPU 或 AP，也不会重做旧失败。
