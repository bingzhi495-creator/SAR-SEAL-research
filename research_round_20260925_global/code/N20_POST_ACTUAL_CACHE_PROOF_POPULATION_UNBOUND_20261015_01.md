# N20 后置 proof 人口闭合（UNBOUND）

`verifyProofPopulation(rows,oldRows,gpuRows,cpuQ)` 为唯一导出，无依赖、I/O、日志、载荷加载或数值重放。文件只保存，未 import/调用/测试。ROOT 必须先确认原 CPU actual0 和完整纯 RAW，并严格 SHA 读取原 JSONL/两报告/Q；本模块不验证实际终态或来源真伪。

- rows 为 IMAGE_PROOFS.jsonl 的741行解析数组，每行恰 dataset_index/image_id/parent_packet/packet/proof。oldRows/gpuRows 为对应旧/新 report.rows，均741，ordinal必须0..740原序且image无重复。
- parent_packet与旧row.packet、packet与新row.packet以及新row.parent_packet逐字典严格相同，实际三字段 ref 的path/bytes/SHA不规范化或改写。原/新增物理packet不得重复或相互别名。
- proof按普通qualifier实际18字段闭合：6项数值/索引检查及ROOT_external_source_and_terminal_checks_required必须true；qualified/extra_feature_values_independently_rerendered/models_loaded/GT_files_opened/GT_used_for_coordinates必须false。每图index/image/N/U/E/rawbytes与对应report rows逐值一致；不复算Torch数组、坐标或特征。
- 每图N4096，合计741图/3035136点/U1786820；E按全部新rows求和，extra F字节=1024E，原始张量字节=1032E+72U。images/N/U/E/F字节同时对原cpuQ汇总。
- cpuQ须是原pending schema/status和原false范围标记，不传ROOT另行签出的qualified结果。返回独立整数和、proof_population_exact及限制字段，qualified仍false，numerical_values_replayed/actual_CPU_terminal_verified_by_this_module均false。

只闭合已执行原 proof 的完整人口、整数与flag，不能替代既有数值资格，也不认证科学Source/GT/权重、AP或新训练。
