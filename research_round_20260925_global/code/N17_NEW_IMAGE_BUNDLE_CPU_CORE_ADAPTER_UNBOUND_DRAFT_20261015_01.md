# N17 新图捕获包到独立 CPU 数值核心：UNBOUND 接口

状态：普通开发副本，未读实际科学文件，未执行科学诊断或赋予资格。GPU outer01→02 只修改 COCO 外层 ABI 与失败18记录；本解码器不处理 RAW 计数，新图捕获包与 CPU tuple 接口不变。

入口：`decode_image(torch, np, old_shared, new_bundle, row, expected_binding, expected_input_carrier, native_predictions) -> ImageDecoder`。所有对象由 ROOT 的 closed-ref loader 预先读取；模块不打开文件、不加载模型，不把内部字符串 member ID 当物理 ref。`row.bundle` 是新包严格 `{path,bytes,sha256}` ref；`expected_input_carrier` 为 `{bundle,shared_member,original_GPU_binding}`，指向原 N16 真实包和共享成员。旧共享对象保留引用，不写入或复制到新包。

解码前核对新包九字段/schema/binding/index/image_id、旧共享完整 identity 与原 qids 顺序、自然100 query、FP32 F/e/L、INT64 union/C、物理 ref 与内部 member ID 区分。所有成员及 registry/manifest locators 顺序、schema/role/query/epoch/mode/chunk、实际 tensor 字节数、2048 chunks、每 query 固定八 profiles、无缺失/未知/重复成员均检查。mask 采用 little packbits，截取真实 H×W 并拒绝非零尾 padding；保留实际 RLE 和原 class/S0/非 segmentation 字段。

`decoder.query(q)` 返回可传给 selected CPU core 的参数：`query_id,native_L_grid,e0,C_flat_indices,K,common_PR_chunks,iter_scout_chunks,read_profile`。它仅用原共享 union 的 exact searchsorted 恢复实际输入，无 full-F 补零、重渲染或旧 scout logits 替代。

- `iter_scout_chunks(epoch,arm)` 逐原顺序返回六 tuple：`(flat_indices,F256,e0,raw_native_L,actual_extra8,actual_GPU_logits)`，全为 CPU tensor；模型为 E24/E8 的 LOCAL_ERROR/CENTER_ERROR。
- `read_profile(epoch,arm,mode)` 返回四 tuple：`(actual_selected_flat_indices,selected_PR_chunks,actual_BOOL_mask,actual_RLE)`；PR 每块六字段为 `flat_indices,sampled_F,e0,native_L,GPU_raw_residual,GPU_combined`。
- `decoder.predictions(policy)` 按原 export qids 顺序返回该固定策略100条预测，保留原非 segmentation 字段。
- 调用所有 query 的四模型 chunk 和八 profile 后，`require_all_consumed()` 要求已解码/消费集合等于完整 registry，检查时浏览元数据不算消费。

CPU 数值核心另行执行真实八邻/中心、两公式绝对1e-4/rtol0、实际 GPU logits 的 FP64 排序、selected PR strict sign、mask/RLE 等门；本模块成功只表示结构与接线通过。ROOT 外层仍须闭合实际物理 SHA、Source/spec/admission/RAW、原载体映射、qualified 模型、分段实际终态、完整 scope/COCO 和主决策；本模块不自证 CUDA marker 或实际终态。

合成检查：一项手工3×4网格、自然100 query、8 profiles/query 的内存 fixture，反向原 qids、K0、union gather、六/四 tuple、mask/RLE 和完整消费通过；错 bitorder、member 顺序、缺成员、错 role 被拒绝。fixture 的 CUDA 字段仅模拟存储 ABI，实际只运行 CPU，不使用 SAR 数据或真实权重。RED stub 拒绝后 GREEN：1 test passed，exit0；不重复已核旧 kernel 或科学数值检查。
