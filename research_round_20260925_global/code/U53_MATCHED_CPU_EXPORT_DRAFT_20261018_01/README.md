# U53 单叶逐图CPU导出草稿

**SOURCE-ONLY / UNEXECUTED。** 仅完成两个微小标准库元数据测试和普通源码AST/差分核对。未导入Torch/NumPy/PIL/pycocotools，未调用entry.main/export/save，未读取实际图像、GT、缓存、权重、资格链或Ops。本目录只含新entry/helper及测试说明；不包含Root控制器、实际准入、账户变更或科学结果。

直接基于公开普通副本 `U51_MATCHED_CPU_EXPORT_DRAFT_20261018_01/{entry.py,matched_export.py}`，不修改U51原件、绑定Source或已导出包。改动目的是把剩余索引64..740的677图作为一个明确预锁任务，减少多批任务重复执行来源检查和启动前缀的开销。这是工程批处理调整，不是科学创新或训练收益。

## 明确的新协议

- `entry.py --spec ABS`：spec schema为 `U53-matched-CPU-export-spec-v1`，operation为 `CPU_U53_MATCHED_TRAINING_INPUT_EXPORT`；REPORT schema为 `U53-matched-CPU-export-report-v1`。不静默接受U51 spec。
- `validate_batch(batch_indices)`：非空Python list，长度至多741，strict int原ordinal在0..740且不重复；返回独立列表，保留预锁原序，不sort、不强加升序。Root实际请求须明确预锁 `list(range(64,741))`；本helper不自行选择、发现或更改剩余集合。
- 每个单图包schema为 `U53-matched-CPU-portable-item-v1`；`selected_indices`记录整份预锁列表，`images=1`仅代表当前包。`whole_population_exported=False`仍保留，不能把677图剩余批次称为本叶完成741图。
- CPU0/1、两线程、CUDA_VISIBLE_DEVICES空、RSS上限16GiB、可用RAM下限16GiB不变；此新CPU协议deadline为1800秒。整卡8GiB等原driver资源约束由未来Root独立准入保持，entry不重写driver或账户规则。没有自动重试、分片、降精度或资源上限放宽。

`source_roles`仍恰为 `entry_source/export_source/rgb_reader_source/matched_reader_source/geometry_source`，三个原reader/geometry SHA保持U51值，export_source必须为本U53 helper新SHA及字节数。十个显式库角色、真实版本Torch2.0.1+cu118/NumPy1.23.5、原三ref外层context、所选图的原metadata/tensor/JPEG严格refs、Source全闭合与source-after检查沿用U51。只限定显式来源，不宣称完整OS或全部binary闭包。

## 逐图内存和不变的导出内容

entry仅一次构造原U37 reader和原741图/1661GT metadata，然后按给定索引逐图load→clone→exclusive save→真实 `weights_only=True,map_location='cpu'` 重载→逐字段核验→释放。弱引用列表在每图开始时重置；上一图必须已通过所有原/导出张量释放检查才可进入下一图。只保留标量审计、包ref、源metadata和当前图张量，不把677张图的像素、mask或tensor留在内存。原COCO/FIT注释解析和当前图GT栅格化仍如实记为执行，不伪称无标注读取。

API不变：`export_item(torch,index,item,record,*,provenance,check,batch_indices)` 与 `save_export(torch,package,destination,*,check)`。原U37.load必须先完成其真实F/M/query/GT来源核验；不能绕过原reader或用任意字典代替资格。包只保存CPU有限detached FP32的 `rgb400/e/M/L/targets` 五字段；GT顺序、原ann/query映射、identity/RGB回执及原800→400几何保持。M已为GT顺序，不能二次按natural query索引，不能由L重建M；L仍为原N16 native_L。F及无用全100-query storage不进入包。

原clone与save/load数值循环完全保留：torch.equal、原/副本raw-byte SHA相同、独立紧凑storage、offset0/contiguous/存储字节等于逻辑字节，以及真实CPU重载逐字段核验。不创建teacher、模型、optimizer、前向、loss或重新匹配，不修改训练或AP规则。输出文件独占创建，失败保留原partial文件和REPORT，不覆盖、不重试。

REPORT的 `requested_batch_images/actual_completed_images/batch_export_complete` 仍区分计划与实际完成；中断或失败不能授整批完成。`original_metadata_population={images:741,GT:1661,tensor_population_revalidated:False}` 只是原metadata范围。跨历史批次的并集、不重复和全1661GT守恒由Root后续另按实际产物核实；本任务不能自行将旧批次与新批次拼成全人口资格。包含真实RGB/GT的包保持私有，不随源码发布。

## 本轮静态证据边界

两个标准库行为测试覆盖677图单预锁列表、741上界/原序、重复/非法索引拒绝，以及entry的新1800秒协议与U51协议拒绝。fixture只有小整数列表和合成ref字典，无图像、tensor或伪ML结果。AST核对原五字段clone和实际重载循环保持，不代表运行过这些数值路径，也不保证677图能在1800秒或现有资源下完成。实际执行、原始终态、资源和独立Root接收均待未来单叶任务。
