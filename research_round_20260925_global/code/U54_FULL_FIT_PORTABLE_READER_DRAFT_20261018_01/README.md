# U54 完整FIT便携包逐图读取器

**SOURCE-ONLY / UNEXECUTED。** 本轮仅标准库合成元数据测试及AST。未导入Torch/NumPy，未调用prepare/load，未打开实际manifest、包、原图、GT、权重、Q/RAW/Ops或访问SSH。没有runner、调度、模型、optimizer、训练预算、AP或新资格叶；未来正式DGX环境和执行协议仍需Root独立预锁。此模块不能拼接本机U40连续链。

直接先例是公开普通U50 `pilot_body.py` 的 `_load_item/_PortableReader` 以及U51/U53导出包。保留其真实五字段、原GT/query轴及几何恢复，删除仅54/710/G1与27和U49格式的试点限定，并改为逐次从磁盘load。原U49两图包明确拒绝，不混充完整人口。

## API与保留范围

`prepare(torch,c,manifest,*,image_geometry,source_refs)`，其中torch由未来真实库闭合的调用者注入，c提供`closed(strictref)`和`check()`；image_geometry为原已固定源码模块，source_refs恰为本模块`portable_reader`与本地`image_geometry`两份严格ref。prepare核本地模块对象/文件字节及原geometry SHA，然后只核manifest元数据，不打开741个tensor包。调用者须先从`c.closed(manifest_ref)`读取manifest，并将两源码和全部本地包ref纳入未来Source。

返回reader：

- `records`是长度741的原顺序metadata list，保留 `dataset_index/image_id/identity/G/query_ids`；构造时核索引0..740恰一次、总G1661。只核声明metadata，不宣称重新验证全体tensor/真实GT。
- `load(index)`仅对该原ordinal调用`c.closed`及 `torch.load(map_location='cpu',weights_only=True)`，逐字段核后返回 `rgb400/e/M/L/targets/G/query_ids/identity/geometry` 九键。这是U39实际消费的下游ABI，不是原含F的U37/N28完整ABI。G为每图真实完整匹配数1..100，保持原ann/query顺序；M已按GT排列，绝不再按natural query_id索引，L仍为原native_L，不是强R修补。
- `visits`只记录成功返回的原index，`load_attempts/failed_loads`计调用，`last_audit`仅保留最近一次的标量检查和原metadata。失败原异常传播，不自动重试，不缓存任何item或tensor。下一load前用弱引用要求上一图返回的五个Tensor对象已释放；调用者仍负责释放所有外部storage别名/view，Root资源监控负责实际内存门，弱引用不是全局别名或RSS证明。

每包须精确等于manifest指定的U51/U53 schema与整批selected_indices，images=1，所有自资格/训练/AP/DEV/TEST/whole_population标志False。字段和equivalence只有五个；CPU FP32、完整shape、finite/detached、offset0/contiguous/精确storage字节、五字段storage互不重叠和raw contiguous字节SHA均核。equivalence还须逐项等于Root提供的原接收摘要；RGB范围[0,1]、逐实例target二值不变。载入后的值直接交给调用者，不重建M、不裁剪/重排GT、不重新resize RGB、不读原JPEG或运行teacher。

原完整identity/RGB回执/三context/三源码及input_refs保留在私有metadata审计；本地geometry仍调用原 `make_geometry(torch,ori_shape,content_shape,800,400)`。原Windows来源ref和历史接收ref仅作来源，不在DGX重新打开；实际读取只有显式本地模块与当前包。Root须负责真实接收资格、manifest来源和传输字节认证，本模块不靠JSON中的schema自行证明这些资格。

## Root未来manifest合同

顶层恰六键：

```text
schema = U54-full-FIT-portable-manifest-v1
population_metadata = {images:741,GT:1661,tensor_population_revalidated:False}
original_context = {FIT_context:strictref,N13_qualified:strictref,N13_report:strictref}
original_reader_sources = {rgb_matched_reader:strictref,matched_reader:strictref,image_geometry:strictref}
exports = [三个批次字典]
rows = [741个原索引顺序的行字典]
```

exports每项恰四键 `root_qualified_ref/root_qualified_schema/package_schema/dataset_indices`，精确为：

| export_index | dataset_indices | Root schema | package schema |
|---|---|---|---|
| 0 | 0..31原序 | ROOT-U51-matched-CPU-export-qualified-v1 | U51-matched-CPU-portable-item-v1 |
| 1 | 32..63原序 | ROOT-U51-matched-CPU-export-qualified-v1 | U51-matched-CPU-portable-item-v1 |
| 2 | 64..740原序 | ROOT-U53-matched-CPU-export-qualified-v1 | U53-matched-CPU-portable-item-v1 |

本轮没有读取或假定第三批已经完成。未来实际U53不足677、失败、缺包或Root未接收时，不能构造合格全清单。不能合并成另一个selected_indices列表以掩盖历史批次来源。

每row恰六键：

```text
dataset_index: 原0..740
record: {dataset_index,image_id,G,query_ids,identity}  # 原已资格FIT记录，完整保留
package_ref: DGX本地严格ref
original_package_ref: 原导出RootQ的package_refs[str(index)]
export_index: 0/1/2
summary: 原RootQ的package_summaries[str(index)]原样
```

本地与原包bytes/SHA必须精确相等，路径可以因真实传输而改变；741个本地路径不能重复。summary仍是原六键`images/matched_instances/fields/package_bytes/package_sha256/weights_only_CPU_roundtrip_verified`，其中fields每字段八键equivalence原样保留，不重写为新数值结果。

**元数据缺口：** 现U51/U53接收记录的package_refs与package_summaries没有完整identity/query_ids。Root不能只据这些摘要编造records；须另用原已资格FIT context/record按dataset_index精确join并保留原identity/query/ann次序。每包load再次要求其中完整metadata等于该原record；原context/source refs也须等于manifest。实际构造manifest和验证RootQ/跨批次无重漏均由未来Root任务执行，本轮没有读取这些对象。

私有manifest及包包含真实图像/标注身份和路径，禁止随源码发布。原全人口1661计数只是原metadata守恒门；只有未来741次实际载入与真实训练任务证据，才可另谈实际覆盖，不能凭prepare或visit列表预授新GT/训练/AP资格。

## 本轮微小证据

缺模块RED→coverage GREEN；单行原GT/query顺序及接收摘要/传输ref冲突RED→GREEN。另做历史三批schema/索引拒错回归，共3项stdlib测试。只用一条合成身份metadata、极小ref字典和整数索引列表，不造图像、mask或tensor，不测试fakeNN；它们不代表真实load、序列化、DGX训练或资源检查通过。
