# U51 matched-only 分批CPU导出 helper

<!-- U51_FIRST_BATCH_ACTUAL_ACCEPTED_20261018_01 -->

**后续实际状态：** 原入口与helper已在Root独立CPU任务中实际完成首批0–31，32图/43匹配实例；原值、紧凑存储和真实CPU重载通过，原任务退出0并接收。worker13.328秒，峰值RSS643702784 B，包共117777312 B，CUDA未初始化。见[本批摘要](../../results/U51_MATCHED_FIT_0000_0032_EXPORT_ACTUAL_20261018.json)。仅首批，不证明741图全量完成或DGX全量训练；下文未执行是开发阶段记录。


**SOURCE-ONLY / UNEXECUTED。** 本轮仅进行3项微小标准库元数据测试及AST/普通源码差分核对；未导入ML库、调用export/save、读取真实图像/GT/缓存/权重/资格或执行SSH。真实张量逐字节等价、紧凑存储和weights_only重载均待Root单叶实际运行。本目录没有entry、runner、controller、模型或optimizer。

直接先例为 U49 `matched_export.py` 的12281B/SHA `30755d77c81632f36e0bc88d442b9fa4a9ee8a9afb121670c04b681b06b8e015`，未改其原件、冻结源或实际包。新增仅是显式批次范围和人口元数据，不是新方法机制或原U40跨环境续训。

## API与最小变化

- `validate_batch(batch_indices)`：纯metadata，必须是非空Python list、长度≤32，元素为strict int且在0..740内、无重复；返回独立列表，**保持预锁给定次序，不sort、不强加升序、不改原始ordinal**。允许Root预锁`[740,0]`，不以重排“修正”它。
- `package_metadata(index,item,record,provenance,*,batch_indices)`：原U49合同增加batch参数；index须是原record.dataset_index且属于批次。完整identity/query/ann/RGB回执仍按原U37输入检查，原query/GT顺序不可变。
- `export_item(torch,index,item,record,*,provenance,check,batch_indices)`：原五字段导出，新增预锁批次在复制前后不变检查。
- `save_export(torch,package,destination,*,check)`：签名不变，仍独占写入并真实 `torch.load(map_location='cpu',weights_only=True)` 逐字段重载核验。额外检查包的原index属于其batch，以及人口元数据口径一致。失败文件保留，不覆盖、不重试。

每包 `schema='U51-matched-CPU-portable-item-v1'`；顶层沿U49：`metadata/tensors/equivalence/selected_indices/images/qualified/new_training/AP/DEV/TEST/whole_population_exported`。`selected_indices`是本次**完整预锁批次**原序列表，`images=1`只表示该单图包，所有资格/训练/评价与whole_population_exported标志均False；不得把selected_indices的长度解释成这个文件已经包含或验证了整批图片。

`metadata` 保留原9项 `dataset_index,image_id,G,query_ids,ann_ids,identity,rgb_receipt,geometry,provenance`，只新增第10项 `population_metadata={images:741,GT:1661,tensor_population_revalidated:False}`。这是原完整人口metadata范围，不是本helper完成741图tensor核验的结论。批次实际完成、多个批次覆盖/不重复、1661GT守恒和独立终态接收都由未来entry/Root负责，不能据本包自行授资格。

## 完全保留的数值/来源合同

调用者必须先真实执行原source-closed U37.prepare/load；底层原reader对全metadata和该图F/M/query/GT的既有校验不可省。helper只证明克隆相对于传入的原item等价，不认证任意伪造字典、不重新前向teacher或匹配，不打开provenance refs来编造独立资格。

仅保存有限detached CPU FP32的 `rgb400[1,3,400,400]`、`e[G,256]`、`M[G,200,200]`、`L[G,H,W]`、单个 `targets[G,H,W]` Tensor。M/e/L已经是原GT顺序，不能再按natural query_ids索引，M不能由L反推；L是原N16 native_L，非强R修补。完整原GT逐实例0/1目标仅用于数据包，不合并/丢弃。原F仅在原reader完成校验后于下游包省略；无F值访问、无100-query多余storage。

每字段实际clone后仍核torch.equal和原/副本raw bytes SHA256相等、独立storage、offset0/contiguous/storage字节数精确等于逻辑字节。输入不要求本来连续，不被修改。SHA算法仍为 `detach().contiguous().numpy().tobytes(order='C')` 的SHA256；shape/dtype另列。equivalence每字段仍恰有 `shape,dtype,logical_bytes,source_sha256,sha256,bitexact,independent_storage,compact_storage` 八键。

原identity/rgb_receipt整份保留；geometry仍仅四项构造元数据，future loader复用原800/400 image_geometry，不序列化闭包或重生成RGB。provenance仍是原三context refs、三固定源码refs、显式input_refs；不发现数据、不修改来源。torch由Root注入，无顶层ML import，不创建模型、前向或计算loss，不持有整批tensor。调用者必须逐图释放item与package，≤32批次不意味着同时加载32图。

真实包含私有RGB、GT及来源，**不得跟源码直接公开**。仅发表经核实且去身份/路径的统计；未执行、部分完成和失败须如实保留。32只是允许批次长度上限，不承诺180秒或任何资源门一定通过；Root可提前预锁更小批次，helper不自动分片、降质或重试。

## 本轮可复核证据

三个stdlib vertical RED→GREEN：缺模块 `79e68c` → 独立列表/原序 `8c2cfa`；非法/重复/超32索引拒绝 `d64450` → 两项 `bf60e7`；新index0/740与批次归属 `ae3f43` → 三项 `8609cc`（3 tests，0.001s）。fixture仅合成JSON metadata，不含tensor、像素或假GT结果，不是ML/序列化/全人口通过的证据。
