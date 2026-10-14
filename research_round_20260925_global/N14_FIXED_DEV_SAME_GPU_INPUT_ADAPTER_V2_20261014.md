# N14 stage24 fixed DEV 适配 v02：单图真实整包

状态：**UNBOUND / 未执行 / 待 ROOT 集成 actual ABI**。v01 Python和说明原样保留；本版仅更改 capture 保存组织与明确对应 CPU 读取接口，不改变 FP32、head数学、采样点、region支持、分数或 mask/COCO 规则。只读取被授权普通摘录/源码草稿；没有读取实际 SCI、SOURCE、RAW、SPEC、Q、缓存、权重或 live输出，没有科学执行。完成后停止改写，由 ROOT 合并、复制和预绑定。

## 固定评价及资格边界

完整主端点仍为共同旧 LOG E24 父权重后的新独立 stage24；BASE_CONTINUATION 40497 参数，MEMORY_ADDON 42698 参数，累计每分支39864更新。旧 E24只作初始化出处，不称连续48轮。只评价旧 DEV187 / 原GT380 / 每图全部100 native queries / 每 control 18700预测；不做中间epoch选择、GT guided matched-only筛选、GEO、NMS、score cutoff或TEST。

沿用原 frozen teacher/packet、`M=eF`精确投影、GPU native renderer、原natural class/query order/quality/score、原 `fixed_score_rows` 与 official COCO adapter。immutable full100 GPU L selector逐字复用；支持唯一为 `abs(actual_original_GPU_native_L)<=2`，region外L不变，两arm顺序、chunk2048。仍只有一份单armrefined100 raster。

ROOT必须实现强闭合的独立stage24 gate：训练GPU+独立CPU source/spec/admission/RAW/report/Q/details、同一对8/14张量checkpoint、三段trace、stage origin、完整24个history/orders/activity和39864 exact optimizer steps。所有正常及旧8+单独failed9 actual RAW在首SHA前终止核验，owner join和所有source/pins一致；原E24与first-stageproofs继承，不要求已训练MEM输出再次等于BASE。旧N13 `endpoint_gate/load_heads`不得作为memory资格门。`stage24_adapter_contract`是新adapter合同，不是伪称actualQ有这些字段。

`load_stage24_heads`仍支持CPUcold严格加载，GPU和CPU调用者分别先完成自己的ROOTowngate，再将真实stage24训练资格映射至同一canonical gate。`packet_selector_gate`保留旧 `n10.packet_check/selector` 的native metadata实际ABI，不能拿canonical训练gate代替。两种mode的owncontext由完整runner作者共同适配，helper不创建owner或自证terminal。

## 已与完整GPU/CPU runner作者统一的API

`load_stage24_heads(torch,spec,root_stage24_gate,bound,base_factory,memory_factory)`不变。

`evaluate_image`全部参数与五项返回不变：返回 `baseline,predictions,summary,selector_row,capture_ref`。其中capture_ref是本图 **一个真实 IMAGE_CAPTURE.pth 的 {path,bytes,sha256} ref**。

`cpu_replay_captured_image(torch,np,masks,n10,heads,capture_ref,sink,load_closed_selector,numeric_protocol,completed_GPU_evaluation_ref)`：原manifest参数位置改为wholepacket真实ref，由helper内部 `sink.read(capture_ref)` 读取一次，然后逐一验证包内dict字段。不能传virtualref、member路径ref或同路径不同内容ref。

capture sink仅需现有 `write(relative_name,value)` 和 `read(actual_ref)`。write在fresh目录中wx保存一个actual .pth，递归CPU/dtype/finite检查、exact all-member读回及真实reference；read先bound实际整包ref。GPU runner已确认直接保存CPU包，**不cpu_tree.clone()再复制整个包**。本版每图一次write；global JSON CAPTURE_MANIFEST由完整GPU runner保存并实际引用187个整包与各图U/T/bytes/binding，不新增query/member文件。

## 整包精确字段

顶层schema为 `N14-fixed-DEV-actual-GPU-point-input-capture-v2`。顶层包含 `shared`、`queries`、`GPU_predictions_by_arm`、`U`、`T`、`raw_tensor_payload_bytes`、`natural_queries=100`、`GT_query_selection=False`、独立CPU未完成/terminal不自证标志，并重复shared的非tensor provenance metadata供报告索引；CPU核这些重复字段完全相同。

`shared`包含原schema、own evaluation `binding`（source/spec/GPU_admission/phase与stage24 GPU report/CPU Q refs）、dataset_index/image_id、identity原H/W、native_packet ref、selector row/carrier ref、natural q0..99、native_export_query_ids、baseline100 RLE/score/class、support_formula、point_chunk与head_parameters，以及三个真实CPU tensor：

- `union_flat_indices[U]`：sorted unique INT64，全部100 queries旧LOG支持的空间并集。
- `sampled_F[U,256]`：从此次GPU `original_feature`真实值按2048 gather后直接复制，FP32；query间共享。没有CPU F rerender或额外whole-union GPU F副本。
- `e0[1,100,256]`：此次实际GPU e的CPU FP32副本，保持原ABI。

`queries`是natural q0..99顺序的100个dict，**空支持/空mask/未匹配query也必须有**。每项exact keys：`query_id,K,flat_indices,native_L,arms`。其中flat_indices为sorted INT64[K]，native_L为此GPU实际FP32[K]；arms的key集合exact为BASE_CONTINUATION与MEMORY_ADDON。每arm exact keys：`query_id,arm,GPU_residual,GPU_refined_logits`，后两者均实际GPU输出直接复制的FP32[K]，combined来源严格是同次 `native_L+residual`。

`GPU_predictions_by_arm`包含两arm此次实际 `fixed_score_rows` 输出的各100完整query RLE/原score/class/order，供CPU独立mask重建逐条精确比较。包内members使用字段和值验证，不伪造文件引用；wholepacket一个actual ref绑定所有members。

## CPU numeric protocol：统一8 keys

新JSONexact keys共8个（含schema）：

|key|固定值/来源|
|---|---|
|schema|`N14-fixed-DEV-same-GPU-point-input-CPU-numeric-protocol-v1`|
|same_actual_GPU_point_inputs|true|
|point_chunk|2048|
|mask_RLE_exact|true|
|locked_before_CPU_execution|true，由ROOT执行前锁定|
|bound_completed_GPU_evaluation_report|**已完成DEV GPU评价report的actual ref**，不能拿训练端点report替代|
|atol|0.0001|
|rtol|0.0001|

这不是旧训练端点CPU numeric protocol，也不是声明旧Q已有此schema。CPUowngate闭合该协议、完成GPUreport及全部187包后，传 `completed_GPU_evaluation_ref`，helper核相等。

CPU按闭合原full100 GPU L selector验证所有query支持和native L逐字、并集精确，按相同chunk和FP32重算8/14张量residual与L+residual。数字allclose后仍强制每点sign完全一致，从baselineRLE复制全mask、只改对应flatidx，逐query与actual GPU exported RLE精确相等，再原official COCO18700读取（segm、maxDets[1,10,100]、无scorefilter、原crowd/ignore，COCO12用原 `stats_exact` 规则）。N13 image113的CPU F rerender问题因此不混入head replay，但 **sameGPU输入也不保证CPU/GPU近零sign一定一致**；如失败须记录资格失败，不能事后放宽或说capture已保证成功。

## 精确payload与实际peak边界

每图 `N=H*W`、`T=sum_100query K`、`U=LOG空间并集`；新增原始tensor payload仍准确为 **`B=1032U+28T+102400 bytes`**：F=1024U，unionidx=8U，e=102400，query输入idx+nativeL=12T，两armresidual+combined=16T。保存组织变化不改变B。

全187图若U未知，上界 `1032*sumN+28*sumT+19148800`；若T亦未知，上界 `3832*sumN+19148800`。原selector `400*sumN` bytes复用，不新增整份selector文件；U=N时sampledF涵盖整图真实数值，不能把它当零保存量。没有读取实际尺寸/K，不能虚构GB总数；由ROOT闭合metadata填实际上界。

单图1真实packet，全DEV **187个新.pth+1个global JSON manifest**，替代v01的302/图=56474个新.pth。每包603个tensor（shared3+100×6），pickle/zip/RLE/metadata overhead另算，实际总file bytes由真实ref记录。全SOURCE覆盖187个physicalrefs和globalmanifest，不用virtualmember ref。

因为本版内存累计全部query inputs+两arm结果，保存前最多一整包B；sink exact readback瞬时有第二整包，按 **2B的tensor读回峰值**预锁，另计serialization/metadata/现存runtime。若sink对CPU成员再clone，则source+clone+readback将成3B；已与GPU runner作者约定不clone。GPU骨架100个indices合计8T与包中flat_indices来自同np数组（可共享storage），保守峰值门可另算8T但不要把它当新增diskpayload；完整runner资源门必须明确保守上界。

还须计原CPU full100 L解码400N、原packet驻留、union bool N、querynative重复/expectedidx/searchsorted、native mask100N、RLE/库；GPU原F为1024N、L与单armrefined各400N、queryindices、head临时与原packet投影。分块F=2097152 bytes（2MiB），helper没有额外整并集GPU F或第二armfull raster。**2B不是总进程peakRAM或GPU证书**；实际freeRAM/disk/allocator门由ROOT启动前锁定。finish一次真实保存后释放shared/queries/indices，CPU helper返回后释放整包，逐图运行不积存全DEV F。

## 交付及结论范围

v02仅完整定义helper，保持v01 loader/evaluate math；本次只AST静态检查，不调用Torch/模型/数据。完整GPU与CPU runner作者已收到统一组织/接口/8key协议，ROOT负责review/合并/actual source/spec/admission/RAW绑定。

ROOT将在任何N14 AP前独立明确第一次AP operational停止规则，helper仅返回真实fixed-stage24/完整CPUqualified AP，不内置选epoch或门槛修改。单seed该DEV/AP配方结果不能自动排除额外容量、证明memory机制/创新/TGRS/SOTA/TEST泛化。没有新增文献或模块。仅普通稿与说明进入持续同步范围；实际F/L/capture cache属于科学载荷，不自动发布。