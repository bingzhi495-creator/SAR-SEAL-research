# U49 matched-only CPU 导出 helper

<!-- U49_ORIGINAL_EXECUTION_ACCEPTED_20261018_01 -->

**后续实际状态：** 原入口已在Root独立CPU任务中实际执行并接收，2图/28匹配实例，五字段等价、独立紧凑存储和真实weights_only CPU重载通过。耗时4.609秒，峰值RSS588648448 B，私有包总43366454 B，CUDA未初始化。见[实际摘要](../../results/U49_MATCHED_CPU_EXPORT_ACTUAL_20261018.json)。这只证明本机两图导出，不证明全量或DGX兼容性；源码及下文开发阶段状态保留，包和GT不发布。


**SOURCE-ONLY / 未绑定 / 未实际导出。** 本轮仅运行3项微小标准库元数据测试，未导入 Torch/NumPy/PIL，未读取真实图像、GT、缓存、权重、资格链或运行状态。真实张量复制、摘要、紧凑存储、序列化和 `weights_only=True` 重载均未执行，须由 Root 后续独立 owned leaf 核验。没有模型、optimizer、前向、训练或 AP。

范围固定为原 FIT **dataset_index 54 和710** 的两图试点，不能把原索引改为0/1；不是741图导出或原U40跨环境续训。包含私有真实RGB、目标和来源，**不得随研发源码直接发布到 GitHub**。Root负责原科学来源准入、实际调用原U37.load、所有资源门、输出目录保护与真实终态接收。

## API

`package_metadata(dataset_index,item,record,provenance)` 是纯标准库公共合同。`record` 必须来自原 `reader.records[dataset_index]`，不能由 item 临时拼造独立证明。函数核 G/query_ids/完整identity与record相等、query/ann一对一且原顺序不变、原RGB回执和800/400整数几何、三份原源码摘要及显式原输入引用；返回独立深拷贝。元数据测试仅检验这些规则，合成ID不代表真实54/710图内容或有效数据资格。

`export_item(torch,dataset_index,item,record,*,provenance,check)` 接受原 U37.load 返回的完整11键 item（包含 F，但不访问其值或存储），返回一个便携包。`check()` 由未来 Root 控制提供。torch由调用者注入；函数不顶层导入任何科学库、不扫描路径或读取原文件。

必须先由原 U37/底层 N28 reader 完成其既有 F/M/query/GT/来源校验，再调用本函数；不能先跳过 F 验证后借省略导出 F 宣称兼容原完整reader。函数证明输出相对于**传入原item**的等价，不独立重做教师前向、matcher或真实GT栅格，也不能认证任意调用者伪造的字典。M直接取原 matched M；无插值、自然查询二次索引或从 L 重建入口。

五个字段均须为有限、detached、CPU FP32，按原G轴原样复制：

| 字段 | 形状 |
| --- | --- |
| rgb400 | `[1,3,400,400]`，原U37输出，值在[0,1] |
| e | `[G,256]`，原matched顺序 |
| M | `[G,200,200]`，已经是GT顺序的原coarse logits |
| L | `[G,H,W]`，原N16 native_L，非强R修补L |
| targets | **单个**`[G,H,W]`张量，原逐实例完整网格0/1目标，不是list |

原reader未要求输入连续，本helper对每字段执行独立clone/contiguous，不改输入。未来实际执行时要求 `torch.equal`、原/副本逐字节SHA相同、storage_offset0、连续且 storage.nbytes 精确等于逻辑字节，源/副本存储指针不同；不携带 F、自然100-query L storage、geometry闭包或原item引用。没有常驻缓存。

`save_export(torch,package,destination,*,check)` 在调用者拥有的现存目录中以独占 `xb` 写单图新文件，再真实 `torch.load(map_location='cpu',weights_only=True)` 并核元数据/五字段/摘要/紧凑存储。返回 `{package_ref:{path,bytes,sha256},weights_only_CPU_roundtrip_verified:True,qualified:False}`。失败不删除已有文件、不覆盖或自动重试；Root须保留实际失败。调用者逐图释放原item和返回package，不能把函数内部没有全人口缓存说成调用者已释放其他载体。

## 精确便携结构

顶层为：`schema='U49-matched-CPU-portable-item-v1'`、`metadata`、`tensors`、`equivalence`、`selected_indices=[54,710]`、`images=1`，以及 `qualified/new_training/AP/DEV/TEST/whole_population_exported=False`。仅 tensor 和原生 primitive，保存时不存 Python 自定义类或 NumPy 数组。

`metadata` 恰有：`dataset_index,image_id,G,query_ids,ann_ids,identity,rgb_receipt,geometry,provenance`。完整 identity 和 rgb_receipt 深拷贝保留原字段与排列；geometry 仅为 `{original_shape,actual_img_shape,canvas_size:800,branch_size:400}`。重载端应调用原 `image_geometry.make_geometry(torch,original,content,800,400)`；不序列化闭包，也不在导出时重新生成RGB或运行renderer。原几何为整数 round-half-up keep-ratio、右/下零pad；resize/render沿原bilinear、align_corners=False、antialias=False。

`provenance` 恰有以下三块，其引用只作元数据保存/结构校验，**本helper不自行重新哈希这些输入文件**：

- `context`：`FIT_context/N13_qualified/N13_report` 三个原strict refs。
- `source_refs`：`rgb_matched_reader/matched_reader/image_geometry` 三个原strict refs，必须匹配代码常量中的原SHA；实际模块和来源闭合由未来caller负责。
- `input_refs`：调用者显式给出原实际输入strict refs，不做数据发现；至少包含原image与三context，通常还包括该图carrier/N16packet/nativeL、原FIT注释来源等实际依赖。不能因便携包省略F而抹去已实际读取的来源。

`tensors` 恰为 `rgb400/e/M/L/targets`。`equivalence` 同五键，各值为 `{shape,dtype,logical_bytes,source_sha256,sha256,bitexact,independent_storage,compact_storage}`；成功的后三bool均True。SHA算法是原CPU张量 `detach().contiguous().numpy().tobytes(order='C')` 的 SHA256，不加header；shape/dtype另行严格记录。字段值都保持原GT顺序，`query_ids`不是0..G-1重编号，不再用它索引已经matched的M/e/L。

## 源码依据与本轮证据

普通发布副本 U37 `rgb_matched_reader.py:96–152`、底层 N28训练 `reader.py:130–175`、U39 `resource_probe.py:153–241`；U34 `shared_initialization.py:178–289`说明后续可独立CPU共同初始化，本helper不调用它。原三源码SHA写在 `SOURCE_SHA256`，geometry闭包没有可移植实例属性，故仅保留有源码依据的四项构造参数。

标准库测试垂直执行：首例缺模块 RED `344a8b` → GREEN `b9d0ad`；query/GT重排拒绝 RED `86f3cd` → 两例GREEN `39e8b8`；geometry/Source/原图引用/primitive拒错 RED `3d095b` → 三例GREEN `5bfb43`（3 tests，0.001s）。这些仅为元数据/错误路径证据，不是tensor等价、真实包导出、GPU或DGX通过。

## 新增 CPU owned entry（仍未运行）

`entry.py --spec ABS` 是供未来原执行器调用的独立薄入口，本轮没有调用 main、prepare/load、科学库导入或实际导出。**既有 matched_export.py 和 test_metadata.py 保持原 STOP 字节不变。** 新入口不创建 Root binding/资格/执行进程，不写Q或自动重试。

spec schema 为 `ROOT-U49-matched-CPU-export-spec-v1`，operation 为 `CPU_U49_MATCHED_TRAINING_INPUT_EXPORT`。必需字段如下：

- `phase`，全新 `output_directory`（canonical D盘绝对路径，独占创建）。
- `dataset_indices=[54,710]`；`expected_versions={torch:'2.0.1+cu118',numpy:'1.23.5'}`。
- `limits={allowed_cpus:[0,1],threads:2,maximum_RSS_bytes:17179869184,minimum_available_RAM_bytes:17179869184,deadline_seconds:180,CUDA_visible_devices:''}`。原执行器须先设置 CPU affinity 及 OMP/MKL/OPENBLAS 三项线程环境为2、隐藏CUDA；whole GPU等原执行器门不在入口伪造。
- `source_roles` 精确五键：`entry_source/export_source/rgb_reader_source/matched_reader_source/geometry_source`。导出helper锁定原12281B/30755d77…，三reader/geometry锁原公开SHA；不能借入口更换实现。
- `context_ref`：原 `ROOT-U38-full-method-CPU-context-v1` JSON，含 `FIT_context/N13_qualified/N13_report` 三个strict refs。
- `library_sources` 精确十键：`psutil,torch,numpy,PIL,PIL.Image,PIL.ImageFile,pycocotools,pycocotools.coco,pycocotools.mask,pycocotools._mask`，均为调用者预登记的**实际canonical文件ref**，包括所选编译扩展；不接收C盘junction别名与实际E盘导入路径混用。它只是显式选定库文件闭合，不是完整二进制/OS环境闭合。
- `input_sources`：原四metadata refs（FIT_context/N13qual/N13report/FIT注释）及两图各carrier/N16packet/nativeL/JPEG，合计12项；context_ref另列，总输入13项。引用只从原JSON精确字段取得，不猜路径。`sources` 纳入全部声明的代码/库/context/inputs及原执行器需要的明确依赖；不得依此增加其他图tensor读取。

入口先流式核全部显式Source字节并建立有限、逐用检查文件身份的hash缓存，然后才导入真实库。实际导入路径、版本、CPU affinity、线程与CUDA冷态均核对。只动态加载四个固定普通模块，随后调用原 `rgb_reader.prepare(torch,np,c,context,matched_reader=...,image_geometry=...,source_refs={matched_reader,image_geometry})`。完整741/1661 metadata仍由原reader验证；**COCO对象确实构造、完整FIT标注记录确实解析**，仅实际load54/710并由原reader重栅格其原GT、核原matcher身份及F/M/query摘要，不重算匹配。

每图来自同一个真实 `reader.records[index]`，依次export/save，输出 `MATCHED_FIT_54.pt` 和 `MATCHED_FIT_710.pt`。逐图以weakref核源/副本/roundtrip tensor对象释放后才读取下一图，清理reader/COCO持有对象与动态源码别名；不缓存整人口tensor。不会构造teacher/UNet/optimizer、执行模型forward或loss，CUDA前后均不得初始化。

`REPORT.json` schema 为 `U49-matched-CPU-export-report-v1`，状态 COMPLETE/FAILED、qualified/new_training/AP/DEV/TEST/whole_population_exported均False。保留每图实际stage和已写包ref，以及显式白名单 `summary`（G计数、字段shape/dtype/bytes/SHA/等价与roundtrip标量，不复制image_id/ann_ids/querymap/私有路径）。Source/input refs与package_ref是私有运行审计，整个REPORT也不直接公开；Root可另取sanitized摘要。失败保存partial REPORT及已有输出并抛出原错误，不清理失败产物。

报告还保留实际CPU/NumPy/Python RNG、默认dtype/grad/线程在owned两线程设置后的保持情况、动态模块别名恢复、CUDA冷态、tensor释放及采样RSS/freeRAM/耗时。资源采样间隔20ms，不声称绝对RSS峰；全部显式源在结束时重新检查身份/必要字节。成功必须两图完成且所有旗标、资源和180秒门通过；这些均是未来实际执行项，本轮没有实测值。

新增 `test_entry_metadata.py` 只有两项纯标准库错误路径/白名单测试：环境/索引/Source会员约束 RED `cf939a` → GREEN `bd0d80`；sanitized摘要不泄露身份路径 RED `f2047b` → 两项GREEN `bfd6ba`（0.001s）。未重跑已STOP的helper测试，未使用fake tensor或假科学结果。
