# 正式 TEST 与跨数据集证据准备度（2026-10-14）

本页仅定位独立发布副本已有的数据协议、元数据核查和历史审阅；未读取实际 SCI/SOURCE/RAW、影像、GT、权重或缓存，未运行科学代码、重新核验数据包或新增联网调查。既有审计直接复用，不能把准备文档当作实际数据或评价资格。

**2026-10-17最新来源增量：** 实际绑定initializer已与官方公开端点的多项指纹一致，见文末后续核对；完整远端SHA256、独立训练谱系及全人口真实像素/raster仍未完成。前文保留各阶段形成时的边界。

**测试暴露已有记录，不能默认恢复盲测身份。** [历史确认实验方案第42行](../paper_tgrs/revision_tgrs_20260905/CONFIRMATORY_EXPERIMENTS_CN.md)明确记录：HRSID test 已用于规则选择，PSeg-SSDD test 结果也已公开查看。[2026-10-11论文源码审阅](../docs/paper_review_20261011.md)进一步定位旧 HRSID TEST 融合超参数选择、best-on-test 及近岸选点；[历史修订报告](../paper_tgrs/revision_tgrs_20260905/REVISION_REPORT_CN.md)同样明确测试参与参数与方案选择。HRSID 旧 TEST 不能通过重命名恢复 blind。**2026-10-16补充：SSDD 已有具体模型选型的历史文本记录。** [历史 TEST 用途台账](HISTORICAL_TEST_USE_LEDGER_TEXT_EVIDENCE_20261016.md)定位到旧实验稿第203/209行：对应对比表的 competitor checkpoint 为 best-on-test，两项 SAR 模块本地复现按近岸子集选择。此为文稿披露，完整测试接触清单、实际参数/权重身份、运行时间关系与原始产物绑定仍缺；不能推及当前 N20，也不能宣称台账已经闭合。

**PSeg-SSDD：官方接口和转换准备已有，正式入场核验尚不完整。** [DATA_ACCESS](../docs/DATA_ACCESS.md)记录 [Official-SSDD作者入口](https://github.com/TianwenZhang0825/Official-SSDD)及[数据论文](https://www.mdpi.com/2072-4292/13/18/3690)：PSeg 真实多边形与 BBox/RBox 版本不同，官方 train928/test232，内部 FIT741/DEV187 不等于 TEST232。已有 `protocol/pseg_to_coco.py` 转换入口，但会跳过缺失 XML/无有效多边形对象，不能凭转换完成认定数量与真值完整。仓库 Apache-2.0 与外链影像压缩包逐文件授权范围未被完整映射；当前发布官方获取入口，不声明原图镜像已获核实授权。

**HRSID（2026-10-17按已核记录更新）：split元数据及本地编码文件已核，像素/mask与外部模型接口仍缺。** [官方PNG同包核验](HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)已实际取得并核验TRAIN3642图/11047实例、TEST1962图/5922实例，前版“官方包内TEST未知”已过时。[本地JPG后续核验](HRSID_LOCAL_JPG_METADATA_AND_VERSION_BOUNDARY_20261015.md)确认5604文件编码/头部，train/test标注与PNG对应split按stem、ID及全部标注字段精确相等；未完成像素、polygon真实栅格化及空间对齐。16951实例的合并文件与split合计16969不等，保留原split、不能互换；本地有两组跨split相同编码异标注图和一组test内部重复。文件名无交集不等于内容无重复或母景独立。保留原协议，预锁去重敏感性补充，不能静默删改标签或恢复盲测。官方JPG归档身份、许可覆盖与历史TEST台账仍待补；不继续等待PNG整包、不发布原数据。原许可及场景证据边界见[官方协议补核](HRSID_OFFICIAL_PROTOCOL_AND_RELEASE_SCOPE_CHECK_20261013.md)。

**现有U40内部评价不能只改人口常数后直接用于跨集。** 原[DEV reader](code/N28_FIXED24_DEV_EVAL_DRAFT_20261017_01/dev_reader.py)绑定187图缓存来源，[指标包装器](code/N28_FIXED24_DEV_EVAL_DRAFT_20261017_01/eval_body.py)也固定187；新E50核心仅面向同一内部DEV。本次未定位现成HRSID1962图全100自然查询e/M/L/S0的合格采集/读取入口，不能改名复用SSDD缓存。下一最短动作是本地JPG流式像素/mask核验、独立人口与载体来源接线，再复用[原通用COCO函数](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/spatial_position_preserving_common24_evaluation_source_20261009/qualify_evaluation_cpu.py)及[既有800→400几何](code/U33_REAL_IMAGE_400_PROBE_DRAFT_20261017_01/image_geometry.py)。这是可复用数学，不是HRSID已运行。须核冻结载体的训练/预训练接触后，才决定能否称零样本；源训练后固定参数评价也不自动取得独立盲测身份。

下一正式验证的最短依赖链，沿用[RESEARCH_GOAL的跨集与正式测试要求](../RESEARCH_GOAL.md)：

1. 完成当前开发评价/独立核验后冻结候选、强对照、权重、推理/选点/评分和停止规则；登记内部 FIT 模型直接评价与官方训练集预锁重训的不同身份，双方采用同一路线。
2. 补齐已有 TEST 使用台账，至少记录数据/图像范围、曾看结果、参数/模型选择用途。未来固定 recipe 在历史 HRSID benchmark 上的零调参 external 结果可报告，但必须披露暴露历史及切片级限制；它不同于真正未参与研究选型的独立确认性证据。独立确认需新未用于选择的场景/集合或可审计的开发—确认方案；重分旧数据不自动清除历史暴露或旧模型训练接触。
3. 由科学 owner 核验选定官方包版本/校验和、完整 train/test 清单、真实实例 mask、图像 ID/路径、转换数量守恒、类别/空图/忽略规则及许可范围。缺合格测试实例真值则停止 mask AP；JPG/PNG、纯背景补充和自建场景协议不得静默混合。
4. 冻结 HRSID 外部路线（当前任务书优先源数据训练后零调参迁移）；实际核验母场景/裁片重叠、同源和重复。未确认场景独立时仅称官方切片协议上的受限外部评价，不称独立采集/地理泛化。原结果和负迁移保留，不能看 TEST 后换阈值、轮次或数据集选优。

本页没有新增正式 TEST、外部 AP、未见测试证明或许可认证；未闭合项目保持缺失状态。

2026-10-17 U41补记：[未绑定单图采集接口](code/U41_UNBOUND_NEW_IMAGE_NATIVE_CAPTURE_DRAFT_20261017_01/README.md)已实现，复用原N07最终mask_embed输出e（不能用旧pre-hook输入h）、M/F、原渲染/导出；自然轴与导出qids分开。只读实际安装库确认：直接删LoadAnnotations仍会在training=True的pad_mask分支访问masks；新稿沿既有data_samples=None guard保留图像归一化/固定800补边，并重建原metadata规则，不造GT。3项标准库接口测试和独立静态审阅通过；没有真实像素/前向等价、模型加载/来源闭合、HRSID1962或SSDD232人口runner，不能据此宣布跨集可运行或已具数据资格。前述像素、实例真值、历史暴露及teacher训练接触缺口不变。

2026-10-17 输入与来源补记：[U42单图像素/掩膜检查草稿](code/U42_UNBOUND_HRSID_PIXEL_MASK_CHECK_DRAFT_20261017_01/README.md)补上强制Image.load及每个原ann的官方COCO栅格链，不复用旧ImageDraw/打平组件的loader。保留原组件、crowd和失败实例；连续area与离散像素面积只记差值，有限越界坐标只诊断不裁剪/提前拒绝。EXIF缺tag按存储坐标记录未旋转，非1/非法/读取异常拒绝；缺tag不是已验证tag1或语义对齐。三项标准库接口/失败注入测试与独立静态审阅通过，真实解码/栅格分支未运行，没有取得全5604人口资格。

当前teacher的上游初始化来源仍未闭合。普通发布源码中的[native COCO初始化检查](multimodal_data_pilot_20260928/native_coco_comparator_initialization_preparation_20261008/qualify_initializers_cpu_v2.py)第147–171行明确将public_checkpoint_origin_and_training_recipe_verified设为False；其COCO路径、配置名和类别兼容不能证明官方字节身份或从未接触HRSID/SSDD留出。源码合同经SPATIAL24完整FIT741训练、N07严格E24载入并另行DEV评价，但不替代实际资格链的数据/权重绑定核验。[官方MMDetection v3.1.0模型索引](https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.1.0/configs/mask2former/metafile.yml)给出同名mask2former_r50_8xb2-lsj-50e_coco配置及候选权重mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth；尚未与当前实际绑定初始checkpoint的完整摘要比对，不能写成已认证同一权重。未发现训练接触证据不等于证明未接触，字节不等也不自动证明SAR训练。下一环是实际绑定初始化ref与可信官方下载字节/作者配方的对应，当前不宣称未见数据泛化；本补记未读取运行中实际资格链或权重，不改变U40。

## 2026-10-17 后续实际初始化来源核对

第7段原actual0及Root接收完成后、下一科学任务启动前，Root按当前已接受context严格JSON引用，读取FIT、N13资格/报告、N16 GPU报告及原START加载回执；每份JSON均核原严格bytes/SHA，五处native0 E24 teacher引用逐值相同，加载回执的original_initialization.checkpoint指向本地m2f_r50_coco.pth。此为原worker加载证据与metadata链，不是独立重放教师网络/状态；没有打开训练后E24权重、tensor、图像或GT。

Root随后仅流式读取该原初始化文件字节，完整SHA与其已绑定ref相等：176432372字节、SHA256 `41b088b62838c986004f94bab2870395b669ec6bd11b6590369b838b0de06879`；文件未改写、未反序列化。官方端点HTTP200响应Content-Length同值，官方文件名SHA前缀41b088b6与本地一致；从未带Range的GET顺序读取的首4194304字节SHA，与本地同一前缀逐摘要相等。本地整文件MD5字串d5bd8b504b06199b5a7de45627447485与ETag字串相同。[实际脱敏摘要](results/M2F_COCO_INITIALIZER_ENDPOINT_FINGERPRINT_LINKAGE_20261017.json)。这些共同支持与指定公开端点的来源关联，不是全量远端SHA256已取得。

[官方直连全流尝试](results/M2F_OFFICIAL_COCO_REMOTE_BYTES_WITNESS_20261017_01.json)在固定90秒后、实际96.282秒退出，仅取得4MiB，保持FAILED_PUBLIC_REMOTE_FETCH_NO_LOCAL_IDENTITY；局部SHA不充作全文件摘要，不重试扩大预算。HTTP ETag是不透明validator，OSS也不保证所有创建方式的ETag均为内容MD5；本次只报告字串相等，不认证服务端MD5语义。[HTTP规范](https://www.rfc-editor.org/rfc/rfc9110.html#section-8.8.3)、[OSS数据校验说明](https://www.alibabacloud.com/help/en/oss/user-guide/data-verification/)。Last-Modified仅为端点元数据，不证明训练时间或数据范围。

[官方v3.1.0索引](https://github.com/open-mmlab/mmdetection/blob/v3.1.0/configs/mask2former/metafile.yml#L115-L130)、[配置](https://github.com/open-mmlab/mmdetection/blob/v3.1.0/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py)、[README](https://github.com/open-mmlab/mmdetection/blob/v3.1.0/configs/mask2former/README.md)将该候选关联到ImageNet-1K预训练R50/COCO train2017实例模型；这是发布方声明，不是本项目独立重建全部训练谱系。八位文件名校验标识也不等于完整SHA256，参见[MMEngine下载命名规则](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.utils.dl_utils.load_url.html)。本次来源证据有实质增量，但历史TEST查看/选择不被清除，母景独立、JPG/PNG像素等价、全人口真实mask及未见泛化仍未证明。当前两臂配方、数据、顺序和固定50轮端点不变。


## 2026-10-17 U43完整人口检查实现（未执行）

[完整人口核心](code/U43_HRSID_POPULATION_PIXEL_MASK_CHECK_DRAFT_20261017_01/README.md)及[CPU入口/工作体](code/U43_HRSID_POPULATION_CHECK_ENTRY_DRAFT_20261017_01/README.md)已完成并保存：固定原train3642图/11047实例、test1962图/5922实例，全5604/16969逐图调用原U42完整解码/官方polygon栅格；原组件、crowd、空图、失败项和split局部ID均保留，跨split裸ID复用合法。两目录共6项标准库元数据/失败路径行为测试通过，另经一次独立静态互接审。没有导入真实PIL/NumPy/mask或Torch，没有打开原图/GT执行人口检查，尚未实际prepare/bind/launch；私有Root控制正在起草，不因代码或测试存在视为已取得资格。

未来单次CPU任务只核输入兼容与计数，保留5604私有逐图报告；仅全图解码及16969原实例栅格成功才申请原actual0/Root接收。失败/中断不删图、修标、自动重试或改预算。全程无模型、预测、训练或AP，实际使用TEST标注会如实披露；结果也不能证明语义标注对齐、JPG等于PNG像素、母景独立或恢复未接触测试身份。它不改变已锁U40主方法与50轮训练。


2026-10-17 执行准备补记：[U43固定检查计划](U43_HRSID_FULL_PIXEL_MASK_INPUT_CHECK_PLAN_20261017.md)已形成，Root已从原完整encoded/header记录提取5604个原JPEG与两split的严格引用候选，并仅按文件字节固定十二个选定库引用；未重新读取原JPEG/GT、导入这些库或执行人口检查。Root私有prepare/bind/register/intake控制草稿已完成少量纯元数据行为与独立静态互接审，实际绑定与科学执行仍待原训练任务终止后统一安排，不因准备完成授予数据资格。

[U44外部自动实例推理方案](U44_HRSID_FIXED_EXTERNAL_PREDICTION_DESIGN_20261017.md)明确采用小CPU窗口、GPU逐模型轮换、只保留最终RLE；不把内部187图reader改常数冒充1962人口，不建立全人口F/M/L密集缓存。新图无GT模型接口、固定E50端点/效用及强R外部来源仍缺，保持planned/not run；输入检查通过也不证明任何跨集AP或未见数据泛化。
