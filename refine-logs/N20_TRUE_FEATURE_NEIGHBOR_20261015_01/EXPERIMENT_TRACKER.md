# N20 执行状态（2026-10-15）

**阅读顺序：** 本文按时间保留执行记录，当前进展以末尾最近的“当前”段落及[状态总览](../../research_round_20260925_global/N20_REPEAT202_TRAINING_COMPLETE_ENDPOINT_CPU_STATUS_20261016.md)为准；早期任务描述按当时快照理解。

**历史快照（2026-10-16，第三段DEV启动时）：** 第二种子202610150202的固定24轮训练、独立端点CPU及外部产物核验已完成。前两个固定DEV分段[0,47)、[47,94)均实际结束0，并在各自新纯终态门通过后接收产物；累计94图/9400query/十策略的14项来源绑定、捕获文件SHA和原导出顺序已核对。这只是元数据与完整性接收，独立tensor数值核验和全187图统一COCO仍待完成。第三分段[94,141)原GPU185任务37955已启动，活动产物不读取；当前185正常原任务终态/26保留科学失败/211终态，活动任务不计完成。尚无第二种子AP；首201主差值仍+0.197741 AP点，重复效用、正式TEST、跨集、机制和创新结论待证。[状态总览](../../research_round_20260925_global/N20_REPEAT202_TRAINING_COMPLETE_ENDPOINT_CPU_STATUS_20261016.md)。

**历史汇总（2026-10-16，训练运行时快照）：首201完整DEV及独立post/Root资格已完成，主LOCAL68.297126、CENTER67.978791、强R68.099385，主差值+0.197741 AP点。第二种子202同配方24轮正式训练任务原supervisor55247已启动并运行，未读取活动批进度/结束产物。当前181正常/26保留科学失败/207终态；活动任务不计正常完成。尚无重复AP、正式TEST、跨集或创新成功结论。此段为历史快照，当前进展以本文末尾最近的“当前”段落为准。**

- 全741图真实F邻域缓存已通过GPU/独立CPU/Root资格：旧样本3,035,136、unique中心1,786,820、额外位置2,159,732，新增序列化缓存2,360,944,527B。旧中心逐byte一致；CPU未独立重渲染额外F。
- 预锁初始化digest a13802fc0bf4f23a8ab368b11b15a24b60552baadf1ab8264daf242173854a68；LOCAL真实九邻域与CENTER重复中心，两臂5张量/18,851参数、相同四层MLP初态/采样/图序/AdamW/固定24轮。
- 原训练首试因输出父目录缺失实际exit1，失败22保留；第二次因旧当前计数门实际exit1，失败23保留；第三次保存完整E8/E24与17,784次成对更新后，在最后CUDA清理门实际exit1，失败24保留。原失败未重写，没有第四次训练，也未创建原GPU REPORT。旧成功脚本含cuBLAS workspace清理，属于可能诊断方向，未证明是唯一原因。
- 独立CPU恢复：原进程CPU01/两线程/coldCUDA，实际0，726.765秒；原终态纯前缀实际0先于任何结果读取。随后Root核原模型与20绑定、五AdamW状态、E8各5,928/E24各17,784更新、私有图序/RNG/历史/保存轨迹子流、1,482次完整canonical输入读取、Q/DETAILS/导入与全部引用闭包。独立Root恢复资格另存，原pending Q未改。
- E24保存GPU训练端点CE：LOCAL 0.005694717780、CENTER 0.005710665533；CPU两公式最大logit差3.05176e-5、与GPU保存均值CE最大差2.33460e-10（原abs1e-4/rtol0）。这是训练输入端点证据，不能推出AP收益、充分收敛、逐点GPUlogit或整轨迹重放。
- 后续DEV前门已适配恢复资格路径：首GPU owner141、failed24、pins165，四逻辑范围[0,47)、[47,94)、[94,141)、[141,187)。仅保存的E8/E24快照；不消费不存在的原GPU报告。N16旧四分片载体与N17五物理公共PR载体严格分开。新增prior carrier metadata hook已保存、一次source-only审阅未见确定阻断，尚未真实执行。
- 新DEV真实F邻域capture/writer和独立CPU解码器正在普通未绑定稿中接线。保留2048点块、8策略、全C FP64全局排序、K0；复用已核全C公共PR参考，但新选择子集仍fresh PR并核原容差/strictsign。开发草稿和metadata门不替代真实张量资格、全量COCO或AP。
- 下一步：完成必要接口检查，绑定并执行第一个固定DEV范围；原任务终态后先纯RAW前缀，再完整GPU/CPU数值与统一全量评价。
- 主预锁不变：Δ=AP_LOCAL−max(AP_CENTER,68.09938494745545)，E24-atmost主、E8/exact辅助。Δ≤0 STOP；正收益才按既有规则复验，不能事后补参数追分。N16/N17/N19旧STOP与所有失败保留。普通DW信息检验不是GLDet复现或核心创新；没有正式TEST、跨集、TGRS成功结论。

[详细恢复证据与范围](../../research_round_20260925_global/N20_SAVED_E8_E24_CPU_RECOVERY_QUALIFIED_STATUS_20261015.md)。

- **固定DEV首分片已启动**：RAW-only启动前检查actual0（0.228秒，无Torch/科学输入哈希），完整入口AST通过；一次独立source-only ABI审阅READY。原supervisor47836持有唯一新GPU任务，范围[0,47)，无自动重试。真实原栅格F9捕获/2048评分/保存GPUlogit+FP64分数及新CPU解码器已接线；CPU手算2×3邻域与反例拒绝1项actual0（2.006秒），只是合成行为检查。运行中仅同步独立TMP普通副本和此状态，不读取/哈希运行载荷或绑定SOURCE用于发布；本段不代表GPU实际数值完成或AP。

- **CPU后续代码已准备，未执行**：独立前门/数值tail与新F9 decoder/body经一次定向静态ABI审阅READY，语法检查通过；四GPU全部原actual0后才按预先划定八子范围做CPU核验。不能把未来owners145–152或完整COCO写成已执行。[八CPU预锁范围](CPU_EXECUTION_PARTITION_PLANNED_20261015_01.json)、[CPU前门合同](../../research_round_20260925_global/code/N20_FIXEDDEV_FEATURE_ERROR_CPU_FRONT_UNBOUND_CONTRACT_20261015_01.md)、[全187 COCO/post合同](../../research_round_20260925_global/code/N20_FULL187_COCO_FINAL_AND_POST_MINIMAL_WIRING_UNBOUND_20261015_01.md)。
- **继续判定器的有限检查**：纯首201解释器的3项合成行为测试实际0，覆盖较强CENTER优先、Δ=0/0<Δ<1/Δ>=1、undefined、错误seed/端点/单位/强参考及辅助字段拒绝。它只给继续规则，不认证输入、不把未来复验或发表标为成功。
- **具体工程疑点，尚非实测瓶颈**：每个2048点F9回调约18MiB，四head及query重复的CPU拷贝可能增加证据采集成本。文件时间点不足以证明瓶颈或独立FPS。当前冻结任务不改；只有真实资源中止后，才可另锁执行版本检验按图唯一输入证明与逐调用GPU位一致门，保留实际logit、独立CPU两公式、FP64排序、fresh PR和严格sign/RLE。CPU allclose本身不能证明输入逐byte一致，不能悄删旧证据或沿用不再成立的旗标。

- **失败25的隔离修复**：原actual held exit1/reap/birthgone及原source/spec/ad/phase/bootstrap/stderr均已严格锁定；原REPORT不存在。tiny测试以NumPy空INT64/FP64、singleton stride0、0D、真实BytesIO torch.save/load复现相同错误（RED：1test/7errors/2.399s），修复后同测试GREEN（1test/2.102s），保留signed-zero逐byte差异及shape/dtype拒绝。AST验证tail除n20_same_bits外完全同一。empty判相等仍须dtype/shape相同，非空保持UINT8精确比较，无allclose替代或容差放宽。
- **修复版新原任务**：完整入口AST与一次定向装配审阅通过；typed25在首Source hash前核原失败、后闭合原源/日志；pre-HASH RAW-only实际0（0.233秒，无Torch）。新原36736独立输出，从[0,47)重新计算，不采用失败31文件作为完成证据。只新增必要检查器/失败链工程修复，不训练新模型，不实现性能猜想。运行中不为同步读/hash科学载荷或绑定Source。
- **CPU装配错误单独复现与修正**：原CPUfront01错误要求全局历史N19 CPU_SCHEMA等于新N20 schema；保留历史常量的纯header测试先RED，front02仅删该多余global比较后GREEN（1test/0.008s）。它仍是未执行草稿，后续当前failure25链必须适配，旧current24版本不能直接启动。

- **后续当前25链源码**：CPUfront03仅修current header与新DEV父25/+25，历史缓存21/恢复24/训练失败23/原DEV失败24不变；新header fixture保留历史N19常量，真实纯header检查通过，非CPU数值资格。当前[CPU范围补充02](CPU_EXECUTION_PARTITION_PLANNED_20261015_02.json)与[final/post合同02](../../research_round_20260925_global/code/N20_FULL187_COCO_FINAL_AND_POST_MINIMAL_WIRING_UNBOUND_20261015_02.md)替代未执行旧current24版本；仍必须先取得完整四GPU原actual0。
- 新字节回归的GREEN复现需显式选择 `--tail N20_FIXEDDEV_FEATURE_ERROR_GPU_RUN_TAIL_UNBOUND_DRAFT_20261015_02.py`；同测试默认旧tail01用于保留RED复现，不能将历史默认RED误当新修复仍失败。两个版本源码与失败证据均留存；不发布原科学载荷/绑定资格链。

- **首GPU实际0后的推进**：首完整REPORT与MANIFEST、START及原RAW/admission/phase/bootstrap引用已由Root在新pure142+failed25实际0后核验；47份新捕获均逐文件SHA一致，native/R及8新策略的预测各4700条，保存状态仍为pending CPU，未把原REPORT改为qualified。新[核验范围与待执行源码](../../research_round_20260925_global/N20_FIRST_DEV_GPU_ACTUAL0_AND_CPU_PIPELINE_STATUS_20261015.md)列出证据等级。
- **当前八CPU完整入口与最终评价入口**：当前25链CPU entry02 AST与纯header检查actual0；full187 final front/tail/完整entry已装配并AST通过，post后置门一次独立静态READY，均UNBOUND_NOT_EXECUTED，不能计为CPU/AP资格。新增完整public proof_query合成CPU集成检查actual0（1test/2.079秒），覆盖两公式/FP64评分/选点/PR/RLE与篡改拒绝；它不消费真实数据或替代原GPU捕获的数值检查。
- **下一唯一科学任务**：第二GPU范围[47,94)，使用与成功首段相同冻结入口；RAW-only启动前检查actual0，无Torch/首次SHA前即停；原supervisor85580独立输出，无自动重试。四GPU全部原actual0后，按预锁八CPU范围进行数值核验，再统一官方COCO汇总，绝不平均分片AP。

- **第二GPU完整实际0与第三GPU启动**：原supervisor85580 actual0（1721.031秒），新pure143+failed25先actual0（0.430秒），随后读取并核原RAW/ad/spec/entry/phase/bootstrap及REPORT/START/MANIFEST13绑定，47份物理捕获共4,243,201,508B逐SHA相同，十策略各4700完整预测及原image/query/class/S0顺序、canonical profiles、零cleanup/预算/范围/无partial复用。保留原pending-CPU状态，不升级为数值qualified。第三段原supervisor97208已启动、[94,141)、同冻结entry与E8/E24；启动前pure RAW-only检查0.289秒actual0，无Torch/首次SHA前停止。当前唯一科学任务为第三段，不新增训练或融合。

- **与失败25直接相关的CPU字节路径定向核对**：仅检查普通未绑定decoder421f、numeric BODY dbf与CPU tail c7ee三份源码，未发现相同可达的Long→Byte视图路径。decoder的registry为finite/numel描述；BODY仅FP32→INT32同宽比较、非scalar输入及n>0点块，K0跳过；tail的FP32模型状态比较调用完成父GPU_outer_source的n20_same_bits，后续须绑定已成功修复源5b09，不能绑定失败副本360f。动态ref旧helper未展开，未导入/测试/修改科学代码；这只是一次有限静态风险核对，不是实际CPU资格。无具体问题故停止额外检查，继续等待原第三GPU97208。

- **第三GPU完整实际0与第四GPU启动**：原supervisor97208 actual0（1651.609秒），新pure144+failed25先actual0（0.677秒），随后读取并核原RAW/ad/spec/entry/phase/bootstrap及REPORT/START/MANIFEST13绑定；47份物理捕获共3,825,291,460B逐SHA相同，十策略各4700完整预测及原image/query/class/S0顺序、canonical profiles、零cleanup/预算/范围/无partial复用。原报告保留pending-CPU状态。第四段原supervisor21000已启动、[141,187)、46图/4600query，同冻结entry与E8/E24；启动前pure RAW-only检查0.319秒actual0，无Torch/首次SHA前停止。当前唯一科学任务为第四段，不新增训练或融合。全部四GPU完成后仍按预锁八CPU范围逐点核验，再统一全187官方COCO；没有分片AP或提前选型。

- **全部四GPU原actual0、首次CPU启动**：第四原supervisor21000 actual0（1217.781秒）；pure145+failed25先actual0（0.367秒）后才读取本段科学产物，Root逐文件SHA与三产物13绑定/原RAW/ad/spec/entry/phase/bootstrap、46图/4600query/十完整预测及canonical roster/原image-query-class-S0顺序、预算、零cleanup均通过。原失败25及31个partial未复用，训练GPU失败24未升级。首CPU fixed entry02已从独立普通稿复制冻结（159357B、SHA9a8cb27d8ff5180615e08009d35524383fc4f26656c4d7471f10b2f53c831841），沿用已通过AST/header/publicAPI合成检查的字节版本，无新模型代码。首CPU145范围[0,24)、映射原GPU0、四原GPU exact10描述符、预锁八范围计划02与完整Sources/pins已绑定；自己的decoder421f/proof dbf/独立endpoint f6b与旧PR/FP64/五tensorhead绑定，原GPU spec中的CEA2未改。pre-HASH RAW-only启动前检查actual0（0.307秒，无Torch），唯一原CPU supervisor89755已启动：CUDA空、CPU01/两线程、1800秒、RSS16GiB、无自动重试。未把运行状态、静态检查或首批输出当CPU资格；八段全部原actual0核验后才统一全187官方COCO。

- **首CPU实际资源中断，保留失败26**：原supervisor89755在1800.343秒held实际exit1、reaped/birthgone，abort为运行资源/截止门；新pure145+failed26先actual0（0.411秒，无Torch），再读取结果。原目录13份逐图partial、无REPORT，不复用也不计CPU完成。当前正常145/失败26/总终态171；不是模型失效或新AP。
- **重新锁定执行资源分片**：[执行计划03](CPU_EXECUTION_PARTITION_PLANNED_20261015_03.json)将每个原GPU父域切成8份、最多6整图，共32份完整覆盖187图。只减小执行范围，原模型/捕获/双公式/容差/评分/主端点不变；历史八份计划、失败25/训练失败24保持原样。新typed26、CPU和final/post合同适配进行中，未绑定/执行，不自动重试、不复用失败partial。

- **32范围接口接线与首份实际启动**：当前26链entry03（168252B，SHA045e5857166193e14403df8948e0a661cad9373c557bccf84b6f99ebc7d9a150）已冻结并绑定首份[0,6)。32scope metadata合成RED→GREEN、typed26合成、完整entry AST/header提取式检查和数值循环AST不变检查均通过，仅是接线证据；四GPU报告仍历史25，失败26独立保留。pre-HASH RAW-only actual0（0.292秒，无Torch），唯一原supervisor40922已启动，CUDA空/CPU01两线程/1800秒/RSS16GiB/无自动重试。当前145正常/26失败/171终态，0/32CPU合格，不提前计完成或计算AP。final/post entry02及32范围合同已保存，尚未绑定/执行。普通源稿和检查器同步，当前绑定Source/运行载荷不为发布而读取。

- **首32范围CPU实际0并通过Root核验**：[0,6)原supervisor40922 held exit0（1088.890秒），新pure146+failed26先actual0（0.407秒，无Torch）再读原结果。RAW-bound stdout REPORT ref、Source/spec/admission/phase/bootstrap及完整23项绑定核对；6份proof、600自然query、4800 profiles、19225注册members全覆盖；两CPU公式对GPU最大4.57763671875e-5，原1e-4/rtol0与strict sign门不变。十策略各600条预测与原父切片全字段相等，原REPORT仍pending/False。纯metadata读取器手工100query/800profiles及11项拒绝RED→GREEN通过，仅辅助Root读取、不执行新数值forward。
- **第二份[6,12)实际启动**：同冻结entry03/原E8-E24，own146/current26/172终态，RAW-only prehash actual0（0.295秒、无Torch）；唯一原supervisor69024运行中。当前1/32完成，失败26与partial不复用，不平均分片AP；全187 AP仍未执行。运行Source/科学载荷未为本次同步读取/复制/哈希，只使用已核验内存事实及普通独立源码。

- **第二份CPU实际0及第三份启动**：[6,12)原supervisor69024 held exit0（1178.000秒），新pure147+failed26先actual0（0.387秒，无Torch），之后Root核原RAW-bound stdout/Source/spec/admission/phase/bootstrap/23绑定、600query/4800profiles/19802注册members以及十策略各600完整预测原父切片exact。双CPU scout公式对GPU最大3.0517578125e-5，原数值与strict sign/RLE门不变，原REPORT保留pendingFalse。第三份[12,18)同冻结entry03已启动，prehash RAW-only actual0（0.285秒）后唯一原supervisor36183运行；当前2/32、12图/1200query，147正常/26失败/173终态，无N20 AP。
- **跨集元数据独立准备**：已另存并同步[HRSID候选版本实文件说明](../../research_round_20260925_global/HRSID_OFFICIAL_ANNOTATION_INTAKE_20261015_01.md)，Root独立核固定提交JSON3573/9870/215无标注关联；只隔离元数据，没有mask decode/图像/跨集训练/AP或原数据发布，完整版本和测试划分仍未知。

- **第三份CPU实际0及第四份启动**：[12,18)原任务held exit0（871.141秒），pure148+failed26先actual0（0.390秒，无Torch），再核Source/spec/admission/phase/bootstrap、23绑定及RAW-bound stdout报告ref；600query/4800profiles/17116注册members，十策略各600完整预测与父域切片全字段一致。双CPU scout公式对GPU最大2.288818359375e-5，原数值门不变，原REPORT pendingFalse保留。第四份[18,24)同冻结Source启动，RAW-only prehash actual0（0.299秒），唯一原supervisor68897，无自动重试。当前3/32、18图/1800query/14400profiles，正常148/保留失败26/174终态，尚无N20 AP。
- **独立数据准备与条件性机制分析**：[官方PNG目录与TEST](../../research_round_20260925_global/HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)已核完整目录与1962图/5922实例JSON的CRC/SHA/关联；同包TRAIN短响应停止，不混用先前Git候选TRAIN。[渲染F9条件性信息说明](../../research_round_20260925_global/N20_RENDERED_F9_CONDITIONAL_INFORMATION_AND_CONTROL_SCOPE_20261015_01.md)限定四基表示的双线性单元、边缘与FP32条件；无实际位置/秩/效用测量，不改变冻结模型/停止规则。仅独立文档同步，运行载荷与绑定Source未为发布读取。

- **第四份CPU实际0及第五份启动**：[18,24)原supervisor68897 held exit0（835.922秒），pure149+failed26先actual0（0.386秒，无Torch），随后核RAW-bound stdout报告ref、Source/spec/admission/phase/bootstrap及23绑定；600自然query/4800profiles/16376注册members，十策略各600完整预测与父切片全字段一致，双CPU scout公式对GPU最大2.288818359375e-5，原数值门不变。第五份[24,30)同冻结Source，prehash actual0（0.342秒）后唯一原supervisor29083运行，无自动重试。当前4/32、24图/2400query/19200profiles，正常149/保留失败26/175终态；尚无全187 AP。
- **HRSID新独立小分块轮失败保留**：新轮≤256KiB/段，首请求206且CR/总长一致，但实际260154/262144B、6.844秒，EOF后立即STOP，无重试/完整TRAIN/partial复用。Root独立核partial实长/SHA和安全日志，未把它计入N20科学失败或导入当前Source。静态审阅指出网络30秒仅软门、写盘失败时计数可能偏少；本次无写盘错误且实长一致。上一轮7361454B为保守预算扣费而非已知实际总量，此处已在[来源记录](../../research_round_20260925_global/HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)明确。

- **第五份CPU实际0及第六份启动**：[24,30)原supervisor29083 held exit0（767.563秒），pure150+failed26先actual0（0.336秒，无Torch），再核RAW-bound stdout报告ref、Source/spec/admission/phase/bootstrap与23绑定；600query/4800profiles/16271注册members，十策略各600预测与父切片全字段一致，双CPU scout公式对GPU最大3.0517578125e-5，原数值门和REPORT pendingFalse保留。第六份[30,36)同冻结Source，prehash actual0（0.304秒）后唯一原supervisor59461运行，无自动重试。累计5/32、30图/3000query/24000profiles，正常150/保留失败26/176终态，无全187 AP。
- **有限独立研究复审**：既有374匹配实例统计同时有路由余量及固定PR动作能力限制，不包含6未匹配GT/全部自然query；像素有害多于有益与强R提升AP并存，不能排序为唯一AP瓶颈。N20检验最终渲染像素邻域，正负结果仍依预锁Delta和重复门解释；详见[当前状态](../../research_round_20260925_global/N20_FIRST_DEV_GPU_ACTUAL0_AND_CPU_PIPELINE_STATUS_20261015.md)。未追加方法/测试或改变主端点。

- **HRSID同包元数据实核完成**：不同客户端有界轮保留所有失败，第三轮完整首块+第四轮完整七块+第五轮四块取得完整TRAIN，Root独立核每块/deflate/raw/CRC/JSON。TRAIN3642/11047、TEST1962/5922，filename交集0/并集5604且全部PNG目录exact；跨split imageID交1962/annID交5922为数值复用，后续需split命名空间。Git候选共同3572/仅候选1/仅PNG TRAIN70，不混配。无图像/mask/AP或加入当前科学Sources；[详细自写来源记录](../../research_round_20260925_global/HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)包含所有取件失败和实际证据范围。

- **第六份CPU实际0及第七份启动**：[30,36)原supervisor59461 held exit0（633.813秒），pure151+failed26先actual0（0.382秒，无Torch），随后核RAW-bound stdout报告ref、Source/spec/admission/phase/bootstrap及23绑定；600query/4800profiles/15002注册members，十策略各600完整预测与父切片全字段一致，双CPU scout公式对GPU最大3.0517578125e-5，原门和REPORT pendingFalse保留。第七份[36,42)同冻结Source，prehash actual0（0.296秒）后唯一原supervisor95848运行，无自动重试。累计6/32、36图/3600query/28800profiles，正常151/保留失败26/177终态，尚无全187 AP。
- **数据入口与影像范围明确**：更新[数据获取说明](../../docs/DATA_ACCESS.md)/[上游索引](../../docs/upstream_data_index.json)，区分实际同包PNG TRAIN/TEST和Git候选及split ID复用。一次有界本地目录盘点仅358个.jpg文件、66018309B，未打开/hash文件，不能认证完整原数据或同PNG像素；不据文件名更换数据。仅普通文档/索引同步，未读取运行科学载荷或绑定Source用于发布。

- **第七份CPU实际0及第八份启动**：[36,42)原supervisor95848 held exit0（645.547秒），pure152+failed26先actual0（0.412秒，无Torch），随后核RAW-bound stdout报告ref、Source/spec/admission/phase/bootstrap/23绑定；600query/4800profiles/15008注册members，十策略各600预测与父切片全字段一致，双CPU scout公式对GPU最大3.0517578125e-5，原门及REPORT pendingFalse保留。第八份[42,47)5图同冻结Source，prehash actual0（0.350秒）后唯一原supervisor66496运行，无自动重试。累计7/32、42图/4200query/33600profiles，正常152/保留失败26/178终态，尚无全187 AP。
- **独立元数据与原图编码文件准备**：[F9几何说明](../../research_round_20260925_global/N20_RENDERED_F9_CONDITIONAL_INFORMATION_AND_CONTROL_SCOPE_20261015_01.md)新增原F空间尺寸未在已审ABI显式保存的有限核对，不拿图像H,W冒充。[官方PNG两文件字节试点](../../research_round_20260925_global/HRSID_OFFICIAL_PNG_ENCODED_IMAGE_BYTES_PILOT_20261015_01.md)实际取得两编码PNG并Root独立核SHA/ZIP与PNG全部chunk CRC/IHDR；header重复CSP误拒失败保留，RED→GREEN最小修复后fresh02成功。未解码IDAT/像素或mask、未评价/加入N20Sources，尚非完整跨集数据。


## 第八份独立CPU通过，第九份已启动

[42,47)原supervisor66496 held实际退出0（1337.344秒），原句柄回收且出生身份消失，error/abort均无。Root先执行新pure153+failed26实际0（0.343秒，无科学库导入），随后核验RAW-bound stdout REPORT引用、Source/spec/admission/phase/bootstrap与完整23项绑定。5图/500自然query/4000策略记录、20065注册成员完整覆盖；双CPU scout公式分别对实际GPU最大3.0517578125e-5，满足原绝对1e-4/rtol0门；十策略各500完整预测与原父域切片逐字段相同。原REPORT仍为pending/qualified=False，外部Root仅对该子范围核验通过，不等于全DEV/AP资格。

第九份[47,53)切入第二个GPU父域[47,94)，沿用冻结Source/模型/主端点。RAW-only prehash实际0（0.307秒，无科学库导入）后唯一原supervisor63376启动，无自动重试。累计8/32份、47图/4700自然query/37600策略记录，正常153/保留科学失败26/179终态。全187图统一官方COCO尚未执行，无N20涨点、创新或TGRS成功结论。


## 第九份独立CPU通过，第十份已启动

[47,53)原supervisor63376 held实际退出0（1189.047秒），原句柄回收、出生身份消失且无error/abort。Root先执行新pure154+failed26实际0（0.325秒，无科学库导入），随后恢复固定原函数并以单次退出的Root元数据流程完成核验；Source/spec/admission/phase/bootstrap、RAW-bound stdout REPORT及23项绑定均通过。6图/600自然query/4800策略记录、20166注册成员完整覆盖，双CPU scout公式分别对实际GPU最大4.57763671875e-5，满足原绝对1e-4/rtol0门；十策略各600完整预测与第二GPU父域切片逐字段一致。原REPORT保持pending/qualified=False，仅该子范围获得外部Root核验，没有全DEV/AP资格。

第十份[53,59)沿用冻结Source、模型、范围方案和主端点。RAW-only prehash实际0（0.333秒，无科学库导入）后唯一原supervisor29173启动，无自动重试。累计9/32份、53图/5300自然query/42400策略记录，正常154/保留科学失败26/180终态。辅助长寿命Node进程的内存不足退出未中断独立科学任务，不计为科学RAW失败；已恢复的原核验函数与纯门顺序经过实际范围核验，后续采用单次进程并限制内存，不改科学配方。统一全187官方COCO尚未执行，没有N20涨点、创新或TGRS成功结论。


## 第十份独立CPU通过，第十一份已启动

[53,59)原supervisor29173 held实际退出0（845.047秒），原句柄回收、出生身份消失且无error/abort。Root先执行新pure155+failed26实际0（0.389秒，无科学库导入），再以原固定核验函数的单次进程流程完成RAW-bound stdout REPORT、Source/spec/admission/phase/bootstrap和23项绑定核验。6图/600自然query/4800策略记录、16968注册成员完整覆盖；双CPU scout公式分别对实际GPU最大3.0517578125e-5，满足原绝对1e-4/rtol0门；十策略各600完整预测与原第二GPU父域切片逐字段一致。原REPORT仍为pending/qualified=False，仅该子范围通过外部Root核验，没有全DEV/AP资格。

第十一份[59,65)沿用相同冻结Source、模型、数值规则和主端点，RAW-only prehash实际0（0.315秒，无科学库导入）后唯一原supervisor59589启动，无自动重试。累计10/32份、59图/5900自然query/47200策略记录，正常155/保留科学失败26/181终态。统一全187图官方COCO尚未执行；没有N20涨点、创新或TGRS成功结论。


## 第十一份独立CPU通过，第十二份已启动

[59,65)原supervisor59589 held实际退出0（894.422秒），原句柄回收、出生身份消失且无error/abort。Root先执行新pure156+failed26实际0（0.466秒，无科学库导入），再用原固定核验函数完成RAW-bound stdout REPORT、Source/spec/admission/phase/bootstrap与23项绑定核验。6图/600自然query/4800策略记录、17309注册成员完整覆盖；双CPU scout公式分别对实际GPU最大3.0517578125e-5，满足原绝对1e-4/rtol0门；十策略各600完整预测与第二GPU父域原切片逐字段相同。原REPORT保持pending/qualified=False；该子范围的外部Root通过不等于全DEV或AP资格。

第十二份[65,71)同冻结Source、模型、数值和主端点，RAW-only prehash实际0（0.349秒，无科学库导入）后唯一原supervisor84549启动，无自动重试。累计11/32份、65图/6500自然query/52000策略记录，正常156/保留科学失败26/182终态。统一全187图官方COCO尚未执行，没有N20涨点、创新或TGRS成功结论。


## 第十二份独立CPU通过，第十三份已启动

[65,71)原supervisor84549 held实际退出0（929.219秒），原句柄回收、出生身份消失且无error/abort。Root先执行新pure157+failed26实际0（0.415秒，无科学库导入），再用原固定核验函数完成RAW-bound stdout REPORT、Source/spec/admission/phase/bootstrap及23项绑定核验。6图/600自然query/4800策略记录、17343注册成员完整覆盖；双CPU scout公式分别对实际GPU最大3.0517578125e-5，满足原绝对1e-4/rtol0门；十策略各600完整预测与第二GPU父域原切片逐字段相同。原REPORT保持pending/qualified=False，仅该子范围通过外部Root核验，不等于全DEV/AP资格。

第十三份[71,77)同冻结Source、模型、数值与主端点，RAW-only prehash实际0（0.330秒，无科学库导入）后唯一原supervisor11499启动，无自动重试。累计12/32份、71图/7100自然query/56800策略记录，正常157/保留科学失败26/183终态。统一全187图官方COCO尚未执行；没有N20涨点、创新或TGRS成功结论。


## 第十三份独立CPU通过，第十四份已启动

[71,77)原supervisor11499 held实际退出0（1042.016秒），原句柄回收、出生身份消失且无error/abort。Root先执行新pure158+failed26实际0（0.366秒，无科学库导入），再用原固定核验函数完成RAW-bound stdout REPORT、Source/spec/admission/phase/bootstrap及23项绑定核验。6图/600自然query/4800策略记录、18239注册成员完整覆盖；双CPU scout公式分别对实际GPU最大2.288818359375e-5，满足原绝对1e-4/rtol0门；十策略各600完整预测与第二GPU父域原切片逐字段相同。原REPORT保持pending/qualified=False，仅该子范围通过外部Root核验，不等于全DEV/AP资格。

第十四份[77,83)同冻结Source、模型、数值与主端点，RAW-only prehash实际0（0.295秒，无科学库导入）后唯一原supervisor59486启动，无自动重试。累计13/32份、77图/7700自然query/61600策略记录，正常158/保留科学失败26/184终态。统一全187图官方COCO尚未执行；没有N20涨点、创新或TGRS成功结论。


## 第十四份独立CPU通过，第十五份已启动

[77,83)原supervisor59486 held实际退出0（1547.109秒），原句柄回收、出生身份消失且无error/abort。Root先执行新pure159+failed26实际0（0.428秒，无科学库导入），再用原固定核验函数完成RAW-bound stdout REPORT、Source/spec/admission/phase/bootstrap及23项绑定核验。6图/600自然query/4800策略记录、23693注册成员完整覆盖；双CPU scout公式分别对实际GPU最大1.9073486328125e-5，满足原绝对1e-4/rtol0门；十策略各600完整预测与第二GPU父域原切片逐字段相同。原REPORT保持pending/qualified=False，仅该子范围通过外部Root核验，不等于全DEV/AP资格。

第十五份[83,89)同冻结Source、模型、数值与主端点，RAW-only prehash实际0（0.339秒，无科学库导入）后唯一原supervisor31915启动，无自动重试。累计14/32份、83图/8300自然query/66400策略记录，正常159/保留科学失败26/185终态。统一全187图官方COCO尚未执行；没有N20涨点、创新或TGRS成功结论。


## 第十五份CPU原任务完成及后续执行（2026-10-15）

第十五份范围[83,89)原supervisor31915实际退出0，1643.422秒；新pure160+failed26实际0先于结果/来源读取，随后Root一次限定范围资格检查通过。六图/600自然query/4800策略记录、25,498个绑定成员；两套CPU公式对原GPU最大误差2.288818359375e-5，十份完整预测数组与父域对应切片精确一致。原REPORT的pending/qualified=False保留，Root资格独立记录。

累计15/32份，连续[0,89)，89图/8900自然query/71200策略记录；160正常/26保留科学失败/186终态。下一份[89,94)已按原预算、原选择/数值规则真实启动，原supervisor97571，无自动重试。尚未开展完整187图统一官方COCO，不能报告N20全量AP、重复种子、跨集或创新成功。


## 第十六份CPU原任务完成及下一范围（2026-10-15）

第十六份[89,94)原supervisor97571实际退出0，957.437秒；新pure161+failed26实际0先于结果读取，Root随后限定范围资格通过。五图/500自然query/4000策略记录、13,765个绑定成员；两套CPU公式对实际GPU最大误差3.0517578125e-5，十份完整预测数组与父域对应切片精确一致。原pending REPORT不改，Root资格独立记录。

累计16/32份、连续[0,94)、94图/9400自然query/75200策略记录；161正常/26保留科学失败/187终态。第十七份[94,100)已按原数值、选择规则及预算启动，原supervisor42186，无自动重试。完整187图统一COCO尚未执行，没有N20全量AP、重复或跨集成功结论。旧自主网络的来源与元数据补查是独立工作，不计入此科学终态数，也没有新增模型训练。


## 第十七份CPU原任务完成及后续执行（2026-10-16）

第十七份[94,100)原supervisor42186实际退出0，原任务计时1441.203秒；新pure162+failed26实际0先于结果及来源读取，Root随后限定范围资格通过。六图/600自然query/4800策略记录、20,172个绑定成员；两套CPU公式对实际GPU最大误差1.9073486328125e-5，十份完整预测数组与父域对应切片精确一致。原REPORT的pending/qualified=False保留，Root资格独立记录。

累计17/32份、连续[0,100)、100图/10000自然query/80000策略记录；162正常/26保留科学失败/188终态。第十八份[100,106)已按原数值/选择规则及资源预算启动，原supervisor27037，无自动重试。完整187图统一COCO尚未执行，没有N20全量AP、重复种子或跨集成功结论。阶段分片数量不代表论文完成百分比，也不是分片AP的平均。


## 第十八份CPU原任务完成及第十九份启动（2026-10-16）

第十八份[100,106)原supervisor27037实际退出0，原任务计时1322.891秒；新pure163+failed26实际0先于结果及来源读取，Root随后限定范围资格通过。六图/600自然query/4800策略记录、19,521个绑定成员；两套CPU公式对实际GPU最大误差3.0517578125e-5，十份完整预测数组与父域对应切片精确一致。原REPORT的pending/qualified=False保留，Root资格独立记录。

累计18/32份、连续[0,106)、106图/10600自然query/84800策略记录；163正常/26保留科学失败/189终态。第十九份[106,112)已按原数值、选择规则及资源预算启动，原supervisor34360，无自动重试。完整187图统一COCO尚未执行，没有N20全量AP、重复种子或跨集成功结论。分片数量不代表论文完成百分比，也不能用分片AP平均替代全量评价。


## 第十九份CPU原任务完成及第二十份启动（2026-10-16）

第十九份[106,112)原supervisor34360实际退出0，原任务计时1121.157秒；新pure164+failed26实际0先于结果及来源读取，Root随后限定范围资格通过。六图/600自然query/4800策略记录、18,782个绑定成员；两套CPU公式对实际GPU最大误差3.0517578125e-5，十份完整预测数组与父域对应切片精确一致。原REPORT的pending/qualified=False保留，Root资格独立记录。

累计19/32份、连续[0,112)、112图/11200自然query/89600策略记录；164正常/26保留科学失败/190终态。第二十份[112,118)已按原数值、选择规则及资源预算启动，原supervisor69634，无自动重试。完整187图统一COCO尚未执行，没有N20全量AP、重复种子或跨集成功结论。分片数量不代表论文完成百分比，也不能用分片AP平均替代全量评价。


## 第二十份CPU原任务完成及第二十一份启动（2026-10-16）

第二十份[112,118)原supervisor69634实际退出0，原任务计时992.0秒；新pure165+failed26实际0先于结果及来源读取，Root随后限定范围资格通过。六图/600自然query/4800策略记录、17,516个绑定成员；两套CPU公式对实际GPU最大误差3.0517578125e-5，十份完整预测数组与父域对应切片精确一致。原REPORT的pending/qualified=False保留，Root资格独立记录。

累计20/32份、连续[0,118)、118图/11800自然query/94400策略记录；165正常/26保留科学失败/191终态。第二十一份[118,124)已按原数值、选择规则及资源预算启动，原supervisor35192，无自动重试。完整187图统一COCO尚未执行，没有N20全量AP、重复种子或跨集成功结论。分片数量不代表论文完成百分比，也不能用分片AP平均替代全量评价。


## 第二十一份CPU原任务完成及第二十二份启动（2026-10-16）

第二十一份[118,124)原supervisor35192实际退出0，原任务计时759.656秒；新pure166+failed26实际0先于结果及来源读取，Root随后限定范围资格通过。六图/600自然query/4800策略记录、15,461个绑定成员；两套CPU公式对实际GPU最大误差3.814697265625e-5，十份完整预测数组与父域对应切片精确一致。原REPORT的pending/qualified=False保留，Root资格独立记录。

累计21/32份、连续[0,124)、124图/12400自然query/99200策略记录；166正常/26保留科学失败/192终态。第二十二份[124,130)已按原数值、选择规则及资源预算启动，原supervisor24228，无自动重试。完整187图统一COCO尚未执行，没有N20全量AP、重复种子或跨集成功结论。分片数量不代表论文完成百分比，也不能用分片AP平均替代全量评价。


## 第二十二份CPU原任务完成及第二十三份启动（2026-10-16）

第二十二份[124,130)原supervisor24228实际退出0，原任务计时1186.266秒；新pure167+failed26实际0先于结果及来源读取，Root随后限定范围资格通过。六图/600自然query/4800策略记录、20,006个绑定成员；两套CPU公式对实际GPU最大误差6.103515625e-5，十份完整预测数组与父域对应切片精确一致。原REPORT的pending/qualified=False保留，Root资格独立记录。

累计22/32份、连续[0,130)、130图/13000自然query/104000策略记录；167正常/26保留科学失败/193终态。第二十三份[130,136)已按原数值、选择规则及资源预算启动，原supervisor55586，无自动重试。完整187图统一COCO尚未执行，没有N20全量AP、重复种子或跨集成功结论。分片数量不代表论文完成百分比，也不能用分片AP平均替代全量评价。


## 第二十三份CPU原任务完成及第二十四份启动（2026-10-16）

第二十三份[130,136)原supervisor55586实际退出0，原任务计时1183.203秒；新pure168+failed26实际0先于结果及来源读取，Root随后限定范围资格通过。六图/600自然query/4800策略记录、19,976个绑定成员；两套CPU公式对实际GPU最大误差4.57763671875e-5，十份完整预测数组与父域对应切片精确一致。原REPORT的pending/qualified=False保留，Root资格独立记录。

累计23/32份、连续[0,136)、136图/13600自然query/108800策略记录；168正常/26保留科学失败/194终态。第二十四份[136,141)已按原数值、选择规则及资源预算启动，原supervisor51614，无自动重试。完整187图统一COCO尚未执行，没有N20全量AP、重复种子或跨集成功结论。分片数量不代表论文完成百分比，也不能用分片AP平均替代全量评价。


## 第二十四份CPU原任务完成及第二十五份启动（2026-10-16）

第二十四份[136,141)原supervisor51614实际退出0，原任务计时743.469秒；新pure169+failed26实际0先于结果及来源读取，Root随后限定范围资格通过。五图/500自然query/4000策略记录、13,475个绑定成员；两套CPU公式对实际GPU最大误差3.0517578125e-5，十份完整预测数组与父域对应切片精确一致。原REPORT的pending/qualified=False保留，Root资格独立记录。

累计24/32份、连续[0,141)、141图/14100自然query/112800策略记录；169正常/26保留科学失败/195终态。第二十五份[141,147)已按原数值、选择规则及资源预算启动，原supervisor24209，无自动重试。完整187图统一COCO尚未执行，没有N20全量AP、重复种子或跨集成功结论。分片数量不代表论文完成百分比，也不能用分片AP平均替代全量评价。


## 第二十五份CPU原任务完成及第二十六份启动（2026-10-16）

第二十五份[141,147)原supervisor24209实际退出0，原任务计时1025.437秒；新pure170+failed26实际0先于结果及来源读取，Root随后限定范围资格通过。六图/600自然query/4800策略记录、18,361个绑定成员；两套CPU公式对实际GPU最大误差4.57763671875e-5，十份完整预测数组与父域对应切片精确一致。原REPORT的pending/qualified=False保留，Root资格独立记录。

累计25/32份、连续[0,147)、147图/14700自然query/117600策略记录；170正常/26保留科学失败/196终态。第二十六份[147,153)已按原数值、选择规则及资源预算启动，原supervisor14723，无自动重试。完整187图统一COCO尚未执行，没有N20全量AP、重复种子或跨集成功结论。分片数量不代表论文完成百分比，也不能用分片AP平均替代全量评价。


## 第二十六份CPU原任务完成及第二十七份启动（2026-10-16）

第二十六份[147,153)原supervisor14723实际退出0，原任务计时943.375秒；新pure171+failed26实际0先于结果及来源读取，Root随后限定范围资格通过。六图/600自然query/4800策略记录、17,786个绑定成员；两套CPU公式对实际GPU最大误差3.0517578125e-5，十份完整预测数组与父域对应切片精确一致。原REPORT的pending/qualified=False保留，Root资格独立记录。

累计26/32份、连续[0,153)、153图/15300自然query/122400策略记录；171正常/26保留科学失败/197终态。第二十七份[153,159)已按原数值、选择规则及资源预算启动，原supervisor79562，无自动重试。完整187图统一COCO尚未执行，没有N20全量AP、重复种子或跨集成功结论。分片数量不代表论文完成百分比，也不能用分片AP平均替代全量评价。


## 第二十七份CPU原任务完成及第二十八份启动（2026-10-16）

第二十七份[153,159)原supervisor79562实际退出0，原任务计时778.860秒；新pure172+failed26实际0先于结果及来源读取，Root随后限定范围资格通过。六图/600自然query/4800策略记录、16,041个绑定成员；两套CPU公式对实际GPU最大误差3.814697265625e-5，十份完整预测数组与父域对应切片精确一致。原REPORT的pending/qualified=False保留，Root资格独立记录。

累计27/32份、连续[0,159)、159图/15900自然query/127200策略记录；172正常/26保留科学失败/198终态。第二十八份[159,165)已按原数值、选择规则及资源预算启动，原supervisor54658，无自动重试。完整187图统一COCO尚未执行，没有N20全量AP、重复种子或跨集成功结论。分片数量不代表论文完成百分比，也不能用分片AP平均替代全量评价。


## 第二十八份CPU原任务完成及第二十九份启动（2026-10-16）

第二十八份[159,165)原supervisor54658实际退出0，原任务计时907.359秒；新pure173+failed26实际0先于结果及来源读取，Root随后限定范围资格通过。六图/600自然query/4800策略记录、18,201个绑定成员；两套CPU公式对实际GPU最大误差3.0517578125e-5，十份完整预测数组与父域对应切片精确一致。原REPORT的pending/qualified=False保留，Root资格独立记录。

累计28/32份、连续[0,165)、165图/16500自然query/132000策略记录；173正常/26保留科学失败/199终态。第二十九份[165,171)已按原数值、选择规则及资源预算启动，原supervisor29686，无自动重试。完整187图统一COCO尚未执行，没有N20全量AP、重复种子或跨集成功结论。分片数量不代表论文完成百分比，也不能用分片AP平均替代全量评价。


## 第二十九份CPU原任务完成及第三十份启动（2026-10-16）

第二十九份[165,171)原supervisor29686实际退出0，原任务计时834.844秒；新pure174+failed26实际0先于结果及来源读取，Root随后限定范围资格通过。六图/600自然query/4800策略记录、16,987个绑定成员；两套CPU公式对实际GPU最大误差6.103515625e-5，十份完整预测数组与父域对应切片精确一致。原REPORT的pending/qualified=False保留，Root资格独立记录。

累计29/32份、连续[0,171)、171图/17100自然query/136800策略记录；174正常/26保留科学失败/200终态。第三十份[171,177)已按原数值、选择规则及资源预算启动，原supervisor95786，无自动重试。完整187图统一COCO尚未执行，没有N20全量AP、重复种子或跨集成功结论。分片数量不代表论文完成百分比，也不能用分片AP平均替代全量评价。


## 第三十份CPU原任务完成及第三十一份启动（2026-10-16）

第三十份[171,177)原supervisor95786实际退出0，原任务计时693.563秒；新pure175+failed26实际0先于结果及来源读取，Root随后限定范围资格通过。六图/600自然query/4800策略记录、15,114个绑定成员；两套CPU公式对实际GPU最大误差4.57763671875e-5，十份完整预测数组与父域对应切片精确一致。原REPORT的pending/qualified=False保留，Root资格独立记录。

累计30/32份、连续[0,177)、177图/17700自然query/141600策略记录；175正常/26保留科学失败/201终态。第三十一份[177,183)已按原数值、选择规则及资源预算启动，原supervisor5759，无自动重试。完整187图统一COCO尚未执行，没有N20全量AP、重复种子或跨集成功结论。分片数量不代表论文完成百分比，也不能用分片AP平均替代全量评价。


## 第三十一份CPU原任务完成及第三十二份启动（2026-10-16）

第三十一份[177,183)原supervisor5759实际退出0，原任务计时647.656秒；新pure176+failed26实际0先于结果及来源读取，Root随后限定范围资格通过。六图/600自然query/4800策略记录、15,048个绑定成员；两套CPU公式对实际GPU最大误差3.814697265625e-5，十份完整预测数组与父域对应切片精确一致。原REPORT的pending/qualified=False保留，Root资格独立记录。

累计31/32份、连续[0,183)、183图/18300自然query/146400策略记录；176正常/26保留科学失败/202终态。第三十二份[183,187)已按原数值、选择规则及资源预算启动，原supervisor67569，无自动重试。完整187图统一COCO尚未执行，没有N20全量AP、重复种子或跨集成功结论。分片数量不代表论文完成百分比，也不能用分片AP平均替代全量评价。


## 第三十二份CPU原任务完成及全187图统一评价启动（2026-10-16）

第三十二份[183,187)原supervisor67569实际退出0，原数值用时671.985秒；新pure177+failed26实际0后才进行结果和来源读取，Root核验该范围通过。四图/400自然query/3200策略记录，10,067个绑定成员；双CPU公式对实际GPU最大误差4.57763671875e-5，十策略完整预测均与父域对应切片精确一致。原REPORT的pending/qualified=False保留，外部Root资格另行记录。

累计32/32份、连续[0,187)、187图/18700自然query/149600策略记录；177正常/26保留科学失败/203终态。全187图统一COCO原supervisor66048已按固定十策略、原GT及评分规则启动，无自动重试。最终入口使用已冻结的32分片完整源码及首201种子主结果解释器，未改训练、选点或评分规则。统一AP及独立post-actual/Root资格仍未完成，不能从分片完成宣称涨点、创新或TGRS就绪，也不平均分片AP。


## 首201全量AP资格完成与第二种子决策（2026-10-16）

统一COCO原supervisor66048实际0（485.359秒），Root新pure178实际0后绑定独立post178；post原supervisor51900实际0（443.015秒），新pure179实际0后Root外部资格实际0。32/32 CPU与四GPU、187图/380GT/18700自然query、十策略完整预测和官方全量COCO均闭合，native与强R的全部预测及12统计严格保持。原Q及post的pending/qualified=False不改，资格另记。当前179正常/26保留科学失败/205终态。

LOCAL 68.297126、同预算CENTER 67.978791、冻结强参考68.099385；主差值LOCAL−max(CENTER,强参考)=+0.197741 AP点，相对CENTER为+0.318335点。固定E24-atmost为唯一主结果；E8/exact仅补充。LOCAL的AP75与AR100低于强R，不能宣称所有定位/召回指标改善或误差机制已证实。下一步仅预锁202610150202同配方重复，尚未启动；只有两seed均高于各自CENTER且平均主差值≥1点才允许203。原训练GPU清理失败24与独立恢复保留，不能称原GPU训练actual0。正式TEST、跨集、独立重复、部署耗时及创新成立仍未完成。

完整结果与范围：[首201结果与重复规则](../../research_round_20260925_global/N20_FIRST201_FULLDEV_QUALIFIED_RESULTS_AND_REPEAT_DECISION_20261016.md)。


## 第二种子初始化及合成清理探针（2026-10-16）

CPU179源码初始化实际0，新pure180实际0后Root核验；GPU180合成清理探针实际0，新pure181实际0后Root核验。当前181正常/26保留科学失败/207终态。两项都不是SAR训练或AP重复。新初始化仅seed202、配对相同独立存储、零空间权重；合成两臂8步/4096点、18,851参数，执行新tail相同清理片段后allocated/reserved归零。保留原failure24，尚未证明旧失败唯一根因。SAR第二种子训练尚未启动。[记录与适用范围](../../research_round_20260925_global/N20_REPEAT202_INITIALIZATION_AND_SYNTHETIC_CLEANUP_STATUS_20261016.md)。


## 第二种子正式训练任务启动（2026-10-16）

原supervisor55247已按已冻结新入口及独立新spec/admission启动，无自动重试。181正常/26失败/207终态保持，活动任务不添加到正常链。未读取活动源码、资格载荷或批训练产物用于发布；尚不能说更新次数、E8/E24或收尾已完成。沿用原FIT741、采样/监督/顺序、五张量18,851参数、固定24轮与原E24-atmost主评价，只改变配对下游初始化202610150202。原初态CPU179与合成清理GPU180已合格；正式训练仍待实际终态及独立端点核验。三个合成metadata测试由RED转GREEN，解决新初始化/探针文件没有旧expected计数字段的接口差异，未改冻结原件或放松旧F/P/cache门。


## 第二种子固定DEV入口准备（未执行）

入口准备时CPU182原97964运行；其后实际退出0并完成新纯183终态门，外部Root复核仍在进行。新DEV前门和尾部保存，metadata seed准入5测试/29场景通过，尚无第二种子AP或完整DEV数值资格。未来装配以原CPU实际0、新纯183实际0及外部Root端点资格为前提；原预算、端点与停止规则不变。[代码与范围](../../research_round_20260925_global/N20_REPEAT202_TRAINING_COMPLETE_ENDPOINT_CPU_STATUS_20261016.md)。


## 第二种子CPU32独立核验入口草稿

CPU草稿准备时首GPU183原63518仍在运行，随后已正常结束并完成首段接收。CPU前门/尾部保留32固定范围与原两公式；静态接口审阅通过，编号回归3测试/12场景、执行锁回归3测试/49场景通过。首次锁回归框架漏注入JSON已修正，未改方法。四GPU真实完成、当前执行锁与全部CPU数值资格仍待执行；没有第二种子AP。[源码及范围](../../research_round_20260925_global/N20_REPEAT202_TRAINING_COMPLETE_ENDPOINT_CPU_STATUS_20261016.md)。


## 第二种子首DEV接收与第二分段启动（历史）

第二种子202610150202的固定24轮训练、独立端点CPU及外部产物核验已完成。首个DEV分段[0,47)原GPU183已实际结束0（1409.078秒），新纯终态184先于产物读取通过；47图/4700query/十策略、14项来源绑定及47份捕获文件的3152238385字节流式SHA和原导出顺序已核对。这只是元数据与完整性接收，独立tensor数值核验和全187图统一COCO仍待完成。第二分段[47,94)原GPU184任务7853已启动，活动产物不读取；当前184正常原任务终态/26保留科学失败/210终态，活动任务不计完成。尚无第二种子AP；首201主差值仍+0.197741 AP点，重复效用、正式TEST、跨集、机制和创新结论待证。

[首段普通核验摘要](REPEAT202_FIRSTDEV_METADATA_SUMMARY_20261016.json)；[结果范围与核验边界](../../research_round_20260925_global/N20_REPEAT202_TRAINING_COMPLETE_ENDPOINT_CPU_STATUS_20261016.md)。


## 第二种子两个DEV分段接收完成，第三分段已启动（历史）

第二种子202610150202的固定24轮训练、独立端点CPU及外部产物核验已完成。前两个固定DEV分段[0,47)、[47,94)均实际结束0，并在各自新纯终态门通过后接收产物；累计94图/9400query/十策略的14项来源绑定、捕获文件SHA和原导出顺序已核对。这只是元数据与完整性接收，独立tensor数值核验和全187图统一COCO仍待完成。第三分段[94,141)原GPU185任务37955已启动，活动产物不读取；当前185正常原任务终态/26保留科学失败/211终态，活动任务不计完成。尚无第二种子AP；首201主差值仍+0.197741 AP点，重复效用、正式TEST、跨集、机制和创新结论待证。

[第二段普通核验摘要](REPEAT202_SECONDDEV_METADATA_SUMMARY_20261016.json)；[范围、接续程序修正与未来CPU装配边界](../../research_round_20260925_global/N20_REPEAT202_TRAINING_COMPLETE_ENDPOINT_CPU_STATUS_20261016.md)。


## UNet++模块内部改造补充核对

按用户补充重新核查模块定义与实际forward：复合残差CBAM后加SimAM、门控DW/PW细节上采样、二项式模糊下采样均有源码依据。此前偏重连接路径的描述不完整，已修正结构审阅的范围。同时静态定位full中未使用BlurPool结果、预BN输出旁路和DSConv批次偏移抵消；尚未执行张量回归或改原源码，不推断全部历史权重受影响。当前N20是独立读出复验，非这些UNet++完整网络的验证，实验配方保持。

[源码行号、文件身份与证据边界](../../research_round_20260925_global/UNETPP_MODULE_INTERNAL_CHANGES_SOURCE_CHECK_20261016.md)。


## 第二种子前三个DEV分段接收完成，最后分段运行（历史）

第二种子202610150202的固定24轮训练和独立端点核验已完成。前三个固定DEV范围[0,47)、[47,94)、[94,141)均实际退出0，并在新的纯终态前置检查通过后接收产物；累计141图/14100 query/十策略的来源绑定、捕获文件SHA及原导出顺序已核对。这是元数据与完整性接收，独立tensor数值资格和全187图统一COCO仍未完成。最后范围[141,187)的原GPU186任务30806已启动，活动任务不计完成。当前186个正常原任务终态、26个保留科学失败、合计212；尚无第二种子AP。首201主差值仍为+0.197741 AP点，重复效用、正式TEST、跨集、机制和创新结论待证。

第三段47图/4700 query的47份捕获共3,825,739,739字节逐SHA通过；三段累计11,221,049,793字节。原报告的pending状态保留。后续CPU32入口尚未实际装配；完整AP/post未绑定草稿及合成测试已保存，不能替代真实数值或AP资格。

[第三段普通核验摘要](REPEAT202_THIRDDEV_METADATA_SUMMARY_20261016.json)；[当前范围及草稿导航](../../research_round_20260925_global/N20_REPEAT202_TRAINING_COMPLETE_ENDPOINT_CPU_STATUS_20261016.md)。


## 历史：全部DEV分段接收完成，首个CPU32数值复核已启动

第二种子202610150202的固定24轮训练与独立端点核验已完成，四个DEV分段均实际退出0。每段均先通过新的纯终态前置检查，再接收产物；全部187图/18700 query/十策略的14项来源绑定、187份捕获SHA及原导出顺序已核对。这只是元数据与完整性接收，tensor数值资格和全187图统一COCO尚未完成。当前CPU32执行计划已锁定，首个[0,6)复核任务187、原supervisor54707已启动，0/32完成，不读取活动产物。当前187个正常原任务终态、26个保留科学失败、合计213，活动CPU不计完成。尚无第二种子AP；首201主差值仍+0.197741 AP点，重复效用、正式TEST、跨集、机制与创新结论待证。

最后46图/4600 query的46份捕获共2,494,559,828字节逐SHA通过，四段累计13,715,609,621字节。首CPU前已经锁定四GPU和32CPU显式身份及原32范围；CPU入口实际装配但数值未完成。公开摘要在CPU绑定前保存，未为同步读取活动Source或载荷。

[全部四段普通核验摘要](REPEAT202_ALLFOURDEV_METADATA_SUMMARY_20261016.json)；[状态与接续边界](../../research_round_20260925_global/N20_REPEAT202_TRAINING_COMPLETE_ENDPOINT_CPU_STATUS_20261016.md)；[原生网格与渲染F9先例补查](../../research_round_20260925_global/N20_NATIVE_GRID_VS_RENDERED_F9_DIRECT_PRIOR_REVIEW_20261016_01.md)。


## 历史：首个CPU32批次接收完成，第二批运行

第二种子202610150202的固定24轮训练、独立端点核验和四段DEV元数据接收已完成。CPU32首批[0,6)原任务187实际退出0，并经新纯终态188前置检查及Root来源、保存的数值证明摘要核对；目前1/32批、6/187图、600/18700 query完成，十组对照均保留。第二批[6,12)原任务188、supervisor11648已启动，活动产物不读取。当前188个正常原任务终态、26个保留科学失败、合计214；活动任务不计完成。全187图数值资格与统一COCO仍待完成，尚无第二种子AP。首201主差值仍+0.197741 AP点，重复效用、正式TEST、跨集、机制与创新结论待证。

[首批普通核验摘要](REPEAT202_FIRSTCPU32_METADATA_SUMMARY_20261016.json)；[执行范围与接续说明](../../research_round_20260925_global/N20_REPEAT202_TRAINING_COMPLETE_ENDPOINT_CPU_STATUS_20261016.md)。


## 历史：前两批CPU32接收完成，第三批运行

第二种子202610150202的固定24轮训练、独立端点核验和四段DEV元数据接收已完成。CPU32前两批[0,6)、[6,12)均实际退出0，并先通过各自新纯终态前置检查，再核对来源及保存的数值证明摘要；目前2/32批、12/187图、1200/18700 query完成，十组对照均保留。第三批[12,18)原任务189、supervisor40058已启动，活动产物不读取。当前189个正常原任务终态、26个保留科学失败、合计215；活动任务不计完成。全187图数值资格与统一COCO仍待完成，尚无第二种子AP。首201主差值仍+0.197741 AP点，重复效用、正式TEST、跨集、机制与创新结论待证。

[第二批普通核验摘要](REPEAT202_SECONDCPU32_METADATA_SUMMARY_20261016.json)；[分段计时与范围](../../research_round_20260925_global/N20_CPU_REVIEW_COST_AND_FUTURE_PROTOCOL_20261016.md)。


## 历史：CPU32已接收3/32批，第4批已启动

第二种子202610150202的固定24轮训练、端点核验和四段DEV元数据接收历史保持。CPU32第3批[12,18)原任务189实际退出0，Root来源及保存数值证明摘要的独立元数据接收已通过；目前3/32批、18/187图、1800/18700自然query完成，十组对照保留。第4批[18,24)原任务190、supervisor42829已启动，活动产物不读取。当前190个正常原任务终态、26个保留科学失败、合计216；活动任务不计完成。全187图数值资格与统一COCO仍待完成，尚无第二种子AP。首201主差值仍+0.197741 AP点；重复效用、正式TEST、跨集、机制与创新结论待证。

[本批普通核验摘要](REPEAT202_CPU32_SCOPE02_METADATA_SUMMARY_20261016.json)。


## 历史：CPU32已接收4/32批，第5批已启动

第二种子202610150202的固定24轮训练、端点核验和四段DEV元数据接收历史保持。CPU32第4批[18,24)原任务190实际退出0，Root来源及保存数值证明摘要的独立元数据接收已通过；目前4/32批、24/187图、2400/18700自然query完成，十组对照保留。第5批[24,30)原任务191、supervisor65257已启动，活动产物不读取。当前191个正常原任务终态、26个保留科学失败、合计217；活动任务不计完成。全187图数值资格与统一COCO仍待完成，尚无第二种子AP。首201主差值仍+0.197741 AP点；重复效用、正式TEST、跨集、机制与创新结论待证。

[本批普通核验摘要](REPEAT202_CPU32_SCOPE03_METADATA_SUMMARY_20261016.json)。


## 历史：CPU32已接收5/32批，第6批已启动

第二种子202610150202的固定24轮训练、端点核验和四段DEV元数据接收历史保持。CPU32第5批[24,30)原任务191实际退出0，Root来源及保存数值证明摘要的独立元数据接收已通过；目前5/32批、30/187图、3000/18700自然query完成，十组对照保留。第6批[30,36)原任务192、supervisor26042已启动，活动产物不读取。当前192个正常原任务终态、26个保留科学失败、合计218；活动任务不计完成。全187图数值资格与统一COCO仍待完成，尚无第二种子AP。首201主差值仍+0.197741 AP点；重复效用、正式TEST、跨集、机制与创新结论待证。

[本批普通核验摘要](REPEAT202_CPU32_SCOPE04_METADATA_SUMMARY_20261016.json)。


## 历史：CPU32已接收6/32批，第7批已启动

第二种子202610150202的固定24轮训练、端点核验和四段DEV元数据接收历史保持。CPU32第6批[30,36)原任务192实际退出0，Root来源及保存数值证明摘要的独立元数据接收已通过；目前6/32批、36/187图、3600/18700自然query完成，十组对照保留。第7批[36,42)原任务193、supervisor10902已启动，活动产物不读取。当前193个正常原任务终态、26个保留科学失败、合计219；活动任务不计完成。全187图数值资格与统一COCO仍待完成，尚无第二种子AP。首201主差值仍+0.197741 AP点；重复效用、正式TEST、跨集、机制与创新结论待证。

[本批普通核验摘要](REPEAT202_CPU32_SCOPE05_METADATA_SUMMARY_20261016.json)。


## 历史：CPU32已接收7/32批，第8批已启动

第二种子202610150202的固定24轮训练、端点核验和四段DEV元数据接收历史保持。CPU32第7批[36,42)原任务193实际退出0，Root来源及保存数值证明摘要的独立元数据接收已通过；目前7/32批、42/187图、4200/18700自然query完成，十组对照保留。第8批[42,47)原任务194、supervisor56946已启动，活动产物不读取。当前194个正常原任务终态、26个保留科学失败、合计220；活动任务不计完成。全187图数值资格与统一COCO仍待完成，尚无第二种子AP。首201主差值仍+0.197741 AP点；重复效用、正式TEST、跨集、机制与创新结论待证。

[本批普通核验摘要](REPEAT202_CPU32_SCOPE06_METADATA_SUMMARY_20261016.json)。


## 历史：CPU32已接收8/32批，第9批已启动

第二种子202610150202的固定24轮训练、端点核验和四段DEV元数据接收历史保持。CPU32第8批[42,47)原任务194实际退出0，Root来源及保存数值证明摘要的独立元数据接收已通过；目前8/32批、47/187图、4700/18700自然query完成，十组对照保留。第9批[47,53)原任务195、supervisor71600已启动，活动产物不读取。当前195个正常原任务终态、26个保留科学失败、合计221；活动任务不计完成。全187图数值资格与统一COCO仍待完成，尚无第二种子AP。首201主差值仍+0.197741 AP点；重复效用、正式TEST、跨集、机制与创新结论待证。

[本批普通核验摘要](REPEAT202_CPU32_SCOPE07_METADATA_SUMMARY_20261016.json)。


## 历史：CPU32已接收9/32批，第10批已启动

第二种子202610150202的固定24轮训练、端点核验和四段DEV元数据接收历史保持。CPU32第9批[47,53)原任务195实际退出0，Root来源及保存数值证明摘要的独立元数据接收已通过；目前9/32批、53/187图、5300/18700自然query完成，十组对照保留。第10批[53,59)原任务196、supervisor86576已启动，活动产物不读取。当前196个正常原任务终态、26个保留科学失败、合计222；活动任务不计完成。全187图数值资格与统一COCO仍待完成，尚无第二种子AP。首201主差值仍+0.197741 AP点；重复效用、正式TEST、跨集、机制与创新结论待证。

[本批普通核验摘要](REPEAT202_CPU32_SCOPE08_METADATA_SUMMARY_20261016.json)。


## 历史：CPU32已接收10/32批，第11批已启动

第二种子202610150202的固定24轮训练、端点核验和四段DEV元数据接收历史保持。CPU32第10批[53,59)原任务196实际退出0，Root来源及保存数值证明摘要的独立元数据接收已通过；目前10/32批、59/187图、5900/18700自然query完成，十组对照保留。第11批[59,65)原任务197、supervisor1554已启动，活动产物不读取。当前197个正常原任务终态、26个保留科学失败、合计223；活动任务不计完成。全187图数值资格与统一COCO仍待完成，尚无第二种子AP。首201主差值仍+0.197741 AP点；重复效用、正式TEST、跨集、机制与创新结论待证。

[本批普通核验摘要](REPEAT202_CPU32_SCOPE09_METADATA_SUMMARY_20261016.json)。


## 历史：CPU32已接收11/32批，第12批已启动

第二种子202610150202的固定24轮训练、端点核验和四段DEV元数据接收历史保持。CPU32第11批[59,65)原任务197实际退出0，Root来源及保存数值证明摘要的独立元数据接收已通过；目前11/32批、65/187图、6500/18700自然query完成，十组对照保留。第12批[65,71)原任务198、supervisor91727已启动，活动产物不读取。当前198个正常原任务终态、26个保留科学失败、合计224；活动任务不计完成。全187图数值资格与统一COCO仍待完成，尚无第二种子AP。首201主差值仍+0.197741 AP点；重复效用、正式TEST、跨集、机制与创新结论待证。

[本批普通核验摘要](REPEAT202_CPU32_SCOPE10_METADATA_SUMMARY_20261016.json)。


## 历史：CPU32已接收12/32批，第13批已启动

第二种子202610150202的固定24轮训练、端点核验和四段DEV元数据接收历史保持。CPU32第12批[65,71)原任务198实际退出0，Root来源及保存数值证明摘要的独立元数据接收已通过；目前12/32批、71/187图、7100/18700自然query完成，十组对照保留。第13批[71,77)原任务199、supervisor82370已启动，活动产物不读取。当前199个正常原任务终态、26个保留科学失败、合计225；活动任务不计完成。全187图数值资格与统一COCO仍待完成，尚无第二种子AP。首201主差值仍+0.197741 AP点；重复效用、正式TEST、跨集、机制与创新结论待证。

[本批普通核验摘要](REPEAT202_CPU32_SCOPE11_METADATA_SUMMARY_20261016.json)。


## 历史：CPU32已接收13/32批，第14批已启动

第二种子202610150202的固定24轮训练、端点核验和四段DEV元数据接收历史保持。CPU32第13批[71,77)原任务199实际退出0，Root来源及保存数值证明摘要的独立元数据接收已通过；目前13/32批、77/187图、7700/18700自然query完成，十组对照保留。第14批[77,83)原任务200、supervisor35775已启动，活动产物不读取。当前200个正常原任务终态、26个保留科学失败、合计226；活动任务不计完成。全187图数值资格与统一COCO仍待完成，尚无第二种子AP。首201主差值仍+0.197741 AP点；重复效用、正式TEST、跨集、机制与创新结论待证。

[本批普通核验摘要](REPEAT202_CPU32_SCOPE12_METADATA_SUMMARY_20261016.json)。


## 历史：CPU32已接收14/32批，第15批已启动

第二种子202610150202的固定24轮训练、端点核验和四段DEV元数据接收历史保持。CPU32第14批[77,83)原任务200实际退出0，Root来源及保存数值证明摘要的独立元数据接收已通过；目前14/32批、83/187图、8300/18700自然query完成，十组对照保留。第15批[83,89)原任务201、supervisor93528已启动，活动产物不读取。当前201个正常原任务终态、26个保留科学失败、合计227；活动任务不计完成。全187图数值资格与统一COCO仍待完成，尚无第二种子AP。首201主差值仍+0.197741 AP点；重复效用、正式TEST、跨集、机制与创新结论待证。

[本批普通核验摘要](REPEAT202_CPU32_SCOPE13_METADATA_SUMMARY_20261016.json)。


## 当前：CPU32已接收15/32批，第16批已启动

第二种子202610150202的固定24轮训练、端点核验和四段DEV元数据接收历史保持。CPU32第15批[83,89)原任务201实际退出0，Root来源及保存数值证明摘要的独立元数据接收已通过；目前15/32批、89/187图、8900/18700自然query完成，十组对照保留。第16批[89,94)原任务202、supervisor30618已启动，活动产物不读取。当前202个正常原任务终态、26个保留科学失败、合计228；活动任务不计完成。全187图数值资格与统一COCO仍待完成，尚无第二种子AP。首201主差值仍+0.197741 AP点；重复效用、正式TEST、跨集、机制与创新结论待证。

[本批普通核验摘要](REPEAT202_CPU32_SCOPE14_METADATA_SUMMARY_20261016.json)。
