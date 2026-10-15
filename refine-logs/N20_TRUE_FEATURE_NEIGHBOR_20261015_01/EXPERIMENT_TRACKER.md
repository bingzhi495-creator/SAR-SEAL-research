# N20 执行状态（2026-10-15）

**最新：四个DEV GPU原任务全部实际exit0；每段先新纯终态前缀actual0再读取产物，Root核验13绑定/来源、全187图/18700自然query、187份物理捕获与十策略各完整预测，总捕获13,714,348,716B。第四段1217.781秒、2,494,659,875B捕获、cleanup零占用；原报告均保持pending CPU。当前145正常/25失败/170终态，首独立CPU[0,24)实际已启动、0/8完成，模型与评分/选择协议未变，暂无AP。下文历史状态以本段与文末为准。**

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
