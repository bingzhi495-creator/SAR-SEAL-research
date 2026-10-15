# N20：全部DEV GPU分片完成及CPU数值核验状态

更新：2026-10-16。当前全部四GPU分片及32/32份独立CPU范围核验完成（187图/18700自然query/149600策略记录），全187图统一COCO原supervisor66048运行中。正常177/保留科学失败26/203终态。下文保留历史记录，当前状态以本段及文末为准。完整AP及独立post-actual/Root资格尚未完成，没有涨点、正式TEST、跨数据集或发表就绪结论。

## 已实际完成

两臂使用相同741张FIT图、每轮3,035,136采样点、AdamW、24轮和17,784次更新。LOCAL使用原图渲染特征的真实3×3邻域，CENTER重复中心特征；各18,851参数、五张量、相同初始化。E8/E24保存端点已由独立CPU及Root恢复核验；原训练GPU最终清理exit1仍保留，没有补造原GPU REPORT，也没有第四次训练。

修复后的首DEV分片[0,47)原任务实际exit0，耗时1411.156秒。Root先执行新纯终态前缀并取得actual0，再读取本轮结果；检查REPORT、CAPTURE_MANIFEST、START的13项绑定及原任务身份/来源，47张图/4700自然query，47份捕获文件共3,151,195,873字节均与保存SHA一致，十策略各4700条完整预测并保持image/query/class/S0顺序。CUDA清理后allocated/reserved均为0，观察到整卡峰值3488MiB；整卡观察不能称为进程显存认证。耗时含大量证据采集与来源校验，不是独立推理FPS。

这些检查确认原任务完成及输出完整，不替代张量数值、选择索引、PR、strict sign、RLE和全量COCO资格。原REPORT保留qualified=False和CPU pending，不改成已资格。首段完成时正常终态142、失败25、合计167；当前计数见下方更新。

## 保留失败与最小修复

第一次首DEV尝试在序列化回读的Long零步长转Byte处实际exit1，无资源abort，无完整REPORT。31份partial未复用。极小CPU回归先复现同错，再验证仅逐字节比较助手的修复；保持dtype/shape、signed zero和非连续张量的精确语义，不放宽科学容差。模型、评分、训练、捕获、范围及预算没有改变。失败25、原日志与来源按原资格规则保留，不把修复解释为方法收益。

## 当前执行与下一步

第二GPU分片[47,94)现已实际exit0，1721.031秒；新pure143+failed25 actual0先于本段科学产物读取。Root核对原终态与来源、三产物13绑定、47图/4700自然query、47份捕获4,243,201,508字节的SHA、十策略各4700完整预测及image/query/class/S0顺序、canonical profiles、预算与零cleanup。两段累计94图/9400query，仍仅GPU完成与输出完整性，非独立CPU数值或AP资格。前两段完成时正常终态143、失败25、合计168；当前计数见下一段。

第三GPU分片[94,141)已实际exit0，1651.609秒；新pure144+failed25 actual0先于本段科学产物读取，Root完成来源/13绑定、47图/4700query、47份捕获3,825,291,460字节逐SHA及十策略各4700完整预测/原顺序、预算、零cleanup核验。累计141图/14100query，仍只是GPU完成与输出完整性。前三段完成时正常终态144、失败25、合计169；当前计数见下方。

第四GPU分片[141,187)已实际exit0，1217.781秒；新pure145+failed25 actual0先于产物读取。Root核46图/4600query、三产物13绑定、46份捕获2,494,659,875字节逐SHA、十策略各4600完整预测与原顺序、预算和零cleanup。四段累计187图/18700query/187份捕获，共13,714,348,716B；这里只确认GPU完成和输出完整性，独立CPU数值及全量AP仍待完成。四GPU完成时正常终态145、失败25、合计170。首个独立CPU[0,24)随后实际资源中断，当前计数和替代执行范围见本次更新；不从未资格partial或分片AP拼装结论。

原计划八CPU按既定原捕获逐点核验两独立公式、GPU误差、canonical FP64评分、全局排序、预算K、fresh selected PR、strict sign、RLE与完整预测。随后十策略统一187张图/18,700自然query做一次各自全量官方COCO评价，保留全部12项stats，核原生与强参考一致；绝不平均分片AP。最终原CPU和后置Root各自须实际退出0。

| 代码与检查 | 实际状态 | 证据限度 |
|---|---|---|
| 历史25链CPU entry02 | 首独立CPU[0,24)实际资源中断 | 无完整REPORT；保留失败26，partial不复用 |
| 历史full187 COCO entry01 | 旧八范围源稿保留 | 本次新32范围使用entry02，仍未执行COCO |
| 历史post完整entry01 | 旧八范围源稿保留 | 本次新32范围使用entry02，未绑定/执行 |
| public proof_query合成CPU集成检查01 | 实际0，1项测试，2.079秒 | 手构2×3数据；验证两公式/评分/选点/PR/RLE及篡改拒绝，不认证真实GPU |

源码：[CPU完整入口](code/N20_FIXEDDEV_FEATURE_ERROR_CPU_ENTRY_UNBOUND_DRAFT_20261015_02.py)、[全量COCO完整入口](code/N20_FIXEDDEV_FULL187_COCO_CPU_ENTRY_UNBOUND_DRAFT_20261015_01.py)、[post完整入口](code/N20_FIXEDDEV_FULL187_POST_ACTUAL_ROOT_CPU_ENTRY_UNBOUND_DRAFT_20261015_01.py)、[合成集成检查](code/test_N20_SELECTED_NUMERIC_CPU_PUBLIC_API_SYNTHETIC_ONLY_20261015_01.py)。这些是可保存研发稿，不能把文件存在或静态READY写为真实实验成功。

## 与GLDet及创新判断的关系

[GLDet原论文综述](GLDET_PRIMARY_PAPER_REVIEW_AND_SEGMENTATION_TRANSFER_20261014.md)和[局部信息检验设计](GLDET_CODE_RECHECK_AND_MINIMAL_LOCAL_INFORMATION_TEST_20261015.md)保留原文引用及范围。GLDet完整空间结构尚未复现；N20是先检验真实局部特征有无增量信息的隔离实验，普通邻域操作本身不是新创新。N17旧原生单元邻域主结果为负，不能直接推出本轮原图渲染邻域也无效。

主端点始终为E24-atmost：Delta=AP_LOCAL-max(AP_CENTER,68.09938494745545)。Delta≤0停止此配方追加预算/种子/融合；0<Delta<1只允许先复验第二种子，第三种子须符合原锁定条件；Delta≥1复验第二/第三种子。E8或exact不能救主端点。正收益仍需排除额外容量/计算与评价因素，说明对直接先例的实际差异，补重复种子和跨集；“加模块后涨点”不会自动成为TGRS充分证据。

## 发布范围

本次仅同步普通独立研发副本、合同和经核验状态；不发布原始数据、标注、权重、第三方全文或运行绑定资格链。当前运行中的科学文件与绑定Source未为同步而读取、复制或哈希。


## 首CPU实际资源中断与新执行范围（本次更新）

原CPU[0,24)在1800.343秒实际退出1，守护程序记录运行资源/截止门中断；held句柄已回收且原出生身份消失。Root先执行145个正常与26个保留失败的纯终态前缀，实际退出0且未导入Torch，再读取本轮结果。原输出有13份逐图证明文件，没有完整REPORT；这些partial不复用、不算完成。观察到的最小可用内存30,790,778,880B，整卡占用峰值2146MiB；这些记录不支持显存不足或方法失效结论。

这是数值核验任务的资源问题，不是额外训练，也不是部署推理速度测量。当前正常145/失败26/总终态171，合格CPU0份，仍没有N20 AP。原四GPU报告与捕获、E8/E24模型以及失败25/训练失败24保持原样。

新[执行补充03](../refine-logs/N20_TRUE_FEATURE_NEIGHBOR_20261015_01/CPU_EXECUTION_PARTITION_PLANNED_20261015_03.json)将每个原GPU父域按最多6张连续整图切成8份，四父域共32份，覆盖187图/18700自然query且无重叠、无遗漏。新owner145–176，之后统一全187 COCO与后置核验另有自己的执行身份。改变仅针对执行资源，不按AP或partial结果选择图片；原八份计划保留为历史，不把部分文件升级为通过。新CPU完整入口03已冻结，首份[0,6)独立核验已启动，0/32完成；统一COCO和post完整入口02已完成文本装配/AST检查，仍未绑定和执行。

源码层面确认：atmost/exact已共享各head的CPU输出，不存在按八策略重复八遍head的情况；仍有重复F9构造、掩膜解包和不变状态检查，尚无实测阶段耗时证明其为瓶颈。本轮优先只减小分片，不改数值函数；仍逐份检查实际捕获输入、四模型各两CPU公式、原2048批次、绝对1e-4/rtol0、保存FP64分数、fresh PR、strict sign、mask/RLE及全部十策略预测。不能减少验证后仍宣称执行了原完整证明。

### 32份核验代码与首份启动

[CPU完整entry03](code/N20_FIXEDDEV_FEATURE_ERROR_CPU_ENTRY_UNBOUND_DRAFT_20261015_03.py)、[CPU合同02](code/N20_FIXEDDEV_FEATURE_ERROR_CPU_FRONT_UNBOUND_CONTRACT_20261015_02.md)、[统一COCO entry02](code/N20_FIXEDDEV_FULL187_COCO_CPU_ENTRY_UNBOUND_DRAFT_20261015_02.py)与[post entry02](code/N20_FIXEDDEV_FULL187_POST_ACTUAL_ROOT_CPU_ENTRY_UNBOUND_DRAFT_20261015_02.py)已保存。合成元数据检查覆盖32份完整人口、旧范围/错误parent/失败partial拒绝；完整入口header的AST提取式检查通过。源码比较确认每图all100数值循环、PR双公式、检查、编码及序列化AST与旧版一致，历史GPU25保持原样。它们只说明装配与合同符合预期，不替代真实GPU/CPU数值或AP资格。

启动前新纯RAW-only检查actual0（0.292秒），无Torch且在首次Source哈希前停止。唯一新原CPU任务为[0,6)，CUDA空、CPU0/1与两线程、1800秒上限、RSS16GiB及最小可用RAM16GiB，无自动重试。该首份启动时为正常145/失败26/171终态，0/32合格；当前进度见下方更新，运行中的任务不计完成。模型和原四GPU捕获未重算，主端点和全187官方COCO仍按原锁定规则。

## 首份独立CPU核验通过，第二份已启动

[0,6)原任务实际退出0，1088.890秒，原句柄已回收、出生身份消失，无资源abort或等待错误。Root先执行新纯终态前缀146正常+26失败，实际0（0.407秒，无Torch），再读取报告、来源和结果。原RAW绑定stdout中的报告ref与物理REPORT完全一致，原entry/spec/admission/phase/bootstrap与来源绑定均核对。原REPORT仍保持pending/qualified=False，不修改为自授资格。

6份逐图proof覆盖600个自然query、4800个策略记录及19225个注册成员。Root读取各strictref并逐项核自然query顺序、K/C、四head双CPU公式统计、2048分块、绝对1e-4/rtol0、严格符号、FP64评分与索引/mask/RLE门以及全部成员覆盖；两CPU公式分别对实际GPU的最大差异均为4.57763671875e-5。十策略各600条完整预测与原GPU父域切片逐值相等，包括image/query导出顺序、类别、S0与RLE；没有重排自然query来掩盖预测顺序差异。

这些统计由实际退出0的冻结数值Source计算，Root结合来源、原终态、实际日志、完整proof及预测作本子范围核验；只看True旗标不能重新证明张量运算。注册成员覆盖是去重集合一致，不代表每个成员只被访问一次。核验耗时含来源验证与重复公式，不是独立推理FPS。

[纯metadata读取检查器](code/N20_ROOT_CPU_PROOF_METADATA_VALIDATOR_SOURCE_ONLY_20261015_01.mjs)及[手工反例](code/test_N20_ROOT_CPU_PROOF_METADATA_VALIDATOR_SYNTHETIC_ONLY_20261015_01.mjs)已保存：100 query/800 profiles，覆盖K0、2048分块边界和11项篡改拒绝；两公式各对GPU≤1e-4时，彼此差异统计允许2e-4，不错误收紧科学门。手工检查不替代真实数据验证。

第二份[6,12)沿用同一冻结完整CPU Source与模型/容差/规则；新RAW-only启动前检查实际0（0.295秒，无Torch/首次SHA前停止），唯一原任务已启动，无自动重试。首份完成时为1/32份，即6/187图、600/18700自然query；正常146、失败26、合计172；当前进度见下方第二份更新。只有全部32份原CPU实际0并核验后才统一全187官方COCO，当前没有新AP、涨点、SOTA或TGRS结论。

## 第二份独立CPU通过，第三份已启动

[6,12)原任务实际退出0（1178.000秒），held句柄回收且原出生身份消失，无资源abort/等待错误。Root先执行pure147+failed26实际0（0.387秒，无Torch），随后核原Source/spec/admission/phase/bootstrap、23项报告绑定和原RAW-bound stdout报告ref；6份proof覆盖600自然query、4800策略记录及19802注册成员，ten策略各600预测与原GPU父切片全字段一致。两个CPU scout公式分别对实际GPU最大差异均为3.0517578125e-5，原2048分块、1e-4/rtol0、FP64评分与选点、PR两公式和strict sign/mask/RLE门不变。原REPORT仍pending/qualified=False，Root本子范围核验不升级为完整DEV资格。

第三份[12,18)沿同一冻结Source启动，pre-HASH RAW-only检查实际0（0.285秒，无Torch），无自动重试。累计2/32份、12图/1200query、9600策略记录；当前正常147/失败26/173终态。全部32份完成前不运行统一AP，不以局部核验代替模型效用。

独立并行的[HRSID候选标注版本实文件核查](HRSID_OFFICIAL_ANNOTATION_INTAKE_20261015_01.md)只完成固定作者提交的隔离JSON元数据验证：3573图/9870标注，215图无标注关联；PNG/JPG版本、完整TEST与母景关系未认证。未把原JSON/影像加入本次N20数据或Source，也未发布原数据。


## 第三份独立CPU通过，第四份已启动

[12,18)原任务held实际退出0（871.141秒），原句柄回收、原出生身份消失，error/abort均无。Root先执行新pure148+failed26实际0（0.390秒，无Torch），随后才读取本段科学结果；原RAW-bound stdout报告ref、Source/spec/admission/phase/bootstrap及完整23项绑定通过。6图、600自然query、4800策略记录、17116注册成员完整覆盖；双CPU scout公式各对实际GPU最大2.288818359375e-5，原1e-4/rtol0、2048分块、FP64评分/索引、PR两公式、strict sign及mask/RLE门不变。十策略各600条预测与原GPU父域切片逐字段一致，原REPORT仍pending/qualified=False。

第四份[18,24)同冻结entry03已启动，pre-HASH RAW-only检查actual0（0.299秒，无Torch）；无自动重试。累计3/32份、18图/1800自然query/14400策略记录，正常148/失败26/总终态174。尚无N20全187官方COCO/AP。完成子范围数值核验不等于涨点、重复种子或核心创新。

独立的[HRSID官方PNG目录与TEST实核](HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)取得同包完整目录及TEST 1962图/5922实例；同包TRAIN短响应未完成，不能与先前Git候选TRAIN混配。[F9条件性信息说明](N20_RENDERED_F9_CONDITIONAL_INFORMATION_AND_CONTROL_SCOPE_20261015_01.md)澄清最终渲染栅格邻域及四基结论的条件，没有读取实际位置/张量或改动当前配方。


## 第四份独立CPU通过，第五份已启动

[18,24)原supervisor68897 held实际退出0（835.922秒），原句柄回收且出生身份消失，error/abort均无。Root先运行新pure149+failed26实际0（0.386秒，无Torch），随后核原RAW-bound stdout REPORT引用、Source/spec/admission/phase/bootstrap及23项绑定。6图/600自然query/4800策略记录、16376注册成员完整覆盖；双CPU scout公式分别对实际GPU最大2.288818359375e-5，原绝对1e-4/rtol0、2048分块、FP64评分/索引、PR两公式、strict sign及mask/RLE检查不变，十策略各600完整预测与父域切片全字段相同。原REPORT pending/qualified=False保留，本子范围外部核验不是完整DEV或AP资格。

第五份[24,30)沿用同一冻结entry03与模型/主端点，RAW-only prehash检查实际0（0.342秒，无Torch）后唯一原supervisor29083启动，无自动重试。累计4/32份、24/187图、2400/18700自然query及19200策略记录，正常149/保留失败26/175终态。尚未运行统一全187官方COCO。

独立HRSID小分块新轮首请求仍短响应，已停止；Root实读partial长度及SHA与安全日志一致。没有完整TRAIN，不混用旧Git候选TRAIN；详情见[官方PNG来源记录](HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)。


## 第五份独立CPU通过，第六份已启动

[24,30)原supervisor29083 held实际退出0（767.563秒），原句柄回收且出生身份消失，error/abort均无。Root先运行新pure150+failed26实际0（0.336秒，无Torch），随后核RAW-bound stdout REPORT引用、Source/spec/admission/phase/bootstrap及完整23项绑定。6图/600自然query/4800策略记录、16271注册成员完整覆盖；双CPU scout公式分别对实际GPU最大3.0517578125e-5，原绝对1e-4/rtol0、2048分块、FP64评分/索引、PR两公式、strict sign和mask/RLE门保持，十策略各600完整预测与原父域切片逐字段一致。原REPORT保持pending/qualified=False，本子范围核验不等同完整DEV或AP资格。

第六份[30,36)同冻结entry03与模型/主端点，RAW-only prehash实际0（0.304秒，无Torch）后唯一原supervisor59461启动，无自动重试。累计5/32份、30图/3000自然query/24000策略记录；正常150/保留失败26/176终态。统一全187官方COCO尚未执行。

### 研究判断的有限独立复审

独立审阅复核[N18能力统计](N18_QUALIFIED_ROUTING_CAPABILITY_RESULT_AND_NEXT_HYPOTHESIS_20261015.md)和[像素与AP边界](N18_PIXEL_COUNTS_VERSUS_AP_INTERPRETATION_20261015.md)：LOCAL漏选6353/10927个预算内可修机会、实际翻转44.389%有害，说明路由有改进余量；固定PR对C内65423原错误可修11146，也说明条件性动作能力有限。二者同时存在，不能排序为唯一全量AP主瓶颈；该群体仅374个原生IoU≥0.5匹配实例，不含6个未匹配GT和其他自然query。强R有益6564/有害8528却将全量AP提升约1.03点，是按像素净量直接解释AP的实际反例。此次仅复核已有合格统计，没有新增数值实验或测量N20实际邻域秩。

N20最多检验当前渲染像素邻域在本初始化/预算/监督/固定PR下的效用；它不是原特征网格相邻单元实验。同名义参数的CENTER未完全匹配输入秩、协方差和优化几何。Delta≤0仍STOP当前配方，包括LOCAL高于CENTER但不超过强R的情况；Delta>0只按既定门复验，不因单次收益跳到创新/泛化主张。下一轮若要改变selector或PR，仍需区分AP敏感实例的漏选/误改与缺少可修动作；本次不启动新的方法实验或改写停止规则。


### HRSID同包标注版本实核完成

[官方PNG来源记录](HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)现已完整取得TRAIN及TEST：Root独立核原生curl完成范围、实际块SHA、完整压缩流与解压字节/CRC/JSON。TRAIN3642图/11047实例、TEST1962图/5922实例，文件名无交集且并集5604精确覆盖同包PNG目录；数值imageID/annID跨split复用，不能据裸ID认定重复或直接合并。Git候选文件名仅共同3572、候选独有1/PNG TRAIN独有70，不能混用。没有图像像素/mask解码/跨集AP或盲测资格恢复；本次N20数据和Source不变，所有原下载失败保留。


## 第六份独立CPU通过，第七份已启动

[30,36)原supervisor59461 held实际退出0（633.813秒），原句柄回收且出生身份消失，error/abort均无。Root先执行新pure151+failed26实际0（0.382秒，无Torch），随后核原RAW-bound stdout REPORT引用、Source/spec/admission/phase/bootstrap与完整23项绑定。6图/600自然query/4800策略记录、15002注册成员完整覆盖；双CPU scout公式分别对实际GPU最大3.0517578125e-5，原绝对1e-4/rtol0、2048分块、FP64评分/索引、PR两公式、strict sign和mask/RLE门保持，十策略各600完整预测与原父域切片逐字段相等。原REPORT保持pending/qualified=False；该子范围核验不是全DEV/AP资格。

第七份[36,42)同冻结entry03、模型/主端点，RAW-only prehash实际0（0.296秒，无Torch）后唯一原supervisor95848已启动，无自动重试。累计6/32份、36图/3600自然query/28800策略记录，正常151/保留失败26/177终态。统一全187官方COCO尚未执行。

[数据获取说明](../docs/DATA_ACCESS.md)及[机器可读上游索引](../docs/upstream_data_index.json)现已区分Git候选与同包PNG的实际标注版本、SHA/CRC、split ID命名空间及未完成影像/mask/AP资格。本机一次有界目录元数据盘点仅确认历史目录内358个.jpg普通文件、66018309B，未打开或hash文件；不能认证完整原数据或与官方PNG像素同版。没有据后缀/前缀改名或替换标注，不改变N20数据或Source。


## 第七份独立CPU通过，第八份已启动

[36,42)原supervisor95848 held实际退出0（645.547秒），原句柄回收且出生身份消失，error/abort均无。Root先运行新pure152+failed26实际0（0.412秒，无Torch），随后核原RAW-bound stdout REPORT引用、Source/spec/admission/phase/bootstrap与完整23项绑定。6图/600自然query/4800策略记录、15008注册成员完整覆盖；双CPU scout公式分别对实际GPU最大3.0517578125e-5，原绝对1e-4/rtol0、2048分块、FP64评分/索引、PR两公式、strict sign和mask/RLE门保持；十策略各600完整预测与父域切片逐字段相同。原REPORT保持pending/qualified=False，该子范围通过不等于全DEV或AP资格。

第八份[42,47)为5图，同冻结entry03/模型/主端点，RAW-only prehash实际0（0.350秒，无Torch）后唯一原supervisor66496启动，无自动重试。累计7/32份、42图/4200自然query/33600策略记录，正常152/保留失败26/178终态。统一全187官方COCO尚未执行。

[原F几何元数据定位](N20_RENDERED_F9_CONDITIONAL_INFORMATION_AND_CONTROL_SCOPE_20261015_01.md)确认已审ABI不显式保存原F空间尺寸；已保留的四清单187行original_shape只是图像H,W，不测原F尺度或秩，不改冻结任务。独立的[官方PNG编码文件试点](HRSID_OFFICIAL_PNG_ENCODED_IMAGE_BYTES_PILOT_20261015_01.md)已取得两文件并由Root核SHA/ZIP与PNG容器CRC/IHDR，未解码IDAT/像素、未mask/AP/模型评价或加入N20 Source，不等于取得完整5604图。原header解析失败及最小修复保留。


## 第八份独立CPU通过，第九份已启动

[42,47)原supervisor66496 held实际退出0（1337.344秒），原句柄回收且出生身份消失，error/abort均无。Root先执行新pure153+failed26实际0（0.343秒，无科学库导入），随后核验RAW-bound stdout REPORT引用、Source/spec/admission/phase/bootstrap与完整23项绑定。5图/500自然query/4000策略记录、20065注册成员完整覆盖；双CPU scout公式分别对实际GPU最大3.0517578125e-5，满足原绝对1e-4/rtol0门；十策略各500完整预测与原父域切片逐字段相同。原REPORT仍为pending/qualified=False，外部Root仅对该子范围核验通过，不等于全DEV/AP资格。

第九份[47,53)切入第二个GPU父域[47,94)，沿用冻结Source/模型/主端点。RAW-only prehash实际0（0.307秒，无科学库导入）后唯一原supervisor63376启动，无自动重试。累计8/32份、47图/4700自然query/37600策略记录，正常153/保留科学失败26/179终态。全187图统一官方COCO尚未执行，无N20涨点、创新或TGRS成功结论。

[官方数据获取](HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)的8MiB传输控制通过，但后续整包下载第3块curl超时，已停止；只有16MiB有效归档前缀，尚未备齐全部影像，不影响当前N20评价。


## 第九份独立CPU通过，第十份已启动

[47,53)原supervisor63376 held实际退出0（1189.047秒），原句柄回收、出生身份消失且无error/abort。Root先执行新pure154+failed26实际0（0.325秒，无科学库导入），随后恢复固定原函数并以单次退出的Root元数据流程完成核验；Source/spec/admission/phase/bootstrap、RAW-bound stdout REPORT及23项绑定均通过。6图/600自然query/4800策略记录、20166注册成员完整覆盖，双CPU scout公式分别对实际GPU最大4.57763671875e-5，满足原绝对1e-4/rtol0门；十策略各600完整预测与第二GPU父域切片逐字段一致。原REPORT保持pending/qualified=False，仅该子范围获得外部Root核验，没有全DEV/AP资格。

第十份[53,59)沿用冻结Source、模型、范围方案和主端点。RAW-only prehash实际0（0.333秒，无科学库导入）后唯一原supervisor29173启动，无自动重试。累计9/32份、53图/5300自然query/42400策略记录，正常154/保留科学失败26/180终态。辅助长寿命Node进程的内存不足退出未中断独立科学任务，不计为科学RAW失败；已恢复的原核验函数与纯门顺序经过实际范围核验，后续采用单次进程并限制内存，不改科学配方。统一全187官方COCO尚未执行，没有N20涨点、创新或TGRS成功结论。

[HRSID获取记录](HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)保留原下载失败，新规则02续取已完成首个8MiB新块（198.735秒），仍未完成整包或跨集资格。


## 第十份独立CPU通过，第十一份已启动

[53,59)原supervisor29173 held实际退出0（845.047秒），原句柄回收、出生身份消失且无error/abort。Root先执行新pure155+failed26实际0（0.389秒，无科学库导入），再以原固定核验函数的单次进程流程完成RAW-bound stdout REPORT、Source/spec/admission/phase/bootstrap和23项绑定核验。6图/600自然query/4800策略记录、16968注册成员完整覆盖；双CPU scout公式分别对实际GPU最大3.0517578125e-5，满足原绝对1e-4/rtol0门；十策略各600完整预测与原第二GPU父域切片逐字段一致。原REPORT仍为pending/qualified=False，仅该子范围通过外部Root核验，没有全DEV/AP资格。

第十一份[59,65)沿用相同冻结Source、模型、数值规则和主端点，RAW-only prehash实际0（0.315秒，无科学库导入）后唯一原supervisor59589启动，无自动重试。累计10/32份、59图/5900自然query/47200策略记录，正常155/保留科学失败26/181终态。统一全187图官方COCO尚未执行；没有N20涨点、创新或TGRS成功结论。

[有限增量先例核查](DIRECT_SAR_INSTANCE_SEGMENTATION_INCREMENTAL_PRIOR_CHECK_20261015_01.md)区分O²Former预印本全文与DACF-net官方域索引，未把未直开全文的数值升级为独立复核；不改变当前端点。[HRSID获取](HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)新规则02已再次受控停止，64MiB前缀不等于完整数据集，官方百度候选格式仍未确认。


## 第十一份独立CPU通过，第十二份已启动

[59,65)原supervisor59589 held实际退出0（894.422秒），原句柄回收、出生身份消失且无error/abort。Root先执行新pure156+failed26实际0（0.466秒，无科学库导入），再用原固定核验函数完成RAW-bound stdout REPORT、Source/spec/admission/phase/bootstrap与23项绑定核验。6图/600自然query/4800策略记录、17309注册成员完整覆盖；双CPU scout公式分别对实际GPU最大3.0517578125e-5，满足原绝对1e-4/rtol0门；十策略各600完整预测与第二GPU父域原切片逐字段相同。原REPORT保持pending/qualified=False；该子范围的外部Root通过不等于全DEV或AP资格。

第十二份[65,71)同冻结Source、模型、数值和主端点，RAW-only prehash实际0（0.349秒，无科学库导入）后唯一原supervisor84549启动，无自动重试。累计11/32份、65图/6500自然query/52000策略记录，正常156/保留科学失败26/182终态。统一全187图官方COCO尚未执行，没有N20涨点、创新或TGRS成功结论。

[有限先例综述](DIRECT_SAR_INSTANCE_SEGMENTATION_INCREMENTAL_PRIOR_CHECK_20261015_01.md)补记O²Former v1三类实现澄清项；这不证明外部算法无效，自行选择实现时须标同协议适配而非完整作者复现，不改当前N20端点。


## 第十二份独立CPU通过，第十三份已启动

[65,71)原supervisor84549 held实际退出0（929.219秒），原句柄回收、出生身份消失且无error/abort。Root先执行新pure157+failed26实际0（0.415秒，无科学库导入），再用原固定核验函数完成RAW-bound stdout REPORT、Source/spec/admission/phase/bootstrap及23项绑定核验。6图/600自然query/4800策略记录、17343注册成员完整覆盖；双CPU scout公式分别对实际GPU最大3.0517578125e-5，满足原绝对1e-4/rtol0门；十策略各600完整预测与第二GPU父域原切片逐字段相同。原REPORT保持pending/qualified=False，仅该子范围通过外部Root核验，不等于全DEV/AP资格。

第十三份[71,77)同冻结Source、模型、数值与主端点，RAW-only prehash实际0（0.330秒，无科学库导入）后唯一原supervisor11499启动，无自动重试。累计12/32份、71图/7100自然query/56800策略记录，正常157/保留科学失败26/183终态。统一全187图官方COCO尚未执行；没有N20涨点、创新或TGRS成功结论。

[独立数据传输控制记录](HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)保留本次10秒连接超时/0B；没有HTTP响应，未证明实际源地址匹配或TAP旁路，未改系统设置、未追加整包任务，不影响当前N20数据与评价。


## 第十三份独立CPU通过，第十四份已启动

[71,77)原supervisor11499 held实际退出0（1042.016秒），原句柄回收、出生身份消失且无error/abort。Root先执行新pure158+failed26实际0（0.366秒，无科学库导入），再用原固定核验函数完成RAW-bound stdout REPORT、Source/spec/admission/phase/bootstrap及23项绑定核验。6图/600自然query/4800策略记录、18239注册成员完整覆盖；双CPU scout公式分别对实际GPU最大2.288818359375e-5，满足原绝对1e-4/rtol0门；十策略各600完整预测与第二GPU父域原切片逐字段相同。原REPORT保持pending/qualified=False，仅该子范围通过外部Root核验，不等于全DEV/AP资格。

第十四份[77,83)同冻结Source、模型、数值与主端点，RAW-only prehash实际0（0.295秒，无科学库导入）后唯一原supervisor59486启动，无自动重试。累计13/32份、77图/7700自然query/61600策略记录，正常158/保留科学失败26/184终态。统一全187图官方COCO尚未执行；没有N20涨点、创新或TGRS成功结论。

[独立HRSID获取记录](HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)补记1MiB新规则03试点首请求300秒超时/381717B、0新增完整块；Root实核新失败body与八旧完整块及复制前缀，仍仅64MiB。有限镜像检索未找到已核原版PNG替代；不改当前N20数据或Source，不把数据获取失败计入科学失败26。


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
