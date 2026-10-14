# N13 固定PR候选的覆盖、纠正能力与选点余量诊断（草案）

2026-10-14。状态：**DRAFT / 未登记科学执行资格 / 未运行真实SAR数据**。这是负结果后的条件诊断，不是新方法收益或创新声明。数学实现的人工布尔例子仅检查有限输入下的算法关系。

## 问题与执行顺序

[三个固定E48主结果](N13_FIXED48_THREE_SEED_PRIMARY_RESULT_AND_NEXT_DIAGNOSTIC_20261014.md)已完整核验，GEO−LOG平均为−0.189583 AP百分点；当前联合配方按预锁规则停止扩展。剩余预锁E08/E24先按原协议完成，不以最好曲线端点替换E48。下一主要问题是：**在固定原生输出、固定PR和逐query原K下，更多候选区域里是否存在当前PR实际能够利用的修复余量？**

N12中的width2区域提高了GT理想覆盖，但候选域为原域约4.545倍；这不是同计算成本提升。不能仅凭GT存在正确答案，就决定训练收益选择器。

## 固定输入及资格门

- 条件载体预先指定第一个预锁下游种子701的LOG E48 PR，不按最高DEV AP挑选。187图内部DEV已有选型历史；只作探索性诊断。
- 配对人口沿原生掩膜.50匹配的固定374对。正式登记逐对image、GT、原query身份及原匹配规则；不得按新输出重新配对。380 GT中的六个未匹配者单列，不临时挑query赋予K或Ω，也不推断它们没有可利用提议。
- Ω = 原LOG候选域 ∪ 原生边界width2。width2的精确定义、图像边界、空mask、距离算法、候选去重、原图flat索引和逐query K必须在执行前沿既有协议核实。K是不同选点数，既不等于候选域大小，也不等于二值翻转数。
- 普通训练/评价ABI记载的PR输入为原栅格F_original[256,H,W]与原生L[H,W]，没有x/y、ROI局部坐标或扩框归一化网络。冻结原渲染来源、逻辑栅格、通道顺序和原图flat索引；扩大Ω只增加候选位置，不更改旧点的F/L。原LOG域与新增halo分开固定2048点分块，避免新增候选重排旧分块。e和零patch49在文档中不影响残差值；实际源码和载体身份仍需ROOT终止后核验。
- 固定PR的每点候选C必须独立于最终选集合产生，保存每点原生值、残差及精确二值结果。原协议的阈值、严格或含等号比较符、插值和数值路径沿用实际来源，不能默认“≥0.5”。排序权重r=0.5与预测概率阈值是不同概念。
- 在原LOG选点上核验C，并重建原LOG最终掩膜：选点外原样，选点内候选一致。提议、query/class/score/order、空输出规则不变。现存full-L allclose诊断失败不授权放宽容差或称位级一致；不能重建时先停止并定位身份、坐标或数值问题。
- 实际数据、检查点、缓存、逐对人口、SOURCE和所有权资格尚未登记。本草案和纯布尔实现不替代这些门。

## 四项主比较及一项约束诊断

全部比较使用同一A（原生二值mask）、非空G（配对GT）、Ω和逐query K：

|分支|可用信息与允许动作|作用|
|---|---|---|
|GT不限K区域oracle|Ω中直接写G，Ω外保留A|该区域可纠正的理想潜力，允许更多动作|
|GT同K行动oracle|从Ω选exact K位置，候选为G|控制动作数后的理想潜力|
|固定C同K行动oracle|从Ω选exact K位置，只能写冻结C|当前PR候选实际提供的最好局部IoU|
|原LOG支持输出|用原支持exact K位置写同一冻结C|当前选择与候选的实际条件行为|

GT只用于离线特权诊断，不进入可部署选择器。固定C oracle按FN修复、FP修复、不变、新增FP、损失TP五类排序，以原图flat索引稳定处理平局；r=0.5是固定类别次序约定，不是每点真实IoU变化。选满K时允许不变填充。

逐实例核验：
**GT不限K IoU ≥ GT同K IoU ≥ 固定C同K IoU ≥ 原支持IoU**。
GT同K不得低于原生；固定C同K可能低于原生，因为exact K可能迫使有害动作。违例先按实现、身份或输入不一致处理，不解释为科学现象。

另列固定C的at-most-K修复oracle：只选有益动作，可能少于K。它只诊断拒绝动作的潜力，不能冒充同exact-K对照。记录最少被迫有害选点数：
max(0, K − 有益候选数 − 不变候选数)，并核对实际oracle所选有害数。

## 必须记录的输出与判读

报告374配对的IoU均值及分布、FP/FN总数、有效翻转/修复/破坏/不变、.50/.75/.90/.95阈值双向跨越和跌破.50。可另报达阈值数/380；配对IoU均值仍明确使用374分母，不混算六个未匹配者。这不是COCO AP，也不是自然18,700 query全部人口的收益。

区域至少分为原LOG域、width2新增域、Ω外；在原生距离带内/外分层，分别统计错误覆盖和固定候选修复/伤害。并列逐实例均值和总像素加权计数，不能互换分母。

优先比较三个非负条件差：GT不限K−GT同K（动作预算），GT同K−固定C同K（当前候选能力），固定C同K−原支持（选择余量）。它们帮助确定下一检验优先级，不是可部署收益、因果贡献或可达AP提升。

- 当前候选能力差占主导，特别是在新增域：先检验相同结构、相同初始化及预算的训练支持匹配；不得由当前外推失败宣称所有宽域PR无效。
- 固定候选已有明显余量而原支持选取有害动作：才优先检验收益选择。必要替代包括同容量error＋actual flip、共同Ω上的原不确定性topK和MagNet式评分；不能仅比弱选择器。
- 被迫有害动作造成下降：先检验保留旧值/拒绝动作约束。at-most-K是另一个决策条件，不与exact-K混称同预算。
- 主要差来自不限K与同K：优先研究覆盖、动作数及全流程成本。没有充分余量时，不直接追加复杂结构。
- 以上都仅描述第一条件载体和这一固定人口；后续是否值得训练、以及唯一主要假设，需在真实结果核验后另锁。

记录Ω枚举、全候选PR计算、排序、替换、数据传输和总时间；原K只匹配动作数，全Ω预测是特权诊断开销，不等于稀疏部署成本。诊断结束后选择一条主要假设，不自动启动多模块长训。

## 直接先例与保守设计方向

[PointRend，CVPR 2020](https://arxiv.org/pdf/1912.08193)的训练保留均匀覆盖，推理采用不确定性点；[作者采样实现](https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_features.py)要求先插值logit再算不确定性。扩大训练覆盖或不确定性采样并非本项目的新原则。

[RefineMask，CVPR 2021](https://arxiv.org/pdf/2104.08569)在后续阶段使用GT与前阶段预测边界的联合监督范围，推理保留预测边界外旧值；[作者实现](https://github.com/zhanggang001/RefineMask/blob/main/mmdet/models/roi_heads/refine_roi_head.py)可作直接控制参考。其固定RoI网格的边界宽度不能等同原图width2。

[Mask Transfiner，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.pdf)的incoherence监督来自GT降采样重建损失；联合点注意力可能使候选依赖所选集合，[作者公开代码](https://github.com/SysCV/transfiner/blob/main/detectron2/modeling/roi_heads/mask_head.py)不能直接替代当前独立候选假设，作者重实现与论文细节也须区分。

[SegFix，ECCV 2020](https://arxiv.org/pdf/2007.04269)将方向转为实际标签搬移动作，再用特权几何检验潜力，是动作受限诊断的相关先例。[SRNet，Remote Sensing 2023](https://www.mdpi.com/2072-4292/15/5/1324)研究SAR旋转框内掩膜采样及目标不同部位监督；尚未核验其作者代码，也不据此推断本项目ROI现状。

只有新增域GT潜力较高而固定PR能力低时，才考虑“相同训练支持下，固定局部上下文与实例条件能否改善候选修正”的结构假设。最先对照支持匹配的原PR和容量匹配MLP；固定邻域独立读取须维持候选不依赖选集合。该方向有直接光学先例，尚无SAR实验、技术差异或TGRS依据。

## 当前实现状态

[纯布尔诊断草案](code/N13_fixed_candidate_action_oracle_draft.py)不加载模型或数据，由调用者提供A/C/G/Ω/S/K。人工穷举检查只证明已检查的小规模输入关系；代码状态保持UNBOUND_DRAFT。原栅格候选适配器草案已保存并通过合成检查；实际来源适配、资格登记、完整SAR数据运行及独立结果核验均未完成。

人工验证：正常模式（optimize=0）实际退出0，检查3,584个完整三像素配置、256个随机配置、9,935项exact-K预算、32,898项候选/GT子集及17项非法输入拒绝。独立静态审阅确认穷举答案未复用排序算法；其指出的优化模式禁用assert问题已增加显式拒绝门，-O自测实际退出1并报预期拒绝，优化模式的纯API仍可调用。以上均为SYNTHETIC_ONLY，不是科学失败、真实任务效用或执行资格；SAR适配层和来源关系仍待核验。

复现人工算法检查：在仓库根目录执行 `python -B research_round_20260925_global/code/N13_fixed_candidate_action_oracle_draft.py --self-test`。默认256个随机配置、种子20261014；正常模式运行。此次查询环境为Python 3.10.13、NumPy 1.26.4，与创建草案前查询一致。`-B`避免生成字节码；自测不读取SAR数据、不执行资格登记或Git发布，自测中的publication标记描述程序未执行上传，仓库发布状态另由Git提交核验。


## 原栅格候选适配器草案与待核验门

[原栅格适配器](code/N13_original_raster_fixed_PR_adapter_draft.py)为UNBOUND_DRAFT / SYNTHETIC_ONLY。输入显式八张量、F_original、exact native L及互不重叠的原LOG/新增halo域，返回升序原图flat索引、未被旧band门裁去的原始残差、L+u及二值候选；域外保持原生L>0。它不构建区域、不读GT或检查点，不接收K或最终选集合，不执行发布。数学API省去残差无关的e/patch；私有合成ABI包装仅检查形状/零patch，不认证真实e来源。

按普通ABI实现40,497参数：F256非仿射LayerNorm(eps=1e-5)，c=clamp(L,−8,8)/8逐层注入三个91维ReLU和最后线性层。候选比较符由实际协议固定。ROOT已在科学任务间隙完成[真实源码文字核对](N13_FIXED_PR_ACTUAL_SOURCE_INTERFACE_REVIEW_20261014.md)：实际导出使用严格logit>0，即strict_gt_zero；含等号≥0仅为合成API分支。该文字核对不代替真实端点、输入、原输出重建和运行资格。旧LOG与新增halo分别以2048点分块；新增halo不改变旧LOG分块。非连续L的merged_C采用可写连续副本，输入和既有梯度不改写。

已有合成自测实际退出0：独立nn.Module/LayerNorm/Linear算术对照、零末层和L=0阈值、旧域分块不变、输入/梯度保持、非法输入拒绝及非连续输出写回。普通导入不加载Torch；-O自测按预期拒绝。环境为host Torch2.2.2+cpu，单CPU线程、未初始化CUDA；与科学Torch2.0.1+cu118不同，不能代替其资格。独立只读审阅未发现草案发布阻塞错误，其指出的非连续输入零末层覆盖缺口已由下面一次非零残差逐点合成检查补齐。完整F有限性检查、非连续reshape副本和CUDA分块数值仍需实际内存/时耗和数值核验；这不取得科学执行资格。

复现已有合成检查：在仓库根目录运行 `python -B research_round_20260925_global/code/N13_original_raster_fixed_PR_adapter_draft.py --self-test`。这不会读取SAR数据，不执行科学资格或上传。自测只能验证小规模算术，不能证明科学运行、任务效用或创新。

ROOT已完成八张量源码键、F原renderer/flat gather、strict_gt_zero及固定原生匹配函数的文字核对，见[接口说明](N13_FIXED_PR_ACTUAL_SOURCE_INTERFACE_REVIEW_20261014.md)。真实诊断仍须先完成剩余预锁曲线，再登记并核验实际端点与八张量证明、exact GPU L/F来源、原LOG的abs(L)≤2和native L>0、width2对应既有generate_block_target(P,2)==1原栅格实现及图边界处理、逐对固定374人口/六未匹配者的实际身份、以及原LOG所选位置和完整最终输出重建；不以通用距离≤2替代width2。合成草案不自动满足这些门；不能重建原输出时先定位身份/数值/坐标问题，不开展收益选择器长训。


## 2026-10-14：非零残差与非连续栅格的独立合成补查

[针对性检查代码](code/N13_original_raster_fixed_PR_nonzero_strided_check.py)一次实际执行正常退出0，耗时2.625秒、无stderr。使用非零八张量权重及非连续F[256,5,7]/L[5,7]，独立nn.Module按F[:,y,x]逐点取特征，不复用适配器reshape/gather或其残差函数。七个小例覆盖两种二值比较符、新增域4→12点、原域23点和三种空域；索引、二值与merged_C精确一致，u最大绝对差4.47035×10⁻⁸、L+u最大差1.19209×10⁻⁷，通过预先固定的rtol=10⁻⁶/atol=10⁻⁷。扩大新增域时旧域索引/残差/logit/二值位级不变，所有输入原样。

Host Python3.10.13、Torch2.2.2+cpu、psutil5.9.0，亲和[0,1]、1/1线程、CUDA未初始化。此次检查不覆盖非连续输入跨2048点分块，也不建立大栅格资源、科学Torch2.0.1+cu118、真实模型/renderer/域或导出比较符资格。原适配器及真实检查源/输出保留；发布副本只统一文本换行。没有重跑既有自测、读取科学载荷或改变当前预锁评价。

复现：在仓库根目录运行 `python -B research_round_20260925_global/code/N13_original_raster_fixed_PR_nonzero_strided_check.py --adapter research_round_20260925_global/code/N13_original_raster_fixed_PR_adapter_draft.py`。CPU亲和[0,1]是此检查的显式要求；不具备这两个可用CPU的环境会拒绝，不能把该拒绝解释为模型失效。


## 从条件余量到完整AP检验的最小桥接

独立审阅认为当前374固定配对方案足以先做条件诊断，无须为启动它扩大人口或追加理论。该人口上的改善只能说明新假设值得检验：背景或重复提议、原分数排序下的COCO匹配变化仍可能使完整AP下降。下一项部署评价另预锁两个必要条件：

1. 策略覆盖自然全部18,700 query，区域、K、候选与是否写入仅由推理可用输入决定，不用374配对身份或GT筛选；逐query核对原class/score/order/空输出及K=0行为，未匹配和背景提议也按同一声明策略处理。完整380 GT人口保留六个原生未匹配者，COCO重新执行自身匹配，不沿用374对匹配计算AP。
2. 行动条件明确。只换选点器时，各对照共享选择前生成、冻结且不依赖最终选集合的C；若新结构使C依赖所选集合，须重新登记候选条件，原固定C oracle不再解释其余量。保留旧值/拒绝写入如改成at-most-K，应单列预算与写入条件；核对实际选点、写入、翻转、修复、伤害及不变计数，不称原exact-K收益。

新假设是否有效以同协议完整COCO AP差值判断，固定配对IoU及修复/伤害仅解释差值出现在哪里。此桥接是未来评价条件，未执行新部署策略、未得到AP提升或核心创新；当前预锁曲线、E48停止决定及后续单一诊断顺序不变。


## 2026-10-14：纯统计连接与独立分母检查

[统计模块](code/N13_fixed_PR_diagnostic_statistics_draft.py)已保存为UNBOUND_DRAFT / SYNTHETIC_ONLY。diagnose_instance和diagnose_population显式复用既有纯布尔oracle，调用者提供A/C/G、互斥old_LOG/new_halo、S/K及实际image/GT/query身份；此协议严格S=old_LOG、K=|S|、Ω=old_LOG∪new_halo。候选独立性仅保存调用者声明，不由统计程序认证。模块不加载模型、权重、GT文件或科学资料，不构造width2，也不取得科学执行资格。

输出分别保留四个主分支和at-most-K，统计逐对IoU、像素TP/FP/FN、选点/字面写入/二值翻转/修复/伤害/不变、三项非负条件差、exact-K被迫伤害，以及相对native和原支持的.50/.75/.90/.95双向跨越。区域分为old/new/outside；可选距离带由调用者显式给布尔mask和来源说明，只检查互斥，不认证距离定义。零错误图的coverage为未定义，逐实例均值只用有定义的分母，与总像素加权比例分别记录。candidate_evaluated_points是所提供Ω中的逻辑候选数量，不是实际PR调用、时耗或资源测量；真实运行须另记录这些成本。

匹配与未匹配GT的身份独立保存，禁止重复GT或image/query身份。未匹配者不赋query、K或域；配对IoU均值仅除matched数，可另报配对达阈数/totalGT，不能称AP或召回。可显式锁定matched/unmatched/total三个期望计数；该计数检查不代替真实374+6配对来源。

首次隐藏host合成执行实际退出0，0.375秒，3个手算小例及14项非法输入拒绝通过。独立静态审阅指出原支持跌破.50、字面写入与翻转、宏均值与像素汇总、域外错误的证据覆盖缺口；随后[独立手算补查](code/N13_fixed_PR_diagnostic_statistics_independent_hand_check.py)一次实际退出0，0.306秒，4个案例的525项显式核对通过，未改原模块、未重跑原自测或oracle穷举。检查包括宏IoU=7/12与像素IoU=5/9、3+1身份、old/halo/outside像素错误覆盖2/5、2/5、1/5及宏覆盖1/3、1/2、1/6，以及K=0和被迫伤害/拒绝的分离。以上均是人工布尔值，不是SAR结果。

两次使用host Python3.10.13、NumPy1.26.4，CPU亲和[0,1]、四个线程环境1、-B、CUDA空且未导入Torch；与科学运行环境的相容性仍待实际核验。正常import统计模块只加载标准库，生产输入校验显式抛异常；自测与独立检查显式拒绝-O。

复现独立手算检查（需NumPy、psutil及可用CPU0/1；以下设置在当前PowerShell会话及后续子进程生效）：

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
python -B research_round_20260925_global/code/N13_fixed_PR_diagnostic_statistics_independent_hand_check.py --statistics research_round_20260925_global/code/N13_fixed_PR_diagnostic_statistics_draft.py --oracle research_round_20260925_global/code/N13_fixed_candidate_action_oracle_draft.py
```

真实载体、exact L/F、width2、374+6身份、原LOG完整重建、资源与独立CPU资格均未由这些小例确认。当前701、702 E24已完成独立CPU/ROOT核验，最后703 E24评价接续中；剩余固定E24曲线完成后再登记真实固定PR诊断，不以准备工具代替SAR效用或核心创新证据。


## 2026-10-14：width2作者定义的外部静态核对

ROOT在线核对了[RefineMask固定提交函数](https://github.com/zhanggang001/RefineMask/blob/bf81cac8090115e80c690cd94c4e25a1e6550603/mmdet/models/losses/cross_entropy_loss.py#L117-L145)，未读取运行中的本地绑定来源。对二值原栅格P，令n为零padding的5×5前景数，静态推导得到：

```text
positive_response = max(25P − n, 0) / 25
negative_response = max(n − 25P, 0) / 25
width2 = (P AND n ≤ 22) OR (NOT P AND n ≥ 3)
```

函数先将输入转FP32；第二路对已经padding的P取补，两路严格>.1后标边界1，再以剩余前景标2。由此推得全零P无width2；全一P在常规H/W≥5时为两像素图像边框；孤立单前景不生成完整5×5 halo。以上是二值输入的源码静态推论，不是本地执行证据；也不能套用sameK距离D的空/满不确定性回退。

701 E24独立CPU实际结束、下一702 E24完整来源门通过后，ROOT已在新GPU准入前只读核对实际RefineMask_author_source及N12 v02 run.py/qualify.py/common.py。绑定函数、FP32/CPU调用、padding后取补、严格>.1、标签覆盖和独立整数/25计数与上述定义一致；sameK D的空/满回退没有用于width2。未加载真实P或运行边界计算，真实诊断域仍未核验，当前曲线及停止决定不变。见[本地接口补充](N13_FIXED_PR_ACTUAL_SOURCE_INTERFACE_REVIEW_20261014.md)。


## 2026-10-14：候选生成、原LOG完整重建与统计已接通

新[连接层及合成证据](N13_FIXED_PR_PIPELINE_SYNTHETIC_INTEGRATION_CHECK_20261014.md)先生成不接收G/S/K的固定C，再仅在旧LOG域写回并与必需的原预测参考逐像素相等核对；通过后才让统计读取G，固定S=old_LOG和K=|S|。真实权重/L/F/参考身份、候选独立性、width2、人口、运行时与资源均未由此认证。

72项首次手算检查实际0；独立八像素例/一次错误参考拒绝的106项核对实际0，补齐新增域真实翻转和非空域外保持的合成证据。完整C不能冒充原LOG输出，exact-K强制伤害与at-most-K分别报告；详见总结中的手算表和便携复现命令。没有再运行已有自测，没有读取实时科学载荷，也没有执行新的SAR诊断或报告AP/创新。702 E24独立CPU原32807已单次启动；完成剩余曲线后，再登记真实固定PR诊断。


## 2026-10-14：显式原栅格区域装配接口

[装配层及有限检查](N13_FIXED_PR_NATIVE_DOMAIN_ASSEMBLY_SYNTHETIC_CHECK_20261014.md)不自行实现width2：由实际调用者提供作者回调，以CPU FP32连续native_P[None]和boundary_width=2计算CPU int64标签，标签1形成native_width2，再与old_LOG去重得到new_halo/Omega。49项字面回调核对和13项拒绝只验证阈值、域拆分与存储隔离，不认证作者函数、实际L/域或科学运行时；真实登记须另核对作者身份及独立整数计数。

该输出可提供连接层所需old_LOG/new_halo，真实组合仍未执行。当前702 E24已完成独立CPU/ROOT曲线核验，最后703 E24接续中；不新增训练或改变原评价/停止规则，不把工具准备当成研究效用。


## 2026-10-14：702 E24已核验，真实诊断顺序保持

702 E24 LOG/GEO为67.953519/67.733877 AP，配对差−0.219643点；来源与实际CPU输出引用已ROOT核验，独立数值审阅一致。最后703 E24 GPU原65204已单次启动；其GPU、独立CPU及ROOT完整核验后才登记本诊断。三条E48均负的主比较与停止决定保持，当前未以合成接口或学习曲线提前选择新结构，也未执行374+6真实行动诊断。


## 2026-10-14：人口汇总与距离分层接口已补齐

[接口及有限检查](N13_FIXED_PR_PRECOMPUTED_POPULATION_AND_DISTANCE_INTERFACE_CHECK_20261014.md)新增matched_diagnostics预计算JSON路径，与原matched_instances二选一，复用同一身份/人口锁/分母/距离带键及汇总代码，不再调用diagnose_instance。连接层显式distance_bands仅在原LOG重建门后透传，不构造距离公式。这样可逐对保留JSON并释放栅格；没有测量真实内存或时间收益。

59项/10拒绝、35项小例检查和独立静态审阅通过，仅SYNTHETIC_ONLY；预计算数据、来源、候选独立性及距离身份仍未认证。最后703 E24 GPU实际0，独立CPU原92886已单次启动；实际结束和完整ROOT曲线资格后才登记真实374+6诊断，不提前宣称效果或调整停止决定。

## 2026-10-14：全部固定曲线合格，真实诊断已登记启动

全部三个E24已核验，LOG/GEO均值67.992117/67.781883，配对差−0.210233点；E48主结论及停止不变。实际载体为预锁第一701 LOG E48，人口来自原生.50匹配已核验的374对与6个未匹配GT。真实执行元数据原24101实际0，完整9,579来源、93常规/8失败/101身份登记后，科学原GPU59604单次启动；尚无诊断结果资格。固定K及候选不依赖所选集合；GT只作身份复核及离线统计。距离额外固定为原生in-domain chessboard D≤1、1<D≤2、2<D≤4、D>4/∞四带，空/满为∞；它不替代作者width2。详见[完整曲线与启动范围](N13_FIXED24_THREE_SEED_CURVE_RESULT_AND_REAL_DIAGNOSTIC_START_20261014.md)。实际GPU后还需独立CPU/ROOT资格，不能将启动当成效用、AP提升或核心创新。

GPU原59604已实际正常结束165.468秒，held handle／reaped／原出生身份消失通过；第94项RAW-only终止合并门实际0。独立CPU资格稿已由子agent编写及另一agent静态核对，ROOT确认GPU owner CUDA参数为0并修正UNBOUND稿，确认实际外部ABI；[CPU研发副本](code/N13_fixed_PR_real_diagnostic_cpu_qualification_draft.py)的独立私有执行副本已登记并单次启动，尚未完成资格。CPU预锁实际carrier.numeric_protocol中的支持atol/rtol=10⁻⁴，并另锁公式比较atol/rtol=10⁻⁴；近零符号差按原实用规则记录、不是认证误差界，不用CPU近似L或符号替换GPU候选。CPU元数据原44623实际0，完整9,967来源、94常规／8失败／102身份通过ROOT准入，原CPU4404单次启动，完整资格仍待实际CPU／ROOT读回。
