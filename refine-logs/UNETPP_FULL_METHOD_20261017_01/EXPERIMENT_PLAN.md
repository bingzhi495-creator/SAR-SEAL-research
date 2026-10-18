# 以自主完整UNet++为主体的实例分割方法设计与实验路线

<!-- U51_FIRST_BATCH_AND_U40_RESUMED_20261018_01 -->

2026-10-18 后续：U51首批32图/43匹配实例已actual0导出并接收（117777312 B，worker13.328秒，RSS643702784 B），全量准备32/741图。DGX原配对试点已6真实step通过但不计本机主训。资源双观察恢复后，原U40 2816→3072已启动；已接收端点仍2816，运行不冒记3072，无新增DEV/AP。[当前实际记录](../../research_round_20260925_global/U50_DGX_PAIRED_PILOT_ACTUAL_AND_FULL_INPUT_NEXT_20261018.md)。


<!-- U40_2816_ACCEPTED_CURRENT_20261018_01 -->

2026-10-18 最新已接收端点：2816/37050（3轮+593图，预算7.60%）。原2560→2816任务actual0，经Root原终态/FIRST/来源/配对检查点和轨迹核验接收；两臂各256更新、561实例访问，512次成功step，原任务1060.828秒，未越原资源限额。FULL/PLAIN训练损失均有限且相近，无新增DEV/TEST/AP，不凭−0.234%相对训练损失宣称效用。运行账户293正常/36失败，与模型更新次数分开。下一段2816→3072仅已准备请求，启动仍须双资源观察。U49/U50仅普通源码完成，未实际导出或DGX运行；所有历史失败保留。详见[本段核验及后续](../../research_round_20260925_global/U40_U02560_02816_ACCEPTED_AND_DGX_EXPORT_STATUS_20261018.md)。


<!-- U40_POST_U48_ACTUAL_LIVE_20261018_01 -->

2026-10-18 接续实况更新：09:09:57 UTC 与09:11:48 UTC两次资源观察均达标，free RAM分别38350995456/38238998528 B、整卡2065/1963 MiB。Root已实际启动并登记原配对2560→2816接续任务，17份训练来源及原配方/限额不变，恢复原2560检查点。当前为运行中，尚无该段终态或新配对预算；有效端点仍2560/37050。此前资源不足描述保留为当时观察，不能据启动记为已完成。


<!-- U48_NATIVE_EXECUTION_CURRENT_20261018_01 -->

2026-10-18 U48 实测补记：原 E24 教师在预选 HRSID TRAIN 首图的完整原生100查询推理已实际退出0并接收，耗时26.453秒、分配器峰值约2.11GiB，清理后allocated/reserved0；原2GiB后处理OOM actual1单独核验保留。单图资源修正版只改2→3GiB分配cap，数学/输入/score不变；无新训练/AP/全人口资格。主训有效端点仍每臂2560/37050；新增运行账户为292正常/36失败，不等于训练进度。最新账户到原f35训练失败及未变2560模型父的元数据继承已实际验证，未绑定/启动接续。主训需两次相隔至少30秒的free RAM≥32GiB与整卡基线≤3584MiB观察；当前未达门，17源/共同图序/配方/固定E50预算不改。详见[单图实测](../../research_round_20260925_global/U48_SINGLE_HRSID_NATIVE_CAPTURE_ACTUAL_AND_NEXT_STEP_20261018.md)。


日期：2026-10-17。当前状态：自主完整UNet++卷积资源版与原plain64两臂各2560正式更新已接收，[2560,2816)原尝试因资源门受控终止并保留，待充足余量独立接续；总预算仍每臂50轮/37050更新。N28固定E24完整DEV按原门槛STOP当前路由配方，主方法独立训练。E50私有执行控制草稿与独立静态互接审已完成，尚未实际准备/绑定/数值执行。完整主方法尚无AP或创新成立结论，后文保留形成时记录。

## 1. 本轮设计决定

后续主方法直接以本项目自行组织的完整UNet++结构为主体。保留复合模块、H/4语义与横纵方向分支、原融合层及完整嵌套解码；实例接口只承担实例区分和掩膜读出。不能再仅用提取的一种注意力路由代表用户全部设计。

### 最终候选的收敛决定（2026-10-17，回应已有涨点基础）

直接以已有自研UNet++设计为最终候选主体，不重开一套无关模型。研究问题收敛为：在保留原实例身份与分数的条件下，语义—方向协同的完整图像特征及嵌套细节重建，能否改善全域实例掩膜。它是待验证的统一方法假设，不是已成立的新颖性结论。

| 组成 | 当前落实 | 收益归属与定稿条件 |
| --- | --- | --- |
| 自研复合编码与H/4语义、横纵方向融合 | 使用当前已进入U40正式训练的full卷积资源版 | 相对普通完整UNet++的固定50轮效用成立后，再以容量接近替代、静态方向及固定增益等最少控制判断作用 |
| 完整五尺度嵌套重建 | 保持真实RGB输入、全域残差及原几何 | 与普通对照共享输入、监督和读出权限；不能把访问原图或更大模型的收益自动归给方向机制 |
| 每图共享编码、实例条件残差读出 | 原自然查询/类别/S0/顺序保持；每图一次主干，逐query小头 | 属当前实例适配实现；已有query/dynamic-head先例，不能仅凭该组合声明首创 |
| 历史已涨点的boundary/frequency_boundary | 保留原收益记录与复现入口；不改当前已锁主训练 | 这些语义F1正信号支持复用优先级，尚未归属到确切full/new1结构。若后续检验辅助监督，普通主干必须获得同样监督，并保留无辅助版本作对照 |

当前full明确关闭且不实例化巨大旧Transformer，不称开启该层的旧模型原样复现。已锁定训练不因本次定稿讨论重启或改动；先完成现有完整候选/普通主干公平主实验，再按原预锁收益门槛开展机制、重复及跨集证据。主效用若为负，只能限定为当前预算和配方的结果；若为正，也不能跳过先例差异与重复验证。最终论文可以围绕一个统一核心组织贡献，不预先承诺三个独立创新。

N28固定24轮/17784更新及完整DEV已完成。SELECTIVE比WHOLE_NODE高0.366014AP点，但低于强R及N26预锁参考，门槛Delta−2.516096，Root决定STOP当前路由配方，不扩种子、预算或融合。其证据限定冻结dense特征路由头；完整骨干U40独立按已接收原状态接续，不能把N28结果当作full成败。

## 2. 已有正收益应被承认，也要正确归属

以下来自现存原始JSON，阈值均0.5；本轮只读取数值记录和训练历史，未重新加载或哈希权重。

| 历史数据与版本 | seed | clean F1 % | 改进F1 % | 差值：百分点 |
| --- | ---: | ---: | ---: | ---: |
| HRSID frequency+boundary | 42 | 76.458428 | 78.859378 | +2.400950 |
| SSDD boundary | 42 | 84.114113 | 85.572212 | +1.458099 |
| SSDD boundary | 43 | 86.460553 | 86.291999 | -0.168554 |
| SSDD boundary | 44 | 85.026929 | 86.813455 | +1.786526 |
| SSDD三种子均值 | — | 85.200532 | 86.225889 | +1.025357 |

SSDD两组F1的均值±样本标准差分别为85.200532±1.182814与86.225889±0.623257个百分点；配对差值的样本标准差为1.046916个百分点，三个种子中两个为正。HRSID此项仅一个种子，不能据此声称重复稳定。

HRSID两份记录均为50轮、每轮1821批，共91050批；3642图训练，1962图评价。SSDD六份记录均为50轮、每轮464批，共23200批；928图训练，232图评价。每对记录内的checkpoint摘要一致，是日志关联证据，本轮未验证现存权重文件内容或当时冻结源码。

原记录在D:/SAR/UNet++/runs/metrics/，HRSID文件为hrsid_clean_unetpp400_full_eval.json与hrsid_frequency_boundary_unetpp400_full_eval.json；SSDD为ssdd_{clean,boundary}_unetpp256_full[_seed43或_seed44]_eval.json。对应runs/logs/内将_eval改为_train；三种子汇总为runs/analysis/ssdd_boundary_vs_clean_multiseed_summary.json。

这些记录明确为语义分割，instance_ap_reported=False；miou字段按既有定义解释为前景IoU。它们说明自主UNet++系列已有实际正信号，不能写成“从未涨点”；它们对应clean、boundary、frequency_boundary类，不是full/new1确切复合模块的公平消融，也不是COCO实例AP。旧full/new1的完整结构—训练—结果关联仍需补齐；未定位到不等于未训练或失效。

## 3. 核心问题与待检验主张

已有N27在当前固定FIT741/1661匹配集合发现，强R剩余像素错误73.8521%处于固定可写域外。这只说明该域内纠正的几何限制，支持检验能够全域读写、保持细节的完整图像结构；不能据此证明方向分支、注意力或AP收益。

唯一主要效用假设：在相同真实图像、实例身份、读出头、监督和训练预算下，自主完整UNet++的语义—双方向融合与分层嵌套重建，比普通完整UNet++提供更有用的实例掩膜特征。

需要排除的主要解释是额外容量/计算、原图访问、辅助监督或评分排序。主效用未建立前，不同时引入新的分支选择器、多层query反馈、频率边界联合损失和采样模块。已有正信号版本保留为后续可复现参考，不按零件数量组织创新。

## 4. 主体结构与实例接口

共享图像主干U(I)保留full中以下实际路径：

1. 双卷积/BN、通道与空间门控、残差、ReLU及z+SimAM(z)的编码复合块。
2. H/4的CBAM语义支路，以及由独立普通卷积输入生成的x/y DSConv支路，经原con2拼接融合。
3. 原五尺度嵌套解码、独立上采样边、低分辨率注意力解码块和最高分辨率普通卷积节点。
4. 最终CONV0_4输出作为共享高分辨率U。原二值语义输出不能充当所有实例的预测。

冻结实例载体提供自然查询e_q、M200、原实际GPU原生logit L_q、类别、分数和查询顺序。小型公共读出产生delta_q，输出L'_q=L_q+Render(delta_q)。MVP采用现成的原生L，不假设已存在完整强R的logit；零残差首先复现原生预测，强R另作固定参考。若未来选择强R作为残差基底，必须先独立恢复并核对其精确logit/渲染关系，不能把R的二值RLE当作原始logit。

每张图只运行一次完整主干；每query只运行同一小头。小头使用U的公共投影、e_q投影及原M200，末层零初始化；按query分批，预测必须与整批及置换后的对应结果一致。查询不反复执行或改写大主干。所有100个原查询、类别、S0分数和顺序保持，不新增筛选/NMS或连通域转换；全域残差经已核验的实际图像几何渲染后才加到L。

新主干需要真实影像。现有200×200单层F缓存不能冒充完整UNet++输入。拟采用与冻结载体相同原图及几何的400×400图像分支；输入预处理、填充、裁剪与原尺寸对应必须先由实际试验接齐，未接齐不启动训练。

适配器返回与U相同网格的残差，拟为400×400；既有N28训练核心的200×200形状契约不能直接套用。需要独立固定400→原800填充网格→实际裁剪→原图尺寸的渲染入口，并核真实几何和零残差身份；不把先降到200或只改shape断言当作已经验证的接线。

整个backbone仍包含旧语义输出头、未调用的普通解码声明，以及旧up块被旁路的bn1参数。它们不都处于实例loss的祖先图内。M0应从实际forward/反向核对有效参数集合，在正式训练前固定其名称、梯度检查和优化器集合；旧输出头不进实例损失时明确排除。分别报告分配、可训练和实际更新参数，不能沿用“全部requires_grad参数必须有梯度”而误判，也不能在训练失败后临时忽略意外缺梯度。

## 5. 原模型保留、显式改版与工程限制

| 内容 | 本轮处理 |
| --- | --- |
| 上述复合块、方向融合和完整嵌套解码 | 主体保留，逐模块核对真正进入forward与梯度的路径 |
| 原始full/model_new1及旧权重 | 保留原件；任何修复均在独立版本 |
| full中conv1(x)覆盖前置BN/ReLU | 先保留实际旧forward作为明确版本；preact修正版单独比较，不混算结构创新 |
| 原按HW设embedding的Transformer | 单独保留为历史配置；本机候选显式use_x2_transformer=False且instantiate_x2_transformer=False，不能称忠实复现开启该层的旧配置 |
| model_new1的up_x选择性传递 | N28独立检验；不未经对照就合并两套解码组织 |
| 旧频率/边界版本 | 保留已有收益与复现入口；若加入实例训练，普通对照获得相同辅助监督 |
| 分支tap条件融合 | 仅后续候选；先用普通相同信息读出建立主干效用 |

原full把[B,256,HW]送入默认序列优先Transformer；其序列轴实际是图像batch，不能叙述为已实现空间自注意力。512输入时E=16384，主要矩阵约6E²=16.1亿参数，仅FP32权重约6GiB；该层不符合当前本机4GiB分配限制。关闭且不实例化是显式资源版本，不是已证明不重要。

DGX现有SSH已于本轮只读连通。GB10统一内存可用约71.48GiB；用户llama-server仍占用约36.22GiB，另有归属未核的CPU任务。它不是空闲机器证明，不停止用户进程、不承诺完整配置可训；实际资源试验后再安排任务。当前Root仍统一拥有数值执行，不因双机可达自动并行启动。

## 6. 最少必要实验

| 块 | 必须回答 | 方案与判断 |
| --- | --- | --- |
| B0 接口/资源 | 完整主干能否作为共享实例特征，零增量是否准确 | 真实图像、全部原查询零残差复现；一次主干/图；查询分批与置换；批次隔离；有效模块梯度；渲染/GT坐标；记录峰值与耗时。未通过只修工程问题 |
| B1 主效用 | 用户完整结构是否提高当前实例分割 | 普通完整UNet++与full卷积资源版，均给相同真实图像、相同实例头、全域权限和原匹配监督；同预算固定端点评价。冻结原生/强R作为参考 |
| B2 原因排查 | 收益是否由所述结构而非规模或监督带来 | 在获得效用后依次做容量/计算接近的普通三路或静态方向强替代、复合块替换；保留相同图像/辅助监督。报告实际参数与成本，不能用仅参数相等代替信息相等 |
| B3 重复与外部范围 | 效果是否稳定且可迁移 | 按预锁门槛追加种子；正式评价及至少另一公共实例数据集，区分固定零调参迁移与目标集重新训练 |

头部实例条件机制可采用已有动态/查询读出思想，不作为独立首创。[QueryInst](https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Instances_As_Queries_ICCV_2021_paper.pdf)已有动态掩膜头；[BlendMask](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_BlendMask_Top-Down_Meets_Bottom-Up_for_Instance_Segmentation_CVPR_2020_paper.html)已有共享密集basis与实例信息融合；[DSCNet](https://openaccess.thecvf.com/content/ICCV2023/html/Qi_Dynamic_Snake_Convolution_Based_on_Topological_Geometric_Constraints_for_Tubular_ICCV_2023_paper.html)已研究动态蛇形算子与多视图融合；[O2Former预印本](https://arxiv.org/abs/2506.11913)已在SAR船实例上结合方向敏感表示和多尺度query。因此，方向＋query或共享主干＋动态头本身不能作为新颖性结论。

直接先例已更新至O2Former的JSTARS2026正式出版元数据及DACF-net出版社可检索的方法/实验段落；O2Former具体配方仍来自arXiv v1，DACF-net完整公式和补充材料未读。方向、多尺度query、跨层调制和残差融合均有直接先例。详见[查新与证据边界](../../research_round_20260925_global/U40_DIRECT_PRIOR_UPDATE_20261017.md)，不能用标题差异或非统一协议数字宣布新颖或胜出。

## 7. 预算、初始化与评价规则

新整网不能沿用N28小头24轮即视为充分训练。拟定主配方为两臂相同50个完整FIT轮次：741更新/轮、37050更新/臂；AdamW lr3e-4、wd1e-4，FP32；固定图序和原实例匹配，按图聚合实例损失，一图一次更新，共享主干图保留至该图全部实例回传。正式训练前由B0实测成本并形成完整执行预锁；如资源需改预算，两臂统一、在看DEV前另行记录，不预先承诺GPU小时数。

固定新种子202610173101、202610173102、202610173103，首先只运行3101。不同结构的公共层采用同源初始化，新增层来源单独记录；所有主对照采用同一初始化资格。不能把已在完整928图上训练的旧权重直接用于当前741/187内部划分而未核对图像重叠。旧权重仅在精确构造、数据谱系及当前留出无泄漏均确认后才进入主实验，否则共同从头初始化，并明确该范围不等于复现旧最佳初始化。

主要指标为完整内部DEV的COCO mask AP，辅以AP75/AR、可计算尺度指标、固定匹配几何和像素FP/FN；固定分数首先隔离几何收益。N26质量乘分只作为另项参考，不能先融合后把评分收益归给结构。全部种子均保留，不挑最好结果。

阶段效用门槛沿用既定原则：相对同预算普通对照与预定强参考中较高者，首组提高至少1AP点且高于配对对照，才优先推进其余种子；小正收益最多补第二组，再按预锁规则判断第三组；非正结束当前配方扩展，未充分收敛时只能限定到该预算。N26三种子均值69.53736694仅是共享冻结载体上的额外强参考，不包装成同预算配对统计量。

## 8. 机制与论文主张的条件

z+SimAM(z)在当前ReLU后包含输入依赖幅值放大，应以固定增益或空间均值门控作替代，区分空间建模与幅值效应。方向分支应用静态条带或普通分支排查；采样、Transformer和边界监督不同时叠加。

当前可写的主张是“提出并实现以自主完整结构为主体的实例适配候选”，不能写成其已优于强方法。只有主效用、直接先例差异、重复稳定性、最少机制控制和跨数据集范围得到支持，才围绕核心机制组织论文贡献。已有语义收益是选择此研究基础的证据，不替代新的实例验证。

相关历史结构与归属以仓库research_round_20260925_global/UNETPP_MODULE_MODIFICATIONS_CONFIRMED_20261017.md及UNETPP_LEGACY_PROVENANCE_AND_CONTAINER_EVIDENCE_20261015.md为准。同步只含自写设计、源码草稿和核验摘要，不包含原数据、全标注、权重或运行资格链。

## 9. 已完成的研发边界（2026-10-17补记）

三项tiny真实nn CPU检查实际通过并经Root接收，包含每图一次主干、分批/置换、精确零残差及有限非零反向；不包含原full、真实影像、400渲染或AP。原完整卷积主体的feature wrapper及下一CPU64两项检查已实现，仅静态/AST。明确图像batch=1以避开旧DSConv批次索引问题，直接返回CONV0_4并跳过旧语义头；不暗改原算子。完整执行与资源/几何资格仍待实际验证。详见[结果与代码导航](../../research_round_20260925_global/UNETPP_SHARED_ADAPTER_CPU_RESULT_AND_FULL_FEATURE_STATUS_20261017.md)。

## 10. 真实full64检查实际通过后的范围

两项原完整卷积特征路径CPU检查已实际正常结束并经Root接收：与原forward的Identity尾头特征逐元素精确相等；245个预先声明的有效参数张量均有有限梯度，四个旧语义头零调用、18个BN buffers保持。总分配67303293参数，有效53122947，原失活14180346；没有优化器或训练。400真实影像/几何、train态BN、内存和AP仍未验证。[结果与范围](../../research_round_20260925_global/U30_REAL_FULL_FEATURE_CPU64_ACTUAL_RESULT_20261017.md)。上一节的未执行描述保留为此前研发快照。

新400分支以共同从头训练为默认，可独立定义两臂完全相同的原图RGB/255输入，不强依赖teacher未恢复的光度默认值；但必须显式锁定实际原图身份、内容整数尺寸、插值/Pad、400映射以及原尺寸GT/实例渲染，不能声称已复刻teacher输入或旧最佳训练配方。具体执行合同仍须在实际B0前固定。

完整真实图像B0已形成[固定候选合同](REAL_IMAGE_B0_DESIGN.md)：新分支共同RGB/255、保留两次插值与原800整数几何、显式antialias=False，检查EXIF/GT方向一致性、原100查询零残差和真实多GT共享图反向；尚未执行。该合同经一次独立静态质疑后补齐方向和插值缺口，不依赖猜测teacher光度默认值。

## 11. 普通主对照与真实输入收口

普通主对照选原plain64 model1（与历史clean32/GN分开），不称参数/计算配平；M100从冻结teacher真实补采，不从L反推。新几何及plain特征接口已完成仅AST，真实补采尚未执行。[来源/缓存缺口/逐轮状态及代码](../../research_round_20260925_global/U31_MAIN_CONTROL_AND_NATIVE100_INPUT_PROGRESS_20261017.md)。后续资源试点拟两次一次性真实optimizer更新，按pilot计数而非正式主训练，实际源与预锁尚未完成。

## 12. 真实100查询输入已取得，400两步资源试点待执行

U32单张真实FIT的100查询M/F已实际退出0并经Root接收，完整原L/e/class/GT/匹配/S0/输出顺序与锚点精确一致；原任务22.297秒，无训练或AP。它解决B0输入缺口，尚不授予400/全FIT训练资格。

U33拟先实际运行几何解析与原plain64特征公共合约，再依次对PLAIN/FULL执行全部100查询零残差、非零Q2分组/置换、每臂两次一次性AdamW更新。真实9GT按4/4/1分块、每图一次编码；第二次包含已建立优化器状态。试点头初始化跨臂精确相同，主干只记录种子、不声称公共层同源；试点参数不复用于正式主训练。固定FP32和既有资源限制，不以单图通过代替741图/收敛/效用。[实测摘要与试点合同](../../research_round_20260925_global/U32_NATIVE100_CAPTURE_ACTUAL_AND_U33_REAL400_PLAN_20261017.md)。前节未执行表述保留为时间线。

## 13. chunk4真实资源失败与独立chunk1修订

U33两CPU合约、普通臂400零残差与非零分组比较、第一步AdamW更新实际通过，第二步已有148个状态时backward OOM；固定分配上限4GiB，物理卡约24GiB。FULL未执行，失败保留，不作总体方法判断。U35只对两臂共同改训练chunk4→1，九GT各1/9，维持每图一次encode/update、两步、FP32、相同输入/监督/优化器/资源；eval25×4和Q2判据不变，源码尚未执行。正式公共算子104状态初始化草稿已形成，只静态，完整seed202610173101，未构造/复制权重或主训练。[结果和修订合同](../../research_round_20260925_global/U33_REAL400_OOM_RESULT_AND_CHUNK1_REVISION_20261017.md)。

## 14. 逐实例普通臂两步通过，自主主干仍需资源修订

U35原actual1已核验保留：PLAIN两步和臂间CUDA双0均通过；FULL真实400 eval/零残差/非零分组通过，但首次train encode在CONV0_3 BN处4GiB配额下OOM，没有FULL更新。U36拟两臂统一FP32保存张量CPU暂存，不改结构/分辨率/块/优化器/监督、不作checkpoint重算，先小型GPU两步梯度等价再真实图，当前未执行。共有16算子104状态初始化仍仅静态；正式主训练未开始。单次VJP只另存候选，不与本轮暂存叠加。[最新实测与设计边界](../../research_round_20260925_global/U35_REAL400_PARTIAL_RESULT_AND_FP32_CPU_ACTIVATION_DESIGN_20261017.md)。


## 15. 以已有自主结构为主体：真实400两步已通过

U36原任务实际退出0并完成外部元数据接收。两臂相同FP32、chunk1、AdamW及CPU激活暂存，均完成9实例真实图像两次更新；独立小模型普通/暂存各两步，损失、逐参数梯度、BN和Adam状态均以torch.equal比较通过。共4小模型+4真实图试点更新，不追认为正式训练；共有104状态初始化仍未实跑。FULL第二步245个有效参数张量梯度有限且非零，GPU分配峰值2119900160字节，采样RSS峰值13953851392字节。该图第二步5.844秒不能外推整轮耗时或加速收益。

用户提出“直接基于这些进行最终方法设计”的方向已落实为主线：保留自主复合编码、H/4语义与双方向融合和完整嵌套解码，采用同一每图共享实例读出。历史语义收益支持选用这一基础；尚不证明full全部组件对当前实例AP的因果贡献。普通plain64得到相同输入、实例条件、全域写入权限、监督与训练预算，不先融合评分收益再归因给结构。

下一项不再增加新模块：先实际验收U37完整FIT741/1661真实JPEG/几何和U34公共104状态初始化，再在共同预锁图序上维持Adam状态完成跨图资源试点。保持U36已通过的反传方式，VJP另立候选。全741的原图及几何检查逐图流式执行，不以少数资源图替代全人口来源核验；少数连续更新也不能保证全741稳态。

原匹配reader仅返回G个实例的M/L/e；native_L来源虽含100查询，不自动拥有全部M100、S0或其实际一致性证明。正式DEV全候选评价所需来源补采须独立接齐，不能由U32单图推定全人口具备。新的资格更新必须事先明确为丢弃试点或正式预算，不事后改称主训练首轮。完整50轮主训练尚未启动，效用、重复与跨集条件保持不变。

[U36真实结果与成本](../../research_round_20260925_global/U36_REAL400_TRAINING_RESOURCE_ACTUAL_RESULT_20261017.md)。前节“尚未执行”为当时快照；U33/U35失败与数值路径均保留，没有重写或重跑来消除失败。

下一项已保存[U38最小验证方案](../../research_round_20260925_global/U38_FULL_METHOD_READER_AND_RESOURCE_GATE_DESIGN_20261017.md)与[CPU组合函数草稿](../../research_round_20260925_global/code/U38_FULL_METHOD_CPU_QUALIFIER_DRAFT_20261017_01/DESIGN.md)，尚未执行；先全741流式读取和共有初始化，再最小跨图连续更新，不要求全FIT M100/S0。

[U38 CPU入口/工作体](../../research_round_20260925_global/code/U38_FULL_METHOD_CPU_ENTRY_DRAFT_20261017_01/DESIGN.md)已形成，严格741原JPEG引用有序摘要接线；只完成静态检查，不代表全人口读入或共同初始化实际通过。


## 16. 全FIT读取及共同初始化已实际通过

U38原实际退出0并外部接收：完整741图/1661匹配实例的原JPEG/RGB400/几何与原GT/query关联均已检查；104公共状态（68参数张量+36BNbuffer）及共享实例头按seed202610173101实际同源，未映射/失活边界保留，模型随后释放。CPU任务289.953秒，完整读取166.826秒、初始化0.864秒；未初始化CUDA、无NN前后向/优化器/训练/AP。此结果不是由静态代码推定通过，Root也没有独立像素重演。[完整结果与范围](../../research_round_20260925_global/U38_FULL_FIT_RGB_AND_COMMON_INITIALIZATION_ACTUAL_RESULT_20261017.md)。

完整人口原shape572种、G1至29，资源代表序列按预定metadata规则为54(G1,190×303)→710(G27,327×489)→54。下一项[U39跨图资源函数](../../research_round_20260925_global/code/U39_FULL_METHOD_CROSS_IMAGE_RESOURCE_DRAFT_20261017_01/DESIGN.md)已形成但未运行：两臂各3次丢弃试点更新，Adam/BN跨图保留，所有27匹配实例按1/G参与。CPU资格和此前U36单图两步均不代替此跨图更新，也不计入正式50轮；不为证明绝对无泄漏扩展大量资格循环。前节未执行描述为此前时间线。


## 17. 跨图执行源码与正式连续训练实现

[U39入口/工作体](../../research_round_20260925_global/code/U39_FULL_METHOD_CROSS_IMAGE_ENTRY_DRAFT_20261017_01/DESIGN.md)已静态复核，调用同一已发布资源函数；source只准入实际54/710六个张量载体，保留全部741元数据/JPEG身份，不以未用缓存重复扫描代替试点。当前尚未运行，不授六次更新或长训资格。

[U40连续训练实现设计](../../research_round_20260925_global/U40_FULL_METHOD_CONTINUOUS_TRAIN_IMPLEMENTATION_DESIGN_20261017.md)建议薄层复用U39单图数值循环，完整保存/恢复模型、BN、Adam、各臂RNG和共同图序游标，两臂同段同累计前缀。拟在固定Python3.8下以独立random.Random(202610173101)按轮打乱0..740，预生成50轮共同图序；此为正式执行预锁前的明确实现选择，尚未生成实际序列或启动主训练。分段长度依据U39实测与保存成本确定，256仅候选；预算仍每臂37050更新，不由分段改变。状态保存/恢复检查纳入正式段，不另加资格叶，不宣称未测全轨迹逐位可重放。已有资源试点不计入正式预算。


## 18. U39实际跨图结果与U40连续训练核心

[U39原实际退出0与Root接收摘要](../../research_round_20260925_global/results/U39_CROSS_IMAGE_RESOURCE_ACTUAL_20261017.md)：两臂各3次，图序小图/27实例图/小图，保持真实Adam/BN；合计6次丢弃更新、正式预算0。40.610秒，实际CUDA分配峰值2,047,209,472字节，进程RSS峰值13,980,127,232字节，各臂和最终显存释放双0。不以短检查替代长程稳定性或AP证据。

[U40训练核心](../../research_round_20260925_global/code/U40_FULL_METHOD_CONTINUOUS_TRAIN_DRAFT_20261017_01/DESIGN.md)复用已运行U39单图循环，全局step传入不重置；保存完整CPU模型/BN/Adam/RNG与共同顺序游标，真实CPU序列化签名回读和恢复检查纳入正式段。Root及独立静态审阅未发现确定阻断；3项纯元数据测试通过，未运行真实state恢复/正式更新。待接入entry/binder与锁定分段及磁盘保留预算。首段验证实际保存，后续段验证实际恢复，不另增资格叶。主方法与普通原版UNet++保持同输入/监督/顺序/初始化规则和每臂50轮37,050更新。历史语义涨点不等于完整实例AP提升已成立。


## 19. 正式50轮执行预锁与入口

[U40执行入口与工作体](../../research_round_20260925_global/code/U40_FULL_METHOD_CONTINUOUS_TRAIN_ENTRY_DRAFT_20261017_01/DESIGN.md)已完成Root静态审阅及10项纯元数据测试，尚未启动实际正式段。以已有自定义完整卷积主干为主体，与普通原版主干使用同一实例读出和完整FIT741/1661匹配监督；首种子202610173101、FP32/RGB400、query1、AdamW固定配方、每臂50轮37,050更新。

CPython3.8.17的独立Random按轮打乱完整741索引，Root纯元数据实际生成共同37,050图序摘要`2afadef62eac9f797c125fa105a3a28f94cdbdd9db4b5b5a2b018d4f0d5f194e`。预锁每段256更新、共145段（末段186），允许跨epoch，段间严格恢复全部模型/BN/Adam/各臂RNG与原始顺序游标；不是重复从头训练。首段[0,256)，图序和端点不依赖DEV或损失选型。

分段依据U39实测时耗及1800秒单任务硬限额，尚不能保证所有图的全程资源行为；真实保存和续段恢复在正式预算内验证，不再单列资格叶。资源维持CPU0/1两线程、4GiB分配、8GiB整卡、16GiB RSS与16GiB可用RAM下限。冻结保留全部检查点；磁盘首段按2GiB/双臂状态×145+32GiB共322GiB保守预留，后段根据首个Root接受双臂状态的实际字节数核剩余保留预算，不把估计体积当事实。仅两臂相同段实际完成、权重与trace写入、终态与来源核验后推进共同前缀。无自动重试，失败的已返回更新与单边产物保留。

固定50轮完成后才进行完整DEV187/18,700查询评价，原候选/类别/S0/顺序不改；普通对照、原生及强R参考一起报告，不将评分收益归给结构。原N28 DEV reader源码已声明完整M100/e/F和原native-L读取路径，可作为评价接线的复用候选；仅静态确认，实际文件读取与零残差原预测校准仍须在固定评价时核验，不冒称全DEV已完成或可从L反推M。


## 20. 首个正式段实测与固定评价研发

[首个正式段安全摘要](../../research_round_20260925_global/results/U40_FORMAL_U00000_00256_ACTUAL_20261017.json)已接收，两个主干均完成256更新并保存完整CPU状态，共512次，属于正式预算；不是50轮完成或AP收益证据。首次只验证保存/回读，续段真实恢复需后续原任务核验。实际双臂检查点合计1,262,052,812字节，后段按此实测字节计算剩余保留预算加32GiB，不再以首段2GiB估计冒称实际体积。下段[256,512)已启动，配置/图序/总预算不变。

[DEV RGB桥实现](../../research_round_20260925_global/code/U40_FIXED50_DEV_RGB_READER_IMPLEMENTATION_DRAFT_20261017_01/README.md)已静态审阅并通过3项纯元数据测试；保留原6对象和自然100查询轴，逐实际query id映射导出，不使用GT/facade。全部187真实JPEG引用、数值读取及固定E50恢复/原预测校准仍待未来评价，未新增数值资格叶或中途AP。

[直接先例更新](../../research_round_20260925_global/U40_DIRECT_PRIOR_UPDATE_20261017.md)补充：O2Former已有JSTARS2026正式记录，其本轮可读数字来自arXiv v1且离岸/近岸分列、500轮配方，不能和内部DEV直接判胜；DACF-net已用高层语义增益、Hadamard调制和残差，基础版与换骨干/Soft-NMS增强版必须区分。当前完整结构的具体差异仍为待检验候选，先完成冻结full/plain效用检验，再按原计划条件实施最少替换对照，不因此改动正在运行的主方法。


## 21. 第1个完整FIT轮次已完成，评价元数据已接齐

[512,768)第三正式段原actual0已Root接收，两臂各累计768次更新、共同游标1轮+27图；本段每臂匹配GT访问624次（重复训练访问，非新增标注），原任务1042.703秒。worker恢复/保存回读及两臂CUDA清0通过，Root没有独立神经重放；无DEV/AP。[安全摘要](../../research_round_20260925_global/results/U40_FORMAL_U00512_00768_ACTUAL_20261017.json)。

利用原N07报告和N16四份JSON采集清单，实际完成内部187图/18700查询的上下文及原JPEG候选引用提取，普通RGB桥的真实metadata函数检查通过。没有读JPEG、tensor、GT正文或权重，也未执行推理/COCO；不是图像/像素资格或额外科学叶。[元数据准备摘要](../../research_round_20260925_global/results/U40_FIXED50_DEV_METADATA_PREPARATION_ACTUAL_20261017.json)。后续正式固定端点仍需实际Source闭合、RGB/全部zero校准与资源/COCO验证。

为更早取得一组已有路线的完整端点，在完整主干段间穿插N28剩余固定24轮训练；当前第20→21轮原任务已启动，仍仅第20轮14820次/臂已接收。模型父和运行记账父已分别核对，不改变任一配方/预算/端点。N28不替代完整骨架主效用，U40仍从各自768状态与原ORDER接续。


第21轮路由对照原actual0已接收，两臂各累计15561更新，原任务354.813秒，完整模型/Adam/RNG恢复由worker检查通过；Root仅核原终态、字节与元数据关系，无tensor重放。[实测](../../research_round_20260925_global/results/N28_E21_CONTINUOUS_TRAINING_ACTUAL_20261017.json)。第21→22轮原任务已从此状态启动。仍固定E24、17784更新才完整DEV评价，无中途AP；评价原runner已存在，正补最薄私有绑定/接收控制，不新增数值pilot。

第22轮原actual0已接收，两臂各累计16302更新，原任务354.312秒，最低可用RAM约31.74GiB、整卡观测峰值3063MiB；模型/Adam/RNG恢复检查由worker完成，Root未重放tensor。[实测](../../research_round_20260925_global/results/N28_E22_CONTINUOUS_TRAINING_ACTUAL_20261017.json)。第22→23轮原任务已启动，仍只在E24/17784更新后评价；U40完整主干仍为各768更新，原状态和50轮预算不变。

N28原固定E24已实际退出0并接收，两臂各17784更新，原任务352.843秒，模型/Adam/RNG恢复由worker检查；Root只作原终态/字节与元数据核验。[训练实测](../../research_round_20260925_global/results/N28_E24_CONTINUOUS_TRAINING_ACTUAL_20261017.json)。首个47图完整查询DEV预测已沿原受控入口启动；后续固定47/47/47/46四分片，再统一187图/380GT/18700查询CPU COCO，原native/R完整12指标必须复现，不平均分片AP、不筛选查询、不增补teacher或临时调参。当前无预测接收或AP结论，U40各768更新仍保留。

N28固定端点及完整COCO已实际完成，原native/R全部12指标精确复现；[结果与限定几何解释](../../research_round_20260925_global/N28_FIXED24_FULLDEV_RESULT_AND_FULL_METHOD_CONTINUATION_20261017.md)。主U40已从各768启动[768,1024)，没有授予未完成更新数；科学源码和原50轮顺序不变。E50最薄外层也已完成未执行草稿与独立静态审阅，实际恢复、像素与COCO仍待端点。

第四正式段[768,1024)已实际退出0并接收，两臂各1024更新（第1轮+283图），本段每臂556匹配GT访问是重复训练访问；原owned1008.344秒，worker956.890秒，模型/Adam/BN/RNG恢复与保存回读检查通过，Root未重放tensor/优化器。[实测](../../research_round_20260925_global/results/U40_FORMAL_U00768_01024_ACTUAL_20261017.json)。第5段[1024,1280)原任务已启动，尚不授予1280完成数；原科学源码/顺序/50轮端点不变。E50控制草稿完成6项纯元数据行为检查及独立静态互接审，仍未执行准备或绑定，不构成任何科学资格。

第五正式段[1024,1280)原actual0已接收，两臂各累计1280更新（第1轮+539图），本段每臂604匹配GT访问为重复训练访问，原owned1010.188秒、worker961.188秒；模型/Adam/BN/RNG恢复与保存回读检查通过，两臂结束CUDA allocated/reserved均0，Root未重放tensor/优化器。[实测](../../research_round_20260925_global/results/U40_FORMAL_U01024_01280_ACTUAL_20261017.json)。第6段[1280,1536)已从原状态启动，源码/顺序/50轮端点不变，没有提前DEV/AP。

第5段实际终止接收后、续段启动前，Root对五段已接收JSON训练trace汇总[前1280更新loss](../../research_round_20260925_global/results/U40_FIRST1280_TRAINING_LOSS_MONITOR_20261017.json)：全部有限，两组接近；第5段均值普通0.038416、自主0.038289。它是匹配训练实例的拟合监测，不能说明全部100候选假阳性、DEV/AP、收敛或创新，不用于改配方/选轮次。

[U41单图原生采集接口](../../research_round_20260925_global/code/U41_UNBOUND_NEW_IMAGE_NATIVE_CAPTURE_DRAFT_20261017_01/README.md)已实现并完成3项标准库接口测试、独立静态审阅。它区分最终e与旧h，保留原自然查询、导出顺序/评分及实际无GT图像补边路径；尚未运行像素/NN、取得新人口或跨集资格，不改U40。

第六正式段[1280,1536)原actual0已接收，两臂各累计1536更新（第2轮+54图），每臂568匹配GT训练访问，原owned986.234秒、worker940.125秒，恢复/保存回读通过，CUDA allocated/reserved清0；Root无神经重放。[实测](../../research_round_20260925_global/results/U40_FORMAL_U01280_01536_ACTUAL_20261017.json)。原[1536,1792)已从各自完整模型/Adam/BN/RNG及同一ORDER接续启动，原50轮端点不变。[第6段训练loss](../../research_round_20260925_global/results/U40_U01280_01536_TRAINING_LOSS_MONITOR_20261017.json)全部有限，同图平均普通0.035898、自主0.035758，仍接近，不支持AP、收敛或创新主张，不用于改配方/选轮次。

[U42单图像素/COCO栅格检查草稿](../../research_round_20260925_global/code/U42_UNBOUND_HRSID_PIXEL_MASK_CHECK_DRAFT_20261017_01/README.md)完成3项标准库接口测试和独立静态审阅，尚未真实解码/栅格运行或取得人口资格；保留原实例、面积差异与越界坐标诊断。teacher上游公开初始化来源仍须实际绑定摘要与官方来源对应，不能由COCO文件名或兼容性推断未见数据泛化；[证据范围](../../research_round_20260925_global/N16_FORMAL_TEST_AND_CROSS_DATASET_EVIDENCE_READINESS_20261014.md)。

第七正式段[1536,1792)原actual0已接收，两臂各累计1792更新（第2轮+310图），每臂563匹配GT训练访问，原owned986.109秒、worker937.172秒，完整恢复/保存回读通过、CUDA allocated/reserved清0。[实测](../../research_round_20260925_global/results/U40_FORMAL_U01536_01792_ACTUAL_20261017.json)。[训练loss监测](../../research_round_20260925_global/results/U40_U01536_01792_TRAINING_LOSS_MONITOR_20261017.json)同图均值普通0.037304、自主0.037189，全部有限但接近，不支持AP或创新主张。原[1792,2048)已从各自完整状态及同一ORDER启动，50轮端点不变。

在原终态/Root接收后、下一科学任务前完成[teacher初始化多指纹核对](../../research_round_20260925_global/results/M2F_COCO_INITIALIZER_ENDPOINT_FINGERPRINT_LINKAGE_20261017.json)：当前FIT/N13/N16五处teacher引用相等，加载回执追到原COCO initializer，本地完整SHA匹配绑定ref，官方前缀/长度/首4MiB指纹和MD5-ETag字串比较均一致。它支持公开端点来源关联；[全流尝试](../../research_round_20260925_global/results/M2F_OFFICIAL_COCO_REMOTE_BYTES_WITNESS_20261017_01.json)仍未完成，ETag不被等同服务端MD5，完整远端SHA/独立训练谱系/未见数据泛化未成立。没有NN、tensor、训练权重或像素重放；完整证据范围见[跨集准备度](../../research_round_20260925_global/N16_FORMAL_TEST_AND_CROSS_DATASET_EVIDENCE_READINESS_20261014.md)。


第八正式段[1792,2048)原actual0已接收，两臂各累计2048更新（第2轮+566图），每臂574匹配GT训练访问，原owned995.063秒、worker946.578秒，完整恢复/保存回读通过、CUDA allocated/reserved清0。[实测](../../research_round_20260925_global/results/U40_FORMAL_U01792_02048_ACTUAL_20261017.json)。[训练loss监测](../../research_round_20260925_global/results/U40_U01792_02048_TRAINING_LOSS_MONITOR_20261017.json)同图均值普通0.035317、自主0.035126，全部有限但差距小，不支持AP、收敛或创新主张。原[2048,2304)已从各自完整状态及同一ORDER启动，固定50轮端点不变。

[U43 HRSID完整人口检查核心](../../research_round_20260925_global/code/U43_HRSID_POPULATION_PIXEL_MASK_CHECK_DRAFT_20261017_01/README.md)与[CPU入口](../../research_round_20260925_global/code/U43_HRSID_POPULATION_CHECK_ENTRY_DRAFT_20261017_01/README.md)已完成6项纯元数据/失败路径测试和独立静态互接审；全5604图/16969原实例的真实检查尚未准备/绑定/执行。它是跨集输入兼容检查，不改U40或授予外部效用/盲测结论。


第九正式段[2048,2304)原actual0已接收，两臂各累计2304更新（3轮+81图），每臂550次匹配GT训练访问，原owned980.078秒、worker934.266秒，恢复/保存回读通过。[实测](../../research_round_20260925_global/results/U40_FORMAL_U02048_02304_ACTUAL_20261017.json)、[训练loss监测](../../research_round_20260925_global/results/U40_U02048_02304_TRAINING_LOSS_MONITOR_20261017.json)。普通/自主均值0.037823/0.037617，全部有限但差距小，没有新增AP或收敛主张。

间插U43全5604输入检查原actual1已保留，在PIL来源门处4.844秒退出，尚未population/像素/原split记录解析/COCO栅格；入口已对编码Source字节做闭合。Root纯文件检查证实十二库仅路径别名与物理路径不同、bytes/SHA全相同，旧源码/绑定/产物保持。Root失败接收没有图像/GT语义重演或数据资格；[失败及范围](../../research_round_20260925_global/results/U43_HRSID_POPULATION_LIBRARY_PATH_FAILURE_ACTUAL_20261017.json)、[明确修复计划](../../research_round_20260925_global/U43_CANONICAL_LIBRARY_REFERENCE_REPAIR_PLAN_20261017.md)。普通/自主主训练的配对模型父仍是2304；运行记账父单独接收该失败。原[2304,2560)已接续启动，固定50轮端点不变。

2026-10-18：第10段实际0与Root接收完成，两臂各2560更新，匹配训练loss普通0.036857、自主0.036750，全部有限、差异小，不支持AP或收敛结论。HRSID路径修复完成全5604/16969访问但13空栅格未过原非空门，actual1保留；主训练从原状态接续[2560,2816)，配方不变。外部单图薄桥已实现，仅元数据测试/静态审通过；外部COCO保留官方原标注，敏感性须另行预锁，尚未执行。

2026-10-18资源状态更新：第11段original actual1，最小可用RAM16993021952字节低于16GiB；wholeGPU8165MiB未超8192，elapsed775.453秒未到deadline。Root仅接收终态/源/日志和失败记账，未授部分CP/每臂成功资格，预算信用0，原paired frontier2560保持。具体进程归因未知，并行全人口合成fixture可能干扰但无RSS/绝对时刻证明；训练运行期间不再做大人口fixture。用户允许资源充足按原限额独立接续，旧attempt与全部原件保留，不放宽配方/预算。外部恢复/小窗口/原TEST合并COCO函数已形成未执行实现，源和静态/元数据范围见U44最新设计。

## 2026-10-18：独立新颖性复核与条件机制对照（尚未执行）

本轮复核不改变当前U40的17科学源码、共同顺序、初始化、50轮预算或端点评价规则。训练loss仅作过程诊断；效用、重复稳定性及机制仍待原门槛检验。

当前H/4语义支路与x/y DSConv独立计算后经concat和con2融合，可描述为语义与方向信息融合；尚不能据此称为语义条件动态采样，或声称已经证明特殊协同。直接SAR先例[O2Former](https://arxiv.org/abs/2506.11913)已公开方向感知和多尺度query增强；本轮核到的是2025-06-13预印本摘要与元数据，未独立复现其AP。[DSCNet作者仓库](https://github.com/YaoleiQi/DSCNet)明确介绍形态约束与多视图特征融合，但本轮未核其精确三路代码是否与本项目同构，也未证明该先例在当前SAR协议下的效用。[SimAM原论文](https://proceedings.mlr.press/v139/yang21o.html)已有无新增参数的逐神经元注意力；采用这些原语的数量不构成创新证据。外审提及但本轮未成功核到原始页面的额外论文不作为此处既定新颖性证据。

最有竞争力的常规解释是：额外有效卷积、感受野差异和输入依赖幅值增益，改善了一般密集精修。当前FULL与PLAIN有效参数约53.12M与47.20M，主比较并非容量匹配，不能将其潜在收益全部归给方向采样或空间注意力。

仅在主效用满足原推进条件后，优先预锁以下最少对照，不能按外部TEST结果选控制：

1. 保留H/4支路输入、通道、con2及其余结构，以静态x/y规则采样或预锁的强静态条带卷积替换动态采样；记录实际有效参数和训练/推理成本。移除或闲置offset网络时，不把无作用参数计作有效容量匹配。具体替代实现与初始化规则须在运行前锁定，本条不是已运行控制。
2. 对z+SimAM(z)=(1+g(z))z，只将空间门g替换成每通道空间均值，得到(1+mean_HW(g))z，保留CBAM及其余残差。它去除空间选择并保留平均门值，不保证输出范数严格相同，亦不独自证明因果机制。

两项均须沿原可比数据、训练预算、评价规则与共同query/S0协议进行。如果静态或均值门控制足以解释收益，应收缩机制与创新主张；保留真实效用及失败结果，不能为保持新颖叙事而改名或事后换评价。是否还需额外容量控制取决于这些具体实现的实际差异，而不是预先宣称两项已足够支持TGRS。

## 2026-10-18：DSConv实现接线核查（静态事实；数值待验证）

Root在已结束的原任务绑定中核到full_dsconv为15207 B，SHA256 `b4b612edf9f87cf29dc48bb675d501658fdec5dbee1ea363a7f02d86d197e5f0`；普通发布副本[U33 dsconv.py](../../research_round_20260925_global/code/U33_REAL_IMAGE_400_PROBE_DRAFT_20261017_01/dsconv.py)逐字节相同。完整full/feature普通副本也与该数值版本对应。本节没有修改任何冻结源码或训练结果。

K=9的morph0在第166行detach().clone后，仅通过第177行循环重新接回内侧六点；两个端点0/8保留前向偏移但不保留其直接offset梯度。morph1第234行clone保留梯度，两方向的range(1,center)均未将端点纳入累计。它们并非相同方向的两个假分支，也不是整个offset网络都失活。第277–347行的邻点整数索引之外仍用连续坐标计算双线性权重，未静态发现该路径整体截断。语义支路在concat处融合，不能声称其并行输出显式输入了offset网络。

可证伪的下一项只有一个B1/K9/H=W=1坐标诊断：对morph0 sum(y)、morph1 sum(x)分别求18通道VJP与中心有限差分。静态预期选中9通道morph0的AD为[0,1,2,3,0,3,2,1,0]、FD为[1,1,2,3,0,3,2,1,1]；morph1 AD/FD均为后者，未选偏移半部均0。以上是待实测预期，不能充当数值结果。

[探针源码](../../research_round_20260925_global/code/U45_DSC_COORDINATE_VJP_DIAGNOSTIC_DRAFT_20261018_01/README.md)仅作局部CPU双精度坐标公式核查，不运行卷积、BN、整网、插值特征或数据。若实际确认差异且目标确为中心固定、全部9点向外累计，最小新版本候选为移除该detach、两处循环包括center；这一改动改变原计算与梯度，必须另建版本和实际验证，不得原位改旧Source、沿用旧结果归因或宣称AP提升。实际训练是否受影响及修复是否有任务效用仍未确认。


## 2026-10-18: actual U45/U46 coordinate diagnostic accepted

Both original CPU jobs exited0 and were Root-accepted. U45 original morph0 AD [0,1,2,3,0,3,2,1,0] differs at endpoints0/8 from FD [1,1,2,3,0,3,2,1,1]; original morph1 AD/FD agree with the latter. Separate three-line U46 candidate has both-axis AD/FD [1,2,3,4,0,4,3,2,1], matching the predeclared complete-accumulation VJP. CPU-double coordinate sums only: no full Jacobian, DSConv feature interpolation, real images, FP32 network, training or AP. Complete endpoint accumulation already has an author-code precedent; no new sampling-method claim.

[Actual evidence, source and next step](../../research_round_20260925_global/U45_U46_DSC_COORDINATE_ACTUAL_FINDING_AND_NEXT_STEP_20261018.md). Current accounting290 normal/35 failed is independent of unchanged U40 model frontier2560. Do not swap U46 into the running protocol. Main recovery02 control is prepared but unbound; resource observation at07:25:18 UTC was insufficient. Earlier pending-numerics descriptions remain historical; this is the current numerical status.


## 2026-10-18: actual U47 teacher CPU restoration accepted

The original teacher-only job exited0 and was Root-accepted. Complete E24 state610/parameter451 and model/BN bytes matched the original loading proof; frozen eval CPU, no gradients, caller RNG/helper bindings restored, anchor/model released. Worker21.5seconds, sampled peakRSS1632632832bytes; CUDA remained cold. No image/dataset/forward/optimizer/AP. [Actual scope and DGX observation](../../research_round_20260925_global/U47_TEACHER_CPU_RESTORE_ACTUAL_AND_DGX_STATUS_20261018.md). U47 supplies the missing teacher restoration function; current outer pipeline still needs resolved preprocessing provenance and real new-image forward before E50 external predictions.

Accounting291normal/35failed is separate from unchanged U40 paired model frontier2560. DGX SSH/OS/package metadata was actually read without ML import: aarch64/GB10 with about69GiB available system RAM, Python3.12/Torch2.10 in the established main environment. That runtime does not satisfy immutable local U40; no direct continuation, numerical port or speed claim. Main resource admission remains unmet; original limits and E50 endpoint are unchanged.
