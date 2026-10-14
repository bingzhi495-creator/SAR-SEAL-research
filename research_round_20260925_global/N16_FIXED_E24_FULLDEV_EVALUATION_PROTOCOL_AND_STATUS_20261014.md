# N16 固定24轮完整开发集评价

记录时间：2026-10-14T10:01:13.099Z。状态：原监督入口已启动，尚待GPU真实结束及独立CPU证明；没有新增AP结论。

Utility与Error使用同一初始四张量、同一FIT点/权重/顺序和固定24轮AdamW训练，各17784更新，已完成独立E8/E24端点核验。不同标签目标的训练交叉熵不能直接比较优劣。

本次覆盖187张DEV图、380个原始实例、每图全部100自然query，共18700候选。保留原类别、分数、输出顺序和全部原标注；选择器不读取评价GT、不设置分数过滤或NMS。主端点固定E24；E8为曲线观察，exact-K为预锁补充，不替代atmost-K主结果。

12组固定预测包括原生、原强E24固定Omega、E8/E24两臂atmost/exact，以及E24实际翻转门控密集误差和MagNet二分类适配。后者为同协议适配，不是作者完整方法复现。Utility主结果相对max(Error同24轮AP,68.09938494745545)至少提高1点且高于Error才进入重复验证；非正收益停止当前配方，正收益不足1点至多补一个预锁种子。

CPU只读闭合原GPU选择缓存的资源核算：29404940原图像素；57056641个C候选点；4573049个K点；每图候选并集累计1530704点；最大单query候选59126点。计数仅用于资源准入，不是训练/评价效果。保守序列化上界约59.000GB，文件上界434843；采用逐图共享实际F/e/L、逐2048点保存实际GPU输出，以及little-bit无损掩膜压缩。共享诊断运行不能据此宣称独立部署FPS。

真正的资格要求：原监督进程真实退出0、原进程已回收；随后独立CPU对全部18700 query使用保存的实际GPU输入/输出检查候选域、严格排名、所选索引、两条PR公式、L>0符号、完整掩膜、原分数/顺序、RLE及官方COCO12一致。数值绝对容差1e-4、相对容差0；索引与符号不能用近似替代。CPU结果自报不等于Root真实结束资格。

运行资源保持CPU0/1、2线程、GPU分配器4096MiB、整卡观察8192MiB、可用RAM16GiB及1800秒截止；不自动重试。历史12个失败原样保留，其中资源截止的部分缓存没有资格、没有被复用。

本次采用新普通GPU入口02；此前未执行01发现局部CPU字典覆盖模式布尔值的问题，02仅改局部变量名称，4项合成回归通过。候选域K返回值及完整掩膜预算的静态接口修正亦保留。上述工程检查不构成科学收益。

[GPU普通研发源码](code/N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_UNBOUND_20261014_02.py)、[捕获与CPU解码合同](code/N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_CONTRACT_20261014_01.md)、[接口修正记录](code/N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_CORRECTION_20261014_02.md)。这些为独立普通发布副本；没有运行中的原始载荷、绑定SOURCE、凭据、数据、权重或第三方PDF。

## 原失败记录与新修正版（2026-10-14T10:13:26.504Z）

首次完整DEV入口原监督真实退出1，用时149.390秒；held handle退出1、回收及原birth消失均核验，abort/error均为空，整卡观察峰值2254MiB，最小可用RAM33256136704字节。第一张图调用旧packet_check时缺少report字段，尚无新增预测或AP；不是方法负结果。原ENTRY/SPEC/RAW/START与日志冻结保留，原REPORT缺失。

新普通GPU03仅补原N07 report对应字段、向旧选择缓存传递原context，并增加原失败13在首次SHA之前的终止/身份检查以及后续SOURCE/owner闭合；科学配方未变。独立新监督入口现已启动，尚待实际成功结束，不是自动重试。

CPU02入口与GPU03前门AST一致，评价后缀与CPU01保持一致；微型CPU合成全100query/12策略/K0边界通过，7项畸形捕获被拒绝。真正的18700query实际证明尚未开始。

[新GPU03普通源码](code/N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_UNBOUND_20261014_03.py)、[接口与失败保留修正](code/N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_CORRECTION_20261014_03.md)、[CPU02普通源码](code/N16_FIXED_E24_ALL100_SCOUT_FULLDEV_CPU_UNBOUND_20261014_02.py)、[CPU合同](code/N16_FIXED_E24_ALL100_SCOUT_FULLDEV_CPU_CONTRACT_20261014_02.md)。旧GPU02为该失败任务的普通源码历史副本，不能替代新入口执行资格。

## 原30分钟截止与逐图存储版（2026-10-14T10:53:50.460Z）

第二次完整DEV原作业在1800.297秒真实受控中断，51/187图，整卡观察峰值4479MiB，最小可用RAM30474997760字节。held exit1、reaped、原birth消失均已核验，abort为QUALIFIED_ORIGINAL_LEAF_RESOURCE_OR_DEADLINE；无报告、无完整AP。部分捕获保留但未获资格，不用于新运行或方法效用判断。

已结束作业的第二图耗时30.672秒，其中捕获回调19.6601秒（约64.1%）；PR2.7386秒、scout1.1341秒、gather0.7972秒、readback0.2006秒、排序0.219秒、评分0.266秒。此单图重叠诊断不等于独立部署速度或完整数据集成本。小文件写入/读回/哈希是具体工程优化依据。

GPU04/05改为每图一个真实闭合bundle、内部仅严格字符串member ID，不构造虚拟文件ref或自引用SHA；仍保留每2048点实际GPU输入/输出、每query各端点/控制完整mask及原Omega。每图1次保存、1次读回、1次哈希在合成spy检查中通过；每图raw<=1GiB、单文件<=2GiB、RSS16GiB和原1800秒等资源门保持。新05仅在04上添加失败14的首次SHA前RAW保留/closed owner join；科学计算与选择规则未变。新独立原监督入口已启动，尚待真实成功结束；不能把合成IO收益当真实提速或AP收益。

CPU03/04改为每图真实bundle只load一次，检查member、registry、使用集合严格完整；原9字段数字证明和12策略COCO规则保留。合成100query/12策略/K0/完整成员通过，8类异常拒绝；独立一次静态审阅未发现确定阻断。CPU真实完整证明尚未执行。

[逐图GPU源码04](code/N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_UNBOUND_20261014_04.py)、[bundle合同](code/N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_BUNDLE_CONTRACT_20261014_04.md)、[现行GPU05](code/N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_UNBOUND_20261014_05.py)、[现行CPU04](code/N16_FIXED_E24_ALL100_SCOUT_FULLDEV_CPU_UNBOUND_20261014_04.py)、[CPU逐图adapter](code/N16_FIXED_E24_ALL100_SCOUT_FULLDEV_CPU_ADAPTER_UNBOUND_20261014_02.py)。本仓库仅发布独立普通副本；运行中的科学捕获、绑定SOURCE与资格链没有为同步而读取、复制或哈希。

## 逐图版实际截止与固定分段补充（2026-10-14T11:30:11.518Z）

逐图版原监督session72971已在1800.375秒真实受控退出1，149/187图；held exit1、reaped、原birth消失均核验，abort为QUALIFIED_ORIGINAL_LEAF_RESOURCE_OR_DEADLINE，error空、wait错误无、GPU观察错误无；整卡观察峰值4497MiB，最小可用RAM31018995712字节。原完整REPORT缺失，无完整AP，失败部分捕获不复用。所有15个失败原样保留，此结果不是方法效用失败。

现已在原实际终止核验后锁定[固定ordinal四段执行补充](N16_FIXED_FULLDEV_EXECUTION_PARTITION_SUPPLEMENT_20261014.md)。每段全100query/12策略、相同冻结模型与数值规则；四段各实际GPU0闭合，独立CPU汇总完整18700后统一12COCO，再独立全scopeCPU证明。各bundle保留原shard binding，不能改造来源；训练owner107/108和后续各owner明确join。总评价时长增加，训练预算、数据、候选、预测、E24主端点/指标/停止门不变；不平均分段AP、不减证据范围、不复用失败partial。首段已于本轮启动，固定global ordinal[0,47)，原监督session17396；原始RAW前缀通过后进入新Source/预算门。完整AP仍未形成，四段实际0、全量汇总与独立CPU证明仍须完成。

[分段GPU普通源码](code/N16_FIXED_E24_ALL100_SCOUT_SHARD_GPU_UNBOUND_20261014_01.py)、[分段合同](code/N16_FIXED_E24_ALL100_SCOUT_SHARD_GPU_CONTRACT_20261014_01.md)。5项最小合成检查通过，固定107/108训练来源、global47分段索引和15号受控失败均保留；它们不替代实际科学执行。第一段保守serialized上界13,647,216,640字节、所需free31,363,956,736字节，按闭合本段真实几何计算；不按样本比例估、不复用149图失败partial。


## 首段实际结束与第二段执行

首段原监督session17396真实退出0，47/47图、4700自然query、12组完整本段预测和47个实际逐图bundle；elapsed693.609秒、整卡观察峰值3482MiB、最小可用RAM31,436,058,624字节。held退出0、reaped、原birth消失、abort/error为空、wait/GPU观察错误均无；原生和原强E24固定Omega的本段预测逐值一致。实际serialized捕获5,416,736,908字节。首段仍是局部预测生成，不计算本段AP，不宣称完整DEV或独立数值资格。

新增normal110+failed15纯RAW前缀实际退出0后，第二段原监督session88419已启动，范围固定[47,94)，训练和选点规则不变；后续两段、独立CPU全量汇总及完整数值证明仍待实际执行。

[CPU全量汇总02普通源码](code/N16_FIXED_E24_ALL100_SCOUT_SHARDS_AGGREGATE_UNBOUND_20261014_02.py)、[汇总合同](code/N16_FIXED_E24_ALL100_SCOUT_SHARDS_AGGREGATE_CONTRACT_20261014_02.md)、[完整独立CPU证明普通源码](code/N16_FIXED_E24_ALL100_SCOUT_SHARDS_FULLDEV_CPU_PROOF_UNBOUND_20261014_01.py)、[证明合同](code/N16_FIXED_E24_ALL100_SCOUT_SHARDS_FULLDEV_CPU_PROOF_CONTRACT_20261014_01.md)。这些已形成的普通研发产物及时保存，但尚未实际执行；合成覆盖/绑定检查不是实际来源资格或真实COCO验证。
