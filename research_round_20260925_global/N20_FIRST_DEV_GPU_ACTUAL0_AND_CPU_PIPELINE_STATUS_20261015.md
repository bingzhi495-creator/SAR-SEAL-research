# N20：全部DEV GPU分片完成及CPU数值核验状态

更新：2026-10-15。证据等级为单初始化、内部DEV、全部四GPU分片实际完成和来源/文件核验；没有N20 AP结果、正式TEST、跨数据集或发表就绪结论。

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

启动前新纯RAW-only检查actual0（0.292秒），无Torch且在首次Source哈希前停止。唯一新原CPU任务为[0,6)，CUDA空、CPU0/1与两线程、1800秒上限、RSS16GiB及最小可用RAM16GiB，无自动重试。当前仍正常145/失败26/171终态，0/32合格；运行中的任务不计完成。模型和原四GPU捕获未重算，主端点和全187官方COCO仍按原锁定规则。
