# N20：前两DEV分片完成及后续评价状态

更新：2026-10-15。证据等级为单初始化、内部DEV、首GPU分片实际完成和来源/文件核验；没有N20 AP结果、正式TEST、跨数据集或发表就绪结论。

## 已实际完成

两臂使用相同741张FIT图、每轮3,035,136采样点、AdamW、24轮和17,784次更新。LOCAL使用原图渲染特征的真实3×3邻域，CENTER重复中心特征；各18,851参数、五张量、相同初始化。E8/E24保存端点已由独立CPU及Root恢复核验；原训练GPU最终清理exit1仍保留，没有补造原GPU REPORT，也没有第四次训练。

修复后的首DEV分片[0,47)原任务实际exit0，耗时1411.156秒。Root先执行新纯终态前缀并取得actual0，再读取本轮结果；检查REPORT、CAPTURE_MANIFEST、START的13项绑定及原任务身份/来源，47张图/4700自然query，47份捕获文件共3,151,195,873字节均与保存SHA一致，十策略各4700条完整预测并保持image/query/class/S0顺序。CUDA清理后allocated/reserved均为0，观察到整卡峰值3488MiB；整卡观察不能称为进程显存认证。耗时含大量证据采集与来源校验，不是独立推理FPS。

这些检查确认原任务完成及输出完整，不替代张量数值、选择索引、PR、strict sign、RLE和全量COCO资格。原REPORT保留qualified=False和CPU pending，不改成已资格。首段完成时正常终态142、失败25、合计167；当前计数见下方更新。

## 保留失败与最小修复

第一次首DEV尝试在序列化回读的Long零步长转Byte处实际exit1，无资源abort，无完整REPORT。31份partial未复用。极小CPU回归先复现同错，再验证仅逐字节比较助手的修复；保持dtype/shape、signed zero和非连续张量的精确语义，不放宽科学容差。模型、评分、训练、捕获、范围及预算没有改变。失败25、原日志与来源按原资格规则保留，不把修复解释为方法收益。

## 当前执行与下一步

第二GPU分片[47,94)现已实际exit0，1721.031秒；新pure143+failed25 actual0先于本段科学产物读取。Root核对原终态与来源、三产物13绑定、47图/4700自然query、47份捕获4,243,201,508字节的SHA、十策略各4700完整预测及image/query/class/S0顺序、canonical profiles、预算与零cleanup。两段累计94图/9400query，仍仅GPU完成与输出完整性，非独立CPU数值或AP资格。当前正常终态143、失败25、合计168。

第三GPU分片[94,141)已启动，仍用前两段的同一冻结源码和E8/E24模型；第四段固定[141,187)。只允许唯一原科学任务，无自动重试。全部四GPU原任务实际0并经Root先终态后读取核验，才开始预锁八CPU范围；不从未资格partial或分片AP拼装结论。

八CPU按既定原捕获逐点核验两独立公式、GPU误差、canonical FP64评分、全局排序、预算K、fresh selected PR、strict sign、RLE与完整预测。随后十策略统一187张图/18,700自然query做一次各自全量官方COCO评价，保留全部12项stats，核原生与强参考一致；绝不平均分片AP。最终原CPU和后置Root各自须实际退出0。

| 代码与检查 | 实际状态 | 证据限度 |
|---|---|---|
| 当前25链完整CPU entry02 | 已装配，AST及纯header实际通过 | 未绑定、未运行真实CPU数值 |
| full187 COCO front/tail及完整entry01 | 已装配，AST通过，一次定向静态审阅READY | 尚未绑定/真实运行COCO |
| post-actual后置门、合同及完整entry01 | 一次独立静态审阅READY；文本装配/AST实际通过 | 不自证post实际退出；未绑定/执行 |
| public proof_query合成CPU集成检查01 | 实际0，1项测试，2.079秒 | 手构2×3数据；验证两公式/评分/选点/PR/RLE及篡改拒绝，不认证真实GPU |

源码：[CPU完整入口](code/N20_FIXEDDEV_FEATURE_ERROR_CPU_ENTRY_UNBOUND_DRAFT_20261015_02.py)、[全量COCO完整入口](code/N20_FIXEDDEV_FULL187_COCO_CPU_ENTRY_UNBOUND_DRAFT_20261015_01.py)、[post完整入口](code/N20_FIXEDDEV_FULL187_POST_ACTUAL_ROOT_CPU_ENTRY_UNBOUND_DRAFT_20261015_01.py)、[合成集成检查](code/test_N20_SELECTED_NUMERIC_CPU_PUBLIC_API_SYNTHETIC_ONLY_20261015_01.py)。这些是可保存研发稿，不能把文件存在或静态READY写为真实实验成功。

## 与GLDet及创新判断的关系

[GLDet原论文综述](GLDET_PRIMARY_PAPER_REVIEW_AND_SEGMENTATION_TRANSFER_20261014.md)和[局部信息检验设计](GLDET_CODE_RECHECK_AND_MINIMAL_LOCAL_INFORMATION_TEST_20261015.md)保留原文引用及范围。GLDet完整空间结构尚未复现；N20是先检验真实局部特征有无增量信息的隔离实验，普通邻域操作本身不是新创新。N17旧原生单元邻域主结果为负，不能直接推出本轮原图渲染邻域也无效。

主端点始终为E24-atmost：Delta=AP_LOCAL-max(AP_CENTER,68.09938494745545)。Delta≤0停止此配方追加预算/种子/融合；0<Delta<1只允许先复验第二种子，第三种子须符合原锁定条件；Delta≥1复验第二/第三种子。E8或exact不能救主端点。正收益仍需排除额外容量/计算与评价因素，说明对直接先例的实际差异，补重复种子和跨集；“加模块后涨点”不会自动成为TGRS充分证据。

## 发布范围

本次仅同步普通独立研发副本、合同和经核验状态；不发布原始数据、标注、权重、第三方全文或运行绑定资格链。当前运行中的科学文件与绑定Source未为同步而读取、复制或哈希。
