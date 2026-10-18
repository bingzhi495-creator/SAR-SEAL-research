# 完整自主UNet++实例方法：执行跟踪

<!-- U51_FIRST_BATCH_AND_U40_RESUMED_20261018_01 -->

2026-10-18 后续：U51首批32图/43匹配实例已actual0导出并接收（117777312 B，worker13.328秒，RSS643702784 B），全量准备32/741图。DGX原配对试点已6真实step通过但不计本机主训。资源双观察恢复后，原U40 2816→3072已启动；已接收端点仍2816，运行不冒记3072，无新增DEV/AP。[当前实际记录](../../research_round_20260925_global/U50_DGX_PAIRED_PILOT_ACTUAL_AND_FULL_INPUT_NEXT_20261018.md)。


<!-- U40_2816_ACCEPTED_CURRENT_20261018_01 -->

2026-10-18 最新已接收端点：2816/37050（3轮+593图，预算7.60%）。原2560→2816任务actual0，经Root原终态/FIRST/来源/配对检查点和轨迹核验接收；两臂各256更新、561实例访问，512次成功step，原任务1060.828秒，未越原资源限额。FULL/PLAIN训练损失均有限且相近，无新增DEV/TEST/AP，不凭−0.234%相对训练损失宣称效用。运行账户293正常/36失败，与模型更新次数分开。下一段2816→3072仅已准备请求，启动仍须双资源观察。U49/U50仅普通源码完成，未实际导出或DGX运行；所有历史失败保留。详见[本段核验及后续](../../research_round_20260925_global/U40_U02560_02816_ACCEPTED_AND_DGX_EXPORT_STATUS_20261018.md)。


<!-- U40_POST_U48_ACTUAL_LIVE_20261018_01 -->

2026-10-18 接续实况更新：09:09:57 UTC 与09:11:48 UTC两次资源观察均达标，free RAM分别38350995456/38238998528 B、整卡2065/1963 MiB。Root已实际启动并登记原配对2560→2816接续任务，17份训练来源及原配方/限额不变，恢复原2560检查点。当前为运行中，尚无该段终态或新配对预算；有效端点仍2560/37050。此前资源不足描述保留为当时观察，不能据启动记为已完成。


<!-- U48_NATIVE_EXECUTION_CURRENT_20261018_01 -->

2026-10-18 U48 实测补记：原 E24 教师在预选 HRSID TRAIN 首图的完整原生100查询推理已实际退出0并接收，耗时26.453秒、分配器峰值约2.11GiB，清理后allocated/reserved0；原2GiB后处理OOM actual1单独核验保留。单图资源修正版只改2→3GiB分配cap，数学/输入/score不变；无新训练/AP/全人口资格。主训有效端点仍每臂2560/37050；新增运行账户为292正常/36失败，不等于训练进度。最新账户到原f35训练失败及未变2560模型父的元数据继承已实际验证，未绑定/启动接续。主训需两次相隔至少30秒的free RAM≥32GiB与整卡基线≤3584MiB观察；当前未达门，17源/共同图序/配方/固定E50预算不改。详见[单图实测](../../research_round_20260925_global/U48_SINGLE_HRSID_NATIVE_CAPTURE_ACTUAL_AND_NEXT_STEP_20261018.md)。


<!-- U40_RESOURCE_RECOVERY_CURRENT_20261018_01 -->

2026-10-18接续终态：原内存中止后发起的独立同段尝试又因整卡显存8398>8192 MiB受控结束，原任务已回收，失败记账已核验；RAM最低17490812928 B高于16 GiB，800.078秒未到时限。当前有效端点仍每臂2560，未读取或复用该失败尝试的部分权重与trace，新增配对预算为0。两个失败尝试的实际资源成本保留，不把资源中止判为模型失效。下一主训启动观察按整卡基线≤3584 MiB、free RAM≥32 GiB，覆盖分配器4 GiB及512 MiB整卡余量；原数值配方和运行限额未改。当前先核精确版本DSConv的局部坐标梯度，不构建网络，不替代主效用评价。

更新：2026-10-18。完整自主卷积主干与普通对照两臂各2560正式更新已接收，[2560,2816)原尝试资源中断已保留，新独立接续任务已整卡超限结束，尚未接收新配对端点。N28固定E24完整DEV按原门槛STOP当前路由配方。表格为当前状态，后文保留形成时记录。

设计收敛确认：最终候选直接以既有自研完整UNet++为主体，保持当前U40连续训练；历史boundary/frequency_boundary正收益有条件复用，不能自动归给full/new1或当前实例AP。本次只明确方法组织与归属，不改动已锁数值源码、配方、总预算、端点或停止规则。

| ID | 目的 | 系统/范围 | 状态 | 下一项实质动作 |
| --- | --- | --- | --- | --- |
| U00 | 找回已有正信号 | HRSID frequency+boundary；SSDD boundary三种子 | 已读原数值记录与完整50轮历史；有语义F1收益 | 保留版本归属；不冒充full模块或实例AP |
| U01 | 确定完整结构与资源版 | 原full复合编码、H/4方向融合、完整解码 | 卷积资源版已通过U36/U39并进入正式训练 | 巨大旧Transformer明确关闭且不实例化；原件保留 |
| U02 | 实现共享实例接口 | 一次图像主干＋小型查询残差头 | tiny三项及真实full64两项均实际通过并接收；原特征精确等价且245张量梯度有限 | 真实400及跨图资源已通过；沿固定主训练和端点评价验证效用 |
| U03 | 检验真实输入和资格 | 原图/几何/原100查询/L/匹配GT | U38全741/1661与104同源初始化已接收；U39跨图检查通过 | 固定E50的完整DEV输入和评价核心独立准备，未数值执行 |
| U04 | 测量真实成本 | 完整full卷积资源版与普通对照 | U40首段双臂256更新已接收；原任务969.312秒，CUDA约1.91GiB/RSS约13.04GiB | 后续段独立记录成本与恢复，不能由局部耗时承诺整轮或加速 |
| U05 | 首组公平主训练 | 普通完整UNet++ vs自主full，seed3101 | 已接收各2560更新；第11段两次资源中断均已保留，当前先核坐标梯度 | 每臂50轮/37050更新才评价；尚无主方法AP |
| U06 | 验证核心归因 | 容量/计算替代、复合块/方向替换 | 条件待执行 | 主效用成立后只做能改变结论的最少控制 |
| U07 | 重复与跨数据集 | 3102/3103；另一公共实例数据集 | 条件待执行 | 按预定收益门槛推进，不承诺必然成功 |
| N28 | 路由固定端点 | SELECTIVE vs WHOLE_NODE；seed2801 | 完整187图COCO已接收；AP67.021271/66.655257；门槛Delta−2.516096 | STOP当前配方，不扩预算/种子/融合；不代替完整主干证据 |
| HW | 确认DGX入口 | Spark GB10只读连接与资源快照 | SSH实际成功；有用户llama服务及未核CPU任务 | 保留用户进程，数值执行仍由Root统一安排 |

科学执行只沿当前原句柄追踪，观察超时不重启。此前N28原session45218已终态；本文为普通跟踪记录，不构成任何任务的绑定输入或执行资格。具体来源与失败保留仍由原执行记录维护。

论文状态：已有历史语义正收益和当前结构研发，不等于完整实例方法涨点或TGRS投稿依据。主张/指标/成本/初始化/划分/端点均以实际完成且核验的结果更新。

补记：CPU tiny测试原任务3.265秒、三项全通过；仅接口范围，未执行原完整结构。原full64测试草稿已形成，尚未执行。实际接线和结果导航见[本轮研发进展](../../research_round_20260925_global/UNETPP_SHARED_ADAPTER_CPU_RESULT_AND_FULL_FEATURE_STATUS_20261017.md)。

真实full64原任务约5.156秒、两项全部通过且经Root接收；400/真实影像/训练/AP仍未执行。[完整结果](../../research_round_20260925_global/U30_REAL_FULL_FEATURE_CPU64_ACTUAL_RESULT_20261017.md)。此前未执行描述为历史记录。

最新普通对照已选plain64 model1；几何/plain两项均已RED后实现，仅AST；all100M实际补采源已完成但未执行。此前session45218记录为已终态历史，当前无该任务。[本轮进展](../../research_round_20260925_global/U31_MAIN_CONTROL_AND_NATIVE100_INPUT_PROGRESS_20261017.md)。

U32真实100查询M/F补采已实际完成并接收；U33新400双臂两步资源试点仍未执行，不是正式主训练。[当前实测与具体试点](../../research_round_20260925_global/U32_NATIVE100_CAPTURE_ACTUAL_AND_U33_REAL400_PLAN_20261017.md)。

U33两CPU/普通臂零残差与分组/首步更新通过，第二步4GiB下OOM，FULL未执行，原失败已核验保留。U35仅两臂共同训练chunk1，未执行；U34公共初始化仅静态草稿，正式主训练未启动。[具体结果与修订](../../research_round_20260925_global/U33_REAL400_OOM_RESULT_AND_CHUNK1_REVISION_20261017.md)。

最新U35：PLAIN两步和CUDA清0通过，FULL eval通过而首个train forward OOM，原失败保留；CPU暂存U36未执行，正式主训练未启动。N28已原状态启动第16轮。详见[实测与设计](../../research_round_20260925_global/U35_REAL400_PARTIAL_RESULT_AND_FP32_CPU_ACTIVATION_DESIGN_20261017.md)。

更新：第16轮原actual0已接收，每臂11856更新；无E16 DEV，仍固定E24。U36含明确4toy+4real一次性更新合同，尚未执行。


最新实际结果：U36原actual0已外部接收，tiny4+real4共8次一次性更新，PLAIN/FULL真实400每臂两步通过；U33/U35失败保留。详见[U36结果](../../research_round_20260925_global/U36_REAL400_TRAINING_RESOURCE_ACTUAL_RESULT_20261017.md)。下一项是全741真实RGB读取、104公共状态初始化和跨图持续资源资格；不是增加新模块或先做VJP。50轮主训练及实例AP仍未开展。N28第17轮已由第16轮原模型/优化器/随机状态接续启动；启动不等于完成，固定端点仍为24轮。

第17轮原actual0已核验接收：每臂12597次累计更新，355.468秒，固定24轮端点保持；未做中途DEV或报告新AP。[实测摘要](../../research_round_20260925_global/results/N28_E17_CONTINUOUS_TRAINING_ACTUAL_20261017.json)。

第18轮原actual0已核验接收，每臂累计13338更新；357.343秒，无中途DEV/AP。[实测摘要](../../research_round_20260925_global/results/N28_E18_CONTINUOUS_TRAINING_ACTUAL_20261017.json)。完整主干下一任务为U38单叶全741读取与公共初始化，尚未执行。

U38原actual0已接收：全741/1661真实读取、公共104状态和adapter同源均通过，CUDA冷、无训练。下一项U39跨图6次一次性更新仅静态草稿；正式50轮/AP未做。[实际结果](../../research_round_20260925_global/U38_FULL_FIT_RGB_AND_COMMON_INITIALIZATION_ACTUAL_RESULT_20261017.md)。N28第19轮已由第18轮原状态接续启动；不将启动计为已完成。

第19轮原actual0已接收，每臂累计14079更新；360秒，无中途DEV/AP。[实测](../../research_round_20260925_global/results/N28_E19_CONTINUOUS_TRAINING_ACTUAL_20261017.json)。U39跨图执行源码已形成，尚未运行；U40仅连续训练实现设计，正式50轮未启动。

U39原actual0已接收，两臂各3次连续丢弃更新，覆盖27实例图；正式预算仍0。U40连续核心已形成、静态审阅及3项纯元数据测试通过，尚未正式数值执行。采用既有自定义卷积主干和共享实例残差读出，不另起骨架，无本次AP或创新成立结论。

U40入口/工作体完成静态和10项纯元数据测试；固定共同50轮图序、256更新×145段（末186），正式训练尚未实际启动。保存/恢复直接在正式段内核验，不另加数值试点。磁盘保留规划为估计，首段后依据实际字节更新；无新增AP。

N28第20轮原actual0已接收，两臂各累计14820更新，完整恢复模型/Adam/RNG；原owned357.656秒，固定24轮端点不变，无中途AP。[实测](../../research_round_20260925_global/results/N28_E20_CONTINUOUS_TRAINING_ACTUAL_20261017.json)。


U40第一正式任务[0,256)已由Root实际启动并登记原执行句柄，当前运行中；本条只记录启动事实，不授予本段完成数或检查点接收。两臂各计划256次，属于每臂50轮37,050次总预算，先普通对照再完整改造主干，不自动重跑。正式原终态、双臂状态与完整trace待统一核验；无AP、长期稳态或创新成立结论。

[固定E50完整DEV设计](../../research_round_20260925_global/U40_FIXED50_DEV_EVALUATION_DESIGN_20261017.md)与[无伪G的RGB桥设计](../../research_round_20260925_global/U40_FIXED50_DEV_RGB_BRIDGE_DESIGN_20261017.md)仅静态。复用既有自然100查询字段读取与几何；按原导出qids逐项索引，所有分数/类别不改。原JPEG187引用待实际闭合，不新增teacher补采或数值资格叶；评价只在最终配对端点执行。


首个正式[0,256)段原actual0已Root接收，两臂各256次（合计512次），共同游标为第1轮内256/741，尚未完成第1轮。实测原任务969.312秒，实际阶段CUDA分配峰值2,051,057,664字节、worker RSS峰值14,004,445,184字节；双臂完整状态合计1,262,052,812字节，序列化回读检查由工作进程完成，Root仅核终态/源码/JSON trace及权重文件字节。两臂与最终CUDA释放双0；首次尚无续段恢复证明，无AP或效用结论。[安全摘要](../../research_round_20260925_global/results/U40_FORMAL_U00000_00256_ACTUAL_20261017.json)。

下段[256,512)原任务已启动，从已接收两臂各自模型/BN/Adam/RNG与同一ORDER继续，待原终态及实际恢复核验。未更换冻结主干、监督、顺序或50轮总预算。E50 RGB桥完成静态/纯元数据源码验证，未数值执行；直接先例笔记已补齐证据边界，均不参与中途选型。


固定E50预测与完整人口CPU评价核心已形成未绑定草稿，各3项纯元数据测试通过；未实际恢复/运行模型、读取真实DEV或执行COCO。仅复用原数学与完整候选，不使用旧E24模型/推进规则；原bbox/area元数据保持并在未来记录实际计数。见[评价设计与两核心导航](../../research_round_20260925_global/U40_FIXED50_DEV_EVALUATION_DESIGN_20261017.md)。当前256正式前缀和原续训任务不受影响。


第二正式段[256,512)原任务实际退出0并经Root接收，两臂各再256次、累计512次；尚未完成第1轮741图。worker实际检查各自完整模型/BN/Adam/RNG恢复与保存回读相等，Root只接收原终态/字节/JSON及trace关系，不作独立tensor重放或全轨迹bitexact主张。原任务1006.062秒、CUDA分配峰值2,048,325,120字节、worker RSS峰值14,006,157,312字节，两臂及最终CUDA清0；无DEV/AP。脱敏摘要已在下一科学任务前复制到普通结果区。[实测](../../research_round_20260925_global/results/U40_FORMAL_U00256_00512_ACTUAL_20261017.json)。下段[512,768)由该各自状态与原ORDER接续启动，尚未接收本段更新。


第三正式[512,768)段原actual0已接收，完整骨架两臂各累计768次（第1轮+27图），本段每臂624匹配GT访问，worker实际恢复/保存回读检查通过，原任务1042.703秒、双臂CUDA清0，无DEV/AP。[实测](../../research_round_20260925_global/results/U40_FORMAL_U00512_00768_ACTUAL_20261017.json)。187原JPEG候选引用已从四份原JSON manifest精确提取并通过RGB桥metadata函数；未读原像素/tensor/GT/权重、没有NN/COCO或新资格叶。[元数据摘要](../../research_round_20260925_global/results/U40_FIXED50_DEV_METADATA_PREPARATION_ACTUAL_20261017.json)。当前穿插N28第20→21轮，之后完整骨架仍从各自768状态接续；原预算/端点不变。


N28第21轮原actual0与完整恢复已核验接收，每臂15561更新，原任务354.813秒；无DEV/AP。[实测](../../research_round_20260925_global/results/N28_E21_CONTINUOUS_TRAINING_ACTUAL_20261017.json)。第22轮原任务已按原模型/Adam/RNG继续启动，固定端点仍E24；完整UNet++状态保持各768更新，后续独立接续，不由路由对照代替主干证据。

N28第22轮actual0已核验接收，两臂各16302更新；原任务354.312秒，无DEV/AP。[实测](../../research_round_20260925_global/results/N28_E22_CONTINUOUS_TRAINING_ACTUAL_20261017.json)。第23轮已沿原模型/Adam/RNG启动；完整UNet++各768更新检查点保留，不由路由对照代替主方法证据。

N28第24轮actual0已核验接收，完成两臂各17784更新；原任务352.843秒，无训练中DEV/AP。[实测](../../research_round_20260925_global/results/N28_E24_CONTINUOUS_TRAINING_ACTUAL_20261017.json)。固定E24的首个47图/4700查询DEV预测原任务已启动并登记；四分片和完整CPU COCO尚未完成。完整主干状态保持各768更新。

N28四GPU及完整CPU原actual0均接收，固定S0/候选/原native/R 12项精确复现；[结果和条件几何](../../research_round_20260925_global/N28_FIXED24_FULLDEV_RESULT_AND_FULL_METHOD_CONTINUATION_20261017.md)。完整主干从768原状态重新启动[768,1024)，其完成数仍768。E50外层源码已实现，仅metadata测试和静态审阅，未数值执行。

U40第四段原actual0已接收，两臂各1024更新；本段每臂556 GT访问，原1008.344秒，CUDA allocated/reserved均清0，worker实际恢复和回读通过。[实测](../../research_round_20260925_global/results/U40_FORMAL_U00768_01024_ACTUAL_20261017.json)。第5段[1024,1280)已启动；E50执行控制草稿6项纯元数据检查与独立静态审阅通过，实际准备/绑定/数值尚未执行。

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

2026-10-18实测：第10段[2304,2560)actual0并接收，GT访问614/臂，模型/Adam/BN/RNG恢复与保存回读通过、无AP。HRSID规范路径修复原任务actual1已保留：5604图均解码，13/16969实例为空栅格，未删图/改GT；该失败不证明数据不可用或方法失败。U44单图桥两项元数据测试与独立静态审通过，数值/窗口/外部AP未运行。U40已原状态启动[2560,2816)，固定E50不变。

2026-10-18资源终态补记：第11段[2560,2816)actual1，原受控abort与失败记账已接收。minfreeRAM低于16GiB，未把部分PLAIN迹线/CP当作配对成功，当前仍2560/臂。未确认游戏或元数据测试等单一触发原因；大人口合成测试已停止并禁止与科学叶并行。用户允许真实余量充足后沿原限额另立attempt，原checkpoint/Source/顺序/预算不变。外部CPU restore、window、split metrics三个未绑定函数层已保存，真实外部数值及owner仍缺。


## 2026-10-18: actual U45/U46 coordinate diagnostic accepted

Both original CPU jobs exited0 and were Root-accepted. U45 original morph0 AD [0,1,2,3,0,3,2,1,0] differs at endpoints0/8 from FD [1,1,2,3,0,3,2,1,1]; original morph1 AD/FD agree with the latter. Separate three-line U46 candidate has both-axis AD/FD [1,2,3,4,0,4,3,2,1], matching the predeclared complete-accumulation VJP. CPU-double coordinate sums only: no full Jacobian, DSConv feature interpolation, real images, FP32 network, training or AP. Complete endpoint accumulation already has an author-code precedent; no new sampling-method claim.

[Actual evidence, source and next step](../../research_round_20260925_global/U45_U46_DSC_COORDINATE_ACTUAL_FINDING_AND_NEXT_STEP_20261018.md). Current accounting290 normal/35 failed is independent of unchanged U40 model frontier2560. Do not swap U46 into the running protocol. Main recovery02 control is prepared but unbound; resource observation at07:25:18 UTC was insufficient. Earlier pending-numerics descriptions remain historical; this is the current numerical status.


## 2026-10-18: actual U47 teacher CPU restoration accepted

The original teacher-only job exited0 and was Root-accepted. Complete E24 state610/parameter451 and model/BN bytes matched the original loading proof; frozen eval CPU, no gradients, caller RNG/helper bindings restored, anchor/model released. Worker21.5seconds, sampled peakRSS1632632832bytes; CUDA remained cold. No image/dataset/forward/optimizer/AP. [Actual scope and DGX observation](../../research_round_20260925_global/U47_TEACHER_CPU_RESTORE_ACTUAL_AND_DGX_STATUS_20261018.md). U47 supplies the missing teacher restoration function; current outer pipeline still needs resolved preprocessing provenance and real new-image forward before E50 external predictions.

Accounting291normal/35failed is separate from unchanged U40 paired model frontier2560. DGX SSH/OS/package metadata was actually read without ML import: aarch64/GB10 with about69GiB available system RAM, Python3.12/Torch2.10 in the established main environment. That runtime does not satisfy immutable local U40; no direct continuation, numerical port or speed claim. Main resource admission remains unmet; original limits and E50 endpoint are unchanged.
