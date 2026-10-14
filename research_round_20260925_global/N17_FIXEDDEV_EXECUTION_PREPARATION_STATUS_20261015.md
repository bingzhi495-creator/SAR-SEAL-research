# N17 固定DEV：执行准备与边界

2026-10-15。完整24轮GPU及独立E8/E24 CPU端点均原实际退出0、Root完整资格完成；新DEV首段原GPU122的01/02（监督18818/11194）均实际退出1，分别为COCO字段接口和Root漏列旧预测Source；未生成START/REPORT/新预测。19个失败完整保留，独立03已单次启动（原监督83596），实际终止和新捕获资格仍待完成；没有新AP。下一步固定主E24 atmost完整187图/380GT/18700自然query，在原冻结teacher/PR/C/K/类别分数和query顺序下比较LOCAL与CENTER。

[数值body](code/N17_ALL100_LOCAL_CENTER_ERROR_SELECTOR_EVAL_BODY_UNBOUND_DRAFT_20261015_01.py)已冻结、一次定向静态审阅READY、发布副本一项tiny SYN原退出0；普通body接口证据不代替实际DEV载体或GPU输出资格。[共享载体必要ABI](N17_FIXEDDEV_SHARED_CARRIER_GPU_REQUIRED_ABI_UNBOUND_20261015_01.md)记录旧物理bundle与shared成员、索引映射、四模型注入、callback与新writer差异，Root实际核了四段metadata完整关联，真实bundle张量校验仍留给下一original owned GPU任务。

沿旧成功四段[0,47)/[47,94)/[94,141)/[141,187)作固定执行分区，不改变训练或科学评价配方；后续8个profile保留E8/E24×两ERROR臂×atmost/exact，主仅E24 atmost，不从输出事后挑最高。四段最终合并后统一全DEV COCO，不平均各段AP。原native与强R预测只有在完整绑定和导出条件吻合时复用；新CUDA模型logits、选点与PR动作需新capture及独立CPU资格。

GPU外层已形成、修正三处确定构造问题并过一次小SYN/定向静态审阅，Root冻结首段Source/spec/admission后原运行已终止1；selected CPU proof core已冻结/审阅但未真实执行，新bundle与参数解码合同已明确。首组增益门与重复停止规则见[预锁计划](N17_LOCAL_NATIVE_LOGIT_ERROR_SELECTOR_EXPERIMENT_PLAN_20261015.md)。[已完成训练资格](N17_ENDPOINT_CPU_QUALIFICATION_ENTRY_AND_STATUS_20261015.md)不能升级为精度、机制创新、SOTA或TGRS结论。


## 独立选点数值core已形成，仍未执行实际DEV资格

[CPU proof core](code/N17_SELECTED_NUMERIC_CPU_PROOF_BODY_UNBOUND_DRAFT_20261015_01.py)及[参数合同](code/N17_SELECTED_NUMERIC_CPU_CORE_UNBOUND_CONTRACT_20261015_01.md)已冻结，发布副本一项tiny capture→hooks合成检查原退出0（test0.059秒），一次定向静态审阅READY；不是真实DEV输出资格。core以独立NumPy核真8/center8字节值、factory与独立CPU公式各对实际GPU logits absolute1e-4；排名只由实际GPU logits的canonicalFP64分数定义，选点/mask/RLE exact，PR两式/完整C与selected兼容及严格sign保持。新bundle解码和Source/全100/query类分数顺序/完整187_scope/实际终止均由尚待绑定执行的外层闭合，core不自证AP或创新。


## 首段original GPU122任务已退出1，原件保留

首47图范围[0,47)，启动时正常122+失败17/139身份pin；原监督18818现在真实退出1，Root先核纯RAW后保留为失败18，总140身份pin。COCO内嵌字典被误当文件引用，失败发生在预测前；原01不可重写，无自动重试。所有E8/E24×LOCAL/CENTER×atmost/exact八策略固定，主仅E24atmost；完整187合并及独立CPU之后才判断主门，不能由单段宣称涨点。新[GPU入口](code/N17_FIXEDDEV_LOCAL_CENTER_ERROR_GPU_SHARD_ROOT_UNBOUND_DRAFT_20261015_01.py)最终SHA1ac666823c07767d4ccf7ae4a3cd2c8dc936e0d9654010c4bef3fcea07637804、[合同](code/N17_FIXEDDEV_LOCAL_CENTER_ERROR_GPU_SHARD_ROOT_UNBOUND_CONTRACT_20261015_01.md)、[小型接口检查](code/test_N17_FIXEDDEV_GPU_SHARD_ROOT_TINY_SYNTHETIC_ONLY_20261015_01.py)保存；发布副本1 tiny SYN原实际0（test0.034秒），一次定向静态审阅READY，只接口证据不代替实际结果。

已有元数据推得四段保守raw上界约5.237/6.853/6.245/4.140GB、单图最大约599.24MB；实际新owned任务仍重算/验证C、输入字节与捕获大小。每段RSS16GiB、单图raw1GiB、总raw8GiB、serialized12GiB、aux1GiB、minfree29GiB；新的捕获输出用空间充足的数据盘，不移动或删除旧冻结载荷。CPU01/两线程/GPU0、allocator4096MiB/整卡观察8192MiB、1800秒、无自动重试。

本次初始入口修正reference函数局部遮蔽、末profile CUDA引用及8×100 mask预算，没有更改科学配方。新writer物理每图1bundle，旧shared输入载体与新评估binding区分；actualGPU extra8/logits/selectedPR及after mask捕获后独立CPUcore核验，不能拿旧模型logits替代。运行中Source/科学载荷/资格链不因同步读取或发布，本文为独立普通状态快照。


[首轮失败与独立02恢复说明](N17_FIRSTDEV_REFERENCE_ABI_FAILURE_AND_RECOVERY_20261015.md)：24轮训练资格不受影响，没有新DEV AP；新执行尚未启动。


## 独立02修正版已原启动

正常122+失败18/140身份pin与全部原失败已核；独立02首47图任务已单次启动（原监督11194），现已原实际退出1，Root遗漏旧预测Source闭包，无START/REPORT/新预测；没有新AP。02修正仅COCO内嵌字段和失败18门，科学配方不变。新[入口02](code/N17_FIXEDDEV_LOCAL_CENTER_ERROR_GPU_SHARD_ROOT_UNBOUND_DRAFT_20261015_02.py)101252字节/SHA c9d82140b39a8d7f664cfefb3290ba5974ddf2e786e2fd0be96c00993960d106，[合同02](code/N17_FIXEDDEV_LOCAL_CENTER_ERROR_GPU_SHARD_ROOT_UNBOUND_CONTRACT_20261015_02.md)与[小型合成检查](code/test_N17_FIXEDDEV_GPU_SHARD_ROOT_TINY_SYNTHETIC_ONLY_20261015_02.py)保存；发布副本同一项SYN原实际0（0.058秒），一次定向静态审阅READY，均不代替真实DEV资格。

Root监督单一科学任务，CPU01/2线程/GPU0，allocator4096MiB/整卡8192MiB、RSS16GiB/1800秒、无自动重试；数据盘新输出，原01未改或复用。同步仅普通发布副本和状态，不读取运行中Source/载荷/资格链。


## 新图捕获解码器已准备

[纯CPU解码器](code/N17_NEW_IMAGE_BUNDLE_CPU_CORE_ADAPTER_UNBOUND_DRAFT_20261015_01.py)及[短合同](code/N17_NEW_IMAGE_BUNDLE_CPU_CORE_ADAPTER_UNBOUND_DRAFT_20261015_01.md)已冻结，一次定向静态审阅READY；发布副本[一项100query×8profile合成检查](code/test_N17_NEW_IMAGE_BUNDLE_CPU_CORE_ADAPTER_SYNTHETIC_ONLY_20261015_01.py)原实际0（test0.403秒）。只核新9key bundle结构、角色/字节/成员顺序、oldshared union恢复、mask littlepadding与core tuple接线，不读取真实DEV或给予实际数值资格。GPU02已原实际退出1，失败19保留；CPU外层尚在构造，未实际启动。


## 第二次实际失败已保留，03尚未启动

见[两次接口/清单失败与恢复](N17_FIRSTDEV_REFERENCE_ABI_FAILURE_AND_RECOVERY_20261015.md)。正常仍122、失败19/总141身份pin，旧结果12个导出strictrefs及复用控制全187顺序已核验并补齐。不存在运行中科学任务；Root等待仅保留19门的新独立03，预定科学配方不改。


## 第三版原GPU已启动

[入口03](code/N17_FIXEDDEV_LOCAL_CENTER_ERROR_GPU_SHARD_ROOT_UNBOUND_DRAFT_20261015_03.py)105070B/SHA4348aba0d60ceacd0afeb6135019972f19a23ac4428ada0866240472573aa5c5，[合同03](code/N17_FIXEDDEV_LOCAL_CENTER_ERROR_GPU_SHARD_ROOT_UNBOUND_CONTRACT_20261015_03.md)及[合成检查](code/test_N17_FIXEDDEV_GPU_SHARD_ROOT_TINY_SYNTHETIC_ONLY_20261015_03.py)保存。发布副本同一项SYN原退出0（test0.141秒），一次定向静态审阅READY。新spec完整保留原02所有Source，并补12个已核旧导出ref；native/R各187图18700query的原顺序全部吻合。正常122/失败19/141身份pin的纯原前缀通过后，Root新单次启动首47图（监督83596）。

03只增加失败19完整身份门及本次显式绑定纠错，COCO、选点、PR、writer和主门不改。尚无新REPORT、AP或创新结论；实际运行中Source/载荷/资格链不为同步读取。后续只观察原监督句柄，终止后按原资格规则推进，其余分区不自动重启。


## 独立CPU数值外层已准备，未实际启动

[单段CPU入口](code/N17_FIXEDDEV_LOCAL_CENTER_ERROR_CPU_NUMERIC_SHARD_ROOT_UNBOUND_DRAFT_20261015_01.py)69212B/SHAd32ef4c5505f94340b75dff5fdd8598fa1a33ea5b7f05097251c24cd356730db与[必填接口合同](code/N17_FIXEDDEV_LOCAL_CENTER_ERROR_CPU_NUMERIC_SHARD_ROOT_UNBOUND_CONTRACT_20261015_01.md)已冻结、一次定向静态审阅READY。发布副本[同一项stdlib假引用header检查](code/test_N17_FIXEDDEV_CPU_NUMERIC_SHARD_HEADER_SYNTHETIC_ONLY_20261015_01.py)原实际0（test0.055秒）；不测试真实权重、SAR数值或执行资格。

CPU own Source gate与CUDAempty独立，失败9–19保留；只有父GPU实际0/完整Source/新bundle与模型资格闭合后才可绑定启动。四新scout的factory与独立CPU式、真实GPU排名/选点、PR两式、mask/RLE和原qids/类分数逐项核验；单段不算AP，最后仍需四段完整scope和统一全187 COCO。尚未实际执行，不自证任何新结果。


## 固定主结果解释规则已实现，尚未输入真实AP

[主结果纯函数](code/N17_FIXED_MAIN_RESULT_INTERPRETER_UNBOUND_DRAFT_20261015_01.py)和[合同](code/N17_FIXED_MAIN_RESULT_INTERPRETER_UNBOUND_DRAFT_20261015_01.md)只接收已资格E24-atmost/percent/171→172→173的有序主摘要；强参考固定68.09938494745545，不接E8、exact或次要最好值。聚合严格为mean(LOCAL)-max(mean(CENTER),strong)，同时单独要求每组LOCAL>CENTER。Root逐行核对预锁规则，发布副本[一项手工边界合成检查](code/test_N17_FIXED_MAIN_RESULT_INTERPRETER_SYNTHETIC_ONLY_20261015_01.py)原实际0（test0.010秒）；没有读取真实AP或赋予资格。

首组非正停止，正但不足1最多172、两组均值门及各组win通过才开173，首组>=1按预锁完成两组重复；三组均值>=1且均pairwin仅支持冻结载体下scout初始化效用。该函数不证明创新、充分收敛、全网重复、SOTA或TGRS。
