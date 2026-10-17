# U38：完整自设计 UNet++ 的输入与资源资格方案草稿

状态：**PLANNED_NOT_EXECUTED**；2026-10-17。仅依据普通发布设计/源码和 U36 普通草稿整理，没有读取绑定 SOURCE、RAW、资格链、权重、图像或缓存，没有执行 Torch/NumPy、实现 binder 或授予训练资格。本方案只检验明确规模下的读取/更新可行性，不支持 AP、收敛、机制有效或创新结论。U36 必须先真实终止并由 Root 接收，不能由其 README 的计划推定通过。

## 已有接口的真实边界

- U34 `build_pair` 的公共初始化为 **68 个参数张量 + 36 个 BN buffer = 104 个 state**，来自明确定义的16个公共根；完整模型容量不同。未来必须实际核名字、shape、dtype、参数/buffer 类型、模块属性和复制结果。以 PLAIN seed **202610173101** 的公共状态原位复制到 FULL，未映射状态与 inactive 边界不变；同种子重建同实例适配器且末层零。U36 旧试点 seed3330/3331 不能追认为已执行 U34。
- U37 调用完整旧 reader 验证，返回 **G 个真实匹配**的 e/M/L/GT，追加 JPEG→RGB400 与 geometry；L 是 **native_L**，不是强 R 输出。旧 native carrier 有 `[100,H,W]` L，旧 F carrier 的 M 是 **matched_M200**，U37 不返回 M100/S0。**这些G版输入已足够全741匹配训练，训练不需要M100/S0。** 单图U32证明全100接口可行，不能外推全部DEV已备齐；也不能因此另设无必要的全FIT M100补采。
- 新训练器须用同一400 geometry渲染 delta，再加原图 native_L；不能直接复用 N28 的200输出 renderer、原损失或主干 runner。RGB uint8→FP32/255，经原整数 content resize、右下800 padding、再400；不声称复刻 teacher 光度规范或旧最佳配方。

## 最小顺序

### 1. 单叶流式全人口 CPU 验证：长训前不能用抽样替代

优先**一个原任务按 `[0,741)` 流式验证全部741图/1661匹配实例**，硬边界 **≤1800s**，CPU affinity0/1、Torch线程2、CUDA不可见且未初始化、RSS≤16GiB、可用RAM≥16GiB。按原 ordinal 加载并释放上一图，不保留全部 F/L/GT/RGB。前版六片128图建议没有可信成本实测依据，现撤销：Root提供的已有N28实测为每叶两臂全741读取并含GPU训练约358s，不能直接等同于新增JPEG解码/几何的CPU耗时，但足以反对无依据地先拆六次准入。此处不捏造新的预计耗时。只有实际超时/资源失败后才保留失败与完成范围，另行审定分片修订；不自动重试、不跳图、不把部分扫描称全量通过。

单叶逐图实际核验：

1. 全741图 metadata 总表的 ordinal/image_id/identity 一一对应；累计1661个原 GT annotation_id 恰一次，无遗漏、重复或重新 matcher。保留低质/empty预测、K0等原实例，不以 IoU 阈值筛选。
2. 原 reader 的 teacher、carrier、native_L、assignment、GT像素摘要等既有严格核验原样保留；每图 G/原 GT 顺序/匹配 query_ids、e/M/L/targets 的 shape、dtype、有限性与原来源一致。未分配 query 不伪标成 GT 或背景训练例。
3. **741个原 JPEG**提前登记精确 path/bytes/sha256，identity.image 原引用闭合；真实 JPEG格式/尺寸、EXIF缺失或1、RGB转换和uint8方向、original/content/canvas整数几何均合格，真实产生有限 `[1,3,400,400]` FP32 [0,1]。不猜文件名、改路径、自动转向或默改 resize rounding。
4. 每图已有 all100 native_L 与自然 query顺序0..99、e0和原matched_M来源/轴映射一致，选出的G条恰对应原GT顺序。**FIT匹配训练的完整性按741图/1661实例判定，不要求额外全FIT M100/S0，不启动无必要的教师补采。** native_L载体不包含M/S0本身不是训练缺陷。
5. 后续完整 **DEV自然100query评价**须独立准备每图对应M100/e0/native_L/原S0/class及ordinal、原图geometry；各字段可分文件，但身份与query轴必须闭合。新M与旧L/S0的同源/渲染校准预先明确，不能按GT重排或以matched_M伪充M100。此评价输入准备与FIT训练资格分开，不阻塞CPUreader、U34共有初始化或小组稳态资格；不因此扩展成全FIT补采任务。

最终报告核 `[0,741)` 恰覆盖一次、GT总数1661、输入身份一致且无失败；无须为接收再重跑全部像素。通过后的引用供训练逐图复用，实际 load 的来源核对仍保留。此全FIT输入资格和独立DEV100评价准备各自记录，不将后者的未完成写成前者失败。若硬边界中断，完成进度仅为失败诊断记录，不授全人口资格。

### 2. U34 + U36 + U37 的连续真实更新资源试点

前置：U36已真实接收、选中图的 CPU输入验证通过；全741 CPU验证为长训硬前置。新源明确整合U34/U37，不能称旧U36已包含它们。

从完整 **metadata** 固定选图，不看DEV/AP：a按 `(G,H×W,ordinal)` 最小，b按 `G×H×W` 最大且并列最小ordinal；优先**同一GPU叶，两臂各按 a→b→a 连续3次更新**。目录同时记录最大H×W/最大G角色及首组实际覆盖情况，不把每个极值都设为额外资格任务。如a/b未覆盖不同shape或G，明确该项未覆盖，并可在执行前从真实人口选择同时具备该差异的高负担b；具体ordinal与选择规则须事前锁定，不能结果出来后换图。人口本来没有差异时如实说明。400编码固定，原图/监督负担仍不同。

两臂同图序/匹配/native_L/损失/几何，U34实际公共104states与适配器一致；FULL保留原有效模块，显式不分配旧巨大Transformer。**每臂模型与AdamW连续存在，跨图不重置**；每图训练 encode一次，仍 save_on_cpu(pin_memory=True)、FP32、query_chunk1、每块1/G、完整图一次更新。分别记录 updates/GT visits，每臂须经历两次以上、不同shape/G的更新，以及已有Adam状态下的下一次forward/backward。旧语义头调用0、inactive梯度边界、有效参数grad存在且finite、原BN一次forward语义保持；允许合法零梯度。

两臂**顺序执行且各自3图位于同一叶**，不同时把两模型常驻GPU，不先人为跨叶保存恢复。Root提供的已接收单图9实例FULL约7.2s实测支持先尝试这个紧凑组；该测量不能替代不同shape/G的真实时限与峰值验证，也不据此编造新组预计秒数。图a第二次出现时保留此前模型/BN/Adam状态，只释放当图临时graph；这些是disposable资格更新，不追计正式训练预算。实际准入后仍以1800s硬限执行；真实超限保留失败，分段仅作为后续明确修订备选，不能自动重试或把多次冷启动拼成同进程稳态证据。

固定 GPU allocator4GiB/整卡观测8GiB、CPU0/1两线程、RSS16GiB/可用RAM16GiB、每叶1800s。400编码固定不等于资源固定：CPU all100 L容器、原H×W、G组target/renderer与反向次数均变。训练只消费U37原G匹配输入，不额外把全部100query全分辨率副本常驻GPU；自然100推理按原序另行分块，S0/class不变。

在相同边界记录资源：CUDA同步、删当图graph/临时tensor后，分别记仍含常驻model/Adam的allocated、reserved、RSS和实际CPU保存/恢复计数；整臂释放后沿既有要求硬核CUDAallocated/reserved均0。RSS首次升高后平台可能来自CPU/pinned allocator高水位，回小图未立即下降不能单独判泄漏；a→b→a能暴露明显的未释放对象/超限，不能仅凭3点判定缓慢泄漏或无限运行稳定。若出现实质持续增长，保留逐阶段证据并暂停长训，区分活引用与缓存，不用累计pack字节充当RSS；不为证明绝对无泄漏预先增加一大组循环。

### 3. 通过与停止

全741/1661 CPU人口通过、U34真实配对通过、两臂连续代表负载更新及释放在限额内后，才能称具备该固定配方主训练的输入/资源条件；不额外要求全FIT M100/S0。DEV100评价输入在固定正式评价前独立补齐。固定50轮或其他主预算沿正式计划，不凭资格loss选型、延长或换模块。失败叶不转成功；本方案不授权自动重试、不改4GiB/分辨率、不加VJP/AMP/checkpoint。资源通过后仍需公平主训、固定评价、重复种子和范围内的科学论证。

## 建议立即启动的下一叶

**U38 CPU full-population：原ordinal0–740，全部741图/1661匹配实例，单叶1800s硬限。** 枚举完整metadata/全部原JPEG严格引用，再逐图运行已审U37 load、原reader验证、JPEG与geometry检查；不建训练optimizer、不初始化CUDA、不训练/AP，不查找或要求全FIT M100/S0。U34共有初始化可在同一已准入CPU叶以真实CPU构造/复制核验后释放，单独记录104states与适配器一致性，不调用NN前向。输出全人口报告、严格输入索引、图shape/G metadata及拟锁定a/b序列；不复制全量张量。DEV100输入准备独立pending。若尚未实现这些组合接线，应先完成普通源码并实际准入后启动，不能凭本方案称已运行。超限则原任务失败、保留真实进度，slice仅是此后新修订备选。本文件没有启动或绑定任何任务。

依据为实际读取的普通发布副本：`github_release_20261011/research_round_20260925_global/code/U34_FULL_PLAIN_INIT_DRAFT_20261017_01/DESIGN.md`；同公开research_round下 `code/U37_FULL_RGB_MATCHED_READER_DRAFT_20261017_01/{DESIGN.md,rgb_matched_reader.py}`、`code/N28_DENSE_ROUTING_TRAIN_SOURCE_ONLY_20261017_01/reader.py`、`code/U36_REAL400_CPU_SAVED_ACTIVATIONS_DRAFT_20261017_01/README.md`。未读取实际绑定科学资格并将其作为已通过证据。
