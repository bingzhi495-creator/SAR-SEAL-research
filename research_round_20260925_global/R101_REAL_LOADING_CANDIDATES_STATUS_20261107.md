# R101真实加载候选：helper与CPU完整模型worker已冻结，尚未执行

2026-11-07。Root已实际将178793939字节V1候选暂存DGX，原tool/SSH0、完整SHA与本地私有副本一致；只完成输入字节准备，没有导入Torch、反序列化checkpoint或取得模型资格。论文实际初值仍未知，当前随机初值512轨迹与主PLAIN/FULL不改。

此前520/586元数据检查器只验证调用方声明。本轮新增可执行真实张量逻辑的[actual_load候选](code/R101_ACTUAL_TENSOR_LOAD_DRAFT_20261107_01/README.md)：未来从真实source_state和model现场提取元数据，独立clone完整源、model state及named buffers，按严格520项白名单生成完整state，strict=True/assign=False加载，然后逐key核dtype/shape/numel和torch.equal值，检查全部非白名单（含nonpersistent buffers与FP64 schedule）、参数对象/可训练标记和CPU Torch RNG不变。相等范围是明确的数值相等，不冒称逐bit。函数不能自行证明来源、模型身份或资源资格。

另一[CPU factory worker候选](code/R101_CPU_MODEL_LOAD_PROBE_DRAFT_20261107_01/README.md)未来调用原Trainer.build_model，从固定作者兼容source/config构造完整DiffSAR，显式CPU-only、seed40244023、worker0；不创建Trainer/优化器，不加载FIT/GT、调用forward或保存checkpoint。它才会以weights_only=True/map_location=cpu读取179MB文件并调用真实helper，验证真实完整R101/FrozenBN/33调制DCN、12项FP64 schedule、500proposals/6heads及前后模型属性；成功report仍qualified=False，必须经Root真实外部tool/parent/child/来源/资源和严格整组闭合后限定接受。

有限验证：helper完成11项标准库fake-tensor测试（0.348秒/实际测试工具0），worker完成6项SYNTHETIC contract测试（0.004秒/实际工具0）。原TDD日志留私有；这些测试没有运行真实Torch、读权重或构造模型，不能把fake返回的成功字段当实际结果。Root读完整两份源码，另一独立继承模型代理做窄集成审阅，未见具体接口阻点，不冒称跨模型审计。

静态核查真实作者属性及接口，纠正了调制DCN stride/dilation存int而普通Conv2d存tuple的差异；不能在运行时随意改其值。另发现归一化均值/std不是model属性或buffer，而是normalizer函数闭包捕获tensor：worker会检查并独立clone它们，加载后再比较，避免遗漏普通state_dict之外的状态。未输入图像，真实图像量纲与是否二次归一化仍未证明。

该闭包也限制后续GPU迁移：不能把CPU模型直接.to(cuda)就声称训练就绪，因为闭包tensor并不必然随注册参数/缓冲迁移。正式GPU初始化/实际有限前向还要另做；本CPU候选不执行CPU DCN前向（作者实现拒绝该路径）。CPU Torch RNG范围也不能扩称Python/NumPy或CUDA全部RNG守恒。

Root已选择独立300秒generic parent候选（只改旧常量1800→300，保留原owner/资源/真实wait/清理），实际stage/original/post/close仍在准备。旧Native训练post/close不能原样用于此CPU验证，Root须另绑定exact spec/lease、检查原主段真实结束并授至多一次资格；控制未就绪时主训练继续，不为候选代码占用GPU空闲。资源闭合与加载成功分别判定，UNKNOWN不释放、非0不升级、不自动重跑。

源码只作为未运行候选公开。实际checkpoint反序列化、完整模型构造、Tensor加载、库/CPU运行兼容性及外部闭合仍待；没有新方法AP、强复现或创新结论。原始依据与已知范围见[公开初值入口](DIFFSAR_INITIALIZATION_GAP_NEW_EVIDENCE_20261107_02.md)、[字节取得状态](PUBLIC_R101_INITIALIZER_BYTE_INTAKE_STATUS_20261107.md)、两个候选README中的原始源码链接。
