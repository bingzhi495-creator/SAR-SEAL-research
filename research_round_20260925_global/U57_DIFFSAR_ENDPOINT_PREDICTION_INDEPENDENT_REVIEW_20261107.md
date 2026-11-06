# DiffSAR端点预测候选的有限独立静态审阅

2026-11-07。审阅对象为冻结v1 `worker.py`（23173字节，SHA256 `90e3a4f951f84d015e47db05d96ca495c5fdaec0601d279f70db911bfcba4d98`）、README及三项metadata/IO测试源码。另只读已发布普通full-worker副本和固定作者提交`bdc3362372213bc8c3a1f6d5b7b74dc224d1b871`的模型构造、mapper、推理及COCO序列化源码。未重跑测试，未SSH、导入Torch、读取检查点/GT正文/绑定SOURCE/实际资格链或执行模型；未修改候选。

**结论：在本次指定接口与静态可见范围内，未确认具体阻断。该结论不证明真实500训练状态能在1000候选模式严格加载或完成预测，不是运行、AP或方法效用资格。**

| 核查点 | 静态依据及结论范围 |
|---|---|
| 固定端点与来源 | `validate_endpoint_header`拒绝非FORMAL、非37050、非同一训练500协议/来源；实际加载后还调用原`validate_checkpoint_header`。训练worker、config adapter和每个原训练库的字节pin与检查点科学pin逐项相接。Root内联admission的model摘要与实际检查点model typed-byte摘要相同才继续。 |
| 完整模型加载 | 原模型构造后核全体state键、shape、dtype，`strict=True`加载后再计算同一typed-byte摘要；没有整体float转换。原float64扩散缓冲不会因默认float32而被默许窄化。作者Trainer的build_model主体调用同一原`build_model`；EMA关闭时没有应保留的额外EMA路径。实际最终状态仍须运行验证。 |
| 原推理与输出 | 非训练原mapper不传GT；原模型1000 proposals、单步、NMS后输出由原序列化函数全部导出。原postprocess浮点0/1掩膜被允许，没有替换为ROI贴mask、额外排序/前裁100。类别仅在导出时0→1映射一次。137条替身测试只说明I/O不裁100，不是模型测试。 |
| 人口与随机性 | 单187图、manifest索引0..186及唯一图像ID；严格加载后一次重置Python/NumPy/Torch/CUDA种子，再创建mapper并顺序预测，不分片、续跑或每图重seed。该约定明确不同于未知作者论文随机轨迹。GT ref仅宣告给未来评价器，正常角色分离的spec/manifest不读取GT正文，也不独立认证380实例。 |
| 失败与指标范围 | 出错保留FAILED/无自动重试；实际预测完毕也保持qualified=False，需外部Root核原工具终态、来源、资源及出生进程/组回收。worker没有optimizer构造/step或COCO指标调用；AP始终False。CPU segm评价适配尚缺，移除bbox以按mask面积分箱与maxDets100仍是后续要求。 |

**必须保留的外部依赖：** 资格文件当前在worker内做字节身份核验，语义投影由新Root lease的admission承载；因此Root发放lease前必须实际核验该原最终资格记录及checkpoint、协议、来源和model摘要，不能用这份程序或lease布尔值代替该审计。此处没有真实最终admission可检，本次未认证它。新评价lease不能修改既有冻结训练lease或旧资格文件。

仍须实际取得合格37050检查点与DEV187原图manifest，验证1000候选真实加载/完整导出、内存与1800秒预算、严格退出/清理，以及独立CPU COCO输入适配与统一GT评价。三项既有测试只覆盖header拒绝、全部137条导出和数量角色区分，未覆盖上述真实路径；本审阅未扩大其证据范围。
