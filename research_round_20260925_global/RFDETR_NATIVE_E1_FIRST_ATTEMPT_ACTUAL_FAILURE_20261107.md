# RF-DETR 原生 BF16 首个 E1 尝试

2026-11-07 在 DGX 上执行第一次实际 E1。固定 RF-DETR Seg Nano 1.12.0、公开 COCO 初始权重、576 像素、BF16、seed 5201、741 张 FIT 图。训练器建立了原生数据加载器，并报告每轮 744 个微批次、累积 4、全程 100 轮调度下的 18,600 次计划更新。187 张 DEV 的标注用于数据模块初始化，训练期间验证和测试关闭。两次 BF16 更新的先前独立验收仍有效。 [RF-DETR 固定源码](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/trainer.py)

原始 SSH 和子进程均退出 1。监督器记录运行约 12.18 秒，49 次资源采样，无资源越限及清理异常。原始工作进程记录 `AttributeError: 'Trainer' object has no attribute 'enable_checkpointing'`，随后独立进程核验确认同一启动周期内原所有者、子进程及其进程组均已退出。此轮完成 **0 次优化器更新**，未产生 checkpoint 或 AP。 [原始候选回调](code/RFDETR_NATIVE_E1_CANDIDATE_20261107/e1_resume.py#L142)

错误来自项目 E1 审计回调在 `on_train_start` 读取 Lightning 的构造参数。固定 Lightning 2.6.6 源码将已注册的检查点回调公开为 `checkpoint_callbacks`，该入口先于首个训练批次。修复应核对实际回调及其归档条件，并单独记录真实恢复结果。E1 的验收仍是 186 次更新、原生断点、独立资源闭合。E2 的第 187 次更新需要另行验证。E1 本身不生成模型效果指标。 [Lightning 检查点接口](https://github.com/Lightning-AI/pytorch-lightning/blob/2.6.6/src/lightning/pytorch/trainer/trainer.py#L1256) · [训练启动次序](https://github.com/Lightning-AI/pytorch-lightning/blob/2.6.6/src/lightning/pytorch/loops/fit_loop.py#L375)

新候选仅将该检查改为读取实际回调列表，要求明确存在保存完整状态的回调。原训练工作体对原生 `ModelCheckpoint` 的检查仍有效。回归测试在旧版重现同类属性错误，新版通过完整归档的接受以及缺失、仅权重和类型不明的拒绝。新候选状态为本地测试通过，真实 E1 结果需独立验收。 [修正源码](code/RFDETR_NATIVE_E1_CHECKPOINT_API_RECOVERY_20261107/e1_resume.py)

## 两次修正记录

第一次修正绑定了新的回调，但执行入口仍记录原回调的字节长度与 SHA-256，启动时按原规则拒绝。原始退出码 1，监督器运行约 0.145 秒，无资源越限。已独立确认原进程和进程组退出。该次在模型加载前中止，完成 0 次更新，无 checkpoint、AP 或 TEST。

第二次修正重新绑定入口，并在启动前检查入口内全部七条依赖指纹与完整清单一致。测试从第一次修正的实际请求复现仅回调一项不一致，新入口对同一组依赖的检查全部通过。数据、模型、初始化、学习率计划、精度和验收规则沿用原设计。新的独立请求已启动，训练和归档结果待同一请求返回后核验。原两次失败均保留。

后续已完成并独立验收第 1 轮。 [实际 E1 结果](RFDETR_NATIVE_E1_ACTUAL_ACCEPTED_20261107.md)
