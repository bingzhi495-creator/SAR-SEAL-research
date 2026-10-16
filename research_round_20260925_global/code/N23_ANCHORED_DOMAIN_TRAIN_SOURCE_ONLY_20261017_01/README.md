# N23 原监督锚定的C经验池训练：源码与合成测试

状态更新：E01–08及E09–16原进程均实际0并经Root接收，每臂累计26,576次更新；741图真实联核及三类实际样本的GPU梯度等价检查通过。E17–24已恢复完整状态启动，最近原进程复查仍运行；没有新AP。详见[实际阶段进度](../../N23_ANCHORED_TRAINING_STAGE_PROGRESS_20261017.md)。下文源码于执行前形成，15项合成测试通过，合成检查与真实检查范围分别记录。

本轮保留原LOG_PR每GT全部1024点、顺序、重复、标签、Omega门控与原mean-BCE损失系数1，三臂共用N16原matched C点作为固定经验池。两种Omega分母对照与C补充监督均从同一个强E24读出器开始，固定追加24轮。该实现用于建立强对照和检验训练域效用，尚不作为核心创新。

文件：`joint_reader.py`连接相同图像/真实GT/query/e/原K/GT摘要；`dataset_adapter.py`保留原C采样；`loss_terms.py`明确三臂分母与零项；`training_core.py`执行统一拼接、前向、反传及每GT一次AdamW；`train_body.py`负责完整接入、初始化、真实数值检查和阶段状态；`run_train.py`负责独立资源与待外部实际终态接收。`point_heads.py`是原40497参数读出器的原字节副本，非新结构。

在本目录运行 `python -B -m unittest discover -p 'test_*.py' -v`。测试依赖NumPy，使用合成元数据/数组，不导入Torch或读真实数据。真实训练另需PyTorch、psutil及调用方提供的已核缓存、权重、私有运行配置与源引用；没有发布这些大载荷和运维信息，完整训练复现入口仍未完成。

训练使用省去重复ABI检查的相同残差数学。正式更新前必须在三种实际批次（有新增Omega、无新增Omega、K0）核canonical/fast残差、目标函数和全部8参数梯度精确相等。测试通过范围仅限被检批次及统一concat，不冒称所有输入全局数值证明。

原PR packet包含完整matched-L字段，反序列化时可能读入；本实现仅用已有点F/L和标签，不重新提取全量特征，不前向原生教师。新增C点的成员身份及真实标签沿用已独立接收的N16缓存，必须先通过canonical GT/query与GT摘要联核；不将背景空代理混入主监督。

[执行前固定训练方案](../../../refine-logs/N23_ANCHORED_DOMAIN_TRAIN_20261017_01/EXPERIMENT_PLAN.md)、[导致本轮修复的真实覆盖结果](../../N23_REAL_FULLFIT_CACHE_COVERAGE_RESULT_20261017.md)。目前没有新AP、SOTA或TGRS就绪结论。
