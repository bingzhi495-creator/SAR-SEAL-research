# U57：DGX 全 FIT 原数学更新成对训练设计

状态：**静态设计，未写训练实现、未建模、未启动；不能据此计入训练、收益或资格。** 本设计只阅读 GitHub 发布副本的普通 U34/U39/U40/U50/U54 源码，没有访问运行状态、原始回执、资格链、冻结执行源码、图像、标注、权重或 SSH，也没有导入 Torch。下文是待 Root 审核确定的新规格，尚未锁定，未因同步而读取任何绑定 SOURCE。

## 目标与边界

在 DGX 的独立环境中，从头完成 PLAIN/FULL 各 50 轮、37,050 次图像更新，检验已有改动结构在相同数据、顺序和预算下是否有效。沿用原 U39 `_update`，不引入 U52 候选、梯度重组或新的模型模块。不得拼接本机已完成的 2,816 次更新；新账本从 0 开始，模型、AdamW、BN、RNG、身份及执行资格均属于新 DGX 任务。相同 seed 不构成跨设备或跨软件版本 bitexact 证明。

这是一组结构效用对照，不是新创新的证明：PLAIN 约 47.20M、FULL 约 53.12M 参数，**并非容量匹配**。即使 FULL 得分更高，也只能先支持本配方下整体结构的相对效用，仍需容量/计算量对照、机制消融、重复种子和跨数据集证据。

## 原更新与 U54 reader 接口

静态结论：U54 `PortableReader` 可直接作为原 U39 `_update` 的 `reader` 参数；不需要包一层数值适配器，也不应调用仅支持三步试点的 `_run_arm`。

| 原 `_update` 消费项 | U54 提供项 | 必须保留的行为 |
|---|---|---|
| `reader.records[index]` | 741 条原始顺序元数据 | 索引 0…740，合计 1,661 个匹配实例；不重排身份或重新 matcher |
| `rgb400` | CPU FP32 `[1,3,400,400]` | 原 RGB/255 与几何，不重新解码或改变光度 |
| `e` / `M` | `[G,256]` / `[G,200,200]` | 原匹配 query 顺序，逐查询搬至 CUDA |
| `L` / `targets` | `[G,H,W]` | 原 native L 与原图二值实例 GT |
| `G/query_ids/identity` | 原元数据的深拷贝 | 与 records 完整比对，不把自然 query 轴当匹配轴 |
| `geometry.render_delta` | 同一原始 400→800→原图函数 | 双线性渲染及裁切参数不变 |

`_update` 每图一次 encode，随后每个查询 `loss.backward(retain_graph=...)`，最后一次 AdamW step；其 `finally` 清空当前 item、RGB、特征和逐查询张量，并清梯度。U54 的下一次 `load` 会要求前次五个 CPU tensor 对象均已释放。训练编排不得将返回 item、tensor、异常 traceback 或 hook output 存进报告；日志只保留标量/身份。`reader.last_audit` 自身为普通元数据，可记录其摘要。此接口兼容仅为静态结论；741 图 CPU 实际校验通过后，一次 tiny 全链 smoke 才能提供真实训练中的释放证据。

## 初始化、顺序与配方如何严格取得

1. **初始化直接调用原 U34 `build_pair`。** 固定首组 seed `202610173101`；两个 CPU 构造分别使用 forked RNG；仅对源码已证明同操作的根，将 PLAIN 的 104 个 state 原位复制到 FULL（68 参数 tensor、36 BN buffers）。不同操作的 state 保留各自构造抽样；两个共同 adapter 以相同独立 CPU seed 构造，核全部初值相等且最终 head 真零。保存原构造回执与全部共同 state 摘要，不加载旧 UNet 权重、不声称两整网同初值。
2. **数据身份顺序须由经实际校验的 U54 manifest 固定。** 两臂读取完全相同的 741 条记录与本地便携文件；原 Windows 引用只作导出来源，不在 DGX 打开。
3. **训练 shuffle 沿用普通 U40 的纯元数据算法。** 在 CPython 3.8 中，从普通 `train_core.generate_order()` 新生成独立封存文件：私有 `random.Random(202610173101)`，每轮新建 `range(741)` 后 shuffle，预生成完整 37,050 列表，保存生产者版本、生成器初末状态和摘要。不能读取或复制正在绑定本机训练的 ORDER。DGX 只消费这份新文件，不再次 shuffle；每轮验证恰好全 741 个索引一次，两臂共用同一全局游标与实际输入链。
4. **新 DGX validator 区分 order 生产者与执行消费者。** 原 U40 `validate_order(require_python38=False)` 仍把元数据同当前 Python 比较，不能直接验证由 3.8 产生、在 3.12 消费的文件。新验证器应显式核固定生产者元数据、全部列表/状态/摘要和每轮覆盖；消费者版本另写入 DGX 实验身份，绝不修改旧 validator。
5. **数值配方不变。** FP32、RGB400、canvas800、B1、query chunk1、每图全部 G 的 query-mean BCEWithLogits + softDice(smooth1)；canonical `save_on_cpu(pin_memory=True)`；AdamW lr `3e-4`、wd `1e-4`、betas `(0.9,0.999)`、eps `1e-8`、foreach/fused false，按排序后的 effective 参数建组。保留 inactive 分配及 `grad=None`，旧语义 head 不 forward，其 BN buffers 不变；有效 BN 与 Adam 跨图连续。无 AMP、TF32、compile、VJP、重算、增强或 scheduler。
6. **RNG 与后端单独预锁。** Python/Torch 用完整 seed，NumPy 用 `%2**32`；图序私有 RNG 与模型 RNG 分开。各臂构造/加载后、下一图前设置或恢复独立 Python/NumPy/Torch CPU/CUDA 状态。至少沿用 U50 的 cudnn deterministic=true、benchmark=false、两处 TF32=false；是否采用严格 deterministic algorithms 及 CUBLAS 环境须在正式首段前单独固定、如实记录其算子支持情况，不能中途静默降级或混合后端策略。后端并非跨环境相等的证明。

## 最小实现边界与执行次序

新建 U57 的 body/entry/Linux owner 与独立 schema、操作名、输出目录、完整身份和资格账本。U40 只提供编排先例，**不能直接调用其 `train_segment`**：它写死 Python3.8/Torch2.0.1、原 RGB reader、旧 context 与检查点身份。U50 只提供 DGX 模块闭合和监督先例，也不能把其六步试点改名为全量训练。

新 body 按 U50 的 11 个原数学组件角色，加 U54 reader，实际闭合所有源码和模块路径；不调用要求旧 reader 模块集合的 U39 `_closed_sources` 来冒充新 reader 来源。另闭合 manifest、独立 order、741 个本地 package、入口/监督器/库足迹，以及续段父状态；来源清单按实际引用去重生成，不能猜一个固定数量。原 U39 `_update` 固定为普通先例中的 SHA `378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d`。

| 顺序 | 必做事项 | 放行条件 |
|---|---|---|
| 1 | 完成传输及 U54 全 741 图实际 CPU 校验 | 741 次加载/五字段字节、1,661 G、逐项释放、无失败、原任务终端/监督/来源均经 Root 接收 |
| 2 | 完成 U57 静态实现及元数据测试、独立审查 | 九键 ABI、版本与来源拒绝规则、图序、全局 step、失败计数和成对提交均明确 |
| 3 | 一次性 tiny 全链 smoke，不反复添加检查轮次 | Root 事先锁定微小同序输入及预算，覆盖真实 reader→原更新→保存→受接收的父状态恢复→下一步→释放→原任务终端；建议两个相接的极小片段检验接续，具体步数/监督安排待 Root 确定。记录真实训练、序列化及吞吐；smoke 状态可丢弃，不计入 E50，不以 loss 选模型 |
| 4 | 全新从 0 开始 E50，并只从原 DGX 父状态接续至每臂 37,050 | tiny 全链通过且来源/资源合格后进入完整训练，不重复 smoke；每段两臂同游标、同实际输入链；完整 model/BN/inactive、Adam 映射及矩、模式和全部 RNG 恢复；检查点 CPU roundtrip 实测 |
| 5 | 固定终点统一评估 | 两臂均满 37,050 后按预锁的相同完整评估人口、100-query/后处理/COCO 规则评价；训练期间不读 DEV/TEST、不算 AP、不依中间收益选模型 |

不预设 741 图或 256 图一定能放入 1,800 秒。首个正式片段应采用保守小段，给初始化、完整状态保存、哈希、清理留余量，具体长度待 Root 结合真实 smoke 耗时确定；首个真实正式片段记录两臂吞吐和 I/O 分解。后续段可按真实资源/耗时调整长度，但不改变两臂共同总预算及输入顺序，也不使用 AP 或方法输赢决定。不能从两图试点推算全程 GPU-hours，现阶段不承诺速度或完成时间。

DGX 延续 CPU0/1、两线程、allocator4GiB、RSS≤16GiB、freeRAM≥16GiB、单叶1,800秒及唯一 owner。GB10 UMA whole-GPU 缺失必须记 null，不能冒称通过 Windows 8GiB 门；这属于独立 DGX 资源合同。单臂失败保留真实已进入/已返回 step、部分状态和终端，禁止把单边游标当成配对成功；恢复需新处置记录，不循环重试。每段只在两份状态及输入链完整、原进程终止/回收经核验后推进 DGX 自己的配对前缀；本机前缀不变。

## 当前接口风险与可支持的结论

- U54 全量校验只能证明导出输入在 DGX 完整可读；尚不证明全部图可在原 4GiB 数值路径完成训练。重图及 CPU pinned allocator 资源压力需真实片段观察。
- 同 seed、共同104 state 和 adapter 配对不消除全部容量/结构差异；这组实验不能独立归因到某个模块机制。
- 新 Torch 的完整状态保存/加载须显式选择安全、受准入的 tensor/primitive ABI，并实测 Adam step/moments 设备及参数顺序；不能以只载入 model 或新建空 Adam 冒充续训。恢复资格须验证该时点状态逐项一致，不承诺下一步或完整轨迹 bitexact。
- 后端确定性诊断与候选加速器验证的结果需独立保留。U57 不用“跨环境完全一样”作为效用宣称；也不因原算子 replay 末位差异就宣称原方法失效。
- 单个完整 seed 的公平结果是继续机制研究的依据；重复种子、强外部对照及跨数据集结果仍是论文证据缺口。本设计没有提供 AP、涨点、速度提升或 TGRS 达标结论。

普通源码依据（实际阅读的 P 发布副本，非绑定执行 SOURCE）：

- [U39 `resource_probe.py`](code/U39_FULL_METHOD_CROSS_IMAGE_RESOURCE_DRAFT_20261017_01/resource_probe.py)：`_update`、`_partition`、`_sample`。
- [U34 `shared_initialization.py`](code/U34_FULL_PLAIN_INIT_DRAFT_20261017_01/shared_initialization.py)：`build_pair`、`verify_operation_sources`、`static_state_contract`。
- [U40 `train_core.py`](code/U40_FULL_METHOD_CONTINUOUS_TRAIN_DRAFT_20261017_01/train_core.py)：`generate_order`、`validate_order`、`_seed_arm`、`_train_arm`、`_save_checkpoint`、`train_segment`。
- [U50 `pilot_body.py`](code/U50_DGX_PAIRED_PILOT_BODY_DRAFT_20261018_01/pilot_body.py)：`SOURCE_PINS`、`pilot_contract`、`_sources`、`run_pilot`。
- [U54 `portable_reader.py`](code/U54_FULL_FIT_PORTABLE_READER_DRAFT_20261018_01/portable_reader.py)：`validate_manifest`、`PortableReader.load`、`prepare`。
- [U54 `validation_body.py`](code/U54_FULL_FIT_VALIDATION_DRAFT_20261018_01/validation_body.py)：`run`、`completed_population`。
