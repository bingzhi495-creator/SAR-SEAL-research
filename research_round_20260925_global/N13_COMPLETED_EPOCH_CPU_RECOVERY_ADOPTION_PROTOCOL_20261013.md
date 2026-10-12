# N13 已完成八轮产物的独立 CPU 恢复与采用协议

状态：**PREPARED / RECOVERY CPU UNEXECUTED**。这是普通研究协议，不是绑定科学 SPEC、运行报告或资格 Q。编制日期：2026-10-13。原 D02 训练 GPU 已由 ROOT 确认实际 exit1、持有进程回收及原进程身份消失；本协议不改变该失败记录。

## 发生了什么

原 D02 第一种子 202610120701 的配对训练完成 epoch1–8，并保存九份产物，随后在 `train.py:534` 构造最终报告时引用未绑定的 `common.RELATION_REF_KEYS`，触发 NameError。原 `REPORT.json` 没有产生。源码中的原 CPU 入口 `qualify_train_cpu.py:460` 存在同一作用域问题；不能将它直接用于本次恢复。

原训练的 `train.py:508` 已保存两臂 checkpoint 及 PROOF 并逐个完整读回检查；513 保存 DECISION；515–526 位于失败之前，包含外部 RNG 不变、GPU allocated_after 等于零、CPU/线程资源、实际导入文件及 SOURCE/SPEC/admission 一致性检查。根据 ROOT 提供的实际 traceback 与冻结代码路径，这些前置断言已执行通过。该控制流证据与独立 CPU 数值验证分开记载。

`train.py:552` 的整叶 deadline 断言位于失败之后，**没有执行**。ROOT 提供的实际持有运行时长为 **635.797 秒**，低于预锁定 1800 秒；这是 ROOT 运行证据，不能表述为原代码的 deadline 断言已通过。553 的报告写出同样没有执行。

## 恢复范围及原件保留

采用一个新的、独立绑定的纯 CPU 恢复入口和 ABI。它仅验证并形成恢复资格，不补写原 REPORT、不更改原失败退出、不改原检查点、PROOF、日志、源码、SPEC 或 admission。

锁定九份原始产物：

1. `START.json`。
2. `INITIAL_LOADED_STATE.pth`。
3. `TRACE.jsonl`。
4. `UPDATE_TRACE.jsonl`。
5. `HEAD_LOG_PR_SEED202610120701_EPOCH08.pth`。
6. 上述 LOG_PR checkpoint 的 `.PROOF.json`。
7. `HEAD_GEO_PR_SEED202610120701_EPOCH08.pth`。
8. 上述 GEO_PR checkpoint 的 `.PROOF.json`。
9. `DECISION.json`。

新恢复清单以这些原件及原 SOURCE/SPEC/admission/实际失败 RAW 的精确引用为来源。清单是新 CPU 核验的派生产物，明确标为 artifact recovery/adoption；它不是原 GPU 的报告。完整缓存、原 PR factory、已资格化的 M→L 关系及其实际祖先链保持原引用。

## 执行资格

新入口首先核对 **40 条 normal RAW＋7 条 failed RAW，共47组原身份/admission pins**；核对实际持有终止码、回收、原身份消失以及冻结 admission 身份后，才允许任何 SHA、SOURCE import 或数值 import。第七条失败保留本次 D02 GPU 的实际 exit1。

原训练 SPEC 的六失败链作为精确历史前缀继承，不将新增失败从当前 SPEC 中裁掉，不把失败 GPU 放入 normal 列表。当前 CPU admission、SOURCE、SPEC、bootstrap 及完整来源 closure 由 ROOT 在实际执行前绑定。运行保持 CPU0/1、线程2、空 CUDA、1800秒单叶 deadline，无自动重试。

恢复 CPU 本身不能认证当前持有终止结果。只有实际 CPU exit0 后，ROOT 核对该次原身份、admission、回收和身份消失，并将实际 RAW 与新恢复 Q 完整关联，才能宣布产物已取得恢复采用资格。

## 独立 CPU 必须核验的已保存证据

CPU 从 START、初始载入状态、两臂最终 checkpoint/PROOF、两份日志及 DECISION 交叉重建八轮证据；不依赖不存在的原 REPORT。它核验两臂配对初始权重、原缓存和 factory 身份、有限 FP32 状态、八个参数张量及40497参数、AdamW 矩结构与 step=13288、私有排序 RNG、全部八轮图像排列、13288条配对更新身份及其累计记录。

每臂固定八轮包含5928图像访问、13288实例/更新和13606912点访问。完整 TRACE/UPDATE_TRACE 应与 checkpoint 的历史、更新数、梯度活动计数及 PROOF 相符。所有旧证明的精确性要求保持原样；不能因报告构造失败而放宽数值规则。

最终端点使用原 CPU 独立 PR 层公式、外部 region gate 和同一已资格化点缓存，遍历741图/1661实例。域外 logits 继承原 L；域内仅应用原 point head 残差。稳定 BCE 公式保持：

`max(z, 0) - z*y + log1p(exp(-abs(z)))`。

每实例1024点的均值再按1661实例取平均；editable BCE 保持原来的逐非空实例域内均值定义，不能改成所有 editable 点的合并均值。实例、点和 editable 人口应精确一致。

沿用原 `NUMERIC`：端点 BCE 比较 `rtol=1e-4, atol=1e-6`；稳定逐点 BCE 与原 functional BCE 的算术比较同为 `rtol=1e-4, atol=1e-6`。不新增容差，不用近似原图 L 替换缓存中真实 GPU L。

不执行 optimizer.step、不重算或重演训练更新、不重新训练、不加载原生模型、不读取新 GT、不评估 DEV/TEST/AP。保存的 AdamW 状态和私有 RNG 作为原件直接验证，供通过资格后的接续使用。

## 缺失运行观察的边界

原 REPORT 未产生，因此原 allocator peak、allocated_before、实际 imported_library_refs 完整向量及报告构造时 elapsed 数值没有在该报告中存档。不能从检查点推导或在派生清单中伪装为原 GPU 观察值；若 ROOT 有独立运行证据，必须保留其真实来源和测量范围。

515–526 的断言通过可依据冻结代码与实际失败位置记录为原控制流证据。恢复 CPU 自己的导入来源、资源和耗时属于新 CPU 运行。GPU 整卡峰值与 PyTorch allocator peak 不是同一指标，不能相互代替。

CPU 核验的是完整更新访问记录、状态结构/步数和最终缓存点端点。它不独立重放每一步梯度或 AdamW 运算，也不独立重算 epoch1–7 的中间端点；这些中间损失保留为绑定 GPU 记录。所有相关标志及证据限制应明确记录。

## 取得资格后的研究顺序

当前科学配方不变：LOG_PR/GEO_PR 两臂、同头/同K、种子202610120701/202610120702/202610120703、每种子每臂固定48轮、8轮单叶、相同 AdamW 和 constant learning rate，不因本次失败重置优化器、改变预算或挑选最优轮次。

若新恢复 CPU 实际 exit0 并经 ROOT 终止资格闭合，链变为 **41 normal＋7 failed**。按固定8轮区块的种子交错顺序，先完成 seed202610120702、seed202610120703 的首段冷启动叶及各自资格，再处理 seed202610120701 从 epoch9 接续。第一种子的两臂从原 epoch8 checkpoint 精确载入权重、AdamW、私有排序 RNG、历史和活动计数；不重复 epoch1–8。

后续训练 v03 必须有明确的 recovery-parent adoption 分支，验证新的恢复清单/Q/实际 CPU RAW 与原失败 GPU 及两臂 checkpoint 的精确关联。旧 v02 的成功 GPU＋CPU 相邻 normal-parent 接口不能接受这次失败 GPU 产物。**该后续训练采用分支目前尚未实现；本协议不表示接续已开始。**

恢复采用成功只支持保留已完成训练工作和按原配方继续。它不提供密集掩膜 AP、泛化、跨数据集、创新性或 TGRS 投稿级证据；这些结论仍依赖后续公平三种子实验与独立评估。

## 编制依据

本协议仅静态读取已终止冻结的 D02 `train.py`、`qualify_train_cpu.py`、`cache_contract.py`，并采用 ROOT 提供的实际运行终止信息。未读取 SCI、GT、缓存、运行 SPEC/Q，未哈希或执行科学代码。新 CPU 源码及 ABI 完成 STOP 后另做一次独立静态审阅；实际科学执行和最终资格闭合均由 ROOT 负责。
