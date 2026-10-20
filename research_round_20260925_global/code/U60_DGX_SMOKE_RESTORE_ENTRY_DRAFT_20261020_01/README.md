# U60：独立 SMOKE2 的 CPU 真实恢复接线草稿

**状态：SOURCE-ONLY / UNEXECUTED。** 本目录仅有普通源码、stdlib 元数据和注入 stub 接线测试。没有运行真实构造、Torch、NumPy、owner、子进程、SSH、权重恢复或评估；测试通过不是实际恢复成功或 Root 接收。不得据此推进 FORMAL 训练计数、报告 AP、宣称可投稿。

目标仅为未来 Root 在先前已独立接收的 **SMOKE2** 端点上实际执行一次 CPU 技术检查：用原 `shared_initialization.build_pair` 构造 PLAIN/FULL 两个不同对象，再分别调用 U58 `restore_smoke_arm`。不接受任何 FORMAL 中途或 E50 端点；不训练、forward、matching 或评价，不读取 FIT/DEV/GT，不写新模型 checkpoint，不恢复优化器或 RNG 训练状态。U58 为验证完整 CP 签名会 CPU 加载其优化器/RNG 载荷，检查后释放；这不等于恢复或接续训练。

## 文件及接线

- `entry.py`：无科学导入的 CLI；`--spec` 和 `--spec-sha` 由真实 owner 传入。闭合实际库和源码、四份 JSON 与两臂 CP；导入真实原模块及依赖别名；执行原 CPU 构造和两臂恢复；保留完整/部分报告，退出前复查、释放、恢复调用方 CPU/NumPy/Python RNG 与模块别名。所有结论待 Root 接收。
- `linux_owner.py`：从普通 U54 owner 作 **U60 专用有限适配**。保留自有 session/group、`PR_SET_PDEATHSIG`、真实 pid/start_ticks/boot、资源监控、仅自有组终止、wait/reap、日志和实际退出生命周期。替换了硬锁的任务契约、输入角色和恢复完成判定，未做任意任务执行框架。
- `test_metadata.py`：仅 stdlib；覆盖 CPU 环境拒错、实际调用构造函数这一外层路径、两臂独立对象与顺序、FORMAL 入口拒绝、Source 角色拒错、第二臂失败后保留部分报告且不重试、owner 资源/终态元数据门禁及原监督结构检查。stub 结果永不构成真实网络证据，也没有重复 U58 内部恢复测试。

真实运行后仍只可产生 `COMPLETE_PENDING_ROOT` 的 body；worker 为 `COMPLETE`，owner 成功状态为 `OWNED_CHILD_COMPLETE_PENDING_ROOT_REVIEW`。保留 worker `REPORT.json`，owner `RUN_STATE.json`、`OWNER_RECEIPT.json`、`stdout.log`、`stderr.log` ABI。没有自资格或自动重试。

## 固定条件及直接 Source 闭合

环境固定 Linux/aarch64、Python **3.12.14**、Torch **2.10.0+cu130**、NumPy **1.26.4**、psutil **6.1.1**；必须实际 CPU0/1、2 线程、可用 RAM≥16 GiB、RSS≤16 GiB、1800 秒，`CUDA_VISIBLE_DEVICES=''`，Torch CUDA 始终未初始化。父 owner 监督整个自有 group；子控制每次闭合块和阶段边界检查。CPU 检查不提供 GPU 显存或速度认证。

Root 必须新建一个精确 `ROOT-U60-DGX-SMOKE2-CPU-restore-spec-v1`，operation 为 `CPU_U60_DGX_INDEPENDENT_SMOKE2_RESTORE`。没有示例真实路径、虚构实际 ref、实际 SPEC 或接收记录。其字段为：

| 字段 | 必须的实际内容 |
| --- | --- |
| `entry_source` / `owner_source` / `restore_helper_source` | 当前 U60 entry、专用 owner、已钉住的 U58 helper 的实际 `{path,bytes,sha256}` |
| `restore_sources` | U58 精确 12 个恢复角色：U57 body/entry、shared_initialization、7 个原构造组件、u36_helpers、u39_training_ops；全部固定 SHA |
| `auxiliary_sources` | 原 image_geometry、portable_reader 两份代码 ref；固定 SHA，只闭合/载入定义，不读取其数据 |
| `library_sources` | genuine psutil/torch/numpy 入口源码的实际 ref，导入前后核验实际 `__file__` 与 SHA；不声称完整 binary/OS closure |
| `endpoint_refs` | Root 已接收 SMOKE2 的 `qualified`、实际末段 `spec`、`pair_commit`、完整 `order` 四份 JSON 的实际 ref |
| `checkpoint_refs` | 同一已接收 SMOKE2 pair 的 PLAIN/FULL 完整 CP ref，不可用 FORMAL 或单臂产物替代 |
| `prelock` | `entry.prelock()` 的精确类型和值；保留 CPU 限额、无数据/训练/forward/AP 约束 |
| `phase` / `output_directory` / `owner_output_directory` | 新独立 attempt，owner/worker 输出互不包含且均不存在；Root 预建其专用父目录 |
| `sources` | 仅上述实际直接读取文件的按路径排序闭合，无额外 FIT 包、manifest、GT、全量训练 Source 表或训练 trace |

U58 的恢复模块参数精确 12 角色；但它还会闭合从实际训练 spec 推导的完整 14 份原代码 ref（含 image_geometry/portable_reader）。因此两者不能混淆，也不能通过省略这两份代码让实际任务失败。四 JSON 的内容会出现训练的历史 ref；**不会**递归读取这些历史 ref、741 FIT 包、训练 manifest 或 GT。原 U57 `source_table` 在这里仅作 JSON 元数据一致性验证。

构造函数必须来自实际闭合的原 `shared_initialization.py`（SHA `4ba968db7688a03d5975c8546607791a5e09e82ef6c5c7d096cd6a368d1b0292`）。其 `SOURCE_SHA256` 精确七组件。入口没有接收外部初始化 receipt 的参数：receipt 必须来自本次实际 `build_pair` 的返回值，再由两臂 CP 验证对照。`restore_pair` 的可注入接口仅供外层单元测试；只有 CLI 的实际来源闭合、原调用与完整 owner/Root 外部终态证据一起，才能支持技术恢复记录。

## 为什么不能原样用 U54 父 CLI

原 U54 `validate_spec` 硬锁 U54 schema/operation、两代码角色、741 package_refs、manifest、741/1661 prelock。其 `completed_validation`、`run_owner` 完成分支及最终 success 又要求741审计、`IMAGE_CHECKS.jsonl`/`private_audit_ref` 与 `CPU_validation_complete=True`。只改两个 validator 不足够，也不得伪装成数据校验。本目录已将这些固定语义改为 `CPU_restore_complete`、两臂列表与 SMOKE2 cursor；没有741审计或数据验证成功声明。`FileAllowlist`、ChildControl、资源和自有进程组生命周期保持原设计。

## 未来 Root 仍须完成

1. 在当前数值 leaf 原任务真实终止并核验后，按所有者规则另行准入一次 U60 技术检查；不得并行启动或从观察超时推断终止。
2. 绑定并逐字节核验真实 SMOKE2 接收 summary/ref、spec、pair、order 和两臂 CP，以及全部真实代码/库来源；不能把 stdlib stub receipt 当作实际已通过证明。
3. 分阶段检查真实环境/资源，闭合新 SPEC 与来源后只调用专用 owner 一次；不在已有输出重启或失败自动重试。
4. Root 外部观察 SSH/owner 真实返回值、owner/child 原 birth 不存在、group 空、source/log/report 逐字节一致；独立审阅两臂完整报告与资源记录。现有只支持 U54/U57 的 Root success helper **不能硬套 U60**，须另保存明确技术记录且不给训练/AP资格。
5. 无论成功失败保留原始终态和部分产物，明确该检查仅支持 SMOKE2 技术恢复，不证明 E50 恢复、真实推理、泛化、效用或创新。

本目录不包含上述真实绑定或执行，故当前仍为 **UNEXECUTED**。
