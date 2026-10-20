# U65 全187 DEV 包 CPU reader 校验草稿

状态：**UNEXECUTED / 私有、未绑定、待 Root 实际接收**。本目录的标准库元数据与 AST 检查不能证明包读取、GPU forward、E50、AP 或方法效用。本文档及入口不会授予运行资格、训练额度或自动重试。

本入口基于普通 U54 `entry.py` / `linux_owner.py` 最小适配，科学正文由同目录的 `validation_body.py` 提供。未来 Root 在真实来源与执行门满足后，才可在 DGX 用原 Python 3.12.14 / Torch 2.10.0+cu130 / NumPy 1.26.4 / Linux aarch64 环境运行。原 owner 还要求真实 psutil 6.1.1。此轮没有运行 owner、child、Torch、NumPy、SSH 或任何真实数据读取，也未制作实际 spec 或资格。

目标是调用固定 U59 `package_io.prepare`，按原 ordinal 0..186 各执行一次 `reader.load(index)`。每次 load 真正返回 `(item, row, originals)`，核验 RGB400/e/M/L 四个 CPU FP32 张量的既有包证明、原100候选顺序和原生 prediction 字典，并在下一次 load 前释放当前返回张量。无模型、teacher、forward、优化器、训练、GT 栅格化/匹配或 RLE 解码。读取 DEV 数据如实记为 `DEV_technical_input=True`，而 `DEV=False`、`AP=False`、`TEST=False` 表示没有执行评价，不表示没有读取 DEV。

## 输入与来源 ABI

spec 恰14键（与其他导出 worker 的14键不同）：

`schema / operation / phase / entry_source / body_source / owner_source / source_refs / library_sources / manifest_ref / package_refs / prelock / output_directory / owner_output_directory / sources`。

- `schema = ROOT-U65-DGX-full-DEV-CPU-reader-validation-spec-v1`。
- `operation = CPU_U65_DGX_DEV_FULL187_PORTABLE_READER_VALIDATION`。
- `package_refs` 恰187个不同 DGX-local canonical POSIX strict refs，顺序0..186。
- `manifest_ref` 指向 Root 根据真实 U64 到达结果形成的 MANIFEST，其 `entries` 为完整187行 U59 entry；每行含 `dataset_index / image_id / query_ids / package_ref`。`entries[*].package_ref` 必须逐项等于 spec 的 `package_refs`。入口不把旧 `rows` 当作新 ABI。
- 包内既有 Windows provenance 保留原始 primitive 历史元数据，不替换其 input/source 记录，不沿历史路径重新打开原 JPEG、bundle、annotation、teacher 或捕获载荷。只有新 manifest 的实际包 `package_ref.path` 由 Root 绑定为 Linux 本地文件。`raw_provenance_inputs_revalidated=False`，不把历史元数据误报成跨主机重新核验。
- `source_refs` 恰4角色，固定 SHA 如下；入口只执行 `portable_reader` 和 `image_geometry` 两模块源码。`dev_reader`、`dev_rgb_reader` 的实际本地源文件参与闭合与 U59 原依赖核验，不执行原 DEV reader 以重开 raw 输入。
- `library_sources` 恰 `torch / numpy / psutil` 三份真实模块源文件 refs；运行时实际导入路径与字节必须匹配。该显式闭合不等于全部 Python、共享库、OS 二进制闭合。
- `sources` 是 entry/body/owner/manifest、4角色、3库与187包 refs 的去重后排序精确 union，不带模型、权重、GT 或额外隐式数据输入。重复同路径只有完全同 ref 可接受。Root 绑定实际 entry/owner refs，body bytes/SHA 由入口固定。
- worker 和 owner 输出必须全新、互不包含，也不得包含任何 admitted source/input。父目录由 Root 预建，不可复用失败 attempt。

入口固定的普通正文源码为 `validation_body.py`，10599 bytes / SHA256 `2550f30b495c68a4680cf17d5615124ad562bba734f00b4cf44ac99022431d49`。此 pin 来自 D 成稿字节核对，未使用 E 不完整副本。

| 角色 | 固定 SHA256 |
|---|---|
| portable_reader | `8fabc800e4d36e2da3988e42acf88058c80790f9c8b3d664a1a4fa9232233473` |
| dev_reader | `daabec8388e23d9ff646ae78eba69ba2168099a72971578a8d34cef04f1268c0` |
| dev_rgb_reader | `b91375e5366da3f74272ce811f453be1379b46a4bb96103f9d8039a4998a1204` |
| image_geometry | `65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7` |

`prelock` 恰以下15键：`schema / environment / dataset_indices / images / queries / CUDA_visible_devices / limits / no_new_training / no_forward / no_matching / no_AP / no_DEV / no_TEST / no_GT_rasterization / DEV_technical_input`。schema 为 `U65-DEV187-CPU-reader-validation-prelock-v1`；indices 为 `list(range(187))`、images187、queries18700。六个原 no 标志及 `no_GT_rasterization`、`DEV_technical_input` 都严格为 True；`no_DEV` 指没有 DEV 评价，与读取 DEV 技术输入相容。

## 实际运行与终态边界

原 CLI 保留：owner 与 worker 都接受 `--spec ABSOLUTE_JSON --spec-sha EXACT_SHA256`。此文档不给实际命令、spec、路径资格或自助启动步骤；由 Root 用真实绑定接入。

资源限额沿 U54 原值：CPU0/1、Torch/OMP/MKL/OpenBLAS 两线程、RSS≤16GiB、可用 RAM≥16GiB、1800秒、`CUDA_VISIBLE_DEVICES=''` 且 Torch CUDA 全程未初始化。父进程单线程、无科学导入；原 preflight 总计时、整个自有 group 采样监控、源码块闭合检查、超限停止保留。RSS/RAM 是原采样与协作检查，不能描述为 OS 硬 cap 或系统峰值保证；CPU-only 检查不能授予 GPU 显存资格。

原 `start_new_session`、真实 PID/start_ticks/boot/session/process-group、`PR_SET_PDEATHSIG` 与父死亡竞态检查、只向自身本次 group 发送 TERM→KILL、真实 wait/reap、PID 重用拒绝、未知 birth 仅清理自有 Popen leader、原 signal handler 恢复及失败日志生命周期不变。不会重试或启动其他任务。Root 仍须独立观测 owner 自身真实终止；owner 自述 returning 不能代替外部终态。

worker 新报告 schema 为 `U65-DEV187-CPU-reader-validation-worker-report-v1`，正文为 `U65-DEV187-CPU-reader-validation-body-report-v1`，资源为 `U65-Linux-child-control-observations-v1`，owner 收据为 `U65-Linux-owned-full-DEV-CPU-reader-validation-receipt-v1`。原 `REPORT.json`、`IMAGE_CHECKS.jsonl`、`stdout.log`、`stderr.log`、`RUN_STATE.json`、`OWNER_RECEIPT.json` 的可收集 ABI 保留。

只有正文实际187次 attempted/validated/released/recorded、18700候选、visits0..186、failed_loads0、剩余返回张量0、全部来源/控制/RNG/dtype/grad/threads/alias/CUDA 旗标真实通过，worker 才可写 `COMPLETE`。每图 scalar audit 仅保存 ordinal/image_id/queries/status/包ref/三个元数据 SHA/CUDA 与释放标志，不保存张量、targets 或 GT 数量。原 matched_instances、741/1661 与 FIT 语义不再属于 U65。

owner 还须核验同一 spec/source/input identity、全部四层资源与状态、真实 child returncode 整数0、已 reap、原 birth absent、自有 group 为空且扫描已解析、无错误/资源违规，才可形成 `OWNED_CHILD_COMPLETE_PENDING_ROOT_REVIEW`。`CPU_validation_complete=True` 仅表示此 CPU reader 技术检查完成；`qualified=False`、训练额度0、创新/提速/AP 结论均不成立。不会套用只认识 U54/U57 的旧 Root 科学成功 helper。

## 本轮验证与尚缺条件

轻量测试只读这几份普通源码、构造 synthetic metadata 并调用 stdlib 元数据 guard，不调用任何 `main`、`run_owner` 或 body 的实际科学执行。覆盖新的187/4角色/精确闭合/原 manifest entries、18700与技术输入边界、原13个 owner 关键函数或类的标签归一 AST 保持以及只加载两模块。没有重复旧 U54 全量测试，测试通过不是实际数据成功。

测试记录：scope `0415be actual1 → 2b954f actual0`；manifest `8db1cf actual1 → 67efbd actual0`；completion `897d3f actual1 → 2c4cdc actual0`。最终4项标准库测试 `1f2eba actual0`，普通源码 AST/旧范围残留检查 `b1c0b0 actual0`；最终测试同时核对 D 正文实际 bytes/SHA 与入口固定 pin 相等。进程保护 AST 是来源等价性检查，并未启动 Linux owner 或替代真实终态验证。

Root 后续必须提供并核验真实187到达包及 MANIFEST strict refs、实际4角色与3库源 refs、此 entry/body/owner 三源码 refs、全新独立输出、当时资源与全局单任务准入，以及真实 Linux owner 终态。任何未读取/未执行部分保留 UNEXECUTED，partial audit/report 与失败原因保留，不因失败删除载荷或自动换任务。E盘先前写入失败的不完整同名草稿没有修改或删除；当前唯一待绑定成稿位于此 D 独立目录。
