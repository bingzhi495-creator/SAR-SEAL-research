# 最小启动兼容补丁工具

`apply_diffsar_startup_compat.py` 是本项目自有的标准库工具，不包含第三方完整源码、运维配置或私有路径，不导入或执行作者代码、Torch 等科学库。它只处理 `FULL_MODEL_ENTRY_COMPAT.md` 论证的三个文件、五处精确行修改，面向作者提交 `bdc3362372213bc8c3a1f6d5b7b74dc224d1b871` 的独立源码副本。

必须显式给出独立副本目录；工具不能自行识别外部已冻结的原件，不能对原件使用。默认仅预览：

```text
python -B apply_diffsar_startup_compat.py --source-root /path/to/independent-source-copy
```

检查报告后，在同一独立副本应用：

```text
python -B apply_diffsar_startup_compat.py --source-root /path/to/independent-source-copy --apply
```

每处原始精确行必须恰好出现一次。缺行、重复行、已应用状态、非 UTF-8 文本、二进制内容、越界路径和符号链接/目录联接均拒绝；所有文件先完成检查，才暂存和写入。保持原有 LF/CRLF 换行及其他文本。报告仅含相对文件名、每文件 before/after SHA-256、五处变更说明和状态，不写入绝对源路径。

退出码：0 为 `PREVIEW` 或经读回核对的 `APPLIED`；2 为 `REFUSED`；3 为 `PARTIAL_APPLY`，表示文件替换开始后发生 I/O/读回/清理错误，必须检查该独立副本。逐文件替换为原子操作，三个文件整体不是文件系统事务；不能把未取得完整 `APPLIED` 的副本投入后续实验。工具不保证对抗同时篡改源码目录的其他进程，应用期间副本应由调用者独占。

`test_apply_diffsar_startup_compat.py` 使用标准库、临时文本目录和正常 CLI，验证预览不写、显式应用及哈希、CRLF 和非目标内容保留、缺行/重复行在任何目标写入前拒绝、重复应用拒绝。本轮四项测试通过；未对作者科学源码进行导入、模型构造或数值测试。

```text
python -B test_apply_diffsar_startup_compat.py
```

这仅是可审计的启动兼容工具。完整模型、真实单批、训练和公平 AP 的状态必须沿各自实际执行记录报告，不能由文本补丁测试替代。
