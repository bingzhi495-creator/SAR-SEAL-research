# 原评分关闭后的占据界诊断入口

`run_diagnostic.py` 是独立 CPU 入口。源码与小型合成测试已完成，实际诊断尚未执行。已有 oracle、adapter、评分数学及原入口均未修改。

入口固定原评分摘要 `e0d9f9f8e6d1878f9e7d61278a54ddd9bfef88314796ac558b4bb675abbd225c` 和唯一自然终态回执 `d3659561c411709aa32842f58416eb59db4925242d1c402c4e1b09a95dc10e2e`。先核实际 exit0、完整 observed footer/gate 与报告相等，以及本次 `AP_vs_LW=S`、`geometry_vs_LW=S` 均为 false，再加载其他来源。读取的是既存 AP 门结果，不重算 AP。

原准入加载器按 `3a77e660fd31570cf1f4b44e818b28cc6743d414591216a5f274a11139955b05` 加载。继续使用其原始 fit-step 加载和 readout 身份校验，使用准确修复版 `run_dev_readout.py` 的 `543bc7ffae51cb0de5b247fef59eeae6309a155440021d3ed50a7b3716ec8e9f`。旧 `87c78e…` 版本不兼容本次一行 CUDA 概率修复，不允许回退。

原 capture、独立 readback、稀疏编辑、readout prelabel、score prelabel、历史 native 字节、历史分数、原 DEV GT 和四个空栅格身份全部按原引用闭合。概率修复仅核同一已消费回执，不再读取或应用修复张量。新 prelabel 写入后才通过原 `native.dev_file` 打开两个固定 DEV 标签文件。没有 TEST 路径或预测模型入口。

候选和点只做元数据连接，不调用 `prepare_reconstruction`、`reconstruct_predictions`、`_reconstruct` 或任何 scorer AP 函数。随后只调用公开 `fixed_match_geometry` 与 adapter。原 `observed_geometry_summary` 和 stock matching counts 必须与已关闭评分精确相等，否则失败，不改容差或重试。

运行参数沿原评分调用保留以下路径参数。

```text
--capture --capture-accepted --image-manifest
--readout-root --readout-accepted
--historical-native --historical-score --original-dev-dir
--code --spot-source --shapely-site --output
--capture-accepted-sha --readout-accepted-sha
--probability-repair --probability-repair-sha
```

新增参数如下。

| 参数 | 内容 |
| --- | --- |
| `--formal-source` | 原评分入口、准确修复版 readout、score_dev、evaluate_geometry 的目录 |
| `--original-score` / `--original-score-sha` | 上述已关闭实际评分及固定摘要 |
| `--original-score-receipt` / `--original-score-receipt-sha` | 上述唯一原始自然终态回执及固定摘要 |
| `--adapter-source` | adapter 源码，固定 `e094367dd03dae02b98406a74d5700f30c44574369cf61ea3d0646ef2e80ade8` |
| `--oracle-source` | oracle 源码，固定 `16e5929035d2e95f23b240d755bf9c9a80b3c53fd5eb6570467ad975b454edcf` |

Root 使用原已核 CPU 解释器和 `-I -B -X utf8`，设置 `CUDA_VISIBLE_DEVICES=-1` 与 OMP/MKL/OpenBLAS 单线程。输出必须是不存在的绝对目录。入口不启动子进程、监听任务或自动重试。

输出含 `PRELABEL_SAFE.json`、`OCCUPANCY_BOUND_WITNESS_PRIVATE.json` 和 `RESULT_SAFE.json`。候选、匹配和原生像素细目只在私有 witness 中。SAFE 包含聚合界、原观察摘要一致性、来源摘要和 scope，明确 `AP_recomputed=False`、`stored_original_score_read=True`、`TEST=False`。异常保留已写产物且不写成功状态。

3 项新增小测试通过，用时 0.140 秒。测试使用手工闭合回执和普通公开数学接口，核错误摘要/原始终态/门拒绝、原 prelabel 与完整点名册、仅两项 DEV 标签调用、原观察几何精确对照。AP 和预测重建函数在测试私有模块内设为触发即报错，实际分派未触发它们。测试未导入 Torch，未读取实际评分、标签或科学载荷，也未调用 CLI main。
