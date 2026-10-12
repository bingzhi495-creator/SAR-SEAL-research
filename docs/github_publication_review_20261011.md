# GitHub 发布源码审阅（2026-10-11）

建议从明确的论文源码清单建立干净发布副本，保留原实验目录。现有 Git 跟踪 589 项，其中 409 项文件名属于数据、运行目录、缓存或权重类；不适合直接整体推送。此次只读源码和文件名，未读取科学 JSON、GT、权重、结果或运行载荷，未执行训练、导入科学库或改动原代码。

## 推荐范围与入口

下表 `R/` 表示 `research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/`。先精选 `.py`、数学 fixture 与接口说明；表中目录是源码来源，不代表全部实验已经完成。

| 作用 | 源码来源及主要入口 |
|---|---|
| 当前 9441 参数读出及联合损失 | `R/query_pixel_posterior_coupled_readout_source_20261010/query_pixel_readout.py`；`R/query_pixel_joint_native_source_20261011/` 的 adapter、capture、objective、loss helpers 与 synthetic fixture |
| 联合训练主线 | `R/query_pixel_joint_native_preflight_source_20261011/run.py`；`R/query_pixel_joint_native_fullFIT24_source_20261011_deadline14h/train.py`；后续 `R/query_pixel_joint_native_E24warm_additional48_source_20261011_deadline28h_02/train.py`；各自 `qualify_cpu.py` |
| 固定端点评价与交叉诊断 | `R/query_pixel_joint_native_evaluation_source_20261011_first24_02/evaluate.py`、`qualify_cpu.py`；`R/factorial_geometry_score_source_20261011/` 的 helper、test、CPU runner |
| 冻结骨干与强对照 | `R/query_pixel_posterior_scalar_control_fullFIT24*_source_20261011/`、对应 warm48/repeat 源码；`R/native_finetune24_fresh_optimizer_source_20261010/`、`R/native_finetune48_E48warm_weights_source_20261010/`；保留各目录用途及状态区别 |
| 基础辅助源码 | `R/spatial_position_preserving_common24*_source_20261009/`、实际调用的早期 initializer/native/renderer helpers；仅发布被选择入口需要的源码依赖，不复制整个 research tree |
| 历史实现，独立归档 | `carrier/`、`mne/`、`losses/`、`seeding/`、`protocol/`；入口包括 `protocol/pseg_to_coco.py`、`protocol/seal_eval.py --preds --gt`，早期 MMDetection 配置在 `carrier/configs/`，不与最新 800 栅格配方混称同一实验 |

最新训练/评价 CLI 为 `--spec ABS --owner-execute`，CPU reader 使用其独立 acceptance 参数。它们依赖 ROOT 创建的来源绑定、原生 E24 初始化及前驱资格记录，不是只装依赖即可运行的公共入口。公开版还需整理路径配置、可重新生成的输入模板及辅助源码清单；原 SCI-used 源码保持原样，不能通过删除校验伪装成已验证的可移植版本。

## 依赖与复现限制

已有 `autodl/setup_env.sh:7–11` 记录 Torch **2.0.1+cu118**、MMEngine **0.10.4**、MMCV **2.1.0**、MMDetection **3.3.0**、`numpy<2`，另安装 SciPy、OpenCV、pycocotools；`autodl/setup.sh:5` 注释指定 torchvision **0.15.2**。`protocol/datasets.py:9–10` 还需 NumPy/Pillow。AutoDL README 写 Python 3.10，最新源码使用 Python 3.8 静态语法检查；不能据此宣称两个环境均已完整重跑。最低兼容版本没有专项验证，应先记录这套版本，而非随意写宽泛最低版本。根目录未见统一 `requirements.txt`/`pyproject.toml`；旧 AutoDL 脚本会修改配置路径并下载初始化，不能直接作为最新联合训练的安装说明。

## 排除与版权、隐私边界

- 排除 `data/`、`runs/`、`work_dirs/`、`outputs/`、`tmp/`、`delivery/`、`node_modules/`、环境副本、模型/预测/日志/缓存；所有实际 SPEC、Q、receipt、admission、AUTH 和科学结果另行选择，不从原目录批量加入。SmartFrost、视频等无关工程不属于论文仓库。
- 根目录未见项目 LICENSE，应由权利人明确自有代码许可。外部框架采用依赖链接优先；[MMDetection](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/LICENSE) 与 [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/LICENSE) 的 Apache-2.0 源码保留原许可/署名。`refs/MP-Former/LICENSE:1` 是 CC-BY-NC-4.0；`refs/DiffSARShipInst/` 顶层未见 LICENSE，不能因包含 Detectron2 就推定全部作者新增代码同许可。不要整包发布 `refs/`。
- 数据公开下载不自动证明本项目可再分发；先提供 [Official-SSDD](https://github.com/TianwenZhang0825/Official-SSDD) 与 [HRSID](https://github.com/chaozhong2010/HRSID) 作者链接及转换代码，明确实际标注版本、划分和权限。内部 FIT741/DEV187 不是官方 SSDD TEST232，README 不应把内部指标写成公开 TEST SOTA。
- 对上述候选范围 439 个源码文本做保守规则扫描，未命中硬编码凭证、私钥、私网 IP 或带凭证 URL；这不是全仓库历史安全证明。发现本机用户路径：`carrier/dump_figure_assets.py:24`、`diagnostics/geofield_fps.py:6`；`carrier/tests/` 下 `cpu_test_baselines.py:23`、`cpu_test_minfoot.py:32`、`cpu_test_r6d_cfg.py:25`、`cpu_test_sga2.py:24`、`cpu_test_swins_cfg.py:22`、`gate0_stemlat.py:31`。只报告位置，不复制路径中的身份值。运维 `.ps1` 未纳入本次内容审阅，建议不随论文源码发布。

发布定位应是“研究源码及明确的复现限制”。当前最新链条尚不能宣称下载后可一键复现，未完成的训练、外部 TEST 和资源资格也不得由源码准备状态代替。
