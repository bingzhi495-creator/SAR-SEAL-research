# 自定义 UNet++ 与历史训练入口的对应证据

核对日期：2026-10-17。本文补充[模块内部改造的源码核对](UNETPP_MODULE_INTERNAL_CHANGES_SOURCE_CHECK_20261016.md)，回答“这些模块是否有训练入口和训练记录”。本轮仅检查允许读取的普通脚本、两份配置和两份约 10 KB 的历史日志；未读取权重、导入模型、执行张量计算或新增训练。当前 N20 配方和预锁决策规则保持。

## 1. model_full：确认训练入口，历史执行身份仍缺失

`D:/SAR/UNet++/trainer_coco.py` 第 5 行导入 `model_full.UnetPlusPlus`，第 62 行实例化；第 74 行设批量为 2，第 79 行声明 100 轮，第 88 行指定保存位置 `weights/current/111_SSDD.pth`。

这证明该训练入口代码使用自定义 full 网络，不能据此推断历史实际完成 100 轮，也不能确认现有同名权重由今天的源码生成。`trainer.py` 第 5 行导入的是 `model_old_zuobiao`，应与 full 版本分开记录。

批量 2 与已定位的 DSConv 批次偏移抵消问题有关：当前实现存在需要修复并做批次隔离回归的问题。但本次没有核验历史源码快照及实际训练批次，不能据此认定历史训练已经受该问题影响。

## 2. SamplingBoundary：配置与历史短训日志能够对应

两份配置均指定 `model_sampling_boundary.SamplingBoundaryUNetPlusPlus`：

| 数据范围 | 配置与模型入口 | 对应小日志 |
| --- | --- | --- |
| HRSID | `configs/baseline_hrsid_sampling_boundary_unetpp400_r142_bounded64.yaml:28` | `runs/logs/hrsid_sampling_boundary_r142_unetpp400_bounded64_train.json:201` |
| SSDD | `configs/baseline_ssdd_sampling_boundary_unetpp256_r142_bounded64.yaml:34` | `runs/logs/ssdd_sampling_boundary_r142_unetpp256_bounded64_train.json:201` |

表内路径均相对于 `D:/SAR/UNet++/`。`train_semantic.py` 第 17 行通过 importlib 导入模型，第 53 行实例化，第 465 行进入训练，第 479 行保存权重，第 502 行写日志。

两份小日志均记录种子 42、5 轮、每轮 64 批、未加载初始检查点，并指向各自的 `sampling_boundary_r142...latest.pth`。两份当前配置的 SHA256 与各自日志记载匹配。因此可以表述为“存在与配置对应的历史短训记录”，不能把这两组记录描述为充分训练、独立重复种子或实例分割 AP 验证。

日志未记录模型源码哈希；本轮未读取权重，也没有重新认证当时的原始进程终态。故“历史实际调用的模型源码与当前文件完全一致”仍未确认。此边界不应被改写成“从未训练过”。

## 3. 对后续研究的影响

内部结构确有定制，且部分版本具有具体历史训练记录。现阶段应区分代码路由、历史日志、权重身份和实际效用四种证据。后续选择某一版本继续研究时，在独立版本中修复已定位的实现问题，记录源码与配置身份，并在同一载体、数据划分、初始化、监督和预算下比较对应的普通模块与定制模块。

短训记录可以支持该预算下的比较，前提是对照采用同等预算；它不能单独证明充分训练后的效果，也不能证明方法总体失效或达到投稿水平。当前 N20 属于冻结实例载体上的下游读出实验，不替代上述完整 UNet++ 网络的验证。

本补充由独立静态核对形成，不新增 AP、SOTA、新颖性或 TGRS 就绪结论。
