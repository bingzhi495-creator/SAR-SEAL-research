# HRSID 仅 COCO 初始化的原生 E24 基线

固定 RF-DETR 1.12.0 Nano 分割和既有原生增强，使用 2,937 张 FIT PNG、9,260 条有效训练实例。每组 164 个查询、13 组、576 输入、BF16、单图批次及累计 4。原生 AdamW、EMA 和训练点损失沿用已完成首轮的设置。此运行从作者的 COCO SegNano 资产重新开始。作者登记的 MD5 和本地 SHA-256 与先前 SSDD 原生基线的 COCO 来源一致。固定种子 5201，先按原生宽度 100 加载并保持单类两槽头，再扩展每组 64 行。独立 CPU 核对 545 个模型张量，并确认两张查询表逐组保留原前缀，其他张量逐值相同。 [作者 RF-DETR](https://arxiv.org/abs/2511.09554) · [官方权重登记](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/assets/model_weights.py#L242-L245) · [初值核验](results/RFDETR_COCO_SEG_NANO_OFFICIAL_START_SAFE_20261109.json)

- 调度固定为 100 轮、73,500 步、2,205 步预热。单次连续运行至 E24，预期 70,560 次 FIT 访问、17,640 次实际 AdamW 和 EMA 更新。每轮独立核对原生补采与目标字段，记录空目标、目标数、有限损失和真实步数。只保留 E1、E8、E24 完整原生检查点及随机状态，E1/E8 不用于模型选择。
- 使用已验收的本机离线环境，单进程 GPU 分配限额 6 GiB、可用 RAM 下限 16 GiB、进程 RSS 上限 10 GiB、实际工作体上限 10 小时。以首轮时间估算约 6 小时，长程实际时间另记。资源中断保存真实退出和完整端点，依据原运行状态处理。
- 训练过程只使用 FIT。完成后以 E24 EMA、固定 `num_select=100` 和已独立核验的 DEV705 做一次原始 GT COCO mask 评价。4 条空栅格的有效掩膜敏感性另报并复用同一批预测。正式 TEST 与训练、筛选均隔离。该基线提供目标域公共参考，单一训练种子及项目内部 DEV 指标的适用范围单列。 [FIT 入口验收](HRSID_NATIVE_FIT_FIRST_EPOCH_PROTOCOL_20261109.md) · [DEV705 资格](HRSID_OFFICIAL_PNG_DEV705_AUDIT_PROTOCOL_20261109.md)

三项合成测试通过，覆盖预锁端点、循环计数及 E100 调度。真实长训待执行。随机状态保留，精确续训需另做连续轨迹对照。 [训练源码](code/HRSID_COCO_NATIVE_FIT_E24_20261109/train_hrsid_coco_native_e24.py)
