# Instance-RSDD 同载体 PointRend 对照

在已经评价的 Instance-RSDD 发行版上增加直接先例对照。保持 HRSID 训练的 RF-DETR E24、740 张目标图、全部原生候选及分数。比较在进入目标评分前锁定，以 583 张高分三号影像为主。原 5217、5218 点头和既定复验停止规则继续单列。 [预锁协议](INSTANCE_RSDD_POINTREND_TRANSFER_PRELOCK_20261122.md) · [原两次迁移结果](INSTANCE_RSDD_FROZEN_HRSID_ACTUAL_20261122.md)

单个 PointRend 适配点头已在 HRSID FIT109 闭合。训练池有 392 个有效正匹配，抽样实际覆盖 119 个 RoI 和 78 个来源组。198,402 个参数完成 200 次 AdamW 更新，使用 39,200 个监督点。训练工作进程实测 8.46 秒，正常退出并核对结果。沿用作者式点头及不确定点采样，使用同载体的原始查询掩膜和投影特征代替独立训练的 Mask R-CNN RoI 粗头及 FPN 特征。 [源拟合汇总](results/RFDETR_POINTREND_SOURCE_FULL_FIT109_ACTUAL_SAFE_20261122.json) · [PointRend 原文](https://arxiv.org/html/1912.08193v2)

首次特征捕获因首图未能逐位复现原输出受控停止。原 RF-DETR 1.12.0 导入会将 CUDA 矩阵乘法精度设为 `high`，首次新程序在导入后覆盖为 `highest`。恢复原导入顺序后，全量在 740 图逐位核对原分数、候选、框、掩膜和 43,252 个合格原生查询平面，所有查询的空间基重构也逐位匹配。修复后的捕获实际结束，退出码 0，核对结果哈希 `fe1ae6783c3062f93a7c8ad5c2b6787f041232a2b5ca8d00b778df81308360a6`。工作耗时 123.03 秒，CUDA 分配峰值约 274 MiB。捕获过程未读取目标标注。 [实现和复现入口](code/INSTANCE_RSDD_POINTREND_TRANSFER_20261122)

目标端的同量回放现已开始。掩膜精度待回放和独立复算闭合后更新。
