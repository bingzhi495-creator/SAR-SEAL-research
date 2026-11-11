# Mask RT-DETR-S 的 HRSID 同预算 E24 训练结果

作者 PaddleDetection v2.9.0、HRSID FIT2937、单类 COCO 转移、种子 20261111、FP32 和批量 4。两臂均自然退出 0，原子进程退出及作业树清空经独立核验。固定初始化、CPU／CUDA 随机流、图像与其他非掩膜输入、监督行序一致。按预锁协议使用两臂第 24 轮 EMA 进行独立端点评分。 [预锁协议](MASK_RTDETR_HRSID_FIT2937_E24_AUTHOR_PROTOCOL_20261111.md) · [完整配对验收](results/MASK_RTDETR_HRSID_FIT2937_E24_PAIR_ACCEPTED_SAFE_20261112.json)

| 训练计数和成本 | 原生 A | 中心采样 B |
| --- | ---: | ---: |
| 完整轮数 | 24 | 24 |
| FIT 访问数，含 72 次末批补齐 | 70,560 | 70,560 |
| AdamW／学习率调度／EMA，各 | 17,640 | 17,640 |
| 原等待计时，秒 | 8,015.844 | 8,022.828 |
| Paddle 峰值保留显存，GiB | 18.165 | 18.165 |
| 进程树峰值主存，GiB | 4.207 | 4.303 |

两臂记录 196,391 条保留监督，其中 191,237 条最终掩膜改变。若不回退，中心独有非空 126 条，原生独有非空 128 条。应用保护后两臂各有 172 条原生空掩膜，非空支持差为 0。B 的原等待比 A 多 6.984 秒，约 0.087%，为单次同机观察。B 的三份检查点和末轮档案验收通过，原训练的范围为 FIT。 [A 独立验收](results/MASK_RTDETR_HRSID_FIT2937_NATIVE_E24_INDEPENDENT_ACTUAL_SAFE_20261111.json) · [B 独立验收](results/MASK_RTDETR_HRSID_FIT2937_CENTER_E24_INDEPENDENT_ACTUAL_SAFE_20261112.json)

DEV705 端点在训练闭合后独立导出及评分。训练验收本身未读取 DEV 或正式 TEST，也未产生 AP。 [固定端点评价流程](MASK_RTDETR_FIXED_ENDPOINT_STREAM_FIT_ACTUAL_20261111.md#固定-dev705-运行边界)

