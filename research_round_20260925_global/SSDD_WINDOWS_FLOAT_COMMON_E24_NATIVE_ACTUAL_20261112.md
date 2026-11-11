# SSDD 官方小数标注原生 E24 实测与中心臂接续

原生 A 完整运行 24 轮，访问 17,856 次，每轮 741 张原图加 3 次固定补齐。AdamW 和 EMA 均实际返回 4,464 次。预热 558 次，沿用 18,600 次更新的 E100 学习率形状，固定评价端点为第 24 轮 EMA。原工作体耗时 3,013.735 秒，原外层和子进程自然退出 0，作业树清空。独立 CPU 读回核验原选定 EMA 档案、阶段标记、实际优化器时钟及逐图访问账本。 [小数训练预锁](PSEG_SSDD_OFFICIAL_FLOAT_COMMON_TRAIN_PROTOCOL_20261111.md) · [串行验收方案](code/SSDD_WINDOWS_FLOAT_COMMON_E24_ACCEPTANCE_20261112/README.md)

训练累计保留 34,804 条共同实例访问，1,095 次访问无目标。原空掩膜保护未移除实例。冻结的两种掩膜几何在 16,760 次访问有差异。本节为原生臂的实际账本和验收，完整 A/B 对照仍需中心臂自然闭合，再比对全程随机状态和共同实例。两臂使用同一 COCO Q100 初值、种子 5201、BF16 精度和固定预算。 [修正启动预锁](code/SSDD_WINDOWS_FLOAT_COMMON_E24_ISOLATED_START_20261112/SSDD_WINDOWS_FLOAT_COMMON_E24_ISOLATED_START_AMENDMENT_20261112.md)

首次独立验收误调用启动前的空输出目录检查，原训练已闭合，因此在权重加载前退出。修正后的验收仍按原绑定摘要读入范围、臂别和初值，核对真实输出路径及回放证据，再进行原 E24 档案检查。原训练及产物逐字节保持。中心 B 在 A 独立验收后已用同一训练工作体启动。 B 原进程已核验首轮 744 次访问、186 次 AdamW／EMA，耗时 125.349 秒，1,440 条共同监督和 53 次无目标访问均与 A 首轮一致，原保护移除为零。DEV187 小数主评分及整数敏感性待两臂验收，正式 TEST 继续隔离。 [闭合验收修正](code/SSDD_WINDOWS_FLOAT_COMMON_E24_CLOSED_SCOPE_20261112/accept_float_e24.py) · [独立评估入口](code/SSDD_FLOAT_PRIMARY_DEV187_WINDOWS_OWNER_20261112/README.md)

