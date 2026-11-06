# 作者运行环境与真实D2算子：实际进展及范围

2026-11-07后续状态：[02和03限定观察](U57_DIFFSAR_MASK_GRADIENT_DIAGNOSTICS_20261107.md)均已实际结束并保留原工具1。02的IAMR参数梯度全零，03在共享输出处记录到存在、有限、全零的总损失梯度。它不能唯一归因饱和，也不证明方法无法训练；01未保存的分组明细仍永久未知，不用后续观察回填。完整训练/AP尚未完成。下文为首次失败和此前准备的历史记录。

最新执行补记：[完整模型首次单FIT实际结果](U57_FULL_DIFFSAR_SINGLE_FIT_FAILURE_ACTUAL_20261106.md)已完成损失和反向调用，随后梯度覆盖失败。原失败不合格，主训练已接续至25088后下一段。下文“尚未执行完整模型”是本页较早阶段的历史状态，当前以新失败报告为准。

2026-11-06。主PLAIN/FULL已各严格接收24832/37050更新（67.02%，33轮+379图），本次新增各512更新/1106次匹配访问，下一段已接续。固定50轮、原始初始化/顺序/预算和主资源限额不变，尚无端点AP。

作者固定提交bdc3362372213bc8c3a1f6d5b7b74dc224d1b871的Detectron2扩展已实际完成两种步长的FP32调制卷积检查。输出与input/offset/mask/weight四种一阶梯度均满足预锁容差；独立参考由grid_sample、掩膜与卷积权重显式组合，未调用另一份DCN。最大绝对误差为1.7881393432617188e-7。ROIAlign与水平NMS另通过解析接线检查；它们委托Torchvision，不宣称独立后端对比。

原工具/SSH/worker及Root后核均0，选定7项父控制引用及12项worker/库/lease引用前后验证，同boot原父子及所属组回收。约3.811秒，组RSS峰值1589837824字节，CUDA allocator峰值17073152字节，清理后allocated/reserved为0/0，无资源违规。算子账户独立1正常/0失败/1总计，不算主训练、重复种子或AP。01未绑定/执行；实际使用02仅增加数值完成后的cuBLAS工作区清理，数学样例、容差均未变。

从官方PyPI逐项核大小和SHA的16个小依赖已装入专用前缀，正常包/扩展、get_cfg、timm实际接口及声明依赖检查通过。该前缀显式共享base site-packages，不是完全隔离；没有向base安装或降级。版本与实际返回值见安全JSON。

原作者完整包首次导入因缺失FPN导出返回1，保留失败。独立源码副本仅改三个文件五处：删除未用且缺失的FPN/FAN/BiFPN导出、删除未调用的poly_focal导入、将Pillow旧LINEAR别名换成同义BILINEAR。其余1118个原文件一致，原归档/源码/编译产物不改；不替换SCJE、DCN、CARAFE、动态头或实际损失公式。

第一轮兼容检查包导入后在未登记的MODEL.BIFPN_1p_5p配置合并处返回1，整次失败不升级。随后按真实训练入口补作者EMA配置初始化，再显式登记这个未用节点的五个typed字段，原YAML不改、new_allowed保持false；新的实际检查返回0。确认R101/SCJE、500 proposals、6 heads、RGB、shortestedge/max512、EMA关闭。此前静态稿误写BGR，发布副本已纠正，附官方RGB与HRSID1024更正说明。

一张按既有741张FIT规范顺序选择的真实原图及原实例标注已独立准备，原网格355×501、1实例。旧原路径不存在，先保留元数据准备失败，再在历史派生目录找回与冻结记录SHA/字节数完全相同的JPEG。不是拿放缩缓存代替原图；尚未解码/执行完整模型。

完整SSDD单FIT候选保留原mapper、完整R101/SCJE、500候选/6头和24项损失，拟一次反向，不做优化器step或权重下载。候选为未绑定/未执行；8GiB allocator、16GiB组RSS等独立资源合同不改变主训练4GiB限额。随机初始化运行检查不能证明作者论文配方复现或效用。固定端点、完整外部训练/AP、容量控制、重复和跨集仍缺，总目标未达成。

[算子实际安全记录](results/U57_D2_OPERATOR_ACTUAL_20261106.json)；[运行/兼容实际安全记录](results/U57_AUTHOR_RUNTIME_ACTUAL_20261106.json)；[可发布启动工具与静态审阅](code/DIFFSAR_STARTUP_COMPAT_TOOLS_20261106_01/FULL_MODEL_ENTRY_COMPAT.md)；[输入说明更正](code/DIFFSAR_STARTUP_COMPAT_TOOLS_20261106_01/FULL_MODEL_INPUT_CORRECTION.md)；[完整模型候选（未执行）](code/DIFFSAR_FULL_MODEL_FIT_PROBE_DRAFT_20261106_01/README.md)。
