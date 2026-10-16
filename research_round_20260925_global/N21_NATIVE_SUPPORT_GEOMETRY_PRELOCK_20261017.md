# N21：FIT 像素九邻域的原生支持几何检验

> 执行后补记（2026-10-17）：原任务实际0、新终态前缀实际0和Root外部接收均已完成，按原规则判为 `SHARED_FOUR_PREMISE_REJECTED`。见[完整结果与范围](N21_NATIVE_SUPPORT_GEOMETRY_RESULT_20261017.md)及[普通汇总](results/N21_NATIVE_SUPPORT_GEOMETRY_SUMMARY_20261017.json)。以下为执行前预锁原文，样本、阈值、分支及限制原样保留；其中“尚无结果”等表述属于执行前历史状态。

2026-10-17。**执行前预锁；尚无真实诊断结果。** N20 两种子已完成外部接收，meanDelta=+0.04503879895506202 AP点，按原规则停止当前配方，不补seed203、预算或融合。N21是一次新的几何前提检验，不是原主终点的补救，也不是新网络或涨点实验。

## 唯一问题与实际路径

在已核当前渲染路径和新固定的FIT均匀整数中心上，像素±1九邻是否至少80%共享一个相邻原生2×2单元的支持？这一比例若不足，应撤回“普遍共享四角”前提。即使足够，也只支持理想插值几何冗余，不证明实际特征、LayerNorm后表示、任务信息或AP存在瓶颈。

Root接收后，对已结束的实际绑定源码作了窄查：`diagnose.py:200–209`将F按通道分组，由原生网格bilinear至800×800，左上裁剪到actual_img_shape，再bilinear至original_shape；两次align_corners=False。`cache_fit.py:204–224`硬检查F为(1,256,200,200)，`capture_body.py:162–180`实际调用该renderer。`coordinates.py:36–40`在原图整数坐标取±1并分别clip。源码身份保留在私有执行契约；未把两次resize合并成一次。

## 执行前固定规则

- 输入仅为已完成FIT邻域缓存REPORT的741行ID和尺寸元数据；不读权重、特征张量、图像、GT或DEV/TEST。原生网格固定200×200、canvas固定800×800；每行crop/original尺寸取identity字段，字段缺失、不一致或越界即停止，不能跳过后另挑图。
- seed固定202610170301。image_key为行内image_id的字符串形式，要求741行唯一，且identity.image_id与行内ID一致。按SHA256(`N21-image-v1|seed|image_key`)排序，image_key破同哈希，取前32图；不按尺寸或误差选图。
- 每图产生64个不重复原图整数中心，若全图不足64像素则取全部。候选序号k从0起，分别以SHA256(`N21-center-v1|seed|image_key|k|y`)和结尾x的完整digest整数，模原图H/W；按首次出现保留。候选上限max(10000,100×目标数)，不足即失败，不换seed。
- 对每个中心的九邻，逐级组合精确有理数双线性权重，输出原生支持索引、分子分母、支持并集数和9行系数矩阵的精确秩。使用Fraction，不引入非零或rank容差；这是理想算子代数，不是实际FP32插值的逐位核验或特征数值秩。
- 主分母只含九点均没有原图邻域clip、第一阶段resize clamp、第二阶段resize clamp的组；边界组单列，全部组也完整报告。主分母不足全部组50%，判为INCONCLUSIVE_CLAMP_COVERAGE，停止该解释分支，不改排除规则。
- “同单元”要求全部正权重支持的y、x各至多两个相邻原生索引，允许边界退化。主分母足够时，以精确整数比较5×same≥4×main判为GEOMETRY_PREMISE_SUPPORTED，否则SHARED_FOUR_PREMISE_REJECTED。

均匀中心不代表历史ERROR采样点或训练点分布。报告比例限于所锁FIT子集与采样规则，不据此推断整个数据集、真实船舶轮廓或失败实例的比例。

## 分支与执行边界

前提被否定：撤回共享四角解释，回到实际误差和可用特征分析；不围绕该解释追加模块。前提被支持：仅允许考虑另行预锁的原生邻域对像素F9比较，仍须控制物理感受野、容量、初始化、监督和预算，不能自动启动长训。

[直接先例核查](N20_NATIVE_GRID_SAMPLING_DIRECT_PRIOR_CANDIDATE_20261016.md)已经记录LIIF等原生邻域先例；原生九点、坐标和局部集成本身不作为新原语。支持代数8项及选样/汇总8项UNIT_FIXTURE均已通过；这不认证实际FIT结果。

通用源码与测试：[插值支持](code/N21_NATIVE_SUPPORT_GEOMETRY_SOURCE_ONLY_20261017_01/native_support.py)、[选样与汇总](code/N21_NATIVE_SUPPORT_GEOMETRY_SOURCE_ONLY_20261017_01/fit_geometry.py)、[支持代数测试](code/N21_NATIVE_SUPPORT_GEOMETRY_SOURCE_ONLY_20261017_01/test_native_support.py)、[选样与汇总测试](code/N21_NATIVE_SUPPORT_GEOMETRY_SOURCE_ONLY_20261017_01/test_fit_geometry.py)。两组测试均为独立UNIT_FIXTURE，不含本项目真实数据。该通用工具公开不等于发布原始FIT缓存或证明新方法有效。

实际执行将新建独立Source、契约和输出；只绑定此诊断所需的直接输入、源码与已完成父结果，不重新运行历史COCO或全模型来源门。既有资格链和失败记录完整保留。单一Root所有者、CPU0/1、两线程、CUDA为空、1800秒、RSS上限16GiB、可用RAM至少16GiB、整卡观察上限8192MiB、无自动重试。输出在原任务实际退出和外部接收前保持pending，不由程序自证成功。
