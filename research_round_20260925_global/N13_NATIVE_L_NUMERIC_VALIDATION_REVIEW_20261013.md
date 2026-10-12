# N13 原生 L 数值验证静态审阅 — 2026-10-13

状态：普通候选验证备忘录，未注册、未执行。作者只读取独立发布 SOURCE 和官方 PyTorch v2.0.1 源码；未读取科学载荷、GT、缓存、权重、实际 SPEC/Q/RAW，也未导入数值库、运行数值、哈希或改写训练/缓存 SOURCE。ROOT 已报告 cache GPU01 实际成功、cache CPU01 因全 L 的原定 allclose 判据失败；该失败应保留。有限 index3 诊断的实际结束状态未由本备忘录确认。本文件不改变既有判据，也不认定失败仅由浮点差造成。

## 可以直接核对的源码事实

独立发布的 qualify_cache_cpu.py 中 render_original_cpu 对保存 matched_M200 执行：bilinear/align_corners=False resize 到800×800，按 actual_img_shape=(h,w) 取左上 crop，再以同模式 resize 到 original_shape=(H,W)。两段之间实际产生 FP32 中间张量，不能把它当作一次没有中间舍入的理想插值。当前全 L 判据是逐点 allclose(rtol=1e-4,atol=1e-4)。CPU 区域、点索引、标签/区域flag与采样 L 的检查使用保存的 GPU matched_native_L；CPU approximate L 没有替换选择器。

官方 UpSample.h 的 align_corners=False 路径用输入/输出大小定义比例，half-pixel 源坐标为 s(j+1/2)-1/2，负坐标按线性插值的边缘规则钳制；CPU helper 还处理相同大小直接复制和索引/权重边界。这是算法坐标定义的来源。[PyTorch v2.0.1 UpSample.h](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/UpSample.h)

官方 CPU kernel 预计算索引及权重，包含通用递归插值和向量化分支。布局、通道数、输出尺寸和线程条件可能决定路径；该源码不能支持“只要 Python 参数相同，所有设备逐bit相同”的结论。[PyTorch v2.0.1 CPU UpSampleKernel.cpp](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/cpu/UpSampleKernel.cpp)

官方 CUDA forward 使用源坐标/四邻点双线性表达式，分别有 channels-last 与非channels-last路径；前者分派条件包含 channels>=16。按 batch/channel 遍历并无跨通道求和，但匹配子集的通道数量/布局变化可能改变分派。此处仅检查 forward，未把 backward 的非确定性问题套到 forward。[PyTorch v2.0.1 CUDA UpSampleBilinear2d.cu](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/cuda/UpSampleBilinear2d.cu)

这些是官方 tag 的源码事实，**不等于已证明实际安装 binary 的指令、编译选项、FMA/FTZ 或分派行为**。现用 Torch2.0.1+cu118 binary没有重编译，本次也未读取/反汇编或运行它。不能将源码意图直接称为已认证 GPU 浮点误差界。

## 数学推断及其边界

令保存的 FP32 M200 为精确输入 M，理想两阶段算子为 L*=T2 C T1 M，C为crop，T1/T2为边缘规则一致的 bilinear。理想权重非负、行和1，故crop和理想插值不放大最大范数误差。但实际舍入权重和中间输出仍须计入误差。

200→800的比例1/4及规定范围内的half-pixel坐标可精确用二进制FP32表示；这不使加权值的乘加精确。第二段 h/H,w/W通常不可精确表示；坐标舍入、运算排列、FMA或非FMA均可能影响结果。当前实际偏差是否由这些因素造成仍待诊断，不能以“平台差异”先行排除wrong carrier/crop/query映射。

即使输出 L 接近0，邻域 M 可以具有较大正负幅值。抵消后的 |L| 不控制乘加误差；高对比邻域也会放大微小坐标差。因此1e-4+1e-4|L_CPU|不是普适 FP32 定理。该事实不足以宣告当前具体输入必然满足另一阈值。

一个条件性单层界为：

\[
E\le\sum_i|\widetilde w_i-w_i||v_i|
 +\gamma_n\sum_i|\widetilde w_i v_i|+E_{\rm underflow},
\quad u=2^{-24},\quad\gamma_n=\frac{nu}{1-nu}.
\]

这里实际邻域/权重区间、乘加深度 n、坐标精度、FMA/FTZ及underflow项必须由已限定实现推导，不能任意选择 n 或乘安全倍数。坐标区间跨整数时必须覆盖两侧邻域，不能固定一次floor后忽略另一分支。第一阶段有不确定输出时，第二阶段自身误差需对其输入区间计算。

对设备 b，可用逐点形式 Eb<=T2 C E1,b+E2,b，或保守最大范数形式；两设备差由 EG+EC约束。该公式尚未成为实际可执行/已认证门槛。更直接的独立CPU候选是向外舍入区间，覆盖允许的坐标及逐操作舍入，逐像素检验保存 GPU L 是否在区间内。以输入形状/幅度推导界可以；以index3观测残差反推界不可以。过宽区间可能包容错误crop或索引，因此数值包含只是关系检查，不能替代来源和几何身份检查。

## 优先候选：同设备关系重放＋独立CPU选择器核验

候选A分开两项资格：

1. 新独立owned GPU入口只读取已冻结carrier与保存L，使用原固定设备/binary和相同两阶段renderer复算，预先锁定精确逐点或逐bit一致门槛。保存 M 的dtype、形状、query身份、布局/stride与原canonical渲染输入的对应必须有可核验证据。重复同一GPU渲染可证明M→保存L的同设备关系，不能独立复现teacher forward、原SAR影像解码、GT匹配或原M的计算。使用同一backend也不能发现所有backend共同错误。
2. 独立CPU只把已绑定保存GPU L作为唯一选择输入，精确重建 P=(L>0)、B=(|L|<=2)、K、Chebyshev几何排序/固定tie、RLE、私有采样RNG、点idx、regionflag和gather后的native_logits；继续独立核验FIT硬标签。这里可有 exact equality，不需要把CPU approximate L的符号/带内结果替换为正式指标。
3. 两项分别保留来源、实际terminal与证据范围。若候选A不再认证跨设备全L allclose，必须明确新资格没有该更强结论；不能将新成功回写成旧CPU01通过。原失败、原1e-4判据与完整诊断均保留。

保存GPU L的选择关系和carrier→GPU L关系是不同命题。对同一个损坏L自洽地重建区域只能证明自洽；必须同时成立第1项及原carrier/packet来源资格，才具有本轮冻结表示的数值身份支持。

## all100缺失风险，不能用伪重建补证据

新carrier仅保留匹配query的 M200；原GPU曾自然all100渲染再取query。**丢失的未匹配M200不可从匹配子集、class logits或输出L唯一恢复。** 重新用e/F算M或填零都不是恢复原all100张量，不能称teacher/all100完整重放。

已有CUDA源码表明插值数学上按通道独立，但G通道子集的布局和分派可能与原100通道不同。要声称匹配输出canonical精确重放，必须先证明两种调用的算术路径不变，或在有原布局证据时重建相同形状/布局且证明未匹配值不影响匹配输出。即便以dummy通道检验shape/dispatch，该对象也应明示为dummy，范围只到matched query输出，不能伪称原all100数据被恢复。

若当前保存材料不足以证明相同路径，候选A只能先给子集重放的有限证据；不能宣布全all100身份闭合。更强方案需另登记新的冻结补充：保存原all100 M200及两阶段布局/参数或必要GPU中间量，由ROOT重新资格执行，不修改旧产物，也不混用新teacher输出作旧cache原件的身份证明。当前训练只消费matched FIT实例，限定匹配子集关系可能足够支持训练资格，但仍要与DEV正式all100评估路径单独对齐。

## 若采用CPU区间方案，必要前瞻门槛

候选B需要在使用当前残差数值之前锁定：明确binary/kernel支持范围；区间坐标、边缘/crop、浮点操作/FMA/FTZ模型及精确输入约定；向外舍入实现与独立静态/算术审阅；逐点通过规则；完整741/1661人口及异常处理。必要证据包括已锁规则下全人口验证、形状/身份/来源一致、错误crop/query排列/布局等负对照应被拒绝。若区间不能约束到有用精度，保留“不能认证”的结论，不能继续任意扩大。

有限index3诊断可定位在哪一段出现差异、哪些坐标/边缘/邻域条件相关，并检验源码推断；它不能单独建立全数据、全device或全kernel误差上限。现有原allclose仍按原规则失败退出；若将来提出新验证原理，应新版本、新登记、新证据，不追溯修改原判据或删除失败。

本备忘录提出的是可审查候选与证据门槛，无实际数值结论；不是方法收益、充分收敛、创新或TGRS投稿证据。训练v01及全部绑定科学原件保持不变。
