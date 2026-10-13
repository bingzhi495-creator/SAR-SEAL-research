# N13：Poly-BRBLE 全文核查

2026-10-13。已取得并阅读 TGRS2025 原论文全部17页文本，方法页与消融表另作视觉核对；未运行作者实现。[官方教师页](https://faculty.ecnu.edu.cn/_s33/tk2/main.psp)指向该作者课题组；[原论文PDF](https://s3.3sobs.cn/paper/2025/Shuwei%20Tang%20et%20al_Poly-BRBLE_A_Boundary_Refinement-Based_Individual_Building_Localization_and_Extraction_Model_Combined_With_Regularization.pdf)题名、作者及[DOI](https://doi.org/10.1109/TGRS.2025.3554255)一致。PDF仅临时阅读，不发布；本次未确认作者代码。

§III-B2、式(8–9)、Algorithm1（p.5–6）明确组合 Laplacian 边界buffer、不确定性及点精修：逐级上采样时先更新几何buffer，再执行3K不确定点＋K随机点的MLP修正，结果进入下一轮SAIU；§III-C另设buffer与point监督。这已覆盖“几何边界＋不确定选点＋局部头修补”的宽泛概念，不能以这些原语宣称首次。

差异在于它学习语义/实例融合并多级重预测，面向光学建筑；N13用冻结SAR特征作残差，仅在所选原图区域生效。N13的K来自LOG面积，GEO按预测异类的Chebyshev距离D、absL和flat index作词典排序，取同K；这是编辑域与监督位置的联合处理，并非其4K采样的同一预算定义。差异尚不能证明效用或创新。

§IV-E、TableVIII（p.12）引入点精修时也改变融合/上采样结构，未提供固定头、固定容量的sameK支持对照。因此本项目关键控制是同教师、100query、类别分数、空预测、40,497参数头、初始化和三seed/48epoch，区域外保持原输出；还需后续区分编辑域与监督采样作用，核验实际AP、修复与破坏、全流程成本及跨集稳定性。未经统一划分、可比监督与预算、评估规则的论文AP不能作为N13胜负证据，也不能凭局部覆盖或此全文核查认定TGRS成熟。

以上N13设置据当前锁定协议与主任务说明，本次文献核查未另行审阅实验源码；据2026-10-13本次主任务状态，N13 AP尚未执行；该文献记录未独立核验实时运行状态。本次核查未访问实验数据或更改冻结的48轮实验；公开同步由主任务完成。
