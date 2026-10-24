# U57直接先例增量核查：SegS-YOLO

访问日2026-10-25（北京时间）。使用research-lit流程，本轮限定核查三篇；仅阅读普通发布副本和公开原始来源，未访问绑定SOURCE、资格链、数据、权重或SSH，未测试或改变当前训练。下列新增证据不构成新颖性认证。

**实质增量：SegS-YOLO已由待核线索升级为官方原文可核的SAR实例分割结构先例。**

Zhang、Sun、Chang、Tang、Hou，*SegS-YOLO: A YOLO-Based Instance Segmentation Approach for Small-Scale Ship Targets in SAR Images*，IEEE JSTARS **19:2658–2679，2026**。[IEEE登记页](https://ieeexplore.ieee.org/document/11261833/)、[出版社提交的Crossref元数据](https://api.crossref.org/works/10.1109/JSTARS.2025.3635282)及[官方PDF](https://ieeexplore.ieee.org/ielx8/4609443/11278119/11261833.pdf)身份一致。首页另载2025-11-20上线、2026-01-05当前版本，不能只按DOI中的2025判断卷年。

- **结构重叠：** §III-B-3，印刷2663–2664页，DSCPKI结合CAA上下文/条带注意、并行多尺度卷积和DSConv。动态蛇形卷积与上下文、多尺度整合已经用于SAR实例掩膜，不能作为本项目的宽泛首创主张。
- **确为实例分割：** §IV-B、§V-A及表II，2668–2670页，讨论和评价的是mask AP、mask AP50及mask APS；已视觉核对表II列为AP、AP50、AP75、APS、APM、APL。不是仅框检测，本笔记不转录其自报分数或与本项目排名。
- **跨集范围：** §V-C-1，2670页，LS-SSDD缺实例标注，仅作定性展示，不能充当跨集mask AP证据。

本项目与它的具体差异仍是：固定外部候选与原分数，以教师最终实例mask embedding、粗掩膜M200和共享UNet++图像特征预测零初始化残差，经原几何渲染后加到外部原生logit L；SegS-YOLO则改造YOLOv8-Seg主干、neck及三尺度分割头。该计算图差异须由公平效用与针对性控制支持，单一先例不能证明当前整体结构没有贡献。应把SegS-YOLO纳入直接先例差异表；本轮未定位可运行官方代码，不将自行适配称为作者完整复现。作者同名与用户的身份关系尚未确认，不据姓名作推断。

| 另两项限定复核 | 本轮结论 |
|---|---|
| [DACF-Net，IJDE，2026-09-08上线](https://www.tandfonline.com/doi/full/10.1080/17538947.2026.2696657?af=R) | [作者仓库固定提交479dfab](https://github.com/yang199809/DACF-Net/tree/479dfab0048ddb7edae1b732ff5ee4452aa82be8)仅.gitignore、LICENSE、README，仍待发布实现；本轮核到唯一main分支及空Releases。旧代码可用性判断未改变。 |
| [O2Former，arXiv:2506.11913](https://arxiv.org/abs/2506.11913)及[作者所属大学正式发表记录](https://pure.kfupm.edu.sa/en/publications/o2-former-direction-aware-and-multiscale-query-enhancement-for-sa/) | 2026正式发表及方向/query重叠已被[U40旧表](U40_DIRECT_PRIOR_UPDATE_20261017.md)覆盖，不冒充本次新发现。 |

IEEE浏览工具曾遇机器人拦截；本轮普通无代理HTTP读取官方PDF返回200，并在内存抽取22页，8,052,579字节，SHA256 `5124d85353314b35a245d79047fbe2a34947fe5cfd22212be96c107fbcb73731`。官方原文读取、必要版面核对完成，全文未落盘或转存；本文仅保存自写归纳及链接。当前配方、预算和[首次端点AP前规则](U57_PROSPECTIVE_UTILITY_AND_REPEAT_POLICY_20261025.md)不变，没有新增外部复现或AP结果。

2026-10-25特征来源另核：[当前教师身份说明](U57_TEACHER_IDENTITY_AND_DATA_EXPOSURE_SCOPE_20261025.md)。上述条件是最终实例mask embedding，不据接口函数名将其写成原始query token。
