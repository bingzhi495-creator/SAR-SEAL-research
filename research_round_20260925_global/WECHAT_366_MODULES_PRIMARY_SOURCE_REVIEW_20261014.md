# 微信“366模块”文章及原始来源核查

2026-10-14。证据：原微信正文/目录截图与可核验原论文、作者代码。未取得完整366项清单，未运行资料包、未联系二维码提供方。

[原文章《366个即插即用的涨点神器分享！》](https://mp.weixin.qq.com/s/BKqVjKESr9aB-LHd4-7Vjg)是模块资料包宣传。正文列模块类别和领取方式，没有统一SAR实验、完整公开模块代码或可比基准。“即用”和数量不是有效性或创新性的证据。截图两项可核验线索如下。

| 样例 | 原始来源及任务 | 对当前项目的适用判断 |
|---|---|---|
| UCMNet | [CVPR2026官方论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_UCMNet_Uncertainty-Aware_Context_Memory_Network_for_Under-Display_Camera_Image_Restoration_CVPR_2026_paper.html)，[arXiv方法与表格](https://arxiv.org/html/2604.00381v1)，[作者代码](https://github.com/kdhRick2222/UCMNet/tree/0cb74dc0602327d6018e0d0f08ed189af8667a29)。屏下相机图像复原。 | 可取learned memory/context检索作最小点头候选，但当前输入仅无序F采样点，不能原样使用整图横纵注意力。 |
| VideoFusion | [CVPR2026官方论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_VideoFusion_A_Spatio-Temporal_Collaborative_Network_for_Multi-modal_Video_Fusion_CVPR_2026_paper.html)，[原论文](https://arxiv.org/html/2503.23359v3)，[作者代码](https://github.com/Linfeng-Tang/VideoFusion/tree/02b3431cedc6f4a809dab5441ed22a71487873e9)。配准红外/可见视频融合。 | 当前单幅SAR实例输入没有配准双模态视频；不列当前试训，不制造第二模态或时序证据。 |

UCMNet用不确定性查询与成对可学习memory/context banks检索上下文，再做栅格注意力；其高频不确定性监督来自图像复原，不能直接套用分割标签或把小absL当校准误差。表1报告3.2M参数、POLED/TOLED PSNR33.81/38.37，是复原结果，不能换算为mask AP。[原论文](https://arxiv.org/html/2604.00381v1)

N14初试只借鉴检索形式，以F+c驱动8-token条件纠错残差，未使用原不确定性损失、频率模块或横纵注意力；不是完整作者复现。先做等预算BASE/MEMORY两分支，若有重复收益再补容量与路由控制。不能把“模块有效”自动归因于不确定性或SAR物理机制，后续技术差异仍须对照直接先例。[预锁协议](N14_CONTEXT_MEMORY_SINGLE_MODULE_EXPLORATION_PROTOCOL_20261014.md)
