# U40直接先例更新：O2Former与DACF-net

检索日2026-10-17；仅文献核查，未读取运行载荷或改变实验。对照公开主计划与U39续研说明；使用research-lit/novelty-check流程，跨模型复核工具不可用，本笔记不认证新颖性。

## O2Former

[arXiv记录](https://arxiv.org/abs/2506.11913)及[v1全文HTML](https://arxiv.org/html/2506.11913v1)已读方法和实验。[作者所属大学记录](https://pure.kfupm.edu.sa/en/publications/o2-former-direction-aware-and-multiscale-query-enhancement-for-sa/)确认已发表于JSTARS 19:11633–11645，2026，DOI10.1109/JSTARS.2026.3677088；正式版正文/与v1差异未核，不能继续只称未发表预印本。本轮未核实作者代码链接，不等于证明代码不存在。

它在Mask2Former上用OQG融合多尺度特征生成query；OAEM以旋转采样、角度卷积分支、极坐标编码和动态融合建模方向，已直接覆盖“方向＋多尺度query”。与本法不同：其学习查询生成及Transformer解码；本法固定教师query/分数，以完整改造UNet++共享图像特征预测原生logit残差。该实现差异尚非创新证明。

v1称PSeg-SSDD为928/232、HRSID为3642/1962训练/测试；ImageNet→HRSID→SSDD初始化，500轮。表I的mask AP为IoU=.50:.05:.95；表II/III分别报离岸/近岸：SSDD 71.9/63.2，HRSID 71.7/43.1（%），不是全测试集AP，不能平均替代。其“官方划分”与后文随机采样表述、尺度64²分界均需配置核验，不能直接当本项目协议。

## DACF-net

[出版社](https://www.tandfonline.com/doi/full/10.1080/17538947.2026.2696657)记录IJDE 19(2)，2026-09-08上线。本轮直连失败，但出版社搜索索引读到摘要及§3.3、4、5正文片段；未取得完整公式、表格/补充材料，不能写“全文已完整审阅”。

MSFFM多分支大核融合；CLFIM用高层语义生成增益，Hadamard调制低/中层并保留残差；DCPBlock结合条带空间与全局通道注意，接GA-RPN/HTC。这是方向、跨层传输和残差融合的直接先例；其proposal级联不同于本法冻结query上的全域掩膜残差，但不支持宣称首次融合方向与语义。

索引正文：HRSID 13:7、PSeg-SSDD 8:2；800/512输入，AdamW、12轮。base分割AP为60.2/64.2；‡为63.3/65.7（HRSID/PSeg），‡另换EdgeNeXt-B并用Soft-NMS，不能全归因模块。作者称COCO分割AP，精确阈值/面积表S2、maxDets及初始化未核。[作者仓库](https://github.com/yang199809/DACF-Net)当前仅.gitignore/LICENSE/README，声明实现待发布，不能称可复现源码已齐。

## 当前主张与最少控制

先完成既定full/plain同输入、实例头、原S0、预算对照；这只检验整套结构效用。若正收益，再先以普通/静态条带分支替换双DSConv、尽量接近容量与计算，区分自适应几何与多分支规模；复合块以固定增益替代z+SimAM(z)，区分空间选择与幅值放大。不得将N28路由结论混归完整full。以上是条件建议，不授权执行、不新增模块。异划分、预训练和预算数字不与本项目内部DEV直接判胜负；尚无TGRS级或核心创新成立结论。


后续[固定query精修与ReSAM点提示协议补核](U40_FIXED_QUERY_REFINEMENT_AND_PROMPT_PROTOCOL_PRIORS_20261017.md)区分CVPR正式索引、v1和作者固定提交；IoU/F1点提示结果不直接作为自动COCO AP基线。CascadePSP为原图加粗掩膜精修的通用直接先例；不改变当前50轮训练。
