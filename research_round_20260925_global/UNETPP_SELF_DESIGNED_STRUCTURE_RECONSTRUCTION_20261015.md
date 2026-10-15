# 自主UNet++结构还原与可保留的研究主线（2026-10-15）

按用户本轮明确说明，这些UNet++结构与项目设计记为本项目自主设计。应肯定自主的结构选择、连接路径和整合工作；UNet++、CBAM、SimAM、Dynamic Snake Conv、BatchFormer、BlurPool等基础先例仍须如实归属，不能因此称为首次提出。本轮只静态复核实际forward，无新代码实现、训练、权重加载或联网查新，旧报告不改。

标准嵌套连接可写为 X[i,j] = H[i,j](Concat(X[i,0],…,X[i,j−1], Up(X[i+1,j−1])))。所读多数变体保留这一UNet++拓扑；自主差异主要在“哪些支路、哪些分辨率、哪些传输边上进行何种运算”，不应把原本稠密连接另起名称当新拓扑。

## 按实际前向还原

| 文件 / forward行号 | 编码与融合 | 解码与连接 | 原生UNet++差异、启用边界 |
|---|---|---|---|
| archive/old_code/Unet++/model1.py / 99-143 | Five ordinary convolution scales 64/128/256/512/1024; MaxPool2d. | Dense nested UNet++ concatenation; separate transposed convolutions per up edge; optional four shallow-output heads. | Local reference implementation of established UNet++ structure, not a new connectivity invention. Ordinary convolution and learned upsampling. No evidence of directional/attention branch in this forward. |
| archive/old_code/root_legacy_models/model_new2.py / 120-164 | Five ordinary convolution scales with MaxPool2d. | Nested dense concatenation and transposed convolution, optional deep supervision. | SqueezeAttentionBlock definition exists in file, but this forward does not call it. Ordinary convolution route. Defined attention class must not be counted as an enabled network contribution. |
| archive/old_code/root_legacy_models/model1.py / 177-243 | Standard five-scale convolutions; quarter and sixteenth-scale half-batch Transformer paths. | Dense nested concatenation; transposed convolution; final CoordConv. | Cross-sample processing at two scales and coordinate prediction, while retaining the UNet++ graph. encoder_x2 and encoder_x4 are called. DSConv x/y are allocated at 138-139 but not called in this forward. |
| archive/old_code/root_legacy_models/model_old.py / 199-276 | Convolution + CBAM at each scale; half-batch Transformers at quarter and sixteenth scale. | Nested decoder; CBAM after intermediate decoder convolutions; conventional final mask head. | Attention affects continuing encoder representations and intermediate dense-decoder representations, not solely the cross-scale transfer branch. encoder_x2/x4 and CBAM. No new topology established by attention insertion alone. |
| archive/old_code/root_legacy_models/model_new1.py / 315-408 | Ordinary convolution encoder; x2 half-batch Transformer. A separate up_x_i_0=CBAM(x_i_0) is created at each relevant scale. | Attention-filtered up_x features are used only for cross-scale transfer; the original x_i_j remains in same-scale dense concatenation. The same CBAM module is reused across columns at a scale. | Explicit separation of cross-scale semantic transport from same-scale/local detail memory; this is the most distinctive existing routing choice in the inspected custom variants. Selective up_x transfer, scale-shared CBAM, encoder_x2. encoder_x3 is allocated at 313 but not called in this forward. |
| archive/old_code/root_legacy_models/model_new3.py / 387-473 | Convolution + CBAM/SimAM; quarter-scale ordinary/attention plus x/y Snake convolution concatenated and fused. | Nested decoder; intermediate features are attention filtered; ordinary final mask head. | Parallel semantic and directional representations meet at quarter scale; decoder still uses original nested dense graph. x/y DSConv, CBAM and SimAM. encoder_x2/x4 are allocated at 380-381, but their forward calls are commented and not executed. |
| model_old_zuobiao.py / 288-361 | Stage0 convolution+CBAM, then five-scale attention encoder. Quarter-scale CBAM and separate convolution/Snake x/y merge. At H/8 and H/16, half-batch BatchFormer output is combined with x/y Snake features. | Dense nested UNet++; middle/lower decoder nodes use CBAM_block_up; highest-resolution nested nodes use ordinary convolution; final attention CoordConv. | Multi-scale contextual/directional fusion rather than only quarter-scale fusion. ds_stage_2/3/4 x/y, batch_for_3/4; old BatchFormer hardcodes 50x50 and 25x25 spatial embeddings. Shape conventions require matching input resolution; no source-only claim of batch-independent inference. |
| model_full.py / 392-472 | Five scales. H/4 has parallel CBAM plus optional encoder_x2 path, and separate ordinary stage_2 feeding x/y DSConv. Three tensors concatenate into con2. Deeper levels use CBAM blocks. | Dense nested graph, distinct learned up edges; CBAM_block_up at six lower-resolution nodes; ordinary Conv blocks at high-resolution x0 columns; final attention CoordConv; deep supervision optional. | Concentrates contextual/directional fusion at H/4, mixes attention placement by decoder depth, preserves highest-resolution dense detail path. ds_stage_2 x/y; encoder_x2 only when use_x2_transformer=True; default True is a constructor default, not proof for every checkpoint. SimAM is active in encoder CBAM blocks. include_legacy_x4_transformer may allocate encoder_x4 but forward never calls it. ChannelAttention.forward:65 computes BlurPool into variable max, which is not used in returned attention. Main downsample remains MaxPool; it is not an effective antialiased representation path. |
| resnetplus.py / 134-163 | BasicBlock/BottleNeck residual stages with configurable layer counts, MaxPool; initial VGG block. | UNet++ nested dense concatenation with bilinear align_corners=True upsampling; optional deep supervision. | Residual encoder plus nested reconstruction; useful carrier/control, with no directional prior or novel nested topology established. Residual stages and bilinear upsampling. No matching training/weight lineage established in the static index. |
| model_unetpp_clean.py / 78-116 | Five scales starting base_channels=32 by default; GroupNorm default; MaxPool. | Dense nested graph with bilinear align_corners=False and 1x1 mask heads. | Provides an explicit simple reference; differs from full in width, normalization, upsampling, attention and head. Plain nested graph; optional deep supervision. No attention, Snake, coordinate head or Transformer. |
| model_sfic_frequency.py / model_sfic_boundary.py / model_sfic_frequency_boundary.py / frequency 84-113; boundary 32-38; combined 39-45 | Frequency variant multiplies shallow x0_0 by 1+scale*sigmoid(project(gray,local-residual,Sobel)). Other encoder scales remain clean baseline. | Inherited clean dense decoder. Boundary head shares final x0_4 with mask head; combined variant has both modifications. | Shallow image-conditioned modulation plus shared-representation boundary supervision; not a changed connection topology. Gate in frequency path, boundary auxiliary head when corresponding class selected. Positive frequency scale only amplifies feature magnitude; physical scattering interpretation is unverified. |
| model_sampling_boundary.py / 117-144 | Each down step performs binomial depthwise BlurPool followed by ConvBlock. | Every scale-transition upsampling uses bilinear plus gated depthwise/pointwise detail residual; modules up1..up4 are shared by columns at each scale. Dense concatenation and boundary/mask heads remain. | Consistent redesign of actual down/up operators across the nested reconstruction graph, not just a named but disconnected BlurPool. True anti-aliased downsampling and detail residual upsampling in the returned computation path. Only bounded 5x64 training history confirmed, not full-convergence or superiority. |
| model_sbe_stps.py / STPS 252-289; BEGA 339-392 | STPS substitutes 2x2 energy-weighted aggregation+max context for all four downsampling stages. BEGA variant separately keeps ordinary MaxPool encoder. | STPS uses plain clean nested concatenation; BEGA gates each same-scale skip using skip/up/energy inputs before concatenation. | Separates sampling preservation from skip selection in different existing variants; they are not a single already-validated fused model. Energy sampling in SBEStps; energy-conditioned skip transport in SBEBegaFusion. No connected numerical training chain established by results audit; uniform/feature controls exist as designs. |

## 最值得保留的三条自主结构候选

### 1. 保留局部细节记忆的选择性跨尺度语义传输

**已有设计：** model_new1 keeps raw x_i_j in dense same-scale memory while passing CBAM-filtered up_x_i_j across scale; scale-shared gates avoid one new head per dense edge.

**可检验假设：** If coarse/background contamination actually propagates through cross-scale edges, select transport without rewriting local detail representation.

**可能形成的技术差异：** A task-conditioned transport rule tied to measured contamination/instance boundary may distinguish this integrated routing from indiscriminate attention placement; exact prior novelty still unverified.

**必须否定的替代解释：** Ordinary CBAM placement or shared regularization explains all gain.；Extra convolution, parameter count, BatchFormer or normalization rather than path separation explains gain.

**最少强控制：** Same carrier, same gate and weights budget: gate all paths versus only cross-scale transport versus identity.；Shared versus unshared gate with cost controlled; remove Transformer only through a separately locked comparison, not silent alteration.；Fixed instance candidates/scores and split-level errors; native instance adapter required before mask AP claims.

**现有证据：** Existing self-designed routing confirmed statically; standalone fair long-train effect/weight lineage not established.

### 2. 中分辨率语义—方向互补融合与分层解码

**已有设计：** full merges H/4 contextual/optional BatchFormer features with separate x/y Snake paths, then uses attention below and plain dense high-resolution reconstruction above; old_zuobiao tests a different multiscale placement.

**可检验假设：** If thin hull continuity and scene/context discrimination need complementary features at different resolutions, fuse them before nested reconstruction while protecting high-resolution detail.

**可能形成的技术差异：** Placement, complementary branch inputs and selective decoder routing form an integrated architecture hypothesis; Snake/CBAM/SimAM/BatchFormer themselves are known components.

**必须否定的替代解释：** Larger width/extra branches/DSConv alone explains benefit.；Batch composition or coordinate position prior explains apparent gain.；Ordinary deformable/directional convolution with same cost is equally effective.

**最少强控制：** Match width, normalization, upsampling and head before comparing full with clean.；Two ordinary branches versus semantic+x/y directional; H/4 only versus additional deeper placements under locked budget.；Coordinate head on/off, attention placement control, sample/batch-inference audit when BatchFormer enabled.

**现有证据：** Existing integration confirmed; some legacy weights exist in overall inventory but source-to-trained-endpoint mapping for these exact forwards remains incomplete.

### 3. 与嵌套解码相协调的采样保持与细节恢复

**已有设计：** SamplingBoundary consistently changes downsampling to binomial blur+refine and every up edge to detail residual, with scale-shared reconstruction operators; STPS/BEGA are separate sampling/transport variants.

**可检验假设：** If measured feature-grid aliasing/resize errors erase small hulls, align analysis and reconstruction operators across all nested edges.

**可能形成的技术差异：** Can become a coherent task-dependent sampling/reconstruction architecture if tested against strong generic resampling and actual sampling-error evidence; current source does not prove a new anti-alias primitive.

**必须否定的替代解释：** Ordinary antialiasing, extra detail-convolution capacity or boundary supervision explains gain.；Lower recall at narrow structures offsets any smooth-boundary benefit.

**最少强控制：** Same budget and boundary loss: baseline, blur only, detail residual only, both.；Uniform/generic-feature versus energy sampling when STPS is the tested variant.；Boundary geometry, pixel FP/FN, small-object mask AP and full cost; no reuse of old AP50/75 as COCO AP.

**现有证据：** SamplingBoundary has bounded 5 epochs x64 batches per dataset; STPS/BEGA numerical chain unconfirmed. No full-convergence success established.

## 训练程度与不能跨越的结论

| 家族 | 已读记录支持什么 | 不能据此推出什么 |
|---|---|---|
| SBRF R118 + generic | Results reviewer confirmed 50 epochs for HRSID/SSDD; 91,050 /23,200 batches respectively. | This is completed budget evidence, not proof of improvement or novelty. |
| HDSC R134 + generic and SamplingBoundary R142 | 5 epochs x64 batches =320 updates per inspected dataset/profile. | Bounded training only, not overall invalidity or full convergence. |
| CoastGate ordinary/targeted/metadata-CX | All inspected train JSONs have 1x4=4 batches. | Smoke-level evidence only. |
| STPS / BEGA | No connected numerical train chain found in 84 train JSONs and full weight inventory by results reviewer. | Design/smoke execution unconfirmed; do not call long trained. |
| full / old_zuobiao / archive / NestedUResnet | Current review confirms source and actual static forward; no exact complete source-init-budget-checkpoint endpoint chain established here. | Unknown is not untrained. Merge external run evidence before selecting or judging performance. |

这些结构通常输出一张二值语义图。面向当前公共SAR船舶实例分割目标，需要在保持实际实例身份的载体/适配上检验，不能以旧连通域AP50/AP75代替COCO mask AP，也不能把边界辅助监督、归一化、通道宽度和初始化差异归给结构机制。clean默认base32+GroupNorm，而full为base64+BatchNorm/不同上采样与头，直接比较不足以定位原因。

特别需要修正结构叙述：full的ChannelAttention.forward第65行计算BlurPool后结果未参与返回，所以有计算开销但不能据此说主表征经过抗混叠；old_zuobiao确有batch_former有意处理半批样本，所以不能仅凭轴序断言bug，但必须讲清与空间注意力的区别及推理批次依赖。model_new1跨尺度up_x与原x记忆分离、full中分辨率方向/语义互补、SamplingBoundary真实下采样/重建联动，是比“加了几个模块”更值得重建和检验的设计主线。

目前三条都只是有源码依据的候选，尚未完成直接先例查新、公平效用和机制证据；本轮不宣布创新成立，也不更改正在运行的N20固定端点。


发布说明：本文件是自写审阅与研究记录。逐文件私有JSON索引留存本地；未发布历史权重、原始数据、全量标注、运维配置或第三方论文全文。相关记录见[扩展总评](UNETPP_EXPANDED_MODEL_ASSET_REVIEW_AND_NEXT_DESIGN_20261015.md)。


## 后续来源与容器元数据补查（2026-10-15）

本文件保留此前静态审阅/只做stat阶段的范围。后续Root已仅打开111、222和两份ALL_NEW的ZIP目录与有界pickle元数据，没有加载模型/张量或哈希整份权重。新增writer/reader声明、参数名样集合及兼容性限制见[后续证据](UNETPP_LEGACY_PROVENANCE_AND_CONTAINER_EVIDENCE_20261015.md)。该补查仍不认证完整原训练、唯一forward或当前AP；不能用元数据相等合并或删除权重。另明确区分旧掩膜IoU阈值通过指标、连通域代理AP与COCO AP。
