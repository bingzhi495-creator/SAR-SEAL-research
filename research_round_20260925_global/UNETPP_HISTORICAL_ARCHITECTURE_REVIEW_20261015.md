# D:/SAR/UNet++ 历史架构静态借鉴审阅（2026-10-15）

本轮只读未绑定源码、配置和 AST；未执行项目代码、未导入 Torch、未打开权重或数据、未改动 D 项目，也未改动当前 N20。以下“配置预算”不是已完成训练的证明；实际历史结果由独立证据核验补充。

## 先确认任务与评价边界

标准 HRSIDDataset 将全部实例多边形合并为一张语义掩膜（dataset1.py:79–91），训练没有保留每条实例 ID。旧 eval_instance.py:46–60 用阈值连通域和平均概率生成候选；319–352 只评价 AP50/AP75，并明确写出 “not COCO API”。这些历史 AP 不能直接作为当前 COCO mask AP[.5:.95] 强对照。

model_mask2former.py:104–122 对所有 query 做 log-mean-exp 得到单张二值图，因此是语义适配；这不等于完整原生 Mask2Former 实例复现。SegFormer/DeepLabV3+ 包装也服务二值语义接口。maskrcnn.py:29–105 的 torchvision 示例才保留逐实例 mask，但预训练/数据路径/实际训练状态仍需独立核验。

## 八组优先借鉴对象

| 家族 | 源码与配置证据 | 可借鉴的具体内容 | 主要反解释/限制 |
|---|---|---|---|
| Clean UNet++ / GroupNorm | model_unetpp_clean.py:35–116；configs/baseline_hrsid_clean_unetpp400.yaml:27–65 | Useful cheap dense-feature reference or generic geometry-control backbone; cannot replace a strong instance baseline solely on semantic Dice. | Any gain may be ordinary high-resolution decoder/deep supervision/capacity rather than a SAR-specific mechanism. |
| Auxiliary boundary + scattering residual gate | model_sfic_boundary.py:6–38；boundary_loss.py:19–59；model_sfic_frequency.py:8–87 | A minimal instance-conditioned boundary head is an appropriate competing explanation for a more complex refinement design. Gate descriptors may be a fixed diagnostic cue. | Extra boundary supervision/capacity or generic local contrast may fully explain gain. This frequency gate only amplifies features for positive scale; it is not an explicit clutter-suppression gate. |
| STPS / BEGA / anti-aliased sampling | model_sbe_stps.py:8–149；model_sbe_stps.py:152–195；model_sbe_stps.py:198–289 | Most concrete reusable architecture idea: test information loss at downsample/interpolation boundaries before choosing an instance feature sampler. A future feature-grid sampler could compare energy/learned/constant weights under identical carrier. | Max+average mixing, extra 1x1 capacity, or antialiasing may explain gain. Energy cues may favor speckle/harbor clutter. Backbone or FPN replacement changes end-to-end training and cannot be inserted into current frozen N20 endpoint. |
| Scattering-Boundary Reliability Field | model_sbrf.py:8–88；model_sbrf.py:117–170；sbrf_loss.py:85–162 | Instance-conditioned hard-negative modeling or a signed residual corrected by foreground/boundary support could be a focused new hypothesis if actual FP neighborhoods justify it. | Additional image cues, hard-negative supervision and logit calibration can each explain gain. Zero field initialization means i=b=c=0.5; default lambdas shift z by +0.875, so this is not identity initialization. |
| Hull geometry and instance-separation targets | model_hdsc.py:7–77；hdsc_loss.py:89–123；hdsc_instance_targets.py:120–158 | Strongest task-specific supervision candidate: query-conditioned separation/near-instance-negative targets while retaining true instance IDs. Useful direct control against a claimed complex neighbor-error mechanism. | Extra auxiliary supervision rather than inferred ship shape may account for improvement. A merged semantic map cannot encode arbitrary overlapping instances. Default zero calibration acts through shared representation training only. |
| Gamma vs Gaussian region-likelihood objective | sar_region_loss.py:13–34；sar_region_loss.py:73–121；sar_region_loss.py:124–179 | Could test a local ROI or per-instance display-intensity contrast prior as a low-parameter regularizer, with generic regional-energy controls. | Displayed JPG/PNG may have unknown clipping, tone mapping and log scaling, so display values are not established calibrated amplitude. Squaring them does not prove Gamma physical likelihood. Global two-region means also pool harbor/clutter/ships and can induce contrast bias. |
| Quality-aligned Deformable DETR head | detector_extensions/mmdet_quality_aligned/deformable_detr_quality_aligned_head_r255.py:23–48；detector_extensions/mmdet_quality_aligned/deformable_detr_quality_aligned_head_r255.py:190–233；detector_extensions/mmdet_quality_aligned/deformable_detr_quality_aligned_head_r255.py:261–300 | Mask-IoU quality estimator with fixed masks/candidates is a required strong control whenever proposed method could improve AP mainly via reranking. It is an adaptation, not source-exact mask model. | Ranking and candidate truncation can raise AP without repairing masks. Box quality is not mask quality; source IoU target is built from live bbox predictions without an explicit detach in this function, so training coupling needs deliberate treatment. |
| Legacy attention/DSConv/CoordConv + lightweight distillation | model_full.py:86–121；model_full.py:318–385；model_full.py:392–472 | Directional sampling could inspire a targeted thin-hull control. Depthwise multiscale student may be useful for a future cost/accuracy study after a strong teacher is confirmed. | Many simultaneous modules, much larger base width and different normalization prevent attributing gain. Transformer input is [B,C,H*W] but batch_first is not set, so default [S,N,E] interpretation attends across the first axis (original batch), not intended spatial tokens. KD softmax dim=0 normalizes over batch for standard [B,C,H,W] output; one-channel semantic KL must be redesigned, not copied. |

## 每组的机制、实际关联与最少强控制

### Clean UNet++ / GroupNorm

**实现：** Five encoder scales, nested dense skip concatenation, bilinear resize, 1-channel mask head; optional four-level deep supervision.

**关联：** train_semantic.py --config configs/baseline_hrsid_clean_unetpp400.yaml; static association only, command not executed.

**仅配置可见的预算：** HRSID config: 50 epochs, batch 2, seed 42, AdamW 3e-4. Configuration is not proof that training completed.

**本地先例边界：** Source explicitly calls this a compact UNet++ baseline; no originality implied.

**最少控制：** Same initialization, data version, supervision, pixels, steps and COCO evaluator; retain true instance candidates/IDs.

### Auxiliary boundary + scattering residual gate

**实现：** Boundary branch reads final x0_4 and adds BCE on morphology-derived boundary. Frequency gate concatenates grayscale, absolute local-mean residual and Sobel magnitude, projects 3 channels to feature channels, and multiplies shallow features by 1+0.5*sigmoid(gate). Combined variant exists.

**关联：** train_semantic.py with paired clean/boundary/frequency/frequency_boundary configs for HRSID and SSDD.

**仅配置可见的预算：** Inspected HRSID boundary and frequency configs request 50 epochs, batch 2, seed 42, AdamW 3e-4; actual run status not inspected here.

**本地先例边界：** Conventional auxiliary edge supervision plus local image-statistic feature gating; source labels are not novelty evidence.

**最少控制：** Plain auxiliary boundary head; same head with raw image/learned generic feature descriptors; constant gate and parameter-comparable controls. Do not call Sobel or average-pool residual a measured physical scattering quantity.

### STPS / BEGA / anti-aliased sampling

**实现：** STPS combines max-pooled context with softmax-weighted 2x2 feature aggregation. Scores use a learned grayscale/contrast/Sobel energy head plus a feature score, merged by 1x1 convolution. Uniform and feature-only variants preserve the two-stream structure. BEGA gates skip tensors with skip/up/energy concatenation. Separate SamplingBoundary model uses fixed binomial BlurPool and gated depthwise detail residual after bilinear upsampling.

**关联：** train_semantic.py; configs pair STPS with generic_uniform/generic_feature and separate BEGA variant.

**仅配置可见的预算：** Inspected STPS profile is 1 epoch x 1 batch, CPU evaluation. SamplingBoundary R142 profile requests 5 epochs capped at 64 batches. Neither is full-convergence proof.

**本地先例边界：** Source identifies generic adaptive sampling controls and depthwise binomial antialiasing. External first-use novelty was not checked in this static pass.

**最少控制：** Max only; max+uniform; max+feature learned; max+image-energy with equal merge capacity; fixed BlurPool. Lock geometric coordinates and grid cell units, not only rendered-pixel offsets.

### Scattering-Boundary Reliability Field

**实现：** 1x1 projection of final decoder feature plus grayscale/normalized contrast/Sobel predicts interior, boundary, clutter fields. Corrected mask logit is z + lambda_i*i + lambda_b*b - lambda_c*(1-i)*(1-b)*c. Interior/boundary are mask-derived targets; clutter target selects top-5% contrast/gradient hard background outside a dilated foreground exclusion region.

**关联：** train_semantic.py; R118 full_screen and generic_control config pairs are present.

**仅配置可见的预算：** Inspected R118 HRSID full_screen requests 50 epochs, batch 2, seed 42. Source/config labels such as positive bounded controls are historical assertions, not independently checked outcomes here.

**本地先例边界：** Source contains an explicit generic field control without SAR descriptors. Local grayscale/contrast/edge descriptors overlap the frequency and SBE modules above.

**最少控制：** Constant +0.875 bias; zero-centered identity residual; same auxiliary labels/heads without image descriptors; matched generic contrast features; fixed candidate and score crossing to separate geometry and ranking effects.

### Hull geometry and instance-separation targets

**实现：** Final feature map predicts core, boundary, near-hull negative and separation-moat fields through a zero-initialized 1x1 projection. HDSCSmokeLoss falls back to erosion/boundary/dilation of merged semantic mask unless metadata supplies per-instance fields. hdsc_instance_targets builds each annotation mask and delegates true per-instance target computation. Default mask-logit calibration weight is zero.

**关联：** train_semantic.py with hdsc_dataset_adapter.HRSIDHDSCInstanceDataset; R141 instance-aware configs differ from earlier semantic smoke configs.

**仅配置可见的预算：** Inspected instance R141 requests 5 epochs, 64 batches cap, separation loss weight 0.1, mask calibration 0.0. Earlier default separation weight is zero.

**本地先例边界：** Source explicitly reserves generic geometry control; most fallback labels are ordinary mask morphology. No external novelty claim established.

**最少控制：** Same four heads with generic geometry; boundary-only; separation-only with authentic instance metadata; identical supervision count. Verify sign/definition of moat before enabling nonzero mask-logit calibration, since source currently adds moat support.

### Gamma vs Gaussian region-likelihood objective

**实现：** Converts mean display-channel value to squared intensity by default; estimates soft foreground/background per-image means; optionally detaches means; adds mean p*(I/mu1+log(mu1))+(1-p)*(I/mu0+log(mu0)) to supervised mixed loss. Gaussian counterpart uses squared deviations.

**关联：** train_semantic.compute_model_loss passes images to loss when uses_input_image; Gamma/Gaussian clean configs and boundary+region variants exist.

**仅配置可见的预算：** Inspected R100 profile requests 50 epochs, batch 2, seed 42, weight 0.1, square, detach_means true. Actual Gamma outcomes are delegated to independent run-evidence review.

**本地先例边界：** Module explicitly derives the term as SAR Gamma analogue of Chan-Vese and compares a Gaussian control; that local derivation is not proof of distribution fit.

**最少控制：** Gamma vs Gaussian and simple intensity/contrast control, identity vs square only on TRAIN/DEV under prelocked selection, loss scale control; measured histogram fit/metadata before any physical claim. No new sweep or training was executed in this review.

### Quality-aligned Deformable DETR head

**实现：** Adds predicted localization quality to Deformable DETR after Hungarian target construction. Positive target is matched predicted-box/GT-box IoU; unmatched target zero with weight 0.25. Inference top-k ranks class_probability^a * predicted_quality^b; it does not rank using observed GT IoU.

**关联：** MMDetection custom_imports loads QualityAlignedDeformableDETRHeadR255; adapted SSDD config requests class/quality powers both 1.

**仅配置可见的预算：** No actual detector environment, trained weights, AP or execution inspected in this pass.

**本地先例边界：** Deformable DETR extension and generic IoU-quality calibration are explicit in source; not a first-use novelty claim.

**最少控制：** Freeze candidate masks; score-only vs geometry-only vs both; compare predicted mask IoU with generic quality head; keep losses/init/budget consistent. Inspect intended target detach semantics before implementation.

### Legacy attention/DSConv/CoordConv + lightweight distillation

**实现：** Legacy full UNet++ combines CBAM, SimAM, x/y Dynamic Snake Convolution at quarter scale, TransformerEncoderLayer on flattened feature spatial axis, transposed-convolution nested decoder and attention CoordConv output. KD student uses depthwise ShuffleUnit/channel shuffle and multiscale decoding; teacher+student trained with hard segmentation loss and KL soft loss.

**关联：** Legacy trainer.py/trainer_coco.py and KD/train_KD.py are separate entry families; no unified fair schedule inferred. Do not run them as-is.

**仅配置可见的预算：** Historical checkpoint name 222.pth is referenced by KD source but was neither opened nor attributed to a completed verified run here.

**本地先例边界：** Source explicitly names Dynamic Snake Convolution; CBAM/SimAM/CoordConv and channel shuffle are known family labels, not newly verified papers in this static pass.

**最少控制：** Isolate directional operator on same carrier against ordinary/deformable depthwise convolution; identity-safe residual. For KD, freeze/detach teacher, correct Bernoulli or two-class distribution axis, check student channel contracts, then compare measured latency and AP with equal student training. These are proposed repairs, not executed fixes.

## 三个必须在复用前处理的实现问题

1. SBRF 零初始化并不保持原预测：三个 sigmoid 均为 0.5，默认公式给 base logit 增加 0.5+0.5−0.125=0.875。先用常数偏置与恒等初始化对照，避免将阈值移动包装为复杂可靠场机制。

2. model_full.py:378–408 没有设置 batch_first，却输入 [B,C,H×W]。按 TransformerEncoderLayer 默认 [S,N,E]，第一轴被当序列轴；这是批轴语义风险，应先修正并复核旧结果的适用范围。不能直接将该实现描述为标准空间自注意力。

3. KD/KD_loss.py:7–10 用 softmax(dim=0)，对通常的 [B,C,H,W] 输出沿 batch 归一化；KD/train_KD.py:50 未显式 detach/no_grad 教师。单 logit 二值分割应设计 Bernoulli 或两类分布蒸馏并冻结教师；学生分支通道也须检查。这里只指出静态问题，未执行修改。

## 建议如何服务当前研究

当前 N20 固定端点保持原样。若需要新假设，优先围绕真实实例错误，将“逐实例分隔/近邻负样本”作为单条假设；将普通边界头、同容量通用邻域聚合、固定掩膜质量排序作为强控制。STPS/抗混叠属于需要重训载体的另一条路线，只有真实采样损失证据支持时再单独锁定实验。Gamma 项先按显示强度区域正则处理，不能未经显示链与分布验证称为成立的 SAR 物理似然。

本轮未查外部论文新颖性、未验证历史训练完成或指标；本地名称/注释不自动成为论文证据。结构化 JSON 保留 EXTRACTED 源码关联与 INFERRED 迁移假设，便于 Root 整合和后续 graphify，推断边不代表已实现。

## HDSC 几何目标补充核实

hdsc_r132_target_diagnostics.py:92–168 为逐实例归一化欧氏距离核心（阈值0.45）、逐实例边界，以及“两个以上实例膨胀区交叠且排除前景”的 separation moat；近船负样本进一步排除 moat。由此确认 model_hdsc.py:57–63 非零校准把背景 moat 作为正支持加到前景 logit，存在静态符号不一致。所读 R141 配置权重为0.0，关闭这条校准支路，不能据此认定既有结果错误；迁移前需按背景定义修改或单独检验，不能照搬。


[总评与有条件复用方案](UNETPP_HISTORICAL_ASSETS_SYNTHESIS_AND_REUSE_PLAN_20261015.md)。源文件位置均相对于用户指定的历史工作区；本次未复制原项目代码、权重或数据。


## 后续扩展与自主结构解释（2026-10-15）

本文件保留首轮八组优先审阅的历史快照，不能代表全目录。用户明确说明UNet++结构为团队自主设计，整体结构与基础算子来源分开；新增完整目录、自主前向还原、权重/长训证据及直接先例见[后续总评](UNETPP_EXPANDED_MODEL_ASSET_REVIEW_AND_NEXT_DESIGN_20261015.md)。旧full与clean并非同一结构或公平配方；未定位旧权重完整来源不等于未训练。BatchFormer可能有意跨样本计算，轴风险不能直接判成编码错误。

候选“邻船作为负类”也不是新的监督定义：原生逐实例mask损失已包含本实例外的负类目标；不确定性点采样支持由配置决定的随机补点。真正的技术差异需落在信息、路由、采样或目标的具体改变，并与强常规替代比较。新候选仍未验证，不以本目录审阅宣称创新成立。
