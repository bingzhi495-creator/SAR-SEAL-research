# UNet++ 完整模型源码/配置索引（2026-10-15）

这是在首轮8组抽查基础上的扩展盘点，不将首轮抽查冒充全目录审阅。只静态读未绑定源码/配置和AST，未执行项目、import Torch、加载/哈希权重或数据、修改D项目或当前N20。

## 覆盖与证据等级

rg --files --hidden --no-ignore 枚举 3739 个指定扩展名文本路径；最终登记 1009 个源码/配置/部署示例候选，其中 AST 索引 423、配置文本索引 517、仅登记文件名未展开 69。根 model*.py 共 14 份，项目 configs 共 126 份；模型族/包装/骨干注册 104 条。完整文件、类名、行号、父类、forward范围、别名、配置关联、排除原因均在配套JSON。所有登记不等于独立创新数。

初次排除research_ops_runs会漏掉真实模型仓库，因此已修正：HGTFN、MSCK-Net、Mamba-YOLO、ultralytics、yolov12源码全部纳入目录；运维记录与源码按内容角色区分。环境/site-packages/数据/权重/cache不作为模型源码。第三方原子层不要求逐一机制解读，但顶层模型族逐项登记，未展开项明确标记。

## 新发现与可借鉴机制

| 家族 | 静态发现 | 适用边界/下一步 |
|---|---|---|
| NestedUResnet | Residual encoder + nested UNet++ decoder; a generic stronger backbone control, not SAR-specific by definition.（resnetplus.py:80-159） | Compare only after task/protocol alignment; no matching local YAML module reference found in this index. |
| CoastGateBoundaryUNetPlusPlus | Zero-centered sigmoid modulation explicitly subtracts its initial gate value, so initialization preserves base features. A useful identity-safe alternative to SBRF initial +0.875 shift.（model_sfic_coast_gate.py:35-68） | Requires authentic coast/hard-region target provenance; existing boundary-as-coast and real metadata labels must not be conflated. |
| LIE / guess drafts | Sigmoid input threshold, learnable-random versus fixed Gaussian filtering, local/neural branch fusion and directional x/y outputs. LIE Gaussian label is not an actual Gaussian parameter constraint.（LIE.py:6-66; guess.py:22-85） | Possible directional local descriptor control; no configured enclosing instance network established; channel contract assumes compatible in/out dimensions. |
| Legacy BatchFormer context | A class explicitly named batch_former intentionally reshapes [B,C,HW] and processes half the batch. Thus cross-sample attention may be intended, not automatically a coding bug.（model_old_zuobiao.py:169-185; model_full.py:378-408） | Earlier eight-group note must be interpreted as an axis/claim risk: if described as spatial attention, source does not implement that. Verify train/eval batch semantics and sample independence before reuse. |
| KD variants | new_Unet is conventional full U-Net; new_student_model_1 defines MultiScaleFusion/CBAM but the shown forward follows only a shallow subset and never calls MultiScaleFusion. Its ShuffleUnit is pointwise-depthwise-pointwise, without shuffle in that forward. shuffleUnet.DSConv means depthwise separable convolution, unlike root Dynamic Snake Conv.（KD/new_student_model_1.py:74-113,162-191; KD/new_Unet.py:86-115; KD/shuffleUnet.py:4-20） | Count executed forward graph rather than class names or all allocated parameters; retain separate model identity for each weight mapping. |
| Mamba-YOLO | SS2D uses cross_selective_scan; VSS/XSS with local/gated blocks and VisionClueMerge build multiscale model. T config ends Detect; independent yolo-mamba-seg config ends Segment.（research_ops_runs/sar_model_benchmark_comparison_20260608_1312/repos/Mamba-YOLO/ultralytics/nn/modules/mamba_yolo.py:6-170,305-416; cfg/models/mamba-yolo/Mamba-YOLO-T.yaml:8-38; yolo-mamba-seg.yaml:17-47） | Detector training does not validate segmentation config. Selective state-space or feature-merge component is a generic transferable mechanism requiring end-to-end fair control, not instant SAR novelty. |
| MSCK-Net | Multi-scale strip/local grouped-convolution branches with residual layer scale and ConvGLU; HGNetv2_MSCK + HybridEncoder + D-FINE are box detector in examined SAR config.（research_ops_runs/sar_model_benchmark_comparison_20260608_1312/repos/MSCK-Net/engine/backbone/StripBlock3.py:45-122; configs/IS/msck-n-hrsid.yml:14-48） | Strong anisotropic/directional operator control, but box AP must not enter mask AP table. |
| MSCK evaluator files are not models | last/checkpoint paths save self.state_dict; eval/latest.pth and eval/{000,050,...}.pth save coco_evaluator.coco_eval[bbox].eval. eval.pth similarly saves evaluator output.（research_ops_runs/sar_model_benchmark_comparison_20260608_1312/repos/MSCK-Net/engine/solver/det_solver.py:106-112,175-184,199） | Exclude these evaluator tensors from trained-model count, without loading them. |
| SCJE_FPN / CARAFE / ACmix | Vendored FPN has content-aware CARAFE first-level upsampling and ACmix-based SCJE_FPN refinement; comments identify CARAFE ICCV2019. These are direct sampling/refinement precedents already locally present.（assets/diffsarshipinst/source/detectron2/modeling/backbone/fpn.py:231-244,292-386,737-772,873+） | Useful strong generic upsampling/attention-convolution control; not a new primitive or proof the configured model trained successfully. |
| HGTFN | Hyperspectral spectral encoder, superpixel hypergraph, key-pixel selection and ViT; output class distribution per pixel, not SAR instances.（research_ops_runs/sar_model_benchmark_comparison_20260608_1312/repos/HGTFN/model.py:74-167） | Only borrow selective-compute hypothesis if relevant evidence exists; exclude from SAR instance performance comparison. |

## 全部顶层模型族/骨干/包装登记

| ID / 模型族 | 任务 | 来源/审阅状态 | 文件数 |
|---|---|---|---|
| local:model_unetpp_clean.py / model_unetpp_clean.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:model_sfic_frequency_boundary.py / model_sfic_frequency_boundary.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:model_sfic_frequency.py / model_sfic_frequency.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:model_sfic_coast_gate.py / model_sfic_coast_gate.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:model_sfic_boundary.py / model_sfic_boundary.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:model_segformer.py / model_segformer.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:model_sbrf.py / model_sbrf.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:model_sbe_stps.py / model_sbe_stps.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:model_sampling_boundary.py / model_sampling_boundary.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:model_old_zuobiao.py / model_old_zuobiao.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:model_mask2former.py / model_mask2former.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:model_hdsc.py / model_hdsc.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:model_full.py / model_full.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:model_deeplabv3plus.py / model_deeplabv3plus.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:KD/new_student_model_1.py / KD/new_student_model_1.py | requires family interpretation | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:KD/shuffleUnet.py / KD/shuffleUnet.py | requires family interpretation | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:KD/new_student_model.py / KD/new_student_model.py | requires family interpretation | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:KD/new_Unet.py / KD/new_Unet.py | requires family interpretation | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:KD/head_student.py / KD/head_student.py | requires family interpretation | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:KD/teacher_model.py / KD/teacher_model.py | requires family interpretation | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:KD/student_model.py / KD/student_model.py | requires family interpretation | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:archive/old_code/Unet++/model1.py / archive/old_code/Unet++/model1.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:archive/old_code/root_legacy_models/model_old.py / archive/old_code/root_legacy_models/model_old.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:archive/old_code/root_legacy_models/model_new3.py / archive/old_code/root_legacy_models/model_new3.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:archive/old_code/root_legacy_models/model_new2.py / archive/old_code/root_legacy_models/model_new2.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:archive/old_code/root_legacy_models/model_new1.py / archive/old_code/root_legacy_models/model_new1.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:archive/old_code/root_legacy_models/model_coord(1).py / archive/old_code/root_legacy_models/model_coord(1).py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:archive/old_code/root_legacy_models/model1.py / archive/old_code/root_legacy_models/model1.py | binary semantic segmentation / semantic component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:maskrcnn.py / maskrcnn.py | native instance segmentation adapter / example | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| component:resnetplus.py / NestedUResnet | semantic segmentation | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| component:LIE.py / LIE learned-filter draft | directional feature component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| component:guess.py / LIE fixed-Gaussian draft | directional feature component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| component:dsconv.py / Dynamic Snake Convolution | directional sampling component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| component:head.py / Attention CoordConv head | semantic prediction component | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| local:modern_instance_factory / Torchvision Mask R-CNN adapter | native instance segmentation | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 3 |
| local:quality_aligned_detr / Quality-aligned Deformable DETR | box detection | local/custom or adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| vendor:diffsarshipinst / DiffSARShipInst | native SAR instance segmentation | vendored SAR method / local modifications possible / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 4 |
| detectron2:rcnn / GeneralizedRCNN / Faster / Mask / Cascade / rotated RCNN / proposal-only | detection or instance segmentation by configured ROI heads | vendored framework / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 9 |
| detectron2:retinanet / RetinaNet | box detection | vendored framework / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| detectron2:fcos / FCOS | box detection | vendored framework / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| detectron2:semantic_seg / SemanticSegmentor / SemSegFPN | semantic segmentation | vendored framework / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| detectron2:panoptic_fpn / PanopticFPN | panoptic segmentation | vendored framework / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| detectron2:backbone:resnet / ResNet / deformable ResNet | feature backbone / neck | vendored framework with extensions / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| detectron2:backbone:resnet-6out / six-output ResNet variant | feature backbone / neck | vendored framework with extensions / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| detectron2:backbone:regnet / AnyNet / RegNet | feature backbone / neck | vendored framework with extensions / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| detectron2:backbone:swin / Swin Transformer | feature backbone / neck | vendored framework with extensions / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| detectron2:backbone:vit / ViT / SimpleFeaturePyramid | feature backbone / neck | vendored framework with extensions / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| detectron2:backbone:mvit / MViTv2 backbone | feature backbone / neck | vendored framework with extensions / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| detectron2:backbone:fpn / FPN / SCJE_FPN / CARAFE / ACmix | feature backbone / neck | vendored framework with extensions / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| detectron2:project:PointRend / PointRend | instance and semantic point refinement | vendored project / AST_CONFIG_INDEXED_MECHANISM_NOT_FULLY_EXPANDED | 20 |
| detectron2:project:PointSup / PointSup | point-supervised instance segmentation | vendored project / AST_CONFIG_INDEXED_MECHANISM_NOT_FULLY_EXPANDED | 10 |
| detectron2:project:TensorMask / TensorMask | dense instance segmentation | vendored project / AST_CONFIG_INDEXED_MECHANISM_NOT_FULLY_EXPANDED | 10 |
| detectron2:project:DeepLab / DeepLab | semantic segmentation | vendored project / AST_CONFIG_INDEXED_MECHANISM_NOT_FULLY_EXPANDED | 11 |
| detectron2:project:Panoptic-DeepLab / Panoptic-DeepLab | panoptic segmentation | vendored project / AST_CONFIG_INDEXED_MECHANISM_NOT_FULLY_EXPANDED | 10 |
| detectron2:project:DensePose / DensePose | dense human pose / task mismatch for SAR instance comparison | vendored project / AST_CONFIG_INDEXED_MECHANISM_NOT_FULLY_EXPANDED | 138 |
| detectron2:project:ViTDet / ViTDet | ViT/Swin/MViT detection and instance-segmentation configurations | vendored project / AST_CONFIG_INDEXED_MECHANISM_NOT_FULLY_EXPANDED | 23 |
| detectron2:project:MViTv2 / MViTv2 | multi-scale vision transformer detection/segmentation configurations | vendored project / AST_CONFIG_INDEXED_MECHANISM_NOT_FULLY_EXPANDED | 9 |
| detectron2:project:TridentNet / TridentNet | multi-branch scale-aware box detection | vendored project / AST_CONFIG_INDEXED_MECHANISM_NOT_FULLY_EXPANDED | 11 |
| detectron2:project:Rethinking-BatchNorm / Rethinking-BatchNorm | normalization experiment variants; not one new SAR model | vendored project / AST_CONFIG_INDEXED_MECHANISM_NOT_FULLY_EXPANDED | 7 |
| repo:Mamba-YOLO:mamba-yolo / Mamba-YOLO / mamba-yolo | box detection (unless config head explicitly indicates otherwise); instance segmentation | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 4 |
| repo:Mamba-YOLO:rt-detr / Mamba-YOLO / rt-detr | box detection (unless config head explicitly indicates otherwise) | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 4 |
| repo:Mamba-YOLO:v3 / Mamba-YOLO / v3 | box detection (unless config head explicitly indicates otherwise) | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 3 |
| repo:Mamba-YOLO:v5 / Mamba-YOLO / v5 | box detection (unless config head explicitly indicates otherwise) | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 2 |
| repo:Mamba-YOLO:v6 / Mamba-YOLO / v6 | box detection (unless config head explicitly indicates otherwise) | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 1 |
| repo:Mamba-YOLO:v8 / Mamba-YOLO / v8 | box detection (unless config head explicitly indicates otherwise); image classification; instance segmentation; keypoint / pose estimation; oriented box detection | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 17 |
| repo:Mamba-YOLO:v9 / Mamba-YOLO / v9 | box detection (unless config head explicitly indicates otherwise); instance segmentation | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 4 |
| repo:Mamba-YOLO:wrapper:sam / SAM / TinySAM / SAM2 or SAM3 according to local build files | prompt segmentation | vendored model wrapper / TOP_LEVEL_AST_INDEXED_ATOMIC_COMPONENTS_PARTLY_NOT_EXPANDED | 8 |
| repo:Mamba-YOLO:wrapper:fastsam / FastSAM | prompt segmentation | vendored model wrapper / TOP_LEVEL_AST_INDEXED_ATOMIC_COMPONENTS_PARTLY_NOT_EXPANDED | 1 |
| repo:Mamba-YOLO:wrapper:nas / YOLO-NAS | configured task wrapper | vendored model wrapper / TOP_LEVEL_AST_INDEXED_ATOMIC_COMPONENTS_PARTLY_NOT_EXPANDED | 1 |
| repo:Mamba-YOLO:wrapper:yolo / YOLO / YOLOWorld / YOLOE task wrappers | configured task wrapper | vendored model wrapper / TOP_LEVEL_AST_INDEXED_ATOMIC_COMPONENTS_PARTLY_NOT_EXPANDED | 1 |
| repo:ultralytics:11 / ultralytics / 11 | box detection (unless config head explicitly indicates otherwise); image classification; instance segmentation; keypoint / pose estimation; oriented box detection | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 8 |
| repo:ultralytics:12 / ultralytics / 12 | box detection (unless config head explicitly indicates otherwise); image classification; instance segmentation; keypoint / pose estimation; oriented box detection | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 5 |
| repo:ultralytics:26 / ultralytics / 26 | box detection (unless config head explicitly indicates otherwise); image classification; instance segmentation; keypoint / pose estimation; oriented box detection; semantic segmentation | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 10 |
| repo:ultralytics:rt-detr / ultralytics / rt-detr | box detection (unless config head explicitly indicates otherwise) | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 4 |
| repo:ultralytics:v10 / ultralytics / v10 | box detection (unless config head explicitly indicates otherwise) | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 6 |
| repo:ultralytics:v3 / ultralytics / v3 | box detection (unless config head explicitly indicates otherwise) | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 3 |
| repo:ultralytics:v5 / ultralytics / v5 | box detection (unless config head explicitly indicates otherwise) | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 2 |
| repo:ultralytics:v6 / ultralytics / v6 | box detection (unless config head explicitly indicates otherwise) | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 1 |
| repo:ultralytics:v8 / ultralytics / v8 | box detection (unless config head explicitly indicates otherwise); image classification; instance segmentation; keypoint / pose estimation; oriented box detection | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 19 |
| repo:ultralytics:v9 / ultralytics / v9 | box detection (unless config head explicitly indicates otherwise); instance segmentation | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 7 |
| repo:ultralytics:wrapper:sam / SAM / TinySAM / SAM2 or SAM3 according to local build files | prompt segmentation | vendored model wrapper / TOP_LEVEL_AST_INDEXED_ATOMIC_COMPONENTS_PARTLY_NOT_EXPANDED | 23 |
| repo:ultralytics:wrapper:fastsam / FastSAM | prompt segmentation | vendored model wrapper / TOP_LEVEL_AST_INDEXED_ATOMIC_COMPONENTS_PARTLY_NOT_EXPANDED | 1 |
| repo:ultralytics:wrapper:nas / YOLO-NAS | configured task wrapper | vendored model wrapper / TOP_LEVEL_AST_INDEXED_ATOMIC_COMPONENTS_PARTLY_NOT_EXPANDED | 1 |
| repo:ultralytics:wrapper:yolo / YOLO / YOLOWorld / YOLOE task wrappers | configured task wrapper | vendored model wrapper / TOP_LEVEL_AST_INDEXED_ATOMIC_COMPONENTS_PARTLY_NOT_EXPANDED | 1 |
| repo:yolov12:11 / yolov12 / 11 | box detection (unless config head explicitly indicates otherwise); image classification; instance segmentation; keypoint / pose estimation; oriented box detection | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 6 |
| repo:yolov12:rt-detr / yolov12 / rt-detr | box detection (unless config head explicitly indicates otherwise) | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 4 |
| repo:yolov12:v10 / yolov12 / v10 | box detection (unless config head explicitly indicates otherwise) | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 6 |
| repo:yolov12:v12 / yolov12 / v12 | box detection (unless config head explicitly indicates otherwise) | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 1 |
| repo:yolov12:v3 / yolov12 / v3 | box detection (unless config head explicitly indicates otherwise) | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 3 |
| repo:yolov12:v5 / yolov12 / v5 | box detection (unless config head explicitly indicates otherwise) | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 2 |
| repo:yolov12:v6 / yolov12 / v6 | box detection (unless config head explicitly indicates otherwise) | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 1 |
| repo:yolov12:v8 / yolov12 / v8 | box detection (unless config head explicitly indicates otherwise); image classification; instance segmentation; keypoint / pose estimation; oriented box detection | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 17 |
| repo:yolov12:v9 / yolov12 / v9 | box detection (unless config head explicitly indicates otherwise); instance segmentation | vendored model/config family / CONFIG_INDEXED_ONLY_UNLESS_NEW_ANALYSIS_BELOW | 7 |
| repo:yolov12:wrapper:sam / SAM / TinySAM / SAM2 or SAM3 according to local build files | prompt segmentation | vendored model wrapper / TOP_LEVEL_AST_INDEXED_ATOMIC_COMPONENTS_PARTLY_NOT_EXPANDED | 11 |
| repo:yolov12:wrapper:fastsam / FastSAM | prompt segmentation | vendored model wrapper / TOP_LEVEL_AST_INDEXED_ATOMIC_COMPONENTS_PARTLY_NOT_EXPANDED | 1 |
| repo:yolov12:wrapper:nas / YOLO-NAS | configured task wrapper | vendored model wrapper / TOP_LEVEL_AST_INDEXED_ATOMIC_COMPONENTS_PARTLY_NOT_EXPANDED | 1 |
| repo:yolov12:wrapper:yolo / YOLO / YOLOWorld / YOLOE task wrappers | configured task wrapper | vendored model wrapper / TOP_LEVEL_AST_INDEXED_ATOMIC_COMPONENTS_PARTLY_NOT_EXPANDED | 1 |
| repo:MSCK-Net:msck / MSCK-Net / HGNetv2_MSCK | box detection | vendored detector/model adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 4 |
| repo:MSCK-Net:deim / DEIM / DEIMv2 | box detection | vendored detector/model adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 2 |
| repo:MSCK-Net:dfine / D-FINE decoder | box detection | vendored detector/model adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| repo:MSCK-Net:rtdetrv2 / RT-DETRv2 decoder | box detection | vendored detector/model adaptation / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 1 |
| repo:MSCK-Net:backbones / MSCK backbone catalogue: HGNetv2, PResNet, CSPDarknet, CSPResNet, ViT, DINOv3, timm/torchvision wrappers | backbone / generic or foundation feature models | vendored backbone collection / AST_INDEXED_MECHANISMS_PARTLY_NOT_EXPANDED | 34 |
| repo:HGTFN / HGTFN / HGCN_MHF | hyperspectral pixel classification | vendored hyperspectral model / STATIC_INDEXED_NOT_EXECUTION_VERIFIED | 4 |
| deployment:demos / TensorRT tutorial/demo models | deployment examples; not evidence of research training | third-party deployment demos / NAME_ONLY_NOT_EXPANDED | 30 |

## 本地与历史模型源码完整列表

| 文件 | 类/定义 | 配置关联 | 状态 |
|---|---|---|---|
| CFAR.py |  | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| Combinedloss.py | CombinedDiceFocalLoss:4 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| DiceLoss.py | SoftDiceLoss:6, FocalLoss:23, MixedLoss:35 | configs/baseline_hrsid.yaml, configs/baseline_hrsid_clean_unetpp400.yaml, configs/baseline_hrsid_clean_unetpp400_seed43.yaml, configs/baseline_hrsid_clean_unetpp400_seed44.yaml, configs/baseline_hrsid_clean_unetpp400_smoke16.yaml, configs/baseline_hrsid_clean_unetpp400_smoke32.yaml, configs/baseline_hrsid_clean_unetpp400_smoke4.yaml, configs/baseline_hrsid_clean_unetpp400_smoke64.yaml, configs/baseline_hrsid_deeplabv3plus_full_protocol_budget10.yaml, configs/baseline_hrsid_frequency_unetpp400_full.yaml, configs/baseline_hrsid_frequency_unetpp400_smoke4.yaml, configs/baseline_hrsid_mask2former_full_protocol_budget10.yaml, configs/baseline_hrsid_segformer_full_protocol_budget10.yaml, configs/baseline_ssdd.yaml, configs/baseline_ssdd_clean_unetpp256.yaml, configs/baseline_ssdd_clean_unetpp256_seed43.yaml, configs/baseline_ssdd_clean_unetpp256_seed44.yaml, configs/baseline_ssdd_clean_unetpp256_smoke16.yaml, configs/baseline_ssdd_clean_unetpp256_smoke32.yaml, configs/baseline_ssdd_clean_unetpp256_smoke64.yaml, configs/baseline_ssdd_clean_unetpp256_smoke8.yaml, configs/baseline_ssdd_clean_unetpp400_scale_full.yaml, configs/baseline_ssdd_clean_unetpp400_smoke4_scale.yaml, configs/baseline_ssdd_deeplabv3plus_full_protocol_budget10.yaml, configs/baseline_ssdd_frequency_unetpp256_full.yaml, configs/baseline_ssdd_frequency_unetpp256_smoke4.yaml, configs/baseline_ssdd_frequency_unetpp400_scale_full.yaml, configs/baseline_ssdd_mask2former_full_protocol_budget10.yaml, configs/baseline_ssdd_mixedloss.yaml, configs/baseline_ssdd_no_x2_transformer.yaml, configs/baseline_ssdd_recovered_legacy256.yaml, configs/baseline_ssdd_segformer_full_protocol_budget10.yaml, configs/cross_hrsid_clean_to_ssdd256.yaml, configs/cross_ssdd_clean_to_hrsid400.yaml, configs/robustness_hrsid_clean_unetpp400_pure_background.yaml, configs/robustness_hrsid_frequency_unetpp400_pure_background.yaml | AST_INDEXED_NOT_EXECUTED |
| FocalLoss.py | WeightedFocalLoss:6 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| KD/KD_loss.py | KD_loss:7 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| KD/head_student.py | AddCoords:9, CoordConv:60 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| KD/new_Unet.py | DoubleConv:9, Down:27, Up:41, OutConv:77, UNet:86 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| KD/new_student_model.py | ShuffleUnit:23, ShuffleUnitV1:60, Down:82, Up:95, OutConv:115, UNet:123 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| KD/new_student_model_1.py | ChannelAttention:6, SpatialAttention:23, CBAMBlock:39, MultiScaleFusion:74, ShuffleUnit:94, Down:116, Up:130, OutConv:153, UNet:162 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| KD/shuffleUnet.py | DSConv:4, DoubleConv:22, Down:39, Up:53, ShuffleUnit:81, UNet:107 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| KD/student_model.py | DoubleConv:9, Down:27, Up:41, OutConv:68, UNet:77 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| KD/teacher_model.py | ContinusParalleConv:9, EncoderConv:38, SqueezeAttentionBlock:53, ChannelAttention:75, SpatialAttention:90, CBAM_block:103, CBAM_block_up:136, batch_former:169, UnetPlusPlus:187 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| KD/train_KD.py | student_train:22, KD_train:39, do_train:62 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| LIE.py | TargetSensitiveConv:6, LIEModule:25 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| archive/old_code/Unet++/model1.py | ContinusParalleConv:5, UnetPlusPlus:35 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| archive/old_code/Unet++/trainer.py | Trainer:10 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| archive/old_code/root_legacy_models/model1.py | ContinusParalleConv:5, EncoderConv:34, AddCoords:49, CoordConv:87, UnetPlusPlus:109 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| archive/old_code/root_legacy_models/model_coord(1).py | AddCoords:1, CoordConv:39 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| archive/old_code/root_legacy_models/model_new1.py | Bottleneck:10, ContinusParalleConv:37, SqueezeAttentionBlock:68, ContinusParalleConv1:89, ContinusParalleConv_down:128, ChannelAttention:161, eca_layer:178, SpatialAttention:206, CBAM_block:222, UnetPlusPlus:234 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| archive/old_code/root_legacy_models/model_new2.py | ContinusParalleConv:5, SqueezeAttentionBlock:34, UnetPlusPlus:55 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| archive/old_code/root_legacy_models/model_new3.py | AddCoords:6, EncoderConv:42, CoordConv:58, ContinusParalleConv:79, ChannelAttention:108, SpatialAttention:126, CBAM_block:142, SimAM:196, channel_shuffle:223, BlurPool:242, UnetPlusPlus:296 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| archive/old_code/root_legacy_models/model_old.py | ContinusParalleConv:5, SqueezeAttentionBlock:34, ChannelAttention:56, SpatialAttention:71, CBAM_block:84, UnetPlusPlus:119 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| archive/old_code/root_legacy_scripts/trainer1.py |  | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| boundary_loss.py | MaskBoundaryLoss:19, RecallAwareBoundaryLoss:62, MaskBoundaryCoastGateLoss:219, MaskBoundaryMetadataCoastGateLoss:280 | configs/baseline_hrsid_boundary_unetpp400_full.yaml, configs/baseline_hrsid_boundary_unetpp400_full_seed43.yaml, configs/baseline_hrsid_boundary_unetpp400_full_seed44.yaml, configs/baseline_hrsid_boundary_unetpp400_smoke4.yaml, configs/baseline_hrsid_coast_gate_targeted_unetpp400_smoke4.yaml, configs/baseline_hrsid_coast_gate_unetpp400_smoke4.yaml, configs/baseline_hrsid_frequency_boundary_unetpp400_full.yaml, configs/baseline_hrsid_frequency_boundary_unetpp400_smoke4.yaml, configs/baseline_hrsid_generic_recall_boundary_unetpp400_smoke4.yaml, configs/baseline_hrsid_metadata_cx_coast_gate_unetpp400_smoke4.yaml, configs/baseline_hrsid_rabf_unetpp400_smoke4.yaml, configs/baseline_hrsid_sampling_boundary_unetpp400_r141_smoke1.yaml, configs/baseline_hrsid_sampling_boundary_unetpp400_r142_bounded64.yaml, configs/baseline_ssdd_boundary_unetpp256_full.yaml, configs/baseline_ssdd_boundary_unetpp256_full_seed43.yaml, configs/baseline_ssdd_boundary_unetpp256_full_seed44.yaml, configs/baseline_ssdd_boundary_unetpp400_scale_full.yaml, configs/baseline_ssdd_boundary_unetpp400_smoke4_scale.yaml, configs/baseline_ssdd_coast_gate_targeted_unetpp256_smoke4.yaml, configs/baseline_ssdd_frequency_boundary_unetpp256_full.yaml, configs/baseline_ssdd_frequency_boundary_unetpp256_smoke4.yaml, configs/baseline_ssdd_frequency_boundary_unetpp400_scale_full.yaml, configs/baseline_ssdd_generic_recall_boundary_unetpp256_smoke4.yaml, configs/baseline_ssdd_rabf_unetpp256_smoke4.yaml, configs/baseline_ssdd_sampling_boundary_unetpp256_r141_smoke1.yaml, configs/baseline_ssdd_sampling_boundary_unetpp256_r142_bounded64.yaml, configs/cross_hrsid_boundary_to_ssdd256.yaml, configs/cross_ssdd_boundary_to_hrsid400.yaml, configs/robustness_hrsid_boundary_unetpp400_pure_background.yaml, configs/robustness_hrsid_frequency_boundary_unetpp400_pure_background.yaml | AST_INDEXED_NOT_EXECUTED |
| cash.py |  | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_boundary_unetpp400_full.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_boundary_unetpp400_full_region.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_boundary_unetpp400_full_seed43.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_boundary_unetpp400_full_seed44.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_boundary_unetpp400_smoke4.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_clean_unetpp400.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_clean_unetpp400_region.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_clean_unetpp400_region_gauss.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_clean_unetpp400_region_seed1_r382.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_clean_unetpp400_seed43.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_clean_unetpp400_seed44.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_clean_unetpp400_smoke16.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_clean_unetpp400_smoke32.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_clean_unetpp400_smoke4.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_clean_unetpp400_smoke64.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_coast_gate_targeted_unetpp400_smoke4.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_coast_gate_unetpp400_smoke4.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_deeplabv3plus_full_protocol_budget10.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_deeplabv3plus_preflight.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_frequency_boundary_unetpp400_full.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_frequency_boundary_unetpp400_smoke4.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_frequency_unetpp400_full.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_frequency_unetpp400_smoke4.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_generic_recall_boundary_unetpp400_smoke4.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_hdsc_generic_geometry_unetpp400_r134_bounded64.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_hdsc_instance_adapter_unetpp400_r140_smoke1.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_hdsc_instance_generic_geometry_unetpp400_r141_bounded64.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_hdsc_instance_unetpp400_r141_bounded64.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_hdsc_unetpp400_r133_smoke1.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_hdsc_unetpp400_r134_bounded64.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_mask2former_full_protocol_budget10.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_mask2former_upernet_preflight.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_maskrcnn400_bounded4.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_maskrcnn400_preflight.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_maskrcnn400_smoke2.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_metadata_cx_coast_gate_unetpp400_smoke4.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_mmdet_deformable_detr400_preflight.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_mmdet_deformable_detr_one_class_r201.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_rabf_unetpp400_smoke4.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_sampling_boundary_unetpp400_r141_smoke1.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_sampling_boundary_unetpp400_r142_bounded64.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_sbe_bega_unetpp400_smoke1.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_sbe_generic_feature_unetpp400_smoke1.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_sbe_generic_uniform_unetpp400_smoke1.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_sbe_stps_unetpp400_smoke1.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_sbrf_generic_control_unetpp400_r115_bounded64.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_sbrf_generic_control_unetpp400_r118_full_screen.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_sbrf_unetpp400_r115_bounded64.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_sbrf_unetpp400_r118_full_screen.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_sbrf_unetpp400_smoke1.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_segformer_full_protocol_budget10.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_segformer_preflight.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_yolo_obb_full_protocol_budget10.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_yolo_obb_preflight.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_yolo_obb_smoke2.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_yolo_seg_full_protocol_budget10.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_yolo_seg_preflight.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_hrsid_yolo_seg_smoke2.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_boundary_unetpp256_full.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_boundary_unetpp256_full_seed43.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_boundary_unetpp256_full_seed44.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_boundary_unetpp400_scale_full.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_boundary_unetpp400_smoke4_scale.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_clean_unetpp256.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_clean_unetpp256_seed43.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_clean_unetpp256_seed44.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_clean_unetpp256_smoke16.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_clean_unetpp256_smoke32.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_clean_unetpp256_smoke64.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_clean_unetpp256_smoke8.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_clean_unetpp400_scale_full.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_clean_unetpp400_smoke4_scale.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_coast_gate_targeted_unetpp256_smoke4.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_deeplabv3plus_full_protocol_budget10.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_frequency_boundary_unetpp256_full.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_frequency_boundary_unetpp256_smoke4.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_frequency_boundary_unetpp400_scale_full.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_frequency_unetpp256_full.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_frequency_unetpp256_smoke4.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_frequency_unetpp400_scale_full.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_generic_recall_boundary_unetpp256_smoke4.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_hdsc_generic_geometry_unetpp256_r134_bounded64.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_hdsc_instance_adapter_unetpp256_r140_smoke1.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_hdsc_instance_generic_geometry_unetpp256_r141_bounded64.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_hdsc_instance_unetpp256_r141_bounded64.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_hdsc_unetpp256_r133_smoke1.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_hdsc_unetpp256_r134_bounded64.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_mask2former_full_protocol_budget10.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_mixedloss.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_mmdet_deformable_detr_one_class_r201.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_no_x2_transformer.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_rabf_unetpp256_smoke4.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_recovered_legacy256.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_sampling_boundary_unetpp256_r141_smoke1.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_sampling_boundary_unetpp256_r142_bounded64.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_sbe_bega_unetpp256_smoke1.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_sbe_generic_feature_unetpp256_smoke1.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_sbe_generic_uniform_unetpp256_smoke1.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_sbe_stps_unetpp256_smoke1.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_sbrf_generic_control_unetpp256_r115_bounded64.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_sbrf_generic_control_unetpp256_r118_full_screen.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_sbrf_unetpp256_r115_bounded64.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_sbrf_unetpp256_r118_full_screen.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_sbrf_unetpp256_smoke1.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_segformer_full_protocol_budget10.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_yolo_obb_full_protocol_budget10.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_yolo_obb_smoke2.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_yolo_seg_full_protocol_budget10.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/baseline_ssdd_yolo_seg_smoke2.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/cross_hrsid_boundary_to_ssdd256.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/cross_hrsid_clean_to_ssdd256.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/cross_hrsid_region_gamma_to_ssdd256_r378.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/cross_ssdd_boundary_to_hrsid400.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/cross_ssdd_clean_to_hrsid400.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/diffsarshipinst/hrsid_eval_r535.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/diffsarshipinst/ssdd_eval_r535.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/mmdet/hrsid_deformable_detr_one_class_r203.py |  | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| configs/mmdet/ssdd_deformable_detr_one_class_r203.py |  | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| configs/mmdet/ssdd_deformable_detr_quality_aligned_dump_safe_r279.py |  | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| configs/mmdet/ssdd_deformable_detr_quality_aligned_r255.py |  | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| configs/robustness_hrsid_boundary_unetpp400_pure_background.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/robustness_hrsid_clean_unetpp400_pure_background.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/robustness_hrsid_frequency_boundary_unetpp400_pure_background.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| configs/robustness_hrsid_frequency_unetpp400_pure_background.yaml |  | 本索引未建立模块关联 | CONFIG_TEXT_INDEXED_NOT_EXECUTED |
| detector_extensions/mmdet_quality_aligned/deformable_detr_quality_aligned_head_r255.py | QualityAlignedDeformableDETRHeadR255:23 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| dsconv.py | My_DSConv:18, DSConv:28, DSC:93 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| guess.py | TargetSensitiveConv:22, LIEModule:44 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| hdsc_loss.py | HDSCSmokeLoss:10 | configs/baseline_hrsid_hdsc_generic_geometry_unetpp400_r134_bounded64.yaml, configs/baseline_hrsid_hdsc_instance_adapter_unetpp400_r140_smoke1.yaml, configs/baseline_hrsid_hdsc_instance_generic_geometry_unetpp400_r141_bounded64.yaml, configs/baseline_hrsid_hdsc_instance_unetpp400_r141_bounded64.yaml, configs/baseline_hrsid_hdsc_unetpp400_r133_smoke1.yaml, configs/baseline_hrsid_hdsc_unetpp400_r134_bounded64.yaml, configs/baseline_ssdd_hdsc_generic_geometry_unetpp256_r134_bounded64.yaml, configs/baseline_ssdd_hdsc_instance_adapter_unetpp256_r140_smoke1.yaml, configs/baseline_ssdd_hdsc_instance_generic_geometry_unetpp256_r141_bounded64.yaml, configs/baseline_ssdd_hdsc_instance_unetpp256_r141_bounded64.yaml, configs/baseline_ssdd_hdsc_unetpp256_r133_smoke1.yaml, configs/baseline_ssdd_hdsc_unetpp256_r134_bounded64.yaml | AST_INDEXED_NOT_EXECUTED |
| head.py | AddCoords:9, CoordConv:64 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| iq.py |  | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| kener2l.py | plot_spectrum:15 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| loss.py | SoftLoULoss:6 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| maskrcnn.py | MyDataset:21 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| model_deeplabv3plus.py | DeepLabV3PlusBinary:4 | configs/baseline_hrsid_deeplabv3plus_full_protocol_budget10.yaml, configs/baseline_ssdd_deeplabv3plus_full_protocol_budget10.yaml | AST_INDEXED_NOT_EXECUTED |
| model_full.py | ContinusParalleConv:8, EncoderConv:37, ChannelAttention:52, SpatialAttention:70, CBAM_block:86, SimAM:140, channel_shuffle:167, BlurPool:186, CBAM_block_up:239, UnetPlusPlus:273 | configs/baseline_ssdd.yaml, configs/baseline_ssdd_mixedloss.yaml, configs/baseline_ssdd_no_x2_transformer.yaml, configs/baseline_ssdd_recovered_legacy256.yaml | AST_INDEXED_NOT_EXECUTED |
| model_hdsc.py | HDSCFieldHead:7, HDSCUNetPlusPlus:28, HDSCGenericGeometryControlUNetPlusPlus:80 | configs/baseline_hrsid_hdsc_generic_geometry_unetpp400_r134_bounded64.yaml, configs/baseline_hrsid_hdsc_instance_adapter_unetpp400_r140_smoke1.yaml, configs/baseline_hrsid_hdsc_instance_generic_geometry_unetpp400_r141_bounded64.yaml, configs/baseline_hrsid_hdsc_instance_unetpp400_r141_bounded64.yaml, configs/baseline_hrsid_hdsc_unetpp400_r133_smoke1.yaml, configs/baseline_hrsid_hdsc_unetpp400_r134_bounded64.yaml, configs/baseline_ssdd_hdsc_generic_geometry_unetpp256_r134_bounded64.yaml, configs/baseline_ssdd_hdsc_instance_adapter_unetpp256_r140_smoke1.yaml, configs/baseline_ssdd_hdsc_instance_generic_geometry_unetpp256_r141_bounded64.yaml, configs/baseline_ssdd_hdsc_instance_unetpp256_r141_bounded64.yaml, configs/baseline_ssdd_hdsc_unetpp256_r133_smoke1.yaml, configs/baseline_ssdd_hdsc_unetpp256_r134_bounded64.yaml | AST_INDEXED_NOT_EXECUTED |
| model_mask2former.py | Mask2FormerBinary:25 | configs/baseline_hrsid_mask2former_full_protocol_budget10.yaml, configs/baseline_ssdd_mask2former_full_protocol_budget10.yaml | AST_INDEXED_NOT_EXECUTED |
| model_old_zuobiao.py | ContinusParalleConv:9, EncoderConv:38, SqueezeAttentionBlock:53, ChannelAttention:75, SpatialAttention:90, CBAM_block:103, CBAM_block_up:136, batch_former:169, UnetPlusPlus:187 | configs/baseline_hrsid.yaml | AST_INDEXED_NOT_EXECUTED |
| model_sampling_boundary.py | FixedBlurPool2d:8, AntiAliasDownsample:28, DetailAwareUpsample:40, SamplingBoundaryUNetPlusPlus:62 | configs/baseline_hrsid_sampling_boundary_unetpp400_r141_smoke1.yaml, configs/baseline_hrsid_sampling_boundary_unetpp400_r142_bounded64.yaml, configs/baseline_ssdd_sampling_boundary_unetpp256_r141_smoke1.yaml, configs/baseline_ssdd_sampling_boundary_unetpp256_r142_bounded64.yaml | AST_INDEXED_NOT_EXECUTED |
| model_sbe_stps.py | SARBoundaryEnergyHead:8, STPSDownsample:50, GenericSamplingDownsample:107, BoundaryEnergyGuidedSkipConcat:152, SBEStpsUNetPlusPlus:198, SBEBegaFusionUNetPlusPlus:292, SBEGenericSamplingControlUNetPlusPlus:395 | configs/baseline_hrsid_sbe_bega_unetpp400_smoke1.yaml, configs/baseline_hrsid_sbe_generic_feature_unetpp400_smoke1.yaml, configs/baseline_hrsid_sbe_generic_uniform_unetpp400_smoke1.yaml, configs/baseline_hrsid_sbe_stps_unetpp400_smoke1.yaml, configs/baseline_ssdd_sbe_bega_unetpp256_smoke1.yaml, configs/baseline_ssdd_sbe_generic_feature_unetpp256_smoke1.yaml, configs/baseline_ssdd_sbe_generic_uniform_unetpp256_smoke1.yaml, configs/baseline_ssdd_sbe_stps_unetpp256_smoke1.yaml | AST_INDEXED_NOT_EXECUTED |
| model_sbrf.py | SBRFDescriptorBuilder:8, SBRFFieldHead:63, SBRFUNetPlusPlus:91, SBRFGenericControlUNetPlusPlus:173 | configs/baseline_hrsid_sbrf_generic_control_unetpp400_r115_bounded64.yaml, configs/baseline_hrsid_sbrf_generic_control_unetpp400_r118_full_screen.yaml, configs/baseline_hrsid_sbrf_unetpp400_r115_bounded64.yaml, configs/baseline_hrsid_sbrf_unetpp400_r118_full_screen.yaml, configs/baseline_hrsid_sbrf_unetpp400_smoke1.yaml, configs/baseline_ssdd_sbrf_generic_control_unetpp256_r115_bounded64.yaml, configs/baseline_ssdd_sbrf_generic_control_unetpp256_r118_full_screen.yaml, configs/baseline_ssdd_sbrf_unetpp256_r115_bounded64.yaml, configs/baseline_ssdd_sbrf_unetpp256_r118_full_screen.yaml, configs/baseline_ssdd_sbrf_unetpp256_smoke1.yaml | AST_INDEXED_NOT_EXECUTED |
| model_segformer.py | SegFormerBinary:23 | configs/baseline_hrsid_segformer_full_protocol_budget10.yaml, configs/baseline_ssdd_segformer_full_protocol_budget10.yaml | AST_INDEXED_NOT_EXECUTED |
| model_sfic_boundary.py | BoundaryUNetPlusPlus:6 | configs/baseline_hrsid_boundary_unetpp400_full.yaml, configs/baseline_hrsid_boundary_unetpp400_full_region.yaml, configs/baseline_hrsid_boundary_unetpp400_full_seed43.yaml, configs/baseline_hrsid_boundary_unetpp400_full_seed44.yaml, configs/baseline_hrsid_boundary_unetpp400_smoke4.yaml, configs/baseline_hrsid_generic_recall_boundary_unetpp400_smoke4.yaml, configs/baseline_hrsid_rabf_unetpp400_smoke4.yaml, configs/baseline_ssdd_boundary_unetpp256_full.yaml, configs/baseline_ssdd_boundary_unetpp256_full_seed43.yaml, configs/baseline_ssdd_boundary_unetpp256_full_seed44.yaml, configs/baseline_ssdd_boundary_unetpp400_scale_full.yaml, configs/baseline_ssdd_boundary_unetpp400_smoke4_scale.yaml, configs/baseline_ssdd_generic_recall_boundary_unetpp256_smoke4.yaml, configs/baseline_ssdd_rabf_unetpp256_smoke4.yaml, configs/cross_hrsid_boundary_to_ssdd256.yaml, configs/cross_ssdd_boundary_to_hrsid400.yaml, configs/robustness_hrsid_boundary_unetpp400_pure_background.yaml | AST_INDEXED_NOT_EXECUTED |
| model_sfic_coast_gate.py | CoastGateBoundaryUNetPlusPlus:9 | configs/baseline_hrsid_coast_gate_targeted_unetpp400_smoke4.yaml, configs/baseline_hrsid_coast_gate_unetpp400_smoke4.yaml, configs/baseline_hrsid_metadata_cx_coast_gate_unetpp400_smoke4.yaml, configs/baseline_ssdd_coast_gate_targeted_unetpp256_smoke4.yaml | AST_INDEXED_NOT_EXECUTED |
| model_sfic_frequency.py | ScatteringResidualGate:8, ScatteringResidualUNetPlusPlus:56 | configs/baseline_hrsid_frequency_unetpp400_full.yaml, configs/baseline_hrsid_frequency_unetpp400_smoke4.yaml, configs/baseline_ssdd_frequency_unetpp256_full.yaml, configs/baseline_ssdd_frequency_unetpp256_smoke4.yaml, configs/baseline_ssdd_frequency_unetpp400_scale_full.yaml, configs/robustness_hrsid_frequency_unetpp400_pure_background.yaml | AST_INDEXED_NOT_EXECUTED |
| model_sfic_frequency_boundary.py | ScatteringBoundaryUNetPlusPlus:7 | configs/baseline_hrsid_frequency_boundary_unetpp400_full.yaml, configs/baseline_hrsid_frequency_boundary_unetpp400_smoke4.yaml, configs/baseline_ssdd_frequency_boundary_unetpp256_full.yaml, configs/baseline_ssdd_frequency_boundary_unetpp256_smoke4.yaml, configs/baseline_ssdd_frequency_boundary_unetpp400_scale_full.yaml, configs/robustness_hrsid_frequency_boundary_unetpp400_pure_background.yaml | AST_INDEXED_NOT_EXECUTED |
| model_unetpp_clean.py | ConvBlock:19, CleanUNetPlusPlus:35 | configs/baseline_hrsid_clean_unetpp400.yaml, configs/baseline_hrsid_clean_unetpp400_region.yaml, configs/baseline_hrsid_clean_unetpp400_region_gauss.yaml, configs/baseline_hrsid_clean_unetpp400_region_seed1_r382.yaml, configs/baseline_hrsid_clean_unetpp400_seed43.yaml, configs/baseline_hrsid_clean_unetpp400_seed44.yaml, configs/baseline_hrsid_clean_unetpp400_smoke16.yaml, configs/baseline_hrsid_clean_unetpp400_smoke32.yaml, configs/baseline_hrsid_clean_unetpp400_smoke4.yaml, configs/baseline_hrsid_clean_unetpp400_smoke64.yaml, configs/baseline_ssdd_clean_unetpp256.yaml, configs/baseline_ssdd_clean_unetpp256_seed43.yaml, configs/baseline_ssdd_clean_unetpp256_seed44.yaml, configs/baseline_ssdd_clean_unetpp256_smoke16.yaml, configs/baseline_ssdd_clean_unetpp256_smoke32.yaml, configs/baseline_ssdd_clean_unetpp256_smoke64.yaml, configs/baseline_ssdd_clean_unetpp256_smoke8.yaml, configs/baseline_ssdd_clean_unetpp400_scale_full.yaml, configs/baseline_ssdd_clean_unetpp400_smoke4_scale.yaml, configs/cross_hrsid_clean_to_ssdd256.yaml, configs/cross_hrsid_region_gamma_to_ssdd256_r378.yaml, configs/cross_ssdd_clean_to_hrsid400.yaml, configs/robustness_hrsid_clean_unetpp400_pure_background.yaml | AST_INDEXED_NOT_EXECUTED |
| modern_detector_adapter_preflight.py | _probe_package:17, package_status:38, _package_root_from_origin:45, discover_detector_config_candidates:54, classify_adapter_readiness:84, _extract_mmcv_range:138, build_toolchain_repair_plan:147, _find_first_instance_sample:246, _dataset_from_config:255, build_preflight_report:263, render_markdown:320, write_outputs:399, load_config:408, main:413 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| modern_instance_baseline_bounded.py | _dataset_from_config:24, _collate_fn:29, _make_loader:34, _set_seed:51, _build_model:59, _build_optimizer:66, _move_targets:74, compute_iou:81, compute_average_precision:89, _match_detections:116, _summarize_threshold:144, summarize_maskrcnn_predictions:164, _prediction_records:255, _skipped_summary:276, _run_training:280, _run_evaluation:369, build_bounded_report:425, render_markdown:513, write_outputs:571, load_config:580, main:585 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| modern_instance_baseline_preflight.py | CocoInstanceDetectionDataset:24 | configs/baseline_hrsid_maskrcnn400_bounded4.yaml, configs/baseline_hrsid_maskrcnn400_preflight.yaml, configs/baseline_hrsid_maskrcnn400_smoke2.yaml, configs/baseline_hrsid_mmdet_deformable_detr400_preflight.yaml | AST_INDEXED_NOT_EXECUTED |
| modern_instance_baseline_smoke.py | _dataset_from_config:24, _collate_fn:29, _make_loader:34, _set_seed:50, _build_model:58, _build_optimizer:65, _move_targets:73, _run_training:80, _run_evaluation:135, _skipped_summary:192, build_smoke_report:199, render_markdown:284, write_outputs:340, load_config:349, main:354 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| resnetplus.py | VGGBlock:5, BasicBlock:26, BottleNeck:52, NestedUResnet:80 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| sar_region_loss.py | _RegionMixin:124, RegionLikelihoodLoss:157, RegionBoundaryLikelihoodLoss:182 | configs/baseline_hrsid_boundary_unetpp400_full_region.yaml, configs/baseline_hrsid_clean_unetpp400_region.yaml, configs/baseline_hrsid_clean_unetpp400_region_gauss.yaml, configs/baseline_hrsid_clean_unetpp400_region_seed1_r382.yaml, configs/cross_hrsid_region_gamma_to_ssdd256_r378.yaml | AST_INDEXED_NOT_EXECUTED |
| sbe_loss.py | SBEMaskOnlyLoss:7 | configs/baseline_hrsid_sbe_bega_unetpp400_smoke1.yaml, configs/baseline_hrsid_sbe_generic_feature_unetpp400_smoke1.yaml, configs/baseline_hrsid_sbe_generic_uniform_unetpp400_smoke1.yaml, configs/baseline_hrsid_sbe_stps_unetpp400_smoke1.yaml, configs/baseline_ssdd_sbe_bega_unetpp256_smoke1.yaml, configs/baseline_ssdd_sbe_generic_feature_unetpp256_smoke1.yaml, configs/baseline_ssdd_sbe_generic_uniform_unetpp256_smoke1.yaml, configs/baseline_ssdd_sbe_stps_unetpp256_smoke1.yaml | AST_INDEXED_NOT_EXECUTED |
| sbrf_loss.py | SBRFSmokeLoss:10 | configs/baseline_hrsid_sbrf_generic_control_unetpp400_r115_bounded64.yaml, configs/baseline_hrsid_sbrf_generic_control_unetpp400_r118_full_screen.yaml, configs/baseline_hrsid_sbrf_unetpp400_r115_bounded64.yaml, configs/baseline_hrsid_sbrf_unetpp400_r118_full_screen.yaml, configs/baseline_hrsid_sbrf_unetpp400_smoke1.yaml, configs/baseline_ssdd_sbrf_generic_control_unetpp256_r115_bounded64.yaml, configs/baseline_ssdd_sbrf_generic_control_unetpp256_r118_full_screen.yaml, configs/baseline_ssdd_sbrf_unetpp256_r115_bounded64.yaml, configs/baseline_ssdd_sbrf_unetpp256_r118_full_screen.yaml, configs/baseline_ssdd_sbrf_unetpp256_smoke1.yaml | AST_INDEXED_NOT_EXECUTED |
| strong_baseline_yolo_smoke.py | _now_utc:27, _resolve_workspace_path:31, _as_hw:38, _set_seed:46, _probe_package:57, package_status:78, _annotations_by_image:91, _baseline_family:98, _label_format_for_family:106, _normalize_point:110, _yolo_lines_for_annotation:117, _yolo_lines_for_image:156, _selected_images:165, _copy_image:179, _materialize_split:184, _write_dataset_yaml:245, _remove_stale_yolo_caches:269, materialize_yolo_manifest:285, _ultralytics_device_arg:327, _simple_results_dict:338, _run_ultralytics_training:351, _run_ultralytics_validation:433, _skipped_summary:487, build_yolo_smoke_report:491, render_markdown:574, write_outputs:642, load_config:651, main:656 | configs/baseline_hrsid_yolo_obb_full_protocol_budget10.yaml, configs/baseline_hrsid_yolo_obb_smoke2.yaml, configs/baseline_hrsid_yolo_seg_full_protocol_budget10.yaml, configs/baseline_hrsid_yolo_seg_smoke2.yaml, configs/baseline_ssdd_yolo_obb_full_protocol_budget10.yaml, configs/baseline_ssdd_yolo_obb_smoke2.yaml, configs/baseline_ssdd_yolo_seg_full_protocol_budget10.yaml, configs/baseline_ssdd_yolo_seg_smoke2.yaml | AST_INDEXED_NOT_EXECUTED |
| train_semantic.py | import_symbol:17, load_config:22, resolve_device:27, set_seed:33, build_dataset:41, build_model:53, build_loss:80, build_optimizer:85, build_data_loader:90, unpack_batch:100, move_batch_metadata_to_device:112, count_trainable_parameters:129, apply_batch_cap:133, iter_declared_output_paths:139, maybe_unwrap_output:155, compute_model_loss:165, compute_batch_accuracy:174, collect_batch_diagnostics:181, run_forward_check:213, build_dry_run_summary:254, train:337, main:410 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| trainer.py | Trainer:13 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |
| trainer_coco.py | Trainer:14 | 本索引未建立模块关联 | AST_INDEXED_NOT_EXECUTED |

## 配置索引与训练证据的关系

JSON保留全部517份配置的关键声明行、检测/分割/OBB/分类/姿态任务标记及126份项目configs的模块关联。所有配置只证明设计/调用声明，不证明实际训练结束或对应权重来源。训练记录与权重stat由独立结果审阅整合，禁止将语义Dice、框AP、旧连通域AP、当前COCO mask AP混为一表。

Mamba-T检测训练不证明同库Segment配置已训练；MSCK eval/*.pth是bbox评价缓存而非模型state_dict；HGTFN为高光谱分类任务。别名UnetPlusPlus、重复仓库中的通用YOLO配置、未进入forward的分支和第三方层不应重复计算为独立创新。

HDSC仍是候选，需要后续检索逐实例分隔、repulsion和边界监督直接先例；本轮没有完成该项查新。当前N20按原固定端点继续，目录盘点不授权插入模块、改评价规则或重启训练。

旧8组报告保留为历史快照；关于Transformer轴的解释以此扩展审阅的BatchFormer语境修正为准。


发布说明：本文件是自写审阅与研究记录。逐文件私有JSON索引留存本地；未发布历史权重、原始数据、全量标注、运维配置或第三方论文全文。相关记录见[扩展总评](UNETPP_EXPANDED_MODEL_ASSET_REVIEW_AND_NEXT_DESIGN_20261015.md)。
