# Teacher真实图像输入合同：普通发布源码静态证据

日期：2026-10-17。仅阅读独立发布源码、配置与普通说明；未导入Torch/NumPy、执行数值、读取原图/标注/权重、Ops、H、运行中Source或科学资格链。以下是源码声明，不能替代下一次实际输入复现。

## 已找到的明确合同

1. **真实主输入与灰度辅助目标分开。** [T:246–290](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/baaa78464cd81ec2e9069e89726d1111bd892d86/research_round_20260925_global/multimodal_data_pilot_20260928/native_ordinal_two_mode_mask_adapter_learning_preparation_20261008/train.py#L246) 的模型输入链为 `LoadImageFromFile(to_float32=True) → LoadAnnotations(with_bbox=True,with_mask=True) → Resize(scale=(800,800),keep_ratio=True) → PackDetInputs → model.data_preprocessor(...,training=True)`。loader显式要求float32，不在这段代码中执行 `/255`、手工Normalize或额外随机增强。它确实读取标注；278–284行仅在复制的模型前向samples中移除GT字段，不是无标注loader。
2. **实际画布与有效内容。** T:267–289要求输入形状`[1,3,800,800]`、`batch_input_shape=(800,800)`，保存`ori_shape/img_shape/scale_factor`。独立公开几何检查 [E:243–260](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/baaa78464cd81ec2e9069e89726d1111bd892d86/research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/training_path_inference_control_source_20261009/evaluate_common_cpu.py#L243) 声明原图`(H,W)`、`s=800/max(H,W)`、内容尺寸`(int(H*s+0.5),int(W*s+0.5))`，scale_factor为实际整数宽高比。必须保存实际尺寸，不能仅存理想比例。
3. **固定Pad声明。** [C:119–139](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/baaa78464cd81ec2e9069e89726d1111bd892d86/research_round_20260925_global/multimodal_data_pilot_20260928/native_coco_comparator_micro_preflight_preparation_20261008/run.py#L119) 覆写`pad_mask=True,pad_size_divisor=32`及`BatchFixedSizePad(size=(800,800),img_pad_value=0,pad_mask=True,mask_pad_value=0,pad_seg=False)`；声明keep-ratio 800 resize、无增强。未在本轮读取外部preprocessor实现，故不额外断言其颜色转换、归一化与Pad的内部执行顺序或默认参数。
4. **辅助灰度不能冒充输入。** T:265–277另用`cv2.IMREAD_GRAYSCALE`读取uint8原灰度，用nearest缩至实际内容尺寸，再放进右/下补零的`gray800`。这是另一目标通路，不是三通道归一化teacher输入。其`native_input_gamma=1`记录也不能补出mean/std或RGB顺序。
5. **原图来源能定位到根目录及拼接关系。** T:601–606使用`CocoDataset(ann_file=spec['inputs']['fit_annotation'],data_prefix=dict(img=spec['image_root']))`产生`info`，T:253绑定真实`info['img_path']`。另一个公开检查 [F:1293–1303](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/baaa78464cd81ec2e9069e89726d1111bd892d86/research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/shared_readout_objective_fullFIT24_source_20261010/qualify_cpu.py#L1293) 明确FIT和DEV共用`E:/PSeg_SSDD/voc_style/JPEGImages_train`，并检查集合分离；E:257要求实际路径等于`image_root / COCO images.file_name`。本轮没有读取任何COCO内容或原图，不能给出已核验的单图清单、现存文件完整性或当前活动任务实际引用。

## 尚未找到，不能猜填

| 项 | 已读源码中的状态 |
| --- | --- |
| mean、std的准确数值及单位 | T和C均不声明；由其绑定模型config继承。未找到可用于当前teacher的有效合并配置数值。 |
| 是否BGR→RGB、RGB→BGR及实际解码顺序 | T未显式给`color_type/channel_order`，C未覆写颜色转换。三通道和`to_float32=True`不能证明RGB。外部loader默认实现和data_preprocessor配置尚未核对。 |
| 是否额外除255、具体归一化公式 | T没有手工除255；不能据此断言外部preprocessor内部一定没有，也不能把旧语义UNet++习惯搬来。 |
| Resize实际插值默认值、Pad默认方向的实现 | T只显式给scale/keep_ratio；灰度nearest不适用于主输入。渲染裁剪代码支持右/下画布约定，但实际库实现仍需绑定核对。 |
| 当前teacher合并config的绝对路径/完整继承值 | T:562仅从`spec['inputs']['config']`strictref加载。未读该私有绑定；普通源码未给出这个ref的实际path。 |

## 已找到的精确配置入口与证据边界

- [I:24–30、254–267](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/baaa78464cd81ec2e9069e89726d1111bd892d86/research_round_20260925_global/multimodal_data_pilot_20260928/native_coco_comparator_initialization_preparation_20261008/qualify_initializers_cpu_v2.py#L24) 将官方Mask2Former标识锁为 **`mask2former_r50_8xb2-lsj-50e_coco.py`**，要求路径含`.mim`；I:86–103递归收集本地`_base_`闭包。C:103–112从`spec['initializers']['mask2former']['config']`加载该配置并要求闭包在sources中。这里定位的是官方初始化入口，不能单凭文件名证明它就是当前teacher的完整派生配置。
- [A:5](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/baaa78464cd81ec2e9069e89726d1111bd892d86/carrier/configs/a0_m2f_r50_pseg_smoke.py#L5) 给出精确命名入口 **`mmdet::mask2former/mask2former_r50_8xb2-lsj-50e_coco.py`**。其安装绝对前缀本轮未见，不能自行拼成已核路径。
- A:14–45是较早的**512**烟雾配置，含RandomFlip/RandomResize/RandomCrop；不能替代T的固定800加载合同。其他carrier继承文件也不能替代当前绑定config。

## 给真实图像B0的最小接线结论

后续由Root在其合法独立执行中取得T对应的**有效合并config、外部loader/preprocessor实现版本与默认值**，把颜色、mean/std、数值范围、插值和Pad顺序显式固定后再定义新400分支。本轮没有替Root读取这些绑定对象。若决定从实际teacher规范化800画布缩到400，须另外固定800→400的插值合同；这是一项新分支定义，尚无真实数值证明。

每图至少保留原`image_id/file_name/img_path`及图像摘要、原/实际/画布尺寸和实际scale_factor；不得重建另一套猜测几何。T:54–62的旧渲染是`M200→800→实际内容裁剪→原图`，双线性`align_corners=False`；新δ400应定义自己的`400→800→裁剪→原图`，不能先降200后声称等价。mean/std和颜色未补齐前，不应宣称已经复现teacher输入。

## 实质逐段读取清单

所有路径均在`E:/SAR/seal/github_release_20261011/`；上文T/C/E/F/I/A链接为完整绝对路径。

- T：1–110、190–300、535–610、706–771；重点246–290、562、601–606。
- C：1–165；重点101–139。
- E：1–65、215–270；重点243–260。
- F：1281–1304；只读发布的普通检查源码，没有执行其中资格或张量读取。
- I：1–65、86–105、140–195、245–270；只读配置标识/闭包/加载逻辑，没有读取它引用的任何checkpoint或资格。
- A：1–78。
- [既有wiring review](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/baaa78464cd81ec2e9069e89726d1111bd892d86/research_round_20260925_global/UNETPP_FULL_BACKBONE_WIRING_STATIC_REVIEW_20261017.md#L85)：1–130，仅作索引和交叉核对。
- 额外仅检索了carrier继承声明：[a0_m2f_r50_pseg.py:5](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/baaa78464cd81ec2e9069e89726d1111bd892d86/carrier/configs/a0_m2f_r50_pseg.py#L5)、[a0_m2f_r50_hrsid.py:4、38–45](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/baaa78464cd81ec2e9069e89726d1111bd892d86/carrier/configs/a0_m2f_r50_hrsid.py#L4)、[c0_m2f_cocoinit_hrsid.py:6](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/baaa78464cd81ec2e9069e89726d1111bd892d86/carrier/configs/c0_m2f_cocoinit_hrsid.py#L6)。这些配置族不是当前teacher的输入合同证据。其他仅路径/单行命中的历史副本没有被用于建立本报告结论。
