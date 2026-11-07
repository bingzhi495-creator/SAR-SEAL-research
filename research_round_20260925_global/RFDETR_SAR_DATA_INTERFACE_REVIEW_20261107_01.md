# RF-DETR SAR 数据接口静态审查

日期：2026-11-07。范围：指定官方固定源码副本的 datasets/coco.py、training/module_data.py、config.py 及直接调用的数据变换/评估源码；并阅读未运行候选 RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml。未导入Torch，未读取任何项目实际图像、GT、活动SOURCE或Ops，未SSH、未执行数据加载/训练。下面行号均相对该固定源码包 src/rfdetr/；没有据此宣布候选已通过运行资格。

**结论：候选可继续准备source-only适配，但真实数据接口和最终评估尚未验证。最重要的两处语义差异是：fixed576仍有随机裁剪；在线验证GT可能经过缩小再放大，不能直接代替原GT主协议。**

## 1. 目录与类别接口

候选使用 dataset_file=roboflow，其COCO目录必须是：

~~~text
dataset_root/
  train/_annotations.coco.json + images referenced by file_name
  valid/_annotations.coco.json + images referenced by file_name
  test/_annotations.coco.json  + images referenced by file_name
~~~

API split是val，实际目录名是**valid**。图像路径是split目录拼接JSON的file_name；保留合法子目录即可，无须复制、重编码图像。依据：datasets/coco.py:1639–1646、761–774；datasets/__init__.py:63–66、89–101。

不要误改成 dataset_file=coco：后者走train2017/、val2017/、annotations/instances_train2017.json等布局，且普通检测/分割默认**不重映射类别**。DataModule的coco test阶段会回到val；roboflow COCO缺test标注会报错，不会因run_test=false自动形成可用测试集。依据：datasets/coco.py:1509–1515、1567–1573；training/module_data.py:463–508。

Roboflow路径按原category id排序，过滤未标注的分组父类，映射为连续label 0..N−1；非训练split优先复用train的cat2label，逆label2cat附在COCO对象上。单类原id=1应映射为label=0，原GT仍保留id=1；预测导出时逆映射回1。**不要为了num_classes=1直接改写主GT类别id。** 若使用raw COCO路径却保留原id=1，只有一个输出槽可能越界。类别名、父类规则、映射与逆映射必须在后续真实接口验证中明示；训练映射缺失/不可读时源码会警告并回落split-local，不能接受为静默成功。依据：datasets/coco.py:101–198、731–750、854–864、1663–1666。

## 2. 掩膜、空目标、crowd及几何处理

| 情况 | 官方代码实际行为 | 本项目适配要求 |
| --- | --- | --- |
| polygon | pycocotools栅格化；同一实例多部分做union；扁平且少于3点的部分跳过 | 保留已有合法polygon；复杂孔洞/原二值GT优先保留其准确RLE表示，不以粗轮廓改变真值 |
| RLE | 接受counts为str/bytes的压缩RLE，或list的非压缩RLE；要求counts、size结构，错误类型报错 | 验证size=[H,W]、解码尺寸和mask像素语义；JSON压缩counts应是可序列化字符串 |
| 空图像 | 无annotation时得到boxes(0,4)、masks(0,H,W) | 合法背景图应保留，不伪造船舶 |
| 有bbox但segmentation缺失/None/空/退化 | 得到**全零mask且保留实例行**，不会自动拒绝 | 对mask实验必须显式区分缺失标注与合法背景；不让“可加载”掩盖缺mask；**禁止用bbox填充mask** |
| iscrowd=1 | ConvertCoco从训练target移除；评估callback可经DataModule的原COCO对象重新附加crowd | 保存原iscrowd、类别及真实crowd mask；独立评估沿用主协议；不绕开DataModule后假定crowd仍自动忽略 |
| bbox | 原COCO xywh→绝对xyxy；clip到图像边界；宽高≤0的实例及同步字段移除 | 后续检查移除数量/原因，不静默改主GT；单位与图像尺寸必须一致 |
| masks/boxes/area | 详见下文；并非所有变换都从mask重新生成bbox | 训练target bookkeeping与原始GT的语义必须分开记录 |

掩膜依据：datasets/coco.py:399–523、843–880、919–932；crowd恢复：同文件589–656、training/callbacks/coco_eval.py:1570–1661。

候选torchvision变换的几何细节：

- Resize对图像bilinear+antialias，对tv_tensors.Mask走nearest；boxes按坐标缩放，area乘x/y比例，更新size，orig_size保留解码原尺寸。它**不会每次从缩放后的离散mask重新计算bbox/像素area**。
- crop对boxes裁剪，删除退化框；原本非空但裁完为空的mask实例会删除，而原本就是box-only的空mask行可保留。同步labels/iscrowd/masks等，area重算为裁后**bbox面积**；bbox仍是裁后的框，不是自动mask-tight框。
- Normalize将boxes从绝对xyxy变为按当前图像W/H归一化的cxcywh。训练中area的上述更新，不意味着主评估可以替换原GT area或mask定义。

依据：datasets/_torchvision.py:59–121、380–430、590–618；datasets/transforms.py:50–95。不要把Albumentations路径中的mask重算框行为套到本候选的torchvision路径。

## 3. 灰度输入、归一化、576与padding

decode_pil_image经Pillow或可用JPEG解码器输出RGB；Pillow路径明确convert("RGB")。对普通8-bit灰度L图，这是三个相同灰度通道的输入适配，**不是三种独立模态**。ToDtype(float32,scale=True)后用ImageNet mean=(0.485,0.456,0.406)、std=(0.229,0.224,0.225)，所以归一化后三个数值通道不再相同。禁止额外重复除255或重复Normalize；若真实源是16-bit/float，必须单独核验量化与辐射处理，不能假定这一RGB路径保真。依据：datasets/io_utils.py:98–124，datasets/coco.py:1186–1188、1222–1228。

mask任务禁用JPEG draft缩小解码；val/test也禁用，避免orig_size随解码变化。依据：datasets/coco.py:296–345。该保护不等于验证mask后续resize完全无损。

候选 square_resize_div_64=true **是直接拉伸到576×576，非letterbox保长宽比**；非方形图会各向异性变形。名字含div_64并不代表所有模型collator按64：实际block_size=patch_size×num_windows，Nano为12×1=12；576恰好可整除。batch零padding放在右/下，NestedTensor.mask=True表示padding；原图区域为False。当前单张固定方形576正常无额外padding。依据：datasets/coco.py:1456–1475、1227–1228，training/module_data.py:306–320，utilities/tensors.py:152–187。

Nano官方默认resolution=312、patch=12、PE=26；候选576/PE48整数上相容，但属于适配配方。必须后续核权重加载/位置编码插值和输出尺寸，不能把“576可整除”当作预训练初始化已经匹配。依据：config.py:1106–1119。

**multi_scale="off"不关闭scale_jitter。** 候选scale_jitter=true时，训练仍以50%选择直接576、50%选择短边400/500/600→随机方形裁剪(384..600并按输入夹紧)→576；默认还有p=0.5水平翻转。val/test只有确定性576 resize。准确描述应为“固定输出576，保留尺度/裁剪增强”。如果要无随机裁剪，另将scale_jitter=false；是否这样改应由公平配方决定，而不是为了文字方便。依据：datasets/coco.py:1037–1087、1190–1228；datasets/_torchvision.py:192–209、452–476。

## 4. maxDets=100及在线GT的两个评价陷阱

**100有三层含义。** 候选num_queries=100、num_select=100、eval_max_dets=100分别限制推理查询容量、输出top-k、评估截断；group_detr=13是训练分组，不会把推理容量变成1300，也不等于有效batch乘13。单类密集场景若GT数量超过100，容量/召回上限必须后续核查，不能只把eval_max_dets调大便解决。即便GT≤100，也不保证错误预测不会占据top-k。

callback使用max_detection_thresholds=[1,10,100]。源码默认eval_max_dets是500；faster_coco_eval汇总AP按最大配置阈值，不能将官方AP500、某方法AP1000与当前AP100直接混表。100适合作为统一COCO主口径的候选，但最终必须与主项目现行GT、maxDets规则一致；如另报500需并列明确。模型输出top-k按query×class score选取，此处单类100与capacity一致。依据：config.py:1372，training/callbacks/coco_eval.py:279–305，training/coco_map.py:418–441，models/postprocess.py:130–153。run_test=false表示训练后未自动形成测试结果，不能把val当test。

**最重要：全尺寸预测不代表原GT未变。** eval_masks_head_resolution=false会将预测mask logits bilinear插值到原尺寸后按logit>0二值化，这个方向合理；但callback._convert_targets读取的是dataloader中**已经nearest缩到576的GT mask**，再nearest放大到预测/原尺寸。它不是重新从canonical原尺寸GT解码，因此小目标和边界可能变化，往返resize不保证恒等；其entry也没有直接透传原target area。默认flag=false避免了低mask-head网格评估，却**不能解决GT往返重采样**。这是源码可见的协议风险，不是本轮实测偏差大小。

依据：models/postprocess.py:223–319；training/callbacks/coco_eval.py:556–570、1631–1661；config.py:1389–1398。建议保留在线验证曲线为“官方loader口径”，最终主表另由**预测回到原尺寸 + 已资格化原GT的独立评估器**完成；不去改写主GT迎合在线callback。额外禁止predict默认置信阈值截断后再计算完整AP；导出时应按统一规则保留连续分数/top-k，阈值预先约定。

## 5. source-only适配方案与最小真实验证门槛

当前只准备reader/config及参数化测试设计，不生成派生标注、不扫描/复制/哈希活动科学载荷。后续得到正式数据读取资格后：

1. 保留主GT、原split成员、image/annotation/category id、真实mask与iscrowd/area语义。适配器只解析固定split、提供图像路径与内存标签映射；如需Roboflow目录视图/派生JSON，另建只读接口视图并记录来源与规则，不能覆盖原件或重新随机划分。
2. 训练允许标准在线增强后的target；最终预测用label2cat还原原类别、原image_id、原尺寸mask，交回主协议评估器。主GT不经过576往返变换。若需要从真实mask重算派生bbox/area，必须在独立训练接口明确记录，最终主评估保留原定义。
3. 不以“框变mask”、插值假GT或补全缺标凑齐segmentation。缺失/空的真实实例mask须阻止资格通过或单列为不支持，而非悄悄删除测试目标。

**最小真实验证必须实际检查，以下本轮均未做：**

- split真实成员/原图路径与image_id、JSON H/W一致，train/valid/test不串集；类别唯一与train映射、逆映射完全往返，未知类别不回退猜测。
- 至少覆盖真实polygon、多部件/RLE、合法空图、边界/细小船、相邻船和crowd（若数据存在）；逐实例检查mask解码非空、尺寸、实例数与boxes/labels同步。保留孔洞/断开部分；与原GT像素相等，不能仅凭可视化“大致相似”。
- 原RGB适配前后的8-bit灰度值、/255及Normalize只执行一次；记录原图/target/576/padded tensor/mask头/原尺寸预测的实际shape、dtype与坐标。
- 分别检查直接resize和裁剪分支的mask/box/area变化、保留/删除计数；核原数据最大/高分位实例数与100容量上限，不能截断GT。
- 用同一批预测对比官方在线metric与canonical原GT评估，核AP、AP50/75、尺寸段、crowd忽略和maxDets；若有差异，定位GT重采样/area/阈值/类别映射，不能选较高者当结论。
- 资格后做最小batch的实际forward/损失有限值、空目标/crowd路径与预测导出回读，并确认实际优化步数/采样次数、微批累积和EMA端点。synthetic fixture只能验证接口分支，不能替代真实输入验证或证明训练就绪。

状态：静态审阅完成；未修改候选YAML或官方源码，未开展真实数据验证/训练。建议Root先明确最终canonical评估通道，再定576增强表述和100容量/评价协议，避免长训后发现不可比。

