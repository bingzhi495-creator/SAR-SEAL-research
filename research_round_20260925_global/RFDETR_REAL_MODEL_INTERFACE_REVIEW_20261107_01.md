# RF-DETR 1.12.0：真实SegNano权重加载与单张forward接口审查

日期：2026-11-07。依据官方1.12.0普通源码副本，版本在`pyproject.toml:7`明确。**本审查只读源码；没有导入Torch、没有读取权重、没有运行模型或训练。** 以下代码是供另行取得资源与执行资格后的建议，不是已经通过的测试结果。所有源码定位均相对官方源码根，不包含私人路径。

## 1. 最小建议

第一次真实模型检查用**CPU、float32、batch1、312×312、官方COCO原分类头、eval、原生eager forward**。先证明官方权重与原架构严格兼容，再考虑CUDA及单类SAR头。直接调用底层`LWDETR`，不要调用`.inference()`，也不要调用`.export()`。

- `.inference()`默认`compile=True`（`src/rfdetr/detr.py:1747`）；即使传`compile=False`，仍深拷贝完整权重并切到export模式（`:1834–1848`），所以不能靠关闭compile避免额外模型副本。
- 官方高层构造最终通过`load_pretrain_weights`加载，但其底层是`load_state_dict(strict=False)`（`src/rfdetr/models/weights.py:666–687`）。构造成功不能写成“strict加载成功”。
- `build_model_from_config`只构建模型，不加载完整RF-DETR权重（`src/rfdetr/models/lwdetr.py:995–1040`），适合受控strict检查。可用`dataclasses.replace(MODEL_DEFAULTS, force_no_pretrain=True)`明确禁止上游backbone预训练下载；该标志属于`ModelDefaults`，不是可直接传给SegNano高层构造器的ModelConfig字段。

## 2. 依赖与额外联网

`pyproject.toml:38–48`的核心依赖是：Python≥3.10、torch≥2.2、torchvision≥0.17、transformers≥5.1且<6、pydantic≥2且<3、requests、numpy、tqdm、supervision≥0.29、pyDeprecate≥0.9且<0.10。Pillow、PyYAML等还有依赖链；**torch与torchvision必须是相容构建**，不能仅分别满足版本下限。

另需确认SciPy：`src/rfdetr/models/matcher.py:29`顶层导入`scipy.optimize.linear_sum_assignment`，而`lwdetr.py:48`顶层导入matcher。即使仅forward、不计算匹配也会经过此导入。它在pyproject中归为训练可选项，实际环境可能由其他核心包间接安装，但不能假定不存在依赖。可只使用项目核心依赖并显式确认SciPy，无需为了单次forward装整套训练、COCO评估、LoRA、ONNX/TensorRT或可视化附加组件。

实际deformable attention使用`ms_deform_attn_core_pytorch`（`src/rfdetr/models/ops/modules/ms_deform_attn.py:28`、`src/rfdetr/models/ops/functions/ms_deform_attn_func.py:26`），不应为这次CPU原生forward额外编译第三方CUDA算子。CPU运算可用性仍须由真正执行确认，本次没有测速度或兼容性。

联网路径须分别控制：

1. **RF-DETR完整模型权重**：高层`RFDETR.__init__`会调用下载逻辑（`detr.py:721–723`）。SegNano注册名为`rf-detr-seg-nano.pt`，实际官方URL是`https://repo.roboflow.com/rfdetr/rf-detr-seg-n-ft.pth`，注册MD5为`9995497791d0ff1664a1d9ddee9cfd20`（`assets/model_weights.py:242–245`）。扩展名不同不代表不同模型。下载/完整性核验由执行方负责，本审查没有读取或哈希权重。
2. **DINOv2上游权重**：`lwdetr.py:879`根据`pretrain_weights is None`决定是否请求backbone预训练。窗口DINO配置在本地构建；`dinov2.py:203–209`只有`load_dinov2_weights=True`时调用`from_pretrained`。`backbone/__init__.py:155`用`force_no_pretrain`进一步关闭。SegNano本身patch12、非原生PE尺寸通常也会使DINO下载关闭，但不应把这个间接条件当作通用离线控制。
3. **非窗口backbone**：`dinov2.py:129–141`要求从hub加载，不属于本次SegNano最小路径。不要顺手更换encoder后继续承诺离线。

## 3. strict加载与单次forward建议代码

下例使用固定1.12.0内部构建接口和官方安全加载器；内部API不应假设跨版本稳定。假定执行方已把正确版本源码放入Python搜索路径，并提供经资格核验的本地官方权重路径。**本段没有执行。**

```python
from dataclasses import replace
from pathlib import Path
import torch
from rfdetr.config import RFDETRSegNanoConfig
from rfdetr.models import MODEL_DEFAULTS, build_model_from_config
from rfdetr.utilities.io import _safe_torch_load

# 构造、加载及任何分类头修改均在inference_mode外完成。
cfg = RFDETRSegNanoConfig(pretrain_weights=None, device="cpu", num_classes=90)
defaults = replace(MODEL_DEFAULTS, force_no_pretrain=True)
net = build_model_from_config(cfg, defaults=defaults)

weights_file = Path("<qualified-local-official-seg-nano-checkpoint>")
checkpoint = _safe_torch_load(str(weights_file), trust=False)
assert isinstance(checkpoint, dict) and "model" in checkpoint
state = checkpoint["model"]
assert isinstance(state, dict)
net.load_state_dict(state, strict=True)
del state, checkpoint

device = torch.device("cpu")  # CUDA须在另行核验资源后显式选择。
net = net.to(device=device, dtype=torch.float32).eval()
image01 = torch.full((1, 3, 312, 312), 0.5, device=device)
mean = image01.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = image01.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
inputs = (image01 - mean) / std
with torch.inference_mode():
    outputs = net(inputs)
    shapes = {k: tuple(outputs[k].shape)
              for k in ("pred_logits", "pred_boxes", "pred_masks")}
    finite = {k: bool(torch.isfinite(outputs[k]).all()) for k in shapes}
print(shapes, finite)
```

strict不通过时保留missing/unexpected/shape mismatch，先比较checkpoint元数据与模型配置。不要立即改成strict=False、随意删键、改变分组或插值PE后仍写“原始权重严格一致”。官方加载器还包含分类头对齐、query按组裁剪、位置编码插值等兼容操作（`weights.py:432–556`、`:642`）；**经声明的规范化后strict通过**和**原始state直接strict通过**是不同证据，应分别记录。本次未读权重，不能保证其键名和分组与默认配置实际完全相同。

## 4. 输入与输出真实契约

| 对象 | 1.12.0实际契约与定位 |
|---|---|
| SegNano结构 | `config.py:1106–1119`：windowed-small、4 decoder层、patch12、1窗口、312输入、PE26、100queries、100select、config.num_classes90；`ModelConfig.group_detr=13`在`:725` |
| 类别头 | `lwdetr.py:853`再加1，COCO原结构**91个logit槽**。其推理是各槽sigmoid，不是softmax后删最后一个背景槽（`postprocess.py:130–153`）。COCO最后槽90仍可能是真实类别，不能套通用DETR的“去最后背景”规则 |
| 输入张量 | 底层接受CHW列表或BCHW Tensor，并转成NestedTensor（`lwdetr.py:497–498`）。**底层不做图像缩放或归一化**。本次直接给3通道、连续float32、312方形；实际图像需RGB/灰度复制到3通道，按[0,1]后减ImageNet mean/std |
| 高层predict的区别 | 高层Tensor输入应是[0,1]，会再缩放和归一化；已手动标准化的张量不能再传predict。真实归一化在`_prediction/core.py:1207–1217`，参数在`detr.py:692–693`。固定当前接口避免双重标准化 |
| 尺寸 | SegNano至少满足边长为12倍数；第一次严格用312，不先改分辨率/patch/窗口/PE（`variants.py:177–185`）。其他Seg尺寸窗口不同，不能直接复用该整除条件 |
| eval输出 | 预期`pred_logits`为(1,100,91)，`pred_boxes`为(1,100,4)归一化cxcywh；`pred_masks`为(1,100,78,78) **原始logits**。78来自312//mask_downsample_ratio4；`heads/segmentation.py:338–357`。这是静态预期，尚非本机运行结果 |
| mode差异 | `.eval()`只保留1组query；训练模式使用全部训练组，mask变成spatial/query/bias分解字典，并算辅助层mask；不能拿训练分解对象当dense mask（`lwdetr.py:509–520`、`heads/segmentation.py:361–405`） |
| 辅助输出 | eval仍可含aux/enc的框和logit；默认不计算这些层的mask。不要设置`all_layer_masks=True`来做首次forward（`lwdetr.py:483–492`） |
| mask后处理 | `PostProcess`按query/class分数选出结果，mask **先双线性上采样logit，再以0阈值二值化**，输出形状(K,1,H_original,W_original) bool（`postprocess.py:223–319`）。先阈值再插值不是同一协议；maskAP须对原图尺寸 |

纯接口检查只需形状、有限值、设备、dtype与strict结果；合成图上有无检测和类别分数不能证明SAR有效，更不能视为新基线AP。

## 5. COCO→单类SAR分类头与设备迁移

先完成COCO91槽的strict检查，再在独立阶段决定单类初值。高层`num_classes=1`最终对应2输出槽；低层`net.reinitialize_detection_head(2)`的入参是**实际槽数**，不能误传1。它同时处理主分类头和two-stage所有encoder分类头（`lwdetr.py:232–249`）。训练标签与输出槽语义要按数据转换核对，不能只凭“单类”猜category ID；本次不读数据也不设定训练映射。

方法名虽叫reinitialize，实际`_resize_linear`是截断/复制已有COCO行（`lwdetr.py:58–85`），**不是随机初始化，更不是自动选择COCO的boat行**。若研究协议需要新头随机初值，须显式处理所有相应分类头并记录；否则应诚实记录官方截断初值。首次接口检查无需改头。

设备风险：

- `_build_model_context`刻意把模型保留在CPU，`device="cuda"`只记录意图（`inference.py:375–380`、`:453–470`）。高层predict会延迟移动；直接取`wrapper.model.model`后forward必须自己移动。
- `_resize_linear`用未指定device/dtype的`nn.Linear`新建模块（`lwdetr.py:76`），所以对已CUDA/低精度模型改头后可能混合CPU/float32新头。**在CPU/float32上完成加载和头修改，之后整体`.to(device,dtype)`**，不要依赖历史device字段。
- 移动底层模型后若还使用高层wrapper，须保持context.device与config一致，否则其延迟设备逻辑可能再次移动。最小验证只用底层模块，避免两套设备状态。
- CPU先用float32；不要为省资源直接用float16，部分CPU算子/初始化和不同Torch构建的支持不一。`.eval()`和`torch.inference_mode()`分工不同，两者都要；mode只包forward，避免创建后续要训练的inference tensor参数。
- CPU线程数、超时和CUDA显存门限由执行方约束；本审查不证明峰值内存、耗时，也没有重试运行。

## 6. checkpoint安全兼容

`utilities/io.py:28–110`先尝试`weights_only=True,map_location="cpu"`；再用局部`safe_globals([argparse.Namespace,types.SimpleNamespace])`兼容旧args；只有trust显式为True才允许full pickle。PyTorch声明的2.2最低版本不能自动证明所有新序列化辅助API均存在：若第一步因Namespace失败，旧Torch没有`serialization.safe_globals`会成为兼容风险。执行前应核对此API；不能用无条件`weights_only=False`掩盖问题。

另外config导入会执行设备探测（`config.py:232–270`），新版走torch.accelerator，旧版退回cuda availability探测；CPU配置不代表完全没有设备探测。正式执行仍应在独立受控进程完成。无需读取任何数据集、创建优化器、调用criterion或训练，便可完成这次真实模型接口检查。

交付状态：静态调用链与风险已核验；strict加载、真实forward、CPU/GPU可用性及资源用量均待执行方实际确认。
