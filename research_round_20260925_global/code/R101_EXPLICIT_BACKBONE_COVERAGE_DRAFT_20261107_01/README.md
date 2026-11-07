# UNBOUND R101骨干显式元数据转换与完整覆盖候选

2026-11-07。状态：**独立候选源码；14项有限SYNTHETIC标准库测试通过；实际权重、模型提取和加载尚未执行。** 本实现不导入Torch/NumPy，不读取checkpoint、模型、数据、运行SOURCE、资格链或Ops，不构造张量、不执行训练/推理，也不提供资源/科学接受资格。

## 接口与输入边界

公共接口是 `check_coverage(source, target)`。两者必须是调用方显式提供的普通字典：每个值为 `dtype`、整数列表 `shape`、整数 `numel`，可另带小写64位十六进制 `sha256`。dtype采用 `torch.float32`；已知源BN计数采用 `torch.int64` 标量。shape、dtype及元素数既逐项比较，又核对独立固定架构，不靠两端相互一致放行。输入不被改写。

`source` 只接受原始Torchvision R101规范名。`target` 必须由调用方明确提取**且仅包含完整 `backbone.bottom_up.*` 子集**，不能把整个模型状态直接传入。FPN、SCJE、IAMR、分类/动态头等额外目标键拒收，本实现不处理它们。已转换D2格式或COCO DiffusionDet权重不属于此接口。

若输入来自JSON，先用 `parse_request(text)`，要求顶层恰为 `schema/source/target`、schema为 `R101-explicit-metadata-request-v1`；它拒绝任何层级的重复键和非有限JSON常量，大小限定8MiB。若调用方先用普通JSON解析器丢失了重复键，后续字典API无法追溯该信息，不能宣称检查过原始JSON。

```python
from coverage_check import parse_request, check_coverage
request = parse_request(request_json_text)
plan = check_coverage(request['source'], request['target'])
```

以上仅消费元数据文本，不是实际加载示例。

## 固定完整规范

规范为R101 `[3,4,23,3]` bottleneck、64通道stem、单group、各stage调制DCN与FrozenBN。必须集不从传入键推断，也不接受调用方缩小的必需集合。

| 范围 | 必须迁移条目 | state元素数 |
|---|---:|---:|
| stem | 5 | 9,664 |
| res2，3块 | 50 | 218,624 |
| res3，4块 | 65 | 1,226,752 |
| res4，23块 | 350 | 26,163,200 |
| res5，3块 | 50 | 14,987,264 |
| 合计 | **520** | **42,605,504** |

每块conv1/2/3权重、各FrozenBN的weight/bias/running_mean/running_var、每stage首块shortcut及其四项norm均须存在。FrozenBN加载器可能自动补缺失running统计，所以不能以一次strict加载未报错代替此加载前检查。state元素含FrozenBN缓冲，**不是全部可训练参数数目**。

目标必须另含33块的66个 `conv2_offset.{weight,bias}`，各块形状为 `[27,width,3,3]` 和 `[27]`，width依stage为64/128/256/512。它们合计1,975,995元素；完整目标为586条/44,581,499元素。只有这66条列入 `target_unloaded_plan`，从不映射为拟加载项；不能因此声称其实际数值零或加载后保持不变。

源fc的weight/bias与104个已知BN位置的 `num_batches_tracked` 可明确排除；旧源文件省略它们也允许。存在时仍按预定shape/dtype检查，未知位置计数、额外偏置、非规范块号、包装前缀及其他未声明键拒收。fc仅支持显式ImageNet1000类规格。没有后缀heuristics、自动前缀剥离或类型转换；目标映射有唯一性检查。

## 测试与可发布材料

运行 `python -B -m unittest -v test_coverage`，只生成明确SYNTHETIC的小型元数据字典。5个RED→GREEN小周期的原始TDD日志留私有；最终14个测试方法在0.228秒通过。可发布文件为 `coverage_check.py`、`test_coverage.py`、本README及 `TEST_SUMMARY.json`。原始日志/本地测试记录器不在发布清单，不为发布改写它们。

覆盖危险情况包括：后部R101块缺项、四项FrozenBN状态遗漏、两端同时给出相同错误shape、错dtype/numel/bool维度/摘要、未知或非规范源键、源端offset混入、目标头/FPN及额外计数混入、目标offset缺项/形状错误、嵌套JSON重复键、可选旧计数缺省，以及输入不可变和范围声明。没有运行科学suite、权重转换器或实际加载。

## 仍需真实执行证明的内容

元数据本身属于调用方陈述。通过只证明所给元数据符合固定规范，**不证明它确从真实官方文件/实际目标模型提取**。可选摘要也只是携带的字符串；源摘要与目标初始化摘要分别保留，不要求加载前相等，不冒称比对过实际张量。

未来Root仍须在独立资格内证明文件来源与字节身份、实际模型和配置、逐张量提取/加载后相等、有限值、RGB量纲/标准化、stride/FrozenBN语义、RNG及未加载参数的前后不变。DCN的zero offset与调制mask=0.5不等于普通卷积函数等价；本候选不改mask、不乘权重，也不建立这一等价。FPN/SCJE/IAMR等未被此子集覆盖，实际加载执行器必须另证它们未被意外修改。当前随机初值轨迹保持，另一预训练轨迹须另从update0建立；没有AP、论文复现、SOTA或TGRS结论。

原始来源：[D2官方转换器](https://github.com/facebookresearch/detectron2/blob/247691a1c292654867deb35cabb27f18d75a1363/tools/convert-torchvision-to-d2.py)、[固定作者R101/DCN构造](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/backbone/resnet.py#L198)、[FrozenBN](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/layers/batch_norm.py#L12)、[固定配置](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-SSDD-train.yaml#L129)。只保存自有规则和摘要，不附第三方全文。
