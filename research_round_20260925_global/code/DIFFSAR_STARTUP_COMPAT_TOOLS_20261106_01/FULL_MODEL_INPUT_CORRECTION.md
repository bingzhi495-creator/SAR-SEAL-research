# 完整模型入口稿的数据配置更正

日期：2026-11-06。本附注明确更正 `FULL_MODEL_ENTRY_COMPAT.md` 初稿中的输入通道和训练尺寸说明。该初稿 SHA-256 为 `698b81c50c3c4f2fd646959cac3498b23cacf960deaf5bd40d9061f9f62ab2c3`，为保持已发送审阅记录不原地改写；发布版本必须合并本更正，不能继续将错误的 BGR 说明作为执行依据。

固定作者提交 `bdc3362372213bc8c3a1f6d5b7b74dc224d1b871` 的两个训练 YAML 均明确 `INPUT.FORMAT: RGB`，不是 BGR。逐行独立核查结果如下：

| 作者训练配置 | FORMAT | MIN/MAX_SIZE_TRAIN、MIN/MAX_SIZE_TEST | CROP |
|---|---|---|---|
| SSDD | RGB | 512 | 启用，absolute_range，512×512 |
| HRSID | RGB | 1024 | 启用，absolute_range，1024×1024 |

两者 `PIXEL_MEAN=[123.675,116.28,103.53]`、`PIXEL_STD=[58.395,57.12,57.375]`。构造真实单批应由作者 mapper 使用所选训练 YAML 读取和联合变换 image/box/mask，再返回 BitMasks；不应按通用 D2 默认格式手工假定 BGR，也不能把 HRSID 的 512 输入称为保留作者训练配方。若为了资源限制降低 HRSID 尺寸，应明确称尺寸适配后的运行检查，并保留对原 1024 配方的执行缺口。

原稿所述完整网络的微型算子预算不足、BitMasks 必要性、三文件启动兼容补丁及损失不变等结论不受此更正影响。此附注仍是静态核查；不代表完整网络或训练已经执行。

精确来源：[SSDD train YAML，第19–34行](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-SSDD-train.yaml#L19)、[HRSID train YAML，第19–34行](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-HRSID-train.yaml#L19)、[作者 mapper](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/dataset_mapper.py#L69)。

依赖补充：官方 [timm 0.6.13 元数据](https://pypi.org/pypi/timm/0.6.13/json) 提供 549,137 字节的 `py3-none-any` wheel，SHA-256 `ea5aed42f94062a80da414e6f1791cb82012fdb54f7db72c607637914a521345`，Python≥3.7；声明依赖为 torch≥1.7、torchvision、pyyaml、huggingface-hub，不含 einops 或 safetensors。版本选择仍是本轮显式兼容选择，不能冒称作者锁定环境。
