# RF-DETR 1.12：COCO→SAR 单类分类头初始化核查

状态：2026-11-07，**仅静态查源；没有导入 Torch、读取权重或开展实验**。未改候选 YAML、冻结控制或主段。官方源码固定 commit `639eac4555ceb89ef399ccda3fde716bbcfe34cb`，不使用会继续变化的 develop 分支作本结论依据。

## P0 结论

若真实输入为官方 COCO 91-slot checkpoint，现候选 YAML 的显式 `num_classes: 1` 经官方 loader，最后得到 **两槽，保留原 COCO 行 `[0, 1]`，不是随机重置，也不是自动选 boat**。原 boat 行 9 会从 decoder 和全部 two-stage encoder 分类头中被截掉。其他共享表征、定位和分割权重仍可迁移，不能把“船行未保留”说成“船舶知识全部消失”，更不能先认定此初始化性能差。

因此必须在正式长训前明确初始化政策。当前 YAML 只能被描述为 **A：官方默认前缀截取迁移配方（预期，尚未实测）**；不能称已经完成语义对齐的强迁移基线。

## 官方函数链及行号

1. `models/lwdetr.py:845–853`：构造输出数为 `args.num_classes + 1`。`models/weights.py:434–471`：显式配置优先，加载前把不匹配的头调整为 checkpoint 宽度；`:642–644` 实际加载；`:658–661` 再裁回任务宽度。`training/module_model.py:461–481` 的原生训练路径使用同一 loader，然后构建 criterion。因此此处应推导 **2→91→加载 COCO→2**，不凭函数名猜“全随机 reinitialize”。[官方 loader](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/weights.py#L434-L471)，[裁回任务宽度](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/weights.py#L642-L661)，[原生训练入口](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/module_model.py#L461-L481)。

2. `models/lwdetr.py:58–85` 的 `_resize_linear` 对旧 weight/bias 做 repeat 后取前 `num_classes` 行，随后复制进新 Linear；缩小时就是前缀截取。`:232–250` 同时处理 `class_embed` 和每个 `transformer.enc_out_class_embed[g]`。只改 decoder 而遗漏 encoder，不是完整的语义对齐。[官方 resize 与双阶段处理](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/lwdetr.py#L58-L85)。

3. COCO 官方组织的类别元数据明确 `id=9, name=boat, isthing=1`；RF-DETR `assets/coco_classes.py:17` 也映射 `9: boat`。`datasets/coco.py:679–681` 明确预训练 COCO 原始约定为输出槽 k 对应 COCO category k。因此这是 **sparse ID 9 / 张量第 9 行（从零计）**，不能误用 80 类压紧索引中的第 8 行。官方元数据同时表明 id1=person、id90=toothbrush。[COCO 官方类别元数据](https://raw.githubusercontent.com/cocodataset/panopticapi/master/panoptic_coco_categories.json)，[RF-DETR 对应类别表](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/assets/coco_classes.py#L8-L18)，[原始槽位约定](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/datasets/coco.py#L666-L681)。

4. **额外槽不是可随意复制的 softmax background 类。** 对当前 remap 单类，GT 为 label0，模型槽为0/1；`criterion.py:792–816` 用另一个临时 sentinel 表达 unmatched，再删掉 sentinel one-hot 列进入 sigmoid focal，所以模型槽1没有正标签。COCO row0是无正类别标签的空缺槽，row90是 toothbrush，不能把 row90 当作背景初始化。原生 `PostProcess` 在所有 query×class 的 sigmoid 值上取 top100，之后再过滤未映射槽；slot1 仍会参与竞争。[分类目标](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/criterion.py#L792-L816)，[原生 top-k](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/postprocess.py#L130-L153)。

## A/B/C 可审查选择

| 配方 | 分类头与监督 | 是否保留 boat 行 | 公平性和成本边界 |
|---|---|---|---|
| **A 官方默认迁移** | config1，2槽；每个头取源行 `[0,1]`；SAR GT label0，导出 `{0:1}` | 否，目标0继承源空缺0，未映射1继承 person1 | 应保留作官方代码迁移参考；不是论文默认312复现，因为当前配方576及训练设置另有调整。 |
| **B 语义对齐迁移** | config1，2槽；主比较建议各头源行 `[9,1]`，只把目标0换成boat9；unused1与A完全一致 | 是，decoder及所有encoder头的 weight **和 bias** 均须映射 | A/B参数量、结构、监督、排序及预算可相同；是额外工程迁移配方，不能冒称未改官方默认或方法创新。另一个候选 `[9,0]` 用源无正标签槽0初始化unused1，必须单独命名；它同时改了两行，收益不能全归因boat先验。 |
| **C 保留完整 COCO 头** | 显式 config90，91槽不裁；SAR GT必须一致地映射到内部label9，评估再映射 `{9:1}` | 是，保留全部COCO分类行 | 只设config90、仍喂label0并不会训练boat槽。当前单类 canonical exporter 不能直接用于C。91路分类、loss、encoder proposal max及all-class top100竞争均与A/B不同，是另一配方，不是等结构初始化消融。 |

C 的额外分类计算约按91/2扩大，但**整体模型速度不能按45.5倍推算**。对 hidden_dim=D、训练group数G、two-stage开启时，仅分类参数差约为 `89×(D+1)×(G+1)`；实际时间/显存须测。`transformer.py:827、1030–1032` 以所有类别最大 logit 选 encoder proposals，所以保留其他COCO类还会改变query来源，不只是最后过滤。对C若先只筛label9再取top100，又改了原生后处理，必须另记协议；不能与“先原生allclass top100后过滤、不补选”混报。[encoder proposal选择](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/transformer.py#L817-L832)。

## 长训前的最小决策与记录

- 保留现 YAML 冻结；在独立运行计划把它标作A的候选，实际 checkpoint 必须绑定官方字节引用，不能由当前占位文件名推定已加载哪份权重。
- 优先准备 **A 与 B `[9,1]` 同预算对照**，以最小改动检验保留boat行是否有用。`[9,0]` 可作为另外的初始化候选；C暂作为独立诊断，不让额外类别竞争和成本混入主比较。哪个成为后续强基线，应由相同预算的训练/验证证据决定，不能靠语义合理性保证B胜出；不得用测试集选择初始化。
- 每次在 optimizer 建立前记录实际两个槽、dataset GT labels、inverse mapping、全部分类头的源行映射和逐行相等证据；共有非分类权重应相等，PE插值、RGB转换和归一化、query/group及额外空schema补全应一致。A/B须控制构造阶段RNG消耗，例如采用同一官方路径后再做确定性行替换并核RNG；只写相同seed不足以证明后续数据顺序和dropout一致。
- 后续新增模块与其强基线共享选定初始化；若模块方案用B、对照只用A，则初始化是混杂因素。短预算只支持该预算内的判断，正式结论仍需可比长训、重复种子与跨数据集证据。

本核查定位的是一个实际的迁移配置决策，不主张创新或收益。源码确定“默认截掉哪一行”；实际权重加载、head映射、训练合法性和效果仍待独立资格与实验确认。
