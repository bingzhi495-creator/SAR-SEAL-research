# GLDet 代码复核与一项最小空间检验

2026-10-15；普通研究设计，未实现、未训练；未读取实际SOURCE/RAW/SPEC/Q/数据/权重。只复用[已发布原PDF评议](GLDET_PRIMARY_PAPER_REVIEW_AND_SEGMENTATION_TRANSFER_20261014.md)及公开一级来源。N16结论仍以ROOT完成独立全scope资格后的固定主端点为准，本文不预判结果；也不据单预算无增益宣称效用选择总体失效。

**代码现况。** Chang等，GLDet，IEEE TGRS63，2025，DOI[10.1109/TGRS.2025.3559551](https://doi.org/10.1109/TGRS.2025.3559551)。今天直接访问[changhao123/GLDet](https://github.com/changhao123/GLDet)仍只见1次commit和README，原始[README](https://raw.githubusercontent.com/changhao123/GLDet/main/README.md)仅“# AIRSAR”；[GLDet1](https://github.com/changhao123/GLDet1)显示空仓。题名/DOI/作者定向检索没有找到可确认归属的实现。这是有限检索中的“作者代码未核验”，并不证明作者没有其他代码。IEEE网页本次触发机器人验证；发表与方法判断沿用已有正式PDF核查，不把网络访问失败当论文问题。当前不能精确复现GIM分支/DSA实施细节。

**唯一假设。** 若N16固定24主实验不提升，下一项仅检验：对于既定PR的help/harm/noop预测，真实局部native-L邻域值能否补充中心F+e+L。UTILITY scout追加同query、原native H×W网格3×3的八邻点，按固定偏移顺序取c=clamp(L,-8,8)/8；边界坐标复制，保持原中心输入/归一化不变。513→521→32→3共16803参数，比16547多256；共同旧初始化权重复制，新增8列全零，保证两新分支初态等同原scout。它是逐点有坐标的邻域输入，不能把采样点序列当图像。固定PR、C、K、atmost主/exact补充、native scores/全100query均不变，只改utility预测输入。

**先例与差异。** GLDet LGIM在有序backbone feature图上做多分支空间卷积与channel关联；原评议未见显式Sobel或loss梯度前向。这里仅取既有query-logit邻域，不是GIM复现，也不借用其整网mask AP作为预期收益。[PointRend，Kirillov等，CVPR2020](https://arxiv.org/abs/1912.08193)已做自适应点式mask细化；[Mask Transfiner，Ke等，CVPR2022](https://arxiv.org/abs/2111.13673)及其[作者代码](https://github.com/SysCV/transfiner)已在错误易发quadtree节点用transformer纠错。本文不主张首次局部空间精修；具体检验是固定action的效用路由是否受益于同query logit邻域值，未改变corrector或采用quadtree。

**同预算强对照。** 只比较真实八邻与八份中心c复制，二者均16803参数、相同四张量初态、样本/私有顺序、GT匹配与共同背景empty代理语义、importance-weighted CE均值、AdamW及24轮图更新预算；控制仍执行同样的邻域索引读取，再复制中心，计算列清。常值邻域两臂应逐点等效；中心复制没有匹配邻值分布/协方差，正收益也可能来自局部平均置信度，不能单凭此对照归因空间排列；若要主张排列机制，另需保留八值而破坏相对位置的机制对照。原N16无邻域结果与固定strong E24 PR AP68.09938494745545保留参考；不能只超过较弱selector就称有效。预先固定主AP与停止门，不选中间epoch；若初筛正收益，才按既定重复规则确认。两分支的新增采样/I/O/训练/部署成本均单列，256 MAC/候选点为首层新增乘加量，不等于整系统成本。

**缓存与否证。** FIT与DEV来源必须分离：现有187张DEV all100-L GPU评价捕获只可用于该固定DEV评价的邻域取值/重放，不能派生741张FIT训练邻域或用于选型。FIT训练源可用性待ROOT独立核验：中心点packet本身不足以证明已保存完整同query L；若有合格FIT全L，须绑定其独立来源并按原坐标生成新空间薄cache；若无，须由冻结教师和既定FIT坐标重新生成邻域，教师前向、L渲染、采样、I/O与资格成本全部计入。无须F邻域或SAR图像patch缓存，不等于无需新增FIT邻域cache；旧冻结packet不补列。八值FP32新增32字节/采样点；DEV全C推理按2048点流式，八值块65536字节，FIT/DEV各自native-L源与元数据成本另计，均须CPU同输入重放并核原形状、padding与采样映射。不用GT选择邻域，不从DEV结果调整训练输入。若真实八邻在重复的固定预算下不胜同容量中心复制，则否定“这项3×3 logit邻域输入改善效用路由”的局部假设；若CE或动作分类改善但完整all100 AP不改善，仍判实际部署无收益，并保留像素效用≠IoU/AP、97%noop/背景proxy偏差、固定PR动作不足等反证。该结果不否定GLDet、更不否定所有局部F信息。即使AP正增益，也需胜强参考与重复/对照证据；不追凑3点，不提前称创新。

未科学执行；本文件由ROOT统一审核和发布，本子任务没有Git提交。


## 2026-10-15 合格主结果后的实际取舍

N16完整主门STOP（Utility67.666732、同预算Error67.895932、强参考68.099385）；因此上文UTILITY邻域仅保留为历史候选，下一最小检验改为较强ERROR真实八邻与同容量中心复制。Root已核FIT合格清单741份全query native-L，尚未做新邻域提取/CPU坐标资格。3×3输入仍非GLDet复现或已成立创新；无需新teacher只是本次缓存条件，不是一般成本优势。[当前一组公平计划](N17_LOCAL_NATIVE_LOGIT_ERROR_SELECTOR_EXPERIMENT_PLAN_20261015.md)、[完整合格结果](N16_FULLDEV_QUALIFIED_FIXED24_RESULT_AND_STOP_20261015.md)。
