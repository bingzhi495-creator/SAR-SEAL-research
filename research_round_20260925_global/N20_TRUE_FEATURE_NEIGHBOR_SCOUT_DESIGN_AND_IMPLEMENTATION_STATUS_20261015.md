# N20：真实 F256 邻域输入的最小公平检验

状态：**UNBOUND_DESIGN / SOURCE_ONLY_SYNTHETIC_PASS**。2026-10-15。结构原型及2项CPU合成行为检查已完成；新邻域缓存、科学入口、训练、真实数值资格和DEV评价均未执行。不得把本文当作已启动训练或涨点报告。

## 来自 GLDet 的判断

[GLDet 原文综述](GLDET_PRIMARY_PAPER_REVIEW_AND_SEGMENTATION_TRANSFER_20261014.md)已核20页原PDF和主要表格；本轮再核所附PDF首页，题名/作者/DOI一致。GLDet 在有序完整空间特征上联合全局扫描、局部卷积和方向通道关联，并含实例分割头；不是把三个名字加到点向量上。其SSDD/HRSID mask AP71.3/58.6来自整网500轮，不与本项目内部DEV直接比高低。[原文DOI](https://doi.org/10.1109/TGRS.2025.3559551)。

本轮只检验真实空间特征是否提供旧输入没有的有效信息。N17用同query的邻域native-L，主结果为负；它不等同真实256维特征邻域。N19固定方向评分同样触发STOP，不再扫描其阈值、权重或旧种子。普通depthwise残差适配是已知结构，本原型是信息基线，不是已确认的核心创新，也不是GLDet GIM/DSA复现。

## 唯一假设及实际作用位置

同初始化、监督、参数量和固定24轮下，真实F邻域能使ERROR scout的全量mask AP超过中心复制控制和强参考R。N18的固定条件人口显示尚有可修动作漏选，但不保证全量AP收益。**本轮仍只改善路由观察，原PR修正器保持冻结，不会因此提高其可修动作上界。** 若继续缺乏收益，应重新检验修正器建模假设，不能把更多评分变体当进展。

## 已核坐标与拟定取数

Root 在原任务终态及纯RAW前缀实际0之后，读取已绑定原renderer及DEV包装；读取前后物理哈希一致。两端均按16通道分组，FP32 bilinear resize到800×800、按actual_img_shape从左上裁剪，再bilinear到original_shape，两次align_corners=False。没有point_sample，也不能把两次插值合成一次并声称bitexact。

最小取数定义为**渲染后原图整数±1像素的九邻**，按dy=-1,0,1内dx=-1,0,1排序，中心slot4；x/y分别clamp到本图边界，replicate，不跨行/图。它不是原生F格点的±1，也不检验所有邻距或尺度。实际取数规则仍须进入新的来源绑定并通过真实核验，不能仅凭此静态源码结论启动训练。

沿用旧unique_flat_indices、sample_to_unique和所有query/点顺序。中心输入逐字节复用旧sampled_F_unique，新冻结teacher capture的offset0必须实核等于旧中心；不一致时记录失败，不能替换旧输入或放宽为近似中心。旧FIT只有稀疏中心F与native-L，没有可直接供八邻采样的完整F载体，需新capture，不称免费缓存复用。CPU资格使用实际GPU输入，不能用近似CPU重渲染冒充。

## 两臂与预算候选

Fprime = Fcenter + sum_offset(w[offset,channel] * Fneighbor[offset,channel])；w为9×256全零。其后为Fprime和e各自non-affine LN(eps1e-5)，拼原L/8，513→32→ReLU→3。**两臂各18,851参数**；LOCAL输入真实九邻，CENTER输入九份相同中心。共同MLP初态、完整state复制，初始化函数相同。中心复制只匹配名义容量，不匹配输入秩、协方差或有效优化条件。

继承FIT741/内部DEV187、原每图4096点及ERROR标签/importance、原AdamW与图序/点序，固定新24轮，每臂17,784更新；teacher/PR/C/K/query/class/S0/COCO不变，主E24 atmost，E8/exact不救主门。首次初始化候选202610150201，后续202/203；必须在真实数据执行前统一锁定，不自动补旧种子。

主Delta = AP_LOCAL - max(AP_CENTER, 68.09938494745545)，沿总目标规则：Delta≤0停止当前配方；0<Delta<1最多补第二组，只有前两组均胜对应CENTER且meanDelta≥1才允许第三组；首组Delta≥1再补202/203。所有数字用未舍入值。当前尚无N20 AP，不改变既有STOP。正结果先支持当前共享teacher和内部DEV下的邻域信息效用；再预锁逐点八邻位置置乱、普通替代结构、重复、正式TEST和跨集，才讨论机制和创新。

## 费用及下一实际执行

旧合格元数据：741图、3,035,136样本、1,786,820个按图去重中心。若每个unique中心保存八邻FP32，新增原张量14,637,629,440B；按全部样本物化九位置则27,971,813,376B。可进一步把邻域位置合并成每图唯一坐标集，旧中心值复用，只存新增坐标F与索引映射；这仍是未执行的缓存设计，不能预报实际磁盘或速度。上述原张量数字不含header、索引、旧缓存、资格读回或备份。

下一项实际工作是新冻结teacher邻域capture及独立坐标/实际输入核验，然后配对24轮训练；不再重复文献综述或N19读回。每个真实leaf沿用单执行者、CPU01/两线程、GPU/内存上限、1800秒及无自动重试规则，失败保留。任何持久载体与科学入口须新建，冻结原件不改写。

## 已实现与验证范围

- [原型](code/N20_FEATURE_ADAPTER_UNBOUND_20261015_01.py)：显式传入neighbors；不隐造中心复制，CPU FP32构造、不读文件、不设设备或全局种子，恢复外部CPU RNG。
- [行为测试](code/N20_FEATURE_ADAPTER_TEST_UNBOUND_20261015_01.py)：先实际RED（缺模块），再GREEN；最终2项检查实际退出0，2.188秒，CPU01/两线程/CUDA为空。零权重下两种不同真实合成网格邻域及中心复制均逐byte保持旧输出；非中心权重能得到有限非零梯度，单一邻居变化的最终logits与手算有效F的独立旧基线逐byte一致。
- 合成检验不验证真实F邻域来源、GPU数值、真实训练、AP、部署耗时、泛化或TGRS就绪。跨模型审阅工具仍不可用；同模型独立代理意见不冒充跨模型审计。
