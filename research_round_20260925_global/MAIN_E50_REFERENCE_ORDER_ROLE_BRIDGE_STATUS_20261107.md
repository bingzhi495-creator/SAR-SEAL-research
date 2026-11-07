# 主E50原参考数据：人口、顺序与角色桥已实际核验

2026-11-07。Root有限数据I/O原工具/SSH实际退出0。此任务为未来主PLAIN/FULL固定端点评价准备，读取已明确登记的6份原输入及1份仅作比较的旧R文件，不读取模型、checkpoint或科学源码，不导入科学库，不解码mask或计算AP。

两份原context的187个image IDs均为唯一整数，全序完全相同，并与原GT.images的顺序一致；当前context的187行也按该序排列。GT保持187图/380个合法实例、单类别1。原native、主R及比较R各18700条，每图恰100条，类别、图像身份、compressed-RLE尺寸与原GT对应，按图的原块序与context相同。7份原数据字节在检查前后不变。

原context→legacy context、两者共同native/R/legacy result角色，以及legacy→原DEV的字节身份均闭合；六个原登记refs保持，既有资格没有被这次数据检查重新授予。历史native/R完整七键COCO字典与12项stats只检查了保存格式，没有重新计算或复现指标。

发现并消除了一个容易误用的点：主评价登记R与N26诊断共同R的原字节refs不同，但18700条解析记录及顺序、全部非score字段、score类型和值完全相同。它们的原引用仍分别保留，不能因为语义相同就以新R替换主登记R，也不能放宽原byte pin。最终评价仍须沿原角色完整复现COCO12。

这次只证明数据层的人口/尺寸/原顺序与角色。context的query字段没有由“直接整数list(range(100))”探针建立该表示；不把它写成query无效，也不冒称隐式query身份已证明。后续真实U66/U67仍必须按原合同核查询映射、零残差校准、同一E50端点和四片完整输出。每图100条记录本身不替代这些证明。

实际解析了含annotations的GT正文和预测JSON，因此不能声称本流程从未接触GT。数据I/O约1.500秒，远端child约0.550秒、RSS峰值83092KiB；CPU2/3、nice19、512MiB地址空间上限、120秒期限。该RSS不当作独立整组或整机峰值。原输入、完整标注、预测、位置与refs均不发布。

当前训练、原整数主协议、初始化、目标37050和重复/停止规则不改。E50权重尚未取得、四片预测/真实ABI/完整指标及外部Root数值接收仍待，AP=False、TEST=False、训练credit0。原生/R/N26的既有AP不因这次数据核验增加新数值。详见[安全聚合](results/MAIN_E50_REFERENCE_ORDER_ROLE_BRIDGE_PUBLIC_SAFE_20261107.json)。
