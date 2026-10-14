# 正式 TEST 与跨数据集证据准备度（2026-10-14）

本页仅定位独立发布副本已有的数据协议、元数据核查和历史审阅；未读取实际 SCI/SOURCE/RAW、影像、GT、权重或缓存，未运行科学代码、重新核验数据包或新增联网调查。既有审计直接复用，不能把准备文档当作实际数据或评价资格。

**测试暴露已有记录，不能默认恢复盲测身份。** [历史确认实验方案第42行](../paper_tgrs/revision_tgrs_20260905/CONFIRMATORY_EXPERIMENTS_CN.md)明确记录：HRSID test 已用于规则选择，PSeg-SSDD test 结果也已公开查看。[2026-10-11论文源码审阅](../docs/paper_review_20261011.md)进一步定位旧 HRSID TEST 融合超参数选择、best-on-test 及近岸选点；[历史修订报告](../paper_tgrs/revision_tgrs_20260905/REVISION_REPORT_CN.md)同样明确测试参与参数与方案选择。HRSID 旧 TEST 不能通过重命名恢复 blind。SSDD **结果暴露不等于已确认具体参数选型**：本次未找到完整的 SSDD 测试接触清单、选型参数/模型及其时间关系，不能由这个缺口推断“没有历史选型”。

**PSeg-SSDD：官方接口和转换准备已有，正式入场核验尚不完整。** [DATA_ACCESS](../docs/DATA_ACCESS.md)记录 [Official-SSDD作者入口](https://github.com/TianwenZhang0825/Official-SSDD)及[数据论文](https://www.mdpi.com/2072-4292/13/18/3690)：PSeg 真实多边形与 BBox/RBox 版本不同，官方 train928/test232，内部 FIT741/DEV187 不等于 TEST232。已有 `protocol/pseg_to_coco.py` 转换入口，但会跳过缺失 XML/无有效多边形对象，不能凭转换完成认定数量与真值完整。仓库 Apache-2.0 与外链影像压缩包逐文件授权范围未被完整映射；当前发布官方获取入口，不声明原图镜像已获核实授权。

**HRSID：已有定向元数据审计，未达到真实实例数据入场资格。** 复用 [外部协议准备度](../docs/HRSID_external_protocol_readiness_20261012.md)及[官方协议/许可补核（含10月14日更新）](../research_round_20260925_global/HRSID_OFFICIAL_PROTOCOL_AND_RELEASE_SCOPE_CHECK_20261013.md)。已有 [作者仓库](https://github.com/chaozhong2010/HRSID)、[原论文DOI](https://doi.org/10.1109/ACCESS.2020.3005861)及 JPG/PNG/近岸标记/400纯背景包入口；65/35 是裁片图像数比例，不证明母场景互斥。固定作者仓库当时只列训练 JSON，官方包内 TEST JSON、真实 segmentation 编码/解码、图像 ID 对应与版本一致性尚未闭合。场景映射、跨划分重叠和同船重复仍待实际核验；不能据此宣称实际泄漏或独立地理泛化。GPL-3.0/论文CC BY4.0 对外链影像和全量标注的覆盖仍未明确；既有审计记录 [许可issue #3](https://github.com/chaozhong2010/HRSID/issues/3)当时未获作者答复，不等于没有许可或禁止使用。

下一正式验证的最短依赖链，沿用[RESEARCH_GOAL的跨集与正式测试要求](../RESEARCH_GOAL.md)：

1. 完成当前开发评价/独立核验后冻结候选、强对照、权重、推理/选点/评分和停止规则；登记内部 FIT 模型直接评价与官方训练集预锁重训的不同身份，双方采用同一路线。
2. 补齐已有 TEST 使用台账，至少记录数据/图像范围、曾看结果、参数/模型选择用途。未来固定 recipe 在历史 HRSID benchmark 上的零调参 external 结果可报告，但必须披露暴露历史及切片级限制；它不同于真正未参与研究选型的独立确认性证据。独立确认需新未用于选择的场景/集合或可审计的开发—确认方案；重分旧数据不自动清除历史暴露或旧模型训练接触。
3. 由科学 owner 核验选定官方包版本/校验和、完整 train/test 清单、真实实例 mask、图像 ID/路径、转换数量守恒、类别/空图/忽略规则及许可范围。缺合格测试实例真值则停止 mask AP；JPG/PNG、纯背景补充和自建场景协议不得静默混合。
4. 冻结 HRSID 外部路线（当前任务书优先源数据训练后零调参迁移）；实际核验母场景/裁片重叠、同源和重复。未确认场景独立时仅称官方切片协议上的受限外部评价，不称独立采集/地理泛化。原结果和负迁移保留，不能看 TEST 后换阈值、轮次或数据集选优。

本页没有新增正式 TEST、外部 AP、未见测试证明或许可认证；未闭合项目保持缺失状态。
