# N25 结果到主张

状态：provisional — no separate integrity audit run；[pending Codex review]。Codex MCP评价器本轮未提供，未调用成功；同团队静态/解释审阅不等于跨模型审计。以下由Root依据真实结果作范围判断，流程继续。

- claim_supported：partial。固定R mask下，预锁GT乘积评分有+5.297021 AP点变化；支持继续检验质量预测可学习性。
- what_results_support：187图/380GT/五组全部100query上，评分替换可以在几何不变时改变AP。R原68.099385、GT乘积73.396406、GT质量单独3.380706；完整S0在本规则中仍有重要排序信息。
- what_results_dont_support：没有可部署新方法收益、训练稳定性、数学AP上界、SAR专属机制、结构创新、跨集泛化或TGRS就绪证明。
- missing_evidence：从FIT可获得的真实匹配监督学习质量；固定端点真实头评价；重复种子、强普通替代、机制与成本、独立数据集。
- suggested_claim_revision：从“新结构已提高AP”收缩为“现有固定mask存在GT辅助乘积评分机会，需检验可学习性”。
- next_experiments_needed：唯一下一项为冻结R的query–mask条件IoU回归基线；准确生成FIT完整R输入与匹配IoU，主评分S0×qhat。质量头本身为已知方法适配，不宣称创新。
- confidence：对本轮固定评分事实高；对真实头能否取得收益尚无证据。
- integrity_status：unavailable。原任务actual0、fresh RAW-first及Root接收均完成；五原COCO12校准，oracle几何逐位核对，93500个query质量/乘积另据N24矩阵核验。Root未独立重算COCO。

接收器状态保存变量重名已恢复，保留原科学产物和失败记录，没有科学重跑；不计为方法失效。N20/N23停止不变，不用oracle补救原方法结果。
