# 固定 E24 强参考导出的 COCO 重算协议

日期：2026-10-12。只补齐旧固定参考的当前执行凭据，不更换参考、方法、评分或主指标。本次不是盲测：A主AP0.646861684933002、B主AP0.6679706764420889、旧native0 E24历史AP0.6707105663749372均已暴露。ROOT已核对同DEV、全部187图像身份、helper/initializer、E24权重及COCO源码。旧CPU实际0记在历史状态，但旧资格没有本轮同等级的独立CPU owner RAW。

任务：使用封闭官方COCO helper复算预先指定的旧native0 E24完整固定GPU导出。只算此参考，不选择其他旧端点。保留187图/380原GT/18700queries、原始顺序、空预测、crowd/ignore；不更改分数/阈值/NMS。

SOURCE/SPEC/CPU admission/输出均使用新文件；不修改旧产物。本作业13个现有RAW严格0先于任何hash和科学库导入。输入按绝对路径/bytes/SHA绑定。单一ROOT owner，CPU01/CUDA空/四线程环境2/wholeGPU8192MiB/RAM16GiB及disk20GiB下限/1800秒上限/无自动重试。

验收：原CPU实际terminal0，ROOT all14 RAW先于新Q读取/hash；新Q的SOURCE/SPEC/CPU admission/phase binding与全部输入匹配。比较全部12项COCO指标。差值<=1e-12只表示固定导出指标复算一致，不宣称重放native forward/训练轨迹、重建几何或重算遗漏的logits质量。若不一致，保留两套并定位原因，不能选高值。

决定：用新复算E24 AP与本轮锁定A_native_official、B_posterior_official判断，保持原0.010/小正收益/非正收益规则，不替换最高budget列。E24训练预算不同，只作强参考；同预算主对照仍是A/B。门槛未满足则保留B>A事实并结束该配方扩展，依据新误差锁定一条有限新假设检验。资源或执行异常不能当作方法失败。
