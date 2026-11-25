# Umbra 补充队列双盲判读包

24 个已锁定的独立采集全部通过原生窗口提取及独立读回。本次在封存名单上为 A、B 两位判读者各创建一份离线判读包。每包有 24 幅完整的 1024 × 1024 原生灰度 PNG，保留补充队列的匿名编号和原名单顺序。两包在技术核验时通过逐图字节哈希、ZIP 校验、匿名名单一致性和界面角色校验。打包进程实际退出 0，回执末行校验通过，耗时约 1.19 秒。六项合成测试和合成观察进程验证通过。[判读包流程](code/UMBRA_SUPPLEMENT_REVIEW_PACKETS_20261126/README.md) · [实际汇总回执](results/UMBRA_SUPPLEMENT_REVIEW_PACKETS_SAFE_20261126.json)

每位判读者在本机打开自己的离线页面，查看同一批原生灰度图。完成后分别保存并私下回传 `SAR_ship_supplement_review_A.json` 和 `SAR_ship_supplement_review_B.json` 的原文件。两包内均无标注、模型输出和地理坐标。真实回传及评分尚未执行。本次发布范围为代码、协议和安全汇总。

