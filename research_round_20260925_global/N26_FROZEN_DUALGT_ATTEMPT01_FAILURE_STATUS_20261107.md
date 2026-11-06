# N26冻结两GT诊断首次实际调用：输入检查失败，未产生AP

2026-11-07。原工具、SSH、parent子进程与worker均实际退出1；后验核验源不变、资源满足、实际exec握手及全部进程组闭合，记为 **FAILED_CLOSED_UNQUALIFIED**。原调用很短，工具没有分配持续session；记录None及真实终态，不编造句柄。后续元数据0不升级原失败。

实际读并解析了两份GT和共同R/三个质量头预测；GT非几何身份检查已通过。`validate_prediction_pair` 在首条记录读取 `bbox` 时抛出KeyError。冻结预测采用segmentation-only结果，原检查错误地要求了可选bbox字段。三份score audit仍未解析；四个整数COCO案例均未开始、AP均null，浮点阶段未开始。这是实现检查错误，不能解释为模型或排序方法无效。

[官方COCO loadRes的固定原始实现](https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/coco.py)提供segmentation结果分支，并可从RLE生成缺失bbox；本次薄检查应允许字段缺省，存在时仍校验原四元组。这个外部源代码事实不替代项目实际canonical运行验证。

旧源码、旧输入、grant、失败产物均保留且冻结。另建最小successor，仅修bbox存在性检查；不补造bbox，不改变原分数、RLE、顺序、逐记录非score完全一致约束、三个seed、固定四整数AP的1e−8百分点门槛或COCO数学。新候选先做真实缺字段合成回归和有限审阅，再保存普通预绑定源码并同步；未来只能以新grant、新目录、单次调用执行。当前未执行新候选，不自动重试旧调用。

主PLAIN/FULL已严格接收29952/37050，诊断credit为0；原调用闭合后已恢复下一29952→30208训练。没有新增AP、跨GT稳健性、外部SOTA或TGRS就绪结论。原始预测/GT/完整执行与资格记录均私有，仅发布此安全摘要。
