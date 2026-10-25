# DEV187实际标注来源：独立元数据与单规则补核

2026-10-25。Root从已关闭评价输入取得字节精确副本，独立只读审阅者核对118319字节与既有SHA256，并解析实际JSON：187图、380个唯一实例、单类ship、全部iscrowd=0，每实例一个非空polygon，无RLE。该检查不做mask解码、评分或影像判读。

本机候选源COCO训练成员包含928图/2041 polygon。按精确file_name和类别名映射，DEV187的文件名与尺寸全部对应，图像/标注ID不共用命名空间；不能按裸ID判断同源。联合segmentation/bbox/area/iscrowd在转换前仅14/380相同，366不同。

Root随后在所有数值叶关闭的窗口实际运行确定性stdlib补核，父/子工具退出0。输入GT、源COCO和既有转换规则记录均先核字节与SHA，只测试一条既有规则：每个坐标int()截断，bbox为坐标极值差，面积为连续polygon鞋带面积绝对值/2，不试舍入、顶点重排或其他规则。结果**380/380联合几何对象相同、380个唯一对应、0多余/缺失**。没有生成或改写GT，也没有mask解码或AP计算。

这证明本机候选与冻结内部GT之间的确定性几何等价，不证明过去实际执行了哪个converter，不认证官方完整归档身份、版本、场景独立或标注质量。记录为polygon实例，不能据此假定无任何更早转换历史，也没有证据将其称为本轮模型生成的伪GT。

官方[SSDD论文](https://www.mdpi.com/2072-4292/13/18/3690)和[固定仓库说明](https://github.com/TianwenZhang0825/Official-SSDD/tree/937ccbec2ee967b7263e201aa085f25c4aecaebd)支持PSeg/LabelMe polygon及VOC/COCO形式；本轮论文直开受429限制，部分核验依赖官方域索引。仓库说明未提供可闭合该外链PSeg整包的逐文件校验；代码许可证不能自动认证外链归档每项授权。因此整包身份和外链文件许可继续分列WARN，原始数据、全量标注不上传。

独立元数据检索曾单点解析一个非COCO的LabelMe JSON，接触其imageData字符串但未打印/解码；未再读取该类候选。核心同源结论仅依据上述两份COCO及确定规则。

“冻结内部GT未改动”与“上游浮点标注未经转换”是两件事。整数/浮点敏感性已在首次FULL端点AP前声明，仍待按[原政策](U57_PROSPECTIVE_ANNOTATION_SENSITIVITY_POLICY_20261025.md)执行；不因本次补核更改主GT、挑更好分数或重新命名为官方TEST。

[安全元数据结果](results/U57_DEV187_GT_PROVENANCE_SUPPLEMENT_20261025.json)不含原标注全文、私有数据路径或逐实例身份。
