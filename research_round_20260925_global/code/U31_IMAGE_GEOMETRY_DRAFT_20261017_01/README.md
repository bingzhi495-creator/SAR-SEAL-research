# 未绑定研发草稿，未完成数值资格

2026-10-17。原图RGB uint8/255→实际整数内容→右下Pad800→分支400；残差400→800→按原整数内容裁剪→原尺寸。双线性align_corners=False、antialias=False。原比例的整数round-half-up严格核对，不推导另一个半尺寸裁剪框。唯一4/8非方几何测试使用解析期望，Root先观察缺模块RED（4cfb45，无Torch），实现后仅AST，尚无GREEN、真实影像/400渲染资格。

代码不包含本机spec/资格/原数据/权重。主线范围见[完整方案](../../../refine-logs/UNETPP_FULL_METHOD_20261017_01/EXPERIMENT_PLAN.md)。
