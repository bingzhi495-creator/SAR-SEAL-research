# NORA 插图与重建

当前公开全文使用本目录的 16 张重绘图。作者审阅版含 17 张图，五张方法图和五组定性图嵌入核验过的 SAR 像素与掩膜。公开定性图使用实际结果的空间汇总，公开方法图使用可编辑示意。所有图保留分面、字体、坐标、公式或线条的可编辑元素。实验值、预算和选例规则沿用已核验版本。

| 图组 | 所含图件 | 生成入口 |
| --- | --- | --- |
| 方法 | 总览、ROH、等计数选择、固定总数分配、Phoenix 对照 | [总览与 ROH](draw_nora_native_flow.py) · [三种写入控制](draw_method_controls.py) |
| 统计与消融 | 转移、参照差异、分配、DEV705、SAR-SSP 汇总、两侧消融、支持与计数 | [统计绘图](plot_nora_evidence.py) |
| 定性 | Instance-RSDD 选择、Gaofen-3、PSeg-SSDD、SegRefiner | [案例重排](rebuild_case_plates.py) |

使用公开稿目录下的旧图件与发布的实验汇总，从本目录运行以下命令并另存输出。定性 PDF 导出使用 Edge 或对应的 Chromium 可执行文件。

~~~
python draw_nora_native_flow.py --output rebuild
python draw_method_controls.py --output rebuild
python plot_nora_evidence.py --paper ../.. --output rebuild
python rebuild_case_plates.py --public-source ../cvpr_20261126 --work rebuilt_cases
~~~

暖白与鞍棕用于背景和层级。橙赭标记主流程或放大定位，矿绿和砖红配合增删符号。酒红用于需要分开的参考。版面说明与图注见[构图文字稿](../../style_review/NORA_METHOD_FIGURE_STORYBOARD_20261126.md)和[GLDet 逐页研读](../../style_review/GLDET_FULL_STYLE_REVIEW_20261126.md)。


[五页 Draw.io 可编辑工程](NORA_method_figures.drawio)可在 draw.io 的本地文件入口打开。五张方法图的方片、连线和文字可单独编辑。需要从已校核的示意图重建时，在本目录执行 `python build_NORA_method_figures.py --source . --output rebuilt_methods.drawio`。
