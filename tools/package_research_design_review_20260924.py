"""Build a small, internal research addendum without source imagery or GT.

This is a review package, not a submission package.  It intentionally omits
the blinded image packs, private reviewer mappings, E COCO annotations,
checkpoints, predictions, credentials, and the unreleased manuscript PDF.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAME = "RESEARCH_DESIGN_REVIEW_20260924"
MANUSCRIPTS = ROOT / "research_round_20260920" / "manuscript"

EXTRA = [
    "TGRS_RELEASE_INDEX_20260920.md",
    "tools/package_research_design_review_20260924.py",
    "carrier/mask_tta.py",
    "carrier/same_pool_trace.py",
    "carrier/tests/cpu_test_same_pool_trace.py",
    "research_round_20260920/automation/run_same_pool_trace_smoke_20260924.py",
    "research_round_20260920/automation/verify_same_pool_trace_smoke_20260924.py",
    "research_round_20260920/automation/run_same_pool_top1_full_20260924.py",
    "research_round_20260920/automation/verify_same_pool_top1_full_20260924.py",
    "research_round_20260920/automation/evaluate_same_pool_top1_full_20260924.py",
    "research_round_20260920/automation/evaluate_same_pool_top1_full_20260925_v2.py",
    "research_round_20260920/automation/run_top1_bpr_historical_20260925.py",
    "research_round_20260920/automation/iceye_cog_window_pilot_20260924.py",
    "research_round_20260920/automation/iceye_csi_window_pilot_20260925.py",
    "research_round_20260920/automation/iceye_static_registration_gate0_20260925.py",
    "research_round_20260920/automation/iceye_wpi_port_screen_20260925.py",
    "research_round_20260920/automation/iceye_wpi_non_dwell_slc_screen_20260925.py",
    "research_round_20260920/automation/alternative_multiview_sar_catalog_20260925.py",
    "research_round_20260920/automation/umbra_sicd_gate0_20260925.py",
    "research_round_20260920/automation/umbra_sicd_full_axis_gate0_20260925.py",
    "research_round_20260920/automation/umbra_sicd_gec_registration_gate0_20260925.py",
    "research_round_20260920/automation/umbra_crossport_numeric_20260925.py",
    "research_round_20260920/automation/umbra_antwerp_rank0_full_axis_20260925.py",
    "research_round_20260920/automation/umbra_aoi_metadata_lock_20260925.py",
    "research_round_20260920/automation/umbra_gec_aoi_validity_pilot_20260925.py",
    "research_round_20260920/automation/umbra_gec_aoi_train_preview_20260925.py",
    "research_round_20260920/automation/umbra_annotation_semantic_validator_20260925.py",
    "research_round_20260920/automation/test_umbra_annotation_semantic_validator_20260925.py",
    "research_round_20260920/automation/umbra_coco_crowd_semantics_smoke_20260925.py",
    "research_round_20260920/automation/umbra_train_rank0_aoi_sicd_projection_20260925.py",
    "research_round_20260920/automation/umbra_train_rank0_aoi_sicd_projection_attempt1_pixel_range_violation_20260925.py",
    "research_round_20260920/automation/umbra_aoi_v2_vector_screen_20260925.py",
    "research_round_20260920/automation/umbra_train_rank0_native_annotation_kit_20260925.py",
    "research_round_20260920/automation/umbra_train_rank0_native_annotation_kit_resume_v2_20260925.py",
    "research_round_20260920/automation/umbra_gate0_public_lidar_tile_audit_20260925.py",
    "research_round_20260920/automation/audit_umbra_five_strip_area_gate0_20260925.py",
    "research_round_20260920/automation/revise_umbra_three_port_gate0_v2_20260925.py",
    "research_round_20260920/automation/revise_umbra_three_port_gate0_v3_20260925.py",
    "research_round_20260920/automation/umbra_coco_record_export_20260925.py",
    "research_round_20260920/automation/test_umbra_coco_record_export_synthetic_20260925.py",
    "tools/audit_umbra_port_catalog_20260925.py",
    "tools/lock_umbra_port_manifest_20260925.py",
    "research_round_20260920/automation/acquisition_benchmark_20260924/audit_exposure_dose_v1.py",
    "research_round_20260920/automation/acquisition_benchmark_20260924/independent_exposure_dose_recalc_20260924.py",
    "research_round_20260920/automation/acquisition_benchmark_20260924/audit_exposure_dose_union_v2.py",
    "research_round_20260920/automation/instance_rsdd_public_probe_20260924.py",
    "research_round_20260920/automation/slc_product_catalog_20260924.py",
    "research_round_20260920/automation/slc_select_8_products_20260924.py",
    "research_round_20260920/automation/slc_pilot_burst_catalog_20260924.py",
    "research_round_20260920/automation/same_pool_top1_audit_20260924.py",
    "research_round_20260920/automation/iceye_open_catalog_audit_20260924.py",
    "research_round_20260920/automation/iceye_candidate_metadata_20260924.py",
    "research_round_20260920/results/acquisition_benchmark_20260924/manifest.json",
    "research_round_20260920/results/acquisition_benchmark_20260924/pairs.csv",
    "research_round_20260920/results/acquisition_benchmark_20260924/replacement.csv",
    "research_round_20260920/results/acquisition_benchmark_20260924/exposure_dose_v1.json",
    "research_round_20260920/results/acquisition_benchmark_20260924/exposure_dose_union_v2.json",
    "research_round_20260920/results/ueikap_shipmask_preflight_20260924/SLC_PRODUCT_CATALOG_METADATA_20260924.json",
    "research_round_20260920/results/ueikap_shipmask_preflight_20260924/SLC_EIGHT_PRODUCT_CANDIDATES_20260924.json",
    "research_round_20260920/results/ueikap_shipmask_preflight_20260924/SLC_PILOT_BURST_METADATA_20260924.json",
    "research_round_20260920/results/same_pool_top1_audit_20260924.json",
    "research_round_20260920/results/same_pool_trace_smoke_20260924/verification.json",
    "research_round_20260920/results/same_pool_top1_full_20260924/pre_ap_action_lock.json",
    "research_round_20260920/results/same_pool_top1_full_20260924/run_identity.json",
    "research_round_20260920/results/same_pool_top1_full_20260924/run_summary.json",
    "research_round_20260920/results/same_pool_top1_full_20260924/verification.json",
    "research_round_20260920/results/same_pool_top1_full_20260924/evaluator_v2_amendment.json",
    "research_round_20260920/results/same_pool_top1_full_20260924/evaluator_v2_proof.json",
    "research_round_20260920/results/same_pool_top1_full_20260924/ap_diagnostic_v2.json",
    "research_round_20260920/results/top1_bpr_historical_20260925/input_identity.json",
    "research_round_20260920/results/top1_bpr_historical_20260925/diagnostic.json",
    "research_round_20260920/results/iceye_open_catalog_20260924/stac_metadata_audit.json",
    "research_round_20260920/results/iceye_open_catalog_20260924/fixed_candidate_metadata.json",
    "research_round_20260920/results/iceye_open_catalog_20260924/cog_window_pilot/window_pilot.json",
    "research_round_20260920/results/iceye_open_catalog_20260924/csi_window_pilot/csi_window_pilot.json",
    "research_round_20260920/results/iceye_open_catalog_20260924/static_registration_gate0/roi_read.json",
    "research_round_20260920/results/iceye_open_catalog_20260924/wpi_port_screen/wpi_screen_20260925.json",
    "research_round_20260920/results/iceye_open_catalog_20260924/wpi_port_screen/non_dwell_slc_wpi_screen_20260925.json",
    "research_round_20260920/results/alternative_multiview_sar_data_20260925/catalog_metadata_audit.json",
    "research_round_20260920/results/alternative_multiview_sar_data_20260925/capella_maritime_item_metadata.json",
    "research_round_20260920/results/alternative_multiview_sar_data_20260925/umbra_csi_cphd_candidate_sizes.json",
    "research_round_20260920/results/umbra_port_catalog_independent_audit_20260925.json",
    "research_round_20260920/results/umbra_sicd_gate0_20260925/probe.json",
    "research_round_20260920/results/umbra_sicd_full_axis_gate0_20260925/metrics.json",
    "research_round_20260920/results/umbra_sicd_gec_registration_gate0_20260925/metrics.json",
    "research_round_20260920/results/umbra_port_prospective_manifest_20260925.json",
    "research_round_20260920/results/umbra_crossport_numeric_20260925/metrics.json",
    "research_round_20260920/results/umbra_crossport_numeric_20260925/metrics_initial_parser_stop.json",
    "research_round_20260920/results/umbra_crossport_numeric_20260925/metrics_second_jebel_head_timeout.json",
    "research_round_20260920/results/umbra_antwerp_rank0_full_axis_20260925.json",
    "research_round_20260920/results/umbra_aoi_stac_geometry_snapshot_20260925.json",
    "research_round_20260920/results/umbra_port_aoi_metadata_lock_20260925.json",
    "research_round_20260920/results/umbra_gec_aoi_validity_pilot_20260925.json",
    "research_round_20260920/results/umbra_annotation_record_schema_20260925.json",
    "research_round_20260920/results/umbra_coco_crowd_semantics_smoke_20260925.json",
    "research_round_20260920/results/umbra_train_rank0_aoi_sicd_projection_20260925.json",
    "research_round_20260920/results/umbra_train_rank0_aoi_sicd_projection_attempt1_pixel_range_violation_20260925.json",
    "research_round_20260920/results/umbra_aoi_v2_osm_vector_screen_20260925/screen.json",
    "research_round_20260920/results/umbra_three_port_2d_physical_gate0_protocol_20260925.json",
    "research_round_20260920/results/umbra_gate0_public_elevation_20260925/long_beach_lariac_tile_audit.json",
    "research_round_20260920/results/umbra_five_strip_area_gate0_audit_20260925.json",
    "research_round_20260920/results/umbra_three_port_2d_physical_gate0_protocol_v2_20260925.json",
    "research_round_20260920/results/umbra_three_port_2d_physical_gate0_protocol_v3_20260925.json",
    "research_round_20260920/results/umbra_rotterdam_train_rank0_native_sicd_gate0_v1_draft_20260925.json",
    "research_round_20260920/results/umbra_coco_record_export_synthetic_20260925.json",
    "research_round_20260920/prototypes/iceye_vid_c5/__init__.py",
    "research_round_20260920/prototypes/iceye_vid_c5/c5.py",
    "research_round_20260920/prototypes/iceye_vid_c5/test_c5.py",
    "research_round_20260920/prototypes/iceye_vid_c5/README.md",
    "research_round_20260920/external_data/instance_rsdd_20260924/acquisition_status.json",
]

README = """# 2026-09-25 SAR 船舶实例分割研究审阅附录

**内部审阅包；不是 TGRS 可直接投稿稿件，也不是最终可复现训练代码。**

先读 `TGRS_RELEASE_INDEX_20260920.md`，再读
`research_round_20260920/manuscript/RESEARCH_DECISION_20260924.md` 和
`research_round_20260920/manuscript/RESULT_TO_CLAIM_20260924.md`。

本包收录本轮方法否证、来源重叠干预设计、剂量独立复算、外部真值寻找、
SLC/ICEYE/Umbra 多视条件性方案、完整同池及旧 BPR 强对照、隔离的 C5 CPU 原型、
Umbra 十港目录/物理格式审计、三预定训练港首景完整方位轴数值试点、SarPy 几何 STOP 试点及独立审阅、
SDFSD 标注复用核验、120 次采集元数据清单、十港固定 AOI 足迹几何锁、
三训练首景 GEC 有效像素核验与非包内缩略图的可行性分流、三港 AOI→SICD 模型投影/物理读取成本、
外部水陆矢量 v2 的 STOP 核验、三港二维物理 Gate0 原始草案与五带面积矛盾独立审计、
分开的完整 AOI 面积门 v2/v3 待签修订、原生 SICD 单网格备选路线和 Rotterdam 单景草案、当前自动监控停用审计、
三训练首景原生 GEC 内部试标包的无图像审计及双盲判读规范、
独立三维高程公开来源与 Long Beach 瓦片索引的无点云审计、
记录模式与合成语义检查、官方 COCO crowd 语义合成核对、
真实记录→COCO 导出的暂停审计和预先规则锁、10/10 纯合成导出核验及仲裁覆盖草案，
以及少量只读脚本/冻结元数据。文件 SHA-256 在
`SHA256SUMS.txt`。原始 JPEG/TIFF、盲审图像包、私有身份映射、E 测试标注、
权重及预测均未收入；脚本若离开主项目数据不能独立运行。当前收窄措辞的
论文/PDF 工作审阅包在 `delivery/TGRS_WORKING_MANUSCRIPT_REVIEW_20260925_v8.zip`；
9 月 20 日冻结审阅包仍保留。本研究附录不替代论文工作包。

完整原八视角同池诊断中，简单每视角 top-1 的官方 mask AP 比旧门控高 2.5834 点；
复用历史 BPR 后官方差为 +0.2455、近岸 +0.6460、远海 -0.0829。这些均为已暴露测试集的历史诊断，不能作为独立确认。
ICEYE 公开 Dwell 数据的港区分布无法满足原预锁两外港测试；Umbra 十港目录虽有 496 个同采集 GEC/SICD 对，
三个预定训练港各一首景的 128 行完整方位轴三视数值重合成通过，十港各 1 km² 的固定 AOI 被元数据足迹包含；
先前已读过后来入选的一景 Antwerp 条带，不能称全队列零影像预见。三个训练首景 AOI 按文件声明的 nodata 均 100% 有效，
但 v1 中心方框的 Jebel Ali 缩略图主要落在工业陆地，船舶正例高度不确定。v2 外部水陆矢量
未取得十港一致且拓扑完整的版本，已 STOP，v1 不变；原始 OSM 响应未入包。
三个 TRAIN rank0 原生 GEC AOI 的 A/B 空白试标材料已内部准备并按原字节核验，但没有真实标签、
实测物理分辨单元或已释放的正式判读任务；其影像和私有身份映射绝不在本包内。
公开三维资料只为 Antwerp/Long Beach 提供未核点级的旧 LiDAR 候选；Jebel Ali 仍缺该 AOI 可核独立控制，
因此是 STOP_DATA，不能把 NOAA 索引覆盖或 SICD 像素高度预算当作 GEC 物理通过。
五带四角投影名义覆盖仅 15.81%/16.38%/20.22%，原草案不能直接达到 ≥25% 有效域；
v3 未执行草案把五带校准与独立预算的完整 AOI 面积验证分开，
锁定三个 GEC 原 AOI 像元分母与高度/局部 PSF/有效掩膜算法的未解封状态。
真实标注转 COCO 的边界与 crowd 规则已预先锁定，纯合成导出 10/10 通过；
裁决变更覆盖尚待机器规则，真实出口维持 STOP_ADJUDICATION_TRACE，无真实主 GT 导出。
原生 SICD 同网格路线已有 Rotterdam TRAIN rank0 的局部五带草案，未执行，
不是完整 AOI 物理通过或已证创新。
同窗几何拟合的二维静态点留出、实测 PSF、GEC 配准、双盲船舶掩膜和 AP 均未通过。
独立红队要求加入完整 SICD 复数 I/Q 强基线。
不能把数值分解称为可发表的物理方法增益。C5 只在合成 CPU 张量上通过正确性检查。
目前没有新模型在独立采集上获得可发表的完整 mask AP；正式训练仍受岸线双审、
可信外部舰体真值、数据互斥及第三模型复现闸门约束。作者元数据继续保留占位符。
"""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(name: str = NAME) -> None:
    if re.fullmatch(r"RESEARCH_DESIGN_REVIEW_20260924(?:_v[2-9][0-9]*)?", name) is None:
        raise ValueError("invalid review bundle name")
    stage = ROOT / "delivery" / name
    archive = ROOT / "delivery" / f"{name}.zip"
    if stage.exists() or archive.exists():
        raise FileExistsError("review bundle already exists; do not overwrite")
    chosen = {path.relative_to(ROOT).as_posix(): path
              for path in MANUSCRIPTS.glob("*.md")}
    for directory in (
        ROOT / "research_round_20260920" / "automation" / "acquisition_benchmark_20260924",
        ROOT / "research_round_20260920" / "automation" / "overlap_ownership_gate0_20260924",
    ):
        for path in directory.glob("*.py"):
            chosen[path.relative_to(ROOT).as_posix()] = path
    for name in ("gate0_protocol.json", "input_hashes.json", "stage2_fusion_lock.json"):
        path = ROOT / "research_round_20260920" / "automation" / "overlap_ownership_gate0_20260924" / name
        if path.is_file():
            chosen[path.relative_to(ROOT).as_posix()] = path
    for relative in EXTRA:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        chosen[relative] = path
    if len(chosen) < 60:
        raise ValueError("too few research reports for an audit package")
    stage.mkdir(parents=True)
    checksums: list[str] = []
    for relative, source in sorted(chosen.items()):
        content = source.read_bytes()
        dest = stage / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as stream:
            stream.write(content)
        checksums.append(f"{digest(content)}  {relative}")
    with (stage / "README_CN.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(README)
    checksums.append(f"{digest((stage / 'README_CN.md').read_bytes())}  README_CN.md")
    (stage / "SHA256SUMS.txt").write_text("\n".join(sorted(checksums)) + "\n",
                                           encoding="utf-8", newline="\n")
    members = sorted(path for path in stage.rglob("*") if path.is_file())
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as package:
        for path in members:
            relative = path.relative_to(stage).as_posix()
            info = zipfile.ZipInfo(relative, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            package.writestr(info, path.read_bytes())
    with zipfile.ZipFile(archive) as package:
        bad = package.testzip()
        if bad is not None:
            raise ValueError(f"archive CRC failed: {bad}")
        if sorted(package.namelist()) != sorted(
            path.relative_to(stage).as_posix() for path in members
        ):
            raise ValueError("archive member list differs from staged files")
        for path in members:
            relative = path.relative_to(stage).as_posix()
            if package.read(relative) != path.read_bytes():
                raise ValueError(f"archive bytes differ: {relative}")
    sha_path = archive.with_suffix(archive.suffix + ".sha256")
    sha_path.write_text(f"{digest(archive.read_bytes())}  {archive.name}\n",
                        encoding="utf-8", newline="\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": digest(archive.read_bytes()),
                      "file_count": len(members)}, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise SystemExit("usage: package_research_design_review_20260924.py [bundle-name]")
