"""Package the current manuscript as an internal review artifact.

The published historical results remain observational and the author fields
remain placeholders. This is deliberately not a submission-ready archive.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper_tgrs" / "revision_tgrs_20260905"
NAME = "TGRS_WORKING_MANUSCRIPT_REVIEW_20260924"

README = """# TGRS 船舶实例分割论文工作审阅包（2026-09-25）

本包是**内部/导师审阅稿，不能直接投稿**。它包含重新编译的 15 页正文、
3 页补充 PDF、LaTeX 源文件、引用、图件及本轮主张审计；作者、单位等仍保留占位符。

先读 `audit/SAME_POOL_TOP1_FULL_DIAGNOSTIC_20260925.md` 和
`audit/TOP1_BPR_HISTORICAL_DIAGNOSTIC_20260925.md`，再读 `paper/main.pdf`。
完整同池对照已精确回放历史输出，简单每视角 top-1 的官方 mask AP 56.4，
高于旧门控 53.8；近岸和远海也同向。稿件因此改为探索性评估，停止把
候选数门控写成必要的新算法贡献。两法都在历史暴露测试集上评估，
不能把这次结果当独立确认。复用历史 BPR 后，top-1/门控官方 AP 为
58.8101/58.5646，近岸 top-1 高 0.6460，远海低 0.0829；旧重排器、
重新 OOF 训练的修正器和完整级联尚未同协议复验。

核心待办是按父场景或采集互斥重新锁定验证，使用 OOF 载体预测训练修正器，
取得可信的外部船体掩膜并测量计算成本。本包没有新训练或外域 AP；
新物理多视建模方案及其数据瓶颈见独立研究附录。
本版 PDF 与 v7 相同；更新的是审计附录：Umbra 三个预定训练港首景仅通过
复数数值重合成，0/3 通过二维静态点/PSF 物理门。固定 AOI 的 Jebel Ali
训练首景主要是工业陆域；统一水域矢量 v2 因十港来源和拓扑不足而暂停，
三景原生 GEC 双人试标包仅完成内部准备，没有合格双人船舶掩膜或新 AP。
三港二维物理 Gate0 仍是待专家签认的协议。五条固定样带在四角投影模型下
名义覆盖仅 15.81%/16.38%/20.22%，原草案不能直接满足 ≥25% 有效域门；
修订 v3 将样带校准与另预算完整 AOI 验证分开，锁定原 AOI 像元分母，
同时将高度传播与逐像元有效域算法标为未解封。不能将这些预检写成论文已验证的创新点。
新增公开三维来源筛查：Antwerp/Long Beach 仅有未核点级的旧 LiDAR 候选，
Jebel Ali 无可核 AOI 级公开控制，故物理门维持 STOP_DATA。COCO 导出已锁
像素边界与 crowd 规则，10/10 纯合成导出通过，但仲裁日志覆盖和真实双人标签仍未完成。
原生 SICD 同网格掩膜是降低跨产品配准混杂的候选研究路线；Rotterdam
TRAIN 首景只有待审的局部五带试点草案，尚无新模型/AP。

所有收录文件的 SHA-256 在 `SHA256SUMS.txt`。此包不含原始影像、
盲审私有映射、测试标注、模型权重或可复现实验代码。
"""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(name: str = NAME) -> None:
    if re.fullmatch(r"TGRS_WORKING_MANUSCRIPT_REVIEW_202609(?:24|25)(?:_v[2-9][0-9]*)?", name) is None:
        raise ValueError("invalid review bundle name")
    stage = ROOT / "delivery" / name
    archive = ROOT / "delivery" / f"{name}.zip"
    if stage.exists() or archive.exists():
        raise FileExistsError("working review bundle exists; do not overwrite")

    chosen: dict[str, Path] = {}
    for suffix in ("*.tex", "*.bib"):
        for path in SOURCE.glob(suffix):
            chosen[f"paper/{path.name}"] = path
    for filename in (
        "build.ps1", "main.pdf", "supplement.pdf", "main.bbl",
        "AUTHOR_INFORMATION.md", "README_CN.md",
    ):
        path = SOURCE / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        chosen[f"paper/{filename}"] = path
    for path in (SOURCE / "figs").rglob("*"):
        if path.is_file():
            chosen[f"paper/figs/{path.relative_to(SOURCE / 'figs').as_posix()}"] = path
    for filename in (
        "PAPER_CLAIM_AUDIT_20260924.md",
        "SAME_POOL_TOP1_FULL_DIAGNOSTIC_20260925.md",
        "TOP1_BPR_HISTORICAL_DIAGNOSTIC_20260925.md",
        "TOP1_BPR_HISTORICAL_PROTOCOL_20260925.md",
        "RESEARCH_DECISION_20260924.md",
        "RESULT_TO_CLAIM_20260924.md",
        "UMBRA_THREE_PORT_2D_GATE0_INDEPENDENT_AUDIT_20260925.md",
        "UMBRA_THREE_PORT_2D_PHYSICAL_GATE0_V2_20260925.md",
        "UMBRA_THREE_PORT_2D_PHYSICAL_GATE0_V3_20260925.md",
        "NATIVE_SICD_VS_GEC_DECISION_20260925.md",
        "UMBRA_ROTTERDAM_TRAIN_RANK0_NATIVE_SICD_GATE0_V1_DRAFT_20260925.md",
    ):
        path = ROOT / "research_round_20260920" / "manuscript" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        chosen[f"audit/{filename}"] = path
    if len(chosen) < 25:
        raise ValueError("too few manuscript files")

    stage.mkdir(parents=True)
    checksums: list[str] = []
    for relative, path in sorted(chosen.items()):
        data = path.read_bytes()
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as out:
            out.write(data)
        checksums.append(f"{digest(data)}  {relative}")
    (stage / "README_CN.md").write_text(README, encoding="utf-8", newline="\n")
    checksums.append(f"{digest((stage / 'README_CN.md').read_bytes())}  README_CN.md")
    (stage / "SHA256SUMS.txt").write_text(
        "\n".join(sorted(checksums)) + "\n", encoding="utf-8", newline="\n"
    )

    members = sorted(path for path in stage.rglob("*") if path.is_file())
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as out:
        for path in members:
            relative = path.relative_to(stage).as_posix()
            info = zipfile.ZipInfo(relative, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            out.writestr(info, path.read_bytes())
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
        if sorted(package.namelist()) != sorted(
            path.relative_to(stage).as_posix() for path in members
        ):
            raise ValueError("ZIP file list differs from staging directory")
        for path in members:
            relative = path.relative_to(stage).as_posix()
            if package.read(relative) != path.read_bytes():
                raise ValueError(f"ZIP bytes differ: {relative}")
    sha = digest(archive.read_bytes())
    archive.with_suffix(".zip.sha256").write_text(
        f"{sha}  {archive.name}\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({
        "archive": str(archive), "bytes": archive.stat().st_size,
        "sha256": sha, "file_count": len(members),
    }, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise SystemExit("usage: package_tgrs_working_review_20260924.py [bundle-name]")
