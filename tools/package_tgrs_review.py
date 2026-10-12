from pathlib import Path
import shutil, hashlib, json, zipfile, os, re

ROOT = Path(r'E:\SAR\seal')
OUT = ROOT / 'delivery' / 'TGRS_REVIEW_PACKAGE_20260920'
if OUT.exists():
    shutil.rmtree(OUT)
(OUT/'paper').mkdir(parents=True)
(OUT/'code').mkdir(parents=True)
(OUT/'audit').mkdir(parents=True)

REV = ROOT/'paper_tgrs'/'revision_tgrs_20260905'

def cp(src, dst):
    src = Path(src); dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def cp_rel(src_root, rel, dst_root):
    src = Path(src_root)/rel
    if not src.exists():
        raise FileNotFoundError(src)
    cp(src, Path(dst_root)/rel)

# Paper package: source, compiled deliverables, referenced figures, and review evidence.
paper_files = [
    'main.tex','sec_intro.tex','sec_related.tex','sec_method.tex','sec_exp.tex','sec_disc.tex',
    'tab_fafn.tex','tab_soupn.tex','tab_univ.tex','ai_disclosure.tex','supplement.tex','refs.bib',
    'main.bbl','main.log','main.pdf','supplement.log','supplement.pdf','build.ps1','README_CN.md','AUTHOR_INFORMATION.md',
    'PAPER_PLAN.md','CONFIRMATORY_EXPERIMENTS_CN.md','REVISION_REPORT_CN.md'
]
for rel in paper_files: cp_rel(REV, rel, OUT/'paper')
figs = [
    'fig1_framework_cvpr.png','fig1_framework_ai.png','architecture_expanded_cvpr.png',
    'figm_fusion_cvpr.png','figm_fusion_ai.png','figm_fusion.pdf',
    'figm_bpr_cvpr.png','figm_bpr.pdf','fig1b_readout.pdf','fig_pr_ssdd.pdf',
    'fig8_qual_hrsid_multi.pdf','fig7b_qual_ssdd8.pdf','fig_univ_qual_2row.pdf'
]
for rel in figs: cp_rel(REV, 'figs/'+rel, OUT/'paper')
review_files = [
    'FINAL_QA.json','EXPERIMENT_NUMERIC_CHECK.json','CARRIER_PROVENANCE_CHECK.json',
    'CARRIER_PROVENANCE_CHECK.md','EXPERIMENT_CHANGELOG.md','FRAMING_CHANGELOG.md',
    'METHOD_CHANGELOG.md','INTEGRATED_CONTENT_REVIEW.md','VISUAL_REVIEW.md',
    'ASSET_PROVENANCE.json','MANUSCRIPT_CHANGES.diff','IMAGEGEN_COMPLETE_RECORD.md',
    'O2FORMER_REFERENCE_UPDATE.md','verify_revision.py','build_output.log','final_typeset.log'
]
for rel in review_files:
    src = REV/'review'/rel
    if src.exists(): cp(src, OUT/'paper'/'review'/rel)

verify_pkg = r'''from pathlib import Path
import re, json
import fitz
ROOT=Path(__file__).resolve().parent.parent
res={}
for name in ['main','supplement']:
    pdf=ROOT/f'{name}.pdf'; log=ROOT/f'{name}.log'
    doc=fitz.open(pdf)
    text='\n'.join(p.get_text() for p in doc)
    logtxt=log.read_text(encoding='utf-8',errors='replace')
    res[name]={'pages':len(doc),'size_bytes':pdf.stat().st_size,'overfull':re.findall(r'Overfull[^\n]+',logtxt),'undefined':re.findall(r'(?i)[^\n]*(?:undefined|multiply defined)[^\n]*',logtxt),'placeholders':re.findall(r'\b(?:XXX|TODO|TBD)\b',text)}
sources=['main.tex','sec_intro.tex','sec_related.tex','sec_method.tex','sec_exp.tex','sec_disc.tex','tab_fafn.tex','tab_soupn.tex','tab_univ.tex','ai_disclosure.tex']
text='\n'.join((ROOT/x).read_text(encoding='utf-8') for x in sources)
labels=set(re.findall(r'\\label\{([^}]+)\}',text)); refs=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',text)); cites={k for g in re.findall(r'\\cite\{([^}]+)\}',text) for k in g.split(',')}; bib=set(re.findall(r'@\w+\{([^,]+),',(ROOT/'refs.bib').read_text(encoding='utf-8')))
res['source']={'missing_labels':sorted(refs-labels),'missing_citations':sorted(cites-bib),'abstract_words':len(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',text,re.S).group(1).split())}
print(json.dumps(res,ensure_ascii=False,indent=2))
if any(v.get('overfull') or v.get('undefined') or v.get('placeholders') for k,v in res.items() if k in ['main','supplement']) or res['source']['missing_labels'] or res['source']['missing_citations']:
    raise SystemExit(1)
'''
(OUT/'paper'/'review'/'verify_package.py').write_text(verify_pkg, encoding='utf-8')

# Code snapshot: all carrier source/config/tests, protocol scripts, figure generators,
# and the queue scripts that are explicitly tied to reported paper results.
for d in ['carrier','protocol']:
    for src in (ROOT/d).rglob('*.py'):
        if '__pycache__' in src.parts: continue
        cp(src, OUT/'code'/src.relative_to(ROOT))
for src in (ROOT/'paper_tgrs'/'figs').glob('*.py'):
    if '__pycache__' in src.parts: continue
    cp(src, OUT/'code'/'paper_tgrs'/'figs'/src.name)
for name in ['tools_cast_engine.py','tools_cast_trivial.py']:
    if (ROOT/name).exists(): cp(ROOT/name, OUT/'code'/name)
queue_names = [
    'run_soupn_queue.sh','run_bpr_queue.sh','run_scic_queue.sh','run_ssdd_queue.sh',
    'run_ssdd2_queue.sh','run_ssddr11_queue.sh','run_univ_queue.sh',
    'run_univ_solov2_queue.sh','run_v2seeds_queue.sh','run_fullmetrics_queue.sh',
    'run_post_queue.sh','run_post_waiter.sh'
]
for name in queue_names:
    if (ROOT/name).exists(): cp(ROOT/name, OUT/'code'/'queues'/name)

# Copy only protocol metadata, not datasets or checkpoint binaries.
for rel in ['data/hrsid/annotations/test_inshore.json','data/hrsid/annotations/test_offshore.json',
            'data/pseg_ssdd/annotations/test_inshore.json','data/pseg_ssdd/annotations/test_offshore.json']:
    # Dataset annotations are omitted from the release package by default; record their names below.
    pass

# Remove generated bytecode/caches from the release snapshot.
for cache in OUT.rglob('__pycache__'):
    shutil.rmtree(cache, ignore_errors=True)
for pyc in OUT.rglob('*.pyc'):
    pyc.unlink(missing_ok=True)

# Audit and release notes.
readme = '''# TGRS Review Package (2026-09-20)

This package freezes the 2026-09-05 evidence-bounded revision as a **working manuscript** and pairs it with a source snapshot of the paper-related code.

## Contents

- `paper/`: LaTeX source, compiled `main.pdf` (14 pages), `supplement.pdf` (3 pages), referenced figures, bibliography, and review/provenance artifacts.
- `code/`: paper-related carrier modules, configurations, protocol scripts, CPU tests, figure generators, and the queue scripts used to produce the reported records.
- `audit/`: TGRS readiness decision, blockers, scope and checksum manifests.

## Status

The manuscript is technically coherent and compiles cleanly, but this package is **not a direct TGRS submission package**. The HRSID test subsets informed configuration selection; independent confirmation is not available. Author affiliations, corresponding-author information, funding, and the public code archive are not finalized. The code snapshot also retains absolute paths and depends on external datasets, checkpoints, MMDetection, and recorded run artifacts.

Do not upload this package as-is. Resolve the P0 blockers in `audit/TGRS_READINESS_REVIEW_CN.md` first.
'''
(OUT/'README.md').write_text(readme, encoding='utf-8')

paper_readme = '''# Paper package

Baseline: `paper_tgrs/revision_tgrs_20260905` (2026-09-05 revision).

Build with `powershell -ExecutionPolicy Bypass -File build.ps1` in this directory after installing IEEEtran, pdfLaTeX and BibTeX. The build is document-only; it does not train or evaluate models.

Expected outputs: `main.pdf` (14 pages) and `supplement.pdf` (3 pages). Run `python review/verify_package.py` to check the frozen PDF/source. The recorded source QA reports no missing labels/citations, no overfull boxes, a 222-word abstract, embedded fonts, and no Type3 fonts.

The author block is provisional. `AUTHOR_INFORMATION.md` records the missing affiliations, final author list, corresponding author, email, funding and dates.
'''
(OUT/'paper'/'PACKAGE_README.md').write_text(paper_readme, encoding='utf-8')

code_readme = '''# Paper-related code snapshot

This is a source snapshot for the experiments described in the accompanying paper package. It is not yet a portable public repository.

Included:

- `carrier/`: model carriers, mask-space fusion, BPR, re-ranking, soup and evaluation configs.
- `protocol/`: HRSID/PSeg-SSDD conversion and split utilities.
- `carrier/tests/`: CPU-level smoke tests for fusion, BPR, re-ranking and carrier interfaces.
- `paper_tgrs/figs/`: figure/table generation helpers.
- `queues/`: historical queue scripts associated with reported runs.

Excluded by design: raw datasets, annotation bundles, checkpoints, prediction caches, `runs/`, `work_dirs/`, third-party repositories, Python bytecode, and unrelated robotics/materials files.

The source still contains machine-specific paths and external `_base_` references because this snapshot preserves the evidence-producing implementation. See `PORTABILITY_BLOCKERS.md` for the exact release work still required. Do not claim that this package reproduces the headline numbers on a clean machine until those blockers are fixed and a fresh run is logged.
'''
(OUT/'code'/'README.md').write_text(code_readme, encoding='utf-8')

req = '''# Observed environment used for the recorded experiments (not a lockfile)
python==3.8.17
torch==2.0.1+cu118
mmengine==0.10.3
mmcv==2.1.0
mmdet==3.3.0
numpy==1.23.5
scikit-learn==1.3.0
opencv-python==4.8.0.76
pycocotools==2.0.7
timm==0.9.2
'''
(OUT/'code'/'requirements-observed.txt').write_text(req, encoding='utf-8')

portability = '''# Portability blockers

The snapshot is traceable to the local research workspace, but it is not portable. Before public release:

1. Replace `D:/SAR/UNet++/...`, `E:/SAR/seal/...`, `E:/miktex/...`, and user-specific interpreter paths with an environment variable or config file.
2. Replace the six `univ_*_mstta.py` configs whose `_base_` points into `runs/` with self-contained configs inside this package.
3. Add a clean `tools/train.py`/`tools/test.py` entry point or document the exact MMDetection installation command.
4. Add generators or versioned ID lists for the HRSID inshore/offshore splits.
5. Add the missing BPR leakage audit, paired bootstrap CI, and cross-carrier correlation scripts if those statistics remain in the manuscript.
6. Publish a clean license and third-party attribution record; do not redistribute datasets or weights without checking their terms.
7. Bind configs, checkpoints, prediction caches and table-generation outputs with immutable hashes in a public archive.
'''
(OUT/'code'/'PORTABILITY_BLOCKERS.md').write_text(portability, encoding='utf-8')
(OUT/'code'/'LICENSE_STATUS.md').write_text('No public license is asserted in this review snapshot. Add a project license and third-party attribution before publication.\n', encoding='utf-8')

validation = '''# Validation record — 2026-09-20

## Passed

- Source compilation from `paper_tgrs/revision_tgrs_20260905/build.ps1`: main 14 pages and supplement 3 pages; all passes exited successfully.
- `paper/review/verify_package.py`: passed with no overfull boxes, undefined references, placeholder text, missing labels or missing citation keys; abstract 222 words.
- Python bytecode compilation of the code snapshot passed before cleanup.

## Environment-limited checks

- `carrier/tests/cpu_test_bpr.py`: not runnable in the current Python environment because `pycocotools` is unavailable, despite being listed in the observed environment.
- `carrier/tests/cpu_test_vote_merge.py`: not runnable because the installed MMCV extension cannot load its `_ext` DLL in this environment.

These failures are recorded as environment limitations; they are not counted as successful functional tests. Re-run them in the pinned GPU/CPU environment before public release.
'''
(OUT/'audit'/'VALIDATION_RESULTS.md').write_text(validation, encoding='utf-8')

readiness = '''# TGRS readiness review — 2026-09-20

## Decision

**Current verdict: NO-GO for direct TGRS submission.** The work is a plausible TGRS candidate and the revised manuscript is suitable for internal review or an advisor review, but the present evidence and release state do not support a submission-ready claim.

## What is already strong

- The topic fits TGRS: SAR ship instance segmentation, inference-time candidate aggregation and pixel-level quality correction.
- The 2026-09-05 revision narrows the claim to a cluster-support-gated mask-fusion rule plus boundary refinement and score adjustment. It explicitly distinguishes adopted ingredients from the paper's evaluated integration.
- The manuscript discloses that HRSID test subsets informed configuration selection, distinguishes HRSID auxiliary quality supervision from the standard PSeg-SSDD head, and avoids claiming a universal preservation law or overall SOTA.
- The revision compiles to a 14-page main manuscript and 3-page supplement with no missing labels/citations, no overfull boxes, a 222-word abstract, embedded fonts and no Type3 fonts in the recorded QA.
- Numerical table preservation is documented for 433 cells, but this is a consistency check rather than a fresh AP re-run.

## Blocking findings

| Severity | Finding | Evidence | Required action |
|---|---|---|---|
| P0 | HRSID test subsets informed gate/view/operator selection; headline results are exploratory | `paper/sec_exp.tex`, `paper/sec_disc.tex`, `paper/CONFIRMATORY_EXPERIMENTS_CN.md` | Run an independent scene- or image-disjoint confirmation set with settings locked before evaluation, or narrow the paper to a protocol/diagnostic study and state that scope in the title/abstract. |
| P0 | No public code/checkpoint archive; code snapshot has hard-coded local paths and external `_base_` references | `code/PORTABILITY_BLOCKERS.md`, `paper/sec_disc.tex` | Build a clean repository, add an environment file and one-command reproduction path, publish split IDs and checkpoint provenance, and re-run a clean smoke test. |
| P0 | Author list, affiliations, corresponding author, email, funding and dates are provisional | `paper/AUTHOR_INFORMATION.md`, `paper/main.tex` | Complete the author block and first-footnote metadata; do not invent missing information. |
| P1 | Main manuscript is 14 pages; TGRS currently charges overlength pages beyond page 10 for submissions after 2026-01-01 | Official TGRS/IEEE author guidance; `paper/main.pdf` | Decide whether to pay for 4 overlength pages or trim the main paper and move nonessential material to supplement. |
| P1 | Several reported statistics are not independently regenerated by scripts in the current workspace | prior review records and `paper/CONFIRMATORY_EXPERIMENTS_CN.md` | Add exact bootstrap, split-generation, BPR audit and correlation scripts, or remove/qualify the associated numbers. |
| P1 | Source-to-checkpoint/prediction binding is incomplete; prior checks inspect metadata but do not load/hash all artifacts | `paper/review/CARRIER_PROVENANCE_CHECK.md` | Hash and record configs, checkpoints, predictions and evaluator outputs; preserve the exact table-generation inputs. |
| P1 | Public release licensing and dataset redistribution terms are unresolved | `code/LICENSE_STATUS.md` | Add a project license, third-party notices and dataset/weight usage instructions. |
| P2 | Runtime and memory are not measured end-to-end; view count is not a latency multiplier | `paper/sec_disc.tex` | Add synchronized latency/memory measurements or remove deployment-style cost language. |

## Claim-level judgment

- **Supported with current evidence:** the described fusion rule, its write behavior, the BPR/re-ranker interfaces, and the reported endpoint changes on the recorded HRSID/PSeg-SSDD runs.
- **Supported only conditionally:** transfer to the listed carriers/datasets and the component decomposition, because the settings and selected seeds are conditioned on the recorded development choices.
- **Not supported:** independent confirmation, universal superiority, general SOTA, a general representation-preservation law, or an information-theoretic SAR sensor ceiling.

## Recommended release gate

Do not label the next archive `submission-ready` until all P0 items are closed and the final source is recompiled from the clean archive. After that, perform a fresh citation/claim audit and a human visual check of the exact upload PDF.
'''
(OUT/'audit'/'TGRS_READINESS_REVIEW_CN.md').write_text(readiness, encoding='utf-8')

scope = '''# Scope and exclusion record

Included in this archive: paper source/PDF/supplement, referenced figures, review evidence, paper-related carrier code/configs/tests/protocols, and selected historical queue scripts.

Excluded: `data/`, `runs/`, `work_dirs/`, `pretrained/`, `refs/`, `site-packages/`, raw images, annotation bundles, checkpoints, prediction caches, bytecode, old manuscript trees, autodl bundles, robotics archives, and unrelated diagnostics.

This exclusion is deliberate. The package is a review and freeze artifact, not a complete public reproduction release. The missing external inputs and exact provenance requirements are listed in `audit/TGRS_READINESS_REVIEW_CN.md` and `code/PORTABILITY_BLOCKERS.md`.
'''
(OUT/'audit'/'SCOPE_AND_EXCLUSIONS.md').write_text(scope, encoding='utf-8')

# File inventory and checksums.
records=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file():
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        records.append({'path':str(p.relative_to(OUT)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':h})
(OUT/'audit'/'RELEASE_MANIFEST.json').write_text(json.dumps({'created':'2026-09-20','baseline':'paper_tgrs/revision_tgrs_20260905','status':'review_only','files':records},ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'audit'/'SHA256SUMS.txt').open('w',encoding='utf-8') as f:
    for r in records:
        f.write(f"{r['sha256']}  {r['path']}\n")

# Zip once, then checksum the archive.
zip_path = OUT.with_suffix('.zip')
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file(): z.write(p, p.relative_to(OUT.parent))
sha=hashlib.sha256(zip_path.read_bytes()).hexdigest()
(zip_path.with_suffix('.zip.sha256')).write_text(f'{sha}  {zip_path.name}\n',encoding='utf-8')
print(json.dumps({'root':str(OUT),'zip':str(zip_path),'zip_bytes':zip_path.stat().st_size,'sha256':sha,'files':len(records)},ensure_ascii=False,indent=2))

