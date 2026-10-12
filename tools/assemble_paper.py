# -*- coding: utf-8 -*-
"""Splice the corrected rewrite body into seal.tex: keep preamble (1..maketitle)
and the bibliography/end, replace the body (abstract..conclusion)."""
import io

st = io.open('paper/seal.tex', encoding='utf-8').read().splitlines(keepends=True)
body = io.open('paper/seal_body_rewrite.tex', encoding='utf-8').read().strip()

# find boundaries
i_abs = next(i for i, l in enumerate(st) if l.strip().startswith(r'\begin{abstract}'))
i_bib = next(i for i, l in enumerate(st) if l.strip().startswith(r'\bibliographystyle'))

preamble = ''.join(st[:i_abs])
tail = ''.join(st[i_bib:])
out = preamble + body + '\n\n' + tail
io.open('paper/seal.tex', 'w', encoding='utf-8').write(out)
print(f'spliced: preamble {i_abs} lines + body {len(body)} chars + tail from line {i_bib}')
print('new seal.tex lines:', out.count(chr(10)))
