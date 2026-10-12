# -*- coding: utf-8 -*-
import io, re
t = io.open('paper/seal.tex', encoding='utf-8').read()
print('=== tables (labels) ===')
for m in re.finditer(r'\\label\{(tab:[^}]+)\}', t):
    print('  ', m.group(1))
print('=== figures (labels) ===')
for m in re.finditer(r'\\label\{(fig:[^}]+)\}', t):
    print('  ', m.group(1))
print('=== TODO / HRSIDtodo content ===')
for m in re.finditer(r'\\(HRSIDtodo|TODO)\{([^}]*)\}', t):
    c = m.group(2).strip()
    print('  -', m.group(1), ':', (c[:95] if c else '(empty)'))
print('=== competitor numbers present? ===')
for kw in ['O$^2', '81.2', '45.9', '66.4', 'DiffSAR', 'MC-SAM', 'SRNet', 'comparison']:
    print(f'  {kw!r}: {t.count(kw)}')
# find the related-work comparison table content
i = t.find('tab:sota')
if i < 0:
    i = t.find('omparison')
if i > 0:
    s = t.rfind('begin{table', 0, i)
    e = t.find('end{table}', i)
    if s > 0 and e > 0:
        print('=== comparison table ===')
        print(t[s - 1:e + 10][:1100])
