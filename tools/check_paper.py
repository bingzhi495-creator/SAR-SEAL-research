# -*- coding: utf-8 -*-
import io, re
t = io.open('paper/seal.tex', encoding='utf-8').read()
print('=== TODO / HRSIDtodo placeholders ===')
for m in re.finditer(r'\\(HRSIDtodo|TODO)\{([^}]*)\}', t):
    print(' -', m.group(1), ':', m.group(2)[:100])
print('\n=== detection seeding sentences ===')
for sent in re.split(r'(?<=\.)\s+', t):
    s = sent.strip().replace('\n', ' ')
    if any(k in s for k in ['AR@', 'high-IoU localiz', 'localization', 'recall']) and len(s) < 300:
        if any(k in s for k in ['seed', 'AR', 'localiz', 'recall']):
            print('  •', s[:260])
