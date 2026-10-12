# -*- coding: utf-8 -*-
import json, glob, re, io
c = glob.glob(r'E:/Temp/UserTemp/claude/E--SAR/*/tasks/w1c8mnl54.output')
d = json.loads(io.open(c[0], encoding='utf-8').read())
res = d['result']
if isinstance(res, str):
    res = json.loads(res)
print('keys:', list(res.keys()))
tex = res.get('integrated_tex', '')
print('integrated_tex len:', len(tex))
# extract the fenced latex block (first ```latex ... ``` ); fall back to whole
fence = '```'
m = re.search(fence + r'latex\s*(.*?)' + fence, tex, re.DOTALL)
body = m.group(1).strip() if m else tex
print('extracted body len:', len(body))
print('body starts:', repr(body[:80]))
print('body ends:', repr(body[-80:]))
io.open('paper/seal_body_rewrite.tex', 'w', encoding='utf-8').write(body)
rr = res.get('rereview', {})
print('rereview total:', rr.get('total'), 'blockers:', len(rr.get('blockers', [])))
json.dump(rr, io.open('paper/rereview.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for f in rr.get('blockers', []):
    print('BLOCKER:', (f.get('issue', '') or '')[:160])
print('verdicts:', rr.get('verdicts'))
