from itertools import combinations
from pathlib import Path
from runpy import run_path
import numpy as np

probe=run_path(str(Path(__file__).with_name('swap_oracle.py')))
mask=np.ones((2,3),dtype=bool)
assert probe['one_sided_budget'](mask,np.array([.5,0.]))==1.
assert probe['one_sided_budget'](mask,np.array([.5,.5]))==2.25
assert probe['one_sided_budget'](mask,np.array([-.5,-.5]))==2.25
assert probe['one_sided_budget'](mask,np.array([0.,0.]))==0.
adds=np.array([.8,1.,.6,.1]); removes=np.array([.2,0.,.9,.4]); cap=3
idx_a,idx_r,delta=probe['select_swaps'](adds,removes,cap)
truth=max(sum(adds[list(a)])-sum(removes[list(b)]) for k in range(cap+1)
          for a in combinations(range(len(adds)),k) for b in combinations(range(len(removes)),k))
assert len(idx_a)==len(idx_r)==3 and np.isclose(delta,truth)
assert probe['select_swaps'](adds,removes,0)[2]==0.
print('Fractional two-dimensional continuous budget and brute-force fixed-area swap fixture passed')
