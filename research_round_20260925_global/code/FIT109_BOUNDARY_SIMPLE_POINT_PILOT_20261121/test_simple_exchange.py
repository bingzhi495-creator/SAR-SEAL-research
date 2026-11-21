"""Generated checks for foreground/background topology and executable pair edits."""
import runpy
import sys

import numpy as np
from scipy.ndimage import generate_binary_structure, label

m = runpy.run_path(sys.argv[1])
fg, bg = generate_binary_structure(2,2), generate_binary_structure(2,1)
rng = np.random.default_rng(27031)

def counts(p):
    p = np.pad(p, 1)
    return label(p, fg)[1], label(~p,bg)[1]-1

checked = 0
for _ in range(2048):
    p = rng.random((9,9)) < rng.random()
    y,x = rng.integers(0,9,2)
    before = counts(p)
    if m["is_simple"](np.pad(p,1),y+1,x+1):
        p[y,x] ^= True
        assert counts(p) == before
        checked += 1
assert checked > 400

p=np.zeros((11,13),dtype=bool)
p[2:9,3:10]=True
coords=np.array([[2*13+1, 5*13+6],
                 [5*13+2, 5*13+9],
                 [5*13+10, 5*13+3],
                 [8*13+2, 3*13+3]])
scores=np.array([[.97,.1],[.90,.2],[.85,.3],[.70,.1]])
pairs=m["constrained_pairs"](p,coords,scores,3)
new=p.copy().reshape(-1)
for a,b in pairs:
    assert new[coords[a,0]] == 0 and new[coords[b,1]] == 1
    new[coords[a,0]],new[coords[b,1]]=True,False
new=new.reshape(p.shape)
assert len(pairs) > 0 and counts(p) == counts(new) and new.sum() == p.sum()
assert all(coords[b,1] != 5*13+6 for _,b in pairs)
print(f"{checked} generated admissible flips and a matched polarity swap fixture passed")
