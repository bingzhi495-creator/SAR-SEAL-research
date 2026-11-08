"""Stable canonical DEV export and equal-width orientation controls."""
import json

import numpy as np

ARMS=("native","uncertainty","center","ring","local_logit")
SEEDS=(5201,5202,5203)
E100_AP=0.6816530127007889
NATIVE_SHA="7d6239e87ccc6aed4a012e491007ce4c48edff053d9edf8f04f7177f5e138321"
DEV_MANIFEST_SHA="18df007e15e3e838e607f178f14e7a26f2b37b992ee39bf091e0b74cf084fb13"
DEV_GT_SHA="5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55"


def encode_json(value):
    return (json.dumps(value,separators=(",",":"),allow_nan=False)+"\n").encode("utf-8")


def unordered_ring(x):
    y=x.copy()
    y[:,6:]=np.sort(x[:,6:].reshape(-1,4,8),axis=2).reshape(-1,32)
    return y


def unchanged_identities(a,b):
    def key(row):
        return row["image_id"],row["category_id"],row["score"].hex(),row["segmentation"]["size"]
    return len(a)==len(b) and all(key(x)==key(y) for x,y in zip(a,b))

