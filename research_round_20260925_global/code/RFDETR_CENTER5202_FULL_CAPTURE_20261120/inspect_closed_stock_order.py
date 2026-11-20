from pathlib import Path
from hashlib import sha256
import json
import sys
path=Path(sys.argv[1])
queries=[json.loads(line) for line in (path/"QUERIES_PRIVATE.jsonl").read_bytes().splitlines()]
stock=(path/"STOCK_NATIVE_PREDICTIONS_PRIVATE.jsonl").read_bytes().splitlines(keepends=True)
different_order=[]
for ordinal,(q,line) in enumerate(zip(queries,stock,strict=True),1):
    rows=[dict(image_id=q["image_id"],category_id=1,score=q["ship_scores"][i],segmentation=q["all_rles"][i])
          for i in q["native_ids"]]
    doc=dict(image_id=q["image_id"],raw_query_count=164,native_count=100,
             unmapped_slot1_count=100-len(rows),native_ranks=q["native_ranks"],coco=rows)
    actual=json.loads(line)
    if doc != actual:
        raise ValueError("Scientific stock field mismatch at "+str(ordinal))
    encoded=(json.dumps(doc,separators=(",",":"),ensure_ascii=False,allow_nan=False)+"\n").encode()
    if encoded != line:
        different_order.append(ordinal)
print(json.dumps(dict(exact_semantic_rows=len(queries),different_serialization_rows=len(different_order),
                     first_order_mismatch=different_order[0],last_order_mismatch=different_order[-1])))

