"""Stage exactly the first eight fixed SSDD FIT content groups for native CPU qualification."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--catalog-code",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(args.catalog_code))
    from ssdd_catalog import catalog,canonical,choose,read_member,verify_archive
    verify_archive(args.bundle)
    args.output.mkdir()
    image_folder=args.output/"images"
    image_folder.mkdir()
    with ZipFile(args.bundle) as archive:
        rows,annotations=catalog(archive)
        selected=choose(rows)[:8]
        gt=dict(images=[],annotations=[],categories=[dict(id=1,name="ship")])
        proof=[]
        for row in selected:
            iid=row["image_id"]
            filename=row["original_basename"]
            pin=row["original_image_ref"]
            h,w=row["original_shape"]
            data=read_member(archive,row["archive_image_entry"],pin["sha256"],pin["bytes"])
            if Path(filename).name!=filename or not filename.endswith(".jpg") or not data.startswith(b"\xff\xd8"):
                raise ValueError("Expected exact SSDD FIT JPEG bytes")
            if len(annotations[iid])==0:
                raise ValueError("All prelocked sources require at least one training annotation")
            with (image_folder/filename).open("xb") as f:
                f.write(data)
            gt["images"].append(dict(id=iid,file_name=filename,width=w,height=h))
            gt["annotations"].extend(annotations[iid])
            proof.append(dict(image_id=iid,basename=filename,sha256=pin["sha256"],bytes=pin["bytes"],
                              original_hw=[h,w],annotations=len(annotations[iid])))
    label_data=canonical(gt)
    with (args.output/"annotations.json").open("xb") as f:
        f.write(label_data)
    readback=dict(status="EXACT_SSDD_EIGHT_FIT_CONTENT_GROUPS_STAGED",images=proof,
                  image_count=len(proof),annotation_count=len(gt["annotations"]),
                  annotation_sha256=sha256(label_data).hexdigest(),
                  official_DEV_opened=False,official_TEST_opened=False)
    raw=canonical(readback)
    with (args.output/"CLOSED_INPUT_PRIVATE.json").open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(images=len(proof),annotations=len(gt["annotations"]),
                         subset_manifest_sha256=sha256(raw).hexdigest()),sort_keys=True),flush=True)

if __name__=="__main__":
    main()

