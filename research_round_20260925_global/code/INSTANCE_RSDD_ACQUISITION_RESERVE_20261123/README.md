# Rebuild the 74-image reserve roster

Use Python 3 and the Poppler or MiKTeX `pdftotext` command. Obtain the SRNet author labels, the original RSDD-SAR paper with Appendix Table 1, and the original RSDD-SAR image ZIP. The admitted COCO release contains 2,810 training images and 740 test images. [SRNet author release](https://github.com/XIDIAN212Lab/SRNet/blob/2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6/readme.md#the-instance-rsdd-dataset) · [RSDD-SAR paper](https://files.sciengine.com/pdfs/2095-283X/23A1C428928844EC809FB12C4F5A7A6B.pdf) · [Original image archive mirror](https://github.com/VisionXLab/point2rbox-v2/blob/932576b104c153b030175e42c5af1baac73999cf/tools/data/rsdd/README.md#download-rsdd-dataset)

Run from this directory with local paths and a new output filename.

```console
python build_reserve_roster.py --author-dir annotation_coco --paper-pdf rsdd-paper.pdf --image-archive RSDD-SAR.zip --output reserve.csv
```

The script reads the author image records and ZIP directory, joins product numbers to the 125 author parent scenes and retains author training records from the 13 scenes absent from the released test. The 2,100-byte CSV has SHA-256 `2614e567e9ef1b2c791c1b357b03adfe7302ece53e911c1963c40e623ad405fc`. Pass the local file to `intake_reserve.py --roster`. Run `python build_reserve_roster.py --help` for the directory-only archive option. [Protocol and evaluation scope](../../INSTANCE_RSDD_PARENT_RESERVE_PRELOCK_20261123.md)
