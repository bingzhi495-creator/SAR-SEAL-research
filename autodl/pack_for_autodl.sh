#!/usr/bin/env bash
# LOCAL (run on E:\SAR\seal in Git-Bash): build seal_autodl.tar.gz (~52MB).
# Stages code + PSeg annotations + JPEGImages into a `seal/` root, then tars it.
set -e
SEAL="E:/SAR/seal"
IMG="E:/PSeg_SSDD/voc_style"
STAGE="$(mktemp -d)/seal"
echo "staging -> $STAGE"
mkdir -p "$STAGE/data/pseg_ssdd" "$STAGE/PSeg_SSDD/voc_style"

# code (no runs/, no .git, no __pycache__)
for d in carrier protocol mne seeding diagnostics autodl; do
  [ -d "$SEAL/$d" ] && cp -r "$SEAL/$d" "$STAGE/"
done
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name '*.pyc' -delete 2>/dev/null || true

# data: instance annotations + the train/test images only (~52MB)
cp -r "$SEAL/data/pseg_ssdd/annotations" "$STAGE/data/pseg_ssdd/"
cp -r "$IMG/JPEGImages_train" "$STAGE/PSeg_SSDD/voc_style/"
cp -r "$IMG/JPEGImages_test"  "$STAGE/PSeg_SSDD/voc_style/" 2>/dev/null || true

OUT="$SEAL/seal_autodl.tar.gz"
# --force-local: treat the "E:" drive colon as a local path, not a host:path (GNU tar)
tar --force-local -czf "$OUT" -C "$(dirname "$STAGE")" seal
rm -rf "$(dirname "$STAGE")"
echo "DONE -> $OUT"; du -h "$OUT"
echo "Upload this to AutoDL /root/autodl-tmp/ then: cd /root/autodl-tmp && tar xzf seal_autodl.tar.gz"
