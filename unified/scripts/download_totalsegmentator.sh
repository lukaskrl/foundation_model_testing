#!/usr/bin/env bash
# Fetch the TotalSegmentatorV2 dataset (Zenodo 10047292, CC BY 4.0) and extract
# it into the layout unified/data/totalsegmentator.py expects.
#
#   bash scripts/download_totalsegmentator.sh
#   DATA_ROOT=/mnt/big bash scripts/download_totalsegmentator.sh
#   KEEP_ZIP=1 bash scripts/download_totalsegmentator.sh      # don't delete the archive
#
# v2.0.1 is pinned deliberately: 1228 subjects / 117 structures, which is the
# split (1082 train / 57 val / 89 test) that configs/base.yaml and the committed
# unified/data/splits/*.txt were built against. A different release would
# silently change the benchmark's subject set.
#
# Zenodo drops long connections routinely — curl exits 18 ("transfer closed")
# partway through a 22 GiB file. It does honour Range on GET, so the fix is to
# re-invoke with -C - until the byte count matches rather than to start over.
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/data}"
DEST="$DATA_ROOT/TotalSegmentatorDataset"
ZIP="$DATA_ROOT/downloads/Totalsegmentator_dataset_v201.zip"
URL="https://zenodo.org/api/records/10047292/files/Totalsegmentator_dataset_v201.zip/content"
WANT_BYTES=23581218285
WANT_MD5=fe250e5718e0a3b5df4c4ea9d58a62fe
MAX_TRIES="${MAX_TRIES:-40}"

size_of() { [[ -f "$1" ]] && stat -c %s "$1" || echo 0; }

mkdir -p "$(dirname "$ZIP")" "$DEST"

need=$(( WANT_BYTES - $(size_of "$ZIP") ))
if (( need > 0 )); then
    avail=$(( $(df --output=avail -B1 "$DATA_ROOT" | tail -1) ))
    # zip + extracted tree, and the extracted tree is about the size of the zip
    # (nii.gz members are already compressed, so zip adds almost nothing).
    if (( avail < need + WANT_BYTES )); then
        echo "refusing to start: need ~$(( (need + WANT_BYTES) / 2**30 )) GiB free" \
             "under $DATA_ROOT, have $(( avail / 2**30 )) GiB" >&2
        exit 1
    fi
fi

for (( try = 1; try <= MAX_TRIES; try++ )); do
    have=$(size_of "$ZIP")
    (( have >= WANT_BYTES )) && break
    echo "[$try/$MAX_TRIES] $(( have / 2**20 )) / $(( WANT_BYTES / 2**20 )) MiB" \
         "($(( have * 100 / WANT_BYTES ))%)"
    curl -fL -C - --retry 5 --retry-delay 5 --retry-all-errors \
         --progress-bar -o "$ZIP" "$URL"
done

have=$(size_of "$ZIP")
if (( have != WANT_BYTES )); then
    echo "incomplete after $MAX_TRIES attempts: $have != $WANT_BYTES bytes" >&2
    exit 1
fi

echo "verifying md5 of $(( have / 2**30 )) GiB (a few minutes)..."
got="$(md5sum "$ZIP" | cut -d' ' -f1)"
if [[ "$got" != "$WANT_MD5" ]]; then
    echo "md5 mismatch: $got != $WANT_MD5 — delete $ZIP and re-run" >&2
    exit 1
fi
echo "md5 OK"

# The archive may or may not carry a top-level wrapper directory; extract to a
# staging dir and hoist whichever layout came out, so DEST always ends up with
# subject dirs + meta.csv directly under it.
if [[ ! -f "$DEST/meta.csv" ]]; then
    stage="$DATA_ROOT/.tsv2_stage.$$"
    mkdir -p "$stage"
    echo "extracting..."
    unzip -q "$ZIP" -d "$stage" || { echo "unzip failed" >&2; exit 1; }
    if [[ -f "$stage/meta.csv" ]]; then
        src="$stage"
    else
        src="$(dirname "$(find "$stage" -maxdepth 2 -name meta.csv | head -1)")"
    fi
    [[ -d "$src" ]] || { echo "no meta.csv in the archive" >&2; exit 1; }
    echo "moving into $DEST ..."
    find "$src" -mindepth 1 -maxdepth 1 -exec mv -t "$DEST" {} +
    rm -rf "$stage"
else
    echo "$DEST/meta.csv already present — skipping extraction"
fi

subjects=$(find "$DEST" -mindepth 1 -maxdepth 1 -type d -name 's*' | wc -l)
echo
echo "dataset root: $DEST"
echo "subject dirs: $subjects (expected 1228)"
python3 - "$DEST/meta.csv" <<'PY'
import csv, sys
from collections import Counter
with open(sys.argv[1], encoding="utf-8-sig") as f:
    counts = Counter(r["split"].strip().lower()
                     for r in csv.DictReader(f, delimiter=";"))
print("meta.csv splits:", dict(counts), "(expected train 1082 / val 57 / test 89)")
PY

if [[ -z "${KEEP_ZIP:-}" ]]; then
    rm -f "$ZIP" && echo "removed the archive (KEEP_ZIP=1 to keep it)"
fi

cat <<EOF

next:
  1. point configs/base.yaml at it:
       data.dataset_root: $DEST
       data.meta_csv:     $DEST/meta.csv
  2. python -m scripts.prepare_data          # splits + merged label.nii.gz per subject
EOF
