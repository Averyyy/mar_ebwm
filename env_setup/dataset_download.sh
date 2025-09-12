#!/usr/bin/env bash
# stream_imagenet_resume_split.sh
set -euo pipefail

# --- CLI args: require --dir, optional --num_workers ---
usage() { echo "Usage: $0 --dir <imagenet_root> [--num_workers <N>]"; exit 1; }
DIR=""
NUM_WORKERS=8
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) DIR="${2:-}"; shift 2 ;;
    --num_workers) NUM_WORKERS="${2:-}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done
[[ -z "$DIR" ]] && { echo "ERROR: --dir is required."; usage; }
# =================

echo "DIR: {$DIR}"
echo "NUM_WORKERS: {$NUM_WORKERS}"

mkdir -p "$DIR"/{train,val,test}

# Use existing login if possible; else require HF_TOKEN
HF_TOKEN="${HF_TOKEN:-$(huggingface-cli whoami -t 2>/dev/null || true)}"
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set and no token found via huggingface-cli."
  echo "Run: huggingface-cli login   (or export HF_TOKEN=...)"
  exit 1
fi

BASE="https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data"

# Map each shard to a split subfolder
declare -A TARGETS=(
  [train_images_0.tar.gz]=train
  [train_images_1.tar.gz]=train
  [train_images_2.tar.gz]=train
  [train_images_3.tar.gz]=train
  [train_images_4.tar.gz]=train
  [val_images.tar.gz]=val
  [test_images.tar.gz]=test
)

FILES=("${!TARGETS[@]}")

# Choose decompressor
if command -v pigz >/dev/null 2>&1; then
  TAR_BASE=(tar --use-compress-program=pigz -x --skip-old-files)
  echo "Using pigz for parallel decompression ${PIGZ:-"(default threads)"}"
else
  TAR_BASE=(tar -xz --skip-old-files)
  echo "Using gzip decompression"
fi

for f in "${FILES[@]}"; do
  split_dir="${TARGETS[$f]}"
  dest="$DIR/$split_dir"
  marker="$dest/.${f}.done"
  
  if [[ -f "$marker" ]]; then
    echo "Skipping $f (already marked done)"
    continue
  fi

  echo "Streaming-extracting $f → $dest"
  url="${BASE}/${f}?download=true"
  mkdir -p "$dest"
  
  curl -fsSL -H "Authorization: Bearer ${HF_TOKEN}" "$url" \
    | "${TAR_BASE[@]}" -C "$dest"

  touch "$marker"
done

echo "✅ All shards extracted under: $DIR/{train,val,test}"

if python util/scripts/reorganize_imagenet_inplace.py \
  --imagenet_root "$DIR" \
  --num_workers "$NUM_WORKERS" \
  --datasets train val test; then
  echo "✅ Reorganization succeeded."
else
  cat <<'MSG'
❌ Reorganization failed, may need to call util/scripts/reorganize_imagenet_inplace.py manually using the following command:
python util/scripts/reorganize_imagenet_inplace.py \
  --imagenet_root "$DIR" \
  --num_workers "$NUM_WORKERS" \
  --datasets train val test
MSG
  exit 1
fi