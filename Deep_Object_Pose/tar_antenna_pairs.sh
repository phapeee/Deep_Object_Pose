#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data/AntennaTrainingData"
DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/antenna_pairs_subset"

COUNT=${1:-0}
OUTPUT_DIR=${2:-$DEFAULT_OUTPUT_DIR}

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[error] Data directory '$DATA_DIR' not found." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
pairs_collected=0
while IFS= read -r -d '' png; do
  json="${png%.png}.json"
  if [[ ! -f "$json" ]]; then
    echo "[warn] Missing JSON for $png (expected $json). Skipping."
    continue
  fi
  rel_png="${png#$DATA_DIR/}"
  rel_json="${json#$DATA_DIR/}"
  dest_png="$OUTPUT_DIR/$rel_png"
  dest_json="$OUTPUT_DIR/$rel_json"
  mkdir -p "$(dirname "$dest_png")"
  mkdir -p "$(dirname "$dest_json")"
  cp "$png" "$dest_png"
  cp "$json" "$dest_json"
  pairs_collected=$((pairs_collected + 1))
  if [[ $COUNT -gt 0 && $pairs_collected -ge $COUNT ]]; then
    break
  fi
done < <(find "$DATA_DIR" -type f -name '*.png' -print0 | sort -z)

if [[ $pairs_collected -eq 0 ]]; then
  echo "[error] No PNG/JSON pairs found." >&2
  exit 1
fi

echo "Copied $pairs_collected PNG/JSON pairs into '$OUTPUT_DIR'."
