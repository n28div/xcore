#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"
URL="https://drive.google.com/drive/folders/18dtd1Qt4h7vezlm2G0hF72aqFcAEFCUo"

# Minimal flag parsing: -o for output dir, -u for URL
while getopts ":o:u:h" opt; do
  case "$opt" in
    o) DATA_DIR="$OPTARG" ;;
    u) URL="$OPTARG" ;;
    h) echo "Usage: $(basename "$0") [-o OUTPUT_DIR] [-u GOOGLE_DRIVE_FOLDER_URL]"; exit 0 ;;
    :) echo "Error: -$OPTARG requires an argument" >&2; exit 1 ;;
    \?) echo "Error: invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

mkdir -p "$DATA_DIR"

if ! command -v gdown >/dev/null 2>&1; then
  echo "Error: 'gdown' not found. Install it and retry, e.g.:" >&2
  echo "  pip install --user gdown    # or: pipx install gdown" >&2
  exit 1
fi

echo "Downloading datasets into '$DATA_DIR' from: $URL"
gdown --folder "$URL" -O "$DATA_DIR" --fuzzy

echo "Datasets downloaded to '$DATA_DIR'"