#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/sem8/satts/NLP"
LOG_FILE="$REPO_DIR/nohup_script.log"
OUTPUT_NOTEBOOK="$REPO_DIR/transformer_akkadian_english_out.ipynb"

cd "$REPO_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] starting script.py" >> "$LOG_FILE"

python3 script.py >> "$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] script.py finished successfully" >> "$LOG_FILE"

if [[ -f "$OUTPUT_NOTEBOOK" ]]; then
    git add "$(basename "$OUTPUT_NOTEBOOK")"
    if ! git diff --cached --quiet; then
        git commit -m "Add executed notebook output"
        git push origin main
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] pushed output notebook to origin/main" >> "$LOG_FILE"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] no output changes to commit" >> "$LOG_FILE"
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] expected output notebook not found" >> "$LOG_FILE"
    exit 1
fi
