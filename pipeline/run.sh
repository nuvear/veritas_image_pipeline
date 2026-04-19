#!/bin/bash
# run.sh — Launch the food image pipeline orchestrator
#
# Usage:
#   bash run.sh [--start-fd FD000031] [--end-fd FD004315] [--workers 8] [--batch-size 250]
#
# Environment variables (set before running or edit below):
#   OPENAI_API_KEY
#   SERP_API_KEY
#   INPUT_JSON     — path to input JSON file
#   OUTPUT_DIR     — where to write final images
#   CHECKPOINT_DIR — where to write logs and checkpoints

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── API Keys (override via environment or edit here) ──────────────────────
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export SERP_API_KEY="${SERP_API_KEY:-}"

# ── Paths ─────────────────────────────────────────────────────────────────
INPUT_JSON="${INPUT_JSON:-$SCRIPT_DIR/../input/records.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/../output/images}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$SCRIPT_DIR/../output/checkpoints}"

# ── Pipeline parameters ───────────────────────────────────────────────────
START_FD="${1:-FD000031}"
END_FD="${2:-FD004315}"
WORKERS="${3:-8}"
BATCH_SIZE="${4:-250}"

echo "============================================================"
echo "Food Image Pipeline"
echo "Input:       $INPUT_JSON"
echo "Output:      $OUTPUT_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "FD range:    $START_FD – $END_FD"
echo "Workers:     $WORKERS | Batch size: $BATCH_SIZE"
echo "============================================================"

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

nohup python3 "$SCRIPT_DIR/orchestrator.py" \
    --input "$INPUT_JSON" \
    --output "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --start-fd "$START_FD" \
    --end-fd "$END_FD" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    >> "$CHECKPOINT_DIR/orchestrator.log" 2>&1 &

ORCH_PID=$!
echo "Orchestrator launched (PID $ORCH_PID)"
echo "Monitor: tail -f $CHECKPOINT_DIR/orchestrator.log"
echo $ORCH_PID > "$CHECKPOINT_DIR/orchestrator.pid"
