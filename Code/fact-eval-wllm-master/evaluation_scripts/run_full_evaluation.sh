#!/bin/bash
"""
Full Evaluation Runner

This script runs all three evaluation scripts for a given algorithm and parameters.

Usage:
    ./run_full_evaluation.sh KGW jsl HQA2 --gamma 0.5 --delta 2
    ./run_full_evaluation.sh EXPEdit biomistral MEQS --nlength 256
    ./run_full_evaluation.sh DIP biomistral HQA --alpha 0.45
"""

set -e  # Exit on any error

# Check if minimum arguments provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <algorithm> <model> <dataset> [additional_args...]"
    echo "Examples:"
    echo "  $0 KGW jsl HQA2 --gamma 0.5 --delta 2"
    echo "  $0 EXPEdit biomistral MEQS --nlength 256"
    echo "  $0 DIP biomistral HQA --alpha 0.45"
    exit 1
fi

ALGORITHM=$1
MODEL=$2
DATASET=$3
shift 3  # Remove first 3 arguments, keep the rest

echo "========================================="
echo "Full Evaluation for $ALGORITHM $MODEL $DATASET"
echo "Additional parameters: $@"
echo "========================================="

# Create output directory
OUTPUT_DIR="results_${ALGORITHM}_${MODEL}_${DATASET}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "=== 1. Quality Evaluation ==="
python quality_evaluation.py \
    --algorithm "$ALGORITHM" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output "${OUTPUT_DIR}/quality_results.json" \
    "$@"

echo ""
echo "=== 2. Task Evaluation ==="
python task_evaluation.py \
    --algorithm "$ALGORITHM" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --reference natural \
    --candidate watermarked \
    --output "${OUTPUT_DIR}/task_results.json" \
    "$@"

echo ""
echo "=== 3. Detection Evaluation ==="
python detection_evaluation.py \
    --algorithm "$ALGORITHM" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --target_fpr 0.0 \
    --output "${OUTPUT_DIR}/detection_results.json" \
    "$@"

echo ""
echo "========================================="
echo "Full evaluation completed!"
echo "Results saved in: $OUTPUT_DIR/"
echo "========================================="

# Display summary
echo ""
echo "=== Summary ==="
if [ -f "${OUTPUT_DIR}/quality_results.json" ]; then
    echo "Quality Results:"
    python -c "
import json
with open('${OUTPUT_DIR}/quality_results.json') as f:
    data = json.load(f)
if 'unwatermarked_ppl' in data: print(f'  Unwatermarked PPL: {data[\"unwatermarked_ppl\"]:.2f}')
if 'watermarked_ppl' in data: print(f'  Watermarked PPL: {data[\"watermarked_ppl\"]:.2f}')  
if 'avg_simcse' in data: print(f'  SimCSE: {data[\"avg_simcse\"]:.4f}')
"
fi

if [ -f "${OUTPUT_DIR}/task_results.json" ]; then
    echo "Task Results:"
    python -c "
import json
with open('${OUTPUT_DIR}/task_results.json') as f:
    data = json.load(f)
if 'rouge2' in data: print(f'  ROUGE-2: {data[\"rouge2\"]:.4f}')
if 'rougeL' in data: print(f'  ROUGE-L: {data[\"rougeL\"]:.4f}')
if 'f1' in data: print(f'  F1: {data[\"f1\"]:.4f}')
if 'alignscore' in data: print(f'  AlignScore: {data[\"alignscore\"]:.4f}')
"
fi

if [ -f "${OUTPUT_DIR}/detection_results.json" ]; then
    echo "Detection Results:"
    python -c "
import json
with open('${OUTPUT_DIR}/detection_results.json') as f:
    data = json.load(f)
print(f'  TPR (at {data[\"target_fpr\"]*100}% FPR): {data[\"tpr\"]:.4f}')
print(f'  AUROC: {data[\"auroc\"]:.4f}')
"
fi