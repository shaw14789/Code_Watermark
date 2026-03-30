#!/bin/bash
"""
Run evaluation with AlignScore environment

This script activates the AlignScore environment and runs the evaluation scripts with AlignScore support.

Usage:
    ./run_with_alignscore.sh run_evaluation.py --algorithm DIP --model meditron --dataset HQA --gamma 0.5 --delta 2 --alpha 0.45 --metrics rouge alignscore
    ./run_with_alignscore.sh task_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2
"""

set -e  # Exit on any error

# Check if arguments provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_name> [script_arguments...]"
    echo "Examples:"
    echo "  $0 run_evaluation.py --algorithm DIP --model meditron --dataset HQA --gamma 0.5 --delta 2 --alpha 0.45 --metrics rouge alignscore"
    echo "  $0 task_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2"
    exit 1
fi

SCRIPT_NAME=$1
shift  # Remove script name, keep the rest

# Check if the script exists
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: Script '$SCRIPT_NAME' not found in current directory"
    exit 1
fi

# Check if AlignScore environment exists
ALIGNSCORE_ENV="../AlignScore/alignscore_env/bin/activate"
if [ ! -f "$ALIGNSCORE_ENV" ]; then
    echo "Error: AlignScore environment not found at $ALIGNSCORE_ENV"
    echo "Please set up AlignScore environment first:"
    echo "  cd ../AlignScore"
    echo "  python -m venv alignscore_env"
    echo "  source alignscore_env/bin/activate"
    echo "  pip install ."
    echo "  pip install pytorch_lightning"
    exit 1
fi

echo "========================================="
echo "Running with AlignScore Environment"
echo "========================================="
echo "Script: $SCRIPT_NAME"
echo "Arguments: $@"
echo "Environment: $ALIGNSCORE_ENV"
echo ""

# Activate AlignScore environment and run the script
source "$ALIGNSCORE_ENV"

# Add AlignScore to Python path
export PYTHONPATH="../AlignScore/src:$PYTHONPATH"

echo "AlignScore environment activated"
echo "Running: python $SCRIPT_NAME $@"
echo ""

# Run the script
python "$SCRIPT_NAME" "$@"

echo ""
echo "========================================="
echo "Evaluation with AlignScore completed!"
echo "========================================="