# Evaluation Scripts

This directory contains standalone Python scripts for evaluating watermarked text generation results. The scripts are based on the implementation in `generate_all_NEW.ipynb` and `visualize.ipynb`.

## Scripts

### 1. `text_generation.py` - Watermarked Text Generation
Generates watermarked and unwatermarked text using various watermarking algorithms.

**Usage:**
```bash
# KGW algorithm for Question-Answering (HQA2)
python text_generation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2

# EXPEdit algorithm for Summarization (MEQS)
python text_generation.py --algorithm EXPEdit --model biomistral --dataset MEQS --nlength 256

# DIP algorithm for Text Completion (HQA)
python text_generation.py --algorithm DIP --model biomistral --dataset HQA --alpha 0.45

# SWEET algorithm for Text Completion (HQA)
python text_generation.py --algorithm SWEET --model jsl --dataset HQA --gamma 0.5 --delta 2 --entropy 0.9

# Custom sample count and save summary
python text_generation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --samples 1000 --output generation_summary.json
```

**Output:** Generates `.pkl` and `.json` files containing watermarked, unwatermarked, and natural texts

### 2. `quality_evaluation.py` - Quality Metrics
Calculates perplexity and SimCSE similarity scores.

**Usage:**
```bash
# KGW algorithm
python quality_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2

# EXPEdit algorithm  
python quality_evaluation.py --algorithm EXPEdit --model biomistral --dataset MEQS --nlength 256

# DIP algorithm
python quality_evaluation.py --algorithm DIP --model biomistral --dataset HQA --alpha 0.45

# Calculate only perplexity
python quality_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --perplexity_only

# Calculate only SimCSE with custom reference/candidate
python quality_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --simcse_only --reference natural --candidate watermarked
```

**Output:** Perplexity and SimCSE scores

### 3. `task_evaluation.py` - Task Performance Metrics
Calculates ROUGE-2, ROUGE-L, F1, and AlignScore.

**Usage:**
```bash
# Full evaluation
python task_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --reference natural --candidate watermarked

# ROUGE scores only
python task_evaluation.py --algorithm EXPEdit --model biomistral --dataset MEQS --nlength 256 --rouge_only

# F1 score only
python task_evaluation.py --algorithm DIP --model biomistral --dataset HQA --alpha 0.45 --f1_only

# Skip AlignScore if not available
python task_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --skip_alignscore
```

**Output:** ROUGE-2, ROUGE-L, F1, and AlignScore

### 4. `detection_evaluation.py` - Detection Performance
Generates detection scores and calculates TPR and AUROC with specified FPR.

**Usage:**
```bash
# Full detection evaluation
python detection_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2

# EXPEdit with custom parameters
python detection_evaluation.py --algorithm EXPEdit --model biomistral --dataset MEQS --nlength 256 --nruns 100

# Use existing scores if available
python detection_evaluation.py --algorithm DIP --model biomistral --dataset HQA --alpha 0.45 --use_existing

# Only calculate metrics from existing scores
python detection_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --metrics_only

# Custom FPR target
python detection_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --target_fpr 0.05
```

**Output:** TPR, AUROC, F1, threshold at target FPR

## Datasets and Models

### Medical Datasets
- **`HQA`**: HealthQA dataset for **text completion/generation tasks**
  - Contains 230-word medical passages
  - Use for: Text completion evaluations
  - Example: `--dataset HQA`

- **`HQA2`**: HealthQA dataset for **question-answering tasks**
  - Medical question-answer pairs
  - Use for: QA task evaluations  
  - Example: `--dataset HQA2`

- **`MEQS`**: MeQSum dataset for **summarization tasks**
  - Medical question summarization
  - Use for: Summarization evaluations
  - Example: `--dataset MEQS`

### Medical Models
- **`meditron`**: Meditron-7B (`epfl-llm/meditron-7b`)
  - Medical domain adaptation of Llama-2
  - Example: `--model meditron`

- **`jsl`**: JSL-MedLlama-3-8B-v2.0 (`johnsnowlabs/JSL-MedLlama-3-8B-v2.0`)
  - John Snow Labs medical LLM
  - Example: `--model jsl`

- **`biomistral`**: BioMistral-7B (`BioMistral/BioMistral-7B`)
  - Medical domain Mistral model
  - Example: `--model biomistral`

## Common Parameters

### Required Parameters
- `--algorithm`: Watermarking algorithm (KGW, SWEET, EXPEdit, DIP)
- `--model`: Model identifier (meditron, jsl, biomistral)
- `--dataset`: Dataset identifier (HQA, HQA2, MEQS)

### Algorithm-Specific Parameters
- **KGW/SWEET**: `--gamma`, `--delta`
- **SWEET**: `--entropy` (default: 0.9)
- **EXPEdit**: `--nlength`
- **DIP**: `--alpha`

### Optional Parameters
- `--dump_path`: Custom path to logs directory (default: `logs/{algorithm}`)
- `--output`: Save results to JSON file
- `--ntokens`: Truncate texts to specified token length
- `--reference`: Reference text type (unwatermarked, natural)
- `--candidate`: Candidate text type (watermarked, unwatermarked)

## Examples

## Flexible Evaluation Runner

### 5. `run_evaluation.py` - Selective Metric Evaluation
Allows you to choose exactly which metrics to calculate without running unnecessary evaluations.

**Usage:**
```bash
# Single metric
python run_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --metrics f1

# Multiple specific metrics
python run_evaluation.py --algorithm EXPEdit --model biomistral --dataset MEQS --nlength 256 --metrics simcse rouge f1

# Quality metrics shorthand (perplexity + SimCSE)
python run_evaluation.py --algorithm DIP --model biomistral --dataset HQA --alpha 0.45 --metrics quality

# Task metrics shorthand (ROUGE + F1 + AlignScore)
python run_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --metrics task

# Detection only (generates scores if needed, then calculates TPR/AUROC)
python run_evaluation.py --algorithm EXPEdit --model biomistral --dataset MEQS --nlength 256 --metrics detection_only

# All metrics
python run_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --metrics all --save_individual
```

### 6. `run_with_alignscore.sh` - AlignScore Environment Wrapper
Automatically uses the AlignScore environment for scripts that require AlignScore.

**Usage:**
```bash
# Run with AlignScore support using existing environment
./run_with_alignscore.sh run_evaluation.py --algorithm DIP --model meditron --dataset HQA --gamma 0.5 --delta 2 --alpha 0.45 --metrics rouge alignscore

# Run task evaluation with AlignScore
./run_with_alignscore.sh task_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --reference natural --candidate watermarked
```

**Available Metrics:**
- `perplexity` - Text perplexity using LLaMA
- `simcse` - Semantic similarity using SimCSE
- `rouge` - ROUGE-2 and ROUGE-L scores
- `f1` - F1 score between texts
- `alignscore` - AlignScore evaluation (requires AlignScore environment)
- `detection` - Detection TPR and AUROC (can use existing scores if available)
- `quality` - Shorthand for perplexity + simcse
- `task` - Shorthand for rouge + f1 + alignscore
- `all` - All available metrics
- `detection_only` - Only detection evaluation (useful for slow detection score generation)

## AlignScore Setup

AlignScore requires a special environment with `pytorch_lightning`. You have two options:

### Option 1: Use Existing Environment (Recommended)
The `alignscore_env` environment is already set up in your AlignScore directory. Use the wrapper script:

```bash
# Use the wrapper for any script that needs AlignScore
./run_with_alignscore.sh run_evaluation.py --algorithm DIP --model meditron --dataset HQA --alpha 0.45 --metrics alignscore
```

### Option 2: Install in Current Environment
```bash
pip install pytorch_lightning
python run_evaluation.py --algorithm DIP --model meditron --dataset HQA --alpha 0.45 --metrics alignscore
```

### Complete evaluation examples

```bash
# Complete evaluation for KGW (without AlignScore)
python run_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --metrics all --skip_alignscore --save_individual

# Complete evaluation for KGW (with AlignScore using environment)
./run_with_alignscore.sh run_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --metrics all --save_individual

# Only ROUGE and F1 for EXPEdit (no special environment needed)
python run_evaluation.py --algorithm EXPEdit --model biomistral --dataset MEQS --nlength 256 --metrics rouge f1

# Quality metrics for DIP (no special environment needed)
python run_evaluation.py --algorithm DIP --model biomistral --dataset HQA --alpha 0.45 --metrics quality

# ROUGE and AlignScore for DIP (using AlignScore environment)
./run_with_alignscore.sh run_evaluation.py --algorithm DIP --model meditron --dataset HQA --gamma 0.5 --delta 2 --alpha 0.45 --metrics rouge alignscore

# Detection with existing scores (fast, no detection score generation)
python run_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --metrics detection_only --use_existing_detection

# Detection generating new scores (slow, full detection pipeline)  
python run_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --metrics detection_only
```