#!/usr/bin/env python3
"""
Detection Evaluation Script

This script calculates detection scores and computes TPR and AUROC with 0% FPR for watermarked text evaluation.
Based on the implementation in generate_all_NEW.ipynb and visualize.ipynb.

Usage:
    python detection_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2
    python detection_evaluation.py --algorithm EXPEdit --model biomistral --dataset MEQS --nlength 256 --nruns 100
    python detection_evaluation.py --algorithm DIP --model biomistral --dataset HQA --alpha 0.45
"""

import os
import gc
import json
import torch
import pickle
import argparse
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from transformers import AutoTokenizer, set_seed

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from constants import PROJECT_ROOT
from our_utils import *
from watermark.auto_watermark import AutoWatermark
from evaluation.dataset import C4Dataset, ClinicDataset
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator


def _detect_single_text_helper(args):
    """Global helper function to detect watermark for a single text"""
    text, prompt, algorithm, params, model_type, ntokens, nruns, model_type_for_processor = args
    
    try:
        # Get the correct config path using PROJECT_ROOT
        config_path = os.path.join(PROJECT_ROOT, 'config', f'{algorithm}.json')
        
        transformers_config = ModelLoader.load_model(model_type, max_new_tokens=200, min_length=230)
        my_watermark = AutoWatermark.load(
            algorithm, 
            algorithm_config=config_path, 
            transformers_config=transformers_config
        )
        
        text_processor_tokenizer = None
        
        if prompt is not None:
            processed_text = TextProcessor.remove_prompt(
                text, 
                prompt, 
                model_type_for_processor, 
                text_processor_tokenizer
            )
        else:
            processed_text = text
            
        if not processed_text:
            return None
        
        if ntokens is not None:
            processed_text = TextProcessor.truncate_text(
                processed_text, 
                my_watermark.config.generation_tokenizer, 
                max_length=ntokens
            )
        
        # Use parallel detection for EXPEdit algorithm for better performance
        if algorithm == "EXPEdit":
            score = my_watermark.detect_watermark_parallel(processed_text, return_dict=True)['score']
        else:
            score = my_watermark.detect_watermark(processed_text, return_dict=True)['score']
        
        del my_watermark
        torch.cuda.empty_cache()
        gc.collect()
        
        return score
    except Exception as e:
        print(f"Error in worker process: {e}")
        return None


def detect_watermark_single_process(dump_path, algorithm, model_type, dataset, params=None, ntokens=None, nruns=100):
    """
    Calculate detection scores for generated texts using single process (like in notebook).
    """
    file_name = FileManager.build_filename(algorithm, params, model_type, dataset)
    watermark_file_path = f'{dump_path}/{file_name}'
    
    with open(watermark_file_path, 'rb') as f:
        watermarked_texts, unwatermarked_texts, natural_texts = pickle.load(f)
    
    dataset_path = FileManager.get_dataset_path(dataset)
    my_dataset = ClinicDataset(dataset_path)
    
    config_params = params.copy() if params else {}
    if algorithm == "EXPEdit":
        config_params["n_runs"] = nruns
    
    # Change to parent directory temporarily for config operations
    original_cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(script_dir, '..')
    os.chdir(parent_dir)
    
    try:
        config = ConfigManager.load_config(algorithm, config_params)
        ConfigManager.save_config(algorithm, config)
    finally:
        os.chdir(original_cwd)
    
    # Initialize watermark detector (single process)
    transformers_config = ModelLoader.load_model(model_type, max_new_tokens=200, min_length=230)
    
    # Get the correct config path relative to the main project directory
    config_path = os.path.join(script_dir, '..', 'config', f'{algorithm}.json')
    my_watermark = AutoWatermark.load(
        algorithm, 
        algorithm_config=config_path, 
        transformers_config=transformers_config
    )
    
    my_watermark.print_config()
    
    # Setup tokenizer for preprocessing
    text_processor_tokenizer = None
    model_type_for_processor = None
    
    print(f"Processing {len(watermarked_texts)} watermarked texts...")
    
    debug_print = True

    # Detect watermarked texts
    w_scores = []
    for text, prompt in tqdm(zip(watermarked_texts, my_dataset.prompts), desc="Detecting watermarked texts"):
        text = TextProcessor.remove_prompt(
            text, 
            prompt, 
            model_type_for_processor, 
            text_processor_tokenizer
        )
        
        if not text:
            continue
        
        if ntokens is not None:
            text = TextProcessor.truncate_text(
                text, 
                my_watermark.config.generation_tokenizer, 
                max_length=ntokens
            )

        if debug_print:
            print("w:",text)
            debug_print = False
        
        # Use parallel detection for EXPEdit algorithm for better performance
        if algorithm == "EXPEdit":
            w_scores.append(my_watermark.detect_watermark_parallel(text, return_dict=True)['score'])
        else:
            w_scores.append(my_watermark.detect_watermark(text, return_dict=True)['score'])

    print(f'Watermarked Score [{min(w_scores) if w_scores else "N/A"}, {max(w_scores) if w_scores else "N/A"}]')
    
    # Save w_scores backup
    ntokens_str = f"-{ntokens}" if ntokens is not None else ""
    nruns_str = f"-{nruns}" if nruns != 100 and algorithm == "EXPEdit" else ""
    w_scores_backup_path = f'{dump_path}/{file_name}-W_SCORES{ntokens_str}{nruns_str}.pkl'
    with open(w_scores_backup_path, 'wb') as f:
        pickle.dump(w_scores, f)
    print(f"Saved w_scores backup to {w_scores_backup_path}")
    
    # Detection for unwatermarked is typically not needed
    uw_scores = []
    
    # Save uw_scores backup
    uw_scores_backup_path = f'{dump_path}/{file_name}-UW_SCORES{ntokens_str}{nruns_str}.pkl'
    with open(uw_scores_backup_path, 'wb') as f:
        pickle.dump(uw_scores, f)
    print(f"Saved uw_scores backup to {uw_scores_backup_path}")
    
    print(f"Processing {len(natural_texts)} natural texts...")
    
    # Detect natural texts
    n_scores = []
    debug_print = True
    for text in tqdm(natural_texts, desc="Detecting natural texts"):
        if ntokens is not None:
            text = TextProcessor.truncate_text(
                text, 
                my_watermark.config.generation_tokenizer, 
                max_length=ntokens
            )

        if debug_print:
            print("n:",text)
            debug_print = False
        
        # Use parallel detection for EXPEdit algorithm for better performance
        if algorithm == "EXPEdit":
            n_scores.append(my_watermark.detect_watermark_parallel(text, return_dict=True)['score'])
        else:
            n_scores.append(my_watermark.detect_watermark(text, return_dict=True)['score'])

    print(f'Natural Score [{min(n_scores) if n_scores else "N/A"}, {max(n_scores) if n_scores else "N/A"}]')
    
    # Save n_scores backup
    n_scores_backup_path = f'{dump_path}/{file_name}-N_SCORES{ntokens_str}{nruns_str}.pkl'
    with open(n_scores_backup_path, 'wb') as f:
        pickle.dump(n_scores, f)
    print(f"Saved n_scores backup to {n_scores_backup_path}")
    
    # Save detection scores
    score_file_path = f'{dump_path}/{file_name}-SCORES{ntokens_str}{nruns_str}.pkl'
    with open(score_file_path, 'wb') as f:
        pickle.dump((w_scores, uw_scores, n_scores), f)
    
    print(f"Saved scores to {score_file_path}")
    
    # Clean up
    del my_watermark
    torch.cuda.empty_cache()
    gc.collect()
    
    return w_scores, uw_scores, n_scores


def calculate_detection_metrics(w_scores, uw_scores, n_scores, target_fpr=0.0):
    """
    Calculate TPR and AUROC from detection scores.
    """
    calculator = DynamicThresholdSuccessRateCalculator(
        labels=['TPR', 'FPR', 'F1', 'AUROC', 'AUROC_fpr', 'AUROC_tpr', 'Threshold'], 
        rule='target_fpr', 
        target_fpr=target_fpr
    )
    
    # For EXPEdit, the order is reversed (n_scores, w_scores)
    # For other algorithms, use (w_scores, n_scores)
    result = calculator.calculate(w_scores, n_scores)
    
    return result


def load_existing_scores(dump_path, algorithm, model_type, dataset, params=None, ntokens=None, nruns=100):
    """
    Load existing detection scores from file if available.
    """
    file_name = FileManager.build_filename(algorithm, params, model_type, dataset)
    
    ntokens_str = f"-{ntokens}" if ntokens is not None else ""
    nruns_str = f"-{nruns}" if nruns != 100 and algorithm == "EXPEdit" else ""
    score_file_path = f'{dump_path}/{file_name}-SCORES{ntokens_str}{nruns_str}.pkl'
    
    try:
        with open(score_file_path, 'rb') as f:
            w_scores, uw_scores, n_scores = pickle.load(f)
        print(f"Loaded existing scores from: {score_file_path}")
        return w_scores, uw_scores, n_scores
    except FileNotFoundError:
        print(f"No existing scores found at: {score_file_path}")
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description='Detection Evaluation: TPR and AUROC calculation')
    parser.add_argument('--algorithm', type=str, required=True, 
                       help='Watermarking algorithm (KGW, SWEET, EXPEdit, DIP)')
    parser.add_argument('--model', type=str, required=True,
                       help='Model type (opt, llama, meditron, jsl, biomistral)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset (C4, HQA, MIMIC, HQA2, MEQS)')
    parser.add_argument('--dump_path', type=str, default=None,
                       help='Path to dump directory (default: logs/{algorithm})')
    
    # Algorithm specific parameters
    parser.add_argument('--gamma', type=float, help='Gamma parameter for KGW/SWEET')
    parser.add_argument('--delta', type=int, help='Delta parameter for KGW/SWEET')
    parser.add_argument('--entropy', type=float, default=0.9, help='Entropy parameter for SWEET')
    parser.add_argument('--nlength', type=int, help='Length parameter for EXPEdit')
    parser.add_argument('--alpha', type=float, help='Alpha parameter for DIP')
    
    # Detection parameters
    parser.add_argument('--ntokens', type=int, help='Number of tokens to evaluate (truncate texts)')
    parser.add_argument('--nruns', type=int, default=100, help='Number of runs for EXPEdit detection')
    parser.add_argument('--target_fpr', type=float, default=0.0, help='Target FPR for threshold selection')
    
    # Options
    parser.add_argument('--use_existing', action='store_true',
                       help='Use existing detection scores if available')
    parser.add_argument('--output', type=str, help='Output file path for results')
    parser.add_argument('--metrics_only', action='store_true',
                       help='Only calculate metrics from existing scores (no detection)')
    
    args = parser.parse_args()
    
    # Build parameters dictionary
    params = {}
    if args.gamma is not None:
        params['gamma'] = args.gamma
    if args.delta is not None:
        params['delta'] = args.delta
    if args.entropy is not None:
        params['entropy'] = args.entropy
    if args.nlength is not None:
        params['nlength'] = args.nlength
    if args.alpha is not None:
        params['alpha'] = args.alpha
    
    # Set default dump path
    if args.dump_path is None:
        args.dump_path = os.path.join(PROJECT_ROOT, "logs", f"{args.algorithm}")
    
    print(f"Detection Evaluation for {args.algorithm} {args.model} {args.dataset}")
    print(f"Parameters: {params}")
    print(f"Dump path: {args.dump_path}")
    print(f"Target FPR: {args.target_fpr}")
    
    w_scores, uw_scores, n_scores = None, None, None
    
    # Try to load existing scores first
    if args.use_existing or args.metrics_only:
        w_scores, uw_scores, n_scores = load_existing_scores(
            args.dump_path, args.algorithm, args.model, args.dataset, 
            params, args.ntokens, args.nruns
        )
    
    # Calculate detection scores if not available or not using existing
    if not args.metrics_only and (w_scores is None or n_scores is None):
        print("\n=== Calculating Detection Scores ===")
        try:
            w_scores, uw_scores, n_scores = detect_watermark_single_process(
                dump_path=args.dump_path,
                algorithm=args.algorithm,
                model_type=args.model,
                dataset=args.dataset,
                params=params,
                ntokens=args.ntokens,
                nruns=args.nruns
            )
        except Exception as e:
            print(f"Error calculating detection scores: {e}")
            return
    
    if w_scores is None or n_scores is None:
        print("Error: Could not load or calculate detection scores")
        return
    
    print(f"\n=== Calculating Detection Metrics ===")
    print(f"Loaded {len(w_scores)} watermarked scores and {len(n_scores)} natural scores")
    
    # Calculate metrics
    if args.algorithm == "EXPEdit":
        # For EXPEdit, the order is reversed
        result = calculate_detection_metrics(n_scores, uw_scores, w_scores, args.target_fpr)
    else:
        result = calculate_detection_metrics(w_scores, uw_scores, n_scores, args.target_fpr)
    
    results = {
        'algorithm': args.algorithm,
        'model': args.model,
        'dataset': args.dataset,
        'target_fpr': args.target_fpr,
        'tpr': result['TPR'],
        'fpr': result['FPR'],
        'f1': result['F1'],
        'auroc': result['AUROC'],
        'threshold': result['Threshold'],
        'num_watermarked_scores': len(w_scores),
        'num_natural_scores': len(n_scores),
        'watermarked_score_range': [min(w_scores), max(w_scores)] if w_scores else None,
        'natural_score_range': [min(n_scores), max(n_scores)] if n_scores else None,
        **params
    }
    
    # Save results if output path specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\n=== Detection Results ===")
    print(f"TPR (at {args.target_fpr*100}% FPR): {result['TPR']:.4f}")
    print(f"AUROC: {result['AUROC']:.4f}")
    print(f"F1 Score: {result['F1']:.4f}")
    print(f"Threshold: {result['Threshold']:.4f}")
    print(f"Actual FPR: {result['FPR']:.4f}")
    
    if w_scores:
        print(f"Watermarked scores: [{min(w_scores):.4f}, {max(w_scores):.4f}]")
    if n_scores:
        print(f"Natural scores: [{min(n_scores):.4f}, {max(n_scores):.4f}]")
    
    return results


if __name__ == "__main__":
    main()