#!/usr/bin/env python3
"""
Flexible Evaluation Runner

This script allows you to selectively run specific evaluation metrics based on your needs.
You can choose exactly which metrics to compute without running unnecessary evaluations.

Usage:
    python run_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --metrics perplexity simcse rouge f1
    python run_evaluation.py --algorithm EXPEdit --model biomistral --dataset MEQS --nlength 256 --metrics rouge f1 --skip_alignscore
    python run_evaluation.py --algorithm DIP --model biomistral --dataset HQA --alpha 0.45 --metrics detection_only
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from constants import PROJECT_ROOT


def run_command(cmd, description):
    """Run a command and handle errors with real-time output"""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Use Popen for real-time output instead of run with capture_output=True
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1, universal_newlines=True)
        
        output_lines = []
        # Read output line by line and print in real-time
        for line in process.stdout:
            print(line, end='')  # Print each line immediately
            output_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            print("✓ Success")
            return True, ''.join(output_lines)
        else:
            print(f"✗ Error: Command exited with code {return_code}")
            return False, ''.join(output_lines)
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Flexible Evaluation Runner')
    
    # Required parameters
    parser.add_argument('--algorithm', type=str, required=True, 
                       help='Watermarking algorithm (KGW, SWEET, EXPEdit, DIP)')
    parser.add_argument('--model', type=str, required=True,
                       help='Model type (opt, llama, meditron, jsl, biomistral)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset (C4, HQA, MIMIC, HQA2, MEQS)')
    
    # Algorithm specific parameters
    parser.add_argument('--gamma', type=float, help='Gamma parameter for KGW/SWEET')
    parser.add_argument('--delta', type=int, help='Delta parameter for KGW/SWEET')
    parser.add_argument('--entropy', type=float, default=0.9, help='Entropy parameter for SWEET')
    parser.add_argument('--nlength', type=int, help='Length parameter for EXPEdit')
    parser.add_argument('--alpha', type=float, help='Alpha parameter for DIP')
    
    # Metrics selection
    parser.add_argument('--metrics', nargs='+', 
                       choices=['perplexity', 'simcse', 'rouge', 'f1', 'alignscore', 'detection', 
                               'quality', 'task', 'all', 'detection_only'],
                       default=['all'],
                       help='Metrics to calculate. Examples: --metrics f1, --metrics simcse rouge f1, --metrics quality (perplexity+simcse), --metrics task (rouge+f1+alignscore), --metrics detection_only (detection with existing/new scores), --metrics all')
    
    # Evaluation parameters
    parser.add_argument('--dump_path', type=str, default=None,
                       help='Path to dump directory (default: logs/{algorithm})')
    parser.add_argument('--reference', type=str, default='unwatermarked',
                       choices=['unwatermarked', 'natural'],
                       help='Reference text type')
    parser.add_argument('--candidate', type=str, default='watermarked',
                       choices=['watermarked', 'unwatermarked'],
                       help='Candidate text type')
    parser.add_argument('--ntokens', type=int, help='Number of tokens to evaluate')
    parser.add_argument('--nruns', type=int, default=100, help='Number of runs for EXPEdit')
    parser.add_argument('--target_fpr', type=float, default=0.0, help='Target FPR for detection')
    
    # AlignScore parameters
    parser.add_argument('--alignscore_model', type=str, default='roberta-large',
                       help='AlignScore model type')
    parser.add_argument('--skip_alignscore', action='store_true',
                       help='Skip AlignScore even if requested')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--save_individual', action='store_true',
                       help='Save individual script outputs')
    parser.add_argument('--use_existing_detection', action='store_true',
                       help='Use existing detection scores if available')
    
    args = parser.parse_args()
    
    # Set default dump path
    if args.dump_path is None:
        args.dump_path = os.path.join(PROJECT_ROOT, "logs", f"{args.algorithm}")
    
    # Set default output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"eval_results_{args.algorithm}_{args.model}_{args.dataset}_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Expand metric groups
    metrics_to_run = set()
    for metric in args.metrics:
        if metric == 'all':
            metrics_to_run.update(['perplexity', 'simcse', 'rouge', 'f1', 'alignscore', 'detection'])
        elif metric == 'quality':
            metrics_to_run.update(['perplexity', 'simcse'])
        elif metric == 'task':
            metrics_to_run.update(['rouge', 'f1', 'alignscore'])
        elif metric == 'detection_only':
            metrics_to_run = {'detection'}
            break
        else:
            metrics_to_run.add(metric)
    
    # Remove alignscore if skipped
    if args.skip_alignscore:
        metrics_to_run.discard('alignscore')
    
    print(f"Running evaluation for {args.algorithm} {args.model} {args.dataset}")
    print(f"Metrics to calculate: {sorted(metrics_to_run)}")
    print(f"Output directory: {args.output_dir}")
    
    # Build common arguments
    common_args = [
        '--algorithm', args.algorithm,
        '--model', args.model,
        '--dataset', args.dataset,
    ]
    
    if args.dump_path:
        common_args.extend(['--dump_path', args.dump_path])
    if args.gamma is not None:
        common_args.extend(['--gamma', str(args.gamma)])
    if args.delta is not None:
        common_args.extend(['--delta', str(args.delta)])
    if args.entropy is not None:
        common_args.extend(['--entropy', str(args.entropy)])
    if args.nlength is not None:
        common_args.extend(['--nlength', str(args.nlength)])
    if args.alpha is not None:
        common_args.extend(['--alpha', str(args.alpha)])
    if args.ntokens is not None:
        common_args.extend(['--ntokens', str(args.ntokens)])
    
    results_summary = {
        'algorithm': args.algorithm,
        'model': args.model,
        'dataset': args.dataset,
        'metrics_requested': sorted(metrics_to_run),
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }
    
    success_count = 0
    total_count = 0
    
    # Run quality evaluation if needed
    quality_metrics = {'perplexity', 'simcse'} & metrics_to_run
    if quality_metrics:
        total_count += 1
        quality_args = ['python', 'quality_evaluation.py'] + common_args
        
        if 'perplexity' not in quality_metrics:
            quality_args.append('--simcse_only')
        elif 'simcse' not in quality_metrics:
            quality_args.append('--perplexity_only')
        
        quality_args.extend(['--reference', args.reference, '--candidate', args.candidate])
        
        if args.save_individual:
            quality_args.extend(['--output', f"{args.output_dir}/quality_results.json"])
        
        success, output = run_command(quality_args, f"Quality Evaluation ({', '.join(quality_metrics)})")
        if success:
            success_count += 1
            # Extract results from output if not saved to file
            if not args.save_individual and output:
                results_summary['results']['quality'] = 'completed'
    
    # Run task evaluation if needed
    task_metrics = {'rouge', 'f1', 'alignscore'} & metrics_to_run
    if task_metrics:
        total_count += 1
        task_args = ['python', 'task_evaluation.py'] + common_args
        
        # Add specific flags for individual metrics
        if 'rouge' in task_metrics and 'f1' not in task_metrics and 'alignscore' not in task_metrics:
            task_args.append('--rouge_only')
        elif 'f1' in task_metrics and 'rouge' not in task_metrics and 'alignscore' not in task_metrics:
            task_args.append('--f1_only')
        elif 'alignscore' in task_metrics and 'rouge' not in task_metrics and 'f1' not in task_metrics:
            task_args.append('--alignscore_only')
        
        if 'alignscore' not in task_metrics or args.skip_alignscore:
            task_args.append('--skip_alignscore')
        
        task_args.extend([
            '--reference', args.reference,
            '--candidate', args.candidate,
            '--alignscore_model', args.alignscore_model
        ])
        
        if args.save_individual:
            task_args.extend(['--output', f"{args.output_dir}/task_results.json"])
        
        success, output = run_command(task_args, f"Task Evaluation ({', '.join(task_metrics)})")
        if success:
            success_count += 1
            if not args.save_individual and output:
                results_summary['results']['task'] = 'completed'
    
    # Run detection evaluation if needed
    if 'detection' in metrics_to_run:
        total_count += 1
        detection_args = ['python', 'detection_evaluation.py'] + common_args
        detection_args.extend([
            '--target_fpr', str(args.target_fpr),
            '--nruns', str(args.nruns)
        ])
        
        if args.use_existing_detection:
            detection_args.append('--use_existing')
        
        if args.save_individual:
            detection_args.extend(['--output', f"{args.output_dir}/detection_results.json"])
        
        success, output = run_command(detection_args, "Detection Evaluation")
        if success:
            success_count += 1
            if not args.save_individual and output:
                results_summary['results']['detection'] = 'completed'
    
    # Save summary
    results_summary['success_rate'] = f"{success_count}/{total_count}"
    with open(f"{args.output_dir}/evaluation_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n========================================")
    print(f"Evaluation Summary")
    print(f"========================================")
    print(f"Algorithm: {args.algorithm}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Metrics requested: {', '.join(sorted(metrics_to_run))}")
    print(f"Success rate: {success_count}/{total_count}")
    print(f"Results saved in: {args.output_dir}/")
    
    # Show quick summary if individual files were saved
    if args.save_individual:
        print(f"\n=== Quick Summary ===")
        
        # Quality results
        quality_file = f"{args.output_dir}/quality_results.json"
        if os.path.exists(quality_file):
            try:
                with open(quality_file) as f:
                    data = json.load(f)
                print("Quality:")
                if 'unwatermarked_ppl' in data:
                    print(f"  Unwatermarked PPL: {data['unwatermarked_ppl']:.2f}")
                if 'watermarked_ppl' in data:
                    print(f"  Watermarked PPL: {data['watermarked_ppl']:.2f}")
                if 'avg_simcse' in data:
                    print(f"  SimCSE: {data['avg_simcse']:.4f}")
            except:
                pass
        
        # Task results
        task_file = f"{args.output_dir}/task_results.json"
        if os.path.exists(task_file):
            try:
                with open(task_file) as f:
                    data = json.load(f)
                print("Task:")
                if 'rouge2' in data:
                    print(f"  ROUGE-2: {data['rouge2']:.4f}")
                if 'rougeL' in data:
                    print(f"  ROUGE-L: {data['rougeL']:.4f}")
                if 'f1' in data:
                    print(f"  F1: {data['f1']:.4f}")
                if 'alignscore' in data:
                    print(f"  AlignScore: {data['alignscore']:.4f}")
            except:
                pass
        
        # Detection results
        detection_file = f"{args.output_dir}/detection_results.json"
        if os.path.exists(detection_file):
            try:
                with open(detection_file) as f:
                    data = json.load(f)
                print("Detection:")
                print(f"  TPR (at {data['target_fpr']*100}% FPR): {data['tpr']:.4f}")
                print(f"  AUROC: {data['auroc']:.4f}")
            except:
                pass


if __name__ == "__main__":
    main()