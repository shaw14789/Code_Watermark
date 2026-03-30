#!/usr/bin/env python3
"""
Task Evaluation Script

This script calculates ROUGE-2, ROUGE-L, F1, and AlignScore for watermarked text evaluation.
Based on the implementation in generate_all_NEW.ipynb.

Usage:
    python task_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2
    python task_evaluation.py --algorithm EXPEdit --model biomistral --dataset MEQS --nlength 256
    python task_evaluation.py --algorithm DIP --model biomistral --dataset HQA --alpha 0.45
"""

import os
import gc
import json
import torch
import pickle
import argparse
import sys
from tqdm import tqdm
from transformers import AutoTokenizer
from rouge_score import rouge_scorer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from constants import PROJECT_ROOT
from our_utils import *
from evaluation.dataset import C4Dataset, ClinicDataset

# Add AlignScore path if available
try:
    # First try with src path added
    sys.path.append('../AlignScore/src')
    from alignscore import AlignScore
    ALIGNSCORE_AVAILABLE = True
    print("AlignScore successfully imported")
except ImportError as e:
    print(f"Warning: AlignScore not available. Error: {e}")
    print("To use AlignScore, you need to:")
    print("1. Activate the alignscore environment: source ../AlignScore/alignscore_env/bin/activate")
    print("2. Or install pytorch_lightning: pip install pytorch_lightning")
    print("3. Then re-run this script")
    ALIGNSCORE_AVAILABLE = False


def get_rouge_score(dump_path, algorithm, model_type, dataset, params=None,
                   reference="natural", candidate="watermarked", 
                   standardize_length=False, token_limit=100, 
                   watermarked_texts=None, unwatermarked_texts=None, natural_texts=None):
    """
    Calculate ROUGE scores between texts.
    """
    scorer = rouge_scorer.RougeScorer(['rouge2', 'rougeL'], use_stemmer=False)
    dataset_path = FileManager.get_dataset_path(dataset)
    
    if watermarked_texts is None:
        file_name = FileManager.build_filename(algorithm, params, model_type, dataset)
        watermark_file_path = f'{dump_path}/{file_name}'
        with open(watermark_file_path, 'rb') as f:
            watermarked_texts, unwatermarked_texts, natural_texts = pickle.load(f)
    
    n = len(watermarked_texts)
    print(f"Samples: {n} watermarked, {len(unwatermarked_texts)} unwatermarked, {len(natural_texts)} natural")
    
    text_processor_tokenizer = None
    text_processor_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[model_type])
    
    rouge2_scores = []
    rougeL_scores = []
    my_dataset = ClinicDataset(dataset_path)
    
    for idtext in tqdm(range(n)):
        prompt = my_dataset.get_prompt(idtext)
        
        wt_text = TextProcessor.remove_prompt(
            watermarked_texts[idtext], 
            prompt, 
            model_type, 
            text_processor_tokenizer
        )
        
        uwt_text = TextProcessor.remove_prompt(
            unwatermarked_texts[idtext], 
            prompt, 
            model_type, 
            text_processor_tokenizer
        )
        
        nt_text = natural_texts[idtext]
        
        if standardize_length:
            wt_tokens = text_processor_tokenizer.encode(wt_text, add_special_tokens=True)
            uwt_tokens = text_processor_tokenizer.encode(uwt_text, add_special_tokens=True)
            nt_tokens = text_processor_tokenizer.encode(nt_text, add_special_tokens=True)
            
            length = min(token_limit, len(wt_tokens), len(uwt_tokens), len(nt_tokens))
            
            wt_text = text_processor_tokenizer.decode(wt_tokens[:length], skip_special_tokens=True)
            uwt_text = text_processor_tokenizer.decode(uwt_tokens[:length], skip_special_tokens=True)
            nt_text = text_processor_tokenizer.decode(nt_tokens[:length], skip_special_tokens=True)
        
        if reference == "unwatermarked" and candidate == "watermarked":
            ref_text = uwt_text
            cand_text = wt_text
        elif reference == "natural" and candidate == "watermarked":
            ref_text = nt_text
            cand_text = wt_text
        elif reference == "natural" and candidate == "unwatermarked":
            ref_text = nt_text
            cand_text = uwt_text

        scores = scorer.score(ref_text, cand_text)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return (sum(rouge2_scores) / len(rouge2_scores), sum(rougeL_scores) / len(rougeL_scores)), (rouge2_scores, rougeL_scores)


def get_f1_score(dump_path, algorithm, model_type, dataset, params=None,
                   reference="unwatermarked", candidate="watermarked",
                   watermarked_texts=None, unwatermarked_texts=None, natural_texts=None):
    """
    Calculate F1 scores between texts.
    """
    dataset_path = FileManager.get_dataset_path(dataset)
    
    if watermarked_texts is None:
        file_name = FileManager.build_filename(algorithm, params, model_type, dataset)
        watermark_file_path = f'{dump_path}/{file_name}'
        
        with open(watermark_file_path, 'rb') as f:
            watermarked_texts, unwatermarked_texts, natural_texts = pickle.load(f)
    
    n = len(watermarked_texts)
    print(f"Samples: {n} watermarked, {len(unwatermarked_texts)} unwatermarked, {len(natural_texts)} natural")
    
    text_processor_tokenizer = None
    text_processor_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[model_type])
    
    f1_scores = []
    my_dataset = ClinicDataset(dataset_path)
    wt_texts_dump = []
    
    for idtext in tqdm(range(n)):
        prompt = my_dataset.get_prompt(idtext)
        
        wt_text = TextProcessor.remove_prompt(
            watermarked_texts[idtext], 
            prompt, 
            model_type, 
            text_processor_tokenizer
        )
        wt_texts_dump.append(wt_text)
        
        uwt_text = TextProcessor.remove_prompt(
            unwatermarked_texts[idtext], 
            prompt, 
            model_type, 
            text_processor_tokenizer
        )
        
        nt_text = natural_texts[idtext]
        
        if reference == "unwatermarked" and candidate == "watermarked":
            ref_text = uwt_text
            cand_text = wt_text
        elif reference == "natural" and candidate == "watermarked":
            ref_text = nt_text
            cand_text = wt_text
        elif reference == "natural" and candidate == "unwatermarked":
            ref_text = nt_text
            cand_text = uwt_text
        
        scores = compute_f1(cand_text, ref_text)
        f1_scores.append(scores)
    
    return sum(f1_scores) / len(f1_scores), f1_scores, wt_texts_dump


def get_alignscore_score(dump_path, algorithm, model_type, dataset, params=None,
                        reference="unwatermarked", candidate="watermarked", 
                        standardize_length=False, token_limit=100,
                        watermarked_texts=None, unwatermarked_texts=None, natural_texts=None,
                        alignscore_model='roberta-base', alignscore_ckpt=None, 
                        evaluation_mode='nli_sp', device='cuda:0', batch_size=32):
    """
    Calculate AlignScore between texts.
    """
    if not ALIGNSCORE_AVAILABLE:
        print("AlignScore not available. Skipping AlignScore calculation.")
        return None, None
    
    if alignscore_ckpt is None:
        if alignscore_model == 'roberta-base':
            alignscore_ckpt = '../AlignScore/model/AlignScore-base.ckpt'
        else:
            alignscore_ckpt = '../AlignScore/model/AlignScore-large.ckpt'
    
    scorer = AlignScore(
        model=alignscore_model,
        batch_size=batch_size,
        device=device,
        ckpt_path=alignscore_ckpt,
        evaluation_mode=evaluation_mode,
        verbose=True
    )
    
    if watermarked_texts is None:
        file_name = FileManager.build_filename(algorithm, params, model_type, dataset)
        watermark_file_path = f'{dump_path}/{file_name}'
        
        with open(watermark_file_path, 'rb') as f:
            watermarked_texts, unwatermarked_texts, natural_texts = pickle.load(f)
    
    n = len(watermarked_texts)
    print(f"Samples: {n} watermarked, {len(unwatermarked_texts)} unwatermarked, {len(natural_texts)} natural")
    
    dataset_path = FileManager.get_dataset_path(dataset)
    my_dataset = ClinicDataset(dataset_path)
    
    text_processor_tokenizer = None
    text_processor_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[model_type])
    
    alignscore_scores = []
    
    for idtext in tqdm(range(n), desc="Computing AlignScore"):
        if idtext < len(my_dataset.prompts):
            prompt = my_dataset.prompts[idtext]
        else:
            prompt = ""
        
        wt_text = TextProcessor.remove_prompt(
            watermarked_texts[idtext], 
            prompt, 
            model_type, 
            text_processor_tokenizer
        )
        
        uwt_text = TextProcessor.remove_prompt(
            unwatermarked_texts[idtext], 
            prompt, 
            model_type, 
            text_processor_tokenizer
        )
        
        nt_text = natural_texts[idtext] if idtext < len(natural_texts) else ""
        
        if reference == "unwatermarked" and candidate == "watermarked":
            ref_text = uwt_text
            cand_text = wt_text
        elif reference == "natural" and candidate == "watermarked":
            ref_text = nt_text
            cand_text = wt_text
        elif reference == "natural" and candidate == "unwatermarked":
            ref_text = nt_text
            cand_text = uwt_text
        else:
            print(f"Invalid reference-candidate combination: {reference}-{candidate}")
            continue
        
        if not ref_text.strip() or not cand_text.strip():
            print(f"Warning: Empty text at index {idtext}, skipping")
            continue
        
        score = scorer.score(contexts=[ref_text], claims=[cand_text])[0]
        alignscore_scores.append(score)
    
    if not alignscore_scores:
        print("No valid AlignScore scores computed")
        return None, None
    
    average_score = sum(alignscore_scores) / len(alignscore_scores)
    print(f"AlignScore ({reference} vs {candidate}): {average_score:.4f}")
    
    return average_score, alignscore_scores


def main():
    parser = argparse.ArgumentParser(description='Task Evaluation: ROUGE, F1, and AlignScore')
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
    
    # Evaluation parameters
    parser.add_argument('--reference', type=str, default='natural',
                       choices=['unwatermarked', 'natural'],
                       help='Reference text type')
    parser.add_argument('--candidate', type=str, default='watermarked',
                       choices=['watermarked', 'unwatermarked'],
                       help='Candidate text type')
    parser.add_argument('--standardize_length', action='store_true',
                       help='Standardize text lengths')
    parser.add_argument('--token_limit', type=int, default=100,
                       help='Token limit when standardizing length')
    
    # AlignScore specific parameters
    parser.add_argument('--alignscore_model', type=str, default='roberta-large',
                       choices=['roberta-base', 'roberta-large'],
                       help='AlignScore model type')
    parser.add_argument('--alignscore_ckpt', type=str, 
                       help='Path to AlignScore checkpoint')
    parser.add_argument('--evaluation_mode', type=str, default='nli_sp',
                       help='AlignScore evaluation mode')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device for AlignScore computation')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for AlignScore')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output file path for results')
    parser.add_argument('--rouge_only', action='store_true',
                       help='Calculate only ROUGE scores')
    parser.add_argument('--f1_only', action='store_true',
                       help='Calculate only F1 scores')
    parser.add_argument('--alignscore_only', action='store_true',
                       help='Calculate only AlignScore')
    parser.add_argument('--skip_alignscore', action='store_true',
                       help='Skip AlignScore calculation')
    
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
    
    # Set default AlignScore checkpoint path
    if args.alignscore_ckpt is None:
        if args.alignscore_model == 'roberta-base':
            args.alignscore_ckpt = os.path.join(PROJECT_ROOT, 'AlignScore', 'model', 'AlignScore-base.ckpt')
        else:
            args.alignscore_ckpt = os.path.join(PROJECT_ROOT, 'AlignScore', 'model', 'AlignScore-large.ckpt')
    
    print(f"Task Evaluation for {args.algorithm} {args.model} {args.dataset}")
    print(f"Parameters: {params}")
    print(f"Dump path: {args.dump_path}")
    print(f"Reference: {args.reference}, Candidate: {args.candidate}")
    
    results = {
        'algorithm': args.algorithm,
        'model': args.model,
        'dataset': args.dataset,
        'reference': args.reference,
        'candidate': args.candidate,
        **params
    }
    
    # Load texts once for all evaluations
    file_name = FileManager.build_filename(args.algorithm, params, args.model, args.dataset)
    watermark_file_path = f'{args.dump_path}/{file_name}'
    
    print(f"Loading data from: {watermark_file_path}")
    with open(watermark_file_path, 'rb') as f:
        watermarked_texts, unwatermarked_texts, natural_texts = pickle.load(f)
    
    # Calculate ROUGE scores
    if not args.f1_only and not args.alignscore_only:
        print("\n=== Calculating ROUGE Scores ===")
        try:
            (rouge2, rougeL), rouge_scores = get_rouge_score(
                dump_path=args.dump_path,
                algorithm=args.algorithm,
                model_type=args.model,
                dataset=args.dataset,
                params=params,
                reference=args.reference,
                candidate=args.candidate,
                standardize_length=args.standardize_length,
                token_limit=args.token_limit,
                watermarked_texts=watermarked_texts,
                unwatermarked_texts=unwatermarked_texts,
                natural_texts=natural_texts
            )
            results['rouge2'] = rouge2
            results['rougeL'] = rougeL
            results['rouge2_scores'] = rouge_scores[0]
            results['rougeL_scores'] = rouge_scores[1]
            print(f"ROUGE-2: {rouge2:.4f}")
            print(f"ROUGE-L: {rougeL:.4f}")
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            results['rouge_error'] = str(e)
    
    # Calculate F1 scores
    if not args.rouge_only and not args.alignscore_only:
        print("\n=== Calculating F1 Scores ===")
        try:
            f1_avg, f1_scores, wt_texts_dump = get_f1_score(
                dump_path=args.dump_path,
                algorithm=args.algorithm,
                model_type=args.model,
                dataset=args.dataset,
                params=params,
                reference=args.reference,
                candidate=args.candidate,
                watermarked_texts=watermarked_texts,
                unwatermarked_texts=unwatermarked_texts,
                natural_texts=natural_texts
            )
            results['f1'] = f1_avg
            results['f1_scores'] = f1_scores
            print(f"F1 Score: {f1_avg:.4f}")
        except Exception as e:
            print(f"Error calculating F1: {e}")
            results['f1_error'] = str(e)
    
    # Calculate AlignScore
    if not args.rouge_only and not args.f1_only and not args.skip_alignscore:
        print("\n=== Calculating AlignScore ===")
        try:
            alignscore_avg, alignscore_scores = get_alignscore_score(
                dump_path=args.dump_path,
                algorithm=args.algorithm,
                model_type=args.model,
                dataset=args.dataset,
                params=params,
                reference=args.reference,
                candidate=args.candidate,
                standardize_length=args.standardize_length,
                token_limit=args.token_limit,
                watermarked_texts=watermarked_texts,
                unwatermarked_texts=unwatermarked_texts,
                natural_texts=natural_texts,
                alignscore_model=args.alignscore_model,
                alignscore_ckpt=args.alignscore_ckpt,
                evaluation_mode=args.evaluation_mode,
                device=args.device,
                batch_size=args.batch_size
            )
            if alignscore_avg is not None:
                results['alignscore'] = alignscore_avg
                results['alignscore_scores'] = alignscore_scores
                print(f"AlignScore: {alignscore_avg:.4f}")
            else:
                results['alignscore_error'] = "Failed to compute AlignScore"
        except Exception as e:
            print(f"Error calculating AlignScore: {e}")
            results['alignscore_error'] = str(e)
    
    # Save results if output path specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\n=== Final Results ===")
    if 'rouge2' in results:
        print(f"ROUGE-2: {results['rouge2']:.4f}")
    if 'rougeL' in results:
        print(f"ROUGE-L: {results['rougeL']:.4f}")
    if 'f1' in results:
        print(f"F1 Score: {results['f1']:.4f}")
    if 'alignscore' in results:
        print(f"AlignScore: {results['alignscore']:.4f}")
    
    return results


if __name__ == "__main__":
    main()