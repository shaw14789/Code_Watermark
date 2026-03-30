#!/usr/bin/env python3
"""
Quality Evaluation Script

This script calculates perplexity and SimCSE similarity scores for watermarked text evaluation.
Based on the implementation in generate_all_NEW.ipynb.

Usage:
    python quality_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2
    python quality_evaluation.py --algorithm EXPEdit --model biomistral --dataset MEQS --nlength 256
    python quality_evaluation.py --algorithm DIP --model biomistral --dataset HQA --alpha 0.45
"""

import os
import gc
import json
import torch
import pickle
import argparse
import sys
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, 
                         set_seed, LlamaTokenizer, AutoModel)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from constants import PROJECT_ROOT
from our_utils import *
from evaluation.dataset import C4Dataset, ClinicDataset
from evaluation.tools.text_quality_analyzer import PPLCalculator


def calculate_perplexity(dump_path, algorithm, model_type, dataset, params=None, ntokens=None):
    """
    Calculate perplexity scores for generated texts.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_model_path = MODEL_PATHS['llama']
    
    tokenizer = LlamaTokenizer.from_pretrained(eval_model_path)
    analyzer = PPLCalculator(
        model=AutoModelForCausalLM.from_pretrained(eval_model_path, device_map='auto'),
        tokenizer=tokenizer,
        device='auto'
    )
    
    dataset_path = FileManager.get_dataset_path(dataset)
    file_name = FileManager.build_filename(algorithm, params, model_type, dataset)
    watermark_file_path = f'{dump_path}/{file_name}'
    
    print(f"Processing: {watermark_file_path}")
    
    with open(watermark_file_path, 'rb') as f:
        watermarked_texts, unwatermarked_texts, natural_texts = pickle.load(f)
    
    my_dataset = ClinicDataset(dataset_path)
    
    text_preprocessing_tokenizer = None
    
    # Calculate perplexity for unwatermarked texts
    uw_ppl_scores = []
    for text, prompt in tqdm(zip(unwatermarked_texts, my_dataset.prompts), desc="Unwatermarked Texts"):
        text = TextProcessor.remove_prompt(text, prompt, model_type, text_preprocessing_tokenizer)
        if not text:
            continue
            
        if ntokens is not None:
            text = TextProcessor.truncate_text(text, tokenizer, max_length=ntokens)
            
        uw_ppl_scores.append(analyzer.analyze(text))
    
    avg_uw_ppl = sum(uw_ppl_scores) / len(uw_ppl_scores) if uw_ppl_scores else 0
    print(f"Unwatermarked Perplexity: {avg_uw_ppl:.2f}")
    
    # Calculate perplexity for watermarked texts
    w_ppl_scores = []
    for text, prompt in tqdm(zip(watermarked_texts, my_dataset.prompts), desc="Watermarked Texts"):
        text = TextProcessor.remove_prompt(text, prompt, model_type, text_preprocessing_tokenizer)
        if len(text) < 5:
            continue
            
        if ntokens is not None:
            text = TextProcessor.truncate_text(text, tokenizer, max_length=ntokens)

        score = analyzer.analyze(text)
        w_ppl_scores.append(score)
    
    avg_w_ppl = sum(w_ppl_scores) / len(w_ppl_scores) if w_ppl_scores else 0
    print(f"Watermarked Perplexity: {avg_w_ppl:.2f}")
    
    return {
        'file': os.path.basename(watermark_file_path),
        'algorithm': algorithm,
        'model': model_type,
        'dataset': dataset,
        'unwatermarked_ppl': avg_uw_ppl,
        'watermarked_ppl': avg_w_ppl,
        'natural_ppl': 0,
        **params
    }


def get_simcse_similarity(dump_path, algorithm, model_type, dataset, params=None, 
                         reference="unwatermarked", candidate="watermarked", 
                         standardize_length=False, token_limit=100,
                         watermarked_texts=None, unwatermarked_texts=None, natural_texts=None):
    """
    Calculate semantic similarity using SimCSE embeddings.
    """
    simcse_path = "princeton-nlp/sup-simcse-roberta-base"
    model_simcse = AutoModel.from_pretrained(simcse_path)
    tokenizer_simcse = AutoTokenizer.from_pretrained(simcse_path)
    model_simcse.eval()
    
    for _, param in model_simcse.named_parameters():
        param.requires_grad = False
    
    dataset_path = FileManager.get_dataset_path(dataset)
    
    if watermarked_texts is None:
        file_name = FileManager.build_filename(algorithm, params, model_type, dataset)
        watermark_file_path = f'{dump_path}/{file_name}'
        with open(watermark_file_path, 'rb') as f:
            watermarked_texts, unwatermarked_texts, natural_texts = pickle.load(f)
        
        if (algorithm == "EXPEdit" and dataset == "MEQS" and model_type == "meditron") or (algorithm=="DIP"):
            temp_file_name = FileManager.build_filename("KGW", params, model_type, dataset)
            temp_watermark_file_path = f'{dump_path.replace(algorithm, "KGW")}/{temp_file_name}'
            with open(temp_watermark_file_path, 'rb') as f:
                _, unwatermarked_texts, _ = pickle.load(f)
    
    n = len(watermarked_texts)
    print(f"Samples: {n} watermarked, {len(unwatermarked_texts)} unwatermarked, {len(natural_texts)} natural")
    
    text_preprocessing_tokenizer = None
    
    simcse_scores = []
    my_dataset = ClinicDataset(dataset_path)
    
    for idtext in tqdm(range(n)):
        prompt = my_dataset.get_prompt(idtext)
        
        wt_text = TextProcessor.remove_prompt(
            watermarked_texts[idtext], 
            prompt, 
            model_type, 
            text_preprocessing_tokenizer
        )
        
        uwt_text = TextProcessor.remove_prompt(
            unwatermarked_texts[idtext], 
            prompt, 
            model_type, 
            text_preprocessing_tokenizer
        )
        
        nt_text = natural_texts[idtext]
        
        if standardize_length:
            wt_tokens = tokenizer_simcse.encode(wt_text, add_special_tokens=True)
            uwt_tokens = tokenizer_simcse.encode(uwt_text, add_special_tokens=True)
            nt_tokens = tokenizer_simcse.encode(nt_text, add_special_tokens=True)
            
            length = min(token_limit, len(wt_tokens), len(uwt_tokens), len(nt_tokens))
            
            wt_text = tokenizer_simcse.decode(wt_tokens[:length], skip_special_tokens=True)
            uwt_text = tokenizer_simcse.decode(uwt_tokens[:length], skip_special_tokens=True)
            nt_text = tokenizer_simcse.decode(nt_tokens[:length], skip_special_tokens=True)
        
        output_w_wm = tokenizer_simcse(wt_text, return_tensors="pt", padding=True)
        output_no_wm = tokenizer_simcse(uwt_text, return_tensors="pt", padding=True)
        output_nat = tokenizer_simcse(nt_text, return_tensors="pt", padding=True)
        
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        
        attention_masks_wm = torch.ones_like(output_w_wm['input_ids'])
        attention_masks_no_wm = torch.ones_like(output_no_wm['input_ids'])
        attention_masks_nat = torch.ones_like(output_nat['input_ids'])
        
        embed_wm = model_simcse(
            output_w_wm['input_ids'],
            attention_mask=attention_masks_wm,
            output_hidden_states=True,
            return_dict=True
        ).pooler_output
        
        embed_no_wm = model_simcse(
            output_no_wm['input_ids'],
            attention_mask=attention_masks_no_wm,
            output_hidden_states=True,
            return_dict=True
        ).pooler_output
        
        embed_nat = model_simcse(
            output_nat['input_ids'],
            attention_mask=attention_masks_nat,
            output_hidden_states=True,
            return_dict=True
        ).pooler_output
        
        if reference == "unwatermarked" and candidate == "watermarked":
            simcse_scores.append(cos(embed_no_wm[0], embed_wm[0]).item())
        elif reference == "natural" and candidate == "watermarked":
            simcse_scores.append(cos(embed_nat[0], embed_wm[0]).item())
        elif reference == "natural" and candidate == "unwatermarked":
            simcse_scores.append(cos(embed_nat[0], embed_no_wm[0]).item())
    
    return sum(simcse_scores) / n, simcse_scores


def main():
    parser = argparse.ArgumentParser(description='Quality Evaluation: Perplexity and SimCSE')
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
    parser.add_argument('--ntokens', type=int, help='Number of tokens to evaluate (truncate texts)')
    parser.add_argument('--reference', type=str, default='unwatermarked',
                       choices=['unwatermarked', 'natural'],
                       help='Reference text type for SimCSE')
    parser.add_argument('--candidate', type=str, default='watermarked',
                       choices=['watermarked', 'unwatermarked'],
                       help='Candidate text type for SimCSE')
    parser.add_argument('--standardize_length', action='store_true',
                       help='Standardize text lengths for SimCSE')
    parser.add_argument('--token_limit', type=int, default=100,
                       help='Token limit when standardizing length')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output file path for results')
    parser.add_argument('--perplexity_only', action='store_true',
                       help='Calculate only perplexity')
    parser.add_argument('--simcse_only', action='store_true',
                       help='Calculate only SimCSE')
    
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
    
    print(f"Quality Evaluation for {args.algorithm} {args.model} {args.dataset}")
    print(f"Parameters: {params}")
    print(f"Dump path: {args.dump_path}")
    
    results = {}
    
    # Calculate perplexity
    if not args.simcse_only:
        print("\n=== Calculating Perplexity ===")
        try:
            ppl_results = calculate_perplexity(
                dump_path=args.dump_path,
                algorithm=args.algorithm,
                model_type=args.model,
                dataset=args.dataset,
                params=params,
                ntokens=args.ntokens
            )
            results.update(ppl_results)
            print(f"Perplexity results: {ppl_results}")
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            results['perplexity_error'] = str(e)
    
    # Calculate SimCSE
    if not args.perplexity_only:
        print("\n=== Calculating SimCSE Similarity ===")
        try:
            avg_simcse, simcse_scores = get_simcse_similarity(
                dump_path=args.dump_path,
                algorithm=args.algorithm,
                model_type=args.model,
                dataset=args.dataset,
                params=params,
                reference=args.reference,
                candidate=args.candidate,
                standardize_length=args.standardize_length,
                token_limit=args.token_limit
            )
            results['avg_simcse'] = avg_simcse
            results['simcse_scores'] = simcse_scores
            results['simcse_reference'] = args.reference
            results['simcse_candidate'] = args.candidate
            print(f"Average SimCSE similarity: {avg_simcse:.4f}")
        except Exception as e:
            print(f"Error calculating SimCSE: {e}")
            results['simcse_error'] = str(e)
    
    # Save results if output path specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\n=== Final Results ===")
    if 'unwatermarked_ppl' in results:
        print(f"Unwatermarked Perplexity: {results['unwatermarked_ppl']:.2f}")
    if 'watermarked_ppl' in results:
        print(f"Watermarked Perplexity: {results['watermarked_ppl']:.2f}")
    if 'avg_simcse' in results:
        print(f"SimCSE Similarity ({args.reference} vs {args.candidate}): {results['avg_simcse']:.4f}")
    
    return results


if __name__ == "__main__":
    main()