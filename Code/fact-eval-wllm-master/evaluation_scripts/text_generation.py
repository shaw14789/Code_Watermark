#!/usr/bin/env python3
"""
Text Generation Script

This script generates watermarked and unwatermarked text using various watermarking algorithms.
Based on the implementation from generate_all_NEW.ipynb.

Usage:
    python text_generation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2
    python text_generation.py --algorithm EXPEdit --model biomistral --dataset MEQS --nlength 256
    python text_generation.py --algorithm DIP --model biomistral --dataset HQA --alpha 0.45
    python text_generation.py --algorithm SWEET --model jsl --dataset HQA --gamma 0.5 --delta 2 --entropy 0.9
"""

import os
import gc
import json
import torch
import pickle
import argparse
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, 
                         set_seed, LlamaTokenizer)

# Add parent directory to path to import utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from constants import PROJECT_ROOT
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark
from evaluation.dataset import C4Dataset, ClinicDataset
from our_utils import *

DEFAULT_DEVICE = 'auto'

def generate_watermarked_text(algorithm, model_type, dataset, params=None, device=DEFAULT_DEVICE, 
                            prefix_output="fixed", samples=200000):
    """
    Generate and save watermarked and unwatermarked text.
    
    Args:
        algorithm: Watermarking algorithm (KGW, SWEET, EXPEdit, DIP)
        model_type: Model type (opt, llama, meditron, jsl, biomistral)
        dataset: Dataset name (C4, HQA, MIMIC, HQA2, MEQS)
        params: Algorithm parameters dictionary
        device: Device to use for generation
        prefix_output: Output folder prefix
        samples: Number of samples to generate
    """
    print(f"Starting generation for {algorithm} with {model_type} on {dataset}")
    set_seed(113)
    
    # Load dataset
    dataset_path = FileManager.get_dataset_path(dataset)
    if dataset.lower() == 'c4':
        my_dataset = C4Dataset(dataset_path)
    else:
        my_dataset = ClinicDataset(dataset_path)
    
    # Setup model parameters
    max_new_tokens = 200
    min_length = 230
    if dataset.lower() == 'hqa2':
        min_length = 210
    if dataset.lower() == 'meqs':
        max_new_tokens = 25
        min_length = 5
    
    print(f"Using max_new_tokens={max_new_tokens}, min_length={min_length}")
    
    # Load model configuration
    transformers_config = ModelLoader.load_model(
        model_type, 
        device, 
        max_new_tokens, 
        min_length
    )
    
    # Load and update algorithm configuration
    config = ConfigManager.load_config(algorithm, params)
    ConfigManager.save_config(algorithm, config)
    
    # Initialize watermark
    config_path = os.path.join(PROJECT_ROOT, "config", f"{algorithm}.json")
    my_watermark = AutoWatermark.load(
        algorithm, 
        algorithm_config=config_path, 
        transformers_config=transformers_config
    )
    
    print("CONFIG: ", end="")
    my_watermark.print_config()

    # Generate unwatermarked text first (IMPORTANT: Don't change the order)
    print("Generating unwatermarked texts...")
    unwatermarked_texts = []
    for i in tqdm(range(min(samples, my_dataset.prompt_nums)), desc="Unwatermarked"):
        unwatermarked_text = my_watermark.generate_unwatermarked_text(my_dataset.get_prompt(i))
        unwatermarked_texts.append(unwatermarked_text)
    
    # Generate watermarked text
    print("Generating watermarked texts...")
    watermarked_texts = []
    for i in tqdm(range(min(samples, my_dataset.prompt_nums)), desc="Watermarked"):
        watermarked_text = my_watermark.generate_watermarked_text(my_dataset.get_prompt(i))
        watermarked_texts.append(watermarked_text)
    
    # Get natural texts
    print("Loading natural texts...")
    natural_texts = []
    for i in tqdm(range(min(samples, my_dataset.natural_text_nums)), desc="Natural"):
        natural_texts.append(my_dataset.get_natural_text(i))
    
    # Print samples
    print("\n--- SAMPLE OUTPUTS ---")
    print("UNWATERMARKED TEXT:", unwatermarked_texts[0][:200] + "..." if len(unwatermarked_texts[0]) > 200 else unwatermarked_texts[0])
    print("\nWATERMARKED TEXT:", watermarked_texts[0][:200] + "..." if len(watermarked_texts[0]) > 200 else watermarked_texts[0])
    print("\nNATURAL TEXT:", natural_texts[0][:200] + "..." if len(natural_texts[0]) > 200 else natural_texts[0])
    
    # Save results
    folder_name = algorithm
    file_name = FileManager.build_filename(
        algorithm,
        params,
        model_type,
        dataset
    )
    
    output_dir = f'logs/{prefix_output}-{folder_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save pickle file
    pickle_path = f'{output_dir}/{file_name}'
    with open(pickle_path, 'wb') as f:
        pickle.dump((watermarked_texts, unwatermarked_texts, natural_texts), f)
    
    print(f"Saved pickle data to: {pickle_path}")
    
    # Save JSON file for easy inspection
    json_data = []
    for i in range(min(samples, my_dataset.prompt_nums)):
        data_point = {
            "prompt": my_dataset.prompts[i] if i < len(my_dataset.prompts) else None,
            "watermarked_text": watermarked_texts[i] if i < len(watermarked_texts) else None,
            "unwatermarked_text": unwatermarked_texts[i] if i < len(unwatermarked_texts) else None,
            "natural_text": natural_texts[i] if i < len(natural_texts) else None
        }
        json_data.append(data_point)
    
    json_path = f'{output_dir}/{file_name}.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    print(f"Saved JSON data to: {json_path}")
    
    # Print statistics
    print(f"\n--- GENERATION STATISTICS ---")
    print(f"Generated {len(watermarked_texts)} watermarked texts")
    print(f"Generated {len(unwatermarked_texts)} unwatermarked texts") 
    print(f"Loaded {len(natural_texts)} natural texts")
    
    # Clean up
    del my_watermark
    torch.cuda.empty_cache()
    gc.collect()
    
    return pickle_path, json_path


def main():
    parser = argparse.ArgumentParser(description='Generate watermarked text using various algorithms')
    
    # Required arguments
    parser.add_argument('--algorithm', required=True, choices=['KGW', 'SWEET', 'EXPEdit', 'DIP'],
                       help='Watermarking algorithm')
    parser.add_argument('--model', required=True, 
                       choices=['opt', 'llama', 'meditron', 'jsl', 'biomistral'],
                       help='Model type')
    parser.add_argument('--dataset', required=True, choices=['C4', 'HQA', 'MIMIC', 'HQA2', 'MEQS'],
                       help='Dataset name')
    
    # Algorithm-specific parameters
    parser.add_argument('--gamma', type=float, help='Gamma parameter for KGW/SWEET')
    parser.add_argument('--delta', type=int, help='Delta parameter for KGW/SWEET')
    parser.add_argument('--entropy', type=float, default=0.9, help='Entropy parameter for SWEET')
    parser.add_argument('--nlength', type=int, help='N-length parameter for EXPEdit')
    parser.add_argument('--alpha', type=float, help='Alpha parameter for DIP')
    
    # Optional arguments
    parser.add_argument('--samples', type=int, default=200000, help='Number of samples to generate')
    parser.add_argument('--device', default='auto', help='Device to use for generation')
    parser.add_argument('--prefix_output', default='fixed', help='Output folder prefix')
    parser.add_argument('--output', help='Save summary to JSON file')
    
    args = parser.parse_args()
    
    # Build parameter dictionary based on algorithm
    params = {}
    if args.algorithm in ['KGW', 'SWEET']:
        if args.gamma is None or args.delta is None:
            parser.error(f'{args.algorithm} requires --gamma and --delta parameters')
        params['gamma'] = args.gamma
        params['delta'] = args.delta
        if args.algorithm == 'SWEET':
            params['entropy'] = args.entropy
    elif args.algorithm == 'EXPEdit':
        if args.nlength is None:
            parser.error('EXPEdit requires --nlength parameter')
        params['nlength'] = args.nlength
    elif args.algorithm == 'DIP':
        if args.alpha is None:
            parser.error('DIP requires --alpha parameter')
        params['alpha'] = args.alpha
    
    # Generate text
    try:
        pickle_path, json_path = generate_watermarked_text(
            algorithm=args.algorithm,
            model_type=args.model,
            dataset=args.dataset,
            params=params,
            device=args.device,
            prefix_output=args.prefix_output,
            samples=args.samples
        )
        
        # Save summary if requested
        if args.output:
            summary = {
                'algorithm': args.algorithm,
                'model': args.model,
                'dataset': args.dataset,
                'parameters': params,
                'samples_generated': args.samples,
                'pickle_file': pickle_path,
                'json_file': json_path,
                'status': 'completed'
            }
            
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=4)
            
            print(f"Summary saved to: {args.output}")
        
        print(f"\n✓ Generation completed successfully!")
        print(f"Pickle file: {pickle_path}")
        print(f"JSON file: {json_path}")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        if args.output:
            summary = {
                'algorithm': args.algorithm,
                'model': args.model,
                'dataset': args.dataset,
                'parameters': params,
                'samples_generated': 0,
                'error': str(e),
                'status': 'failed'
            }
            
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=4)
        
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())