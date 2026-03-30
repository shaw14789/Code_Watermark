import os
import gc
import json
import torch
import pickle
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, 
                         set_seed, LlamaTokenizer)
from rouge_score import rouge_scorer
from constants import PROJECT_ROOT
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark
from evaluation.dataset import C4Dataset, ClinicDataset
from evaluation.tools.text_editor import TruncatePromptTextEditor
from evaluation.tools.text_quality_analyzer import PPLCalculator
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from evaluation.pipelines.detection import (WatermarkedTextDetectionPipeline, 
                                          UnWatermarkedTextDetectionPipeline, 
                                          DetectionPipelineReturnType)

# Global constants for paths
MODEL_PATHS = {
    "opt": "facebook/opt-1.3b",
    "llama": "meta-llama/Llama-2-7b-hf",
    "meditron": "epfl-llm/meditron-7b",
    "jsl":"johnsnowlabs/JSL-MedLlama-3-8B-v2.0",
    "biomistral":"BioMistral/BioMistral-7B"
}

DATASET_PATHS = {
    "hqa": os.path.join(PROJECT_ROOT, "dataset", "hqa", "hqa_processed_230words.json"),
    "hqa2": os.path.join(PROJECT_ROOT, "dataset", "hqa2", "hqa2_processed_4.json"),
    "meqs": os.path.join(PROJECT_ROOT, "dataset", "meqsum", "meqsum_processed.json")
}

# Default device
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class ModelLoader:
    """Handles consistent model loading across the project."""
    
    @staticmethod
    def load_model(model_type, device=DEFAULT_DEVICE, max_new_tokens=200, min_length=230):
        """
        Load a model and its tokenizer with consistent parameters.
        
        Args:
            model_type: String identifier for model (opt, llama, meditron, jsl, biomistral)
            device: Device to load model to
            max_new_tokens: Maximum number of tokens to generate
            min_length: Minimum length for generation
            
        Returns:
            TransformersConfig object with loaded model and tokenizer
        """
        model_path = MODEL_PATHS.get(model_type)
        if not model_path:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Adjust min_length based on dataset/model
        if model_type == "meditron" and min_length > 210:
            min_length = 210
            
        # Load model based on type
        if model_type == "opt":
            return TransformersConfig(
                model=AutoModelForCausalLM.from_pretrained(model_path).to(device),
                tokenizer=AutoTokenizer.from_pretrained(model_path),
                vocab_size=50272,
                device=device,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                do_sample=True,
                no_repeat_ngram_size=4
            )
        elif model_type == "meditron":
            return TransformersConfig(
                model=AutoModelForCausalLM.from_pretrained(model_path, device_map="auto"),
                tokenizer=AutoTokenizer.from_pretrained(model_path, device_map="auto"),
                vocab_size=32000,
                device=device,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                do_sample=True,
                no_repeat_ngram_size=4,
                pad_token_id=AutoTokenizer.from_pretrained(model_path).eos_token_id
            )
        elif model_type == "jsl":
            return TransformersConfig(
                model=AutoModelForCausalLM.from_pretrained(model_path, device_map="auto"),
                tokenizer=AutoTokenizer.from_pretrained(model_path, device_map="auto"),
                vocab_size=128256,
                device=device,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                do_sample=True,
                no_repeat_ngram_size=4,
                pad_token_id=AutoTokenizer.from_pretrained(model_path).eos_token_id
            )
        elif model_type == "biomistral":
            return TransformersConfig(
                model=AutoModelForCausalLM.from_pretrained(model_path, device_map="auto"),
                tokenizer=AutoTokenizer.from_pretrained(model_path, device_map="auto"),
                vocab_size=32000,
                device=device,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                do_sample=True,
                no_repeat_ngram_size=4,
                pad_token_id=AutoTokenizer.from_pretrained(model_path).eos_token_id
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class TextProcessor:
    """Handles text preprocessing consistently across the project."""
    
    @staticmethod
    def remove_prompt(text, prompt, model_type=None, tokenizer=None):
        """
        Remove prompt from generated text consistently.
        
        Args:
            text: The full text (prompt + generated)
            prompt: The prompt to remove
            model_type: The model type (for biogpt special handling)
            tokenizer: Optional tokenizer (required for biogpt)
            
        Returns:
            Text with prompt removed
        """
        text_editor = TruncatePromptTextEditor()
        return text_editor.edit(text, prompt)
    
    @staticmethod
    def truncate_text(text, tokenizer, max_length=50):
        """
        Truncate text to a specified token length.
        
        Args:
            text: Text to truncate
            tokenizer: Tokenizer to use
            max_length: Maximum token length
            
        Returns:
            Truncated text
        """
        tokens = tokenizer(text, return_tensors="pt", padding=True, 
                          truncation=True, max_length=max_length)['input_ids']
        return tokenizer.decode(tokens[0], skip_special_tokens=True)


class ConfigManager:
    """Manages algorithm configurations consistently."""
    
    @staticmethod
    def load_config(algorithm, params=None):
        """
        Load and update configuration from file.
        
        Args:
            algorithm: Algorithm name (e.g., "SWEET", "KGW")
            params: Dictionary of parameters to update
            
        Returns:
            Updated configuration dictionary
        """
        config_path = os.path.join(PROJECT_ROOT, "config", f"{algorithm}.json")
        with open(config_path) as f:
            config = json.load(f)
            
        # Update configuration with provided parameters
        if params:
            for key, value in params.items():
                if value is not None:
                    if key == "nlength":
                        config["pseudo_length"] = value
                    else:
                        config[key] = value
                    
        # Special handling for specific algorithms/datasets
        if algorithm == "EXPEdit" and params.get("dataset") == "MEQS":
            config["sequence_length"] = 25
            
        return config
    
    @staticmethod
    def save_config(algorithm, config):
        """
        Save configuration to file.
        
        Args:
            algorithm: Algorithm name
            config: Configuration dictionary
        """
        config_path = os.path.join(PROJECT_ROOT, "config", f"{algorithm}.json")
        with open(config_path, "w") as f:
            json.dump(config, f)


class FileManager:
    """Handles file operations consistently."""
    
    @staticmethod
    def get_dataset_path(dataset):
        """Get dataset path from identifier."""
        normalized_dataset = dataset.lower()
        return DATASET_PATHS.get(normalized_dataset)
    
    @staticmethod
    def get_model_path(model):
        """Get model path from identifier."""
        return MODEL_PATHS.get(model.lower())
    
    @staticmethod
    def build_filename(algorithm, params, model, dataset):
        """
        Build consistent filename based on parameters.
        
        Args:
            algorithm: Algorithm name
            params: Algorithm parameters
            model: Model identifier
            dataset: Dataset identifier
            
        Returns:
            Constructed filename
        """
        def format_param(value):
            """Format parameter value to remove unnecessary decimal places"""
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            return str(value)
        
        if algorithm == "KGW":
            return f"{algorithm}-g{format_param(params['gamma'])}-d{format_param(params['delta'])}-{model}-{dataset}.pkl"
        elif algorithm == "SWEET":
            return f"{algorithm}-e{format_param(params['entropy'])}-g{format_param(params['gamma'])}-d{format_param(params['delta'])}-{model}-{dataset}.pkl"
        elif algorithm == "EXPEdit":
            return f"{algorithm}-n{format_param(params['nlength'])}-{model}-{dataset}.pkl"
        elif algorithm == "PF":
            return f"{algorithm}-{model}-{dataset}.pkl"
        elif algorithm == "DIP":
            return f"{algorithm}-a{format_param(params['alpha'])}-{model}-{dataset}.pkl"
        elif algorithm == "SIR":
            return f"{algorithm}-{model}-{dataset}.pkl"
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    @staticmethod
    def parse_filename(filename):
        """
        Parse parameters from filename.
        
        Args:
            filename: Filename to parse
            
        Returns:
            Dictionary of extracted parameters
        """
        parts = filename.split('-')
        if "KGW" in parts:
            return {
                'algorithm': parts[0],
                'gamma': float(parts[1][1:]),
                'delta': float(parts[2][1:]),
                'model': parts[3],
                'dataset': parts[4].split('.')[0]
            }
        elif "SWEET" in parts:
            return {
                'algorithm': parts[0],
                'entropy': float(parts[1][1:]),
                'gamma': float(parts[2][1:]),
                'delta': float(parts[3][1:]),
                'model': parts[4],
                'dataset': parts[5].split('.')[0]
            }
        elif "EXPEdit" in parts:
            return {
                'algorithm': parts[0],
                'nlength': int(parts[1][1:]),
                'model': parts[2],
                'dataset': parts[3].split('.')[0]
            }
        elif "PF" in parts:
            return {
                'algorithm': parts[0],
                'model': parts[1],
                'dataset': parts[2].split('.')[0]
            }
        elif "DIP" in parts:
            return {
                'algorithm': parts[0],
                'alpha': float(parts[1][1:]),
                'model': parts[2],
                'dataset': parts[3].split('.')[0]
            }
        elif "SIR" in parts:
            return {
                'algorithm': parts[0],
                'model': parts[1],
                'dataset': parts[2].split('.')[0]
            }


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)
