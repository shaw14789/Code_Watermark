from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TCDWConfig:
    # paths
    gen_model_path: str = "/share/sda1/FX+ZK/Code-fx/Watermark/WaterMark/Model/meditron-7b"
    nli_path: str = "/share/sda1/FX+ZK/Code-fx/Watermark/WaterMark/Model/nli_model"
    pqa_data_path: str = "/share/sda1/FX+ZK/Code-fx/Watermark/WaterMark/dataset/PubMedQA/pqa_labeled"
    output_csv: str = "tcdw_unified_results.csv"

    # experiment mode
    experiment_mode: str = "wm_span"   # no_wm / wm_only / wm_span

    # generation
    max_new_tokens: int = 48
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True

    # watermark
    gamma: float = 0.5
    seed_offset: int = 42
    watermark_bias: float = 2.0

    # span-aware boost
    anchor_start_boost: float = 0.8
    anchor_continue_boost: float = 5.0
    anchor_finish_boost: float = 7.0

    # prompt
    prompt_template: str = (
        "Question: {question}\n"
        "Context: {context}\n"
        "Answer in one concise sentence strictly supported by the context: "
    )
    force_bos_token_id: Optional[int] = 1

    # runtime
    num_samples: int = 100
    fp16: bool = True
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # spacy
    scispacy_model: str = "en_core_sci_lg"
