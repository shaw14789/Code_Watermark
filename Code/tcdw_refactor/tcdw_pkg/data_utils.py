import os
import csv
import math
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk, load_dataset


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def smart_load_medical_data(path: str):
    if os.path.exists(os.path.join(path, "dataset_info.json")):
        print(">>> 检测到 Arrow/save_to_disk 格式，使用 load_from_disk 加载")
        return load_from_disk(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"路径不存在: {path}")

    files = os.listdir(path)
    for f in files:
        file_path = os.path.join(path, f)
        if f.endswith(".parquet"):
            print(f">>> 检测到 parquet: {f}")
            ds = load_dataset("parquet", data_files=file_path)
            return ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        if f.endswith(".jsonl") or f.endswith(".json"):
            print(f">>> 检测到 json/jsonl: {f}")
            ds = load_dataset("json", data_files=file_path)
            return ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    raise FileNotFoundError(f"目录 {path} 下未找到可识别的数据文件")


def normalize_pubmedqa_item(item: Dict[str, Any]) -> Tuple[str, str]:
    question = item.get("question", "")
    context = (
        item.get("long_answer", None)
        or item.get("context", None)
        or item.get("contexts", None)
        or item.get("answer", None)
        or ""
    )

    if isinstance(context, list):
        context = " ".join([str(x) for x in context])
    elif isinstance(context, dict):
        context = str(context)

    if not isinstance(question, str):
        question = str(question)
    if not isinstance(context, str):
        context = str(context)

    return question, context


def _green_list_from_prev(
    prev_token_id: int,
    vocab_size: int,
    gamma: float,
    seed_offset: int,
    device: str,
) -> set:
    dev = torch.device(device)
    g = torch.Generator(device=dev)
    g.manual_seed(int(prev_token_id) + seed_offset)
    perm = torch.randperm(vocab_size, generator=g, device=dev)
    green_size = int(vocab_size * gamma)
    return set(perm[:green_size].cpu().tolist())


def detect_z_from_token_ids(
    prompt_ids: List[int],
    generated_ids: List[int],
    vocab_size: int,
    gamma: float,
    seed_offset: int,
    device: str,
) -> Dict[str, Any]:
    if len(prompt_ids) == 0 or len(generated_ids) == 0:
        return {
            "green_hits": 0,
            "green_total": 0,
            "green_ratio": 0.0,
            "z_score": 0.0,
        }

    prev_id = prompt_ids[-1]
    hits = 0
    total = 0

    for curr_id in generated_ids:
        green = _green_list_from_prev(prev_id, vocab_size, gamma, seed_offset, device)
        if curr_id in green:
            hits += 1
        total += 1
        prev_id = curr_id

    expected = total * gamma
    var = total * gamma * (1.0 - gamma)
    z = (hits - expected) / (math.sqrt(var) + 1e-6)

    return {
        "green_hits": hits,
        "green_total": total,
        "green_ratio": hits / total if total > 0 else 0.0,
        "z_score": float(z),
    }
