# -*- coding: utf-8 -*-

import os
import os
import re
import csv
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from medical_claim_extractor import extract_must_have_claims, claims_to_anchor_texts
from medical_claim_scorer import score_must_have_claims
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    LogitsProcessor,
    LogitsProcessorList,
)

import spacy

import random
import numpy as np
import torch

#统一设种子的函数
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =========================
# 1. 配置
# =========================

@dataclass
class TCDWConfig:
    # paths
    llama_path: str ="/share/sda1/FX+ZK/Code-fx/Watermark/WaterMark/Model/meditron-7b" 
    #nli_path: str = "/share/sda1/FX+ZK/Code-fx/Watermark/WaterMark/Model/nli_model"
    nli_path: str = "/share/sda1/FX+ZK/Code-fx/Watermark/WaterMark/0320/nli_model_offline"
    pqa_data_path: str = "/share/sda1/FX+ZK/Code-fx/Watermark/WaterMark/dataset/PubMedQA/pqa_labeled"
    output_csv: str = "tcdw_unified_results_compare_v1.csv"
    # ablation mode
    # 可选: "no_wm", "wm_only", "wm_span"
    experiment_mode: str = "wm_span"

    # generation
    max_new_tokens: int = 48
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True


    # watermark
    gamma: float = 0.5
    seed_offset: int = 42
    watermark_bias: float = 2.0
    random_seed: int = 42

    # factual anchors
    anchor_boost: float = 8.0
    max_anchors: int = 16

    # =========================
    # 【新增】span-aware anchor 参数
    # 说明：原来的 anchor_boost 是“扁平 token 级硬拉”；
    # 这里新增 start / continue / finish 三段式 boost，
    # 用于把 anchor 从 token bag 升级成 span-aware 状态机。
    # =========================
    anchor_start_boost: float = 1.0
    anchor_continue_boost: float = 4.0#前一版参数是3.5
    anchor_finish_boost: float = 6.0#前一版参数是5.0

    # prompt
    prompt_template: str = "Question: {question}\nContext: {context}\nAnswer: "
    # prompt_template = (
    # "Question: {question}\n"
    # "Context: {context}\n"
    # "Answer in one concise sentence strictly supported by the context: "
    # )
    force_bos_token_id: Optional[int] = 1   # Llama 常见 BOS = 1，若 tokenizer 自带则可不手工强加

    # runtime
    num_samples: int = 150
    fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # spacy
    scispacy_model: str = "en_core_sci_lg"  # 本地已安装时使用


# =========================
# 2. 数据加载
# =========================

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


# =========================
# 3. Anchor 提取
# =========================

class MedicalAnchorExtractor:
    def __init__(self, cfg: TCDWConfig, tokenizer: AutoTokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.nlp = spacy.load(cfg.scispacy_model)

        # 可继续扩展
        self.regex_patterns = [
            r"\b\d+(\.\d+)?\s?(mg|g|ml|mcg|kg)\b",
            r"\b\d+(\.\d+)?\s?(day|days|week|weeks|month|months|year|years)\b",
            r"\b\d+(\.\d+)?%\b",
        ]



    def extract_anchor_texts(self, text: str) -> List[str]:
        anchors = []

        if not isinstance(text, str) or not text.strip():
            return anchors

        doc = self.nlp(text)

        # 1) NER
        for ent in doc.ents:
            val = ent.text.strip()
            if val and len(val) > 1:
                anchors.append(val)

        # 2) regex 补充
        # 原逻辑保留（re.findall 有分组时会返回 tuple）
        for pat in self.regex_patterns:
            anchors.extend(re.findall(pat, text))

        # re.findall 有分组时可能返回 tuple
        cleaned = []
        for a in anchors:
            if isinstance(a, tuple):
                a = "".join(a).strip()
            if isinstance(a, str):
                a = a.strip()
                if a:
                    cleaned.append(a)

        # 去重 + 保序
        deduped = []
        seen = set()
        for a in cleaned:
            low = a.lower()
            if low not in seen:
                seen.add(low)
                deduped.append(a)

        return deduped[: self.cfg.max_anchors]

    def anchor_texts_to_token_ids(self, anchor_texts: List[str]) -> List[int]:
        # =========================
        # 【保留旧逻辑，仅作对比】
        # 旧逻辑：把所有 anchor 文本打平成 token ids 列表，
        # 这会导致后续只是在做 token-level boost，而不是 span-level 约束。
        # =========================
        token_ids = []
        for a in anchor_texts:
            ids = self.tokenizer.encode(" " + a, add_special_tokens=False)
            token_ids.extend(ids)

        # token 级去重
        deduped = []
        seen = set()
        for tid in token_ids:
            if tid not in seen:
                seen.add(tid)
                deduped.append(tid)

        return deduped

    # =========================
    # 【新增】span-aware 版本：返回 List[List[int]]
    # 说明：每个 anchor 文本保留为一个 token 序列，
    # 例如 "10 mg daily" -> [tid1, tid2, tid3]
    # 这样后续就能按“短语进度”做状态机式约束，而不是 token bag。
    # =========================
    def anchor_texts_to_token_seqs(self, anchor_texts: List[str]) -> List[List[int]]:
        anchor_token_seqs = []
        for a in anchor_texts:
            ids = self.tokenizer.encode(" " + a, add_special_tokens=False)
            if len(ids) > 0:
                anchor_token_seqs.append(ids)
        return anchor_token_seqs


# =========================
# 4. LogitsProcessor
# =========================

class TCDWMedicalLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        gamma: float,
        seed_offset: int,
        watermark_bias: float,
        anchor_ids: List[int],
        anchor_boost: float,
        # =========================
        # 【新增】span-aware 参数
        # 说明：保留旧参数不删，便于你对比；
        # 真正启用的是下面这组 anchor_token_seqs + 三段式 boost。
        # =========================
        anchor_token_seqs: Optional[List[List[int]]] = None,
        start_boost: float = 1.0,
        continue_boost: float = 3.5,
        finish_boost: float = 5.0,
        prompt_length: int = 0, 

    ):

        self.gamma = gamma
        self.seed_offset = seed_offset
        self.watermark_bias = watermark_bias
        # 旧版 token-flat 参数保留，便于对照
        self.anchor_ids = anchor_ids
        self.anchor_boost = anchor_boost

        # =========================
        # 【新增】保存 span-aware 状态机所需字段
        # =========================
        self.anchor_token_seqs = anchor_token_seqs if anchor_token_seqs is not None else []
        self.start_boost = start_boost
        self.continue_boost = continue_boost
        self.finish_boost = finish_boost
        self.prompt_length = prompt_length

        # =========================
        # 【新增】span-aware 诊断统计
        # =========================
        self.num_steps = 0
        self.num_span_boost_steps = 0
        self.num_empty_boost_steps = 0
        self.num_start_boost_hits = 0
        self.num_continue_boost_hits = 0
        self.num_finish_boost_hits = 0
        self.total_next_token_candidates = 0
        # 用于记录哪些 anchor 最终完整出
        self.completed_anchor_spans = set()   

    def _get_green_list(self, prev_token_id: int, vocab_size: int, device: torch.device) -> torch.Tensor:
        # 局部 generator，避免污染全局随机状态
        g = torch.Generator(device=device)
        g.manual_seed(int(prev_token_id) + self.seed_offset)
        perm = torch.randperm(vocab_size, generator=g, device=device)
        green_size = int(vocab_size * self.gamma)
        return perm[:green_size]

    # =========================
    # 【新增】辅助函数1：判断某个 anchor span 是否已经完整出现过
    # 说明：一旦某个 anchor 已完整出现，就不再继续推它，
    # 避免重复生成同一短语。
    # =========================
    def _is_seq_satisfied(self, generated_ids: List[int], target_seq: List[int]) -> bool:
        n = len(target_seq)
        if n == 0 or len(generated_ids) < n:
            return False
        for i in range(len(generated_ids) - n + 1):
            if generated_ids[i:i+n] == target_seq:
                return True
        return False

    # =========================
    # 【新增】辅助函数2：计算 suffix-prefix 最大匹配长度
    # 说明：用来判断“当前已经匹配到 anchor 的第几步”。
    # 例如 generated 结尾是 [10, 23]，target 是 [10, 23, 45]，
    # 则返回 2，表示下一步该鼓励 45。
    # =========================
    def _matched_prefix_len(self, generated_ids: List[int], target_seq: List[int]) -> int:
        max_k = min(len(generated_ids), len(target_seq))
        for k in range(max_k, 0, -1):
            if generated_ids[-k:] == target_seq[:k]:
                return k
        return 0

    def get_diagnostics(self) -> Dict[str, Any]:
        avg_candidates = (
            self.total_next_token_candidates / self.num_span_boost_steps
            if self.num_span_boost_steps > 0 else 0.0
        )

        return {
            "num_steps": self.num_steps,
            "num_span_boost_steps": self.num_span_boost_steps,
            "num_empty_boost_steps": self.num_empty_boost_steps,
            "span_boost_step_ratio": (
                self.num_span_boost_steps / self.num_steps if self.num_steps > 0 else 0.0
            ),
            "num_start_boost_hits": self.num_start_boost_hits,
            "num_continue_boost_hits": self.num_continue_boost_hits,
            "num_finish_boost_hits": self.num_finish_boost_hits,
            "avg_next_token_candidates": avg_candidates,
            "num_completed_anchor_spans": len(self.completed_anchor_spans),
            "num_anchor_token_seqs": len(self.anchor_token_seqs),
            "avg_anchor_seq_len": (
                sum(len(seq) for seq in self.anchor_token_seqs) / len(self.anchor_token_seqs)
                if len(self.anchor_token_seqs) > 0 else 0.0
            ),
        }

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 当前只处理 batch=1，更直观；如果你后续要扩 batch，可以再矢量化
        if input_ids.size(0) != 1:
            return scores

        vocab_size = scores.size(-1)
        device = scores.device


        full_ids = input_ids[0].tolist()
        # =========================
        # 【关键修复】span-aware 只看新生成部分
        # =========================
        generated = full_ids[self.prompt_length:]
        # watermark 仍然按真正上一 token 来走
        prev_token_id = int(input_ids[0, -1].item())

        # ======================================
        # 【保留旧逻辑，仅作对比】
        # 旧逻辑：token-level watermark + 扁平 token boost
        # 注意：下面整段旧逻辑不再执行，只保留注释供你对比。
        # ======================================
        # green_list = self._get_green_list(prev_token_id, vocab_size, device)
        # mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        # mask[green_list] = True
        # for aid in self.anchor_ids:
        #     if 0 <= aid < vocab_size:
        #         mask[aid] = False
        # scores[0, mask] += self.watermark_bias
        # existing_tokens = set(input_ids[0].tolist())
        # candidate_anchors = [aid for aid in self.anchor_ids if 0 <= aid < vocab_size and aid not in existing_tokens]
        # if candidate_anchors:
        #     best_aid = max(candidate_anchors, key=lambda x: float(scores[0, x].item()))
        #     scores[0, best_aid] += self.anchor_boost

        next_token_boosts: Dict[int, float] = {}

        self.num_steps += 1

        active_candidates = []   # 已经启动，优先推进
        start_candidates = []    # 还没启动，只有没有 active 时才允许启动

        for seq_idx, seq in enumerate(self.anchor_token_seqs):
            if not seq:
                continue

            # 如果该 anchor 已在“新生成部分”中完整出现，则标记完成，不再参与
            if self._is_seq_satisfied(generated, seq):
                self.completed_anchor_spans.add(seq_idx)
                continue

            k = self._matched_prefix_len(generated, seq)

            if 0 < k < len(seq):
                # 已经启动：候选是下一个 token
                tid = seq[k]
                boost = self.finish_boost if k == len(seq) - 1 else self.continue_boost

                active_candidates.append({
                    "seq_idx": seq_idx,
                    "tid": tid,
                    "boost": boost,
                    "k": k,
                    "seq_len": len(seq),
                })

            elif k == 0:
                # 还没启动：候选是首 token
                tid = seq[0]
                boost = self.start_boost

                start_candidates.append({
                    "seq_idx": seq_idx,
                    "tid": tid,
                    "boost": boost,
                    "k": 0,
                    "seq_len": len(seq),
                })

        # =========================
        # 【核心修改】聚焦策略
        # 1. 有 active anchors 时，只推进 active anchor
        # 2. 没有 active anchors 时，才启动 1 个新的 anchor
        # =========================
        selected_candidates = []

        if len(active_candidates) > 0:
            # 优先推进“匹配进度最高”的 active anchor
            # 如果进度相同，再看当前 token logits 分数
            active_candidates = sorted(
                active_candidates,
                key=lambda x: (x["k"], scores[0, x["tid"]].item()),
                reverse=True
            )

            # 每步只保留 1 个 active candidate
            selected_candidates = active_candidates[:1]

        else:
            # 没有 active anchor，才允许启动一个新的 anchor
            start_candidates = sorted(
                start_candidates,
                key=lambda x: scores[0, x["tid"]].item(),
                reverse=True
            )

            # 每步最多只启动 1 个
            selected_candidates = start_candidates[:1]

        for cand in selected_candidates:
            tid = cand["tid"]
            boost = cand["boost"]
            k = cand["k"]
            seq_len = cand["seq_len"]

            if 0 <= tid < vocab_size:
                next_token_boosts[tid] = next_token_boosts.get(tid, 0.0) + boost

                if k == 0:
                    self.num_start_boost_hits += 1
                elif k == seq_len - 1:
                    self.num_finish_boost_hits += 1
                else:
                    self.num_continue_boost_hits += 1

        if len(next_token_boosts) > 0:
            self.num_span_boost_steps += 1
            self.total_next_token_candidates += len(next_token_boosts)
        else:
            self.num_empty_boost_steps += 1
        # 实际加到 logits 上
        for tid, boost in next_token_boosts.items():
            scores[0, tid] += boost

        # ======================================
        # 【新增】watermark bias（兼容 span-aware 版本）
        # 说明：保留原 watermark 协议不变；
        # 同时把“当前合法的 next token”从 watermark mask 中豁免，
        # 避免事实约束和 watermark 直接打架。
        # ======================================
        green_list = self._get_green_list(prev_token_id, vocab_size, device)
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        mask[green_list] = True

        for tid in next_token_boosts.keys():
            if 0 <= tid < vocab_size:
                mask[tid] = False

        scores[0, mask] += self.watermark_bias

        return scores
    



# =========================
# 5. 统一实验对象
# =========================

class UnifiedMedicalExperiment:
    def __init__(self, cfg: TCDWConfig):


        self.cfg = cfg
        self.device = cfg.device

        # 新增：固定全局随机种子
        set_global_seed(cfg.random_seed)
        print(f">>> 使用 random_seed: {cfg.random_seed}")

        print(f">>> 初始化 tokenizer: {cfg.llama_path}")
        self.gen_tokenizer = AutoTokenizer.from_pretrained(cfg.llama_path)

        if self.gen_tokenizer.pad_token is None:
            self.gen_tokenizer.pad_token = self.gen_tokenizer.eos_token

        print(f">>> 初始化生成模型: {cfg.llama_path}")
        dtype = torch.float16 if cfg.fp16 and self.device == "cuda" else torch.float32
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            cfg.llama_path,
            torch_dtype=dtype,
        ).to(self.device)
        self.gen_model.eval()

        print(f">>> 初始化 NLI tokenizer/model: {cfg.nli_path}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(cfg.nli_path)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(cfg.nli_path).to(self.device)
        self.nli_model.eval()

        self.anchor_extractor = MedicalAnchorExtractor(cfg, self.gen_tokenizer)

        # 动态解析标签
        self.label2id = {str(v).lower(): int(k) for k, v in self.nli_model.config.id2label.items()}
        # 常见标签名兼容
        self.entail_label_id = self._find_label_id(["entailment", "entails"])
        self.contra_label_id = self._find_label_id(["contradiction", "contradict"])

        print(f">>> NLI id2label: {self.nli_model.config.id2label}")
        print(f">>> entail_label_id={self.entail_label_id}, contra_label_id={self.contra_label_id}")

    def _find_label_id(self, candidates: List[str]) -> Optional[int]:
        for name, idx in self.label2id.items():
            for cand in candidates:
                if cand in name:
                    return idx
        return None

    def build_prompt(self, question: str, context: str) -> str:
        return self.cfg.prompt_template.format(question=question, context=context)

    def encode_prompt(self, prompt: str) -> torch.LongTensor:
        input_ids = self.gen_tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = input_ids.to(self.device)

        if self.cfg.force_bos_token_id is not None:
            bos_id = self.cfg.force_bos_token_id
            if input_ids.size(1) == 0 or int(input_ids[0, 0].item()) != bos_id:
                bos = torch.tensor([[bos_id]], dtype=input_ids.dtype, device=input_ids.device)
                input_ids = torch.cat([bos, input_ids], dim=1)

        return input_ids

    @torch.no_grad()
    def generate_one(self, question: str, ref_fact: str) -> Dict[str, Any]:
        # =========================
        # 1. 构造 prompt / 编码
        # =========================
        prompt = self.build_prompt(question, ref_fact)
        input_ids = self.encode_prompt(prompt)
        prompt_len = input_ids.size(1)

        mode = self.cfg.experiment_mode

        # =========================
        # 2. 先准备 Must-Have claims 和 anchors
        #    这一步必须在 processor 创建之前完成
        # =========================
        must_have_claims = extract_must_have_claims(ref_fact, self.anchor_extractor.nlp)
        high_anchor_texts, medium_anchor_texts = claims_to_anchor_texts(must_have_claims)

        # 当前 span-aware 主保护对象：只用高优先级 anchor
        anchor_texts = high_anchor_texts

        anchor_ids = self.anchor_extractor.anchor_texts_to_token_ids(anchor_texts)
        anchor_token_seqs = self.anchor_extractor.anchor_texts_to_token_seqs(anchor_texts)

        # =========================
        # 3. 根据实验模式设置参数
        # =========================
        watermark_bias = self.cfg.watermark_bias
        start_boost = self.cfg.anchor_start_boost
        continue_boost = self.cfg.anchor_continue_boost
        finish_boost = self.cfg.anchor_finish_boost

        if mode == "no_wm":
            watermark_bias = 0.0
            start_boost = 0.0
            continue_boost = 0.0
            finish_boost = 0.0

        elif mode == "wm_only":
            watermark_bias = self.cfg.watermark_bias
            start_boost = 0.0
            continue_boost = 0.0
            finish_boost = 0.0

        elif mode == "wm_span":
            watermark_bias = self.cfg.watermark_bias
            start_boost = self.cfg.anchor_start_boost
            continue_boost = self.cfg.anchor_continue_boost
            finish_boost = self.cfg.anchor_finish_boost

        else:
            raise ValueError(f"Unknown experiment_mode: {mode}")

        # =========================
        # 4. 生成
        # =========================
        if mode == "no_wm":
            output_ids = self.gen_model.generate(
                input_ids=input_ids,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                pad_token_id=self.gen_tokenizer.pad_token_id,
                eos_token_id=self.gen_tokenizer.eos_token_id,
            )

            # no_wm 没有 processor，所以诊断字段手动给默认值
            span_diag = {
                "num_steps": 0,
                "num_span_boost_steps": 0,
                "num_empty_boost_steps": 0,
                "span_boost_step_ratio": 0.0,
                "num_start_boost_hits": 0,
                "num_continue_boost_hits": 0,
                "num_finish_boost_hits": 0,
                "avg_next_token_candidates": 0.0,
                "num_completed_anchor_spans": 0,
                "num_anchor_token_seqs": len(anchor_token_seqs),
                "avg_anchor_seq_len": (
                    sum(len(seq) for seq in anchor_token_seqs) / len(anchor_token_seqs)
                    if len(anchor_token_seqs) > 0 else 0.0
                ),
            }

        else:
            processor = TCDWMedicalLogitsProcessor(
                gamma=self.cfg.gamma,
                seed_offset=self.cfg.seed_offset,
                watermark_bias=watermark_bias,
                anchor_ids=anchor_ids,
                anchor_boost=self.cfg.anchor_boost,
                anchor_token_seqs=anchor_token_seqs,
                start_boost=start_boost,
                continue_boost=continue_boost,
                finish_boost=finish_boost,
                prompt_length=prompt_len,   # 关键：只看新生成部分
            )

            logits_processor = LogitsProcessorList([processor])

            output_ids = self.gen_model.generate(
                input_ids=input_ids,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                logits_processor=logits_processor,
                pad_token_id=self.gen_tokenizer.pad_token_id,
                eos_token_id=self.gen_tokenizer.eos_token_id,
            )

            # 生成结束后，读取 span-aware 诊断信息
            span_diag = processor.get_diagnostics()

        # =========================
        # 5. 解码生成文本
        # =========================
        gen_ids = output_ids[0, prompt_len:]
        gen_text = self.gen_tokenizer.decode(gen_ids, skip_special_tokens=True)

        # =========================
        # 6. Z-score
        # =========================
        z_info = detect_z_from_token_ids(
            prompt_ids=input_ids[0].tolist(),
            generated_ids=gen_ids.tolist(),
            vocab_size=len(self.gen_tokenizer),
            gamma=self.cfg.gamma,
            seed_offset=self.cfg.seed_offset,
            device=self.device,
        )

        # =========================
        # 7. 整体语义审计
        # =========================
        entail_score, contra_score, neutral_score = self.audit_semantic(ref_fact, gen_text)

        # =========================
        # 8. Claim-level 评估
        # =========================
        claim_eval = score_must_have_claims(
            must_have_claims=must_have_claims,
            generated_text=gen_text,
            audit_fn=self.audit_semantic,
            entail_threshold=0.50,
            contra_threshold=0.50,
        )

        # =========================
        # 9. 返回完整结果
        # =========================
        return {
            "prompt": prompt,
            "prompt_token_ids": input_ids[0].tolist(),
            "generated_token_ids": gen_ids.tolist(),
            "text": gen_text,
            "num_generated_tokens": len(gen_ids),

            # anchor / claims
            "must_have_claims": must_have_claims,
            "anchor_texts": anchor_texts,
            "anchor_ids": anchor_ids,
            "anchor_token_seqs": anchor_token_seqs,

            # sentence / semantic metrics
            "entail_score": entail_score,
            "contra_score": contra_score,
            "neutral_score": neutral_score,

            # z-score info
            **z_info,

            # claim-level scores
            "claim_support_score": claim_eval["support_score"],
            "claim_contradiction_score": claim_eval["contradiction_score"],
            "claim_missing_score": claim_eval["missing_score"],
            "claim_results": claim_eval["claim_results"],

            # span-aware diagnostics
            "num_span_steps": span_diag["num_steps"],
            "num_span_boost_steps": span_diag["num_span_boost_steps"],
            "num_empty_boost_steps": span_diag["num_empty_boost_steps"],
            "span_boost_step_ratio": span_diag["span_boost_step_ratio"],
            "num_start_boost_hits": span_diag["num_start_boost_hits"],
            "num_continue_boost_hits": span_diag["num_continue_boost_hits"],
            "num_finish_boost_hits": span_diag["num_finish_boost_hits"],
            "avg_next_token_candidates": span_diag["avg_next_token_candidates"],
            "num_completed_anchor_spans": span_diag["num_completed_anchor_spans"],
            "num_anchor_token_seqs": span_diag["num_anchor_token_seqs"],
            "avg_anchor_seq_len": span_diag["avg_anchor_seq_len"],
        }
    


    def first_sentence(self,text: str) -> str:
        text = text.strip()
        if not text:
            return text
        parts = re.split(r'(?<=[.!?])\s+', text)
        return parts[0].strip() if parts else text
    

    @torch.no_grad()
    def audit_semantic(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        if not isinstance(premise, str):
            premise = "" if premise is None else str(premise)
        if not isinstance(hypothesis, str):
            hypothesis = "" if hypothesis is None else str(hypothesis)

        # 新增：只取生成结果的第一句做 NLI
        nli_hypothesis = self.first_sentence(hypothesis)
        inputs = self.nli_tokenizer(
            premise,
            nli_hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        logits = self.nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

        entail_score = float(probs[self.entail_label_id].item()) if self.entail_label_id is not None else float("nan")
        contra_score = float(probs[self.contra_label_id].item()) if self.contra_label_id is not None else float("nan")

        # 尝试找 neutral
        neutral_id = None
        for name, idx in self.label2id.items():
            if "neutral" in name:
                neutral_id = idx
                break
        neutral_score = float(probs[neutral_id].item()) if neutral_id is not None else float("nan")

        return entail_score, contra_score, neutral_score


# =========================
# 6. Z-score 检测（直接吃 token ids）
# =========================

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
    if len(prompt_ids) == 0:
        return {
            "green_hits": 0,
            "green_total": 0,
            "green_ratio": 0.0,
            "z_score": 0.0,
        }

    if len(generated_ids) == 0:
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
        green = _green_list_from_prev(
            prev_token_id=prev_id,
            vocab_size=vocab_size,
            gamma=gamma,
            seed_offset=seed_offset,
            device=device,
        )
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


# =========================
# 7. 样本字段兼容
# =========================

def normalize_pubmedqa_item(item: Dict[str, Any]) -> Tuple[str, str]:
    question = item.get("question", "")

    # 你原先主要用 long_answer；这里做兼容
    context = (
        item.get("long_answer", None)
        or item.get("context", None)
        or item.get("contexts", None)
        or item.get("answer", None)
        or ""
    )

    # 处理 list / dict
    if isinstance(context, list):
        context = " ".join([str(x) for x in context])
    elif isinstance(context, dict):
        context = str(context)

    if not isinstance(question, str):
        question = str(question)
    if not isinstance(context, str):
        context = str(context)

    return question, context


# =========================
# 8. 主实验
# =========================

def run_batch_experiment(cfg: TCDWConfig):
    dataset = smart_load_medical_data(cfg.pqa_data_path)
    exp = UnifiedMedicalExperiment(cfg)

    results = []
    total_num = min(cfg.num_samples, len(dataset))

    print(f">>> 开始一体化实验，总样本数: {total_num}")

    for idx in tqdm(range(total_num)):
        try:
            item = dataset[idx]
            question, ref_fact = normalize_pubmedqa_item(item)

            result = exp.generate_one(question=question, ref_fact=ref_fact)

            row = {
                "id": idx,
                "question": question,
                "ref_fact": ref_fact,
                "text": result["text"],
                "num_generated_tokens": result["num_generated_tokens"],
                "anchor_texts": " ||| ".join(result["anchor_texts"]),
                "anchor_ids": str(result["anchor_ids"]),
                # =========================
                # 【新增】额外导出 span-aware anchor 序列，方便你比对
                # =========================
                "anchor_token_seqs": str(result["anchor_token_seqs"]),
                "green_hits": result["green_hits"],
                "green_total": result["green_total"],
                "green_ratio": round(result["green_ratio"], 6),
                "z_score": round(result["z_score"], 6),
                "entail_score": round(result["entail_score"], 6) if not math.isnan(result["entail_score"]) else result["entail_score"],
                "contra_score": round(result["contra_score"], 6) if not math.isnan(result["contra_score"]) else result["contra_score"],
                "neutral_score": round(result["neutral_score"], 6) if not math.isnan(result["neutral_score"]) else result["neutral_score"],
                "prompt": result["prompt"],
                "prompt_token_ids": str(result["prompt_token_ids"]),
                "generated_token_ids": str(result["generated_token_ids"]),
                "gamma": cfg.gamma,
                "seed_offset": cfg.seed_offset,
                "watermark_bias": cfg.watermark_bias,
                "anchor_boost": cfg.anchor_boost,
                "experiment_mode": cfg.experiment_mode,
                # =========================
                # 【新增】导出 span-aware boost 参数
                # =========================
                "anchor_start_boost": cfg.anchor_start_boost,
                "anchor_continue_boost": cfg.anchor_continue_boost,
                "anchor_finish_boost": cfg.anchor_finish_boost,
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "must_have_claims": str(result["must_have_claims"]),
                "claim_support_score": round(result["claim_support_score"], 6),
                "claim_contradiction_score": round(result["claim_contradiction_score"], 6),
                "claim_missing_score": round(result["claim_missing_score"], 6),
                "claim_results": str(result["claim_results"]),
                "random_seed": cfg.random_seed,

                #新增
                "num_span_steps": result["num_span_steps"],
                "num_span_boost_steps": result["num_span_boost_steps"],
                "num_empty_boost_steps": result["num_empty_boost_steps"],
                "span_boost_step_ratio": round(result["span_boost_step_ratio"], 6),
                "num_start_boost_hits": result["num_start_boost_hits"],
                "num_continue_boost_hits": result["num_continue_boost_hits"],
                "num_finish_boost_hits": result["num_finish_boost_hits"],
                "avg_next_token_candidates": round(result["avg_next_token_candidates"], 6),
                "num_completed_anchor_spans": result["num_completed_anchor_spans"],
                "num_anchor_token_seqs": result["num_anchor_token_seqs"],
                "avg_anchor_seq_len": round(result["avg_anchor_seq_len"], 6),
                
            }
            results.append(row)

        except Exception as e:
            results.append({
                "id": idx,
                "question": "",
                "ref_fact": "",
                "text": "",
                "num_generated_tokens": 0,
                "anchor_texts": "",
                "anchor_ids": "",
                "anchor_token_seqs": "",
                "green_hits": 0,
                "green_total": 0,
                "green_ratio": 0.0,
                "z_score": 0.0,
                "entail_score": float("nan"),
                "contra_score": float("nan"),
                "neutral_score": float("nan"),
                "prompt": "",
                "prompt_token_ids": "",
                "generated_token_ids": "",
                "gamma": cfg.gamma,
                "seed_offset": cfg.seed_offset,
                "watermark_bias": cfg.watermark_bias,
                "anchor_boost": cfg.anchor_boost,
                "anchor_start_boost": cfg.anchor_start_boost,
                "anchor_continue_boost": cfg.anchor_continue_boost,
                "anchor_finish_boost": cfg.anchor_finish_boost,
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "error": str(e),
                "experiment_mode": cfg.experiment_mode,
                "must_have_claims": "",
                "claim_support_score": float("nan"),
                "claim_contradiction_score": float("nan"),
                "claim_missing_score": float("nan"),
                "claim_results": "",
                "random_seed": cfg.random_seed,
            })

    df = pd.DataFrame(results)
    df.to_csv(cfg.output_csv, index=False, encoding="utf-8")

    valid_df = df[df["text"].astype(str).str.len() > 0].copy()

    print("\n>>> 实验完成")
    print(f">>> 输出文件: {cfg.output_csv}")
    print(f">>> 总样本数: {len(df)}")
    print(f">>> 有效样本数: {len(valid_df)}")

    if len(valid_df) > 0:
        if "z_score" in valid_df.columns:
            print(f">>> 平均 Z-score: {valid_df['z_score'].mean():.4f}")
        if "entail_score" in valid_df.columns:
            print(f">>> 平均 entail_score: {valid_df['entail_score'].mean():.4f}")
        if "contra_score" in valid_df.columns:
            print(f">>> 平均 contra_score: {valid_df['contra_score'].mean():.4f}")
        if "neutral_score" in valid_df.columns:
            print(f">>> 平均 neutral_score: {valid_df['neutral_score'].mean():.4f}")
        if "claim_support_score" in valid_df.columns:
            print(f">>> 平均 claim_support_score: {valid_df['claim_support_score'].mean():.4f}")
        if "claim_contradiction_score" in valid_df.columns:
            print(f">>> 平均 claim_contradiction_score: {valid_df['claim_contradiction_score'].mean():.4f}")
        if "claim_missing_score" in valid_df.columns:
            print(f">>> 平均 claim_missing_score: {valid_df['claim_missing_score'].mean():.4f}")

def run_all_ablation_with_seeds():
    #modes = ["no_wm", "wm_only", "wm_span"]
    modes = ["wm_span"]
    seeds = [1]
  

    for seed in seeds:
        for mode in modes:
            print(f"\n===== Running mode: {mode}, seed: {seed} =====")
            cfg = TCDWConfig(
                num_samples=50,
                experiment_mode=mode,
                random_seed=seed,
                output_csv=f"tcdw_{mode}_seed{seed}_results.csv",
            )
            run_batch_experiment(cfg)

if __name__ == "__main__":
    run_all_ablation_with_seeds()