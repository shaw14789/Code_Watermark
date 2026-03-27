import re
import math
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    LogitsProcessorList,
)
import spacy

from .config import TCDWConfig
from .data_utils import set_global_seed, smart_load_medical_data, normalize_pubmedqa_item, detect_z_from_token_ids
from .anchor_extractor import PubMedQAAnchorExtractor
from .claim_extractor import extract_must_have_claims
from .claim_scorer import score_must_have_claims
from .processor import TCDWMedicalLogitsProcessor


class UnifiedMedicalExperiment:
    def __init__(self, cfg: TCDWConfig):
        self.cfg = cfg
        self.device = cfg.device
        set_global_seed(cfg.random_seed)
        print(f">>> 使用 random_seed: {cfg.random_seed}")

        print(f">>> 初始化 tokenizer: {cfg.gen_model_path}")
        self.gen_tokenizer = AutoTokenizer.from_pretrained(cfg.gen_model_path)
        if self.gen_tokenizer.pad_token is None:
            self.gen_tokenizer.pad_token = self.gen_tokenizer.eos_token

        print(f">>> 初始化生成模型: {cfg.gen_model_path}")
        dtype = torch.float16 if cfg.fp16 and self.device == "cuda" else torch.float32
        self.gen_model = AutoModelForCausalLM.from_pretrained(cfg.gen_model_path, torch_dtype=dtype).to(self.device)
        self.gen_model.eval()

        print(f">>> 初始化 NLI tokenizer/model: {cfg.nli_path}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(cfg.nli_path)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(cfg.nli_path).to(self.device)
        self.nli_model.eval()

        print(f">>> 初始化 scispaCy: {cfg.scispacy_model}")
        self.nlp = spacy.load(cfg.scispacy_model)
        self.anchor_extractor = PubMedQAAnchorExtractor(self.gen_tokenizer)

        self.label2id = {str(v).lower(): int(k) for k, v in self.nli_model.config.id2label.items()}
        self.entail_label_id = self._find_label_id(["entailment", "entails"])
        self.contra_label_id = self._find_label_id(["contradiction", "contradict"])
        print(f">>> NLI id2label: {self.nli_model.config.id2label}")
        print(f">>> entail_label_id={self.entail_label_id}, contra_label_id={self.contra_label_id}")

    def _find_label_id(self, candidates: List[str]):
        for name, idx in self.label2id.items():
            for cand in candidates:
                if cand in name:
                    return idx
        return None

    def build_prompt(self, question: str, context: str) -> str:
        return self.cfg.prompt_template.format(question=question, context=context)

    def encode_prompt(self, prompt: str) -> torch.LongTensor:
        input_ids = self.gen_tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        if self.cfg.force_bos_token_id is not None:
            bos_id = self.cfg.force_bos_token_id
            if input_ids.size(1) == 0 or int(input_ids[0, 0].item()) != bos_id:
                bos = torch.tensor([[bos_id]], dtype=input_ids.dtype, device=input_ids.device)
                input_ids = torch.cat([bos, input_ids], dim=1)
        return input_ids

    def first_sentence(self, text: str) -> str:
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

        # 整体答案评估仍然只看第一句
        nli_hypothesis = self.first_sentence(hypothesis)
        inputs = self.nli_tokenizer(premise, nli_hypothesis, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        logits = self.nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        entail_score = float(probs[self.entail_label_id].item()) if self.entail_label_id is not None else float("nan")
        contra_score = float(probs[self.contra_label_id].item()) if self.contra_label_id is not None else float("nan")
        neutral_id = None
        for name, idx in self.label2id.items():
            if "neutral" in name:
                neutral_id = idx
                break
        neutral_score = float(probs[neutral_id].item()) if neutral_id is not None else float("nan")
        return entail_score, contra_score, neutral_score

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
        # 2. 评估层：先提取 Must-Have claims
        #    注意：claim 只用于评估，不再作为生成 anchor
        # =========================
        must_have_claims = extract_must_have_claims(ref_fact, self.nlp)

        # =========================
        # 3. 生成层：只从 ref_fact 中抽自然短片段作为 anchor
        # =========================
        anchor_texts = self.anchor_extractor.extract_anchor_texts(ref_fact)
        anchor_token_seqs = self.anchor_extractor.anchor_texts_to_token_seqs(anchor_texts)

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
                anchor_token_seqs=anchor_token_seqs,
                start_boost=start_boost,
                continue_boost=continue_boost,
                finish_boost=finish_boost,
                prompt_length=prompt_len,
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

            span_diag = processor.get_diagnostics()

        # =========================
        # 5. 解码
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
        # 7. 整体 NLI
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
        # 9. 返回结果
        # =========================
        return {
            "prompt": prompt,
            "prompt_token_ids": input_ids[0].tolist(),
            "generated_token_ids": gen_ids.tolist(),
            "text": gen_text,
            "num_generated_tokens": len(gen_ids),

            # 评估层
            "must_have_claims": must_have_claims,

            # 生成层 anchors
            "anchor_texts": anchor_texts,
            "anchor_token_seqs": anchor_token_seqs,

            # semantic metrics
            "entail_score": entail_score,
            "contra_score": contra_score,
            "neutral_score": neutral_score,

            # z-score
            **z_info,

            # claim-level
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
            result = exp.generate_one(question, ref_fact)
            row = {
                "id": idx,
                "question": question,
                "ref_fact": ref_fact,
                "text": result["text"],
                "num_generated_tokens": result["num_generated_tokens"],
                "must_have_claims": str(result["must_have_claims"]),
                "anchor_texts": " ||| ".join(result["anchor_texts"]),
                "anchor_token_seqs": str(result["anchor_token_seqs"]),
                "green_hits": result["green_hits"],
                "green_total": result["green_total"],
                "green_ratio": round(result["green_ratio"], 6),
                "z_score": round(result["z_score"], 6),
                "entail_score": round(result["entail_score"], 6) if not math.isnan(result["entail_score"]) else result["entail_score"],
                "contra_score": round(result["contra_score"], 6) if not math.isnan(result["contra_score"]) else result["contra_score"],
                "neutral_score": round(result["neutral_score"], 6) if not math.isnan(result["neutral_score"]) else result["neutral_score"],
                "claim_support_score": round(result["claim_support_score"], 6),
                "claim_contradiction_score": round(result["claim_contradiction_score"], 6),
                "claim_missing_score": round(result["claim_missing_score"], 6),
                "claim_results": str(result["claim_results"]),
                "prompt": result["prompt"],
                "prompt_token_ids": str(result["prompt_token_ids"]),
                "generated_token_ids": str(result["generated_token_ids"]),
                "gamma": cfg.gamma,
                "seed_offset": cfg.seed_offset,
                "watermark_bias": cfg.watermark_bias,
                "anchor_start_boost": cfg.anchor_start_boost,
                "anchor_continue_boost": cfg.anchor_continue_boost,
                "anchor_finish_boost": cfg.anchor_finish_boost,
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "random_seed": cfg.random_seed,
                "experiment_mode": cfg.experiment_mode,
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
            print(f"[ERROR] idx={idx}, mode={cfg.experiment_mode}, err={repr(e)}")
            results.append({
                "id": idx,
                "question": "",
                "ref_fact": "",
                "text": "",
                "num_generated_tokens": 0,
                "must_have_claims": "",
                "anchor_texts": "",
                "anchor_token_seqs": "",
                "green_hits": 0,
                "green_total": 0,
                "green_ratio": 0.0,
                "z_score": 0.0,
                "entail_score": float("nan"),
                "contra_score": float("nan"),
                "neutral_score": float("nan"),
                "claim_support_score": float("nan"),
                "claim_contradiction_score": float("nan"),
                "claim_missing_score": float("nan"),
                "claim_results": "",
                "prompt": "",
                "prompt_token_ids": "",
                "generated_token_ids": "",
                "gamma": cfg.gamma,
                "seed_offset": cfg.seed_offset,
                "watermark_bias": cfg.watermark_bias,
                "anchor_start_boost": cfg.anchor_start_boost,
                "anchor_continue_boost": cfg.anchor_continue_boost,
                "anchor_finish_boost": cfg.anchor_finish_boost,
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "random_seed": cfg.random_seed,
                "experiment_mode": cfg.experiment_mode,
                "num_span_steps": 0,
                "num_span_boost_steps": 0,
                "num_empty_boost_steps": 0,
                "span_boost_step_ratio": 0.0,
                "num_start_boost_hits": 0,
                "num_continue_boost_hits": 0,
                "num_finish_boost_hits": 0,
                "avg_next_token_candidates": 0.0,
                "num_completed_anchor_spans": 0,
                "num_anchor_token_seqs": 0,
                "avg_anchor_seq_len": 0.0,
                "error": str(e),
            })

    df = pd.DataFrame(results)
    df.to_csv(cfg.output_csv, index=False, encoding="utf-8")
    valid_df = df[df["text"].astype(str).str.len() > 0].copy()

    print("\n>>> 实验完成")
    print(f">>> 输出文件: {cfg.output_csv}")
    print(f">>> 总样本数: {len(df)}")
    print(f">>> 有效样本数: {len(valid_df)}")

    if len(valid_df) > 0:
        print(f">>> 平均 Z-score: {valid_df['z_score'].mean():.4f}")
        print(f">>> 平均 entail_score: {valid_df['entail_score'].mean():.4f}")
        print(f">>> 平均 contra_score: {valid_df['contra_score'].mean():.4f}")
        print(f">>> 平均 neutral_score: {valid_df['neutral_score'].mean():.4f}")
        print(f">>> 平均 claim_support_score: {valid_df['claim_support_score'].mean():.4f}")
        print(f">>> 平均 claim_contradiction_score: {valid_df['claim_contradiction_score'].mean():.4f}")
        print(f">>> 平均 claim_missing_score: {valid_df['claim_missing_score'].mean():.4f}")
        print(f">>> 平均 span_boost_step_ratio: {valid_df['span_boost_step_ratio'].mean():.4f}")
        print(f">>> 平均 num_completed_anchor_spans: {valid_df['num_completed_anchor_spans'].mean():.4f}")
        print(f">>> 平均 avg_anchor_seq_len: {valid_df['avg_anchor_seq_len'].mean():.4f}")
        print(f">>> 平均 avg_next_token_candidates: {valid_df['avg_next_token_candidates'].mean():.4f}")


def run_all_ablation_with_seeds():

    # modes = ["no_wm", "wm_only", "wm_span"]
    # seeds = [1, 2, 3]
    modes = [ "wm_span"]
    seeds = [1]

    for seed in seeds:
        for mode in modes:
            print(f"\n===== Running mode: {mode}, seed: {seed} =====")
            cfg = TCDWConfig(
                num_samples=20,
                experiment_mode=mode,
                random_seed=seed,
                output_csv=f"tcdw_{mode}_seed{seed}_results.csv",
            )
            run_batch_experiment(cfg)
