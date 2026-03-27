from typing import List, Dict, Any, Optional
import torch
from transformers import LogitsProcessor


class TCDWMedicalLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        gamma: float,
        seed_offset: int,
        watermark_bias: float,
        anchor_token_seqs: Optional[List[List[int]]] = None,
        start_boost: float = 0.8,
        continue_boost: float = 5.0,
        finish_boost: float = 7.0,
        prompt_length: int = 0,
    ):
        self.gamma = gamma
        self.seed_offset = seed_offset
        self.watermark_bias = watermark_bias
        self.anchor_token_seqs = anchor_token_seqs if anchor_token_seqs is not None else []
        self.start_boost = start_boost
        self.continue_boost = continue_boost
        self.finish_boost = finish_boost
        self.prompt_length = prompt_length

        # diagnostics
        self.num_steps = 0
        self.num_span_boost_steps = 0
        self.num_empty_boost_steps = 0
        self.num_start_boost_hits = 0
        self.num_continue_boost_hits = 0
        self.num_finish_boost_hits = 0
        self.total_next_token_candidates = 0
        self.completed_anchor_spans = set()

    def _get_green_list(self, prev_token_id: int, vocab_size: int, device: torch.device) -> torch.Tensor:
        g = torch.Generator(device=device)
        g.manual_seed(int(prev_token_id) + self.seed_offset)
        perm = torch.randperm(vocab_size, generator=g, device=device)
        green_size = int(vocab_size * self.gamma)
        return perm[:green_size]

    def _is_seq_satisfied(self, generated_ids: List[int], target_seq: List[int]) -> bool:
        n = len(target_seq)
        if n == 0 or len(generated_ids) < n:
            return False
        for i in range(len(generated_ids) - n + 1):
            if generated_ids[i:i+n] == target_seq:
                return True
        return False

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

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if input_ids.size(0) != 1:
            return scores

        vocab_size = scores.size(-1)
        device = scores.device
        full_ids = input_ids[0].tolist()
        generated = full_ids[self.prompt_length:]
        prev_token_id = int(input_ids[0, -1].item())

        self.num_steps += 1
        next_token_boosts: Dict[int, float] = {}

        active_candidates = []
        start_candidates = []

        for seq_idx, seq in enumerate(self.anchor_token_seqs):
            if not seq:
                continue
            if self._is_seq_satisfied(generated, seq):
                self.completed_anchor_spans.add(seq_idx)
                continue
            k = self._matched_prefix_len(generated, seq)
            if 0 < k < len(seq):
                tid = seq[k]
                boost = self.finish_boost if k == len(seq) - 1 else self.continue_boost
                active_candidates.append({"seq_idx": seq_idx, "tid": tid, "boost": boost, "k": k, "seq_len": len(seq)})
            elif k == 0:
                tid = seq[0]
                boost = self.start_boost
                start_candidates.append({"seq_idx": seq_idx, "tid": tid, "boost": boost, "k": 0, "seq_len": len(seq)})

        selected_candidates = []
        if len(active_candidates) > 0:
            active_candidates = sorted(active_candidates, key=lambda x: (x["k"], scores[0, x["tid"]].item()), reverse=True)
            selected_candidates = active_candidates[:1]
        else:
            start_candidates = sorted(start_candidates, key=lambda x: scores[0, x["tid"]].item(), reverse=True)
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

        for tid, boost in next_token_boosts.items():
            scores[0, tid] += boost

        green_list = self._get_green_list(prev_token_id, vocab_size, device)
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        mask[green_list] = True
        for tid in next_token_boosts.keys():
            mask[tid] = False
        scores[0, mask] += self.watermark_bias
        return scores
