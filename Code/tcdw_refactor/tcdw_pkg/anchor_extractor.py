import re
from typing import List

# 生成层 anchor：只保留自然语言里本来就会出现的短片段，不再用 claim 文本
ANCHOR_RELATION_PATTERNS = [
    r"\bassociated with\b",
    r"\bnot associated with\b",
    r"\brelated to\b",
    r"\bnot related to\b",
    r"\blinked to\b",
    r"\brecommended\b",
    r"\bcontraindicated\b",
    r"\beffective\b",
    r"\bineffective\b",
    r"\bsynergistic\b",
    r"\bnot synergistic\b",
]

ANCHOR_RISK_PATTERNS = [
    r"\bhigher risk\b",
    r"\blower risk\b",
    r"\bincreased risk\b",
    r"\bdecreased risk\b",
    r"\bmore likely\b",
    r"\bless likely\b",
]

ANCHOR_MODALITY_PATTERNS = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bappears? to\b",
    r"\bsuggests?\b",
    r"\bpossible\b",
    r"\bunclear\b",
]

ANCHOR_NUMERIC_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s?(?:mg|g|ml|mcg|kg)\b",
    r"\b\d+(?:\.\d+)?%\b",
    r"\b\d+(?:\.\d+)?\s?(?:day|days|week|weeks|month|months|year|years)\b",
    r"\btwice daily\b",
    r"\bonce daily\b",
    r"\bweekly\b",
    r"\bmonthly\b",
]


def _find_matches(patterns: List[str], text: str) -> List[str]:
    hits = []
    for p in patterns:
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            hits.append(m.group(0).strip())

    out = []
    seen = set()
    for h in hits:
        k = h.lower()
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out


class NaturalAnchorExtractor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract_anchor_texts(self, ref_fact: str) -> List[str]:
        anchors = []
        anchors.extend(_find_matches(ANCHOR_NUMERIC_PATTERNS, ref_fact))
        anchors.extend(_find_matches(ANCHOR_RISK_PATTERNS, ref_fact))
        anchors.extend(_find_matches(ANCHOR_RELATION_PATTERNS, ref_fact))
        anchors.extend(_find_matches(ANCHOR_MODALITY_PATTERNS, ref_fact))

        # 去重保序
        out = []
        seen = set()
        for a in anchors:
            k = a.lower()
            if k not in seen:
                seen.add(k)
                out.append(a)

        # 别太多，避免过散
        return out[:6]

    def anchor_texts_to_token_seqs(self, anchor_texts: List[str]) -> List[List[int]]:
        seqs = []
        for a in anchor_texts:
            ids = self.tokenizer.encode(" " + a, add_special_tokens=False)
            if len(ids) > 0:
                seqs.append(ids)
        return seqs
