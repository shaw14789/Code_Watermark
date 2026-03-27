import re
from typing import List


# =========================
# PubMedQA 风格：高优先级 anchors
# 这些是更接近“结论骨架”的自然短片段
# =========================

DECISION_PATTERNS = [
    r"\bno difference\b",
    r"\bnot significantly\b",
    r"\bnot associated with\b",
    r"\bnot related to\b",
    r"\bnot recommended\b",
    r"\bno evidence\b",
    r"\bdid not\b",
    r"\bwas not\b",
    r"\bwere not\b",
    r"\bcan be\b",
    r"\bplays? a role\b",
    r"\bassociated with\b",
    r"\brelated to\b",
    r"\beffective\b",
    r"\bineffective\b",
    r"\brecommended\b",
    r"\bcontraindicated\b",
]

COMPARISON_PATTERNS = [
    r"\bbetter than\b",
    r"\bworse than\b",
    r"\bequal to\b",
    r"\bnot equal\b",
    r"\bcomparable to\b",
    r"\bcompared with\b",
    r"\bcompared to\b",
    r"\bsimilar to\b",
    r"\bsuperior to\b",
    r"\binferior to\b",
]

RISK_DIRECTION_PATTERNS = [
    r"\bhigher risk\b",
    r"\blower risk\b",
    r"\bincreased risk\b",
    r"\bdecreased risk\b",
    r"\bmore likely\b",
    r"\bless likely\b",
    r"\bincrease(?:d|s)?\b",
    r"\bdecrease(?:d|s)?\b",
    r"\breduced\b",
    r"\benhanced\b",
    r"\bimproved\b",
    r"\bworsened\b",
]

MODALITY_PATTERNS = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bappears? to\b",
    r"\bsuggests?\b",
    r"\bpossible\b",
    r"\bunclear\b",
    r"\bmore studies\b",
    r"\bdefinitive conclusion\b",
    r"\bevidence is limited\b",
]

SIGNIFICANCE_PATTERNS = [
    r"\bsignificant(?:ly)?\b",
    r"\bnot significant(?:ly)?\b",
    r"\bno significant difference\b",
    r"\bstatistically significant\b",
]

NUMERIC_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s?(?:mg|g|ml|mcg|kg)\b",
    r"\b\d+(?:\.\d+)?%\b",
    r"\b\d+(?:\.\d+)?\s?(?:day|days|week|weeks|month|months|year|years)\b",
    r"\btwice daily\b",
    r"\bonce daily\b",
    r"\bweekly\b",
    r"\bmonthly\b",
]

# 这些词太泛，不建议拿来做主保护 anchor
LOW_VALUE_SINGLE_WORDS = {
    "study", "studies", "result", "results", "analysis", "outcome", "outcomes",
    "patient", "patients", "group", "groups", "effect", "effects",
    "treatment", "intervention", "context", "question", "answer",
}


def _find_matches(patterns: List[str], text: str) -> List[str]:
    hits = []
    for p in patterns:
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            hits.append(m.group(0).strip())

    # 去重保序
    out = []
    seen = set()
    for h in hits:
        k = h.lower()
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out


def _dedup_keep_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        k = x.lower().strip()
        if not k:
            continue
        if k not in seen:
            seen.add(k)
            out.append(x.strip())
    return out


def _is_low_value_anchor(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return True
    if t in LOW_VALUE_SINGLE_WORDS:
        return True
    # 过短单词一般信息量太低
    if len(t.split()) == 1 and len(t) <= 3:
        return True
    return False


def _prefer_two_to_three_token_anchors(items: List[str]) -> List[str]:
    """
    优先保留 2~3 token 的自然短片段。
    如果没有，再保留 1 token 或更长片段。
    """
    items = _dedup_keep_order(items)

    two_three = []
    others = []

    for x in items:
        n = len(x.split())
        if 2 <= n <= 3:
            two_three.append(x)
        else:
            others.append(x)

    return two_three + others


class PubMedQAAnchorExtractor:
    """
    专门给 PubMedQA 风格结论型医疗 QA 用的 anchor extractor。

    核心思想：
    - 不优先抽实体
    - 优先抽决定 yes/no/maybe 结论的自然短片段
    - 优先保护 relation / polarity / comparison / modality / numeric
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract_anchor_texts(self, ref_fact: str) -> List[str]:
        anchors = []

        # 1. 先抽最关键的结论骨架
        anchors.extend(_find_matches(DECISION_PATTERNS, ref_fact))
        anchors.extend(_find_matches(COMPARISON_PATTERNS, ref_fact))
        anchors.extend(_find_matches(RISK_DIRECTION_PATTERNS, ref_fact))
        anchors.extend(_find_matches(MODALITY_PATTERNS, ref_fact))
        anchors.extend(_find_matches(SIGNIFICANCE_PATTERNS, ref_fact))
        anchors.extend(_find_matches(NUMERIC_PATTERNS, ref_fact))

        anchors = _dedup_keep_order(anchors)

        # 2. 过滤低价值 anchor
        anchors = [a for a in anchors if not _is_low_value_anchor(a)]

        # 3. 优先保留 2~3 token 的短片段
        anchors = _prefer_two_to_three_token_anchors(anchors)

        # 4. 最多只保留少量，避免过散
        anchors = anchors[:6]

        return anchors

    def anchor_texts_to_token_seqs(self, anchor_texts: List[str]) -> List[List[int]]:
        """
        只保留 seq_len >= 2 的 anchor，避免单 token anchor 太脆弱。
        """
        seqs = []
        for a in anchor_texts:
            ids = self.tokenizer.encode(" " + a, add_special_tokens=False)
            if len(ids) >= 2:
                seqs.append(ids)
        return seqs