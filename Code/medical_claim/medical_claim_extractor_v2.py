import re
from typing import List, Dict, Any, Tuple

# =========================
# High-priority patterns: 这些是真正该重点保护的医学结论骨架
# =========================
HIGH_PRIORITY_RELATION_PATTERNS = [
    r"\bassociated with\b",
    r"\bnot associated with\b",
    r"\brelated to\b",
    r"\bnot related to\b",
    r"\blinked to\b",
    r"\bpredict(?:s|ed)?\b",
    r"\bdoes not predict\b",
    r"\brecommended\b",
    r"\bnot recommended\b",
    r"\bcontraindicated\b",
    r"\beffective\b",
    r"\bineffective\b",
    r"\bpreferred\b",
    r"\bnot preferred\b",
    r"\bsynergistic\b",
    r"\bnot synergistic\b",
]

HIGH_PRIORITY_RISK_PATTERNS = [
    r"\bhigher risk\b",
    r"\blower risk\b",
    r"\bincreased risk\b",
    r"\bdecreased risk\b",
    r"\bsignificantly higher\b",
    r"\bsignificantly lower\b",
    r"\bmore likely\b",
    r"\bless likely\b",
    r"\bincrease(?:d|s)?\b",
    r"\bdecrease(?:d|s)?\b",
    r"\breduced\b",
    r"\benhanced\b",
    r"\bworsened\b",
    r"\bimproved\b",
]

HIGH_PRIORITY_NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bdoes not\b",
    r"\bdo not\b",
    r"\bwithout\b",
    r"\babsence of\b",
    r"\bnegative\b",
]

HIGH_PRIORITY_MODALITY_PATTERNS = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bappears? to\b",
    r"\bsuggests?\b",
    r"\bpossible\b",
    r"\bunclear\b",
    r"\bmore studies are necessary\b",
    r"\bevidence is limited\b",
]

HIGH_PRIORITY_NUMERIC_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s?(?:mg|g|ml|mcg|kg)\b",
    r"\b\d+(?:\.\d+)?%\b",
    r"\b\d+(?:\.\d+)?\s?(?:day|days|week|weeks|month|months|year|years)\b",
    r"\btwice daily\b",
    r"\bonce daily\b",
    r"\bweekly\b",
    r"\bmonthly\b",
]

# =========================
# Low-information generic terms: 这些词太泛，尽量不过度保护
# =========================
GENERIC_LOW_VALUE_TERMS = {
    "patient", "patients", "treatment", "study", "studies", "result", "results",
    "analysis", "outcome", "outcomes", "group", "groups", "system", "effect",
    "effects", "response", "responses", "condition", "conditions",
}


def simple_sentence_split(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def find_matches(patterns: List[str], text: str) -> List[str]:
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


def pick_entity_candidates(nlp, sent: str) -> List[str]:
    doc = nlp(sent)
    ents = []
    for ent in doc.ents:
        t = ent.text.strip()
        if len(t) > 1:
            ents.append(t)

    out = []
    seen = set()
    for e in ents:
        k = e.lower()
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out[:6]


def is_low_value_entity(text: str) -> bool:
    t = text.strip().lower()
    if t in GENERIC_LOW_VALUE_TERMS:
        return True
    if len(t.split()) == 1 and len(t) <= 3:
        return True
    return False


def is_too_long_phrase(text: str, max_words: int = 5) -> bool:
    return len(text.strip().split()) > max_words


def dedup_claims(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for c in claims:
        key = (c["type"], c["claim"].lower(), c["priority"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def build_claim(claim_type: str, claim_text: str, priority: str) -> Dict[str, Any]:
    return {
        "type": claim_type,
        "claim": claim_text.strip(),
        "priority": priority,
    }


def extract_must_have_claims(ref_fact: str, nlp) -> List[Dict[str, Any]]:
    """
    轻量版 claim / anchor 抽取：输出高优先级/中优先级两档。

    高优先级：
    - numeric / dosage / frequency
    - risk direction
    - relation polarity
    - modality / hedging

    中优先级：
    - 关键实体（辅助，不应主导 span-aware）
    """
    claims: List[Dict[str, Any]] = []
    sents = simple_sentence_split(ref_fact)

    for sent in sents:
        entities = pick_entity_candidates(nlp, sent)

        relation_hits = find_matches(HIGH_PRIORITY_RELATION_PATTERNS, sent)
        risk_hits = find_matches(HIGH_PRIORITY_RISK_PATTERNS, sent)
        neg_hits = find_matches(HIGH_PRIORITY_NEGATION_PATTERNS, sent)
        modality_hits = find_matches(HIGH_PRIORITY_MODALITY_PATTERNS, sent)
        numeric_hits = find_matches(HIGH_PRIORITY_NUMERIC_PATTERNS, sent)

        # -------------------------
        # 高优先级：numeric
        # -------------------------
        for hit in numeric_hits[:3]:
            if entities:
                claims.append(build_claim("numeric", f"{entities[0]} involves {hit}", "high"))
            else:
                claims.append(build_claim("numeric", hit, "high"))

        # -------------------------
        # 高优先级：risk direction
        # -------------------------
        for hit in risk_hits[:2]:
            if len(entities) >= 2:
                claims.append(build_claim("risk", f"{entities[0]} {hit} {entities[1]}", "high"))
            elif len(entities) == 1:
                claims.append(build_claim("risk", f"{entities[0]} {hit}", "high"))
            else:
                claims.append(build_claim("risk", hit, "high"))

        # -------------------------
        # 高优先级：relation / polarity
        # -------------------------
        for hit in relation_hits[:2]:
            if len(entities) >= 2:
                claims.append(build_claim("relation", f"{entities[0]} {hit} {entities[1]}", "high"))
            elif len(entities) == 1:
                claims.append(build_claim("relation", f"{entities[0]} {hit}", "high"))
            else:
                claims.append(build_claim("relation", hit, "high"))

        # -------------------------
        # 高优先级：negation / modality
        # -------------------------
        for hit in neg_hits[:2]:
            claims.append(build_claim("polarity", hit, "high"))

        for hit in modality_hits[:2]:
            claims.append(build_claim("modality", hit, "high"))

        # -------------------------
        # 中优先级：关键实体（辅助）
        # 只保留少量、不太长、不是泛词的实体
        # -------------------------
        for ent in entities[:3]:
            if is_low_value_entity(ent):
                continue
            if is_too_long_phrase(ent):
                continue
            claims.append(build_claim("entity", ent, "medium"))

    claims = dedup_claims(claims)

    # 控制总量：高优先级优先保留
    high_claims = [c for c in claims if c["priority"] == "high"]
    med_claims = [c for c in claims if c["priority"] == "medium"]

    # 你现在最重要的是别让 anchor 太多太散
    high_claims = high_claims[:6]
    med_claims = med_claims[:2]

    return high_claims + med_claims


def claims_to_anchor_texts(must_have_claims: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    把 claim 列表拆成：
    - high_priority_texts: 用于 span-aware 主保护
    - medium_priority_texts: 可选辅助
    """
    high_priority_texts = []
    medium_priority_texts = []

    for item in must_have_claims:
        text = item["claim"].strip()
        if not text:
            continue
        if item["priority"] == "high":
            high_priority_texts.append(text)
        else:
            medium_priority_texts.append(text)

    def dedup_keep_order(xs: List[str]) -> List[str]:
        out = []
        seen = set()
        for x in xs:
            k = x.lower()
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out

    return dedup_keep_order(high_priority_texts), dedup_keep_order(medium_priority_texts)


__all__ = [
    "extract_must_have_claims",
    "claims_to_anchor_texts",
]