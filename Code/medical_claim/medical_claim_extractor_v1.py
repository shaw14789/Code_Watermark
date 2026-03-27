import re
from typing import List, Dict, Any

RISK_PATTERNS = [
    r"\bassociated with\b",
    r"\brelated to\b",
    r"\blinked to\b",
    r"\bpredict(?:s|ed)?\b",
    r"\bincreased risk\b",
    r"\bhigher risk\b",
    r"\blower risk\b",
    r"\bmore likely\b",
    r"\bless likely\b",
    r"\bincrease(?:d|s)?\b",
    r"\bdecrease(?:d|s)?\b",
    r"\beffective\b",
    r"\bineffective\b",
    r"\brecommended\b",
    r"\bcontraindicated\b",
    r"\bsynergistic\b",
]

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bdoes not\b",
    r"\bdo not\b",
    r"\bwithout\b",
    r"\babsence of\b",
    r"\bnegative\b",
]

MODALITY_PATTERNS = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bappears? to\b",
    r"\bsuggests?\b",
    r"\bpossible\b",
    r"\bunclear\b",
    r"\bmore studies are necessary\b",
    r"\bevidence is limited\b",
]

NUMERIC_PATTERNS = [
    r"\b\d+(\.\d+)?\s?(mg|g|ml|mcg|kg)\b",
    r"\b\d+(\.\d+)?%\b",
    r"\b\d+(\.\d+)?\s?(day|days|week|weeks|month|months|year|years)\b",
    r"\btwice daily\b",
    r"\bonce daily\b",
    r"\bweekly\b",
    r"\bmonthly\b",
]


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

    # 去重保序
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

    # 去重保序
    out = []
    seen = set()
    for e in ents:
        k = e.lower()
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out[:4]


def build_relation_claim(sent: str, entities: List[str], relation_hits: List[str]) -> List[Dict[str, Any]]:
    claims = []
    if not relation_hits:
        return claims

    rel = relation_hits[0]

    if len(entities) >= 2:
        claim = f"{entities[0]} {rel} {entities[1]}"
    elif len(entities) == 1:
        claim = f"{entities[0]} {rel}"
    else:
        claim = sent

    claims.append({
        "type": "relation",
        "claim": claim,
        "priority": "must",
    })
    return claims


def build_numeric_claims(sent: str, numeric_hits: List[str], entities: List[str]) -> List[Dict[str, Any]]:
    claims = []
    for hit in numeric_hits[:2]:
        if entities:
            claim = f"{entities[0]} involves {hit}"
        else:
            claim = f"value or dosage is {hit}"
        claims.append({
            "type": "numeric",
            "claim": claim,
            "priority": "must",
        })
    return claims


def build_modality_claims(modality_hits: List[str]) -> List[Dict[str, Any]]:
    claims = []
    if modality_hits:
        claims.append({
            "type": "modality",
            "claim": "the statement is hedged or uncertain",
            "priority": "must",
        })
    return claims


def build_negation_claims(neg_hits: List[str]) -> List[Dict[str, Any]]:
    claims = []
    if neg_hits:
        claims.append({
            "type": "polarity",
            "claim": "the statement contains negation or negative polarity",
            "priority": "must",
        })
    return claims


def extract_must_have_claims(ref_fact: str, nlp) -> List[Dict[str, Any]]:
    """
    Extract lightweight medical must-have claims from a reference fact string.

    Args:
        ref_fact: medical reference text
        nlp: a loaded spaCy / scispaCy pipeline, e.g. en_core_sci_lg

    Returns:
        A list of claim dicts with fields:
        - type
        - claim
        - priority
    """
    claims = []
    sents = simple_sentence_split(ref_fact)

    for sent in sents:
        entities = pick_entity_candidates(nlp, sent)

        relation_hits = find_matches(RISK_PATTERNS, sent)
        numeric_hits = find_matches(NUMERIC_PATTERNS, sent)
        neg_hits = find_matches(NEGATION_PATTERNS, sent)
        modality_hits = find_matches(MODALITY_PATTERNS, sent)

        sent_claims = []

        # 1. 数字/剂量优先
        sent_claims.extend(build_numeric_claims(sent, numeric_hits, entities))

        # 2. 风险/关系
        sent_claims.extend(build_relation_claim(sent, entities, relation_hits))

        # 3. 极性/否定
        sent_claims.extend(build_negation_claims(neg_hits))

        # 4. 模态/谨慎性
        sent_claims.extend(build_modality_claims(modality_hits))

        # 控制每句最多 3 个 claim
        claims.extend(sent_claims[:3])

    # 全局去重
    dedup = []
    seen = set()
    for c in claims:
        key = (c["type"], c["claim"].lower())
        if key not in seen:
            seen.add(key)
            dedup.append(c)

    return dedup[:6]


__all__ = [
    "extract_must_have_claims",
]