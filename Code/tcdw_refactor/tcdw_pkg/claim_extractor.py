import re
from typing import List, Dict, Any

RISK_PATTERNS = [
    r"\bhigher risk\b",
    r"\blower risk\b",
    r"\bincreased risk\b",
    r"\bdecreased risk\b",
    r"\bmore likely\b",
    r"\bless likely\b",
    r"\bincrease(?:d|s)?\b",
    r"\bdecrease(?:d|s)?\b",
]

RELATION_PATTERNS = [
    r"\bassociated with\b",
    r"\bnot associated with\b",
    r"\brelated to\b",
    r"\bnot related to\b",
    r"\blinked to\b",
    r"\bpredict(?:s|ed)?\b",
    r"\brecommended\b",
    r"\bcontraindicated\b",
    r"\beffective\b",
    r"\bineffective\b",
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
    r"\b\d+(?:\.\d+)?\s?(?:mg|g|ml|mcg|kg)\b",
    r"\b\d+(?:\.\d+)?%\b",
    r"\b\d+(?:\.\d+)?\s?(?:day|days|week|weeks|month|months|year|years)\b",
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
    return out[:4]


def extract_must_have_claims(ref_fact: str, nlp) -> List[Dict[str, Any]]:
    claims = []
    sents = simple_sentence_split(ref_fact)

    for sent in sents:
        entities = pick_entity_candidates(nlp, sent)
        relation_hits = find_matches(RELATION_PATTERNS, sent)
        risk_hits = find_matches(RISK_PATTERNS, sent)
        numeric_hits = find_matches(NUMERIC_PATTERNS, sent)
        neg_hits = find_matches(NEGATION_PATTERNS, sent)
        modality_hits = find_matches(MODALITY_PATTERNS, sent)

        for hit in numeric_hits[:2]:
            claims.append({"type": "numeric", "claim": hit, "priority": "must"})

        for hit in risk_hits[:2]:
            claims.append({"type": "risk", "claim": hit, "priority": "must"})

        for hit in relation_hits[:2]:
            claims.append({"type": "relation", "claim": hit, "priority": "must"})

        for hit in neg_hits[:2]:
            claims.append({"type": "polarity", "claim": hit, "priority": "must"})

        for hit in modality_hits[:2]:
            claims.append({"type": "modality", "claim": hit, "priority": "must"})

        # 少量实体作为辅助评估对象
        for ent in entities[:2]:
            claims.append({"type": "entity", "claim": ent, "priority": "aux"})

    out = []
    seen = set()
    for c in claims:
        key = (c["type"], c["claim"].lower(), c["priority"])
        if key not in seen:
            seen.add(key)
            out.append(c)

    must = [x for x in out if x["priority"] == "must"][:8]
    aux = [x for x in out if x["priority"] == "aux"][:2]
    return must + aux
