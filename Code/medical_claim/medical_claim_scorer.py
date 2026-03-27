<<<<<<< HEAD
import re
from typing import List, Dict, Any


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def score_single_claim_with_nli(
    claim: str,
    answer_sentences: List[str],
    audit_fn,
) -> Dict[str, Any]:
    """
    audit_fn(claim, sentence) -> (entail_score, contra_score, neutral_score)
    """
    best_entail = 0.0
    best_contra = 0.0
    best_neutral = 0.0
    best_entail_sentence = ""
    best_contra_sentence = ""

    for sent in answer_sentences:
        entail, contra, neutral = audit_fn(claim, sent)

        if entail > best_entail:
            best_entail = entail
            best_entail_sentence = sent

        if contra > best_contra:
            best_contra = contra
            best_contra_sentence = sent

        if neutral > best_neutral:
            best_neutral = neutral

    return {
        "best_entail": best_entail,
        "best_contra": best_contra,
        "best_neutral": best_neutral,
        "best_entail_sentence": best_entail_sentence,
        "best_contra_sentence": best_contra_sentence,
    }


def classify_claim_status(
    best_entail: float,
    best_contra: float,
    entail_threshold: float = 0.50,
    contra_threshold: float = 0.50,
) -> str:
    if best_contra >= contra_threshold:
        return "contradicted"
    elif best_entail >= entail_threshold:
        return "supported"
    else:
        return "missing"


def score_must_have_claims(
    must_have_claims: List[Dict[str, Any]],
    generated_text: str,
    audit_fn,
    entail_threshold: float = 0.50,
    contra_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Args:
        must_have_claims: e.g. [{"type": "...", "claim": "...", "priority": "must"}, ...]
        generated_text: model output
        audit_fn: a function like self.audit_semantic(claim, sentence)
    """
    sentences = split_sentences(generated_text)

    if len(must_have_claims) == 0:
        return {
            "support_score": 0.0,
            "contradiction_score": 0.0,
            "missing_score": 0.0,
            "claim_results": [],
        }

    claim_results = []
    supported_count = 0
    contradicted_count = 0
    missing_count = 0

    for item in must_have_claims:
        claim_text = item["claim"]

        local_scores = score_single_claim_with_nli(
            claim=claim_text,
            answer_sentences=sentences,
            audit_fn=audit_fn,
        )

        status = classify_claim_status(
            best_entail=local_scores["best_entail"],
            best_contra=local_scores["best_contra"],
            entail_threshold=entail_threshold,
            contra_threshold=contra_threshold,
        )

        if status == "supported":
            supported_count += 1
        elif status == "contradicted":
            contradicted_count += 1
        else:
            missing_count += 1

        claim_results.append({
            "type": item.get("type", ""),
            "claim": claim_text,
            "priority": item.get("priority", "must"),
            "status": status,
            "best_entail": local_scores["best_entail"],
            "best_contra": local_scores["best_contra"],
            "best_neutral": local_scores["best_neutral"],
            "best_entail_sentence": local_scores["best_entail_sentence"],
            "best_contra_sentence": local_scores["best_contra_sentence"],
        })

    total = len(must_have_claims)

    return {
        "support_score": supported_count / total,
        "contradiction_score": contradicted_count / total,
        "missing_score": missing_count / total,
        "claim_results": claim_results,
=======
import re
from typing import List, Dict, Any


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def score_single_claim_with_nli(
    claim: str,
    answer_sentences: List[str],
    audit_fn,
) -> Dict[str, Any]:
    """
    audit_fn(claim, sentence) -> (entail_score, contra_score, neutral_score)
    """
    best_entail = 0.0
    best_contra = 0.0
    best_neutral = 0.0
    best_entail_sentence = ""
    best_contra_sentence = ""

    for sent in answer_sentences:
        entail, contra, neutral = audit_fn(claim, sent)

        if entail > best_entail:
            best_entail = entail
            best_entail_sentence = sent

        if contra > best_contra:
            best_contra = contra
            best_contra_sentence = sent

        if neutral > best_neutral:
            best_neutral = neutral

    return {
        "best_entail": best_entail,
        "best_contra": best_contra,
        "best_neutral": best_neutral,
        "best_entail_sentence": best_entail_sentence,
        "best_contra_sentence": best_contra_sentence,
    }


def classify_claim_status(
    best_entail: float,
    best_contra: float,
    entail_threshold: float = 0.50,
    contra_threshold: float = 0.50,
) -> str:
    if best_contra >= contra_threshold:
        return "contradicted"
    elif best_entail >= entail_threshold:
        return "supported"
    else:
        return "missing"


def score_must_have_claims(
    must_have_claims: List[Dict[str, Any]],
    generated_text: str,
    audit_fn,
    entail_threshold: float = 0.50,
    contra_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Args:
        must_have_claims: e.g. [{"type": "...", "claim": "...", "priority": "must"}, ...]
        generated_text: model output
        audit_fn: a function like self.audit_semantic(claim, sentence)
    """
    sentences = split_sentences(generated_text)

    if len(must_have_claims) == 0:
        return {
            "support_score": 0.0,
            "contradiction_score": 0.0,
            "missing_score": 0.0,
            "claim_results": [],
        }

    claim_results = []
    supported_count = 0
    contradicted_count = 0
    missing_count = 0

    for item in must_have_claims:
        claim_text = item["claim"]

        local_scores = score_single_claim_with_nli(
            claim=claim_text,
            answer_sentences=sentences,
            audit_fn=audit_fn,
        )

        status = classify_claim_status(
            best_entail=local_scores["best_entail"],
            best_contra=local_scores["best_contra"],
            entail_threshold=entail_threshold,
            contra_threshold=contra_threshold,
        )

        if status == "supported":
            supported_count += 1
        elif status == "contradicted":
            contradicted_count += 1
        else:
            missing_count += 1

        claim_results.append({
            "type": item.get("type", ""),
            "claim": claim_text,
            "priority": item.get("priority", "must"),
            "status": status,
            "best_entail": local_scores["best_entail"],
            "best_contra": local_scores["best_contra"],
            "best_neutral": local_scores["best_neutral"],
            "best_entail_sentence": local_scores["best_entail_sentence"],
            "best_contra_sentence": local_scores["best_contra_sentence"],
        })

    total = len(must_have_claims)

    return {
        "support_score": supported_count / total,
        "contradiction_score": contradicted_count / total,
        "missing_score": missing_count / total,
        "claim_results": claim_results,
>>>>>>> 6668a76 (修改了config.py里的参数配置)
    }