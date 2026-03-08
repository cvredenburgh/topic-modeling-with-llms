"""Reliability helpers for repeated LLM inference and consensus."""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List


def summary_consensus(
    summaries: List[str],
    min_agreement: float,
) -> Dict[str, object]:
    """Pick the most central summary and score agreement in [0, 1]."""
    cleaned = [s.strip() for s in summaries if s and s.strip()]
    if not cleaned:
        return {
            "summary": "",
            "reliability_score": 0.0,
            "reliability_consistent": False,
            "candidate_summaries": [],
        }
    if len(cleaned) == 1:
        return {
            "summary": cleaned[0],
            "reliability_score": 1.0,
            "reliability_consistent": True,
            "candidate_summaries": cleaned,
        }

    avg_sims = []
    for i, s_i in enumerate(cleaned):
        sims = []
        for j, s_j in enumerate(cleaned):
            if i == j:
                continue
            sims.append(SequenceMatcher(None, s_i, s_j).ratio())
        avg_sims.append(sum(sims) / max(len(sims), 1))

    best_idx = max(range(len(cleaned)), key=lambda idx: avg_sims[idx])
    score = float(avg_sims[best_idx])
    return {
        "summary": cleaned[best_idx],
        "reliability_score": score,
        "reliability_consistent": score >= min_agreement,
        "candidate_summaries": cleaned,
    }


def tag_consensus(
    sampled_tags: List[List[str]],
    min_agreement: float,
) -> Dict[str, object]:
    """Aggregate repeated tag generations into stable/unstable tag sets."""
    normalized = [[t.strip().lower() for t in tags if t and t.strip()] for tags in sampled_tags]
    normalized = [tags for tags in normalized if tags]
    if not normalized:
        return {
            "tags": [],
            "reliability_score": 0.0,
            "reliability_consistent": False,
            "candidate_tags": [],
        }

    freq: Dict[str, int] = {}
    first_seen: Dict[str, int] = {}
    seen_idx = 0
    for tags in normalized:
        for tag in tags:
            if tag not in first_seen:
                first_seen[tag] = seen_idx
                seen_idx += 1
        for tag in set(tags):
            freq[tag] = freq.get(tag, 0) + 1

    n = len(normalized)
    tag_rows = []
    ranked_tags = sorted(freq.items(), key=lambda item: (-item[1], first_seen[item[0]], item[0]))
    for tag, count in ranked_tags:
        agreement = count / n
        tag_rows.append(
            {"tag": tag, "consistent": agreement >= min_agreement, "agreement": round(agreement, 4)}
        )

    # Global agreement = average pairwise Jaccard overlap between sampled tag sets.
    if n == 1:
        reliability = 1.0
    else:
        scores = []
        for i in range(n):
            a = set(normalized[i])
            for j in range(i + 1, n):
                b = set(normalized[j])
                denom = len(a | b)
                scores.append(0.0 if denom == 0 else len(a & b) / denom)
        reliability = float(sum(scores) / max(len(scores), 1))

    return {
        "tags": tag_rows,
        "reliability_score": reliability,
        "reliability_consistent": reliability >= min_agreement,
        "candidate_tags": normalized,
    }
