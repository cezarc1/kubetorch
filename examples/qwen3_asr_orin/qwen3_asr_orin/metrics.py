from __future__ import annotations

import re


def normalize_transcript(text: str) -> str:
    """Normalize transcripts for sanity WER checks, not leaderboard scoring."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"_+", " ", text)
    return " ".join(text.split())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return word error rate using Levenshtein distance over normalized words."""
    ref_words = normalize_transcript(reference).split()
    hyp_words = normalize_transcript(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    previous = list(range(len(hyp_words) + 1))
    for ref_index, ref_word in enumerate(ref_words, start=1):
        current = [ref_index]
        for hyp_index, hyp_word in enumerate(hyp_words, start=1):
            substitution_cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1] / len(ref_words)
