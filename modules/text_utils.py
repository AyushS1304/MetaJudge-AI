"""
Small text helpers shared across the pipeline, UI, and baselines.
"""

from __future__ import annotations

import re


def split_sentences(text: str) -> list[str]:
    """Split text into simple sentence-like chunks while preserving punctuation."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


def find_best_matching_sentence(text: str, fragment: str) -> str:
    """
    Return the sentence from ``text`` with the highest token overlap with ``fragment``.
    """
    sentences = split_sentences(text)
    if not sentences:
        return text.strip()

    fragment_words = set(fragment.lower().split())
    best_sentence = sentences[0]
    best_score = -1

    for sentence in sentences:
        score = len(set(sentence.lower().split()) & fragment_words)
        if score > best_score:
            best_sentence = sentence
            best_score = score

    return best_sentence
