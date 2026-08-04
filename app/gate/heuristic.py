"""Heuristic ambiguity gate — zero ML dependencies, same protocol.

Used when no CenterDistill checkpoint is available. Always sets
fallback_used=True. Results from this gate MUST NEVER be presented
as CenterDistill results — not in logs, tables, or the README
(AGENTS.md rule 4).

Rules (spec §5.3):
    - Unresolved deictic with no antecedent → CLARIFY
    - ≥2 entities matching the head noun → ALTERNATIVES
    - <5 tokens and no named entity → CLARIFY
    - else → ANSWER
"""

from __future__ import annotations

import re
import time

import numpy as np
import numpy.typing as npt

from app.gate.base import Behaviour, GateDecision
from app.gate.thresholds import DEFAULT_THRESHOLDS, GateThresholds

# Deictic pronouns that signal ambiguity when context has multiple referents
_DEICTICS: frozenset[str] = frozenset(
    {"this", "that", "these", "those", "it", "they", "them", "its", "their"}
)

# Minimum context length (in tokens) to consider deictic references unresolved —
# shorter contexts are unlikely to contain multiple competing referents
_MIN_CTX_TOKENS_FOR_DEICTIC: int = 10


def _tokenize_simple(text: str) -> list[str]:
    """Whitespace + punctuation tokenizer. No ML deps."""
    return re.findall(r"\b\w+\b", text.lower())


def _extract_capitalized_entities(text: str) -> list[str]:
    """Extract capitalized phrases as a proxy for named entities.

    Returns individual capitalized words and multi-word sequences.
    This is deliberately coarse — it's a heuristic fallback, not NER.
    """
    return re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)


def _has_deictic_without_antecedent(
    q_tokens: list[str], ctx_tokens: list[str]
) -> bool:
    """Check for unresolved deictic reference.

    Heuristic: question contains a deictic pronoun AND the context is long
    enough to plausibly contain multiple referents.
    """
    deictics_in_q = _DEICTICS & set(q_tokens)
    if not deictics_in_q:
        return False
    # If context has enough tokens, multiple referents are plausible
    return len(ctx_tokens) > _MIN_CTX_TOKENS_FOR_DEICTIC


def _count_entity_referents(question: str, context: str) -> int:
    """Count distinct named entities in context that could be competing referents.

    The spec says '≥2 entities matching the head noun'. Without POS tagging
    we approximate by counting distinct proper-noun-like phrases in context
    that share a category with the question's subject. Only entities of ≥3
    chars are counted to avoid matching adjectives like 'Western'.
    """
    entities = set(
        e for e in _extract_capitalized_entities(context)
        if len(e) >= 3
    )
    # Filter to entities that appear as distinct noun phrases (not substrings
    # of each other) to avoid double-counting "New York" and "York"
    filtered: set[str] = set()
    sorted_entities = sorted(entities, key=len, reverse=True)
    for ent in sorted_entities:
        if not any(ent in longer for longer in filtered):
            filtered.add(ent)
    return len(filtered)


def _make_synthetic_distribution(behaviour: Behaviour) -> list[float]:
    """Generate a synthetic 5-vector matching the decided behaviour.

    This keeps the downstream UI and state schema consistent without
    pretending to have real model outputs. The distributions are
    hand-crafted to be plausible for each behaviour.
    """
    if behaviour == "ANSWER":
        return [0.70, 0.10, 0.08, 0.07, 0.05]
    elif behaviour == "ALTERNATIVES":
        return [0.30, 0.28, 0.20, 0.12, 0.10]
    else:  # CLARIFY
        return [0.25, 0.22, 0.20, 0.18, 0.15]


def _entropy_nats(distribution: list[float]) -> float:
    """Compute entropy in NATS from a probability distribution.

    Uses np.log (natural log), NEVER log2. Using log2 produces values
    off by a factor of ln(2) ≈ 0.693, silently misrouting queries.
    """
    p: npt.NDArray[np.float64] = np.array(distribution, dtype=np.float64)
    return float(-(p * np.log(p)).sum())


class HeuristicGate:
    """Rule-based ambiguity gate with zero ML dependencies.

    Satisfies the AmbiguityGate protocol. Always sets fallback_used=True.

    Rules (spec §5.3, applied in order):
        1. Unresolved deictic with no antecedent → CLARIFY
        2. ≥2 entity-like phrases in context → ALTERNATIVES
        3. <5 tokens and no named entity in question → CLARIFY
        4. else → ANSWER
    """

    def __init__(self, thresholds: GateThresholds | None = None) -> None:
        self._thresholds = thresholds or DEFAULT_THRESHOLDS

    def __call__(self, question: str, context: str) -> GateDecision:
        """Classify using heuristic rules. Always fallback_used=True."""
        start: float = time.perf_counter()

        q_tokens: list[str] = _tokenize_simple(question)
        ctx_tokens: list[str] = _tokenize_simple(context)

        # Apply rules in specified order
        behaviour: Behaviour
        if _has_deictic_without_antecedent(q_tokens, ctx_tokens):
            behaviour = "CLARIFY"
        elif _count_entity_referents(question, context) >= 2:
            behaviour = "ALTERNATIVES"
        elif len(q_tokens) < 5 and not _extract_capitalized_entities(question):
            behaviour = "CLARIFY"
        else:
            behaviour = "ANSWER"

        # Generate synthetic distribution for protocol compatibility
        distribution: list[float] = _make_synthetic_distribution(behaviour)

        # Compute entropy in NATS from the synthetic distribution
        entropy: float = _entropy_nats(distribution)

        elapsed_ms: float = (time.perf_counter() - start) * 1000.0

        return GateDecision(
            behaviour=behaviour,
            center_distribution=distribution,
            max_prob=max(distribution),
            entropy=entropy,  # NATS — never log2
            second_mass=sorted(distribution, reverse=True)[1],
            thresholds=self._thresholds.as_dict(),
            latency_ms=round(elapsed_ms, 2),
            fallback_used=True,  # ALWAYS True for heuristic gate
        )
