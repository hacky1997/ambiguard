"""LLM judge comparison arm — asks an LLM directly about ambiguity.

This is the arm the comparison table exists to beat. Its cost, latency,
and non-determinism are the argument for the learned gate (spec §3.1).

Prompt template (few-shot, temperature 0):
    "Is this question ambiguous given this context?
     Answer ANSWER, CLARIFY, or ALTERNATIVES."
"""

from __future__ import annotations

from typing import Any

from app.llm.base import LLMResponse
from app.llm.registry import get_provider
from app.settings import get_settings
from eval.arms import ArmResult

_JUDGE_PROMPT_TEMPLATE: str = """You are an ambiguity classifier. Given a question and context, decide the routing behaviour.

Rules:
- ANSWER: The question has a single clear answer given the context.
- CLARIFY: The question is ambiguous — multiple interpretations exist and the user should clarify.
- ALTERNATIVES: The question has multiple valid interpretations that can each be answered separately.

Question: {question}
Context: {context}

Respond with exactly one word: ANSWER, CLARIFY, or ALTERNATIVES."""


def _parse_behaviour(raw: str) -> str:
    """Extract a valid behaviour label from LLM output.

    Handles common variations: extra whitespace, lowercase, explanation text.
    """
    upper = raw.strip().upper()
    for label in ("ANSWER", "CLARIFY", "ALTERNATIVES"):
        if label in upper:
            return label
    # Conservative default if parsing fails
    return "ANSWER"


class LLMJudgeArm:
    """Comparison arm that asks an LLM to judge ambiguity.

    Non-deterministic, costs real money (unless using mock provider),
    and adds a full generation round-trip per decision. These downsides
    ARE the argument — the comparison table quantifies them.
    """

    def __init__(
        self,
        provider_name: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        settings = get_settings()
        self._provider_name: str = provider_name or settings.llm_provider
        self._provider = get_provider(
            self._provider_name,
            api_key=api_key or settings.openai_api_key,
            model=model or settings.openai_model,
        )

    @property
    def name(self) -> str:
        return f"LLM judge ({self._provider.model_name})"

    @property
    def deterministic(self) -> bool:
        return False  # LLM output varies even at temperature 0

    def predict(self, question: str, context: str) -> ArmResult:
        """Ask the LLM to classify ambiguity."""
        prompt: str = _JUDGE_PROMPT_TEMPLATE.format(
            question=question, context=context
        )
        response: LLMResponse = self._provider.complete(prompt, temperature=0.0)

        return ArmResult(
            prediction=_parse_behaviour(response["content"]),
            latency_ms=response["latency_ms"],
            cost_usd=response["cost_usd"],
            metadata={
                "model": response["model"],
                "raw_response": response["content"],
                "deterministic": response["deterministic"],
            },
        )
