"""In-memory vector retrieval store — zero external DB dependencies.

Used as default so the app boots without Qdrant / external services
(AGENTS.md rule 3). Implements simple term-frequency / cosine similarity
search over a 4-domain corpus (medical, legal, e-commerce, education).
"""

from __future__ import annotations

import math
import re
from collections import Counter

from app.graph.state import Evidence
from app.retrieval.base import Retriever

# Seed corpus across 4 domains matching SPEC.md §1.5
_DEFAULT_CORPUS: list[dict[str, str]] = [
    {
        "id": "doc_med_01",
        "domain": "medical",
        "title": "Medication Side Effects & Guidelines",
        "text": (
            "Medication Alpha is prescribed for hypertension and causes mild nausea in 5% "
            "of patients. Medication Beta is an anti-inflammatory drug used for arthritis, "
            "with common side effects including headaches and dizziness. Always consult a "
            "healthcare provider before combining medications."
        ),
    },
    {
        "id": "doc_med_02",
        "domain": "medical",
        "title": "Pediatric Dosage Recommendations",
        "text": (
            "Pediatric dosages differ significantly from adult dosages. For children aged 5-12, "
            "standard dosage of Medication Alpha is 5mg daily, whereas adult dosage is 10mg "
            "to 20mg daily depending on severity."
        ),
    },
    {
        "id": "doc_leg_01",
        "domain": "legal",
        "title": "US Marriage Laws & Age Limits",
        "text": (
            "The legal age of marriage without parental consent in most US states is 18. "
            "In Nebraska, the legal age is 19, and in Mississippi it is 21. Exceptions "
            "exist for emancipated minors with judicial approval."
        ),
    },
    {
        "id": "doc_ecom_01",
        "domain": "e-commerce",
        "title": "Smartphone Comparison: Galaxy vs iPhone",
        "text": (
            "The Samsung Galaxy S24 features a 6.7-inch AMOLED screen and 12GB RAM priced at $999. "
            "The Apple iPhone 15 Pro features a 6.1-inch Super Retina display and titanium body "
            "priced at $1099. Both support 5G networks."
        ),
    },
    {
        "id": "doc_edu_01",
        "domain": "education",
        "title": "University Admission Requirements",
        "text": (
            "Undergraduate admission requires a minimum high school GPA of 3.0, two recommendation "
            "letters, and standardized test scores (SAT > 1200 or ACT > 25). International "
            "students must demonstrate English proficiency via TOEFL or IELTS."
        ),
    },
]


def _tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def _cosine_similarity(vec1: Counter[str], vec2: Counter[str]) -> float:
    """Compute cosine similarity between two word frequency vectors."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[w] * vec2[w] for w in intersection)

    sum1 = sum(v**2 for v in vec1.values())
    sum2 = sum(v**2 for v in vec2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return numerator / denominator


class InMemoryRetriever(Retriever):
    """In-memory retriever implementation with zero external DB dependencies."""

    def __init__(self, corpus: list[dict[str, str]] | None = None) -> None:
        self._corpus = corpus or _DEFAULT_CORPUS
        self._vectorized_corpus = [
            (doc, Counter(_tokenize(doc["text"] + " " + doc["title"]))) for doc in self._corpus
        ]

    def search(self, query: str, top_k: int = 3) -> list[Evidence]:
        """Search in-memory corpus by cosine similarity."""
        q_vec = Counter(_tokenize(query))
        scored: list[tuple[dict[str, str], float]] = []

        for doc, doc_vec in self._vectorized_corpus:
            score = _cosine_similarity(q_vec, doc_vec)
            scored.append((doc, score))

        # Sort by score descending
        scored.sort(key=lambda item: item[1], reverse=True)

        results: list[Evidence] = []
        for doc, score in scored[:top_k]:
            results.append(
                Evidence(
                    doc_id=doc["id"],
                    text=doc["text"],
                    score=round(score, 4),
                    source=doc.get("domain", "corpus"),
                    retrieved_by=f"in_memory_query:{query[:30]}",
                )
            )

        return results
