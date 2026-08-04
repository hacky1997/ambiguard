"""Retrieval base protocol and evidence types."""

from __future__ import annotations

from typing import Protocol

from app.graph.state import Evidence


class Retriever(Protocol):
    """Protocol for retrieval engines (in-memory, Qdrant, etc.)."""

    def search(self, query: str, top_k: int = 3) -> list[Evidence]:
        """Search corpus for top_k relevant evidence documents."""
        ...
