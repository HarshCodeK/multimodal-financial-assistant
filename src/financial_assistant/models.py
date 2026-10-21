"""Typed records shared by PDF extraction, retrieval, and grounded answer generation."""

from dataclasses import dataclass, field
from enum import Enum


class SourceType(str, Enum):
    pdf = "pdf"
    image = "image"


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    source_type: SourceType
    text: str
    page: int | None = None


@dataclass(frozen=True)
class RetrievedChunk(DocumentChunk):
    distance: float = 0.0


@dataclass
class Answer:
    supported: bool
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    context_word_count: int = 0
    context_token_estimate: int = 0
    prompt_preview: str = ""
