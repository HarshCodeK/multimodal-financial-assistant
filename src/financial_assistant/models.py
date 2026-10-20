"""Small typed records shared by the local extraction and retrieval pipeline."""

from dataclasses import dataclass, field
from enum import Enum


class SourceType(str, Enum):
    pdf = "pdf"
    image = "image"
    text = "text"


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
