from enum import Enum
from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    pdf = "pdf"
    image = "image"
    text = "text"


class DocumentChunk(BaseModel):
    source: str
    source_type: SourceType
    text: str = Field(min_length=1, max_length=20_000)
    page: int | None = Field(default=None, ge=1)


class RetrievedChunk(DocumentChunk):
    distance: float


class Answer(BaseModel):
    supported: bool
    answer: str
    sources: list[RetrievedChunk] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    supported: bool
    answer: str
    sources: list[RetrievedChunk] = Field(default_factory=list)


class ImageRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    image_data_url: str = Field(min_length=20, max_length=12_000_000)

    @field_validator("image_data_url")
    @classmethod
    def validate_data_url(cls, value: str) -> str:
        if not value.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
            raise ValueError("image_data_url must be a PNG or JPEG data URL")
        return value


class ImageResponse(BaseModel):
    answer: str


class IngestResponse(BaseModel):
    source: str
    chunks_indexed: int
    pages_extracted: int
