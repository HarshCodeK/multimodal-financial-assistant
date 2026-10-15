import base64
import binascii
import io
from enum import Enum
from PIL import Image
from pydantic import BaseModel, Field, field_validator

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


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
    image_data_url: str = Field(min_length=30, max_length=12_000_000)

    @field_validator("image_data_url")
    @classmethod
    def validate_data_url(cls, value: str) -> str:
        if not value.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
            raise ValueError("image_data_url must be a PNG or JPEG data URL")
        header, encoded = value.split(",", 1)
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image_data_url must contain valid base64") from exc
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("image payload is empty or exceeds the 8 MiB limit")
        try:
            image = Image.open(io.BytesIO(raw))
            expected_format = "PNG" if header.startswith("data:image/png") else "JPEG"
            if image.format != expected_format:
                raise ValueError("image bytes do not match the declared image type")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("image dimensions exceed the supported pixel limit")
            image.verify()
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError("image_data_url is not a valid supported image") from exc
        return value


class ImageResponse(BaseModel):
    answer: str


class IngestResponse(BaseModel):
    source: str
    chunks_indexed: int
    pages_extracted: int
