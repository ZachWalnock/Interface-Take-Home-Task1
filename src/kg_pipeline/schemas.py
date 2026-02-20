from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    left: float
    top: float
    right: float
    bottom: float
    label: str | None = None

    @field_validator("left", "top", "right", "bottom")
    @classmethod
    def validate_bounds(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("Bounding box values must be in [0.0, 1.0].")
        return value


class Measurement(BaseModel):
    kind: str = Field(description="pressure, temperature, setpoint, flow, etc")
    value: str | float | None = None
    unit: str | None = None
    raw_text: str | None = None
    confidence: float | None = None
    bbox: BoundingBox | None = None


class Entity(BaseModel):
    entity_id: str | None = None
    name: str
    entity_type: str
    description: str | None = None
    page_number: int
    bbox: BoundingBox | None = None
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    measurements: list[Measurement] = Field(default_factory=list)
    confidence: float | None = None


class Relation(BaseModel):
    relation_id: str | None = None
    source_ref: str
    target_ref: str
    relation_type: str
    line_id: str | None = None
    description: str | None = None
    page_number: int
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None


class PageExtraction(BaseModel):
    page_number: int
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DocumentExtraction(BaseModel):
    doc_id: str
    pages: list[PageExtraction] = Field(default_factory=list)
