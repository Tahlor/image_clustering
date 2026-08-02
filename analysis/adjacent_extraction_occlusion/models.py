from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

FIELD_WEIGHTS = {
    "given_name": 1.5,
    "surname": 2.25,
    "petition_number": 3.0,
    "birth_year": 1.1,
    "birth_city": 0.85,
    "birth_country": 0.9,
    "birth_day": 0.55,
    "birth_month": 0.55,
    "birth_county": 0.65,
    "birth_state": 0.65,
    "marriage_year": 0.8,
    "marriage_city": 0.65,
    "marriage_country": 0.65,
    "immigration_city": 0.7,
    "immigration_country": 0.75,
    "arrival_year": 0.7,
    "vessel": 0.75,
    "mode_of_travel": 0.55,
    "age": 0.45,
    "gender": 0.35,
    "race": 0.35,
    "event_year": 0.3,
    "event_day": 0.15,
    "event_month": 0.15,
    "event_city": 0.2,
    "event_county": 0.2,
    "event_state": 0.2,
}


@dataclass(frozen=True)
class Config:
    max_gap: int = 3
    automatic_link_threshold: float = 0.72
    review_link_threshold: float = 0.58
    max_conflict_rate: float = 0.28
    min_extraction_containment: float = 0.52
    min_identity_similarity: float = 0.45
    strong_embedding_similarity: float = 0.93
    strong_cv_same_scene: float = 0.70
    min_occlusion_asymmetry: float = 0.12
    min_role_margin: float = 0.08
    extraction_weight: float = 0.68
    embedding_weight: float = 0.18
    cv_weight: float = 0.14
    conflict_penalty: float = 0.42

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "Config":
        if not value:
            return cls()
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**value)


@dataclass
class Record:
    role: str
    event_type: str | None
    fields: dict[str, str]


@dataclass
class Extraction:
    image_id: str
    source_filename: str
    sequence_key: str
    sequence_index: int
    records: list[Record]
    quality: dict[str, float]
    run_count: int = 1
    run_sources: list[str] = field(default_factory=list)

    @property
    def field_mass(self) -> float:
        return sum(
            FIELD_WEIGHTS.get(key, 0.25)
            for record in self.records
            for key, value in record.fields.items()
            if value
        )


@dataclass
class CvEvidence:
    same_scene_probability: float | None = None
    occlusion_probability: float | None = None
    relation: str | None = None
