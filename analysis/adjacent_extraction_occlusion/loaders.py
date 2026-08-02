from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from models import CvEvidence, Extraction, Record

ROLE_ALIASES = {
    "primary": "self",
    "principal": "self",
    "applicant": "self",
    "petitioner": "self",
    "self": "self",
    "spouse": "spouse",
    "child": "child",
}
SOURCE_PATTERNS = (
    re.compile(r"(?P<s>63129_[A-Za-z0-9]+_[A-Za-z0-9]+-\d+)$"),
    re.compile(r"(?P<s>i\d+-\d+)$", re.I),
)
SEQUENCE_RE = re.compile(r"^(?P<series>.+)-(?P<index>\d+)$")


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def canonical_image_id(value: str) -> str:
    name = Path(str(value)).name
    stem = re.sub(r"(?i)\.(jpg|jpeg|png|j2k|jp2|tif|tiff|webp)$", "", name)
    stem = re.sub(r"\(\d+\)$", "", stem)
    for pattern in SOURCE_PATTERNS:
        match = pattern.search(stem)
        if match:
            return match.group("s")
    return stem


def sequence_parts(image_id: str) -> tuple[str, int]:
    match = SEQUENCE_RE.match(image_id)
    if not match:
        raise ValueError(f"Cannot derive sequence from {image_id!r}")
    return match.group("series"), int(match.group("index"))


def pair_key(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((canonical_image_id(first), canonical_image_id(second))))


def role_name(value: Any) -> str:
    normalized = norm(value)
    return ROLE_ALIASES.get(normalized, normalized or "self")


def clean_fields(fields: Mapping[str, Any]) -> dict[str, str]:
    ignored = {"event_type", "role", "record_index", "record_id"}
    return {
        str(key): str(value)
        for key, value in fields.items()
        if key not in ignored
        and value not in (None, "")
        and not isinstance(value, (dict, list))
    }


def _event_fields(event: Mapping[str, Any]) -> dict[str, Any]:
    place = event.get("event_place") or {}
    result = {
        "event_city": place.get("city"),
        "event_county": place.get("county"),
        "event_state": place.get("state"),
    }
    dates = event.get("date_observations") or []
    if dates:
        first_date_types = {"Declaration", "Petition", "Deposition", "Oath"}
        date = dates[0] if event.get("event_type") in first_date_types else dates[-1]
        result.update(
            event_day=date.get("day"),
            event_month=date.get("month"),
            event_year=date.get("year"),
        )
    return result


def _person_fields(record: Mapping[str, Any], spouse: bool = False) -> dict[str, Any]:
    if spouse:
        person = ((record.get("spouse") or {}).get("person") or {})
    else:
        person = record.get("person") or {}
    name = person.get("name") or {}
    birth = person.get("birth") or {}
    date = birth.get("date") or {}
    place = birth.get("place") or {}
    return {
        "prefix": name.get("prefix"),
        "given_name": name.get("given_name"),
        "surname": name.get("surname"),
        "suffix": name.get("suffix"),
        "birth_day": date.get("day"),
        "birth_month": date.get("month"),
        "birth_year": date.get("year"),
        "birth_city": place.get("city"),
        "birth_county": place.get("county"),
        "birth_state": place.get("state"),
        "birth_country": place.get("country"),
    }


def _flatten_nested(
    record: Mapping[str, Any],
    event_type: str | None,
    base: Mapping[str, Any],
) -> list[Record]:
    fields = dict(base)
    fields.update(_person_fields(record))
    person = record.get("person") or {}
    marriage = record.get("marriage") or {}
    marriage_date = marriage.get("date") or {}
    marriage_place = marriage.get("place") or {}
    immigration = record.get("immigration_place") or {}
    arrival = record.get("arrival_date") or {}
    departure = record.get("departure_date") or {}
    fields.update(
        petition_number=record.get("petition_number"),
        age=person.get("age"),
        gender=person.get("gender"),
        race=person.get("race"),
        marriage_day=marriage_date.get("day"),
        marriage_month=marriage_date.get("month"),
        marriage_year=marriage_date.get("year"),
        marriage_city=marriage_place.get("city"),
        marriage_county=marriage_place.get("county"),
        marriage_state=marriage_place.get("state"),
        marriage_country=marriage_place.get("country"),
        immigration_city=immigration.get("city"),
        immigration_county=immigration.get("county"),
        immigration_state=immigration.get("state"),
        immigration_country=immigration.get("country"),
        arrival_day=arrival.get("day"),
        arrival_month=arrival.get("month"),
        arrival_year=arrival.get("year"),
        departure_day=departure.get("day"),
        departure_month=departure.get("month"),
        departure_year=departure.get("year"),
        mode_of_travel=record.get("mode_of_travel"),
        vessel=record.get("vessel"),
        has_photo=record.get("has_photo"),
    )
    output = [
        Record(
            role_name(record.get("record_role") or record.get("role")),
            event_type,
            clean_fields(fields),
        )
    ]
    if output[0].role == "self":
        spouse_fields = dict(base)
        spouse_fields.update(_person_fields(record, spouse=True))
        spouse_fields = clean_fields(spouse_fields)
        if any(
            key in spouse_fields
            for key in ("given_name", "surname", "birth_year", "birth_city")
        ):
            output.append(Record("spouse", event_type, spouse_fields))
    return output


def parse_payload(payload: Mapping[str, Any]) -> list[Record]:
    if isinstance(payload.get("records"), list):
        return [
            Record(
                role_name(record.get("role") or record.get("record_role")),
                record.get("event_type"),
                clean_fields(record.get("fields") or {}),
            )
            for record in payload["records"]
            if isinstance(record, Mapping)
        ]
    output: list[Record] = []
    for event in payload.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        for record in event.get("records") or []:
            if isinstance(record, Mapping):
                output.extend(
                    _flatten_nested(
                        record,
                        event.get("event_type"),
                        _event_fields(event),
                    )
                )
    return output


def iter_json(path: Path) -> Iterator[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, Mapping):
                    yield value
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        yield from (item for item in value if isinstance(item, Mapping))
    elif isinstance(value, Mapping):
        for key in ("items", "results", "extractions"):
            if isinstance(value.get(key), list):
                yield from (
                    item for item in value[key] if isinstance(item, Mapping)
                )
                return
        yield value


def extraction_from_object(obj: Mapping[str, Any], source: str) -> Extraction:
    parsed = obj.get("parsed_response")
    payload = parsed if isinstance(parsed, Mapping) else obj
    raw_name = (
        obj.get("source_filename")
        or payload.get("source_filename")
        or obj.get("image_stem")
        or obj.get("image_id")
        or obj.get("filename")
    )
    if not raw_name:
        raise ValueError(f"No image identifier in {source}")
    image_id = canonical_image_id(str(raw_name))
    sequence_key, sequence_index = sequence_parts(image_id)
    quality = payload.get("quality") or obj.get("quality") or {}
    return Extraction(
        image_id=image_id,
        source_filename=str(obj.get("source_filename") or raw_name),
        sequence_key=sequence_key,
        sequence_index=sequence_index,
        records=parse_payload(payload),
        quality={
            str(key): float(value)
            for key, value in quality.items()
            if isinstance(value, (int, float))
        },
        run_sources=[source],
    )


def _record_similarity(first: Record, second: Record) -> float:
    if first.role != second.role:
        return 0.0
    fields = ("given_name", "surname", "petition_number", "birth_year")
    scores = []
    for key in fields:
        left, right = norm(first.fields.get(key)), norm(second.fields.get(key))
        if left and right:
            scores.append(1.0 if left == right else 0.0)
    if first.event_type and second.event_type:
        scores.append(float(first.event_type == second.event_type))
    return sum(scores) / len(scores) if scores else 0.45


def _consensus_record(records: Sequence[Record]) -> Record:
    fields: dict[str, str] = {}
    for key in {key for record in records for key in record.fields}:
        values = [record.fields[key] for record in records if record.fields.get(key)]
        normalized_counts = Counter(norm(value) for value in values)
        winner = normalized_counts.most_common(1)[0][0]
        fields[key] = max(
            (value for value in values if norm(value) == winner),
            key=len,
        )
    event_types = [record.event_type for record in records if record.event_type]
    return Record(
        records[0].role,
        Counter(event_types).most_common(1)[0][0] if event_types else None,
        fields,
    )


def merge_extraction_runs(runs: Sequence[Extraction]) -> Extraction:
    if not runs:
        raise ValueError("No extraction runs supplied")
    groups: list[list[Record]] = []
    for run in runs:
        used: set[int] = set()
        for group in groups:
            best = max(
                (
                    (_record_similarity(group[0], record), index)
                    for index, record in enumerate(run.records)
                    if index not in used
                ),
                default=(0.0, -1),
            )
            if best[0] >= 0.55:
                group.append(run.records[best[1]])
                used.add(best[1])
        groups.extend([[record] for index, record in enumerate(run.records) if index not in used])
    base = max(runs, key=lambda run: (run.field_mass, len(run.records)))
    quality: dict[str, float] = {}
    for key in {key for run in runs for key in run.quality}:
        values = [run.quality[key] for run in runs if key in run.quality]
        quality[key] = sum(values) / len(values)
    return Extraction(
        image_id=base.image_id,
        source_filename=base.source_filename,
        sequence_key=base.sequence_key,
        sequence_index=base.sequence_index,
        records=[_consensus_record(group) for group in groups],
        quality=quality,
        run_count=len(runs),
        run_sources=sorted({source for run in runs for source in run.run_sources}),
    )


merge_runs = merge_extraction_runs


def load_extractions(paths: Sequence[Path]) -> list[Extraction]:
    by_image: dict[str, list[Extraction]] = defaultdict(list)
    for path in paths:
        for obj in iter_json(path):
            item = extraction_from_object(obj, str(path))
            by_image[item.image_id].append(item)
    return sorted(
        (merge_extraction_runs(runs) for runs in by_image.values()),
        key=lambda item: (item.sequence_key, item.sequence_index),
    )


def load_embeddings(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=False)
        id_key = "filenames" if "filenames" in data.files else "image_ids" if "image_ids" in data.files else None
        if id_key and "embeddings" in data.files:
            return {
                canonical_image_id(str(image_id)): np.asarray(vector, dtype=np.float32)
                for image_id, vector in zip(data[id_key], data["embeddings"], strict=True)
            }
        return {
            canonical_image_id(key): np.asarray(data[key], dtype=np.float32)
            for key in data.files
        }
    output = {}
    for obj in iter_json(path):
        raw_id = obj.get("image_id") or obj.get("filename") or obj.get("source_filename")
        vector = obj.get("embedding") or obj.get("vector")
        if raw_id and isinstance(vector, list):
            output[canonical_image_id(str(raw_id))] = np.asarray(vector, dtype=np.float32)
    return output


def _float(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def load_cv(path: Path | None) -> dict[tuple[str, str], CvEvidence]:
    if path is None:
        return {}
    rows: list[Mapping[str, Any]]
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = list(iter_json(path))
    output = {}
    for row in rows:
        first = row.get("first_image_id") or row.get("image_a")
        second = row.get("second_image_id") or row.get("image_b")
        if not first or not second:
            continue
        output[pair_key(str(first), str(second))] = CvEvidence(
            same_scene_probability=_float(
                row,
                "same_document_probability",
                "same_scene_probability",
                "same_probability",
            ),
            occlusion_probability=_float(
                row,
                "occluded_given_same_probability",
                "same_occluded_probability",
                "occlusion_probability",
            ),
            relation=str(row.get("relation") or row.get("branch") or "") or None,
        )
    return output


def load_quality(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    output = {}
    for obj in iter_json(path):
        raw_id = obj.get("image_id") or obj.get("source_filename") or obj.get("filename")
        quality = obj.get("quality") or obj
        if raw_id:
            output[canonical_image_id(str(raw_id))] = {
                str(key): float(value)
                for key, value in quality.items()
                if isinstance(value, (int, float))
            }
    return output
