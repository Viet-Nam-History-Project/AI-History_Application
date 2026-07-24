"""Entity and relationship extraction for the canonical PDF ingestion pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .entity_resolution import (
    clean_entity_name,
    is_valid_entity_surface,
    normalize_search_key,
    resolve_entity_name,
)
from .graph_ontology import (
    CANONICAL_RELATIONSHIP_TYPES,
    canonicalize_relationship_type,
)
from .pdf_ingestion import PdfChunk


ENTITY_TYPES = (
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "EVENT",
    "PERIOD",
    "DOCUMENT",
    "STATE",
    "WEAPON",
)
SEMANTIC_RELATIONSHIP_TYPES = tuple(sorted(
    CANONICAL_RELATIONSHIP_TYPES
    - {"HAS_STAGE", "HAS_EVENT", "HAS_PAGE", "HAS_CHUNK", "MENTIONS", "SUPPORTED_BY"}
))


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add_response(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_tokens = int(
            getattr(usage, "prompt_tokens", None)
            or getattr(usage, "input_tokens", None)
            or 0
        )
        output_tokens = int(
            getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_tokens", None)
            or 0
        )
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += int(
            getattr(usage, "total_tokens", None)
            or input_tokens + output_tokens
        )

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "totalTokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ExtractedEntity:
    canonical_id: str
    name: str
    entity_type: str
    aliases: tuple[str, ...] = ()
    search_keys: tuple[str, ...] = ()
    surface_forms: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractedRelationship:
    source_canonical_id: str
    target_canonical_id: str
    relationship_type: str
    evidence: str
    raw_predicate: str = ""


@dataclass
class ChunkGraphExtraction:
    chunk_id: str
    page_number: int
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {
            "chunkId": self.chunk_id,
            "pageNumber": self.page_number,
            "entities": [
                {
                    "canonicalId": entity.canonical_id,
                    "name": entity.name,
                    "entityType": entity.entity_type,
                    "aliases": list(entity.aliases),
                    "searchKeys": list(entity.search_keys),
                    "surfaceForms": list(entity.surface_forms),
                }
                for entity in self.entities
            ],
            "relationships": [
                {
                    "sourceCanonicalId": relation.source_canonical_id,
                    "targetCanonicalId": relation.target_canonical_id,
                    "type": relation.relationship_type,
                    "evidence": relation.evidence,
                    "rawPredicate": relation.raw_predicate,
                }
                for relation in self.relationships
            ],
        }


def _normalized_name(value: str) -> str:
    return clean_entity_name(value)


def _search_key(value: str) -> str:
    return normalize_search_key(value)


def canonical_entity_id(name: str, entity_type: str) -> str:
    """Create a stable ID without merging different entity types by accident."""
    key = _search_key(name)
    slug = re.sub(r"\s+", "-", key)[:96] or "entity"
    fingerprint = hashlib.sha256(
        f"{entity_type}:{key}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{entity_type.casefold()}:{slug}:{fingerprint}"


class PdfGraphExtractionService:
    def __init__(self, client: OpenAI, model: str, batch_size: int = 6) -> None:
        self.client = client
        self.model = model
        self.batch_size = max(1, min(batch_size, 12))

    @staticmethod
    def _response_format() -> dict[str, Any]:
        entity_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": list(ENTITY_TYPES)},
            },
            "required": ["name", "type"],
        }
        relationship_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "source": {"type": "string"},
                "source_type": {
                    "type": "string",
                    "enum": list(ENTITY_TYPES),
                },
                "target": {"type": "string"},
                "target_type": {
                    "type": "string",
                    "enum": list(ENTITY_TYPES),
                },
                "type": {
                    "type": "string",
                    "enum": list(SEMANTIC_RELATIONSHIP_TYPES),
                },
                "evidence": {"type": "string"},
                "raw_predicate": {"type": "string"},
            },
            "required": [
                "source",
                "source_type",
                "target",
                "target_type",
                "type",
                "evidence",
                "raw_predicate",
            ],
        }
        chunk_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "chunk_id": {"type": "string"},
                "entities": {"type": "array", "items": entity_schema},
                "relationships": {"type": "array", "items": relationship_schema},
            },
            "required": ["chunk_id", "entities", "relationships"],
        }
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "pdf_history_graph_extraction",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "chunks": {"type": "array", "items": chunk_schema},
                    },
                    "required": ["chunks"],
                },
            },
        }

    def extract(
        self,
        chunks: list[PdfChunk],
    ) -> tuple[list[ChunkGraphExtraction], TokenUsage]:
        results: list[ChunkGraphExtraction] = []
        usage = TokenUsage()
        for offset in range(0, len(chunks), self.batch_size):
            batch = chunks[offset:offset + self.batch_size]
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format=self._response_format(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn trích xuất đồ thị tri thức Lịch sử Việt Nam. "
                            "Chỉ lấy thực thể có tên riêng hoặc sự kiện/tổ chức/địa danh/vũ khí "
                            "cụ thể xuất hiện trong chính chunk. Không biến khái niệm chung, đại từ, "
                            "chức danh đứng một mình hoặc câu mô tả thành entity. Các từ như cao su, "
                            "đường sắt, địa chủ, nhân dân, chính phủ nói chung, hòa bình hoặc cách "
                            "mạng nói chung không phải entity nếu không có tên riêng đi kèm. Quan hệ phải có "
                            "bằng chứng trực tiếp, có hướng và cả hai đầu phải nằm trong entities "
                            "của cùng chunk. Giữ nguyên cách viết tên trong nguồn. Phân loại quốc gia "
                            "và chính thể là STATE; không biến ngày tháng, năm, số lượng hoặc quân số "
                            "thành EVENT/PERSON. LOCATION chỉ là địa điểm địa lý; không gán hiệp định, "
                            "hội nghị, chiến dịch hoặc trận đánh thành LOCATION. Trong ontology của "
                            "hệ thống này, hiệp định và hội nghị lịch sử được phân loại là EVENT; "
                            "DOCUMENT dành cho văn bản, sắc lệnh, nghị quyết, tuyên bố, sách hoặc tài "
                            "liệu. Nếu cùng một tên xuất hiện ở nhiều chunk, phải giữ nhất quán type. "
                            "raw_predicate là cụm động từ ngắn trong nguồn, còn "
                            "type bắt buộc chọn từ ontology đã cho."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "allowed_relationships": SEMANTIC_RELATIONSHIP_TYPES,
                                "chunks": [
                                    {
                                        "chunk_id": chunk.chunk_id,
                                        "page_number": chunk.page_start,
                                        "text": chunk.text,
                                    }
                                    for chunk in batch
                                ],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            )
            usage.add_response(response)
            payload = json.loads(response.choices[0].message.content or "{}")
            by_chunk_id = {
                str(item.get("chunk_id") or ""): item
                for item in payload.get("chunks", [])
                if isinstance(item, dict)
            }
            for chunk in batch:
                results.append(self._normalize_chunk(
                    chunk,
                    by_chunk_id.get(chunk.chunk_id, {}),
                ))
        return results, usage

    @staticmethod
    def _normalize_chunk(
        chunk: PdfChunk,
        payload: dict[str, Any],
    ) -> ChunkGraphExtraction:
        entities_by_id: dict[str, dict[str, Any]] = {}
        entity_id_by_typed_key: dict[tuple[str, str], str] = {}
        normalized_chunk_text = _search_key(chunk.text)
        for raw_entity in payload.get("entities", []):
            if not isinstance(raw_entity, dict):
                continue
            name = _normalized_name(raw_entity.get("name", ""))
            entity_type = str(raw_entity.get("type") or "").upper()
            key = _search_key(name)
            if (
                len(key) < 2
                or entity_type not in ENTITY_TYPES
                or key not in normalized_chunk_text
                or not is_valid_entity_surface(name)
            ):
                continue
            resolution = resolve_entity_name(name, entity_type)
            resolved_type = resolution.entity_type
            canonical_id = canonical_entity_id(
                resolution.canonical_name,
                resolved_type,
            )
            aggregate = entities_by_id.setdefault(canonical_id, {
                "name": resolution.canonical_name,
                "entityType": resolved_type,
                "aliases": [],
                "searchKeys": [],
                "surfaceForms": [],
            })
            if name not in aggregate["surfaceForms"]:
                aggregate["surfaceForms"].append(name)
            for alias in resolution.aliases:
                if alias not in aggregate["aliases"]:
                    aggregate["aliases"].append(alias)
            for search_key in resolution.search_keys:
                if search_key not in aggregate["searchKeys"]:
                    aggregate["searchKeys"].append(search_key)
                entity_id_by_typed_key[(resolved_type, search_key)] = canonical_id
                # Quan hệ do model trả về vẫn có thể dùng type thô trước khi
                # canonicalizer sửa; giữ ánh xạ này để không làm rơi cạnh.
                entity_id_by_typed_key[(entity_type, search_key)] = canonical_id
            entity_id_by_typed_key[(resolved_type, key)] = canonical_id
            entity_id_by_typed_key[(entity_type, key)] = canonical_id

        entities = [
            ExtractedEntity(
                canonical_id=canonical_id,
                name=item["name"],
                entity_type=item["entityType"],
                aliases=tuple(item["aliases"]),
                search_keys=tuple(item["searchKeys"]),
                surface_forms=tuple(item["surfaceForms"]),
            )
            for canonical_id, item in entities_by_id.items()
        ]
        entity_by_id = {
            entity.canonical_id: entity
            for entity in entities
        }

        relationships: list[ExtractedRelationship] = []
        seen_relationships: set[tuple[str, str, str]] = set()
        for raw_relation in payload.get("relationships", []):
            if not isinstance(raw_relation, dict):
                continue
            source_type = str(raw_relation.get("source_type") or "").upper()
            target_type = str(raw_relation.get("target_type") or "").upper()
            source_id = entity_id_by_typed_key.get((
                source_type,
                _search_key(raw_relation.get("source", "")),
            ))
            target_id = entity_id_by_typed_key.get((
                target_type,
                _search_key(raw_relation.get("target", "")),
            ))
            source = entity_by_id.get(source_id or "")
            target = entity_by_id.get(target_id or "")
            if source is None or target is None or source == target:
                continue
            relation_type = canonicalize_relationship_type(
                str(raw_relation.get("type") or "")
            )
            if relation_type not in SEMANTIC_RELATIONSHIP_TYPES:
                continue
            evidence = _normalized_name(raw_relation.get("evidence", ""))[:500]
            if not evidence:
                continue
            relation_key = (
                source.canonical_id,
                target.canonical_id,
                relation_type,
            )
            if relation_key in seen_relationships:
                continue
            seen_relationships.add(relation_key)
            relationships.append(ExtractedRelationship(
                source_canonical_id=source.canonical_id,
                target_canonical_id=target.canonical_id,
                relationship_type=relation_type,
                evidence=evidence,
                raw_predicate=_normalized_name(
                    raw_relation.get("raw_predicate", "")
                )[:200],
            ))

        return ChunkGraphExtraction(
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_start,
            entities=entities,
            relationships=relationships,
        )
