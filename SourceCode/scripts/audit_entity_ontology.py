#!/usr/bin/env python3
"""Audit entity type conflicts and safely backfill curated identities.

Default mode is read-only. ``--apply-curated`` only merges identities whose
aliases and canonical type are explicitly curated in entity_resolution.py.
``--openai-review`` asks the configured model to review unresolved conflicts
using excerpts from multiple trusted indexed sources; it never mutates Neo4j.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from openai import OpenAI

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from src.api.config import get_settings
from src.api.entity_resolution import (
    forced_curated_identities,
    infer_unambiguous_entity_type,
    is_valid_entity_surface,
    normalize_search_key,
)
from src.api.graph_extraction import canonical_entity_id
from src.api.graph_ontology import CANONICAL_RELATIONSHIP_TYPES


def _unique(values: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _load_entities(session: Any) -> list[dict[str, Any]]:
    return [
        record.data()
        for record in session.run(
            """
            MATCH (entity:Entity {pipeline: 'pdf'})
            RETURN entity.canonicalId AS canonicalId,
                   entity.name AS name,
                   entity.entityType AS entityType,
                   coalesce(entity.aliases, []) AS aliases,
                   coalesce(entity.searchKeys, []) AS searchKeys,
                   coalesce(entity.sourceIds, []) AS sourceIds
            ORDER BY entity.name
            """
        )
    ]


def _conflicts(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for entity in entities:
        keys = {
            normalize_search_key(str(entity.get("name") or "")),
            *(
                normalize_search_key(str(alias))
                for alias in entity.get("aliases") or []
            ),
        }
        for key in keys:
            if key:
                groups[key][str(entity["canonicalId"])] = entity

    conflicts: list[dict[str, Any]] = []
    seen_id_sets: set[tuple[str, ...]] = set()
    for key, by_id in groups.items():
        types = {
            str(entity.get("entityType") or "")
            for entity in by_id.values()
        }
        if len(types) < 2:
            continue
        ids = tuple(sorted(by_id))
        if ids in seen_id_sets:
            continue
        seen_id_sets.add(ids)
        conflicts.append({
            "key": key,
            "types": sorted(types),
            "entities": [
                {
                    "canonicalId": entity["canonicalId"],
                    "name": entity.get("name") or "",
                    "entityType": entity.get("entityType") or "",
                    "sourceCount": len(entity.get("sourceIds") or []),
                }
                for entity in by_id.values()
            ],
        })
    return sorted(
        conflicts,
        key=lambda item: (-len(item["entities"]), item["key"]),
    )


def _curated_merge_groups(
    entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for canonical_name, entity_type, aliases in forced_curated_identities():
        identity_keys = {
            normalize_search_key(value)
            for value in (canonical_name, *aliases)
        }
        matched = []
        for entity in entities:
            entity_keys = {
                normalize_search_key(str(entity.get("name") or "")),
                *(
                    normalize_search_key(str(alias))
                    for alias in entity.get("aliases") or []
                ),
                *(str(key) for key in entity.get("searchKeys") or []),
            }
            if identity_keys & entity_keys:
                matched.append(entity)
        if not matched:
            continue
        target_id = canonical_entity_id(canonical_name, entity_type)
        groups.append({
            "canonicalName": canonical_name,
            "entityType": entity_type,
            "targetId": target_id,
            "aliases": _unique([canonical_name, *aliases, *[
                alias
                for entity in matched
                for alias in (
                    [str(entity.get("name") or "")]
                    + list(entity.get("aliases") or [])
                )
            ]]),
            "searchKeys": _unique([
                normalize_search_key(value)
                for value in (canonical_name, *aliases)
            ] + [
                str(key)
                for entity in matched
                for key in entity.get("searchKeys") or []
            ]),
            "sourceIds": _unique([
                str(source_id)
                for entity in matched
                for source_id in entity.get("sourceIds") or []
            ]),
            "oldIds": sorted({
                str(entity["canonicalId"])
                for entity in matched
                if str(entity["canonicalId"]) != target_id
            }),
        })
    return groups


def _deterministic_merge_groups(
    entities: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    excluded_ids: set[str],
) -> list[dict[str, Any]]:
    by_id = {str(entity["canonicalId"]): entity for entity in entities}
    groups: list[dict[str, Any]] = []
    for conflict in conflicts:
        entity_type = infer_unambiguous_entity_type(conflict["key"])
        conflict_ids = {
            str(item["canonicalId"]) for item in conflict["entities"]
        }
        if not entity_type or conflict_ids & excluded_ids:
            continue
        matched = [
            by_id[entity_id]
            for entity_id in conflict_ids
            if entity_id in by_id
        ]
        preferred = next(
            (
                entity for entity in matched
                if str(entity.get("entityType") or "") == entity_type
                and normalize_search_key(str(entity.get("name") or ""))
                == conflict["key"]
            ),
            matched[0] if matched else None,
        )
        if preferred is None:
            continue
        canonical_name = str(preferred.get("name") or "").strip()
        target_id = canonical_entity_id(canonical_name, entity_type)
        groups.append({
            "canonicalName": canonical_name,
            "entityType": entity_type,
            "targetId": target_id,
            "aliases": _unique([
                canonical_name,
                *[
                    value
                    for entity in matched
                    for value in (
                        [str(entity.get("name") or "")]
                        + list(entity.get("aliases") or [])
                    )
                ],
            ]),
            "searchKeys": _unique([
                normalize_search_key(canonical_name),
                *[
                    str(key)
                    for entity in matched
                    for key in entity.get("searchKeys") or []
                ],
            ]),
            "sourceIds": _unique([
                str(source_id)
                for entity in matched
                for source_id in entity.get("sourceIds") or []
            ]),
            "oldIds": sorted(
                entity_id for entity_id in conflict_ids
                if entity_id != target_id
            ),
            "typeResolvedBy": "deterministic_name_prefix_backfill",
        })
    return groups


def _combine_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for group in groups:
        target_id = str(group["targetId"])
        current = combined.setdefault(target_id, {
            **group,
            "aliases": [],
            "searchKeys": [],
            "sourceIds": [],
            "oldIds": [],
        })
        for field in ("aliases", "searchKeys", "sourceIds", "oldIds"):
            current[field] = _unique([
                *current[field],
                *list(group.get(field) or []),
            ])
    return list(combined.values())


def _merge_groups(session: Any, groups: list[dict[str, Any]]) -> dict:
    mapping = {
        old_id: group["targetId"]
        for group in groups
        for old_id in group["oldIds"]
    }
    old_ids = sorted(mapping)
    relationships = []
    if old_ids:
        relationships = [
            record.data()
            for record in session.run(
                """
                MATCH (source:Entity)-[relation]->(target:Entity)
                WHERE source.canonicalId IN $oldIds
                   OR target.canonicalId IN $oldIds
                RETURN source.canonicalId AS sourceId,
                       target.canonicalId AS targetId,
                       type(relation) AS relationshipType,
                       properties(relation) AS properties
                """,
                oldIds=old_ids,
            )
        ]

    for group in groups:
        session.run(
            """
            MERGE (target:Entity {canonicalId: $targetId})
            ON CREATE SET target.createdAt = datetime()
            SET target.name = $canonicalName,
                target.entityType = $entityType,
                target.pipeline = 'pdf',
                target.aliases = $aliases,
                target.searchKeys = $searchKeys,
                target.sourceIds = $sourceIds,
                target.typeResolvedBy = $typeResolvedBy,
                target.updatedAt = datetime()
            """,
            **{
                **group,
                "typeResolvedBy": group.get(
                    "typeResolvedBy",
                    "curated_identity_backfill",
                ),
            },
        ).consume()
        if group["oldIds"]:
            session.run(
                """
                MATCH (chunk:AIChunk)-[mention:MENTIONS]->(old:Entity)
                WHERE old.canonicalId IN $oldIds
                MATCH (target:Entity {canonicalId: $targetId})
                MERGE (chunk)-[replacement:MENTIONS]->(target)
                ON CREATE SET replacement.createdAt = datetime()
                SET replacement.pipeline = 'pdf',
                    replacement.sourceId = coalesce(
                      replacement.sourceId,
                      mention.sourceId
                    ),
                    replacement.pageNumber = coalesce(
                      replacement.pageNumber,
                      mention.pageNumber
                    ),
                    replacement.surfaceForms = reduce(
                      values = coalesce(replacement.surfaceForms, []),
                      value IN coalesce(mention.surfaceForms, []) |
                      CASE
                        WHEN value IN values THEN values
                        ELSE values + value
                      END
                    )
                DELETE mention
                """,
                oldIds=group["oldIds"],
                targetId=group["targetId"],
            ).consume()

    recreated = 0
    for relationship in relationships:
        relationship_type = str(relationship["relationshipType"])
        if relationship_type not in CANONICAL_RELATIONSHIP_TYPES:
            continue
        source_id = mapping.get(
            str(relationship["sourceId"]),
            str(relationship["sourceId"]),
        )
        target_id = mapping.get(
            str(relationship["targetId"]),
            str(relationship["targetId"]),
        )
        if source_id == target_id:
            continue
        session.run(
            f"""
            MATCH (source:Entity {{canonicalId: $sourceId}})
            MATCH (target:Entity {{canonicalId: $targetId}})
            MERGE (source)-[relation:`{relationship_type}`]->(target)
            SET relation += $properties,
                relation.pipeline = 'pdf',
                relation.updatedAt = datetime()
            """,
            sourceId=source_id,
            targetId=target_id,
            properties=relationship.get("properties") or {},
        ).consume()
        recreated += 1

    if old_ids:
        session.run(
            """
            MATCH (old:Entity)
            WHERE old.canonicalId IN $oldIds
            DETACH DELETE old
            """,
            oldIds=old_ids,
        ).consume()

    # Curated geographical context for the 1954 agreement.
    agreement_id = canonical_entity_id("Hiệp định Genève", "EVENT")
    geneva_id = canonical_entity_id("Genève", "LOCATION")
    switzerland_id = canonical_entity_id("Thụy Sĩ", "STATE")
    session.run(
        """
        MERGE (agreement:Entity {canonicalId: $agreementId})
        SET agreement.name = 'Hiệp định Genève',
            agreement.entityType = 'EVENT',
            agreement.pipeline = 'pdf',
            agreement.updatedAt = datetime()
        MERGE (geneva:Entity {canonicalId: $genevaId})
        ON CREATE SET geneva.createdAt = datetime()
        SET geneva.name = 'Genève',
            geneva.entityType = 'LOCATION',
            geneva.pipeline = 'pdf',
            geneva.aliases = ['Genève', 'Geneva', 'Giơnevơ', 'Giơnơvơ'],
            geneva.searchKeys = ['geneve', 'geneva', 'gionevo', 'gionovo'],
            geneva.typeResolvedBy = 'curated_identity_backfill',
            geneva.updatedAt = datetime()
        MERGE (switzerland:Entity {canonicalId: $switzerlandId})
        ON CREATE SET switzerland.createdAt = datetime()
        SET switzerland.name = 'Thụy Sĩ',
            switzerland.entityType = 'STATE',
            switzerland.pipeline = 'pdf',
            switzerland.aliases = ['Thụy Sĩ', 'Switzerland'],
            switzerland.searchKeys = ['thuy si', 'switzerland'],
            switzerland.typeResolvedBy = 'curated_identity_backfill',
            switzerland.updatedAt = datetime()
        MERGE (agreement)-[signedAt:OCCURRED_AT]->(geneva)
        SET signedAt.evidence = 'Hiệp định Genève năm 1954 được ký tại Genève, Thụy Sĩ.',
            signedAt.pipeline = 'curated',
            signedAt.updatedAt = datetime()
        MERGE (geneva)-[partOf:PART_OF]->(switzerland)
        SET partOf.evidence = 'Genève là một thành phố của Thụy Sĩ.',
            partOf.pipeline = 'curated',
            partOf.updatedAt = datetime()
        """,
        agreementId=agreement_id,
        genevaId=geneva_id,
        switzerlandId=switzerland_id,
    ).consume()
    return {
        "mergedNodeCount": len(old_ids),
        "recreatedRelationshipCount": recreated,
        "curatedGroupCount": len(groups),
    }


def _evidence_for_conflict(session: Any, conflict: dict[str, Any]) -> list[dict]:
    ids = [item["canonicalId"] for item in conflict["entities"]]
    return [
        record.data()
        for record in session.run(
            """
            MATCH (source:KnowledgeSource)-[:HAS_PAGE]->(:KnowledgePage)
                  -[:HAS_CHUNK]->(chunk:AIChunk)-[:MENTIONS]->(entity:Entity)
            WHERE entity.canonicalId IN $ids
              AND source.trustLevel IN ['official', 'academic']
            RETURN source.title AS source,
                   source.trustLevel AS trustLevel,
                   entity.name AS entityName,
                   entity.entityType AS currentType,
                   chunk.pageStart AS page,
                   substring(chunk.text, 0, 700) AS excerpt
            ORDER BY source.title, chunk.pageStart
            LIMIT 12
            """,
            ids=ids,
        )
    ]


def _openai_review(
    client: OpenAI,
    model: str,
    session: Any,
    conflicts: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    payload = []
    for conflict in conflicts:
        # Một âm tiết sau khi bỏ dấu thường gom nhầm các từ đồng âm như
        # đường/Đường/Đuống hoặc Diêm/Diệm; không gửi các nhóm này để model
        # "đoán" một canonical identity.
        if len(str(conflict["key"]).split()) < 2:
            continue
        evidence = _evidence_for_conflict(session, conflict)
        distinct_sources = {
            str(item.get("source") or "") for item in evidence
        }
        if len(distinct_sources) < 2:
            continue
        payload.append({
            **conflict,
            "evidence": evidence,
        })
        if len(payload) >= limit:
            break
    if not payload:
        return []

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "entity_type_audit",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "key": {"type": "string"},
                                    "canonical_name": {"type": "string"},
                                    "canonical_type": {
                                        "type": "string",
                                        "enum": [
                                            "PERSON", "LOCATION",
                                            "ORGANIZATION", "EVENT",
                                            "PERIOD", "DOCUMENT", "STATE",
                                            "WEAPON", "NOT_AN_ENTITY",
                                            "UNRESOLVED",
                                        ],
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                    "reason": {"type": "string"},
                                },
                                "required": [
                                    "key", "canonical_name",
                                    "canonical_type", "confidence", "reason",
                                ],
                            },
                        },
                    },
                    "required": ["items"],
                },
            },
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn kiểm định ontology lịch sử từ bằng chứng trong nhiều "
                    "nguồn official/academic. LOCATION chỉ là địa điểm; EVENT "
                    "gồm chiến dịch, hội nghị và hiệp định theo ontology ứng "
                    "dụng; DOCUMENT là văn bản/sắc lệnh/nghị quyết/tác phẩm. "
                    "Danh từ chung hoặc khái niệm như cao su, đường sắt, địa "
                    "chủ không phải tên riêng thì trả NOT_AN_ENTITY. "
                    "Chỉ kết luận khi các đoạn cùng nói về một đối tượng. Nếu "
                    "tên đồng âm, chỉ khác nhau do bỏ dấu, bằng chứng mâu "
                    "thuẫn hoặc chưa đủ, trả UNRESOLVED. Mỗi key đầu vào chỉ "
                    "được trả đúng một item tổng hợp; không trả một item cho "
                    "từng node/type. Không dùng kiến thức ngoài bằng chứng."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
    )
    parsed = json.loads(response.choices[0].message.content or "{}")
    return list(parsed.get("items") or [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-curated", action="store_true")
    parser.add_argument("--apply-deterministic", action="store_true")
    parser.add_argument("--openai-review", action="store_true")
    parser.add_argument("--review-limit", type=int, default=30)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    try:
        with driver.session(database=settings.neo4j_database) as session:
            entities = _load_entities(session)
            before_conflicts = _conflicts(entities)
            curated_groups = _curated_merge_groups(entities)
            curated_ids = {
                old_id
                for group in curated_groups
                for old_id in group["oldIds"]
            }
            deterministic_groups = _deterministic_merge_groups(
                entities,
                before_conflicts,
                curated_ids,
            )
            selected_groups = [
                *(curated_groups if args.apply_curated else []),
                *(
                    deterministic_groups
                    if args.apply_deterministic
                    else []
                ),
            ]
            migration = {
                "mode": "dry-run",
                "curatedGroupCount": len(curated_groups),
                "curatedMergeCandidateCount": sum(
                    len(group["oldIds"]) for group in curated_groups
                ),
                "deterministicGroupCount": len(deterministic_groups),
                "deterministicMergeCandidateCount": sum(
                    len(group["oldIds"]) for group in deterministic_groups
                ),
                "deterministicPreview": [
                    {
                        "canonicalName": group["canonicalName"],
                        "entityType": group["entityType"],
                        "mergeCount": len(group["oldIds"]),
                    }
                    for group in deterministic_groups[:40]
                ],
            }
            if selected_groups:
                migration = _merge_groups(
                    session,
                    _combine_groups(selected_groups),
                )
                migration["mode"] = "applied"
                migration["curatedApplied"] = args.apply_curated
                migration["deterministicApplied"] = args.apply_deterministic
                entities = _load_entities(session)

            report: dict[str, Any] = {
                "entityCount": len(entities),
                "conflictCountBefore": len(before_conflicts),
                "conflictCountAfter": len(_conflicts(entities)),
                "curatedMigration": migration,
                "conflicts": _conflicts(entities),
                "invalidSurfaceCount": sum(
                    1 for entity in entities
                    if not is_valid_entity_surface(
                        str(entity.get("name") or "")
                    )
                ),
                "invalidSurfacePreview": [
                    {
                        "canonicalId": entity["canonicalId"],
                        "name": entity.get("name") or "",
                        "entityType": entity.get("entityType") or "",
                    }
                    for entity in entities
                    if not is_valid_entity_surface(
                        str(entity.get("name") or "")
                    )
                ][:80],
            }
            if args.openai_review:
                report["openaiReview"] = _openai_review(
                    OpenAI(api_key=settings.openai_api_key),
                    settings.openai_entity_model,
                    session,
                    report["conflicts"],
                    max(1, min(args.review_limit, 100)),
                )
            rendered = json.dumps(report, ensure_ascii=False, indent=2)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as output:
                    output.write(rendered)
                    output.write("\n")
            print(rendered)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
