import json
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from neo4j import GraphDatabase

from .config import get_settings
from .entity_resolution import (
    expand_known_entity_aliases,
    normalize_search_key,
)
from .graph_extraction import ChunkGraphExtraction
from .history_facets import infer_history_facets
from .models import KnowledgeIndexMetadata
from .pdf_ingestion import PdfChunk, PdfPage
from .graph_ontology import (
    CANONICAL_RELATIONSHIP_TYPES,
    canonicalize_relationship_type,
    is_canonical_relationship_type,
)


VIETNAMESE_STOP_WORDS = {
    "ai", "bao", "biet", "cac", "cho", "co", "cua", "duoc", "gi", "hay", "khi", "la",
    "minh", "mot", "nam", "nay", "nhung", "noi", "o", "the", "thi", "toi", "trong",
    "tu", "va", "ve",
}

PDF_GRAPH_LABELS = frozenset({
    "KnowledgeSource",
    "KnowledgePage",
    "AIChunk",
    "Entity",
    "AIUsageLog",
    "AIQueryLog",
    "AIConfig",
})


def _json_safe_neo4j_value(value: Any, include_embeddings: bool = False) -> Any:
    """Convert Neo4j values to JSON-safe data without exposing vector payloads."""
    if isinstance(value, dict):
        return {
            str(key): _json_safe_neo4j_value(item, include_embeddings)
            for key, item in value.items()
            if include_embeddings or key != "embedding"
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_neo4j_value(item, include_embeddings) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    iso_format = getattr(value, "iso_format", None)
    if callable(iso_format):
        return iso_format()
    return str(value)


def _neo4j_log_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a payload containing only values valid as Neo4j properties."""

    def is_primitive(value: Any) -> bool:
        return value is None or isinstance(value, (str, int, float, bool))

    safe_payload: dict[str, Any] = {}
    for key, value in payload.items():
        if is_primitive(value):
            safe_payload[key] = value
            continue
        if isinstance(value, (list, tuple)) and all(
            is_primitive(item) and item is not None for item in value
        ):
            safe_payload[key] = list(value)
            continue

        # Neo4j properties cannot contain maps, nested arrays, or arbitrary
        # Python objects. Preserve those diagnostics as queryable JSON text.
        safe_payload[key] = json.dumps(
            _json_safe_neo4j_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return safe_payload


class Neo4jKnowledgeRepository:
    def __init__(self) -> None:
        settings = get_settings()
        self.database = settings.neo4j_database
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    def ensure_schema(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "CREATE CONSTRAINT ai_source_id IF NOT EXISTS "
                "FOR (s:KnowledgeSource) REQUIRE s.id IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT ai_chunk_id IF NOT EXISTS "
                "FOR (c:AIChunk) REQUIRE c.id IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT ai_page_id IF NOT EXISTS "
                "FOR (p:KnowledgePage) REQUIRE p.id IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT ai_config_key IF NOT EXISTS "
                "FOR (c:AIConfig) REQUIRE c.key IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT ai_query_log_id IF NOT EXISTS "
                "FOR (q:AIQueryLog) REQUIRE q.id IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT ai_usage_log_id IF NOT EXISTS "
                "FOR (u:AIUsageLog) REQUIRE u.id IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT pdf_entity_canonical_id IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.canonicalId IS UNIQUE"
            ).consume()
            session.run(
                "CREATE INDEX ai_query_log_created_at IF NOT EXISTS "
                "FOR (q:AIQueryLog) ON (q.createdAt)"
            ).consume()
            session.run(
                "CREATE INDEX ai_page_lookup IF NOT EXISTS "
                "FOR (p:KnowledgePage) ON (p.sourceId, p.pageNumber)"
            ).consume()
            session.run(
                "CREATE INDEX ai_chunk_source_lookup IF NOT EXISTS "
                "FOR (c:AIChunk) ON (c.sourceId)"
            ).consume()
            session.run(
                "CREATE INDEX pdf_entity_name IF NOT EXISTS "
                "FOR (e:Entity) ON (e.name)"
            ).consume()
            session.run(
                "CREATE FULLTEXT INDEX ai_chunk_fulltext IF NOT EXISTS "
                "FOR (c:AIChunk) ON EACH [c.text, c.heading]"
            ).consume()
            session.run(
                """
                CREATE VECTOR INDEX ai_chunk_vector IF NOT EXISTS
                FOR (c:AIChunk) ON (c.embedding)
                OPTIONS {indexConfig: {
                  `vector.dimensions`: 1536,
                  `vector.similarity_function`: 'cosine'
                }}
                """
            ).consume()

    def replace_source(
        self,
        metadata: KnowledgeIndexMetadata,
        pages: list[PdfPage],
        chunks: list[PdfChunk],
        embeddings: list[list[float]],
        graph_extractions: list[ChunkGraphExtraction] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        source = metadata.model_dump()
        page_rows = [
            {
                "id": page.page_id,
                "pageNumber": page.page_number,
                "charCount": page.char_count,
                "contentHash": page.content_hash,
                "preview": page.preview,
                "extractionStatus": page.extraction_status,
                "chunkCount": page.chunk_count,
                "footnoteRemoved": page.footnote_removed,
                "removedFootnoteChars": page.removed_footnote_chars,
                "runningHeaderRemoved": page.running_header_removed,
                "removedRunningHeaderChars": (
                    page.removed_running_header_chars
                ),
            }
            for page in pages
        ]
        chunk_rows = [
            {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "pageStart": chunk.page_start,
                "pageEnd": chunk.page_end,
                "contentHash": chunk.content_hash,
                "embedding": embedding,
                "sequence": chunk.sequence,
                "pageChunkIndex": chunk.page_chunk_index,
                "charCount": chunk.char_count,
                "heading": chunk.heading,
                "wordCount": chunk.word_count,
                "facets": list(chunk.facets),
                "yearStart": chunk.year_start,
                "yearEnd": chunk.year_end,
                "pageId": f"{metadata.source_id}-p{chunk.page_start}",
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        graph_rows = [
            extraction.as_row()
            for extraction in (graph_extractions or [])
        ]
        relationship_rows: dict[str, list[dict[str, Any]]] = {}
        for graph_row in graph_rows:
            for relation in graph_row["relationships"]:
                relationship_rows.setdefault(relation["type"], []).append({
                    **relation,
                    "chunkId": graph_row["chunkId"],
                    "pageNumber": graph_row["pageNumber"],
                })

        with self.driver.session(database=self.database) as session:
            session.execute_write(
                self._replace_source_tx,
                source,
                page_rows,
                chunk_rows,
                graph_rows,
                relationship_rows,
                usage or {},
            )

    def source_chunk_cache(self, source_id: str) -> dict[str, Any]:
        """Read reusable OpenAI artifacts from the latest PDF index."""
        result: dict[str, Any] = {
            "pipelineVersion": "",
            "embeddingModel": "",
            "entityModel": "",
            "chunks": {},
        }
        with self.driver.session(database=self.database) as session:
            source_record = session.run(
                """
                OPTIONAL MATCH (source:KnowledgeSource {id: $sourceId})
                RETURN source.pipelineVersion AS pipelineVersion,
                       source.embeddingModel AS embeddingModel,
                       source.entityModel AS entityModel
                """,
                sourceId=source_id,
            ).single()
            if source_record:
                result.update({
                    "pipelineVersion": str(
                        source_record["pipelineVersion"] or ""
                    ),
                    "embeddingModel": str(
                        source_record["embeddingModel"] or ""
                    ),
                    "entityModel": str(source_record["entityModel"] or ""),
                })

            for record in session.run(
                """
                MATCH (chunk:AIChunk {sourceId: $sourceId})
                WHERE chunk.contentHash IS NOT NULL
                RETURN chunk.contentHash AS contentHash,
                       chunk.embedding AS embedding,
                       coalesce(chunk.graphExtracted, false) AS graphExtracted
                """,
                sourceId=source_id,
            ):
                content_hash = str(record["contentHash"] or "")
                if not content_hash:
                    continue
                result["chunks"][content_hash] = {
                    "embedding": list(record["embedding"] or []),
                    "graphExtracted": bool(record["graphExtracted"]),
                    "entities": [],
                    "relationships": [],
                }

            for record in session.run(
                """
                MATCH (chunk:AIChunk {sourceId: $sourceId})
                      -[mention:MENTIONS]->(entity:Entity {pipeline: 'pdf'})
                WHERE chunk.contentHash IS NOT NULL
                RETURN chunk.contentHash AS contentHash,
                       entity.canonicalId AS canonicalId,
                       entity.name AS name,
                       entity.entityType AS entityType,
                       coalesce(entity.aliases, []) AS aliases,
                       coalesce(entity.searchKeys, []) AS searchKeys,
                       coalesce(mention.surfaceForms, []) AS surfaceForms
                """,
                sourceId=source_id,
            ):
                cached = result["chunks"].get(str(record["contentHash"] or ""))
                if cached is None:
                    continue
                cached["entities"].append({
                    "canonicalId": str(record["canonicalId"] or ""),
                    "name": str(record["name"] or ""),
                    "entityType": str(record["entityType"] or ""),
                    "aliases": list(record["aliases"] or []),
                    "searchKeys": list(record["searchKeys"] or []),
                    "surfaceForms": list(record["surfaceForms"] or []),
                })

            for record in session.run(
                """
                MATCH (chunk:AIChunk {sourceId: $sourceId})
                MATCH (sourceEntity:Entity)-[relation]->(targetEntity:Entity)
                WHERE chunk.contentHash IS NOT NULL
                  AND relation.pipeline = 'pdf'
                  AND relation.sourceId = $sourceId
                  AND relation.chunkId = chunk.id
                RETURN chunk.contentHash AS contentHash,
                       sourceEntity.canonicalId AS sourceCanonicalId,
                       targetEntity.canonicalId AS targetCanonicalId,
                       type(relation) AS type,
                       coalesce(relation.evidence, '') AS evidence,
                       coalesce(relation.rawPredicate, '') AS rawPredicate
                """,
                sourceId=source_id,
            ):
                cached = result["chunks"].get(str(record["contentHash"] or ""))
                if cached is None:
                    continue
                cached["relationships"].append({
                    "sourceCanonicalId": str(
                        record["sourceCanonicalId"] or ""
                    ),
                    "targetCanonicalId": str(
                        record["targetCanonicalId"] or ""
                    ),
                    "type": str(record["type"] or ""),
                    "evidence": str(record["evidence"] or ""),
                    "rawPredicate": str(record["rawPredicate"] or ""),
                })
        return result

    def ensure_source_structure(
        self,
        source_id: str,
        expected_page_count: int | None = None,
        expected_chunk_count: int | None = None,
    ) -> dict[str, Any] | None:
        """Repair source/page/chunk links and verify the persisted graph structure."""
        with self.driver.session(database=self.database) as session:
            source = session.run(
                "MATCH (source:KnowledgeSource {id: $sourceId}) RETURN source.id AS id",
                sourceId=source_id,
            ).single()
            if not source:
                return None

            counts = self._source_structure_counts(session, source_id)
            repaired_page_links = 0
            repaired_chunk_links = 0

            # Only repair when counts differ. This keeps normal detail reads cheap even
            # for books containing thousands of chunks.
            if counts["linkedPageCount"] != counts["pageNodeCount"]:
                repaired = session.run(
                    """
                    MATCH (source:KnowledgeSource {id: $sourceId})
                    MATCH (page:KnowledgePage {sourceId: $sourceId})
                    WHERE NOT (source)-[:HAS_PAGE]->(page)
                    MERGE (source)-[:HAS_PAGE]->(page)
                    RETURN count(page) AS count
                    """,
                    sourceId=source_id,
                ).single()
                repaired_page_links = int(repaired["count"] if repaired else 0)

            if counts["linkedChunkCount"] != counts["chunkNodeCount"]:
                repaired = session.run(
                    """
                    MATCH (chunk:AIChunk {sourceId: $sourceId})
                    WHERE NOT (:KnowledgePage)-[:HAS_CHUNK]->(chunk)
                    WITH chunk,
                         coalesce(
                           chunk.pageId,
                           $sourceId + '-p' + toString(chunk.pageStart)
                         ) AS expectedPageId
                    MATCH (page:KnowledgePage {id: expectedPageId})
                    MERGE (page)-[:HAS_CHUNK]->(chunk)
                    SET chunk.pageId = page.id
                    RETURN count(chunk) AS count
                    """,
                    sourceId=source_id,
                ).single()
                repaired_chunk_links = int(repaired["count"] if repaired else 0)

            if repaired_page_links or repaired_chunk_links:
                counts = self._source_structure_counts(session, source_id)

        result = {
            "sourceId": source_id,
            "pageCount": counts["linkedPageCount"],
            "chunkCount": counts["linkedChunkCount"],
            "pageNodeCount": counts["pageNodeCount"],
            "chunkNodeCount": counts["chunkNodeCount"],
            "repairedPageLinks": repaired_page_links,
            "repairedChunkLinks": repaired_chunk_links,
        }
        result["valid"] = (
            result["pageCount"] == result["pageNodeCount"]
            and result["chunkCount"] == result["chunkNodeCount"]
            and (
                expected_page_count is None
                or result["pageCount"] == expected_page_count
            )
            and (
                expected_chunk_count is None
                or result["chunkCount"] == expected_chunk_count
            )
        )
        return result

    @staticmethod
    def _source_structure_counts(session: Any, source_id: str) -> dict[str, int]:
        queries = {
            "linkedPageCount": (
                "MATCH (:KnowledgeSource {id: $sourceId})-[:HAS_PAGE]->"
                "(node:KnowledgePage) RETURN count(node) AS count"
            ),
            "linkedChunkCount": (
                "MATCH (:KnowledgeSource {id: $sourceId})-[:HAS_PAGE]->"
                "(:KnowledgePage)-[:HAS_CHUNK]->(node:AIChunk) "
                "RETURN count(node) AS count"
            ),
            "pageNodeCount": (
                "MATCH (node:KnowledgePage {sourceId: $sourceId}) "
                "RETURN count(node) AS count"
            ),
            "chunkNodeCount": (
                "MATCH (node:AIChunk {sourceId: $sourceId}) "
                "RETURN count(node) AS count"
            ),
        }
        result: dict[str, int] = {}
        for key, query in queries.items():
            record = session.run(query, sourceId=source_id).single()
            result[key] = int(record["count"] if record else 0)
        return result

    @staticmethod
    def _replace_source_tx(
        tx: Any,
        source: dict,
        pages: list[dict],
        chunks: list[dict],
        graph_rows: list[dict],
        relationship_rows: dict[str, list[dict]],
        usage: dict[str, Any],
    ) -> None:
        tx.run(
            """
            MATCH ()-[relation]->()
            WHERE relation.pipeline = 'pdf' AND relation.sourceId = $sourceId
            DELETE relation
            """,
            sourceId=source["source_id"],
        ).consume()
        tx.run(
            """
            MATCH (entity:Entity {pipeline: 'pdf'})
            WHERE $sourceId IN coalesce(entity.sourceIds, [])
            SET entity.sourceIds = [
              item IN entity.sourceIds
              WHERE item <> $sourceId
            ]
            """,
            sourceId=source["source_id"],
        ).consume()
        tx.run(
            """
            MERGE (source:KnowledgeSource {id: $source.source_id})
            SET source.title = $source.title,
                source.author = $source.author,
                source.publisher = $source.publisher,
                source.publicationYear = $source.publication_year,
                source.sourceType = $source.source_type,
                source.trustLevel = $source.trust_level,
                source.storagePath = $source.storage_path,
                source.relatedPeriods = $source.related_periods,
                source.relatedStages = $source.related_stages,
                source.relatedEvents = $source.related_events,
                source.relatedPersons = $source.related_persons,
                source.active = true,
                source.indexedAt = datetime(),
                source.entityCount = coalesce($usage.entityCount, 0),
                source.semanticRelationshipCount = coalesce($usage.relationshipCount, 0),
                source.lastIndexEmbeddingTokens = coalesce($usage.embeddingInputTokens, 0),
                source.lastIndexEntityInputTokens = coalesce($usage.entityInputTokens, 0),
                source.lastIndexEntityOutputTokens = coalesce($usage.entityOutputTokens, 0),
                source.lastIndexTotalTokens = coalesce($usage.totalTokens, 0),
                source.pipelineVersion = $usage.pipelineVersion,
                source.embeddingModel = $usage.embeddingModel,
                source.entityModel = $usage.entityModel,
                source.reusedChunkCount = coalesce($usage.reusedChunkCount, 0),
                source.openaiChunkCount = coalesce($usage.openaiChunkCount, 0),
                source.totalPageCount = coalesce($usage.pageCount, 0),
                source.indexedPageCount = coalesce($usage.indexedPageCount, 0),
                source.contentStartPage = coalesce($usage.contentStartPage, 1),
                source.contentEndPage = coalesce($usage.contentEndPage, 0),
                source.skippedPageCount = coalesce($usage.skippedPageCount, 0),
                source.footnotePageCount = coalesce(
                    $usage.footnotePageCount,
                    0
                  ),
                source.removedFootnoteChars = coalesce(
                    $usage.removedFootnoteChars,
                    0
                  ),
                source.runningHeaderPageCount = coalesce(
                    $usage.runningHeaderPageCount,
                    0
                  ),
                source.removedRunningHeaderChars = coalesce(
                    $usage.removedRunningHeaderChars,
                    0
                  ),
                source.boundaryDetection = coalesce(
                    $usage.boundaryDetection,
                    'full_document'
                  )
            WITH source
            OPTIONAL MATCH (source)-[:HAS_CHUNK]->(legacyChunk:AIChunk)
            WITH source, collect(legacyChunk) AS legacyChunks
            FOREACH (node IN legacyChunks | DETACH DELETE node)
            WITH source
            OPTIONAL MATCH (source)-[:HAS_PAGE]->(oldPage:KnowledgePage)
            OPTIONAL MATCH (oldPage)-[:HAS_CHUNK]->(oldChunk:AIChunk)
            WITH source, collect(DISTINCT oldChunk) AS oldChunks, collect(DISTINCT oldPage) AS oldPages
            FOREACH (node IN oldChunks | DETACH DELETE node)
            FOREACH (node IN oldPages | DETACH DELETE node)
            """,
            source=source,
            usage=usage,
        ).consume()
        # Remove orphaned nodes left by an interrupted older indexing job. The
        # sourceId is stable when admins rename metadata, so cleanup is deterministic.
        tx.run(
            "MATCH (node:AIChunk {sourceId: $sourceId}) DETACH DELETE node",
            sourceId=source["source_id"],
        ).consume()
        tx.run(
            "MATCH (node:KnowledgePage {sourceId: $sourceId}) DETACH DELETE node",
            sourceId=source["source_id"],
        ).consume()
        tx.run(
            """
            MATCH (source:KnowledgeSource {id: $sourceId})
            UNWIND $pages AS page
            CREATE (node:KnowledgePage {
              id: page.id,
              sourceId: $sourceId,
              pageNumber: page.pageNumber,
              charCount: page.charCount,
              contentHash: page.contentHash,
              preview: page.preview,
              extractionStatus: page.extractionStatus,
              chunkCount: page.chunkCount,
              footnoteRemoved: coalesce(page.footnoteRemoved, false),
              removedFootnoteChars: coalesce(page.removedFootnoteChars, 0),
              runningHeaderRemoved: coalesce(
                page.runningHeaderRemoved,
                false
              ),
              removedRunningHeaderChars: coalesce(
                page.removedRunningHeaderChars,
                0
              )
            })
            MERGE (source)-[:HAS_PAGE]->(node)
            """,
            sourceId=source["source_id"],
            pages=pages,
        ).consume()
        tx.run(
            """
            MATCH (source:KnowledgeSource {id: $sourceId})-[relation:RELATED_TO]->()
            DELETE relation
            """,
            sourceId=source["source_id"],
        ).consume()
        tx.run(
            """
            UNWIND $chunks AS chunk
            MATCH (page:KnowledgePage {id: chunk.pageId})
            CREATE (node:AIChunk {
              id: chunk.id,
              sourceId: $sourceId,
              sourceTitle: $sourceTitle,
              pageId: chunk.pageId,
              text: chunk.text,
              pageStart: chunk.pageStart,
              pageEnd: chunk.pageEnd,
              contentHash: chunk.contentHash,
              embedding: chunk.embedding,
              sequence: chunk.sequence,
              pageChunkIndex: chunk.pageChunkIndex,
              charCount: chunk.charCount,
              heading: chunk.heading,
              wordCount: chunk.wordCount,
              facets: chunk.facets,
              yearStart: chunk.yearStart,
              yearEnd: chunk.yearEnd,
              extractionStatus: page.extractionStatus,
              sourcePriority: CASE $trustLevel
                WHEN 'official' THEN 1.0
                WHEN 'academic' THEN 0.9
                ELSE 0.75
              END,
              graphExtracted: $graphExtracted,
              active: true
            })
            MERGE (page)-[:HAS_CHUNK]->(node)
            """,
            sourceId=source["source_id"],
            sourceTitle=source["title"],
            trustLevel=source["trust_level"],
            chunks=chunks,
            graphExtracted=bool(usage.get("entityExtractionEnabled")),
        ).consume()
        if graph_rows:
            tx.run(
                """
                UNWIND $rows AS row
                MATCH (chunk:AIChunk {id: row.chunkId})
                UNWIND row.entities AS item
                MERGE (entity:Entity {canonicalId: item.canonicalId})
                ON CREATE SET entity.createdAt = datetime()
                SET entity.name = item.name,
                    entity.entityType = item.entityType,
                    entity.pipeline = 'pdf',
                    entity.aliases = reduce(
                      values = [],
                      alias IN coalesce(entity.aliases, []) + item.aliases |
                      CASE
                        WHEN alias IN values THEN values
                        ELSE values + alias
                      END
                    ),
                    entity.searchKeys = reduce(
                      values = [],
                      searchKey IN coalesce(entity.searchKeys, []) + item.searchKeys |
                      CASE
                        WHEN searchKey IN values THEN values
                        ELSE values + searchKey
                      END
                    ),
                    entity.sourceIds = reduce(
                      values = [],
                      itemSourceId IN coalesce(entity.sourceIds, []) + [$sourceId] |
                      CASE
                        WHEN itemSourceId IN values THEN values
                        ELSE values + itemSourceId
                      END
                    ),
                    entity.extractorVersion = $pipelineVersion,
                    entity.updatedAt = datetime()
                MERGE (chunk)-[mention:MENTIONS]->(entity)
                SET mention.sourceId = $sourceId,
                    mention.pageNumber = row.pageNumber,
                    mention.pipeline = 'pdf',
                    mention.surfaceForms = item.surfaceForms,
                    mention.extractorVersion = $pipelineVersion
                """,
                rows=graph_rows,
                sourceId=source["source_id"],
                pipelineVersion=usage.get("pipelineVersion", ""),
            ).consume()
            for relationship_type, rows in relationship_rows.items():
                if relationship_type not in CANONICAL_RELATIONSHIP_TYPES:
                    raise ValueError(
                        f"Loại quan hệ không thuộc ontology: {relationship_type}"
                    )
                tx.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (sourceEntity:Entity {{canonicalId: row.sourceCanonicalId}})
                    MATCH (targetEntity:Entity {{canonicalId: row.targetCanonicalId}})
                    MERGE (sourceEntity)-[relation:`{relationship_type}` {{
                      sourceId: $sourceId,
                      chunkId: row.chunkId
                    }}]->(targetEntity)
                    SET relation.pageNumber = row.pageNumber,
                        relation.evidence = row.evidence,
                        relation.rawPredicate = row.rawPredicate,
                        relation.pipeline = 'pdf',
                        relation.extractorVersion = $pipelineVersion,
                        relation.updatedAt = datetime()
                    """,
                    rows=rows,
                    sourceId=source["source_id"],
                    pipelineVersion=usage.get("pipelineVersion", ""),
                ).consume()
        tx.run(
            """
            MATCH (entity:Entity {pipeline: 'pdf'})
            WHERE NOT (entity)<-[:MENTIONS]-(:AIChunk)
              AND NOT (entity)-[]-()
            DELETE entity
            """
        ).consume()
        if usage:
            tx.run(
                """
                CREATE (log:AIUsageLog {
                  id: randomUUID(),
                  sourceId: $sourceId,
                  operation: 'pdf_index',
                  embeddingModel: $usage.embeddingModel,
                  entityModel: $usage.entityModel,
                  embeddingInputTokens: coalesce($usage.embeddingInputTokens, 0),
                  entityInputTokens: coalesce($usage.entityInputTokens, 0),
                  entityOutputTokens: coalesce($usage.entityOutputTokens, 0),
                  totalTokens: coalesce($usage.totalTokens, 0),
                  entityCount: coalesce($usage.entityCount, 0),
                  relationshipCount: coalesce($usage.relationshipCount, 0),
                  reusedChunkCount: coalesce($usage.reusedChunkCount, 0),
                  openaiChunkCount: coalesce($usage.openaiChunkCount, 0),
                  pageCount: coalesce($usage.pageCount, 0),
                  indexedPageCount: coalesce($usage.indexedPageCount, 0),
                  contentStartPage: coalesce($usage.contentStartPage, 1),
                  contentEndPage: coalesce($usage.contentEndPage, 0),
                  skippedPageCount: coalesce($usage.skippedPageCount, 0),
                  footnotePageCount: coalesce(
                    $usage.footnotePageCount,
                    0
                  ),
                  removedFootnoteChars: coalesce(
                    $usage.removedFootnoteChars,
                    0
                  ),
                  runningHeaderPageCount: coalesce(
                    $usage.runningHeaderPageCount,
                    0
                  ),
                  removedRunningHeaderChars: coalesce(
                    $usage.removedRunningHeaderChars,
                    0
                  ),
                  boundaryDetection: coalesce(
                    $usage.boundaryDetection,
                    'full_document'
                  ),
                  pipelineVersion: $usage.pipelineVersion,
                  createdAt: datetime()
                })
                """,
                sourceId=source["source_id"],
                usage=usage,
            ).consume()
    @staticmethod
    def _fulltext_query(question: str) -> tuple[str, list[str]]:
        terms = []
        for word in re.findall(r"[0-9A-Za-zÀ-ỹĐđ_-]+", question.casefold()):
            normalized = word.strip("_-")
            search_key = normalize_search_key(normalized)
            if len(search_key) < 2 or search_key in VIETNAMESE_STOP_WORDS:
                continue
            if normalized not in terms:
                terms.append(normalized)
        selected = terms[:14]
        return " OR ".join(f'"{term}"' for term in selected), selected

    @staticmethod
    def _exact_phrase_query(phrase: str) -> str:
        """Build a safe Lucene phrase query for Neo4j full-text search."""
        escaped = re.sub(r'([+\-!(){}\[\]^"~*?:\\/])', r'\\\1', phrase.strip())
        return f'"{escaped}"'

    @staticmethod
    def _result_projection() -> str:
        return """
               node.id AS id,
               node.text AS text,
               node.heading AS heading,
               node.sequence AS sequence,
               node.pageStart AS pageStart,
               node.pageEnd AS pageEnd,
               node.facets AS facets,
               node.yearStart AS yearStart,
               node.yearEnd AS yearEnd,
               node.sourcePriority AS sourcePriority,
               node.extractionStatus AS extractionStatus,
               source.id AS sourceId,
               source.title AS sourceTitle,
               source.trustLevel AS trustLevel,
               score,
               collect(DISTINCT entity.name) AS relatedEntities,
               collect(DISTINCT entity.canonicalId) AS relatedEntityIds
        """

    def retrieve_hybrid(
        self,
        question: str,
        embedding: list[float],
        candidate_k: int,
        *,
        exact_phrases: list[str] | None = None,
        primary_entity: str = "",
    ) -> dict[str, Any]:
        fulltext_query, query_terms = self._fulltext_query(question)
        exact_phrases = [phrase.strip() for phrase in (exact_phrases or []) if phrase.strip()]
        with self.driver.session(database=self.database) as session:
            vector_records = session.run(
                f"""
                CALL db.index.vector.queryNodes('ai_chunk_vector', $topK, $embedding)
                YIELD node, score
                MATCH (source:KnowledgeSource)
                WHERE (source)-[:HAS_CHUNK]->(node)
                   OR EXISTS {{
                     MATCH (source)-[:HAS_PAGE]->(:KnowledgePage)-[:HAS_CHUNK]->(node)
                   }}
                WITH source, node, score
                WHERE coalesce(node.active, true) = true
                  AND coalesce(source.active, true) = true
                OPTIONAL MATCH (node)-[:MENTIONS]->(entity:Entity {{pipeline: 'pdf'}})
                RETURN {self._result_projection()}
                ORDER BY score DESC
                """,
                topK=candidate_k,
                embedding=embedding,
            )
            vector_items = [record.data() for record in vector_records]

            lexical_items: list[dict] = []
            if fulltext_query:
                lexical_records = session.run(
                    f"""
                    CALL db.index.fulltext.queryNodes('ai_chunk_fulltext', $searchQuery, {{limit: $topK}})
                    YIELD node, score
                    MATCH (source:KnowledgeSource)
                    WHERE (source)-[:HAS_CHUNK]->(node)
                       OR EXISTS {{
                         MATCH (source)-[:HAS_PAGE]->(:KnowledgePage)-[:HAS_CHUNK]->(node)
                       }}
                    WITH source, node, score
                    WHERE coalesce(node.active, true) = true
                      AND coalesce(source.active, true) = true
                    OPTIONAL MATCH (node)-[:MENTIONS]->(entity:Entity {{pipeline: 'pdf'}})
                    RETURN {self._result_projection()}
                    ORDER BY score DESC
                    """,
                    searchQuery=fulltext_query,
                    topK=candidate_k,
                )
                lexical_items = [record.data() for record in lexical_records]

            exact_items: list[dict] = []
            for phrase in exact_phrases:
                exact_records = session.run(
                    f"""
                    CALL db.index.fulltext.queryNodes(
                      'ai_chunk_fulltext', $searchQuery, {{limit: $scanLimit}}
                    )
                    YIELD node, score
                    MATCH (source:KnowledgeSource)
                    WHERE (source)-[:HAS_CHUNK]->(node)
                       OR EXISTS {{
                         MATCH (source)-[:HAS_PAGE]->(:KnowledgePage)-[:HAS_CHUNK]->(node)
                       }}
                    WITH source, node, score
                    WHERE coalesce(node.active, true) = true
                      AND coalesce(source.active, true) = true
                      AND toLower(coalesce(node.heading, '') + ' ' + coalesce(node.text, ''))
                          CONTAINS toLower($phrase)
                    OPTIONAL MATCH (node)-[:MENTIONS]->(entity:Entity {{pipeline: 'pdf'}})
                    RETURN {self._result_projection()}
                    ORDER BY score DESC
                    LIMIT $topK
                    """,
                    searchQuery=self._exact_phrase_query(phrase),
                    phrase=phrase,
                    scanLimit=max(candidate_k * 4, 40),
                    topK=candidate_k,
                )
                for record in exact_records:
                    item = record.data()
                    item["matchedPhrase"] = phrase
                    item["entityMatch"] = bool(
                        primary_entity
                        and normalize_search_key(phrase) in normalize_search_key(
                            f"{item.get('heading') or ''} {item.get('text') or ''}"
                        )
                    )
                    exact_items.append(item)

            entity_items: list[dict] = []
            entity_key = normalize_search_key(primary_entity)
            if entity_key:
                entity_records = session.run(
                    f"""
                    MATCH (matchedEntity:Entity {{pipeline: 'pdf'}})
                          <-[:MENTIONS]-(node:AIChunk)
                    WHERE $entityKey IN coalesce(matchedEntity.searchKeys, [])
                      AND coalesce(node.active, true) = true
                    MATCH (source:KnowledgeSource)
                    WHERE (
                      (source)-[:HAS_CHUNK]->(node)
                      OR EXISTS {{
                        MATCH (source)-[:HAS_PAGE]->(:KnowledgePage)
                              -[:HAS_CHUNK]->(node)
                      }}
                    )
                      AND coalesce(source.active, true) = true
                    WITH DISTINCT source, node, 1.0 AS score
                    OPTIONAL MATCH (node)-[:MENTIONS]->
                                   (entity:Entity {{pipeline: 'pdf'}})
                    RETURN {self._result_projection()}
                    ORDER BY node.sequence
                    LIMIT $topK
                    """,
                    entityKey=entity_key,
                    topK=candidate_k,
                )
                for record in entity_records:
                    item = record.data()
                    item["matchedPhrase"] = primary_entity
                    item["entityMatch"] = True
                    entity_items.append(item)

        fused: dict[str, dict] = {}
        channel_weights = {
            "exact": 4.0,
            "entity": 2.5,
            "vector": 1.0,
            "lexical": 1.0,
        }
        for channel, items in (
            ("exact", exact_items),
            ("entity", entity_items),
            ("vector", vector_items),
            ("lexical", lexical_items),
        ):
            for rank, item in enumerate(items, start=1):
                entry = fused.setdefault(item["id"], {
                    **item,
                    "rrfScore": 0.0,
                    "exactScore": 0.0,
                    "entityScore": 0.0,
                    "vectorScore": 0.0,
                    "lexicalScore": 0.0,
                    "channels": [],
                    "directEvidence": False,
                    "entityMatch": False,
                    "matchedPhrases": [],
                })
                entry["rrfScore"] += channel_weights[channel] / (60 + rank)
                entry[f"{channel}Score"] = float(item["score"])
                if channel not in entry["channels"]:
                    entry["channels"].append(channel)
                if channel == "exact":
                    entry["directEvidence"] = True
                    entry["entityMatch"] = entry["entityMatch"] or bool(item.get("entityMatch"))
                    phrase = item.get("matchedPhrase")
                    if phrase and phrase not in entry["matchedPhrases"]:
                        entry["matchedPhrases"].append(phrase)
                elif channel == "entity":
                    entry["entityMatch"] = True
                    phrase = item.get("matchedPhrase")
                    if phrase and phrase not in entry["matchedPhrases"]:
                        entry["matchedPhrases"].append(phrase)
        ordered = sorted(
            fused.values(),
            key=lambda item: (
                item["rrfScore"] + (0.002 if item.get("trustLevel") == "official" else 0),
                item["vectorScore"],
            ),
            reverse=True,
        )
        max_rrf = max((item["rrfScore"] for item in ordered), default=1.0)
        for item in ordered:
            item["score"] = round(item["rrfScore"] / max_rrf, 6)
        return {
            "items": ordered[:candidate_k],
            "candidateCount": len(ordered),
            "queryTerms": query_terms,
            "exactMatchCount": len({item["id"] for item in exact_items}),
            "channels": [
                channel
                for channel, items in (
                    ("exact", exact_items),
                    ("entity", entity_items),
                    ("vector", vector_items),
                    ("lexical", lexical_items),
                )
                if items
            ],
        }

    def retrieve_timeline_windows(
        self,
        question: str,
        windows: list[tuple[int, int]],
        *,
        per_window: int = 4,
    ) -> list[dict[str, Any]]:
        """Recall evidence evenly across an F9 range without defining periods.

        ``yearStart``/``yearEnd`` are only a coarse recall filter here. They
        never become historical boundaries by themselves; the F9 planner must
        still ground every accepted boundary in the chunk content.
        """
        if not windows:
            return []

        recalled: list[dict[str, Any]] = []
        scan_limit = max(120, min(600, per_window * 45))
        with self.driver.session(database=self.database) as session:
            for window_start, window_end in windows:
                midpoint = (window_start + window_end) // 2
                search_query, _ = self._fulltext_query(question)
                year_query = " OR ".join(
                    f'"{year}"'
                    for year in dict.fromkeys(
                        (window_start, midpoint, window_end)
                    )
                )
                search_query = " OR ".join(
                    part for part in (search_query, year_query) if part
                )
                if not search_query:
                    continue
                records = session.run(
                    f"""
                    CALL db.index.fulltext.queryNodes(
                      'ai_chunk_fulltext', $searchQuery, {{limit: $scanLimit}}
                    )
                    YIELD node, score
                    MATCH (source:KnowledgeSource)
                    WHERE (source)-[:HAS_CHUNK]->(node)
                       OR EXISTS {{
                         MATCH (source)-[:HAS_PAGE]->(:KnowledgePage)
                               -[:HAS_CHUNK]->(node)
                       }}
                    WITH DISTINCT source, node, score
                    WHERE coalesce(node.active, true) = true
                      AND coalesce(source.active, true) = true
                      AND coalesce(
                            node.yearStart, source.yearStart, $windowStart
                          ) <= $windowEnd
                      AND coalesce(
                            node.yearEnd, source.yearEnd, $windowEnd
                          ) >= $windowStart
                    OPTIONAL MATCH (node)-[:MENTIONS]->
                                   (entity:Entity {{pipeline: 'pdf'}})
                    RETURN {self._result_projection()}
                    ORDER BY score DESC
                    LIMIT $perWindow
                    """,
                    searchQuery=search_query,
                    scanLimit=scan_limit,
                    windowStart=window_start,
                    windowEnd=window_end,
                    perWindow=max(1, min(per_window, 8)),
                )
                for rank, record in enumerate(records, start=1):
                    item = record.data()
                    item["timelineWindowStart"] = window_start
                    item["timelineWindowEnd"] = window_end
                    item["timelineWindowRank"] = rank
                    item["channels"] = ["timeline_window"]
                    recalled.append(item)
        return recalled

    def source_detail(
        self,
        source_id: str,
        page_number: int | None = None,
        query: str = "",
        limit: int = 80,
    ) -> dict[str, Any] | None:
        integrity = self.ensure_source_structure(source_id)
        if integrity is None:
            return None
        if not integrity["valid"]:
            raise RuntimeError(
                "Cấu trúc nguồn tri thức chưa nhất quán sau khi tự phục hồi "
                f"(pages={integrity['pageCount']}/{integrity['pageNodeCount']}, "
                f"chunks={integrity['chunkCount']}/{integrity['chunkNodeCount']})."
            )

        search_aliases = list(expand_known_entity_aliases(query))
        if not search_aliases and query.strip():
            search_aliases = [query.strip()]

        with self.driver.session(database=self.database) as session:
            source_record = session.run(
                """
                MATCH (source:KnowledgeSource {id: $sourceId})
                OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:KnowledgePage)
                OPTIONAL MATCH (page)-[:HAS_CHUNK]->(chunk:AIChunk)
                RETURN source {.*, indexedAt: toString(source.indexedAt)} AS source,
                       count(DISTINCT page) AS pageCount,
                       count(DISTINCT chunk) AS chunkCount
                """,
                sourceId=source_id,
            ).single()
            if not source_record:
                return None
            pages = [record.data()["page"] for record in session.run(
                """
                MATCH (:KnowledgeSource {id: $sourceId})-[:HAS_PAGE]->(page:KnowledgePage)
                RETURN page {.*} AS page
                ORDER BY page.pageNumber
                """,
                sourceId=source_id,
            )]
            chunk_records = session.run(
                """
                MATCH (:KnowledgeSource {id: $sourceId})-[:HAS_PAGE]->(page:KnowledgePage)-[:HAS_CHUNK]->(chunk:AIChunk)
                WHERE ($pageNumber IS NULL OR page.pageNumber = $pageNumber)
                  AND (
                    $searchQuery = ''
                    OR any(
                      alias IN $searchAliases
                      WHERE toLower(chunk.text) CONTAINS toLower(alias)
                    )
                  )
                OPTIONAL MATCH (chunk)-[:MENTIONS]->(entity:Entity {pipeline: 'pdf'})
                RETURN chunk.id AS id, chunk.heading AS heading, chunk.sequence AS sequence,
                       chunk.pageChunkIndex AS pageChunkIndex, chunk.pageStart AS pageStart,
                       chunk.pageEnd AS pageEnd, chunk.charCount AS charCount,
                       chunk.wordCount AS wordCount, chunk.facets AS facets,
                       chunk.yearStart AS yearStart, chunk.yearEnd AS yearEnd,
                       chunk.extractionStatus AS extractionStatus,
                       chunk.active AS active, chunk.text AS text,
                       collect(DISTINCT entity.name) AS entities,
                       collect(DISTINCT entity {
                         .canonicalId,
                         .name,
                         .entityType,
                         .aliases
                       }) AS entityDetails
                ORDER BY chunk.sequence
                LIMIT $limit
                """,
                sourceId=source_id,
                pageNumber=page_number,
                searchQuery=query.strip(),
                searchAliases=search_aliases,
                limit=max(1, min(limit, 200)),
            )
            detail = {
                "source": source_record["source"],
                "pageCount": int(source_record["pageCount"]),
                "chunkCount": int(source_record["chunkCount"]),
                "pages": pages,
                "chunks": [record.data() for record in chunk_records],
                "integrity": integrity,
            }
            return _json_safe_neo4j_value(detail)

    def graph_summary(self) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            labels = [record.data() for record in session.run(
                """
                MATCH (node)
                WHERE any(label IN labels(node) WHERE label IN $labels)
                  AND (NOT node:Entity OR node.pipeline = 'pdf')
                UNWIND [label IN labels(node) WHERE label IN $labels] AS label
                RETURN label, count(*) AS count
                ORDER BY count DESC
                """,
                labels=sorted(PDF_GRAPH_LABELS),
            )]
            relationships = [record.data() for record in session.run(
                """
                MATCH (source)-[relation]->(target)
                WHERE any(label IN labels(source) WHERE label IN $labels)
                  AND any(label IN labels(target) WHERE label IN $labels)
                  AND (NOT source:Entity OR source.pipeline = 'pdf')
                  AND (NOT target:Entity OR target.pipeline = 'pdf')
                RETURN type(relation) AS type, count(*) AS count
                ORDER BY count DESC
                """,
                labels=sorted(PDF_GRAPH_LABELS),
            )]
            usage_record = session.run(
                """
                MATCH (usage:AIUsageLog)
                RETURN count(usage) AS indexRuns,
                       sum(coalesce(usage.embeddingInputTokens, 0)) AS embeddingInputTokens,
                       sum(coalesce(usage.entityInputTokens, 0)) AS entityInputTokens,
                       sum(coalesce(usage.entityOutputTokens, 0)) AS entityOutputTokens,
                       sum(coalesce(usage.totalTokens, 0)) AS totalTokens
                """
            ).single()
            relationship_type_count = len(relationships)
            canonical_type_count = sum(
                1 for item in relationships if is_canonical_relationship_type(item["type"])
            )
            return {
                "labels": labels,
                "relationships": relationships,
                "nodeCount": sum(int(item["count"]) for item in labels),
                "edgeCount": sum(int(item["count"]) for item in relationships),
                "relationshipTypeCount": relationship_type_count,
                "canonicalTypeCount": canonical_type_count,
                "legacyTypeCount": relationship_type_count - canonical_type_count,
                "canonicalTypeLimit": len(CANONICAL_RELATIONSHIP_TYPES),
                "usage": {
                    key: int(usage_record[key] or 0) if usage_record else 0
                    for key in (
                        "indexRuns",
                        "embeddingInputTokens",
                        "entityInputTokens",
                        "entityOutputTokens",
                        "totalTokens",
                    )
                },
            }

    def relationship_ontology_preview(self) -> dict[str, Any]:
        summary = self.graph_summary()
        mappings = []
        affected_edges = 0
        for item in summary["relationships"]:
            source_type = str(item["type"])
            canonical_type = canonicalize_relationship_type(source_type)
            requires_migration = source_type != canonical_type
            count = int(item["count"])
            if requires_migration:
                affected_edges += count
            mappings.append({
                "sourceType": source_type,
                "canonicalType": canonical_type,
                "count": count,
                "requiresMigration": requires_migration,
            })
        return {
            **summary,
            "affectedEdgeCount": affected_edges,
            "mappings": mappings,
        }

    def migrate_relationship_ontology(self) -> dict[str, Any]:
        preview = self.relationship_ontology_preview()
        migrated_edges = 0
        migrated_types = 0
        with self.driver.session(database=self.database) as session:
            for mapping in preview["mappings"]:
                if not mapping["requiresMigration"]:
                    continue
                canonical_type = mapping["canonicalType"]
                if canonical_type not in CANONICAL_RELATIONSHIP_TYPES:
                    raise ValueError(f"Loại quan hệ đích không hợp lệ: {canonical_type}")
                result = session.run(
                    f"""
                    MATCH (source)-[old]->(target)
                    WHERE type(old) = $source_type
                    CREATE (source)-[new:`{canonical_type}`]->(target)
                    SET new = properties(old),
                        new.rawType = coalesce(old.rawType, $source_type)
                    DELETE old
                    RETURN count(*) AS migrated
                    """,
                    source_type=mapping["sourceType"],
                ).single()
                migrated_edges += int(result["migrated"] if result else 0)
                migrated_types += 1
        return {
            "migratedEdgeCount": migrated_edges,
            "migratedTypeCount": migrated_types,
            "summary": self.graph_summary(),
        }

    def graph_search(self, query: str, label: str = "", limit: int = 40) -> list[dict]:
        selected_label = label.strip()
        if selected_label and selected_label not in PDF_GRAPH_LABELS:
            raise ValueError("Loại node không thuộc pipeline PDF mới.")
        with self.driver.session(database=self.database) as session:
            records = session.run(
                """
                MATCH (node)
                WHERE ($label = '' OR $label IN labels(node))
                  AND any(item IN labels(node) WHERE item IN $allowedLabels)
                  AND (NOT node:Entity OR node.pipeline = 'pdf')
                  AND ($searchQuery = '' OR toLower(
                    coalesce(node.title, node.name, node.sourceTitle,
                             node.canonicalId, node.id, '')
                  ) CONTAINS toLower($searchQuery)
                  OR any(
                    alias IN coalesce(node.aliases, [])
                    WHERE toLower(alias) CONTAINS toLower($searchQuery)
                  )
                  OR any(
                    searchKey IN coalesce(node.searchKeys, [])
                    WHERE searchKey CONTAINS $normalizedSearchQuery
                  ))
                OPTIONAL MATCH (node)-[relation]-(related)
                WHERE any(item IN labels(related) WHERE item IN $allowedLabels)
                  AND (NOT related:Entity OR related.pipeline = 'pdf')
                RETURN coalesce(node.id, node.canonicalId, elementId(node)) AS id,
                       labels(node) AS labels,
                       coalesce(node.title, node.name, node.sourceTitle, node.id) AS title,
                       properties(node) AS properties,
                       [item IN collect(DISTINCT CASE
                          WHEN relation IS NULL THEN NULL
                          ELSE {
                            type: type(relation),
                            direction: CASE WHEN startNode(relation) = node THEN 'out' ELSE 'in' END,
                            relatedId: coalesce(related.id, related.canonicalId),
                            relatedTitle: coalesce(
                              related.title, related.name, related.sourceTitle, related.id
                            )
                          }
                        END) WHERE item IS NOT NULL][0..12] AS relationships
                ORDER BY title
                LIMIT $limit
                """,
                searchQuery=query.strip(),
                normalizedSearchQuery=normalize_search_key(query),
                label=selected_label,
                allowedLabels=sorted(PDF_GRAPH_LABELS),
                limit=max(1, min(limit, 100)),
            )
            return [_json_safe_neo4j_value(record.data()) for record in records]

    def legacy_graph_preview(self) -> dict[str, int]:
        with self.driver.session(database=self.database) as session:
            record = session.run(
                """
                OPTIONAL MATCH (legacyNode)
                WHERE any(
                  label IN labels(legacyNode)
                  WHERE label IN ['Chunk', 'Period', 'Stage', 'Event', 'Person']
                )
                   OR (
                     legacyNode:Entity
                     AND coalesce(legacyNode.pipeline, '') <> 'pdf'
                   )
                WITH count(DISTINCT legacyNode) AS nodeCount,
                     count(DISTINCT CASE WHEN legacyNode:Chunk THEN legacyNode END) AS chunkCount,
                     count(DISTINCT CASE WHEN legacyNode:Entity THEN legacyNode END) AS entityCount
                OPTIONAL MATCH (source)-[relation]->(target)
                WHERE NOT (
                  (source:KnowledgeSource AND target:KnowledgePage
                   AND type(relation) = 'HAS_PAGE')
                  OR (source:KnowledgePage AND target:AIChunk
                      AND type(relation) = 'HAS_CHUNK')
                  OR coalesce(relation.pipeline, '') = 'pdf'
                )
                RETURN nodeCount, chunkCount, entityCount,
                       count(DISTINCT relation) AS relationshipCount
                """
            ).single()
            return {
                "nodeCount": int(record["nodeCount"] if record else 0),
                "chunkCount": int(record["chunkCount"] if record else 0),
                "entityCount": int(record["entityCount"] if record else 0),
                "relationshipCount": int(
                    record["relationshipCount"] if record else 0
                ),
            }

    def cleanup_legacy_graph(self) -> dict[str, Any]:
        preview = self.legacy_graph_preview()
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (source)-[relation]->(target)
                WHERE NOT (
                  (source:KnowledgeSource AND target:KnowledgePage
                   AND type(relation) = 'HAS_PAGE')
                  OR (source:KnowledgePage AND target:AIChunk
                      AND type(relation) = 'HAS_CHUNK')
                  OR coalesce(relation.pipeline, '') = 'pdf'
                )
                DELETE relation
                """
            ).consume()
            session.run(
                """
                MATCH (node)
                WHERE any(
                  label IN labels(node)
                  WHERE label IN ['Chunk', 'Period', 'Stage', 'Event', 'Person']
                )
                   OR (node:Entity AND coalesce(node.pipeline, '') <> 'pdf')
                DETACH DELETE node
                """
            ).consume()
        return {
            "deleted": preview,
            "remaining": self.legacy_graph_preview(),
            "summary": self.graph_summary(),
        }

    def export_graph_component(
        self,
        component_type: str,
        name: str,
        include_embeddings: bool = False,
    ) -> dict[str, Any]:
        """Export one node label or relationship type with its graph context."""
        if component_type not in {"node", "relationship"}:
            raise ValueError("component_type phải là node hoặc relationship.")

        component_name = name.strip()
        if not component_name or len(component_name) > 160:
            raise ValueError("Tên thành phần graph không hợp lệ.")
        if component_type == "node" and component_name not in PDF_GRAPH_LABELS:
            raise ValueError("Loại node không thuộc pipeline PDF mới.")
        if component_type == "relationship":
            visible_types = {
                str(item["type"])
                for item in self.graph_summary()["relationships"]
            }
            if component_name not in visible_types:
                raise ValueError("Quan hệ không thuộc pipeline PDF mới.")

        with self.driver.session(database=self.database) as session:
            if component_type == "node":
                exists = session.run(
                    """
                    MATCH (node)
                    WHERE $name IN labels(node)
                      AND (NOT node:Entity OR node.pipeline = 'pdf')
                    RETURN count(node) AS count
                    """,
                    name=component_name,
                ).single()
                component_count = int(exists["count"] if exists else 0)
                if component_count == 0:
                    raise ValueError(f"Không tìm thấy loại node {component_name}.")

                node_records = session.run(
                    """
                    MATCH (node)
                    WHERE $name IN labels(node)
                      AND (NOT node:Entity OR node.pipeline = 'pdf')
                    RETURN elementId(node) AS elementId,
                           labels(node) AS labels,
                           properties(node) AS properties
                    ORDER BY coalesce(node.title, node.name, node.sourceTitle,
                                      node.id, node.canonicalId, elementId(node))
                    """,
                    name=component_name,
                )
                selected_nodes = {
                    record["elementId"]: {
                        "elementId": record["elementId"],
                        "labels": list(record["labels"]),
                        "properties": _json_safe_neo4j_value(
                            record["properties"], include_embeddings
                        ),
                        "componentNode": True,
                    }
                    for record in node_records
                }
                relation_records = session.run(
                    """
                    MATCH (node)
                    WHERE $name IN labels(node)
                      AND (NOT node:Entity OR node.pipeline = 'pdf')
                    MATCH (node)-[relation]-(related)
                    WHERE any(label IN labels(related) WHERE label IN $allowedLabels)
                      AND (NOT related:Entity OR related.pipeline = 'pdf')
                    WITH DISTINCT relation
                    WITH relation, startNode(relation) AS source, endNode(relation) AS target
                    RETURN elementId(relation) AS elementId,
                           type(relation) AS type,
                           properties(relation) AS properties,
                           elementId(source) AS sourceElementId,
                           labels(source) AS sourceLabels,
                           properties(source) AS sourceProperties,
                           elementId(target) AS targetElementId,
                           labels(target) AS targetLabels,
                           properties(target) AS targetProperties
                    ORDER BY type, sourceElementId, targetElementId
                    """,
                    name=component_name,
                    allowedLabels=sorted(PDF_GRAPH_LABELS),
                )
            else:
                exists = session.run(
                    """
                    MATCH (source)-[relation]->(target)
                    WHERE type(relation) = $name
                      AND any(label IN labels(source) WHERE label IN $allowedLabels)
                      AND any(label IN labels(target) WHERE label IN $allowedLabels)
                      AND (NOT source:Entity OR source.pipeline = 'pdf')
                      AND (NOT target:Entity OR target.pipeline = 'pdf')
                    RETURN count(relation) AS count
                    """,
                    name=component_name,
                    allowedLabels=sorted(PDF_GRAPH_LABELS),
                ).single()
                component_count = int(exists["count"] if exists else 0)
                if component_count == 0:
                    raise ValueError(f"Không tìm thấy loại quan hệ {component_name}.")

                selected_nodes = {}
                relation_records = session.run(
                    """
                    MATCH (source)-[relation]->(target)
                    WHERE type(relation) = $name
                      AND any(label IN labels(source) WHERE label IN $allowedLabels)
                      AND any(label IN labels(target) WHERE label IN $allowedLabels)
                      AND (NOT source:Entity OR source.pipeline = 'pdf')
                      AND (NOT target:Entity OR target.pipeline = 'pdf')
                    RETURN elementId(relation) AS elementId,
                           type(relation) AS type,
                           properties(relation) AS properties,
                           elementId(source) AS sourceElementId,
                           labels(source) AS sourceLabels,
                           properties(source) AS sourceProperties,
                           elementId(target) AS targetElementId,
                           labels(target) AS targetLabels,
                           properties(target) AS targetProperties
                    ORDER BY sourceElementId, targetElementId
                    """,
                    name=component_name,
                    allowedLabels=sorted(PDF_GRAPH_LABELS),
                )

            nodes = dict(selected_nodes)
            relationships: list[dict[str, Any]] = []
            for record in relation_records:
                source_id = record["sourceElementId"]
                target_id = record["targetElementId"]
                for node_id, labels, properties in (
                    (source_id, record["sourceLabels"], record["sourceProperties"]),
                    (target_id, record["targetLabels"], record["targetProperties"]),
                ):
                    if node_id not in nodes:
                        nodes[node_id] = {
                            "elementId": node_id,
                            "labels": list(labels),
                            "properties": _json_safe_neo4j_value(
                                properties, include_embeddings
                            ),
                            "componentNode": False,
                        }
                relationships.append({
                    "elementId": record["elementId"],
                    "type": record["type"],
                    "sourceElementId": source_id,
                    "targetElementId": target_id,
                    "properties": _json_safe_neo4j_value(
                        record["properties"], include_embeddings
                    ),
                })

        label_counts = Counter(
            label for node in nodes.values() for label in node["labels"]
        )
        relationship_counts = Counter(item["type"] for item in relationships)
        return {
            "schemaVersion": "1.0",
            "exportedAt": datetime.now(UTC).isoformat(),
            "database": self.database,
            "component": {
                "type": component_type,
                "name": component_name,
                "count": component_count,
            },
            "options": {"embeddingsIncluded": include_embeddings},
            "statistics": {
                "nodeCount": len(nodes),
                "componentNodeCount": sum(
                    1 for node in nodes.values() if node["componentNode"]
                ),
                "relationshipCount": len(relationships),
                "labels": dict(label_counts.most_common()),
                "relationshipTypes": dict(relationship_counts.most_common()),
            },
            "nodes": list(nodes.values()),
            "relationships": relationships,
        }

    def get_active_prompt(self) -> dict[str, str] | None:
        with self.driver.session(database=self.database) as session:
            record = session.run(
                "MATCH (config:AIConfig {key: 'active_prompt'}) RETURN config {.*} AS config"
            ).single()
            return dict(record["config"]) if record else None

    def set_active_prompt(
        self,
        version_id: str,
        name: str,
        system_prompt: str,
        query_normalization_instruction: str,
        answer_planning_instruction: str,
        presentation_instruction: str,
        output_contract: str,
    ) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (config:AIConfig {key: 'active_prompt'})
                SET config.versionId = $versionId,
                    config.name = $name,
                    config.systemPrompt = $systemPrompt,
                    config.queryNormalizationInstruction = $queryNormalizationInstruction,
                    config.answerPlanningInstruction = $answerPlanningInstruction,
                    config.presentationInstruction = $presentationInstruction,
                    config.outputContract = $outputContract,
                    config.updatedAt = datetime()
                """,
                versionId=version_id,
                name=name,
                systemPrompt=system_prompt,
                queryNormalizationInstruction=query_normalization_instruction,
                answerPlanningInstruction=answer_planning_instruction,
                presentationInstruction=presentation_instruction,
                outputContract=output_contract,
            ).consume()

    def log_query(self, payload: dict[str, Any]) -> None:
        safe_payload = _neo4j_log_payload(payload)
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                CREATE (log:AIQueryLog {id: randomUUID()})
                SET log += $payload, log.createdAt = datetime()
                """,
                payload=safe_payload,
            ).consume()

    def query_insights(self, limit: int = 80) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            count_record = session.run(
                "MATCH (log:AIQueryLog) RETURN count(log) AS total"
            ).single()
            total = int(count_record["total"] if count_record else 0)
            if total == 0:
                return {
                    "summary": {
                        "total": 0,
                        "unanswered": 0,
                        "averageScore": 0,
                        "averageLatencyMs": 0,
                    },
                    "items": [],
                }

            summary = session.run(
                """
                MATCH (log:AIQueryLog)
                RETURN count(CASE WHEN coalesce(log['answerable'], false) = false THEN 1 END) AS unanswered,
                       avg(coalesce(log['retrievalScore'], 0.0)) AS averageScore,
                       avg(coalesce(log['latencyMs'], 0)) AS averageLatencyMs
                """
            ).single()
            records = session.run(
                """
                MATCH (log:AIQueryLog)
                RETURN log['question'] AS question, log['answerable'] AS answerable,
                       log['intent'] AS intent, log['scope'] AS scope,
                       log['retrievalScore'] AS retrievalScore,
                       log['candidateCount'] AS candidateCount,
                       log['selectedCount'] AS selectedCount,
                       log['comparisonIntent'] AS comparisonIntent,
                       log['comparisonDomain'] AS comparisonDomain,
                       log['explicitFacets'] AS explicitFacets,
                       log['balancedFacets'] AS balancedFacets,
                       log['missingComparisonCells'] AS missingComparisonCells,
                       log['answerStructure'] AS answerStructure,
                       log['latencyMs'] AS latencyMs,
                       toString(log['createdAt']) AS createdAt
                ORDER BY log['createdAt'] DESC
                LIMIT $limit
                """,
                limit=max(1, min(limit, 200)),
            )
            return {
                "summary": {
                    "total": total,
                    "unanswered": int(summary["unanswered"] if summary else 0),
                    "averageScore": float(summary["averageScore"] or 0) if summary else 0,
                    "averageLatencyMs": float(summary["averageLatencyMs"] or 0) if summary else 0,
                },
                "items": [record.data() for record in records],
            }

    def delete_source(self, source_id: str) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH ()-[relation]->()
                WHERE relation.pipeline = 'pdf'
                  AND relation.sourceId = $sourceId
                DELETE relation
                """,
                sourceId=source_id,
            ).consume()
            session.run(
                """
                MATCH (entity:Entity {pipeline: 'pdf'})
                WHERE $sourceId IN coalesce(entity.sourceIds, [])
                SET entity.sourceIds = [
                  item IN entity.sourceIds
                  WHERE item <> $sourceId
                ]
                """,
                sourceId=source_id,
            ).consume()
            session.run(
                """
                MATCH (source:KnowledgeSource {id: $sourceId})
                OPTIONAL MATCH (source)-[:HAS_CHUNK]->(chunk:AIChunk)
                OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:KnowledgePage)
                OPTIONAL MATCH (page)-[:HAS_CHUNK]->(pageChunk:AIChunk)
                WITH source, collect(DISTINCT chunk) + collect(DISTINCT pageChunk) AS chunks,
                     collect(DISTINCT page) AS pages
                FOREACH (node IN chunks | DETACH DELETE node)
                FOREACH (node IN pages | DETACH DELETE node)
                DETACH DELETE source
                """,
                sourceId=source_id,
            ).consume()
            session.run(
                """
                MATCH (entity:Entity {pipeline: 'pdf'})
                WHERE NOT (entity)<-[:MENTIONS]-(:AIChunk)
                  AND NOT (entity)-[]-()
                DELETE entity
                """
            ).consume()

    def update_source_metadata(self, source_id: str, metadata: dict) -> bool:
        with self.driver.session(database=self.database) as session:
            record = session.run(
                """
                MATCH (source:KnowledgeSource {id: $sourceId})
                SET source.title = $metadata.title,
                    source.author = $metadata.author,
                    source.publisher = $metadata.publisher,
                    source.publicationYear = $metadata.publication_year,
                    source.sourceType = $metadata.source_type,
                    source.trustLevel = $metadata.trust_level,
                    source.relatedPeriods = $metadata.related_periods,
                    source.relatedStages = $metadata.related_stages,
                    source.relatedEvents = $metadata.related_events,
                    source.relatedPersons = $metadata.related_persons,
                    source.updatedAt = datetime()
                WITH source
                OPTIONAL MATCH (source)-[:HAS_PAGE]->(:KnowledgePage)-[:HAS_CHUNK]->(chunk:AIChunk)
                WITH source, collect(DISTINCT chunk) AS chunks
                FOREACH (node IN chunks | SET node.sourceTitle = $metadata.title)
                RETURN source.id AS id
                """,
                sourceId=source_id,
                metadata=metadata,
            ).single()
            if not record:
                return False

            return True

    def stats(self) -> dict[str, int]:
        with self.driver.session(database=self.database) as session:
            record = session.run(
                """
                OPTIONAL MATCH (source:KnowledgeSource)
                WITH count(source) AS sources
                OPTIONAL MATCH (chunk:AIChunk)
                WITH sources, count(chunk) AS chunks
                OPTIONAL MATCH (page:KnowledgePage)
                RETURN sources, chunks, count(page) AS pages
                """
            ).single()
            return {
                "sources": int(record["sources"] if record else 0),
                "chunks": int(record["chunks"] if record else 0),
                "pages": int(record["pages"] if record else 0),
            }

    def backfill_chunk_facets(
        self,
        batch_size: int = 500,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Recompute AIChunk facets from stored text without OpenAI or embeddings."""
        batch_size = max(10, min(batch_size, 2000))
        last_id = ""
        scanned = 0
        changed = 0
        facet_counts: Counter[str] = Counter()

        with self.driver.session(database=self.database) as session:
            while True:
                records = list(session.run(
                    """
                    MATCH (chunk:AIChunk)
                    WHERE chunk.id > $lastId
                    RETURN chunk.id AS id, chunk.text AS text,
                           coalesce(chunk.facets, []) AS facets
                    ORDER BY chunk.id
                    LIMIT $batchSize
                    """,
                    lastId=last_id,
                    batchSize=batch_size,
                ))
                if not records:
                    break

                updates: list[dict[str, Any]] = []
                for record in records:
                    chunk_id = str(record["id"] or "")
                    facets = list(infer_history_facets(str(record["text"] or "")))
                    scanned += 1
                    facet_counts.update(facets)
                    if facets != list(record["facets"] or []):
                        changed += 1
                        updates.append({"id": chunk_id, "facets": facets})
                    last_id = chunk_id

                if updates and not dry_run:
                    session.run(
                        """
                        UNWIND $rows AS row
                        MATCH (chunk:AIChunk {id: row.id})
                        SET chunk.facets = row.facets,
                            chunk.facetVersion = 'history-facets-v3'
                        """,
                        rows=updates,
                    ).consume()

        return {
            "dryRun": dry_run,
            "scannedChunks": scanned,
            "changedChunks": changed,
            "facetCounts": dict(sorted(facet_counts.items())),
            "openAiCalls": 0,
            "embeddingCalls": 0,
        }
