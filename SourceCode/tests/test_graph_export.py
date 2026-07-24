import unittest
from datetime import datetime

from neo4j.time import DateTime

from src.api.neo4j_repository import _json_safe_neo4j_value, _neo4j_log_payload


class GraphExportSerializationTest(unittest.TestCase):
    def test_log_payload_serializes_nested_diagnostics(self) -> None:
        result = _neo4j_log_payload({
            "question": "Điện Biên Phủ diễn ra như thế nào?",
            "selectedChunkIds": ["chunk-1", "chunk-2"],
            "confidenceFactors": {
                "direct_evidence": 1.0,
                "entity_match": 1.0,
            },
        })

        self.assertEqual(
            result["selectedChunkIds"],
            ["chunk-1", "chunk-2"],
        )
        self.assertEqual(
            result["confidenceFactors"],
            '{"direct_evidence":1.0,"entity_match":1.0}',
        )

    def test_log_payload_serializes_arrays_containing_null(self) -> None:
        result = _neo4j_log_payload({
            "evolutionBoundaryYears": [None, 1911, 1930, 1939],
        })

        self.assertEqual(
            result["evolutionBoundaryYears"],
            "[null,1911,1930,1939]",
        )

    def test_embedding_is_omitted_by_default(self) -> None:
        value = {
            "id": "chunk-1",
            "embedding": [0.1, 0.2, 0.3],
            "nested": {"embedding": [0.4], "title": "Dien Bien Phu"},
        }

        result = _json_safe_neo4j_value(value)

        self.assertNotIn("embedding", result)
        self.assertNotIn("embedding", result["nested"])
        self.assertEqual(result["nested"]["title"], "Dien Bien Phu")

    def test_embedding_can_be_included_explicitly(self) -> None:
        value = {"id": "chunk-1", "embedding": [0.1, 0.2, 0.3]}

        result = _json_safe_neo4j_value(value, include_embeddings=True)

        self.assertEqual(result["embedding"], [0.1, 0.2, 0.3])

    def test_non_json_neo4j_value_uses_iso_format(self) -> None:
        class Neo4jDateLike:
            @staticmethod
            def iso_format() -> str:
                return "1954-05-07"

        result = _json_safe_neo4j_value({"date": Neo4jDateLike()})

        self.assertEqual(result, {"date": "1954-05-07"})

    def test_real_neo4j_datetime_is_json_safe(self) -> None:
        value = {
            "source": {
                "title": "1945-1965",
                "indexedAt": DateTime.from_native(datetime(2026, 7, 16, 13, 5, 57)),
            },
            "pages": [{"pageNumber": 1}],
        }

        result = _json_safe_neo4j_value(value)

        self.assertEqual(result["source"]["title"], "1945-1965")
        self.assertIsInstance(result["source"]["indexedAt"], str)
        self.assertTrue(result["source"]["indexedAt"].startswith("2026-07-16T13:05:57"))


if __name__ == "__main__":
    unittest.main()
