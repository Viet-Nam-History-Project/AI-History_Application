import unittest

from src.api.neo4j_repository import Neo4jKnowledgeRepository


class FakeRecord:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def data(self) -> dict:
        return dict(self.payload)


class FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def run(self, query: str, **_params):
        if "matchedEntity:Entity" not in query:
            return []
        return [FakeRecord({
            "id": "chunk-ho-chi-minh",
            "text": "Nguyễn Ái Quốc thành lập Hội Việt Nam Cách mạng Thanh niên.",
            "heading": "",
            "sequence": 1,
            "pageStart": 10,
            "pageEnd": 10,
            "facets": ["leadership"],
            "yearStart": 1925,
            "yearEnd": 1925,
            "sourcePriority": 1.0,
            "extractionStatus": "ready",
            "sourceId": "source",
            "sourceTitle": "Lịch sử Việt Nam",
            "trustLevel": "official",
            "score": 1.0,
            "relatedEntities": ["person:ho-chi-minh"],
        })]


class FakeDriver:
    @staticmethod
    def session(**_kwargs):
        return FakeSession()


class EntityAwareRetrievalTest(unittest.TestCase):
    def test_alias_entity_channel_returns_mentioning_chunk(self) -> None:
        repository = Neo4jKnowledgeRepository.__new__(
            Neo4jKnowledgeRepository
        )
        repository.driver = FakeDriver()
        repository.database = "neo4j"

        result = repository.retrieve_hybrid(
            "Bác Hồ đã làm gì năm 1925?",
            [0.0] * 1536,
            12,
            primary_entity="Bác Hồ",
        )

        self.assertEqual(result["items"][0]["id"], "chunk-ho-chi-minh")
        self.assertEqual(result["items"][0]["channels"], ["entity"])
        self.assertTrue(result["items"][0]["entityMatch"])
        self.assertFalse(result["items"][0]["directEvidence"])
        self.assertIn("entity", result["channels"])


if __name__ == "__main__":
    unittest.main()
