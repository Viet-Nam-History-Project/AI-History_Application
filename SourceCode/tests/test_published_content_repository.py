import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.api.rag.retrieval.published_content_repository import (
    CompositeKnowledgeRepository,
    PublishedContentRepository,
)


class _PrimaryRepository:
    def retrieve_hybrid(self, question, embedding, candidate_k, **kwargs):
        del question, embedding, candidate_k, kwargs
        return {
            "items": [{
                "id": "pdf-unrelated",
                "text": "Một đoạn PDF không trả lời đủ quan hệ được hỏi.",
                "heading": "", "score": 1.0, "sourceId": "pdf",
                "sourceTitle": "PDF", "pageStart": 31, "pageEnd": 31,
                "channels": ["vector"],
            }],
            "candidateCount": 1, "queryTerms": ["su kien"],
            "exactMatchCount": 0, "channels": ["vector"],
        }

    def retrieve_subject_evidence(self, subjects, *, limit=240):
        del subjects, limit
        return []

    def retrieve_timeline_windows(self, question, windows, *, per_window=4):
        del question, windows, per_window
        return []


class PublishedContentRepositoryTest(unittest.TestCase):
    def _manifest(self, root: Path) -> Path:
        relative = "v-test/periods/p/stages/s/events.json"
        event_path = root / relative
        event_path.parent.mkdir(parents=True)
        payload = json.dumps([{
            "id": "event-a", "status": "published",
            "title": "Sự kiện A tại Địa danh B",
            "summary": "Lực lượng A bắt đầu hoạt động tại Địa danh B ngày 2/3/1901.",
            "startDate": "1901-03-02T00:00:00.000Z",
            "content": {"warSummary": [{
                "detail": "Hoạt động mở đầu tại Địa danh B.",
                "diadiem": {"content": "Địa danh B"},
            }]},
        }], ensure_ascii=False).encode()
        event_path.write_bytes(payload)
        person_relative = "v-test/persons/p/person-a.json"
        person_path = root / person_relative
        person_path.parent.mkdir(parents=True, exist_ok=True)
        person_payload = json.dumps({
            "id": "person-a", "status": "published",
            "name": "Nhân vật A",
            "title": "Nhân vật tiêu biểu thời kỳ P",
            "overview": "Nhân vật A có nhiều đóng góp trong thời kỳ P.",
            "achievements": ["Lãnh đạo hoạt động A."],
        }, ensure_ascii=False).encode()
        person_path.write_bytes(person_payload)
        manifest = {
            "contentVersion": "v-test",
            "files": {
                "periods/p/stages/s/events.json": {
                    "url": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                },
                "persons/p/person-a.json": {
                    "url": person_relative,
                    "sha256": hashlib.sha256(person_payload).hexdigest(),
                    "size": len(person_payload),
                },
            },
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        return manifest_path

    def test_search_reads_verified_published_event(self) -> None:
        with TemporaryDirectory() as temporary:
            repository = PublishedContentRepository(
                enabled=True,
                manifest_path=str(self._manifest(Path(temporary))),
                cache_ttl_seconds=300,
            )
            results = repository.search(
                "Lực lượng A bắt đầu ở đâu, vào thời gian nào?", 5,
            )
            self.assertEqual(len(results), 1)
            self.assertIn("Địa danh B", results[0]["text"])
            self.assertIn("2/3/1901", results[0]["text"])
            self.assertEqual(results[0]["yearStart"], 1901)
            self.assertIn("published_content", results[0]["channels"])
            self.assertEqual(
                results[0]["publishedContentPath"],
                "periods/p/stages/s/events.json",
            )
            self.assertEqual(results[0]["publishedContentId"], "event-a")

    def test_search_reads_verified_person_profile(self) -> None:
        with TemporaryDirectory() as temporary:
            repository = PublishedContentRepository(
                enabled=True,
                manifest_path=str(self._manifest(Path(temporary))),
            )

            results = repository.search("Nhân vật A đóng góp gì?", 5)

            person = next(
                item for item in results
                if item["publishedContentType"] == "person"
            )
            self.assertEqual(person["heading"], "Nhân vật A")
            self.assertEqual(person["publishedContentPath"], "persons/p/person-a.json")

    def test_suggestion_search_is_not_limited_by_evidence_selection(self) -> None:
        with TemporaryDirectory() as temporary:
            repository = PublishedContentRepository(
                enabled=True,
                manifest_path=str(self._manifest(Path(temporary))),
            )

            results = repository.suggest(
                "Ai là nhân vật tiêu biểu thời kỳ P?",
                "Những người tiêu biểu gồm Nhân vật A.",
                "person",
            )

            self.assertEqual([item["heading"] for item in results], ["Nhân vật A"])
            self.assertIn("published_content_suggestion", results[0]["channels"])

    def test_composite_promotes_matching_reviewed_content(self) -> None:
        with TemporaryDirectory() as temporary:
            published = PublishedContentRepository(
                enabled=True,
                manifest_path=str(self._manifest(Path(temporary))),
            )
            repository = CompositeKnowledgeRepository(_PrimaryRepository(), published)
            result = repository.retrieve_hybrid(
                "Lực lượng A bắt đầu ở đâu, vào thời gian nào?", [], 8,
            )
            self.assertEqual(result["items"][0]["heading"], "Sự kiện A tại Địa danh B")
            self.assertIn("published_content", result["channels"])

    def test_hash_mismatch_does_not_admit_unverified_content(self) -> None:
        with TemporaryDirectory() as temporary:
            manifest_path = self._manifest(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            next(iter(manifest["files"].values()))["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            repository = PublishedContentRepository(enabled=True, manifest_path=str(manifest_path))
            self.assertEqual(repository.search("Sự kiện A", 5), [])


if __name__ == "__main__":
    unittest.main()
