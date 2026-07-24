import tempfile
import unittest
from pathlib import Path

from src.api.graph_extraction import (
    ChunkGraphExtraction,
    ExtractedEntity,
    TokenUsage,
)
from src.api.index_checkpoint import IndexCheckpointStore
from src.api.pdf_ingestion import ParsedPdf, PdfChunk, PdfPage


class IndexCheckpointStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = IndexCheckpointStore(
            str(Path(self.temp_dir.name) / "checkpoints.sqlite3")
        )
        self.chunk = PdfChunk(
            chunk_id="source-p12-c1",
            text="Hà Nội là thủ đô Việt Nam.",
            page_start=12,
            page_end=12,
            content_hash="chunk-hash",
            sequence=1,
            page_chunk_index=1,
            char_count=27,
            heading="",
            word_count=7,
            facets=(),
        )
        self.parsed = ParsedPdf(
            page_count=12,
            text_page_count=1,
            content_start_page=12,
            content_end_page=12,
            skipped_page_count=11,
            boundary_detection="chapter_1_to_end",
            footnote_page_count=0,
            removed_footnote_chars=0,
            running_header_page_count=0,
            removed_running_header_chars=0,
            chunks=[self.chunk],
            pages=[
                PdfPage(
                    page_id="source-p12",
                    page_number=12,
                    char_count=27,
                    content_hash="page-hash",
                    preview="Hà Nội là thủ đô Việt Nam.",
                    extraction_status="ready",
                    chunk_count=1,
                )
            ],
            duplicate_chunk_count=0,
            ocr_applied=False,
            ocr_languages="",
            ocr_engine="",
            ocr_failed_pages=[],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _begin(self, job_id="job-1"):
        self.store.begin_job(
            source_id="source",
            document_hash="document",
            pipeline_version="pipeline",
            embedding_model="embedding",
            entity_model="entity",
            job_id=job_id,
        )

    def test_parsed_pdf_and_openai_results_survive_a_restart(self):
        self._begin()
        self.store.save_parsed_pdf("source", "document", self.parsed)
        self.store.save_embeddings(
            "source",
            "document",
            [self.chunk],
            [[0.1, 0.2]],
            TokenUsage(input_tokens=8, total_tokens=8),
        )
        extraction = ChunkGraphExtraction(
            chunk_id=self.chunk.chunk_id,
            page_number=12,
            entities=[
                ExtractedEntity(
                    canonical_id="location:ha-noi",
                    name="Hà Nội",
                    entity_type="LOCATION",
                )
            ],
        )
        self.store.save_graph(
            "source",
            "document",
            [self.chunk],
            [extraction],
            TokenUsage(input_tokens=10, output_tokens=4, total_tokens=14),
        )

        reopened = IndexCheckpointStore(str(self.store.database_path))
        parsed = reopened.load_parsed_pdf("source", "document")
        results = reopened.load_chunk_results("source", "document")

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.chunks[0].page_start, 12)
        self.assertEqual(results["chunk-hash"]["embedding"], [0.1, 0.2])
        self.assertEqual(
            results["chunk-hash"]["graph"]["entities"][0]["name"],
            "Hà Nội",
        )
        self.assertEqual(reopened.token_usage("source"), (8, 10, 4))

    def test_interrupted_job_becomes_paused_and_resumable(self):
        self._begin()
        self.assertEqual(self.store.active_job_count(), 1)
        self.store.set_progress(
            "source",
            status="running",
            phase="entities",
            total_chunks=100,
            completed_chunks=25,
            current_page=18,
        )
        self.store.recover_interrupted_jobs()

        job = self.store.persisted_job("job-1")

        self.assertEqual(job["status"], "paused")
        self.assertTrue(job["resumable"])
        self.assertEqual(job["completed_chunks"], 25)
        self.assertEqual(job["current_page"], 18)
        self.assertEqual(self.store.active_job_count(), 0)

    def test_new_document_discards_incompatible_checkpoint(self):
        self._begin()
        self.store.save_embeddings(
            "source",
            "document",
            [self.chunk],
            [[0.1]],
            TokenUsage(input_tokens=3, total_tokens=3),
        )
        self.store.begin_job(
            source_id="source",
            document_hash="new-document",
            pipeline_version="pipeline",
            embedding_model="embedding",
            entity_model="entity",
            job_id="job-2",
        )

        self.assertEqual(
            self.store.load_chunk_results("source", "new-document"),
            {},
        )
        self.assertEqual(self.store.token_usage("source"), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
