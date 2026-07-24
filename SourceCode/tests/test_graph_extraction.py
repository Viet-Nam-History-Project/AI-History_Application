import unittest
from types import SimpleNamespace

from src.api.graph_extraction import (
    PdfGraphExtractionService,
    TokenUsage,
    canonical_entity_id,
)
from src.api.pdf_ingestion import PdfChunk


def make_chunk() -> PdfChunk:
    return PdfChunk(
        chunk_id="source-p1-c0",
        text=(
            "Nguyễn Thị Định lãnh đạo phong trào Đồng khởi tại Bến Tre "
            "trong thời kỳ kháng chiến."
        ),
        page_start=1,
        page_end=1,
        content_hash="hash",
        sequence=0,
        page_chunk_index=0,
        char_count=96,
        heading="",
    )


class GraphExtractionNormalizationTest(unittest.TestCase):
    def test_canonical_id_is_stable_across_case_and_accents(self) -> None:
        first = canonical_entity_id("Nguyễn Thị Định", "PERSON")
        second = canonical_entity_id("nguyen thi dinh", "PERSON")

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("person:nguyen-thi-dinh:"))

    def test_canonical_id_keeps_entity_types_separate(self) -> None:
        person = canonical_entity_id("Hà Nội", "PERSON")
        location = canonical_entity_id("Hà Nội", "LOCATION")

        self.assertNotEqual(person, location)

    def test_named_person_and_direct_relationship_are_normalized(self) -> None:
        extraction = PdfGraphExtractionService._normalize_chunk(
            make_chunk(),
            {
                "entities": [
                    {"name": " Nguyễn Thị Định ", "type": "PERSON"},
                    {"name": "Đồng khởi", "type": "EVENT"},
                    {"name": "Bến Tre", "type": "LOCATION"},
                    {"name": "nguyen thi dinh", "type": "PERSON"},
                    {"name": "phong trào", "type": "CONCEPT"},
                    {"name": "Hồ Chí Minh", "type": "PERSON"},
                ],
                "relationships": [
                    {
                        "source": "Nguyễn Thị Định",
                        "source_type": "PERSON",
                        "target": "Đồng khởi",
                        "target_type": "EVENT",
                        "type": "LED",
                        "evidence": "Nguyễn Thị Định lãnh đạo phong trào Đồng khởi.",
                        "raw_predicate": "lãnh đạo",
                    },
                    {
                        "source": "Nguyễn Thị Định",
                        "source_type": "PERSON",
                        "target": "kháng chiến",
                        "target_type": "EVENT",
                        "type": "PARTICIPATES_IN",
                        "evidence": "Không có entity đích tương ứng.",
                        "raw_predicate": "tham gia",
                    },
                ],
            },
        )

        names = {entity.name for entity in extraction.entities}
        self.assertEqual(names, {"Nguyễn Thị Định", "Đồng khởi", "Bến Tre"})
        self.assertEqual(len(extraction.relationships), 1)
        self.assertEqual(
            extraction.relationships[0].source_canonical_id,
            canonical_entity_id("Nguyễn Thị Định", "PERSON"),
        )
        self.assertEqual(
            extraction.relationships[0].target_canonical_id,
            canonical_entity_id("Đồng khởi", "EVENT"),
        )
        self.assertEqual(extraction.relationships[0].relationship_type, "LED")
        self.assertEqual(
            extraction.relationships[0].raw_predicate,
            "lãnh đạo",
        )

    def test_known_aliases_merge_into_one_canonical_person(self) -> None:
        chunk = make_chunk()
        chunk.text = (
            "Nguyễn Ái Quốc, sau này được biết đến với tên Hồ Chí Minh, "
            "là một lãnh tụ cách mạng."
        )
        extraction = PdfGraphExtractionService._normalize_chunk(
            chunk,
            {
                "entities": [
                    {"name": "Nguyễn Ái Quốc", "type": "PERSON"},
                    {"name": "Hồ Chí Minh", "type": "PERSON"},
                ],
                "relationships": [],
            },
        )

        self.assertEqual(len(extraction.entities), 1)
        entity = extraction.entities[0]
        self.assertEqual(entity.name, "Hồ Chí Minh")
        self.assertIn("Nguyễn Ái Quốc", entity.aliases)
        self.assertIn("bac ho", entity.search_keys)

    def test_quantities_dates_and_generic_mentions_are_rejected(self) -> None:
        chunk = make_chunk()
        chunk.text = (
            "Ngày 10/3/1975 có 140.000 người; nhân dân và quân đội "
            "tiếp tục chiến đấu."
        )
        extraction = PdfGraphExtractionService._normalize_chunk(
            chunk,
            {
                "entities": [
                    {"name": "10/3/1975", "type": "EVENT"},
                    {"name": "140.000 người", "type": "PERSON"},
                    {"name": "nhân dân", "type": "ORGANIZATION"},
                    {"name": "quân đội", "type": "ORGANIZATION"},
                ],
                "relationships": [],
            },
        )

        self.assertEqual(extraction.entities, [])

    def test_agreement_alias_repairs_wrong_location_type(self) -> None:
        chunk = make_chunk()
        chunk.text = (
            "Hiệp định Giơnevơ được ký tại Thụy Sĩ và quy định việc "
            "đình chỉ chiến sự ở Đông Dương."
        )
        extraction = PdfGraphExtractionService._normalize_chunk(
            chunk,
            {
                "entities": [
                    {"name": "Hiệp định Giơnevơ", "type": "LOCATION"},
                    {"name": "Thụy Sĩ", "type": "STATE"},
                ],
                "relationships": [
                    {
                        "source": "Hiệp định Giơnevơ",
                        "source_type": "LOCATION",
                        "target": "Thụy Sĩ",
                        "target_type": "STATE",
                        "type": "OCCURRED_AT",
                        "evidence": "Hiệp định Giơnevơ được ký tại Thụy Sĩ.",
                        "raw_predicate": "được ký tại",
                    },
                ],
            },
        )

        agreement = next(
            item for item in extraction.entities
            if item.name == "Hiệp định Genève"
        )
        self.assertEqual(agreement.entity_type, "EVENT")
        self.assertIn("Hiệp định Giơnevơ", agreement.aliases)
        self.assertIn("hiep dinh gionevo", agreement.search_keys)
        self.assertEqual(len(extraction.relationships), 1)
        self.assertEqual(
            extraction.relationships[0].source_canonical_id,
            canonical_entity_id("Hiệp định Genève", "EVENT"),
        )

    def test_lowercase_common_nouns_are_not_entities(self) -> None:
        chunk = make_chunk()
        chunk.text = (
            "cao su, đường sắt và địa chủ đều xuất hiện trong phần "
            "phân tích kinh tế xã hội."
        )
        extraction = PdfGraphExtractionService._normalize_chunk(
            chunk,
            {
                "entities": [
                    {"name": "cao su", "type": "EVENT"},
                    {"name": "đường sắt", "type": "LOCATION"},
                    {"name": "địa chủ", "type": "ORGANIZATION"},
                ],
                "relationships": [],
            },
        )

        self.assertEqual(extraction.entities, [])


class TokenUsageTest(unittest.TestCase):
    def test_usage_supports_chat_and_embedding_responses(self) -> None:
        usage = TokenUsage()
        usage.add_response(SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
        )))
        usage.add_response(SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=80,
            total_tokens=80,
        )))

        self.assertEqual(usage.input_tokens, 200)
        self.assertEqual(usage.output_tokens, 30)
        self.assertEqual(usage.total_tokens, 230)


if __name__ == "__main__":
    unittest.main()
