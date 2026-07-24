import json
import unittest
from dataclasses import replace
from types import SimpleNamespace

from src.api.query_analysis import analyze_query, build_query_plan, normalize_query
from src.api.rag_service import (
    OBSOLETE_EVOLUTION_PLANNING_INSTRUCTION_V23,
    OBSOLETE_EVOLUTION_PLANNING_INSTRUCTION_V24,
    RagService,
    effective_answer_planning_instruction,
    naturalize_source_meta_language,
)
from src.api.neo4j_repository import Neo4jKnowledgeRepository


class _FakeCompletions:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def create(self, **_kwargs):
        message = SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class RagServiceRetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = RagService.__new__(RagService)
        self.service.settings = SimpleNamespace(
            ai_enable_rerank=True,
            ai_retrieval_top_k=4,
            ai_retrieval_candidate_k=12,
            openai_chat_model="test-model",
        )

    def test_log_failure_does_not_discard_chat_response(self) -> None:
        class FailingRepository:
            @staticmethod
            def log_query(_payload: dict) -> None:
                raise RuntimeError("Neo4j logging unavailable")

        self.service.repository = FailingRepository()

        with self.assertLogs("src.api.rag_service", level="ERROR") as logs:
            self.service._log_query({
                "confidenceFactors": {"entity_match": 1.0},
            })

        self.assertIn("Không thể ghi AIQueryLog vào Neo4j", logs.output[0])

    def test_source_meta_language_is_rewritten_as_natural_answer(self) -> None:
        answer = naturalize_source_meta_language(
            "Ngoài ra, nguồn tài liệu cũng nêu rõ nhiều loại thuế. "
            "Theo nguồn tài liệu, Pháp còn duy trì các cơ sở độc quyền."
        )

        self.assertEqual(
            answer,
            "Ngoài ra, có nhiều loại thuế. Pháp còn duy trì các cơ sở độc quyền.",
        )

    def test_active_prompt_receives_mandatory_comparison_rule(self) -> None:
        effective = effective_answer_planning_instruction(
            "QUY TẮC LẬP CÂU TRẢ LỜI:\n- Chỉ dùng bằng chứng đã truy xuất.",
        )

        self.assertIn(
            "ADAPTIVE_COMPARISON_PLANNING_V22",
            effective,
        )
        self.assertEqual(
            effective.count("ADAPTIVE_COMPARISON_PLANNING_V22"),
            1,
        )
        self.assertIn(
            "RECOVERABLE_BOUNDARY_EVOLUTION_PLANNING_F9_V28",
            effective,
        )
        self.assertEqual(
            effective.count(
                "RECOVERABLE_BOUNDARY_EVOLUTION_PLANNING_F9_V28"
            ),
            1,
        )

    def test_effective_prompt_replaces_obsolete_f9_v23_rule(self) -> None:
        effective = effective_answer_planning_instruction(
            "Chỉ dẫn Admin\n\n"
            + OBSOLETE_EVOLUTION_PLANNING_INSTRUCTION_V23
        )

        self.assertNotIn("ADAPTIVE_EVOLUTION_PLANNING_F9_V23", effective)
        self.assertIn(
            "RECOVERABLE_BOUNDARY_EVOLUTION_PLANNING_F9_V28",
            effective,
        )

    def test_effective_prompt_replaces_obsolete_f9_v24_rule(self) -> None:
        effective = effective_answer_planning_instruction(
            "Chỉ dẫn Admin\n\n"
            + OBSOLETE_EVOLUTION_PLANNING_INSTRUCTION_V24
        )

        self.assertNotIn("DYNAMIC_EVOLUTION_PLANNING_F9_V24", effective)
        self.assertEqual(
            effective.count(
                "RECOVERABLE_BOUNDARY_EVOLUTION_PLANNING_F9_V28"
            ),
            1,
        )

    def test_effective_prompt_replaces_obsolete_f9_v25_rule(self) -> None:
        effective = effective_answer_planning_instruction(
            "Chỉ dẫn Admin\n\n"
            "- EVIDENCE_VERIFIED_EVOLUTION_PLANNING_F9_V25:\n"
            "- Quy tắc cũ của F9.\n"
            "- Chỉ khẳng định điều có bằng chứng; "
            "không tự điền bằng kiến thức ngoài nguồn."
        )

        self.assertNotIn(
            "EVIDENCE_VERIFIED_EVOLUTION_PLANNING_F9_V25",
            effective,
        )
        self.assertEqual(
            effective.count(
                "RECOVERABLE_BOUNDARY_EVOLUTION_PLANNING_F9_V28"
            ),
            1,
        )

    def test_effective_prompt_replaces_obsolete_f9_v26_rule(self) -> None:
        effective = effective_answer_planning_instruction(
            "Chỉ dẫn Admin\n\n"
            "- TIMELINE_COVERAGE_EVOLUTION_PLANNING_F9_V26:\n"
            "- Quy tắc cũ của F9.\n"
            "- Nếu thiếu period, chỉ nêu các thay đổi có bằng chứng "
            "và nói ngắn gọn giới hạn độ phủ."
        )

        self.assertNotIn(
            "TIMELINE_COVERAGE_EVOLUTION_PLANNING_F9_V26",
            effective,
        )
        self.assertEqual(
            effective.count(
                "RECOVERABLE_BOUNDARY_EVOLUTION_PLANNING_F9_V28"
            ),
            1,
        )

    def test_model_refines_f9_facets_without_accepting_wrong_domain_facet(self) -> None:
        plan = build_query_plan(
            "Chính sách cai trị của Pháp thay đổi như thế nào "
            "từ 1897 đến 1945?",
        )
        self.service.client = SimpleNamespace(chat=SimpleNamespace(
            completions=_FakeCompletions({
                "evolution_intent": "evolution_over_time",
                "evolution_domain": "political_administration",
                "subject_type": "colonial_governance_policy",
                "selected_facets": [
                    "historical_evolution",
                    "continuity_change",
                    "cause",
                    "governance_methods",
                    "wartime_mobilization",
                    "agriculture",
                ],
                "rationale": "Chọn facet đúng đối tượng cai trị.",
            }),
        ))

        refined, used_model = self.service._refine_evolution_plan(
            plan.subject,
            plan,
        )

        self.assertTrue(used_model)
        self.assertIn("governance_methods", refined.required_facets)
        self.assertIn("wartime_mobilization", refined.required_facets)
        self.assertNotIn("agriculture", refined.required_facets)
        self.assertEqual(
            refined.evolution_subject_type,
            "colonial_governance_policy",
        )

    def test_f9_subject_excludes_change_request_and_date_range(self) -> None:
        plan = build_query_plan(
            "Chính sách cai trị của Pháp thay đổi như thế nào "
            "từ 1897 đến 1945?",
        )

        self.assertEqual(plan.subject, "Chính sách cai trị của Pháp")
        self.assertEqual(plan.explicit_date_range, "1897–1945")

    def test_revenue_source_wording_is_not_removed(self) -> None:
        answer = "Nguồn thu chính gồm thuế thân, thuế ruộng và thuế muối."
        self.assertEqual(naturalize_source_meta_language(answer), answer)

    @staticmethod
    def _candidates() -> list[dict]:
        relevant = [
            {
                "id": "dbp-1",
                "sourceTitle": "Lịch sử kháng chiến chống Pháp",
                "pageStart": 120,
                "heading": "Chiến dịch Điện Biên Phủ",
                "text": "Chiến dịch Điện Biên Phủ do Đại tướng Võ Nguyên Giáp chỉ huy.",
                "score": 0.91,
            },
            {
                "id": "dbp-2",
                "sourceTitle": "Lịch sử Việt Nam",
                "pageStart": 125,
                "heading": "Kết quả",
                "text": "Kết quả chiến dịch Điện Biên Phủ là tập đoàn cứ điểm bị tiêu diệt.",
                "score": 0.84,
            },
        ]
        generic = [
            {
                "id": f"other-{index}",
                "sourceTitle": "Lịch sử Việt Nam",
                "pageStart": index,
                "heading": "Bối cảnh",
                "text": "Một đoạn lịch sử không liên quan trực tiếp.",
                "score": 0.3 - index * 0.01,
            }
            for index in range(1, 4)
        ]
        return relevant + generic

    def test_rerank_normalizes_selected_ids(self) -> None:
        self.service.client = SimpleNamespace(chat=SimpleNamespace(
            completions=_FakeCompletions({
                "intent": "giải thích",
                "scope": "Chiến dịch Điện Biên Phủ",
                "date_range": "1954",
                "answerable": True,
                "covered_facets": ["chỉ huy"],
                "evidence_coverage": 0.25,
                "selected_ids": [" dbp-1 "],
            }),
        ))
        selected, analysis = self.service._rerank("Hỏi về Điện Biên Phủ", self._candidates())
        self.assertEqual([item["id"] for item in selected], ["dbp-1"])
        self.assertEqual(analysis["date_range"], "1954")
        self.assertEqual(analysis["evidence_coverage"], 0.25)

    def test_rerank_returns_contextually_normalized_question(self) -> None:
        original = "Nói cho tôi biết cách chính sách thuộc địa của Pháp"
        normalization = normalize_query(original)
        self.service.client = SimpleNamespace(chat=SimpleNamespace(
            completions=_FakeCompletions({
                "intent": "giải thích",
                "scope": "chính sách thuộc địa của Pháp",
                "date_range": "",
                "answerable": True,
                "covered_facets": [],
                "evidence_coverage": 1.0,
                "selected_ids": ["dbp-1"],
                "normalized_question": normalization.normalized_question,
                "corrections": [{
                    "original": "cách chính sách",
                    "replacement": "các chính sách",
                    "reason": "Lỗi gõ theo ngữ cảnh",
                }],
                "rewrite_confidence": 0.99,
                "ambiguous": False,
                "ambiguity_notes": [],
            }),
        ))

        _, analysis = self.service._rerank(
            normalization.normalized_question,
            self._candidates(),
            normalization=normalization,
        )

        self.assertEqual(
            analysis["normalized_question"],
            "Nói cho tôi biết các chính sách thuộc địa của Pháp",
        )
        self.assertFalse(analysis["query_ambiguous"])
        self.assertGreaterEqual(len(analysis["query_corrections"]), 1)

    def test_contextual_rewrite_cannot_change_explicit_year(self) -> None:
        normalization = normalize_query("Điện Biên Phủ diễn ra năm 1954?")
        analysis = self.service._model_query_analysis(
            normalization.normalized_question,
            {
                "normalized_question": "Điện Biên Phủ diễn ra năm 1955?",
                "corrections": [],
                "rewrite_confidence": 0.99,
                "ambiguous": False,
                "ambiguity_notes": [],
            },
            normalization,
        )

        self.assertEqual(
            analysis["normalized_question"],
            "Điện Biên Phủ diễn ra năm 1954?",
        )

    def test_contextual_rewrite_cannot_replace_named_subject(self) -> None:
        normalization = normalize_query("Nguyễn Thị Định là ai?")
        analysis = self.service._model_query_analysis(
            normalization.normalized_question,
            {
                "normalized_question": "Nguyễn Thị Minh Khai là ai?",
                "corrections": [],
                "rewrite_confidence": 0.99,
                "ambiguous": False,
                "ambiguity_notes": [],
            },
            normalization,
        )

        self.assertEqual(analysis["normalized_question"], "Nguyễn Thị Định là ai?")

    def test_contextual_rewrite_cannot_append_inferred_scope(self) -> None:
        normalization = normalize_query(
            "Nói cho tôi biết cách chính sách thuộc địa của Pháp",
        )
        analysis = self.service._model_query_analysis(
            normalization.normalized_question,
            {
                "normalized_question": (
                    "Nói cho tôi biết các chính sách thuộc địa "
                    "của Pháp ở Việt Nam"
                ),
                "corrections": [],
                "rewrite_confidence": 0.99,
                "ambiguous": False,
                "ambiguity_notes": [],
            },
            normalization,
        )

        self.assertEqual(
            analysis["normalized_question"],
            "Nói cho tôi biết các chính sách thuộc địa của Pháp",
        )

    def test_unchanged_ambiguous_abbreviation_stays_ambiguous(self) -> None:
        normalization = normalize_query("HCM có vai trò gì?")
        analysis = self.service._model_query_analysis(
            normalization.normalized_question,
            {
                "normalized_question": "HCM có vai trò gì?",
                "corrections": [],
                "rewrite_confidence": 0.99,
                "ambiguous": False,
                "ambiguity_notes": [],
            },
            normalization,
        )

        self.assertTrue(analysis["query_ambiguous"])

    def test_fulltext_stop_words_are_filtered_even_with_accents(self) -> None:
        _, terms = Neo4jKnowledgeRepository._fulltext_query(
            "Nói cho tôi biết các chính sách thuộc địa của Pháp có gì",
        )

        self.assertNotIn("nói", terms)
        self.assertNotIn("tôi", terms)
        self.assertNotIn("biết", terms)
        self.assertNotIn("các", terms)
        self.assertNotIn("của", terms)
        self.assertNotIn("có", terms)
        self.assertEqual(terms, ["chính", "sách", "thuộc", "địa", "pháp"])

    def test_invalid_model_ids_do_not_erase_answerable_candidates(self) -> None:
        self.service.client = SimpleNamespace(chat=SimpleNamespace(
            completions=_FakeCompletions({
                "answerable": True,
                "selected_ids": ["unknown-id"],
            }),
        ))
        selected, _ = self.service._rerank("Hỏi về Điện Biên Phủ", self._candidates())
        self.assertEqual(len(selected), 4)

    def test_recovery_query_puts_inferred_year_before_original_question(self) -> None:
        query = self.service._recovery_query(
            "Chiến dịch Điện Biên Phủ diễn ra như thế nào?",
            {
                "scope": "Chiến dịch Điện Biên Phủ năm 1954",
                "date_range": "1954",
            },
        )
        self.assertTrue(query.startswith("Chiến dịch Điện Biên Phủ năm 1954"))

    def test_known_aliases_expand_exact_retrieval_phrases(self) -> None:
        signals = analyze_query("Hiệp định Genève là gì?")

        phrases = self.service._expanded_exact_phrases(signals)

        self.assertIn("Hiệp định Genève", phrases)
        self.assertIn("Hiệp định Giơnevơ", phrases)
        self.assertIn("Hiệp định Giơnơvơ", phrases)

    def test_known_subject_grounding_rejects_another_campaign(self) -> None:
        signals = analyze_query(
            "Chiến thắng Điện Biên Phủ có kết quả và ý nghĩa gì?",
        )
        candidates = [
            {
                "id": "wrong",
                "sourceId": "1975",
                "sourceTitle": "1965-1975",
                "pageStart": 500,
                "text": (
                    "Chiến dịch Hồ Chí Minh kết thúc thắng lợi năm 1975, "
                    "phát huy tinh thần Điện Biên Phủ."
                ),
            },
            {
                "id": "agreement",
                "sourceId": "1954",
                "pageStart": 19,
                "yearStart": 1954,
                "yearEnd": 1960,
                "text": (
                    "Sau thất bại của Pháp ở Điện Biên Phủ, "
                    "Hiệp định Giơnevơ được ký kết."
                ),
            },
        ]

        grounded, applied = self.service._ground_candidates_by_known_subject(
            candidates,
            signals,
        )

        self.assertTrue(applied)
        self.assertEqual([item["id"] for item in grounded], ["agreement"])

    def test_comparison_expands_queries_for_each_side_and_facet(self) -> None:
        plan = build_query_plan(
            "So sánh hai cuộc khai thác thuộc địa của Pháp",
        )

        queries = self.service._expanded_recovery_queries(
            "So sánh hai cuộc khai thác thuộc địa của Pháp",
            {"scope": plan.inferred_scope, "date_range": ""},
            plan,
        )

        self.assertEqual(len(queries), 1 + 2 * len(plan.required_facets))
        self.assertTrue(any(
            "Cuộc khai thác thuộc địa lần thứ nhất" in query
            and "nông nghiệp" in query
            for query in queries
        ))
        self.assertTrue(any(
            "Cuộc khai thác thuộc địa lần thứ hai" in query
            and "thương nghiệp" in query
            for query in queries
        ))

    def test_model_selects_only_relevant_comparison_criteria(self) -> None:
        plan = build_query_plan(
            "So sánh chiến lược “Chiến tranh đặc biệt” và "
            "“Chiến tranh cục bộ” của Mỹ ở miền Nam Việt Nam",
        )
        self.service.client = SimpleNamespace(chat=SimpleNamespace(
            completions=_FakeCompletions({
                "selected_facets": [
                    "comparison_context",
                    "forces",
                    "geographic_scope",
                    "strategy_methods",
                    "agriculture",
                ],
                "rationale": "Chọn tiêu chí quân sự có giá trị phân biệt.",
            }),
        ))

        refined, used_model = self.service._refine_comparison_plan(
            plan.subject,
            plan,
        )

        self.assertTrue(used_model)
        self.assertEqual(
            refined.required_facets,
            (
                "comparison_context",
                "forces",
                "geographic_scope",
                "strategy_methods",
                "representative_events",
                "result",
            ),
        )
        self.assertNotIn("agriculture", refined.required_facets)

    def test_military_comparison_expansion_does_not_search_economic_sectors(self) -> None:
        plan = build_query_plan(
            "So sánh chiến lược “Chiến tranh đặc biệt” và "
            "“Chiến tranh cục bộ” của Mỹ ở miền Nam Việt Nam",
        )

        queries = self.service._expanded_recovery_queries(
            plan.subject,
            {"scope": plan.inferred_scope, "date_range": ""},
            plan,
        )

        combined = "\n".join(queries).casefold()
        self.assertIn("ấp chiến lược", combined)
        self.assertIn("lực lượng tham chiến", combined)
        self.assertNotIn("nông nghiệp ruộng đất", combined)
        self.assertNotIn("thương nghiệp thị trường", combined)

    def test_comparison_side_prefers_subject_text_over_broad_source_period(self) -> None:
        plan = build_query_plan(
            "So sánh chiến lược “Chiến tranh đặc biệt” và "
            "“Chiến tranh cục bộ” của Mỹ ở miền Nam Việt Nam",
        )

        sides = self.service._comparison_side_indexes(
            {
                "sourceTitle": "Lịch sử Việt Nam 1945-1965",
                "heading": "Chiến tranh cục bộ",
                "text": "Mỹ chuyển sang chiến lược Chiến tranh cục bộ.",
            },
            plan,
        )

        self.assertEqual(sides, (1,))

    def test_agreement_comparison_uses_alias_without_matching_both_agreements(self) -> None:
        plan = build_query_plan(
            "So sánh Hiệp định Genève và Hiệp định Paris",
        )

        sides = self.service._comparison_side_indexes(
            {
                "sourceTitle": "Lịch sử ngoại giao Việt Nam",
                "heading": "Hiệp định Giơnevơ năm 1954",
                "text": "Hiệp định Giơnevơ quy định đình chỉ chiến sự.",
            },
            plan,
        )
        queries = self.service._expanded_recovery_queries(
            plan.subject,
            {"scope": plan.inferred_scope, "date_range": ""},
            plan,
        )

        self.assertEqual(sides, (0,))
        self.assertTrue(any("Hiệp định Giơnevơ" in query for query in queries))

    def test_comparison_selection_keeps_evidence_for_both_sides(self) -> None:
        plan = build_query_plan(
            "So sánh hai cuộc khai thác thuộc địa của Pháp",
        )
        candidates = [
            {
                "id": "first-agriculture",
                "sourceTitle": "1897-1918",
                "pageStart": 102,
                "text": "Nông nghiệp đồn điền phát triển, ruộng đất bị tập trung.",
                "score": 0.8,
            },
            {
                "id": "second-agriculture",
                "sourceTitle": "1919-1930",
                "pageStart": 131,
                "text": "Vốn đầu tư nông nghiệp và đồn điền cao su tăng mạnh.",
                "score": 0.8,
            },
        ]

        selected = self.service._select_with_comparison_coverage(
            [],
            candidates,
            plan,
            4,
        )

        self.assertEqual(
            {item["id"] for item in selected},
            {"first-agriculture", "second-agriculture"},
        )

    def test_comparison_evidence_matrix_tracks_each_side_and_missing_cell(self) -> None:
        plan = build_query_plan(
            "So sánh hai cuộc khai thác thuộc địa của Pháp về nông nghiệp",
        )
        details = self.service._comparison_evidence_details(
            [
                {
                    "id": "first-agriculture",
                    "sourceTitle": "1897-1918",
                    "text": "Nông nghiệp đồn điền và ruộng đất được mở rộng.",
                },
            ],
            plan,
        )

        matrix = details["comparison_evidence"]["agriculture"]
        self.assertEqual(
            matrix["Cuộc khai thác thuộc địa lần thứ nhất của Pháp"],
            ["first-agriculture"],
        )
        self.assertEqual(
            matrix["Cuộc khai thác thuộc địa lần thứ hai của Pháp"],
            [],
        )
        self.assertEqual(details["balanced_facets"], [])
        self.assertIn(
            "agriculture::Cuộc khai thác thuộc địa lần thứ hai của Pháp",
            details["missing_comparison_cells"],
        )

    def test_comparison_period_gate_rejects_unrelated_later_volume(self) -> None:
        plan = build_query_plan(
            "So sánh hai cuộc khai thác thuộc địa của Pháp",
        )
        candidates, filtered = (
            self.service._filter_comparison_period_candidates(
                [
                    {"id": "first", "sourceTitle": "1897-1918"},
                    {"id": "second", "sourceTitle": "1919-1930"},
                    {"id": "wrong", "sourceTitle": "1945-1965"},
                    {"id": "unknown", "sourceTitle": "Lịch sử kinh tế Việt Nam"},
                ],
                plan,
            )
        )

        self.assertTrue(filtered)
        self.assertEqual(
            [item["id"] for item in candidates],
            ["first", "second", "unknown"],
        )

    def test_final_period_gate_removes_later_retrospective_volume(self) -> None:
        signals = analyze_query(
            "Chiến thắng Điện Biên Phủ có kết quả và ý nghĩa gì?",
        )
        results, filtered = self.service._filter_results_by_known_period(
            [
                {
                    "id": "correct",
                    "sourceTitle": "1945-1965",
                    "text": "Hiệp định Giơnevơ được ký ngày 20-7-1954.",
                },
                {
                    "id": "retrospective",
                    "sourceTitle": "1965-1975",
                    "yearStart": 1954,
                    "yearEnd": 1954,
                    "text": "Nhắc lại Chiến dịch Điện Biên Phủ năm 1954.",
                },
            ],
            signals,
        )

        self.assertTrue(filtered)
        self.assertEqual([item["id"] for item in results], ["correct"])

    def test_missing_inferred_year_triggers_recovery_even_with_initial_results(self) -> None:
        analysis = {
            "scope": "Chiến dịch Điện Biên Phủ",
            "date_range": "1954",
        }
        self.assertTrue(self.service._needs_query_recovery(
            "Chiến dịch Điện Biên Phủ diễn ra như thế nào?",
            analysis,
            True,
        ))
        self.assertFalse(self.service._needs_query_recovery(
            "Chiến dịch Điện Biên Phủ 1954 diễn ra như thế nào?",
            analysis,
            True,
        ))

    def test_recovery_queries_cover_each_requested_facet(self) -> None:
        queries = self.service._expanded_recovery_queries(
            "Chiến dịch Điện Biên Phủ diễn ra thế nào, gồm lực lượng gì, "
            "ai là người lạnh đão và kết quả ra sao?",
            {
                "scope": "Chiến dịch Điện Biên Phủ",
                "date_range": "1954",
            },
        )
        self.assertEqual(len(queries), 5)
        self.assertTrue(all("1954" in query for query in queries))
        self.assertTrue(any("diễn biến" in query for query in queries))
        self.assertTrue(any("lực lượng tham chiến" in query for query in queries))
        self.assertTrue(any("lãnh đạo" in query for query in queries))
        self.assertTrue(any("kết quả" in query for query in queries))
        self.assertTrue(any("mốc thời gian" in query for query in queries))
        self.assertTrue(any("quân số" in query for query in queries))

    def test_multi_part_question_expands_even_when_year_is_explicit(self) -> None:
        self.assertTrue(self.service._needs_query_recovery(
            "Chiến dịch Điện Biên Phủ 1954 diễn ra thế nào, "
            "lực lượng gì, ai lãnh đạo và kết quả ra sao?",
            {
                "scope": "Chiến dịch Điện Biên Phủ",
                "date_range": "1954",
            },
            True,
        ))

    def test_policy_overview_expands_each_required_facet_without_date(self) -> None:
        question = "Nói cho tôi biết các chính sách thuộc địa của Pháp"
        plan = build_query_plan(question)
        queries = self.service._expanded_recovery_queries(
            question,
            {"scope": plan.inferred_scope, "date_range": "1897–1945"},
            plan,
        )

        self.assertGreaterEqual(len(queries), 6)
        self.assertTrue(any("nam kỳ bắc kỳ" in query for query in queries))
        self.assertTrue(any("độc quyền muối" in query for query in queries))
        self.assertTrue(any("trường học chương trình" in query for query in queries))
        self.assertTrue(any("giai cấp công nhân" in query for query in queries))
        self.assertTrue(all("1897" not in query for query in queries))
        self.assertTrue(any("độc quyền muối" in query for query in queries))
        self.assertTrue(any("nam kỳ bắc kỳ trung kỳ" in query for query in queries))

    def test_f9_first_pass_searches_turning_points_without_fixed_periods(self) -> None:
        question = (
            "Chính sách cai trị của Pháp thay đổi như thế nào "
            "từ 1897 đến 1945?"
        )
        plan = build_query_plan(question)
        queries = self.service._expanded_recovery_queries(
            question,
            {
                "scope": plan.inferred_scope,
                "date_range": plan.explicit_date_range,
            },
            plan,
        )
        combined = "\n".join(queries)

        self.assertEqual(plan.evolution_periods, ())
        self.assertIn("bước ngoặt", combined)
        self.assertIn("trạng thái đầu kỳ", combined)
        self.assertTrue(any("nhượng bộ" in query for query in queries))
        self.assertTrue(any("thời chiến" in query for query in queries))

    def test_f9_source_year_metadata_cannot_become_boundary_evidence(self) -> None:
        cards = self.service._evolution_evidence_cards(
            [{
                "id": "broad-volume",
                "sourceTitle": "Lịch sử Việt Nam 1897-1945",
                "heading": "Kết luận",
                "text": (
                    "Chính sách cai trị thay đổi qua nhiều chặng nhưng "
                    "mục tiêu thống trị vẫn được duy trì."
                ),
                "yearStart": 1897,
                "yearEnd": 1945,
                "score": 0.9,
            }],
            requested_range=(1897, 1945),
            subject="chính sách cai trị của Pháp",
        )

        self.assertEqual(cards[0]["explicit_content_years"], [])
        self.assertFalse(cards[0]["boundary_eligible"])

    def test_f9_boundary_verifier_corrects_inverted_actor(self) -> None:
        self.service.settings.openai_entity_model = "test-verifier"
        self.service.client = SimpleNamespace(chat=SimpleNamespace(
            completions=_FakeCompletions({
                "checks": [{
                    "period_index": 0,
                    "evidence_id": "turn-1939",
                    "evidence_sufficient": True,
                    "agency_was_correct": False,
                    "actor": "Chính phủ Pháp ngả sang hữu",
                    "action": "xóa bỏ dần các cải cách",
                    "cause": (
                        "Chính phủ Pháp ngả sang hữu và xóa bỏ dần "
                        "các cải cách của Mặt trận Nhân dân"
                    ),
                    "reason": "Mặt trận Nhân dân là bên tiến hành cải cách.",
                }],
                "all_boundaries_groundable": True,
            }),
        ))
        periods = [(
            1939,
            1945,
            "Huy động và đàn áp thời chiến",
            "Siết chặt kiểm soát",
            "Mặt trận Nhân dân xóa bỏ cải cách",
            1939,
            "Mặt trận Nhân dân",
            "xóa bỏ cải cách",
            "turn-1939",
            (
                "Chính phủ Pháp ngả hẳn sang hữu, xóa bỏ dần những "
                "cải cách mà Mặt trận Nhân dân tiến hành"
            ),
            ("turn-1939",),
        )]

        verified = self.service._verify_evolution_boundary_agency(
            "chính sách cai trị của Pháp",
            periods,
        )

        self.assertIsNotNone(verified)
        self.assertEqual(
            verified[0][6],
            "Chính phủ Pháp ngả sang hữu",
        )
        self.assertNotIn(
            "Mặt trận Nhân dân xóa bỏ",
            verified[0][4],
        )

    def test_f9_boundary_verifier_downgrades_only_uncertain_agency(self) -> None:
        self.service.settings.openai_entity_model = "test-verifier"
        self.service.client = SimpleNamespace(chat=SimpleNamespace(
            completions=_FakeCompletions({
                "checks": [{
                    "period_index": 0,
                    "evidence_id": "turn-1939",
                    "evidence_sufficient": False,
                    "agency_was_correct": False,
                    "actor": "",
                    "action": "",
                    "cause": "",
                    "reason": "Đoạn trích chỉ đủ chứng minh trạng thái đổi.",
                }],
                "all_boundaries_groundable": False,
            }),
        ))
        periods = [(
            1939,
            1945,
            "Kiểm soát thời chiến",
            "Chế độ cai trị chuyển sang thời chiến",
            "Nguyên nhân chưa chắc chắn",
            1939,
            "Chủ thể suy đoán",
            "Hành động suy đoán",
            "turn-1939",
            "Từ năm 1939, Đông Dương chuyển sang chế độ thời chiến.",
            ("turn-1939",),
        )]

        verified = self.service._verify_evolution_boundary_agency(
            "chính sách cai trị của Pháp",
            periods,
        )

        self.assertIsNotNone(verified)
        self.assertEqual(verified[0][0:4], periods[0][0:4])
        self.assertEqual(verified[0][5], 1939)
        self.assertEqual(verified[0][8:11], periods[0][8:11])
        self.assertEqual(verified[0][4], "")
        self.assertEqual(verified[0][6], "")
        self.assertEqual(verified[0][7], "")

    def test_f9_second_pass_expands_dynamic_evidence_periods(self) -> None:
        question = (
            "Chính sách cai trị của Pháp thay đổi như thế nào "
            "từ 1897 đến 1945?"
        )
        plan = replace(
            build_query_plan(question),
            evolution_periods=((1897, 1918), (1919, 1939), (1939, 1945)),
        )
        queries = self.service._expanded_recovery_queries(
            question,
            {
                "scope": plan.inferred_scope,
                "date_range": plan.explicit_date_range,
            },
            plan,
        )
        combined = "\n".join(queries)

        for start, end in plan.evolution_periods:
            self.assertIn(f"{start}-{end}", combined)

    def test_f9_selection_reserves_evidence_across_dynamic_periods(self) -> None:
        plan = build_query_plan(
            "Chính sách cai trị của Pháp thay đổi như thế nào "
            "từ 1897 đến 1945?",
        )
        plan = replace(
            plan,
            evolution_periods=(
                (1897, 1918),
                (1919, 1939),
                (1939, 1945),
            ),
            evolution_period_evidence_ids=(
                ("period-1897-1918",),
                ("period-1919-1939",),
                ("period-1939-1945",),
            ),
        )
        candidates = [
            {
                "id": f"period-{start}-{end}",
                "sourceTitle": "Lịch sử Việt Nam",
                "heading": f"Giai đoạn {start}-{end}",
                "text": (
                    f"Năm {start}, chính sách cai trị chuyển biến; "
                    f"đến năm {end} xuất hiện một bước ngoặt mới."
                ),
                "yearStart": start,
                "yearEnd": end,
                "score": 0.7,
            }
            for start, end in plan.evolution_periods
        ]
        selected = self.service._select_with_evolution_coverage(
            candidates[:2],
            candidates,
            plan,
            9,
        )

        self.assertEqual(
            {item["id"] for item in selected},
            {item["id"] for item in candidates},
        )

    def test_f9_dynamic_period_plan_detects_subject_lifetime_mismatch(self) -> None:
        question = (
            "Chiến tranh đặc biệt thay đổi như thế nào "
            "từ 1961 đến 1970?"
        )
        plan = replace(
            build_query_plan(question),
            evolution_subject_type="military_strategy",
        )
        candidates = [
            {
                "id": "special-war",
                "sourceTitle": "Lịch sử Việt Nam 1954-1965",
                "heading": "Chiến tranh đặc biệt",
                "text": (
                    "Từ năm 1961 đến năm 1965, Mỹ thực hiện chiến lược "
                    "Chiến tranh đặc biệt và chiến lược này bị thất bại."
                ),
                "yearStart": 1961,
                "yearEnd": 1965,
                "score": 0.9,
            },
            {
                "id": "local-war",
                "sourceTitle": "Lịch sử Việt Nam 1965-1975",
                "heading": "Chuyển sang Chiến tranh cục bộ",
                "text": (
                    "Từ năm 1965 đến năm 1970, Mỹ chuyển sang chiến lược "
                    "Chiến tranh cục bộ, đưa quân Mỹ trực tiếp tham chiến."
                ),
                "yearStart": 1965,
                "yearEnd": 1968,
                "score": 0.88,
            },
        ]
        self.service.client = SimpleNamespace(chat=SimpleNamespace(
            completions=_FakeCompletions({
                "periods": [
                    {
                        "start": 1962,
                        "end": 1965,
                        "label": "Chiến tranh đặc biệt",
                        "dominant_state": "Quân đội Sài Gòn giữ vai trò chính",
                        "boundary_cause": "Chiến lược thất bại",
                        "boundary_year": 1961,
                        "boundary_actor": "Mỹ",
                        "boundary_action": "thực hiện Chiến tranh đặc biệt",
                        "boundary_evidence_id": "special-war",
                        "boundary_excerpt": (
                            "Từ năm 1961 đến năm 1965, Mỹ thực hiện "
                            "chiến lược Chiến tranh đặc biệt"
                        ),
                        "evidence_ids": ["special-war"],
                    },
                    {
                        "start": 1965,
                        "end": 1969,
                        "label": "Bị thay thế bởi chiến lược kế tiếp",
                        "dominant_state": "Mỹ trực tiếp đưa quân tham chiến",
                        "boundary_cause": "Mỹ chuyển sang Chiến tranh cục bộ",
                        # Boundary may describe the transition out of this
                        # period and therefore align with its end.
                        "boundary_year": 1970,
                        # Planner proposals may omit semantic roles. The
                        # locked excerpt remains sufficient for the runtime
                        # semantic-role verifier to recover them.
                        "boundary_actor": "",
                        "boundary_action": "",
                        "boundary_evidence_id": "local-war",
                        "boundary_excerpt": (
                            "Từ năm 1965, Mỹ chuyển sang chiến lược "
                            "Chiến tranh cục bộ"
                        ),
                        "evidence_ids": ["local-war"],
                    },
                ],
                "subject_lifetime_start": 1961,
                "subject_lifetime_end": 1965,
                "range_mismatch": True,
                "range_resolution": "explain_transition_to_successor",
                "confidence": 0.93,
            }),
        ))

        refined, used_model = self.service._derive_evolution_period_plan(
            question,
            plan,
            candidates,
            analyze_query(question),
        )

        self.assertTrue(used_model)
        self.assertEqual(refined.evolution_subject_lifetime, (1961, 1965))
        self.assertTrue(refined.evolution_range_mismatch)
        self.assertEqual(
            refined.evolution_range_resolution,
            "explain_transition_to_successor",
        )
        self.assertEqual(
            refined.evolution_periods,
            ((1961, 1965), (1965, 1970)),
        )

    def test_f9_timeline_windows_cover_range_without_becoming_periods(self) -> None:
        windows = self.service._evolution_timeline_windows((1897, 1945))

        self.assertGreaterEqual(len(windows), 4)
        self.assertLessEqual(len(windows), 9)
        self.assertEqual(windows[0][0], 1897)
        self.assertEqual(windows[-1][1], 1945)
        for previous, current in zip(windows, windows[1:], strict=False):
            self.assertEqual(current[0], previous[1] + 1)

    def test_f9_timeline_coverage_uses_repository_window_recall(self) -> None:
        question = (
            "Chính sách cai trị của Pháp thay đổi như thế nào "
            "từ 1897 đến 1945?"
        )
        plan = build_query_plan(question)
        captured: dict = {}

        class Repository:
            @staticmethod
            def retrieve_timeline_windows(
                query: str,
                windows: list[tuple[int, int]],
                *,
                per_window: int,
            ) -> list[dict]:
                captured.update({
                    "query": query,
                    "windows": windows,
                    "per_window": per_window,
                })
                return [{"id": "timeline-evidence"}]

        self.service.repository = Repository()
        recalled = self.service._retrieve_evolution_timeline_coverage(
            question,
            plan,
        )

        self.assertEqual(recalled, [{"id": "timeline-evidence"}])
        self.assertIn(question, captured["query"])
        self.assertEqual(captured["per_window"], 4)
        self.assertEqual(captured["windows"][0][0], 1897)
        self.assertEqual(captured["windows"][-1][1], 1945)

    def test_f9_evidence_cards_distinguish_heading_year_from_metadata(self) -> None:
        cards = self.service._evolution_evidence_cards(
            [{
                "id": "popular-front",
                "heading": "Giai đoạn 1936-1939",
                "text": (
                    "Chính phủ mới nới lỏng một số biện pháp kiểm soát "
                    "đối với phong trào dân chủ."
                ),
                "yearStart": 1930,
                "yearEnd": 1945,
                "score": 0.9,
            }],
            requested_range=(1897, 1945),
            subject="chính sách cai trị của Pháp",
        )

        self.assertEqual(cards[0]["heading_years"], [1936, 1939])
        self.assertEqual(cards[0]["text_years"], [])
        self.assertTrue(
            cards[0]["temporal_evidence_levels"]["section_heading_year"]
        )
        self.assertFalse(
            cards[0]["temporal_evidence_levels"]["metadata_range_only"]
        )

    def test_concrete_evidence_score_rewards_named_quantified_examples(self) -> None:
        generic = {
            "text": "Chính sách thuộc địa nhằm phục vụ lợi ích của chính quốc.",
        }
        concrete = {
            "text": (
                "Năm 1930, diện tích đồn điền đạt 909.300 ha; "
                "Pháp đồng thời đặt ra các loại thuế và xây dựng đồn điền."
            ),
            "relatedEntities": ["Pháp", "Việt Nam"],
        }

        self.assertLess(self.service._concrete_evidence_score(generic), 0.32)
        self.assertGreaterEqual(
            self.service._concrete_evidence_score(concrete),
            0.7,
        )

    def test_facet_selection_prefers_specific_example_over_generic_summary(self) -> None:
        candidates = [
            {
                "id": "generic-economy",
                "text": "Pháp khai thác kinh tế và đặt ra thuế khóa.",
                "score": 0.95,
            },
            {
                "id": "specific-economy",
                "text": (
                    "Năm 1897 Pháp thành lập Sở Thuế quan và Độc quyền; "
                    "muối, rượu và thuốc phiện trở thành ba khoản độc quyền."
                ),
                "relatedEntities": ["Sở Thuế quan và Độc quyền"],
                "score": 0.72,
            },
        ]

        selected = self.service._select_with_facet_coverage(
            candidates[:1],
            candidates,
            ["economic_taxation"],
            1,
        )

        self.assertEqual(selected[0]["id"], "specific-economy")

    def test_open_question_adds_generic_detail_recovery_query(self) -> None:
        question = "Hãy giải thích Cách mạng tháng Tám"
        plan = build_query_plan(question)
        queries = self.service._expanded_recovery_queries(
            question,
            {"scope": "", "date_range": ""},
            plan,
        )

        self.assertGreaterEqual(len(queries), 2)
        self.assertTrue(any("dẫn chứng cụ thể" in query for query in queries))

    def test_merge_keeps_top_result_from_focused_query_above_threshold(self) -> None:
        broad_items = [
            {"id": f"generic-{index}", "score": 0.9 - index * 0.01}
            for index in range(8)
        ]
        retrievals = [
            {"items": broad_items, "queryTerms": [], "channels": ["vector"]}
            for _ in range(5)
        ]
        retrievals.append({
            "items": [
                {"id": "focused-example", "score": 0.91},
                *broad_items,
            ],
            "queryTerms": [],
            "channels": ["lexical"],
        })

        merged = self.service._merge_retrievals(retrievals, 12)
        by_id = {item["id"]: item for item in merged["items"]}

        self.assertIn("focused-example", by_id)
        self.assertGreaterEqual(by_id["focused-example"]["score"], 0.62)

    def test_facet_selection_reserves_evidence_for_broad_question(self) -> None:
        candidates = [
            {"id": "timeline", "text": "Chính sách thay đổi qua các giai đoạn.", "score": 1.0},
            {"id": "politics", "text": "Bộ máy cai trị và chính quyền thuộc địa.", "score": 0.8},
            {"id": "economy", "text": "Khai thác kinh tế, ruộng đất và thuế khóa.", "score": 0.8},
            {"id": "culture", "text": "Chính sách giáo dục, trường học và báo chí.", "score": 0.8},
            {"id": "society", "text": "Xã hội phân hóa thành nhiều giai cấp.", "score": 0.8},
            {"id": "effects", "text": "Mục đích thống trị và hậu quả phụ thuộc.", "score": 0.8},
        ]
        plan = build_query_plan("Các chính sách thuộc địa của Pháp là gì?")

        selected = self.service._select_with_facet_coverage(
            candidates[:1],
            candidates,
            plan.required_facets,
            6,
        )
        coverage = self.service._facet_evidence_map(
            plan.required_facets,
            selected,
        )

        self.assertTrue(all(coverage.values()))
        self.assertEqual(
            self.service._estimate_evidence_coverage(
                "Các chính sách thuộc địa của Pháp là gì?",
                selected,
                plan.required_facets,
            ),
            1.0,
        )

    def test_phase_question_expands_progress_result_and_leadership_queries(self) -> None:
        question = "Chiến dịch Điện Biên Phủ diễn ra qua những đợt nào?"
        plan = build_query_plan(question)

        queries = self.service._expanded_recovery_queries(
            question,
            {"scope": "", "date_range": ""},
            plan,
        )
        normalized = " ".join(queries).casefold()

        self.assertGreaterEqual(len(queries), 4)
        self.assertIn("mốc thời gian bắt đầu kết thúc", normalized)
        self.assertIn("kết quả ngày kết thúc", normalized)
        self.assertIn("nhân vật", normalized)

    def test_phase_grounding_rejects_another_campaign_and_brief_mentions(self) -> None:
        signals = analyze_query(
            "Chiến dịch Điện Biên Phủ diễn ra qua những đợt nào?",
        )
        candidates = [
            {
                "id": "ho-chi-minh-campaign",
                "sourceId": "1975",
                "pageStart": 511,
                "heading": "Chiến dịch Hồ Chí Minh",
                "text": "Chiến dịch diễn ra qua nhiều giai đoạn năm 1975.",
            },
            {
                "id": "comparison",
                "sourceId": "1975",
                "pageStart": 446,
                "heading": "",
                "text": (
                    "Lực lượng hậu cần lớn gấp nhiều lần so với "
                    "Chiến dịch Điện Biên Phủ."
                ),
            },
        ]

        grounded = self.service._ground_event_phase_results(
            candidates,
            candidates,
            signals,
            8,
        )

        self.assertEqual(grounded, [])

    def test_phase_grounding_keeps_dated_phase_and_adjacent_result(self) -> None:
        signals = analyze_query(
            "Chiến dịch Điện Biên Phủ diễn ra qua những đợt nào?",
        )
        candidates = [
            {
                "id": "phase",
                "sourceId": "dbp",
                "pageStart": 100,
                "heading": "Chiến dịch Điện Biên Phủ",
                "text": (
                    "Đợt 1 diễn ra từ 13-3 đến 17-3-1954, "
                    "quân ta tiến công Him Lam."
                ),
            },
            {
                "id": "result",
                "sourceId": "dbp",
                "pageStart": 102,
                "heading": "",
                "text": (
                    "Ngày 7-5-1954, sở chỉ huy đối phương bị chiếm "
                    "và chiến dịch kết thúc thắng lợi."
                ),
            },
            {
                "id": "unrelated",
                "sourceId": "other",
                "pageStart": 20,
                "heading": "Một sự kiện khác",
                "text": "Đợt 1 bắt đầu ngày 1-1-1960.",
            },
        ]

        grounded = self.service._ground_event_phase_results(
            candidates,
            candidates,
            signals,
            8,
        )

        self.assertEqual(
            [item["id"] for item in grounded],
            ["phase", "result"],
        )

    def test_phase_best_effort_omits_precise_ungrounded_details(self) -> None:
        captured: dict = {}

        class Repository:
            @staticmethod
            def get_active_prompt():
                return {}

        class Completions:
            @staticmethod
            def create(**kwargs):
                captured.update(kwargs)
                return SimpleNamespace(choices=[
                    SimpleNamespace(message=SimpleNamespace(
                        content="Chiến dịch diễn ra qua ba đợt chính.",
                    )),
                ])

        self.service.repository = Repository()
        self.service.client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions()),
        )

        answer = self.service._answer_event_phases_best_effort(
            "Chiến dịch Điện Biên Phủ diễn ra qua những đợt nào?",
        )

        self.assertEqual(answer, "Chiến dịch diễn ra qua ba đợt chính.")
        system_message = captured["messages"][0]["content"]
        self.assertIn("CHẾ ĐỘ BEST-EFFORT KHÔNG NGUỒN", system_message)
        self.assertIn(
            "Không bổ sung ngày tháng chính xác, tên nhân vật",
            system_message,
        )
        self.assertIn(
            "vẫn phải nêu các địa danh, cứ điểm hoặc khu vực",
            system_message,
        )
        self.assertIn("Him Lam, Độc Lập và Bản Kéo", system_message)

    def test_evidence_fallback_tolerates_vietnamese_typo(self) -> None:
        selected = self.service._evidence_fallback(
            "Chiến dịch Điện Biên Phủ có ai là người lạnh đão và kết quả ra sao?",
            self._candidates(),
            4,
        )
        self.assertEqual({item["id"] for item in selected}, {"dbp-1", "dbp-2"})

    def test_evidence_coverage_reflects_requested_facets(self) -> None:
        coverage = self.service._estimate_evidence_coverage(
            "Chiến dịch Điện Biên Phủ diễn ra thế nào, lực lượng gì, "
            "ai lãnh đạo và kết quả ra sao?",
            self._candidates()[:2],
        )
        self.assertEqual(coverage, 0.5)

    def test_rerank_keeps_direct_person_evidence(self) -> None:
        direct_evidence = {
            "id": "nguyen-thi-dinh-259",
            "sourceId": "1945-1965",
            "sourceTitle": "Lịch sử Việt Nam 1945-1965",
            "pageStart": 259,
            "heading": "Phong trào Đồng Khởi",
            "text": (
                "Một trong những tấm gương tiêu biểu là bà Nguyễn Thị Định, "
                "lúc đó là Phó Bí thư Tỉnh ủy Bến Tre."
            ),
            "score": 0.65,
            "directEvidence": True,
            "entityMatch": True,
            "matchedPhrases": ["Nguyễn Thị Định"],
            "channels": ["exact"],
            "trustLevel": "official",
        }
        generic = {
            "id": "generic-person",
            "sourceId": "1945-1965",
            "sourceTitle": "Lịch sử Việt Nam 1945-1965",
            "pageStart": 100,
            "heading": "Kháng chiến chống Mỹ",
            "text": "Nhiều nhân vật lịch sử đã tham gia phong trào cách mạng.",
            "score": 0.95,
        }
        self.service.client = SimpleNamespace(chat=SimpleNamespace(
            completions=_FakeCompletions({
                "answerable": True,
                "evidence_coverage": 0.2,
                "selected_ids": ["generic-person"],
            }),
        ))

        selected, analysis = self.service._rerank(
            "Nguyễn Thị Định là ai?",
            [generic, direct_evidence],
            analyze_query("Nguyễn Thị Định là ai?"),
        )

        self.assertIn("nguyen-thi-dinh-259", {item["id"] for item in selected})
        self.assertTrue(analysis["answerable"])

    def test_identity_confidence_is_capped_without_direct_evidence(self) -> None:
        confidence, factors = self.service._calibrate_confidence(
            analyze_query("Nguyễn Thị Định là ai?"),
            [{
                "id": "generic",
                "sourceId": "source",
                "pageStart": 1,
                "score": 1.0,
                "trustLevel": "official",
            }],
            {"evidence_coverage": 1.0},
        )

        self.assertLessEqual(confidence, 0.45)
        self.assertEqual(factors["direct_evidence"], 0.0)
        self.assertEqual(factors["entity_match"], 0.0)

    def test_direct_identity_evidence_raises_confidence(self) -> None:
        confidence, factors = self.service._calibrate_confidence(
            analyze_query("Nguyễn Thị Định là ai?"),
            [{
                "id": "direct",
                "sourceId": "source",
                "pageStart": 259,
                "score": 1.0,
                "trustLevel": "official",
                "directEvidence": True,
                "entityMatch": True,
            }],
            {"evidence_coverage": 1.0},
        )

        self.assertGreater(confidence, 0.45)
        self.assertEqual(factors["direct_evidence"], 1.0)
        self.assertEqual(factors["entity_match"], 1.0)

    def test_confidence_falls_back_when_model_coverage_is_null(self) -> None:
        confidence, factors = self.service._calibrate_confidence(
            analyze_query("Nguyễn Thị Định là ai?"),
            [{
                "id": "direct",
                "sourceId": "1945-1965",
                "pageStart": 259,
                "score": 0.95,
                "text": "Nguyễn Thị Định là Phó Bí thư Tỉnh ủy Bến Tre.",
                "trustLevel": "official",
                "directEvidence": True,
                "entityMatch": True,
            }],
            {"evidence_coverage": None},
            "Nguyễn Thị Định là ai?",
        )

        self.assertGreater(confidence, 0.75)
        self.assertEqual(factors["evidence_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
