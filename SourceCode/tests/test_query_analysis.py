import unittest

from src.api.query_analysis import analyze_query, build_query_plan, normalize_query


class QueryAnalysisTest(unittest.TestCase):
    def test_corrects_contextual_cach_to_cac(self) -> None:
        result = normalize_query(
            "Nói cho tôi biết cách chính sách thuộc địa của Pháp",
        )

        self.assertEqual(
            result.normalized_question,
            "Nói cho tôi biết các chính sách thuộc địa của Pháp",
        )
        self.assertTrue(result.changed)
        self.assertEqual(result.corrections[0].original, "cách chính sách")

    def test_removes_consecutive_duplicate_words(self) -> None:
        result = normalize_query("Ai ai lãnh đạo chiến dịch Điện Biên Phủ?")

        self.assertEqual(
            result.normalized_question,
            "Ai lãnh đạo chiến dịch Điện Biên Phủ?",
        )
        self.assertIn("lặp", result.corrections[0].reason)

    def test_expands_safe_history_abbreviations(self) -> None:
        result = normalize_query("ĐBP và CMT8 có ý nghĩa gì?")

        self.assertEqual(
            result.normalized_question,
            "Điện Biên Phủ và Cách mạng tháng Tám có ý nghĩa gì?",
        )
        self.assertEqual(len(result.corrections), 2)

    def test_marks_ambiguous_abbreviation_without_rewriting_it(self) -> None:
        result = normalize_query("HCM có vai trò gì?")

        self.assertEqual(result.normalized_question, "HCM có vai trò gì?")
        self.assertTrue(result.ambiguous_terms)

    def test_preserves_proper_names_and_years(self) -> None:
        question = "Nguyễn Thị Định hoạt động thế nào năm 1960?"

        self.assertEqual(normalize_query(question).normalized_question, question)

    def test_plans_colonial_policy_question_as_faceted_overview(self) -> None:
        normalized = normalize_query(
            "Nói cho tôi biết cách chính sách thuộc địa của Pháp",
        ).normalized_question
        plan = build_query_plan(normalized)

        self.assertEqual(plan.intent, "overview")
        self.assertEqual(plan.inferred_scope, "Việt Nam")
        self.assertEqual(plan.explicit_date_range, "")
        self.assertEqual(plan.answer_structure, "facets_first_then_timeline")
        self.assertIn("political_administrative", plan.required_facets)
        self.assertIn("economic_taxation", plan.required_facets)
        self.assertIn("culture_education", plan.required_facets)
        self.assertIn("social_transformation", plan.required_facets)
        self.assertIn("objectives_consequences", plan.required_facets)
        self.assertFalse(plan.allow_inferred_date_recovery)

    def test_distinguishes_policy_evolution_from_overview(self) -> None:
        plan = build_query_plan(
            "Chính sách thuộc địa của Pháp thay đổi như thế nào từ 1897 đến 1945?",
        )

        self.assertEqual(plan.intent, "historical_evolution")
        self.assertEqual(plan.evolution_intent, "evolution_over_time")
        self.assertEqual(plan.evolution_domain, "political_administration")
        self.assertIn("historical_evolution", plan.required_facets)
        self.assertIn("continuity_change", plan.required_facets)
        self.assertIn("governance_methods", plan.required_facets)
        self.assertIn("wartime_mobilization", plan.required_facets)
        self.assertEqual(plan.explicit_date_range, "1897–1945")
        self.assertEqual(
            plan.answer_structure,
            "evolution_evolution_over_time",
        )
        # Query parser chỉ xác định subject/domain/range. Phân kỳ F9 phải
        # được tạo sau retrieval từ bằng chứng bước ngoặt, không hard-code.
        self.assertEqual(plan.evolution_periods, ())

    def test_same_date_range_does_not_predefine_same_f9_periods(self) -> None:
        strategy = build_query_plan(
            "Chiến lược chiến tranh của Mỹ thay đổi như thế nào "
            "từ 1965 đến 1976?",
        )
        country = build_query_plan(
            "Việt Nam thay đổi như thế nào từ 1965 đến 1976?",
        )

        self.assertEqual(strategy.intent, "historical_evolution")
        self.assertEqual(country.intent, "historical_evolution")
        self.assertEqual(strategy.evolution_periods, ())
        self.assertEqual(country.evolution_periods, ())

    def test_classifies_continuity_change_question_without_comparison(self) -> None:
        plan = build_query_plan(
            "Những yếu tố nào trong chính sách cai trị của Pháp thay đổi "
            "và những yếu tố nào vẫn được duy trì từ 1897 đến 1945?",
        )

        self.assertEqual(plan.intent, "historical_evolution")
        self.assertEqual(plan.evolution_intent, "continuity_and_change")
        self.assertEqual(
            plan.answer_structure,
            "evolution_continuity_and_change",
        )
        self.assertIn("continuity_change", plan.required_facets)

    def test_classifies_turning_point_and_change_cause_subtypes(self) -> None:
        turning_points = build_query_plan(
            "Những mốc nào đánh dấu bước ngoặt trong chính sách cai trị "
            "của Pháp từ 1897 đến 1945?",
        )
        cause = build_query_plan(
            "Vì sao chính sách cai trị của Pháp thay đổi trong "
            "giai đoạn 1936 đến 1939?",
        )

        self.assertEqual(turning_points.evolution_intent, "turning_points")
        self.assertEqual(cause.evolution_intent, "cause_of_change")
        self.assertIn("cause", cause.required_facets)

    def test_plans_event_phase_question_with_dates_people_and_result(self) -> None:
        plan = build_query_plan(
            "Chiến dịch Điện Biên Phủ diễn ra qua những đợt nào?",
        )

        self.assertEqual(plan.intent, "event_phases")
        self.assertEqual(plan.required_facets, ("progress", "result"))
        self.assertEqual(plan.optional_facets, ("leadership",))
        self.assertEqual(
            plan.answer_structure,
            "phases_with_dates_then_result",
        )

    def test_plans_result_and_significance_as_required_facets(self) -> None:
        plan = build_query_plan(
            "Chiến thắng Điện Biên Phủ có kết quả và ý nghĩa gì?",
        )

        self.assertEqual(plan.intent, "event_outcome")
        self.assertEqual(plan.subject, "Chiến thắng Điện Biên Phủ")
        self.assertEqual(plan.required_facets, ("result", "significance"))
        self.assertEqual(plan.answer_structure, "result_then_significance")

    def test_plans_two_colonial_exploitations_as_comparison_matrix(self) -> None:
        plan = build_query_plan(
            "So sánh hai cuộc khai thác thuộc địa của Pháp",
        )

        self.assertEqual(plan.intent, "comparison")
        self.assertEqual(
            plan.answer_structure,
            "comparison_matrix_then_similarities_differences",
        )
        self.assertEqual(
            plan.comparison_subjects,
            (
                "Cuộc khai thác thuộc địa lần thứ nhất của Pháp",
                "Cuộc khai thác thuộc địa lần thứ hai của Pháp",
            ),
        )
        self.assertEqual(
            plan.comparison_date_ranges,
            ((1897, 1918), (1919, 1930)),
        )
        for facet in (
            "comparison_context",
            "organizer",
            "objectives_consequences",
            "scale_investment",
            "agriculture",
            "industry",
            "commerce",
        ):
            self.assertIn(facet, plan.required_facets)
        self.assertLessEqual(len(plan.required_facets), 7)

    def test_extracts_explicit_comparison_subjects(self) -> None:
        plan = build_query_plan(
            "So sánh Cách mạng tháng Tám và Chiến dịch Điện Biên Phủ",
        )

        self.assertEqual(plan.intent, "comparison")
        self.assertEqual(
            plan.comparison_subjects,
            ("Cách mạng tháng Tám", "Chiến dịch Điện Biên Phủ"),
        )

    def test_military_strategy_comparison_uses_relevant_facets_only(self) -> None:
        plan = build_query_plan(
            "So sánh chiến lược “Chiến tranh đặc biệt” và "
            "“Chiến tranh cục bộ” của Mỹ ở miền Nam Việt Nam",
        )

        self.assertEqual(plan.intent, "comparison")
        self.assertEqual(plan.comparison_domain, "military_strategy")
        self.assertEqual(
            plan.comparison_subjects,
            ("Chiến tranh đặc biệt", "Chiến tranh cục bộ"),
        )
        self.assertEqual(
            plan.comparison_date_ranges,
            ((1961, 1965), (1965, 1968)),
        )
        for facet in (
            "comparison_context",
            "forces",
            "geographic_scope",
            "strategy_methods",
            "representative_events",
        ):
            self.assertIn(facet, plan.required_facets)
        for irrelevant_facet in (
            "agriculture",
            "industry",
            "commerce",
            "transport",
        ):
            self.assertNotIn(irrelevant_facet, plan.required_facets)

    def test_comparison_main_difference_gets_direct_answer_structure(self) -> None:
        plan = build_query_plan(
            "Điểm khác biệt cơ bản giữa Chiến tranh đặc biệt và "
            "Chiến tranh cục bộ là gì?",
        )

        self.assertEqual(plan.comparison_intent, "main_difference")
        self.assertEqual(plan.answer_structure, "comparison_main_difference")
        self.assertFalse(plan.include_similarities)
        self.assertTrue(plan.include_differences)
        self.assertFalse(plan.include_judgement)
        self.assertEqual(
            plan.comparison_subjects,
            ("Chiến tranh đặc biệt", "Chiến tranh cục bộ"),
        )

    def test_explicit_comparison_facets_are_not_broadened(self) -> None:
        plan = build_query_plan(
            "So sánh Chiến tranh đặc biệt và Chiến tranh cục bộ "
            "về lực lượng và phạm vi",
        )

        self.assertEqual(plan.comparison_intent, "focused_facets")
        self.assertEqual(
            plan.explicit_facets,
            ("forces", "geographic_scope"),
        )
        self.assertEqual(plan.required_facets, plan.explicit_facets)
        self.assertEqual(plan.answer_structure, "comparison_selected_facets")

    def test_comparison_cause_intent_retrieves_only_causal_facets(self) -> None:
        plan = build_query_plan(
            "So sánh nguyên nhân thắng lợi của Cách mạng tháng Tám "
            "và kháng chiến chống Pháp",
        )

        self.assertEqual(plan.comparison_intent, "cause_comparison")
        self.assertEqual(plan.explicit_facets, ("cause",))
        self.assertEqual(plan.required_facets, ("cause",))
        self.assertEqual(plan.answer_structure, "comparison_focused_causes")
        self.assertEqual(
            plan.comparison_subjects,
            ("Cách mạng tháng Tám", "kháng chiến chống Pháp"),
        )

    def test_diplomatic_agreements_use_agreement_domain(self) -> None:
        plan = build_query_plan(
            "So sánh Hiệp định Giơnevơ và Hiệp định Paris",
        )

        self.assertEqual(plan.comparison_domain, "diplomatic_agreement")
        self.assertIn("agreement_content", plan.required_facets)
        self.assertIn("participants", plan.required_facets)
        self.assertNotIn("agriculture", plan.required_facets)

    def test_extracts_person_from_identity_question(self) -> None:
        signals = analyze_query("Nguyễn Thị Định là ai?")

        self.assertEqual(signals.primary_entity, "Nguyễn Thị Định")
        self.assertEqual(signals.exact_phrases, ("Nguyễn Thị Định",))
        self.assertTrue(signals.is_identity_query)

    def test_removes_polite_request_before_person_name(self) -> None:
        signals = analyze_query("Hãy cho tôi biết Nguyễn Thị Định là ai")

        self.assertEqual(signals.primary_entity, "Nguyễn Thị Định")
        self.assertTrue(signals.is_identity_query)

    def test_extracts_event_without_requiring_a_year(self) -> None:
        signals = analyze_query(
            "Chiến dịch Điện Biên Phủ diễn ra như thế nào, gồm lực lượng gì?",
        )

        self.assertEqual(signals.primary_entity, "Chiến dịch Điện Biên Phủ")
        self.assertIn("Chiến dịch Điện Biên Phủ", signals.exact_phrases)
        self.assertFalse(signals.is_identity_query)

    def test_does_not_treat_generic_question_as_named_subject(self) -> None:
        signals = analyze_query("Ai là nhân vật lịch sử?")

        self.assertEqual(signals.primary_entity, "")
        self.assertEqual(signals.exact_phrases, ())

    def test_extracts_event_after_leadership_question(self) -> None:
        signals = analyze_query("Ai lãnh đạo Chiến dịch Điện Biên Phủ?")

        self.assertEqual(
            signals.primary_entity,
            "Chiến dịch Điện Biên Phủ",
        )

    def test_extracts_event_from_significance_question(self) -> None:
        signals = analyze_query("Ý nghĩa của Cách mạng tháng Tám là gì?")

        self.assertEqual(signals.primary_entity, "Cách mạng tháng Tám")

    def test_extracts_alias_subject_before_auxiliary(self) -> None:
        signals = analyze_query("Bác Hồ đã làm gì năm 1925?")

        self.assertEqual(signals.primary_entity, "Bác Hồ")


if __name__ == "__main__":
    unittest.main()
