import json
import unittest
from types import SimpleNamespace

from src.api.models import ChatRequest
from src.api.rag.contracts import EvidenceItem, GeneratedAnswer, UnifiedPlan
from src.api.rag.evidence import EvidenceTools, ModelEvidenceSelector
from src.api.rag.generator import (
    memory_anchor_instruction,
    remove_redundant_opening_heading,
)
from src.api.rag.pipeline import GroundedRagPipeline
from src.api.rag.verifier import should_verify


class _ChatCompletions:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))
        )])


class _Embeddings:
    def create(self, *, input, **_kwargs):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2]) for _ in input])


class _Client:
    def __init__(self, payloads):
        self.chat = SimpleNamespace(completions=_ChatCompletions(payloads))
        self.embeddings = _Embeddings()


class _FailingChatCompletions:
    @staticmethod
    def create(**_kwargs):
        raise RuntimeError("selector unavailable")


class _FailingClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FailingChatCompletions())


class _Published:
    @staticmethod
    def search(_query, _limit):
        return []


class _Repository:
    published = _Published()

    @staticmethod
    def retrieve_hybrid(_query, _embedding, _limit, **_kwargs):
        return {
            "items": [{
                "id": "e1",
                "text": "Liên quân Pháp - Tây Ban Nha nổ súng vào Đà Nẵng ngày 1/9/1858.",
                "heading": "Mở đầu cuộc xâm lược",
                "sourceId": "book-1",
                "sourceTitle": "Lịch sử Việt Nam",
                "pageStart": 20,
                "score": 0.98,
                "channels": ["fulltext"],
                "relatedEntities": ["Đà Nẵng"],
            }],
            "candidateCount": 1,
            "queryTerms": ["xâm lược", "Đà Nẵng"],
            "exactMatchCount": 1,
            "channels": ["fulltext"],
        }


class _QueryDiversityRepository:
    published = _Published()

    @staticmethod
    def retrieve_hybrid(query, _embedding, limit, **_kwargs):
        configured = {
            "câu hỏi chung": ("broad", 0.99),
            "tìm các nhóm": ("groups", 0.75),
            "tìm vai trò đóng góp": ("roles", 0.25),
        }
        prefix, base_score = configured.get(
            query,
            (("groups", 0.75) if "nhóm" in query else ("roles", 0.25)),
        )
        return {
            "items": [
                {
                    "id": f"{prefix}-{index}",
                    "text": f"{prefix} evidence {index}",
                    "score": base_score - index * 0.001,
                }
                for index in range(limit)
            ],
            "candidateCount": limit,
            "queryTerms": [prefix],
            "exactMatchCount": 0,
            "channels": ["vector"],
        }

    @staticmethod
    def retrieve_subject_evidence(_subjects, *, limit=240):
        return [
            {
                "id": "subject-broad",
                "text": "Sự kiện có sự tham gia rộng rãi của toàn dân.",
                "score": 0.9,
                "channels": ["subject_scan"],
            },
            {
                "id": "subject-groups",
                "text": (
                    "Trong sự kiện, công nhân giữ vai trò tiên phong; "
                    "nông dân là lực lượng cơ bản và trí thức tham gia tổ chức."
                ),
                "score": 0.5,
                "channels": ["subject_scan"],
            },
        ][:limit]


class _TemporalRepository:
    published = _Published()

    @staticmethod
    def retrieve_hybrid(_query, _embedding, _limit, **_kwargs):
        return {
            "items": [
                {
                    "id": "correct-period",
                    "text": "Hỏa lực trong Chiến tranh cục bộ.",
                    "yearStart": 1965,
                    "yearEnd": 1968,
                    "score": 0.8,
                },
                {
                    "id": "wrong-period",
                    "text": "Tên lửa TOW trong mùa hè 1972.",
                    "yearStart": 1972,
                    "yearEnd": 1972,
                    "score": 0.99,
                },
                {
                    "id": "undated",
                    "text": "Đoạn không có metadata năm.",
                    "score": 0.7,
                },
            ],
            "candidateCount": 3,
            "queryTerms": ["chiến tranh cục bộ"],
            "exactMatchCount": 0,
            "channels": ["vector"],
        }


PROMPTS = {
    "system": "Trợ lý lịch sử",
    "query_normalization": "Sửa lỗi rõ nghĩa",
    "answer_planning": "Lập kế hoạch linh hoạt",
    "presentation": "Trả lời tự nhiên",
    "output_contract": "Không ghi [1]",
}


class UnifiedRagArchitectureTest(unittest.TestCase):
    def test_temporal_gate_removes_only_provably_disjoint_evidence(self):
        tools = EvidenceTools(_Client([]), "embedding", _TemporalRepository())
        plan = UnifiedPlan.from_dict({
            "standaloneQuestion": "Hỏa lực Mỹ trong Chiến tranh cục bộ?",
            "yearStart": 1965,
            "yearEnd": 1968,
            "requirements": [{
                "id": "firepower", "question": "Hỏa lực thay đổi thế nào?",
                "answerType": "timeline", "required": True,
            }],
        }, "question")

        items, diagnostics = tools.retrieve(plan)

        self.assertEqual(
            {item.id for item in items},
            {"correct-period", "undated"},
        )
        self.assertEqual(diagnostics["temporal_filtered_count"], 1)

    def test_multi_window_requirement_needs_dated_evidence_from_each_side(self):
        client = _Client([{
            "requirementEvidence": [{
                "requirementId": "change", "evidenceIds": ["before"],
            }],
            "missingRequirements": [],
            "conflicts": [],
        }])
        selector = ModelEvidenceSelector(client, "model")
        plan = UnifiedPlan.from_dict({
            "standaloneQuestion": "Điểm thay đổi trước và sau năm 1968?",
            "timeWindows": [
                {"id": "before", "start": 1965, "end": 1968, "label": "Trước"},
                {"id": "after", "start": 1969, "end": 1972, "label": "Sau"},
            ],
            "requirements": [{
                "id": "change", "question": "Điểm thay đổi?",
                "answerType": "comparison", "required": True,
                "timeWindowIds": ["before", "after"],
            }],
        }, "question")
        selected = selector.select(plan, [
            EvidenceItem("before", "Giai đoạn trước", year_start=1965, year_end=1968),
        ])

        self.assertIn("change", selected.missing_requirements)

    def test_memory_anchor_guidance_adapts_to_answer_mode(self):
        direct = UnifiedPlan.from_dict({
            "standaloneQuestion": "Ai lãnh đạo?",
            "mode": "direct_fact",
            "requirements": [
                {"id": "leader", "question": "Ai lãnh đạo?", "required": True},
            ],
        }, "Ai lãnh đạo?")
        explanation = UnifiedPlan.from_dict({
            "standaloneQuestion": "Vì sao sự kiện thành công?",
            "mode": "explanatory_rag",
            "requirements": [
                {"id": "cause", "question": "Vì sao?", "required": True},
            ],
        }, "Vì sao sự kiện thành công?")
        comparison = UnifiedPlan.from_dict({
            "standaloneQuestion": "So sánh hai sự kiện",
            "mode": "comparison",
            "requirements": [
                {"id": "compare", "question": "Giống và khác?", "required": True},
            ],
        }, "So sánh hai sự kiện")
        participants = UnifiedPlan.from_dict({
            "standaloneQuestion": "Những lực lượng nào tham gia?",
            "mode": "explanatory_rag",
            "requirements": [
                {
                    "id": "groups", "question": "Có những nhóm nào?",
                    "answerType": "participant_groups", "required": True,
                },
                {
                    "id": "roles", "question": "Mỗi nhóm có vai trò gì?",
                    "answerType": "participant_roles", "required": True,
                },
            ],
        }, "Những lực lượng nào tham gia?")

        self.assertIn("không thêm mục từ khóa", memory_anchor_instruction(direct))
        self.assertIn("Từ khóa ghi nhớ", memory_anchor_instruction(explanation))
        self.assertIn("lãnh đạo đúng đắn, kịp thời", memory_anchor_instruction(explanation))
        self.assertIn("Từ khóa đối chiếu", memory_anchor_instruction(comparison))
        self.assertIn("vai trò hoặc hình thức tham gia", memory_anchor_instruction(participants))
        self.assertIn("Phân biệt cơ quan lãnh đạo", memory_anchor_instruction(participants))

    def test_redundant_question_heading_is_removed_but_content_heading_is_kept(self):
        question = "Vì sao Cách mạng tháng Tám năm 1945 thắng lợi nhanh chóng?"
        repeated = (
            "## Vì sao Cách mạng tháng Tám 1945 thắng lợi nhanh chóng?\n\n"
            "Có hai nhóm nguyên nhân chính."
        )
        shortened = (
            "## Vì sao thắng lợi diễn ra nhanh chóng?\n\n"
            "Có hai nhóm nguyên nhân chính."
        )
        useful = "## Nền tảng chuẩn bị lâu dài\n\nCó sự chuẩn bị về lực lượng."

        self.assertEqual(
            remove_redundant_opening_heading(repeated, question),
            "Có hai nhóm nguyên nhân chính.",
        )
        self.assertEqual(
            remove_redundant_opening_heading(shortened, question),
            "Có hai nhóm nguyên nhân chính.",
        )
        self.assertEqual(remove_redundant_opening_heading(useful, question), useful)

    def test_plan_keeps_only_general_modes_and_atomic_requirements(self):
        plan = UnifiedPlan.from_dict({
            "standaloneQuestion": "Điện Biên Phủ diễn ra thế nào và ai chỉ huy?",
            "mode": "graph_multihop",
            "subject": "Chiến dịch Điện Biên Phủ",
            "requirements": [
                {"id": "progress", "question": "Diễn biến thế nào?", "answerType": "progress", "required": True},
                {"id": "leader", "question": "Ai chỉ huy?", "answerType": "person_role", "required": True},
            ],
            "retrievalQueries": ["diễn biến Điện Biên Phủ", "chỉ huy Điện Biên Phủ"],
            "outputStyle": "sections",
        }, "original")
        self.assertEqual(plan.mode, "graph_multihop")
        self.assertEqual([item.id for item in plan.requirements], ["progress", "leader"])

    def test_evidence_tools_merge_duplicate_candidates(self):
        tools = EvidenceTools(_Client([]), "embedding", _Repository())
        plan = UnifiedPlan.from_dict({
            "standaloneQuestion": "Pháp bắt đầu xâm lược ở đâu?",
            "requirements": [{"id": "r1", "question": "ở đâu", "answerType": "location", "required": True}],
            "retrievalQueries": ["Pháp mở đầu xâm lược Việt Nam"],
        }, "question")
        items, diagnostics = tools.retrieve(plan)
        self.assertEqual([item.id for item in items], ["e1"])
        self.assertGreaterEqual(diagnostics["candidate_count"], 1)

    def test_evidence_tools_reserve_candidates_for_each_semantic_query(self):
        tools = EvidenceTools(_Client([]), "embedding", _QueryDiversityRepository())
        plan = UnifiedPlan.from_dict({
            "standaloneQuestion": "câu hỏi chung",
            "requirements": [
                {
                    "id": "groups", "question": "Các nhóm?",
                    "answerType": "participant_groups", "required": True,
                },
                {
                    "id": "roles", "question": "Vai trò?",
                    "answerType": "participant_roles", "required": True,
                },
            ],
            "retrievalQueries": ["tìm các nhóm", "tìm vai trò đóng góp"],
        }, "câu hỏi chung")

        items, _ = tools.retrieve(plan, candidate_k=18)

        self.assertLessEqual(len(items), 48)
        self.assertTrue(any(item.id.startswith("groups-") for item in items))
        self.assertTrue(any(item.id.startswith("roles-") for item in items))

    def test_participant_inventory_reserves_detailed_same_subject_evidence(self):
        tools = EvidenceTools(_Client([]), "embedding", _QueryDiversityRepository())
        plan = UnifiedPlan.from_dict({
            "standaloneQuestion": "câu hỏi chung",
            "subject": "Sự kiện",
            "requirements": [
                {
                    "id": "groups", "question": "Các nhóm?",
                    "answerType": "participant_groups", "required": True,
                },
                {
                    "id": "roles", "question": "Vai trò?",
                    "answerType": "participant_roles", "required": True,
                },
            ],
            "retrievalQueries": [
                "tìm các nhóm công nhân nông dân trí thức",
                "tìm vai trò đóng góp",
            ],
        }, "câu hỏi chung")

        items, diagnostics = tools.retrieve(plan, candidate_k=18)

        self.assertIn("subject-groups", [item.id for item in items])
        self.assertIn("subject_detail_scan", diagnostics["channels"])

    def test_selector_only_accepts_candidate_ids(self):
        client = _Client([{
            "requirementEvidence": [{"requirementId": "r1", "evidenceIds": ["made-up", "e1"]}],
            "missingRequirements": [],
            "conflicts": [],
        }])
        selector = ModelEvidenceSelector(client, "model")
        plan = UnifiedPlan.from_dict({
            "standaloneQuestion": "câu hỏi",
            "requirements": [{"id": "r1", "question": "câu hỏi", "answerType": "fact", "required": True}],
            "retrievalQueries": ["câu hỏi"],
        }, "câu hỏi")
        selected = selector.select(plan, [EvidenceItem("e1", "bằng chứng")])
        self.assertEqual(selected.evidence_ids, ("e1",))

    def test_selector_receives_participant_completeness_contract(self):
        client = _Client([{
            "requirementEvidence": [
                {"requirementId": "groups", "evidenceIds": ["e1"]},
            ],
            "missingRequirements": ["roles"],
            "conflicts": [],
        }])
        selector = ModelEvidenceSelector(client, "model")
        plan = UnifiedPlan.from_dict({
            "standaloneQuestion": "Những lực lượng nào tham gia?",
            "requirements": [
                {
                    "id": "groups", "question": "Những nhóm nào?",
                    "answerType": "participant_groups", "required": True,
                },
                {
                    "id": "roles", "question": "Vai trò từng nhóm?",
                    "answerType": "participant_roles", "required": True,
                },
            ],
        }, "Những lực lượng nào tham gia?")

        selector.select(plan, [EvidenceItem("e1", "Toàn dân tham gia")])
        prompt = client.chat.completions.calls[0]["messages"][1]["content"]

        self.assertIn("toàn dân", prompt)
        self.assertIn("hình thức tham gia hoặc đóng góp", prompt)

    def test_selector_failure_fails_closed_instead_of_reusing_top_chunks(self):
        selector = ModelEvidenceSelector(_FailingClient(), "model")
        plan = UnifiedPlan.from_dict({
            "standaloneQuestion": "Ai chỉ huy và diễn ra khi nào?",
            "requirements": [
                {"id": "leader", "question": "Ai chỉ huy?", "answerType": "person_role", "required": True},
                {"id": "time", "question": "Khi nào?", "answerType": "date", "required": True},
            ],
            "retrievalQueries": ["chỉ huy và thời gian"],
        }, "Ai chỉ huy và diễn ra khi nào?")
        selected = selector.select(plan, [EvidenceItem("e1", "một đoạn có điểm cao")])
        self.assertEqual(selected.evidence_ids, ())
        self.assertEqual(set(selected.missing_requirements), {"leader", "time"})
        self.assertTrue(selected.conflicts)

    def test_high_risk_question_runs_single_verifier(self):
        plan = UnifiedPlan.from_dict({
            "standaloneQuestion": "Khi nào?",
            "requirements": [{"id": "r1", "question": "Khi nào?", "answerType": "date", "required": True}],
            "retrievalQueries": ["khi nào"],
            "requiresVerification": True,
        }, "Khi nào?")
        generated = SimpleNamespace(answer="Năm 1945", claims=())
        self.assertTrue(should_verify(plan, generated, ()))

    def test_generated_claims_expose_requirement_coverage(self):
        generated = GeneratedAnswer(
            answer="Một điều kiện trực tiếp và một nền tảng lâu dài.",
            claims=(
                {
                    "text": "Điều kiện trực tiếp.",
                    "evidenceIds": ["e1"],
                    "requirementIds": ["trigger"],
                    "risk": "low",
                },
                {
                    "text": "Nền tảng lâu dài.",
                    "evidenceIds": ["e2"],
                    "requirementIds": ["foundation", "foundation"],
                    "risk": "low",
                },
            ),
        )
        self.assertEqual(
            generated.covered_requirement_ids,
            ("trigger", "foundation"),
        )

    def test_selector_computes_missing_required_requirement(self):
        client = _Client([{
            "requirementEvidence": [
                {"requirementId": "foundation", "evidenceIds": ["e1"]},
            ],
            "missingRequirements": [],
            "conflicts": [],
        }])
        selector = ModelEvidenceSelector(client, "model")
        plan = UnifiedPlan.from_dict({
            "standaloneQuestion": "Vì sao sự kiện thành công nhanh chóng?",
            "requirements": [
                {
                    "id": "trigger",
                    "question": "Điều kiện trực tiếp nào mở ra thời cơ?",
                    "answerType": "immediate_causal_condition",
                    "required": True,
                },
                {
                    "id": "foundation",
                    "question": "Nền tảng nào giúp tận dụng thời cơ?",
                    "answerType": "enabling_causes",
                    "required": True,
                },
            ],
            "retrievalQueries": ["nguyên nhân trực tiếp và nền tảng"],
        }, "Vì sao sự kiện thành công nhanh chóng?")
        selected = selector.select(plan, [EvidenceItem("e1", "Nền tảng lâu dài")])
        self.assertEqual(selected.missing_requirements, ("trigger",))

    def test_pipeline_preserves_api_and_uses_corrected_verified_answer(self):
        client = _Client([
            {
                "standaloneQuestion": "Thực dân Pháp bắt đầu xâm lược Việt Nam ở đâu, khi nào?",
                "mode": "direct_fact", "subject": "cuộc xâm lược Việt Nam của Pháp",
                "requirements": [
                    {"id": "place", "question": "Bắt đầu ở đâu?", "answerType": "location", "required": True},
                    {"id": "time", "question": "Bắt đầu khi nào?", "answerType": "date", "required": True},
                ],
                "retrievalQueries": ["Pháp Tây Ban Nha nổ súng mở đầu xâm lược Việt Nam"],
                "scope": "Việt Nam", "dateRange": "1858", "outputStyle": "direct",
                "requiresVerification": True, "ambiguity": "",
            },
            {
                "requirementEvidence": [
                    {"requirementId": "place", "evidenceIds": ["e1"]},
                    {"requirementId": "time", "evidenceIds": ["e1"]},
                ],
                "missingRequirements": [], "conflicts": [],
            },
            {
                "answer": "Pháp bắt đầu xâm lược Việt Nam tại Đà Nẵng ngày 1/9/1858.",
                "claims": [{
                    "text": "Pháp nổ súng tại Đà Nẵng ngày 1/9/1858.",
                    "evidenceIds": ["e1"],
                    "requirementIds": ["place", "time"],
                    "risk": "high",
                }],
            },
            {
                "status": "supported",
                "correctedAnswer": "",
                "unsupportedClaims": [], "missingRequirementIds": [],
                "requirementCoverage": [
                    {"requirementId": "place", "status": "covered", "note": ""},
                    {"requirementId": "time", "status": "covered", "note": ""},
                ],
                "notes": [],
            },
        ])
        logs = []
        pipeline = GroundedRagPipeline(
            client=client, model="model", embedding_model="embedding",
            repository=_Repository(),
            settings=SimpleNamespace(ai_retrieval_candidate_k=12, ai_retrieval_top_k=8),
            prompt_provider=lambda: PROMPTS,
            query_logger=logs.append,
        )
        response = pipeline.answer(ChatRequest(question="Pháp bắt đầu xâm lược ở đâu, khi nào?"), user_id="u1")
        self.assertIn("Đà Nẵng", response.answer)
        self.assertEqual(response.retrieval.claim_verification_status, "supported")
        self.assertEqual(response.citations[0].chunk_id, "e1")
        self.assertEqual(logs[0]["userId"], "u1")


if __name__ == "__main__":
    unittest.main()
