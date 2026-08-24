import unittest
from types import SimpleNamespace

from src.api.rag_service import (
    ANSWER_PLANNING_INSTRUCTION,
    RagService,
    effective_answer_planning_instruction,
)


class RagServiceFacadeTest(unittest.TestCase):
    def test_legacy_versioned_prompt_is_retired(self) -> None:
        actual = effective_answer_planning_instruction(
            "ADAPTIVE_COMPARISON_PLANNING_V22\nquy tắc cũ"
        )
        self.assertEqual(actual, ANSWER_PLANNING_INSTRUCTION)

    def test_admin_generic_prompt_is_preserved(self) -> None:
        actual = effective_answer_planning_instruction(
            "Ưu tiên trả lời trực tiếp rồi mới giải thích."
        )
        self.assertEqual(actual, "Ưu tiên trả lời trực tiếp rồi mới giải thích.")

    def test_log_failure_does_not_discard_answer(self) -> None:
        class FailingRepository:
            @staticmethod
            def log_query(_payload):
                raise RuntimeError("offline")

        service = RagService.__new__(RagService)
        service.repository = FailingRepository()
        with self.assertLogs("src.api.rag_service", level="ERROR"):
            service._log_query({"question": "test"})

    def test_answer_is_only_a_pipeline_delegation(self) -> None:
        expected = object()
        service = RagService.__new__(RagService)
        service.pipeline = SimpleNamespace(
            answer=lambda request, user_id="": (expected, request, user_id)
        )
        request = object()
        self.assertEqual(
            service.answer(request, "u1"),
            (expected, request, "u1"),
        )


if __name__ == "__main__":
    unittest.main()
