"""Public RAG service.

This module deliberately contains only the stable API used by FastAPI and the
PDF indexer.  The answer workflow lives in :mod:`src.api.rag.pipeline`.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

from .config import get_settings
from .graph_extraction import TokenUsage
from .models import ChatRequest, ChatResponse
from .rag.pipeline import GroundedRagPipeline


logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """
Bạn là trợ lý học Lịch sử Việt Nam. Hãy trả lời chính xác, sáng rõ và có chọn
lọc từ kho tri thức đã được quản trị viên kiểm duyệt. Không dùng trí nhớ riêng
của mô hình để bù cho dữ kiện lịch sử không có trong bằng chứng truy xuất.
""".strip()

QUERY_NORMALIZATION_INSTRUCTION = """
Hiểu câu hỏi theo ngữ cảnh, âm thầm sửa lỗi gõ, từ lặp và viết tắt khi cách
hiểu đã rõ. Không tự đổi tên riêng, mốc thời gian hoặc phạm vi lịch sử.
""".strip()

ANSWER_PLANNING_INSTRUCTION = """
Lập kế hoạch theo đúng yêu cầu thực tế của câu hỏi. Tách câu hỏi nhiều ý thành
các yêu cầu nguyên tử; chọn bố cục phù hợp với nội dung thay vì áp một checklist
cố định. Câu hỏi trực tiếp cần trả lời giá trị chính trước; câu hỏi so sánh,
diễn biến hoặc thay đổi theo thời gian mới dùng bảng hay các chặng khi hữu ích.
Chỉ chọn chi tiết có bằng chứng và ưu tiên mốc thời gian, nhân vật, địa điểm,
hành động hoặc kết quả cụ thể có giá trị giải thích.
Với câu hỏi “vì sao”, tách các vai trò nhân quả thiết yếu theo chính nội dung
câu hỏi thay vì gom thành một mục nguyên nhân chung. Khi phù hợp, cần phân biệt
điều kiện hoặc tác nhân trực tiếp với nền tảng chuẩn bị lâu dài; không thay một
dữ kiện nhân quả cụ thể bằng nhãn mơ hồ như “thời cơ thuận lợi”.
Với câu hỏi số nhiều về lực lượng, nhóm, thành phần hay các bên tham gia, coi
đây là một kiểm kê có cấu trúc: nhận diện các nhóm khác nhau và, trừ khi người
dùng chỉ yêu cầu kể tên thật ngắn, giải thích vai trò hoặc hình thức tham gia
có căn cứ của từng nhóm. Phân biệt cơ quan lãnh đạo/tổ chức với các lực lượng
tham gia; không để nhãn bao quát “toàn dân” thay thế các nhóm cụ thể và không
áp sẵn một danh sách giai cấp hay quân sự cho mọi sự kiện. Model được dùng
tri thức của mình để đề xuất các nhóm có khả năng liên quan làm giả thuyết tìm
kiếm, nhưng mọi nhóm xuất hiện trong đáp án vẫn phải được corpus xác nhận.
""".strip()

PRESENTATION_INSTRUCTION = """
Trình bày tự nhiên, dễ đọc trên điện thoại. Không lặp lại nguyên câu hỏi làm
tiêu đề, không nói về quy trình truy xuất hoặc “nguồn tài liệu” trong câu trả
lời. Dùng Markdown, danh sách hay bảng chỉ khi chúng làm nội dung rõ hơn. Với
câu trả lời giải thích có nhiều phần, cô đọng cuối mỗi phần bằng 2–5 từ khóa
ghi nhớ rút ra từ chính nội dung đã có căn cứ; không áp dụng máy móc cho câu tra
cứu trực tiếp hoặc tạo thêm dữ kiện trong dòng từ khóa.
""".strip()

OUTPUT_CONTRACT = """
Không chèn ký hiệu nguồn như [1], [2] vào câu trả lời và không tự tạo mục nguồn
tham khảo. Backend trả citations riêng cho Web Admin; App chỉ hiển thị đáp án.
""".strip()


def effective_answer_planning_instruction(value: str) -> str:
    """Return the exact planning instruction used by the runtime.

    Old installations may still store prompt revisions containing a growing
    list of version-specific validator tags.  Those revisions are intentionally
    retired: the unified model planner now owns semantic planning.
    """
    text = str(value or "").strip()
    legacy_markers = (
        "ADAPTIVE_COMPARISON_PLANNING_",
        "EVOLUTION_PLANNING_F9_",
        "MULTI_FACET_EVENT_PLANNING_",
        "RECOVERABLE_BOUNDARY_",
        "SEMANTIC_RELATION_",
        "REQUIREMENT_PRIMARY_",
    )
    if not text or any(marker in text for marker in legacy_markers):
        return ANSWER_PLANNING_INSTRUCTION
    return text


class RagService:
    """Compatibility facade for chat and PDF embedding operations."""

    def __init__(self, repository: Any) -> None:
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.repository = repository
        self.pipeline = GroundedRagPipeline(
            client=self.client,
            model=self.settings.openai_chat_model,
            embedding_model=self.settings.openai_embedding_model,
            repository=repository,
            settings=self.settings,
            prompt_provider=self._active_prompt_layers,
            query_logger=self._log_query,
        )

    def _embed_with_usage(
        self,
        texts: list[str],
    ) -> tuple[list[list[float]], TokenUsage]:
        response = self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=texts,
        )
        usage = getattr(response, "usage", None)
        return (
            [item.embedding for item in response.data],
            TokenUsage(
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=0,
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            ),
        )

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        embeddings, _ = self._embed_with_usage(texts)
        return embeddings

    def embed_chunks_with_usage(
        self,
        texts: list[str],
    ) -> tuple[list[list[float]], TokenUsage]:
        return self._embed_with_usage(texts)

    def _active_prompt_layers(self) -> dict[str, str]:
        try:
            config = self.repository.get_active_prompt() or {}
        except Exception:
            logger.exception("Không thể đọc prompt đang kích hoạt; dùng mặc định")
            config = {}
        return {
            "system": str(config.get("systemPrompt") or DEFAULT_SYSTEM_PROMPT),
            "query_normalization": str(
                config.get("queryNormalizationInstruction")
                or QUERY_NORMALIZATION_INSTRUCTION
            ),
            "answer_planning": effective_answer_planning_instruction(
                str(config.get("answerPlanningInstruction") or "")
            ),
            "presentation": str(
                config.get("presentationInstruction")
                or PRESENTATION_INSTRUCTION
            ),
            "output_contract": str(
                config.get("outputContract") or OUTPUT_CONTRACT
            ),
        }

    def _log_query(self, payload: dict[str, Any]) -> None:
        try:
            self.repository.log_query(payload)
        except Exception:
            logger.exception("Không thể ghi AIQueryLog vào Neo4j")

    def answer(self, request: ChatRequest, user_id: str = "") -> ChatResponse:
        return self.pipeline.answer(request, user_id=user_id)
