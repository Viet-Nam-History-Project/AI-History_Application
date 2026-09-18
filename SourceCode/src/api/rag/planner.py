"""One model-owned query understanding and planning step."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from .contracts import UnifiedPlan


PLAN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "unified_history_query_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "standaloneQuestion": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": [
                        "direct_fact", "explanatory_rag", "comparison",
                        "timeline_evolution", "graph_multihop",
                        "reliability_and_conversation",
                    ],
                },
                "subject": {"type": "string"},
                "requirements": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string"},
                            "question": {"type": "string"},
                            "answerType": {"type": "string"},
                            "required": {"type": "boolean"},
                            "timeWindowIds": {
                                "type": "array", "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "id", "question", "answerType", "required",
                            "timeWindowIds",
                        ],
                    },
                },
                "retrievalQueries": {
                    "type": "array", "minItems": 1, "maxItems": 12,
                    "items": {"type": "string"},
                },
                "scope": {"type": "string"},
                "dateRange": {"type": "string"},
                "yearStart": {"type": ["integer", "null"]},
                "yearEnd": {"type": ["integer", "null"]},
                "timeWindows": {
                    "type": "array", "maxItems": 6,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string"},
                            "start": {"type": "integer"},
                            "end": {"type": "integer"},
                            "label": {"type": "string"},
                        },
                        "required": ["id", "start", "end", "label"],
                    },
                },
                "outputStyle": {
                    "type": "string",
                    "enum": ["direct", "sections", "table", "timeline"],
                },
                "requiresVerification": {"type": "boolean"},
                "ambiguity": {"type": "string"},
            },
            "required": [
                "standaloneQuestion", "mode", "subject", "requirements",
                "retrievalQueries", "scope", "dateRange", "outputStyle",
                "yearStart", "yearEnd", "timeWindows", "requiresVerification",
                "ambiguity",
            ],
        },
    },
}


def _history_text(messages: Sequence[Any]) -> str:
    return "\n".join(
        f"{getattr(item, 'role', '')}: {getattr(item, 'content', '')}"
        for item in messages[-6:]
    )


class UnifiedQueryPlanner:
    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def plan(
        self,
        question: str,
        messages: Sequence[Any],
        instructions: dict[str, str],
    ) -> UnifiedPlan:
        prompt = f"""
Bạn là planner duy nhất của một hệ thống RAG Lịch sử Việt Nam.

NHIỆM VỤ:
- Viết lại câu hỏi thành câu độc lập, sửa lỗi gõ rõ nghĩa nhưng không thêm dữ kiện.
- Chọn đúng một mode tổng quát.
- Tách mọi yêu cầu người dùng thực sự hỏi thành yêu cầu nguyên tử. Không dùng
  checklist cố định: “ở đâu, khi nào” là hai yêu cầu; câu hỏi nhiều ý có thể có
  nhiều yêu cầu; câu hỏi đơn giản chỉ có một.
- Với câu hỏi giải thích “vì sao”, không gom tất cả thành một yêu cầu chung.
  Hãy xác định tập nguyên nhân tối thiểu nhưng đủ sức giải thích kết quả theo
  chính ngữ nghĩa của câu hỏi. Tách riêng những vai trò nhân quả không thể thay
  thế cho nhau, nhất là điều kiện/kích hoạt trực tiếp với nền tảng chuẩn bị lâu
  dài khi cả hai đều cần để giải thích tốc độ, thời điểm hoặc khả năng thành
  công. Đây là nguyên tắc phân rã nhân quả tổng quát, không phải checklist cố
  định; không tạo khía cạnh không liên quan.
- Câu hỏi hỏi số nhiều về “những lực lượng/nhóm/thành phần/bên nào tham gia”
  là yêu cầu kiểm kê có cấu trúc, không phải một scalar fact. Trừ khi người
  dùng yêu cầu chỉ kể tên thật ngắn, hãy dùng `explanatory_rag`, `sections` và
  tách tối thiểu: (1) nhận diện các nhóm/thành phần khác nhau với answerType
  `participant_groups`; (2) vai trò, hình thức tham gia hoặc đóng góp của các
  nhóm được corpus hỗ trợ với answerType `participant_roles`. Nếu cần để tránh
  nhập nhằng, có thể tách thêm cơ quan lãnh đạo/tổ chức khỏi lực lượng trực
  tiếp tham gia, nhưng không tự đoán tên nhóm trong requirement.
- Với dạng kiểm kê lực lượng, tạo truy vấn bổ trợ khác nhau cho: tên các nhóm;
  vai trò/hình thức tham gia/đóng góp; cơ cấu lãnh đạo hoặc tổ chức nếu câu hỏi
  cần phân biệt. Trong `retrievalQueries`, phải tận dụng tri thức của model để
  nêu các nhóm hoặc cách phân loại CÓ KHẢ NĂNG liên quan đến đúng chủ thể như
  các giả thuyết truy xuất; có thể chia thành nhiều truy vấn theo nhóm xã hội,
  chính trị, quân sự hoặc địa phương khi phù hợp với sự kiện. Không áp đủ mọi
  loại cho mọi câu hỏi. Không dồn tất cả tên giả thuyết vào một truy vấn dài:
  tạo các truy vấn ngắn, mỗi truy vấn gồm chủ thể + một nhóm (hoặc một cặp gần
  nghĩa) + quan hệ vai trò/đóng góp; dùng nhiều truy vấn độc lập để các nhóm
  thiểu số không bị nhóm bao quát lấn át. Không đưa giả thuyết vào requirement
  hay đáp án nếu corpus không xác nhận. Không xem cụm bao quát như “toàn dân”
  là thay thế cho các nhóm cụ thể khi kho dữ liệu có thể chứa phân loại chi tiết.
- Chỉ ghi `ambiguity` khi thực sự cần người dùng chọn giữa các cách hiểu làm
  thay đổi chủ thể hoặc phạm vi. Việc người dùng không nói rõ muốn đáp án ngắn
  hay có thêm vai trò không phải là nhập nhằng nếu vẫn có thể trả lời hữu ích.
- Mỗi requirement phải tự mô tả rõ quan hệ cần chứng minh, không dùng nhãn mơ
  hồ như “nguyên nhân khác”. Đánh dấu `required=true` khi bỏ khía cạnh đó sẽ
  làm câu trả lời sai trọng tâm hoặc thiếu lời giải thích thiết yếu.
- Tạo truy vấn tìm kiếm giàu ngữ nghĩa cho từng yêu cầu. Có thể thêm từ đồng
  nghĩa, tên gọi khác, quan hệ cần tìm và giả thuyết truy xuất dựa
  trên tri thức của model. Giả thuyết chỉ dùng để tìm: tuyệt đối không
  coi nó là đáp án nếu corpus không xác nhận.
- Chuẩn hóa phạm vi thời gian thành `yearStart`, `yearEnd`; nếu câu hỏi
  không giới hạn thời gian thì trả null. Với so sánh/tiến trình qua
  nhiều giai đoạn, tạo `timeWindows` riêng và gắn `timeWindowIds` cho
  requirement. Yêu cầu về tiếp nối/thay đổi phải gắn tất cả cửa
  sổ cần đối chiếu.
- `requiresVerification=true` khi câu trả lời dự kiến chứa tên người, mốc ngày,
  số liệu, quan hệ nhân quả, so sánh, nhiều nguồn hoặc khả năng xung đột.
- Chọn outputStyle theo bản chất câu hỏi, không theo một taxonomy nội dung cứng.

Chỉ dẫn chuẩn hóa của Admin:
{instructions['query_normalization']}

Chỉ dẫn lập kế hoạch của Admin:
{instructions['answer_planning']}

Hội thoại gần đây:
{_history_text(messages) or '(không có)'}

Câu hỏi hiện tại: {question}
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format=PLAN_SCHEMA,
                messages=[
                    {"role": "system", "content": "Chỉ lập kế hoạch JSON; không trả lời lịch sử."},
                    {"role": "user", "content": prompt},
                ],
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            return UnifiedPlan.from_dict(parsed, question)
        except Exception:
            # The fallback is intentionally only a transport fallback, not a
            # second semantic planner made from historical regex rules.
            verify = bool(re.search(r"\d|ai |người|khi nào|bao nhiêu", question, re.I))
            return UnifiedPlan.from_dict({
                "standaloneQuestion": question,
                "mode": "explanatory_rag",
                "requirements": [{
                    "id": "r1", "question": question,
                    "answerType": "fact", "required": True,
                }],
                "retrievalQueries": [question],
                "requiresVerification": verify,
                "yearStart": None,
                "yearEnd": None,
                "timeWindows": [],
            }, question)
