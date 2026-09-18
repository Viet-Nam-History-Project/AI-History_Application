"""Grounded answer generation with a compact claim contract."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from .contracts import (
    EvidenceItem,
    EvidenceSelection,
    GeneratedAnswer,
    UnifiedPlan,
    is_participant_inventory_plan,
)


GENERATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "grounded_history_answer_v3",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer": {"type": "string"},
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "text": {"type": "string"},
                            "evidenceIds": {"type": "array", "items": {"type": "string"}},
                            "requirementIds": {"type": "array", "items": {"type": "string"}},
                            "risk": {"type": "string", "enum": ["low", "high"]},
                        },
                        "required": ["text", "evidenceIds", "requirementIds", "risk"],
                    },
                },
            },
            "required": ["answer", "claims"],
        },
    },
}


def memory_anchor_instruction(plan: UnifiedPlan) -> str:
    """Return adaptive learning-oriented presentation guidance.

    Memory anchors are presentation aids, not additional historical claims.
    They are disabled for direct lookups and adapted to the answer shape so
    tables and timelines do not become repetitive.
    """
    if is_participant_inventory_plan(plan):
        return (
            "Đây là câu hỏi kiểm kê lực lượng/thành phần. Mở đầu bằng nhận định "
            "khái quát về phạm vi tham gia, sau đó trình bày mỗi nhóm có căn cứ "
            "theo mẫu linh hoạt `**Tên nhóm:** vai trò hoặc hình thức tham gia`. "
            "Phân biệt cơ quan lãnh đạo/tổ chức với lực lượng trực tiếp tham gia; "
            "không biến tên tổ chức bao trùm và các nhóm thành các mục đồng cấp "
            "nếu quan hệ của chúng khác nhau. Không bịa nhóm để đủ mẫu. Cuối đáp "
            "án có thể thêm đúng một dòng `**Ghi nhớ:** ...` cô đọng bản chất lực "
            "lượng và vai trò nổi bật từ các claim đã được hỗ trợ."
        )
    if plan.mode == "direct_fact" and len(plan.requirements) <= 2:
        return (
            "Đây là câu tra cứu trực tiếp: không thêm mục từ khóa hoặc ghi nhớ "
            "chỉ để kéo dài đáp án."
        )
    if plan.mode == "comparison":
        return (
            "Sau phần so sánh, thêm đúng một dòng `**Từ khóa đối chiếu:** ...` "
            "gồm 2–5 cặp hoặc cụm ngắn thể hiện khác biệt cốt lõi."
        )
    if plan.mode == "timeline_evolution":
        return (
            "Sau mỗi chặng lớn, có thể thêm một dòng `**Mạch ghi nhớ:** ...` "
            "gồm 2–5 cụm ngắn cho thấy trạng thái và hướng chuyển đổi."
        )
    return (
        "Nếu câu trả lời có nhiều phần giải thích, cuối MỖI phần lớn hãy thêm "
        "một dòng `**Từ khóa ghi nhớ:** ...` gồm 2–5 cụm từ ngắn, cô đọng kết "
        "luận hoặc cơ chế quan trọng nhất của chính phần đó. Ví dụ về hình thức: "
        "`**Từ khóa ghi nhớ:** lãnh đạo đúng đắn – chuẩn bị lâu dài – chớp thời "
        "cơ kịp thời`. Từ khóa phải được suy ra từ các claim đã có bằng chứng, "
        "không đưa dữ kiện mới và không lặp nguyên câu mô tả. Với phần giải "
        "thích nguyên nhân, ưu tiên cụm thể hiện vai trò hoặc kết luận học thuật "
        "như `lãnh đạo đúng đắn, kịp thời`, không chỉ liệt kê tên riêng, địa danh "
        "hay danh từ đã xuất hiện. Nếu đáp án chỉ có một ý ngắn thì không cần "
        "tạo dòng này."
    )


def remove_redundant_opening_heading(answer: str, question: str) -> str:
    """Strip only a leading Markdown heading that merely repeats the question."""
    lines = answer.splitlines()
    first_index = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first_index is None:
        return answer.strip()
    match = re.match(r"^#{1,6}\s+(.+?)\s*$", lines[first_index].strip())
    if not match:
        return answer.strip()

    normalize = lambda value: re.sub(r"[^\w\s]", " ", value.casefold(), flags=re.UNICODE)
    heading = " ".join(normalize(match.group(1)).split())
    normalized_question = " ".join(normalize(question).split())
    heading_tokens = set(heading.split())
    question_tokens = set(normalized_question.split())
    token_overlap = (
        len(heading_tokens & question_tokens) / max(1, min(len(heading_tokens), len(question_tokens)))
    )
    similarity = SequenceMatcher(None, heading, normalized_question).ratio()
    # A shortened heading often omits the subject but still repeats the
    # requested relation (for example "Vì sao thắng lợi nhanh chóng?"). High
    # token containment is enough in that case; useful content headings have
    # little overlap with the wording of the question.
    if token_overlap < 0.75 or (similarity < 0.55 and len(heading_tokens & question_tokens) < 4):
        return answer.strip()

    remaining = lines[:first_index] + lines[first_index + 1:]
    while remaining and not remaining[0].strip():
        remaining.pop(0)
    return "\n".join(remaining).strip()


class GroundedAnswerModel:
    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def generate(
        self,
        plan: UnifiedPlan,
        evidence: list[EvidenceItem],
        selection: EvidenceSelection,
        instructions: dict[str, str],
    ) -> GeneratedAnswer:
        allowed = set(selection.evidence_ids)
        selected = [item for item in evidence if item.id in allowed]
        requirement_payload = [
            {
                "id": item.id,
                "question": item.question,
                "answerType": item.answer_type,
                "evidenceIds": selection.requirement_evidence.get(item.id, []),
                "timeWindowIds": list(item.time_window_ids),
            }
            for item in plan.requirements
        ]
        learning_presentation = memory_anchor_instruction(plan)
        prompt = f"""
{instructions['presentation']}

{instructions['output_contract']}

Trả lời câu hỏi bằng trí thông minh ngôn ngữ của bạn, nhưng mọi dữ kiện lịch
sử phải nằm trong bằng chứng đã chọn. Tổng hợp và diễn đạt tự nhiên; không chép
nguyên đoạn. Không nhắc “nguồn”, “bằng chứng”, “kho dữ liệu”, “validator” hay
phần còn thiếu trong câu trả lời cho người dùng. Phần thiếu được backend hiển
thị riêng cho Admin.

Không tạo tiêu đề lặp lại câu hỏi. Với câu tra cứu trực tiếp, mở đầu bằng đáp
án và chỉ thêm một câu bối cảnh hữu ích. Với câu nhiều ý, trả lời đủ từng yêu
cầu. Với so sánh hoặc tiến trình, tự chọn tiêu chí/chặng phù hợp từ bằng chứng;
không điền các mục rỗng chỉ để đủ mẫu.

Hỗ trợ người học ghi nhớ theo đúng độ dài và dạng câu hỏi:
{learning_presentation}
Các dòng ghi nhớ chỉ là cách cô đọng claim đã được hỗ trợ; chúng không phải nơi
bổ sung kiến thức mới. Không đồng thời thêm một kết luận ghi nhớ toàn bài nếu
các phần đã có dòng từ khóa và việc lặp lại không tạo thêm giá trị.

Mỗi requirement `required=true` đã có evidenceIds là một nghĩa vụ nội dung:
answer phải thể hiện rõ ít nhất một nhận định trả lời requirement đó. Với câu
hỏi nguyên nhân, không được thay một nguyên nhân hoặc điều kiện trực tiếp cụ
thể bằng nhãn khái quát như “thời cơ chín muồi”. Hãy nêu ngắn gọn điều gì đã
xảy ra và vì sao nó tạo điều kiện cho kết quả, nếu bằng chứng có chi tiết đó.

Với yêu cầu `participant_groups`, không dừng ở các nhãn “toàn dân”, “quần
chúng” hoặc “lực lượng vũ trang” nếu evidenceIds còn cho biết các nhóm cụ thể.
Với `participant_roles`, không chỉ kể tên: nói ngắn gọn vai trò, hình thức tham
gia hoặc đóng góp mà bằng chứng xác nhận. Chỉ nhóm các lực lượng theo quan hệ
thực sự có trong evidence; không áp một danh sách giai cấp hay quân sự cố định.

Mỗi dữ kiện chỉ nên xuất hiện một lần ở phần phù hợp, không lặp lại giữa
diễn biến, lực lượng và kết quả. Chỉ nêu mức chi tiết cần cho yêu cầu.
Tên người, ngày/giờ và số liệu quá cụ thể là dữ kiện rủi ro cao:
không thêm độ chính xác không cần thiết từ một đoạn đơn lẻ; ưu tiên
bản ghi curated, nguồn official hoặc chi tiết được nhiều đoạn xác nhận.
Không gán thuật ngữ, chính sách, văn kiện, vũ khí hay kết quả của
giai đoạn sau cho giai đoạn trước. Với câu tiếp nối/thay đổi,
chỉ kết luận sau khi evidence bao phủ tất cả time window được gắn.

Mỗi dữ kiện có thể kiểm chứng phải xuất hiện trong `claims`, gắn evidenceId và
requirementIds mà claim thực sự trả lời.
Không dùng evidenceId ngoài danh sách cho phép.

Câu hỏi: {plan.standalone_question}
Mode: {plan.mode}
Phong cách: {plan.output_style}
Phạm vi năm: {plan.year_start} đến {plan.year_end}
Time windows: {json.dumps([
    {'id': item.id, 'start': item.start, 'end': item.end, 'label': item.label}
    for item in plan.time_windows
], ensure_ascii=False)}
Các yêu cầu: {json.dumps(requirement_payload, ensure_ascii=False)}
Yêu cầu chưa đủ dữ liệu (không viết vào answer): {json.dumps(selection.missing_requirements, ensure_ascii=False)}
Bằng chứng: {json.dumps([item.prompt_payload(max_chars=2400) for item in selected], ensure_ascii=False)}
""".strip()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format=GENERATION_SCHEMA,
            messages=[
                {"role": "system", "content": instructions["system"]},
                {"role": "user", "content": prompt},
            ],
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        answer = remove_redundant_opening_heading(
            str(parsed.get("answer") or "").strip(),
            plan.standalone_question,
        )
        claims: list[dict[str, Any]] = []
        allowed_requirements = {item.id for item in plan.requirements}
        for raw in parsed.get("claims") or []:
            if not isinstance(raw, dict):
                continue
            ids = [str(value) for value in raw.get("evidenceIds") or [] if str(value) in allowed]
            if not ids:
                continue
            requirement_ids = [
                str(value) for value in raw.get("requirementIds") or []
                if str(value) in allowed_requirements
            ]
            claims.append({
                "text": str(raw.get("text") or "").strip(),
                "evidenceIds": list(dict.fromkeys(ids)),
                "requirementIds": list(dict.fromkeys(requirement_ids)),
                "risk": str(raw.get("risk") or "low"),
            })
        return GeneratedAnswer(answer=answer, claims=tuple(claims))
