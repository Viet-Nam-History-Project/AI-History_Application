"""A single conditional semantic claim/evidence verifier."""

from __future__ import annotations

import json
import re
from typing import Any

from .contracts import (
    EvidenceItem,
    EvidenceSelection,
    GeneratedAnswer,
    UnifiedPlan,
    VerificationResult,
)


VERIFY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "semantic_evidence_verification_v2",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": ["supported", "corrected", "insufficient"]},
                "correctedAnswer": {"type": "string"},
                "unsupportedClaims": {"type": "array", "items": {"type": "string"}},
                "missingRequirementIds": {"type": "array", "items": {"type": "string"}},
                "requirementCoverage": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "requirementId": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["covered", "repaired", "missing"],
                            },
                            "note": {"type": "string"},
                        },
                        "required": ["requirementId", "status", "note"],
                    },
                },
                "notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "status", "correctedAnswer", "unsupportedClaims",
                "missingRequirementIds", "requirementCoverage", "notes",
            ],
        },
    },
}


def should_verify(plan: UnifiedPlan, generated: GeneratedAnswer, conflicts: tuple[str, ...]) -> bool:
    if plan.requires_verification or conflicts or len(plan.requirements) > 1:
        return True
    if any(claim.get("risk") == "high" for claim in generated.claims):
        return True
    required_ids = {item.id for item in plan.requirements if item.required}
    if not required_ids.issubset(generated.covered_requirement_ids):
        return True
    return bool(re.search(r"\b\d{3,4}\b|\d{1,2}[/.-]\d{1,2}", generated.answer))


class SemanticEvidenceVerifier:
    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def verify(
        self,
        plan: UnifiedPlan,
        generated: GeneratedAnswer,
        evidence: list[EvidenceItem],
        selection: EvidenceSelection,
        conflicts: tuple[str, ...],
    ) -> VerificationResult:
        evidence_by_id = {item.id: item for item in evidence}
        used_ids = {
            evidence_id
            for claim in generated.claims
            for evidence_id in claim.get("evidenceIds") or []
        }
        # Include evidence assigned to omitted requirements. The verifier can
        # then repair a coverage omission in this same semantic pass instead
        # of requiring a growing chain of post-generation validators.
        used_ids.update(selection.evidence_ids)
        used = [evidence_by_id[item_id] for item_id in used_ids if item_id in evidence_by_id]
        prompt = f"""
Kiểm tra toàn bộ nhận định trong câu trả lời theo bằng chứng. Đây là verifier
ngữ nghĩa duy nhất, không phải một tập luật cho từng câu hỏi.

Một claim chỉ được giữ khi bằng chứng hỗ trợ đúng chủ thể, đúng quan hệ, đúng
phạm vi và đúng lần xuất hiện lịch sử. Đặc biệt kiểm tra tên người, địa điểm,
ngày tháng, số liệu, phủ định và quan hệ nhân quả. Không từ chối chỉ vì cách
diễn đạt khác từ khóa. Nếu có thể sửa hoàn toàn bằng chính bằng chứng, trả
`corrected`; nếu không, loại claim không được hỗ trợ. Không thêm kiến thức riêng.
Không giữ mốc giờ/số liệu quá chi tiết nếu nó không cần để trả lời và chỉ
xuất hiện trong một đoạn không curated; có thể sửa bằng cách bỏ độ chính xác
thừa thay vì loại cả câu trả lời.

Khi viết `correctedAnswer`, giữ bố cục học tập hữu ích của draft nếu nội dung
vẫn đúng. Dòng `Từ khóa ghi nhớ`, `Từ khóa đối chiếu` hoặc `Mạch ghi nhớ` chỉ
được cô đọng các claim đã có căn cứ; không xóa chúng chỉ vì cách trình bày,
nhưng cũng không dùng chúng để đưa thêm dữ kiện chưa được evidence hỗ trợ.

Đồng thời kiểm tra độ phủ theo requirements. Một requirement chỉ được xem là
đã trả lời khi câu trả lời có một nhận định rõ ràng giải quyết đúng quan hệ của
nó, không phải chỉ nhắc một nhãn khái quát. Nếu draft bỏ sót requirement nhưng
bằng chứng được gán cho requirement đó đã đủ, hãy viết `correctedAnswer` bổ sung
ngắn gọn và trả `corrected`. Nếu bằng chứng thật sự chưa đủ, ghi id vào
`missingRequirementIds` và không tự bịa phần thiếu. Không tự tạo requirement.

Với `participant_groups`, một nhãn bao quát như “toàn dân” không đủ để đánh
dấu covered nếu evidence đã gán còn nêu các nhóm/thành phần cụ thể mà draft bỏ
sót. Với `participant_roles`, một danh sách tên không đủ: draft phải nêu vai
trò, hình thức tham gia hoặc đóng góp có trong evidence. Khi có đủ evidence,
hãy sửa trong chính semantic pass này thay vì tạo thêm validator chuyên biệt.

Phải tạo đúng một dòng `requirementCoverage` cho MỖI requirement được cung cấp:
- `covered`: draft đã trả lời rõ;
- `repaired`: draft thiếu/mơ hồ nhưng correctedAnswer đã bổ sung từ evidence;
- `missing`: evidence chưa đủ hoặc correctedAnswer vẫn chưa trả lời được.
Không được đánh dấu covered nếu draft chỉ dùng một nhãn chung chung thay cho
cơ chế, điều kiện hay dữ kiện mà requirement yêu cầu.

`notes` chỉ dùng để ghi rủi ro, giới hạn hoặc lý do đã sửa/loại claim. Nếu
toàn bộ câu trả lời được hỗ trợ và không có vấn đề cần quản trị viên xem lại,
hãy trả `notes: []`; không ghi lời xác nhận tích cực vào trường này.

Câu hỏi: {plan.standalone_question}
Câu trả lời: {generated.answer}
Claims: {json.dumps(generated.claims, ensure_ascii=False)}
Requirement draft chưa gắn claim: {json.dumps([
    item.id for item in plan.requirements
    if item.required and item.id not in generated.covered_requirement_ids
], ensure_ascii=False)}
Requirements và evidence đã gán: {json.dumps([
    {
        'id': item.id,
        'question': item.question,
        'required': item.required,
        'evidenceIds': selection.requirement_evidence.get(item.id, []),
    }
    for item in plan.requirements
], ensure_ascii=False)}
Xung đột selector phát hiện: {json.dumps(conflicts, ensure_ascii=False)}
Bằng chứng: {json.dumps([item.prompt_payload(max_chars=2600) for item in used], ensure_ascii=False)}
""".strip()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format=VERIFY_SCHEMA,
            messages=[
                {"role": "system", "content": "Bạn kiểm chứng claim lịch sử bằng ngữ nghĩa và chỉ bằng corpus."},
                {"role": "user", "content": prompt},
            ],
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        status = str(parsed.get("status") or "insufficient")
        corrected_answer = str(parsed.get("correctedAnswer") or "").strip()
        answer = corrected_answer
        if status == "supported":
            answer = generated.answer
        allowed_requirement_ids = {item.id for item in plan.requirements}
        required_requirement_ids = {
            item.id for item in plan.requirements if item.required
        }
        coverage_statuses: dict[str, str] = {}
        for row in parsed.get("requirementCoverage") or []:
            if not isinstance(row, dict):
                continue
            requirement_id = str(row.get("requirementId") or "")
            row_status = str(row.get("status") or "missing")
            if requirement_id in allowed_requirement_ids:
                coverage_statuses[requirement_id] = row_status
        reported_missing = {
            str(item) for item in parsed.get("missingRequirementIds") or []
            if str(item) in allowed_requirement_ids
        }
        coverage_missing = {
            requirement_id
            for requirement_id in required_requirement_ids
            if coverage_statuses.get(requirement_id) not in {"covered", "repaired"}
        }
        if any(value == "repaired" for value in coverage_statuses.values()):
            status = "corrected"
            answer = corrected_answer
        elif coverage_missing and status == "supported":
            status = "insufficient"
        if status == "corrected" and not answer:
            status = "insufficient"
        return VerificationResult(
            status=status,
            answer=answer,
            unsupported_claims=tuple(str(item) for item in parsed.get("unsupportedClaims") or []),
            notes=tuple(str(item) for item in parsed.get("notes") or []),
            missing_requirement_ids=tuple(sorted(reported_missing | coverage_missing)),
        )
