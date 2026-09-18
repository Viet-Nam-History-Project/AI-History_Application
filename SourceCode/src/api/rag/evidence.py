"""Evidence retrieval and model-based selection."""

from __future__ import annotations

import json
import re
from typing import Any

from ..entity_resolution import normalize_search_key
from .contracts import (
    EvidenceItem,
    EvidenceSelection,
    UnifiedPlan,
    is_participant_inventory_plan,
)


# Query-control words do not identify a concrete participant.  Removing them
# before ranking an exact-subject scan lets terms proposed by the planner
# (for example, names of social groups or organisations) carry the score.
# This vocabulary describes the retrieval operation, not any historical
# event or its expected answer.
_QUERY_CONTROL_TERMS = {
    "ai", "bao", "cac", "cho", "co", "cua", "duoc", "gi", "gom",
    "hay", "khi", "la", "luc", "luong", "moi", "nao", "nhom", "nhung",
    "o", "phan", "tham", "gia", "thanh", "the", "thuoc", "tim", "trong",
    "vai", "tro", "va", "ve", "voi", "dong", "gop", "hinh", "thuc",
}


def _content_tokens(value: str) -> set[str]:
    normalized = normalize_search_key(value)
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) >= 2
    }


def _normalized_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_search_key(value))


def _detail_query_phrases(query: str, subject: str) -> set[str]:
    """Extract concrete multi-word hypotheses from a planner query.

    Token overlap is too weak for Vietnamese history: ``công`` in ``công
    cuộc`` must not be treated as evidence for ``công nhân``. Exact two-
    and three-word phrases retain that semantic distinction while remaining
    domain-independent.
    """
    words = _normalized_words(query)
    subject_tokens = _content_tokens(subject)
    phrases: set[str] = set()
    for size in (2, 3):
        for index in range(max(0, len(words) - size + 1)):
            gram = words[index:index + size]
            if any(token in _QUERY_CONTROL_TERMS for token in gram):
                continue
            if set(gram).issubset(subject_tokens):
                continue
            phrases.add(" ".join(gram))
    return phrases


class EvidenceTools:
    """Expose two evidence channels without embedding domain judgements."""

    def __init__(self, client: Any, embedding_model: str, repository: Any) -> None:
        self.client = client
        self.embedding_model = embedding_model
        self.repository = repository

    def retrieve(self, plan: UnifiedPlan, candidate_k: int = 18) -> tuple[list[EvidenceItem], dict[str, Any]]:
        queries = list(dict.fromkeys((plan.standalone_question, *plan.retrieval_queries)))[:12]
        embeddings = self.client.embeddings.create(
            model=self.embedding_model,
            input=queries,
        )
        records: dict[str, EvidenceItem] = {}
        channels: list[str] = []
        query_terms: list[str] = []
        candidate_count = 0
        exact_count = 0
        per_query_ids: dict[str, list[str]] = {query: [] for query in queries}
        per_window_ids: dict[str, list[str]] = {
            window.id: [] for window in plan.time_windows
        }
        for query, vector in zip(queries, embeddings.data, strict=True):
            result = self.repository.retrieve_hybrid(
                query, vector.embedding, candidate_k,
                exact_phrases=[plan.subject] if plan.subject else [],
                primary_entity=plan.subject,
            )
            candidate_count += int(result.get("candidateCount") or 0)
            exact_count += int(result.get("exactMatchCount") or 0)
            channels.extend(result.get("channels") or [])
            query_terms.extend(result.get("queryTerms") or [])
            for raw in result.get("items") or []:
                item = EvidenceItem.from_record(raw)
                if item is None:
                    continue
                per_query_ids[query].append(item.id)
                previous = records.get(item.id)
                if previous is None or item.score > previous.score:
                    records[item.id] = item

        # Timeline/comparison questions need evidence from every requested
        # period, not merely the globally highest vector matches.
        timeline_retrieval = getattr(
            self.repository, "retrieve_timeline_windows", None
        )
        if callable(timeline_retrieval) and plan.time_windows:
            try:
                timeline_records = timeline_retrieval(
                    plan.standalone_question,
                    [(window.start, window.end) for window in plan.time_windows],
                    per_window=5,
                )
            except Exception:
                timeline_records = []
            if timeline_records:
                channels.append("planned_time_windows")
            window_by_range = {
                (window.start, window.end): window.id
                for window in plan.time_windows
            }
            for raw in timeline_records:
                item = EvidenceItem.from_record(raw)
                if item is None:
                    continue
                window_id = window_by_range.get((
                    raw.get("timelineWindowStart"),
                    raw.get("timelineWindowEnd"),
                ))
                if window_id:
                    per_window_ids[window_id].append(item.id)
                previous = records.get(item.id)
                if previous is None or item.score > previous.score:
                    records[item.id] = item

        # A plural force/participant request is an inventory question. Hybrid
        # top-k is good at finding a broad event summary, but that summary can
        # crowd out later pages which describe individual groups and their
        # roles. Scan the bounded set of chunks that explicitly name the same
        # subject, then reserve the chunks that overlap each planner-proposed
        # detail query. The planner supplies the hypotheses; no event, person
        # or social group is encoded here.
        subject_detail_ids: list[str] = []
        subject_scan = getattr(self.repository, "retrieve_subject_evidence", None)
        if (
            callable(subject_scan)
            and plan.subject
            and is_participant_inventory_plan(plan)
        ):
            try:
                scanned_records = subject_scan([plan.subject], limit=240)
            except Exception:
                scanned_records = []
            if scanned_records:
                channels.append("subject_detail_scan")
            scanned_items = [
                item
                for raw in scanned_records
                for item in [EvidenceItem.from_record(raw)]
                if item is not None
                and normalize_search_key(plan.subject)
                in normalize_search_key(" ".join((
                    item.heading,
                    item.text,
                    *item.related_entities,
                )))
            ]
            for query in queries:
                detail_phrases = _detail_query_phrases(query, plan.subject)
                if not detail_phrases:
                    continue
                ranked_scan: list[tuple[float, EvidenceItem]] = []
                for item in scanned_items:
                    normalized_item = " " + normalize_search_key(
                        " ".join((item.heading, item.text, *item.related_entities))
                    ) + " "
                    overlap = {
                        phrase for phrase in detail_phrases
                        if f" {phrase} " in normalized_item
                    }
                    if not overlap:
                        continue
                    coverage = len(overlap) / max(1, len(detail_phrases))
                    # Number of independently matched detail terms matters
                    # more than the original subject-scan rank. The latter is
                    # retained only as a stable tie breaker.
                    phrase_weight = sum(len(phrase.split()) for phrase in overlap)
                    lexical_score = phrase_weight + coverage + item.score * 0.05
                    ranked_scan.append((lexical_score, item))
                ranked_scan.sort(key=lambda row: row[0], reverse=True)
                for lexical_score, item in ranked_scan[:4]:
                    subject_detail_ids.append(item.id)
                    # Keep the normalised value in the same score range as
                    # hybrid candidates while preserving lexical ordering.
                    item.score = max(item.score, min(0.97, 0.70 + lexical_score / 20))
                    previous = records.get(item.id)
                    if previous is None or item.score > previous.score:
                        records[item.id] = item

        # Curated JSON is a fact-store channel. Search it directly as well as
        # through composite hybrid retrieval, because direct facts may not
        # share the PDF's wording or embedding neighbourhood.
        published = getattr(self.repository, "published", None)
        if published is not None:
            for query in queries:
                for raw in published.search(query, max(6, candidate_k // 2)):
                    item = EvidenceItem.from_record(raw)
                    if item is None:
                        continue
                    per_query_ids[query].append(item.id)
                    previous = records.get(item.id)
                    if previous is None or item.score > previous.score:
                        records[item.id] = item
            channels.append("verified_fact_store")

        # Remove only provably disjoint evidence. Undated chunks remain
        # available for semantic inspection instead of being discarded.
        temporal_filtered_ids: set[str] = set()
        if plan.year_start is not None and plan.year_end is not None:
            temporal_filtered_ids = {
                item.id
                for item in records.values()
                if item.overlaps(plan.year_start, plan.year_end) is False
            }
            for item_id in temporal_filtered_ids:
                records.pop(item_id, None)
            for ids in per_query_ids.values():
                ids[:] = [item_id for item_id in ids if item_id in records]
            for ids in per_window_ids.values():
                ids[:] = [item_id for item_id in ids if item_id in records]

        # Reserve room for reviewed fact records. A large PDF corpus can
        # otherwise crowd a directly relevant curated record out of the model
        # selection window with many broadly similar vector matches.
        all_items = list(records.values())
        curated = sorted(
            (item for item in all_items if item.curated),
            key=lambda item: (item.source_priority, item.score),
            reverse=True,
        )[:12]
        curated_ids = {item.id for item in curated}

        subject_detail: list[EvidenceItem] = []
        subject_detail_seen = set(curated_ids)
        for item_id in subject_detail_ids:
            if item_id in records and item_id not in subject_detail_seen:
                subject_detail.append(records[item_id])
                subject_detail_seen.add(item_id)
            if len(subject_detail) >= 16:
                break

        window_reserved: list[EvidenceItem] = []
        window_seen = {*curated_ids, *subject_detail_seen}
        for window in plan.time_windows:
            for item_id in per_window_ids.get(window.id, [])[:5]:
                if item_id in records and item_id not in window_seen:
                    window_reserved.append(records[item_id])
                    window_seen.add(item_id)

        # Preserve a small, independent evidence window for every semantic
        # retrieval query.  Without this, a broad summary query can fill all
        # 48 slots and crowd out the more specific query about roles,
        # contributions or a secondary facet.  This is query diversity, not a
        # domain-specific reranking rule.
        query_reserved: list[EvidenceItem] = []
        reserved_ids = {*curated_ids, *subject_detail_seen, *window_seen}
        for query in queries:
            query_items = sorted(
                (
                    records[item_id]
                    for item_id in dict.fromkeys(per_query_ids.get(query, []))
                    if item_id in records and item_id not in reserved_ids
                ),
                key=lambda item: item.score,
                reverse=True,
            )[:5]
            for item in query_items:
                if item.id not in reserved_ids:
                    query_reserved.append(item)
                    reserved_ids.add(item.id)

        narrative = sorted(
            (item for item in all_items if item.id not in reserved_ids),
            key=lambda item: item.score,
            reverse=True,
        )[: max(0, 48 - len(curated) - len(subject_detail) - len(window_reserved) - len(query_reserved))]
        ranked = [
            *curated, *subject_detail, *window_reserved, *query_reserved,
            *narrative,
        ][:48]
        return ranked, {
            "candidate_count": candidate_count,
            "channels": list(dict.fromkeys(channels)),
            "query_terms": list(dict.fromkeys(query_terms)),
            "exact_match_count": exact_count,
            "temporal_filtered_count": len(temporal_filtered_ids),
        }


SELECTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "evidence_selection",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "requirementEvidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "requirementId": {"type": "string"},
                            "evidenceIds": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["requirementId", "evidenceIds"],
                    },
                },
                "missingRequirements": {"type": "array", "items": {"type": "string"}},
                "conflicts": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["requirementEvidence", "missingRequirements", "conflicts"],
        },
    },
}


class ModelEvidenceSelector:
    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def select(self, plan: UnifiedPlan, candidates: list[EvidenceItem], limit: int = 12) -> EvidenceSelection:
        if not candidates:
            return EvidenceSelection((), {}, tuple(r.id for r in plan.requirements))
        allowed_ids = {item.id for item in candidates}
        prompt = f"""
Chọn bằng chứng cho từng yêu cầu của câu hỏi lịch sử. Đánh giá bằng ngữ nghĩa,
không chỉ trùng từ. Một bằng chứng chỉ hợp lệ khi đúng chủ thể, đúng quan hệ
được hỏi và đúng lần xuất hiện/phạm vi thời gian. Phân biệt mốc bắt đầu với
mốc tái xâm lược, địa điểm khởi đầu với nơi phong trào lan tới, người trực tiếp
chỉ huy với cơ quan lãnh đạo. Các ví dụ này là nguyên tắc quan hệ tổng quát,
không phải đáp án cho một câu cụ thể.

Không suy ra đáp án. Nếu chưa đủ thì để trống và đánh dấu missing. Khi các đoạn
mâu thuẫn về cùng một quan hệ, ghi vào conflicts; ưu tiên dữ liệu đã kiểm duyệt
và đoạn nói trực tiếp, cụ thể.

Metadata `yearStart/yearEnd` của evidence là khoảng thời gian của đoạn.
Không chọn evidence có khoảng năm tách rời phạm vi requirement. Với
requirement gắn nhiều `timeWindowIds` (tiếp nối, thay đổi, so sánh),
tập evidence phải có căn cứ cho từng cửa sổ; một phía không thể đại
diện cho cả hai. Evidence không có metadata năm chỉ được chọn khi nội
dung tự nó xác nhận đúng giai đoạn.

Với yêu cầu nhân quả, đoạn chỉ nói chung rằng “thời cơ thuận lợi”, “chuẩn bị
tốt” hoặc “lãnh đạo đúng” chưa đủ thay cho bằng chứng nêu cơ chế hay điều kiện
cụ thể. Hãy chọn bằng chứng trực tiếp cho từng vai trò nhân quả mà planner đã
tách; không dùng một đoạn chung để giả vờ bao phủ nhiều yêu cầu khác nhau.

Với requirement `participant_groups`, một cụm bao quát như “toàn dân”, “quần
chúng” hay “nhiều lực lượng” chỉ chứng minh tính rộng rãi, chưa đủ chứng minh
danh sách các nhóm/thành phần nếu các ứng viên còn có phân loại cụ thể. Hãy
chọn các đoạn bổ sung lẫn nhau để bao phủ những nhóm có căn cứ trực tiếp.
Với `participant_roles`, đoạn chỉ kể tên không đủ: bằng chứng phải mô tả vai
trò, hình thức tham gia hoặc đóng góp của nhóm. Ưu tiên kết hợp bản ghi curated
để xác nhận chủ thể với các đoạn PDF chính thức giàu chi tiết; không để bản tóm
tắt rộng thay thế toàn bộ bằng chứng chi tiết.

Kế hoạch: {json.dumps({
    'question': plan.standalone_question,
    'subject': plan.subject,
    'scope': plan.scope,
    'dateRange': plan.date_range,
    'yearStart': plan.year_start,
    'yearEnd': plan.year_end,
    'timeWindows': [
        {'id': window.id, 'start': window.start, 'end': window.end, 'label': window.label}
        for window in plan.time_windows
    ],
    'requirements': [r.__dict__ if hasattr(r, '__dict__') else {
        'id': r.id, 'question': r.question, 'answerType': r.answer_type,
        'required': r.required,
        'timeWindowIds': list(r.time_window_ids),
    } for r in plan.requirements],
}, ensure_ascii=False)}

Ứng viên: {json.dumps([item.prompt_payload() for item in candidates], ensure_ascii=False)}
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format=SELECTION_SCHEMA,
                messages=[
                    {"role": "system", "content": "Bạn là bộ chọn bằng chứng RAG; không trả lời người dùng."},
                    {"role": "user", "content": prompt},
                ],
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
        except Exception:
            # A selector outage must never turn the highest-scoring chunks
            # into evidence for every requirement.  That old "best effort"
            # behaviour was the source of several confident but wrong
            # answers.  Fail closed here; the pipeline can still return a
            # transparent corpus-insufficient response.
            parsed = {
                "requirementEvidence": [],
                "missingRequirements": [item.id for item in plan.requirements],
                "conflicts": ["Không thể hoàn tất bước chọn bằng chứng."],
            }
        allowed_requirement_ids = {item.id for item in plan.requirements}
        requirement_by_id = {item.id: item for item in plan.requirements}
        window_by_id = {window.id: window for window in plan.time_windows}
        evidence_by_id = {item.id: item for item in candidates}
        mapping: dict[str, list[str]] = {}
        selected: list[str] = []
        for row in parsed.get("requirementEvidence") or []:
            if not isinstance(row, dict):
                continue
            requirement_id = str(row.get("requirementId") or "")
            if requirement_id not in allowed_requirement_ids:
                continue
            ids = [str(item) for item in row.get("evidenceIds") or [] if str(item) in allowed_ids]
            ids = list(dict.fromkeys(ids))[:8]
            requirement = requirement_by_id[requirement_id]
            assigned_windows = [
                window_by_id[window_id]
                for window_id in requirement.time_window_ids
                if window_id in window_by_id
            ]
            if assigned_windows:
                ids = [
                    item_id for item_id in ids
                    if evidence_by_id[item_id].year_start is None
                    or any(
                        evidence_by_id[item_id].overlaps(window.start, window.end)
                        is True
                        for window in assigned_windows
                    )
                ]
            mapping[requirement_id] = ids
            selected.extend(ids)
        reported_missing = {
            str(item) for item in parsed.get("missingRequirements") or []
            if str(item) in allowed_requirement_ids
        }
        computed_missing = {
            requirement.id for requirement in plan.requirements
            if requirement.required and not mapping.get(requirement.id)
        }
        # For a comparison/evolution requirement, dated evidence must cover
        # every assigned side. Undated evidence is left to the semantic
        # selector because its text may state the period explicitly.
        for requirement in plan.requirements:
            ids = mapping.get(requirement.id, [])
            if not ids or not requirement.time_window_ids:
                continue
            selected_items = [evidence_by_id[item_id] for item_id in ids]
            if any(item.year_start is None and item.year_end is None for item in selected_items):
                continue
            if any(
                not any(item.overlaps(window_by_id[window_id].start, window_by_id[window_id].end) for item in selected_items)
                for window_id in requirement.time_window_ids
                if window_id in window_by_id
            ):
                computed_missing.add(requirement.id)
        return EvidenceSelection(
            evidence_ids=tuple(dict.fromkeys(selected))[:limit],
            requirement_evidence=mapping,
            missing_requirements=tuple(sorted(reported_missing | computed_missing)),
            conflicts=tuple(str(item) for item in parsed.get("conflicts") or []),
        )
